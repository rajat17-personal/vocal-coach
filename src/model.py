"""
VocalCoach Models — lightweight TCN and Conformer architectures for
multi-task singing analysis and vocal coaching.

=== What These Models Do ===

Given an audio clip represented as a mel spectrogram, each model predicts
three things at every 10ms time step:

  1. VAD (Voice Activity Detection): Is the singer vocalising right now?
     Output: probability in [0, 1].

  2. Pitch Posteriorgram: What pitch is being sung?
     Output: 360 probabilities over 20-cent bins, covering B0 (~31.7 Hz)
     through ~B6 (~2006 Hz). Decoded to Hz via Viterbi.

  3. Technique Classification: Which singing technique is active?
     Output: N independent probabilities (multi-label sigmoid, not softmax).
     A singer can simultaneously be vibrato AND breathy — these are not
     mutually exclusive, so each class is modelled independently.
     Default classes: vibrato, breathy, falsetto, belt, straight.

These per-frame outputs feed a post-processing layer (features.py, Phase 2)
that derives vibrato metrics, note-level accuracy, HNR, spectral features,
dynamics, and breath patterns — producing a full per-session coaching report.

=== Baseline for Comparison ===

NanoPitch (GRU, training/model.py): ~333K params, 96.1% rtRPA, 14.1c median.
That model was designed for real-time pitch tracking only. VocalCoach is
designed for multi-task coaching — pitch accuracy parity is desirable but
technique classification F1 is the primary evaluation criterion.

=== Architecture A: VocalCoachTCN ===

Temporal Convolutional Network with exponentially dilated (optionally causal)
convolutions. Each block doubles the dilation, growing the receptive field
geometrically while keeping the sequence length constant:

  Block 0: dilation=1   → 30 ms context
  Block 1: dilation=2   → 70 ms context
  Block 2: dilation=4   → 150 ms context
  ...
  Block 7: dilation=128 → ~3.8 s context  (captures phrase-level patterns)

Signal flow:
    mel (B, T, 40)
      │
      ▼
    Conv1d(40 → hidden, k=1) + GELU        ← project mel bands to hidden dim
      │
      ▼
    TCNBlock × n_blocks                    ← dilated conv stack
    (dilation = 1, 2, 4, 8, …, 2^(n-1))   ← each block: conv → LN → GELU → residual
      │
      ▼
    LayerNorm
      │
      ├──→ Linear(hidden → 1)   + sigmoid  → VAD probability
      ├──→ Linear(hidden → 360) + sigmoid  → pitch posteriorgram
      └──→ Linear(hidden → N)   + sigmoid  → technique labels (multi-label)

`causal=True`  (default): left-padding only → eligible for streaming / WASM
`causal=False`: symmetric padding → slightly better accuracy for offline use

~450K parameters with default hidden=128, n_blocks=8.

=== Architecture B: VocalCoachConformer ===

Conformer (Convolution-augmented Transformer) combines depthwise convolution
for local pattern capture with multi-head self-attention for global context.
Uses the "Macaron" structure from the original Conformer paper (Gulati et al.,
2020), where two half-step feed-forward modules sandwich the attention and
convolution modules:

  x → FF(½) → Self-Attention → ConvModule → FF(½) → LayerNorm

The convolution module's depthwise kernel (default k=31, i.e. 310 ms) is
well-matched to vibrato period (~125–200 ms) and breath phrase boundaries.
Self-attention gives global phrase-level context that GRUs and TCNs lack.

Signal flow:
    mel (B, T, 40)
      │
      ▼
    Linear(40 → hidden)                    ← project mel bands
      │
      ▼
    ConformerBlock × n_layers              ← FF → Attn → Conv → FF → LN
      │
      ▼
    LayerNorm
      │
      ├──→ Linear(hidden → 1)   + sigmoid  → VAD
      ├──→ Linear(hidden → 360) + sigmoid  → pitch posteriorgram
      └──→ Linear(hidden → N)   + sigmoid  → technique labels

Offline-only: full sequence attention cannot run causally without a windowed
approximation. No streaming support.

~295K parameters with default hidden=64, n_layers=4.

=== Why Two Architectures? ===

TCN:
  + Causal → can be deployed to browser via WASM (Phase 4 stretch goal)
  + Faster training (fully parallelisable, no attention O(T²))
  - Receptive field fixed at training time by dilation schedule

Conformer:
  + Global context → better technique / quality classification
  + Depthwise conv naturally captures vibrato periodicity
  - Offline only (attention over full clip)
  - WASM export very difficult

Phase 1 trains both at similar parameter budgets and selects the winner based
on technique F1 (primary) and pitch accuracy (secondary).

=== Experiment C (Phase 1) ===

A third experiment uses MERT-v1-95M (music understanding transformer, trained
on 160k hours, m-a-p/MERT-v1-95M on HuggingFace) as a frozen backbone with
lightweight task heads fine-tuned on VocalSet/GTSinger. This tests whether
music pretraining significantly reduces the labelled singing data requirement.
That experiment is NOT implemented here — it requires the transformers library
and is set up separately in train_mert.py (Phase 1, TBD).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Constants — shared with NanoPitch for compatibility
# ═══════════════════════════════════════════════════════════════════════

PITCH_BINS = 360
PITCH_FMIN = 31.7          # Hz — ~B0
PITCH_CENTS_PER_BIN = 20   # 20-cent resolution, 6 octaves total

N_MELS = 40

# Singing technique classes (multi-label — order matters for evaluation)
TECHNIQUE_NAMES = ['vibrato', 'breathy', 'falsetto', 'belt', 'straight']
N_TECHNIQUES = len(TECHNIQUE_NAMES)


# ═══════════════════════════════════════════════════════════════════════
# Pitch Utilities — identical to NanoPitch for cross-model compatibility
# ═══════════════════════════════════════════════════════════════════════

def f0_to_bin(f0_hz):
    """Convert fundamental frequency (Hz) to pitch bin index.

    bin = 1200 * log2(f0 / f_min) / cents_per_bin

    Returns -1 for unvoiced frames (f0 <= 0).
    """
    f0_hz = np.asarray(f0_hz, dtype=np.float64)
    result = np.full_like(f0_hz, -1.0)
    voiced = f0_hz > 0
    result[voiced] = 1200.0 * np.log2(f0_hz[voiced] / PITCH_FMIN) / PITCH_CENTS_PER_BIN
    return result


def bin_to_f0(bins):
    """Convert pitch bin index to Hz. Inverse of f0_to_bin."""
    bins = np.asarray(bins, dtype=np.float64)
    return PITCH_FMIN * 2.0 ** (bins * PITCH_CENTS_PER_BIN / 1200.0)


def f0_to_posteriorgram(f0_hz, n_frames=None, sigma_bins=1.2):
    """Create a Gaussian-blurred pitch posteriorgram from f0 values.

    Soft labels: each voiced frame gets a Gaussian bump centred at the true
    pitch bin instead of a hard one-hot vector. Helps the model learn that
    pitch is continuous rather than discrete.

    Args:
        f0_hz:      (T,) array of f0 in Hz (0 = unvoiced)
        sigma_bins: Gaussian width in bins (1.2 bins ≈ 24 cents)

    Returns:
        (T, 360) float32 posteriorgram
    """
    if n_frames is None:
        n_frames = len(f0_hz)

    f0_hz = np.asarray(f0_hz[:n_frames], dtype=np.float64)
    bins = f0_to_bin(f0_hz)
    posteriorgram = np.zeros((n_frames, PITCH_BINS), dtype=np.float32)
    bin_indices = np.arange(PITCH_BINS, dtype=np.float64)

    for t in range(n_frames):
        if bins[t] < 0:
            continue
        dist = bin_indices - bins[t]
        posteriorgram[t] = np.exp(-0.5 * (dist / sigma_bins) ** 2)

    return posteriorgram


def viterbi_decode(posteriorgram, transition_width=12, voicing_threshold=0.3,
                   onset_penalty=1.0):
    """Offline Viterbi decoder: posteriorgram → smooth f0 track.

    Finds the globally optimal pitch sequence using dynamic programming.
    Identical to NanoPitch's offline decoder — use this for post-session
    analysis where full-clip context is available.

    Args:
        posteriorgram:    (T, 360) pitch probabilities from model
        transition_width: max pitch change per frame in bins (12 = 240 cents)
        voicing_threshold: min peak probability to initialise as voiced
        onset_penalty:    log-domain cost for voiced↔unvoiced transitions

    Returns:
        f0_hz: (T,) float32 decoded pitch in Hz (0 = unvoiced)
    """
    T, N = posteriorgram.shape
    if T == 0:
        return np.zeros(0, dtype=np.float32)

    tw = int(transition_width)
    W = 2 * tw + 1
    log_obs = np.log(posteriorgram + 1e-10)

    V = np.full((T, N + 1), -np.inf, dtype=np.float64)
    bp = np.zeros((T, N + 1), dtype=np.int32)

    max_post = posteriorgram[0].max()
    if max_post > voicing_threshold:
        V[0, :N] = log_obs[0]
    V[0, N] = np.log(1.0 - max_post + 1e-10)

    for t in range(1, T):
        max_post_t = posteriorgram[t].max()
        prev = V[t - 1, :N]

        padded = np.pad(prev, (tw, tw), constant_values=-np.inf)
        windows = np.lib.stride_tricks.as_strided(
            padded, shape=(N, W),
            strides=(padded.strides[0], padded.strides[0]))
        best_k = np.argmax(windows, axis=1)
        best_val = windows[np.arange(N), best_k]
        best_from_voiced = np.clip(np.arange(N) - tw + best_k, 0, N - 1)

        from_unvoiced = V[t - 1, N] - onset_penalty
        use_voiced = best_val >= from_unvoiced
        V[t, :N] = np.where(use_voiced, best_val, from_unvoiced) + log_obs[t]
        bp[t, :N] = np.where(use_voiced, best_from_voiced, N)

        best_voiced_score = prev.max()
        best_voiced_idx = prev.argmax()
        from_voiced = best_voiced_score - onset_penalty
        stay_uv = V[t - 1, N]
        uv_obs = np.log(1.0 - max_post_t + 1e-10)

        if stay_uv >= from_voiced:
            V[t, N] = stay_uv + uv_obs
            bp[t, N] = N
        else:
            V[t, N] = from_voiced + uv_obs
            bp[t, N] = best_voiced_idx

    path = np.zeros(T, dtype=np.int32)
    path[T - 1] = np.argmax(V[T - 1])
    for t in range(T - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]

    f0_hz = np.zeros(T, dtype=np.float32)
    voiced_mask = path < N
    if voiced_mask.any():
        f0_hz[voiced_mask] = bin_to_f0(path[voiced_mask].astype(np.float64))

    return f0_hz


# ═══════════════════════════════════════════════════════════════════════
# Architecture A: VocalCoachTCN
# ═══════════════════════════════════════════════════════════════════════

class TCNBlock(nn.Module):
    """Single TCN residual block with a dilated (optionally causal) convolution.

    The dilation parameter controls how far apart the kernel samples are:
      dilation=1  → kernel sees frames [t-2, t-1, t] (causal, kernel=3)
      dilation=4  → kernel sees frames [t-8, t-4, t]
      dilation=128 → kernel sees frames [t-256, t-128, t]

    Causal padding (left-only) ensures output[t] never depends on future
    input, preserving the streaming invariant for potential WASM deployment.
    Symmetric padding (non-causal) gives slightly better accuracy for offline.

    Residual connection: output = conv(norm(act(x))) + x
    LayerNorm before the residual add stabilises multi-task gradient flow
    better than BatchNorm (which conflates batch and time statistics).
    """

    def __init__(self, channels, kernel_size=3, dilation=1, causal=True,
                 dropout=0.1):
        super().__init__()
        self.causal = causal
        pad = (kernel_size - 1) * dilation
        # Causal: all padding on the left.  Non-causal: split symmetrically.
        self.pad_left  = pad if causal else pad // 2
        self.pad_right = 0   if causal else pad - pad // 2

        self.conv    = nn.Conv1d(channels, channels, kernel_size,
                                 dilation=dilation, padding=0)
        self.norm    = nn.LayerNorm(channels)
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C)
        residual = x
        h = x.permute(0, 2, 1)                          # (B, C, T)
        h = F.pad(h, (self.pad_left, self.pad_right))    # (B, C, T + pad)
        h = self.conv(h)                                 # (B, C, T)
        h = h.permute(0, 2, 1)                          # (B, T, C)
        h = self.norm(h)
        h = self.act(h)
        h = self.dropout(h)
        return h + residual


class VocalCoachTCN(nn.Module):
    """Lightweight TCN vocal coaching model.

    Args:
        n_mels:       mel spectrogram input bands (40)
        hidden:       internal channel width (128)
        n_blocks:     number of TCN blocks; dilation doubles each block (8)
        kernel_size:  dilated conv kernel size (3)
        causal:       left-pad only — streaming-safe; set False for offline (True)
        dropout:      dropout rate applied inside each block (0.1)
        n_techniques: number of technique classes (5)
    """

    def __init__(self, n_mels=N_MELS, hidden=128, n_blocks=8, kernel_size=3,
                 causal=True, dropout=0.1, n_techniques=N_TECHNIQUES,
                 quality_head=0, deep_technique_head=False, note_head=False,
                 deep_note_head=False, n_attn_layers=0, n_heads=4):
        super().__init__()
        self.causal  = causal
        self.hidden  = hidden
        self.n_blocks = n_blocks
        self.quality_dims = quality_head

        # Project mel bands into the hidden dimension (1×1 conv = linear per frame)
        self.input_proj = nn.Conv1d(n_mels, hidden, kernel_size=1)

        # Dilated TCN stack — dilation doubles each block, so 8 blocks give
        # a receptive field of sum((k-1)*2^i, i=0..7)*10ms ≈ 3.8 s.
        self.blocks = nn.ModuleList([
            TCNBlock(hidden, kernel_size, dilation=2 ** i,
                     causal=causal, dropout=dropout)
            for i in range(n_blocks)
        ])

        # Optional Conformer blocks after the TCN stack.
        # Using full ConformerBlock (FF → Attn → Conv → FF → Norm) rather than
        # a bare MHA+residual because the FF modules buffer attention gradients,
        # preventing the global attention signal from overwriting the local VAD
        # features built by the TCN conv stack. A bare MHA layer lets the pitch
        # loss (dominant under tight sigma or high w-pitch) flow directly into
        # the shared representation and suppresses the VAD head gradient.
        # n_attn_layers=0 (default) preserves the original pure-TCN behaviour.
        self.attn_layers = nn.ModuleList([
            ConformerBlock(hidden, n_heads=n_heads, dropout=dropout, causal=causal)
            for _ in range(n_attn_layers)
        ])

        self.norm = nn.LayerNorm(hidden)

        # ── Output heads (multi-task) ──
        self.head_vad   = nn.Linear(hidden, 1)
        self.head_pitch = nn.Linear(hidden, PITCH_BINS)
        # Technique head: single Linear (default) or 2-layer MLP for probe-mode
        # where the backbone is frozen and the head must do more heavy lifting.
        if deep_technique_head:
            self.head_technique = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, n_techniques),
            )
        else:
            self.head_technique = nn.Linear(hidden, n_techniques)

        # ── Optional quality scoring head ──
        # Clip-level quality via mean-pooled backbone hidden states → 2-layer MLP.
        # quality_head=0: disabled
        # quality_head=1: single scalar (Variant 1 contrastive, Variant 3 MSE distil)
        # quality_head=9: 9-dim output matching ccmusic expert dimensions (Variant 2)
        # No output activation: contrastive ranking loss needs raw logits; caller
        # normalises to display scale. MSE targets are also passed unnormalised.
        if quality_head > 0:
            self.head_quality = nn.Sequential(
                nn.Linear(hidden, hidden // 4),
                nn.GELU(),
                nn.Linear(hidden // 4, quality_head),
            )

        # ── Optional note segmentation head (Variant 4) ──
        # Two independent binary classifiers sharing the same backbone.
        # note_onset:  1 at frames where a new note begins
        # note_offset: 1 at frames where a note ends
        self.has_note_head = note_head
        if note_head:
            if deep_note_head:
                self.head_note_onset  = nn.Sequential(
                    nn.Linear(hidden, hidden // 4), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(hidden // 4, 1))
                self.head_note_offset = nn.Sequential(
                    nn.Linear(hidden, hidden // 4), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(hidden // 4, 1))
            else:
                self.head_note_onset  = nn.Linear(hidden, 1)
                self.head_note_offset = nn.Linear(hidden, 1)

        self._init_weights()
        n = sum(p.numel() for p in self.parameters())
        attn_str = f", attn={n_attn_layers}×{n_heads}h" if n_attn_layers else ""
        note_str = (", deep_note_head=True" if (note_head and deep_note_head)
                    else ", note_head=True" if note_head else "")
        print(f"VocalCoachTCN: {n:,} parameters "
              f"(hidden={hidden}, blocks={n_blocks}, causal={causal}{attn_str}"
              f"{note_str}"
              f"{f', quality_head={quality_head}' if quality_head else ''})")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, mel, return_embeddings=False):
        """Run the model on a batch of mel spectrograms.

        Args:
            mel:               (B, T, 40) — log-mel spectrogram
            return_embeddings: if True, also return backbone hidden states (B, T, hidden)

        Returns:
            vad:       (B, T, 1)   — voice activity probability
            pitch:     (B, T, 360) — pitch posteriorgram
            technique: (B, T, N)   — per-technique probability (multi-label)
            quality:   (B, Q)      — clip-level quality scores (Q=quality_head dims), or None
            embeddings (only when return_embeddings=True): (B, T, hidden)
        """
        x = mel.permute(0, 2, 1)           # (B, 40, T)
        x = F.gelu(self.input_proj(x))     # (B, hidden, T)
        x = x.permute(0, 2, 1)            # (B, T, hidden)

        for block in self.blocks:
            x = block(x)

        for layer in self.attn_layers:
            x = layer(x)

        x = self.norm(x)

        vad       = torch.sigmoid(self.head_vad(x))        # (B, T, 1)
        pitch     = torch.sigmoid(self.head_pitch(x))      # (B, T, 360)
        technique = torch.sigmoid(self.head_technique(x))  # (B, T, N)
        quality   = self.head_quality(x.mean(dim=1)) if self.quality_dims else None
        note_onset  = torch.sigmoid(self.head_note_onset(x))  if self.has_note_head else None
        note_offset = torch.sigmoid(self.head_note_offset(x)) if self.has_note_head else None

        if return_embeddings:
            return vad, pitch, technique, quality, note_onset, note_offset, x
        return vad, pitch, technique, quality, note_onset, note_offset

    def receptive_field_ms(self, hop_ms=10):
        """Temporal receptive field of the TCN stack in milliseconds."""
        rf_frames = sum((3 - 1) * (2 ** i) for i in range(self.n_blocks)) + 1
        return rf_frames * hop_ms


# ═══════════════════════════════════════════════════════════════════════
# Architecture B: VocalCoachConformer
# ═══════════════════════════════════════════════════════════════════════

class FeedForwardModule(nn.Module):
    """Conformer feed-forward module with half-step residual scaling.

    Structure: LayerNorm → Linear(expand) → SiLU → Dropout → Linear(project) → Dropout
    Output: x + 0.5 * FFN(x)  (the 0.5 is from the original Conformer paper)

    SiLU (Swish) is used as the activation following the original paper.
    """

    def __init__(self, hidden, expansion=4, dropout=0.1):
        super().__init__()
        self.norm    = nn.LayerNorm(hidden)
        self.linear1 = nn.Linear(hidden, hidden * expansion)
        self.linear2 = nn.Linear(hidden * expansion, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        h = F.silu(self.linear1(h))   # Swish activation
        h = self.dropout(h)
        h = self.linear2(h)
        h = self.dropout(h)
        return x + 0.5 * h


class ConvolutionModule(nn.Module):
    """Conformer convolution module.

    Structure: LayerNorm → Pointwise(expand 2×) → GLU → Depthwise → BN → SiLU → Pointwise

    The GLU gate halves the channel count after pointwise expansion, so the
    effective channel count entering the depthwise conv equals `hidden`.
    Depthwise conv (groups=hidden) applies one filter per channel — captures
    local temporal patterns efficiently with few parameters.

    Kernel size of 31 → 310 ms receptive field, well-matched to vibrato
    period (125–200 ms) and syllable-level phrasing.

    causal=True:  left-pad only — output at t cannot depend on frames after t.
    causal=False: symmetric padding — sees past and future equally.
    """

    def __init__(self, hidden, kernel_size=31, dropout=0.1, causal=False):
        super().__init__()
        self.norm       = nn.LayerNorm(hidden)
        self.pointwise1 = nn.Linear(hidden, 2 * hidden)   # expand + GLU gate

        if causal:
            # Left-pad by (kernel_size - 1) so output[t] only sees input[≤t]
            self._conv_pad = kernel_size - 1
            self.depthwise = nn.Conv1d(hidden, hidden, kernel_size,
                                       padding=0, groups=hidden)
        else:
            # Symmetric padding — half kernel on each side
            self._conv_pad = 0
            self.depthwise = nn.Conv1d(hidden, hidden, kernel_size,
                                       padding=(kernel_size - 1) // 2,
                                       groups=hidden)

        self.bn         = nn.BatchNorm1d(hidden)
        self.pointwise2 = nn.Linear(hidden, hidden)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, hidden)
        residual = x
        x = self.norm(x)

        # Expand then gate: split into two halves, multiply element-wise
        x = self.pointwise1(x)               # (B, T, 2*hidden)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * torch.sigmoid(x2)           # GLU → (B, T, hidden)

        # Depthwise conv along the time axis
        x = x.permute(0, 2, 1)              # (B, hidden, T)
        if self._conv_pad:
            x = F.pad(x, (self._conv_pad, 0))   # left-pad only (causal)
        x = self.depthwise(x)
        x = self.bn(x)
        x = F.silu(x)
        x = x.permute(0, 2, 1)             # (B, T, hidden)

        x = self.pointwise2(x)
        x = self.dropout(x)
        return residual + x


class ConformerBlock(nn.Module):
    """Single Conformer block (Macaron structure).

    FF(½) → Self-Attention → ConvModule → FF(½) → LayerNorm

    The two half-step feed-forward modules act as a "sandwich" around
    the attention + conv core. This structure from Gulati et al. 2020
    consistently outperforms the standard Transformer-only approach on
    audio tasks.

    Pre-norm pattern for self-attention (LayerNorm before attention input)
    stabilises training with smaller learning rates.

    causal=True:  attention mask prevents frame t from seeing frames t+1…T.
                  Depthwise conv uses left-padding only.
    causal=False: full bidirectional attention; symmetric conv padding.
    """

    def __init__(self, hidden, n_heads=4, ff_expansion=4, conv_kernel=31,
                 dropout=0.1, causal=False):
        super().__init__()
        self.causal      = causal
        self.ff1         = FeedForwardModule(hidden, ff_expansion, dropout)
        self.attn_norm   = nn.LayerNorm(hidden)
        self.attn        = nn.MultiheadAttention(hidden, n_heads,
                                                  dropout=dropout,
                                                  batch_first=True)
        self.attn_drop   = nn.Dropout(dropout)
        self.conv        = ConvolutionModule(hidden, conv_kernel, dropout,
                                             causal=causal)
        self.ff2         = FeedForwardModule(hidden, ff_expansion, dropout)
        self.norm        = nn.LayerNorm(hidden)

    def forward(self, x):
        # x: (B, T, hidden) throughout
        x = self.ff1(x)

        # Pre-norm self-attention
        h = self.attn_norm(x)
        if self.causal:
            # Lower-triangular mask: True = "block this position".
            # Frame t can attend to frames 0..t but not t+1..T-1.
            T = x.size(1)
            mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=1)
        else:
            mask = None
        h, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + self.attn_drop(h)

        x = self.conv(x)
        x = self.ff2(x)
        return self.norm(x)


class VocalCoachConformer(nn.Module):
    """Lightweight Conformer vocal coaching model.

    Offline-only: self-attention attends over the full clip, which requires
    the complete sequence before producing outputs. Use for post-session
    analysis where latency is not a constraint.

    Args:
        n_mels:       mel spectrogram input bands (40)
        hidden:       model dimension (64)
        n_layers:     number of Conformer blocks (4)
        n_heads:      attention heads — must divide hidden evenly (4)
        ff_expansion: feed-forward inner dimension multiplier (4)
        conv_kernel:  depthwise conv kernel size in frames; 31 → 310 ms (31)
        dropout:      dropout rate (0.1)
        n_techniques: number of technique classes (5)
    """

    def __init__(self, n_mels=N_MELS, hidden=64, n_layers=4, n_heads=4,
                 ff_expansion=4, conv_kernel=31, dropout=0.1,
                 causal=False, n_techniques=N_TECHNIQUES, quality_head=0,
                 deep_technique_head=False, note_head=False, deep_note_head=False):
        super().__init__()
        assert hidden % n_heads == 0, (
            f"hidden ({hidden}) must be divisible by n_heads ({n_heads})")

        self.hidden = hidden
        self.causal = causal
        self.quality_dims = quality_head

        # Linear projection from mel bands into the model dimension.
        # Unlike TCN which uses Conv1d(k=1), here a simple Linear suffices
        # because the Conformer blocks handle temporal structure via attention.
        self.input_proj = nn.Linear(n_mels, hidden)

        self.blocks = nn.ModuleList([
            ConformerBlock(hidden, n_heads, ff_expansion, conv_kernel,
                           dropout, causal=causal)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden)

        # ── Output heads (multi-task, same as TCN) ──
        self.head_vad   = nn.Linear(hidden, 1)
        self.head_pitch = nn.Linear(hidden, PITCH_BINS)
        if deep_technique_head:
            self.head_technique = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, n_techniques),
            )
        else:
            self.head_technique = nn.Linear(hidden, n_techniques)

        if quality_head > 0:
            self.head_quality = nn.Sequential(
                nn.Linear(hidden, hidden // 4),
                nn.GELU(),
                nn.Linear(hidden // 4, quality_head),
            )

        # ── Optional note segmentation head (Variant 4) ──
        self.has_note_head = note_head
        if note_head:
            if deep_note_head:
                self.head_note_onset  = nn.Sequential(
                    nn.Linear(hidden, hidden // 4), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(hidden // 4, 1))
                self.head_note_offset = nn.Sequential(
                    nn.Linear(hidden, hidden // 4), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(hidden // 4, 1))
            else:
                self.head_note_onset  = nn.Linear(hidden, 1)
                self.head_note_offset = nn.Linear(hidden, 1)

        n = sum(p.numel() for p in self.parameters())
        note_str = (", deep_note_head=True" if (note_head and deep_note_head)
                    else ", note_head=True" if note_head else "")
        print(f"VocalCoachConformer: {n:,} parameters "
              f"(hidden={hidden}, layers={n_layers}, heads={n_heads}, "
              f"causal={causal}{note_str}"
              f"{f', quality_head={quality_head}' if quality_head else ''})")

    def forward(self, mel, return_embeddings=False):
        """Run the model on a batch of mel spectrograms.

        Args:
            mel:               (B, T, 40) — log-mel spectrogram (full clip)
            return_embeddings: if True, also return backbone hidden states (B, T, hidden)

        Returns:
            vad:       (B, T, 1)   — voice activity probability
            pitch:     (B, T, 360) — pitch posteriorgram
            technique: (B, T, N)   — per-technique probability (multi-label)
            quality:   (B, Q)      — clip-level quality scores (Q=quality_head dims), or None
            embeddings (only when return_embeddings=True): (B, T, hidden)
        """
        x = self.input_proj(mel)    # (B, T, hidden)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        vad       = torch.sigmoid(self.head_vad(x))
        pitch     = torch.sigmoid(self.head_pitch(x))
        technique = torch.sigmoid(self.head_technique(x))
        quality   = self.head_quality(x.mean(dim=1)) if self.quality_dims else None
        note_onset  = torch.sigmoid(self.head_note_onset(x))  if self.has_note_head else None
        note_offset = torch.sigmoid(self.head_note_offset(x)) if self.has_note_head else None

        if return_embeddings:
            return vad, pitch, technique, quality, note_onset, note_offset, x
        return vad, pitch, technique, quality, note_onset, note_offset


# ═══════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════

def build_model(arch='tcn', **kwargs):
    """Instantiate a VocalCoach model by architecture name.

    Args:
        arch:     'tcn' or 'conformer'
        **kwargs: forwarded to VocalCoachTCN or VocalCoachConformer

    Returns:
        Model instance with weights randomly initialised.

    Example::

        model = build_model('tcn', hidden=64, n_blocks=4, causal=False)
        model = build_model('conformer', hidden=128, n_layers=6)
    """
    if arch == 'tcn':
        return VocalCoachTCN(**kwargs)
    elif arch == 'conformer':
        return VocalCoachConformer(**kwargs)
    else:
        raise ValueError(
            f"Unknown architecture {arch!r}. Choose 'tcn' or 'conformer'.")


# ═══════════════════════════════════════════════════════════════════════
# Quick test — run this file directly to verify both models work
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    B, T = 2, 500   # 2 clips, 500 frames = 5 seconds at 10 ms hop

    print("=" * 60)
    print("Experiment A — VocalCoachTCN (causal, streaming-safe)")
    print("=" * 60)
    tcn = VocalCoachTCN(hidden=128, n_blocks=8, causal=True)
    x = torch.randn(B, T, N_MELS)
    vad, pitch, technique, _, _, _ = tcn(x)
    print(f"  Input:       {tuple(x.shape)}")
    print(f"  VAD:         {tuple(vad.shape)}  "
          f"range [{vad.min():.3f}, {vad.max():.3f}]")
    print(f"  Pitch:       {tuple(pitch.shape)}")
    print(f"  Technique:   {tuple(technique.shape)}  "
          f"classes: {TECHNIQUE_NAMES}")
    print(f"  Receptive field: {tcn.receptive_field_ms()} ms")

    print()
    print("Experiment A (non-causal variant, offline)")
    tcn_nc = VocalCoachTCN(hidden=128, n_blocks=8, causal=False)
    vad, pitch, technique, _, _, _ = tcn_nc(x)
    print(f"  Technique: {tuple(technique.shape)}  ✓")

    print()
    print("=" * 60)
    print("Experiment B — VocalCoachConformer (causal=False, offline)")
    print("=" * 60)
    conformer = VocalCoachConformer(hidden=64, n_layers=4, n_heads=4,
                                    causal=False)
    vad, pitch, technique, _, _, _ = conformer(x)
    print(f"  Input:       {tuple(x.shape)}")
    print(f"  VAD:         {tuple(vad.shape)}  "
          f"range [{vad.min():.3f}, {vad.max():.3f}]")
    print(f"  Pitch:       {tuple(pitch.shape)}")
    print(f"  Technique:   {tuple(technique.shape)}  ✓")

    print()
    print("=" * 60)
    print("Experiment B — VocalCoachConformer (causal=True, streaming)")
    print("=" * 60)
    conformer_c = VocalCoachConformer(hidden=64, n_layers=4, n_heads=4,
                                      causal=True)
    vad, pitch, technique, _, _, _ = conformer_c(x)
    print(f"  Input:       {tuple(x.shape)}")
    print(f"  VAD:         {tuple(vad.shape)}  "
          f"range [{vad.min():.3f}, {vad.max():.3f}]")
    print(f"  Pitch:       {tuple(pitch.shape)}")
    print(f"  Technique:   {tuple(technique.shape)}  ✓")

    print()
    print("=" * 60)
    print("build_model factory")
    print("=" * 60)
    for arch, kw in [('tcn', dict(hidden=64, n_blocks=4)),
                     ('conformer', dict(hidden=32, n_layers=2, n_heads=4))]:
        m = build_model(arch, **kw)
        x_small = torch.randn(1, 200, N_MELS)
        v, p, t, _, _, _ = m(x_small)
        print(f"  {arch:12s} → vad {tuple(v.shape)}, "
              f"pitch {tuple(p.shape)}, technique {tuple(t.shape)}  ✓")

    print()
    print("=" * 60)
    print("Pitch utility sanity check")
    print("=" * 60)
    f0 = np.array([0.0, 261.63, 440.0, 880.0])   # silence, C4, A4, A5
    bins = f0_to_bin(f0)
    f0_rt = bin_to_f0(np.maximum(bins, 0))
    for hz, b, rt in zip(f0, bins, f0_rt):
        marker = '(unvoiced)' if hz == 0 else f'→ {rt:.1f} Hz roundtrip'
        print(f"  {hz:7.2f} Hz  bin={b:6.1f}  {marker}")
