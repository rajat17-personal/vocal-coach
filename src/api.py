"""
VocalCoach FastAPI Demo Server — Phase 4
=========================================

Runs server-side on the 4080 Super; browser uploads audio and receives a full
coaching report JSON + optional LLM critique.

Endpoints
---------
  POST /analyse                  — full offline analysis
  POST /analyse/critique         — same + LLM natural-language critique
  GET  /health                   — liveness check
  POST /sessions/{song_id}       — save a take + return progress diff
  GET  /sessions/{song_id}       — list all takes for a song
  GET  /sessions                 — list all song names
  DELETE /sessions/{song_id}/{n} — delete take n

Sessions are stored as JSON files under VOCALCOACH_SESSIONS_DIR
(default: ./vocalcoach_sessions/).  Each song gets one file:
    {song_id}.json  →  {"song_id": "...", "takes": [...report dicts...]}

Usage
-----
    uvicorn vocalcoach.api:app --host 0.0.0.0 --port 8000 --reload

Dependencies
------------
    pip install fastapi uvicorn python-multipart soundfile
"""

import io
import json
import os
import re
import tempfile
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import soundfile as sf
    _HAS_SF = True
except ImportError:
    _HAS_SF = False

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False
    class FastAPI:  # type: ignore
        def __init__(self, **kw): pass
        def post(self, *a, **kw): return lambda f: f
        def get(self, *a, **kw): return lambda f: f
        def delete(self, *a, **kw): return lambda f: f
        def mount(self, *a, **kw): pass
        def add_middleware(self, *a, **kw): pass

import torch

from model import bin_to_f0
from features import (
    extract_all, summarise, phrase_aggregate, compute_dtw_distance,
    SR, HOP_LENGTH
)
from coach import build_report, generate_critique, score_report, compare_to_baselines

# ── App init ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VocalCoach Demo API",
    description="Offline singing analysis: pitch, VAD, technique, vibrato, DTW",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static UI ───────────────────────────────────────────────────────────────

_UI_DIR = Path(__file__).parent / "ui"
if _UI_DIR.exists() and _HAS_FASTAPI:
    app.mount("/ui", StaticFiles(directory=str(_UI_DIR), html=True), name="ui")

# ── Sessions (JSON file store) ───────────────────────────────────────────────

_SESSIONS_DIR = Path(os.environ.get("VOCALCOACH_SESSIONS_DIR", "./vocalcoach_sessions"))

# Phrase segmentation (musical, not speech). A phrase = a voiced span; a new phrase
# starts after a silence gap >= VOCALCOACH_PHRASE_GAP_MS. The default 500 ms groups
# a sung line ("I want to dream baby") into one phrase rather than splitting on every
# brief inter-word pause (the old 150 ms speech threshold over-segmented). Phrases
# shorter than VOCALCOACH_PHRASE_MIN_MS are dropped. Both in milliseconds.
_PHRASE_GAP_MS = float(os.environ.get("VOCALCOACH_PHRASE_GAP_MS", "500"))
_PHRASE_MIN_MS = float(os.environ.get("VOCALCOACH_PHRASE_MIN_MS", "300"))

# Named phrasing presets the UI exposes per upload. "default" uses the env values.
# Ballads breathe slowly between long lines → larger gap; up-tempo/staccato songs
# have short rapid phrases → smaller gap. (gap_ms, min_ms)
PHRASE_MODES = {
    "default": (_PHRASE_GAP_MS, _PHRASE_MIN_MS),
    "ballad":  (800.0, 400.0),   # slow: long lines, long breaths between them
    "uptempo": (300.0, 200.0),   # fast: short rapid phrases
}


def _sessions_dir() -> Path:
    d = _SESSIONS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _slug(song_id: str) -> str:
    """Safe filename slug from a song name."""
    return re.sub(r"[^\w\-]", "_", song_id.strip())[:80]


def _session_path(song_id: str) -> Path:
    return _sessions_dir() / f"{_slug(song_id)}.json"


def _load_session(song_id: str) -> dict:
    p = _session_path(song_id)
    if p.exists():
        return json.loads(p.read_text())
    return {"song_id": song_id, "takes": []}


def _save_session(data: dict) -> None:
    p = _session_path(data["song_id"])
    p.write_text(json.dumps(data, indent=2, default=str))


def _diff_takes(prev: dict, curr: dict) -> dict:
    """Return a human-readable delta between two coaching reports."""
    out = {}

    def _safe(report, *path):
        node = report
        for k in path:
            if not isinstance(node, dict):
                return None
            node = node.get(k)
        if isinstance(node, float) and (node != node):  # nan
            return None
        return node

    METRICS = [
        ("Overall score",       ["coaching", "overall_score"],          True,  None),
        ("Pitch stability",     ["pitch", "f0_stability_std_hz"],       False, "¢ std"),
        ("Vibrato coverage",    ["vibrato", "phrase_fraction"],         True,  "%"),
        ("Vibrato rate",        ["vibrato", "rate_hz_mean"],            None,  "Hz"),
        ("DTW deviation",       ["reference_comparison", "mean_deviation_cents"], False, "¢"),
        ("MOS quality",         ["mos", "score"],                       True,  "/5"),
    ]
    for label, path, higher_better, unit in METRICS:
        pv = _safe(prev, *path)
        cv = _safe(curr, *path)
        if pv is None or cv is None:
            continue
        if label == "Pitch stability":
            # convert Hz std → cents (approximate for small deviations)
            f0 = _safe(curr, "pitch", "f0_mean_hz") or 220.0
            import math
            pv = round(abs(1200 * math.log2((f0 + pv) / f0)), 1)
            cv = round(abs(1200 * math.log2((f0 + cv) / f0)), 1)
        if label == "Vibrato coverage":
            pv = round(pv * 100, 1)
            cv = round(cv * 100, 1)
        delta = round(cv - pv, 2) if isinstance(cv, (int, float)) else None
        if delta is None:
            continue
        direction = None
        if higher_better is True:
            direction = "improved" if delta > 0 else ("regressed" if delta < 0 else "unchanged")
        elif higher_better is False:
            direction = "improved" if delta < 0 else ("regressed" if delta > 0 else "unchanged")
        else:
            direction = "changed"
        out[label] = {
            "previous": pv,
            "current": cv,
            "delta": delta,
            "direction": direction,
            "unit": unit or "",
        }
    return out


# ── Model singleton ─────────────────────────────────────────────────────────

_model = None
_device = None
_args = None  # argparse namespace used during training (for arch/seq_len)

TECHNIQUE_NAMES = ["vibrato", "breathy", "falsetto", "belt", "straight"]

# Optional second technique specialist for two-head fusion. When set, the primary
# checkpoint (VOCALCOACH_CHECKPOINT) is the VocalSet specialist — its backbone runs
# once for pitch/VAD/note/technique — and this GTSinger specialist runs as a +1
# pass to supply falsetto (and a second opinion on vibrato/breathy). See
# vocalcoach/technique_fusion.py for the union/trained fusion rules.
_gt_model = None        # GTSinger specialist (or None → single-head, no fusion)
_fusion_head = None     # trained fusion MLP (or None → union ensemble)
_fusion_mode = "union"

# Optional quality model — a checkpoint trained with our quality head (variant 2 =
# 9-dim expert scores, or variant 3 = scalar). Loaded as a +1 pass; replaces the
# external SingMOS dependency with our own model's quality output. Set
# VOCALCOACH_QUALITY_CHECKPOINT to enable.
_quality_model = None
_quality_norm = None      # {'ccmusic_mean','ccmusic_std',...} for de-normalising
_quality_dims = 0

# Optional note model — a checkpoint trained with the note head (onset/offset).
# Loaded as a +1 pass; supplies real note segmentation (the demo otherwise derives
# notes from f0/phrases). Its own VAD is unused (the primary provides VAD), so it
# may be a backbone-finetune optimised purely for note F1. Set VOCALCOACH_NOTE_CHECKPOINT.
_note_model = None
CCMUSIC_DIMS = ['pitch', 'rhythm', 'vocal_range', 'timbre', 'pronunciation',
                'vibrato', 'dynamic', 'breath_control', 'overall_performance']


def _build_aux_model(ckpt):
    """Build a +1 auxiliary model (GTSinger / quality / note) with the FULL saved
    architecture. Prefer the checkpoint's stored model_kwargs (has n_attn_layers,
    n_heads, head flags, etc.) — partial reconstruction from the state_dict alone
    drops n_attn_layers, leaving the attention layers RANDOM and the model broken.
    Falls back to inferring from state_dict + args if model_kwargs is absent."""
    from vocalcoach.model import VocalCoachTCN
    sd = ckpt["state_dict"]
    kw = dict(ckpt.get("model_kwargs", {}) or {})
    kw.pop("causal", None)  # passed explicitly below
    if not kw:  # legacy checkpoints without model_kwargs — infer
        a = ckpt.get("args", {}); a = vars(a) if not isinstance(a, dict) else a
        kw = dict(hidden=int(sd["input_proj.weight"].shape[0]),
                  n_blocks=max(int(k.split(".")[1]) for k in sd if k.startswith("blocks.")) + 1,
                  n_attn_layers=a.get("n_attn_layers", 0), n_heads=a.get("n_heads", 4),
                  deep_technique_head=a.get("deep_technique_head", False),
                  deep_note_head=a.get("deep_note_head", False))
    # Ensure heads present in the weights are constructed (strict load needs them).
    if any(k.startswith("head_note") for k in sd):
        kw["note_head"] = True
    if any(k.startswith("head_quality") for k in sd):
        qw = [k for k in sd if k.startswith("head_quality") and "weight" in k][-1]
        kw["quality_head"] = int(sd[qw].shape[0])
    causal = ckpt.get("model_kwargs", {}).get("causal", False)
    m = VocalCoachTCN(causal=causal, **kw)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    # Only HEAD params we didn't construct should be missing; backbone must match.
    bad = [k for k in missing if k.startswith(("input_proj", "blocks", "attn_layers", "norm"))]
    if bad:
        raise RuntimeError(f"aux model backbone mismatch — missing {bad[:3]} "
                           f"(model_kwargs={ckpt.get('model_kwargs')})")
    return m


def _load_model():
    global _model, _device, _args, _gt_model, _fusion_head, _fusion_mode
    global _quality_model, _quality_norm, _quality_dims, _note_model

    if _model is not None:
        return  # already loaded

    checkpoint_path = os.environ.get("VOCALCOACH_CHECKPOINT", "")
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise RuntimeError(
            "Set VOCALCOACH_CHECKPOINT env var to the path of best_loss.pth"
        )

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from vocalcoach.model import VocalCoachTCN, VocalCoachConformer
    ckpt = torch.load(checkpoint_path, map_location=_device)
    saved_args = ckpt.get("args", {})

    import argparse
    _args = argparse.Namespace(**saved_args) if isinstance(saved_args, dict) else saved_args

    # hidden / n_layers may be None in probe-mode checkpoints (inherited from
    # a resumed run). Derive them directly from the state dict instead.
    sd = ckpt["state_dict"]
    hidden_from_sd = int(sd["input_proj.weight"].shape[0])
    n_layers_from_sd = max(
        int(k.split(".")[1]) for k in sd if k.startswith("blocks.")
    ) + 1

    arch = getattr(_args, "arch", "conformer")
    if arch == "tcn":
        model = VocalCoachTCN(
            hidden=hidden_from_sd,
            n_blocks=n_layers_from_sd,
            causal=getattr(_args, "causal", False),
            deep_technique_head=getattr(_args, "deep_technique_head", False),
        )
    else:
        model = VocalCoachConformer(
            hidden=hidden_from_sd,
            n_layers=n_layers_from_sd,
            causal=getattr(_args, "causal", False),
            deep_technique_head=getattr(_args, "deep_technique_head", False),
        )
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(_device).eval()
    _model = model

    # ── Optional two-head technique fusion ───────────────────────────────────
    # Primary (_model) = VocalSet specialist: its single forward already gives
    # pitch/VAD/note + the vibrato/breathy/belt/straight technique head. If a
    # GTSinger specialist is provided, load it as a +1 pass to supply falsetto.
    # Default fusion = "union" (no extra checkpoint); set VOCALCOACH_FUSION_HEAD
    # to a trained fusion MLP to use "trained" mode. (Eval showed union ≥ trained,
    # so union is the recommended default.)
    gt_path = os.environ.get("VOCALCOACH_GTSINGER_CHECKPOINT", "")
    if gt_path and os.path.exists(gt_path):
        gt_ckpt = torch.load(gt_path, map_location=_device)
        _gt_model = _build_aux_model(gt_ckpt)
        _gt_model.to(_device).eval()
        fh_path = os.environ.get("VOCALCOACH_FUSION_HEAD", "")
        if fh_path and os.path.exists(fh_path):
            import torch.nn as nn
            _fusion_head = nn.Sequential(
                nn.Linear(2 * len(TECHNIQUE_NAMES), 32), nn.GELU(),
                nn.Linear(32, len(TECHNIQUE_NAMES)))
            _fusion_head.load_state_dict(
                torch.load(fh_path, map_location=_device)["state_dict"])
            _fusion_head.to(_device).eval()
            _fusion_mode = "trained"
        print(f"[VocalCoach API] Two-head technique fusion enabled (mode={_fusion_mode})")

    # ── Optional quality model (our quality head, replaces external SingMOS) ──
    q_path = os.environ.get("VOCALCOACH_QUALITY_CHECKPOINT", "")
    if q_path and os.path.exists(q_path):
        qckpt = torch.load(q_path, map_location=_device)
        qsd = qckpt["state_dict"]
        qw = [k for k in qsd if k.startswith("head_quality") and "weight" in k][-1]
        _quality_dims = int(qsd[qw].shape[0])
        _quality_model = _build_aux_model(qckpt)
        _quality_model.to(_device).eval()
        _quality_norm = qckpt.get("quality_norm", {}) or {}
        print(f"[VocalCoach API] Quality head enabled "
              f"({_quality_dims}-dim, our model — SingMOS not used)")

    # ── Optional note model (note onset/offset head, +1 pass) ────────────────
    n_path = os.environ.get("VOCALCOACH_NOTE_CHECKPOINT", "")
    if n_path and os.path.exists(n_path):
        nckpt = torch.load(n_path, map_location=_device)
        _note_model = _build_aux_model(nckpt)
        _note_model.to(_device).eval()
        print("[VocalCoach API] Note head enabled (real onset/offset + note-with-pitch)")


# Per-class fusion routing — index aligns with TECHNIQUE_NAMES.
# vibrato/breathy: both heads learned them → average. belt/straight: VocalSet
# only. falsetto: GTSinger only.
_VS_CLASSES = {"vibrato", "breathy", "belt", "straight"}
_GT_CLASSES = {"vibrato", "breathy", "falsetto"}


def _fuse_technique(vs_probs, gt_probs):
    """Fuse the two specialists' (T, 5) technique probs into one (T, 5).
    Union routing by default; trained MLP if _fusion_head is loaded."""
    import numpy as _np
    if _fusion_mode == "trained" and _fusion_head is not None:
        x = torch.from_numpy(_np.concatenate([vs_probs, gt_probs], axis=-1)).to(_device)
        with torch.no_grad():
            return torch.sigmoid(_fusion_head(x.float())).cpu().numpy()
    both = _np.array([n in _VS_CLASSES and n in _GT_CLASSES for n in TECHNIQUE_NAMES])
    vs_only = _np.array([n in _VS_CLASSES and n not in _GT_CLASSES for n in TECHNIQUE_NAMES])
    gt_only = _np.array([n in _GT_CLASSES and n not in _VS_CLASSES for n in TECHNIQUE_NAMES])
    fused = _np.zeros_like(vs_probs)
    fused[:, both] = 0.5 * (vs_probs[:, both] + gt_probs[:, both])
    fused[:, vs_only] = vs_probs[:, vs_only]
    fused[:, gt_only] = gt_probs[:, gt_only]
    return fused


def _score_quality(mel_t):
    """Run our quality head on the mel and return a dict:
      {'overall_100': int, 'dimensions': {name: 1-5}, 'scale': '0-100'} or None.
    De-normalises the z-scored head output via quality_norm, then maps the
    overall to 0-100 (1-5 → (x-1)/4 × 100). Replaces external SingMOS."""
    import numpy as _np
    if _quality_model is None:
        return None
    with torch.no_grad():
        q = _quality_model(mel_t)[3]   # (B, dims)
    if q is None:
        return None
    raw = q.squeeze(0).cpu().numpy()    # (dims,)
    if _quality_dims == 9 and _quality_norm.get("ccmusic_mean") is not None:
        mean = _np.asarray(_quality_norm["ccmusic_mean"], dtype=_np.float32)
        std  = _np.asarray(_quality_norm["ccmusic_std"],  dtype=_np.float32)
        vals = raw * std + mean         # de-normalise per dimension → ~1-5
        dims = {CCMUSIC_DIMS[i]: round(float(vals[i]), 2) for i in range(min(9, len(vals)))}
        overall_15 = float(dims.get("overall_performance",
                                    float(_np.mean(vals))))
    else:  # scalar variant
        mean = _quality_norm.get("mse_mean", 0.0); std = _quality_norm.get("mse_std", 1.0)
        overall_15 = float(raw.reshape(-1)[0] * std + mean)
        dims = {}
    overall_15 = max(1.0, min(5.0, overall_15))
    return {
        "overall_100": round((overall_15 - 1.0) / 4.0 * 100),
        "overall_5": round(overall_15, 2),
        "dimensions": dims,
        "scale": "0-100 (from quality head)",
    }


def _detect_notes(mel_t, pitch_post):
    """Run the note head (+1 pass); return a list of detected notes:
      [{"onset_s","offset_s","midi","note"}].
    Onsets/offsets via peak-picking the head's sigmoid outputs; each note's pitch
    is the median argmax of the pitch posteriorgram over [onset, offset]. Reuses
    the same peak-pick / bin→MIDI helpers as scripts/evalNoteHead.py."""
    import numpy as _np, os as _os, sys as _sys
    _sd = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "scripts")
    if _sd not in _sys.path:
        _sys.path.insert(0, _sd)
    try:
        from evalNoteHead import _pick_peaks, _bin_to_midi
    except Exception:
        return []
    with torch.no_grad():
        out = _note_model(mel_t)
    on, off = out[4], out[5]
    if on is None or off is None:
        return []
    p_on = on.squeeze(0).squeeze(-1).cpu().numpy()
    p_off = off.squeeze(0).squeeze(-1).cpu().numpy()
    pk_on = _pick_peaks(p_on, 0.5, 5)
    pk_off = _pick_peaks(p_off, 0.5, 5)
    hop_s = HOP_LENGTH / SR
    notes = []
    off_sorted = _np.sort(pk_off)
    for k, o in enumerate(_np.sort(pk_on)):
        # offset = first detected offset after this onset, else next onset, else clip end
        later = off_sorted[off_sorted > o]
        end = int(later[0]) if len(later) else (
            int(pk_on[pk_on > o].min()) if _np.any(pk_on > o) else len(p_on))
        span = pitch_post[int(o):max(int(o) + 1, end)]
        midi = float(_bin_to_midi(_np.median(span.argmax(axis=1)))) if len(span) else float("nan")
        m = int(round(midi)) if midi == midi else None
        names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note = f"{names[m % 12]}{m // 12 - 1}" if m is not None else "—"
        notes.append({"onset_s": round(float(o) * hop_s, 3),
                      "offset_s": round(float(end) * hop_s, 3),
                      "midi": m, "note": note})

    # ── Cleanup: merge over-segmented held notes ──────────────────────────────
    # The onset detector re-fires within a sustained note (vibrato/wobble crosses
    # the threshold repeatedly), splitting one perceptual note into G3 G3 G3…
    # Merge adjacent notes of the SAME MIDI separated by a small gap (<120 ms),
    # and drop ultra-short fragments (<60 ms). Genuine pitch transitions (different
    # MIDI) are preserved — so real note variation still shows, just not the repeats.
    MERGE_GAP_S = 0.12
    MIN_DUR_S = 0.06
    merged = []
    for nt in notes:
        if (merged and nt["midi"] is not None
                and merged[-1]["midi"] == nt["midi"]
                and nt["onset_s"] - merged[-1]["offset_s"] <= MERGE_GAP_S):
            merged[-1]["offset_s"] = nt["offset_s"]      # extend the held note
        else:
            merged.append(dict(nt))
    cleaned = [nt for nt in merged
               if nt["midi"] is None or (nt["offset_s"] - nt["onset_s"]) >= MIN_DUR_S]
    return cleaned


@app.on_event("startup")
async def startup():
    try:
        _load_model()
        print(f"[VocalCoach API] Model loaded on {_device}")
    except Exception as e:
        print(f"[VocalCoach API] WARNING: model not loaded at startup: {e}")


# ── Audio helpers ───────────────────────────────────────────────────────────

def _load_audio(upload: bytes, target_sr: int = SR) -> np.ndarray:
    """Load uploaded audio bytes, resample to target_sr, return mono float32."""
    if not _HAS_SF:
        raise RuntimeError("pip install soundfile to handle audio uploads")

    with io.BytesIO(upload) as buf:
        y, sr = sf.read(buf, dtype="float32", always_2d=False)

    if y.ndim > 1:
        y = y.mean(axis=1)

    if sr != target_sr:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    return y.astype(np.float32)


def _run_model(y: np.ndarray) -> dict:
    """Run VocalCoach model on a waveform, return per-frame numpy outputs."""
    _load_model()

    import librosa
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=512, hop_length=HOP_LENGTH,
        n_mels=40, fmin=50, fmax=8000,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Shape: (1, T, n_mels) — batch of 1
    mel_t = torch.tensor(mel_db.T[None], dtype=torch.float32).to(_device)

    with torch.no_grad():
        out_vad, out_pitch, out_technique, _, _, _ = _model(mel_t)

    # (B, T, 1) → (T,)
    vad = out_vad.squeeze(0).squeeze(-1).cpu().numpy()
    # (B, T, 360) → (T, 360)
    pitch_post = out_pitch.squeeze(0).cpu().numpy()

    # Voicing: use pitch-peak confidence as proxy (VAD sigmoid may be
    # near-zero in probe-mode checkpoints whose VAD head wasn't retrained).
    # A frame is voiced when its peak bin probability exceeds 30% of the
    # clip maximum — equivalent to the per-class threshold sweep logic.
    pitch_confidence = pitch_post.max(axis=-1)           # (T,)
    voicing_thresh = max(0.05, pitch_confidence.max() * 0.30)
    voiced_mask = pitch_confidence > voicing_thresh

    # F0 from posteriorgram argmax → Hz via model's own bin_to_f0()
    pitch_bin = np.argmax(pitch_post, axis=-1).astype(float)  # (T,)
    f0_hz = np.where(voiced_mask, bin_to_f0(pitch_bin), 0.0).astype(np.float32)

    # Calibrate VAD: three failure modes seen across checkpoints:
    #   1. VAD near-zero everywhere (head not trained, stage1) → fall back to
    #      pitch confidence, but use a stricter absolute threshold so only
    #      genuinely confident pitch frames count as voiced.
    #   2. VAD saturated high everywhere (stage2 overfit) → the model thinks
    #      the whole clip is voiced; fall back to pitch confidence mask.
    #   3. VAD fires normally → use as-is, clipped to [0,1].
    vad_range = float(vad.max() - vad.min())
    pc_thresh = max(0.15, float(pitch_confidence.max()) * 0.35)
    if vad.max() < 0.1:
        # Case 1: dead VAD head (stage1) — use pitch confidence
        vad = (pitch_confidence > pc_thresh).astype(np.float32)
    elif vad_range < 0.05 or vad.min() > 0.6:
        # Case 2: saturated — no silence/voice discrimination at all.
        # vad_range < 0.05: flat (all same value).
        # vad.min() > 0.6: shifted high (stage2 overfit, min=0.76).
        # Both cases: fall back to pitch confidence.
        vad = (pitch_confidence > pc_thresh).astype(np.float32)
    # Case 3: VAD has meaningful dynamic range → use as-is

    result = {"f0_hz": f0_hz, "vad": vad, "mel_db": mel_db, "pitch_post": pitch_post}

    if out_technique is not None:
        # (B, T, N) → (T, N) — already sigmoid'd
        vs_probs = out_technique.squeeze(0).cpu().numpy()
        if _gt_model is not None:
            # +1 pass on the GTSinger specialist, then fuse (union/trained).
            with torch.no_grad():
                gt_out = _gt_model(mel_t)[2]
            gt_probs = gt_out.squeeze(0).cpu().numpy()
            # The two heads may have different class counts only if mis-trained;
            # both use the canonical 5-class TECHNIQUE_NAMES here.
            result["technique_probs"] = _fuse_technique(vs_probs, gt_probs)
        else:
            result["technique_probs"] = vs_probs

    # Quality score from our quality head (+1 pass), if a quality model is loaded.
    if _quality_model is not None:
        result["quality"] = _score_quality(mel_t)

    # Real note onset/offset + note pitch from the note head (+1 pass).
    if _note_model is not None:
        result["notes"] = _detect_notes(mel_t, pitch_post)

    return result


# ── Core analysis pipeline ───────────────────────────────────────────────────

def _analyse(y: np.ndarray, y_ref: Optional[np.ndarray] = None,
             return_arrays: bool = False,
             phrase_gap_ms: Optional[float] = None,
             phrase_min_ms: Optional[float] = None) -> dict:
    """Full pipeline: model → features → phrase aggregation → report.

    phrase_gap_ms / phrase_min_ms override the phrase-segmentation thresholds for
    this request (e.g. from a UI phrasing-mode selector); fall back to env defaults.
    """
    model_out = _run_model(y)
    f0_hz = model_out["f0_hz"]
    vad   = model_out["vad"]
    tech_probs = model_out.get("technique_probs")

    feats = extract_all(y, sr=SR, f0_hz=f0_hz, vad=vad)
    summary = summarise(feats, f0_hz=f0_hz)

    rms_db = feats["rms_db"]
    # ms → frames at the 10 ms hop. Per-request override (UI mode) else env default.
    _gap_ms = phrase_gap_ms if phrase_gap_ms is not None else _PHRASE_GAP_MS
    _min_ms = phrase_min_ms if phrase_min_ms is not None else _PHRASE_MIN_MS
    _gap_frames = max(1, round(_gap_ms / (HOP_LENGTH / SR * 1000)))
    _min_frames = max(1, round(_min_ms / (HOP_LENGTH / SR * 1000)))
    phrases = phrase_aggregate(
        f0_hz, vad,
        technique_probs=tech_probs,
        rms_db=rms_db,
        sr=SR, hop_length=HOP_LENGTH,
        min_gap_frames=_gap_frames,
        min_phrase_frames=_min_frames,
    )

    # Clip-level technique means (voiced frames only)
    technique_clip = None
    if tech_probs is not None:
        voiced_mask = vad > 0.5
        if voiced_mask.any():
            voiced_tech = tech_probs[voiced_mask]
            technique_clip = {
                name: float(np.mean(voiced_tech[:, k]))
                for k, name in enumerate(TECHNIQUE_NAMES)
                if k < voiced_tech.shape[1]
            }

    # DTW vs reference
    dtw_result = None
    if y_ref is not None:
        ref_out = _run_model(y_ref)
        dtw_result = compute_dtw_distance(f0_hz, ref_out["f0_hz"])

    clip_dur = len(y) / SR
    report = build_report(phrases, summary, technique_clip, dtw_result, clip_dur)
    report["_feats_summary"] = summary

    # Quality — from OUR quality head (replaces external SingMOS). The quality
    # model runs as a +1 pass in _run_model; here we surface its 0-100 score and
    # (for variant-2) the 9 expert dimensions.
    quality = model_out.get("quality")
    if quality is not None:
        report["quality"] = quality
        # Keep a 'mos'-shaped block for backward compat with coach.py / UI that
        # read report['mos'], but source it from our head (overall_5), not SingMOS.
        report["mos"] = {"score": quality.get("overall_5"),
                         "grade": None, "scale": "0-100 (quality head)"}

    # Real detected notes from the note head (+1 pass), if a note model is loaded.
    notes = model_out.get("notes")
    if notes is not None:
        report["notes"] = notes

    # D5 — PopBuTFy population context (graceful: {} if baselines not yet built)
    baselines_path = os.path.join(os.path.dirname(__file__), "..", "data", "popbutfy_baselines.json")
    population_context = compare_to_baselines(report, baselines_path=baselines_path)
    if population_context:
        report["population_context"] = population_context

    if return_arrays:
        report["_arrays"] = {
            "mel_db":     model_out["mel_db"],      # (40, T) float32
            "pitch_post": model_out["pitch_post"],  # (T, 360) float32
            "vad":        vad,                      # (T,) float32
            "f0_hz":      f0_hz,                    # (T,) float32
        }

    return report


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    model_loaded = _model is not None
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "device": str(_device) if _device else None,
        "cuda_available": torch.cuda.is_available(),
    }


def _sanitize(obj):
    """Recursively replace nan/inf with None so JSONResponse doesn't choke."""
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


@app.post("/analyse")
async def analyse(
    audio: UploadFile = File(..., description="Singing audio file (WAV/MP3/FLAC)"),
    reference: Optional[UploadFile] = File(None, description="Reference audio for DTW"),
    phrasing: str = Form("default", description="Phrasing mode: 'default', 'ballad' (slow), or 'uptempo' (fast)"),
):
    """Analyse a singing clip and return structured coaching metrics."""
    try:
        y = _load_audio(await audio.read())
        y_ref = None
        if reference is not None:
            y_ref = _load_audio(await reference.read())

        gap_ms, min_ms = PHRASE_MODES.get(phrasing, PHRASE_MODES["default"])
        report = _analyse(y, y_ref, phrase_gap_ms=gap_ms, phrase_min_ms=min_ms)
        report["coaching"] = score_report(report)
        return JSONResponse(content=_sanitize(report))

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.post("/analyse/arrays")
async def analyse_arrays(
    audio: UploadFile = File(..., description="Singing audio file (WAV/MP3/FLAC)"),
):
    """Return mel spectrogram + pitch posteriorgram + VAD + F0 as base64-encoded float32 arrays.

    Response JSON:
        mel_db:     base64 float32, shape (40, T)  — log-mel, row-major
        pitch_post: base64 float32, shape (T, 360) — pitch posteriorgram, row-major
        vad:        base64 float32, shape (T,)
        f0_hz:      base64 float32, shape (T,)
        T:          int — number of time frames
    """
    import base64
    try:
        y = _load_audio(await audio.read())
        report = _analyse(y, return_arrays=True)
        arrays = report.pop("_arrays", {})

        def enc(arr):
            return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()

        mel = arrays["mel_db"]    # (40, T)
        pit = arrays["pitch_post"]  # (T, 360)
        T   = mel.shape[1]

        return JSONResponse(content={
            "T":          T,
            "mel_db":     enc(mel),
            "pitch_post": enc(pit),
            "vad":        enc(arrays["vad"]),
            "f0_hz":      enc(arrays["f0_hz"]),
        })
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.post("/analyse/critique")
async def analyse_with_critique(
    audio: UploadFile = File(..., description="Singing audio file"),
    reference: Optional[UploadFile] = File(None, description="Reference audio for DTW"),
    mode: str = Form("expert", description="Feedback mode: 'expert' or 'beginner'"),
):
    """Analyse + generate LLM natural-language coaching critique."""
    if mode not in ("expert", "beginner"):
        raise HTTPException(status_code=422, detail="mode must be 'expert' or 'beginner'")

    try:
        y = _load_audio(await audio.read())
        y_ref = None
        if reference is not None:
            y_ref = _load_audio(await reference.read())

        report = _analyse(y, y_ref)
        report["coaching"] = score_report(report)
        critique = generate_critique(report, mode=mode)
        report["critique"] = {"mode": mode, "text": critique}
        return JSONResponse(content=_sanitize(report))

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


# ── Sessions endpoints ────────────────────────────────────────────────────

@app.get("/sessions")
async def list_songs():
    """List all song IDs that have saved sessions."""
    songs = []
    for p in sorted(_sessions_dir().glob("*.json")):
        try:
            data = json.loads(p.read_text())
            songs.append({
                "song_id": data.get("song_id", p.stem),
                "n_takes": len(data.get("takes", [])),
                "last_score": (data["takes"][-1].get("coaching", {}) or {}).get("overall_score")
                              if data.get("takes") else None,
            })
        except Exception:
            pass
    return JSONResponse(content=songs)


@app.get("/sessions/{song_id}")
async def get_session(song_id: str):
    """Return all takes for a song."""
    data = _load_session(song_id)
    return JSONResponse(content=_sanitize(data))


@app.post("/sessions/{song_id}")
async def save_take(
    song_id: str,
    audio: UploadFile = File(..., description="Singing audio (WAV/MP3/FLAC)"),
    reference: Optional[UploadFile] = File(None, description="Reference audio for DTW"),
    phrasing: str = Form("default", description="Phrasing mode: 'default', 'ballad', or 'uptempo'"),
):
    """Analyse audio, save as a new take for song_id, return report + progress diff."""
    try:
        y = _load_audio(await audio.read())
        y_ref = _load_audio(await reference.read()) if reference else None

        gap_ms, min_ms = PHRASE_MODES.get(phrasing, PHRASE_MODES["default"])
        report = _analyse(y, y_ref, phrase_gap_ms=gap_ms, phrase_min_ms=min_ms)
        report["coaching"] = score_report(report)

        session = _load_session(song_id)
        session["song_id"] = song_id

        prev_take = session["takes"][-1] if session["takes"] else None
        take_number = len(session["takes"]) + 1
        report["take_number"] = take_number

        progress_diff = _diff_takes(prev_take, report) if prev_take else {}

        session["takes"].append(report)
        _save_session(session)

        return JSONResponse(content=_sanitize({
            "song_id": song_id,
            "take_number": take_number,
            "report": report,
            "progress_diff": progress_diff,
            "all_scores": [
                t.get("coaching", {}).get("overall_score") for t in session["takes"]
            ],
        }))

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.delete("/sessions/{song_id}/{take_n}")
async def delete_take(song_id: str, take_n: int):
    """Delete take number take_n (1-indexed) from a session."""
    session = _load_session(song_id)
    takes = session.get("takes", [])
    if take_n < 1 or take_n > len(takes):
        raise HTTPException(status_code=404, detail=f"Take {take_n} not found")
    takes.pop(take_n - 1)
    # Renumber
    for i, t in enumerate(takes):
        t["take_number"] = i + 1
    session["takes"] = takes
    _save_session(session)
    return {"deleted": take_n, "remaining": len(takes)}


# ── UI redirect ────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    ui_path = _UI_DIR / "index.html"
    if ui_path.exists():
        return FileResponse(str(ui_path))
    return {"message": "VocalCoach API running. UI not found at vocalcoach/ui/index.html"}


# ── Direct launch ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="", help="Path to best_loss.pth")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    cli = parser.parse_args()

    if cli.checkpoint:
        os.environ["VOCALCOACH_CHECKPOINT"] = cli.checkpoint

    uvicorn.run("vocalcoach.api:app", host=cli.host, port=cli.port, reload=cli.reload)
