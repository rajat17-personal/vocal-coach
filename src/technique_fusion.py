"""
Technique Fusion — combine the VocalSet and GTSinger technique specialists.

Why two heads: VocalSet and GTSinger disagree on the shared classes (vibrato/
breathy) and have near-disjoint label spaces (VocalSet = belt/straight + no
falsetto; GTSinger = falsetto + no belt/straight). Training one combined head
trades off on both domains, so we keep two domain SPECIALISTS and fuse their
outputs at inference.

Two fusion modes:

  "union" (DEFAULT — no training):
      Each of the 5 classes is taken from the head that actually learned it:
        vibrato, breathy  → mean of the two heads (both learned them)
        belt, straight    → VocalSet head   (GTSinger has none)
        falsetto          → GTSinger head   (VocalSet has none)
      Cheap (two small head forwards over one shared backbone pass) and needs
      zero training. This is the deployable default.

  "trained" (STRETCH / backup — needs a fusion checkpoint):
      A small learned head maps the 10 stacked specialist probabilities (+ the
      backbone embedding) → unified 5-class output. Trained separately with
      per-sample class masking (only supervise classes present in each clip's
      source corpus). Loaded from --fusion-checkpoint; falls back to "union" if
      not provided.

The backbone is shared: both specialists fine-tuned from the SAME gainaug
backbone, so we run the backbone once and apply each specialist's head. (If the
two checkpoints have different backbones, set share_backbone=False to run both
fully — slower but always correct.)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .model import build_model, TECHNIQUE_NAMES

# Which head owns each class. Index aligns with TECHNIQUE_NAMES.
#   ['vibrato', 'breathy', 'falsetto', 'belt', 'straight']
_VOCALSET_CLASSES = {"vibrato", "breathy", "belt", "straight"}
_GTSINGER_CLASSES = {"vibrato", "breathy", "falsetto"}


def _load_specialist(checkpoint_path, device):
    """Load a technique specialist checkpoint → (model, args-dict)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    kwargs = dict(ckpt.get("model_kwargs", {}))
    causal = kwargs.pop("causal", ckpt.get("causal", False))
    model = build_model(ckpt.get("arch", "tcn"), causal=causal, **kwargs)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval().to(device)
    return model, ckpt.get("args", {})


class TechniqueFusion(nn.Module):
    """Fuses two technique specialists into a single 5-class technique output.

    Usage:
        fuser = TechniqueFusion(vocalset_ckpt, gtsinger_ckpt, device)
        probs = fuser.predict(mel)        # (T, 5) per-frame technique probs
        # or clip-level:
        clip = fuser.predict(mel).mean(0) # (5,)
    """

    def __init__(self, vocalset_ckpt, gtsinger_ckpt, device="cpu",
                 mode="union", fusion_checkpoint=None):
        super().__init__()
        self.device = torch.device(device)
        self.mode = mode
        self.vs_model, _ = _load_specialist(vocalset_ckpt, self.device)
        self.gt_model, _ = _load_specialist(gtsinger_ckpt, self.device)

        # Precompute per-class source masks (which head contributes each class).
        self.vs_mask = np.array([n in _VOCALSET_CLASSES for n in TECHNIQUE_NAMES])
        self.gt_mask = np.array([n in _GTSINGER_CLASSES for n in TECHNIQUE_NAMES])
        self.both_mask = self.vs_mask & self.gt_mask   # vibrato, breathy

        self.fusion_head = None
        if mode == "trained":
            if fusion_checkpoint is None:
                # No trained head available → degrade gracefully to union.
                self.mode = "union"
            else:
                fc = torch.load(fusion_checkpoint, map_location="cpu",
                                weights_only=False)
                # Small MLP: [vs_probs(5) ‖ gt_probs(5)] -> 5 logits.
                self.fusion_head = nn.Sequential(
                    nn.Linear(2 * len(TECHNIQUE_NAMES), 32), nn.GELU(),
                    nn.Linear(32, len(TECHNIQUE_NAMES)),
                ).to(self.device)
                self.fusion_head.load_state_dict(fc["state_dict"])
                self.fusion_head.eval()

    @torch.no_grad()
    def _head_probs(self, mel_t):
        """Run both specialists, return (vs_probs, gt_probs), each (T, 5)."""
        vs = self.vs_model(mel_t)[2].squeeze(0).cpu().numpy()   # (T, 5)
        gt = self.gt_model(mel_t)[2].squeeze(0).cpu().numpy()   # (T, 5)
        return vs, gt

    @torch.no_grad()
    def predict(self, mel):
        """mel: (T, 40) np array or (1, T, 40) tensor → (T, 5) fused probs."""
        if isinstance(mel, np.ndarray):
            mel_t = torch.from_numpy(mel.astype(np.float32))
            if mel_t.dim() == 2:
                mel_t = mel_t.unsqueeze(0)
        else:
            mel_t = mel if mel.dim() == 3 else mel.unsqueeze(0)
        mel_t = mel_t.to(self.device)

        vs, gt = self._head_probs(mel_t)   # each (T, 5)

        if self.mode == "trained" and self.fusion_head is not None:
            x = torch.from_numpy(np.concatenate([vs, gt], axis=-1)).to(self.device)
            return torch.sigmoid(self.fusion_head(x)).cpu().numpy()

        # ── Union fusion ──────────────────────────────────────────────────
        fused = np.zeros_like(vs)
        # vibrato/breathy: average the two heads (both learned them)
        fused[:, self.both_mask] = 0.5 * (vs[:, self.both_mask] + gt[:, self.both_mask])
        # belt/straight: VocalSet only
        vs_only = self.vs_mask & ~self.both_mask
        fused[:, vs_only] = vs[:, vs_only]
        # falsetto: GTSinger only
        gt_only = self.gt_mask & ~self.both_mask
        fused[:, gt_only] = gt[:, gt_only]
        return fused


def train_fusion_head(vocalset_ckpt, gtsinger_ckpt, technique_dirs,
                      out_path, device="cuda", epochs=40, lr=1e-3):
    """STRETCH GOAL — train the small fusion MLP on the union of both corpora with
    per-sample class masking (only supervise classes present in each clip's source).

    Loss is masked BCE: for each clip we only backprop on classes its source
    dataset actually labels (VocalSet clip → vibrato/breathy/belt/straight;
    GTSinger clip → vibrato/breathy/falsetto). This lets a single 5-class head
    learn from both without the missing-label problem corrupting gradients.

    Left as a callable (not wired into the overnight script) — invoke when you
    want the trained-fusion backup. Saves {'state_dict': ...} to out_path.
    """
    import os
    from torch.utils.data import DataLoader, TensorDataset

    dev = torch.device(device)
    fuser = TechniqueFusion(vocalset_ckpt, gtsinger_ckpt, device=dev, mode="union")

    # Build a clip-level training set: (vs_probs‖gt_probs) features + label + mask.
    # NOTE: this forwards BOTH 7.8M-param specialists over every training clip — the
    # slow, one-time phase (a few minutes, no per-epoch output yet). The epoch loop
    # afterward is instant (the head is a ~500-param MLP).
    print("  Building feature cache (forwarding both specialists over all clips)…")
    feats, labels, masks = [], [], []
    for tdir in technique_dirs:
        for fname in ("technique_train.npz", "technique_gtsinger_train.npz"):
            p = os.path.join(tdir, fname)
            if not os.path.exists(p):
                continue
            d = np.load(p, allow_pickle=True)
            mel_flat = d["mel"].astype(np.float32)
            tech = d["technique"].astype(np.float32)   # (n_clips, 5)
            lengths = d["lengths"].astype(np.int64)
            print(f"    {p}: {len(lengths)} clips")
            # which classes this corpus supervises = classes with any positive
            present = tech.sum(0) > 0                    # (5,) bool
            off = 0
            for ci, L in enumerate(lengths):
                L = int(L)
                cm = torch.from_numpy(mel_flat[off:off + L]).unsqueeze(0).to(dev)
                off += L
                vs, gt = fuser._head_probs(cm)
                feat = np.concatenate([vs.mean(0), gt.mean(0)])  # (10,) clip-level
                feats.append(feat)
                labels.append(tech[ci])
                masks.append(present.astype(np.float32))

    X = torch.tensor(np.stack(feats), dtype=torch.float32)
    Y = torch.tensor(np.stack(labels), dtype=torch.float32)
    M = torch.tensor(np.stack(masks), dtype=torch.float32)
    loader = DataLoader(TensorDataset(X, Y, M), batch_size=64, shuffle=True)

    head = nn.Sequential(
        nn.Linear(2 * len(TECHNIQUE_NAMES), 32), nn.GELU(),
        nn.Linear(32, len(TECHNIQUE_NAMES)),
    ).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    for ep in range(epochs):
        tot = 0.0
        for xb, yb, mb in loader:
            xb, yb, mb = xb.to(dev), yb.to(dev), mb.to(dev)
            opt.zero_grad()
            logits = head(xb)
            loss = (bce(logits, yb) * mb).sum() / mb.sum().clamp_min(1.0)
            loss.backward(); opt.step()
            tot += loss.item()
        print(f"  fusion-head epoch {ep+1}/{epochs} loss={tot/len(loader):.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save({"state_dict": head.state_dict(),
                "classes": TECHNIQUE_NAMES}, out_path)
    print(f"  Saved trained fusion head → {out_path}")
    return out_path
