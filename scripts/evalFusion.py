"""
Union-Fusion Technique Evaluation
=================================
Measures the TechniqueFusion union ensemble's per-class accuracy as a single
5-class system, using both held-out test sets:

  - belt, straight   → scored on the VocalSet test set (GTSinger has none)
  - falsetto         → scored on the GTSinger test set (VocalSet has none)
  - vibrato, breathy → scored on BOTH (concatenated), since both corpora label them

For each class we run the fused predictor over the clips of whichever test set(s)
contain that class, take the clip-level mean probability, and report AP (threshold-
free), F1@0.5, and F1 at the per-class best threshold (mirrors the in-training
technique eval). This is the deployable system's real accuracy — neither single
specialist's self-eval reflects it.

Usage
-----
  python scripts/evalFusion.py \\
      --vocalset-ckpt vocalcoach/runs/stage2_vocalset_aug_warmjoint_probe/checkpoints/best_metric.pth \\
      --gtsinger-ckpt vocalcoach/runs/stage2_gtsinger_aug_specialist/checkpoints/best_metric.pth \\
      --vocalset-test data/vocalset_aug/technique_test.npz \\
      --gtsinger-test data/gtsinger_aug/technique_gtsinger_test.npz
"""

import argparse
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import TECHNIQUE_NAMES
from src.technique_fusion import TechniqueFusion

# Which test set(s) supervise each class.
_CLASS_SOURCES = {
    "vibrato":  ["vocalset", "gtsinger"],
    "breathy":  ["vocalset", "gtsinger"],
    "falsetto": ["gtsinger"],
    "belt":     ["vocalset"],
    "straight": ["vocalset"],
}


def _clip_preds(fuser, npz_path):
    """Run the fused predictor over every clip → (clip_probs (N,5), labels (N,5))."""
    d = np.load(npz_path, allow_pickle=True)
    mel = d["mel"].astype(np.float32)
    tech = d["technique"].astype(np.float32)
    lengths = d["lengths"].astype(np.int64)
    probs, off = [], 0
    for L in lengths:
        L = int(L)
        p = fuser.predict(mel[off:off + L])      # (T, 5)
        probs.append(p.mean(0))                   # clip-level
        off += L
    return np.stack(probs), tech


def _best_threshold_f1(true_k, prob_k):
    # Floor 0.2 so the sweep can't collapse to all-positive on a majority class
    # (inflates F1 to the base rate). AP is the base-rate-aware metric to trust.
    best = 0.0
    for t in np.linspace(0.2, 0.95, 16):
        _, _, f1, _ = precision_recall_fscore_support(
            true_k, (prob_k > t).astype(int), average="binary", zero_division=0)
        best = max(best, f1)
    return best


def main():
    ap = argparse.ArgumentParser(description="Evaluate the union-fusion technique system")
    ap.add_argument("--vocalset-ckpt", required=True)
    ap.add_argument("--gtsinger-ckpt", required=True)
    ap.add_argument("--vocalset-test", default="data/vocalset_aug/technique_test.npz")
    ap.add_argument("--gtsinger-test", default="data/gtsinger_aug/technique_gtsinger_test.npz")
    ap.add_argument("--mode", default="union", choices=["union", "trained"])
    ap.add_argument("--fusion-checkpoint", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    device = (args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    fuser = TechniqueFusion(args.vocalset_ckpt, args.gtsinger_ckpt, device=device,
                            mode=args.mode, fusion_checkpoint=args.fusion_checkpoint)

    print(f"Fusion mode: {fuser.mode}")
    vs_probs, vs_true = _clip_preds(fuser, args.vocalset_test)
    gt_probs, gt_true = _clip_preds(fuser, args.gtsinger_test)
    sources = {"vocalset": (vs_probs, vs_true), "gtsinger": (gt_probs, gt_true)}
    print(f"  VocalSet test: {len(vs_true)} clips   GTSinger test: {len(gt_true)} clips")

    print(f"\n  {'Technique':<10} {'AP':>7} {'F1@.5':>7} {'F1*':>7} {'src':>16}  n_pos")
    print(f"  {'-'*58}")
    aps, f1s, f1bests = [], [], []
    for k, name in enumerate(TECHNIQUE_NAMES):
        srcs = _CLASS_SOURCES[name]
        true = np.concatenate([sources[s][1][:, k] for s in srcs])
        prob = np.concatenate([sources[s][0][:, k] for s in srcs])
        npos = int(true.sum())
        if npos == 0:
            print(f"  {name:<10} {'—':>7} {'—':>7} {'—':>7} {'+'.join(srcs):>16}  0")
            continue
        ap_ = average_precision_score(true, prob)
        _, _, f1, _ = precision_recall_fscore_support(
            true, (prob > 0.5).astype(int), average="binary", zero_division=0)
        f1b = _best_threshold_f1(true, prob)
        aps.append(ap_); f1s.append(f1); f1bests.append(f1b)
        print(f"  {name:<10} {ap_:>7.3f} {f1:>7.3f} {f1b:>7.3f} {'+'.join(srcs):>16}  {npos}")

    print(f"  {'-'*58}")
    print(f"  FUSED macro AP: {np.mean(aps):.3f}   "
          f"macro F1@.5: {np.mean(f1s):.3f}   macro F1*: {np.mean(f1bests):.3f}")
    print(f"  (5-class system; each class scored on the test set(s) that label it)")


if __name__ == "__main__":
    main()
