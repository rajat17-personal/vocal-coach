"""
Train the technique fusion head (STRETCH GOAL / backup).

Learns a small MLP that maps the two specialists' stacked probabilities → unified
5-class technique output, trained on the union of both corpora with per-sample
class masking (only supervise classes the clip's source actually labels). This is
the alternative to the default union ensemble — useful where the union's naive
averaging of the shared classes (vibrato/breathy) underperforms.

Run locally while a backbone trains elsewhere — it only forwards the (already
trained) specialists, no backbone training. CPU is fine; GPU faster.

  python scripts/trainFusionHead.py \\
      --vocalset-ckpt vocalcoach/runs/stage2_vocalset_aug_warmjoint_probe/checkpoints/best_metric.pth \\
      --gtsinger-ckpt vocalcoach/runs/stage2_gtsinger_aug_specialist/checkpoints/best_metric.pth \\
      --technique-dirs data/vocalset_aug data/gtsinger_aug \\
      --out vocalcoach/runs/fusion_head/fusion.pth

Then evaluate the trained head vs the union default:
  python scripts/evalFusion.py --mode trained \\
      --fusion-checkpoint vocalcoach/runs/fusion_head/fusion.pth \\
      --vocalset-ckpt ... --gtsinger-ckpt ...
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.technique_fusion import train_fusion_head


def main():
    p = argparse.ArgumentParser(description="Train the technique fusion head")
    p.add_argument("--vocalset-ckpt", required=True)
    p.add_argument("--gtsinger-ckpt", required=True)
    p.add_argument("--technique-dirs", nargs="+",
                   default=["data/vocalset_aug", "data/gtsinger_aug"])
    p.add_argument("--out", default="vocalcoach/runs/fusion_head/fusion.pth")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_fusion_head(
        args.vocalset_ckpt, args.gtsinger_ckpt, args.technique_dirs,
        args.out, device=device, epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()
