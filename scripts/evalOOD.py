"""
OOD Pitch + VAD Evaluation
===========================
Evaluates a VocalCoach checkpoint on out-of-distribution datasets that provide
frame-level F0 annotations (Vocadito, and later iKala).

Results are appended to a single tracker JSON (--log) that accumulates every
run. The run name is inferred from the checkpoint folder. A regression check
compares each new run against the best-so-far on each metric and flags any
metric that degrades by more than --regression-threshold (default 10% relative).

The script uses the same mel feature extraction as api.py / train.py:
  SR=16000, hop=160 (10 ms), n_fft=512, n_mels=40, fmin=50, fmax=8000,
  power_to_db(ref=np.max)

Reports: VAD Acc | VDR | VFA↓ | vF1 | RPA | RCA | Gross | Med.c

Usage
-----
  python scripts/evalOOD.py \\
      --checkpoint vocalcoach/runs/stage1_conformer_128_vadfix/checkpoints/best_metric.pth \\
      --dataset vocadito \\
      --data-dir data/vocadito \\
      --log results/ood_log.json

  # Custom regression threshold (default 10%)
  python scripts/evalOOD.py ... --regression-threshold 0.05

  # Export per-clip CSV alongside log
  python scripts/evalOOD.py ... --log results/ood_log.json --csv results/ood_clips.csv
"""

import argparse
import csv
import json
import os
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import build_model, viterbi_decode
from src.evaluate import pitch_metrics, print_pitch_table

# ── Audio / feature constants — must match api.py / train.py ────────
SR         = 16000
HOP_LENGTH = 160    # 10 ms
N_FFT      = 512
N_MELS     = 40
FMIN       = 50
FMAX       = 8000

# Metrics where higher = better (used for regression direction check)
HIGHER_IS_BETTER = {'vad_acc', 'vdr', 'vf1', 'rpa', 'rca'}
# Metrics where lower = better
LOWER_IS_BETTER  = {'vfa', 'gross', 'median_cents'}

# Metrics that trigger a warning but never block logging.
# gross and median_cents naturally worsen when VDR rises (more voiced frames decoded
# means more pitch attempts on hard/ambiguous frames). Blocking on these would
# reject better VAD models simply because they detect more voiced frames.
# vfa is warn-only too — a model with high VDR may accept slightly more false alarms
# as part of the precision/recall tradeoff (pos-weight training).
WARN_ONLY = {'gross', 'median_cents', 'vfa', 'rca'}


def extract_mel(y: np.ndarray) -> np.ndarray:
    """Return (T, 40) log-mel matching api.py pipeline."""
    import librosa
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.T.astype(np.float32)               # (T, 40)


def load_audio(path: str) -> np.ndarray:
    """Load audio, resample to SR, mix to mono."""
    import soundfile as sf
    import librosa
    y, sr = sf.read(path, always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y


def regrid_f0(times: np.ndarray, f0_hz: np.ndarray, n_frames: int) -> np.ndarray:
    """Nearest-neighbour resample annotation timestamps to 10 ms model grid."""
    frame_times = np.arange(n_frames) * (HOP_LENGTH / SR)
    idx = np.searchsorted(times, frame_times)
    idx = np.clip(idx, 0, len(times) - 1)
    idx_prev = np.maximum(idx - 1, 0)
    dist_next = np.abs(frame_times - times[idx])
    dist_prev = np.abs(frame_times - times[idx_prev])
    nearest = np.where(dist_prev < dist_next, idx_prev, idx)
    return f0_hz[nearest].astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# Dataset loaders
# ══════════════════════════════════════════════════════════════════════

def _vocadito_clips(data_dir: str):
    audio_dir = os.path.join(data_dir, "Audio")
    f0_dir    = os.path.join(data_dir, "Annotations", "F0")
    clips = []
    for fname in sorted(os.listdir(audio_dir)):
        if not fname.endswith(".wav") or "Zone" in fname:
            continue
        stem = fname.replace(".wav", "")
        f0_path = os.path.join(f0_dir, f"{stem}_f0.csv")
        if not os.path.exists(f0_path):
            print(f"  [warn] no F0 annotation for {fname}, skipping")
            continue
        clips.append((stem, os.path.join(audio_dir, fname), f0_path))
    return clips


def _load_f0_csv(path: str):
    times, f0s = [], []
    with open(path, newline='') as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            try:
                times.append(float(row[0]))
                f0s.append(float(row[1]))
            except ValueError:
                continue
    return np.array(times, dtype=np.float64), np.array(f0s, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════
# Core OOD eval loop
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_ood(model, clips, device, label="VocalCoach",
             voicing_threshold=0.3, onset_penalty=1.0):
    from tqdm import tqdm

    model.eval()
    clip_results = []

    for name, wav_path, f0_path in tqdm(clips, desc=f"  OOD eval {label}", leave=False):
        y   = load_audio(wav_path)
        mel = extract_mel(y)
        T   = mel.shape[0]

        times, f0_ann = _load_f0_csv(f0_path)
        f0_ref = regrid_f0(times, f0_ann, T)

        mel_t = torch.from_numpy(mel).unsqueeze(0).to(device)
        v, p, _, _, _, _ = model(mel_t)
        pv = v.squeeze().cpu().numpy()
        pp = p.squeeze(0).cpu().numpy()

        f0_dec = viterbi_decode(pp, voicing_threshold=voicing_threshold,
                                onset_penalty=onset_penalty)

        m = pitch_metrics(f0_dec, f0_ref, vad_pred=pv)
        m['name'] = name
        clip_results.append(m)

    if not clip_results:
        return {}

    def smean(lst, key):
        vals = [x[key] for x in lst
                if not np.isnan(x.get(key, float('nan')))]
        return float(np.mean(vals)) if vals else float('nan')

    keys = ['vad_acc', 'vdr', 'vf1', 'vfa', 'rpa', 'rca', 'gross', 'median_cents']
    overall = {k: smean(clip_results, k) for k in keys}
    return {
        'overall': overall,
        '_macro_rpa': overall['rpa'],
        '_macro_vf1': overall['vf1'],
        '_clip_results': clip_results,
    }


# ══════════════════════════════════════════════════════════════════════
# Tracker JSON — load / save / regression check
# ══════════════════════════════════════════════════════════════════════

def load_tracker(log_path: str) -> dict:
    """Load existing tracker JSON, or return empty structure."""
    if os.path.exists(log_path):
        with open(log_path) as f:
            return json.load(f)
    return {"runs": []}


def save_tracker(tracker: dict, log_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(tracker, f, indent=2)


# Runs whose VDR is below this are treated as VAD-collapsed: their pitch
# metrics (RPA/RCA) are computed on a tiny, easy subset of voiced frames and
# are not representative, so they must not set any "best-so-far" baseline.
# (e.g. a model predicting ~everything unvoiced can report RPA≈100% at VDR≈0%.)
MIN_VALID_VDR = 0.30


def best_so_far(tracker: dict, dataset: str) -> dict:
    """
    For each metric, return the best value seen across all runs on this dataset.
    Higher-is-better metrics take the max; lower-is-better take the min.
    Returns {} if no prior runs exist for this dataset.

    Runs with VDR < MIN_VALID_VDR are excluded from baseline computation: a
    collapsed VAD makes that run's pitch metrics meaningless (RPA/RCA measured
    on a thin subset), so it must not poison the baseline other runs compete
    against.
    """
    prior = [r for r in tracker.get("runs", []) if r.get("dataset") == dataset]
    # Drop VAD-collapsed runs so they can't set a spurious best.
    valid = [r for r in prior
             if r.get("overall", {}).get("vdr", 0.0) >= MIN_VALID_VDR]
    prior = valid if valid else prior  # fall back if nothing clears the floor
    if not prior:
        return {}
    best = {}
    all_metrics = HIGHER_IS_BETTER | LOWER_IS_BETTER
    for metric in all_metrics:
        vals = [r["overall"][metric] for r in prior
                if metric in r.get("overall", {}) and not np.isnan(r["overall"][metric])]
        if not vals:
            continue
        best[metric] = max(vals) if metric in HIGHER_IS_BETTER else min(vals)
    return best


def regression_check(overall: dict, best: dict, threshold: float = 0.25) -> tuple[list[dict], list[dict]]:
    """
    Compare current run's overall metrics against the best-so-far.

    Returns (blocking, warn_only):
        blocking:  metrics in HIGHER_IS_BETTER | LOWER_IS_BETTER - WARN_ONLY that degraded
                   > threshold. A non-empty list prevents writing to the log.
        warn_only: metrics in WARN_ONLY that degraded > threshold. These are always
                   printed but never block logging (gross/median_cents degrade naturally
                   when VDR rises; vfa trades off against VDR under pos-weight training).

    For higher-is-better: regression if (best - current) / best > threshold
    For lower-is-better:  regression if (current - best) / best > threshold
    """
    blocking, warn_only = [], []
    for metric, current_val in overall.items():
        if np.isnan(current_val) or metric not in best:
            continue
        best_val = best[metric]
        if np.isnan(best_val) or best_val == 0:
            continue

        if metric in HIGHER_IS_BETTER:
            rel_change = (best_val - current_val) / abs(best_val)
        else:
            rel_change = (current_val - best_val) / abs(best_val)

        if rel_change > threshold:
            entry = {"metric": metric, "current": current_val,
                     "best": best_val, "rel_drop": rel_change}
            if metric in WARN_ONLY:
                warn_only.append(entry)
            else:
                blocking.append(entry)
    return blocking, warn_only


def infer_run_name(checkpoint_path: str) -> str:
    """
    Derive a human-readable run name from the checkpoint path.
    e.g. 'vocalcoach/runs/stage1_conformer_128_vadfix/checkpoints/best_metric.pth'
      → 'stage1_conformer_128_vadfix/best_metric'
    """
    parts = os.path.normpath(checkpoint_path).split(os.sep)
    # Find 'runs' in the path and take everything after it up to the filename
    try:
        runs_idx = next(i for i, p in enumerate(parts) if p == "runs")
        # runs/<run_name>/checkpoints/<file.pth>  → <run_name>/<stem>
        run_dir  = parts[runs_idx + 1]
        ckpt_stem = os.path.splitext(parts[-1])[0]
        return f"{run_dir}/{ckpt_stem}"
    except (StopIteration, IndexError):
        return os.path.splitext(os.path.basename(checkpoint_path))[0]


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="VocalCoach OOD pitch+VAD evaluation")
    p.add_argument("--checkpoint", required=True,
                   help="VocalCoach checkpoint (.pth)")
    p.add_argument("--dataset", required=True, choices=["vocadito"],
                   help="OOD dataset to evaluate")
    p.add_argument("--data-dir", required=True,
                   help="Root directory of the dataset")
    p.add_argument("--log", default="results/ood_log.json",
                   help="Tracker JSON that accumulates all runs (default: results/ood_log.json)")
    p.add_argument("--run-name", default=None,
                   help="Override auto-inferred run name (default: inferred from checkpoint path)")
    p.add_argument("--device", default="auto",
                   help="cpu / cuda / mps / auto")
    p.add_argument("--voicing-threshold", type=float, default=0.3)
    p.add_argument("--onset-penalty",     type=float, default=1.0)
    p.add_argument("--regression-threshold", type=float, default=0.25,
                   help="Relative degradation threshold for blocking regression (default: 0.25 = 25%%). "
                        "gross, median_cents, and vfa are warn-only and never block logging.")
    p.add_argument("--csv", default=None,
                   help="Also save per-clip CSV to this path")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Device ───────────────────────────────────────────────────────
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Load model ───────────────────────────────────────────────────
    warnings.warn("Loading checkpoint via torch.load — trusted source only.",
                  RuntimeWarning)
    ckpt   = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    arch   = ckpt.get("arch", "tcn")
    causal = ckpt.get("causal", False)
    kwargs = dict(ckpt.get("model_kwargs", {}))
    kwargs.pop("causal", None)
    model  = build_model(arch, causal=causal, **kwargs).to(device)
    model.load_state_dict(ckpt["state_dict"])
    run_label = f"VocalCoach{arch.upper()} (causal={causal}, epoch={ckpt.get('epoch', '?')})"
    print(f"Loaded: {run_label}")

    run_name = args.run_name or infer_run_name(args.checkpoint)
    print(f"Run name: {run_name}")

    # ── Dataset ──────────────────────────────────────────────────────
    data_dir = os.path.abspath(args.data_dir)
    if args.dataset == "vocadito":
        clips = _vocadito_clips(data_dir)
        dataset_label = f"Vocadito ({len(clips)} clips)"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    print(f"Dataset: {dataset_label}")

    # ── Evaluate ─────────────────────────────────────────────────────
    results = eval_ood(model, clips, device, label=run_label,
                       voicing_threshold=args.voicing_threshold,
                       onset_penalty=args.onset_penalty)
    if not results:
        print("No results — check dataset path.")
        return

    overall   = results['overall']
    clip_r    = results.get('_clip_results', [])

    # ── Display ──────────────────────────────────────────────────────
    display = {'overall': overall,
               '_macro_rpa': results['_macro_rpa'],
               '_macro_vf1': results['_macro_vf1']}
    print_pitch_table(display, label=f"{run_label} | OOD: {dataset_label}")

    clip_r_sorted = sorted(
        [c for c in clip_r if not np.isnan(c.get('vdr', float('nan')))],
        key=lambda x: x['vdr'],
    )
    if clip_r_sorted:
        print(f"\n  Worst 5 clips by VDR:")
        for c in clip_r_sorted[:5]:
            print(f"    {c['name']:<20}  VDR={c['vdr']:.1%}  VFA={c.get('vfa',float('nan')):.1%}"
                  f"  RPA={c['rpa']:.1%}  vF1={c['vf1']:.1%}")
        print(f"\n  Best 5 clips by VDR:")
        for c in clip_r_sorted[-5:]:
            print(f"    {c['name']:<20}  VDR={c['vdr']:.1%}  VFA={c.get('vfa',float('nan')):.1%}"
                  f"  RPA={c['rpa']:.1%}  vF1={c['vf1']:.1%}")

    # ── Load tracker + regression check ──────────────────────────────
    tracker = load_tracker(args.log)
    best    = best_so_far(tracker, args.dataset)
    blocking, warn_only = regression_check(overall, best, args.regression_threshold)

    def _fmt_metric(metric, val):
        return f"{val:.1%}" if metric != 'median_cents' else f"{val:.1f}"

    print(f"\n{'═'*60}")
    if not best:
        print("  First run on this dataset — no baseline to compare against.")
        status = "first"
    elif blocking:
        print(f"  ⚠  REGRESSION DETECTED (threshold: {args.regression_threshold:.0%})")
        print(f"  {'Metric':<16}  {'Current':>10}  {'Best':>10}  {'Drop':>8}")
        print(f"  {'─'*16}  {'─'*10}  {'─'*10}  {'─'*8}")
        for r in blocking:
            print(f"  {r['metric']:<16}  {_fmt_metric(r['metric'], r['current']):>10}"
                  f"  {_fmt_metric(r['metric'], r['best']):>10}  {r['rel_drop']:>7.1%}")
        if warn_only:
            print(f"\n  ⚡ Warn-only (not blocking): {', '.join(r['metric'] for r in warn_only)}")
        print(f"\n  Run REJECTED — results not written to {args.log}.")
        status = "regressed"
    else:
        print(f"  ✓  No blocking regressions vs best-so-far  (threshold: {args.regression_threshold:.0%})")
        if warn_only:
            print(f"\n  ⚡ Warn-only regressions (gross/vfa trade off with VDR — not blocking):")
            print(f"  {'Metric':<16}  {'Current':>10}  {'Best':>10}  {'Drop':>8}")
            print(f"  {'─'*16}  {'─'*10}  {'─'*10}  {'─'*8}")
            for r in warn_only:
                print(f"  {r['metric']:<16}  {_fmt_metric(r['metric'], r['current']):>10}"
                      f"  {_fmt_metric(r['metric'], r['best']):>10}  {r['rel_drop']:>7.1%}")
        improvements = []
        for metric in HIGHER_IS_BETTER | LOWER_IS_BETTER:
            cur = overall.get(metric, float('nan'))
            bst = best.get(metric, float('nan'))
            if np.isnan(cur) or np.isnan(bst) or bst == 0:
                continue
            if metric in HIGHER_IS_BETTER:
                delta = cur - bst
                if delta > 0.001:
                    improvements.append((metric, cur, bst, delta))
            else:
                delta = bst - cur
                if delta > 0.001:
                    improvements.append((metric, cur, bst, delta))
        if improvements:
            print(f"\n  Improvements over previous best:")
            for metric, cur, bst, delta in improvements:
                print(f"    {metric:<16}  {_fmt_metric(metric, cur)} (was {_fmt_metric(metric, bst)}, +{delta:.1%})")
        status = "ok"
    print(f"{'═'*60}")

    # ── Append to tracker (only if no regressions) ───────────────────
    if status == "regressed":
        print(f"\nNot written to {args.log} — fix regressions before logging.")
        return

    entry = {
        "run_name":       run_name,
        "dataset":        args.dataset,
        "checkpoint":     args.checkpoint,
        "arch":           arch,
        "causal":         causal,
        "epoch":          ckpt.get("epoch", None),
        "onset_penalty":  args.onset_penalty,
        "n_clips":        len(clip_r),
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "status":         status,
        "overall":        overall,
        "per_clip":       clip_r,
    }
    tracker["runs"].append(entry)
    save_tracker(tracker, args.log)
    print(f"\nAppended to {args.log}  ({len(tracker['runs'])} total runs)")

    # ── Optional per-clip CSV ─────────────────────────────────────────
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
        import csv as csv_mod
        fields = ['run_name', 'name', 'vad_acc', 'vdr', 'vf1', 'vfa',
                  'rpa', 'rca', 'gross', 'median_cents']
        write_header = not os.path.exists(args.csv)
        with open(args.csv, 'a', newline='') as f:
            writer = csv_mod.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            if write_header:
                writer.writeheader()
            for c in clip_r:
                row = {k: (f"{c[k]:.4f}" if not np.isnan(c.get(k, float('nan'))) else '')
                       for k in fields if k not in ('run_name', 'name')}
                row['run_name'] = run_name
                row['name']     = c['name']
                writer.writerow(row)
        print(f"Per-clip CSV appended to {args.csv}")


if __name__ == "__main__":
    main()
