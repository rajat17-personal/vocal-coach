"""
VocalCoach Evaluation Script
=============================

Evaluates a trained VocalCoach model and produces a report comparing it
against the NanoPitch GRU baseline on the same test set.

Metrics
-------
  Pitch / VAD (from test.npz — same file NanoPitch uses):
    VAD Acc  — fraction of frames with correct voice/silence label
    VDR      — Voicing Detection Rate (of truly voiced frames, how many detected)
    RPA      — Raw Pitch Accuracy (voiced frames within 50 cents)
    RCA      — Raw Chroma Accuracy (RPA ignoring octave errors)
    Gross    — Gross error rate (> 50 cents)
    Med.c    — Median pitch error in cents

  Technique (from technique_test.npz — VocalSet held-out singers):
    Per-class F1 and Average Precision
    Macro F1, Macro AP

Usage
-----
  # Evaluate VocalCoach checkpoint vs NanoPitch baseline
  python vocalcoach/evaluate.py \\
      --checkpoint vocalcoach/runs/tcn_noncausal/checkpoints/best_metric.pth \\
      --data-dir   data \\
      --technique-dir data/vocalset

  # Compare with NanoPitch checkpoint
  python vocalcoach/evaluate.py \\
      --checkpoint vocalcoach/runs/tcn_noncausal/checkpoints/best_metric.pth \\
      --data-dir   data \\
      --nanopitch  training/runs/best/checkpoints/best_macro_rpa.pth \\
      --technique-dir data/vocalset

  # Save results to JSON
  python vocalcoach/evaluate.py \\
      --checkpoint ... --data-dir data --json results/exp_a.json
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.model import (
    build_model, viterbi_decode,
    PITCH_BINS, TECHNIQUE_NAMES,
)


# ═══════════════════════════════════════════════════════════════════════
# Arguments
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="VocalCoach evaluation harness")
    p.add_argument("--checkpoint", required=True,
                   help="path to VocalCoach checkpoint (.pth)")
    p.add_argument("--data-dir", default=None,
                   help="directory with test.npz. Omit to skip pitch/VAD eval.")
    p.add_argument("--skip-pitch", action="store_true",
                   help="skip pitch/VAD eval even if --data-dir is set")
    p.add_argument("--technique-dir", default=None,
                   help="directory with technique_test.npz "
                        "(default: None — skip technique evaluation)")
    p.add_argument("--nanopitch", default=None,
                   help="NanoPitch checkpoint for side-by-side comparison")
    p.add_argument("--device", default="auto",
                   help="cpu / cuda / mps / auto")
    p.add_argument("--json", default=None,
                   help="save results to this JSON file")
    p.add_argument("--csv", default=None,
                   help="save pitch results to this CSV file")
    p.add_argument("--voicing-threshold", type=float, default=0.3,
                   help="Viterbi init threshold: min pitch posterior peak at frame 0 to start "
                        "as voiced. Has minimal effect beyond the first frame — tune "
                        "--onset-penalty instead for VDR control.")
    p.add_argument("--onset-penalty", type=float, default=1.0,
                   help="Viterbi voiced<->unvoiced transition cost (log-domain). "
                        "Default 1.0 suits multi-task models with diffuse posteriors (peaks "
                        "0.15-0.25). Pitch-only models with sharper posteriors may prefer 2.0. "
                        "Grid-search VDR vs FAR tradeoff: lower = more voiced frames decoded.")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# Metric helpers
# ═══════════════════════════════════════════════════════════════════════

def pitch_metrics(f0_dec, f0_ref, vad_pred=None):
    """Compute the standard NanoPitch pitch evaluation metrics.

    Args:
        f0_dec:   (T,) decoded F0 in Hz (0 = unvoiced)
        f0_ref:   (T,) reference F0 in Hz (0 = unvoiced)
        vad_pred: (T,) optional — VAD head probability, used for VAD Acc

    Returns:
        dict with keys: vad_acc, vdr, rpa, rca, gross, median_cents
    """
    voiced_gt   = f0_ref  > 0
    voiced_pred = f0_dec  > 0

    # VAD accuracy
    if vad_pred is not None:
        vad_bin = vad_pred > 0.5
        vad_acc = float(np.mean(vad_bin == voiced_gt))
    else:
        vad_acc = float(np.mean(voiced_pred == voiced_gt))

    # Voicing Detection Rate (recall on voiced frames)
    vdr = float(np.mean(voiced_pred[voiced_gt])) if voiced_gt.sum() > 0 else float('nan')

    # Voicing F1 and False Alarm Rate — from raw TP/FP/FN counts.
    # VFA (Voicing False Alarm) = FP / total_silence_frames: how often a silence
    # frame is wrongly called voiced. Directly exposes dead/saturated VAD heads
    # that vF1 and VAD Acc both partially hide (vF1 trades off recall vs precision,
    # VAD Acc is dominated by the majority class).
    tp = int((voiced_pred & voiced_gt).sum())
    fp = int((voiced_pred & ~voiced_gt).sum())
    fn = int((~voiced_pred & voiced_gt).sum())
    tn = int((~voiced_pred & ~voiced_gt).sum())
    prec_v = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    rec_v  = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    if not (np.isnan(prec_v) or np.isnan(rec_v)) and (prec_v + rec_v) > 0:
        vf1 = float(2 * prec_v * rec_v / (prec_v + rec_v))
    else:
        vf1 = float('nan')
    # VFA: low = good (model respects silence); target < 15% for a usable backbone
    vfa = float(fp / (fp + tn)) if (fp + tn) > 0 else float('nan')

    # Pitch metrics on doubly-voiced frames
    both = voiced_gt & voiced_pred
    if both.sum() > 0:
        cents_err = np.abs(1200.0 * np.log2(
            f0_dec[both] / (f0_ref[both] + 1e-10) + 1e-10))
        rpa   = float(np.mean(cents_err < 50))
        gross = float(np.mean(cents_err >= 50))
        med_c = float(np.median(cents_err))

        chroma_err = np.mod(cents_err, 1200.0)
        chroma_circular = np.minimum(chroma_err, 1200.0 - chroma_err)
        rca = float(np.mean(chroma_circular < 50))
    else:
        rpa = gross = rca = med_c = float('nan')

    return dict(vad_acc=vad_acc, vdr=vdr, vf1=vf1, vfa=vfa, rpa=rpa, rca=rca,
                gross=gross, median_cents=med_c)


# ═══════════════════════════════════════════════════════════════════════
# Pitch / VAD evaluation on test.npz
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_pitch(model, data_dir, device, label="VocalCoach", voicing_threshold=0.3, onset_penalty=1.0):
    """Evaluate pitch accuracy and VAD on test.npz across SNR conditions."""
    test_path = os.path.join(data_dir, "test.npz")
    if not os.path.exists(test_path):
        print(f"  [skip] {test_path} not found")
        return {}

    test   = np.load(test_path)
    clips  = test['clips'].astype(np.float32)   # (N, T, 40)
    f0_all = test['f0'].astype(np.float32)       # (N, T)
    vad_all = test['vad'].astype(np.float32)     # (N, T)
    snrs   = test['snr']                          # (N,)
    N      = clips.shape[0]

    model.eval()
    clip_results = []
    for i in tqdm(range(N), desc=f"  Evaluating {label}", leave=False):
        mel = torch.from_numpy(clips[i]).unsqueeze(0).to(device)
        v, p, _, _, _, _ = model(mel)
        pv = v.squeeze().cpu().numpy()    # (T,)
        pp = p.squeeze(0).cpu().numpy()   # (T, 360)
        T  = pv.shape[0]

        f0_ref = f0_all[i, :T]
        f0_dec = viterbi_decode(pp, voicing_threshold=voicing_threshold, onset_penalty=onset_penalty)
        metrics = pitch_metrics(f0_dec, f0_ref, vad_pred=pv)
        metrics['snr'] = float(snrs[i])
        clip_results.append(metrics)

    # Group by SNR
    by_snr = {}
    for r in clip_results:
        by_snr.setdefault(r['snr'], []).append(r)

    def smean(lst, key):
        vals = [x[key] for x in lst if not np.isnan(x.get(key, float('nan')))]
        return float(np.mean(vals)) if vals else float('nan')

    results = {}
    for snr in sorted(by_snr.keys(), key=lambda x: x if np.isfinite(x) else 1e9):
        c   = by_snr[snr]
        tag = "clean" if not np.isfinite(snr) else f"{snr:+.0f} dB"
        results[tag] = {k: smean(c, k) for k in
                        ['vad_acc', 'vdr', 'vf1', 'vfa', 'rpa', 'rca', 'gross', 'median_cents']}

    macro_rpa = float(np.nanmean([v['rpa'] for v in results.values()]))
    macro_vf1 = float(np.nanmean([v['vf1'] for v in results.values()]))
    results['_macro_rpa'] = macro_rpa
    results['_macro_vf1'] = macro_vf1
    return results


def print_pitch_table(results, label="Model"):
    print(f"\n{'═'*94}")
    print(f"  {label} — Pitch & VAD Evaluation")
    print(f"{'═'*94}")
    hdr = f"  {'Condition':<10}  {'VAD Acc':>8}  {'VDR':>8}  {'VFA↓':>8}  {'vF1':>8}  "
    hdr += f"{'RPA':>8}  {'RCA':>8}  {'Gross':>8}  {'Med.c':>7}"
    print(hdr)
    print(f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*7}")
    for tag, m in results.items():
        if tag.startswith('_'):
            continue
        def fmt(v, pct=True):
            if np.isnan(v): return '     n/a'
            return f"{v:8.1%}" if pct else f"{v:7.1f}"
        row = (f"  {tag:<10}  {fmt(m['vad_acc'])}  {fmt(m['vdr'])}  {fmt(m.get('vfa', float('nan')))}  {fmt(m['vf1'])}"
               f"  {fmt(m['rpa'])}  {fmt(m['rca'])}"
               f"  {fmt(m['gross'])}  {fmt(m['median_cents'], pct=False)}")
        print(row)
    macro_rpa = results.get('_macro_rpa', float('nan'))
    macro_vf1 = results.get('_macro_vf1', float('nan'))
    print(f"\n  Macro RPA: {macro_rpa:.1%}  |  Macro vF1: {macro_vf1:.1%}")


# ═══════════════════════════════════════════════════════════════════════
# Technique evaluation on technique_test.npz
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_technique(model, technique_dir, device):
    """Evaluate technique classification on technique_test.npz.

    Also accepts technique_gtsinger_test.npz (GTSinger held-out split) —
    the script tries technique_test.npz first, then the GTSinger filename.
    """
    path = os.path.join(technique_dir, "technique_test.npz")
    if not os.path.exists(path):
        # GTSinger extraction saves under a different name
        path = os.path.join(technique_dir, "technique_gtsinger_test.npz")
    if not os.path.exists(path):
        print(f"  [skip] no technique_test.npz or technique_gtsinger_test.npz in {technique_dir}")
        return {}

    data     = np.load(path, allow_pickle=False)
    mel_flat = data["mel"].astype(np.float32)        # (total_frames, 40)
    f0_flat  = data["f0"].astype(np.float32)
    vad_flat = data["vad"].astype(np.float32)
    tech_all = data["technique"].astype(np.float32)  # (n_clips, N_TECH)
    lengths  = data["lengths"]

    model.eval()
    all_pred, all_true = [], []
    offset = 0
    for clip_idx, length in enumerate(tqdm(lengths, desc="  Technique eval", leave=False)):
        length = int(length)
        mel_clip = mel_flat[offset:offset + length]    # (T, 40)
        offset += length
        if length == 0:
            continue

        mel_t = torch.from_numpy(mel_clip).unsqueeze(0).to(device)
        _, _, pred_tech, _, _, _ = model(mel_t)
        # Average over time to get clip-level prediction
        pred_clip = pred_tech.squeeze(0).mean(0).cpu().numpy()   # (N_TECH,)
        all_pred.append(pred_clip)
        all_true.append(tech_all[clip_idx])

    if not all_pred:
        return {}

    all_pred = np.stack(all_pred)   # (N, N_TECH)
    all_true = np.stack(all_true)   # (N, N_TECH)
    pred_bin = (all_pred > 0.5).astype(int)

    results = {}
    # Floor 0.2 (not ~0) so the per-class F1 sweep can't collapse to the all-positive
    # degenerate threshold on a majority class (which just reproduces the base rate).
    thresholds = np.linspace(0.2, 0.95, 76)    # sweep 0.2…0.95 in 0.01 steps
    best_thresholds = {}
    for k, name in enumerate(TECHNIQUE_NAMES):
        n_pos = int(all_true[:, k].sum())
        if n_pos == 0:
            results[name] = dict(precision=float('nan'), recall=float('nan'),
                                 f1=float('nan'), ap=float('nan'), n_pos=0,
                                 threshold=float('nan'))
            best_thresholds[name] = 0.5
            continue
        ap = average_precision_score(all_true[:, k], all_pred[:, k])
        # Per-class threshold sweep: find t* that maximises F1 on this test set.
        best_f1, best_t = -1.0, 0.5
        for t in thresholds:
            pb = (all_pred[:, k] > t).astype(int)
            _, _, f1_t, _ = precision_recall_fscore_support(
                all_true[:, k], pb, average='binary', zero_division=0)
            if f1_t > best_f1:
                best_f1, best_t = f1_t, float(t)
        best_thresholds[name] = best_t
        pred_bin_best = (all_pred[:, k] > best_t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            all_true[:, k], pred_bin_best, average='binary', zero_division=0)
        # Also compute fixed-0.5 metrics for comparison
        pred_bin_05 = (all_pred[:, k] > 0.5).astype(int)
        _, _, f1_05, _ = precision_recall_fscore_support(
            all_true[:, k], pred_bin_05, average='binary', zero_division=0)
        results[name] = dict(precision=float(p), recall=float(r),
                             f1=float(f1), f1_at_05=float(f1_05),
                             ap=float(ap), n_pos=n_pos,
                             threshold=best_t)

    valid_f1 = [v['f1'] for v in results.values() if not np.isnan(v['f1'])]
    valid_ap = [v['ap'] for v in results.values() if not np.isnan(v['ap'])]
    results['_macro_f1'] = float(np.mean(valid_f1)) if valid_f1 else float('nan')
    results['_macro_ap'] = float(np.mean(valid_ap)) if valid_ap else float('nan')
    f1_at_05_vals = [v['f1_at_05'] for v in results.values()
                     if isinstance(v, dict) and not np.isnan(v.get('f1_at_05', float('nan')))]
    results['_macro_f1_at_05'] = float(np.mean(f1_at_05_vals)) if f1_at_05_vals else float('nan')
    results['_best_thresholds'] = best_thresholds

    # Single-label clip accuracy using per-class best thresholds.
    pred_bin_tuned = np.stack(
        [(all_pred[:, k] > best_thresholds[n]).astype(int)
         for k, n in enumerate(TECHNIQUE_NAMES)], axis=1)
    true_labels = np.argmax(all_true, axis=1)
    pred_labels = np.argmax(all_pred, axis=1)
    has_label   = all_true.sum(axis=1) > 0
    if has_label.sum() > 0:
        clip_acc = float(np.mean(pred_labels[has_label] == true_labels[has_label]))
    else:
        clip_acc = float('nan')
    results['_clip_accuracy'] = clip_acc

    return results


def print_technique_table(results, label="Model"):
    print(f"\n{'═'*82}")
    print(f"  {label} — Technique Classification")
    print(f"{'═'*82}")
    has_tuned = any('threshold' in results.get(n, {}) for n in TECHNIQUE_NAMES)
    if has_tuned:
        print(f"  {'Technique':<12}  {'Prec':>7}  {'Recall':>7}  {'F1':>7}  {'F1@0.5':>7}  {'AP':>7}  {'Thresh':>6}  {'n_clips':>7}")
        print(f"  {'─'*12}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*7}")
    else:
        print(f"  {'Technique':<12}  {'Prec':>7}  {'Recall':>7}  {'F1':>7}  {'AP':>7}  {'n_clips':>7}")
        print(f"  {'─'*12}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")
    for name in TECHNIQUE_NAMES:
        if name not in results:
            continue
        m = results[name]
        def _fmt(key):
            v = m.get(key, float('nan'))
            return f"{v:7.3f}" if not np.isnan(v) else "    n/a"
        if has_tuned:
            t = m.get('threshold', float('nan'))
            t_str = f"{t:6.2f}" if not np.isnan(t) else "   n/a"
            print(f"  {name:<12}  {_fmt('precision')}  {_fmt('recall')}  {_fmt('f1')}  {_fmt('f1_at_05')}  {_fmt('ap')}  {t_str}  {m['n_pos']:7d}")
        else:
            print(f"  {name:<12}  {_fmt('precision')}  {_fmt('recall')}  {_fmt('f1')}  {_fmt('ap')}  {m['n_pos']:7d}")
    clip_acc = results.get('_clip_accuracy', float('nan'))
    clip_str = f"{clip_acc:.1%}" if not np.isnan(clip_acc) else "n/a"
    mf1      = results.get('_macro_f1', float('nan'))
    mf1_05   = results.get('_macro_f1_at_05', float('nan'))
    gain_str = (f"  (+{mf1 - mf1_05:.3f} vs fixed-0.5)"
                if not np.isnan(mf1) and not np.isnan(mf1_05) else "")
    print(f"\n  Macro F1 (tuned thresh): {mf1:.4f}{gain_str}"
          f"\n  Macro AP: {results.get('_macro_ap', float('nan')):.4f}"
          f"   Clip Acc (argmax): {clip_str}"
          f"  [SOTA: MuQ 81.5%, AST 82.0%]")


# ═══════════════════════════════════════════════════════════════════════
# NanoPitch baseline (for side-by-side comparison)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_nanopitch(ckpt_path, data_dir, device):
    """Load and evaluate a NanoPitch checkpoint for baseline comparison."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))
        from model import NanoPitch, viterbi_decode as np_viterbi
    except ImportError:
        print("  [skip] Could not import NanoPitch model from training/model.py")
        return {}

    warnings.warn("Loading NanoPitch checkpoint via torch.load — trusted source only.",
                  RuntimeWarning)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    kwargs = ckpt.get("model_kwargs", {})
    np_model = NanoPitch(**kwargs).to(device)
    np_model.load_state_dict(ckpt["state_dict"])

    test_path = os.path.join(data_dir, "test.npz")
    if not os.path.exists(test_path):
        return {}
    test   = np.load(test_path)
    clips  = test['clips'].astype(np.float32)
    f0_all = test['f0'].astype(np.float32)
    snrs   = test['snr']
    N      = clips.shape[0]

    np_model.eval()
    clip_results = []
    for i in tqdm(range(N), desc="  NanoPitch baseline", leave=False):
        mel = torch.from_numpy(clips[i]).unsqueeze(0).to(device)
        v, p, _ = np_model(mel)
        pv = v.squeeze().cpu().numpy()
        pp = p.squeeze(0).cpu().numpy()
        T  = pv.shape[0]
        f0_ref = f0_all[i, :T]
        f0_dec = np_viterbi(pp)
        metrics = pitch_metrics(f0_dec, f0_ref, vad_pred=pv)
        metrics['snr'] = float(snrs[i])
        clip_results.append(metrics)

    by_snr = {}
    for r in clip_results:
        by_snr.setdefault(r['snr'], []).append(r)

    def smean(lst, key):
        vals = [x[key] for x in lst if not np.isnan(x.get(key, float('nan')))]
        return float(np.mean(vals)) if vals else float('nan')

    results = {}
    for snr in sorted(by_snr.keys(), key=lambda x: x if np.isfinite(x) else 1e9):
        c   = by_snr[snr]
        tag = "clean" if not np.isfinite(snr) else f"{snr:+.0f} dB"
        results[tag] = {k: smean(c, k) for k in
                        ['vad_acc', 'vdr', 'vf1', 'vfa', 'rpa', 'rca', 'gross', 'median_cents']}
    results['_macro_rpa'] = float(np.nanmean([v['rpa'] for v in results.values()]))
    results['_macro_vf1'] = float(np.nanmean([v['vf1'] for v in results.values()]))
    return results


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

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

    # Load VocalCoach checkpoint
    warnings.warn("Loading checkpoint via torch.load — trusted source only.",
                  RuntimeWarning)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    arch   = ckpt.get("arch", "tcn")
    causal = ckpt.get("causal", False)
    kwargs = dict(ckpt.get("model_kwargs", {}))
    kwargs.pop("causal", None)   # already passed explicitly; avoid duplicate kwarg
    model  = build_model(arch, causal=causal, **kwargs).to(device)
    model.load_state_dict(ckpt["state_dict"])
    label = f"VocalCoach{arch.upper()} (causal={causal}, epoch={ckpt.get('epoch','?')})"
    print(f"Loaded: {label}")

    data_dir     = os.path.abspath(args.data_dir) if args.data_dir else None
    tech_dir     = os.path.abspath(args.technique_dir) if args.technique_dir else None

    all_results = {}

    # ── Pitch / VAD ──────────────────────────────────────────────────
    if data_dir and not getattr(args, 'skip_pitch', False):
        pitch_res = eval_pitch(model, data_dir, device, label=label,
                               voicing_threshold=args.voicing_threshold,
                               onset_penalty=args.onset_penalty)
        if pitch_res:
            print_pitch_table(pitch_res, label=label)
            all_results['pitch'] = pitch_res

    # ── NanoPitch baseline ───────────────────────────────────────────
    if args.nanopitch:
        np_res = eval_nanopitch(args.nanopitch, data_dir, device)
        if np_res:
            print_pitch_table(np_res, label="NanoPitch GRU (baseline)")
            all_results['nanopitch_pitch'] = np_res
            # Delta summary
            rpa_vc = pitch_res.get('_macro_rpa', float('nan'))
            rpa_np = np_res.get('_macro_rpa', float('nan'))
            if not (np.isnan(rpa_vc) or np.isnan(rpa_np)):
                delta = rpa_vc - rpa_np
                sign  = "+" if delta >= 0 else ""
                print(f"\n  Δ Macro RPA vs NanoPitch: {sign}{delta:.4f} "
                      f"({'↑ better' if delta >= 0 else '↓ worse'})")

    # ── Technique ────────────────────────────────────────────────────
    if tech_dir:
        tech_res = eval_technique(model, tech_dir, device)
        if tech_res:
            print_technique_table(tech_res, label=label)
            all_results['technique'] = tech_res

    # ── Save ─────────────────────────────────────────────────────────
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.json}")

    if args.csv and 'pitch' in all_results:
        import csv
        with open(args.csv, 'w', newline='') as f:
            fields = ['condition', 'vad_acc', 'vdr', 'vf1', 'vfa', 'rpa', 'rca', 'gross', 'median_cents']
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for tag, m in all_results['pitch'].items():
                if tag.startswith('_'):
                    continue
                writer.writerow({'condition': tag, **m})
        print(f"CSV saved to {args.csv}")


if __name__ == "__main__":
    main()
