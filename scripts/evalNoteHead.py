"""
Note-Head Evaluation
====================
Evaluates a VocalCoach checkpoint's note onset/offset heads against the
annotated VocalSet held-out set (note_test.npz, 203 clips).

The note head learns two per-frame binary sequences:
  note_onset[t]  = 1 where a new note begins
  note_offset[t] = 1 where a note ends
Combined with the pitch posteriorgram these segment continuous singing into
discrete note events.

Metric: onset / offset F1 with a tolerance window (default ±50 ms = ±5 frames
at the 10 ms hop), the standard mir_eval note-segmentation convention. A
predicted onset counts as a true positive if it lies within `tol` frames of a
reference onset (greedy 1-to-1 matching, each reference matched at most once).

NPZ schema (scripts/extractAnnotatedVocalSet.py):
  mel:          (total_frames, 40)  float16  — concatenated clips
  lengths:      (n_clips,)          int32
  note_onsets:  (n_notes,)          int32   — GLOBAL frame indices (into mel)
  note_offsets: (n_notes,)          int32   — GLOBAL frame indices
  note_clip:    (n_notes,)          int32   — clip index each note belongs to
  n_notes:      (n_clips,)          int32   — notes per clip

Usage
-----
  python scripts/evalNoteHead.py \\
      --checkpoint vocalcoach/runs/stage1_attn4_notehead/checkpoints/best_metric.pth \\
      --note-npz data/annotated_vocalset/note_test.npz \\
      --log results/note_log.json
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import build_model

HOP_MS = 10.0   # 10 ms hop (160 samples @ 16 kHz)


# ── Peak picking ────────────────────────────────────────────────────────────

def _pick_peaks(prob, threshold, min_distance):
    """Greedy local-maxima peak picking on a 1-D probability sequence.

    Returns frame indices where prob exceeds `threshold` and is a local max,
    enforcing a minimum spacing of `min_distance` frames between picks
    (highest-probability-first, standard for onset detection).
    """
    above = np.where(prob >= threshold)[0]
    if len(above) == 0:
        return np.array([], dtype=np.int64)
    # local maxima among above-threshold frames
    cand = [i for i in above
            if (i == 0 or prob[i] >= prob[i - 1])
            and (i == len(prob) - 1 or prob[i] >= prob[i + 1])]
    cand.sort(key=lambda i: -prob[i])
    picked, taken = [], np.zeros(len(prob), dtype=bool)
    for i in cand:
        lo, hi = max(0, i - min_distance), min(len(prob), i + min_distance + 1)
        if not taken[lo:hi].any():
            picked.append(i)
            taken[i] = True
    return np.array(sorted(picked), dtype=np.int64)


def _match_f1(pred, ref, tol):
    """Greedy 1-to-1 matching within ±tol frames. Returns (P, R, F1, tp)."""
    if len(ref) == 0 and len(pred) == 0:
        return 1.0, 1.0, 1.0, 0
    if len(ref) == 0 or len(pred) == 0:
        return 0.0, 0.0, 0.0, 0
    ref_taken = np.zeros(len(ref), dtype=bool)
    tp = 0
    for p in pred:
        # nearest unmatched reference within tolerance
        dists = np.abs(ref - p)
        dists[ref_taken] = tol + 1
        j = int(np.argmin(dists))
        if dists[j] <= tol:
            ref_taken[j] = True
            tp += 1
    precision = tp / len(pred)
    recall    = tp / len(ref)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1, tp


# ── Inference ───────────────────────────────────────────────────────────────

# Pitch posteriorgram geometry — must match vocalcoach/model.py.
PITCH_FMIN = 31.7           # Hz, bin 0
PITCH_CENTS_PER_BIN = 20.0  # 20 cents/bin


def _bin_to_midi(bin_idx):
    """Pitch-posteriorgram bin → MIDI note number.
    bin → Hz: FMIN * 2**(bin*cents_per_bin/1200); Hz → MIDI: 69 + 12*log2(f/440)."""
    hz = PITCH_FMIN * 2.0 ** (bin_idx * PITCH_CENTS_PER_BIN / 1200.0)
    return 69.0 + 12.0 * np.log2(hz / 440.0)


def _predict_clip(model, mel_clip, device, chunk=1200):
    """Run model over one clip (possibly chunked). Returns
    (onset_prob (T,), offset_prob (T,), pitch_post (T, n_bins))."""
    onsets, offsets, pitches = [], [], []
    with torch.no_grad():
        for s in range(0, len(mel_clip), chunk):
            c = mel_clip[s:s + chunk].astype(np.float32)
            t = torch.from_numpy(c).unsqueeze(0).to(device)
            out = model(t)
            pitch, on, off = out[1], out[4], out[5]   # pitch (1,T,bins); on/off (1,T,1)
            if on is None or off is None:
                raise RuntimeError("Checkpoint has no note head "
                                   "(train with --note-head).")
            onsets.append(on.squeeze(0).squeeze(-1).cpu().numpy())
            offsets.append(off.squeeze(0).squeeze(-1).cpu().numpy())
            pitches.append(pitch.squeeze(0).cpu().numpy())
    return (np.concatenate(onsets), np.concatenate(offsets),
            np.concatenate(pitches, axis=0))


def _octave_collapse(midi_per_frame):
    """Fold per-frame MIDI to a single octave: snap each frame to the nearest
    octave of the span's median, then re-median. A cheap guard against per-frame
    octave/harmonic jumps in the posteriorgram argmax (the dominant note-with-pitch
    error — argmax is octave-confused on a minority of frames even where overall
    RPA is high)."""
    m = midi_per_frame[~np.isnan(midi_per_frame)]
    if len(m) == 0:
        return float("nan")
    center = np.median(m)
    folded = m + 12.0 * np.round((center - m) / 12.0)
    return float(np.median(folded))


def _note_pitch(pitch_post, onset, next_bound):
    """Predicted MIDI for a note spanning [onset, next_bound): per-frame argmax bin
    over the span → MIDI → octave-collapse → MEDIAN (robust to onset/offset
    transients and to per-frame octave jumps). Returns NaN for an empty span.

    Note: a Viterbi-smoothed f0 track was tried here but regressed note-with-pitch
    F1 on the real (predicted-onset) path — its voicing gate left ~half of detected
    note spans unvoiced, and where it did fire it sometimes locked onto a transition
    rather than the sustained pitch. Plain argmax + octave-collapse is the small but
    consistent net win (note-with-pitch F1 0.504 → 0.510)."""
    lo, hi = int(onset), int(max(onset + 1, next_bound))
    span = pitch_post[lo:hi]
    if len(span) == 0:
        return float("nan")
    midi = _bin_to_midi(span.argmax(axis=1).astype(float))
    return _octave_collapse(midi)


def _match_notes_with_pitch(pred_on, pred_midi, ref_on, ref_midi, tol, cent_tol):
    """Greedy 1-to-1 note matching: a predicted note is correct iff its onset is
    within `tol` frames AND its pitch within `cent_tol` cents of an unmatched
    reference note. Returns (precision, recall, f1, tp)."""
    if len(ref_on) == 0 and len(pred_on) == 0:
        return 1.0, 1.0, 1.0, 0
    if len(ref_on) == 0 or len(pred_on) == 0:
        return 0.0, 0.0, 0.0, 0
    order = np.argsort(pred_on)
    pred_on, pred_midi = np.asarray(pred_on)[order], np.asarray(pred_midi)[order]
    ref_taken = np.zeros(len(ref_on), dtype=bool)
    tp = 0
    for po, pm in zip(pred_on, pred_midi):
        best_j, best_d = -1, tol + 1
        for j in range(len(ref_on)):
            if ref_taken[j]:
                continue
            dt = abs(ref_on[j] - po)
            if dt > tol:
                continue
            cents = abs(1200.0 * np.log2(
                (440.0 * 2 ** ((pm - 69) / 12)) /
                (440.0 * 2 ** ((ref_midi[j] - 69) / 12))))
            if cents <= cent_tol and dt < best_d:
                best_j, best_d = j, dt
        if best_j >= 0:
            ref_taken[best_j] = True
            tp += 1
    precision = tp / len(pred_on)
    recall    = tp / len(ref_on)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1, tp


def evaluate(model, npz_path, device, tol_ms, threshold, min_distance,
             cent_tol=50.0):
    d = np.load(npz_path, allow_pickle=True)
    mel      = d["mel"]
    lengths  = d["lengths"].astype(np.int64)
    on_idx   = d["note_onsets"].astype(np.int64)
    off_idx  = d["note_offsets"].astype(np.int64)
    note_clip = d["note_clip"].astype(np.int64)
    has_midi  = "note_midi" in d
    midi_arr  = d["note_midi"].astype(np.float32) if has_midi else None

    tol = int(round(tol_ms / HOP_MS))

    print(f"\n  note_test: {len(lengths):,} clips, {len(on_idx):,} notes, "
          f"tol=±{tol_ms:.0f}ms (±{tol} frames)"
          + (f", pitch tol ±{cent_tol:.0f} cents" if has_midi else " (no note_midi — pitch eval skipped)"))

    agg = {"onset": dict(tp=0, n_pred=0, n_ref=0),
           "offset": dict(tp=0, n_pred=0, n_ref=0)}
    per_clip_f1 = {"onset": [], "offset": []}
    # note-with-pitch aggregate (onset+pitch must both match)
    np_tp = np_pred = np_ref = 0
    cent_errs = []   # |cents| for matched notes

    offset = 0
    for ci, L in enumerate(lengths):
        L = int(L)
        mel_clip = mel[offset:offset + L]
        clip_start = offset
        offset += L

        # reference onset/offset frames LOCAL to this clip
        sel = note_clip == ci
        ref_on  = on_idx[sel]  - clip_start
        ref_off = off_idx[sel] - clip_start
        in_range = (ref_on >= 0) & (ref_on < L)
        ref_on_r  = ref_on[in_range]
        ref_off   = ref_off[(ref_off >= 0) & (ref_off < L)]
        ref_midi_r = midi_arr[sel][in_range] if has_midi else None

        p_on, p_off, p_pitch = _predict_clip(model, mel_clip, device)
        pk_on  = _pick_peaks(p_on,  threshold, min_distance)
        pk_off = _pick_peaks(p_off, threshold, min_distance)

        for kind, pred, ref in (("onset", pk_on, ref_on_r),
                                ("offset", pk_off, ref_off)):
            _, _, f1, tp = _match_f1(pred, np.sort(ref), tol)
            agg[kind]["tp"]     += tp
            agg[kind]["n_pred"] += len(pred)
            agg[kind]["n_ref"]  += len(ref)
            if len(ref) > 0:
                per_clip_f1[kind].append(f1)

        # ── Note-with-pitch: each detected onset → pitch from the posteriorgram
        #    over [onset, next detected boundary). Match on onset AND pitch. ──
        if has_midi and len(ref_on_r) > 0:
            sorted_on = np.sort(pk_on)
            pred_midi = []
            for k, o in enumerate(sorted_on):
                nb = sorted_on[k + 1] if k + 1 < len(sorted_on) else L
                pred_midi.append(_note_pitch(p_pitch, o, nb))
            order = np.argsort(ref_on_r)
            _, _, _, tp = _match_notes_with_pitch(
                sorted_on, pred_midi, ref_on_r[order], ref_midi_r[order],
                tol, cent_tol)
            np_tp  += tp
            np_pred += len(sorted_on)
            np_ref  += len(ref_on_r)

    results = {}
    print(f"\n  {'Boundary':<10} {'Precision':>10} {'Recall':>9} {'F1 (micro)':>11} {'F1 (macro)':>11}")
    print(f"  {'-'*54}")
    for kind in ("onset", "offset"):
        a = agg[kind]
        p = a["tp"] / a["n_pred"] if a["n_pred"] else 0.0
        r = a["tp"] / a["n_ref"]  if a["n_ref"]  else 0.0
        micro = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        macro = float(np.mean(per_clip_f1[kind])) if per_clip_f1[kind] else 0.0
        results[kind] = {"precision": p, "recall": r,
                         "f1_micro": micro, "f1_macro": macro,
                         "tp": a["tp"], "n_pred": a["n_pred"], "n_ref": a["n_ref"]}
        print(f"  {kind:<10} {p:>10.3f} {r:>9.3f} {micro:>11.3f} {macro:>11.3f}")

    mean_f1 = float(np.mean([results["onset"]["f1_micro"],
                             results["offset"]["f1_micro"]]))
    print(f"  {'-'*54}")
    print(f"  Mean onset/offset F1 (micro): {mean_f1:.3f}")
    results["mean_f1_micro"] = mean_f1

    if has_midi:
        p = np_tp / np_pred if np_pred else 0.0
        r = np_tp / np_ref  if np_ref  else 0.0
        npf1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        results["note_with_pitch"] = {
            "precision": p, "recall": r, "f1": npf1,
            "tp": np_tp, "n_pred": np_pred, "n_ref": np_ref}
        print(f"  Note-with-pitch F1 (onset±{tol_ms:.0f}ms & pitch±{cent_tol:.0f}¢): "
              f"{npf1:.3f}  (P={p:.3f} R={r:.3f})")
    return results


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Evaluate VocalCoach note head")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--note-npz", default="data/annotated_vocalset/note_test.npz")
    p.add_argument("--tol-ms", type=float, default=50.0,
                   help="onset/offset matching tolerance in ms (default 50)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="peak-picking probability threshold (default 0.5)")
    p.add_argument("--min-distance", type=int, default=5,
                   help="min frames between detected boundaries (default 5 = 50ms)")
    p.add_argument("--cent-tol", type=float, default=50.0,
                   help="pitch tolerance in cents for note-with-pitch F1 (default 50, "
                        "the mir_eval convention). A detected note counts only if its "
                        "onset is within --tol-ms AND its pitch within this many cents.")
    p.add_argument("--log", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    warnings.warn("Loading checkpoint via torch.load — trusted source only.",
                  RuntimeWarning)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", "tcn")
    kwargs = dict(ckpt.get("model_kwargs", {}))
    causal = kwargs.pop("causal", ckpt.get("causal", False))
    model = build_model(arch, causal=causal, **kwargs).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    if not getattr(model, "has_note_head", False):
        print("ERROR: checkpoint has no note head (train with --note-head).")
        sys.exit(1)

    run_name = args.run_name or _infer_run_name(args.checkpoint)
    print(f"Loaded: VocalCoach{arch.upper()} epoch={ckpt.get('epoch','?')}  | {run_name}")

    w = 60
    print("\n" + "═" * w)
    print(f"  Note-Head Evaluation  |  {run_name}")
    print("═" * w)

    results = evaluate(model, args.note_npz, device,
                       args.tol_ms, args.threshold, args.min_distance,
                       cent_tol=args.cent_tol)
    print("═" * w)

    if args.log:
        record = {"run_name": run_name, "checkpoint": args.checkpoint,
                  "epoch": ckpt.get("epoch", "?"), "tol_ms": args.tol_ms,
                  "cent_tol": args.cent_tol, "threshold": args.threshold,
                  "timestamp": datetime.now(timezone.utc).isoformat(),
                  "results": results}
        log = {}
        if os.path.exists(args.log):
            with open(args.log) as f:
                log = json.load(f)
        log.setdefault("note_runs", []).append(record)
        os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
        with open(args.log, "w") as f:
            json.dump(log, f, indent=2)
        print(f"\n  Results appended → {args.log}")


def _infer_run_name(ckpt_path):
    parts = os.path.normpath(ckpt_path).split(os.sep)
    try:
        i = parts.index("runs")
        return f"{parts[i+1]}/{os.path.splitext(parts[-1])[0]}"
    except (ValueError, IndexError):
        return os.path.splitext(os.path.basename(ckpt_path))[0]


if __name__ == "__main__":
    main()
