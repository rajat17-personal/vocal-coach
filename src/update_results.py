"""
update_results.py
=================

Evaluates a completed VocalCoach run and upserts rows into
VOCALCOACH_RESULTS.md (four tables: Runs, Leaderboard,
Per-condition offRPA, Per-technique F1).

Usage
-----
  # Pitch-only run
  python vocalcoach/update_results.py \\
      --run-dir vocalcoach/runs/tcn_gtsinger_noncausal \\
      --data-dir data

  # With technique evaluation
  python vocalcoach/update_results.py \\
      --run-dir vocalcoach/runs/tcn_gtsinger_noncausal \\
      --data-dir data \\
      --technique-dir data/vocalset

  # Delete one or more runs from all tables
  python vocalcoach/update_results.py --delete tcn_gtsinger_noncausal
  python vocalcoach/update_results.py --delete run_a run_b run_c

  # Preview without writing
  python vocalcoach/update_results.py \\
      --run-dir vocalcoach/runs/tcn_gtsinger_noncausal \\
      --data-dir data --print-only
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile

import torch

# ── VocalCoach training defaults (must match train.py argparse defaults) ──────
TRAIN_DEFAULTS = {
    # Architecture
    "arch": "tcn", "causal": False, "hidden": None, "n_blocks": None,
    "deep_technique_head": False,
    # Training strategy
    "epochs": 100, "batch_size": 32, "lr": 3e-4, "seq_len": 300,
    "num_workers": 4,
    "probe_mode": False, "freeze_backbone_epochs": 0, "freeze_n_blocks": 0,
    "lr_backbone": None,
    "balance_datasets": False,
    "curriculum": False, "curriculum_warmup": 30, "curriculum_ramp": 10,
    "scheduler": "cosine_warmup", "grad_clip": 5.0,
    # Loss weights
    "w_vad": 0.5, "w_pitch": 1.0, "w_technique": 2.0,
    "vad_pos_weight": 2.3, "technique_clip_weight": 1.0,
    "technique_pos_weights": None,
    "pitch_sigma": 1.2,
    # Technique contrastive (SupCon)
    "contrastive_technique": False, "w_contrastive_technique": 0.5,
    # Note head (Variant 4)
    "note_head": False, "deep_note_head": False, "note_probe": False,
    "w_note": 1.0, "note_pos_weight": 1.0, "note_batch_size": 16, "w_note_metric": 1.0,
    # Quality scoring head
    "quality_variant": 0,
    "quality_pairs_npz": None, "quality_mse_npz": None, "quality_ccmusic_npz": None,
    "ranking_margin": 0.5, "w_ranking": 1.0, "w_quality_mse": 1.0,
    "quality_epochs_mse": 30,
    # Augmentation
    "augment": "none",
    "snr_range": [-10.0, 30.0], "p_clean": 0.0, "snr_bias": 1.0,
    "gain_aug_db": 0.0, "spec_tilt_db": 0.0,
    "freq_mask_param": 4, "n_freq_masks": 2,
    "time_mask_param": 10, "n_time_masks": 2,
    # Misc
    "eval_every": 5, "patience": 0,
    "resume": None,
}

# Args shown in "Key args" column — hyperparams only; data paths shown separately.
INTERESTING_ARGS = [
    # Architecture
    "arch", "causal", "hidden", "n_blocks",
    "deep_technique_head",
    # Training strategy
    "seq_len", "epochs", "batch_size", "lr", "lr_backbone",
    "probe_mode", "freeze_backbone_epochs", "freeze_n_blocks",
    "balance_datasets",
    "curriculum", "curriculum_warmup", "curriculum_ramp",
    "scheduler", "grad_clip",
    # Loss weights
    "w_vad", "w_pitch", "w_technique",
    "vad_pos_weight", "technique_clip_weight",
    "technique_pos_weights",
    "pitch_sigma",
    # Technique contrastive (SupCon)
    "contrastive_technique", "w_contrastive_technique",
    # Note head (Variant 4)
    "note_head", "deep_note_head", "note_probe",
    "w_note", "note_pos_weight", "note_batch_size", "w_note_metric",
    # Quality scoring head (NPZ paths excluded — shown as data sources)
    "quality_variant", "ranking_margin", "w_ranking", "w_quality_mse",
    "quality_epochs_mse",
    # Augmentation
    "augment", "snr_range", "p_clean", "snr_bias",
    "gain_aug_db", "spec_tilt_db",
    "freq_mask_param", "n_freq_masks", "time_mask_param", "n_time_masks",
    # Misc
    "patience", "resume",
]

# Short labels for known dataset paths shown in Key args data-source field.
_DATA_LABELS = {
    None:                      "—",
    "data":                    "GTSinger",
    "data/vocalset":           "VocalSet",
    "data/gtsinger_technique": "GTSinger-tech",
}

TECHNIQUE_NAMES = ["vibrato", "breathy", "falsetto", "belt", "straight"]
CONDITIONS      = ["-5 dB", "+0 dB", "+5 dB", "+10 dB", "+20 dB", "clean"]

# ── NanoPitch reference scores (from NanoPitch RESULTS.MD) ────────────────────
# Run 2  `wpitch2.0`   GRU-64, no augmentation — architecture-only comparison
#   Trained: --w-pitch 2.0  (all other args at NanoPitch defaults)
#   NanoPitch path (no longer on disk): training/runs/wpitch2.0
#
# Run 26 `seq600_btch16_wVad0.05wPitch2_cosine_specaug_n10t30SNR_pitchsigma0.8`
#   GRU-64 + noise+SpecAugment — best balanced run (no clean-VAD trick)
#   Trained: --seq-len 600 --epochs 100 --batch-size 16 --w-vad 0.05
#            --w-pitch 2.0 --scheduler cosine_warmup
#            --augment noise_specaug --snr-range -10 30 --pitch-sigma 0.8
#   NanoPitch path (no longer on disk): training/runs/seq600_btch16_...
NANOPITCH_REF = {
    # Tier 1: architecture comparison — no augmentation, GTSinger only
    "np_run2_arch": {
        "label":   "NanoPitch Run 2  (GRU-64, no aug — arch baseline)",
        "vad_acc": 80.2,   # VAD accuracy %
        "rt_rpa":  91.6,   # macro RPA %
        "rt_vdr":  69.3,   # VDR %
        "rt_med":  31.0,   # median pitch error ¢
        "conditions": {    # per-condition RPA %
            "-5 dB": 89.0, "+0 dB": 89.5, "+5 dB": 88.5,
            "+10 dB": 92.3, "+20 dB": 93.9, "clean": 96.0,
        },
    },
    # Tier 2: best-effort comparison — noise+SpecAugment
    "np_run26_aug": {
        "label":   "NanoPitch Run 26 (GRU-64, noise+SpecAug — best balanced)",
        "vad_acc": 81.6,
        "rt_rpa":  96.1,
        "rt_vdr":  65.9,
        "rt_med":  15.8,
        "conditions": {
            "-5 dB": 95.1, "+0 dB": 94.3, "+5 dB": 94.4,
            "+10 dB": 96.9, "+20 dB": 97.9, "clean": 98.1,
        },
    },
}

# Regex matching any data row in any table (| N | `name` | ...).
DATA_ROW_RE = re.compile(r"^\|\s*[^|]*\|\s*`([^`]+)`\s*\|")


# ── Formatting helpers ─────────────────────────────────────────────────────────

def fmt_val(v):
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, list):
        return " ".join(fmt_val(x) for x in v)
    if isinstance(v, bool):
        return "on" if v else "off"
    return str(v)


def _parse_num(cell):
    m = re.match(r"\s*([\d.]+)", cell.strip())
    return float(m.group(1)) if m else None


def _delta(val, base):
    if base is None or val is None:
        return f"{val:.1f}" if val is not None else "—"
    d = val - base
    sign = "+" if d >= 0 else ""
    return f"{val:.1f} ({sign}{d:.1f})"


def _pct(v):
    return f"{v*100:.1f}" if v is not None and v == v else "—"


def _f(v, decimals=3):
    return f"{v:.{decimals}f}" if v is not None and v == v else "—"


# ── Diff args against defaults ────────────────────────────────────────────────

def _resume_base(path):
    p = os.path.normpath(path)
    parts = p.split(os.sep)
    if "checkpoints" in parts:
        i = parts.index("checkpoints")
        if i > 0:
            return parts[i - 1]
    return os.path.basename(p)


def diff_args(args, data_dir=None, technique_dirs=None):
    out = []

    # Data sources first — always shown regardless of defaults.
    def _label(p):
        if p is None:
            return None
        key = os.path.normpath(p).replace(os.sep, "/")
        # Try exact match, then basename fallback.
        for candidate in (key, os.path.basename(key)):
            if candidate in _DATA_LABELS:
                return _DATA_LABELS[candidate]
        return os.path.basename(key)

    pitch_src = _label(data_dir)
    tech_srcs = [_label(d) for d in (technique_dirs or [])]
    if pitch_src:
        out.append(f"data={pitch_src}")
    if tech_srcs:
        out.append(f"tech={'+'.join(t for t in tech_srcs if t)}")

    # Hyperparameter diffs vs defaults.
    for k in INTERESTING_ARGS:
        if k not in args:
            continue
        v = args[k]
        dv = TRAIN_DEFAULTS.get(k)
        if v == dv:
            continue
        if k == "resume" and v:
            out.append(f"`--resume`={_resume_base(v)}")
        else:
            out.append(f"`--{k.replace('_', '-')}`={fmt_val(v)}")
    return out


# ── Run evaluate.py, return parsed JSON ───────────────────────────────────────

def run_evaluate(ckpt_path, data_dir, technique_dir, label="", voicing_threshold=0.3, onset_penalty=1.0):
    script = os.path.join(os.path.dirname(__file__), "evaluate.py")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        json_path = tmp.name

    cmd = [sys.executable, "-u", script,
           "--checkpoint", ckpt_path,
           "--json",       json_path,
           "--voicing-threshold", str(voicing_threshold),
           "--onset-penalty", str(onset_penalty)]
    if data_dir:
        cmd += ["--data-dir", data_dir]
    if technique_dir:
        cmd += ["--technique-dir", technique_dir]

    if label:
        print(f"\n── {label} ──")
    print(f"Checkpoint: {os.path.basename(ckpt_path)}")
    print(f"Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=False, text=True)
    if res.returncode != 0:
        sys.exit(f"evaluate.py failed (exit {res.returncode})")

    with open(json_path) as f:
        data = json.load(f)
    os.unlink(json_path)
    return data


# ── Extract metrics from evaluate JSON ───────────────────────────────────────

def extract_pitch(ev):
    pitch = ev.get("pitch", {})
    return {
        "macro_rpa": pitch.get("_macro_rpa"),
        "conditions": {c: pitch.get(c) for c in CONDITIONS},
    }


def extract_technique(ev, key="technique"):
    tech = ev.get(key, {})
    if not tech:
        return None
    per = {n: tech.get(n) for n in TECHNIQUE_NAMES}
    return {
        "per_class": per,
        "macro_f1":     tech.get("_macro_f1"),
        "macro_ap":     tech.get("_macro_ap"),
        "clip_accuracy": tech.get("_clip_accuracy"),
    }


def extract_overall(ev):
    pitch = ev.get("pitch", {})
    import numpy as np
    conds = [pitch[c] for c in CONDITIONS if c in pitch]
    def avg(key):
        vals = [c[key] for c in conds if c and key in c and c[key] == c[key]]
        return float(np.mean(vals)) if vals else float("nan")
    return {
        "vad_acc": avg("vad_acc"),
        "rt_rpa":  avg("rpa"),
        "rt_vdr":  avg("vdr"),
        "rt_vf1":  avg("vf1"),
        "rt_vfa":  avg("vfa"),
        "rt_med":  avg("median_cents"),
    }


# ── Markdown section helpers ──────────────────────────────────────────────────

def _section_slice(lines, title):
    start = None
    for i, line in enumerate(lines):
        if line.strip() == f"## {title}":
            start = i + 1
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break
    header_end = start
    while header_end < end and not DATA_ROW_RE.match(lines[header_end]):
        header_end += 1
    data_end = header_end
    while data_end < end and DATA_ROW_RE.match(lines[data_end]):
        data_end += 1
    rows = [l.rstrip("\n") for l in lines[header_end:data_end]]
    return header_end, data_end, rows


def _write_section(path, title, new_rows):
    with open(path) as f:
        lines = f.readlines()
    sl = _section_slice(lines, title)
    if sl is None:
        sys.exit(f"Section '## {title}' not found in {path}")
    he, de, _ = sl
    # Strip blank lines immediately before data rows so the markdown table
    # separator stays adjacent to its data (blank lines break table rendering).
    header = list(lines[:he])
    while header and header[-1].strip() == "":
        header.pop()
    with open(path, "w") as f:
        f.writelines(header + [r + "\n" for r in new_rows] + lines[de:])


def _renumber(rows):
    out = []
    for i, row in enumerate(rows, 1):
        out.append(re.sub(r"^\|\s*[^|]*\|", f"| {i} |", row, count=1))
    return out


def _renumber_from_map(rows, name_to_num):
    out = []
    for row in rows:
        m = DATA_ROW_RE.match(row)
        name = m.group(1) if m else None
        label = str(name_to_num[name]) if name in name_to_num else "—"
        out.append(re.sub(r"^\|\s*[^|]*\|", f"| {label} |", row, count=1))
    return out


def _runs_name_to_number(path):
    with open(path) as f:
        lines = f.readlines()
    sl = _section_slice(lines, "Runs")
    if sl is None:
        return {}
    _, _, rows = sl
    out = {}
    num_re = re.compile(r"^\|\s*(\d+)\s*\|")
    for row in rows:
        nm = DATA_ROW_RE.match(row)
        nu = num_re.match(row)
        if nm and nu:
            out[nm.group(1)] = int(nu.group(1))
    return out


def _extract_baseline(path):
    with open(path) as f:
        lines = f.readlines()
    sl = _section_slice(lines, "Runs")
    if sl is None:
        return None
    _, _, rows = sl
    for row in rows:
        m = DATA_ROW_RE.match(row)
        if m and m.group(1) == "baseline":
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            # cells: [num, name, arch, key_args, note, vad, rpa, vdr, vf1, med, mf1, map]
            if len(cells) >= 9:
                return {
                    "vad":    _parse_num(cells[5]),
                    "rt_rpa": _parse_num(cells[6]),
                    "rt_vdr": _parse_num(cells[7]),
                    "rt_med": _parse_num(cells[9]) if len(cells) > 9 else _parse_num(cells[8]),
                }
    return None


def _all_runs_rows(path):
    with open(path) as f:
        lines = f.readlines()
    sl = _section_slice(lines, "Runs")
    if sl is None:
        return []
    _, _, rows = sl
    return rows


def upsert(path, section, run_name, new_row, sort_key=None,
           renumber_from=None):
    with open(path) as f:
        lines = f.readlines()
    sl = _section_slice(lines, section)
    if sl is None:
        sys.exit(f"Section '## {section}' not found in {path}")
    _, _, rows = sl
    replaced = False
    for i, row in enumerate(rows):
        m = DATA_ROW_RE.match(row)
        if m and m.group(1) == run_name:
            rows[i] = new_row
            replaced = True
            break
    if not replaced:
        rows.append(new_row)
    if sort_key is not None:
        rows.sort(key=sort_key, reverse=True)
    # Pin first run (baseline) to top in Runs section only.
    if section == "Runs" and rows:
        first_name = DATA_ROW_RE.match(rows[0])
        if first_name and first_name.group(1) != run_name:
            pass  # keep insertion order
    rows = (_renumber_from_map(rows, renumber_from)
            if renumber_from is not None else _renumber(rows))
    _write_section(path, section, rows)
    print(f"{'Updated' if replaced else 'Appended'} '{run_name}' → '{section}'")


# ── Row builders ──────────────────────────────────────────────────────────────

def _arch_cell(ckpt_args, ckpt):
    arch   = ckpt_args.get("arch",   ckpt.get("arch",   "tcn"))
    causal = ckpt_args.get("causal", ckpt.get("causal", False))
    c_str  = "causal" if causal else "noncausal"
    return f"{arch}/{c_str}"


def _pct(v, decimals=1):
    """Format a [0,1] fraction as a percentage string, or '—' if nan/None."""
    if v is None or math.isnan(v): return "—"
    return f"{v*100:.{decimals}f}"


def make_runs_row(num, name, arch_cell, args_diff, note,
                  overall, tech, baseline=None):
    ka = ", ".join(args_diff) if args_diff else "defaults"
    note_cell = note or "_tbd_"
    b = baseline or {}
    vad_s = _delta(overall["vad_acc"] * 100, b.get("vad"))
    rpa_s = _delta(overall["rt_rpa"]  * 100, b.get("rt_rpa"))
    vdr_s = _delta(overall["rt_vdr"]  * 100, b.get("rt_vdr"))
    vf1_s = _pct(overall.get("rt_vf1"))
    vfa_s = _pct(overall.get("rt_vfa"))
    med_s = _delta(overall["rt_med"],         b.get("rt_med"))
    mf1_s = _f(tech["macro_f1"])   if tech else "—"
    map_s = _f(tech["macro_ap"])   if tech else "—"
    return (f"| {num} | `{name}` | {arch_cell} | {ka} | {note_cell} | "
            f"{vad_s} | {rpa_s} | {vdr_s} | {vf1_s} | {vfa_s} | {med_s} | {mf1_s} | {map_s} |")


def make_leaderboard_row(num, name, arch_cell, overall, tech, primary):
    vad_s = _pct(overall.get("vad_acc"))
    rpa_s = _pct(overall.get("rt_rpa"))
    vdr_s = _pct(overall.get("rt_vdr"))
    vf1_s = _pct(overall.get("rt_vf1"))
    vfa_s = _pct(overall.get("rt_vfa"))
    med_s = f"{overall['rt_med']:.1f}" if not math.isnan(overall.get("rt_med", float('nan'))) else "—"
    mf1_s = _f(tech["macro_f1"])  if tech else "—"
    map_s = _f(tech["macro_ap"])  if tech else "—"
    pri_s = f"{primary:.4f}"
    return (f"| {num} | `{name}` | {arch_cell} | "
            f"{vad_s} | {rpa_s} | {vdr_s} | {vf1_s} | {vfa_s} | {med_s} | "
            f"{mf1_s} | {map_s} | {pri_s} |")


def make_rpa_row(num, name, pitch):
    def c(cond):
        d = pitch["conditions"].get(cond)
        return f"{d['rpa']*100:.1f}" if d else "—"
    macro = f"{pitch['macro_rpa']*100:.1f}" if pitch["macro_rpa"] else "—"
    return (f"| {num} | `{name}` | "
            f"{c('-5 dB')} | {c('+0 dB')} | {c('+5 dB')} | "
            f"{c('+10 dB')} | {c('+20 dB')} | {c('clean')} | {macro} |")


def make_technique_row(num, name, tech, gt_tech=None):
    def c(cls, src):
        if src is None:
            return "—"
        d = src["per_class"].get(cls)
        if d is None:
            return "—"
        f1 = d.get("f1")
        return _f(f1) if f1 is not None else "—"
    mf1 = _f(tech["macro_f1"]) if tech else "—"
    map_ = _f(tech.get("macro_ap")) if tech else "—"
    clip_acc = tech.get("clip_accuracy") if tech else None
    clip_s = f"{clip_acc:.1%}" if clip_acc is not None and clip_acc == clip_acc else "—"
    return (f"| {num} | `{name}` | "
            + " | ".join(c(n, tech) for n in TECHNIQUE_NAMES)
            + f" | {mf1} | {map_} | {clip_s} |")


def make_gt_technique_row(num, name, gt_tech):
    """Per-class F1 row for GTSinger held-out set (vibrato/breathy/falsetto only)."""
    def c(cls):
        d = gt_tech["per_class"].get(cls)
        if d is None:
            return "—"
        f1 = d.get("f1")
        return _f(f1) if f1 is not None else "—"
    mf1 = _f(gt_tech["macro_f1"])
    map_ = _f(gt_tech.get("macro_ap"))
    clip_acc = gt_tech.get("clip_accuracy")
    clip_s = f"{clip_acc:.1%}" if clip_acc is not None and clip_acc == clip_acc else "—"
    # Only the 3 GTSinger classes (belt/straight absent)
    return (f"| {num} | `{name}` | "
            f"{c('vibrato')} | {c('breathy')} | {c('falsetto')} | "
            f"{mf1} | {map_} | {clip_s} |")


# ── Rebuild leaderboard from Runs rows ───────────────────────────────────────

def _rebuild_leaderboard(path):
    """Re-read all Runs rows, extract numerics, re-sort, write Leaderboard."""
    with open(path) as f:
        lines = f.readlines()
    sl = _section_slice(lines, "Runs")
    if sl is None:
        return
    _, _, runs_rows = sl

    lb_rows = []
    for row in runs_rows:
        m = DATA_ROW_RE.match(row)
        if not m:
            continue
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        # cells: [num, name, arch, key_args, note, vad, rpa, vdr, vf1, vfa, med, mf1, map]
        #  idx:    0     1     2      3         4    5    6    7    8    9    10   11   12
        if len(cells) < 11:
            continue
        name     = m.group(1)
        arch     = cells[2]
        vad_num  = _parse_num(cells[5])
        rpa_num  = _parse_num(cells[6])
        vdr_num  = _parse_num(cells[7])
        n = len(cells)
        # Support old rows (11 cols: no vf1/vfa), new+vf1 (12 cols), new+vf1+vfa (13 cols)
        vf1_num  = _parse_num(cells[8])  if n > 11 else None
        vfa_num  = _parse_num(cells[9])  if n > 12 else None
        med_idx  = 10 if n > 12 else (9 if n > 11 else 8)
        mf1_idx  = med_idx + 1
        map_idx  = med_idx + 2
        med_num  = _parse_num(cells[med_idx]) if n > med_idx else None
        mf1_num  = _parse_num(cells[mf1_idx]) if n > mf1_idx else None
        map_num  = _parse_num(cells[map_idx]) if n > map_idx else None
        primary  = mf1_num if mf1_num is not None else (rpa_num or 0.0)
        vad_s = f"{vad_num:.1f}" if vad_num is not None else "—"
        rpa_s = f"{rpa_num:.1f}" if rpa_num is not None else "—"
        vdr_s = f"{vdr_num:.1f}" if vdr_num is not None else "—"
        vf1_s = f"{vf1_num:.1f}" if vf1_num is not None else "—"
        vfa_s = f"{vfa_num:.1f}" if vfa_num is not None else "—"
        med_s = f"{med_num:.1f}" if med_num is not None else "—"
        mf1_s = cells[mf1_idx] if n > mf1_idx else "—"
        map_s = cells[map_idx] if n > map_idx else "—"
        pri_s = f"{primary:.4f}"
        lb_row = (f"| _ | `{name}` | {arch} | "
                  f"{vad_s} | {rpa_s} | {vdr_s} | {vf1_s} | {vfa_s} | {med_s} | "
                  f"{mf1_s} | {map_s} | {pri_s} |")
        lb_rows.append((primary, lb_row, name))

    lb_rows.sort(key=lambda x: x[0], reverse=True)
    final = [r for _, r, _ in lb_rows]
    # Preserve run numbers from the Runs table (same as Per-condition offRPA).
    name_to_num = _runs_name_to_number(path)
    final = _renumber_from_map(final, name_to_num)
    _write_section(path, "Leaderboard", final)
    print(f"Rebuilt Leaderboard ({len(final)} rows).")


# ── Delete ────────────────────────────────────────────────────────────────────

def delete_row(path, run_name):
    sections = ["Runs", "Per-condition offRPA", "Per-technique F1",
                "Per-technique F1 (GTSinger held-out)", "Leaderboard"]
    removed_any = False
    for section in sections:
        with open(path) as f:
            lines = f.readlines()
        sl = _section_slice(lines, section)
        if sl is None:
            continue
        _, _, rows = sl
        new_rows, removed = [], False
        for row in rows:
            m = DATA_ROW_RE.match(row)
            if m and m.group(1) == run_name:
                removed = True
            else:
                new_rows.append(row)
        if not removed:
            continue
        removed_any = True
        new_rows = _renumber(new_rows)
        _write_section(path, section, new_rows)
        print(f"Removed '{run_name}' from '{section}'")

    # Re-sync per-condition and per-technique numbering to Runs.
    name_to_num = _runs_name_to_number(path)
    for section in ["Per-condition offRPA", "Per-technique F1",
                    "Per-technique F1 (GTSinger held-out)"]:
        with open(path) as f:
            lines = f.readlines()
        sl = _section_slice(lines, section)
        if sl is None:
            continue
        _, _, rows = sl
        _write_section(path, section, _renumber_from_map(rows, name_to_num))

    _rebuild_leaderboard(path)

    if not removed_any:
        print(f"No row matching '{run_name}' found.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Update VOCALCOACH_RESULTS.md")
    p.add_argument("--run-dir",       default=None)
    p.add_argument("--data-dir",      default="data")
    p.add_argument("--technique-dir", default=None,
                   help="directory with technique_test.npz (VocalSet held-out)")
    p.add_argument("--gtsinger-technique-dir", default=None,
                   help="directory with technique_test.npz (GTSinger held-out) "
                        "for cross-dataset generalisation metrics")
    p.add_argument("--checkpoint",    default="best_loss.pth",
                   help="checkpoint filename inside <run-dir>/checkpoints/ "
                        "(best_loss.pth = lowest training loss across all heads; "
                        "best_metric.pth = best eval F1 or RPA, can be premature "
                        "when technique F1 fires before pitch converges)")
    p.add_argument("--name",          default=None,
                   help="row label (default: run-dir basename)")
    p.add_argument("--note",          default=None)
    p.add_argument("--results-md",    default=None)
    p.add_argument("--voicing-threshold", type=float, default=0.3,
                   help="Viterbi frame-0 init threshold (minor effect — see --onset-penalty).")
    p.add_argument("--onset-penalty", type=float, default=1.0,
                   help="Viterbi voiced<->unvoiced transition cost. Default 1.0 suits multi-task "
                        "models with diffuse posteriors. Pitch-only models with sharper posteriors "
                        "diffuse pitch posteriors make the unvoiced state more competitive.")
    p.add_argument("--print-only",    action="store_true")
    p.add_argument("--delete",        nargs="+", default=None, metavar="NAME")
    args = p.parse_args()

    results_md = args.results_md or os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "VOCALCOACH_RESULTS.md")
    results_md = os.path.abspath(results_md)

    if args.delete:
        for name in args.delete:
            delete_row(results_md, name)
        return

    if not args.run_dir:
        sys.exit("--run-dir is required (or use --delete NAME)")

    ckpt_path = os.path.join(args.run_dir, "checkpoints", args.checkpoint)
    if not os.path.isfile(ckpt_path):
        fallback = os.path.join(args.run_dir, "checkpoints", "best_metric.pth")
        if os.path.isfile(fallback) and args.checkpoint != "best_metric.pth":
            print(f"[warn] {ckpt_path} not found; using {fallback}")
            ckpt_path = fallback
        else:
            sys.exit(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {}) or {}

    if ckpt_args.get("quality_variant", 0) > 0:
        print(f"[note] quality_variant={ckpt_args['quality_variant']} checkpoint — "
              "pitch/technique heads are frozen; eval metrics reflect the base checkpoint. "
              "Use --note to annotate the quality variant in the results table.")

    # Resolve paths — prefer CLI args, fall back to what was saved in checkpoint.
    data_dir_abs = os.path.abspath(args.data_dir)
    tech_dir_abs = os.path.abspath(args.technique_dir) if args.technique_dir else None
    gt_tech_dir_abs = (os.path.abspath(args.gtsinger_technique_dir)
                       if args.gtsinger_technique_dir else None)

    # Primary eval (pitch + VocalSet technique)
    ev = run_evaluate(ckpt_path, data_dir_abs, tech_dir_abs,
                      label="VocalSet + pitch eval",
                      voicing_threshold=args.voicing_threshold,
                      onset_penalty=args.onset_penalty)

    # GTSinger technique eval (cross-dataset generalisation).
    # data_dir=None skips redundant pitch eval — pitch results already captured above.
    ev_gt = {}
    if gt_tech_dir_abs:
        ev_gt = run_evaluate(ckpt_path, None, gt_tech_dir_abs,
                             label="GTSinger technique eval",
                             voicing_threshold=args.voicing_threshold,
                             onset_penalty=args.onset_penalty)

    pitch   = extract_pitch(ev)
    tech    = extract_technique(ev)
    gt_tech = extract_technique(ev_gt) if ev_gt else None
    overall = extract_overall(ev)

    primary = (tech["macro_f1"] if tech and tech["macro_f1"] == tech["macro_f1"]
               else overall["rt_rpa"])

    name      = args.name or os.path.basename(os.path.normpath(args.run_dir))
    arch_cell = _arch_cell(ckpt_args, ckpt)
    # Pass the actual data paths used so they appear in Key args.
    ckpt_data_dir  = ckpt_args.get("data_dir")
    ckpt_tech_dirs = ckpt_args.get("technique_dirs") or []
    args_diff = diff_args(ckpt_args,
                          data_dir=ckpt_data_dir,
                          technique_dirs=ckpt_tech_dirs)
    if args.voicing_threshold != 0.3:
        args_diff.append(f"`--voicing-threshold`={args.voicing_threshold}")
    if args.onset_penalty != 2.0:
        args_diff.append(f"`--onset-penalty`={args.onset_penalty}")
    baseline  = _extract_baseline(results_md)

    runs_row    = make_runs_row(
        "_", name, arch_cell, args_diff, args.note, overall, tech, baseline)
    lb_row      = make_leaderboard_row(
        "_", name, arch_cell, overall, tech, primary)
    rpa_row     = make_rpa_row("_", name, pitch)
    tech_row    = make_technique_row("_", name, tech) if tech else None
    gt_tech_row = make_gt_technique_row("_", name, gt_tech) if gt_tech else None

    # ── NanoPitch comparison (stdout only, pitch-only runs) ───────────────────
    # Shown only when no technique data — RPA comparison is the primary metric
    # then. Note: NP scores are realtime Viterbi; VC uses offline Viterbi, so
    # deltas are optimistic by ~1–2% independent of architecture.
    if tech is None:
        rpa_val = overall["rt_rpa"] * 100
        vdr_val = overall["rt_vdr"] * 100
        med_val = overall["rt_med"]
        print("\n--- NanoPitch comparison (NP scores = realtime Viterbi; VC = offline Viterbi) ---")
        print(f"  {'Metric':<12}  {'This run':>10}  {'NP Run2 (rt)':>14}  {'NP Run26 (rt)':>15}")
        for label, this_v, ref2, ref26 in [
            ("RPA %",  rpa_val, 91.6, 96.1),
            ("VDR %",  vdr_val, 69.3, 65.9),
            ("Med ¢",  med_val, 31.0, 15.8),
        ]:
            d2  = f"({this_v - ref2:+.1f})"
            d26 = f"({this_v - ref26:+.1f})"
            print(f"  {label:<12}  {this_v:>10.1f}  {ref2:>6.1f} {d2:>7}  {ref26:>6.1f} {d26:>7}")
        print()

    if args.print_only:
        return

    # 1. Upsert Runs (sequential renumber, no sort)
    upsert(results_md, "Runs", name, runs_row)

    # 2. Rebuild Leaderboard from Runs (always sorted by primary metric)
    _rebuild_leaderboard(results_md)

    # 3. Per-condition offRPA (sorted by Macro col, numbers synced to Runs)
    name_to_num = _runs_name_to_number(results_md)

    def rpa_sort(row):
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        return _parse_num(cells[-1]) or 0.0

    upsert(results_md, "Per-condition offRPA", name, rpa_row,
           sort_key=rpa_sort, renumber_from=name_to_num)

    def _f1_sort(row):
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        # columns: # | name | ... | mf1 | clip_acc  (mf1 is second-to-last)
        return _parse_num(cells[-2]) or 0.0

    # 4. Per-technique F1 — VocalSet eval (sorted by macro F1)
    if tech_row:
        upsert(results_md, "Per-technique F1", name, tech_row,
               sort_key=_f1_sort, renumber_from=name_to_num)

    # 5. Per-technique F1 (GTSinger held-out) — only when GTSinger eval was run
    if gt_tech_row:
        upsert(results_md, "Per-technique F1 (GTSinger held-out)", name, gt_tech_row,
               sort_key=_f1_sort, renumber_from=name_to_num)

    print(f"\nDone — {results_md}")


if __name__ == "__main__":
    main()
