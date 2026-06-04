"""
VocalSet Feature Extraction
============================

Extracts mel / F0 / VAD / technique labels from VocalSet and saves two NPZ
files in the flat format expected by vocalcoach/train.py:

  technique_train.npz — training singers (default: all except test singers)
  technique_test.npz  — test singers  (default: m2, m4, f4, f8)

VocalSet directory structure (works for v1.1 and v1.2 — structure unchanged):
  VocalSet*/
    data_by_singer/
      m1/
        belt/         a_belt.wav, e_belt.wav, ...
        breathy/      a_breathy.wav, ...
        vibrato/      ...
        straight/     ...  (some versions spell it "strait")
        ...
      m2/
      f1/ ... f9/

VocalSet has 18 technique folders (as of v1.2):
  belt, breathy, fast_forte, fast_piano, forte, inhaled, lip_trill, messa,
  pp, slow_forte, slow_piano, spoken, straight, trill, trillo, vibrado,
  vibrato, vocal_fry

Technique → our label mapping
------------------------------
VocalSet folder  →  TECHNIQUE_NAMES index   Notes
  vibrato        →  0  (vibrato)
  vibrado        →  0  (vibrato)            Spanish spelling; same technique
  breathy        →  1  (breathy)
  belt           →  3  (belt)
  straight       →  4  (straight)
  strait         →  4  (straight)           Older VocalSet typo

Skipped (not in our 5-class taxonomy):
  trill, trillo    — rapid pitch alternation between two notes; distinct from
                     vibrato (which is continuous oscillation around one note)
  vocal_fry        — distinct register, not in scope for Phase 1
  forte, pp        — dynamic levels, not technique classes
  fast_forte, fast_piano, slow_forte, slow_piano — dynamic/tempo exercises
  inhaled, lip_trill, messa, spoken — exercises or non-singing

"falsetto" is absent from VocalSet; those labels come from GTSinger.

Test singer split
-----------------
No canonical split is defined in the VocalSet paper for technique
classification. The default here (m2, m4, f4, f8) follows the most commonly
used held-out set in published follow-up work (2 male, 2 female).
Use --test-singers to override.

Output NPZ schema (compatible with vocalcoach/train.py TechniqueDataset):
  mel:       (total_frames, 40)          float16  — log-mel
  f0:        (total_frames,)             float16  — Hz (0 = unvoiced)
  vad:       (total_frames,)             float16  — per-frame binary
  technique: (n_clips, N_TECHNIQUES)     float32  — clip-level binary labels
  lengths:   (n_clips,)                  int32    — frame count per clip
  singers:   (n_clips,)  object str      — singer id per clip (for analysis)

Usage
-----
  python scripts/extractVocalSet.py \\
      --dataset-dir data/VocalSet/data_by_singer \\
      --output-dir  data/vocalset \\
      --rmvpe-model rmvpe.pt \\
      --device cuda

  # Override test singers:
  python scripts/extractVocalSet.py \\
      --dataset-dir data/VocalSet/data_by_singer \\
      --output-dir  data/vocalset \\
      --test-singers m2 m4 f4 f8
"""

import argparse
import os
import sys

import librosa
import numpy as np
from tqdm import tqdm

# ── Constants — must match vocalcoach/model.py ────────────────────────
SR          = 16000
N_MELS      = 40
HOP_LENGTH  = 160       # 10 ms
WIN_LENGTH  = 400       # 25 ms
N_FFT       = 512
FMIN        = 31.7
FMAX        = 8000.0
VAD_TOP_DB  = 30

# Technique names from vocalcoach/model.py (order matters)
TECHNIQUE_NAMES = ['vibrato', 'breathy', 'falsetto', 'belt', 'straight']

# Maps VocalSet folder names → index in TECHNIQUE_NAMES.
# Only folders listed here produce training samples; all others are skipped.
VOCALSET_MAP = {
    'vibrato':  0,   # vibrato
    'vibrado':  0,   # vibrato — Spanish spelling, same technique (merge)
    'breathy':  1,   # breathy
    'belt':     3,   # belt
    'straight': 4,   # straight
    'strait':   4,   # straight — older VocalSet typo
}
# Intentionally skipped:
#   trill, trillo  — pitch alternation (≠ vibrato oscillation), not in taxonomy
#   vocal_fry      — distinct register, not in Phase 1 scope
#   forte, pp, fast_forte, fast_piano, slow_forte, slow_piano — dynamics, not technique
#   inhaled, lip_trill, messa, spoken — exercises / non-singing

# Default test singers: 2 male + 2 female, following the most common split in
# published VocalSet technique classification work.
DEFAULT_TEST_SINGERS = {'m2', 'm4', 'f4', 'f8'}


def parse_args():
    p = argparse.ArgumentParser(description="Extract VocalSet features with technique labels.")
    p.add_argument("--dataset-dir", required=True,
                   help="path to data_by_singer/ inside VocalSet1.1/")
    p.add_argument("--output-dir", default="data/vocalset",
                   help="output directory (default: data/vocalset)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--rmvpe-model", default="rmvpe.pt",
                   help="path to RMVPE checkpoint (default: rmvpe.pt)")
    p.add_argument("--test-singers", nargs="+", default=list(DEFAULT_TEST_SINGERS),
                   help="singer IDs to put in the test split "
                        "(default: m2 f4)")
    p.add_argument("--min-duration", type=float, default=0.5,
                   help="skip clips shorter than this many seconds (default: 0.5)")
    p.add_argument("--pitch-shift-semitones", type=float, nargs="*", default=[],
                   help="augment the TRAIN split with pitch-shifted copies (semitones, "
                        "e.g. -2 2). Technique is pitch-invariant (a belt stays a belt), "
                        "and f0/VAD/mel are re-derived from shifted audio, so all labels "
                        "stay exact. Test split is NEVER augmented. Crosses with "
                        "--time-stretch-rates as a grid.")
    p.add_argument("--time-stretch-rates", type=float, nargs="*", default=[],
                   help="augment the TRAIN split with time-stretched copies (rate factors, "
                        "e.g. 0.9 1.1). Technique is tempo-invariant; labels re-derived. "
                        "Test split is NEVER augmented.")
    return p.parse_args()


# ── Audio feature extraction ──────────────────────────────────────────

def extract_mel(y):
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        window="hann", center=True,
    )
    return librosa.power_to_db(mel, ref=1.0).T.astype(np.float32)   # (T, 40)


def extract_vad(y, n_frames):
    frame_vad = np.zeros(n_frames, dtype=np.float32)
    for s, e in librosa.effects.split(y, top_db=VAD_TOP_DB):
        sf = s // HOP_LENGTH
        ef = min(e // HOP_LENGTH, n_frames)
        if sf < ef:
            frame_vad[sf:ef] = 1.0
    return frame_vad


# ── Discovery ────────────────────────────────────────────────────────

def _dir_to_singer_id(dirname):
    """Convert 'female3' → 'f3', 'male10' → 'm10'. Returns None if not a singer dir."""
    import re
    m = re.fullmatch(r'(female|male)(\d+)', dirname, re.IGNORECASE)
    if m:
        return ('f' if m.group(1).lower() == 'female' else 'm') + m.group(2)
    # Already short-form (f1, m2, etc.)
    if re.fullmatch(r'[fm]\d+', dirname):
        return dirname
    return None


def discover_clips(dataset_dir):
    """Walk dataset_dir and return list of (singer_id, technique_idx, audio_path).

    Handles both flat structure (singer/technique/wav) and the VocalSet1-2
    structure (singer/exercise_type/technique/wav). Singer dirs may be named
    'female1'/'male1' or 'f1'/'m1'; both are normalised to the short form.
    """
    clips = []
    audio_exts = {'.wav', '.flac', '.mp3'}
    for singer_dir in sorted(os.listdir(dataset_dir)):
        singer_id = _dir_to_singer_id(singer_dir)
        if singer_id is None:
            continue
        singer_path = os.path.join(dataset_dir, singer_dir)
        if not os.path.isdir(singer_path):
            continue
        # Walk all subdirectories; pick up any folder whose name is a technique.
        for root, _, fnames in os.walk(singer_path):
            # Skip macOS AppleDouble junk (__MACOSX/ tree + "._name.wav" stubs)
            # that Mac-created zips carry — they aren't real audio.
            if "__MACOSX" in root.split(os.sep):
                continue
            tech_folder = os.path.basename(root)
            if tech_folder not in VOCALSET_MAP:
                continue
            tech_idx = VOCALSET_MAP[tech_folder]
            for fname in sorted(fnames):
                if os.path.splitext(fname)[1].lower() not in audio_exts:
                    continue
                if fname.startswith("._"):
                    continue
                clips.append((singer_id, tech_idx, os.path.join(root, fname)))
    return clips


# ── Per-split extraction ──────────────────────────────────────────────

def _augment_variants(y, sr, pitch_semitones, time_rates):
    """Yield (label, audio): the original plus every pitch×time variant.

    Transforms are applied to raw audio BEFORE feature extraction, so mel/F0/VAD
    re-derive correctly aligned. Technique labels are pitch- and tempo-invariant,
    so they carry over unchanged. The unmodified original is always included."""
    pitches = [0.0] + [s for s in pitch_semitones if abs(s) > 1e-6]
    rates   = [1.0] + [r for r in time_rates if abs(r - 1.0) > 1e-6]
    for n_steps in pitches:
        y_p = (y if abs(n_steps) < 1e-6
               else librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps))
        for rate in rates:
            y_pr = (y_p if abs(rate - 1.0) < 1e-6
                    else librosa.effects.time_stretch(y=y_p, rate=rate))
            tag = f"p{n_steps:+g}_r{rate:g}" if (n_steps or rate != 1.0) else "orig"
            yield tag, y_pr


def extract_split(clips, rmvpe, device, min_frames, split_name,
                  pitch_semitones=(), time_rates=()):
    mel_chunks, f0_chunks, vad_chunks = [], [], []
    technique_labels, lengths, singer_ids = [], [], []
    skipped = 0

    for singer, tech_idx, fpath in tqdm(clips, desc=f"  {split_name}"):
        try:
            y, _ = librosa.load(fpath, sr=SR, mono=True)
        except Exception as exc:
            print(f"    [skip] {os.path.basename(fpath)}: {exc}")
            skipped += 1
            continue

        if len(y) < min_frames * HOP_LENGTH:
            skipped += 1
            continue

        # Clip-level binary technique label — same for every augmented variant.
        label = np.zeros(len(TECHNIQUE_NAMES), dtype=np.float32)
        label[tech_idx] = 1.0

        for _tag, y_var in _augment_variants(y, SR, pitch_semitones, time_rates):
            log_mel = extract_mel(y_var)                                     # (T_mel, 40)
            f0_hz   = rmvpe.infer_from_audio(
                y_var, sample_rate=SR, device=device).astype(np.float32)     # (T_rmvpe,)
            T       = min(len(log_mel), len(f0_hz))
            if T < min_frames:
                skipped += 1
                continue

            log_mel = log_mel[:T]
            f0_hz   = f0_hz[:T]
            frame_vad = extract_vad(y_var, T)

            mel_chunks.append(log_mel)
            f0_chunks.append(f0_hz)
            vad_chunks.append(frame_vad)
            technique_labels.append(label.copy())
            lengths.append(T)
            singer_ids.append(singer)

    return (mel_chunks, f0_chunks, vad_chunks,
            technique_labels, lengths, singer_ids, skipped)


# ── Save ─────────────────────────────────────────────────────────────

def save_split(output_path, mel_chunks, f0_chunks, vad_chunks,
               technique_labels, lengths, singer_ids):
    if not mel_chunks:
        print(f"  [warn] No clips — skipping {output_path}")
        return

    mel_all  = np.concatenate(mel_chunks).astype(np.float16)   # (total_frames, 40)
    f0_all   = np.concatenate(f0_chunks).astype(np.float16)    # (total_frames,)
    vad_all  = np.concatenate(vad_chunks).astype(np.float16)   # (total_frames,)
    tech_all = np.stack(technique_labels)                       # (n_clips, N_TECH)
    len_arr  = np.array(lengths, dtype=np.int32)               # (n_clips,)
    singer_arr = np.array(singer_ids, dtype=object)

    np.savez(
        output_path,
        mel=mel_all, f0=f0_all, vad=vad_all,
        technique=tech_all,
        lengths=len_arr,
        singers=singer_arr,
    )
    total_frames = len(mel_all)
    voiced_pct   = float(np.mean(vad_all > 0)) * 100
    print(f"  Saved {output_path}")
    print(f"    clips={len(lengths)}, frames={total_frames:,} "
          f"({total_frames * HOP_LENGTH / SR / 3600:.2f} hrs), "
          f"voiced={voiced_pct:.1f}%")
    print(f"    technique distribution:")
    for i, name in enumerate(TECHNIQUE_NAMES):
        n = int(tech_all[:, i].sum())
        print(f"      {name:<12}: {n:4d} clips")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    test_singers = set(args.test_singers)
    min_frames   = int(args.min_duration * SR / HOP_LENGTH)

    try:
        from rmvpe import RMVPE
    except ImportError:
        raise SystemExit(
            "RMVPE not installed. Run: pip install rmvpe")

    if not os.path.exists(args.rmvpe_model):
        raise SystemExit(
            f"RMVPE checkpoint not found: {args.rmvpe_model}\n"
            "Download from: https://huggingface.co/lj1995/VoiceConversionWebUI")

    print(f"Loading RMVPE from {args.rmvpe_model} ...")
    rmvpe = RMVPE(args.rmvpe_model, hop_length=HOP_LENGTH)

    print(f"Discovering clips in {args.dataset_dir} ...")
    all_clips = discover_clips(args.dataset_dir)
    print(f"Found {len(all_clips)} clips across "
          f"{len(set(c[0] for c in all_clips))} singers")

    train_clips = [(s, t, p) for s, t, p in all_clips if s not in test_singers]
    test_clips  = [(s, t, p) for s, t, p in all_clips if s in test_singers]
    print(f"Train: {len(train_clips)} clips | Test: {len(test_clips)} clips "
          f"(test singers: {sorted(test_singers)})")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.pitch_shift_semitones or args.time_stretch_rates:
        n_var = (len(args.pitch_shift_semitones) + 1) * (len(args.time_stretch_rates) + 1)
        print(f"\nTRAIN-split augmentation: pitch={args.pitch_shift_semitones or '—'} "
              f"time={args.time_stretch_rates or '—'} → {n_var}× variants/clip "
              f"(test split NOT augmented)")

    print("\nExtracting training split ...")
    tr = extract_split(train_clips, rmvpe, args.device, min_frames, "train",
                       pitch_semitones=args.pitch_shift_semitones,
                       time_rates=args.time_stretch_rates)
    save_split(
        os.path.join(args.output_dir, "technique_train.npz"),
        *tr[:-1],   # mel, f0, vad, technique, lengths, singers
    )
    print(f"  Skipped: {tr[-1]}")

    print("\nExtracting test split ...")
    # Test split is never augmented — augmenting eval data inflates metrics.
    te = extract_split(test_clips, rmvpe, args.device, min_frames, "test")
    save_split(
        os.path.join(args.output_dir, "technique_test.npz"),
        *te[:-1],   # mel, f0, vad, technique, lengths, singers
    )
    print(f"  Skipped: {te[-1]}")

    print("\nDone. Files written to:", args.output_dir)
    print("Technique label mapping used:")
    for folder, idx in sorted(VOCALSET_MAP.items()):
        print(f"  {folder:<10} → {TECHNIQUE_NAMES[idx]}")
    print("(falsetto labels absent from VocalSet — use GTSinger for those)")


if __name__ == "__main__":
    main()
