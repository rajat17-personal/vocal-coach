"""
Feature Extraction Script for VocalCoach / NanoPitch Training
=============================================================

Extracts mel spectrograms, RMVPE F0, and VAD labels from a directory of
vocal audio files and saves them in the flat NPZ format expected by the
trainers (training/train.py and vocalcoach/train.py).

Output NPZ schema (compatible with clean.npz from NanoPitch-PreExtract):
    mel:     (total_frames, 40)  — log-mel spectrogram, float32
    f0:      (total_frames,)     — F0 in Hz (0.0 = unvoiced), float32
    vad:     (total_frames,)     — per-frame binary VAD (0.0 or 1.0), float32
    lengths: (n_clips,)          — frame count per clip, int64

Usage
-----
    python scripts/extractFeatures.py \\
        --dataset-dir path/to/vocals \\
        --output     data/clean.npz \\
        --device     cuda

    # For VocalSet / GTSinger with CUDA:
    python scripts/extractFeatures.py \\
        --dataset-dir data/vocalset/audio \\
        --output     data/vocalset_clean.npz \\
        --device     cuda

Notes
-----
- Audio is resampled to 16 kHz mono.
- Mel: 40 bands, 25 ms Hann window (win_length=400), 10 ms hop (hop_length=160),
  fmin=31.7 Hz (=PITCH_FMIN), fmax=8000 Hz. Stored as log-power (dB re 1.0).
- F0: extracted by RMVPE with the same 10 ms hop. Raw Hz values are stored;
  the posteriorgram is built on-the-fly at training time via f0_to_posteriorgram.
- VAD: RMS energy threshold at -30 dB (top_db=30) via librosa.effects.split,
  converted to a per-frame binary float32 array aligned to the mel hop grid.
- Temporal alignment: mel and RMVPE may differ by ±1 frame; the shorter length
  is used to keep arrays consistent.
"""

import argparse
import os

import librosa
import numpy as np
from tqdm import tqdm

# ── Constants — must match vocalcoach/model.py and training/model.py ──
SR          = 16000
N_MELS      = 40
HOP_LENGTH  = 160       # 10 ms at 16 kHz
WIN_LENGTH  = 400       # 25 ms at 16 kHz
N_FFT       = 512       # next power-of-2 ≥ WIN_LENGTH
FMIN        = 31.7      # Hz — matches PITCH_FMIN in model.py (~B0)
FMAX        = 8000.0    # Hz — upper mel bound
VAD_TOP_DB  = 30        # dB below peak to treat as silence (README: -30 dB RMS)


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract mel / F0 / VAD features into a training-ready NPZ.")
    p.add_argument("--dataset-dir", required=True,
                   help="directory containing .wav / .flac vocal files")
    p.add_argument("--output", default="data/clean.npz",
                   help="path for the output NPZ file (default: data/clean.npz)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="device for RMVPE inference (default: cpu)")
    p.add_argument("--rmvpe-model", default="rmvpe.pt",
                   help="path to the RMVPE checkpoint (default: rmvpe.pt)")
    p.add_argument("--min-duration", type=float, default=0.5,
                   help="skip clips shorter than this many seconds (default: 0.5)")
    p.add_argument("--pitch-shift-semitones", type=float, nargs="*", default=[],
                   help="extra pitch-shifted copies of every clip, in semitones "
                        "(e.g. -2 -1 1 2). Each value adds one augmented variant. "
                        "f0/VAD/mel are RE-DERIVED from the shifted audio, so labels "
                        "stay exact — no manual label transform. The unshifted original "
                        "is always kept. Combined with --time-stretch-rates as a grid.")
    p.add_argument("--time-stretch-rates", type=float, nargs="*", default=[],
                   help="extra time-stretched copies, as rate factors (e.g. 0.9 1.1; "
                        ">1 = faster/shorter, <1 = slower/longer). Labels are re-derived "
                        "from the stretched audio so frame counts and onsets stay correct. "
                        "The rate-1.0 original is always kept. Crosses with "
                        "--pitch-shift-semitones (P shifts × R rates variants per clip).")
    return p.parse_args()


def _augment_variants(y, sr, pitch_semitones, time_rates):
    """Yield (label, audio) pairs: the original plus every pitch×time variant.

    pitch_shift / time_stretch are applied on raw audio BEFORE feature
    extraction, so re-running mel/F0/VAD on each variant produces correctly
    aligned labels (no manual label warping — the source of the reverb-aug
    label-misalignment failure)."""
    # Always include the unmodified original (0 semitones, rate 1.0).
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


def extract_mel(y: np.ndarray) -> np.ndarray:
    """Compute log-mel spectrogram, shape (T, 40), fixed reference (dB re 1.0)."""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        window="hann",
        center=True,
    )
    # Fixed reference: dB re 1.0 power unit — keeps scale consistent across files.
    # Do NOT use ref=np.max (normalises per-file, destroys inter-file dynamics).
    log_mel = librosa.power_to_db(mel, ref=1.0)
    return log_mel.T.astype(np.float32)   # (T, 40)


def extract_vad(y: np.ndarray, n_frames: int) -> np.ndarray:
    """Energy-based VAD: per-frame binary float32 array aligned to mel hop grid.

    Uses librosa.effects.split with top_db=30 (silence = >30 dB below peak RMS),
    then converts sample-level intervals to frame indices.
    """
    frame_vad = np.zeros(n_frames, dtype=np.float32)
    for start_s, end_s in librosa.effects.split(y, top_db=VAD_TOP_DB):
        start_f = start_s // HOP_LENGTH
        end_f   = min(end_s // HOP_LENGTH, n_frames)
        if start_f < end_f:
            frame_vad[start_f:end_f] = 1.0
    return frame_vad


def main():
    args = parse_args()

    # ── Load RMVPE ──────────────────────────────────────────────────────
    try:
        from rmvpe import RMVPE
    except ImportError:
        raise SystemExit(
            "RMVPE not installed. Install with:\n"
            "  pip install rmvpe\n"
            "or clone https://github.com/yxlllc/RMVPE and add it to PYTHONPATH.")

    if not os.path.exists(args.rmvpe_model):
        raise SystemExit(
            f"RMVPE checkpoint not found: {args.rmvpe_model}\n"
            "Download from: https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt")

    print(f"Loading RMVPE from {args.rmvpe_model} ...")
    rmvpe = RMVPE(args.rmvpe_model, hop_length=HOP_LENGTH)

    # ── Collect audio files ─────────────────────────────────────────────
    # Skip macOS AppleDouble junk: a Mac-created zip carries a parallel
    # __MACOSX/ tree of "._name.wav" resource-fork stubs that aren't real audio
    # — they only fail to load (slowly, via the audioread fallback). Excluding
    # them here avoids thousands of guaranteed-fail loads.
    audio_exts = {".wav", ".flac", ".mp3", ".ogg"}
    files = sorted(
        os.path.join(root, f)
        for root, _, fnames in os.walk(args.dataset_dir)
        if "__MACOSX" not in root.split(os.sep)
        for f in fnames
        if os.path.splitext(f)[1].lower() in audio_exts
        and not f.startswith("._")
    )
    if not files:
        raise SystemExit(f"No audio files found in {args.dataset_dir}")
    print(f"Found {len(files)} audio files in {args.dataset_dir}")

    min_frames = int(args.min_duration * SR / HOP_LENGTH)

    mel_chunks, f0_chunks, vad_chunks, lengths = [], [], [], []
    skipped = 0

    for file_path in tqdm(files, desc="Extracting features"):
        try:
            y, _ = librosa.load(file_path, sr=SR, mono=True)
        except Exception as exc:
            print(f"  [skip] {file_path}: load error — {exc}")
            skipped += 1
            continue

        if len(y) < min_frames * HOP_LENGTH:
            skipped += 1
            continue

        # Original clip + every pitch×time augmentation variant. Features are
        # re-extracted per variant, so labels are always correctly aligned.
        for _tag, y_var in _augment_variants(
                y, SR, args.pitch_shift_semitones, args.time_stretch_rates):

            # ── Mel ──────────────────────────────────────────────────────
            log_mel = extract_mel(y_var)          # (T_mel, 40)

            # ── F0 (RMVPE) ───────────────────────────────────────────────
            # infer() returns f0 in Hz; 0.0 marks unvoiced frames.
            f0_hz = rmvpe.infer_from_audio(
                y_var, sample_rate=SR, device=args.device).astype(np.float32)

            # ── Temporal alignment ───────────────────────────────────────
            # mel and RMVPE use the same hop but may differ by ±1 frame due to
            # different internal padding conventions — use the shorter length.
            T = min(len(log_mel), len(f0_hz))
            if T < min_frames:
                skipped += 1
                continue
            log_mel = log_mel[:T]
            f0_hz   = f0_hz[:T]

            # ── VAD ──────────────────────────────────────────────────────
            frame_vad = extract_vad(y_var, T)    # (T,)

            mel_chunks.append(log_mel)
            f0_chunks.append(f0_hz)
            vad_chunks.append(frame_vad)
            lengths.append(T)

    if not mel_chunks:
        raise SystemExit("No valid clips extracted — check --dataset-dir and --min-duration.")

    # ── Save ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    mel_all = np.concatenate(mel_chunks, axis=0)   # (total_frames, 40)
    f0_all  = np.concatenate(f0_chunks,  axis=0)   # (total_frames,)
    vad_all = np.concatenate(vad_chunks, axis=0)   # (total_frames,)
    len_arr = np.array(lengths, dtype=np.int64)    # (n_clips,)

    # Save as float16 to match the dtype of the pre-extracted NanoPitch data
    # (smulelabs/NanoPitch-PreExtract on HuggingFace stores float16).
    np.savez(
        args.output,
        mel=mel_all.astype(np.float16),
        f0=f0_all.astype(np.float16),
        vad=vad_all.astype(np.float16),
        lengths=len_arr,
    )

    total_frames = len(mel_all)
    total_sec    = total_frames * HOP_LENGTH / SR
    voiced_pct   = float(np.mean(vad_all > 0.5)) * 100
    print(f"\nSaved {args.output}")
    print(f"  Clips   : {len(lengths)} processed, {skipped} skipped")
    print(f"  Frames  : {total_frames:,}  ({total_sec / 3600:.2f} hrs)")
    print(f"  mel     : {mel_all.shape}  float32")
    print(f"  f0      : {f0_all.shape}   float32  "
          f"(voiced: {voiced_pct:.1f}%)")
    print(f"  vad     : {vad_all.shape}  float32")
    print(f"  lengths : {len_arr.shape}  int64")


if __name__ == "__main__":
    main()
