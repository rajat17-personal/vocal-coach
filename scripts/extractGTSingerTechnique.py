"""
GTSinger Technique Label Extraction
=====================================

Downloads technique-labeled clips from the GTSinger HuggingFace dataset,
extracts mel / F0 / VAD, and saves them in the flat NPZ format used by
vocalcoach/train.py's TechniqueDataset.

Why a separate script from extractFeatures.py
----------------------------------------------
The pre-extracted clean.npz (from smulelabs/NanoPitch-PreExtract) has no
technique labels — it was created for pitch/VAD training only. This script
re-processes GTSinger clips that carry technique annotations and outputs a
SEPARATE technique_gtsinger_*.npz. The two NPZ files serve different Dataset
classes in train.py:

  clean.npz                → PitchVADDataset  (pitch + VAD supervision)
  technique_gtsinger_*.npz → TechniqueDataset (technique head supervision)

GTSinger technique → our label mapping
---------------------------------------
  vibrato    → 0  (vibrato)
  breathy    → 1  (breathy)
  falsetto   → 2  (falsetto)  ← unique to GTSinger; VocalSet has no falsetto
  glissando  → skip  (pitch slide ornament, not in our 5-class taxonomy)
  mixed_voice→ skip
  pharyngeal → skip

GTSinger dataset on HuggingFace (AaronZ345/GTSinger)
-----------------------------------------------------
The dataset has per-clip technique annotations.  Each example has:
  audio        — waveform dict with 'array' and 'sampling_rate'
  technique    — one of the strings above  (field may also be 'singing_method'
                 or 'style' depending on the dataset version; the script tries
                 common names in order)
  split        — 'train' / 'test' (used to produce two output files)

If the HuggingFace dataset schema changes, set --technique-field to the
correct field name and --train-split / --test-split as needed.

Output NPZ schema (same as extractVocalSet.py):
  mel:       (total_frames, 40)         float16
  f0:        (total_frames,)            float16
  vad:       (total_frames,)            float16
  technique: (n_clips, N_TECHNIQUES)   float32  — clip-level binary labels
  lengths:   (n_clips,)                int32
  ids:       (n_clips,)                object   — clip identifier for debugging

Usage
-----
  # Stream from HuggingFace (no download required, slow on large datasets):
  python scripts/extractGTSingerTechnique.py \\
      --output-dir data/gtsinger_technique \\
      --rmvpe-model rmvpe.pt \\
      --device cuda

  # From a local clone of the HuggingFace dataset:
  python scripts/extractGTSingerTechnique.py \\
      --local-dir /path/to/GTSinger \\
      --output-dir data/gtsinger_technique \\
      --rmvpe-model rmvpe.pt \\
      --device cuda

  # Limit clips per technique (useful for quick tests):
  python scripts/extractGTSingerTechnique.py \\
      --output-dir data/gtsinger_technique \\
      --max-per-technique 200
"""

import argparse
import os
import sys
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

# ── Constants — must match vocalcoach/model.py ────────────────────────
SR          = 16000
N_MELS      = 40
HOP_LENGTH  = 160
WIN_LENGTH  = 400
N_FFT       = 512
FMIN        = 31.7
FMAX        = 8000.0
VAD_TOP_DB  = 30

TECHNIQUE_NAMES = ['vibrato', 'breathy', 'falsetto', 'belt', 'straight']

# AaronZ345/GTSinger provides per-note binary technique indicator fields:
#   vibrato_tech, breathy_tech, falsetto_tech (list<int64>, 1 = technique active)
# belt and straight are absent — those come from VocalSet.
# singing_method is the music genre (pop/classical/…), not a vocal technique.

HF_REPO_ID = "AaronZ345/GTSinger"


def _gtsinger_features():
    """Explicit Features schema for AaronZ345/GTSinger.

    Derived from the CastError traceback (2026-05-10). The dataset card schema
    does not match the actual JSON fields, so we must pass this explicitly.
    audio=decode=False keeps the raw {bytes, path} struct for manual decoding.
    """
    try:
        from datasets import Features, Audio, Value, Sequence
        return Features({
            "item_name":       Value("string"),
            "txt":             Sequence(Value("string")),
            "ph":              Sequence(Value("string")),
            "ph_durs":         Sequence(Value("float64")),
            "word_durs":       Sequence(Value("float64")),
            "ep_pitches":      Sequence(Value("int64")),
            "ep_notedurs":     Sequence(Value("float64")),
            "ep_types":        Sequence(Value("int64")),
            "ph2words":        Sequence(Value("int64")),
            "mix_tech":        Sequence(Value("int64")),
            "falsetto_tech":   Sequence(Value("int64")),
            "breathy_tech":    Sequence(Value("int64")),
            "pharyngeal_tech": Sequence(Value("int64")),
            "vibrato_tech":    Sequence(Value("int64")),
            "glissando_tech":  Sequence(Value("int64")),
            "tech":            Sequence(Value("string")),
            "wav_fn":          Value("string"),
            "language":        Value("string"),
            "singer":          Value("string"),
            "speech_fn":       Value("string"),
            "emotion":         Value("string"),
            "singing_method":  Value("string"),
            "pace":            Value("string"),
            "range":           Value("string"),
            "note_start":      Sequence(Value("float64")),
            "ph_start":        Sequence(Value("float64")),
            "ph_end":          Sequence(Value("float64")),
            "mix":             Sequence(Value("string")),
            "note_end":        Sequence(Value("float64")),
            "label":           Value("int64"),
            "word":            Value("string"),
            "glissando":       Sequence(Value("string")),
            "pharyngeal":      Sequence(Value("string")),
            "note":            Sequence(Value("int64")),
            "breathy":         Sequence(Value("string")),
            "note_dur":        Sequence(Value("float64")),
            "falsetto":        Sequence(Value("string")),
            "vibrato":         Sequence(Value("string")),
            "audio":           Audio(decode=False),
            "start_time":      Value("float64"),
            "end_time":        Value("float64"),
        })
    except ImportError:
        return None


def _patch_tech_features(repo_id, token):
    """Return the dataset's own declared Features with 'tech' patched to string.

    The dataset card wrongly declares tech as ClassLabel (int64) but the actual
    parquet/JSON data stores it as a comma-separated string (e.g. "0", "2,6").
    Keeping all other fields — especially Audio() — at their declared defaults
    preserves the proper streaming audio loading pipeline.  Falls back to the
    fully explicit schema if the builder info cannot be fetched.
    """
    try:
        from datasets import load_dataset_builder, Value, Features
        builder = load_dataset_builder(repo_id, token=token)
        if builder.info.features is not None:
            patched = Features({**builder.info.features, "tech": Value("string")})
            print("  Patched 'tech' field to string; all other features from dataset card.")
            return patched
    except Exception as e:
        print(f"  Warning: could not read dataset builder info ({e}); using explicit schema.")
    return _gtsinger_features()


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract GTSinger technique labels into flat NPZ format.")
    p.add_argument("--output-dir", default="data/gtsinger_technique",
                   help="directory for output NPZ files")
    p.add_argument("--local-dir", default=None,
                   help="path to a locally downloaded GTSinger dataset "
                        "(HuggingFace serialized DatasetDict — skips HuggingFace download). "
                        "For raw WAV folders use --audio-dir instead.")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--mode", default="technique", choices=["technique", "notes"],
                   help="'technique' (default) = per-note technique clips; "
                        "'notes' = NOTE-label test set (onset/offset/MIDI) in the "
                        "note_test.npz schema for scripts/evalNoteHead.py. 'notes' "
                        "requires the HF dataset (--local-dir), not --audio-dir.")
    p.add_argument("--rmvpe-model", default="rmvpe.pt")
    p.add_argument("--train-split", default="train",
                   help="HF dataset split name for training data (default: train)")
    p.add_argument("--test-split", default="test",
                   help="HF dataset split name for test data (default: test)")
    p.add_argument("--no-streaming", action="store_true",
                   help="load from HF cache instead of streaming (faster if already "
                        "downloaded; Arrow schema avoids CastErrors)")
    p.add_argument("--hf-name", default=None,
                   help="HuggingFace dataset config/name (e.g. 'meta' if the cache "
                        "path contains 'AaronZ345___gt_singer/meta/')")
    p.add_argument("--audio-dir", default=None,
                   help="local directory containing downloaded GTSinger WAV files "
                        "(e.g. data/gtsinger_audio); wav_fn paths are resolved "
                        "relative to this dir before falling back to hf_hub_download")
    p.add_argument("--max-per-technique", type=int, default=None,
                   help="max clips per technique per split (None = all)")
    p.add_argument("--min-duration", type=float, default=0.5)
    p.add_argument("--pitch-shift-semitones", type=float, nargs="*", default=[],
                   help="augment the TRAIN split with pitch-shifted copies (semitones, "
                        "e.g. -2 2). Labels re-derived from shifted audio (exact). Test "
                        "split is NEVER augmented. Use --pitch-backend gpu for speed.")
    p.add_argument("--time-stretch-rates", type=float, nargs="*", default=[],
                   help="augment the TRAIN split with time-stretched copies (rate factors, "
                        "e.g. 0.9 1.1). Test split is NEVER augmented.")
    p.add_argument("--pitch-backend", default="cpu", choices=["cpu", "gpu"],
                   help="pitch-shift backend: 'cpu' (librosa, ~3.2s/clip) or 'gpu' "
                        "(torchaudio PitchShift, ms/clip — use for GTSinger-scale data). "
                        "GPU uses --device. Default cpu.")
    return p.parse_args()


# ── Audio helpers ─────────────────────────────────────────────────────

def resample_to_16k(audio_array, orig_sr):
    if orig_sr == SR:
        return audio_array.astype(np.float32)
    return librosa.resample(audio_array.astype(np.float32),
                            orig_sr=orig_sr, target_sr=SR)


def extract_mel(y):
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        window="hann", center=True,
    )
    return librosa.power_to_db(mel, ref=1.0).T.astype(np.float32)  # (T, 40)


def extract_vad(y, n_frames):
    frame_vad = np.zeros(n_frames, dtype=np.float32)
    for s, e in librosa.effects.split(y, top_db=VAD_TOP_DB):
        sf = s // HOP_LENGTH
        ef = min(e // HOP_LENGTH, n_frames)
        if sf < ef:
            frame_vad[sf:ef] = 1.0
    return frame_vad


# ── Pitch / time augmentation ─────────────────────────────────────────
# librosa.effects.pitch_shift is brutally slow (phase vocoder, ~3.2 s per
# 10 s clip). GTSinger is ~9.6k clips, so a CPU pitch-shift grid takes hours.
# torchaudio's PitchShift runs the same operation on the GPU in milliseconds.
# The CPU path stays available as a fallback (no torchaudio / no CUDA).

_PITCH_SHIFTERS = {}   # (n_steps_rounded) -> torchaudio PitchShift module on device


def _get_gpu_shifter(n_steps, device):
    """Cache a torchaudio PitchShift per integer semitone step (modules are
    expensive to build; n_steps is the only varying parameter)."""
    import torch
    from torchaudio.transforms import PitchShift
    key = round(n_steps)
    if key not in _PITCH_SHIFTERS:
        _PITCH_SHIFTERS[key] = PitchShift(sample_rate=SR, n_steps=key).to(device)
    return _PITCH_SHIFTERS[key]


def _augment_variants(y, pitch_semitones, time_rates, pitch_backend, device):
    """Yield (tag, audio): the original plus every pitch×time variant.

    Transforms apply to raw audio BEFORE feature extraction so mel/F0/VAD are
    re-derived (labels stay exact). Technique labels are pitch/tempo-invariant.
    pitch_backend: 'cpu' (librosa) or 'gpu' (torchaudio, much faster).
    Time-stretch always uses librosa (it is cheap: ~28 ms/clip)."""
    import torch
    pitches = [0.0] + [s for s in pitch_semitones if abs(s) > 1e-6]
    rates   = [1.0] + [r for r in time_rates if abs(r - 1.0) > 1e-6]
    for n_steps in pitches:
        if abs(n_steps) < 1e-6:
            y_p = y
        elif pitch_backend == "gpu":
            shifter = _get_gpu_shifter(n_steps, device)
            with torch.no_grad():
                t = torch.from_numpy(np.ascontiguousarray(y)).float().to(device)
                y_p = shifter(t).cpu().numpy().astype(np.float32)
        else:
            y_p = librosa.effects.pitch_shift(y=y, sr=SR, n_steps=n_steps)
        for rate in rates:
            y_pr = (y_p if abs(rate - 1.0) < 1e-6
                    else librosa.effects.time_stretch(y=y_p, rate=rate))
            tag = f"p{n_steps:+g}_r{rate:g}" if (n_steps or rate != 1.0) else "orig"
            yield tag, y_pr


# ── Dataset loading ───────────────────────────────────────────────────

def load_gtsinger(local_dir, train_split, test_split, streaming, hf_name):
    """Load GTSinger from HuggingFace (cached or streaming) or a local directory.

    Returns (train_dataset, test_dataset) — iterable HF Dataset objects.
    """
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError:
        raise SystemExit(
            "HuggingFace 'datasets' library not installed.\n"
            "  pip install datasets")

    token = os.environ.get("HF_TOKEN") or True

    if local_dir:
        print(f"Loading GTSinger from local path: {local_dir}")
        ds = load_from_disk(local_dir)
        train_ds = ds[train_split] if train_split in ds else ds
        test_ds  = ds[test_split]  if test_split  in ds else None
    elif streaming:
        print(f"Streaming GTSinger from HuggingFace ({HF_REPO_ID}) ...")
        print("  Tip: download first for faster repeated runs (--no-streaming).")
        features = _gtsinger_features()  # explicit schema to fix CastError
        kwargs = dict(split=train_split, streaming=True, token=token, features=features)
        if hf_name:
            kwargs["name"] = hf_name
        train_ds = load_dataset(HF_REPO_ID, **kwargs)
        try:
            kwargs["split"] = test_split
            test_ds = load_dataset(HF_REPO_ID, **kwargs)
        except Exception:
            test_ds = None
            print(f"  Note: split '{test_split}' not found — train only.")
    else:
        # Non-streaming: reads from HF cache (Arrow file already has correct schema).
        print(f"Loading GTSinger from HuggingFace cache ({HF_REPO_ID}) ...")
        kwargs = dict(token=token)
        if hf_name:
            kwargs["name"] = hf_name
            print(f"  Using config/name: {hf_name}")
        train_ds = load_dataset(HF_REPO_ID, split=train_split, **kwargs)
        try:
            test_ds = load_dataset(HF_REPO_ID, split=test_split, **kwargs)
        except Exception:
            test_ds = None
            print(f"  Note: split '{test_split}' not found — train only.")

    return train_ds, test_ds


# ── Per-split extraction ──────────────────────────────────────────────

def _load_audio(example, audio_dir):
    """Load audio array from a GTSinger example dict. Returns (arr, sr) or (None, None)."""
    import io
    audio_data = example.get("audio")
    if isinstance(audio_data, dict):
        raw_bytes = audio_data.get("bytes")
        raw_path  = audio_data.get("path")
        try:
            if raw_bytes:
                return librosa.load(io.BytesIO(raw_bytes), sr=None, mono=True)
            elif raw_path:
                return librosa.load(raw_path, sr=None, mono=True)
        except Exception:
            pass

    wav_fn = example.get("wav_fn")
    if wav_fn:
        local_path = os.path.join(audio_dir, wav_fn) if audio_dir else None
        try:
            if local_path and os.path.exists(local_path):
                return librosa.load(local_path, sr=None, mono=True)
            from huggingface_hub import hf_hub_download
            cached = hf_hub_download(repo_id=HF_REPO_ID, filename=wav_fn,
                                     repo_type="dataset",
                                     token=os.environ.get("HF_TOKEN") or True)
            return librosa.load(cached, sr=None, mono=True)
        except Exception:
            pass
    return None, None


def extract_split(dataset, rmvpe, device, min_frames, max_per_tech, split_name,
                  audio_dir=None):
    """Iterate through GTSinger and emit one clip per active-technique note.

    GTSinger provides per-note timestamps (note_start, note_end) and per-note
    binary technique indicators (vibrato_tech, breathy_tech, falsetto_tech).
    Instead of labelling the entire song phrase with any() over all notes
    (which assigns technique=1 to frames where the technique is not active),
    we extract each note window individually and label only that window.

    This matches VocalSet's structure: short isolated clips with clean labels.
    It also gives falsetto clips that VocalSet lacks entirely.

    Returns (mel_chunks, f0_chunks, vad_chunks, technique_labels,
             lengths, clip_ids, skipped_count)
    """
    mel_chunks, f0_chunks, vad_chunks = [], [], []
    technique_labels, lengths, clip_ids = [], [], []
    counts  = {i: 0 for i in range(len(TECHNIQUE_NAMES))}
    skip_reasons = {"tech_unmapped": 0, "audio_bad": 0,
                    "audio_short": 0, "frame_short": 0, "no_timestamps": 0}

    # Per-note technique fields present in GTSinger
    NOTE_TECH_FIELDS = [
        ('vibrato_tech',  0),   # vibrato
        ('breathy_tech',  1),   # breathy
        ('falsetto_tech', 2),   # falsetto
        # belt=3, straight=4 absent in GTSinger — provided by VocalSet
    ]

    for idx, example in enumerate(tqdm(dataset, desc=f"  {split_name}")):
        note_starts = example.get('note_start') or []
        note_ends   = example.get('note_end')   or []

        # Fall back to clip-level any() when note timestamps are absent
        if not note_starts or len(note_starts) != len(note_ends):
            skip_reasons["no_timestamps"] += 1
            continue

        n_notes = len(note_starts)

        # Build per-note label matrix: (n_notes, N_TECHNIQUES)
        note_labels = np.zeros((n_notes, len(TECHNIQUE_NAMES)), dtype=np.float32)
        for field, tech_idx in NOTE_TECH_FIELDS:
            tech_flags = example.get(field) or []
            for ni, flag in enumerate(tech_flags[:n_notes]):
                if flag:
                    note_labels[ni, tech_idx] = 1.0

        # Skip song entirely if no technique-active notes
        if note_labels.sum() == 0:
            skip_reasons["tech_unmapped"] += 1
            continue

        # Load audio once per song — reused across all notes
        audio_arr, orig_sr = _load_audio(example, audio_dir)
        if audio_arr is None or len(audio_arr) == 0:
            skip_reasons["audio_bad"] += 1
            continue

        y_full = resample_to_16k(np.asarray(audio_arr), orig_sr)
        # Extract mel + F0 for the full clip once; slice per note below
        log_mel_full = extract_mel(y_full)
        f0_full      = rmvpe.infer_from_audio(
            y_full, sample_rate=SR, device=device).astype(np.float32)
        T_full = min(len(log_mel_full), len(f0_full))
        log_mel_full = log_mel_full[:T_full]
        f0_full      = f0_full[:T_full]
        vad_full     = extract_vad(y_full, T_full)

        for ni in range(n_notes):
            label = note_labels[ni]
            if label.sum() == 0:
                continue  # no technique active for this note

            active = [i for i in range(len(TECHNIQUE_NAMES)) if label[i] > 0]
            primary = active[0]
            if max_per_tech and counts[primary] >= max_per_tech:
                continue

            # Convert note timestamps to frame indices
            t_start = float(note_starts[ni])
            t_end   = float(note_ends[ni])
            f_start = int(t_start * SR / HOP_LENGTH)
            f_end   = min(int(t_end * SR / HOP_LENGTH), T_full)

            if f_end - f_start < min_frames:
                skip_reasons["frame_short"] += 1
                continue

            note_mel = log_mel_full[f_start:f_end]
            note_f0  = f0_full[f_start:f_end]
            note_vad = vad_full[f_start:f_end]
            T_note   = len(note_mel)

            mel_chunks.append(note_mel)
            f0_chunks.append(note_f0)
            vad_chunks.append(note_vad)
            technique_labels.append(label)
            lengths.append(T_note)
            clip_ids.append(f"{idx}_n{ni}")
            counts[primary] += 1

    total_skipped = sum(skip_reasons.values())
    print(f"\n  Extracted per technique (note-level clips):")
    for i, name in enumerate(TECHNIQUE_NAMES):
        print(f"    {name:<12}: {counts[i]:4d} clips")
    print(f"  Skipped total: {total_skipped}")


def extract_note_split(dataset, rmvpe, device, min_frames, split_name,
                       max_clips=None, audio_dir=None):
    """Build a NOTE-label test set from GTSinger in the SAME schema as
    data/annotated_vocalset/note_test.npz, so scripts/evalNoteHead.py can score
    the note head against GTSinger's note onsets/offsets/MIDI.

    Unlike extract_split (one clip per technique-note), here each SONG is one clip
    and we record EVERY note's (onset, offset, midi) as GLOBAL frame indices into
    the concatenated mel — matching the note_test.npz layout exactly.

    RMVPE F0 runs on `device` (GPU when --device cuda), like extract_split.

    Returns dict of arrays ready for np.savez (mel/f0/vad/lengths +
    note_onsets/note_offsets/note_midi/note_clip/n_notes), plus a 'technique'
    placeholder (zeros) so the file is interchangeable with note_test.npz.
    """
    mel_chunks, f0_chunks, vad_chunks, lengths = [], [], [], []
    on_idx, off_idx, midi_arr, note_clip, n_notes = [], [], [], [], []
    global_offset = 0           # running frame count (for global note indices)
    clip_i = 0
    skipped = {"no_timestamps": 0, "audio_bad": 0, "frame_short": 0, "no_valid_notes": 0}

    for idx, example in enumerate(tqdm(dataset, desc=f"  {split_name} [notes]")):
        if max_clips and clip_i >= max_clips:
            break
        note_starts = example.get('note_start') or []
        note_ends   = example.get('note_end')   or []
        notes_midi  = example.get('note')        or []
        if (not note_starts or len(note_starts) != len(note_ends)
                or len(notes_midi) != len(note_starts)):
            skipped["no_timestamps"] += 1
            continue

        audio_arr, orig_sr = _load_audio(example, audio_dir)
        if audio_arr is None or len(audio_arr) == 0:
            skipped["audio_bad"] += 1
            continue
        y = resample_to_16k(np.asarray(audio_arr), orig_sr)
        log_mel = extract_mel(y)
        f0 = rmvpe.infer_from_audio(y, sample_rate=SR, device=device).astype(np.float32)
        T = min(len(log_mel), len(f0))
        if T < min_frames:
            skipped["frame_short"] += 1
            continue
        log_mel = log_mel[:T]; f0 = f0[:T]; vad = extract_vad(y, T)

        # Per-note global frame indices (onset/offset clamped into this clip).
        kept = 0
        for ns, ne, mid in zip(note_starts, note_ends, notes_midi):
            f_on  = int(float(ns) * SR / HOP_LENGTH)
            f_off = int(float(ne) * SR / HOP_LENGTH)
            if f_on < 0 or f_on >= T or mid is None or int(mid) <= 0:
                continue
            f_off = min(max(f_off, f_on + 1), T - 1)
            on_idx.append(global_offset + f_on)
            off_idx.append(global_offset + f_off)
            midi_arr.append(int(mid))
            note_clip.append(clip_i)
            kept += 1
        if kept == 0:
            skipped["no_valid_notes"] += 1
            continue

        mel_chunks.append(log_mel); f0_chunks.append(f0); vad_chunks.append(vad)
        lengths.append(T); n_notes.append(kept)
        global_offset += T
        clip_i += 1

    print(f"\n  {split_name}: {clip_i} clips, {len(on_idx)} notes")
    print(f"  Skipped: {skipped}")
    if clip_i == 0:
        return None
    return {
        "mel":     np.concatenate(mel_chunks).astype(np.float16),
        "f0":      np.concatenate(f0_chunks).astype(np.float16),
        "vad":     np.concatenate(vad_chunks).astype(np.float16),
        "technique": np.zeros((clip_i, len(TECHNIQUE_NAMES)), dtype=np.float32),
        "lengths": np.array(lengths, dtype=np.int32),
        "note_onsets":  np.array(on_idx, dtype=np.int32),
        "note_offsets": np.array(off_idx, dtype=np.int32),
        "note_midi":    np.array(midi_arr, dtype=np.uint8),
        "note_clip":    np.array(note_clip, dtype=np.int32),
        "n_notes":      np.array(n_notes, dtype=np.int32),
    }
    for reason, n in skip_reasons.items():
        if n:
            print(f"    {reason:<16}: {n}")

    return (mel_chunks, f0_chunks, vad_chunks,
            technique_labels, lengths, clip_ids, total_skipped)


# ── Filesystem walker (raw WAV folder, no HF dataset needed) ─────────

# Folder name → technique index mapping for GTSinger's group structure:
#   language/singer/TechniqueName/song/TechniqueName_Group/*.wav
_FOLDER_TO_TECH = {
    "vibrato_group":           0,   # vibrato
    "breathy_group":           1,   # breathy
    "mixed_voice_and_falsetto": 2,  # falsetto (technique folder level)
    "falsetto_group":          2,   # falsetto (group folder level)
    "mixed_voice_group":       2,   # falsetto variant
    # Glissando, Pharyngeal, Control_Group, Paired_Speech_Group → skipped
}


def extract_from_wav_dir(audio_dir, rmvpe, device, min_frames, max_per_tech,
                         test_singers=None, pitch_semitones=(), time_rates=(),
                         pitch_backend="cpu"):
    """Walk a raw GTSinger WAV directory and extract technique clips.

    Expected layout:
        audio_dir/
          <language>/
            <singer>/
              <TechniqueFolder>/    e.g. Vibrato, Breathy, Mixed_Voice_and_Falsetto
                <song>/
                  <Group>/          e.g. Vibrato_Group, Control_Group
                    0000.wav ...

    The technique label comes from the Group folder name
    (Vibrato_Group → vibrato, Falsetto_Group / Mixed_Voice_Group → falsetto,
    Breathy_Group → breathy). Control_Group and Paired_Speech_Group are skipped.

    test_singers: set of singer names held out for the test split (e.g. {'EN-Alto-1'}).
                  If None all clips go to train.
    Returns two result tuples (train, test), each matching the extract_split
    return format.
    """
    train_res = [[], [], [], [], [], []]
    test_res  = [[], [], [], [], [], []]
    train_skip = test_skip = 0

    counts_tr = {i: 0 for i in range(len(TECHNIQUE_NAMES))}
    counts_te = {i: 0 for i in range(len(TECHNIQUE_NAMES))}
    skip_reasons = {"no_technique": 0, "audio_short": 0, "frame_short": 0}

    wav_paths = sorted(
        p for p in Path(audio_dir).rglob("*.wav")
    )
    print(f"  Found {len(wav_paths):,} WAV files under {audio_dir}")

    for wav_path in tqdm(wav_paths, desc="  walking"):
        parts = wav_path.parts
        # Expect at least: audio_dir / language / singer / technique_dir / song / group / file
        if len(parts) < 6:
            continue

        group_name  = parts[-2].lower()   # e.g. "vibrato_group"
        singer_name = parts[-5]           # e.g. "EN-Alto-1"

        tech_idx = _FOLDER_TO_TECH.get(group_name)
        if tech_idx is None:
            skip_reasons["no_technique"] += 1
            continue

        is_test = test_singers and singer_name in test_singers
        counts  = counts_te if is_test else counts_tr
        res     = test_res  if is_test else train_res

        if max_per_tech and counts[tech_idx] >= max_per_tech:
            continue

        try:
            audio_arr, orig_sr = librosa.load(str(wav_path), sr=None, mono=True)
        except Exception:
            skip_reasons["audio_short"] += 1
            continue

        y = resample_to_16k(np.asarray(audio_arr), orig_sr)
        if len(y) < min_frames * HOP_LENGTH:
            skip_reasons["audio_short"] += 1
            continue

        label = np.zeros(len(TECHNIQUE_NAMES), dtype=np.float32)
        label[tech_idx] = 1.0
        rel_path = str(wav_path.relative_to(audio_dir))

        # Augment train clips only; test split must stay clean (no inflated eval).
        # RMVPE runs per variant so f0 is ground-truth on each transformed clip
        # (a derived-f0 shortcut was rejected: it under-labels voicing ~8.6% of
        # frames at phrase edges, creating a VAD↔f0 mismatch).
        variants = (_augment_variants(y, pitch_semitones, time_rates,
                                      pitch_backend, device)
                    if (not is_test and (pitch_semitones or time_rates))
                    else [("orig", y)])
        for tag, y_var in variants:
            log_mel = extract_mel(y_var)
            f0_hz   = rmvpe.infer_from_audio(
                y_var, sample_rate=SR, device=device).astype(np.float32)
            T       = min(len(log_mel), len(f0_hz))
            if T < min_frames:
                skip_reasons["frame_short"] += 1
                continue
            res[0].append(log_mel[:T])
            res[1].append(f0_hz[:T])
            res[2].append(extract_vad(y_var, T))
            res[3].append(label.copy())
            res[4].append(T)
            res[5].append(rel_path if tag == "orig" else f"{rel_path}#{tag}")
        counts[tech_idx] += 1

    total_skip = sum(skip_reasons.values())
    for label_str, counts in [("train", counts_tr), ("test", counts_te)]:
        print(f"\n  {label_str} — extracted per technique:")
        for i, name in enumerate(TECHNIQUE_NAMES):
            print(f"    {name:<12}: {counts[i]:4d} clips")
    print(f"  Skipped total: {total_skip}")
    for reason, n in skip_reasons.items():
        if n:
            print(f"    {reason:<16}: {n}")

    return (
        (*train_res, train_skip),
        (*test_res,  test_skip),
    )


# ── Save ─────────────────────────────────────────────────────────────

def save_split(output_path, mel_chunks, f0_chunks, vad_chunks,
               technique_labels, lengths, clip_ids):
    if not mel_chunks:
        print(f"  [warn] No clips extracted — skipping {output_path}")
        return

    mel_all  = np.concatenate(mel_chunks).astype(np.float16)
    f0_all   = np.concatenate(f0_chunks).astype(np.float16)
    vad_all  = np.concatenate(vad_chunks).astype(np.float16)
    tech_all = np.stack(technique_labels)
    len_arr  = np.array(lengths, dtype=np.int32)
    ids_arr  = np.array(clip_ids, dtype=object)

    np.savez(output_path, mel=mel_all, f0=f0_all, vad=vad_all,
             technique=tech_all, lengths=len_arr, ids=ids_arr)

    total_frames = len(mel_all)
    voiced_pct   = float(np.mean(vad_all > 0)) * 100
    print(f"\n  Saved {output_path}")
    print(f"    clips={len(lengths)}, frames={total_frames:,} "
          f"({total_frames * HOP_LENGTH / SR / 3600:.2f} hrs), "
          f"voiced={voiced_pct:.1f}%")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    min_frames = int(args.min_duration * SR / HOP_LENGTH)

    try:
        from rmvpe import RMVPE
    except ImportError:
        raise SystemExit("RMVPE not installed. Run: pip install rmvpe")

    if not os.path.exists(args.rmvpe_model):
        raise SystemExit(
            f"RMVPE checkpoint not found: {args.rmvpe_model}\n"
            "Download from: https://huggingface.co/lj1995/VoiceConversionWebUI")

    print(f"Loading RMVPE from {args.rmvpe_model} ...")
    rmvpe = RMVPE(args.rmvpe_model, hop_length=HOP_LENGTH)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Note-label mode: build note_test.npz-schema files for evalNoteHead.py ──
    if args.mode == "notes":
        if not args.local_dir:
            raise SystemExit("--mode notes needs the HF dataset (--local-dir); the "
                             "raw --audio-dir wavs carry no note_start/note_end/note "
                             "metadata.")
        train_ds, test_ds = load_gtsinger(
            args.local_dir, args.train_split, args.test_split,
            streaming=not args.no_streaming, hf_name=args.hf_name)
        print(f"\nExtracting GTSinger NOTE labels (RMVPE on {args.device}) …")
        for ds, split, fname in ((test_ds, "test", "note_gtsinger_test.npz"),
                                  (train_ds, "train", "note_gtsinger_train.npz")):
            if ds is None:
                continue
            d = extract_note_split(ds, rmvpe, args.device, min_frames, split,
                                   max_clips=args.max_per_technique,
                                   audio_dir=args.audio_dir)
            if d is not None:
                out = os.path.join(args.output_dir, fname)
                np.savez(out, **d)
                print(f"  Saved {out}  ({len(d['lengths'])} clips, "
                      f"{len(d['note_onsets'])} notes)")
        return

    if args.audio_dir and not args.local_dir:
        # Raw WAV folder — walk filesystem directly, no HF dataset needed.
        # Test singers held out: one per language to match VocalSet split style.
        # One small singer per language group held out for test.
        # Chosen to be the smallest singer by clip count to keep train ~80%.
        # EN-Alto-1 (1792), KO-Soprano-1 (377), IT-Soprano-1 (550),
        # JA-Tenor-1 (910), ES-Soprano-1 (1665) → ~5% each, ~20% total.
        test_singers = {"EN-Alto-1", "KO-Soprano-1", "IT-Soprano-1",
                        "JA-Tenor-1", "ES-Soprano-1"}
        print(f"\nExtracting from WAV directory: {args.audio_dir}")
        print(f"  Test singers ({len(test_singers)}): {sorted(test_singers)}")
        if args.pitch_shift_semitones or args.time_stretch_rates:
            n_var = (len(args.pitch_shift_semitones) + 1) * (len(args.time_stretch_rates) + 1)
            print(f"  TRAIN augmentation: pitch={args.pitch_shift_semitones or '—'} "
                  f"time={args.time_stretch_rates or '—'} → {n_var}× variants/clip "
                  f"(backend={args.pitch_backend}, test NOT augmented)")
        tr, te = extract_from_wav_dir(
            args.audio_dir, rmvpe, args.device, min_frames,
            args.max_per_technique, test_singers=test_singers,
            pitch_semitones=args.pitch_shift_semitones,
            time_rates=args.time_stretch_rates,
            pitch_backend=args.pitch_backend)
        save_split(os.path.join(args.output_dir, "technique_gtsinger_train.npz"),
                   *tr[:-1])
        save_split(os.path.join(args.output_dir, "technique_gtsinger_test.npz"),
                   *te[:-1])
    else:
        # HuggingFace dataset path (streaming or cached Arrow file).
        train_ds, test_ds = load_gtsinger(
            args.local_dir, args.train_split, args.test_split,
            streaming=not args.no_streaming, hf_name=args.hf_name)

        mapped = ['vibrato', 'breathy', 'falsetto']
        print(f"\nExtracting training split (mapped via *_tech fields: {mapped}) ...")
        tr = extract_split(train_ds, rmvpe, args.device, min_frames,
                           args.max_per_technique, "train", audio_dir=args.audio_dir)
        save_split(os.path.join(args.output_dir, "technique_gtsinger_train.npz"),
                   *tr[:-1])

        if test_ds is not None:
            print("\nExtracting test split ...")
            te = extract_split(test_ds, rmvpe, args.device, min_frames,
                               args.max_per_technique, "test", audio_dir=args.audio_dir)
            save_split(os.path.join(args.output_dir, "technique_gtsinger_test.npz"),
                       *te[:-1])

    print(f"\nDone. Output in {args.output_dir}/")
    print(f"  technique_gtsinger_train.npz — training clips")
    print(f"  technique_gtsinger_test.npz  — held-out test clips")
    print(f"\nTo include in probe-mode training:")
    print(f"  --technique-dirs data/vocalset data/gtsinger_technique")


if __name__ == "__main__":
    main()
