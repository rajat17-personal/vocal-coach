"""
VocalCoach Signal Processing Features (Phase 0 / Phase 2)
==========================================================

Per-frame and per-note acoustic features extracted from raw audio + the model's
F0/VAD outputs. All functions accept numpy arrays and return numpy arrays —
no PyTorch required. Designed to run offline (post-session analysis).

Features implemented (from the full taxonomy):
  5-7 Vibrato rate, depth, regularity  per voiced phrase (autocorrelation of F0)
  14  RMS dynamics            per frame
  15  HNR                     per frame (requires F0 for pitch-lag autocorr)
  16  Spectral centroid        per frame
  17  Spectral tilt (H1-H2)   per voiced frame (requires F0)
  18  Jitter (local)           per voiced segment
  19  Shimmer (local)          per voiced segment
  20  MFCCs (13 coeffs)        per frame
  21  Breath detection         returns breath event intervals
  22  Vocal onset steepness    per note onset
  23  DTW reference distance   per-phrase scalar vs reference F0
  Ph  Phrase aggregation       per-phrase coaching metrics

Usage
-----
    from vocalcoach.features import extract_all, phrase_aggregate

    feats = extract_all(y, sr=16000, f0_hz=f0, vad=vad)
    phrases = phrase_aggregate(f0_hz, vad, technique_probs, sr=16000)
    # feats is a dict with keys matching the taxonomy IDs above
"""

import numpy as np
import librosa

try:
    from scipy.spatial.distance import cdist
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ── Shared constants ────────────────────────────────────────────────
SR          = 16000
HOP_LENGTH  = 160       # 10 ms — must match model hop
WIN_LENGTH  = 400       # 25 ms
N_FFT       = 512
N_MELS      = 40
N_MFCC      = 13


# ═══════════════════════════════════════════════════════════════════════
# Feature 14 — RMS Dynamics
# ═══════════════════════════════════════════════════════════════════════

def compute_rms(y, hop_length=HOP_LENGTH, frame_length=WIN_LENGTH):
    """Per-frame RMS energy in dBFS.

    Returns:
        rms_db: (T,) float32 — values in dB (negative; 0 dBFS = full scale)
    """
    rms = librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop_length, center=True
    )[0]
    return librosa.amplitude_to_db(rms, ref=1.0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# Feature 15 — HNR (Harmonics-to-Noise Ratio)
# ═══════════════════════════════════════════════════════════════════════

def compute_hnr(y, f0_hz, sr=SR, hop_length=HOP_LENGTH,
                frame_length=WIN_LENGTH):
    """Per-frame Harmonics-to-Noise Ratio in dB via autocorrelation.

    For each voiced frame, the normalised autocorrelation r(τ) at the pitch
    lag τ = 1/f0 gives:
        HNR_dB = 10 * log10(r / (1 - r + eps))

    Unvoiced frames are filled with NaN.

    Args:
        y:       (N,) waveform
        f0_hz:   (T,) per-frame F0 in Hz (0 = unvoiced), aligned to hop_length

    Returns:
        hnr_db: (T,) float32
    """
    T = len(f0_hz)
    hnr_db = np.full(T, np.nan, dtype=np.float32)

    for t in range(T):
        if f0_hz[t] <= 0:
            continue
        s = t * hop_length
        e = s + frame_length
        if e > len(y):
            break
        frame = y[s:e].astype(np.float64)
        frame -= frame.mean()

        # Normalised autocorrelation
        ac = np.correlate(frame, frame, mode='full')
        ac = ac[len(frame) - 1:]          # one-sided
        if ac[0] < 1e-12:
            continue
        ac /= ac[0]

        # Pitch lag in samples
        lag = int(round(sr / f0_hz[t]))
        if lag < 1 or lag >= len(ac):
            continue

        r = np.clip(ac[lag], 1e-7, 1.0 - 1e-7)  # avoid log10(0)
        hnr_db[t] = np.float32(10.0 * np.log10(r / (1.0 - r)))

    return hnr_db


# ═══════════════════════════════════════════════════════════════════════
# Feature 16 — Spectral Centroid
# ═══════════════════════════════════════════════════════════════════════

def compute_spectral_centroid(y, sr=SR, hop_length=HOP_LENGTH, n_fft=N_FFT):
    """Per-frame spectral centroid in Hz (brightness / resonance indicator).

    Returns:
        centroid_hz: (T,) float32
    """
    return librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, center=True
    )[0].astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# Feature 17 — Spectral Tilt (H1-H2 proxy)
# ═══════════════════════════════════════════════════════════════════════

def compute_spectral_tilt(y, f0_hz, sr=SR, hop_length=HOP_LENGTH,
                          frame_length=WIN_LENGTH, n_fft=N_FFT):
    """Per-voiced-frame H1-H2 spectral tilt (breathiness indicator).

    H1 = amplitude of first harmonic (at f0)
    H2 = amplitude of second harmonic (at 2*f0)
    Tilt = H1_dB - H2_dB  (positive = breathier, flatter harmonic slope)

    Unvoiced frames are NaN.

    Returns:
        h1_h2: (T,) float32 in dB
    """
    T = len(f0_hz)
    h1_h2 = np.full(T, np.nan, dtype=np.float32)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    for t in range(T):
        if f0_hz[t] <= 0:
            continue
        s = t * hop_length
        e = s + frame_length
        if e > len(y):
            break
        frame = y[s:e] * np.hanning(frame_length)
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)))
        spec = np.abs(np.fft.rfft(frame, n=n_fft))

        # Find bin closest to f0 and 2*f0
        def peak_amp(target_hz, search_cents=50):
            search_hz = target_hz * (2 ** (search_cents / 1200.0) - 1)
            lo = max(0, np.searchsorted(freqs, target_hz - search_hz))
            hi = min(len(freqs), np.searchsorted(freqs, target_hz + search_hz))
            if lo >= hi:
                return 1e-10
            return float(spec[lo:hi].max())

        a1 = peak_amp(f0_hz[t])
        a2 = peak_amp(2.0 * f0_hz[t])
        h1_h2[t] = np.float32(
            20.0 * np.log10(a1 + 1e-10) - 20.0 * np.log10(a2 + 1e-10))

    return h1_h2


# ═══════════════════════════════════════════════════════════════════════
# Feature 18 — Jitter
# ═══════════════════════════════════════════════════════════════════════

def compute_jitter(f0_hz, hop_s=HOP_LENGTH / SR):
    """Local jitter: mean absolute cycle-to-cycle F0 variation / mean F0.

    Computed per continuous voiced segment. Returns a scalar per segment.

    Args:
        f0_hz:  (T,) — F0 in Hz (0 = unvoiced)
        hop_s:  hop size in seconds (default 10 ms)

    Returns:
        jitter_pct: (T,) float32  — local jitter interpolated back to frames
                    (segment average broadcast over voiced frames, NaN elsewhere)
    """
    T = len(f0_hz)
    jitter_arr = np.full(T, np.nan, dtype=np.float32)

    voiced = f0_hz > 0
    # Find voiced segments
    changes = np.diff(voiced.astype(int), prepend=0, append=0)
    starts  = np.where(changes == 1)[0]
    ends    = np.where(changes == -1)[0]

    for s, e in zip(starts, ends):
        seg = f0_hz[s:e]
        if len(seg) < 2:
            continue
        periods = 1.0 / seg                     # cycle duration in seconds
        abs_diffs = np.abs(np.diff(periods))
        local_jitter = float(np.mean(abs_diffs) / (np.mean(periods) + 1e-10))
        jitter_arr[s:e] = np.float32(local_jitter * 100.0)  # percent

    return jitter_arr


# ═══════════════════════════════════════════════════════════════════════
# Feature 19 — Shimmer
# ═══════════════════════════════════════════════════════════════════════

def compute_shimmer(y, f0_hz, sr=SR, hop_length=HOP_LENGTH):
    """Local shimmer: mean absolute cycle-to-cycle amplitude variation / mean amp.

    Amplitude is estimated as peak absolute value within each pitch period.
    Returns per-frame shimmer (segment average, NaN on unvoiced).

    Returns:
        shimmer_pct: (T,) float32
    """
    T = len(f0_hz)
    shimmer_arr = np.full(T, np.nan, dtype=np.float32)

    voiced = f0_hz > 0
    changes = np.diff(voiced.astype(int), prepend=0, append=0)
    starts  = np.where(changes == 1)[0]
    ends    = np.where(changes == -1)[0]

    for s, e in zip(starts, ends):
        amps = []
        for t in range(s, e):
            f0 = f0_hz[t]
            if f0 <= 0:
                continue
            period_samples = int(round(sr / f0))
            sample_s = t * hop_length
            sample_e = min(sample_s + period_samples, len(y))
            if sample_s >= len(y):
                break
            cycle = y[sample_s:sample_e]
            if len(cycle) == 0:
                break
            amps.append(float(np.max(np.abs(cycle)) + 1e-10))

        if len(amps) < 2:
            continue
        amps = np.array(amps)
        abs_diffs = np.abs(np.diff(amps))
        local_shimmer = float(np.mean(abs_diffs) / (np.mean(amps) + 1e-10))
        shimmer_arr[s:e] = np.float32(local_shimmer * 100.0)  # percent

    return shimmer_arr


# ═══════════════════════════════════════════════════════════════════════
# Feature 20 — MFCCs
# ═══════════════════════════════════════════════════════════════════════

def compute_mfcc(y, sr=SR, hop_length=HOP_LENGTH, n_mfcc=N_MFCC, n_fft=N_FFT):
    """Per-frame MFCCs (timbre characterisation).

    Returns:
        mfcc: (T, n_mfcc) float32
    """
    return librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc,
        hop_length=hop_length, n_fft=n_fft,
        center=True,
    ).T.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# Feature 21 — Breath Detection
# ═══════════════════════════════════════════════════════════════════════

def detect_breaths(y, vad, rms_db, sr=SR, hop_length=HOP_LENGTH,
                   min_gap_frames=3, min_energy_db=-45.0,
                   spectral_flatness_thresh=0.3):
    """Detect breath events in unvoiced gaps between voiced phrases.

    A gap between two voiced segments is classified as a breath (rather than
    silence) when it contains audible energy above min_energy_db AND has
    elevated high-frequency content (turbulent airflow signature).

    Args:
        y:          (N,) waveform
        vad:        (T,) per-frame binary VAD (0 or 1)
        rms_db:     (T,) per-frame RMS in dBFS (from compute_rms)
        min_gap_frames:  minimum gap length to examine
        min_energy_db:   minimum RMS threshold for a breath (dBFS)
        spectral_flatness_thresh: above this → noisy (breath-like)

    Returns:
        breaths: list of (start_frame, end_frame) for each detected breath
    """
    T = len(vad)
    # High-frequency energy ratio (breath = high spectral flatness)
    spec_flatness = librosa.feature.spectral_flatness(
        y=y, hop_length=hop_length, center=True
    )[0].astype(np.float32)   # (T,)
    # Pad/trim to T
    spec_flatness = spec_flatness[:T]
    if len(spec_flatness) < T:
        spec_flatness = np.pad(spec_flatness, (0, T - len(spec_flatness)))

    # Find unvoiced gaps between voiced segments
    voiced = vad > 0.5
    changes = np.diff(voiced.astype(int), prepend=0, append=0)
    v_starts = np.where(changes == 1)[0]
    v_ends   = np.where(changes == -1)[0]

    breaths = []
    for i in range(len(v_starts) - 1):
        gap_s = v_ends[i]
        gap_e = v_starts[i + 1]
        if gap_e - gap_s < min_gap_frames:
            continue
        gap_rms  = float(np.mean(rms_db[gap_s:gap_e]))
        gap_flat = float(np.mean(spec_flatness[gap_s:gap_e]))
        if gap_rms > min_energy_db and gap_flat > spectral_flatness_thresh:
            breaths.append((int(gap_s), int(gap_e)))

    return breaths


# ═══════════════════════════════════════════════════════════════════════
# Feature 22 — Vocal Onset Steepness
# ═══════════════════════════════════════════════════════════════════════

def compute_onset_steepness(rms_db, vad, hop_s=HOP_LENGTH / SR,
                            attack_window_frames=10):
    """Attack steepness at each voiced onset.

    Measures the RMS rise rate (dB/s) over `attack_window_frames` frames
    starting from each VAD onset. Hard onsets → high steepness;
    soft/breathy onsets → low steepness.

    Returns:
        steepness: list of (onset_frame, db_per_second)
    """
    voiced = vad > 0.5
    changes = np.diff(voiced.astype(int), prepend=0)
    onsets  = np.where(changes == 1)[0]

    results = []
    for onset in onsets:
        end = min(onset + attack_window_frames, len(rms_db))
        if end - onset < 2:
            continue
        window = rms_db[onset:end]
        duration_s = (end - onset) * hop_s
        rise_db = float(window[-1] - window[0])
        results.append((int(onset), rise_db / duration_s))

    return results


# ═══════════════════════════════════════════════════════════════════════
# Features 5-7 — Vibrato Rate, Depth, Regularity
# ═══════════════════════════════════════════════════════════════════════

def compute_vibrato(f0_hz, sr=SR, hop_length=HOP_LENGTH,
                    min_phrase_frames=30, rate_range=(4.0, 8.0),
                    depth_cents_min=20.0):
    """Detect vibrato in voiced phrases via short-time F0 autocorrelation.

    For each continuous voiced phrase, converts F0 to cents, removes the
    linear trend (pitch contour), then finds the dominant periodic component
    via autocorrelation. A phrase is considered to have vibrato when the peak
    autocorrelation lag falls within the classical vibrato rate range (4-8 Hz)
    and depth exceeds `depth_cents_min`.

    Args:
        f0_hz:          (T,) F0 in Hz (0 = unvoiced)
        sr:             sample rate
        hop_length:     hop in samples (sets frame duration)
        min_phrase_frames: minimum voiced segment length to analyse
        rate_range:     (min_hz, max_hz) for valid vibrato rate
        depth_cents_min: minimum peak-to-trough extent to report vibrato

    Returns:
        list of dicts, one per phrase:
            start_frame, end_frame,
            has_vibrato (bool),
            rate_hz (float | None),
            depth_cents (float | None),  — peak-to-trough range in cents
            regularity (float | None)    — normalised autocorrelation peak [0,1]
    """
    hop_s = hop_length / sr
    frame_rate = 1.0 / hop_s  # frames per second

    voiced = f0_hz > 0
    changes = np.diff(voiced.astype(int), prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends   = np.where(changes == -1)[0]

    results = []
    for s, e in zip(starts, ends):
        rec = {"start_frame": int(s), "end_frame": int(e),
               "has_vibrato": False, "rate_hz": None,
               "depth_cents": None, "regularity": None}

        if e - s < min_phrase_frames:
            results.append(rec)
            continue

        seg = f0_hz[s:e].astype(np.float64)

        # Convert to cents relative to mean pitch
        cents = 1200.0 * np.log2(seg / (np.mean(seg) + 1e-10))

        # Remove linear trend (slow pitch glide)
        t = np.arange(len(cents))
        trend = np.polyval(np.polyfit(t, cents, 1), t)
        residual = cents - trend

        # Normalised autocorrelation of residual
        ac = np.correlate(residual, residual, mode='full')
        ac = ac[len(residual) - 1:]
        if ac[0] < 1e-12:
            results.append(rec)
            continue
        ac /= ac[0]

        # Search lags corresponding to rate_range
        lag_min = max(1, int(frame_rate / rate_range[1]))
        lag_max = min(len(ac) - 1, int(frame_rate / rate_range[0]))
        if lag_min >= lag_max:
            results.append(rec)
            continue

        peak_lag = lag_min + int(np.argmax(ac[lag_min:lag_max + 1]))
        peak_val = float(ac[peak_lag])

        rate_hz = float(frame_rate / peak_lag)
        depth_cents = float(np.max(residual) - np.min(residual))

        if peak_val > 0.3 and depth_cents >= depth_cents_min:
            rec["has_vibrato"] = True
            rec["rate_hz"] = rate_hz
            rec["depth_cents"] = depth_cents
            rec["regularity"] = peak_val

        results.append(rec)

    return results


# ═══════════════════════════════════════════════════════════════════════
# Feature 23 — DTW Reference Pitch Comparison
# ═══════════════════════════════════════════════════════════════════════

def compute_dtw_distance(f0_user, f0_ref, hop_length=HOP_LENGTH, sr=SR,
                         voiced_only=True):
    """DTW alignment of user F0 against a reference F0 track.

    Both sequences are converted to cents relative to their mean pitch so that
    absolute key/transposition differences are removed. Only voiced frames (>0)
    participate when `voiced_only=True`.

    Uses a simple pure-numpy DTW (no scipy) so it works in any environment.
    For large arrays (>10s clips), consider downsampling before calling.

    Args:
        f0_user:    (T,)  user F0 in Hz (0 = unvoiced)
        f0_ref:     (T',) reference F0 in Hz (0 = unvoiced), need not match T
        hop_length: shared hop in samples
        sr:         sample rate
        voiced_only: if True, strips unvoiced frames before alignment

    Returns:
        dict with:
            dtw_distance:       normalised DTW path cost (cents, lower = better)
            mean_deviation_cents: mean absolute deviation along path
            max_deviation_cents:  max absolute deviation along path
            path_length:        number of (i,j) steps in optimal path
    """
    def _to_cents(f0):
        v = f0[f0 > 0] if voiced_only else f0[f0 > 0]
        if len(v) < 2:
            return None
        mean_hz = np.mean(v)
        return 1200.0 * np.log2(v / (mean_hz + 1e-10))

    c_user = _to_cents(f0_user)
    c_ref  = _to_cents(f0_ref)

    if c_user is None or c_ref is None or len(c_user) == 0 or len(c_ref) == 0:
        return {"dtw_distance": float("nan"), "mean_deviation_cents": float("nan"),
                "max_deviation_cents": float("nan"), "path_length": 0}

    N, M = len(c_user), len(c_ref)

    # Cost matrix (absolute cents difference)
    cost = np.abs(c_user[:, None] - c_ref[None, :])  # (N, M)

    # DTW accumulation (standard DP)
    D = np.full((N, M), np.inf, dtype=np.float64)
    D[0, 0] = cost[0, 0]
    for i in range(1, N):
        D[i, 0] = D[i - 1, 0] + cost[i, 0]
    for j in range(1, M):
        D[0, j] = D[0, j - 1] + cost[0, j]
    for i in range(1, N):
        for j in range(1, M):
            D[i, j] = cost[i, j] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    # Traceback to collect deviations
    i, j = N - 1, M - 1
    deviations = []
    path_len = 0
    while i > 0 or j > 0:
        deviations.append(cost[i, j])
        path_len += 1
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            step = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
            if step == 0:
                i -= 1; j -= 1
            elif step == 1:
                i -= 1
            else:
                j -= 1
    deviations.append(cost[0, 0])
    path_len += 1

    deviations = np.array(deviations, dtype=np.float32)
    normalised = float(D[N - 1, M - 1]) / (path_len + 1e-10)

    return {
        "dtw_distance": normalised,
        "mean_deviation_cents": float(np.mean(deviations)),
        "max_deviation_cents":  float(np.max(deviations)),
        "path_length": path_len,
    }


# ═══════════════════════════════════════════════════════════════════════
# Feature Ph — Phrase Aggregation (UX critical)
# ═══════════════════════════════════════════════════════════════════════

_TECHNIQUE_NAMES = ["vibrato", "breathy", "falsetto", "belt"]

def phrase_aggregate(f0_hz, vad, technique_probs=None, rms_db=None,
                     sr=SR, hop_length=HOP_LENGTH,
                     min_gap_frames=15, min_phrase_frames=10):
    """Segment audio into phrases and compute per-phrase coaching metrics.

    Phrases are defined as voiced regions separated by VAD gaps ≥ `min_gap_frames`
    (default 15 frames = 150ms at 10ms hop). Short phrases below
    `min_phrase_frames` are skipped.

    Args:
        f0_hz:          (T,) per-frame F0 in Hz (0 = unvoiced)
        vad:            (T,) per-frame VAD probability
        technique_probs:(T, 4) sigmoid outputs for [vibrato, breathy, falsetto, belt];
                        or None to skip technique per-phrase means
        rms_db:         (T,) RMS energy in dBFS; or None to skip dynamics
        sr:             sample rate
        hop_length:     hop in samples
        min_gap_frames: minimum unvoiced gap to split phrases (default = 150ms)
        min_phrase_frames: minimum phrase duration to report (default = 100ms)

    Returns:
        list of dicts, one per phrase:
            start_frame, end_frame, duration_s,
            f0_mean_hz, f0_std_hz, f0_range_cents,
            vibrato: {has_vibrato, rate_hz, depth_cents, regularity},
            technique_means: {vibrato: float, breathy: float, ...} | None,
            rms_mean_db, rms_arc_db: None if rms_db not provided
    """
    hop_s = hop_length / sr
    voiced = vad > 0.5

    # Find voiced phrase boundaries
    changes = np.diff(voiced.astype(int), prepend=0, append=0)
    v_starts = np.where(changes == 1)[0]
    v_ends   = np.where(changes == -1)[0]

    # Merge phrases separated by gaps smaller than min_gap_frames
    if len(v_starts) == 0:
        return []

    merged_starts = [v_starts[0]]
    merged_ends   = [v_ends[0]]
    for i in range(1, len(v_starts)):
        gap = v_starts[i] - merged_ends[-1]
        if gap < min_gap_frames:
            merged_ends[-1] = v_ends[i]
        else:
            merged_starts.append(v_starts[i])
            merged_ends.append(v_ends[i])

    phrases = []
    vib_results = compute_vibrato(f0_hz, sr=sr, hop_length=hop_length)
    # Index vibrato results by start_frame for O(1) lookup
    vib_by_start = {r["start_frame"]: r for r in vib_results}

    for ps, pe in zip(merged_starts, merged_ends):
        if pe - ps < min_phrase_frames:
            continue

        seg_f0 = f0_hz[ps:pe]
        voiced_f0 = seg_f0[seg_f0 > 0]
        duration_s = (pe - ps) * hop_s

        phrase = {
            "start_frame": int(ps),
            "end_frame":   int(pe),
            "duration_s":  float(duration_s),
            "f0_mean_hz":  float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else float("nan"),
            "f0_std_hz":   float(np.std(voiced_f0))  if len(voiced_f0) > 1 else float("nan"),
            "f0_range_cents": float(
                1200.0 * np.log2((voiced_f0.max() / (voiced_f0.min() + 1e-10)) + 1e-10)
            ) if len(voiced_f0) > 1 else float("nan"),
        }

        # Vibrato — find the sub-phrase that overlaps with [ps, pe]
        # compute_vibrato works on the full f0_hz array; pick matching entry
        phrase_vib = vib_by_start.get(ps)
        if phrase_vib is None:
            # Nearest start within 5 frames (merge may shift boundary slightly)
            for vr in vib_results:
                if abs(vr["start_frame"] - ps) <= 5 and vr["end_frame"] <= pe + 5:
                    phrase_vib = vr
                    break
        phrase["vibrato"] = {
            "has_vibrato": phrase_vib["has_vibrato"] if phrase_vib else False,
            "rate_hz":     phrase_vib["rate_hz"]    if phrase_vib else None,
            "depth_cents": phrase_vib["depth_cents"] if phrase_vib else None,
            "regularity":  phrase_vib["regularity"]  if phrase_vib else None,
        }

        # Technique means
        if technique_probs is not None:
            tp = np.asarray(technique_probs)
            seg_tech = tp[ps:pe]
            phrase["technique_means"] = {
                name: float(np.mean(seg_tech[:, k]))
                for k, name in enumerate(_TECHNIQUE_NAMES)
                if k < seg_tech.shape[1]
            }
        else:
            phrase["technique_means"] = None

        # RMS dynamics
        if rms_db is not None:
            seg_rms = rms_db[ps:pe]
            phrase["rms_mean_db"] = float(np.mean(seg_rms))
            # "arc" = difference between peak (climax) and start/end mean
            third = max(1, (pe - ps) // 3)
            rms_start = float(np.mean(seg_rms[:third]))
            rms_end   = float(np.mean(seg_rms[-third:]))
            rms_peak  = float(np.max(seg_rms))
            phrase["rms_arc_db"] = rms_peak - max(rms_start, rms_end)
        else:
            phrase["rms_mean_db"] = None
            phrase["rms_arc_db"]  = None

        phrases.append(phrase)

    return phrases


# ═══════════════════════════════════════════════════════════════════════
# Convenience: extract all features at once
# ═══════════════════════════════════════════════════════════════════════

def extract_all(y, sr=SR, f0_hz=None, vad=None,
                hop_length=HOP_LENGTH, frame_length=WIN_LENGTH):
    """Extract all Phase 0/2 acoustic features from a waveform.

    Args:
        y:      (N,) float32 waveform at `sr`
        f0_hz:  (T,) per-frame F0 in Hz (0 = unvoiced); if None, features
                requiring F0 are skipped
        vad:    (T,) per-frame binary VAD; if None, derived from f0_hz > 0

    Returns:
        dict with keys: rms_db, hnr_db, spectral_centroid, h1_h2,
                        jitter_pct, shimmer_pct, mfcc, breaths, onsets
        All per-frame arrays are (T,) or (T, K); events are lists of tuples.
    """
    feats = {}

    # Feature 14
    feats['rms_db'] = compute_rms(y, hop_length=hop_length,
                                   frame_length=frame_length)
    T = len(feats['rms_db'])

    # Align F0 / VAD to T
    if f0_hz is None:
        f0_hz = np.zeros(T, dtype=np.float32)
    else:
        f0_hz = np.asarray(f0_hz[:T], dtype=np.float32)

    if vad is None:
        vad = (f0_hz > 0).astype(np.float32)
    else:
        vad = np.asarray(vad[:T], dtype=np.float32)

    # Feature 15
    feats['hnr_db'] = compute_hnr(y, f0_hz, sr=sr,
                                   hop_length=hop_length,
                                   frame_length=frame_length)

    # Feature 16
    feats['spectral_centroid'] = compute_spectral_centroid(
        y, sr=sr, hop_length=hop_length)[:T]

    # Feature 17
    feats['h1_h2'] = compute_spectral_tilt(
        y, f0_hz, sr=sr, hop_length=hop_length,
        frame_length=frame_length)

    # Feature 18
    feats['jitter_pct'] = compute_jitter(f0_hz, hop_s=hop_length / sr)

    # Feature 19
    feats['shimmer_pct'] = compute_shimmer(y, f0_hz, sr=sr,
                                            hop_length=hop_length)

    # Feature 20
    feats['mfcc'] = compute_mfcc(y, sr=sr, hop_length=hop_length)[:T]

    # Feature 21
    feats['breaths'] = detect_breaths(y, vad, feats['rms_db'],
                                       sr=sr, hop_length=hop_length)

    # Feature 22
    feats['onsets'] = compute_onset_steepness(feats['rms_db'], vad,
                                               hop_s=hop_length / sr)

    # Features 5-7 — Vibrato (per phrase)
    feats['vibrato_phrases'] = compute_vibrato(f0_hz, sr=sr, hop_length=hop_length)

    return feats


def summarise(feats, f0_hz=None):
    """Return a flat dict of scalar coaching metrics from extract_all output.

    Suitable for inclusion in a coaching report JSON.
    """
    voiced = ~np.isnan(feats.get('hnr_db', np.array([np.nan])))

    def vmean(arr):
        v = arr[~np.isnan(arr)]
        return float(np.mean(v)) if len(v) > 0 else float('nan')

    # Vibrato summary across all voiced phrases
    vib_phrases = feats.get('vibrato_phrases', [])
    vib_phrases_with = [p for p in vib_phrases if p['has_vibrato']]
    n_phrases = len(vib_phrases)

    summary = {
        'rms_mean_db':           vmean(feats['rms_db']),
        'rms_range_db':          float(np.nanmax(feats['rms_db']) -
                                       np.nanmin(feats['rms_db'])),
        'hnr_mean_db':           vmean(feats['hnr_db']),
        'spectral_centroid_hz':  vmean(feats['spectral_centroid']),
        'h1_h2_mean_db':         vmean(feats['h1_h2']),
        'jitter_mean_pct':       vmean(feats['jitter_pct']),
        'shimmer_mean_pct':      vmean(feats['shimmer_pct']),
        'n_breaths':             len(feats['breaths']),
        'n_onsets':              len(feats['onsets']),
        'onset_steepness_mean':  (float(np.mean([s for _, s in feats['onsets']]))
                                  if feats['onsets'] else float('nan')),
        'n_voiced_segments':     n_phrases,
        'vibrato_phrase_frac':   (len(vib_phrases_with) / n_phrases
                                  if n_phrases > 0 else float('nan')),  # over voiced segments
        'vibrato_rate_hz_mean':  (float(np.mean([p['rate_hz'] for p in vib_phrases_with]))
                                  if vib_phrases_with else float('nan')),
        'vibrato_depth_cents_mean': (float(np.mean([p['depth_cents'] for p in vib_phrases_with]))
                                     if vib_phrases_with else float('nan')),
    }
    return summary
