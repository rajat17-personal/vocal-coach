"""
VocalCoach — SingMOS-Pro Perceptual Quality Scorer (D6)
========================================================

Wraps the pretrained SingMOS-Pro predictor (wav2vec2-large backbone, trained on
singing voice SVS/SVC/SVR data) to add a MOS score to the coaching report.

Why SingMOS-Pro over UTMOS or base SingMOS:
  - UTMOS was trained on TTS speech — generalises poorly to singing
  - SingMOS-v1 uses wav2vec2-base (weaker backbone)
  - SingMOS-Pro uses wav2vec2-large, trained specifically on singing voice data
  - Both return a single MOS scalar at inference; sub-dimension labels
    (lyrics_score, melody_score) exist in the dataset but are NOT model outputs

SingMOS repo: https://github.com/South-Twilight/SingMOS  tag v1.1.2
Paper: arXiv:2510.01812 (SingMOS-Pro, ICASSP 2026)

Dependencies: torch, librosa, s3prl
  pip install s3prl   # needed by the wav2vec2 backbone

Graceful degradation: if the model cannot load, every function returns None —
the rest of the pipeline continues normally.

Usage
-----
    from src.singmos import score_mos, mos_grade

    mos = score_mos(y, sr=16000)        # float in [1,5] or None
    print(mos_grade(mos))               # "excellent" / "good" / ...
"""

import warnings
import numpy as np

_mos_model = None
_mos_device = None
_mos_lengths = None   # cached length tensor — reused across calls
_load_attempted = False

_SINGMOS_SR = 16000
_HUB_REPO   = "South-Twilight/SingMOS:v1.1.2"
_HUB_MODEL  = "singmos_pro"   # wav2vec2-large, singing-specific


def _load_model(device=None):
    """Load SingMOS-Pro via torch.hub once and cache as module singleton."""
    global _mos_model, _mos_device, _load_attempted

    if _load_attempted:
        return _mos_model
    _load_attempted = True

    try:
        import torch
        _mos_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # s3prl uses two torchaudio APIs removed in torchaudio 2.x:
        #   torchaudio.set_audio_backend()       — removed entirely
        #   torchaudio.sox_effects.apply_effects_tensor — removed with sox backend
        # Patch both as no-ops/stubs before s3prl imports to avoid AttributeError.
        # This is safe: SingMOS-Pro's wav2vec2 backbone does not use sox at all;
        # the broken imports are in unrelated s3prl upstream modules (byol_s,
        # mos_prediction) that are only imported as part of s3prl's __init__ sweep.
        import torchaudio as _ta
        import types as _types
        if not hasattr(_ta, "set_audio_backend"):
            _ta.set_audio_backend = lambda *a, **kw: None
        if not hasattr(_ta, "sox_effects"):
            _sox = _types.ModuleType("torchaudio.sox_effects")
            _sox.apply_effects_tensor = lambda waveform, sr, effects, **kw: (waveform, sr)
            _ta.sox_effects = _sox
            import sys as _sys
            _sys.modules["torchaudio.sox_effects"] = _sox

        print(f"[SingMOS] Loading {_HUB_MODEL} from {_HUB_REPO} on {_mos_device} ...")
        predictor = torch.hub.load(
            _HUB_REPO,
            _HUB_MODEL,
            trust_repo=True,
        )
        predictor = predictor.to(_mos_device).eval()
        _mos_model = predictor
        print("[SingMOS] SingMOS-Pro loaded successfully.")
        return _mos_model

    except Exception as e:
        warnings.warn(
            f"[SingMOS] Could not load SingMOS-Pro: {e}\n"
            "MOS scoring will be disabled. To enable:\n"
            "  pip install s3prl\n"
            "  # torch.hub will download weights automatically on first call",
            stacklevel=2,
        )
        return None


def score_mos(y: np.ndarray, sr: int = _SINGMOS_SR, device=None) -> "float | None":
    """Predict perceptual MOS for a singing waveform using SingMOS-Pro.

    Args:
        y:      (N,) float32 waveform
        sr:     sample rate of y — resampled to 16 kHz internally if needed
        device: torch device string, or None to auto-detect

    Returns:
        float in [1.0, 5.0] — predicted MOS (higher = better perceptual quality)
        None if SingMOS-Pro is unavailable
    """
    model = _load_model(device)
    if model is None:
        return None

    try:
        import torch
        import librosa

        if sr != _SINGMOS_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=_SINGMOS_SR)

        wave = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(_mos_device)  # (1, T)
        length = torch.tensor([wave.shape[1]], dtype=torch.long).to(_mos_device)

        with torch.no_grad():
            score = model(wave, length)

        # model returns a tensor or scalar
        if hasattr(score, "item"):
            score = score.item()
        return float(np.clip(score, 1.0, 5.0))

    except Exception as e:
        warnings.warn(f"[SingMOS] Inference failed: {e}", stacklevel=2)
        return None


def mos_grade(mos: "float | None") -> str:
    """Map a MOS score to a human-readable grade.

    Calibrated against SingMOS-Pro dataset distributions:
        professional SVS systems score ~3.8-4.3
        amateur singers typically ~2.5-3.2
    """
    if mos is None:
        return "unavailable"
    if mos >= 4.2:
        return "excellent"
    if mos >= 3.5:
        return "good"
    if mos >= 2.8:
        return "acceptable"
    if mos >= 2.0:
        return "needs_work"
    return "poor"
