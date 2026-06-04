"""
VocalCoach — Hugging Face Spaces entry point (Gradio + ZeroGPU).

Models are loaded onto CPU at startup.  Each analysis call temporarily moves
them to the ZeroGPU-allocated GPU for inference, then returns them to CPU so
the GPU can be released back to the pool.
"""

import os

# Set checkpoint paths before importing src.api (which reads env vars lazily
# inside _load_model, but _PHRASE_GAP_MS / _PHRASE_MIN_MS are read at import).
os.environ.setdefault("VOCALCOACH_CHECKPOINT",          "ckpts/spectilt_tech_vocalset_probe.pth")
# os.environ.setdefault("VOCALCOACH_GTSINGER_CHECKPOINT", "ckpts/spectilt_tech_gtsinger_probe.pth")
os.environ.setdefault("VOCALCOACH_QUALITY_CHECKPOINT",  "ckpts/spectilt_quality_v4_lightrank_m0.005.pth")
os.environ.setdefault("VOCALCOACH_NOTE_CHECKPOINT",     "ckpts/spectilt_note_finetune.pth")

import torch
import soundfile as sf
import librosa
import gradio as gr
import spaces

import src.api as _api
from src.api import _load_model, _analyse, score_report, _sanitize, PHRASE_MODES
from src.features import SR

# ── Model management ─────────────────────────────────────────────────────────

def _models_to(device_str: str) -> None:
    """Move every loaded model (and update the global device pointer)."""
    device = torch.device(device_str)
    _api._device = device
    for m in [_api._model, _api._gt_model, _api._quality_model, _api._note_model]:
        if m is not None:
            m.to(device)


# Load all checkpoints onto CPU now.  CUDA is not available outside @spaces.GPU,
# so _device will be set to "cpu" here and models will sit in CPU RAM.
print("[VocalCoach] loading models onto CPU …")
_load_model()
print(f"[VocalCoach] models ready on {_api._device}")


# ── Inference ────────────────────────────────────────────────────────────────

def _read_audio(path: str) -> "np.ndarray":
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y.astype("float32")


@spaces.GPU
def analyse(audio_path, reference_path, phrasing):
    if audio_path is None:
        return "Upload an audio file to get started.", None

    # Move models to ZeroGPU-allocated GPU for the duration of this call.
    _models_to("cuda")
    try:
        y = _read_audio(audio_path)
        y_ref = _read_audio(reference_path) if reference_path else None

        gap_ms, min_ms = PHRASE_MODES.get(phrasing, PHRASE_MODES["default"])
        report = _analyse(y, y_ref, phrase_gap_ms=gap_ms, phrase_min_ms=min_ms)
        report["coaching"] = score_report(report)
        report = _sanitize(report)

        return _format_summary(report), report

    finally:
        _models_to("cpu")
        torch.cuda.empty_cache()


# ── Summary formatter ─────────────────────────────────────────────────────────

def _fmt(val, unit="", decimals=1):
    if val is None or val != val:   # None or NaN
        return "—"
    return f"{round(val, decimals)}{unit}"


def _format_summary(report: dict) -> str:
    coaching  = report.get("coaching", {}) or {}
    pitch     = report.get("pitch", {}) or {}
    vib       = report.get("vibrato", {}) or {}
    tech      = report.get("technique", {}) or {}
    quality   = report.get("quality", {}) or {}
    vq        = report.get("voice_quality", {}) or {}
    ref_cmp   = report.get("reference_comparison", {}) or {}

    score = coaching.get("overall_score")
    lines = [f"## Overall score: {score}/100\n"]

    # Pitch
    f0    = pitch.get("f0_mean_hz")
    stab  = pitch.get("f0_stability_std_hz")
    lines.append(f"**Pitch:** {_fmt(f0, ' Hz', 0)} avg · stability ±{_fmt(stab, ' Hz')} std")

    # Vibrato
    frac  = vib.get("phrase_fraction")
    rate  = vib.get("rate_hz_mean")
    depth = vib.get("depth_cents_mean")
    lines.append(
        f"**Vibrato:** {_fmt(frac and frac*100, '%', 0)} of phrases · "
        f"{_fmt(rate, ' Hz')} rate · {_fmt(depth, ' ¢')} depth"
    )

    # Technique
    if tech:
        ranked = sorted(tech.items(), key=lambda x: -(x[1] or 0))
        tech_str = "  ".join(f"{k} {_fmt(v, decimals=2)}" for k, v in ranked)
        lines.append(f"**Technique:** {tech_str}")

    # Quality
    if quality:
        lines.append(f"**Quality:** {quality.get('overall_100')}/100")

    # Voice quality
    hnr = vq.get("hnr_mean_db")
    if hnr is not None:
        lines.append(f"**Voice quality:** HNR {_fmt(hnr, ' dB')}")

    # Reference comparison
    if ref_cmp:
        dev = ref_cmp.get("mean_deviation_cents")
        lines.append(f"**vs. reference:** {_fmt(dev, ' ¢')} mean deviation")

    # Coaching feedback
    strengths = coaching.get("strengths") or []
    improve   = coaching.get("areas_to_improve") or []
    if strengths:
        lines.append("\n**Strengths:** " + " · ".join(strengths))
    if improve:
        lines.append("**Work on:** " + " · ".join(improve))

    return "\n".join(lines)


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="VocalCoach", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🎤 VocalCoach\n"
        "Upload a singing clip to get pitch, vibrato, technique and quality analysis.\n"
        "Optionally upload a reference recording to compare against."
    )

    with gr.Row():
        audio_in = gr.Audio(label="Your singing", type="filepath")
        ref_in   = gr.Audio(label="Reference (optional)", type="filepath")

    phrasing = gr.Dropdown(
        choices=["default", "ballad", "uptempo"],
        value="default",
        label="Phrasing mode",
        info="'ballad' for slow songs with long phrases; 'uptempo' for fast/staccato.",
    )

    analyse_btn = gr.Button("Analyse", variant="primary")

    summary_out = gr.Markdown(label="Summary")
    report_out  = gr.JSON(label="Full report")

    analyse_btn.click(
        fn=analyse,
        inputs=[audio_in, ref_in, phrasing],
        outputs=[summary_out, report_out],
    )

demo.launch()
