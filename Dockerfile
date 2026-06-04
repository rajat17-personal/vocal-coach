FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Sessions dir must exist and be writable at runtime
RUN mkdir -p vocalcoach_sessions && chmod 777 vocalcoach_sessions

# Actual checkpoint filenames in ckpts/
ENV VOCALCOACH_CHECKPOINT=ckpts/spectilt_tech_vocalset_probe.pth
# ENV VOCALCOACH_GTSINGER_CHECKPOINT=ckpts/spectilt_tech_gtsinger_probe.pths
ENV VOCALCOACH_QUALITY_CHECKPOINT=ckpts/spectilt_quality_v4_lightrank_m0.005.pth
ENV VOCALCOACH_NOTE_CHECKPOINT=ckpts/spectilt_note_finetune.pth
ENV VOCALCOACH_PHRASE_GAP_MS=500
ENV VOCALCOACH_PHRASE_MIN_MS=300

# HF Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
