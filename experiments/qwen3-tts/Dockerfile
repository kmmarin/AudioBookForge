# Qwen3-TTS test bench: zero-shot cloning now, fine-tuning later (on a bigger GPU).
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git sox libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -U qwen-tts soundfile
ENV HF_HOME=/hf
WORKDIR /work
