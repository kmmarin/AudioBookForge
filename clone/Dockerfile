# The clone server (server.py): Qwen3-TTS voices cloned from a reference clip.
# An image of its own: qwen-tts needs transformers 4.57.3, and Forge's image has
# 5.x for Kokoro, which qwen-tts's model code doesn't run on.
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends sox libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
# qwen-tts brings transformers 4.57.3, accelerate and gradio (with FastAPI and
# Uvicorn); these versions ran together on the GTX 1660 test bench.
RUN pip install --no-cache-dir qwen-tts==0.1.1 fastapi==0.162.0 uvicorn==0.54.0 soundfile==0.14.0

ENV HF_HOME=/hf CLONE_DATA=/data PYTHONUNBUFFERED=1
WORKDIR /srv
COPY server.py /srv/server.py
EXPOSE 8100
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8100"]
