# Voice creator for Audiobook Forge: trains Piper voices from narration.
# Built from the project root, because it shares a few modules with Forge:
#   docker build -f voicelab/Dockerfile -t voicelab .
FROM python:3.11-slim-bookworm

# cpu, or a CUDA build such as cu124 for training on an NVIDIA GPU
ARG TORCH_VARIANT=cpu
# The same torch Forge's GPU image runs; cu124 has no newer build.
ARG TORCH_VERSION=2.6.0
# piper1-gpl v1.3.0, the release Forge voices with. Its wheel leaves out the
# VITS training code (piper/train/vits), so that part comes from the source.
ARG PIPER_REF=fee9b9cefae4ebf9e196cfe994dea418f051506c

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LAB_DATA=/labdata \
    MODELS_DIR=/models \
    HF_HOME=/models/hf

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg espeak-ng gosu ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY voicelab/requirements.txt /tmp/requirements.txt
RUN pip install "torch==${TORCH_VERSION}" "torchaudio==${TORCH_VERSION}" \
      --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}" \
 && pip freeze | grep -iE '^(torch|torchaudio)==' > /tmp/torch-pin.txt \
 && pip install -r /tmp/requirements.txt -c /tmp/torch-pin.txt

# The training code, with its Cython alignment search built for this Python.
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
 && site=$(python -c 'import piper, os; print(os.path.dirname(piper.__file__))') \
 && curl -fsSL "https://github.com/OHF-voice/piper1-gpl/archive/${PIPER_REF}.tar.gz" | tar -xz -C /tmp \
 && cp -r "/tmp/piper1-gpl-${PIPER_REF}/src/piper/train/vits" "$site/train/vits" \
 && cp "/tmp/piper1-gpl-${PIPER_REF}/COPYING" "$site/train/COPYING" \
 && cd "$site/train/vits/monotonic_align" \
 && cythonize -i core.pyx && mkdir -p monotonic_align && mv core*.so monotonic_align/ \
 && rm -rf "/tmp/piper1-gpl-${PIPER_REF}" core.c build \
 && apt-get purge -y gcc libc6-dev && apt-get autoremove -y && rm -rf /var/lib/apt/lists/* \
 && python -c 'from piper.train.vits.monotonic_align import maximum_path; from piper.train.vits.lightning import VitsModel'

# faster-whisper and torch on the GPU need these CUDA libraries; the CUDA torch
# wheels ship them here. The folders don't exist on the CPU build.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.11/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.11/site-packages/nvidia/curand/lib:/usr/local/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.11/site-packages/nvidia/nvjitlink/lib

# Forge's package: the GPU sharing rules, the Whisper word-error measure and
# the stylesheet are shared with it, so the two never drift apart.
COPY app ./app
COPY voicelab/lab ./lab
COPY voicelab/entrypoint.sh /usr/local/bin/voicelab-entrypoint.sh
RUN chmod +x /usr/local/bin/voicelab-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["voicelab-entrypoint.sh"]
CMD ["uvicorn", "lab.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
