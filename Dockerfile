# Audiobook Forge: one image runs both the web app and the worker.
FROM python:3.11-slim-bookworm

# cpu (default) or a CUDA build such as cu124 for NVIDIA GPUs
ARG TORCH_VARIANT=cpu

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATA_DIR=/data \
    OUTPUT_DIR=/output \
    MODELS_DIR=/models \
    HF_HOME=/models/hf \
    QT_QPA_PLATFORM=offscreen

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ffmpeg pandoc espeak-ng poppler-utils antiword fonts-dejavu-core gosu ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
# Install torch from the matching index first, then pin it so nothing swaps it
# for the (much larger) default CUDA build.
RUN pip install torch --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}" \
 && pip freeze | grep -i '^torch==' > /tmp/torch-pin.txt \
 && pip install -r requirements.txt -c /tmp/torch-pin.txt \
 && python -m spacy download en_core_web_sm
# GPU build: Piper on CUDA. onnxruntime-gpu 1.22 is the last release built for
# CUDA 12 (1.23+ need CUDA 13 and a newer driver), matching the torch wheels.
RUN if [ "$TORCH_VARIANT" != "cpu" ]; then \
      pip uninstall -y onnxruntime && pip install onnxruntime-gpu==1.22.0; \
    fi
# true adds Calibre for Kindle (.mobi .azw3) and other legacy ebook formats (~600 MB).
# Declared here, after the Python layers, so changing it doesn't download torch again.
ARG INSTALL_CALIBRE=false
RUN if [ "$INSTALL_CALIBRE" = "true" ]; then \
      apt-get update && apt-get install -y --no-install-recommends calibre \
      && rm -rf /var/lib/apt/lists/*; \
    fi
# faster-whisper and onnxruntime-gpu need these CUDA libraries; the CUDA torch
# wheels ship them here. The folders don't exist on the CPU build.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.11/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.11/site-packages/nvidia/curand/lib:/usr/local/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.11/site-packages/nvidia/nvjitlink/lib

COPY app ./app
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
