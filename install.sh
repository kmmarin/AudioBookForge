#!/usr/bin/env bash
# One-step install on any Linux host with Docker (VM, bare metal or Proxmox LXC).
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker isn't installed. Install it with:  curl -fsSL https://get.docker.com | sh"
  exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
  echo "The Docker Compose plugin is missing. Install the docker-compose-plugin package."
  exit 1
fi

if [ ! -f .env ]; then
  cp .env.example .env
  sed -i "s/^PUID=.*/PUID=$(id -u)/; s/^PGID=.*/PGID=$(id -g)/" .env
  echo "Created .env (edit it to change ports, folders or speed settings)."
fi
mkdir -p data output

files=(-f docker-compose.yml)
if command -v nvidia-smi >/dev/null 2>&1 && docker info 2>/dev/null | grep -qi nvidia; then
  echo "NVIDIA GPU found; using the GPU build."
  files+=(-f docker-compose.gpu.yml)
  # The voice creator (unless VOICELAB=false) and the clone server on this GPU too:
  # CLONE_LOCAL=auto (the default) runs it on a card with 12 GB or more.
  if ! grep -qE '^VOICELAB=false' .env; then
    files+=(-f docker-compose.voice.yml)
    mkdir -p voicelab-data
    echo "Voice creator: on."
  fi
  clone_local=$(sed -n 's/^CLONE_LOCAL=//p' .env | tail -n1)
  if [ "${clone_local:-auto}" = auto ]; then
    mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -dc 0-9)
    clone_local=$([ "${mib:-0}" -ge 11500 ] && echo true || echo false)
  fi
  if [ "$clone_local" = true ]; then
    files+=(-f docker-compose.clone.yml)
    grep -qE '^CLONE_URL=.+' .env || sed -i 's|^CLONE_URL=.*|CLONE_URL=http://clone:8100|' .env
    grep -qE '^CLONE_GPU_SHARE=' .env || echo "CLONE_GPU_SHARE=0.6" >> .env
    echo "Clone server: on (cloned voices and Watch and read's pictures)."
  fi
fi

docker compose "${files[@]}" up -d --build
echo "Downloading voices and the Whisper model (one time, a few minutes)..."
docker compose "${files[@]}" run --rm worker python -m app.prefetch || \
  echo "Prefetch didn't finish; models will download on first use instead."

port=$(grep -E '^PORT=' .env | cut -d= -f2)
echo
echo "Audiobook Forge is running at http://$(hostname -I | awk '{print $1}'):${port:-8080}"
