#!/usr/bin/env bash
# Deploy Audiobook Forge inside WSL 2 (Ubuntu), on the GPU when Windows shares one.
#
#   wsl -u root -- bash deploy/wsl-deploy.sh        from the project folder in Windows
#   sudo deploy/wsl-deploy.sh                        from a WSL shell
#
# Installs Docker Engine and, with an NVIDIA GPU, the NVIDIA Container Toolkit,
# copies the app into WSL's own disk (SQLite and builds are slow on /mnt/c),
# builds and starts it. Run it again to update: books, .env and models are kept.
# --settings-stdin: KEY=VALUE lines on stdin go into .env (deploy/windows-converter.ps1
# sends the server's address and token this way, so they never show in a command line).
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive

die() { printf '\nError: %s\n' "$*" >&2; exit 1; }
say() { printf '\n==> %s\n' "$*"; }
trap 'die "Line $LINENO failed: $BASH_COMMAND"' ERR

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
APP_DIR=/opt/audiobook-forge PORT="" PREFETCH=true GPU=auto SETTINGS=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --app-dir) APP_DIR=$2; shift ;;
    --port) PORT=$2; shift ;;
    --cpu) GPU=false ;;
    --no-prefetch) PREFETCH=false ;;
    # Windows may add a byte-order mark and carriage returns on the way in.
    --settings-stdin) SETTINGS=$(sed '1s/^\xEF\xBB\xBF//; s/\r$//' | grep -E '^[A-Z][A-Z0-9_]*=' || true) ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
  shift
done
[[ $EUID -eq 0 ]] || die "Run this as root: wsl -u root -- bash deploy/wsl-deploy.sh"
grep -qi microsoft /proc/version || echo "Note: this doesn't look like WSL; continuing anyway."
[[ $(ps -p 1 -o comm=) == systemd ]] ||
  die "systemd isn't running. Add [boot] systemd=true to /etc/wsl.conf, run wsl --shutdown in Windows, then try again."

if [[ $GPU == auto ]]; then
  GPU=false
  # WSL puts the Windows driver's nvidia-smi here; no Linux driver is installed.
  if /usr/lib/wsl/lib/nvidia-smi -L >/dev/null 2>&1 || nvidia-smi -L >/dev/null 2>&1; then GPU=true; fi
fi

say "Installing Docker"
if command -v docker >/dev/null && docker compose version >/dev/null 2>&1; then
  echo "Already installed: $(docker --version)"
else
  . /etc/os-release
  apt-get update -q
  apt-get install -y -q ca-certificates curl gnupg rsync
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL "https://download.docker.com/linux/$ID/gpg" -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/$ID ${UBUNTU_CODENAME:-$VERSION_CODENAME} stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -q
  apt-get install -y -q docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
command -v rsync >/dev/null || apt-get install -y -q rsync

if $GPU; then
  say "Setting up the GPU for Docker"
  if ! command -v nvidia-ctk >/dev/null; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |
      gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -q
    apt-get install -y -q nvidia-container-toolkit
  fi
  docker info 2>/dev/null | grep -q 'Runtimes:.*nvidia' || { nvidia-ctk runtime configure --runtime=docker; systemctl restart docker; }
fi
systemctl enable --now docker >/dev/null

say "Copying the app into $APP_DIR"
mkdir -p "$APP_DIR"
# --delete drops files removed upstream; local state and settings stay.
rsync -a --delete --exclude data/ --exclude output/ --exclude voicelab-data/ --exclude certs/ --exclude .env --exclude https.env --exclude __pycache__/ \
      --exclude '*.pyc' --exclude '*:Zone.Identifier' --exclude docker-compose.override.yml --exclude .git \
      "$PROJECT_DIR/" "$APP_DIR/"
cd "$APP_DIR"

set_env() {
  if grep -q "^$1=" .env; then sed -i "s|^$1=.*|$1=$2|" .env; else echo "$1=$2" >> .env; fi
}
owner=$(stat -c %u "$PROJECT_DIR")
if [[ ! -f .env ]]; then
  cp .env.example .env
  tz=$(timedatectl show -p Timezone --value 2>/dev/null || true)
  set_env TZ "${tz:-UTC}"
  echo "Created .env (edit it for speed, format and Whisper settings)."
fi
[[ -n $PORT ]] && set_env PORT "$PORT"
while IFS= read -r line; do
  [[ -n $line ]] && set_env "${line%%=*}" "${line#*=}"
done <<< "$SETTINGS"
# Finished books belong to your WSL user (the owner of the project folder), not root.
set_env PUID "$owner"
set_env PGID "$(stat -c %g "$PROJECT_DIR")"
mkdir -p data output

chmod +x https-setup.sh docker/https/*.sh 2>/dev/null || true
files=(-f docker-compose.yml)
services=(worker)
if $GPU; then
  files+=(-f docker-compose.gpu.yml)
  echo "GPU: $( (/usr/lib/wsl/lib/nvidia-smi -L 2>/dev/null || nvidia-smi -L) | head -n1)"
  # The voice creator (VOICELAB=true in .env) trains voices on the GPU, taking turns with Forge.
  if grep -qE '^VOICELAB=true' .env; then
    files+=(-f docker-compose.voice.yml)
    services+=(voicelab)
    mkdir -p voicelab-data
    echo "Voice creator: on (Create a voice, at the top of Forge's page)"
  fi
  # The clone server (cloned voices, Watch and read's pictures) on this GPU too:
  # Forge and the voice creator reach it by name, and it takes a smaller share of
  # the card, which it now shares with them.
  if grep -qE '^CLONE_LOCAL=true' .env; then
    files+=(-f docker-compose.clone.yml)
    services+=(clone)
    grep -qE '^CLONE_URL=.+' .env || set_env CLONE_URL http://clone:8100
    grep -qE '^CLONE_GPU_SHARE=.+' .env || set_env CLONE_GPU_SHARE 0.6
    grep -qE '^CLONE_BATCH=.+' .env || set_env CLONE_BATCH 8
    echo "Clone server: on, on this GPU (cloned voices and pictures)"
  fi
fi
# HTTPS stays on once ./https-setup.sh has set it up (a name and DNS credentials).
if grep -qE '^DOMAIN=.+' .env && [[ -f certs/dns-credentials.ini ]]; then
  files+=(-f docker-compose.https.yml)
  echo "HTTPS is on for $(grep -E '^DOMAIN=' .env | cut -d= -f2)."
fi
say "Building and starting (the first GPU build downloads about 3 GB)"
# web and worker share one image: build it once, not twice in parallel.
docker compose "${files[@]}" build "${services[@]}"
docker compose "${files[@]}" up -d --no-build --remove-orphans
docker image prune -f >/dev/null

if $PREFETCH; then
  say "Downloading voices and the Whisper model"
  docker compose "${files[@]}" run --rm worker python -m app.prefetch ||
    echo "Prefetch didn't finish; models will download on first use instead."
fi

say "Waiting for the web app"
port=$(grep -E '^PORT=' .env | cut -d= -f2)
port=${port:-8080}
for _ in $(seq 60); do
  if curl -fsS -m 3 "http://127.0.0.1:$port/api/health" >/dev/null 2>&1; then
    docker compose "${files[@]}" ps
    say "Audiobook Forge is running: http://localhost:$port (from Windows too)"
    echo "  Logs:    cd $APP_DIR && docker compose ${files[*]} logs -f"
    echo "  Output:  \\\\wsl.localhost\\${WSL_DISTRO_NAME:-Ubuntu}${APP_DIR//\//\\}\\output"
    exit 0
  fi
  sleep 2
done
docker compose "${files[@]}" ps
die "The web app didn't answer on port $port. See: cd $APP_DIR && docker compose ${files[*]} logs web"
