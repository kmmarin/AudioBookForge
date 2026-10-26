#!/usr/bin/env bash
# Set up (or update) the clone server on a Windows PC with an NVIDIA GPU, in WSL
# (Ubuntu with systemd), from a bundle of this repo's clone/ folder:
#
#   wsl -u root -- bash clone-setup.sh --bundle /mnt/c/Users/me/forge-clone.tgz
#
# The token Forge uses (its CLONE_TOKEN, or SHARE_TOKEN) comes on stdin, so it
# never appears in a command line or log:  ... clone-setup.sh --bundle X < token-file
# Installs Docker and NVIDIA's container toolkit if missing, then builds and
# starts the clone server on port 8100. Reaching it from other machines also
# needs a port proxy and a firewall rule on the Windows side (clone/README.md).
set -euo pipefail
APP_DIR=/opt/audiobook-forge-clone
BUNDLE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle) BUNDLE="$2"; shift 2 ;;
    --app-dir) APP_DIR="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
say() { printf '\n==> %s\n' "$*"; }
die() { echo "Error: $*" >&2; exit 1; }
[[ $EUID -eq 0 ]] || die "Run this as root: wsl -u root -- bash clone-setup.sh ..."
[[ -n $BUNDLE && -f $BUNDLE ]] || die "Give the bundle: --bundle <path to forge-clone.tgz>"
[[ $(ps -p 1 -o comm=) == systemd ]] ||
  die "systemd isn't running. Add [boot] systemd=true to /etc/wsl.conf, run wsl --shutdown in Windows, then try again."
/usr/lib/wsl/lib/nvidia-smi -L >/dev/null 2>&1 || nvidia-smi -L >/dev/null 2>&1 ||
  die "No NVIDIA GPU visible in WSL. Install the current NVIDIA driver in Windows (not in WSL), then try again."

TOKEN=""
if [[ ! -t 0 ]]; then read -r TOKEN || true; fi

say "Installing Docker"
if command -v docker >/dev/null && docker compose version >/dev/null 2>&1; then
  echo "Already installed: $(docker --version)"
else
  . /etc/os-release
  apt-get update -q
  apt-get install -y -q ca-certificates curl gnupg
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL "https://download.docker.com/linux/$ID/gpg" -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/$ID ${UBUNTU_CODENAME:-$VERSION_CODENAME} stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -q
  apt-get install -y -q docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

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
systemctl enable --now docker >/dev/null

say "Unpacking into $APP_DIR"
mkdir -p "$APP_DIR"
tar -xzf "$BUNDLE" -C "$APP_DIR"
cd "$APP_DIR"
touch .env && chmod 600 .env
if [[ -n $TOKEN ]]; then
  grep -v '^CLONE_TOKEN=' .env > .env.new || true
  printf 'CLONE_TOKEN=%s\n' "$TOKEN" >> .env.new
  mv .env.new .env && chmod 600 .env
  echo "Token set."
fi

say "Building and starting the clone server"
docker compose -f docker-compose.clone.yml up -d --build
for _ in $(seq 1 60); do
  curl -sf http://127.0.0.1:8100/api/health >/dev/null && break
  sleep 2
done
curl -s http://127.0.0.1:8100/api/health | python3 -c "
import json, sys
h = json.load(sys.stdin)
print('Clone server:', h['device'], h.get('gpu'), h.get('gpu_gb'), 'GB,', h['dtype'])
for m in h['models']:
    print(' ', m['label'] + ':', 'fits' if m['fits'] else 'does not fit', '-', m.get('note') or '')
" || die "The clone server didn't answer on port 8100: docker compose -f docker-compose.clone.yml logs"
