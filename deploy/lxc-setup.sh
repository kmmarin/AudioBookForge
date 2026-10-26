#!/usr/bin/env bash
# Runs inside the container; proxmox-deploy.sh calls it. Installs Docker from
# Docker's apt repository, unpacks the app, builds and starts it, and
# downloads the models. Safe to run again: books, .env and models are kept.
set -Eeuo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export DEBIAN_FRONTEND=noninteractive LC_ALL=C.UTF-8

die() { printf '\nError: %s\n' "$*" >&2; exit 1; }
say() { printf '\n==> %s\n' "$*"; }
trap 'die "Line $LINENO failed: $BASH_COMMAND"' ERR

APP_DIR=/opt/audiobook-forge BUNDLE="" PORT="" TZ_NAME="" PREFETCH=true
DOCKER_BIP=10.202.0.1/24 DOCKER_POOL=10.203.0.0/16
while [[ $# -gt 0 ]]; do
  case $1 in
    --app-dir) APP_DIR=$2; shift ;;
    --bundle) BUNDLE=$2; shift ;;
    --port) PORT=$2; shift ;;
    --tz) TZ_NAME=$2; shift ;;
    --docker-bip) DOCKER_BIP=$2; shift ;;
    --docker-pool) DOCKER_POOL=$2; shift ;;
    --no-prefetch) PREFETCH=false ;;
    *) die "Unknown option: $1" ;;
  esac
  shift
done
[[ -f $BUNDLE ]] || die "Code bundle not found: $BUNDLE"
[[ $EUID -eq 0 ]] || die "Run this as root inside the container."

say "Installing Docker"
# Written before Docker first starts, so its networks never use 172.x.
docker_config_new=false
if [[ ! -f /etc/docker/daemon.json ]]; then
  mkdir -p /etc/docker
  cat > /etc/docker/daemon.json <<EOF
{
  "bip": "$DOCKER_BIP",
  "default-address-pools": [{"base": "$DOCKER_POOL", "size": 24}],
  "log-driver": "json-file",
  "log-opts": {"max-size": "10m", "max-file": "3"}
}
EOF
  docker_config_new=true
fi
if command -v docker >/dev/null && docker compose version >/dev/null 2>&1; then
  echo "Already installed: $(docker --version)"
  $docker_config_new && systemctl restart docker
else
  apt-get update -q
  apt-get install -y -q ca-certificates curl
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  codename=$(. /etc/os-release && echo "$VERSION_CODENAME")
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian $codename stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -q
  apt-get install -y -q docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
systemctl enable --now docker >/dev/null
command -v curl >/dev/null || apt-get install -y -q curl

driver=$(docker info --format '{{.Driver}}')
if [[ $driver == vfs ]]; then
  echo "Warning: Docker is using the vfs storage driver, which is slow and uses a lot of disk."
  echo "         Put the container on LVM-thin or a directory storage, or use ZFS 2.2+ (Proxmox 8.1+)."
fi

say "Unpacking the app into $APP_DIR"
mkdir -p "$APP_DIR"
rm -rf "$APP_DIR/app" "$APP_DIR/docker" "$APP_DIR/deploy" "$APP_DIR/voicelab"  # drop files removed upstream
tar -xzmf "$BUNDLE" -C "$APP_DIR" --no-same-owner  # m: fresh timestamps
cd "$APP_DIR"

set_env() {
  if grep -q "^$1=" .env; then
    sed -i "s|^$1=.*|$1=$2|" .env
  else
    echo "$1=$2" >> .env
  fi
}
if [[ ! -f .env ]]; then
  cp .env.example .env
  set_env TZ "${TZ_NAME:-UTC}"
  echo "Created .env (edit it for speed, format and Whisper settings)."
fi
[[ -n $PORT ]] && set_env PORT "$PORT"
set_env PUID 1000
set_env PGID 1000
mkdir -p data output
chmod +x https-setup.sh docker/https/*.sh 2>/dev/null || true

files=(-f docker-compose.yml)
# HTTPS stays on once ./https-setup.sh has set it up (a name and DNS credentials).
https_on() { grep -qE '^DOMAIN=.+' .env && [[ -f certs/dns-credentials.ini ]]; }
if https_on; then
  files+=(-f docker-compose.https.yml)
  echo "HTTPS is on for $(grep -E '^DOMAIN=' .env | cut -d= -f2)."
fi

say "Building and starting (the first build downloads about 1.5 GB)"
# web and worker share one image: build it once, not twice in parallel.
docker compose "${files[@]}" build worker
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
healthy=false
for _ in $(seq 60); do
  if curl -fsS -m 3 "http://127.0.0.1:$port/api/health" >/dev/null 2>&1; then
    healthy=true
    break
  fi
  sleep 2
done
docker compose "${files[@]}" ps
$healthy || die "The web app didn't answer on port $port. See: docker compose --project-directory $APP_DIR logs web"
echo "Healthy on port $port."
if https_on; then
  echo "HTTPS: $(grep -E '^SITE_URL=' .env | cut -d= -f2)"
else
  echo "For https with a certificate, run: cd $APP_DIR && ./https-setup.sh"
fi
