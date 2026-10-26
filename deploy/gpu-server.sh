#!/usr/bin/env bash
# Set a Linux machine with an NVIDIA GPU up as a GPU server for Audiobook Forge:
# the clone server, which speaks cloned voices and draws Watch and read's pictures
# for every Forge that lists it in Settings → GPU servers. One line:
#
#   curl -fsSL https://raw.githubusercontent.com/kmmarin/AudioBookForge/main/deploy/gpu-server.sh | sudo bash
#
# It installs Docker and NVIDIA's container toolkit if they're missing, builds and
# starts the clone server on port 8100, and ends by showing the address and token
# to paste into Forge's Settings. Run it again to update: the token and the
# downloaded models are kept. TOKEN=<token> and BRANCH=<branch> override.
set -euo pipefail
BRANCH=${BRANCH:-main}
[[ $EUID -eq 0 ]] || { echo "Run it as root: ... | sudo bash" >&2; exit 1; }

# The project: this checkout when run from one, else downloaded.
SRC=""
if [[ -f "${BASH_SOURCE[0]:-}" ]]; then
  here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
  [[ -f "$here/docker-compose.clone.yml" ]] && SRC=$here
fi
if [[ -z $SRC ]]; then
  SRC=/opt/audiobook-forge-src
  echo "==> Downloading Audiobook Forge ($BRANCH) to $SRC"
  rm -rf "$SRC" && mkdir -p "$SRC"
  curl -fsSL "https://github.com/kmmarin/AudioBookForge/archive/refs/heads/$BRANCH.tar.gz" | tar -xz -C "$SRC" --strip-components=1
fi

# The token: given, or the one this server already has (an update keeps it), or a new one.
token=${TOKEN:-$(sed -n 's/^CLONE_TOKEN=//p' /opt/audiobook-forge-clone/.env 2>/dev/null | tail -n1)}
[[ -n $token ]] || token=$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')

printf '%s\n' "$token" | bash "$SRC/deploy/clone-setup.sh" --from "$SRC"

ip=$(hostname -I 2>/dev/null | awk '{print $1}')
echo
echo "Done. In Forge, open Settings > GPU servers > Add a GPU server, and paste:"
echo "  Address: http://${ip:-<this machine>}:8100"
echo "  Token:   $token"
echo "Give this machine a fixed address (a DHCP reservation) so Forge keeps finding it."
