#!/bin/sh
# Runs as root just long enough to fix volume ownership, then drops to PUID:PGID
# so finished audiobooks are owned by your user on the host.
set -e
PUID="${PUID:-1000}"
PGID="${PGID:-1000}"

if [ "$(id -u)" = "0" ]; then
  getent group "$PGID" >/dev/null 2>&1 || groupadd -o -g "$PGID" forge
  getent passwd "$PUID" >/dev/null 2>&1 || useradd -o -u "$PUID" -g "$PGID" -M -d /models/home -s /usr/sbin/nologin forge
  mkdir -p /data /output /models/home
  for dir in /data /models; do
    [ "$(stat -c %u "$dir")" = "$PUID" ] || chown -R "$PUID:$PGID" "$dir"
  done
  [ "$(stat -c %u /output)" = "$PUID" ] || chown "$PUID:$PGID" /output 2>/dev/null || true
  export HOME=/models/home
  exec gosu "$PUID:$PGID" "$@"
fi
exec "$@"
