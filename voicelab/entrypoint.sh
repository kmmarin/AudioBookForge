#!/bin/sh
# Runs as root just long enough to give PUID:PGID its folders, then drops to
# that user, the same one Forge runs as, so both can share the models volume.
set -e
PUID="${PUID:-1000}"
PGID="${PGID:-1000}"

if [ "$(id -u)" = "0" ]; then
  getent group "$PGID" >/dev/null 2>&1 || groupadd -o -g "$PGID" forge
  getent passwd "$PUID" >/dev/null 2>&1 || useradd -o -u "$PUID" -g "$PGID" -M -d /models/home -s /usr/sbin/nologin forge
  mkdir -p "${LAB_DATA:-/labdata}" /models/home /models/custom-voices /models/gpu-share /models/voicelab /models/hf /models/whisper
  # Only this service's folders: Forge looks after the rest of /models.
  for dir in "${LAB_DATA:-/labdata}" /models/custom-voices /models/gpu-share /models/voicelab; do
    [ "$(stat -c %u "$dir")" = "$PUID" ] || chown -R "$PUID:$PGID" "$dir"
  done
  for dir in /models /models/home /models/hf /models/whisper; do
    [ "$(stat -c %u "$dir")" = "$PUID" ] || chown "$PUID:$PGID" "$dir"
  done
  export HOME=/models/home
  exec gosu "$PUID:$PGID" "$@"
fi
exec "$@"
