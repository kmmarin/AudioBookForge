#!/bin/sh
# Runs in the nginx container before nginx starts: wait for the first
# certificate, then watch it and reload nginx within ten minutes of a renewal.
cert="/etc/letsencrypt/live/$DOMAIN/fullchain.pem"
until [ -f "$cert" ]; do
  echo "Waiting for the certificate for $DOMAIN (see: docker compose logs certbot)"
  sleep 10
done
(
  last=$(stat -L -c %Y "$cert")
  while sleep 600; do
    now=$(stat -L -c %Y "$cert" 2>/dev/null || echo "$last")
    if [ "$now" != "$last" ]; then
      last=$now
      nginx -s reload && echo "Renewed certificate for $DOMAIN is in use"
    fi
  done
) &
