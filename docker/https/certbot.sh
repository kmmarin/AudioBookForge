#!/bin/sh
# Gets a Let's Encrypt certificate for $DOMAIN through your DNS provider (no
# ports need to be open to the internet), then keeps it renewed by itself:
# certbot checks twice a day and renews when it's within 30 days of expiring.
# nginx notices a renewed certificate and reloads within ten minutes.
# Set up with ./https-setup.sh, which writes the credentials file used here.
set -eu

provider=${DNS_PROVIDER:-cloudflare}
creds=/etc/letsencrypt/dns-credentials.ini
if [ ! -f "$creds" ]; then
  echo "No DNS credentials yet. Run ./https-setup.sh in the Audiobook Forge folder."
  sleep 3600
  exit 1
fi
chmod 600 "$creds"

case "$provider" in
  route53)  # the AWS plugin reads a standard AWS credentials file
    export AWS_SHARED_CREDENTIALS_FILE="$creds"
    auth="--dns-route53" ;;
  cloudflare)
    auth="--dns-cloudflare --dns-cloudflare-credentials $creds --dns-cloudflare-propagation-seconds 30" ;;
  *)
    auth="--dns-$provider --dns-$provider-credentials $creds" ;;
esac
if [ -n "${ACME_EMAIL:-}" ]; then account="--email $ACME_EMAIL"; else account="--register-unsafely-without-email"; fi
staging=""
[ "${ACME_STAGING:-false}" = "true" ] && staging="--staging"

if [ ! -f "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ]; then
  echo "Getting a certificate for $DOMAIN from Let's Encrypt (DNS check through $provider)"
  # shellcheck disable=SC2086
  if ! certbot certonly --non-interactive --agree-tos $account $staging $auth \
      --cert-name "$DOMAIN" -d "$DOMAIN"; then
    # Let's Encrypt limits failed attempts, so don't retry in a tight loop.
    echo "Couldn't get a certificate. Check the DNS name and that the API key can edit DNS for it, then run ./https-setup.sh again. Retrying in an hour."
    sleep 3600
    exit 1
  fi
fi

echo "Certificate ready for $DOMAIN. It renews by itself; checking twice a day."
while :; do
  sleep 12h
  certbot renew --non-interactive --quiet || echo "Renewal check failed; trying again in 12 hours."
done
