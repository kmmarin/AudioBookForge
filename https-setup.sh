#!/usr/bin/env bash
# HTTPS for Audiobook Forge in one step. Asks for a DNS name and an API key,
# gets a Let's Encrypt certificate through your DNS provider (nothing has to be
# open to the internet), and serves the app at https://NAME on port 443 (or a
# port you pick). The certificate renews by itself.
#
#   ./https-setup.sh                set up, or change the name, key or port; in the
#                                   project folder, sets up the server named in https.env
#   ./https-setup.sh --env FILE     the same, with the answers from a file (see https.env)
#   ./https-setup.sh --status       show the certificate and when it expires
#   ./https-setup.sh --off          go back to plain http on port 8080
set -Eeuo pipefail

die() { printf '\nError: %s\n' "$*" >&2; exit 1; }
say() { printf '\n==> %s\n' "$*"; }

# Answers from a file instead of questions: read it before changing folder.
answers=""
if [[ ${1:-} == --env ]]; then
  [[ -f ${2:-} ]] || die "No settings file at “${2:-}”."
  answers=$(cat "$2")
  shift 2
fi
from_file() {  # a KEY=value from the answers file, without quotes; never run as code
  printf '%s\n' "$answers" | grep -E "^$1=" | tail -n1 | cut -d= -f2- | sed -E "s/^[\"']//; s/[\"']$//; s/[[:space:]]+$//"
}

cd "$(dirname "$0")"
if [[ ! -f .env && -f .env.example ]]; then
  # Run from the project folder: send the answers in https.env to the server
  # named in it, set HTTPS up there, then clear the key from here.
  [[ -f https.env ]] || die "Copy https.env.example to https.env next to this script, fill it in, then run it again."
  answers=$(cat https.env)
  server=$(from_file SERVER)
  app_dir=$(from_file APP_DIR); app_dir=${app_dir:-/opt/audiobook-forge}
  [[ -n $server ]] || die "Put the server in https.env, like SERVER=root@192.168.1.10."
  [[ -n $(from_file DOMAIN) ]] || die "Fill in DOMAIN in https.env, like books.example.com."
  [[ -n $(from_file CLOUDFLARE_API_TOKEN)$(from_file API_KEY) ]] || die "Fill in the API key in https.env (CLOUDFLARE_API_TOKEN for Cloudflare)."
  say "Setting up HTTPS on $server"
  # shellcheck disable=SC2016  # expanded on the server
  if ssh "$server" 'f=$(mktemp); cat > "$f"; cd "$1" && ./https-setup.sh --env "$f"; s=$?; rm -f "$f"; exit $s' _ "$app_dir" < https.env; then
    sed -i 's/^CLOUDFLARE_API_TOKEN=.*/CLOUDFLARE_API_TOKEN=/; s/^API_KEY=.*/API_KEY=/; s/^AWS_SECRET_ACCESS_KEY=.*/AWS_SECRET_ACCESS_KEY=/' https.env
    echo "Cleared the key from https.env; the server keeps it for renewals."
  else
    die "The setup on $server didn't finish (see above). Nothing was cleared from https.env."
  fi
  exit 0
fi
[[ -f .env ]] || die "Run this in the Audiobook Forge folder (there's no .env here)."
command -v docker >/dev/null || die "Docker isn't installed here."
[[ $EUID -eq 0 ]] || docker info >/dev/null 2>&1 || die "Run it as root (or as a user in the docker group)."

get_env() { grep -E "^$1=" .env | tail -n1 | cut -d= -f2- || true; }
set_env() {
  if grep -q "^$1=" .env; then sed -i "s|^$1=.*|$1=$2|" .env; else echo "$1=$2" >> .env; fi
}

# The same compose files the deploy scripts use: GPU if this Forge has one.
files=(-f docker-compose.yml)
if docker inspect audiobook-forge-worker-1 --format '{{.Config.Image}}' 2>/dev/null | grep -q ':gpu$'; then
  files+=(-f docker-compose.gpu.yml)
  # The voice creator, if it's on, so --remove-orphans leaves it running.
  if [[ $(get_env VOICELAB) == true && -f docker-compose.voice.yml ]]; then files+=(-f docker-compose.voice.yml); fi
fi
compose() { docker compose "${files[@]}" "$@"; }
https_files=("${files[@]}" -f docker-compose.https.yml)

creds=certs/dns-credentials.ini
domain=$(get_env DOMAIN)

case "${1:-}" in
  --status)
    [[ -n $domain ]] || { echo "HTTPS is off."; exit 0; }
    docker compose "${https_files[@]}" exec -T certbot certbot certificates 2>/dev/null \
      || echo "HTTPS is set up for $domain, but the certbot service isn't running."
    exit 0 ;;
  --off)
    set_env DOMAIN ""
    set_env SITE_URL ""
    compose up -d --remove-orphans
    echo "HTTPS is off. Forge is on http://$(hostname -I | awk '{print $1}'):$(get_env PORT)."
    echo "The certificate stays in ./certs, so turning HTTPS back on is quick."
    exit 0 ;;
  "") ;;
  *) sed -n '2,13p' "$0"; exit 0 ;;
esac

echo "HTTPS setup for Audiobook Forge"
[[ -n $answers ]] && echo "Using the answers in the settings file." || echo "Press Enter to keep what's in [brackets]."

# 1. DNS provider -------------------------------------------------------------
providers=(cloudflare digitalocean linode dnsimple route53)
names=("Cloudflare" "DigitalOcean" "Linode (Akamai)" "DNSimple" "Amazon Route 53")
current=$(get_env DNS_PROVIDER); current=${current:-cloudflare}
if [[ -n $answers ]]; then
  provider=$(from_file DNS_PROVIDER | tr 'A-Z' 'a-z'); provider=${provider:-cloudflare}
  pick=0
  for i in "${!providers[@]}"; do [[ ${providers[$i]} == "$provider" ]] && pick=$((i + 1)); done
  (( pick )) || die "DNS_PROVIDER should be one of: ${providers[*]}."
else
  default_n=1
  for i in "${!providers[@]}"; do [[ ${providers[$i]} == "$current" ]] && default_n=$((i + 1)); done
  echo
  echo "Who runs DNS for your domain?"
  for i in "${!providers[@]}"; do echo "  $((i + 1))) ${names[$i]}"; done
  read -r -p "Choose [$default_n]: " pick
  pick=${pick:-$default_n}
  [[ $pick =~ ^[1-5]$ ]] || die "Pick a number from 1 to 5."
  provider=${providers[$((pick - 1))]}
fi

# 2. DNS name -----------------------------------------------------------------
if [[ -n $answers ]]; then
  name=$(from_file DOMAIN)
  [[ -n $name ]] || die "Fill in DOMAIN in the settings file, like books.example.com."
else
  echo
  read -r -p "DNS name for Forge, like books.example.com${domain:+ [$domain]}: " name
fi
name=${name:-$domain}
name=$(echo "$name" | tr 'A-Z' 'a-z' | sed 's|^https\?://||; s|/.*$||')
[[ $name =~ ^([a-z0-9]([a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}$ ]] || die "“$name” isn't a DNS name like books.example.com."

# 3. API key --------------------------------------------------------------------
keep=false
if [[ -f $creds && $provider == "$current" ]]; then keep=true; fi
echo
case "$provider" in
  cloudflare)   prompt="Cloudflare API token (needs Zone > DNS > Edit)"; field=dns_cloudflare_api_token ;;
  digitalocean) prompt="DigitalOcean API token (read and write)"; field=dns_digitalocean_token ;;
  linode)       prompt="Linode API token (Domains: read/write)"; field=dns_linode_key ;;
  dnsimple)     prompt="DNSimple API token"; field=dns_dnsimple_token ;;
  route53)      prompt="AWS access key ID (with Route 53 change rights)"; field="" ;;
esac
secret=""
if [[ -n $answers ]]; then
  if [[ $provider == cloudflare ]]; then key=$(from_file CLOUDFLARE_API_TOKEN); fi
  key=${key:-$(from_file API_KEY)}
  secret=$(from_file AWS_SECRET_ACCESS_KEY)
else
  read -r -s -p "$prompt$($keep && echo ' [keep the saved one]'): " key; echo
  if [[ $provider == route53 && -n $key ]]; then
    read -r -s -p "AWS secret access key: " secret; echo
  fi
fi
if [[ -z $key ]]; then
  $keep || die "An API key is needed to prove to Let's Encrypt that you control $name."
fi

# 4. Port ---------------------------------------------------------------------
port_now=$(get_env HTTPS_PORT)
if [[ -n $answers ]]; then
  port=$(from_file HTTPS_PORT)
else
  echo
  read -r -p "HTTPS port [${port_now:-443}]: " port
fi
port=${port:-${port_now:-443}}
[[ $port =~ ^[0-9]+$ ]] && (( port > 0 && port < 65536 )) || die "A port is a number from 1 to 65535."

# Check the Cloudflare token before using it.
if [[ $provider == cloudflare && -n $key ]]; then
  ok=$(curl -fsS -m 15 -H "Authorization: Bearer $key" https://api.cloudflare.com/client/v4/user/tokens/verify 2>/dev/null \
       | python3 -c 'import json,sys; print(json.load(sys.stdin)["result"]["status"])' 2>/dev/null || true)
  [[ $ok == active ]] || die "Cloudflare didn't accept that token. Make one at dash.cloudflare.com > My Profile > API Tokens (template: Edit zone DNS)."
fi

# Save the key where certbot reads it, readable by root only.
mkdir -p certs
if [[ -n $key ]]; then
  umask 077
  if [[ $provider == route53 ]]; then
    printf '[default]\naws_access_key_id = %s\naws_secret_access_key = %s\n' "$key" "$secret" > "$creds"
  elif [[ $provider == linode ]]; then
    printf '%s = %s\ndns_linode_version = 4\n' "$field" "$key" > "$creds"
  else
    printf '%s = %s\n' "$field" "$key" > "$creds"
  fi
  umask 022
fi

# Point the name at this machine: automatic for Cloudflare.
ip=$(hostname -I | awk '{print $1}')
if [[ $provider == cloudflare ]]; then
  token=${key:-$(sed -n 's/^dns_cloudflare_api_token = //p' "$creds")}
  say "Pointing $name at $ip in Cloudflare"
  python3 - "$name" "$ip" "$token" <<'EOF' || echo "Couldn't set the DNS record; add an A record for $name -> $ip yourself (DNS only, not proxied)."
import json, sys, urllib.request
name, ip, token = sys.argv[1:]
def cf(method, path, body=None):
    req = urllib.request.Request("https://api.cloudflare.com/client/v4" + path, method=method,
                                 data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)["result"]
parts = name.split(".")
zone = None
for i in range(len(parts) - 1):  # books.example.com, then example.com
    found = cf("GET", "/zones?name=" + ".".join(parts[i:]))
    if found:
        zone = found[0]["id"]
        break
if not zone:
    sys.exit(f"No Cloudflare zone for {name} that this token can see.")
record = {"type": "A", "name": name, "content": ip, "ttl": 300, "proxied": False,
          "comment": "Audiobook Forge (LAN address)"}
existing = cf("GET", f"/zones/{zone}/dns_records?type=A&name={name}")
if existing:
    cf("PUT", f"/zones/{zone}/dns_records/{existing[0]['id']}", record)
    print(f"Updated: {name} -> {ip}")
else:
    cf("POST", f"/zones/{zone}/dns_records", record)
    print(f"Added: {name} -> {ip}")
EOF
else
  echo
  echo "Point $name at $ip: add an A record at ${names[$((pick - 1))]} (or in your local DNS)."
fi

# Port 80 redirects plain http to https, unless something else already uses it.
if docker ps --format '{{.Names}}' | grep -q '^audiobook-forge-proxy-1$' \
   || ! (exec 3<>/dev/tcp/127.0.0.1/80) 2>/dev/null; then
  set_env HTTP_BIND 0.0.0.0; set_env HTTP_PORT 80
else
  echo "Port 80 is in use by something else, so http won't redirect to https; open https://$name directly."
  set_env HTTP_BIND 127.0.0.1; set_env HTTP_PORT 0
fi

# A new name or provider means a new certificate.
if [[ $name != "$domain" || $provider != "$current" ]]; then
  rm -rf "certs/live/$name" "certs/archive/$name" "certs/renewal/$name.conf"
fi
set_env DNS_PROVIDER "$provider"
set_env DOMAIN "$name"
set_env HTTPS_PORT "$port"
suffix=""; [[ $port != 443 ]] && suffix=":$port"
set_env HTTPS_SUFFIX "$suffix"
site="https://$name$suffix"
set_env SITE_URL "$site"
chmod +x docker/https/*.sh

say "Getting the certificate (this takes about a minute)"
docker compose "${https_files[@]}" up -d --quiet-pull --remove-orphans 2>&1 | grep -v -E 'Pulling|Pulled|Waiting|Download|Extract|Verifying|fs layer' || true
for _ in $(seq 60); do
  if [[ -f certs/live/$name/fullchain.pem ]]; then break; fi
  if docker compose "${https_files[@]}" logs --tail 5 certbot 2>/dev/null | grep -q "Couldn't get a certificate"; then
    docker compose "${https_files[@]}" logs --tail 30 certbot
    die "Let's Encrypt didn't issue the certificate (details above)."
  fi
  sleep 5
done
[[ -f certs/live/$name/fullchain.pem ]] || die "No certificate after 5 minutes. See: docker compose ${https_files[*]} logs certbot"

say "Checking that renewal will work by itself"
extra=()
[[ $provider == route53 ]] && extra=(-e AWS_SHARED_CREDENTIALS_FILE=/etc/letsencrypt/dns-credentials.ini)
if docker compose "${https_files[@]}" exec -T "${extra[@]}" certbot certbot renew --dry-run --cert-name "$name" >/tmp/forge-renew.log 2>&1; then
  echo "Renewal works. It happens by itself about a month before the certificate expires."
else
  tail -n 15 /tmp/forge-renew.log
  echo "The renewal rehearsal failed (above). The certificate works now, but check the API key before it's due."
fi

say "Checking https://$name"
for _ in $(seq 20); do
  if curl -fsS -m 5 --resolve "$name:$port:127.0.0.1" "https://$name:$port/api/health" >/dev/null 2>&1; then
    echo
    echo "Audiobook Forge is at $site"
    echo "  Certificate: Let's Encrypt, renews by itself. Status: ./https-setup.sh --status"
    echo "  Plain http still works on port $(get_env PORT) on your network."
    exit 0
  fi
  sleep 3
done
die "The certificate is in place but https didn't answer. See: docker compose ${https_files[*]} logs proxy"
