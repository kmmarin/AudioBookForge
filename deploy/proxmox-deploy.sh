#!/usr/bin/env bash
# Deploy Audiobook Forge to a Proxmox VE host: an unprivileged Debian 12 LXC
# with Docker, the app built and started, and the voices downloaded.
#
#   deploy/proxmox-deploy.sh root@pve        from your workstation (over SSH)
#   deploy/proxmox-deploy.sh                 on the Proxmox host itself
#
# Run it again to update. It finds the container by hostname, pushes the new
# code and rebuilds, keeping books, settings (.env) and downloaded models.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: proxmox-deploy.sh [user@host] [options]

Container
  --vmid N                Container ID. Default: the next free ID, or the
                          existing container with the same hostname (updated).
  --hostname NAME         Default: audiobook-forge
  --cores N               Default: 8, or every host core if there are fewer
  --memory MB             Default: 8192
  --disk GB               Root disk size. Default: 32
  --storage NAME          Storage for the root disk. Default: local-lvm,
                          local-zfs, or the first storage that holds containers.
  --template-storage NAME Where the Debian template is kept. Default: local
  --template VOLID        Use this template, e.g. local:vztmpl/debian-12-...zst
  --ssh-key FILE          Public key(s) allowed to log in to the container as root

Network
  --bridge NAME           Default: vmbr0
  --vlan N                VLAN tag
  --ip CIDR               Static address, e.g. 172.16.0.50/20. Default: DHCP
  --gw ADDRESS            Gateway for a static address
  --port N                Web port. Default: 8080
  --docker-bip CIDR       Docker's bridge inside the container. Default: 10.202.0.1/24
  --docker-pool CIDR      Docker's pool for app networks. Default: 10.203.0.0/16
                          (both kept off 172.x so they can't shadow LAN subnets)

App
  --output-dir PATH       Host folder to mount as the audiobook output, such as
                          an Audiobookshelf library. Default: inside the container.
  --no-prefetch           Don't download voices and Whisper now (first use will)

Other
  --yes                   Don't ask before creating the container
  --dry-run               Show what would change, change nothing
  --bundle-out FILE       Only write the code bundle to FILE, then stop
  -h, --help
EOF
}

die() { printf '\nError: %s\n' "$*" >&2; exit 1; }
say() { printf '\n==> %s\n' "$*"; }
trap 'die "Line $LINENO failed: $BASH_COMMAND"' ERR

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
APP_DIR=/opt/audiobook-forge

TARGET="" VMID="" CT_NAME=audiobook-forge CORES="" MEMORY=8192 DISK=32 STORAGE=""
TEMPLATE_STORAGE=local TEMPLATE="" SSH_KEY="" BRIDGE=vmbr0 VLAN="" IP="" GW="" PORT=""
DOCKER_BIP=10.202.0.1/24 DOCKER_POOL=10.203.0.0/16 OUTPUT_DIR="" PREFETCH=true
YES=false DRY=false BUNDLE="" BUNDLE_OUT="" TMP_DIR=""
PASS=()  # options forwarded to the host in SSH mode

if [[ $# -gt 0 && $1 != -* ]]; then TARGET=$1; shift; fi
while [[ $# -gt 0 ]]; do
  opt=$1
  case $opt in
    -h|--help) usage; exit 0 ;;
    --yes) YES=true ;;
    --dry-run) DRY=true ;;
    --no-prefetch) PREFETCH=false ;;
    --vmid|--hostname|--cores|--memory|--disk|--storage|--template-storage|--template|\
    --ssh-key|--bridge|--vlan|--ip|--gw|--port|--docker-bip|--docker-pool|--output-dir|\
    --bundle|--bundle-out)
      [[ $# -ge 2 ]] || die "$opt needs a value."
      val=$2; shift
      case $opt in
        --vmid) VMID=$val ;;            --hostname) CT_NAME=$val ;;
        --cores) CORES=$val ;;          --memory) MEMORY=$val ;;
        --disk) DISK=${val%[Gg]} ;;     --storage) STORAGE=$val ;;
        --template-storage) TEMPLATE_STORAGE=$val ;;
        --template) TEMPLATE=$val ;;    --ssh-key) SSH_KEY=$val ;;
        --bridge) BRIDGE=$val ;;        --vlan) VLAN=$val ;;
        --ip) IP=$val ;;                --gw) GW=$val ;;
        --port) PORT=$val ;;            --docker-bip) DOCKER_BIP=$val ;;
        --docker-pool) DOCKER_POOL=$val ;;
        --output-dir) OUTPUT_DIR=$val ;;
        --bundle) BUNDLE=$val ;;        --bundle-out) BUNDLE_OUT=$val ;;
      esac
      # The key file and bundle are local paths; SSH mode sends them separately.
      [[ $opt == --ssh-key || $opt == --bundle || $opt == --bundle-out ]] || PASS+=("$opt" "$val")
      shift; continue ;;
    *) die "Unknown option: $opt (see --help)" ;;
  esac
  PASS+=("$opt")
  shift
done

for n in "$MEMORY" "$DISK" ${VMID:+"$VMID"} ${CORES:+"$CORES"} ${PORT:+"$PORT"} ${VLAN:+"$VLAN"}; do
  [[ $n =~ ^[0-9]+$ ]] || die "Expected a number, got: $n"
done
[[ -z $GW || -n $IP ]] || die "--gw only applies with a static --ip."
[[ -z $IP || $IP == */* ]] || die "Give --ip with its prefix length, e.g. $IP/24."
[[ -z $SSH_KEY || -f $SSH_KEY ]] || die "SSH key file not found: $SSH_KEY"

# Runs a command that changes something; --dry-run only prints it.
run() {
  printf '+ %s\n' "$*"
  $DRY || "$@"
}

make_bundle() {
  local out=$1 items=() f
  for f in app docker deploy Dockerfile docker-compose.yml requirements.txt .env.example; do
    [[ -e $PROJECT_DIR/$f ]] || die "$PROJECT_DIR doesn't look like the Audiobook Forge project ($f is missing)."
    items+=("$f")
  done
  for f in docker-compose.gpu.yml docker-compose.https.yml docker-compose.voice.yml voicelab https-setup.sh .dockerignore install.sh README.md HANDOFF.md; do
    [[ -e $PROJECT_DIR/$f ]] && items+=("$f")
  done
  # Local state and this machine's docker-compose.override.yml stay behind.
  tar -C "$PROJECT_DIR" --exclude='__pycache__' --exclude='*.pyc' --exclude='*:Zone.Identifier' \
      -czf "$out" "${items[@]}"
}

# ---------------------------------------------------------------------------
# SSH mode: copy the code and this script to the host, run it there
# ---------------------------------------------------------------------------
remote_deploy() {
  local rdir status=0 tty=()
  TMP_DIR=$(mktemp -d)
  trap 'rm -rf "$TMP_DIR"' EXIT
  local ssh_opts=(-o ControlMaster=auto -o "ControlPath=$TMP_DIR/ssh-%C" -o ControlPersist=120)
  [[ -t 0 && -t 1 ]] && tty=(-t)

  make_bundle "$TMP_DIR/forge.tgz"
  say "Connecting to $TARGET"
  ssh "${ssh_opts[@]}" "$TARGET" 'command -v pct >/dev/null' ||
    die "$TARGET isn't a Proxmox VE host (it has no pct command), or SSH failed."
  rdir=$(ssh "${ssh_opts[@]}" "$TARGET" 'mktemp -d /tmp/forge-deploy.XXXXXX')
  local files=("$TMP_DIR/forge.tgz" "${BASH_SOURCE[0]}")
  local args=(--bundle "$rdir/forge.tgz" "${PASS[@]}")
  if [[ -n $SSH_KEY ]]; then
    cp "$SSH_KEY" "$TMP_DIR/authorized_keys"
    files+=("$TMP_DIR/authorized_keys")
    args+=(--ssh-key "$rdir/authorized_keys")
  fi
  scp -q "${ssh_opts[@]}" "${files[@]}" "$TARGET:$rdir/"

  local cmd
  cmd=$(printf '%q ' bash "$rdir/proxmox-deploy.sh" "${args[@]}")
  ssh "${tty[@]}" "${ssh_opts[@]}" "$TARGET" "$cmd; s=\$?; rm -rf $(printf %q "$rdir"); exit \$s" || status=$?
  ssh "${ssh_opts[@]}" -O exit "$TARGET" 2>/dev/null || true
  exit "$status"
}

# ---------------------------------------------------------------------------
# On the Proxmox host
# ---------------------------------------------------------------------------
ct_name_of() { pct config "$1" 2>/dev/null | awk '$1 == "hostname:" {print $2}'; }

find_ct_by_name() {
  local id
  for id in $(pct list | awk 'NR > 1 {print $1}'); do
    [[ $(ct_name_of "$id") == "$CT_NAME" ]] && echo "$id"
  done
  return 0
}

pick_storage() {
  local active
  active=$(pvesm status --content rootdir 2>/dev/null | awk 'NR > 1 && $3 == "active" {print $1}')
  if [[ -n $STORAGE ]]; then
    grep -qx "$STORAGE" <<<"$active" || die "Storage $STORAGE isn't active or can't hold containers. Choices: $(echo $active)"
    echo "$STORAGE"; return
  fi
  local s
  for s in local-lvm local-zfs; do
    grep -qx "$s" <<<"$active" && { echo "$s"; return; }
  done
  [[ -n $active ]] || die "No active storage can hold containers. Pick one with --storage."
  head -n1 <<<"$active"
}

# Newest Debian 12 template (supported by Proxmox 7.4 and later), else the newest Debian.
pick_template() {
  if [[ -n $TEMPLATE ]]; then echo "$TEMPLATE"; return; fi
  pveam update >/dev/null 2>&1 || echo "Couldn't refresh the template list; using the cached one." >&2
  local names name
  names=$(pveam available --section system | awk '{print $2}' | grep -E '^debian-[0-9]+-standard_.*_amd64\.tar\.(zst|gz|xz)$' || true)
  name=$(grep '^debian-12-' <<<"$names" | sort -V | tail -n1)
  [[ -n $name ]] || name=$(sort -V <<<"$names" | tail -n1)
  [[ -n $name ]] || die "No Debian template is available from pveam. Download one and pass --template."
  echo "$TEMPLATE_STORAGE:vztmpl/$name"
}

ensure_template() {
  local volid=$1 name
  name=${volid#*:vztmpl/}
  if pveam list "${volid%%:*}" 2>/dev/null | awk '{print $1}' | grep -qxF "$volid"; then
    echo "Template is already downloaded: $volid"
  else
    run pveam download "${volid%%:*}" "$name"
  fi
}

wait_for_network() {
  $DRY && return

  for _ in $(seq 60); do
    if pct exec "$VMID" -- sh -c 'ip -4 route show default | grep -q . && getent hosts deb.debian.org >/dev/null' 2>/dev/null; then
      return
    fi
    sleep 2
  done
  die "Container $VMID has no working network after 2 minutes. Check the bridge, VLAN and DHCP settings (pct config $VMID)."
}

confirm() {
  { $YES || $DRY; } && return
  [[ -t 0 ]] || die "Add --yes to run without a terminal."
  local answer
  read -r -p "$1 [y/N] " answer
  [[ $answer =~ ^[Yy] ]] || die "Stopped. Nothing was changed."
}

# An unprivileged container's uid 1000 is this uid on the host.
host_uid_for_ct_1000() {
  local base
  base=$(awk -F: '$1 == "root" {print $2; exit}' /etc/subuid 2>/dev/null)
  echo $(( ${base:-100000} + 1000 ))
}

create_ct() {
  local template storage host_cores host_mem cores
  host_cores=$(nproc)
  host_mem=$(awk '/^MemTotal:/ {print int($2 / 1024)}' /proc/meminfo)
  cores=${CORES:-$(( host_cores < 8 ? host_cores : 8 ))}
  storage=$(pick_storage)
  template=$(pick_template)

  local net="name=eth0,bridge=$BRIDGE,ip=${IP:-dhcp}"
  [[ -n $GW ]] && net+=",gw=$GW"
  [[ -n $VLAN ]] && net+=",tag=$VLAN"
  ip link show "$BRIDGE" >/dev/null 2>&1 || die "Bridge $BRIDGE doesn't exist on this host. Pick one with --bridge."

  cat <<EOF

New container
  ID / hostname   $VMID / $CT_NAME
  Template        $template
  Resources       $cores cores, $MEMORY MB memory, ${DISK} GB on $storage
  Network         $net
  Output folder   ${OUTPUT_DIR:-inside the container ($APP_DIR/output)}
  App             http://<container address>:${PORT:-8080}, open to the LAN
EOF
  (( MEMORY + 1024 <= host_mem )) || echo "Warning: the host has only $host_mem MB of memory."
  (( MEMORY >= 4096 )) || echo "Warning: under 4096 MB leaves room for only one voice worker."
  confirm "Create it?"

  ensure_template "$template"
  if [[ -n $OUTPUT_DIR ]]; then
    local uid
    uid=$(host_uid_for_ct_1000)
    if [[ ! -d $OUTPUT_DIR ]]; then
      run mkdir -p "$OUTPUT_DIR"
      run chown "$uid:$uid" "$OUTPUT_DIR"
    elif [[ $(stat -c %u "$OUTPUT_DIR") != "$uid" ]]; then
      echo "Note: $OUTPUT_DIR isn't owned by uid $uid (the container's uid 1000), so the app may not be able to write to it."
      echo "      If it's safe to change: chown -R $uid:$uid $(printf %q "$OUTPUT_DIR")"
    fi
  fi

  # shellcheck disable=SC2054  # the commas are part of the pct values
  local args=(create "$VMID" "$template" --hostname "$CT_NAME" --cores "$cores" --memory "$MEMORY"
              --swap 1024 --rootfs "$storage:$DISK" --net0 "$net" --unprivileged 1
              --features nesting=1,keyctl=1 --onboot 1
              --description "Audiobook Forge (Docker). Managed by deploy/proxmox-deploy.sh.")
  [[ -n $OUTPUT_DIR ]] && args+=(--mp0 "$OUTPUT_DIR,mp=$APP_DIR/output")
  [[ -n $SSH_KEY ]] && args+=(--ssh-public-keys "$SSH_KEY")
  say "Creating container $VMID"
  run pct "${args[@]}"
  run pct start "$VMID"
}

host_deploy() {
  [[ $EUID -eq 0 ]] || die "Run this as root on the Proxmox host."
  command -v pveversion >/dev/null && say "$(pveversion | cut -d/ -f1-2)"
  TMP_DIR=$(mktemp -d)
  trap 'rm -rf "$TMP_DIR"' EXIT
  if [[ -z $BUNDLE ]]; then
    BUNDLE=$TMP_DIR/forge.tgz
    make_bundle "$BUNDLE"
  fi
  [[ -f $BUNDLE ]] || die "Code bundle not found: $BUNDLE"

  local update=false
  if [[ -n $VMID ]] && pct status "$VMID" >/dev/null 2>&1; then
    local name
    name=$(ct_name_of "$VMID")
    [[ $name == "$CT_NAME" ]] ||
      die "Container $VMID already exists and is named $name. Use another --vmid, or --hostname $name to update it."
    update=true
  elif [[ -z $VMID ]]; then
    local found
    found=$(find_ct_by_name)
    if [[ $(wc -w <<<"$found") -gt 1 ]]; then
      die "Several containers are named $CT_NAME ($(echo $found)). Pick one with --vmid."
    elif [[ -n $found ]]; then
      VMID=$found
      update=true
    else
      VMID=$(pvesh get /cluster/nextid)
    fi
  fi

  if $update; then
    say "Updating container $VMID ($CT_NAME); books, settings and models are kept"
    [[ -z $OUTPUT_DIR && -z $IP && -z $SSH_KEY ]] ||
      echo "Note: --output-dir, --ip and --ssh-key only apply when the container is created. Change them with pct set."
    if [[ $(pct status "$VMID") != *running* ]]; then
      run pct start "$VMID"
    fi
  else
    create_ct
  fi

  say "Waiting for the container's network"
  wait_for_network

  say "Copying the app into the container"
  local work=/root/forge-deploy
  run pct exec "$VMID" -- rm -rf "$work"
  run pct exec "$VMID" -- mkdir -p "$work"
  run pct push "$VMID" "$BUNDLE" "$work/forge.tgz"
  run pct exec "$VMID" -- tar -xzf "$work/forge.tgz" -C "$work" deploy/lxc-setup.sh

  local setup=(bash "$work/deploy/lxc-setup.sh" --bundle "$work/forge.tgz" --app-dir "$APP_DIR"
               --docker-bip "$DOCKER_BIP" --docker-pool "$DOCKER_POOL")
  [[ -n $PORT ]] && setup+=(--port "$PORT")
  $PREFETCH || setup+=(--no-prefetch)
  $update || setup+=(--tz "$(timedatectl show -p Timezone --value 2>/dev/null || cat /etc/timezone 2>/dev/null || echo UTC)")
  run pct exec "$VMID" -- "${setup[@]}"
  run pct exec "$VMID" -- rm -rf "$work"

  if $DRY; then
    say "Dry run finished. Nothing was changed."
    return
  fi
  local addr port
  addr=$(pct exec "$VMID" -- sh -c "ip -4 -o addr show dev eth0 | awk '{print \$4}' | cut -d/ -f1 | head -n1")
  port=$(pct exec "$VMID" -- sh -c "grep -E '^PORT=' $APP_DIR/.env | cut -d= -f2")
  say "Audiobook Forge is running in container $VMID ($CT_NAME)"
  cat <<EOF
  Open:      http://${addr:-<container address>}:${port:-8080}
  Shell:     pct enter $VMID          (app in $APP_DIR, settings in $APP_DIR/.env)
  Logs:      pct exec $VMID -- docker compose --project-directory $APP_DIR logs -f
  Update:    run this script again
EOF
}

if [[ -n $BUNDLE_OUT ]]; then
  make_bundle "$BUNDLE_OUT"
  echo "Wrote $BUNDLE_OUT"
  exit 0
fi
if [[ -n $TARGET ]]; then
  remote_deploy
fi
command -v pct >/dev/null ||
  die "This machine isn't a Proxmox VE host. Give the host to deploy to: $0 root@your-proxmox-host"
host_deploy
