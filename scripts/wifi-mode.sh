#!/usr/bin/env bash
# Switch Orange Pi WiFi between client and access-point modes.
# Usage: wifi-mode.sh <ap|client|status>
#
# Environment overrides:
#   COOKIE_FINDER_AP_SSID        (default: cookie-finder)
#   COOKIE_FINDER_AP_PASSPHRASE  (default: cookie-finder)
#   COOKIE_FINDER_AP_GATEWAY     (default: 192.168.12.1)

set -euo pipefail

MODE="${1:-}"
SSID="${COOKIE_FINDER_AP_SSID:-cookie-finder}"
PASSPHRASE="${COOKIE_FINDER_AP_PASSPHRASE:-cookie-finder}"
GATEWAY="${COOKIE_FINDER_AP_GATEWAY:-192.168.12.1}"
RUNTIME_DIR="${COOKIE_FINDER_WIFI_RUNTIME:-/run/cookie-finder-wifi}"
HOSTAPD_CONF="${RUNTIME_DIR}/hostapd.conf"
DNSMASQ_CONF="${RUNTIME_DIR}/dnsmasq.conf"
PID_FILE="${RUNTIME_DIR}/create_ap.pid"
LOCK_FILE="${RUNTIME_DIR}/wifi-mode.lock"

log() { echo "[wifi-mode] $*"; }
die() { echo "[wifi-mode] ERROR: $*" >&2; exit 1; }

# Serialize ap/client switches so the GPIO button service and web UI cannot race.
acquire_mode_lock() {
  mkdir -p "${RUNTIME_DIR}"
  exec 9>"${LOCK_FILE}"
  if ! flock -n 9; then
    die "another wifi-mode switch is in progress"
  fi
  # Marker so the GPIO LED daemon (separate process) can show fast-blink.
  echo "$$" > "${RUNTIME_DIR}/switching"
  trap 'rm -f "${RUNTIME_DIR}/switching"' EXIT
}

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    die "must run as root (configure passwordless sudo for this script)"
  fi
}

find_iface() {
  if command -v iw >/dev/null 2>&1; then
    local iface
    iface="$(iw dev 2>/dev/null | awk '/Interface/ {print $2; exit}')"
    if [[ -n "${iface}" ]]; then
      echo "${iface}"
      return 0
    fi
  fi
  for path in /sys/class/net/*/wireless /sys/class/net/*/phy80211; do
    if [[ -e "${path}" ]]; then
      basename "$(dirname "${path}")"
      return 0
    fi
  done
  return 1
}

iface_type() {
  local iface="$1"
  iw dev "${iface}" info 2>/dev/null | awk '/type/ {print $2; exit}'
}

stop_create_ap() {
  local iface="$1"
  if command -v create_ap >/dev/null 2>&1; then
    create_ap --stop "${iface}" >/dev/null 2>&1 || true
  fi
  if [[ -f "${PID_FILE}" ]]; then
    local pid
    pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      sleep 1
      kill -9 "${pid}" 2>/dev/null || true
    fi
    rm -f "${PID_FILE}"
  fi
  pkill -f "create_ap .*${iface}" >/dev/null 2>&1 || true
}

stop_hostapd_dnsmasq() {
  if [[ -f "${RUNTIME_DIR}/hostapd.pid" ]]; then
    local pid
    pid="$(cat "${RUNTIME_DIR}/hostapd.pid" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  fi
  if [[ -f "${RUNTIME_DIR}/dnsmasq.pid" ]]; then
    local pid
    pid="$(cat "${RUNTIME_DIR}/dnsmasq.pid" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  fi
  pkill -f "${HOSTAPD_CONF}" >/dev/null 2>&1 || true
  pkill -f "${DNSMASQ_CONF}" >/dev/null 2>&1 || true
}

stop_nm_hotspot() {
  if ! command -v nmcli >/dev/null 2>&1; then
    return 0
  fi
  local ssid_lc
  ssid_lc="$(printf '%s' "${SSID}" | tr '[:upper:]' '[:lower:]')"
  # Bring down any active hotspot-style connections
  while IFS=: read -r name _type device; do
    [[ -z "${name}" ]] && continue
    local lname
    lname="$(printf '%s' "${name}" | tr '[:upper:]' '[:lower:]')"
    if [[ "${lname}" == *hotspot* || "${lname}" == *"${ssid_lc}"* ]]; then
      nmcli connection down "${name}" >/dev/null 2>&1 || true
      nmcli connection delete "${name}" >/dev/null 2>&1 || true
    fi
  done < <(nmcli -t -f NAME,TYPE,DEVICE connection show --active 2>/dev/null || true)
}

restore_client_networking() {
  local iface="$1"

  # Prefer NetworkManager when present. Avoid unconditional NM restarts —
  # they drop all interfaces and were a common cause of "stuck offline"
  # after leaving AP mode.
  if command -v nmcli >/dev/null 2>&1; then
    nmcli device set "${iface}" managed yes >/dev/null 2>&1 || true
    nmcli radio wifi on >/dev/null 2>&1 || true

    local state
    state="$(nmcli -t -f GENERAL.STATE device show "${iface}" 2>/dev/null || true)"
    if [[ "${state}" == *unmanaged* ]] \
        && systemctl list-unit-files NetworkManager.service >/dev/null 2>&1; then
      log "wlan still unmanaged; restarting NetworkManager once"
      systemctl restart NetworkManager >/dev/null 2>&1 || true
      sleep 2
      nmcli device set "${iface}" managed yes >/dev/null 2>&1 || true
    fi

    # Try every saved WiFi profile until one associates (not just the first).
    local conn brought_up=0
    while IFS= read -r conn; do
      [[ -z "${conn}" ]] && continue
      log "trying nmcli connection up: ${conn}"
      if nmcli -w 25 connection up "${conn}" ifname "${iface}" >/dev/null 2>&1; then
        log "connected via ${conn}"
        brought_up=1
        break
      fi
    done < <(nmcli -t -f NAME,TYPE connection show 2>/dev/null \
      | awk -F: '$2 ~ /wireless|wifi/ {print $1}')

    if [[ "${brought_up}" -eq 0 ]]; then
      nmcli device connect "${iface}" >/dev/null 2>&1 || true
    fi
    return 0
  fi

  # Fallback: wpa_supplicant + dhcp
  if systemctl list-unit-files "wpa_supplicant@${iface}.service" >/dev/null 2>&1; then
    systemctl restart "wpa_supplicant@${iface}.service" >/dev/null 2>&1 || true
  elif systemctl list-unit-files wpa_supplicant.service >/dev/null 2>&1; then
    systemctl restart wpa_supplicant >/dev/null 2>&1 || true
  fi

  if command -v wpa_cli >/dev/null 2>&1; then
    wpa_cli -i "${iface}" reconfigure >/dev/null 2>&1 || true
    wpa_cli -i "${iface}" reconnect >/dev/null 2>&1 || true
  fi

  if command -v dhclient >/dev/null 2>&1; then
    dhclient -r "${iface}" >/dev/null 2>&1 || true
    dhclient "${iface}" >/dev/null 2>&1 || true
  elif command -v dhcpcd >/dev/null 2>&1; then
    dhcpcd -n "${iface}" >/dev/null 2>&1 || true
  fi
}

start_ap_create_ap() {
  local iface="$1"
  command -v create_ap >/dev/null 2>&1 || return 1
  mkdir -p "${RUNTIME_DIR}"

  # -n: no internet sharing (local AP only); --daemon keeps it in background
  if create_ap --daemon -n --no-virt -g "${GATEWAY}" \
      "${iface}" "${SSID}" "${PASSPHRASE}"; then
    # Best-effort PID capture for cleanup
    pgrep -n -f "create_ap .*${iface}" > "${PID_FILE}" 2>/dev/null || true
    log "started AP via create_ap (ssid=${SSID} gateway=${GATEWAY})"
    return 0
  fi
  return 1
}

start_ap_nmcli() {
  local iface="$1"
  command -v nmcli >/dev/null 2>&1 || return 1

  nmcli device wifi hotspot ifname "${iface}" ssid "${SSID}" password "${PASSPHRASE}" \
    || return 1

  # Try to pin gateway/address when possible (NM versions vary)
  local hs
  hs="$(nmcli -t -f NAME,TYPE connection show --active 2>/dev/null | awk -F: 'tolower($1) ~ /hotspot/ {print $1; exit}')"
  if [[ -n "${hs}" ]]; then
    nmcli connection modify "${hs}" ipv4.addresses "${GATEWAY}/24" >/dev/null 2>&1 || true
    nmcli connection modify "${hs}" ipv4.method shared >/dev/null 2>&1 || true
    nmcli connection up "${hs}" >/dev/null 2>&1 || true
  fi
  log "started AP via nmcli hotspot (ssid=${SSID})"
  return 0
}

start_ap_hostapd() {
  local iface="$1"
  command -v hostapd >/dev/null 2>&1 || return 1
  command -v dnsmasq >/dev/null 2>&1 || return 1

  mkdir -p "${RUNTIME_DIR}"

  # Unmanage interface if NetworkManager is present
  if command -v nmcli >/dev/null 2>&1; then
    nmcli device set "${iface}" managed no >/dev/null 2>&1 || true
  fi

  ip link set "${iface}" down || true
  ip addr flush dev "${iface}" || true
  ip addr add "${GATEWAY}/24" dev "${iface}"
  ip link set "${iface}" up

  cat > "${HOSTAPD_CONF}" <<EOF
interface=${iface}
driver=nl80211
ssid=${SSID}
hw_mode=g
channel=6
ieee80211n=1
wmm_enabled=1
auth_algs=1
ignore_broadcast_ssid=0
wpa=2
wpa_passphrase=${PASSPHRASE}
wpa_key_mgmt=WPA-PSK
rsn_pairwise=CCMP
EOF

  cat > "${DNSMASQ_CONF}" <<EOF
interface=${iface}
bind-interfaces
dhcp-range=192.168.12.50,192.168.12.200,255.255.255.0,12h
dhcp-option=3,${GATEWAY}
dhcp-option=6,${GATEWAY}
EOF

  hostapd -B -P "${RUNTIME_DIR}/hostapd.pid" "${HOSTAPD_CONF}"
  dnsmasq --conf-file="${DNSMASQ_CONF}" --pid-file="${RUNTIME_DIR}/dnsmasq.pid"
  log "started AP via hostapd+dnsmasq (ssid=${SSID} gateway=${GATEWAY})"
  return 0
}

cmd_status() {
  local iface
  iface="$(find_iface || true)"
  if [[ -z "${iface}" ]]; then
    echo "interface="
    echo "mode=unknown"
    exit 0
  fi
  local type
  type="$(iface_type "${iface}" || echo unknown)"
  echo "interface=${iface}"
  echo "type=${type}"
  if [[ "${type}" == "AP" ]]; then
    echo "mode=ap"
  else
    echo "mode=client"
  fi
}

cmd_ap() {
  require_root
  acquire_mode_lock
  local iface
  iface="$(find_iface)" || die "no wireless interface found"

  log "switching ${iface} to AP mode (ssid=${SSID})"
  stop_create_ap "${iface}"
  stop_hostapd_dnsmasq
  stop_nm_hotspot

  # Drop any existing client association
  if command -v nmcli >/dev/null 2>&1; then
    nmcli device disconnect "${iface}" >/dev/null 2>&1 || true
  fi
  ip link set "${iface}" down >/dev/null 2>&1 || true
  sleep 1
  ip link set "${iface}" up >/dev/null 2>&1 || true

  if start_ap_create_ap "${iface}"; then
    exit 0
  fi
  if start_ap_nmcli "${iface}"; then
    exit 0
  fi
  if start_ap_hostapd "${iface}"; then
    exit 0
  fi
  die "no AP backend available (install create_ap, NetworkManager, or hostapd+dnsmasq)"
}

cmd_client() {
  require_root
  acquire_mode_lock
  local iface
  iface="$(find_iface)" || die "no wireless interface found"

  log "switching ${iface} to client mode"
  stop_create_ap "${iface}"
  stop_hostapd_dnsmasq
  stop_nm_hotspot

  ip addr flush dev "${iface}" >/dev/null 2>&1 || true
  ip link set "${iface}" up >/dev/null 2>&1 || true
  restore_client_networking "${iface}"
  log "client mode restore requested"
}

case "${MODE}" in
  ap) cmd_ap ;;
  client) cmd_client ;;
  status) cmd_status ;;
  *)
    echo "Usage: $0 <ap|client|status>" >&2
    exit 2
    ;;
esac
