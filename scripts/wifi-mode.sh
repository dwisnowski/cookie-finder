#!/usr/bin/env bash
# Switch Orange Pi WiFi between client and access-point modes.
# Usage: wifi-mode.sh <ap|client|fix|status>
#
# Environment overrides:
#   COOKIE_FINDER_AP_SSID        (default: cookie-finder)
#   COOKIE_FINDER_AP_PASSPHRASE  (default: empty = open SoftAP, no password)
#   COOKIE_FINDER_AP_GATEWAY     (default: 192.168.12.1)
#
# Captive portal: in AP mode DNS is hijacked to the gateway so phones open the
# web app (served on :80 / :443). See cookie_finder/web/server.py probe routes.

set -euo pipefail

# Non-login shells often omit /usr/sbin (iw, ip live there on Armbian).
export PATH="/usr/sbin:/sbin:/usr/bin:/bin:${PATH:-}"

MODE="${1:-}"
SSID="${COOKIE_FINDER_AP_SSID:-cookie-finder}"
# Empty passphrase = open network (what works reliably on Zero 2W SoftAP).
PASSPHRASE="${COOKIE_FINDER_AP_PASSPHRASE:-}"
GATEWAY="${COOKIE_FINDER_AP_GATEWAY:-192.168.12.1}"
RUNTIME_DIR="${COOKIE_FINDER_WIFI_RUNTIME:-/run/cookie-finder-wifi}"
HOSTAPD_CONF="${RUNTIME_DIR}/hostapd.conf"
DNSMASQ_CONF="${RUNTIME_DIR}/dnsmasq.conf"
PID_FILE="${RUNTIME_DIR}/create_ap.pid"
LOCK_FILE="${RUNTIME_DIR}/wifi-mode.lock"
# NetworkManager shared-mode dnsmasq drop-in (captive DNS hijack).
NM_DNSMASQ_SHARED_DIR="/etc/NetworkManager/dnsmasq-shared.d"
NM_CAPTIVE_CONF="${NM_DNSMASQ_SHARED_DIR}/cookie-finder-captive.conf"

log() { echo "[wifi-mode] $*"; }
die() { echo "[wifi-mode] ERROR: $*" >&2; exit 1; }

# WPA2-PSK requires 8–63 ASCII characters when a passphrase is configured.
assert_passphrase() {
  local n=${#PASSPHRASE}
  if (( n == 0 )); then
    return 0
  fi
  if (( n < 8 || n > 63 )); then
    die "WPA2 passphrase must be 8–63 characters (got ${n}), or empty for an open SoftAP."
  fi
}

IW_BIN="$(command -v iw || true)"
IP_BIN="$(command -v ip || true)"

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
  if [[ -n "${IW_BIN}" ]]; then
    local iface
    iface="$("${IW_BIN}" dev 2>/dev/null | awk '/Interface/ {print $2; exit}')"
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
  [[ -n "${IW_BIN}" ]] || return 1
  "${IW_BIN}" dev "${iface}" info 2>/dev/null | awk '/type/ {print $2; exit}'
}

iface_has_gateway() {
  local iface="$1"
  [[ -n "${IP_BIN}" ]] || return 1
  "${IP_BIN}" -4 -o addr show dev "${iface}" 2>/dev/null | grep -q "${GATEWAY}"
}

ap_backend_running() {
  local iface="$1"
  pgrep -af "create_ap .*${iface}" >/dev/null 2>&1 && return 0
  pgrep -af "hostapd.*${HOSTAPD_CONF}" >/dev/null 2>&1 && return 0
  pgrep -af "hostapd.*${iface}" >/dev/null 2>&1 && return 0
  return 1
}

NM_AP_CONN="cookie-finder-ap"

# Release station/wpa hold and (optionally) hand the iface to hostapd/create_ap.
prepare_iface_for_ap() {
  local iface="$1"
  local set_ap_type="${2:-0}"

  command -v rfkill >/dev/null 2>&1 && rfkill unblock wifi >/dev/null 2>&1 || true
  command -v rfkill >/dev/null 2>&1 && rfkill unblock all >/dev/null 2>&1 || true

  # wpa_supplicant holding the nl80211 socket is the usual cause of:
  #   "Failed to set beacon parameters" / "Could not connect to kernel driver"
  pkill -x wpa_supplicant >/dev/null 2>&1 || true
  sleep 0.5

  if command -v nmcli >/dev/null 2>&1; then
    nmcli device disconnect "${iface}" >/dev/null 2>&1 || true
    nmcli device set "${iface}" managed no >/dev/null 2>&1 || true
  fi

  if [[ -n "${IP_BIN}" ]]; then
    "${IP_BIN}" link set "${iface}" down >/dev/null 2>&1 || true
    "${IP_BIN}" addr flush dev "${iface}" >/dev/null 2>&1 || true
  fi

  if [[ "${set_ap_type}" == "1" ]] && [[ -n "${IW_BIN}" ]]; then
    # Explicit SoftAP type helps UWE5622/sprdwl before hostapd attaches.
    "${IW_BIN}" dev "${iface}" set type __ap >/dev/null 2>&1 \
      || "${IW_BIN}" dev "${iface}" set type ap >/dev/null 2>&1 \
      || log "WARNING: could not set ${iface} type to AP via iw"
  elif [[ -n "${IW_BIN}" ]]; then
    "${IW_BIN}" dev "${iface}" set type managed >/dev/null 2>&1 || true
  fi

  if [[ -n "${IP_BIN}" ]]; then
    "${IP_BIN}" link set "${iface}" up >/dev/null 2>&1 || true
  fi
  sleep 1
}

# After a failed hostapd/create_ap attempt, put the iface back so NM can use it.
recover_iface_for_nm() {
  local iface="$1"
  stop_hostapd_dnsmasq
  pkill -x wpa_supplicant >/dev/null 2>&1 || true
  if [[ -n "${IP_BIN}" ]]; then
    "${IP_BIN}" link set "${iface}" down >/dev/null 2>&1 || true
  fi
  if [[ -n "${IW_BIN}" ]]; then
    "${IW_BIN}" dev "${iface}" set type managed >/dev/null 2>&1 || true
  fi
  if [[ -n "${IP_BIN}" ]]; then
    "${IP_BIN}" link set "${iface}" up >/dev/null 2>&1 || true
  fi
  if command -v nmcli >/dev/null 2>&1; then
    nmcli device set "${iface}" managed yes >/dev/null 2>&1 || true
    nmcli radio wifi on >/dev/null 2>&1 || true
  fi
  sleep 2
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
  # Our AP dnsmasq must own :53 on the AP subnet; stop the system one if present.
  if systemctl list-unit-files dnsmasq.service >/dev/null 2>&1; then
    systemctl stop dnsmasq.service >/dev/null 2>&1 || true
  fi
}

stop_nm_hotspot() {
  if ! command -v nmcli >/dev/null 2>&1; then
    return 0
  fi
  nmcli connection down "${NM_AP_CONN}" >/dev/null 2>&1 || true
  nmcli connection delete "${NM_AP_CONN}" >/dev/null 2>&1 || true
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

# Install NM shared-mode dnsmasq drop-in so SoftAP DHCP clients resolve every
# name to the Pi (captive portal → web app on :80).
install_nm_captive_dns() {
  mkdir -p "${NM_DNSMASQ_SHARED_DIR}"
  cat > "${NM_CAPTIVE_CONF}" <<EOF
# Cookie Finder captive portal — sinkhole all DNS to the SoftAP gateway.
# Only used by NetworkManager "shared" (AP) connections, not client WiFi.
address=/#/${GATEWAY}
EOF
  log "installed captive DNS drop-in: ${NM_CAPTIVE_CONF}"
}

# hostapd-path dnsmasq: DHCP + captive DNS (no upstream forwarding).
write_hostapd_dnsmasq_conf() {
  local iface="$1"
  cat > "${DNSMASQ_CONF}" <<EOF
interface=${iface}
bind-interfaces
listen-address=${GATEWAY}
dhcp-range=192.168.12.50,192.168.12.200,255.255.255.0,12h
dhcp-option=3,${GATEWAY}
dhcp-option=6,${GATEWAY}
no-resolv
no-hosts
# Captive portal: every name → gateway (web app on :80 / :443)
address=/#/${GATEWAY}
EOF
}

client_already_associated() {
  local iface="$1"
  local ssid=""
  if [[ -n "${IW_BIN}" ]]; then
    ssid="$("${IW_BIN}" dev "${iface}" link 2>/dev/null | awk '/SSID:/ {$1=""; sub(/^ /,""); print; exit}')"
  fi
  if [[ -z "${ssid}" ]] && command -v nmcli >/dev/null 2>&1; then
    ssid="$(nmcli -t -f ACTIVE,SSID dev wifi 2>/dev/null | awk -F: '$1=="yes"{print $2; exit}')"
  fi
  if [[ -n "${ssid}" ]]; then
    log "already associated with ${ssid}; leaving client networking alone"
    return 0
  fi
  return 1
}

restore_client_networking() {
  local iface="$1"

  # Prefer NetworkManager when present. Avoid unconditional NM restarts —
  # they drop all interfaces and were a common cause of "stuck offline"
  # after leaving AP mode.
  if command -v nmcli >/dev/null 2>&1; then
    nmcli device set "${iface}" managed yes >/dev/null 2>&1 || true
    nmcli radio wifi on >/dev/null 2>&1 || true

    if client_already_associated "${iface}"; then
      return 0
    fi

    local state
    state="$(nmcli -t -f GENERAL.STATE device show "${iface}" 2>/dev/null || true)"
    if [[ "${state}" == *unmanaged* ]] \
        && systemctl list-unit-files NetworkManager.service >/dev/null 2>&1; then
      log "wlan still unmanaged; restarting NetworkManager once"
      # --no-block: never deadlock systemd from inside another unit's start
      systemctl restart --no-block NetworkManager >/dev/null 2>&1 || true
      sleep 3
      nmcli device set "${iface}" managed yes >/dev/null 2>&1 || true
    fi

    # Try saved WiFi profiles (short waits; cap attempts so we cannot hang for minutes).
    local conn brought_up=0 attempts=0
    while IFS= read -r conn; do
      [[ -z "${conn}" ]] && continue
      attempts=$((attempts + 1))
      [[ "${attempts}" -gt 3 ]] && break
      log "trying nmcli connection up: ${conn}"
      if nmcli -w 10 connection up "${conn}" ifname "${iface}" >/dev/null 2>&1; then
        log "connected via ${conn}"
        brought_up=1
        break
      fi
    done < <(nmcli -t -f NAME,TYPE connection show 2>/dev/null \
      | awk -F: '$2 ~ /wireless|wifi/ {print $1}')

    if [[ "${brought_up}" -eq 0 ]]; then
      nmcli -w 10 device connect "${iface}" >/dev/null 2>&1 || true
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
  local logf="${RUNTIME_DIR}/create_ap.log"
  rm -f "${logf}"

  prepare_iface_for_ap "${iface}" 0

  # -n: no internet sharing (local AP only); --daemon keeps it in background
  # Passphrase omitted → open SoftAP (UWE5622 + Apple clients work reliably open).
  local -a create_args=(--daemon -n --no-virt -c 6 -g "${GATEWAY}" "${iface}" "${SSID}")
  if [[ -n "${PASSPHRASE}" ]]; then
    create_args+=("${PASSPHRASE}")
  fi
  if create_ap "${create_args[@]}" >"${logf}" 2>&1; then
    pgrep -n -f "create_ap .*${iface}" > "${PID_FILE}" 2>/dev/null || true
    sleep 3
    if ! ap_backend_running "${iface}" && ! iface_has_gateway "${iface}"; then
      log "create_ap exited immediately; log:"
      tail -n 30 "${logf}" 2>/dev/null || true
      return 1
    fi
    log "started AP via create_ap (ssid=${SSID} gateway=${GATEWAY})"
    return 0
  fi
  log "create_ap failed; log:"
  tail -n 30 "${logf}" 2>/dev/null || true
  return 1
}

start_ap_nmcli() {
  local iface="$1"
  command -v nmcli >/dev/null 2>&1 || return 1

  recover_iface_for_nm "${iface}"
  stop_nm_hotspot

  # Captive DNS before bringing the shared connection up so NM's dnsmasq
  # inherits the sinkhole drop-in.
  install_nm_captive_dns

  # Explicit AP profile is more reliable on UWE5622 than `wifi hotspot`.
  # Open SoftAP (no wifi-sec): WPA SoftAP often fails phone joins on this chip;
  # key-mgmt "none" wrongly becomes WEP on Bookworm NM — omit security entirely.
  nmcli connection delete "${NM_AP_CONN}" >/dev/null 2>&1 || true
  nmcli connection add type wifi ifname "${iface}" con-name "${NM_AP_CONN}" \
    autoconnect no ssid "${SSID}" \
    802-11-wireless.mode ap \
    802-11-wireless.band bg \
    802-11-wireless.channel 6 \
    ipv4.method shared \
    ipv4.addresses "${GATEWAY}/24" >/dev/null || return 1

  if [[ -n "${PASSPHRASE}" ]]; then
    nmcli connection modify "${NM_AP_CONN}" \
      wifi-sec.key-mgmt wpa-psk \
      wifi-sec.proto rsn \
      wifi-sec.pairwise ccmp \
      wifi-sec.group ccmp \
      wifi-sec.pmf disable \
      wifi-sec.psk "${PASSPHRASE}" >/dev/null || return 1
    log "NM SoftAP: WPA2-PSK (passphrase set via COOKIE_FINDER_AP_PASSPHRASE)"
  else
    log "NM SoftAP: open network (no password)"
  fi

  if ! nmcli -w 25 connection up "${NM_AP_CONN}" ifname "${iface}"; then
    log "nmcli connection up ${NM_AP_CONN} failed"
    return 1
  fi

  sleep 2
  local type
  type="$(iface_type "${iface}" || echo unknown)"
  if [[ "${type}" != "AP" ]] && ! iface_has_gateway "${iface}"; then
    if ! nmcli -t -f NAME,DEVICE connection show --active 2>/dev/null \
        | grep -qiE "(${NM_AP_CONN}|hotspot).*${iface}|${iface}.*(${NM_AP_CONN}|hotspot)"; then
      local active
      active="$(nmcli -t -f NAME,DEVICE connection show --active 2>/dev/null || true)"
      if ! printf '%s\n' "${active}" | grep -qi hotspot \
          && ! printf '%s\n' "${active}" | grep -q "${NM_AP_CONN}"; then
        log "nmcli AP did not become active (type=${type})"
        return 1
      fi
    fi
  fi
  log "started AP via NetworkManager (ssid=${SSID} gateway=${GATEWAY} type=${type} captive-dns=on)"
  return 0
}

start_ap_hostapd() {
  local iface="$1"
  local channel="${2:-6}"
  command -v hostapd >/dev/null 2>&1 || return 1

  mkdir -p "${RUNTIME_DIR}"
  rm -f "${RUNTIME_DIR}/hostapd.log"

  prepare_iface_for_ap "${iface}" 1

  if [[ -n "${IP_BIN}" ]]; then
    "${IP_BIN}" addr flush dev "${iface}" || true
    "${IP_BIN}" addr add "${GATEWAY}/24" dev "${iface}" || true
    "${IP_BIN}" link set "${iface}" up || true
  fi
  sleep 1

  # Keep hostapd.conf minimal — extra keys break older builds / flaky SoftAP drivers.
  # Default: open SoftAP (no WPA). Optional passphrase via COOKIE_FINDER_AP_PASSPHRASE.
  if [[ -n "${PASSPHRASE}" ]]; then
    cat > "${HOSTAPD_CONF}" <<EOF
interface=${iface}
driver=nl80211
ssid=${SSID}
hw_mode=g
channel=${channel}
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0
wpa=2
wpa_passphrase=${PASSPHRASE}
wpa_key_mgmt=WPA-PSK
rsn_pairwise=CCMP
EOF
  else
    cat > "${HOSTAPD_CONF}" <<EOF
interface=${iface}
driver=nl80211
ssid=${SSID}
hw_mode=g
channel=${channel}
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0
EOF
  fi

  if ! hostapd -B -P "${RUNTIME_DIR}/hostapd.pid" \
      -f "${RUNTIME_DIR}/hostapd.log" "${HOSTAPD_CONF}"; then
    log "hostapd failed to start (channel=${channel}; see ${RUNTIME_DIR}/hostapd.log)"
    if [[ -f "${RUNTIME_DIR}/hostapd.log" ]]; then
      tail -n 40 "${RUNTIME_DIR}/hostapd.log" 2>/dev/null || true
    fi
    return 1
  fi
  sleep 2
  if ! ap_backend_running "${iface}"; then
    log "hostapd exited immediately (channel=${channel})"
    if [[ -f "${RUNTIME_DIR}/hostapd.log" ]]; then
      tail -n 40 "${RUNTIME_DIR}/hostapd.log" 2>/dev/null || true
    fi
    stop_hostapd_dnsmasq
    return 1
  fi

  if command -v dnsmasq >/dev/null 2>&1; then
    write_hostapd_dnsmasq_conf "${iface}"
    if systemctl list-unit-files dnsmasq.service >/dev/null 2>&1; then
      systemctl stop dnsmasq.service >/dev/null 2>&1 || true
    fi
    pkill -x dnsmasq >/dev/null 2>&1 || true
    if ! dnsmasq --conf-file="${DNSMASQ_CONF}" --pid-file="${RUNTIME_DIR}/dnsmasq.pid"; then
      log "WARNING: dnsmasq failed — AP is up but DHCP/captive DNS may not work"
    fi
  else
    log "WARNING: dnsmasq not installed — AP is up but DHCP may not work"
  fi

  sleep 1
  local itype
  itype="$(iface_type "${iface}" || echo unknown)"
  if ! ap_backend_running "${iface}" && ! iface_has_gateway "${iface}"; then
    log "hostapd did not stay up (channel=${channel})"
    stop_hostapd_dnsmasq
    return 1
  fi
  log "started AP via hostapd (ssid=${SSID} gateway=${GATEWAY} type=${itype} ch=${channel} captive-dns=on)"
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
  if [[ "${type}" == "AP" ]] \
      || ap_backend_running "${iface}" \
      || iface_has_gateway "${iface}"; then
    echo "mode=ap"
  else
    echo "mode=client"
  fi
}

cmd_ap() {
  require_root
  assert_passphrase
  acquire_mode_lock
  local iface
  iface="$(find_iface)" || die "no wireless interface found"

  log "switching ${iface} to AP mode (ssid=${SSID})"
  stop_create_ap "${iface}"
  stop_hostapd_dnsmasq
  stop_nm_hotspot

  # Orange Pi Zero 2W (UWE5622): NetworkManager SoftAP is usually the only
  # reliable path. hostapd often fails with "Could not connect to kernel driver"
  # unless wpa_supplicant is killed and the iface type is set to AP first.
  log "trying NetworkManager SoftAP first…"
  if start_ap_nmcli "${iface}"; then
    exit 0
  fi

  log "NetworkManager path failed; trying hostapd (channel 6)…"
  if start_ap_hostapd "${iface}" 6; then
    exit 0
  fi
  log "hostapd ch6 failed; trying hostapd (channel 1)…"
  if start_ap_hostapd "${iface}" 1; then
    exit 0
  fi

  log "hostapd path failed; trying create_ap…"
  if start_ap_create_ap "${iface}"; then
    exit 0
  fi

  die "no AP backend available (see journal + /run/cookie-finder-wifi/*.log)"
}

cmd_client() {
  require_root
  acquire_mode_lock
  local iface
  iface="$(find_iface)" || die "no wireless interface found"

  local type
  type="$(iface_type "${iface}" || echo unknown)"
  # Already a healthy client — do not flush addresses / force nmcli reconnect
  # (that was making `make on-the-pi-wifi-gpio-daemon` hang for a long time).
  if [[ "${type}" == "managed" || "${type}" == "station" ]] \
      && client_already_associated "${iface}"; then
    stop_create_ap "${iface}"
    stop_hostapd_dnsmasq
    stop_nm_hotspot
    log "client mode already active; nothing to do"
    exit 0
  fi

  log "switching ${iface} to client mode"
  stop_create_ap "${iface}"
  stop_hostapd_dnsmasq
  stop_nm_hotspot

  ip addr flush dev "${iface}" >/dev/null 2>&1 || true
  ip link set "${iface}" up >/dev/null 2>&1 || true
  restore_client_networking "${iface}"
  log "client mode restore requested"
}

# Saved WiFi profile names, highest autoconnect-priority first.
wifi_profile_names() {
  nmcli -t -f NAME,TYPE,AUTOCONNECT-PRIORITY connection show 2>/dev/null \
    | awk -F: '$2 ~ /wireless|wifi/ {print $3 "\t" $1}' \
    | sort -t$'\t' -k1,1nr \
    | cut -f2
}

# When NetworkManager leaves UWE5622 as unavailable/unmanaged after SoftAP,
# associate with a saved profile via wpa_supplicant + DHCP instead.
connect_via_wpa_supplicant() {
  local iface="$1"
  command -v wpa_supplicant >/dev/null 2>&1 || die "wpa_supplicant not found"
  command -v wpa_passphrase >/dev/null 2>&1 || die "wpa_passphrase not found"
  command -v nmcli >/dev/null 2>&1 || die "nmcli needed to read saved WiFi PSKs"

  nmcli device set "${iface}" managed no >/dev/null 2>&1 || true
  pkill -x wpa_supplicant >/dev/null 2>&1 || true
  sleep 1

  mkdir -p "${RUNTIME_DIR}"
  local conn ssid psk conf="${RUNTIME_DIR}/wpa-fix.conf"

  while IFS= read -r conn; do
    [[ -z "${conn}" ]] && continue
    ssid="$(nmcli -g 802-11-wireless.ssid connection show "${conn}" 2>/dev/null || true)"
    psk="$(nmcli -s -g 802-11-wireless-security.psk connection show "${conn}" 2>/dev/null || true)"
    if [[ -z "${ssid}" || -z "${psk}" ]]; then
      log "skip ${conn}: missing ssid/psk in NetworkManager profile"
      continue
    fi
    log "wpa_supplicant trying SSID=${ssid} (profile ${conn})"
    wpa_passphrase "${ssid}" "${psk}" > "${conf}"
    if ! wpa_supplicant -B -i "${iface}" -c "${conf}" >/dev/null 2>&1; then
      log "wpa_supplicant failed to start for ${ssid}"
      continue
    fi
    sleep 6
    if [[ -n "${IW_BIN}" ]] \
        && "${IW_BIN}" dev "${iface}" link 2>/dev/null | grep -q "SSID:"; then
      if command -v dhclient >/dev/null 2>&1; then
        dhclient -r "${iface}" >/dev/null 2>&1 || true
        dhclient "${iface}" >/dev/null 2>&1 || true
      elif command -v dhcpcd >/dev/null 2>&1; then
        dhcpcd -n "${iface}" >/dev/null 2>&1 || true
      fi
      sleep 2
      log "connected via wpa_supplicant to ${ssid}"
      log "left ${iface} unmanaged so NetworkManager does not fight this link"
      return 0
    fi
    log "no association for ${ssid}; trying next profile"
    pkill -x wpa_supplicant >/dev/null 2>&1 || true
    sleep 1
  done < <(wifi_profile_names)

  return 1
}

# Recover a wedged client radio (common after SoftAP on Orange Pi Zero 2W).
# Tries NetworkManager first; falls back to wpa_supplicant using saved PSKs.
cmd_fix() {
  require_root
  acquire_mode_lock
  local iface
  iface="$(find_iface)" || die "no wireless interface found"

  log "fixing ${iface} (clear AP leftovers, recover client)"
  command -v rfkill >/dev/null 2>&1 && rfkill unblock wifi >/dev/null 2>&1 || true
  command -v rfkill >/dev/null 2>&1 && rfkill unblock all >/dev/null 2>&1 || true

  stop_create_ap "${iface}"
  stop_hostapd_dnsmasq
  stop_nm_hotspot
  pkill -x wpa_supplicant >/dev/null 2>&1 || true

  if [[ -n "${IP_BIN}" ]]; then
    "${IP_BIN}" addr flush dev "${iface}" >/dev/null 2>&1 || true
  fi
  recover_iface_for_nm "${iface}"

  if ! command -v nmcli >/dev/null 2>&1; then
    restore_client_networking "${iface}"
    log "wifi fix finished (no nmcli; used wpa/dhcp fallback path)"
    exit 0
  fi

  # Restart NM, then re-assert managed — order matters on this chip.
  if systemctl list-unit-files NetworkManager.service >/dev/null 2>&1; then
    log "restarting NetworkManager"
    systemctl restart NetworkManager >/dev/null 2>&1 || true
    sleep 4
  fi
  nmcli networking on >/dev/null 2>&1 || true
  nmcli radio wifi on >/dev/null 2>&1 || true
  nmcli device set "${iface}" managed yes >/dev/null 2>&1 || true
  sleep 2

  local state
  state="$(nmcli -t -f DEVICE,STATE device status 2>/dev/null \
    | awk -F: -v d="${iface}" '$1 == d { print $2; exit }')"
  log "nmcli ${iface} state=${state:-unknown}"

  local brought_up=0 conn
  while IFS= read -r conn; do
    [[ -z "${conn}" ]] && continue
    log "trying nmcli connection up: ${conn}"
    if nmcli -w 20 connection up "${conn}" ifname "${iface}" >/dev/null 2>&1; then
      log "connected via nmcli ${conn}"
      brought_up=1
      break
    fi
  done < <(wifi_profile_names)

  if [[ "${brought_up}" -eq 1 ]]; then
    log "wifi fix complete (NetworkManager)"
    exit 0
  fi

  if [[ "${state}" == "unavailable" || "${state}" == "unmanaged" \
      || "${state}" == "disconnected" || -z "${state}" ]]; then
    log "nmcli activate failed (state=${state:-unknown}); trying wpa_supplicant"
  else
    log "nmcli activate failed; trying wpa_supplicant"
  fi

  if connect_via_wpa_supplicant "${iface}"; then
    log "wifi fix complete (wpa_supplicant)"
    exit 0
  fi

  die "could not associate with any saved WiFi profile (in range? run: make wifi-configure-clients)"
}

case "${MODE}" in
  ap) cmd_ap ;;
  client) cmd_client ;;
  fix) cmd_fix ;;
  status) cmd_status ;;
  *)
    echo "Usage: $0 <ap|client|fix|status>" >&2
    exit 2
    ;;
esac
