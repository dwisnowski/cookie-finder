#!/usr/bin/env bash
# Graceful Orange Pi power-off for Cookie Finder.
# Usage: system-power.sh poweroff
#
# Writes a runtime marker so cookie-finder-wifi can chirp the WiFi LED,
# de-energizes the gimbal, then halts. Do not stop cookie-finder-wifi here —
# that service owns the LED until systemd tears the machine down.

set -euo pipefail

export PATH="/usr/sbin:/sbin:/usr/bin:/bin:${PATH:-}"

ACTION="${1:-}"
RUNTIME_DIR="${COOKIE_FINDER_WIFI_RUNTIME:-/run/cookie-finder-wifi}"
MARKER="${RUNTIME_DIR}/powering-off"
LOCK_FILE="${RUNTIME_DIR}/poweroff.lock"
LED_GRACE_S="${COOKIE_FINDER_POWEROFF_GRACE_S:-5}"

log() { echo "[system-power] $*"; }
die() { echo "[system-power] ERROR: $*" >&2; exit 1; }

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    die "must run as root (configure passwordless sudo for this script)"
  fi
}

if [[ "${ACTION}" != "poweroff" ]]; then
  die "usage: system-power.sh poweroff"
fi

require_root

mkdir -p "${RUNTIME_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  die "shutdown already in progress"
fi

echo "$$" > "${MARKER}"
log "shutdown requested (LED marker ${MARKER})"

# Drop stepper hold current before the 5 V rail dies. Leave cookie-finder-wifi
# running so the shutdown LED chirp stays visible.
if command -v systemctl >/dev/null 2>&1; then
  systemctl stop cookie-finder.service >/dev/null 2>&1 || true
fi

log "waiting ${LED_GRACE_S}s for shutdown LED"
sleep "${LED_GRACE_S}"
sync || true
log "powering off"
systemctl poweroff
