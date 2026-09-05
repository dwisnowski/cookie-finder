#!/usr/bin/env bash
# Apply a Cookie Finder software update: git pull --ff-only origin main,
# uv sync, refresh the web systemd unit, and restart cookie-finder-web.
# Invoked by the oneshot systemd unit installed via `make init-software-update`.
# Do not call with untrusted arguments; repo root is fixed by the unit file.

set -euo pipefail
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"

STATUS_NAME="software-update.state"

usage() {
  echo "usage: $0 apply <repo_root>" >&2
  exit 2
}

write_status() {
  local phase="$1"
  local message="${2:-}"
  local path="$REPO_ROOT/data/$STATUS_NAME"
  mkdir -p "$REPO_ROOT/data"
  # Atomic-ish write so the web process never reads a partial JSON object.
  local tmp
  tmp="$(mktemp "$REPO_ROOT/data/.software-update.XXXXXX")"
  printf '{"phase":"%s","message":%s,"updated_at":"%s"}\n' \
    "$phase" \
    "$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$message")" \
    "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
    >"$tmp"
  mv -f "$tmp" "$path"
}

fail() {
  local message="$1"
  write_status "error" "$message"
  echo "error: $message" >&2
  exit 1
}

[[ "${1:-}" == "apply" ]] || usage
REPO_ROOT="${2:-}"
[[ -n "$REPO_ROOT" && -d "$REPO_ROOT" ]] || usage
[[ -d "$REPO_ROOT/.git" ]] || fail "Not a git checkout: $REPO_ROOT"

cd "$REPO_ROOT"

if [[ "$(id -u)" -eq 0 ]]; then
  fail "Refusing to run as root (would break .venv ownership). Run via the systemd unit User=."
fi

write_status "pulling" "Fetching and fast-forwarding origin/main…"
git fetch origin main || fail "git fetch origin main failed"
git merge --ff-only origin/main || fail "git merge --ff-only origin/main failed (dirty tree or diverged history)"

write_status "building" "Syncing dependencies (uv sync)…"
if command -v uv >/dev/null 2>&1; then
  uv sync || fail "uv sync failed"
else
  fail "uv not found on PATH (install uv, then retry)"
fi

write_status "restarting" "Refreshing unit and restarting cookie-finder-web…"
UNIT_IN="$REPO_ROOT/systemd/cookie-finder-web.service.in"
UNIT_OUT="/etc/systemd/system/cookie-finder-web.service"
WEB_PYTHON="${REPO_ROOT}/.venv/bin/python"
WEB_HOST="${WEB_HOST:-0.0.0.0}"
WEB_PORT="${WEB_PORT:-80}"
WEB_HTTPS_PORT="${WEB_HTTPS_PORT:-443}"
TLS_DIR="${COOKIE_FINDER_TLS_DIR:-/var/lib/cookie-finder/tls}"

if [[ ! -x "$WEB_PYTHON" ]]; then
  fail "missing $WEB_PYTHON — run make install on the Pi first"
fi
if [[ ! -f "$UNIT_IN" ]]; then
  fail "missing $UNIT_IN"
fi

sed \
  -e "s|@REPO_ROOT@|${REPO_ROOT}|g" \
  -e "s|@PYTHON@|${WEB_PYTHON}|g" \
  -e "s|@WEB_HOST@|${WEB_HOST}|g" \
  -e "s|@WEB_PORT@|${WEB_PORT}|g" \
  -e "s|@WEB_HTTPS_PORT@|${WEB_HTTPS_PORT}|g" \
  -e "s|@TLS_DIR@|${TLS_DIR}|g" \
  "$UNIT_IN" | sudo -n tee "$UNIT_OUT" >/dev/null \
  || fail "could not write $UNIT_OUT (passwordless sudo required — re-run make init-software-update)"

sudo -n systemctl daemon-reload \
  || fail "systemctl daemon-reload failed"
sudo -n systemctl restart cookie-finder-web.service \
  || fail "systemctl restart cookie-finder-web failed"

write_status "done" "Update complete."
echo "Cookie Finder software update complete."
