#!/usr/bin/env bash
# Install the body systemd units for the *current* user and repo location.
#
# The unit files in body/services/ ship with a reference setup (user "pidog",
# code in /home/pidog). Installing them verbatim on any other machine makes
# systemd fail with the cryptic "failed because of unavailable resources or
# another system error" (issue #5). This script rewrites User=, paths and
# EnvironmentFile= to match the machine it runs on.
#
# Usage:
#   sudo ./scripts/install-body.sh            # nox-body nox-bridge nox-voice
#   sudo ./scripts/install-body.sh nox-vision # additionally install extra units
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Run with sudo: sudo $0 $*" >&2
  exit 1
fi

BODY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../body" && pwd)"
RUN_USER="${SUDO_USER:-$USER}"
UNITS=(nox-body nox-bridge nox-voice "$@")

if ! id "$RUN_USER" >/dev/null 2>&1 || [[ "$RUN_USER" == "root" ]]; then
  echo "Could not determine the non-root user to run the services as." >&2
  echo "Run via sudo from your normal user account." >&2
  exit 1
fi

RUN_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"

ENV_CREATED=0
if [[ ! -f "$BODY_DIR/nox.env" ]]; then
  cp "$BODY_DIR/nox.env.example" "$BODY_DIR/nox.env"
  chown "$RUN_USER": "$BODY_DIR/nox.env"
  ENV_CREATED=1
fi

# Old templates shipped literal <PLACEHOLDER> values; services started with
# those fail with cryptic DNS errors. Refuse to (re)start until they're gone.
ENV_HAS_PLACEHOLDERS=0
if grep -q '[<>]' "$BODY_DIR/nox.env"; then
  ENV_HAS_PLACEHOLDERS=1
fi

for unit in "${UNITS[@]}"; do
  src="$BODY_DIR/services/${unit}.service"
  if [[ ! -f "$src" ]]; then
    echo "Unknown unit: $unit (no $src)" >&2
    exit 1
  fi
  # ~/.local/bin belongs to the user's home, everything else to the repo dir.
  sed -e "s|^User=.*|User=${RUN_USER}|" \
      -e "s|/home/pidog/.local/bin|${RUN_HOME}/.local/bin|g" \
      -e "s|/home/pidog|${BODY_DIR}|g" \
      "$src" > "/etc/systemd/system/${unit}.service"
  echo "Installed /etc/systemd/system/${unit}.service (User=${RUN_USER}, dir=${BODY_DIR})"
done

systemctl daemon-reload
systemctl enable "${UNITS[@]}"

if [[ $ENV_HAS_PLACEHOLDERS -eq 1 ]]; then
  echo
  echo "WARNING: $BODY_DIR/nox.env still contains <PLACEHOLDER> values." >&2
  echo "Edit it (at least BRAIN_HOST — the IP of your brain machine, or 127.0.0.1" >&2
  echo "if brain and body share one machine), then:" >&2
  echo "  sudo systemctl restart ${UNITS[*]}" >&2
elif [[ $ENV_CREATED -eq 1 ]]; then
  echo
  echo "Created $BODY_DIR/nox.env from the example."
  echo "Edit it (at least BRAIN_HOST — use 127.0.0.1 if brain and body share one machine), then:"
  echo "  sudo systemctl start ${UNITS[*]}"
else
  systemctl restart "${UNITS[@]}"
  echo
  echo "Services restarted. Check with: systemctl status ${UNITS[*]}"
fi
