#!/usr/bin/env bash
# Install the brain systemd unit for the *current* user and repo location.
# Same rationale as install-body.sh: the shipped unit hardcodes a reference
# user and path, which breaks on any other machine (issue #5).
#
# Usage: sudo ./scripts/install-brain.sh
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Run with sudo: sudo $0" >&2
  exit 1
fi

BRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../brain" && pwd)"
RUN_USER="${SUDO_USER:-$USER}"

if ! id "$RUN_USER" >/dev/null 2>&1 || [[ "$RUN_USER" == "root" ]]; then
  echo "Could not determine the non-root user to run the service as." >&2
  echo "Run via sudo from your normal user account." >&2
  exit 1
fi

sed -e "s|^User=.*|User=${RUN_USER}|" \
    -e "s|^WorkingDirectory=.*|WorkingDirectory=${BRAIN_DIR}|" \
    "$BRAIN_DIR/services/nox-brain.service" > /etc/systemd/system/nox-brain.service
echo "Installed /etc/systemd/system/nox-brain.service (User=${RUN_USER}, dir=${BRAIN_DIR})"

systemctl daemon-reload
systemctl enable nox-brain

echo
echo "Set your API key and robot address in /etc/default/nox-brain, e.g.:"
echo "  OPENAI_API_KEY=your-key        # only needed for OpenAI-compatible APIs, not Ollama"
echo "  PIDOG_HOST=your-robot.local"
echo "then: sudo systemctl start nox-brain"
