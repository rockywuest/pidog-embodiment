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

# The unit reads /etc/default/nox-brain (optional). Create a commented
# template on first install so users don't have to guess the keys (issue #5).
if [[ ! -f /etc/default/nox-brain ]]; then
  cat > /etc/default/nox-brain <<'EOF'
# Nox Brain configuration — read by nox-brain.service on start.
# Uncomment and adjust what you need, then: sudo systemctl restart nox-brain

# Where the robot body's bridge runs (127.0.0.1 if brain and body share one machine):
#PIDOG_HOST=pidog.local

# LLM backend — any OpenAI-compatible chat-completions endpoint.
# Cloud (OpenAI): set the key, keep the default URL/model:
#OPENAI_API_KEY=sk-...
# Local (Ollama): no key needed, point at Ollama and pick your model:
#OPENAI_URL=http://127.0.0.1:11434/v1/chat/completions
#LLM_MODEL=llama3.2
EOF
  chmod 600 /etc/default/nox-brain
  echo "Created /etc/default/nox-brain (template — edit it, then start the service)"
else
  echo "Kept existing /etc/default/nox-brain"
fi

echo
echo "Edit /etc/default/nox-brain (robot address + LLM backend),"
echo "then: sudo systemctl start nox-brain"
