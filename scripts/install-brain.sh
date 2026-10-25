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
# Same value as NOX_API_TOKEN in the robot's body/nox.env, if you set one there:
#NOX_API_TOKEN=

# Language the dog answers in: auto (the language you speak to it, default),
# or always de / en / fr — pick the one the robot's Piper voice speaks.
# Check it arrived after a restart: journalctl -u nox-brain | grep Language
#NOX_LANG=auto

# Voice mode: chat (default — the LLM answers anything) or commands —
# ONLY keyword commands run (sit/assis/sitz, ...), everything else is ignored:
# deterministic, free, and immune to a TV talking in the room.
#NOX_MODE=commands
# Pairs well with commands mode, in body/nox.env on the ROBOT:
# CONVERSATION_TIMEOUT=0  → every command needs the wake word (no 45 s window)

# The LLM can live on ANY machine — point OPENAI_URL at a bigger PC's Ollama:
#OPENAI_URL=http://192.168.1.50:11434/v1/chat/completions
#LLM_MODEL=qwen2.5:14b

# Battery warning threshold in volts (default 6.8; raise/lower for your pack):
#BATTERY_LOW_V=6.8

# LLM backend — any OpenAI-compatible chat-completions endpoint.
# Cloud (OpenAI): set the key, keep the default URL/model:
#OPENAI_API_KEY=sk-...
# Local (Ollama): no key needed, point at Ollama and pick your model:
#OPENAI_URL=http://127.0.0.1:11434/v1/chat/completions
#LLM_MODEL=llama3.2
# Claude, via Anthropic's OpenAI-compatible endpoint (an API key from
# console.anthropic.com — not a Claude subscription token, sk-ant-oat...):
#OPENAI_URL=https://api.anthropic.com/v1/chat/completions
#OPENAI_API_KEY=sk-ant-api03-...
#LLM_MODEL=claude-opus-5-5
#LLM_TEMPERATURE=off     # current Claude models refuse any temperature
EOF
  chmod 600 /etc/default/nox-brain
  echo "Created /etc/default/nox-brain (template — edit it, then start the service)"
else
  echo "Kept existing /etc/default/nox-brain"
fi

echo
echo "Edit /etc/default/nox-brain (robot address + LLM backend),"
echo "then: sudo systemctl start nox-brain"
