#!/usr/bin/env bash
# install.sh — one entry point for the whole setup.
#
# Detects whether this machine is the BODY (the robot: SunFounder SDK present),
# the BRAIN, or both; asks for the one address each side needs; then runs the
# existing role installers. New users had to discover two scripts and three
# config files before anything moved — this collapses it to:
#
#   sudo ./install.sh                       # interactive
#   sudo ./install.sh --role body  --peer 192.168.1.98   # non-interactive
#   sudo ./install.sh --role brain --peer pidog.local
#   ./install.sh --mock                     # no hardware: fake dog on :8888
#
# --peer is "the other machine": the brain's address when installing the body,
# the robot's address when installing the brain. Use 127.0.0.1 when both run
# on one machine.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROLE=""
PEER=""

usage() { sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --role) ROLE="${2:?--role needs body|brain|both}"; shift 2 ;;
    --peer) PEER="${2:?--peer needs an address}"; shift 2 ;;
    --mock) exec python3 "$REPO_DIR/examples/mock_body.py" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ $EUID -ne 0 ]]; then
  echo "Run with sudo: sudo $0 ${ROLE:+--role $ROLE} ${PEER:+--peer $PEER}" >&2
  echo "(no robot? try the hardware-free dog:  $0 --mock)" >&2
  exit 1
fi
RUN_USER="${SUDO_USER:-$USER}"

# ── Which machine is this? ──────────────────────────────────────────────────
if [[ -z "$ROLE" ]]; then
  if sudo -u "$RUN_USER" python3 -c "import pidog" >/dev/null 2>&1; then
    ROLE=body
    echo "Detected the SunFounder SDK → this is the BODY (the robot)."
  else
    echo "No SunFounder SDK here, so probably the BRAIN."
    read -r -p "Install as [brain], body, or both? " ROLE
    ROLE="${ROLE:-brain}"
  fi
fi
case "$ROLE" in body|brain|both) ;; *) echo "Role must be body, brain or both." >&2; exit 1 ;; esac

if [[ -z "$PEER" ]]; then
  if [[ "$ROLE" == "both" ]]; then
    PEER=127.0.0.1
  elif [[ "$ROLE" == "body" ]]; then
    read -r -p "Address of the BRAIN machine (127.0.0.1 if this one): " PEER
  else
    read -r -p "Address of the ROBOT (e.g. pidog.local or its IP): " PEER
  fi
  PEER="${PEER:-127.0.0.1}"
fi

# ── Body ────────────────────────────────────────────────────────────────────
if [[ "$ROLE" == "body" || "$ROLE" == "both" ]]; then
  "$REPO_DIR/scripts/install-body.sh"
  ENV_FILE="$REPO_DIR/body/nox.env"
  # install-body.sh created nox.env from the example; fill in the one value
  # every body needs. sed keeps user edits elsewhere intact.
  BRAIN_ADDR="$([[ "$ROLE" == "both" ]] && echo 127.0.0.1 || echo "$PEER")"
  if grep -qE '^BRAIN_HOST=' "$ENV_FILE"; then
    sed -i.bak -E "s|^BRAIN_HOST=.*|BRAIN_HOST=${BRAIN_ADDR}|" "$ENV_FILE" && rm -f "$ENV_FILE.bak"
  else
    printf '\nBRAIN_HOST=%s\n' "$BRAIN_ADDR" >> "$ENV_FILE"
  fi
  chown "$RUN_USER": "$ENV_FILE"
  echo "BRAIN_HOST=${BRAIN_ADDR} written to body/nox.env"
  systemctl restart nox-body nox-bridge 2>/dev/null || true
fi

# ── Brain ───────────────────────────────────────────────────────────────────
if [[ "$ROLE" == "brain" || "$ROLE" == "both" ]]; then
  "$REPO_DIR/scripts/install-brain.sh"
  CFG=/etc/default/nox-brain
  ROBOT_ADDR="$([[ "$ROLE" == "both" ]] && echo 127.0.0.1 || echo "$PEER")"
  if grep -qE '^PIDOG_HOST=' "$CFG"; then
    sed -i.bak -E "s|^PIDOG_HOST=.*|PIDOG_HOST=${ROBOT_ADDR}|" "$CFG" && rm -f "$CFG.bak"
  else
    printf '\nPIDOG_HOST=%s\n' "$ROBOT_ADDR" >> "$CFG"
  fi
  echo "PIDOG_HOST=${ROBOT_ADDR} written to $CFG"
  systemctl restart nox-brain 2>/dev/null || true
fi

# ── What's next ─────────────────────────────────────────────────────────────
echo
echo "Done. Health check:   ./scripts/doctor.sh"
if [[ "$ROLE" != "body" ]]; then
  echo "LLM for free speech:  edit /etc/default/nox-brain (OPENAI_API_KEY, or"
  echo "                      OPENAI_URL=http://127.0.0.1:11434/v1/chat/completions + LLM_MODEL for Ollama),"
  echo "                      then: sudo systemctl restart nox-brain"
fi
if [[ "$ROLE" != "brain" ]]; then
  echo "Voice input (mic):    see '🎤 Talking to the Dog' in the README (Vosk model + VOSK_MODEL_PATH)"
fi
echo "Claude as the brain:  see '🔌 Claude as the Brain (MCP)' in the README"
