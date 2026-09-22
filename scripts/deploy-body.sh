#!/usr/bin/env bash
# Copy the body code to a robot over SSH and restart its services.
#
# This replaces the earlier deploy.sh / deploy_all.sh, which pointed at paths
# that no longer existed, hard-coded one particular robot's IP addresses, and
# copied only a subset of the modules — enough to leave a robot running a mix of
# old and new code, or crash the daemon on a missing import.
#
# Usage:
#   ./scripts/deploy-body.sh                      # pidog@pidog.local
#   ./scripts/deploy-body.sh cat1@cat1.local
#   PIDOG_HOST=192.168.1.42 ./scripts/deploy-body.sh
#   ./scripts/deploy-body.sh cat1@cat1.local --no-restart
#
# Prefer `git pull` on the robot when it has network access; this script is for
# robots that do not, or for testing uncommitted changes.
set -euo pipefail

TARGET="${1:-}"
if [[ -z "$TARGET" || "$TARGET" == --* ]]; then
  TARGET="${PIDOG_USER:-pidog}@${PIDOG_HOST:-pidog.local}"
fi
RESTART=1
for arg in "$@"; do
  [[ "$arg" == "--no-restart" ]] && RESTART=0
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_DIR="${PIDOG_DIR:-}"

echo "🐕 Deploying body code to $TARGET"

if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$TARGET" true 2>/dev/null; then
  echo "❌ Cannot SSH to $TARGET (needs a key — password prompts are not supported here)" >&2
  exit 1
fi

# Where the services expect the code: read it from the installed unit, so a
# deploy can never land in a directory the robot does not run from.
if [[ -z "$REMOTE_DIR" ]]; then
  REMOTE_DIR="$(ssh "$TARGET" "systemctl show nox-body -p ExecStart --value 2>/dev/null | sed -n 's|.*[[:space:]]\(/[^[:space:]]*\)/nox_daemon.py.*|\1|p'" || true)"
fi
if [[ -z "$REMOTE_DIR" ]]; then
  REMOTE_DIR="$(ssh "$TARGET" 'echo $HOME')/pidog-embodiment/body"
  echo "   nox-body not installed yet — defaulting to $REMOTE_DIR"
fi
echo "   target directory: $REMOTE_DIR"

# Every module the services import. Missing one of these is how a deploy breaks
# the daemon with an ImportError instead of an obvious failure.
FILES=(body/*.py)
echo "   files: ${#FILES[@]} python modules"

ssh "$TARGET" "mkdir -p '$REMOTE_DIR'"
if command -v rsync >/dev/null 2>&1; then
  rsync -az --info=NAME "${FILES[@]/#/$REPO_DIR/}" "$TARGET:$REMOTE_DIR/"
else
  scp -q "${FILES[@]/#/$REPO_DIR/}" "$TARGET:$REMOTE_DIR/"
  printf '   %s\n' "${FILES[@]##*/}"
fi

if [[ $RESTART -eq 0 ]]; then
  echo "✅ Files copied. Restart skipped (--no-restart)."
  exit 0
fi

echo "🔄 Restarting services..."
ssh "$TARGET" "sudo systemctl restart nox-body nox-bridge 2>&1 | tail -3 || true"

PORT="${PIDOG_BRIDGE_PORT:-8888}"
echo "   waiting for the bridge on :$PORT ..."
for _ in $(seq 1 15); do
  if ssh "$TARGET" "curl -sf --max-time 3 http://127.0.0.1:$PORT/status >/dev/null"; then
    echo "✅ Bridge is up."
    ssh "$TARGET" "curl -s http://127.0.0.1:$PORT/status | head -c 400; echo"
    exit 0
  fi
  sleep 2
done

echo "⚠️  Bridge did not answer within 30s. Check on the robot:" >&2
echo "    journalctl -u nox-body -n 40 --no-pager" >&2
exit 1
