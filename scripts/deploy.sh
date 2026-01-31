#!/bin/bash
# deploy.sh — Deploy Nox embodiment files to PiDog
# Run from Nox's Pi (the brain)

set -e

PIDOG_HOST="pidog@pidog.local"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Deploying Nox Embodiment to PiDog ==="

# 1. Copy Python files
echo "[1/5] Copying Python files..."
scp "$SRC_DIR/nox_brain_bridge.py" "$PIDOG_HOST:/home/pidog/"
scp "$SRC_DIR/nox_voice_loop_v2.py" "$PIDOG_HOST:/home/pidog/"

# 2. Copy systemd service
echo "[2/5] Installing systemd service..."
scp "$SRC_DIR/nox-bridge.service" "$PIDOG_HOST:/tmp/"
ssh "$PIDOG_HOST" "sudo cp /tmp/nox-bridge.service /etc/systemd/system/ && sudo systemctl daemon-reload"

# 3. Create directories on PiDog
echo "[3/5] Creating directories..."
ssh "$PIDOG_HOST" "mkdir -p /home/pidog/nox_face_db /home/pidog/nox_memory"

# 4. Start/restart services
echo "[4/5] Starting services..."
ssh "$PIDOG_HOST" "sudo systemctl enable nox-bridge && sudo systemctl restart nox-bridge"

# Wait for bridge to be ready
echo "  Waiting for bridge..."
sleep 3
for i in $(seq 1 10); do
    if curl -s --connect-timeout 2 "http://pidog.local:8888/status" > /dev/null 2>&1; then
        echo "  ✅ Bridge is up!"
        break
    fi
    sleep 1
done

# 5. Install new pidog.sh locally
echo "[5/5] Updating local pidog script..."
cp "$SRC_DIR/pidog_v2.sh" "$HOME/clawd/scripts/pidog_v2.sh"
chmod +x "$HOME/clawd/scripts/pidog_v2.sh"

echo ""
echo "=== Deployment Complete ==="
echo "Bridge: http://pidog.local:8888"
echo "Test:   curl http://pidog.local:8888/status"
echo "CLI:    ~/clawd/scripts/pidog_v2.sh status"
