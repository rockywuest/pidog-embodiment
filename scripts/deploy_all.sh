#!/bin/bash
# deploy_all.sh — Deploy everything to PiDog and restart services
# Run from Pi 5 (brain)

set -e

PIDOG_HOST="${PIDOG_HOST:-192.168.68.119}"
PIDOG_USER="pidog"
PIDOG_DIR="/home/pidog"
SRC_DIR="$(dirname "$0")"

echo "🐕 Deploying to PiDog at $PIDOG_HOST..."

# Test connectivity
if ! ping -c 1 -W 3 "$PIDOG_HOST" > /dev/null 2>&1; then
    echo "❌ PiDog not reachable at $PIDOG_HOST"
    # Try Tailscale
    if ping -c 1 -W 3 100.67.236.125 > /dev/null 2>&1; then
        PIDOG_HOST="100.67.236.125"
        echo "✅ Using Tailscale IP: $PIDOG_HOST"
    else
        echo "❌ PiDog not reachable via Tailscale either. Aborting."
        exit 1
    fi
fi

echo "📦 Copying files..."

# Core scripts
scp "$SRC_DIR/nox_voice_loop_v2.py" "$PIDOG_USER@$PIDOG_HOST:$PIDOG_DIR/"
scp "$SRC_DIR/nox_brain_bridge.py" "$PIDOG_USER@$PIDOG_HOST:$PIDOG_DIR/"
scp "$SRC_DIR/nox_autonomous_v2.py" "$PIDOG_USER@$PIDOG_HOST:$PIDOG_DIR/"
scp "$SRC_DIR/nox_face_recognition.py" "$PIDOG_USER@$PIDOG_HOST:$PIDOG_DIR/"

# WiFi watchdog
scp "$SRC_DIR/wifi_watchdog.sh" "$PIDOG_USER@$PIDOG_HOST:$PIDOG_DIR/"
ssh "$PIDOG_USER@$PIDOG_HOST" "chmod +x $PIDOG_DIR/wifi_watchdog.sh"

# Service files
scp "$SRC_DIR/nox-wifi-watchdog.service" "$PIDOG_USER@$PIDOG_HOST:/tmp/"
ssh "$PIDOG_USER@$PIDOG_HOST" "sudo cp /tmp/nox-wifi-watchdog.service /etc/systemd/system/ && sudo systemctl daemon-reload"

echo "🔧 Configuring services..."

# Enable WiFi watchdog
ssh "$PIDOG_USER@$PIDOG_HOST" "sudo systemctl enable --now nox-wifi-watchdog 2>/dev/null || true"

# Disable WiFi power management (major cause of drops on Pi 4)
ssh "$PIDOG_USER@$PIDOG_HOST" "sudo iw dev wlan0 set power_save off 2>/dev/null || true"

# Make power_save off persistent
ssh "$PIDOG_USER@$PIDOG_HOST" "echo 'wireless-power off' | sudo tee -a /etc/network/interfaces.d/wlan0 2>/dev/null; sudo bash -c 'echo \"options 8192cu rtw_power_mgnt=0 rtw_enusbss=0\" > /etc/modprobe.d/wifi-powersave.conf' 2>/dev/null || true"

echo "🔄 Restarting services..."
ssh "$PIDOG_USER@$PIDOG_HOST" "sudo systemctl restart nox-voice nox-bridge 2>/dev/null"

# Restart brain-side voice brain on Pi 5
systemctl --user restart nox-voice-brain 2>/dev/null || true

echo "✅ Deployment complete!"
echo ""
echo "Services on PiDog:"
ssh "$PIDOG_USER@$PIDOG_HOST" "sudo systemctl status nox-body nox-bridge nox-voice nox-wifi-watchdog --no-pager 2>&1 | grep -E '●|Active'"
echo ""
echo "Services on Brain (Pi 5):"
systemctl --user status nox-voice-brain --no-pager 2>&1 | grep -E '●|Active'
