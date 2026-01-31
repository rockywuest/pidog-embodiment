#!/bin/bash
# pidog_v2.sh — Enhanced PiDog control via Brain Bridge
# Falls back to direct TCP if bridge is not available
#
# Usage: pidog_v2.sh <command> [args...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDOG_HOST="${PIDOG_HOST:-pidog.local}"
BRIDGE_PORT="${PIDOG_BRIDGE_PORT:-8888}"
DAEMON_PORT="${PIDOG_DAEMON_PORT:-9999}"

# Check if bridge is running
bridge_available() {
    curl -s --connect-timeout 2 "http://${PIDOG_HOST}:${BRIDGE_PORT}/status" > /dev/null 2>&1
}

# Bridge mode (preferred)
bridge_cmd() {
    local method="$1"
    local path="$2"
    local data="$3"
    
    if [ "$method" = "GET" ]; then
        curl -s --connect-timeout 5 --max-time 30 "http://${PIDOG_HOST}:${BRIDGE_PORT}${path}" 2>&1
    else
        curl -s --connect-timeout 5 --max-time 30 -X POST \
            -H "Content-Type: application/json" \
            -d "$data" \
            "http://${PIDOG_HOST}:${BRIDGE_PORT}${path}" 2>&1
    fi
}

# Direct TCP mode (fallback)
tcp_cmd() {
    local json="$1"
    python3 -c "
import socket, sys, os
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(30)
try:
    s.connect(('${PIDOG_HOST}', ${DAEMON_PORT}))
    s.sendall(('${json}' + '\n').encode('utf-8'))
    print(s.recv(4096).decode().strip())
except Exception as e:
    print('{\"error\":\"' + str(e) + '\"}')
finally:
    s.close()
" 2>&1
}

CMD="$1"
shift
ARGS="$*"

if [ -z "$CMD" ]; then
    echo "Usage: pidog_v2.sh <command> [args...]"
    echo ""
    echo "Commands:"
    echo "  status       Full status (bridge mode)"
    echo "  look         Take photo + perception data"
    echo "  photo [path] Take and save photo locally"
    echo "  move <act>   Movement action"
    echo "  head <y> <r> <p>  Move head"
    echo "  rgb <r> <g> <b> [mode]  Set LEDs"
    echo "  speak <text> Speak text"
    echo "  sound <name> Play sound"
    echo "  wake         Wake up"
    echo "  sleep        Go to sleep"
    echo "  reset        Reset position"
    echo "  battery      Battery voltage"
    echo "  express <emotion> [text]  Express emotion"
    echo "  register <name>  Register face"
    echo "  faces        List known faces"
    echo "  voice-check  Check voice inbox"
    echo "  combo <json> Execute combo (JSON)"
    echo ""
    echo "Emotions: happy, sad, curious, excited, alert, sleepy, angry, love, think"
    exit 0
fi

# Try bridge first, fall back to direct TCP
if bridge_available; then
    case "$CMD" in
        status)     bridge_cmd GET "/status" ;;
        look)       bridge_cmd GET "/look" ;;
        photo)      
            result=$(bridge_cmd GET "/photo")
            if [ -n "$1" ]; then
                echo "$result" | python3 -c "
import sys, json, base64
r = json.load(sys.stdin)
if r.get('photo_b64'):
    with open('$1', 'wb') as f:
        f.write(base64.b64decode(r['photo_b64']))
    print(json.dumps({'ok': True, 'saved': '$1'}))
else:
    print(json.dumps({'error': 'no photo data'}))
"
            else
                echo "$result" | python3 -c "
import sys, json
r = json.load(sys.stdin)
if 'photo_b64' in r: r['photo_b64'] = f'[{len(r[\"photo_b64\"])} chars]'
for f in r.get('faces', []):
    if 'crop_b64' in f: f['crop_b64'] = f'[{len(f[\"crop_b64\"])} chars]'
print(json.dumps(r, indent=2, ensure_ascii=False))
"
            fi
            ;;
        move)       bridge_cmd POST "/command" "{\"cmd\":\"move\",\"action\":\"${1:-stand}\"}" ;;
        head)       bridge_cmd POST "/head" "{\"yaw\":${1:-0},\"roll\":${2:-0},\"pitch\":${3:-0}}" ;;
        rgb)        bridge_cmd POST "/rgb" "{\"r\":${1:-128},\"g\":${2:-0},\"b\":${3:-255},\"mode\":\"${4:-breath}\"}" ;;
        speak)      bridge_cmd POST "/speak" "{\"text\":\"$ARGS\"}" ;;
        sound)      bridge_cmd POST "/command" "{\"cmd\":\"sound\",\"name\":\"${1:-single_bark_1}\"}" ;;
        wake)       bridge_cmd POST "/command" "{\"cmd\":\"wake\"}" ;;
        sleep)      bridge_cmd POST "/command" "{\"cmd\":\"sleep\"}" ;;
        reset)      bridge_cmd POST "/command" "{\"cmd\":\"reset\"}" ;;
        battery)    bridge_cmd GET "/status" | python3 -c "import sys,json; print(json.load(sys.stdin).get('battery_v','?'))" ;;
        express)    bridge_cmd POST "/combo" "{\"actions\":[],\"speak\":\"${2:-}\"}" ;;  # TODO: proper emotion mapping
        register)   bridge_cmd POST "/face/register" "{\"name\":\"${1:-}\"}" ;;
        faces)      bridge_cmd GET "/faces" ;;
        voice-check) bridge_cmd GET "/voice/inbox" ;;
        combo)      bridge_cmd POST "/combo" "$1" ;;
        ping)       bridge_cmd POST "/command" "{\"cmd\":\"ping\"}" ;;
        *)          bridge_cmd POST "/command" "{\"cmd\":\"$CMD\"}" ;;
    esac
else
    # Fallback: direct TCP to daemon
    echo '{"note": "bridge unavailable, using direct TCP"}'
    case "$CMD" in
        move)   tcp_cmd "{\"cmd\":\"move\",\"action\":\"${1:-stand}\",\"steps\":${2:-3},\"speed\":${3:-80}}" ;;
        head)   tcp_cmd "{\"cmd\":\"head\",\"yaw\":${1:-0},\"roll\":${2:-0},\"pitch\":${3:-0}}" ;;
        rgb)    tcp_cmd "{\"cmd\":\"rgb\",\"r\":${1:-128},\"g\":${2:-0},\"b\":${3:-255},\"mode\":\"${4:-breath}\"}" ;;
        speak)  tcp_cmd "{\"cmd\":\"speak\",\"text\":\"$ARGS\"}" ;;
        sound)  tcp_cmd "{\"cmd\":\"sound\",\"name\":\"${1:-single_bark_1}\"}" ;;
        wake)   tcp_cmd "{\"cmd\":\"wake\"}" ;;
        sleep)  tcp_cmd "{\"cmd\":\"sleep\"}" ;;
        reset)  tcp_cmd "{\"cmd\":\"reset\"}" ;;
        status) tcp_cmd "{\"cmd\":\"status\"}" ;;
        ping)   tcp_cmd "{\"cmd\":\"ping\"}" ;;
        photo)  tcp_cmd "{\"cmd\":\"photo\"}" ;;
        *)      tcp_cmd "{\"cmd\":\"$CMD\"}" ;;
    esac
fi
