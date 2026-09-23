#!/bin/bash
# wifi_watchdog.sh — Keeps PiDog's WiFi alive
# Deploy to PiDog as systemd service or cron
# Checks connectivity every 60s, recovers if lost

# Router address. Detected from the routing table so this works on any network;
# override with GATEWAY=... if the default route is not what you want to probe.
GATEWAY="${GATEWAY:-$(ip route 2>/dev/null | awk '/^default/{print $3; exit}')}"
if [[ -z "$GATEWAY" ]]; then
  echo "$(date -Is) no default route found — set GATEWAY=<router-ip>" >&2
  exit 1
fi
CHECK_INTERVAL=60
MAX_FAILURES=3
LOG="/tmp/wifi_watchdog.log"

failures=0

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" >> "$LOG"
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1"
}

log "WiFi watchdog started"

while true; do
    if ping -c 1 -W 3 "$GATEWAY" > /dev/null 2>&1; then
        if [ $failures -gt 0 ]; then
            log "WiFi recovered after $failures failures"
        fi
        failures=0
    else
        failures=$((failures + 1))
        log "WiFi check failed ($failures/$MAX_FAILURES)"
        
        if [ $failures -ge $MAX_FAILURES ]; then
            log "WiFi appears dead. Attempting recovery..."
            
            # Step 1: Restart wpa_supplicant
            sudo systemctl restart wpa_supplicant 2>/dev/null
            sleep 5
            
            if ping -c 1 -W 3 "$GATEWAY" > /dev/null 2>&1; then
                log "Recovery: wpa_supplicant restart worked"
                failures=0
                continue
            fi
            
            # Step 2: Bring interface down/up
            sudo ip link set wlan0 down
            sleep 2
            sudo ip link set wlan0 up
            sleep 10
            
            if ping -c 1 -W 3 "$GATEWAY" > /dev/null 2>&1; then
                log "Recovery: interface bounce worked"
                failures=0
                continue
            fi
            
            # Step 3: Full network restart
            sudo systemctl restart NetworkManager 2>/dev/null || sudo systemctl restart dhcpcd 2>/dev/null
            sleep 15
            
            if ping -c 1 -W 3 "$GATEWAY" > /dev/null 2>&1; then
                log "Recovery: network service restart worked"
                failures=0
            else
                log "Recovery FAILED. WiFi still dead. Will retry in ${CHECK_INTERVAL}s"
            fi
            
            failures=0  # Reset counter to avoid spam
        fi
    fi
    
    sleep "$CHECK_INTERVAL"
done
