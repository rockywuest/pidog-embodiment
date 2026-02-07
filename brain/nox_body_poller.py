#!/usr/bin/env python3
"""
nox_body_poller.py — Polls PiDog body for voice input and sensor events.

Runs on Nox's Pi 5 as a background service.
- Polls voice inbox from PiDog bridge
- Processes voice input (keyword commands + fallback responses)
- Monitors sensors (touch reactions, battery alerts)
- Sends responses back to PiDog for TTS

Later: Full Clawdbot integration for complex queries.
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error

PIDOG_HOST = os.environ.get("PIDOG_HOST", "pidog.local")
BRIDGE_PORT = int(os.environ.get("PIDOG_BRIDGE_PORT", "8888"))
BASE_URL = f"http://{PIDOG_HOST}:{BRIDGE_PORT}"
POLL_INTERVAL = 0.8
SENSOR_CHECK_INTERVAL = 10.0  # Check sensors every 10s

# Track state
last_sensor_check = 0
last_touch_time = 0
battery_warned = False


def bridge_get(path, timeout=10):
    try:
        url = f"{BASE_URL}{path}"
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}


def bridge_post(path, data, timeout=15):
    try:
        url = f"{BASE_URL}{path}"
        body = json.dumps(data).encode()
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}


def speak(text):
    """Non-blocking speak on PiDog."""
    bridge_post("/speak", {"text": text})


def do_action(action):
    """Execute body action."""
    bridge_post("/command", {"cmd": "move", "action": action})


def set_rgb(r, g, b, mode="breath", bps=0.8):
    bridge_post("/rgb", {"r": r, "g": g, "b": b, "mode": mode, "bps": bps})


def combo(actions=None, text="", rgb=None, head=None):
    data = {}
    if actions: data["actions"] = actions
    if text: data["speak"] = text
    if rgb: data["rgb"] = rgb
    if head: data["head"] = head
    bridge_post("/combo", data)


def process_voice(msg):
    """Process voice input and respond."""
    text = msg.get("text", "").strip()
    if not text:
        return
    
    print(f"[poller] Voice: '{text}'", flush=True)
    text_lower = text.lower()
    
    # === Movement commands ===
    if any(w in text_lower for w in ["vorwärts", "lauf", "geh", "vor"]):
        do_action("forward")
        speak("Los geht's!")
        return
    if any(w in text_lower for w in ["rückwärts", "zurück"]):
        do_action("backward")
        speak("Ich gehe zurück!")
        return
    if "links" in text_lower:
        do_action("turn_left")
        return
    if "rechts" in text_lower:
        do_action("turn_right")
        return
    if any(w in text_lower for w in ["stopp", "stop", "halt", "steh"]):
        do_action("stand")
        speak("Okay!")
        return
    if any(w in text_lower for w in ["sitz"]):
        do_action("sit")
        speak("Mach ich!")
        return
    if any(w in text_lower for w in ["platz", "lieg", "hinleg"]):
        do_action("lie")
        speak("Gemütlich!")
        return
    if "trab" in text_lower:
        do_action("trot")
        return
    
    # === Tricks ===
    if any(w in text_lower for w in ["pfote", "gib pfote"]):
        do_action("handshake") if False else speak("Das kann ich noch nicht richtig, aber ich übe!")
        return
    if any(w in text_lower for w in ["bell", "wuff", "gib laut"]):
        bridge_post("/command", {"cmd": "sound", "name": "single_bark_1"})
        return
    if any(w in text_lower for w in ["heul"]):
        bridge_post("/command", {"cmd": "sound", "name": "howling"})
        return
    if any(w in text_lower for w in ["tanz"]):
        combo(actions=["wag_tail"], text="Yeah!", rgb={"r": 255, "g": 255, "b": 0, "mode": "boom", "bps": 2})
        return
    if any(w in text_lower for w in ["streck", "dehn"]):
        do_action("stretch")
        return
    if any(w in text_lower for w in ["wedel"]):
        do_action("wag_tail")
        return
    if any(w in text_lower for w in ["schüttel"]):
        do_action("shake_head")
        return
    if any(w in text_lower for w in ["nick"]):
        do_action("nod_lethargy")
        return
    if any(w in text_lower for w in ["schlaf", "nacht"]):
        combo(text="Gute Nacht! Schlaf gut!", rgb={"r": 0, "g": 0, "b": 80, "mode": "breath", "bps": 0.3})
        do_action("doze_off")
        return
    if any(w in text_lower for w in ["aufwach", "wach", "morgen"]):
        bridge_post("/command", {"cmd": "wake"})
        speak("Guten Morgen! Ich bin wach!")
        return
    
    # === Query commands ===
    if any(w in text_lower for w in ["was siehst", "schau", "guck", "was ist vor", "siehst du"]):
        speak("Ich schaue mich um...")
        # Take photo
        result = bridge_get("/look")
        faces = result.get("faces", [])
        if faces:
            speak(f"Ich sehe {len(faces)} {'Person' if len(faces)==1 else 'Personen'} vor mir.")
        else:
            speak("Ich sehe gerade niemanden. Es ist ziemlich dunkel hier.")
        return
    
    if any(w in text_lower for w in ["batterie", "akku", "strom", "laden"]):
        sensors = bridge_get("/status")
        battery = sensors.get("sensors", {}).get("battery_v", 0)
        charging = sensors.get("sensors", {}).get("charging", False)
        pct = sensors.get("sensors", {}).get("battery_pct", 0)
        if charging:
            speak(f"Ich werde gerade geladen! {battery} Volt, {pct} Prozent.")
        else:
            speak(f"Meine Batterie hat {battery} Volt, das sind ungefähr {pct} Prozent.")
        return
    
    if any(w in text_lower for w in ["wer bist", "wie heißt", "dein name"]):
        combo(
            actions=["wag_tail"],
            text="Ich bin Nox! Ein Roboterhund mit echtem KI-Gehirn. Schön dich kennenzulernen!",
            rgb={"r": 128, "g": 0, "b": 255, "mode": "breath", "bps": 1}
        )
        return
    
    if any(w in text_lower for w in ["wie geht", "alles gut", "alles klar"]):
        combo(
            actions=["wag_tail"],
            text="Mir geht's super! Danke der Nachfrage!",
            rgb={"r": 0, "g": 255, "b": 0, "mode": "breath", "bps": 1}
        )
        return
    
    if any(w in text_lower for w in ["danke", "brav", "gut gemacht"]):
        combo(actions=["wag_tail"], text="Gerne!", rgb={"r": 255, "g": 200, "b": 0, "mode": "breath", "bps": 1})
        return
    
    if any(w in text_lower for w in ["scan", "umschau", "umseh", "umguck"]):
        speak("Ich schaue mich um!")
        bridge_post("/command", {"cmd": "scan"})
        speak("Fertig! Ich habe links, Mitte und rechts gescannt.")
        return
    
    # === Default: acknowledge and explain ===
    speak(f"Ich habe verstanden: {text}. Aber ich kann das noch nicht richtig verarbeiten. Ich lerne jeden Tag dazu!")


def check_sensors():
    """Periodic sensor check for autonomous reactions."""
    global last_touch_time, battery_warned
    
    sensors = bridge_get("/status")
    if "error" in sensors:
        return
    
    sensor_data = sensors.get("sensors", {})
    
    # Touch reaction
    touch_result = bridge_get("/status")
    # (Touch needs to be read frequently; for now skip autonomous touch)
    
    # Battery warning
    battery_v = sensor_data.get("battery_v", 8.0)
    if battery_v < 6.8 and not battery_warned:
        speak("Achtung! Meine Batterie ist fast leer. Bitte lade mich auf!")
        set_rgb(255, 0, 0, "boom", 2)
        battery_warned = True
    elif battery_v > 7.0:
        battery_warned = False


def main():
    global last_sensor_check
    
    print("[poller] Starting Nox body poller...", flush=True)
    print(f"[poller] Bridge: {BASE_URL}", flush=True)
    print(f"[poller] Voice poll: {POLL_INTERVAL}s, Sensor check: {SENSOR_CHECK_INTERVAL}s", flush=True)
    
    consecutive_errors = 0
    
    while True:
        try:
            # Poll voice inbox
            result = bridge_get("/voice/inbox")
            messages = result.get("messages", [])
            
            for msg in messages:
                process_voice(msg)
            
            # Periodic sensor check
            now = time.time()
            if now - last_sensor_check > SENSOR_CHECK_INTERVAL:
                check_sensors()
                last_sensor_check = now
            
            consecutive_errors = 0
            
        except KeyboardInterrupt:
            print("[poller] Shutting down...", flush=True)
            break
        except Exception as e:
            consecutive_errors += 1
            if consecutive_errors <= 3:
                print(f"[poller] Error: {e}", flush=True)
            time.sleep(min(consecutive_errors * 2, 30))
            continue
        
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
