#!/usr/bin/env python3
"""
nox_body_client.py — Nox's Brain-side client for controlling the PiDog body.

Runs on Nox's Pi 5 (Clawdbot). Provides a clean Python API for:
- Taking photos and analyzing them
- Controlling movement, head, RGB LEDs
- Speaking via TTS
- Managing face recognition
- Processing voice input from the body

Used by Clawdbot (via shell commands or imported as module).
"""

import json
import base64
import time
import os
import sys
import urllib.request
import urllib.error

# ─── Configuration ───
PIDOG_HOST = os.environ.get("PIDOG_HOST", "pidog.local")
BRIDGE_PORT = int(os.environ.get("PIDOG_BRIDGE_PORT", "8888"))
DAEMON_PORT = int(os.environ.get("PIDOG_DAEMON_PORT", "9999"))
BASE_URL = f"http://{PIDOG_HOST}:{BRIDGE_PORT}"
TIMEOUT = 30


def _request(method, path, data=None, timeout=TIMEOUT):
    """Make HTTP request to PiDog bridge."""
    url = f"{BASE_URL}{path}"
    
    if data is not None:
        body = json.dumps(data).encode()
        req = urllib.request.Request(url, data=body, method=method)
        req.add_header("Content-Type", "application/json")
    else:
        req = urllib.request.Request(url, method=method)
    
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            return json.loads(body)
        except:
            return {"error": f"HTTP {e.code}: {body}"}
    except Exception as e:
        return {"error": str(e)}


def _get(path, timeout=TIMEOUT):
    return _request("GET", path, timeout=timeout)


def _post(path, data=None, timeout=TIMEOUT):
    return _request("POST", path, data=data, timeout=timeout)


# ─── Public API ───

def status():
    """Get full status (sensors, perception, faces)."""
    return _get("/status")


def look():
    """Take a photo and get full perception data (faces, objects, sensors)."""
    return _get("/look", timeout=30)


def photo(save_path=None):
    """Take a photo and optionally save locally."""
    result = _get("/photo", timeout=30)
    
    if save_path and result.get("photo_b64"):
        img_data = base64.b64decode(result["photo_b64"])
        with open(save_path, "wb") as f:
            f.write(img_data)
        result["saved_to"] = save_path
    
    return result


def save_photo(result, path="/tmp/pidog_photo.jpg"):
    """Save a photo result to disk."""
    if result.get("photo_b64"):
        img_data = base64.b64decode(result["photo_b64"])
        with open(path, "wb") as f:
            f.write(img_data)
        return path
    return None


def perception():
    """Get current perception state (cached)."""
    return _get("/perception")


def move(action, steps=3, speed=80):
    """Execute a movement action."""
    return _post("/command", {
        "cmd": "move",
        "action": action,
        "steps": steps,
        "speed": speed
    })


def head(yaw=0, roll=0, pitch=0):
    """Move head."""
    return _post("/head", {"yaw": yaw, "roll": roll, "pitch": pitch})


def rgb(r=128, g=0, b=255, mode="breath", bps=0.8):
    """Set RGB LEDs."""
    return _post("/rgb", {"r": r, "g": g, "b": b, "mode": mode, "bps": bps})


def speak(text):
    """Speak text via TTS."""
    return _post("/speak", {"text": text})


def sound(name):
    """Play a built-in sound."""
    return _post("/command", {"cmd": "sound", "name": name})


def combo(actions=None, speak_text="", rgb_settings=None, head_pos=None):
    """Execute a combination of actions, speech, RGB, and head movement."""
    data = {}
    if actions:
        data["actions"] = actions
    if speak_text:
        data["speak"] = speak_text
    if rgb_settings:
        data["rgb"] = rgb_settings
    if head_pos:
        data["head"] = head_pos
    return _post("/combo", data)


def wake():
    """Wake up sequence."""
    return _post("/command", {"cmd": "wake"})


def sleep_mode():
    """Sleep sequence."""
    return _post("/command", {"cmd": "sleep"})


def reset():
    """Reset to neutral standing position."""
    return _post("/command", {"cmd": "reset"})


def register_face(name):
    """Take a photo and register the detected face as 'name'."""
    return _post("/face/register", {"name": name})


def list_faces():
    """List all known faces."""
    return _get("/faces")


def voice_inbox():
    """Get pending voice messages from PiDog's mic."""
    return _get("/voice/inbox")


def voice_respond(text):
    """Send a voice response to be spoken through PiDog."""
    return _post("/voice/respond", {"text": text})


def battery():
    """Get battery voltage."""
    s = status()
    return s.get("sensors", {}).get("battery_v", "unknown")


def express(emotion, text=""):
    """Express an emotion with matching actions, RGB, and optional speech.
    
    Emotions: happy, sad, curious, excited, alert, sleepy, angry, love, think
    """
    EMOTION_MAP = {
        "happy": {
            "actions": ["wag_tail"],
            "rgb": {"r": 0, "g": 255, "b": 0, "mode": "breath", "bps": 1.5},
        },
        "sad": {
            "actions": ["lie"],
            "rgb": {"r": 0, "g": 0, "b": 128, "mode": "breath", "bps": 0.3},
        },
        "curious": {
            "actions": [],
            "rgb": {"r": 0, "g": 255, "b": 255, "mode": "breath", "bps": 1.0},
            "head": {"yaw": 20, "roll": 0, "pitch": -10},
        },
        "excited": {
            "actions": ["wag_tail", "bark"],
            "rgb": {"r": 255, "g": 255, "b": 0, "mode": "boom", "bps": 2.0},
        },
        "alert": {
            "actions": ["stand"],
            "rgb": {"r": 255, "g": 100, "b": 0, "mode": "boom", "bps": 1.5},
        },
        "sleepy": {
            "actions": ["doze_off"],
            "rgb": {"r": 0, "g": 0, "b": 80, "mode": "breath", "bps": 0.3},
        },
        "angry": {
            "actions": ["bark"],
            "rgb": {"r": 255, "g": 0, "b": 0, "mode": "boom", "bps": 2.0},
        },
        "love": {
            "actions": ["wag_tail"],
            "rgb": {"r": 255, "g": 50, "b": 150, "mode": "breath", "bps": 1.0},
        },
        "think": {
            "actions": [],
            "rgb": {"r": 128, "g": 0, "b": 255, "mode": "breath", "bps": 0.8},
            "head": {"yaw": 15, "roll": -10, "pitch": 10},
        },
    }
    
    settings = EMOTION_MAP.get(emotion, EMOTION_MAP["curious"])
    return combo(
        actions=settings.get("actions", []),
        speak_text=text,
        rgb_settings=settings.get("rgb"),
        head_pos=settings.get("head"),
    )


# ─── CLI Interface ───
def main():
    """CLI interface for quick body control."""
    if len(sys.argv) < 2:
        print("""Nox Body Client — Control PiDog from Nox's Brain

Usage: nox_body_client.py <command> [args...]

Commands:
  status          Full status
  look            Take photo + perception
  photo [path]    Take and save photo
  move <action>   Movement (forward, backward, turn_left, etc.)
  head <y> <r> <p> Move head
  rgb <r> <g> <b> [mode] Set LEDs
  speak <text>    Speak text
  sound <name>    Play sound
  wake            Wake up
  sleep           Go to sleep
  reset           Reset position
  battery         Battery voltage
  express <emotion> [text]  Express emotion
  register <name> Register face
  faces           List known faces
  voice-check     Check voice inbox
  
Emotions: happy, sad, curious, excited, alert, sleepy, angry, love, think
""")
        return
    
    cmd = sys.argv[1]
    args = sys.argv[2:]
    
    try:
        if cmd == "status":
            result = status()
        elif cmd == "look":
            result = look()
            # Don't dump the huge base64 photo
            if "photo_b64" in result:
                result["photo_b64"] = f"[{len(result['photo_b64'])} chars]"
            for face in result.get("faces", []):
                if "crop_b64" in face:
                    face["crop_b64"] = f"[{len(face['crop_b64'])} chars]"
        elif cmd == "photo":
            path = args[0] if args else "/tmp/pidog_photo.jpg"
            result = photo(save_path=path)
            if "photo_b64" in result:
                result["photo_b64"] = f"[saved to {path}]"
        elif cmd == "move":
            action = args[0] if args else "stand"
            result = move(action)
        elif cmd == "head":
            y = float(args[0]) if len(args) > 0 else 0
            r = float(args[1]) if len(args) > 1 else 0
            p = float(args[2]) if len(args) > 2 else 0
            result = head(y, r, p)
        elif cmd == "rgb":
            r_v = int(args[0]) if len(args) > 0 else 128
            g_v = int(args[1]) if len(args) > 1 else 0
            b_v = int(args[2]) if len(args) > 2 else 255
            mode = args[3] if len(args) > 3 else "breath"
            result = rgb(r_v, g_v, b_v, mode)
        elif cmd == "speak":
            text = " ".join(args)
            result = speak(text)
        elif cmd == "sound":
            result = sound(args[0] if args else "single_bark_1")
        elif cmd == "wake":
            result = wake()
        elif cmd == "sleep":
            result = sleep_mode()
        elif cmd == "reset":
            result = reset()
        elif cmd == "battery":
            result = {"battery_v": battery()}
        elif cmd == "express":
            emotion = args[0] if args else "happy"
            text = " ".join(args[1:]) if len(args) > 1 else ""
            result = express(emotion, text)
        elif cmd == "register":
            name = args[0] if args else ""
            result = register_face(name)
        elif cmd == "faces":
            result = list_faces()
        elif cmd == "voice-check":
            result = voice_inbox()
        elif cmd == "combo":
            # JSON from stdin
            import sys
            data = json.loads(sys.stdin.read())
            result = _post("/combo", data)
        else:
            result = {"error": f"unknown command: {cmd}"}
        
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
