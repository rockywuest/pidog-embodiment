#!/usr/bin/env python3
"""
telegram_bot.py — Control your robot from anywhere via Telegram.

Commands:
  /status  — Robot status (battery, sensors, pose)
  /photo   — Take a photo and send it
  /look    — Photo + face detection
  /speak <text> — Make the robot speak
  /move <action> — Execute action (sit, stand, forward, etc.)
  /face list — List known faces
  /face register <name> — Register face from next photo
  /voice <text> — Process as voice input (full LLM)
  /battery — Battery status
  /help — Show commands

Requires: TELEGRAM_BOT_TOKEN environment variable
Optional: TELEGRAM_ALLOWED_USERS (comma-separated Telegram user IDs)
"""

import os
import sys
import json
import time
import base64
import urllib.request
import urllib.error
import tempfile
from threading import Thread

# ─── Config ───
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
ALLOWED_USERS = os.environ.get("TELEGRAM_ALLOWED_USERS", "").split(",")
ALLOWED_USERS = [u.strip() for u in ALLOWED_USERS if u.strip()]

BODY_HOST = os.environ.get("PIDOG_HOST", "pidog.local")
BODY_PORT = int(os.environ.get("PIDOG_BRIDGE_PORT", "8888"))
BODY_URL = f"http://{BODY_HOST}:{BODY_PORT}"

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
POLL_TIMEOUT = 30


def tg_request(method, data=None, files=None):
    """Make a Telegram Bot API request."""
    url = f"{TELEGRAM_API}/{method}"
    
    if files:
        # Multipart form for file upload
        import io
        boundary = "----NoxBotBoundary"
        body = io.BytesIO()
        
        for key, val in (data or {}).items():
            body.write(f"--{boundary}\r\n".encode())
            body.write(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
            body.write(f"{val}\r\n".encode())
        
        for key, (filename, content, content_type) in files.items():
            body.write(f"--{boundary}\r\n".encode())
            body.write(f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'.encode())
            body.write(f"Content-Type: {content_type}\r\n\r\n".encode())
            body.write(content)
            body.write(b"\r\n")
        
        body.write(f"--{boundary}--\r\n".encode())
        body_bytes = body.getvalue()
        
        req = urllib.request.Request(url, data=body_bytes)
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    elif data:
        body_bytes = json.dumps(data).encode()
        req = urllib.request.Request(url, data=body_bytes)
        req.add_header("Content-Type", "application/json")
    else:
        req = urllib.request.Request(url)
    
    try:
        with urllib.request.urlopen(req, timeout=POLL_TIMEOUT + 10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"ok": False, "error": str(e)}


def send_message(chat_id, text, parse_mode="Markdown"):
    """Send a text message."""
    return tg_request("sendMessage", {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
    })


def send_photo(chat_id, photo_bytes, caption=""):
    """Send a photo."""
    return tg_request("sendPhoto", 
        {"chat_id": str(chat_id), "caption": caption},
        {"photo": ("photo.jpg", photo_bytes, "image/jpeg")}
    )


def bridge_get(path, timeout=15):
    """GET request to body bridge."""
    try:
        url = f"{BODY_URL}{path}"
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}


def bridge_post(path, data, timeout=15):
    """POST request to body bridge."""
    try:
        url = f"{BODY_URL}{path}"
        body = json.dumps(data).encode()
        req = urllib.request.Request(url, data=body)
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}


# ─── Command Handlers ───

def cmd_status(chat_id, args):
    status = bridge_get("/status")
    if "error" in status:
        return send_message(chat_id, f"❌ Bridge error: {status['error']}")
    
    sensors = status.get("sensors", {})
    batt_v = sensors.get("battery_v", "?")
    batt_pct = sensors.get("battery_pct", "?")
    charging = "⚡ Charging" if sensors.get("charging") else "🔋"
    touch = sensors.get("touch", "N")
    
    body = status.get("body_state", {})
    posture = body.get("posture", "?")
    
    text = f"""🐕 *Robot Status*
{charging} Battery: {batt_v}V ({batt_pct}%)
🦴 Posture: {posture}
👆 Touch: {touch}
🕐 Uptime: {status.get('uptime_s', '?')}s"""
    
    return send_message(chat_id, text)


def cmd_photo(chat_id, args):
    send_message(chat_id, "📸 Taking photo...")
    result = bridge_get("/photo")
    
    if "error" in result:
        return send_message(chat_id, f"❌ {result['error']}")
    
    if "image" in result:
        img_bytes = base64.b64decode(result["image"])
        return send_photo(chat_id, img_bytes, "📸 Current view")
    
    return send_message(chat_id, "❌ No image data received")


def cmd_look(chat_id, args):
    send_message(chat_id, "👀 Looking...")
    result = bridge_get("/look", timeout=30)
    
    if "error" in result:
        return send_message(chat_id, f"❌ {result['error']}")
    
    faces = result.get("faces", [])
    caption = f"👁️ Detected {len(faces)} face(s)"
    
    for f in faces:
        name = f.get("name", "unknown")
        conf = f.get("confidence", 0)
        if name != "unknown":
            caption += f"\n• {name} ({conf:.0%})"
    
    if "image" in result:
        img_bytes = base64.b64decode(result["image"])
        return send_photo(chat_id, img_bytes, caption)
    
    return send_message(chat_id, caption)


def cmd_speak(chat_id, args):
    text = " ".join(args) if args else "Hallo!"
    result = bridge_post("/speak", {"text": text})
    
    if "error" in result:
        return send_message(chat_id, f"❌ {result['error']}")
    
    return send_message(chat_id, f'🗣️ Speaking: "{text}"')


def cmd_move(chat_id, args):
    if not args:
        actions = "forward, backward, turn_left, turn_right, stand, sit, lie, wag_tail, bark, trot, stretch, push_up, howling"
        return send_message(chat_id, f"Usage: /move <action>\n\nAvailable: {actions}")
    
    action = args[0].lower()
    result = bridge_post("/action", {"action": action})
    
    if "error" in result:
        return send_message(chat_id, f"❌ {result['error']}")
    
    return send_message(chat_id, f"🐕 Executing: {action}")


def cmd_voice(chat_id, args):
    text = " ".join(args) if args else ""
    if not text:
        return send_message(chat_id, "Usage: /voice <what you want to say to the robot>")
    
    result = bridge_post("/voice/input", {"text": text})
    
    if "error" in result:
        return send_message(chat_id, f"❌ {result['error']}")
    
    return send_message(chat_id, f'🎤 Voice input: "{text}"\n_Processing..._')


def cmd_face(chat_id, args):
    if not args:
        return send_message(chat_id, "Usage:\n/face list\n/face register <name>")
    
    subcmd = args[0].lower()
    
    if subcmd == "list":
        result = bridge_get("/face/list")
        if "error" in result:
            return send_message(chat_id, f"❌ {result['error']}")
        
        faces = result.get("faces", {})
        if not faces:
            return send_message(chat_id, "👤 No faces registered yet")
        
        text = "👤 *Known Faces:*\n"
        for name, count in faces.items():
            text += f"• {name} ({count} samples)\n"
        return send_message(chat_id, text)
    
    elif subcmd == "register" and len(args) > 1:
        name = " ".join(args[1:])
        result = bridge_post("/face/register", {"name": name})
        
        if "error" in result:
            return send_message(chat_id, f"❌ {result['error']}")
        
        return send_message(chat_id, f"✅ Registered face: {name}")
    
    return send_message(chat_id, "Usage:\n/face list\n/face register <name>")


def cmd_battery(chat_id, args):
    result = bridge_get("/status")
    if "error" in result:
        return send_message(chat_id, f"❌ {result['error']}")
    
    sensors = result.get("sensors", {})
    v = sensors.get("battery_v", 0)
    pct = sensors.get("battery_pct", 0)
    charging = sensors.get("charging", False)
    
    bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
    icon = "⚡" if charging else ("🪫" if pct < 20 else "🔋")
    
    return send_message(chat_id, f"{icon} [{bar}] {pct}% ({v:.2f}V)")


def cmd_combo(chat_id, args):
    """Execute a combo: /combo sit+wag_tail+speak:Hallo!"""
    if not args:
        return send_message(chat_id, "Usage: /combo sit+wag_tail+speak:Hello!")
    
    combo = {"actions": [], "speak": None, "rgb": None}
    
    for part in " ".join(args).split("+"):
        part = part.strip()
        if part.startswith("speak:"):
            combo["speak"] = part[6:]
        elif part.startswith("rgb:"):
            try:
                r, g, b = part[4:].split(",")
                combo["rgb"] = {"r": int(r), "g": int(g), "b": int(b)}
            except:
                pass
        else:
            combo["actions"].append(part)
    
    result = bridge_post("/combo", combo)
    if "error" in result:
        return send_message(chat_id, f"❌ {result['error']}")
    
    return send_message(chat_id, f"🎭 Combo executed!")


def cmd_help(chat_id, args):
    text = """🐕 *Nox Robot Commands*

/status — Robot status
/photo — Take a photo
/look — Photo + face detection
/speak <text> — Make robot speak
/move <action> — Move (sit, stand, forward...)
/voice <text> — Voice input (AI processes it)
/face list — List known faces
/face register <name> — Register a face
/battery — Battery level
/combo <actions> — Combo (sit+wag\\_tail+speak:Hi)
/help — This help message"""
    
    return send_message(chat_id, text)


COMMANDS = {
    "status": cmd_status,
    "photo": cmd_photo,
    "look": cmd_look,
    "speak": cmd_speak,
    "move": cmd_move,
    "voice": cmd_voice,
    "face": cmd_face,
    "battery": cmd_battery,
    "combo": cmd_combo,
    "help": cmd_help,
    "start": cmd_help,
}


# ─── Main Loop ───

def handle_update(update):
    """Process a Telegram update."""
    msg = update.get("message", {})
    chat_id = msg.get("chat", {}).get("id")
    user_id = str(msg.get("from", {}).get("id", ""))
    text = msg.get("text", "").strip()
    
    if not chat_id or not text:
        return
    
    # Access control
    if ALLOWED_USERS and user_id not in ALLOWED_USERS:
        send_message(chat_id, "⛔ Unauthorized. Your user ID: " + user_id)
        return
    
    # Parse command
    if text.startswith("/"):
        parts = text[1:].split(None, 1)
        cmd = parts[0].lower().split("@")[0]  # Remove @botname
        args = parts[1].split() if len(parts) > 1 else []
        
        handler = COMMANDS.get(cmd)
        if handler:
            try:
                handler(chat_id, args)
            except Exception as e:
                send_message(chat_id, f"❌ Error: {e}")
        else:
            send_message(chat_id, f"Unknown command: /{cmd}\nTry /help")
    else:
        # Treat non-command messages as voice input
        bridge_post("/voice/input", {"text": text})
        send_message(chat_id, f'🎤 → "{text}"')


def main():
    if not BOT_TOKEN:
        print("ERROR: Set TELEGRAM_BOT_TOKEN environment variable")
        sys.exit(1)
    
    print(f"🤖 Nox Telegram Bot starting...")
    print(f"   Body: {BODY_URL}")
    print(f"   Allowed users: {ALLOWED_USERS or 'ALL (no restriction!)'}")
    
    # Test connection
    me = tg_request("getMe")
    if me.get("ok"):
        bot_name = me["result"].get("username", "?")
        print(f"   Bot: @{bot_name}")
    else:
        print(f"   ⚠️ Bot connection failed: {me}")
    
    offset = 0
    consecutive_errors = 0
    
    while True:
        try:
            updates = tg_request("getUpdates", {
                "offset": offset,
                "timeout": POLL_TIMEOUT,
                "allowed_updates": ["message"],
            })
            
            if updates.get("ok"):
                for update in updates.get("result", []):
                    offset = update["update_id"] + 1
                    Thread(target=handle_update, args=(update,), daemon=True).start()
                consecutive_errors = 0
            else:
                raise Exception(updates.get("error", "Unknown error"))
        
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped.")
            break
        except Exception as e:
            consecutive_errors += 1
            wait = min(consecutive_errors * 5, 60)
            print(f"❌ Error: {e} (retry in {wait}s)")
            time.sleep(wait)


if __name__ == "__main__":
    main()
