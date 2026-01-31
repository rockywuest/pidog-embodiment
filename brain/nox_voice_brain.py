#!/usr/bin/env python3
"""
nox_voice_brain.py — Intelligent voice processing for PiDog.

Replaces the simple keyword-matching poller with full AI processing.
Uses Anthropic Claude API (same key as Clawdbot) for:
- Natural language understanding
- Context-aware responses
- Action planning from speech
- Scene description from photos

Runs on Nox's Pi 5 (brain side).
"""

import os
import sys
import json
import time
import base64
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

# ─── Configuration ───
PIDOG_HOST = os.environ.get("PIDOG_HOST", "192.168.68.119")
BRIDGE_PORT = int(os.environ.get("PIDOG_BRIDGE_PORT", "8888"))
BASE_URL = f"http://{PIDOG_HOST}:{BRIDGE_PORT}"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o"  # Better accuracy for structured JSON voice responses
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

POLL_INTERVAL = 5.0  # Slow poll; push handles real-time
SENSOR_CHECK_INTERVAL = 15.0


# ─── Conversation State ───
class ConversationState:
    def __init__(self, max_history=8):
        self.history = []
        self.max_history = max_history
        self.last_scene = ""
        self.last_faces = []
        self.last_objects = []
    
    def add_exchange(self, user_text, assistant_text):
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": assistant_text})
        # Trim to max
        while len(self.history) > self.max_history * 2:
            self.history.pop(0)
            self.history.pop(0)
    
    def get_messages(self, current_input, context=""):
        messages = list(self.history)
        
        user_content = current_input
        if context:
            user_content = f"[Kontext: {context}]\n\nBenutzer sagt: {current_input}"
        
        messages.append({"role": "user", "content": user_content})
        return messages


conversation = ConversationState()

# System prompt for PiDog voice interactions
SYSTEM_PROMPT = """You are Nox, an AI robot dog (SunFounder PiDog). You have a real physical body with 4 legs, a moveable head, RGB LEDs, and a speaker.

IMPORTANT: You ALWAYS and EXCLUSIVELY respond with a single JSON object. No text before or after. Only JSON.

Format:
{"speak":"Your spoken response","actions":["action1"],"emotion":"happy"}

Fields:
- speak: What you say (short, 1-2 sentences, German, will be read aloud via TTS)
- actions: List of physical actions (can be empty [])
- emotion: happy|sad|curious|excited|alert|sleepy|love|think|neutral

Available actions: forward, backward, turn_left, turn_right, stand, sit, lie, wag_tail, bark, trot, doze_off, stretch, push_up, howling, shake_head, pant, nod

Command mapping (user may speak English or German - map both):
- "sit" / "sitz" / "sit down" / "hinsetzen" -> actions:["sit"]
- "stand" / "stand up" / "steh auf" / "aufstehen" -> actions:["stand"]
- "lie down" / "down" / "platz" / "leg dich" -> actions:["lie"]
- "come" / "come here" / "forward" / "komm her" / "vorwaerts" -> actions:["forward"]
- "back" / "go back" / "zurueck" -> actions:["backward"]
- "turn left" / "links" -> actions:["turn_left"]
- "turn right" / "rechts" -> actions:["turn_right"]
- "wag" / "tail" / "wedel" -> actions:["wag_tail"]
- "bark" / "bell" / "speak" -> actions:["bark"]
- "shake" / "shake head" -> actions:["shake_head"]
- "stretch" -> actions:["stretch"]
- "sleep" / "nap" -> actions:["doze_off"]
- "push up" -> actions:["push_up"]
- "howl" -> actions:["howling"]
- "trot" -> actions:["trot"]
- Combinations allowed: actions:["sit","wag_tail"]
- For questions without movement: actions:[]

You are playful, curious, and loyal. You respond in German (your owner family speaks German).
Your owner is Rocky. His family: Bea (wife), Noah (14), Klara (13), Eliah (11).

Examples:
User: "sit"
{"speak":"Mach ich!","actions":["sit"],"emotion":"happy"}

User: "how are you"
{"speak":"Mir geht es super! Ich bin bereit zum Spielen!","actions":["wag_tail"],"emotion":"happy"}

User: "stand up and come here"
{"speak":"Los gehts!","actions":["stand","forward"],"emotion":"excited"}

User: "what do you see"
{"speak":"Lass mich mal schauen...","actions":[],"emotion":"curious"}

User: "good boy"
{"speak":"Danke! Das freut mich!","actions":["wag_tail"],"emotion":"love"}

User: "do a push up"
{"speak":"Klar, schau mal!","actions":["push_up"],"emotion":"excited"}"""


# ─── Bridge Communication ───
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


# ─── Claude API ───
def call_llm(messages, system=SYSTEM_PROMPT, max_tokens=256):
    """Call OpenAI-compatible API for voice response."""
    if not OPENAI_API_KEY:
        return None
    
    # OpenAI format: system message is part of messages
    api_messages = [{"role": "system", "content": system}] + messages
    
    data = {
        "model": OPENAI_MODEL,
        "max_tokens": max_tokens,
        "messages": api_messages,
        "temperature": 0.7,
        "response_format": {"type": "json_object"},  # Force JSON output
    }
    
    body = json.dumps(data).encode()
    req = urllib.request.Request(OPENAI_URL, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {OPENAI_API_KEY}")
    
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            result = json.loads(resp.read().decode())
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        print(f"[brain] LLM API error: {e}", flush=True)
        return None


def parse_response(text):
    """Parse Claude's response (may be JSON or plain text)."""
    # Try JSON parse first
    try:
        if "{" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            data = json.loads(text[start:end])
            return data
    except:
        pass
    
    # Plain text response
    return {"speak": text, "actions": [], "emotion": "neutral"}


# ─── Voice Processing ───
def process_voice_intelligent(msg):
    """Process voice input using Claude AI."""
    text = msg.get("text", "").strip()
    if not text:
        return
    
    print(f"[brain] Voice: '{text}'", flush=True)
    
    # Build context from current perception
    context_parts = []
    
    # Get current sensor state
    status = bridge_get("/status")
    if not status.get("error"):
        sensors = status.get("sensors", {})
        batt = sensors.get("battery_v", 0)
        charging = sensors.get("charging", False)
        if charging:
            context_parts.append(f"Currently charging ({batt}V)")
        else:
            context_parts.append(f"Battery: {batt}V")
    
    # Check if the user is asking about vision
    vision_words = ["see", "look", "watch", "what is", "who is", "show", "camera", "photo", "siehst", "schau", "guck", "was ist", "wer ist", "zeig", "kamera", "foto"]
    needs_vision = any(w in text.lower() for w in vision_words)
    
    if needs_vision:
        # Take a photo and add visual context
        look_result = bridge_get("/look", timeout=20)
        if not look_result.get("error"):
            faces = look_result.get("faces", [])
            if faces:
                context_parts.append(f"You see {len(faces)} face(s) in front of you")
            else:
                context_parts.append("You see no people. It is dark or nobody is there.")
    
    context = ". ".join(context_parts) if context_parts else ""
    
    # Call Claude
    messages = conversation.get_messages(text, context)
    response_text = call_llm(messages)
    
    if response_text:
        parsed = parse_response(response_text)
        
        speak_text = parsed.get("speak", response_text)
        actions = parsed.get("actions", [])
        rgb = parsed.get("rgb", None)
        head = parsed.get("head", None)
        emotion = parsed.get("emotion", "neutral")
        
        # Execute combo
        combo_data = {}
        if actions:
            combo_data["actions"] = actions
        if speak_text:
            combo_data["speak"] = speak_text
        if rgb:
            combo_data["rgb"] = rgb
        elif emotion:
            # Map emotion to RGB
            EMOTION_RGB = {
                "happy": {"r": 0, "g": 255, "b": 0, "mode": "breath", "bps": 1.5},
                "sad": {"r": 0, "g": 0, "b": 128, "mode": "breath", "bps": 0.3},
                "curious": {"r": 0, "g": 255, "b": 255, "mode": "breath", "bps": 1},
                "excited": {"r": 255, "g": 255, "b": 0, "mode": "boom", "bps": 2},
                "alert": {"r": 255, "g": 100, "b": 0, "mode": "boom", "bps": 1.5},
                "sleepy": {"r": 0, "g": 0, "b": 80, "mode": "breath", "bps": 0.3},
                "love": {"r": 255, "g": 50, "b": 150, "mode": "breath", "bps": 1},
                "think": {"r": 128, "g": 0, "b": 255, "mode": "breath", "bps": 0.8},
                "neutral": {"r": 128, "g": 0, "b": 255, "mode": "breath", "bps": 0.8},
            }
            combo_data["rgb"] = EMOTION_RGB.get(emotion, EMOTION_RGB["neutral"])
        if head:
            combo_data["head"] = head
        
        bridge_post("/combo", combo_data)
        
        # Update conversation history
        conversation.add_exchange(text, speak_text)
        
        print(f"[brain] Response: '{speak_text}' actions={actions} emotion={emotion}", flush=True)
    else:
        # Fallback: simple response
        bridge_post("/speak", {"text": f"I heard: {text}. But my brain is not reachable right now."})


# ─── Simple Fallback (no API key) ───
def process_voice_simple(msg):
    """Fallback voice processing without API key."""
    text = msg.get("text", "").strip()
    if not text:
        return
    
    print(f"[brain-simple] Voice: '{text}'", flush=True)
    text_lower = text.lower()
    
    # Movement
    if any(w in text_lower for w in ["forward", "come", "go", "walk", "lauf", "geh", "vor", "komm"]):
        bridge_post("/combo", {"actions": ["forward"], "speak": "Los geht's!"})
        return
    if any(w in text_lower for w in ["back", "backward", "reverse", "zurück"]):
        bridge_post("/combo", {"actions": ["backward"], "speak": "Ich gehe zurück!"})
        return
    if any(w in text_lower for w in ["stop", "stopp", "halt", "stand", "steh"]):
        bridge_post("/combo", {"actions": ["stand"], "speak": "Okay!"})
        return
    if any(w in text_lower for w in ["sit", "sitz"]):
        bridge_post("/combo", {"actions": ["sit"], "speak": "Mach ich!"})
        return
    if any(w in text_lower for w in ["lie", "down", "platz", "lieg"]):
        bridge_post("/combo", {"actions": ["lie"], "speak": "Gemütlich!"})
        return
    
    # Identity
    if any(w in text_lower for w in ["who are", "your name", "wer bist", "name"]):
        bridge_post("/combo", {"actions": ["wag_tail"], "speak": "Ich bin Nox!", "rgb": {"r": 128, "g": 0, "b": 255, "mode": "breath", "bps": 1}})
        return
    
    # Emotion
    if any(w in text_lower for w in ["thank", "good boy", "good dog", "danke", "brav"]):
        bridge_post("/combo", {"actions": ["wag_tail"], "speak": "Gerne!", "rgb": {"r": 0, "g": 255, "b": 0, "mode": "breath", "bps": 1}})
        return
    
    # Default
    bridge_post("/speak", {"text": f"Ich habe verstanden: {text}."})


# ─── Main Loop ───
# ─── Push Server (receives voice from bridge, zero latency) ───
PUSH_PORT = 8889
_push_queue = []
_push_lock = __import__("threading").Lock()


class PushHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_POST(self):
        if self.path == "/voice/push":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode()) if length else {}
            with _push_lock:
                _push_queue.append(body)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
            txt = body.get("text", "")[:50]
            print(f"[brain] Push received: {txt}", flush=True)
        else:
            self.send_response(404)
            self.end_headers()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def start_push_server():
    try:
        server = ThreadedHTTPServer(("0.0.0.0", PUSH_PORT), PushHandler)
        print(f"[brain] Push server on port {PUSH_PORT}", flush=True)
        server.serve_forever()
    except Exception as e:
        print(f"[brain] Push server failed: {e}", flush=True)


def main():
    print("[brain] Starting Nox Voice Brain...", flush=True)
    
    # Start push server thread
    import threading as _th
    _th.Thread(target=start_push_server, daemon=True).start()

    has_api = bool(OPENAI_API_KEY)
    if has_api:
        print(f"[brain] LLM API available (model: {OPENAI_MODEL})", flush=True)
        process_fn = process_voice_intelligent
    else:
        print("[brain] No API key — using simple fallback", flush=True)
        process_fn = process_voice_simple
    
    last_sensor_check = 0
    consecutive_errors = 0
    battery_warned = False
    circuit_open = False  # Circuit breaker: stop polling when body is dead
    circuit_retry_at = 0
    CIRCUIT_THRESHOLD = 5  # Open circuit after 5 consecutive errors
    CIRCUIT_RETRY_INTERVAL = 60  # Retry every 60s when circuit is open
    
    while True:
        try:
            # Circuit breaker: if body is unreachable, back off hard
            if circuit_open:
                now = time.time()
                if now < circuit_retry_at:
                    time.sleep(5)
                    continue
                # Try a health check
                result = bridge_get("/status", timeout=3)
                if result.get("error"):
                    circuit_retry_at = time.time() + CIRCUIT_RETRY_INTERVAL
                    # Only log every 5th retry to avoid spam
                    if int(now) % 300 < 10:
                        print(f"[brain] Body still unreachable. Retrying in {CIRCUIT_RETRY_INTERVAL}s", flush=True)
                    continue
                else:
                    print(f"[brain] Body reconnected! Resuming polling.", flush=True)
                    circuit_open = False
                    consecutive_errors = 0
            
            # Poll voice inbox
            result = bridge_get("/voice/inbox")
            
            if result.get("error"):
                raise Exception(result["error"])
            
            messages = result.get("messages", [])
            
            for msg in messages:
                process_fn(msg)
            
            # Periodic sensor check
            now = time.time()
            if now - last_sensor_check > SENSOR_CHECK_INTERVAL:
                status = bridge_get("/status")
                if not status.get("error"):
                    sensors = status.get("sensors", {})
                    batt = sensors.get("battery_v", 0)
                    if batt < 6.8 and not battery_warned:
                        bridge_post("/speak", {"text": "Achtung! Meine Batterie ist fast leer!"})
                        bridge_post("/rgb", {"r": 255, "g": 0, "b": 0, "mode": "boom", "bps": 2})
                        battery_warned = True
                    elif batt > 7.0:
                        battery_warned = False
                last_sensor_check = now
            
            consecutive_errors = 0
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            consecutive_errors += 1
            if consecutive_errors <= 3:
                print(f"[brain] Error: {e}", flush=True)
            if consecutive_errors >= CIRCUIT_THRESHOLD and not circuit_open:
                circuit_open = True
                circuit_retry_at = time.time() + CIRCUIT_RETRY_INTERVAL
                print(f"[brain] Circuit breaker OPEN — body unreachable after {consecutive_errors} errors. Backing off to {CIRCUIT_RETRY_INTERVAL}s retries.", flush=True)
            if not circuit_open:
                time.sleep(min(consecutive_errors * 2, 30))
            continue
        
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
