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
import queue
import re
import threading
import time
import base64
import socket
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

# ─── Configuration ───
PIDOG_HOST = os.environ.get("PIDOG_HOST", "pidog.local")
BRIDGE_PORT = int(os.environ.get("PIDOG_BRIDGE_PORT", "8888"))
BASE_URL = f"http://{PIDOG_HOST}:{BRIDGE_PORT}"
# Must match NOX_API_TOKEN in the robot's body/nox.env when that is set (issue #30).
API_TOKEN = os.environ.get("NOX_API_TOKEN", "").strip()


def _bridge_request(url, data=None):
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if API_TOKEN:
        req.add_header("Authorization", f"Bearer {API_TOKEN}")
    return req

# Who the robot belongs to — from the environment, never from source: real names
# (children's especially) do not belong in a public repository.
HOUSEHOLD_LINE = os.environ.get(
    "NOX_HOUSEHOLD",
    "You do not know your household yet — ask for names when you need them.")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
# Any OpenAI-compatible chat-completions endpoint works. For a local Ollama:
#   OPENAI_URL=http://127.0.0.1:11434/v1/chat/completions  LLM_MODEL=llama3.2
# api.openai.com requires OPENAI_API_KEY; local endpoints don't.
OPENAI_MODEL = os.environ.get("LLM_MODEL", "gpt-4o")  # gpt-4o: better accuracy for structured JSON voice responses
OPENAI_URL = os.environ.get("OPENAI_URL", "https://api.openai.com/v1/chat/completions")
LLM_CONFIGURED = bool(OPENAI_API_KEY) or "api.openai.com" not in OPENAI_URL
# Local CPU inference (Ollama on a PC) routinely needs 60-120s per reply,
# cloud APIs a few seconds — default accordingly, override via LLM_TIMEOUT.
LLM_TIMEOUT = int(os.environ.get(
    "LLM_TIMEOUT", "30" if "api.openai.com" in OPENAI_URL else "120"))

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

You are playful, curious, and loyal. You respond in the language you are spoken to.
{HOUSEHOLD_LINE}

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
SYSTEM_PROMPT = SYSTEM_PROMPT.replace("{HOUSEHOLD_LINE}", HOUSEHOLD_LINE)


# ─── Bridge Communication ───
def bridge_get(path, timeout=10):
    try:
        url = f"{BASE_URL}{path}"
        with urllib.request.urlopen(_bridge_request(url), timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}


def bridge_post(path, data, timeout=15):
    try:
        url = f"{BASE_URL}{path}"
        body = json.dumps(data).encode()
        with urllib.request.urlopen(_bridge_request(url, body), timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}


# ─── Claude API ───
def call_llm(messages, system=SYSTEM_PROMPT, max_tokens=256):
    """Call OpenAI-compatible API for voice response."""
    if not LLM_CONFIGURED:
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
    if OPENAI_API_KEY:
        req.add_header("Authorization", f"Bearer {OPENAI_API_KEY}")
    
    try:
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
            result = json.loads(resp.read().decode())
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        # urllib wraps timeouts in URLError(reason=socket.timeout)
        reason = getattr(e, "reason", e)
        if isinstance(reason, (TimeoutError, socket.timeout)):
            print(f"[brain] LLM timed out after {LLM_TIMEOUT}s (model: {OPENAI_MODEL}) — "
                  f"local CPU inference can be slow; raise LLM_TIMEOUT if this repeats", flush=True)
        else:
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
    except Exception:
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
        # Fallback: the LLM call failed (timeout, API error) — the bridge
        # itself is fine, so don't claim the brain is unreachable.
        bridge_post("/speak", {"text": f"I heard: {text}. I'm still thinking — give me a moment and try again."})


# ─── Simple Fallback (no API key) ───
# Keyword commands for the no-LLM fallback (issue #42). Whole words only —
# substring matching made "good dog" walk forward ("go") — and every command
# in a sentence counts, in the order spoken: "Setz dich hin und wedel mit dem
# Schwanz" is sit + wag_tail. German, English and French.
_INTENTS = [
    # (action, patterns per language)
    ("sit", {"de": r"setz\w*|sitz\w*|hinsetzen", "en": r"sit", "fr": r"assis|assieds|asseoir"}),
    ("lie", {"de": r"platz|leg dich|lieg\w*|hinlegen", "en": r"lie down|lay down|(?<!sit )down",
             "fr": r"couch[ée]e?|allonge\w*"}),
    ("stand", {"de": r"steh\w*|aufstehen|stopp?|halt", "en": r"stand( up)?|stop",
               "fr": r"debout|l[eè]ve-toi|arr[eê]te\w*|stop"}),
    ("forward", {"de": r"komm( her)?|vorw[aä]rts|lauf\w*|geh los", "en": r"come( here)?|forward|walk|go(?! back)",
                 "fr": r"viens|avance\w*|marche"}),
    ("backward", {"de": r"zur[uü]ck", "en": r"(?<!come )back(wards?)?|reverse", "fr": r"recule\w*|arri[eè]re"}),
    ("turn_left", {"de": r"links", "en": r"(turn )?left", "fr": r"gauche"}),
    ("turn_right", {"de": r"rechts", "en": r"(turn )?right", "fr": r"droite"}),
    ("wag_tail", {"de": r"wedel\w*|schwanz", "en": r"wag\w*|tail", "fr": r"remue\w*|queue"}),
    ("bark", {"de": r"bell\w*|gib laut", "en": r"bark", "fr": r"aboie\w*|aboyer"}),
    ("shake_head", {"de": r"kopf sch[uü]tteln|sch[uü]ttel\w*", "en": r"shake( your)? head", "fr": r"secoue\w*"}),
    ("stretch", {"de": r"streck\w*", "en": r"stretch", "fr": r"[ée]tire\w*"}),
    ("doze_off", {"de": r"schlaf\w*", "en": r"sleep|nap", "fr": r"dors|dormir"}),
    ("push_up", {"de": r"liegest[uü]tz\w*", "en": r"push ?ups?", "fr": r"pompes?"}),
    ("howling", {"de": r"heul\w*", "en": r"howl\w*", "fr": r"hurle\w*"}),
]
_PRAISE = {"de": r"danke|brav\w*|guter hund", "en": r"thanks?|thank you|good (boy|girl|dog)",
           "fr": r"merci|bon chien|bravo"}
_WHO = {"de": r"wer bist du|wie hei[sß]t du|dein name", "en": r"who are you|your name",
        "fr": r"qui es-tu|qui es tu|comment tu t'appelles|ton nom"}
_REPLIES = {
    "ok": {"de": "Mach ich!", "en": "On it!", "fr": "D'accord !"},
    "praise": {"de": "Gerne!", "en": "You're welcome!", "fr": "Avec plaisir !"},
    "who": {"de": "Ich bin Nox!", "en": "I'm Nox!", "fr": "Je suis Nox !"},
    "unknown": {"de": "Das verstehe ich nicht: {text}", "en": "I don't understand: {text}",
                "fr": "Je ne comprends pas : {text}"},
}
# Frequent function words, to answer in the speaker's language when no
# command word gave it away.
_LANG_HINTS = {
    "de": r"ich|du|dich|und|der|die|das|nicht|mit|bitte",
    "en": r"i|you|the|and|please|your|is|to",
    "fr": r"je|tu|toi|le|la|les|et|pas|s'il|avec",
}


def _find(pattern, text):
    # A hyphen may follow ("assis-toi") but not precede ("ex-sit" is no command).
    return [m.start() for m in re.finditer(r"(?<![\w-])(?:" + pattern + r")(?!\w)", text)]


def parse_simple_command(text):
    """Keyword parse: {"actions": [...], "intent": ..., "lang": "de"|"en"|"fr"}."""
    t = " ".join((text or "").lower().replace("’", "'").split())
    votes = {"de": 0, "en": 0, "fr": 0}
    hits = []
    for action, patterns in _INTENTS:
        for lang, pattern in patterns.items():
            positions = _find(pattern, t)
            if positions:
                votes[lang] += 2
                hits.append((positions[0], action))
    actions = []
    for _, action in sorted(hits):
        if action not in actions:
            actions.append(action)
    intent = "command" if actions else None
    for name, patterns in (("who", _WHO), ("praise", _PRAISE)):
        for lang, pattern in patterns.items():
            if _find(pattern, t):
                votes[lang] += 2
                intent = intent or name
    for lang, pattern in _LANG_HINTS.items():
        votes[lang] += len(_find(pattern, t))
    lang = max(("de", "en", "fr"), key=lambda k: votes[k])  # ties → German
    if intent == "praise" and not actions:
        actions = ["wag_tail"]
    if intent == "who" and not actions:
        actions = ["wag_tail"]
    return {"actions": actions, "intent": intent, "lang": lang}


def process_voice_simple(msg):
    """Fallback voice processing without an LLM: keyword commands."""
    text = msg.get("text", "").strip()
    if not text:
        return

    print(f"[brain-simple] Voice: '{text}'", flush=True)
    cmd = parse_simple_command(text)
    lang = cmd["lang"]
    if cmd["intent"] is None:
        print(f"[brain-simple] No command recognised (lang={lang}) — set OPENAI_API_KEY "
              "or OPENAI_URL for free-form understanding", flush=True)
        bridge_post("/speak", {"text": _REPLIES["unknown"][lang].format(text=text)})
        return
    reply = _REPLIES["ok" if cmd["intent"] == "command" else cmd["intent"]][lang]
    combo = {"actions": cmd["actions"], "speak": reply}
    if cmd["intent"] in ("praise", "who"):
        combo["rgb"] = ({"r": 0, "g": 255, "b": 0, "mode": "breath", "bps": 1} if cmd["intent"] == "praise"
                        else {"r": 128, "g": 0, "b": 255, "mode": "breath", "bps": 1})
    print(f"[brain-simple] → {cmd['actions']} ({lang})", flush=True)
    bridge_post("/combo", combo)


# ─── Main Loop ───
# ─── Push Server (receives voice from bridge, zero latency) ───
PUSH_PORT = 8889
# Filled by the push server, drained by the main loop while it waits between
# inbox polls — so a pushed command runs at once instead of never: this used to
# be a plain list nobody read, and every command waited for the 5 s inbox poll
# (issue #42).
_push_queue = queue.Queue()


_seen = []  # (ts, text) of recently handled messages, newest last
_seen_lock = threading.Lock()


def first_time(msg):
    """True the first time a message is seen. A bridge from before issue #42
    pushes AND keeps every message in /voice/inbox; without this a new brain
    would sit twice for one "Platz"."""
    key = (msg.get("ts"), msg.get("text"))
    if key[0] is None:
        return True  # no timestamp, nothing to compare — process it
    with _seen_lock:
        if key in _seen:
            return False
        _seen.append(key)
        del _seen[:-200]
    return True


def handle_message(msg, process_fn):
    """Run one voice message once. A failure here is a failed reply, not an
    unreachable body — it must not count towards the circuit breaker."""
    if not first_time(msg):
        return
    try:
        process_fn(msg)
    except Exception as e:  # noqa: BLE001 - keep listening
        print(f"[brain] Voice processing failed for {msg.get('text', '')[:50]!r}: "
              f"{type(e).__name__}: {e}", flush=True)


def wait_for_push(process_fn, wait):
    """Handle pushed voice messages as they arrive, for up to `wait` seconds."""
    deadline = time.time() + wait
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            return
        try:
            msg = _push_queue.get(timeout=remaining)
        except queue.Empty:
            return
        handle_message(msg, process_fn)


class PushHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_POST(self):
        if self.path == "/voice/push":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode()) if length else {}
            _push_queue.put(body)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            # "processes": tells the bridge it need not also keep the message in
            # /voice/inbox — an older brain without it only queued pushes.
            self.wfile.write(b'{"ok":true,"processes":true}')
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

    has_api = LLM_CONFIGURED
    if has_api:
        print(f"[brain] LLM API available (model: {OPENAI_MODEL}, url: {OPENAI_URL})", flush=True)
        process_fn = process_voice_intelligent
    else:
        print("[brain] No LLM configured (set OPENAI_API_KEY, or OPENAI_URL for a local endpoint) — using simple fallback", flush=True)
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
                handle_message(msg, process_fn)
            
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
        
        wait_for_push(process_fn, POLL_INTERVAL)


if __name__ == "__main__":
    main()
