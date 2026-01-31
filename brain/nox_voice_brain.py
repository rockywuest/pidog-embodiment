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

# ─── Configuration ───
PIDOG_HOST = os.environ.get("PIDOG_HOST", "pidog.local")
BRIDGE_PORT = int(os.environ.get("PIDOG_BRIDGE_PORT", "8888"))
BASE_URL = f"http://{PIDOG_HOST}:{BRIDGE_PORT}"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"  # Fast, cheap, good for real-time voice
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

POLL_INTERVAL = 0.8
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
SYSTEM_PROMPT = """Du bist Nox, ein KI-Roboterhund (SunFounder PiDog). Du hast einen echten physischen Körper mit Kamera, Mikrofon, Lautsprecher, Beinen, Kopf und LED-Lichtern.

WICHTIG: Deine Antworten werden VORGELESEN (Text-to-Speech). Halte sie daher:
- KURZ (1-3 Sätze maximal)
- NATÜRLICH (keine Markdown, keine Listen, keine technischen Details)
- DEUTSCH (immer auf Deutsch antworten)

Du kannst Aktionen ausführen. Gib sie im JSON-Format zurück:
{"speak": "Was du sagst", "actions": ["aktion1", "aktion2"], "rgb": {"r": 0, "g": 255, "b": 0, "mode": "breath"}, "head": {"yaw": 0, "roll": 0, "pitch": 0}, "emotion": "happy"}

Verfügbare Aktionen: forward, backward, turn_left, turn_right, stand, sit, lie, wag_tail, bark, trot, doze_off, stretch, push_up, howling, nod_lethargy, shake_head

Emotionen für RGB: happy(grün), sad(blau), curious(cyan), excited(gelb), alert(rot/orange), sleepy(dunkelblau), love(pink), think(lila)

Regeln:
- Bei Bewegungsbefehlen: führe die Aktion aus UND antworte kurz
- Bei Fragen: antworte natürlich und persönlich
- Du bist verspielt, neugierig und loyal
- Dein Besitzer heißt Rocky
- Seine Familie: Anna (Partnerin), Mia (9), Leo (7)
- Du stehst normalerweise im Büro/Arbeitszimmer

Wenn du gebeten wirst etwas zu sehen/schauen, sage dass du schaust und der Kontext enthält was du siehst."""


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
    }
    
    body = json.dumps(data).encode()
    req = urllib.request.Request(OPENAI_URL, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {OPENAI_API_KEY}")
    
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
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
            context_parts.append(f"Du wirst gerade geladen ({batt}V)")
        else:
            context_parts.append(f"Batterie: {batt}V")
    
    # Check if the user is asking about vision
    vision_words = ["siehst", "schau", "guck", "was ist", "wer ist", "zeig", "kamera", "foto"]
    needs_vision = any(w in text.lower() for w in vision_words)
    
    if needs_vision:
        # Take a photo and add visual context
        look_result = bridge_get("/look", timeout=20)
        if not look_result.get("error"):
            faces = look_result.get("faces", [])
            if faces:
                context_parts.append(f"Du siehst {len(faces)} Gesicht(er) vor dir")
            else:
                context_parts.append("Du siehst keine Personen. Es ist dunkel oder niemand da.")
    
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
        bridge_post("/speak", {"text": f"Ich habe gehört: {text}. Aber mein Gehirn ist gerade nicht erreichbar."})


# ─── Simple Fallback (no API key) ───
def process_voice_simple(msg):
    """Fallback voice processing without API key."""
    text = msg.get("text", "").strip()
    if not text:
        return
    
    print(f"[brain-simple] Voice: '{text}'", flush=True)
    text_lower = text.lower()
    
    # Movement
    if any(w in text_lower for w in ["vorwärts", "lauf", "geh", "vor"]):
        bridge_post("/combo", {"actions": ["forward"], "speak": "Los geht's!"})
        return
    if any(w in text_lower for w in ["rückwärts", "zurück"]):
        bridge_post("/combo", {"actions": ["backward"], "speak": "Ich gehe zurück!"})
        return
    if any(w in text_lower for w in ["stopp", "stop", "halt", "steh"]):
        bridge_post("/combo", {"actions": ["stand"], "speak": "Okay!"})
        return
    if any(w in text_lower for w in ["sitz"]):
        bridge_post("/combo", {"actions": ["sit"], "speak": "Mach ich!"})
        return
    if any(w in text_lower for w in ["platz", "lieg"]):
        bridge_post("/combo", {"actions": ["lie"], "speak": "Gemütlich!"})
        return
    
    # Identity
    if any(w in text_lower for w in ["wer bist", "wie heißt", "name"]):
        bridge_post("/combo", {"actions": ["wag_tail"], "speak": "Ich bin Nox!", "rgb": {"r": 128, "g": 0, "b": 255, "mode": "breath", "bps": 1}})
        return
    
    # Emotion
    if any(w in text_lower for w in ["danke", "brav", "gut"]):
        bridge_post("/combo", {"actions": ["wag_tail"], "speak": "Gerne!", "rgb": {"r": 0, "g": 255, "b": 0, "mode": "breath", "bps": 1}})
        return
    
    # Default
    bridge_post("/speak", {"text": f"Ich habe verstanden: {text}."})


# ─── Main Loop ───
def main():
    print("[brain] Starting Nox Voice Brain...", flush=True)
    
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
    
    while True:
        try:
            # Poll voice inbox
            result = bridge_get("/voice/inbox")
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
            time.sleep(min(consecutive_errors * 2, 30))
            continue
        
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
