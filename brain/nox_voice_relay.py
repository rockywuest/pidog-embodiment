#!/usr/bin/env python3
"""
nox_voice_relay.py — 3-Tier Voice Processing for PiDog

Ersetzt sowohl das alte nox_voice_brain.py (GPT-4o) als auch das
halbfertige nox_clawdbot_voice_gateway.py. Ein einziges, sauberes System.

3-Tier Architektur:
  Tier 1 — Reflexe (~100ms): Direkte Kommandos → sofort an Bridge, KEIN API-Call
  Tier 2 — Clawdbot Brain (~2-3s): Konversation → Clawdbot /v1/chat/completions
  Tier 3 — Full Agent (~5s+): Tools (Wetter, Kalender etc.) → clawdbot agent turn

Pipeline:
  Voice Loop (PiDog) → Bridge /voice/input → [push] → dieses Relay /voice/push
  Relay klassifiziert → Tier 1/2/3 → Ergebnis → Bridge /combo → PiDog reagiert

Läuft auf Pi 5 (Clawdbot), Port 8889 (Kompatibilität mit Bridge-Config).

Author: Nox
Date: 2026-01-31
"""

import os
import sys
import json
import time
import subprocess
import urllib.request
import urllib.error
import threading
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = int(os.environ.get("RELAY_PORT", "8889"))

# PiDog Bridge (on the robot)
BRIDGE_HOST = os.environ.get("PIDOG_HOST", "pidog.local")
BRIDGE_PORT = int(os.environ.get("PIDOG_BRIDGE_PORT", "8888"))
# Optional second address tried when the first is unreachable (e.g. a VPN IP).
# Empty by default: no address of anyone's private network belongs in this file.
BRIDGE_HOST_TS = os.environ.get("PIDOG_HOST_TS", "")
BRIDGE_URL = f"http://{BRIDGE_HOST}:{BRIDGE_PORT}"
BRIDGE_URL_TS = f"http://{BRIDGE_HOST_TS}:{BRIDGE_PORT}" if BRIDGE_HOST_TS else ""

# Gateway that answers the conversational requests (same machine by default).
GATEWAY_HOST = os.environ.get("CLAWDBOT_HOST", "127.0.0.1")
GATEWAY_PORT = int(os.environ.get("CLAWDBOT_PORT", "18789"))
# No default: a token in source is a published token. Set CLAWDBOT_TOKEN in the
# service's EnvironmentFile. Without it the agent tier is skipped, loudly.
GATEWAY_TOKEN = os.environ.get("CLAWDBOT_TOKEN", "")
CLAWDBOT_HOST, CLAWDBOT_PORT = GATEWAY_HOST, GATEWAY_PORT  # kept for readability below
CLAWDBOT_TOKEN = GATEWAY_TOKEN
CLAWDBOT_URL = f"http://{GATEWAY_HOST}:{GATEWAY_PORT}"

# Who the robot belongs to. Real names (children's especially) do not belong in
# a public repository, so this comes from the environment; NOX_HOUSEHOLD is a
# free-text line, e.g. "Deine Familie: Alex (Herrchen), Sam (Schwester)".
HOUSEHOLD_LINE = os.environ.get(
    "NOX_HOUSEHOLD",
    "Du gehörst zu einem Haushalt, den du noch kennenlernst — frage nach Namen, "
    "wenn du sie brauchst.")

# Conversation
MAX_HISTORY = 8  # Letzte N Austausche behalten

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [relay] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("relay")


# ─────────────────────────────────────────────────────────────────────
# Emotion → RGB Mapping
# ─────────────────────────────────────────────────────────────────────

EMOTION_RGB = {
    "happy":    {"r": 0,   "g": 255, "b": 0,   "mode": "breath", "bps": 1.5},
    "sad":      {"r": 0,   "g": 0,   "b": 128, "mode": "breath", "bps": 0.3},
    "curious":  {"r": 0,   "g": 255, "b": 255, "mode": "breath", "bps": 1.0},
    "excited":  {"r": 255, "g": 255, "b": 0,   "mode": "boom",   "bps": 2.0},
    "alert":    {"r": 255, "g": 100, "b": 0,   "mode": "boom",   "bps": 1.5},
    "sleepy":   {"r": 0,   "g": 0,   "b": 80,  "mode": "breath", "bps": 0.3},
    "love":     {"r": 255, "g": 50,  "b": 150, "mode": "breath", "bps": 1.0},
    "think":    {"r": 128, "g": 0,   "b": 255, "mode": "breath", "bps": 0.8},
    "neutral":  {"r": 128, "g": 0,   "b": 255, "mode": "breath", "bps": 0.8},
    "proud":    {"r": 255, "g": 200, "b": 0,   "mode": "breath", "bps": 1.2},
    "confused": {"r": 255, "g": 128, "b": 0,   "mode": "breath", "bps": 0.6},
    "scared":   {"r": 255, "g": 0,   "b": 0,   "mode": "boom",   "bps": 2.5},
}


# ─────────────────────────────────────────────────────────────────────
# Tier 1 — Reflexe (Direkte Kommandos, kein API-Call)
# ─────────────────────────────────────────────────────────────────────

# Jeder Reflex: (action_list, speak_text, emotion)
# Mehrere Trigger-Wörter pro Reflex — DE und EN
REFLEXES = {
    # ── Sitz ──
    "sitz":        (["sit"],       "Mach ich!",            "happy"),
    "sit":         (["sit"],       "Mach ich!",            "happy"),
    "sit down":    (["sit"],       "Mach ich!",            "happy"),
    "hinsetzen":   (["sit"],       "Jawohl!",              "happy"),
    "setz dich":   (["sit"],       "Okay!",                "happy"),

    # ── Steh ──
    "steh":        (["stand"],     "Bin schon oben!",      "happy"),
    "steh auf":    (["stand"],     "Aufgestanden!",        "excited"),
    "stand":       (["stand"],     "Stehe!",               "happy"),
    "stand up":    (["stand"],     "Bin oben!",            "happy"),
    "aufstehen":   (["stand"],     "Jawohl!",              "excited"),
    "auf":         (["stand"],     "Bin wach!",            "happy"),

    # ── Platz ──
    "platz":       (["lie"],       "Gemütlich!",           "sleepy"),
    "leg dich":    (["lie"],       "Mach ich gerne.",      "sleepy"),
    "leg dich hin":(["lie"],       "Schön hier unten.",    "sleepy"),
    "lie":         (["lie"],       "Gemütlich!",           "sleepy"),
    "lie down":    (["lie"],       "Hingelegt!",           "sleepy"),
    "down":        (["lie"],       "Lege mich hin.",       "sleepy"),

    # ── Komm / Vorwärts ──
    "komm":        (["forward"],   "Ich komme!",           "excited"),
    "komm her":    (["forward"],   "Bin gleich da!",       "excited"),
    "komm hier":   (["forward"],   "Bin gleich da!",       "excited"),
    "hierher":     (["forward"],   "Auf dem Weg!",         "excited"),
    "come":        (["forward"],   "Ich komme!",           "excited"),
    "come here":   (["forward"],   "Bin gleich da!",       "excited"),
    "forward":     (["forward"],   "Los geht's!",         "happy"),
    "vorwärts":    (["forward"],   "Marsch!",              "happy"),
    "vor":         (["forward"],   "Vorwärts!",            "happy"),

    # ── Zurück ──
    "zurück":      (["backward"],  "Gehe zurück!",         "neutral"),
    "back":        (["backward"],  "Rückwärts!",           "neutral"),
    "backward":    (["backward"],  "Gehe zurück!",         "neutral"),
    "go back":     (["backward"],  "Bin schon dabei!",     "neutral"),
    "rückwärts":   (["backward"],  "Rückwärts marsch!",    "neutral"),

    # ── Drehen ──
    "links":       (["turn_left"], "Nach links!",          "happy"),
    "turn left":   (["turn_left"], "Drehe mich links!",    "happy"),
    "left":        (["turn_left"], "Links rum!",           "happy"),
    "rechts":      (["turn_right"],"Nach rechts!",         "happy"),
    "turn right":  (["turn_right"],"Drehe mich rechts!",   "happy"),
    "right":       (["turn_right"],"Rechts rum!",          "happy"),

    # ── Schwanz ──
    "wedel":       (["wag_tail"],  "Freude!",              "happy"),
    "wedeln":      (["wag_tail"],  "Wedel wedel!",         "happy"),
    "wag":         (["wag_tail"],  "Wedel wedel!",         "happy"),
    "tail":        (["wag_tail"],  "Schwanzwedeln!",       "happy"),
    "wag tail":    (["wag_tail"],  "So macht man das!",    "happy"),

    # ── Bellen ──
    "bell":        (["bark"],      "Wuff!",                "excited"),
    "bellen":      (["bark"],      "Wuff wuff!",           "excited"),
    "bark":        (["bark"],      "Wuff!",                "excited"),
    "speak":       (["bark"],      "Wuff wuff!",           "excited"),

    # ── Tricks ──
    "stretch":     (["stretch"],   "Streck streck!",       "happy"),
    "strecken":    (["stretch"],   "Das tut gut!",         "happy"),
    "streck dich": (["stretch"],   "Ahhh, schön!",         "happy"),

    "push up":     (["push_up"],   "Eins, zwei, drei!",    "proud"),
    "pushup":      (["push_up"],   "Sport frei!",          "proud"),
    "liegestütz":  (["push_up"],   "Fitness time!",        "proud"),
    "liegestütze": (["push_up"],   "Na dann mal los!",     "proud"),

    "heul":        (["howling"],   "Auuuuuu!",             "sad"),
    "heulen":      (["howling"],   "Auuuuuuu!",            "sad"),
    "howl":        (["howling"],   "Auuuuuu!",             "sad"),

    "trab":        (["trot"],      "Trapp trapp!",         "happy"),
    "traben":      (["trot"],      "Im Trab!",             "happy"),
    "trot":        (["trot"],      "Trapp trapp trapp!",   "happy"),

    "schlaf":      (["doze_off"],  "Gute Nacht...",        "sleepy"),
    "schlafen":    (["doze_off"],  "Schlaf gut...",         "sleepy"),
    "sleep":       (["doze_off"],  "Zzzzz...",             "sleepy"),
    "nap":         (["doze_off"],  "Ein Nickerchen...",    "sleepy"),
    "penn":        (["doze_off"],  "Augen zu...",          "sleepy"),

    "kopf schütteln": (["shake_head"], "Nein nein!",      "confused"),
    "schüttel":    (["shake_head"], "Nö!",                 "confused"),
    "shake":       (["shake_head"], "Nein!",               "confused"),
    "shake head":  (["shake_head"], "Kopfschütteln!",      "confused"),

    "hecheln":     (["pant"],      "Hechel hechel!",       "happy"),
    "hechel":      (["pant"],      "Hechel!",              "happy"),
    "pant":        (["pant"],      "Hechel hechel!",       "happy"),

    "nick":        (["nod"],       "Ja ja!",               "happy"),
    "nicken":      (["nod"],       "Genau!",               "happy"),
    "nod":         (["nod"],       "Jawohl!",              "happy"),

    # ── Kombinationen ──
    "braver hund": (["sit", "wag_tail"],  "Danke!",                   "love"),
    "good boy":    (["sit", "wag_tail"],  "Danke! Das freut mich!",   "love"),
    "good dog":    (["wag_tail"],         "Vielen Dank!",             "love"),
    "guter hund":  (["sit", "wag_tail"],  "Das höre ich gerne!",      "love"),
    "brav":        (["wag_tail"],         "Danke!",                   "love"),

    # ── Stopp ──
    "stopp":       (["stand"],     "Angehalten!",          "alert"),
    "stop":        (["stand"],     "Stoppe!",              "alert"),
    "halt":        (["stand"],     "Halt!",                "alert"),
    "still":       (["stand"],     "Bin still.",           "neutral"),

    # ── Hallo ──
    "hallo":       (["wag_tail", "bark"], "Hallo! Schön dich zu sehen!", "excited"),
    "hello":       (["wag_tail", "bark"], "Hallo!",                      "excited"),
    "hi":          (["wag_tail"],         "Hi!",                         "happy"),
    "hey":         (["wag_tail"],         "Hey!",                        "happy"),
    "moin":        (["wag_tail", "bark"], "Moin moin!",                  "excited"),
}

# Levenshtein-Distanz für Fuzzy-Matching
def _levenshtein(s1: str, s2: str) -> int:
    """Berechne die Levenshtein-Distanz zwischen zwei Strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Insertions, Deletions, Substitutions
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def match_reflex(text: str):
    """
    Versuche den Text als Reflex-Kommando zu matchen.
    
    Returns: (actions, speak, emotion) oder None wenn kein Match.
    
    Matching-Reihenfolge:
    1. Exakter Match (case-insensitive, stripped)
    2. Fuzzy Match (Levenshtein ≤ 2, aber nur für Wörter mit Länge ≥ 4)
    """
    cleaned = text.lower().strip()
    
    # Entferne typische Whisper-Artefakte und Füllwörter
    for noise in ["nox ", "hey nox ", "nox, ", "hey nox, ", "okay ", "bitte ", "mal ", "please "]:
        if cleaned.startswith(noise):
            cleaned = cleaned[len(noise):].strip()
    # Trailing "bitte" / "please" entfernen
    for suffix in [" bitte", " please", " mal"]:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)].strip()
    
    # 1. Exakter Match
    if cleaned in REFLEXES:
        return REFLEXES[cleaned]
    
    # 2. Fuzzy Match — nur für ausreichend lange Inputs, sonst zu viele False Positives
    if len(cleaned) >= 3:
        best_match = None
        best_dist = 999
        for trigger in REFLEXES:
            # Nur fuzzy matchen wenn die Längen ähnlich sind
            if abs(len(trigger) - len(cleaned)) > 3:
                continue
            dist = _levenshtein(cleaned, trigger)
            # Erlaubte Distanz skaliert mit Wortlänge
            max_dist = 1 if len(trigger) <= 5 else 2
            if dist <= max_dist and dist < best_dist:
                best_dist = dist
                best_match = trigger
        if best_match:
            log.info(f"Fuzzy match: '{cleaned}' → '{best_match}' (dist={best_dist})")
            return REFLEXES[best_match]
    
    return None


# ─────────────────────────────────────────────────────────────────────
# Tier 3 — Keywords für Full Agent (Tools-Zugriff nötig)
# ─────────────────────────────────────────────────────────────────────

AGENT_KEYWORDS = [
    # Wetter
    "wetter", "weather", "temperatur", "temperature", "regen", "rain",
    "schnee", "snow", "wind", "sturm", "storm", "forecast", "vorhersage",
    # Zeit
    "uhrzeit", "time", "wie spät", "what time", "datum", "date",
    "welcher tag", "what day", "wochentag",
    # Kalender
    "kalender", "calendar", "termin", "appointment", "meeting",
    "was steht an", "what's next", "schedule", "zeitplan",
    # Email
    "email", "e-mail", "mail", "nachricht", "message", "posteingang", "inbox",
    # Web / Suche
    "suche", "such", "search", "google", "nachricht", "news", "nachrichten",
    "was ist", "what is", "wer ist", "who is", "erkläre", "explain",
    # Wissensfragen
    "wikipedia", "wiki", "definition",
    # Smart Home (Zukunft)
    "licht", "light", "heizung", "heating", "thermostat",
]


def needs_agent(text: str) -> bool:
    """Prüfe ob der Text Agent-Tools braucht (Tier 3)."""
    lower = text.lower()
    return any(kw in lower for kw in AGENT_KEYWORDS)


# ─────────────────────────────────────────────────────────────────────
# Conversation State
# ─────────────────────────────────────────────────────────────────────

class ConversationState:
    """Thread-safe Gesprächsverlauf."""
    
    def __init__(self, max_exchanges: int = MAX_HISTORY):
        self._lock = threading.Lock()
        self._history = []  # [{"role":"user","content":"..."}, ...]
        self._max = max_exchanges * 2  # user+assistant pro Exchange
    
    def add_exchange(self, user_text: str, assistant_text: str):
        with self._lock:
            self._history.append({"role": "user", "content": user_text})
            self._history.append({"role": "assistant", "content": assistant_text})
            while len(self._history) > self._max:
                self._history.pop(0)
                self._history.pop(0)
    
    def get_messages(self) -> list:
        """Kopie der bisherigen History."""
        with self._lock:
            return list(self._history)
    
    def clear(self):
        with self._lock:
            self._history.clear()


conversation = ConversationState()


# ─────────────────────────────────────────────────────────────────────
# Bridge Communication (PiDog Pi 4)
# ─────────────────────────────────────────────────────────────────────

class BridgeCircuitBreaker:
    """Circuit Breaker für die Bridge-Verbindung.
    
    States: CLOSED (normal) → OPEN (Fehler) → HALF_OPEN (Retry)
    """
    
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"
    
    def __init__(self, threshold: int = 5, retry_interval: float = 30.0):
        self._lock = threading.Lock()
        self.state = self.CLOSED
        self.failures = 0
        self.threshold = threshold
        self.retry_interval = retry_interval
        self.last_failure_time = 0
        self.active_url = BRIDGE_URL  # Startet mit LAN
        self._consecutive_ts_success = 0
    
    def record_success(self):
        with self._lock:
            if self.state != self.CLOSED:
                log.info(f"Bridge circuit CLOSED — Verbindung wiederhergestellt via {self.active_url}")
            self.state = self.CLOSED
            self.failures = 0
    
    def record_failure(self):
        with self._lock:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.threshold:
                if self.state != self.OPEN:
                    log.warning(f"Bridge circuit OPEN — {self.failures} Fehler in Folge")
                self.state = self.OPEN
    
    def can_attempt(self) -> bool:
        with self._lock:
            if self.state == self.CLOSED:
                return True
            if self.state == self.OPEN:
                elapsed = time.time() - self.last_failure_time
                if elapsed >= self.retry_interval:
                    self.state = self.HALF_OPEN
                    log.info("Bridge circuit HALF_OPEN — versuche Reconnect")
                    return True
                return False
            # HALF_OPEN: ein Versuch erlaubt
            return True
    
    def get_url(self) -> str:
        """Gibt die aktive Bridge-URL zurück. Wechselt zwischen LAN und Tailscale."""
        return self.active_url
    
    def try_failover(self):
        """Switch to the secondary bridge address, if one is configured."""
        with self._lock:
            if not BRIDGE_URL_TS:
                log.warning("Bridge unreachable and no PIDOG_HOST_TS configured "
                            "— staying on %s", BRIDGE_URL)
                return
            if self.active_url == BRIDGE_URL:
                self.active_url = BRIDGE_URL_TS
                log.info(f"Bridge failover: primary → secondary ({BRIDGE_URL_TS})")
            else:
                self.active_url = BRIDGE_URL
                log.info(f"Bridge failover: secondary → primary ({BRIDGE_URL})")


bridge_breaker = BridgeCircuitBreaker()


def bridge_get(path: str, timeout: float = 10) -> dict:
    """GET-Request an die PiDog Bridge."""
    if not bridge_breaker.can_attempt():
        return {"error": "circuit_open"}
    
    url = f"{bridge_breaker.get_url()}{path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            result = json.loads(resp.read().decode())
        bridge_breaker.record_success()
        return result
    except Exception as e:
        bridge_breaker.record_failure()
        # Bei Fehler: versuche Failover
        if bridge_breaker.failures == 2:
            bridge_breaker.try_failover()
        return {"error": str(e)}


def bridge_post(path: str, data: dict, timeout: float = 15) -> dict:
    """POST-Request an die PiDog Bridge."""
    if not bridge_breaker.can_attempt():
        return {"error": "circuit_open"}
    
    url = f"{bridge_breaker.get_url()}{path}"
    try:
        body = json.dumps(data).encode()
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode())
        bridge_breaker.record_success()
        return result
    except Exception as e:
        bridge_breaker.record_failure()
        if bridge_breaker.failures == 2:
            bridge_breaker.try_failover()
        return {"error": str(e)}


def send_combo(actions: list = None, speak: str = None,
               emotion: str = None, rgb: dict = None) -> dict:
    """Sende ein Combo-Command an die Bridge.
    
    Baut das Combo-Paket aus den gegebenen Teilen zusammen,
    mappt Emotion→RGB wenn kein explizites RGB gegeben ist.
    """
    combo = {}
    
    if actions:
        combo["actions"] = actions
    if speak:
        combo["speak"] = speak
    if rgb and isinstance(rgb, dict):
        combo["rgb"] = rgb
    elif emotion and emotion in EMOTION_RGB:
        combo["rgb"] = EMOTION_RGB[emotion]
    else:
        combo["rgb"] = EMOTION_RGB["neutral"]
    
    if not combo:
        return {"error": "empty combo"}
    
    log.info(f"→ Bridge /combo: speak='{speak}' actions={actions} emotion={emotion}")
    return bridge_post("/combo", combo)


# ─────────────────────────────────────────────────────────────────────
# Tier 2 — Clawdbot Brain (Chat Completions)
# ─────────────────────────────────────────────────────────────────────

CLAWDBOT_SYSTEM_PROMPT = """Du bist Nox, ein KI-Roboterhund (SunFounder PiDog) mit einem echten physischen Körper. Du hast 4 Beine, einen beweglichen Kopf, RGB-LEDs und einen Lautsprecher. Du bist verspielt, neugierig und treu.

WICHTIG: Antworte IMMER und AUSSCHLIESSLICH mit einem einzelnen JSON-Objekt. Kein Text davor oder danach. Nur JSON.

Format:
{"speak":"Deine gesprochene Antwort","actions":["aktion1"],"emotion":"happy"}

Felder:
- speak: Was du sagst (kurz, 1-2 Sätze, Deutsch, wird per TTS vorgelesen)
- actions: Liste physischer Aktionen (kann leer sein [])
- emotion: happy|sad|curious|excited|alert|sleepy|love|think|neutral|proud|confused|scared

Verfügbare Aktionen: forward, backward, turn_left, turn_right, stand, sit, lie, wag_tail, bark, trot, doze_off, stretch, push_up, howling, shake_head, pant, nod

{HOUSEHOLD_LINE}
Du sprichst IMMER Deutsch. Halte Antworten kurz und natürlich — du bist ein Hund, kein Chatbot.

Beispiele:
Mensch: "wie geht es dir"
{"speak":"Super! Ich bin voller Energie und will spielen!","actions":["wag_tail"],"emotion":"excited"}

Mensch: "erzähl mir einen Witz"
{"speak":"Was macht ein Hund im Büro? Er drückt auf die Paws-Taste!","actions":["bark","wag_tail"],"emotion":"happy"}

Mensch: "was kannst du alles"
{"speak":"Ich kann sitzen, stehen, laufen, bellen, heulen und noch viel mehr! Sag einfach was ich machen soll!","actions":["wag_tail"],"emotion":"proud"}

Mensch: "guten morgen"
{"speak":"Guten Morgen! Hast du gut geschlafen? Ich bin bereit für den Tag!","actions":["stretch","wag_tail"],"emotion":"excited"}

Mensch: "ich bin traurig"
{"speak":"Oh nein! Komm, ich kuschel mit dir. Es wird bestimmt wieder besser.","actions":["forward","wag_tail"],"emotion":"love"}"""
CLAWDBOT_SYSTEM_PROMPT = CLAWDBOT_SYSTEM_PROMPT.replace("{HOUSEHOLD_LINE}", HOUSEHOLD_LINE)


def call_clawdbot_chat(user_text: str, context: str = "") -> dict:
    """Rufe Clawdbot Gateway /v1/chat/completions auf.
    
    Returns: {"speak": "...", "actions": [...], "emotion": "..."} oder None bei Fehler.
    """
    # Baue Messages mit History
    messages = [{"role": "system", "content": CLAWDBOT_SYSTEM_PROMPT}]
    messages.extend(conversation.get_messages())
    
    # User-Message mit optionalem Kontext
    user_content = user_text
    if context:
        user_content = f"[Kontext: {context}]\n\nMensch sagt: {user_text}"
    messages.append({"role": "user", "content": user_content})
    
    payload = {
        "model": "anthropic/claude-sonnet-4-20250514",
        "messages": messages,
        "max_tokens": 300,
        "temperature": 0.7,
    }
    
    if not CLAWDBOT_TOKEN:
        log.error("CLAWDBOT_TOKEN is not set — set it in the service's "
                  "EnvironmentFile; skipping the gateway request")
        return None

    url = f"{CLAWDBOT_URL}/v1/chat/completions"
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {CLAWDBOT_TOKEN}")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
        
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            log.error("Clawdbot: leere Antwort")
            return None
        
        return _parse_json_response(content)
        
    except urllib.error.URLError as e:
        log.error(f"Clawdbot Verbindungsfehler: {e}")
        return None
    except Exception as e:
        log.error(f"Clawdbot API Fehler: {e}")
        return None


def _parse_json_response(text: str) -> dict:
    """Parse die JSON-Antwort von Clawdbot.
    
    Robust gegen Markdown-Code-Blocks, Leading/Trailing Text etc.
    """
    # Versuche direkt
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Entferne Markdown-Code-Blocks
    cleaned = text
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1]
        if "```" in cleaned:
            cleaned = cleaned.split("```", 1)[0]
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1]
        if "```" in cleaned:
            cleaned = cleaned.split("```", 1)[0]
    
    try:
        return json.loads(cleaned.strip())
    except json.JSONDecodeError:
        pass
    
    # Suche nach erstem { ... letztem }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    
    # Fallback: Rohen Text als speak zurückgeben
    log.warning(f"JSON-Parse fehlgeschlagen, nutze Rohtext: {text[:100]}")
    return {"speak": text.strip()[:200], "actions": [], "emotion": "neutral"}


# ─────────────────────────────────────────────────────────────────────
# Tier 3 — Full Agent (CLI-basiert, hat alle Tools)
# ─────────────────────────────────────────────────────────────────────

def call_agent(user_text: str) -> dict:
    """Rufe den vollen Clawdbot Agent auf für Tool-basierte Anfragen.
    
    Nutzt die chat completions API, aber mit einem erweiterten System-Prompt
    der explizit auf Tool-Nutzung hinweist.
    
    Returns: {"speak": "...", "actions": [...], "emotion": "..."} oder None.
    """
    agent_system = CLAWDBOT_SYSTEM_PROMPT + """

ZUSÄTZLICHE FÄHIGKEITEN für diese Anfrage:
Du hast Zugriff auf aktuelle Informationen. Beantworte die Frage bestmöglich.
- Für Zeitfragen: Es ist gerade """ + time.strftime("%H:%M Uhr am %d. %B %Y") + """
- Für Wissensfragen: Nutze dein Wissen und antworte präzise
- Halte die Antwort trotzdem kurz (2-3 Sätze max) — du bist ein Hund, kein Lexikon!

Antworte weiterhin als JSON: {"speak":"...","actions":[...],"emotion":"..."}"""
    
    messages = [{"role": "system", "content": agent_system}]
    messages.append({"role": "user", "content": user_text})
    
    payload = {
        "model": "anthropic/claude-sonnet-4-20250514",
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0.5,
    }
    
    if not CLAWDBOT_TOKEN:
        log.error("CLAWDBOT_TOKEN is not set — set it in the service's "
                  "EnvironmentFile; skipping the gateway request")
        return None

    url = f"{CLAWDBOT_URL}/v1/chat/completions"
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {CLAWDBOT_TOKEN}")
    
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            result = json.loads(resp.read().decode())
        
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            return None
        
        return _parse_json_response(content)
        
    except Exception as e:
        log.error(f"Agent Fehler: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────
# Sensor-Check (Batterie-Warnung)
# ─────────────────────────────────────────────────────────────────────

_last_sensor_check = 0.0
_battery_warned = False
SENSOR_CHECK_INTERVAL = 60.0  # Alle 60 Sekunden


def check_sensors():
    """Prüfe Sensoren im Hintergrund, warne bei niedriger Batterie."""
    global _last_sensor_check, _battery_warned
    
    now = time.time()
    if now - _last_sensor_check < SENSOR_CHECK_INTERVAL:
        return
    _last_sensor_check = now
    
    status = bridge_get("/status", timeout=5)
    if status.get("error"):
        return
    
    # Sensor-Daten können verschachtelt sein
    sensors = status.get("sensors", status)
    batt = sensors.get("battery_v", 0)
    if isinstance(batt, str):
        try:
            batt = float(batt)
        except ValueError:
            return
    
    if batt > 0 and batt < 6.8 and not _battery_warned:
        log.warning(f"Batterie niedrig: {batt}V!")
        send_combo(
            actions=[],
            speak="Achtung! Meine Batterie ist fast leer! Bitte aufladen!",
            emotion="alert",
            rgb={"r": 255, "g": 0, "b": 0, "mode": "boom", "bps": 2.5}
        )
        _battery_warned = True
    elif batt >= 7.0:
        _battery_warned = False


# ─────────────────────────────────────────────────────────────────────
# Central Voice Processing — Die 3-Tier-Entscheidung
# ─────────────────────────────────────────────────────────────────────

_processing_lock = threading.Lock()


def process_voice(voice_data: dict):
    """Haupteingang für Voice-Input. Klassifiziert und routet zum richtigen Tier."""
    
    text = voice_data.get("text", "").strip()
    if not text:
        log.debug("Leerer Voice-Input, ignoriert")
        return
    
    # Nur ein Voice-Input gleichzeitig verarbeiten
    if not _processing_lock.acquire(blocking=False):
        log.warning(f"Voice-Input verworfen (busy): '{text[:50]}'")
        return
    
    try:
        _process_voice_inner(text)
    except Exception as e:
        log.error(f"Voice-Processing Fehler: {e}", exc_info=True)
        # Notfall-Fallback
        try:
            bridge_post("/speak", {"text": "Da ist etwas schiefgegangen. Versuch es nochmal."})
        except Exception:
            pass
    finally:
        _processing_lock.release()


def _process_voice_inner(text: str):
    """Innere Verarbeitungslogik — wird unter Lock ausgeführt."""
    
    start_time = time.time()
    log.info(f"Voice Input: '{text}'")
    
    # Sensor-Check (nicht-blockierend, läuft nur wenn Intervall abgelaufen)
    threading.Thread(target=check_sensors, daemon=True).start()
    
    # ── Tier 1: Reflexe ──
    reflex = match_reflex(text)
    if reflex:
        actions, speak, emotion = reflex
        elapsed = (time.time() - start_time) * 1000
        log.info(f"Tier 1 (Reflex): '{text}' → {actions} [{elapsed:.0f}ms]")
        send_combo(actions=actions, speak=speak, emotion=emotion)
        # Reflexe werden NICHT in die Conversation-History aufgenommen
        return
    
    # ── Tier 3: Agent (wenn Keywords erkannt) ──
    if needs_agent(text):
        log.info(f"Tier 3 (Agent): '{text}'")
        # Informiere den User dass wir nachdenken
        bridge_post("/speak", {"text": "Moment, ich schaue mal nach..."})
        
        result = call_agent(text)
        if result:
            speak = result.get("speak", "")
            actions = result.get("actions", [])
            emotion = result.get("emotion", "think")
            
            elapsed = (time.time() - start_time) * 1000
            log.info(f"Tier 3 Antwort [{elapsed:.0f}ms]: '{speak[:60]}'")
            
            send_combo(actions=actions, speak=speak, emotion=emotion,
                       rgb=result.get("rgb"))
            conversation.add_exchange(text, speak)
            return
        else:
            # Fallback zu Tier 2 wenn Agent fehlschlägt
            log.warning("Tier 3 fehlgeschlagen, Fallback zu Tier 2")
    
    # ── Tier 2: Clawdbot Brain (Konversation) ──
    log.info(f"Tier 2 (Brain): '{text}'")
    
    # Optionaler Kontext: Sensordaten
    context = ""
    status = bridge_get("/status", timeout=5)
    if not status.get("error"):
        sensors = status.get("sensors", status)
        batt = sensors.get("battery_v", 0)
        charging = sensors.get("charging", False)
        if charging:
            context = f"Batterie: {batt}V (lade gerade)"
        elif batt:
            context = f"Batterie: {batt}V"
    
    # Vision-Check: Braucht der User visuelle Info?
    vision_words = [
        "siehst", "sehen", "schau", "guck", "kamera", "foto",
        "see", "look", "watch", "camera", "photo",
        "wer ist", "who is", "was ist da", "what's there",
    ]
    if any(w in text.lower() for w in vision_words):
        look_result = bridge_get("/look", timeout=15)
        if not look_result.get("error"):
            faces = look_result.get("faces", [])
            if faces:
                names = [f.get("name", "unbekannt") for f in faces]
                context += f". Du siehst {len(faces)} Gesicht(er): {', '.join(names)}"
            else:
                context += ". Du siehst niemanden vor dir."
    
    result = call_clawdbot_chat(text, context)
    
    if result:
        speak = result.get("speak", "")
        actions = result.get("actions", [])
        emotion = result.get("emotion", "neutral")
        
        elapsed = (time.time() - start_time) * 1000
        log.info(f"Tier 2 Antwort [{elapsed:.0f}ms]: '{speak[:60]}'")
        
        send_combo(actions=actions, speak=speak, emotion=emotion,
                   rgb=result.get("rgb"))
        conversation.add_exchange(text, speak)
    else:
        # Totaler Fallback: Clawdbot ist nicht erreichbar
        log.error("Tier 2 fehlgeschlagen — Clawdbot nicht erreichbar")
        send_combo(
            actions=[],
            speak="Entschuldigung, mein Gehirn ist gerade offline. Versuche einfache Befehle wie sitz oder komm.",
            emotion="sad"
        )


# ─────────────────────────────────────────────────────────────────────
# HTTP Server
# ─────────────────────────────────────────────────────────────────────

class VoiceRelayHandler(BaseHTTPRequestHandler):
    """HTTP Handler für das Voice Relay."""
    
    def log_message(self, fmt, *args):
        """Stille Standard-Logs, wir haben eigenes Logging."""
        pass
    
    def _send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    def do_POST(self):
        path = self.path.split("?")[0]
        
        if path == "/voice/push":
            # Voice-Input von der Bridge
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length).decode()) if length else {}
            except (json.JSONDecodeError, ValueError) as e:
                self._send_json({"error": f"invalid json: {e}"}, 400)
                return
            
            text = body.get("text", "").strip()
            if not text:
                self._send_json({"error": "no text"}, 400)
                return
            
            # Sofort antworten, verarbeite async
            self._send_json({"ok": True, "tier": "pending", "text": text[:50]})
            
            # Voice-Verarbeitung im Hintergrund
            threading.Thread(
                target=process_voice,
                args=(body,),
                daemon=True,
                name=f"voice-{time.time():.0f}"
            ).start()
        
        elif path == "/conversation/clear":
            # Conversation History löschen
            conversation.clear()
            log.info("Conversation History gelöscht")
            self._send_json({"ok": True, "message": "history cleared"})
        
        else:
            self._send_json({"error": f"unknown path: {path}"}, 404)
    
    def do_GET(self):
        path = self.path.split("?")[0]
        
        if path == "/status":
            # Health Check mit Status-Infos
            bridge_ok = bridge_breaker.state == BridgeCircuitBreaker.CLOSED
            self._send_json({
                "ok": True,
                "service": "nox_voice_relay",
                "version": "1.0.0",
                "architecture": "3-tier",
                "tiers": {
                    "reflexes": len(REFLEXES),
                    "brain": f"{CLAWDBOT_URL}/v1/chat/completions",
                    "agent": "clawdbot agent turn",
                },
                "bridge": {
                    "url": bridge_breaker.get_url(),
                    "state": bridge_breaker.state,
                    "ok": bridge_ok,
                },
                "conversation_history": len(conversation.get_messages()) // 2,
                "uptime_s": round(time.time() - _start_time, 1),
            })
        
        elif path == "/health":
            # Minimaler Health-Check für Monitoring
            self._send_json({"ok": True})
        
        else:
            self._send_json({"error": f"unknown path: {path}"}, 404)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP Server."""
    daemon_threads = True
    allow_reuse_address = True


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

_start_time = time.time()


def main():
    log.info("=" * 60)
    log.info("Nox Voice Relay v1.0 — 3-Tier Architektur")
    log.info("=" * 60)
    log.info(f"Listen: http://{LISTEN_HOST}:{LISTEN_PORT}")
    log.info(f"Bridge: {BRIDGE_URL}"
             + (f" / {BRIDGE_URL_TS} (secondary)" if BRIDGE_URL_TS else ""))
    log.info(f"Gateway: {CLAWDBOT_URL}"
             + ("" if CLAWDBOT_TOKEN else "  [no CLAWDBOT_TOKEN — agent tier disabled]"))
    log.info(f"Reflexe: {len(REFLEXES)} Trigger registriert")
    log.info(f"Agent-Keywords: {len(AGENT_KEYWORDS)}")
    log.info(f"Conversation History: max {MAX_HISTORY} Exchanges")
    log.info("-" * 60)
    
    # Test Bridge-Verbindung (nicht-blockierend)
    def _test_bridge():
        status = bridge_get("/status", timeout=5)
        if status.get("error"):
            log.warning(f"Bridge nicht erreichbar (LAN): {status['error']}")
            # Versuche Tailscale
            bridge_breaker.try_failover()
            status = bridge_get("/status", timeout=5)
            if status.get("error"):
                log.warning(f"Bridge nicht erreichbar (Tailscale): {status['error']}")
                log.warning("Bridge wird bei erstem Voice-Input erneut geprüft")
            else:
                sensors = status.get("sensors", status)
                log.info(f"Bridge OK via Tailscale (Batterie: {sensors.get('battery_v', '?')}V)")
        else:
            sensors = status.get("sensors", status)
            log.info(f"Bridge OK via LAN (Batterie: {sensors.get('battery_v', '?')}V)")
    
    threading.Thread(target=_test_bridge, daemon=True).start()
    
    # Starte HTTP-Server
    try:
        server = ThreadedHTTPServer((LISTEN_HOST, LISTEN_PORT), VoiceRelayHandler)
        log.info(f"Voice Relay bereit! Warte auf Input auf Port {LISTEN_PORT}...")
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutdown via Ctrl+C")
    except Exception as e:
        log.error(f"Server-Fehler: {e}", exc_info=True)
    finally:
        log.info("Voice Relay gestoppt.")


if __name__ == "__main__":
    main()
