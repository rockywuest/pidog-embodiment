#!/usr/bin/env python3
"""
nox_voice_loop_v2.py — Improved voice listener for Nox's PiDog body.

Changes from v1:
- Posts recognized speech to local brain bridge (port 8888)
- Bridge handles routing to Nox's brain (Clawdbot on Pi 5)
- Better silence handling and wake word detection
- Conversation state tracking
- Sound direction awareness

Runs as systemd service alongside nox_daemon.py and nox_brain_bridge.py.
"""

import os
import sys
import json
import time
import socket
import subprocess
import threading
import urllib.request

os.environ["SDL_AUDIODRIVER"] = "alsa"
os.environ["AUDIODEV"] = "plughw:3,0"

# ─── Config ───
VOSK_MODEL_PATH = "/home/pidog/vosk-models/vosk-model-small-de-0.15"
BRIDGE_HOST = "localhost"
BRIDGE_PORT = 8888
DAEMON_HOST = "localhost"
DAEMON_PORT = 9999

SAMPLE_RATE = 16000
WAKE_WORDS = ["nox", "knox", "rocks", "hallo nox", "hey nox", "dog", "box"]
SILENCE_TIMEOUT = 2.0
MIN_PHRASE_LENGTH = 2
CONVERSATION_TIMEOUT = 30.0  # Stay in conversation mode for 30s after last interaction

# ─── State ───
class VoiceState:
    def __init__(self):
        self.in_conversation = False
        self.last_interaction = 0
        self.conversation_history = []
    
    def start_conversation(self):
        self.in_conversation = True
        self.last_interaction = time.time()
    
    def update(self):
        if self.in_conversation and time.time() - self.last_interaction > CONVERSATION_TIMEOUT:
            self.in_conversation = False
            self.conversation_history.clear()
    
    def add_exchange(self, user_text, response_text=""):
        self.conversation_history.append({
            "user": user_text,
            "response": response_text,
            "ts": time.time()
        })
        # Keep last 5 exchanges
        if len(self.conversation_history) > 5:
            self.conversation_history.pop(0)

state = VoiceState()


# ─── Helpers ───
def send_to_daemon(cmd_json):
    """Send command to nox_daemon via TCP."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(10)
        s.connect((DAEMON_HOST, DAEMON_PORT))
        s.sendall((json.dumps(cmd_json) + "\n").encode())
        resp = s.recv(4096).decode()
        s.close()
        return json.loads(resp) if resp else {}
    except Exception as e:
        print(f"[voice] daemon error: {e}", flush=True)
        return {"error": str(e)}


def set_rgb(r, g, b, mode="breath", bps=0.8):
    send_to_daemon({"cmd": "rgb", "r": r, "g": g, "b": b, "mode": mode, "bps": bps})


def speak_via_daemon(text):
    send_to_daemon({"cmd": "speak", "text": text})


def post_to_bridge(path, data):
    """Post data to the local brain bridge."""
    try:
        url = f"http://{BRIDGE_HOST}:{BRIDGE_PORT}{path}"
        body = json.dumps(data).encode()
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"[voice] bridge error: {e}", flush=True)
        return None


def strip_wake_word(text):
    """Remove wake word prefix if present. Returns (cleaned_text, was_wake_word)."""
    text_lower = text.lower().strip()
    for ww in WAKE_WORDS:
        if text_lower.startswith(ww):
            cleaned = text[len(ww):].strip()
            # Handle cases like "Nox, was siehst du?" → "was siehst du?"
            if cleaned.startswith(",") or cleaned.startswith("!") or cleaned.startswith("."):
                cleaned = cleaned[1:].strip()
            return cleaned, True
    return text, False


def is_just_noise(text):
    """Check if the recognized text is just noise/fragments."""
    noise_words = {"", "ja", "nee", "ähm", "hm", "hmm", "oh", "ah", "ach", "nun", "die", "der", "das", "und"}
    return text.lower().strip() in noise_words


def main():
    print("[voice-v2] Starting Nox voice loop v2...", flush=True)

    from vosk import Model, KaldiRecognizer

    print(f"[voice-v2] Loading Vosk model: {VOSK_MODEL_PATH}", flush=True)
    model = Model(VOSK_MODEL_PATH)
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    rec.SetWords(True)

    # Audio input
    print("[voice-v2] Starting audio capture...", flush=True)
    process = subprocess.Popen(
        ["arecord", "-D", "plughw:4,0", "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    # Signal we're listening
    set_rgb(0, 100, 50, "breath", 0.5)
    speak_via_daemon("Ich höre jetzt zu.")
    time.sleep(1)
    set_rgb(128, 0, 255, "breath", 0.8)

    print("[voice-v2] Listening...", flush=True)

    try:
        while True:
            data = process.stdout.read(4000)
            if len(data) == 0:
                break

            # Update conversation state
            state.update()

            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "").strip()

                if len(text) < MIN_PHRASE_LENGTH:
                    continue

                # Skip noise
                if is_just_noise(text):
                    print(f"[voice-v2] Noise: '{text}' (skipping)", flush=True)
                    continue

                print(f"[voice-v2] Heard: '{text}'", flush=True)

                # Check for wake word
                cleaned, had_wake_word = strip_wake_word(text)

                # If in conversation mode, process everything
                # If not in conversation, require wake word
                if not state.in_conversation and not had_wake_word:
                    print(f"[voice-v2] Not in conversation, no wake word. Ignoring.", flush=True)
                    continue

                # Activate conversation mode
                state.start_conversation()

                # If just the wake word with no content, acknowledge
                if had_wake_word and len(cleaned) < MIN_PHRASE_LENGTH:
                    send_to_daemon({"cmd": "move", "action": "wag_tail", "steps": 2, "speed": 80})
                    speak_via_daemon("Ja? Was ist?")
                    continue

                # We have actual content to process
                process_text = cleaned if had_wake_word else text

                # Visual feedback: thinking
                set_rgb(128, 0, 255, "speak", 2.0)
                print(f"[voice-v2] Processing: '{process_text}'", flush=True)

                # Post to bridge (which will be picked up by brain)
                post_to_bridge("/voice/input", {
                    "text": process_text,
                    "had_wake_word": had_wake_word,
                    "in_conversation": state.in_conversation,
                    "recent_context": [
                        ex["user"] for ex in state.conversation_history[-3:]
                    ]
                })

                # Add to conversation history
                state.add_exchange(process_text)

                # Don't speak acknowledgment - let the brain respond directly
                # Small head nod to show we heard
                send_to_daemon({"cmd": "move", "action": "nod_lethargy", "steps": 1, "speed": 90})

                # Back to listening mode after brief think color
                time.sleep(0.5)
                set_rgb(128, 0, 255, "breath", 0.8)

            else:
                # Partial result — visual feedback
                partial = json.loads(rec.PartialResult())
                partial_text = partial.get("partial", "")
                if partial_text and len(partial_text) > 3:
                    set_rgb(0, 200, 100, "listen", 2.0)

    except KeyboardInterrupt:
        pass
    finally:
        process.terminate()
        speak_via_daemon("Ich gehe schlafen. Gute Nacht!")
        set_rgb(0, 0, 80, "breath", 0.3)
        print("[voice-v2] Voice loop stopped.", flush=True)


if __name__ == "__main__":
    main()
