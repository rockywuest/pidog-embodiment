#!/usr/bin/env python3
"""
nox_voice_loop.py — Continuous voice listener for Nox's PiDog body.
Listens via Vosk (offline, German), sends recognized text to Nox's brain
(Clawdbot Gateway on Pi), receives response, speaks it via Piper TTS.

Runs alongside nox_daemon.py as a separate service.
"""

import os
import sys
import json
import time
import queue
import socket
import subprocess
import threading
import urllib.request

os.environ["SDL_AUDIODRIVER"] = "alsa"
os.environ["AUDIODEV"] = "plughw:3,0"

# ─── Config ───
VOSK_MODEL_PATH = os.environ.get(
    "VOSK_MODEL_PATH",
    os.path.expanduser("~/vosk-models/vosk-model-small-de-0.15"),
)
PIPER_BIN = os.path.expanduser("~/.local/bin/piper")
PIPER_MODEL = os.path.expanduser("~/.local/share/piper-voices/de_DE-thorsten-high.onnx")
NOX_DAEMON_HOST = "localhost"
NOX_DAEMON_PORT = 9999
CLAWDBOT_HOST = os.environ.get("CLAWDBOT_HOST", "192.168.1.18")
CLAWDBOT_PORT = 18789

SAMPLE_RATE = 16000
WAKE_WORDS = ["nox", "knox", "rocks", "hallo nox", "hey nox", "dog"]
SILENCE_TIMEOUT = 2.0  # seconds of silence = end of utterance
MIN_PHRASE_LENGTH = 2  # minimum chars to process

# ─── Nox Daemon helper ───
def send_to_daemon(cmd_json):
    """Send command to nox_daemon via TCP."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(10)
        s.connect((NOX_DAEMON_HOST, NOX_DAEMON_PORT))
        s.sendall((json.dumps(cmd_json) + "\n").encode())
        resp = s.recv(4096).decode()
        s.close()
        return json.loads(resp) if resp else {}
    except Exception as e:
        print(f"[voice] daemon error: {e}", flush=True)
        return {"error": str(e)}


def speak(text):
    """Speak text via Piper TTS."""
    send_to_daemon({"cmd": "speak", "text": text})


def set_rgb(r, g, b, mode="breath", bps=0.8):
    """Set LEDs via daemon."""
    send_to_daemon({"cmd": "rgb", "r": r, "g": g, "b": b, "mode": mode, "bps": bps})


def send_to_clawdbot(text):
    """Send message to Clawdbot Gateway and get response.
    Uses the Gateway HTTP API to inject a message into the main session."""
    try:
        # Use the gateway's RPC endpoint
        url = f"http://{CLAWDBOT_HOST}:{CLAWDBOT_PORT}/api/v1/rpc"
        payload = {
            "method": "pidog.voice",
            "params": {"text": text, "source": "pidog_voice"}
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        req.timeout = 30
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            return result.get("result", {}).get("reply", "")
    except Exception as e:
        print(f"[voice] clawdbot error: {e}", flush=True)
        # Fallback: write to a file that Nox can poll
        msg_file = "/tmp/nox_voice_inbox.json"
        msg = {"text": text, "ts": time.time()}
        with open(msg_file, "w") as f:
            json.dump(msg, f)
        return None


def main():
    print("[voice] Starting Nox voice loop...", flush=True)

    # Import audio libs
    try:
        import sounddevice as sd
    except ImportError:
        print("[voice] sounddevice not installed, trying pyaudio...", flush=True)
        sd = None

    from vosk import Model, KaldiRecognizer

    print(f"[voice] Loading Vosk model: {VOSK_MODEL_PATH}", flush=True)
    model = Model(VOSK_MODEL_PATH)
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    rec.SetWords(True)

    # Audio input via subprocess (most reliable on Pi)
    print("[voice] Starting audio capture...", flush=True)
    process = subprocess.Popen(
        ["arecord", "-D", "plughw:4,0", "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    # Signal we're listening
    set_rgb(0, 100, 50, "breath", 0.5)
    speak("Ich höre jetzt zu.")
    set_rgb(128, 0, 255, "breath", 0.8)

    print("[voice] Listening... say something!", flush=True)

    last_speech_time = 0
    current_text = ""
    waiting_for_response = False

    try:
        while True:
            data = process.stdout.read(4000)
            if len(data) == 0:
                break

            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "").strip()

                if len(text) < MIN_PHRASE_LENGTH:
                    continue

                print(f"[voice] Heard: '{text}'", flush=True)

                # Check for wake word or just process everything
                text_lower = text.lower()

                # Remove wake word prefix if present
                for ww in WAKE_WORDS:
                    if text_lower.startswith(ww):
                        text = text[len(ww):].strip()
                        break

                if len(text) < MIN_PHRASE_LENGTH:
                    # Just wake word, acknowledge
                    send_to_daemon({"cmd": "move", "action": "wag_tail", "steps": 2, "speed": 80})
                    speak("Ja?")
                    continue

                # Show we're thinking
                set_rgb(0, 100, 255, "speak", 2.0)
                print(f"[voice] Processing: '{text}'", flush=True)

                # Write to inbox file for Nox to pick up
                msg_file = "/tmp/nox_voice_inbox.json"
                msg = {"text": text, "ts": time.time(), "processed": False}
                with open(msg_file, "w") as f:
                    json.dump(msg, f)

                # Try direct gateway call
                response = send_to_clawdbot(text)

                if response:
                    print(f"[voice] Response: '{response}'", flush=True)
                    set_rgb(128, 0, 255, "speak", 1.5)
                    speak(response)
                else:
                    # No direct response, Nox will pick up from inbox
                    speak("Moment, ich denke nach...")

                set_rgb(128, 0, 255, "breath", 0.8)

            else:
                # Partial result
                partial = json.loads(rec.PartialResult())
                partial_text = partial.get("partial", "")
                if partial_text and len(partial_text) > 3:
                    # Visual feedback that we're hearing something
                    set_rgb(0, 200, 100, "listen", 2.0)

    except KeyboardInterrupt:
        pass
    finally:
        process.terminate()
        speak("Ich gehe schlafen. Gute Nacht!")
        set_rgb(0, 0, 80, "breath", 0.3)
        print("[voice] Voice loop stopped.", flush=True)


if __name__ == "__main__":
    main()
