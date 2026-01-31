#!/usr/bin/env python3
"""
nox_voice_loop_v2.py — Improved voice listener for Nox's PiDog body.

Changes from v1:
- Posts recognized speech to local brain bridge (port 8888)
- Bridge handles routing to Nox's brain (Clawdbot on Pi 5)
- Better silence handling and wake word detection
- Conversation state tracking
- Sound direction awareness
- v2.1: Fuzzy wake word matching (Vosk small model is bad at "Nox")
- v2.2: Energy gate to filter phantom recognition from ambient noise
- v2.3: German Vosk model + stricter wake words + disabled phonetic catch-all

Runs as systemd service alongside nox_daemon.py and nox_brain_bridge.py.
"""

import os
import sys
import json
import time
import struct
import math
import socket
import subprocess
import threading
import urllib.request

os.environ["SDL_AUDIODRIVER"] = "alsa"

# ─── Config ───
VOSK_MODEL_PATH = "/home/pidog/vosk-models/vosk-model-small-de-0.15"
BRIDGE_HOST = "localhost"
BRIDGE_PORT = 8888
DAEMON_HOST = "localhost"
DAEMON_PORT = 9999

SAMPLE_RATE = 16000


def find_usb_mic():
    """Auto-detect USB microphone ALSA device (card number changes across reboots)."""
    try:
        result = subprocess.run(["arecord", "-l"], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if "USB" in line and "card" in line:
                card_num = line.split("card ")[1].split(":")[0]
                device = f"plughw:{card_num},0"
                print(f"[voice-v2] Found USB mic: {device}", flush=True)
                return device
    except Exception as e:
        print(f"[voice-v2] Mic detection error: {e}", flush=True)
    
    print("[voice-v2] WARNING: No USB mic found, falling back to plughw:3,0", flush=True)
    return "plughw:3,0"


CAPTURE_DEVICE = find_usb_mic()
os.environ["AUDIODEV"] = CAPTURE_DEVICE

# Software gain: amplify mic input before feeding to Vosk
# USB mic on Pi 4 has very low sensitivity even at max hardware gain
SOFTWARE_GAIN = 8.0  # Multiply signal by this factor

def amplify_audio(data, gain=SOFTWARE_GAIN):
    """Amplify 16-bit PCM audio data by gain factor with clipping."""
    try:
        import array
        n_samples = len(data) // 2
        if n_samples == 0:
            return data
        # Use array module (much faster than struct for bulk operations)
        samples = array.array('h')
        samples.frombytes(data[:n_samples * 2])
        for i in range(len(samples)):
            v = int(samples[i] * gain)
            samples[i] = max(-32768, min(32767, v))
        return samples.tobytes()
    except Exception as e:
        # If amplification fails, return original data
        return data
# ─── Energy Gate (v2.2) ───
# RMS energy threshold to filter silence/ambient noise BEFORE feeding to Vosk.
# This prevents phantom word recognition from background noise.
# Value is on the AMPLIFIED signal (after 8x gain).
ENERGY_THRESHOLD = 500   # RMS value; lowered for better speech detection
ENERGY_WINDOW = 2        # Number of consecutive above-threshold chunks needed

_energy_above_count = 0  # rolling counter of chunks above threshold

def compute_rms(data):
    """Compute RMS energy of 16-bit PCM audio data."""
    n_samples = len(data) // 2
    if n_samples == 0:
        return 0
    try:
        samples = struct.unpack(f'<{n_samples}h', data[:n_samples * 2])
        sum_sq = sum(s * s for s in samples)
        return math.sqrt(sum_sq / n_samples)
    except Exception:
        return 0


# Exact wake words
WAKE_WORDS_EXACT = ["nox", "knox", "hallo nox", "hey nox", "na nox", "hi nox"]
# Fuzzy patterns: German Vosk model may mis-transcribe "Nox" as these
WAKE_WORDS_FUZZY = [
    "nox", "knox", "noks", "nocks", "noxx",
    "fox", "box",  # phonetically close
]
WAKE_PREFIXES = ["hallo", "hey", "hi", "na"]  # "hallo nox" etc.

SILENCE_TIMEOUT = 2.0
MIN_PHRASE_LENGTH = 2
CONVERSATION_TIMEOUT = 45.0  # Extended to 45s

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
            print("[voice-v2] Conversation timed out.", flush=True)
    
    def add_exchange(self, user_text, response_text=""):
        self.conversation_history.append({
            "user": user_text,
            "response": response_text,
            "ts": time.time()
        })
        if len(self.conversation_history) > 5:
            self.conversation_history.pop(0)

state = VoiceState()


# ─── Helpers ───
def send_to_daemon(cmd_json, timeout=10):
    """Send command to nox_daemon via TCP."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
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


_speaking_until = 0  # timestamp when TTS should be done

def speak_via_daemon(text):
    global _speaking_until
    # Estimate TTS duration: ~80ms per char + 500ms buffer
    est_duration = len(text) * 0.08 + 0.5
    _speaking_until = time.time() + est_duration
    send_to_daemon({"cmd": "speak", "text": text}, timeout=30)  # TTS can be slow


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


def levenshtein(s1, s2):
    """Simple Levenshtein distance."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def fuzzy_wake_word_check(text):
    """
    Check if text contains a wake word, using fuzzy matching.
    Returns (cleaned_text, is_wake_word)
    """
    text_lower = text.lower().strip()
    words = text_lower.split()
    
    # 1. Exact match (original logic)
    for ww in WAKE_WORDS_EXACT:
        if text_lower.startswith(ww):
            cleaned = text[len(ww):].strip()
            if cleaned and cleaned[0] in ",.!?;:":
                cleaned = cleaned[1:].strip()
            return cleaned, True
    
    # 2. Single word fuzzy match: any word within edit distance 1 of "nox"
    for i, word in enumerate(words):
        if levenshtein(word, "nox") <= 1:
            # Found fuzzy "nox" — rest of text is the content
            remaining = " ".join(words[i+1:])
            return remaining, True
    
    # 3. Prefix + fuzzy: "hallo" + word close to "nox"
    for i, word in enumerate(words):
        if word in WAKE_PREFIXES and i + 1 < len(words):
            next_word = words[i + 1]
            if levenshtein(next_word, "nox") <= 2:  # more lenient with prefix
                remaining = " ".join(words[i+2:])
                return remaining, True
    
    # 4. Phonetic pattern: DISABLED — too many false positives
    #    German Vosk model handles "Nox" much better than English model
    #    If needed, re-enable with stricter rules
    
    return text, False


def is_just_noise(text):
    """Check if the recognized text is just noise/fragments."""
    noise_words = {
        "", "ja", "nein", "oh", "ah", "uh", "hm", "hmm", "ähm",
        "ich", "du", "er", "sie", "es", "wir", "ihr",
        "ist", "war", "bin", "hat", "und", "oder", "aber",
        "so", "da", "ja", "ok", "okay", "gut", "na",
        "das", "die", "der", "den", "dem", "ein", "eine",
        "was", "wie", "wo", "wer", "noch", "auch", "mal",
        "nicht", "nur", "schon", "doch", "dann", "hier",
        "the", "a", "i", "it", "is", "yes", "no",  # English leftovers
    }
    return text.lower().strip() in noise_words


def main():
    global _speaking_until, _energy_above_count
    print("[voice-v2] Starting Nox voice loop v2.2 (fuzzy wake words + energy gate)...", flush=True)

    from vosk import Model, KaldiRecognizer

    print(f"[voice-v2] Loading Vosk model: {VOSK_MODEL_PATH}", flush=True)
    model = Model(VOSK_MODEL_PATH)
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    rec.SetWords(True)

    # Audio input
    print("[voice-v2] Starting audio capture...", flush=True)
    process = subprocess.Popen(
        ["arecord", "-D", CAPTURE_DEVICE, "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    # Signal we're listening (no TTS — it kills the audio pipeline)
    try:
        set_rgb(0, 100, 50, "breath", 0.5)
        time.sleep(0.5)
        set_rgb(128, 0, 255, "breath", 0.8)
    except:
        pass  # Don't crash on RGB failure

    print("[voice-v2] Listening... (fuzzy wake word matching + energy gate active)", flush=True)

    try:
        while True:
            data = process.stdout.read(4000)
            if len(data) == 0:
                break

            state.update()

            # Amplify before feeding to Vosk
            data = amplify_audio(data)

            # Energy gate (v2.2): skip silent/ambient chunks
            global _energy_above_count
            rms = compute_rms(data)
            if rms < ENERGY_THRESHOLD:
                _energy_above_count = 0
                # Still feed to Vosk (to finalize any pending recognition)
                # but don't start new recognitions from silence
                rec.AcceptWaveform(data)
                continue
            _energy_above_count += 1
            if _energy_above_count < ENERGY_WINDOW:
                rec.AcceptWaveform(data)
                continue

            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "").strip()

                if len(text) < MIN_PHRASE_LENGTH:
                    continue

                if is_just_noise(text):
                    continue

                # Echo protection: ignore audio while TTS is playing
                if time.time() < _speaking_until:
                    print(f"[voice-v2] Echo suppressed: '{text}'", flush=True)
                    continue

                print(f"[voice-v2] Heard: '{text}'", flush=True)

                # Fuzzy wake word check
                cleaned, had_wake_word = fuzzy_wake_word_check(text)

                if not state.in_conversation and not had_wake_word:
                    print(f"[voice-v2] No wake word detected. Ignoring.", flush=True)
                    continue

                # Activate conversation mode
                state.start_conversation()
                print(f"[voice-v2] {'Wake word!' if had_wake_word else 'Continuing conversation.'} Processing: '{cleaned if had_wake_word else text}'", flush=True)

                # If just the wake word with no content, acknowledge
                if had_wake_word and len(cleaned) < MIN_PHRASE_LENGTH:
                    send_to_daemon({"cmd": "move", "action": "wag_tail", "steps": 2, "speed": 80})
                    speak_via_daemon("Ja? Was ist?")
                    continue

                process_text = cleaned if had_wake_word else text

                # Visual feedback: thinking
                set_rgb(128, 0, 255, "speak", 2.0)

                # Post to bridge
                post_to_bridge("/voice/input", {
                    "text": process_text,
                    "had_wake_word": had_wake_word,
                    "in_conversation": state.in_conversation,
                    "recent_context": [
                        ex["user"] for ex in state.conversation_history[-3:]
                    ]
                })

                state.add_exchange(process_text)

                # Small head nod
                send_to_daemon({"cmd": "move", "action": "nod_lethargy", "steps": 1, "speed": 90})

                # Echo protection: suppress audio for estimated brain response + TTS time
                # Brain takes ~2s, TTS takes ~3-5s → suppress for 8s
                _speaking_until = time.time() + 8.0
                print(f"[voice-v2] Echo suppression active for 8s", flush=True)

                time.sleep(0.5)
                set_rgb(128, 0, 255, "breath", 0.8)

            else:
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
