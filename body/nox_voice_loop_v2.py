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
import urllib.request

os.environ["SDL_AUDIODRIVER"] = "alsa"

# ─── Config ───
VOSK_MODEL_PATH = os.environ.get(
    "VOSK_MODEL_PATH",
    os.path.expanduser("~/vosk-models/vosk-model-small-de-0.15"),
)
BRIDGE_HOST = "localhost"
BRIDGE_PORT = 8888
DAEMON_HOST = "localhost"
DAEMON_PORT = 9999

SAMPLE_RATE = 16000


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nox_audio import find_capture_device, spoken, watch_stderr  # noqa: E402

_mic = find_capture_device(os.environ.get("MIC_DEVICE"))
CAPTURE_DEVICE = _mic["device"]
print(f"[voice-v2] Microphone: {CAPTURE_DEVICE or 'NONE'} ({_mic['reason']})", flush=True)
if CAPTURE_DEVICE:
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
    except Exception:
        # If amplification fails, return original data
        return data
# ─── Energy Gate (v2.2, semantics fixed after issue #45) ───
# Filters phantom recognitions born of silence/ambient noise. The original
# version measured the chunks at the MOMENT a phrase finalized — but Vosk
# finalizes a phrase when silence begins, so real speech that ends in silence
# (i.e. every normal utterance) was dropped by lottery: the reporter's
# stt_debug output showed perfectly recognized commands dying here. Energy is
# now judged over the WHOLE utterance: a final is kept when enough chunks
# anywhere in it were above the threshold.
# Value is on the AMPLIFIED signal (after 8x gain).
# Per-microphone setting (nox.env): quiet mics drop real phrases here — the
# reporter's VoiceHAT lost "prends une photo" at the 500 default (issue #45,
# log 6). stt_debug.py shows the measured levels; lower this to match.
try:
    ENERGY_THRESHOLD = int(float(os.environ.get("ENERGY_THRESHOLD", "500") or 500))
except ValueError:
    print("[voice-v2] WARNING: ENERGY_THRESHOLD is not a number — using 500", flush=True)
    ENERGY_THRESHOLD = 500
ENERGY_WINDOW = 2        # chunks above threshold an utterance needs


class UtteranceEnergy:
    """Above-threshold chunks in the CURRENT utterance. feed() every chunk;
    harvest() when Vosk finalizes — it returns the count and starts the next
    utterance at zero. Shared with scripts/stt_debug.py so the tool can never
    drift from the service."""

    def __init__(self):
        self.chunks = 0
        self.peak = 0

    def feed(self, rms):
        self.peak = max(self.peak, rms)
        if rms >= ENERGY_THRESHOLD:
            self.chunks += 1

    def harvest(self):
        chunks, self.chunks = self.chunks, 0
        self.peak = 0
        return chunks

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
# The base wake word is configurable: small Vosk models mangle "Nox" badly in
# some languages (French hears "inox" or "knox"), and a word that exists in
# the model's vocabulary triggers far more reliably (issue #45 follow-up).
def _plain(word):
    """Lowercase and strip accents: Vosk writes "medor" where the config says
    "médor" — the match must not care (#45 interaction review)."""
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", word.lower())
                   if not unicodedata.combining(c))


# Several wake words are allowed, comma-separated: WAKE_WORD=milou,tintin
# (requested in issue #45 — pick words the STT model knows well).
WAKE_WORDS = []
for _w in (os.environ.get("WAKE_WORD") or "nox").replace(";", ",").split(","):
    _w = _plain(_w.strip())
    if _w and _w not in WAKE_WORDS:
        WAKE_WORDS.append(_w)
if not WAKE_WORDS:
    WAKE_WORDS = ["nox"]
WAKE_WORD = WAKE_WORDS[0]  # the primary word, shown in prompts and tools
# An everyday word as wake word fires on everything: WAKE_WORD=hey woke the dog
# on every "hey ..." a French TV said (issue #45 log). Warn, loudly.
_COMMON_WORDS = {"hey", "he", "eh", "hi", "na", "hallo", "ok", "oui", "non", "ja",
                 "nein", "yes", "no", "le", "la", "de", "un", "une", "et", "the", "a"}
for _w in WAKE_WORDS:
    if _w in _COMMON_WORDS or len(_w) < 3:
        print(f"[voice-v2] WARNING: WAKE_WORD='{_w}' is an everyday word — the dog "
              "will wake constantly. Pick a distinctive word (e.g. milou, rex, nox).", flush=True)
print(f"[voice-v2] Wake word{'s' if len(WAKE_WORDS) > 1 else ''}: "
      f"{', '.join(WAKE_WORDS)} (also with hey/hallo/... in front, "
      f"and one letter off)", flush=True)
WAKE_PREFIXES = ["hallo", "hey", "hi", "na", "he", "eh"]  # "hallo nox" etc.
WAKE_WORDS_EXACT = list(WAKE_WORDS) + [f"{p} {w}" for w in WAKE_WORDS
                                       for p in WAKE_PREFIXES]
if "nox" in WAKE_WORDS:
    # Mis-transcriptions seen in the wild for the default name
    WAKE_WORDS_EXACT = ["knox", "inox"] + WAKE_WORDS_EXACT

SILENCE_TIMEOUT = 2.0
MIN_PHRASE_LENGTH = 2
# Seconds the dog keeps listening WITHOUT a wake word after an exchange.
# 0 turns the window off: every command needs the wake word — the right
# setting next to NOX_MODE=commands, where ignored chatter gets no reply
# that would justify a conversation (issue #45 interaction review).
try:
    CONVERSATION_TIMEOUT = float(os.environ.get("CONVERSATION_TIMEOUT", "45") or 45)
except ValueError:
    print("[voice-v2] WARNING: CONVERSATION_TIMEOUT is not a number — using 45", flush=True)
    CONVERSATION_TIMEOUT = 45.0

# ─── State ───
class VoiceState:
    def __init__(self):
        self.in_conversation = False
        self.last_interaction = 0
        self.conversation_history = []
    
    def start_conversation(self):
        if CONVERSATION_TIMEOUT <= 0:
            return  # wake word required every time
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
    plain_words = [_plain(w) for w in words]  # accent-free, same word count
    plain_text = " ".join(plain_words)
    
    # 1. Exact match (original logic, accent-free)
    for ww in WAKE_WORDS_EXACT:
        if plain_text.startswith(ww):
            cleaned = " ".join(words[len(ww.split()):]).strip()
            if cleaned and cleaned[0] in ",.!?;:":
                cleaned = cleaned[1:].strip()
            return cleaned, True
    
    # 2. Single word fuzzy match: any word within edit distance 1 of a wake word
    for i, word in enumerate(plain_words):
        if any(levenshtein(word, ww) <= 1 for ww in WAKE_WORDS):
            # Found fuzzy "nox" — rest of text is the content
            remaining = " ".join(words[i+1:])
            return remaining, True

    # 3. Prefix + fuzzy: "hallo" + word close to a wake word
    for i, word in enumerate(plain_words):
        if word in WAKE_PREFIXES and i + 1 < len(plain_words):
            next_word = plain_words[i + 1]
            if any(levenshtein(next_word, ww) <= 2 for ww in WAKE_WORDS):  # more lenient with prefix
                remaining = " ".join(words[i+2:])
                return remaining, True
    
    # 4. Phonetic pattern: DISABLED — too many false positives
    #    German Vosk model handles "Nox" much better than English model
    #    If needed, re-enable with stricter rules
    
    return text, False


# ─── Self-echo detection (issue #45, voiceInput3 log) ───
# The fixed 8 s window after sending text to the brain missed everything the
# dog said LATER — a slow LLM reply, a battery warning — and the mic fed those
# words back in as commands: the dog held a conversation with itself. The
# daemon now remembers what it spoke ({"cmd": "speech_state"}); a recognised
# phrase whose words are mostly the dog's own recent words is an echo.
ECHO_TEXT_WINDOW = 12.0   # seconds after TTS ended in which a near-copy is echo
ECHO_MIN_TOKENS = 3       # shorter phrases are too easy to match by accident
ECHO_OVERLAP = 0.7        # fraction of heard words found in one spoken phrase


def _echo_tokens(text):
    tokens = []
    for word in (text or "").split():
        word = _plain(word).strip(".,!?;:'\"()«»")
        if word:
            tokens.append(word)
    return tokens


def is_own_speech(heard, speech_state):
    """True when `heard` is the dog's own TTS coming back through the mic."""
    if not isinstance(speech_state, dict) or not speech_state.get("ok"):
        return False  # old daemon or daemon down — fall back to the time window
    if speech_state.get("speaking"):
        return True
    heard_tokens = _echo_tokens(heard)
    if len(heard_tokens) < ECHO_MIN_TOKENS:
        return False  # never eat short real commands ("assis", "couché")
    for entry in speech_state.get("recent") or []:
        age = entry.get("age_s")
        if age is None or age > ECHO_TEXT_WINDOW:
            continue
        spoken_set = set(_echo_tokens(entry.get("text")))
        if not spoken_set:
            continue
        overlap = sum(t in spoken_set for t in heard_tokens) / len(heard_tokens)
        if overlap >= ECHO_OVERLAP:
            return True
    return False


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
    global _speaking_until
    energy = UtteranceEnergy()
    print("[voice-v2] Starting Nox voice loop v2.2 (fuzzy wake words + energy gate)...", flush=True)

    from vosk import Model, KaldiRecognizer

    if not os.path.isdir(VOSK_MODEL_PATH):
        # Exit 0 on purpose: Restart=on-failure would otherwise crash-loop the
        # service on every machine that hasn't downloaded a model yet.
        print(f"[voice-v2] Vosk model not found: {VOSK_MODEL_PATH}", flush=True)
        print("[voice-v2] Download one from https://alphacephei.com/vosk/models and unzip it there,", flush=True)
        print("[voice-v2] or set VOSK_MODEL_PATH in nox.env. Voice input stays off until then.", flush=True)
        sys.exit(0)

    print(f"[voice-v2] Loading Vosk model: {VOSK_MODEL_PATH}", flush=True)
    model = Model(VOSK_MODEL_PATH)
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    rec.SetWords(True)

    # Audio input. No microphone is a clean, EXPLAINED stop (like no model) —
    # the old path invented plughw:3,0, arecord died on it, and the loop ended
    # with nothing but "Voice loop stopped" (issue #45).
    if not CAPTURE_DEVICE:
        print(f"[voice-v2] No microphone: {_mic['reason']}. Voice input stays off.", flush=True)
        return
    print("[voice-v2] Starting audio capture...", flush=True)
    process = subprocess.Popen(
        ["arecord", "-D", CAPTURE_DEVICE, "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stderr_lines = watch_stderr(process)  # drained live — an undrained PIPE stalls arecord

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
                # arecord ended — say WHY, and exit non-zero so systemd retries
                # (a mic that is still enumerating usually works seconds later).
                time.sleep(0.2)  # let the drain thread catch the last lines
                print(f"[voice-v2] audio capture ended: "
                      f"{stderr_lines[-1] if stderr_lines else 'no error output'} "
                      f"(device {CAPTURE_DEVICE})", flush=True)
                sys.exit(1)

            state.update()

            # Amplify before feeding to Vosk
            data = amplify_audio(data)
            energy.feed(compute_rms(data))

            if rec.AcceptWaveform(data):
                speech_chunks = energy.harvest()
                result = json.loads(rec.Result())
                text = result.get("text", "").strip()

                if len(text) < MIN_PHRASE_LENGTH:
                    continue

                if is_just_noise(text):
                    continue

                # Energy gate: an utterance with no real speech energy is a
                # phantom from ambient noise. Logged — a dropped phrase used
                # to vanish without a trace (issue #45).
                if speech_chunks < ENERGY_WINDOW:
                    print(f"[voice-v2] Energy gate dropped: '{text}' "
                          f"({speech_chunks} chunk(s) above level {ENERGY_THRESHOLD}, "
                          f"need {ENERGY_WINDOW} — speak louder or closer)", flush=True)
                    continue

                # Echo protection: ignore audio while TTS is playing
                if time.time() < _speaking_until:
                    print(f"[voice-v2] Echo suppressed: '{text}'", flush=True)
                    continue

                # The dog's own words coming back through the mic — brain
                # replies and battery warnings arrive AFTER the time window.
                speech_state = send_to_daemon({"cmd": "speech_state", "_internal": True},
                                              timeout=3)
                if is_own_speech(text, speech_state):
                    print(f"[voice-v2] Self-echo suppressed: '{text}'", flush=True)
                    continue

                print(f"[voice-v2] Heard: '{text}'", flush=True)

                # Fuzzy wake word check
                cleaned, had_wake_word = fuzzy_wake_word_check(text)

                if not state.in_conversation and not had_wake_word:
                    print("[voice-v2] No wake word detected. Ignoring.", flush=True)
                    continue

                # Activate conversation mode
                state.start_conversation()
                print(f"[voice-v2] {'Wake word!' if had_wake_word else 'Continuing conversation.'} Processing: '{cleaned if had_wake_word else text}'", flush=True)

                # If just the wake word with no content, acknowledge
                if had_wake_word and len(cleaned) < MIN_PHRASE_LENGTH:
                    send_to_daemon({"cmd": "move", "action": "wag_tail", "steps": 2, "speed": 80})
                    speak_via_daemon(spoken("wake_ack"))
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

                # Brief quiet window for the dog's own motion noise (the nod).
                # This used to be a blind 8 s guess at the brain's reply time
                # and ate the user's real follow-ups ('cathy assis', 'remue la
                # queue' in issue #45 log 6) — actual TTS echo is caught by the
                # daemon's speech_state ground truth above.
                _speaking_until = time.time() + 2.5
                print("[voice-v2] Echo suppression active for 2.5s", flush=True)

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
        speak_via_daemon(spoken("goodnight"))
        set_rgb(0, 0, 80, "breath", 0.3)
        print("[voice-v2] Voice loop stopped.", flush=True)


if __name__ == "__main__":
    main()
