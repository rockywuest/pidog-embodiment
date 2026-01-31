#!/usr/bin/env python3
"""
nox_voice_loop_v3.py — Voice listener using faster-whisper for PiDog.

Architecture change from v2:
- Replaced Vosk (streaming, low accuracy) with faster-whisper (batch, high accuracy)
- Uses webrtcvad for Voice Activity Detection (speech start/end)
- Records complete utterances, then transcribes as a batch
- Much higher transcription accuracy at the cost of slight latency
- Keeps: wake word detection, conversation state, echo suppression, energy gate

Pipeline: Mic → arecord 16kHz → Amplify → VAD speech detection → Buffer utterance
          → faster-whisper transcribe → Wake word check → Bridge → Brain
"""

import os
import sys
import json
import time
import struct
import math
import array
import socket
import subprocess
import threading
import urllib.request
import io
import wave
import tempfile

os.environ["SDL_AUDIODRIVER"] = "alsa"

# ─── Config ───
BRIDGE_HOST = "localhost"
BRIDGE_PORT = 8888
DAEMON_HOST = "localhost"
DAEMON_PORT = 9999

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit = 2 bytes

# Whisper config
WHISPER_MODEL = "base"  # tiny=39MB, base=74MB, small=244MB
WHISPER_LANGUAGE = "de"  # German — wir sprechen Deutsch hier
WHISPER_BEAM_SIZE = 5    # Lower = faster, higher = more accurate

# VAD config
VAD_AGGRESSIVENESS = 3       # 0-3: higher = more aggressive filtering
VAD_FRAME_MS = 30            # Frame size: 10, 20, or 30 ms
SPEECH_START_FRAMES = 5      # Consecutive voiced frames to start recording
SPEECH_END_FRAMES = 15       # Consecutive silent frames to stop recording (~450ms)
MAX_SPEECH_SECONDS = 8       # Max utterance length
MIN_SPEECH_SECONDS = 0.8     # Min utterance length (ignore short blips)

# Audio gain
SOFTWARE_GAIN = 12.0

# Wake words
WAKE_WORDS_EXACT = ["nox", "knox", "knocks", "hello nox", "hey nox", "hi nox"]
WAKE_PREFIXES = ["hello", "hey", "hi", "yo"]

CONVERSATION_TIMEOUT = 45.0
MIN_PHRASE_LENGTH = 2


def find_usb_mic():
    """Auto-detect USB microphone ALSA device."""
    try:
        result = subprocess.run(["arecord", "-l"], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if "USB" in line and "card" in line:
                card_num = line.split("card ")[1].split(":")[0]
                device = f"plughw:{card_num},0"
                print(f"[voice-v3] Found USB mic: {device}", flush=True)
                return device
    except Exception as e:
        print(f"[voice-v3] Mic detection error: {e}", flush=True)
    print("[voice-v3] WARNING: No USB mic found, falling back to plughw:3,0", flush=True)
    return "plughw:3,0"


CAPTURE_DEVICE = find_usb_mic()
os.environ["AUDIODEV"] = CAPTURE_DEVICE


# ─── Audio Processing ───
def amplify_audio(data, gain=SOFTWARE_GAIN):
    """Amplify 16-bit PCM audio data by gain factor with clipping."""
    try:
        n_samples = len(data) // 2
        if n_samples == 0:
            return data
        samples = array.array('h')
        samples.frombytes(data[:n_samples * 2])
        for i in range(len(samples)):
            v = int(samples[i] * gain)
            samples[i] = max(-32768, min(32767, v))
        return samples.tobytes()
    except Exception:
        return data


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


def audio_to_wav_bytes(pcm_data):
    """Convert raw PCM data to WAV format in memory."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_data)
    buf.seek(0)
    return buf


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
            print("[voice-v3] Conversation timed out.", flush=True)

    def add_exchange(self, user_text, response_text=""):
        self.conversation_history.append({
            "user": user_text,
            "response": response_text,
            "ts": time.time()
        })
        if len(self.conversation_history) > 5:
            self.conversation_history.pop(0)


state = VoiceState()


# ─── Communication Helpers ───
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
        print(f"[voice-v3] daemon error: {e}", flush=True)
        return {"error": str(e)}


def set_rgb(r, g, b, mode="breath", bps=0.8):
    send_to_daemon({"cmd": "rgb", "r": r, "g": g, "b": b, "mode": mode, "bps": bps})


_speaking_until = 0


def speak_via_daemon(text):
    global _speaking_until
    est_duration = len(text) * 0.08 + 0.5
    _speaking_until = time.time() + est_duration
    send_to_daemon({"cmd": "speak", "text": text}, timeout=30)


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
        print(f"[voice-v3] bridge error: {e}", flush=True)
        return None


def get_from_bridge(path):
    """GET data from the local brain bridge."""
    try:
        url = f"http://{BRIDGE_HOST}:{BRIDGE_PORT}{path}"
        with urllib.request.urlopen(url, timeout=3) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def update_echo_suppression():
    """Check bridge for actual TTS end time and update echo suppression."""
    global _speaking_until
    result = get_from_bridge("/voice/echo_until")
    if result and "echo_until" in result:
        bridge_echo = result["echo_until"]
        if bridge_echo > _speaking_until:
            _speaking_until = bridge_echo
            print(f"[voice-v3] Echo extended to {bridge_echo - time.time():.1f}s (TTS active)", flush=True)


def _echo_monitor():
    """Background thread: poll bridge for TTS end time."""
    global _speaking_until
    time.sleep(2.0)
    for _ in range(6):
        update_echo_suppression()
        if time.time() > _speaking_until + 0.5:
            break
        time.sleep(2.0)


# ─── Wake Word Detection ───
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
    """Check if text contains a wake word. Returns (cleaned_text, is_wake_word)."""
    text_lower = text.lower().strip()
    words = text_lower.split()

    # 1. Exact match
    for ww in WAKE_WORDS_EXACT:
        if text_lower.startswith(ww):
            cleaned = text[len(ww):].strip()
            if cleaned and cleaned[0] in ",.!?;:":
                cleaned = cleaned[1:].strip()
            return cleaned, True

    # 2. Single word fuzzy match: edit distance 1 from "nox"
    for i, word in enumerate(words):
        if levenshtein(word, "nox") <= 1:
            remaining = " ".join(words[i+1:])
            return remaining, True

    # 3. Prefix + fuzzy: "hey" + word close to "nox"
    for i, word in enumerate(words):
        if word in WAKE_PREFIXES and i + 1 < len(words):
            next_word = words[i + 1]
            if levenshtein(next_word, "nox") <= 2:
                remaining = " ".join(words[i+2:])
                return remaining, True

    return text, False


def is_just_noise(text):
    """Check if the recognized text is just noise/fragments."""
    noise_words = {
        "", "the", "a", "an", "i", "he", "she", "it", "we", "you",
        "is", "was", "are", "am", "be", "do", "did", "has", "had",
        "to", "of", "in", "on", "at", "by", "for", "and", "or",
        "but", "so", "if", "no", "yes", "oh", "ah", "um", "uh",
        "hm", "hmm", "yeah", "yep", "nah", "huh", "okay",
        "that", "this", "what", "who", "how", "not", "just",
        "one", "two", "like", "well", "now", "then", "here",
        "thank you", "thanks", "you know",
    }
    return text.lower().strip() in noise_words


# ─── Main Loop ───
def main():
    global _speaking_until
    print("[voice-v3] Starting Nox voice loop v3.0 (faster-whisper + webrtcvad)...", flush=True)

    # Load whisper model
    from faster_whisper import WhisperModel
    print(f"[voice-v3] Loading faster-whisper model '{WHISPER_MODEL}'...", flush=True)
    t0 = time.time()
    whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    print(f"[voice-v3] Model loaded in {time.time()-t0:.1f}s", flush=True)

    # Load VAD
    import webrtcvad
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

    # Audio input via arecord
    print("[voice-v3] Starting audio capture...", flush=True)
    process = subprocess.Popen(
        ["arecord", "-D", CAPTURE_DEVICE, "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    # Startup signal
    try:
        set_rgb(0, 200, 50, "breath", 0.5)
        time.sleep(0.5)
        set_rgb(128, 0, 255, "breath", 0.8)
    except:
        pass

    print("[voice-v3] Listening... (faster-whisper + VAD active)", flush=True)

    # VAD state
    frame_bytes = int(SAMPLE_RATE * VAD_FRAME_MS / 1000) * SAMPLE_WIDTH  # bytes per VAD frame
    speech_buffer = bytearray()  # Accumulates speech audio
    voiced_count = 0  # Consecutive voiced frames
    silent_count = 0  # Consecutive silent frames
    is_speaking = False  # Currently recording speech
    speech_start_time = 0

    # Read buffer — arecord gives us continuous data, we process in VAD-sized frames
    read_buf = bytearray()

    try:
        while True:
            # Read a chunk from arecord
            raw = process.stdout.read(frame_bytes * 4)  # Read ~120ms at a time
            if len(raw) == 0:
                break

            state.update()

            # Amplify
            raw = amplify_audio(raw)

            # Add to read buffer
            read_buf.extend(raw)

            # Process complete VAD frames
            while len(read_buf) >= frame_bytes:
                frame = bytes(read_buf[:frame_bytes])
                read_buf = read_buf[frame_bytes:]

                # Echo suppression: skip while TTS is playing
                if time.time() < _speaking_until:
                    continue

                # VAD: is this frame voiced?
                try:
                    is_voiced = vad.is_speech(frame, SAMPLE_RATE)
                except Exception:
                    is_voiced = False

                if is_voiced:
                    voiced_count += 1
                    silent_count = 0
                else:
                    silent_count += 1
                    if not is_speaking:
                        voiced_count = 0

                # Start recording when enough voiced frames
                if not is_speaking and voiced_count >= SPEECH_START_FRAMES:
                    is_speaking = True
                    speech_start_time = time.time()
                    speech_buffer = bytearray()
                    # Visual: listening
                    set_rgb(0, 200, 100, "listen", 2.0)
                    print("[voice-v3] Speech detected, recording...", flush=True)

                # Buffer audio while speaking
                if is_speaking:
                    speech_buffer.extend(frame)

                    duration = time.time() - speech_start_time

                    # Stop conditions: silence or max length
                    if silent_count >= SPEECH_END_FRAMES or duration >= MAX_SPEECH_SECONDS:
                        is_speaking = False
                        voiced_count = 0
                        silent_count = 0

                        # Check minimum duration
                        if duration < MIN_SPEECH_SECONDS:
                            print(f"[voice-v3] Too short ({duration:.1f}s), skipping", flush=True)
                            continue

                        # Check RMS energy of the buffer
                        rms = compute_rms(bytes(speech_buffer))
                        if rms < 800:
                            print(f"[voice-v3] Too quiet (RMS={rms:.0f}), skipping", flush=True)
                            continue

                        # Visual: transcribing
                        set_rgb(128, 0, 255, "speak", 2.0)
                        print(f"[voice-v3] Transcribing {duration:.1f}s of audio (RMS={rms:.0f})...", flush=True)

                        # Transcribe with faster-whisper using temp file
                        t0 = time.time()
                        tmp_path = "/tmp/_nox_speech.wav"
                        with wave.open(tmp_path, 'wb') as wf:
                            wf.setnchannels(CHANNELS)
                            wf.setsampwidth(SAMPLE_WIDTH)
                            wf.setframerate(SAMPLE_RATE)
                            wf.writeframes(bytes(speech_buffer))
                        segments, info = whisper_model.transcribe(
                            tmp_path,
                            language=WHISPER_LANGUAGE,
                            beam_size=WHISPER_BEAM_SIZE,
                            vad_filter=False,  # We already did VAD externally
                            no_speech_threshold=0.85,  # More permissive for weak USB mic
                            log_prob_threshold=-1.5,    # Accept lower-confidence transcriptions
                            condition_on_previous_text=False,  # Avoid hallucination loops
                        )
                        # Collect segments and filter high no_speech ones
                        all_segs = list(segments)
                        text_parts = []
                        for seg in all_segs:
                            if seg.no_speech_prob < 0.85:  # Only keep likely-speech segments
                                text_parts.append(seg.text.strip())
                            else:
                                print(f"[voice-v3] Segment filtered (no_speech={seg.no_speech_prob:.2f}): '{seg.text.strip()}'", flush=True)
                        text = " ".join(text_parts).strip()
                        transcribe_time = time.time() - t0

                        # Hallucination filter: if text has way more words than
                        # plausible for the audio duration, it's a hallucination
                        word_count = len(text.split())
                        max_plausible_words = int(duration * 4) + 3  # ~4 words/sec max
                        if word_count > max_plausible_words and duration < 3.0:
                            print(f"[voice-v3] Hallucination filtered ({word_count} words for {duration:.1f}s): '{text[:60]}...'", flush=True)
                            text = ""

                        print(f"[voice-v3] Whisper ({transcribe_time:.1f}s): '{text}'", flush=True)

                        if len(text) < MIN_PHRASE_LENGTH:
                            set_rgb(128, 0, 255, "breath", 0.8)
                            continue

                        if is_just_noise(text):
                            set_rgb(128, 0, 255, "breath", 0.8)
                            continue

                        # Process the transcribed text
                        process_transcription(text)

    except KeyboardInterrupt:
        pass
    finally:
        process.terminate()
        speak_via_daemon("Ich gehe schlafen. Gute Nacht!")
        set_rgb(0, 0, 80, "breath", 0.3)
        print("[voice-v3] Voice loop stopped.", flush=True)


def process_transcription(text):
    """Process a complete transcription: wake word check, bridge post."""
    global _speaking_until

    # Wake word check
    cleaned, had_wake_word = fuzzy_wake_word_check(text)

    if not state.in_conversation and not had_wake_word:
        print(f"[voice-v3] No wake word. Ignoring: '{text}'", flush=True)
        set_rgb(128, 0, 255, "breath", 0.8)
        return

    # Activate conversation mode
    state.start_conversation()
    print(f"[voice-v3] {'Wake word!' if had_wake_word else 'Continuing.'} Processing: '{cleaned if had_wake_word else text}'", flush=True)

    # If just the wake word with no content, acknowledge
    if had_wake_word and len(cleaned) < MIN_PHRASE_LENGTH:
        send_to_daemon({"cmd": "move", "action": "wag_tail", "steps": 2, "speed": 80})
        speak_via_daemon("Yes? What is it?")
        return

    process_text = cleaned if had_wake_word else text

    # Visual: thinking
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

    # Head nod
    send_to_daemon({"cmd": "move", "action": "nod_lethargy", "steps": 1, "speed": 90})

    # Smart echo suppression
    _speaking_until = time.time() + 3.0
    print(f"[voice-v3] Echo suppression: 3s initial (bridge extends for TTS)", flush=True)
    threading.Thread(target=_echo_monitor, daemon=True).start()

    time.sleep(0.3)
    set_rgb(128, 0, 255, "breath", 0.8)


if __name__ == "__main__":
    main()
