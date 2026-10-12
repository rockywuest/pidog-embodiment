#!/usr/bin/env python3
"""stt_debug.py — what does the dog actually hear? (issue #45)

Live speech-to-text monitor: prints every word Vosk recognises, through
EXACTLY the pipeline the voice service uses — same body/nox.env, same
microphone, same software gain, same energy gate, same wake-word matching.
Nothing is sent to the brain and nothing is spoken: just you, the mic,
and the words on the screen.

    sudo systemctl stop nox-voice     # free the microphone
    python3 scripts/stt_debug.py
    ... speak ...                     # Ctrl+C to stop
    sudo systemctl start nox-voice

Reading the output:
    [partial] ...          what Vosk thinks while you are still speaking
    HEARD 'text'           a finished phrase — exactly what the service sees
      wake word: ...       whether this phrase would wake the dog
    dropped (energy gate)  the phrase ended on a too-quiet chunk; the service
                           would have discarded it (speak louder / closer)
    dropped (noise filter) single filler words the service ignores
"""
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_env_file(path, environ=None):
    """KEY=VALUE pairs from nox.env; comments and malformed lines are ignored.
    Returns only keys NOT already set in the environment — a variable you
    export on the command line wins, like systemd's Environment= would."""
    environ = os.environ if environ is None else environ
    loaded = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in environ:
                    loaded[key] = value
    except OSError:
        pass
    return loaded


def main():
    env_file = os.path.join(REPO, "body", "nox.env")
    loaded = load_env_file(env_file)
    os.environ.update(loaded)
    if loaded:
        print(f"[stt-debug] Loaded {len(loaded)} value(s) from {env_file}")
    else:
        print(f"[stt-debug] No values loaded from {env_file} "
              "(missing, or everything already set in the environment)")

    # Import AFTER the env is in place: the voice loop reads WAKE_WORD,
    # MIC_DEVICE and VOSK_MODEL_PATH at import time — that is the point,
    # this tool must see the microphone through the service's eyes.
    sys.path.insert(0, os.path.join(REPO, "body"))
    import nox_voice_loop_v2 as vl  # prints mic + wake word, like the service

    print(f"[stt-debug] Vosk model: {vl.VOSK_MODEL_PATH}")
    if not os.path.isdir(vl.VOSK_MODEL_PATH):
        print("[stt-debug] Model directory not found — set VOSK_MODEL_PATH in "
              "body/nox.env (see the README's voice section).")
        sys.exit(1)
    if not vl.CAPTURE_DEVICE:
        print(f"[stt-debug] No microphone found: {vl._mic['reason']}")
        sys.exit(1)

    try:
        from vosk import Model, KaldiRecognizer, SetLogLevel
    except ImportError:
        print("[stt-debug] vosk is not installed — pip3 install vosk")
        sys.exit(1)
    SetLogLevel(-1)  # keep Kaldi's boot log off the screen
    print("[stt-debug] Loading model (takes a few seconds)...")
    model = Model(vl.VOSK_MODEL_PATH)
    rec = KaldiRecognizer(model, vl.SAMPLE_RATE)

    try:
        process = subprocess.Popen(
            ["arecord", "-D", vl.CAPTURE_DEVICE, "-f", "S16_LE",
             "-r", str(vl.SAMPLE_RATE), "-c", "1", "-t", "raw"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("[stt-debug] arecord not found — sudo apt install alsa-utils")
        sys.exit(1)
    from nox_audio import watch_stderr
    stderr_lines = watch_stderr(process)

    print(f"[stt-debug] Listening on {vl.CAPTURE_DEVICE} — speak! "
          f"(wake word '{vl.WAKE_WORD}', Ctrl+C to stop)\n")

    show_partials = sys.stdout.isatty()
    energy_above = 0     # the service's energy gate, replicated chunk by chunk
    peak_rms = 0
    last_partial_len = 0

    def clear_partial():
        nonlocal last_partial_len
        if show_partials and last_partial_len:
            print("\r" + " " * last_partial_len + "\r", end="", flush=True)
            last_partial_len = 0

    try:
        while True:
            data = process.stdout.read(4000)
            if len(data) == 0:
                time.sleep(0.2)  # let the drain thread catch the last lines
                tail = stderr_lines[-1] if stderr_lines else "no error output"
                clear_partial()
                print(f"[stt-debug] audio capture ended: {tail}")
                if "busy" in tail.lower():
                    print("[stt-debug] The microphone is in use — stop the "
                          "service first:  sudo systemctl stop nox-voice")
                sys.exit(1)

            data = vl.amplify_audio(data)
            rms = vl.compute_rms(data)
            peak_rms = max(peak_rms, rms)
            if rms < vl.ENERGY_THRESHOLD:
                energy_above = 0
            else:
                energy_above += 1
            gate_open = energy_above >= vl.ENERGY_WINDOW

            if rec.AcceptWaveform(data):
                text = json.loads(rec.Result()).get("text", "").strip()
                phrase_peak, peak_rms = peak_rms, 0
                if not text:
                    continue
                clear_partial()
                # Same verdicts, same order as the service — but visible.
                if not gate_open:
                    print(f"  dropped (energy gate): '{text}'  "
                          f"[peak level {phrase_peak:.0f} < {vl.ENERGY_THRESHOLD}"
                          " at the end — speak louder or closer]")
                    continue
                if len(text) < vl.MIN_PHRASE_LENGTH or vl.is_just_noise(text):
                    print(f"  dropped (noise filter): '{text}'")
                    continue
                print(f"HEARD '{text}'  [peak level {phrase_peak:.0f}]")
                cleaned, woke = vl.fuzzy_wake_word_check(text)
                if woke:
                    print(f"  wake word: MATCHED → the dog would process: "
                          f"'{cleaned or '(nothing — it would just answer)'}'")
                else:
                    print("  wake word: no — ignored unless already in a "
                          "conversation")
            elif show_partials:
                partial = json.loads(rec.PartialResult()).get("partial", "")
                if partial:
                    line = f"  [partial] {partial}"
                    pad = max(0, last_partial_len - len(line))
                    print("\r" + line + " " * pad, end="", flush=True)
                    last_partial_len = len(line)
    except KeyboardInterrupt:
        clear_partial()
        print("\n[stt-debug] Stopped. Remember:  sudo systemctl start nox-voice")
    finally:
        process.terminate()


if __name__ == "__main__":
    main()
