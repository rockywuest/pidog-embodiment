#!/usr/bin/env python3
"""
nox_vision.py — Local vision engine for PiDog.

Runs SmolVLM-256M via llama.cpp subprocess for scene understanding.
Captures frames from the daemon camera, runs inference, writes results
to a JSON file that the bridge and behavior engine can read.

Architecture:
- Standalone process (NOT a thread in bridge — memory isolation)
- Captures photos via daemon TCP (localhost:9999)
- Runs llama-llava-cli / llama-mtmd-cli as subprocess
- Writes results to /tmp/nox_vision_latest.json
- Periodic mode: analyze every INTERVAL seconds
"""

import json
import time
import os
import sys
import socket
import subprocess
import base64
import signal

# ─── Configuration ───────────────────────────────────────────────────────────

# llama.cpp binary — try mtmd first (newer), fallback to llava
_LLAMA_BINS = [
    os.path.expanduser("~/llama.cpp/build/bin/llama-mtmd-cli"),
    os.path.expanduser("~/llama.cpp/build/bin/llama-llava-cli"),
]
MODEL_PATH = os.path.expanduser("~/models/smolvlm/SmolVLM-256M-Instruct-Q8_0.gguf")
MMPROJ_PATH = os.path.expanduser("~/models/smolvlm/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf")
RESULT_FILE = "/tmp/nox_vision_latest.json"
PHOTO_PATH = "/tmp/nox_vision_frame.jpg"

INTERVAL = 60       # seconds between periodic analyses (inference takes ~37s on Pi 4)
MAX_TOKENS = 64     # max output tokens (keep it concise, saves ~7s)
TEMPERATURE = 0.1   # low temp for factual descriptions
TIMEOUT = 90        # kill subprocess after this many seconds (warmup can be slow)

DAEMON_HOST = "localhost"
DAEMON_PORT = 9999
# The first photo starts the camera inside the daemon: importing vilib (OpenCV,
# picamera2) plus camera_start takes well over 10 s on a Pi 4. With the old 10 s
# the warmup always failed and /vision showed "timed out" for a whole minute
# (issue #36). Later photos come from the running camera in about a second.
PHOTO_TIMEOUT = 45

# ─── Prompts ─────────────────────────────────────────────────────────────────

PROMPTS = {
    # One plain sentence, no numbered checklist: the 256M model copied the
    # list's first item back ("1) Any people.") instead of answering (#36).
    "patrol": (
        "Describe this image in one or two short sentences. "
        "Mention any people and anything blocking the way."
    ),
    "describe": "Describe this scene in detail.",
    "obstacles": "Are there any obstacles ahead? How far away? Is it safe to walk forward?",
    "people": "How many people do you see? Where are they relative to the camera?",
}

DEFAULT_PROMPT = "patrol"

# ─── Globals ─────────────────────────────────────────────────────────────────

running = True
llama_bin = None  # resolved at startup


def _signal_handler(sig, frame):
    global running
    print(f"[vision] Signal {sig} received, shutting down...", flush=True)
    running = False


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


# ─── Daemon Communication ────────────────────────────────────────────────────

def _daemon_cmd(cmd_dict, timeout=10):
    """Send a command to the daemon via TCP socket."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((DAEMON_HOST, DAEMON_PORT))
        sock.sendall(json.dumps(cmd_dict).encode() + b"\n")
        chunks = []
        while True:
            data = sock.recv(65536)
            if not data:
                break
            chunks.append(data)
        sock.close()
        return json.loads(b"".join(chunks).decode())
    except Exception as e:
        return {"error": str(e)}


def capture_frame(timeout=PHOTO_TIMEOUT):
    """Take a photo via daemon and save as JPEG."""
    result = _daemon_cmd({"cmd": "photo"}, timeout=timeout)
    if result.get("error"):
        return None, result["error"]

    # Daemon may return file path or base64
    photo_path = result.get("photo")
    if photo_path and os.path.isfile(photo_path):
        return photo_path, None

    b64 = result.get("photo_b64")
    if b64:
        try:
            img_data = base64.b64decode(b64)
            with open(PHOTO_PATH, "wb") as f:
                f.write(img_data)
            return PHOTO_PATH, None
        except Exception as e:
            return None, f"decode error: {e}"

    return None, f"no photo in response: {list(result.keys())}"


def capture_with_retry(wait=5):
    """The warmup photo: one more try before giving up for a whole INTERVAL —
    a camera that is still starting usually answers a few seconds later."""
    photo_path, err = capture_frame()
    if err:
        print(f"[vision] Warmup capture failed ({err}), retrying once...", flush=True)
        time.sleep(wait)
        photo_path, err = capture_frame()
    return photo_path, err


# ─── Inference ───────────────────────────────────────────────────────────────

def run_inference(image_path, prompt_type="patrol"):
    """Run SmolVLM inference via llama.cpp subprocess."""
    prompt = PROMPTS.get(prompt_type, PROMPTS["patrol"])

    cmd = [
        llama_bin,
        "-m", MODEL_PATH,
        "--mmproj", MMPROJ_PATH,
        "--image", image_path,
        "-p", prompt,
        "-n", str(MAX_TOKENS),
        "--temp", str(TEMPERATURE),
    ]

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
        elapsed = time.time() - start

        if proc.returncode != 0:
            stderr = proc.stderr.strip()[-200:] if proc.stderr else "unknown"
            return None, elapsed, f"exit code {proc.returncode}: {stderr}"

        # Extract the generated text (skip model loading messages on stderr)
        output = proc.stdout.strip()
        if not output:
            output = proc.stderr.strip()  # some versions output to stderr

        return output, elapsed, None

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return None, elapsed, f"timeout after {TIMEOUT}s"
    except Exception as e:
        elapsed = time.time() - start
        return None, elapsed, str(e)


# ─── Result Writing ──────────────────────────────────────────────────────────

def write_result(description, prompt_type, inference_s, error=None):
    """Write result to JSON file (atomic write via rename)."""
    result = {
        "ts": time.time(),
        "description": description,
        "prompt_type": prompt_type,
        "inference_s": round(inference_s, 1) if inference_s else None,
        "model": "SmolVLM-256M-Q8",
        "error": error,
    }
    tmp = RESULT_FILE + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(result, f)
        os.rename(tmp, RESULT_FILE)
    except Exception as e:
        print(f"[vision] Write failed: {e}", flush=True)


# ─── Main Loop ───────────────────────────────────────────────────────────────

def verify_setup():
    """Check that model files and binary exist. Returns None, or what is missing."""
    global llama_bin

    # Find llama binary
    for bin_path in _LLAMA_BINS:
        if os.path.isfile(bin_path) and os.access(bin_path, os.X_OK):
            llama_bin = bin_path
            break

    if not llama_bin:
        return (f"no llama.cpp binary found (tried {', '.join(_LLAMA_BINS)}) — "
                "build llama.cpp, see 'Local Vision' in the README")

    if not os.path.isfile(MODEL_PATH):
        return f"model not found: {MODEL_PATH}"

    if not os.path.isfile(MMPROJ_PATH):
        return f"projector not found: {MMPROJ_PATH}"

    print(f"[vision] Setup OK:", flush=True)
    print(f"  Binary:    {llama_bin}", flush=True)
    print(f"  Model:     {MODEL_PATH}", flush=True)
    print(f"  Projector: {MMPROJ_PATH}", flush=True)
    print(f"  Interval:  {INTERVAL}s", flush=True)
    return None


def main():
    """Main vision loop."""
    print(f"[vision] Nox Vision Engine starting...", flush=True)

    problem = verify_setup()
    if problem:
        # Leave the reason where /vision reads it — otherwise the API can only
        # say "no result file", whatever actually went wrong (issue #36).
        print(f"[vision] ERROR: {problem}", flush=True)
        write_result(None, "setup", None, error=problem)
        print("[vision] Setup failed, exiting.", flush=True)
        sys.exit(1)

    # Wait for daemon to be ready
    for attempt in range(10):
        result = _daemon_cmd({"cmd": "ping"})
        if result.get("pong"):
            print(f"[vision] Daemon connected (attempt {attempt + 1})", flush=True)
            break
        print(f"[vision] Waiting for daemon... (attempt {attempt + 1})", flush=True)
        time.sleep(3)
    else:
        print("[vision] WARNING: Could not connect to daemon, starting anyway", flush=True)

    # First inference (warmup, may be slower)
    print("[vision] Running warmup inference...", flush=True)
    photo_path, err = capture_with_retry()
    if err:
        print(f"[vision] Warmup capture failed: {err}", flush=True)
        write_result(None, "patrol", None, error=err)
    else:
        desc, elapsed, err = run_inference(photo_path)
        if err:
            print(f"[vision] Warmup inference failed ({elapsed:.1f}s): {err}", flush=True)
            write_result(None, "patrol", elapsed, error=err)
        else:
            print(f"[vision] Warmup OK ({elapsed:.1f}s): {desc[:80]}...", flush=True)
            write_result(desc, "patrol", elapsed)

    # Main loop
    cycle = 0
    while running:
        time.sleep(INTERVAL)
        if not running:
            break

        cycle += 1

        # Capture
        photo_path, err = capture_frame()
        if err:
            print(f"[vision] Capture failed: {err}", flush=True)
            write_result(None, DEFAULT_PROMPT, None, error=err)
            continue

        # Inference
        desc, elapsed, err = run_inference(photo_path, DEFAULT_PROMPT)
        if err:
            print(f"[vision] Inference failed ({elapsed:.1f}s): {err}", flush=True)
            write_result(None, DEFAULT_PROMPT, elapsed, error=err)
        else:
            preview = desc[:100].replace("\n", " ") if desc else "empty"
            print(f"[vision] #{cycle} ({elapsed:.1f}s): {preview}", flush=True)
            write_result(desc, DEFAULT_PROMPT, elapsed)

    print("[vision] Shutting down.", flush=True)


if __name__ == "__main__":
    main()
