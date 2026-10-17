#!/usr/bin/env python3
"""llm_check.py — can the brain reach its LLM, and how fast? (issue #45)

When the dog only ever answers "I'm still thinking, try again", the brain
reached the LLM endpoint and failed — and the reason sits in the brain's
journal, not in the voice log people usually watch. This tool asks the LLM
one question EXACTLY the way the brain does (same /etc/default/nox-brain,
same URL, model, timeout and prompt) and prints the answer, the time it
took, or the error together with how to fix it.

Run it on the machine where nox-brain runs (the env file is root-only):

    sudo python3 scripts/llm_check.py
    sudo python3 scripts/llm_check.py /path/to/other/env-file
"""
import os
import socket
import subprocess
import sys
import time
import urllib.parse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ENV = "/etc/default/nox-brain"
TEST_PHRASES = {"de": "Setz dich und wedel mit dem Schwanz",
                "fr": "Assieds-toi et remue la queue",
                "en": "Sit down and wag your tail"}


def read_env_file(path, environ=None):
    """KEY=VALUE pairs not already set in the environment (like systemd's
    EnvironmentFile, an exported variable wins). Raises OSError."""
    environ = os.environ if environ is None else environ
    loaded = {}
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
    return loaded


def probe_port(url, timeout=5.0):
    """Can we open a TCP connection to the endpoint at all? Separates "the
    machine never answers" (wrong address, firewall dropping packets) from
    "the model is slow" — both look like a timeout to the HTTP request.
    Returns None when the port answers, else the socket error."""
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as e:
        return e
    if not host:
        return ValueError(f"no host in {url!r}")
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return None
    except OSError as e:
        return e


def same_port_on_loopback(url):
    """The same endpoint on 127.0.0.1, or None when url already is local."""
    try:
        parsed = urllib.parse.urlparse(url)
        host, port = parsed.hostname, parsed.port
    except ValueError:
        return None
    if not host or host in ("127.0.0.1", "localhost", "::1") or not port:
        return None
    return parsed._replace(netloc=f"127.0.0.1:{port}").geturl()


def brain_service_state():
    try:
        out = subprocess.run(["systemctl", "is-active", "nox-brain"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def main(argv):
    env_path = argv[1] if len(argv) > 1 else DEFAULT_ENV
    try:
        loaded = read_env_file(env_path)
        os.environ.update(loaded)
        print(f"[llm-check] Read {len(loaded)} value(s) from {env_path}")
    except PermissionError:
        print(f"[llm-check] {env_path} is readable by root only — run:  "
              "sudo python3 scripts/llm_check.py")
        return 1
    except OSError:
        print(f"[llm-check] {env_path} not found — checking the environment "
              "as it is (scripts/install-brain.sh creates that file)")

    state = brain_service_state()
    if state not in ("active", "unknown"):
        print(f"[llm-check] Note: the nox-brain service is '{state}' on this "
              "machine. The dog talks to the brain named by BRAIN_HOST in "
              "its body/nox.env — make sure that is THIS machine.")

    # Import AFTER the env is in place: the brain reads OPENAI_URL, LLM_MODEL
    # and LLM_TIMEOUT at import time — this tool must see what the service sees.
    sys.path.insert(0, os.path.join(REPO, "brain"))
    import nox_voice_brain as vb

    print(f"[llm-check] URL:     {vb.OPENAI_URL}")
    print(f"[llm-check] Model:   {vb.OPENAI_MODEL}")
    print(f"[llm-check] Timeout: {vb.LLM_TIMEOUT}s   Mode: {vb.NOX_MODE}")
    if vb.NOX_MODE == "commands":
        print("[llm-check] NOX_MODE=commands — the brain never asks the LLM in "
              "this mode; set NOX_MODE=chat to use it.")
    if not vb.LLM_CONFIGURED:
        print("[llm-check] No LLM configured: set OPENAI_URL (local endpoint, "
              "e.g. http://127.0.0.1:11434/v1/chat/completions) or "
              f"OPENAI_API_KEY in {env_path}, then: sudo systemctl restart nox-brain")
        return 1

    problem = probe_port(vb.OPENAI_URL)
    if isinstance(problem, (socket.timeout, TimeoutError)):
        host = urllib.parse.urlparse(vb.OPENAI_URL).hostname
        print(f"[llm-check] FAILED: {host} does not answer at all (no reply "
              "within 5s) — wrong IP address, the machine is off, or a "
              "firewall silently drops the port.")
        return 1
    if problem is not None:
        local = same_port_on_loopback(vb.OPENAI_URL)
        if local and isinstance(problem, ConnectionRefusedError) and probe_port(local) is None:
            # Issue #45, log 10: the brain and Ollama ran on the SAME machine,
            # but OPENAI_URL used its LAN address, where Ollama does not listen.
            print(f"[llm-check] FAILED: {problem} — but Ollama DOES answer on this "
                  f"machine at 127.0.0.1. If nox-brain runs here too, simply use "
                  f"OPENAI_URL={local} in /etc/default/nox-brain (then: sudo "
                  "systemctl restart nox-brain). OLLAMA_HOST only matters when "
                  "the brain runs on a DIFFERENT machine.")
            return 1
        print(f"[llm-check] FAILED: {problem}{vb.llm_error_hint(problem)}")
        return 1

    phrase = TEST_PHRASES.get(vb.NOX_LANG, TEST_PHRASES["en"])
    print(f"[llm-check] Asking: '{phrase}' ...", flush=True)
    started = time.time()
    try:
        raw = vb.request_llm([{"role": "user", "content": phrase}])
    except Exception as e:
        took = time.time() - started
        if vb.is_timeout(e):
            print(f"[llm-check] FAILED: no answer within {vb.LLM_TIMEOUT}s. The "
                  "model may still be loading (first request after a start), or "
                  "the machine is too slow — try again, or raise LLM_TIMEOUT.")
        else:
            print(f"[llm-check] FAILED after {took:.1f}s: {e}{vb.llm_error_hint(e)}")
        return 1

    took = time.time() - started
    if not (raw or "").strip():
        print(f"[llm-check] FAILED: the model answered after {took:.1f}s, but with "
              f"NOTHING{vb.EMPTY_ANSWER_HINT} The dog would only say it is "
              "still thinking.")
        return 1
    parsed = vb.parse_response(raw or "")
    print(f"[llm-check] Answer in {took:.1f}s:")
    print(f"    raw:     {(raw or '')[:300]}")
    print(f"    speak:   {parsed.get('speak')!r}")
    print(f"    actions: {parsed.get('actions')}")
    if parsed.get("_no_json"):
        print("[llm-check] The model answered without the JSON the brain asks "
              "for — the dog will speak this but not move. Larger models "
              "(llama3.1:8b, qwen2.5:7b) follow the format more reliably.")
    if took > 15:
        print("[llm-check] Slow: the dog will be silent this long after every "
              "sentence. A smaller model, or a faster machine, helps.")
    print("[llm-check] OK — this brain reaches its LLM. If the dog still only "
          "says it is thinking, it talks to a DIFFERENT brain: check BRAIN_HOST "
          "in the robot's body/nox.env, then on the robot: "
          "sudo systemctl restart nox-bridge")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
