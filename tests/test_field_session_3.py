"""Round three of field findings from issue #45 (log 7, 2026-10-15).

The reporter's dog "Cathy" introduced itself as "Nox" (Piper even said
"knox"), his remote-Ollama brain looked dead while `ollama run` worked fine
on that machine (OLLAMA_HOST), and every gait crawled at the old fixed
speed. These tests pin the fixes: ROBOT_NAME, the unreachable-Ollama hint,
and the WALK_SPEED default for gaits.
"""
import importlib

import pytest

import brain.nox_voice_brain as vb
from body.nox_audio import env_int


# ─── the dog is called what its owner called it ───

@pytest.fixture
def named(monkeypatch):
    def load(value=None):
        if value is None:
            monkeypatch.delenv("ROBOT_NAME", raising=False)
        else:
            monkeypatch.setenv("ROBOT_NAME", value)
        return importlib.reload(vb)
    yield load
    monkeypatch.delenv("ROBOT_NAME", raising=False)
    importlib.reload(vb)


def test_robot_name_reaches_prompt_and_canned_replies(named):
    m = named("Cathy")
    assert m.SYSTEM_PROMPT.startswith("You are Cathy, an AI robot dog")
    assert m._REPLIES["who"]["fr"] == "Je suis Cathy !"
    assert m._REPLIES["who"]["de"] == "Ich bin Cathy!"
    assert "Nox" not in m._REPLIES["who"]["en"]


def test_default_and_blank_name_stay_nox(named):
    assert named().ROBOT_NAME == "Nox"
    m = named("   ")
    assert m.ROBOT_NAME == "Nox"
    assert m.SYSTEM_PROMPT.startswith("You are Nox,")


# ─── "connection refused" against a remote Ollama names OLLAMA_HOST ───
# (the full hint matrix lives in test_field_session_4.py)

def test_refused_remote_ollama_gets_the_hint():
    hint = vb.llm_error_hint(
        "[Errno 111] Connection refused",
        "http://192.0.2.1:11434/v1/chat/completions")
    assert "OLLAMA_HOST=0.0.0.0" in hint and "11434" in hint


def test_only_remote_ollama_is_told_about_ollama_host():
    refused = "[Errno 111] Connection refused"
    # Local Ollama refusing = Ollama not running; OLLAMA_HOST is irrelevant.
    for url in ("http://127.0.0.1:11434/v1/chat/completions",
                "http://localhost:11434/v1/chat/completions",
                "https://api.openai.com/v1/chat/completions"):
        assert "OLLAMA_HOST" not in vb.llm_error_hint(refused, url)


def test_other_errors_get_no_hint():
    url = "http://192.0.2.1:11434/v1/chat/completions"
    assert vb.llm_error_hint("timed out", url) == ""
    assert vb.llm_error_hint("HTTP Error 500: boom", url) == ""


# ─── gaits walk at WALK_SPEED, tunable from nox.env ───

def test_env_int_parses_numbers_and_survives_garbage(capsys):
    assert env_int("X", 95, {"X": "80"}) == 80
    assert env_int("X", 95, {"X": "80.0"}) == 80
    assert env_int("X", 95, {"X": "80,5"}) == 80
    assert env_int("X", 95, {}) == 95
    assert env_int("X", 95, {"X": "fast"}) == 95
    assert "fast" in capsys.readouterr().out


def test_daemon_defaults_gaits_to_walk_speed():
    # nox_daemon needs the SunFounder SDK — source-level, like its other tests.
    src = open("body/nox_daemon.py").read()
    assert 'WALK_SPEED = max(1, min(100, env_int("WALK_SPEED", 95)))' in src
    assert "def cmd_move(action, steps=3, speed=None" in src
    assert "speed = WALK_SPEED if action in GAIT_ACTIONS else 80" in src
    # The dispatch table must not reimpose a fixed default.
    assert 'args.get("speed", 80)' not in src


def test_no_caller_pins_gaits_to_the_old_crawl():
    """Issue #45 log 7: "avance et recul mouvements tres lents, comme en
    patrol" — the engine and the bridge hardcoded 60-70 for every gait,
    so WALK_SPEED could never reach the servos."""
    for path in ("body/nox_behavior_engine.py", "body/nox_brain_bridge.py",
                 "body/nox_autonomous_v2.py"):
        src = open(path).read()
        for gait in ("forward", "backward", "turn_left", "turn_right"):
            for line in src.splitlines():
                if f'"action": "{gait}"' in line:
                    assert '"speed"' not in line, f"{path}: {line.strip()}"
