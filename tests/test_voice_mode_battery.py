"""Commands-only mode and the battery warning loop — issue #45 field report.

The reporter's room has a TV: mid-"conversation" the LLM received noise
fragments and chatted about Dakar, and the battery warning repeated every
two minutes because a pack sagging under load crossed the narrow 6.8/7.0
band over and over.
"""
import importlib

import pytest

import brain.nox_voice_brain as vb


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    yield
    for k in ("NOX_MODE", "BATTERY_LOW_V"):
        monkeypatch.delenv(k, raising=False)
    importlib.reload(vb)


def test_commands_mode_executes_commands(monkeypatch):
    sent = []
    monkeypatch.setattr(vb, "bridge_post", lambda p, d, timeout=15: sent.append((p, d)))
    vb.process_voice_commands({"text": "Assis !"})
    assert sent == [("/combo", {"actions": ["sit"], "speak": "D'accord !"})]


def test_commands_mode_ignores_chatter_instead_of_answering(monkeypatch):
    sent = []
    monkeypatch.setattr(vb, "bridge_post", lambda p, d, timeout=15: sent.append((p, d)))
    vb.process_voice_commands({"text": "tu as besoin d'aide à régler mon fils"})
    assert sent == []  # no LLM, no TTS reply — silence is the feature


def test_nox_mode_env_selects_commands(monkeypatch):
    monkeypatch.setenv("NOX_MODE", "commands")
    m = importlib.reload(vb)
    assert m.NOX_MODE == "commands"
    monkeypatch.setenv("NOX_MODE", "quatsch")
    m = importlib.reload(vb)
    assert m.NOX_MODE == "chat"


# ─── battery ───

def test_battery_warns_once_not_in_a_loop():
    w = vb.BatteryWatch(low=6.8, cooldown_s=900)
    t = 1000.0
    assert w.should_warn(6.7, t) is True
    # sagging under load, recovering, sagging — the old code warned each time
    for dv, dt in [(7.1, 10), (6.7, 20), (7.05, 30), (6.6, 40)]:
        assert w.should_warn(dv, t + dt) is False
    # even after a real reset above the wide threshold, the cooldown holds
    assert w.should_warn(7.5, t + 50) is False
    assert w.should_warn(6.7, t + 60) is False
    # after the cooldown AND a reset, it may warn again
    assert w.should_warn(7.5, t + 901) is False
    assert w.should_warn(6.7, t + 902) is True


def test_battery_unreadable_never_warns():
    w = vb.BatteryWatch()
    assert w.should_warn(None, 0) is False


def test_battery_threshold_from_env(monkeypatch):
    monkeypatch.setenv("BATTERY_LOW_V", "7.2")
    w = importlib.reload(vb).BatteryWatch()
    assert w.low == 7.2 and w.reset_above == 7.8
