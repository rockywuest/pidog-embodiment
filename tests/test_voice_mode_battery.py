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
    assert sent == [("/combo", {"actions": ["sit"], "speak": "OK !"})]


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
    # Debounced: one low reading is a servo-load sag (a 7.4 V pack logged
    # 6.56 V under load, issue #45) — only the second in a row warns.
    assert w.should_warn(6.7, t) is False
    assert w.should_warn(6.7, t + 5) is True
    # sagging under load, recovering, sagging — the old code warned each time
    for dv, dt in [(7.1, 10), (6.7, 20), (7.05, 30), (6.6, 40)]:
        assert w.should_warn(dv, t + dt) is False
    # even after a real reset above the wide threshold, the cooldown holds
    assert w.should_warn(7.5, t + 50) is False
    assert w.should_warn(6.7, t + 60) is False
    assert w.should_warn(6.7, t + 75) is False
    # after the cooldown AND a reset, two consecutive lows may warn again
    assert w.should_warn(7.5, t + 901) is False
    assert w.should_warn(6.7, t + 902) is False
    assert w.should_warn(6.7, t + 917) is True


def test_a_single_load_sag_never_warns():
    w = vb.BatteryWatch(low=6.8, cooldown_s=900)
    # low, recovered, low, recovered — isolated sags, never two in a row
    for v, t in [(6.6, 0), (7.3, 15), (6.5, 30), (7.3, 45), (6.4, 60), (7.2, 75)]:
        assert w.should_warn(v, t) is False


def test_battery_unreadable_never_warns():
    w = vb.BatteryWatch()
    assert w.should_warn(None, 0) is False


def test_battery_threshold_from_env(monkeypatch):
    monkeypatch.setenv("BATTERY_LOW_V", "7.2")
    w = importlib.reload(vb).BatteryWatch()
    assert w.low == 7.2 and w.reset_above == 7.8


@pytest.mark.parametrize("raw,expected", [("6.8V", 6.8), ("7,0", 7.0), ("high", 6.8)])
def test_battery_threshold_survives_config_typos(monkeypatch, raw, expected):
    monkeypatch.setenv("BATTERY_LOW_V", raw)
    assert importlib.reload(vb).BatteryWatch().low == expected


def test_brain_stays_silent_while_the_body_handles_battery(monkeypatch):
    """Body and brain both spoke at 6.8 V — two voices, one battery (#47-#58 review)."""
    sent = []
    responses = {"/status": {"ok": True, "sensors": {"battery_v": 6.5},
                             "behavior": {"low_battery": True}}}
    monkeypatch.setattr(vb, "bridge_get", lambda p, timeout=10: responses.get(p, {}))
    monkeypatch.setattr(vb, "bridge_post", lambda p, d, timeout=15: sent.append(p))
    w = vb.BatteryWatch(low=6.8, cooldown_s=900)
    status = responses["/status"]
    behavior = status.get("behavior") or {}
    body_handles = bool(behavior.get("low_battery") or behavior.get("critical_battery"))
    assert body_handles is True
    # the deferred path must NOT burn the brain's cooldown: when the engine
    # later stops, the brain must be able to warn immediately
    if not body_handles:
        w.should_warn(6.5, 100)
    assert w.warned is False and w.last_warn_at == float("-inf")  # pristine: never warned
    # engine gone → brain speaks on the next confirmed (second) low reading
    assert w.should_warn(6.5, 101) is False
    assert w.should_warn(6.5, 116) is True


def test_brain_warns_when_no_engine_is_on_the_case():
    behavior = {}  # NOX_NO_AUTO=1: no behavior engine → brain owns the warning
    assert not (behavior.get("low_battery") or behavior.get("critical_battery"))
