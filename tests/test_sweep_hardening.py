"""Guards from the 2026-10-12 full-repo sweep (after issue #45).

Two classes of field bug kept resurfacing: hardcoded German reaching TTS on
non-German robots, and author-setup constants (battery thresholds) without an
env override. These tests make both classes impossible to reintroduce quietly.
"""
import importlib
import re

import pytest

import body.nox_audio as na
import body.nox_autonomous_v2 as av
import body.nox_behavior_engine as be

# Active body files whose speak calls reach TTS. Legacy files (voice loop v3,
# the old relay) are not installed and not held to this.
ACTIVE_BODY_FILES = [
    "body/nox_daemon.py", "body/nox_brain_bridge.py", "body/nox_behavior_engine.py",
    "body/nox_autonomous_v2.py", "body/nox_voice_loop_v2.py",
]


# ─── no hardcoded spoken strings in active code ───

def test_no_speak_call_hardcodes_its_text():
    """Every spoken sentence must come from nox_audio.SPOKEN (via spoken()) or
    a variable — a string literal in a speak call is how "Ja? Was ist?" and
    "Meine Batterie wird schwach" shipped to French robots."""
    offenders = []
    for path in ACTIVE_BODY_FILES:
        src = open(path).read()
        for pattern in (r'"cmd":\s*"speak",\s*"text":\s*"([^"]+)"',
                        r'speak_via_daemon\(\s*"([^"]+)"'):
            offenders += [f"{path}: {m}" for m in re.findall(pattern, src)]
    assert offenders == []


def test_env_volts_parses_field_typos():
    assert na.env_volts("X", 6.8, {"X": "6,5"}) == 6.5
    assert na.env_volts("X", 6.8, {"X": "6.5V"}) == 6.5
    assert na.env_volts("X", 6.8, {"X": "7 v"}) == 7.0
    assert na.env_volts("X", 6.8, {}) == 6.8
    assert na.env_volts("X", 6.8, {"X": "low"}) == 6.8  # garbage → default


# ─── battery thresholds are per-robot settings on the BODY too ───
# The brain honors BATTERY_LOW_V and defers to the body when the body handles
# low battery itself — so a threshold only the brain can change is useless.

@pytest.fixture
def reload_battery_modules(monkeypatch):
    def load(low=None, critical=None):
        for name, value in (("BATTERY_LOW_V", low), ("BATTERY_CRITICAL_V", critical)):
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        return importlib.reload(be), importlib.reload(av)
    yield load
    monkeypatch.delenv("BATTERY_LOW_V", raising=False)
    monkeypatch.delenv("BATTERY_CRITICAL_V", raising=False)
    importlib.reload(be)
    importlib.reload(av)


def test_battery_thresholds_default_to_2s_lipo(reload_battery_modules):
    engine, auto = reload_battery_modules()
    assert engine.BATTERY_LOW_V == 6.8 and engine.BATTERY_CRITICAL_V == 6.2
    assert auto.BATTERY_LOW_V == 6.8


def test_battery_thresholds_come_from_nox_env(reload_battery_modules):
    engine, auto = reload_battery_modules(low="6.5", critical="6.0")
    assert engine.BATTERY_LOW_V == 6.5 and engine.BATTERY_CRITICAL_V == 6.0
    assert auto.BATTERY_LOW_V == 6.5  # the engine's fallback uses the same value


def test_inverted_thresholds_fall_back_to_defaults(reload_battery_modules):
    engine, _ = reload_battery_modules(low="6.0", critical="6.5")
    assert engine.BATTERY_LOW_V == 6.8 and engine.BATTERY_CRITICAL_V == 6.2


def test_the_fallback_speaks_the_localized_battery_phrase():
    src = open("body/nox_autonomous_v2.py").read()
    assert 'spoken("battery_low")' in src
    assert "Meine Batterie" not in src


# ─── duplicated constants must not drift ───

def test_daemon_and_bridge_ports_agree_across_all_body_files():
    """DAEMON_PORT/BRIDGE_PORT are defined independently in several body files
    (and the brain reaches them via env-overridable defaults). Editing one
    occurrence must fail loudly here, not silently split the services."""
    import glob
    ports = {"DAEMON_PORT": set(), "BRIDGE_PORT": set()}
    for path in glob.glob("body/*.py"):
        src = open(path).read()
        for name in ports:
            ports[name] |= {int(v) for v in re.findall(rf"^{name} = (\d+)", src, re.M)}
    assert ports["DAEMON_PORT"] == {9999}
    assert ports["BRIDGE_PORT"] == {8888}


def test_examples_speak_english():
    src = open("examples/multi_body.py").read()
    for german in ("Ich wechsle", "Jetzt bin ich", "Ich bin bereit"):
        assert german not in src
