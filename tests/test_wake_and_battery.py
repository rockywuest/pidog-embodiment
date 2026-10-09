"""Wake word configurability and honest battery context — issue #45 follow-up.

The reporter's dog now holds French conversations, but waking it is a lottery
(French Vosk hears "Nox" as "inox"/"knox") and mid-answer it invented battery
drama: the LLM context said "Battery: errorV" whenever the daemon's ADC read
failed, and comparing that string raised TypeError in the sensor loop.
"""
import importlib

import pytest

import brain.nox_voice_brain as vb


# ─── battery ───

@pytest.mark.parametrize("value,expected", [
    (7.5, 7.5), (8, 8.0), ("error", None), (None, None), (True, None),
])
def test_numeric_battery(value, expected):
    assert vb.numeric_battery({"battery_v": value}) == expected


def test_numeric_battery_with_no_reading_at_all():
    assert vb.numeric_battery({}) is None
    assert vb.numeric_battery(None) is None


def test_an_unreadable_battery_never_reaches_the_llm(monkeypatch):
    seen = {}
    monkeypatch.setattr(vb, "bridge_get", lambda path, timeout=10: {
        "ok": True, "sensors": {"battery_v": "error"}})
    monkeypatch.setattr(vb, "bridge_post", lambda path, data, timeout=15: None)

    def fake_llm(messages):
        seen["prompt"] = messages[-1]["content"]
        return '{"speak":"Salut !","actions":[]}'

    monkeypatch.setattr(vb, "call_llm", fake_llm)
    vb.process_voice_intelligent({"text": "Bonjour !"})
    assert "error" not in seen["prompt"]
    assert "Battery" not in seen["prompt"]


def test_a_real_battery_reading_still_reaches_the_llm(monkeypatch):
    seen = {}
    monkeypatch.setattr(vb, "bridge_get", lambda path, timeout=10: {
        "ok": True, "sensors": {"battery_v": 7.9}})
    monkeypatch.setattr(vb, "bridge_post", lambda path, data, timeout=15: None)
    monkeypatch.setattr(vb, "call_llm",
                        lambda m: seen.update(prompt=m[-1]["content"]) or '{"speak":"!","actions":[]}')
    vb.process_voice_intelligent({"text": "Bonjour !"})
    assert "Battery: 7.9V" in seen["prompt"]


# ─── wake word ───

def load_v2(monkeypatch, wake=None):
    if wake is None:
        monkeypatch.delenv("WAKE_WORD", raising=False)
    else:
        monkeypatch.setenv("WAKE_WORD", wake)
    import body.nox_voice_loop_v2 as v2
    return importlib.reload(v2)


@pytest.fixture(autouse=True)
def _restore_v2(monkeypatch):
    yield
    monkeypatch.delenv("WAKE_WORD", raising=False)
    import body.nox_voice_loop_v2 as v2
    importlib.reload(v2)


@pytest.mark.parametrize("heard", ["nox assis", "knox assis", "inox assis"])
def test_default_wake_word_hears_its_mistranscriptions(monkeypatch, heard):
    v2 = load_v2(monkeypatch)
    cleaned, woke = v2.fuzzy_wake_word_check(heard)
    assert woke and cleaned == "assis"


def test_a_custom_wake_word_works_with_edit_distance(monkeypatch):
    v2 = load_v2(monkeypatch, "milou")
    assert v2.fuzzy_wake_word_check("milou assis")[1] is True
    assert v2.fuzzy_wake_word_check("milu assis")[1] is True      # distance 1
    assert v2.fuzzy_wake_word_check("nox assis")[1] is False      # old word off
    assert v2.fuzzy_wake_word_check("bonjour milou couché") == ("couché", True)


def test_unrelated_speech_does_not_wake(monkeypatch):
    v2 = load_v2(monkeypatch)
    assert v2.fuzzy_wake_word_check("il fait beau aujourd'hui")[1] is False
