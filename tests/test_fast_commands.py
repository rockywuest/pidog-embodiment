"""A plain command skips the LLM (field: 2-4 s between "Sitz" and the paws)."""
import pytest

import brain.nox_voice_brain as vb


@pytest.mark.parametrize("text, actions", [
    ("Sitz", ["sit"]),
    ("Nox, setz dich bitte hin", ["sit"]),
    ("Platz!", ["lie"]),
    ("stopp", ["stand"]),
    ("Setz dich hin und wedel mit dem Schwanz!", ["sit", "wag_tail"]),
    ("dreh dich nach rechts", ["turn_right"]),
    ("sit down please", ["sit"]),
    ("assis-toi", ["sit"]),
])
def test_plain_commands_are_recognised(text, actions):
    assert vb.plain_command(text) == actions


@pytest.mark.parametrize("text", [
    "Was siehst du links?",
    "setz dich und sag mir was du siehst",
    "kannst du dich hinsetzen",
    "geh nach links zum Sofa",
    "wie geht es dir",
    "good dog",
    "",
])
def test_anything_more_than_a_command_is_not(text):
    assert vb.plain_command(text) is None


def _run(monkeypatch, text):
    sent, llm = [], []
    monkeypatch.setattr(vb, "bridge_post", lambda p, d, timeout=15: sent.append((p, d)))
    monkeypatch.setattr(vb, "bridge_get", lambda p, timeout=10: {"ok": True, "sensors": {}})
    monkeypatch.setattr(vb, "call_llm", lambda messages: llm.append(messages) or '{"speak":"Hallo","actions":[]}')
    vb.process_voice_intelligent({"text": text})
    return sent, llm


def test_a_plain_command_goes_straight_to_the_body(monkeypatch):
    sent, llm = _run(monkeypatch, "Nox, Platz")
    assert not llm
    assert sent == [("/combo", {"actions": ["lie"]})]


def test_a_question_still_goes_to_the_llm(monkeypatch):
    sent, llm = _run(monkeypatch, "Was siehst du?")
    assert llm and sent[-1][1].get("speak") == "Hallo"


def test_the_fast_path_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(vb, "FAST_COMMANDS", False)
    _, llm = _run(monkeypatch, "Sitz")
    assert llm
