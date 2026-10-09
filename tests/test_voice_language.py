"""Which language the dog speaks — issue #42 follow-up.

The reporter's dog, with a French voice and a local Ollama brain, answered
"stand and bark" in German: the prompt asked for German replies and for the
user's language at once, and every example answered an English question in
German. Battery warnings were hard-coded German too.
"""
import importlib

import pytest

import body.nox_audio as audio


@pytest.fixture(autouse=True)
def _restore_brain_module(monkeypatch):
    """Reloading with NOX_LANG=fr must not leak into other test files."""
    yield
    monkeypatch.delenv("NOX_LANG", raising=False)
    import brain.nox_voice_brain as vb
    importlib.reload(vb)


def load_brain(monkeypatch, lang=None):
    if lang is None:
        monkeypatch.delenv("NOX_LANG", raising=False)
    else:
        monkeypatch.setenv("NOX_LANG", lang)
    import brain.nox_voice_brain as vb
    return importlib.reload(vb)


def test_prompt_no_longer_demands_german(monkeypatch):
    vb = load_brain(monkeypatch)
    assert "German, will be read aloud" not in vb.SYSTEM_PROMPT
    assert "language of the user's message" in vb.SYSTEM_PROMPT
    # an English request answered in English among the examples
    assert '"stand up and bark"\n{"speak":"Woof! Here I am!"' in vb.SYSTEM_PROMPT


def test_fixed_language_is_stated_in_the_prompt(monkeypatch):
    vb = load_brain(monkeypatch, "fr")
    assert "Always answer in French" in vb.SYSTEM_PROMPT
    assert vb.reply_language("stand and bark") == "fr"


def test_unknown_nox_lang_falls_back_to_auto(monkeypatch):
    vb = load_brain(monkeypatch, "klingon")
    assert vb.NOX_LANG == "auto"


def test_auto_follows_the_speaker_and_remembers(monkeypatch):
    vb = load_brain(monkeypatch)
    assert vb.reply_language("Assieds-toi et remue la queue") == "fr"
    assert vb.reply_language() == "fr"  # e.g. the battery warning afterwards
    assert vb._REPLIES["battery"]["fr"].startswith("Attention")


def test_simple_mode_answers_english_in_english(monkeypatch):
    vb = load_brain(monkeypatch)
    sent = []
    monkeypatch.setattr(vb, "bridge_post", lambda path, data, timeout=15: sent.append(data))
    vb.process_voice_simple({"text": "stand and bark"})
    assert sent == [{"actions": ["stand", "bark"], "speak": "On it!"}]


@pytest.mark.parametrize("env,lang", [
    ({"NOX_LANG": "en", "PIPER_MODEL": "/v/fr_FR-siwis-medium.onnx"}, "en"),
    ({"PIPER_MODEL": "/home/cat1/.piper_models/fr_FR-siwis-medium.onnx"}, "fr"),
    ({"PIPER_MODEL": "/x/en_US-amy-low.onnx"}, "en"),
    ({"PIPER_MODEL": "/x/de_DE-thorsten-high.onnx"}, "de"),
    ({}, "de"),
    ({"PIPER_MODEL": "/x/nl_NL-mls-medium.onnx"}, "de"),
])
def test_body_speaks_the_language_of_its_voice(env, lang):
    assert audio.body_language(env) == lang


def test_battery_warning_in_french_with_a_french_voice():
    env = {"PIPER_MODEL": "/v/fr_FR-siwis-medium.onnx"}
    assert audio.spoken("battery_critical", env) == "Batterie critique. Je me couche."
