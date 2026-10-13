"""Round two of field findings from issue #45 (logs 5 and 6, 2026-10-13).

The reporter now holds real conversations with the dog — these pin what his
logs showed still going wrong: a dead LLM muted even keyword commands, Piper
read roleplay asterisks aloud ("astérisque remue la queue astérisque"), the
blind 8 s window ate his real follow-ups, and his quiet VoiceHAT needed a
tunable energy threshold. Plus his feature ask: several wake words.
"""
import importlib

import pytest

import body.nox_voice_loop_v2 as v2
import brain.nox_voice_brain as vb


# ─── several wake words (WAKE_WORD=milou,tintin) ───

@pytest.fixture
def wake(monkeypatch):
    def load(value=None):
        if value is None:
            monkeypatch.delenv("WAKE_WORD", raising=False)
        else:
            monkeypatch.setenv("WAKE_WORD", value)
        return importlib.reload(v2)
    yield load
    monkeypatch.delenv("WAKE_WORD", raising=False)
    importlib.reload(v2)


def test_each_listed_wake_word_wakes_the_dog(wake):
    m = wake("milou,tintin")
    assert m.fuzzy_wake_word_check("milou assis") == ("assis", True)
    assert m.fuzzy_wake_word_check("tintin couché") == ("couché", True)
    assert m.fuzzy_wake_word_check("nox assis")[1] is False


def test_fuzzy_and_prefix_matching_cover_every_listed_word(wake):
    m = wake("milou, tintin")  # spaces around the comma must not matter
    assert m.fuzzy_wake_word_check("tintan debout")[1] is True   # distance 1
    assert m.fuzzy_wake_word_check("hey tintin marche") == ("marche", True)
    assert m.fuzzy_wake_word_check("bonjour milu viens")[1] is True


def test_a_single_wake_word_behaves_as_before(wake):
    m = wake("milou")
    assert m.WAKE_WORDS == ["milou"] and m.WAKE_WORD == "milou"
    assert m.fuzzy_wake_word_check("milou assis") == ("assis", True)


def test_default_and_degenerate_values_fall_back_to_nox(wake):
    assert wake().WAKE_WORDS == ["nox"]
    assert wake(" , ,").WAKE_WORDS == ["nox"]  # only separators → default
    m = wake("nox,milou")
    assert m.fuzzy_wake_word_check("knox assis")[1] is True  # nox variants kept


# ─── TTS never reads roleplay aloud ───

def test_stage_directions_are_stripped_before_tts():
    # Piper read this aloud as "astérisque remue la queue ... astérisque"
    assert vb.sanitize_speech("Oh oui *remue la queue avec enthousiasme* !") == "Oh oui !"
    assert vb.sanitize_speech("*wags tail* Here I am!") == "Here I am!"


def test_a_pure_stage_direction_leaves_nothing_to_say():
    assert vb.sanitize_speech("*remue la queue*") is None
    assert vb.sanitize_speech("") is None
    assert vb.sanitize_speech(None) is None


def test_normal_speech_passes_untouched():
    assert vb.sanitize_speech("D'accord ! J'arrive.") == "D'accord ! J'arrive."


def test_stray_markdown_is_cleaned_not_spoken():
    assert vb.sanitize_speech("Je suis **très** content") == "Je suis très content"


# ─── a dead LLM must not mute keyword commands ───

def test_llm_down_still_executes_keyword_commands(monkeypatch):
    sent = []
    monkeypatch.setattr(vb, "call_llm", lambda messages: None)
    monkeypatch.setattr(vb, "bridge_get", lambda p, timeout=10: {"ok": True, "sensors": {}})
    monkeypatch.setattr(vb, "bridge_post", lambda p, d, timeout=15: sent.append((p, d)))
    vb.process_voice_intelligent({"text": "debout"})
    paths = [p for p, _ in sent]
    assert "/combo" in paths
    combo = dict(sent)["/combo"]
    assert combo["actions"] == ["stand"]


def test_llm_down_with_no_command_still_says_thinking(monkeypatch):
    sent = []
    monkeypatch.setattr(vb, "call_llm", lambda messages: None)
    monkeypatch.setattr(vb, "bridge_get", lambda p, timeout=10: {"ok": True, "sensors": {}})
    monkeypatch.setattr(vb, "bridge_post", lambda p, d, timeout=15: sent.append((p, d)))
    vb.process_voice_intelligent({"text": "raconte une histoire"})
    paths = [p for p, _ in sent]
    assert "/speak" in paths and "/combo" not in paths


# ─── the quiet-mic knob and the shrunken blind window ───

def test_energy_threshold_comes_from_nox_env(monkeypatch):
    monkeypatch.setenv("ENERGY_THRESHOLD", "300")
    m = importlib.reload(v2)
    try:
        assert m.ENERGY_THRESHOLD == 300
    finally:
        monkeypatch.delenv("ENERGY_THRESHOLD")
        importlib.reload(v2)


def test_garbage_energy_threshold_falls_back(monkeypatch):
    monkeypatch.setenv("ENERGY_THRESHOLD", "loud")
    m = importlib.reload(v2)
    try:
        assert m.ENERGY_THRESHOLD == 500
    finally:
        monkeypatch.delenv("ENERGY_THRESHOLD")
        importlib.reload(v2)


def test_the_blind_echo_window_stays_short():
    """8 s of deafness after every command ate the user's real follow-ups
    ('cathy assis', 'remue la queue' — issue #45 log 6). Real TTS echo is the
    daemon speech_state's job now; this window only covers motion noise."""
    src = open("body/nox_voice_loop_v2.py").read()
    assert "time.time() + 8.0" not in src
    assert "time.time() + 2.5" in src
