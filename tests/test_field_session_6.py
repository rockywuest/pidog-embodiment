"""Round six of field findings (issue #45, log 11, 2026-10-30).

The fixed replies say "OK !" since round five, yet 'dakar' still showed up in
the voice log: in chat mode the LLM writes its own sentences and can pick
"d'accord" by itself. The word is now swapped for "OK" on the way to the TTS.
"""
import inspect

import brain.nox_voice_brain as vb


def test_d_accord_from_the_llm_becomes_ok():
    assert vb.echo_safe_speech("D'accord ! J'arrive.") == "OK ! J'arrive."
    assert vb.echo_safe_speech("d'accord") == "OK"


def test_typographic_apostrophe_and_case_are_covered():
    # LLMs like the curly apostrophe
    assert vb.echo_safe_speech("D’accord, je m’assois.") == "OK, je m’assois."
    assert vb.echo_safe_speech("D'ACCORD !") == "OK !"


def test_word_in_the_middle_of_a_sentence():
    assert vb.echo_safe_speech("Je suis d'accord avec toi.") == "Je suis OK avec toi."


def test_other_words_and_apostrophes_stay():
    for sentence in ("J'arrive tout de suite.", "Un accord parfait.", "Raccordé !",
                     "Sitz! Mach ich.", "On it!"):
        assert vb.echo_safe_speech(sentence) == sentence


def test_nothing_to_say_stays_nothing():
    assert vb.echo_safe_speech(None) is None
    assert vb.echo_safe_speech("") == ""


def test_sanitize_speech_itself_is_unchanged():
    # its job is stage directions and markdown — the apostrophe test of
    # round two keeps holding
    assert vb.sanitize_speech("D'accord ! J'arrive.") == "D'accord ! J'arrive."


def test_the_llm_reply_passes_through_the_swap_before_the_tts():
    src = inspect.getsource(vb.process_voice_intelligent)
    assert "echo_safe_speech(sanitize_speech(" in src
