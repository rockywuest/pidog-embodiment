"""Second, grammar-restricted decode for commands — trusted only when confident.
Confidences below are the ones measured on the dog (Pi 4, small German model)."""
import body.nox_voice_loop_v2 as v2


def _res(*pairs):
    return {"result": [{"word": w, "conf": c} for w, c in pairs]}


def test_a_confident_command_is_taken():
    assert v2.confident_grammar_text(_res(("komm", 1.0), ("her", 1.0), ("und", 1.0), ("wedel", 1.0),
                                          ("mit", 1.0), ("dem", 1.0), ("schwanz", 1.0)),
                                     "komm her und weder mit dem schwanz") \
        == "komm her und wedel mit dem schwanz"


def test_a_question_forced_into_commands_is_refused():
    # "Nox, was siehst du gerade vor dir?" -> grammar: nox sitz dreh hallo geh
    assert v2.confident_grammar_text(_res(("nox", 1.0), ("sitz", 0.83), ("dreh", 0.72),
                                          ("hallo", 1.0), ("geh", 0.47)),
                                     "nox was siehst du gerade") is None


def test_a_reading_that_dropped_words_is_refused():
    # out-of-grammar words are not emitted: "nox wie spät ist es" -> "nox"
    assert v2.confident_grammar_text(_res(("nox", 1.0)), "nox wie spät ist es") is None
    assert v2.confident_grammar_text(_res(("nox", 1.0), ("komm", 1.0), ("her", 1.0)),
                                     "nox komm her und schau") is None


def test_unknown_words_or_nothing_mean_no_grammar_result():
    assert v2.confident_grammar_text(_res(("[unk]", 1.0), ("sitz", 1.0)), "was sitz") is None
    assert v2.confident_grammar_text({"result": []}, "x") is None
    assert v2.confident_grammar_text({}, "x") is None


def test_the_grammar_replaces_text_only_after_the_echo_filters():
    src = open("body/nox_voice_loop_v2.py").read()
    assert src.index("is_own_speech(better, speech_state)") < src.index("text = better")
    assert "del utterance[:-GRAMMAR_MAX_CHUNKS]" in src


def test_language_comes_from_the_model_name():
    assert v2.model_language("/home/pidog/vosk-models/vosk-model-small-de-0.15") == "de"
    assert v2.model_language("/x/vosk-model-small-fr-0.22/") == "fr"
    assert v2.model_language("/x/vosk-model-en-us-0.22") == "en"
    assert v2.model_language("/x/my-model") is None


def test_the_grammar_knows_the_wake_words_and_unk():
    g = v2.command_grammar("de")
    assert "sitz" in g and v2.WAKE_WORD in g and "[unk]" in g
    assert v2.command_grammar(None) is None
