"""Second, grammar-restricted decode for commands — trusted only when confident.
Confidences below are the ones measured on the dog (Pi 4, small German model)."""
import body.nox_voice_loop_v2 as v2


def _res(*pairs):
    return {"result": [{"word": w, "conf": c} for w, c in pairs]}


def test_a_confident_command_is_taken():
    assert v2.confident_grammar_text(_res(("nox", 1.0), ("leg", 1.0), ("dich", 1.0), ("hin", 1.0))) \
        == "nox leg dich hin"


def test_a_question_forced_into_commands_is_refused():
    # "Nox, was siehst du gerade vor dir?" -> grammar: nox sitz dreh hallo geh
    assert v2.confident_grammar_text(_res(("nox", 1.0), ("sitz", 0.83), ("dreh", 0.72),
                                          ("hallo", 1.0), ("geh", 0.47))) is None


def test_unknown_words_or_nothing_mean_no_grammar_result():
    assert v2.confident_grammar_text(_res(("[unk]", 1.0), ("sitz", 1.0))) is None
    assert v2.confident_grammar_text({"result": []}) is None
    assert v2.confident_grammar_text({}) is None


def test_language_comes_from_the_model_name():
    assert v2.model_language("/home/pidog/vosk-models/vosk-model-small-de-0.15") == "de"
    assert v2.model_language("/x/vosk-model-small-fr-0.22/") == "fr"
    assert v2.model_language("/x/vosk-model-en-us-0.22") == "en"
    assert v2.model_language("/x/my-model") is None


def test_the_grammar_knows_the_wake_words_and_unk():
    g = v2.command_grammar("de")
    assert "sitz" in g and v2.WAKE_WORD in g and "[unk]" in g
    assert v2.command_grammar(None) is None
