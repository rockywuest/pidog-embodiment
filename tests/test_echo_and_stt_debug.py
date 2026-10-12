"""Self-echo suppression and the STT debug tool — issue #45, voiceInput3 log.

The reporter's dog answered itself: the brain's reply ("J'ai entendu ...") and
the battery warning arrived AFTER the fixed 8 s echo window, were picked up by
the microphone and processed as new commands — a conversation loop with one
participant. The daemon now remembers what it spoke; the voice loop drops any
phrase that is mostly the dog's own recent words. The test phrases below are
the literal lines from the reporter's journal.
"""
import importlib.util
import re

import body.nox_voice_loop_v2 as v2
from body.nox_audio import SPOKEN, spoken


def state(speaking=False, recent=()):
    return {"ok": True, "speaking": speaking, "recent": list(recent)}


# ─── self-echo detection ───

def test_the_battery_warning_heard_back_is_an_echo():
    # 12:52:18 Heard: 'attention ma batterie est presque vide' — the dog's own
    # announcement, spoken seconds earlier, fed back in as a command.
    s = state(recent=[{"text": "Attention ! Ma batterie est presque vide !", "age_s": 4.1}])
    assert v2.is_own_speech("attention ma batterie est presque vide", s) is True


def test_the_brains_filler_reply_heard_back_is_an_echo():
    # 12:52:57 — the dog processed its own "thinking" reply, imperfectly
    # transcribed ("rester" for "réessaie"); the overlap must still win.
    s = state(recent=[{"text": "J'ai entendu : ne vois-tu. Je réfléchis encore, "
                               "réessaie dans un instant.", "age_s": 3.0}])
    heard = "j'ai entendu ne vois-tu je réfléchis encore rester dans un instant"
    assert v2.is_own_speech(heard, s) is True


def test_speaking_right_now_suppresses_everything():
    assert v2.is_own_speech("assis", state(speaking=True)) is True


def test_a_short_real_command_is_never_eaten():
    # The dog said "Batterie critique. Je me couche." — the user saying
    # "couche-toi" right after must still work: too short to be an echo.
    s = state(recent=[{"text": "Batterie critique. Je me couche.", "age_s": 2.0}])
    assert v2.is_own_speech("couche-toi", s) is False


def test_genuinely_new_speech_passes():
    s = state(recent=[{"text": "Attention ! Ma batterie est presque vide !", "age_s": 2.0}])
    assert v2.is_own_speech("milou viens ici tout de suite", s) is False


def test_old_speech_is_no_longer_an_echo():
    s = state(recent=[{"text": "Attention ! Ma batterie est presque vide !",
                       "age_s": v2.ECHO_TEXT_WINDOW + 5}])
    assert v2.is_own_speech("attention ma batterie est presque vide", s) is False


def test_an_old_daemon_without_speech_state_changes_nothing():
    # Unknown command → {"error": ...}; daemon down → {"error": ...}.
    for reply in ({"error": "unknown command: speech_state"}, {"error": "timeout"},
                  None, {}):
        assert v2.is_own_speech("attention ma batterie est presque vide", reply) is False


def test_matching_ignores_accents_and_punctuation():
    s = state(recent=[{"text": "Oui ? Qu'est-ce qu'il y a ?", "age_s": 1.0}])
    assert v2.is_own_speech("oui qu'est-ce qu'il y a", s) is True


# ─── the daemon side (source-level: nox_daemon needs the SunFounder SDK) ───

def test_daemon_registers_speech_state_and_records_speech():
    src = open("body/nox_daemon.py").read()
    assert re.search(r'"speech_state":\s*lambda', src)
    assert "_recent_speech.append(entry)" in src
    # the end timestamp lands in a finally: a crashed TTS must not report
    # "speaking" forever
    assert re.search(r"finally:\s*\n\s*entry\[\"end\"\] = time\.time\(\)", src)


# ─── localized body phrases ("ja ossist?" was German TTS in a French voice) ───

def test_wake_ack_and_goodnight_exist_in_all_languages():
    for key in ("wake_ack", "goodnight"):
        assert set(SPOKEN[key]) == {"de", "en", "fr"}
    assert spoken("wake_ack", {"NOX_LANG": "fr"}) == "Oui ? Qu'est-ce qu'il y a ?"


def test_the_voice_loop_has_no_hardcoded_german_lines_left():
    src = open("body/nox_voice_loop_v2.py").read()
    assert "Ja? Was ist?" not in src
    assert "Gute Nacht" not in src
    assert 'spoken("wake_ack")' in src and 'spoken("goodnight")' in src


# ─── scripts/stt_debug.py ───

spec = importlib.util.spec_from_file_location("stt_debug", "scripts/stt_debug.py")
stt_debug = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stt_debug)


def test_env_loader_parses_like_systemd(tmp_path):
    env = tmp_path / "nox.env"
    env.write_text("# comment\nWAKE_WORD=milou\nVOSK_MODEL_PATH=\"/home/x/model\"\n"
                   "MALFORMED LINE\nMIC_DEVICE='plughw:2,0'\n")
    loaded = stt_debug.load_env_file(str(env), environ={})
    assert loaded == {"WAKE_WORD": "milou", "VOSK_MODEL_PATH": "/home/x/model",
                      "MIC_DEVICE": "plughw:2,0"}


def test_env_loader_never_overrides_the_real_environment(tmp_path):
    env = tmp_path / "nox.env"
    env.write_text("WAKE_WORD=milou\n")
    assert stt_debug.load_env_file(str(env), environ={"WAKE_WORD": "rex"}) == {}


def test_env_loader_survives_a_missing_file():
    assert stt_debug.load_env_file("/no/such/nox.env", environ={}) == {}
