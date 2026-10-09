"""The example env must pass the placeholder check, and the voice unit must
not be able to take the Pi down — issue #45 follow-up.

A user's robot became unreachable (SD card pulled): a 1.4 GB Vosk model
OOM-looped nox-voice until the system thrashed. And every fresh install
FAILed the doctor, because our own example comments contain "/home/<you>/..."
and the placeholder check read comments too.
"""
import re


def placeholder_lines(text):
    """The doctor's and installer's check: <...> OUTSIDE comments only."""
    return [line for line in text.splitlines() if re.search(r"^[^#]*[<>]", line)]


def test_the_shipped_example_has_no_placeholders_outside_comments():
    with open("body/nox.env.example") as f:
        assert placeholder_lines(f.read()) == []


def test_an_uncommented_placeholder_is_still_caught():
    assert placeholder_lines("BRAIN_HOST=<brain ip>\n") == ["BRAIN_HOST=<brain ip>"]
    assert placeholder_lines("# BRAIN_HOST=<brain ip>\nBRAIN_HOST=1.2.3.4\n") == []


def test_the_voice_unit_cannot_take_the_pi_down():
    with open("body/services/nox-voice.service") as f:
        unit = f.read()
    assert "MemoryMax=" in unit        # a too-large model kills the service, not the Pi
    assert "StartLimitBurst=" in unit  # and the restart loop gives up


def test_doctor_has_exactly_one_vosk_model_check():
    with open("scripts/doctor.sh") as f:
        doctor = f.read()
    warns = [l for l in doctor.splitlines() if l.strip().startswith('warn "no Vosk model')]
    assert len(warns) == 1  # the stale duplicate printed PASS and this warn at once
    assert "vosk_model=" not in doctor  # the shell-env duplicate variable is gone
