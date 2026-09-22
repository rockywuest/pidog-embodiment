"""Audio fallback when the SDK's sound engine is missing — issue #24.

The reporter's robot printed "sound_effect init ... fail", so Pidog had no
.music attribute. `bark` then died with AttributeError: 'Pidog' object has no
attribute 'music' — and since bark/howling/pant call dog.speak() in the middle
of their motion, the movement was lost too, while the same action worked under
SunFounder's sudo examples.
"""
import pytest

from body.nox_audio import AplayMusic, ensure_music


class Recorder:
    """subprocess.run stand-in."""

    def __init__(self, missing=()):
        self.calls = []
        self.missing = set(missing)

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        if cmd[0] in self.missing:
            raise FileNotFoundError(cmd[0])
        return object()

    @property
    def programs(self):
        return [c[0] for c in self.calls]


class DogWithoutMusic:
    """What the SDK leaves behind after 'sound_effect init ... fail'."""


class DogWithMusic:
    def __init__(self):
        self.music = "real pygame mixer"


def test_wav_plays_through_aplay_on_the_configured_device():
    run = Recorder()
    assert AplayMusic(device="plughw:3,0", runner=run).sound_play("/x/bark.wav") is True
    assert run.calls == [["aplay", "-D", "plughw:3,0", "/x/bark.wav"]]


def test_without_a_device_aplay_uses_the_system_default():
    run = Recorder()
    AplayMusic(runner=run).sound_play("/x/bark.wav")
    assert run.calls == [["aplay", "/x/bark.wav"]]


def test_mp3_falls_through_the_player_chain():
    """The SDK's sound files are .mp3; aplay cannot play those."""
    run = Recorder(missing=["ffplay"])
    assert AplayMusic(runner=run).sound_play("/x/howling.mp3") is True
    assert run.programs == ["ffplay", "mpg123"]


def test_no_player_available_reports_instead_of_raising():
    """A failed sound must never propagate: it would abort the whole action."""
    run = Recorder(missing=["ffplay", "mpg123"])
    music = AplayMusic(runner=run)
    assert music.sound_play("/x/howling.mp3") is False
    assert "ffplay not installed" in music.last_error
    assert "mpg123 not installed" in music.last_error


def test_volume_is_accepted_and_ignored():
    """The SDK always passes a volume; it must not become a TypeError."""
    run = Recorder()
    AplayMusic(runner=run).sound_play("/x/bark.wav", 100)
    assert run.calls


def test_threading_variant_returns_and_still_plays():
    """Pidog.speak() uses sound_play_threading mid-action."""
    run = Recorder()
    t = AplayMusic(device="hw:0,0", runner=run).sound_play_threading("/x/bark.wav", 80)
    t.join(timeout=5)
    assert not t.is_alive()
    assert run.calls == [["aplay", "-D", "hw:0,0", "/x/bark.wav"]]


def test_ensure_music_attaches_when_the_sdk_has_none():
    dog = DogWithoutMusic()
    run = Recorder()
    result = ensure_music(dog, device="plughw:3,0", runner=run)
    assert result["attached"] is True
    assert isinstance(dog.music, AplayMusic)
    # This is the call the SDK makes inside bark() — it must now work.
    dog.music.sound_play_threading("/x/single_bark_1.mp3", 100).join(timeout=5)
    assert run.programs[0] == "ffplay"


def test_ensure_music_leaves_a_healthy_robot_alone():
    dog = DogWithMusic()
    result = ensure_music(dog, device="plughw:3,0", runner=Recorder())
    assert result["attached"] is False
    assert dog.music == "real pygame mixer"


def test_ensure_music_reports_when_no_player_exists(monkeypatch):
    monkeypatch.setattr("body.nox_audio._player_available", lambda runner=None: False)
    dog = DogWithoutMusic()
    result = ensure_music(dog)
    assert result["attached"] is False
    assert "sound stays unavailable" in result["reason"]
    assert not hasattr(dog, "music")


@pytest.mark.parametrize("path", ["/x/a.MP3", "/x/a.mp3"])
def test_mp3_detection_is_case_insensitive(path):
    run = Recorder()
    AplayMusic(runner=run).sound_play(path)
    assert run.programs == ["ffplay"]
