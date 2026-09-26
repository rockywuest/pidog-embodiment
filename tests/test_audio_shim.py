"""Audio fallback when the SDK's sound engine is missing — issue #24.

The reporter's robot printed "sound_effect init ... fail", so Pidog had no
.music attribute. `bark` then died with AttributeError: 'Pidog' object has no
attribute 'music' — and since bark/howling/pant call dog.speak() in the middle
of their motion, the movement was lost too, while the same action worked under
SunFounder's sudo examples.

The first fix attached an aplay-backed stand-in, which made the dog move again —
and then it moved in silence, because SunFounder's sounds are .mp3 and aplay
cannot play those. The failure was recorded in an attribute nobody read. So
these tests pin both halves: the stand-in works, and when it cannot play, it
says so.
"""
import pytest

from body.nox_audio import AplayMusic, audio_capability, ensure_music


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


class Logger:
    def __init__(self):
        self.lines = []

    def __call__(self, message):
        self.lines.append(message)


class DogWithoutMusic:
    """What the SDK leaves behind after 'sound_effect init ... fail'."""


class DogWithMusic:
    def __init__(self):
        self.music = "real pygame mixer"


@pytest.fixture
def wav(tmp_path):
    f = tmp_path / "single_bark_1.wav"
    f.write_bytes(b"RIFF")
    return str(f)


@pytest.fixture
def mp3(tmp_path):
    """The format the SDK actually ships in ~/pidog/sounds."""
    f = tmp_path / "single_bark_1.mp3"
    f.write_bytes(b"ID3")
    return str(f)


# ─── playing ────────────────────────────────────────────────────────────────

def test_wav_plays_through_aplay_on_the_configured_device(wav):
    run = Recorder()
    assert AplayMusic(device="plughw:3,0", runner=run).sound_play(wav) is True
    assert run.calls == [["aplay", "-D", "plughw:3,0", wav]]


def test_without_a_device_aplay_uses_the_system_default(wav):
    run = Recorder()
    AplayMusic(runner=run).sound_play(wav)
    assert run.calls == [["aplay", wav]]


def test_mp3_falls_through_the_player_chain(mp3):
    run = Recorder(missing=["ffplay"])
    assert AplayMusic(runner=run).sound_play(mp3) is True
    assert run.programs == ["ffplay", "mpg123"]


def test_sox_is_invoked_with_the_default_device(mp3):
    """sox needs an explicit -d, unlike the other two players."""
    run = Recorder(missing=["ffplay", "mpg123"])
    assert AplayMusic(runner=run, log=Logger()).sound_play(mp3) is True
    assert run.calls[-1] == ["sox", mp3, "-d"]


def test_volume_is_accepted_and_ignored(wav):
    """The SDK always passes a volume; it must not become a TypeError."""
    run = Recorder()
    AplayMusic(runner=run).sound_play(wav, 100)
    assert run.calls


def test_threading_variant_returns_and_still_plays(wav):
    """Pidog.speak() uses sound_play_threading mid-action."""
    run = Recorder()
    t = AplayMusic(device="hw:0,0", runner=run).sound_play_threading(wav, 80)
    t.join(timeout=5)
    assert not t.is_alive()
    assert run.calls == [["aplay", "-D", "hw:0,0", wav]]


@pytest.mark.parametrize("suffix", [".MP3", ".mp3"])
def test_mp3_detection_is_case_insensitive(tmp_path, suffix):
    f = tmp_path / ("howling" + suffix)
    f.write_bytes(b"ID3")
    run = Recorder()
    AplayMusic(runner=run).sound_play(str(f))
    assert run.programs == ["ffplay"]


# ─── failing loudly ─────────────────────────────────────────────────────────

def test_a_failed_sound_is_logged_not_swallowed(mp3):
    """A silent failure is the worst outcome: the first version of this class
    only stored last_error, so a robot without an mp3 player barked mutely and
    every layer above still answered ok."""
    log = Logger()
    music = AplayMusic(runner=Recorder(missing=["ffplay", "mpg123", "sox"]), log=log)

    assert music.sound_play(mp3) is False
    assert len(log.lines) == 1
    assert "single_bark_1.mp3" in log.lines[0]
    assert "not installed" in log.lines[0]
    assert "apt install mpg123" in log.lines[0], "must name the fix"
    assert "mpg123 not installed" in music.last_error


def test_a_missing_sound_file_says_so(tmp_path):
    log = Logger()
    music = AplayMusic(runner=Recorder(), log=log)
    assert music.sound_play(str(tmp_path / "nope.wav")) is False
    assert "no such sound file" in music.last_error
    assert log.lines and "nope.wav" in log.lines[0]


def test_a_successful_sound_logs_nothing(wav):
    log = Logger()
    assert AplayMusic(device="plughw:3,0", runner=Recorder(), log=log).sound_play(wav) is True
    assert log.lines == []


def test_a_wav_failure_points_at_the_audio_device(wav):
    log = Logger()
    music = AplayMusic(runner=Recorder(missing=["aplay"]), log=log)
    assert music.sound_play(wav) is False
    assert "aplay -l" in log.lines[0] and "AUDIODEV" in log.lines[0]


# ─── attaching ──────────────────────────────────────────────────────────────

def test_ensure_music_attaches_when_the_sdk_has_none(mp3):
    dog = DogWithoutMusic()
    run = Recorder()
    result = ensure_music(dog, device="plughw:3,0", runner=run)
    assert result["attached"] is True
    assert isinstance(dog.music, AplayMusic)
    # This is the call the SDK makes inside bark() — it must now work.
    dog.music.sound_play_threading(mp3, 100).join(timeout=5)
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


def test_ensure_music_warns_when_no_mp3_player_exists(monkeypatch):
    """Attaching the fallback is not enough — the SDK's sounds are mp3."""
    monkeypatch.setattr("body.nox_audio.mp3_player_available", lambda: None)
    monkeypatch.setattr("body.nox_audio._player_available", lambda runner=None: True)
    result = ensure_music(DogWithoutMusic(), device="plughw:3,0")
    assert result["attached"] is True
    assert result["mp3_player"] is None
    assert "mpg123" in result["warning"]


def test_ensure_music_reports_the_player_it_found(monkeypatch):
    monkeypatch.setattr("body.nox_audio.mp3_player_available", lambda: "mpg123")
    monkeypatch.setattr("body.nox_audio._player_available", lambda runner=None: True)
    result = ensure_music(DogWithoutMusic(), device="plughw:3,0")
    assert result["mp3_player"] == "mpg123"
    assert "warning" not in result


def test_audio_capability_shape(monkeypatch):
    monkeypatch.setattr("body.nox_audio.shutil.which", lambda name: "/usr/bin/" + name)
    cap = audio_capability()
    assert cap["wav"] is True
    assert cap["mp3"] in ("ffplay", "mpg123", "sox")
