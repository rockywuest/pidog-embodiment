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
    run = Recorder(missing=["mpg123"])
    assert AplayMusic(runner=run).sound_play(mp3) is True
    assert run.programs == ["mpg123", "ffmpeg"]


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
    assert run.programs == ["mpg123"], "an .mp3 must not go to aplay"


# ─── failing loudly ─────────────────────────────────────────────────────────

def test_a_failed_sound_is_logged_not_swallowed(mp3):
    """A silent failure is the worst outcome: the first version of this class
    only stored last_error, so a robot without an mp3 player barked mutely and
    every layer above still answered ok."""
    log = Logger()
    music = AplayMusic(runner=Recorder(missing=["mpg123", "ffmpeg", "sox", "ffplay"]), log=log)

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
    assert run.programs[0] == "mpg123"


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


# ─── which device to play to (issue #24, third round) ───────────────────────

APLAY_TWO_CARDS = """**** List of PLAYBACK Hardware Devices ****
card 0: Headphones [bcm2835 Headphones], device 0: bcm2835 Headphones [bcm2835 Headphones]
  Subdevices: 8/8
card 1: vc4hdmi [vc4-hdmi], device 0: MAI PCM i2s-hifi-0 [MAI PCM i2s-hifi-0]
  Subdevices: 1/1
"""

# The reporter's actual robot: two HDMI outputs, and the PiDog's speaker on
# card 2 as a Google voiceHAT card. Auto-detection picked card 0 and was mute.
APLAY_PIDOG = """**** List of PLAYBACK Hardware Devices ****
card 0: vc4hdmi0 [vc4-hdmi-0], device 0: MAI PCM i2s-hifi-0 [MAI PCM i2s-hifi-0]
  Subdevices: 1/1
card 1: vc4hdmi1 [vc4-hdmi-1], device 0: MAI PCM i2s-hifi-0 [MAI PCM i2s-hifi-0]
  Subdevices: 1/1
card 2: sndrpigooglevoi [snd_rpi_googlevoicehat_soundcar], device 0: Google voiceHAT SoundCard HiFi
  Subdevices: 1/1
"""

APLAY_HDMI_ONLY = """**** List of PLAYBACK Hardware Devices ****
card 0: vc4hdmi0 [vc4-hdmi-0], device 0: MAI PCM i2s-hifi-0 [MAI PCM i2s-hifi-0]
  Subdevices: 1/1
"""

APLAY_WITH_DAC = APLAY_TWO_CARDS + """card 3: sndrpihifiberry [snd_rpi_hifiberry_dac], device 0: HifiBerry DAC HiFi
  Subdevices: 1/1
"""


class AplayOutput:
    """subprocess.run stand-in that answers `aplay -l`."""

    def __init__(self, stdout="", fail=False):
        self.stdout = stdout
        self.fail = fail

    def __call__(self, cmd, **kwargs):
        if self.fail:
            raise FileNotFoundError("aplay")
        return type("R", (), {"stdout": self.stdout, "returncode": 0})()


def test_cards_are_parsed_from_aplay():
    from body.nox_audio import list_playback_cards
    cards = list_playback_cards(AplayOutput(APLAY_WITH_DAC))
    assert [c["index"] for c in cards] == [0, 1, 3]
    assert cards[2]["id"] == "sndrpihifiberry"


def test_nothing_requested_means_the_alsa_default():
    """SunFounder's own tools set no device and work; overriding that default is
    what silenced the robot twice (issue #24)."""
    from body.nox_audio import resolve_device
    r = resolve_device(runner=AplayOutput(APLAY_PIDOG))
    assert r["device"] is None
    assert r["usable"] is True
    assert "ALSA default" in r["reason"]
    assert "2:sndrpigooglevoi" in r["reason"], "must list the cards it saw"
    assert "AUDIODEV" in r["reason"], "must say how to pin one"


def test_auto_prefers_the_pidog_speaker_over_hdmi():
    """The card list from the reporter's robot: the speaker is card 2, and the
    previous auto-pick chose card 0 (HDMI, not even plugged in)."""
    from body.nox_audio import resolve_device
    r = resolve_device("auto", runner=AplayOutput(APLAY_PIDOG))
    assert r["device"] == "plughw:2,0"
    assert r["usable"] is True
    assert "2:sndrpigooglevoi" in r["reason"]


def test_auto_prefers_a_dac_over_the_onboard_outputs():
    from body.nox_audio import resolve_device
    r = resolve_device("auto", runner=AplayOutput(APLAY_WITH_DAC))
    assert r["device"] == "plughw:3,0"
    assert "3:sndrpihifiberry" in r["reason"]


def test_auto_says_so_when_only_hdmi_exists():
    from body.nox_audio import resolve_device
    r = resolve_device("auto", runner=AplayOutput(APLAY_HDMI_ONLY))
    assert r["usable"] is False
    assert "probably not the robot's speaker" in r["reason"]


def test_auto_skips_hdmi_for_the_headphone_jack():
    from body.nox_audio import resolve_device
    r = resolve_device("auto", runner=AplayOutput(APLAY_TWO_CARDS))
    assert r["device"] == "plughw:0,0", "card 1 is HDMI"


def test_a_configured_device_on_a_missing_card_is_refused():
    """The actual bug: the daemon exported AUDIODEV=plughw:3,0 on a robot with
    only cards 0 and 1, which silenced everything and broke the SDK's mixer."""
    from body.nox_audio import resolve_device
    r = resolve_device("plughw:3,0", runner=AplayOutput(APLAY_TWO_CARDS))
    assert r["device"] is None, "must fall back to the ALSA default"
    assert r["usable"] is False
    assert "does not exist" in r["reason"]
    assert "0:Headphones" in r["reason"], "must list what is present"


def test_a_configured_device_on_a_present_card_is_kept():
    from body.nox_audio import resolve_device
    r = resolve_device("plughw:3,0", runner=AplayOutput(APLAY_WITH_DAC))
    assert r["device"] == "plughw:3,0"
    assert r["usable"] is True


def test_a_named_device_is_respected_unverified():
    from body.nox_audio import resolve_device
    r = resolve_device("default", runner=AplayOutput(APLAY_TWO_CARDS))
    assert r["device"] == "default"
    assert r["usable"] is True


def test_no_cards_at_all_is_reported():
    from body.nox_audio import resolve_device
    r = resolve_device("auto", runner=AplayOutput("**** List of PLAYBACK Hardware Devices ****\n"))
    assert r["device"] is None
    assert r["usable"] is False
    assert "no playback card" in r["reason"]


def test_missing_aplay_does_not_raise():
    from body.nox_audio import list_playback_cards, resolve_device
    assert list_playback_cards(AplayOutput(fail=True)) == []
    r = resolve_device("plughw:3,0", runner=AplayOutput(fail=True))
    assert r["device"] == "plughw:3,0", "unverifiable: respect the setting"


def test_mp3_players_are_told_which_device_to_use(mp3):
    """ffplay was picked first and silently played to the default device while
    the speaker sat on another card — the same silence as no player at all."""
    run = Recorder()
    AplayMusic(device="plughw:3,0", runner=run).sound_play(mp3)
    assert run.calls == [["mpg123", "-q", mp3, "-a", "plughw:3,0"]]


def test_ffmpeg_is_used_when_mpg123_is_absent(mp3):
    run = Recorder(missing=["mpg123"])
    AplayMusic(device="plughw:3,0", runner=run).sound_play(mp3)
    assert run.calls[-1] == ["ffmpeg", "-loglevel", "quiet", "-i", mp3,
                             "-f", "alsa", "plughw:3,0"]


def test_ffplay_is_the_last_resort_and_gets_no_device(mp3):
    """It has no device option; it follows AUDIODEV."""
    run = Recorder(missing=["mpg123", "ffmpeg", "sox"])
    AplayMusic(device="plughw:3,0", runner=run).sound_play(mp3)
    assert run.programs == ["mpg123", "ffmpeg", "sox", "ffplay"]
    assert "plughw:3,0" not in run.calls[-1]


def test_sox_uses_the_default_output_when_no_device_is_known(mp3):
    run = Recorder(missing=["mpg123", "ffmpeg"])
    AplayMusic(runner=run).sound_play(mp3)
    assert run.calls[-1] == ["sox", mp3, "-d"]
