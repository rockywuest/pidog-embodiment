"""Audio fallback for a PiDog whose sound engine failed to initialise (issue #24).

``Pidog.__init__`` prints ``sound_effect init ... fail`` and leaves ``self.music``
unset when the pygame mixer cannot open the audio device — which happens inside
the systemd services on some robots while SunFounder's ``sudo`` examples work.
Every SDK sound path then raises:

    AttributeError: 'Pidog' object has no attribute 'music'

and that kills whole actions, not just their sound: ``bark``, ``howling`` and
``pant`` call ``my_dog.speak()`` in the middle of their motion sequence, so the
dog stops moving too.

``aplay`` works on those robots (the daemon already uses it for TTS), so rather
than stripping the sound out of the SDK's actions, this module gives the SDK the
interface it expects, backed by a subprocess player. ``ensure_music()`` attaches
it only when the real one is missing, so a healthy robot keeps the pygame mixer.

Kept free of SDK and pygame imports so it can be tested anywhere.
"""
import os
import shutil
import subprocess
import threading

# SunFounder ships its sounds as .mp3 in ~/pidog/sounds, and aplay cannot play
# those, so a fallback without one of these players is mute. Same order as the
# daemon's own cmd_sound, so both paths behave alike.
MP3_PLAYERS = (
    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
    ["mpg123", "-q"],
    ["sox"],  # completed with "-d" (default device) in _command
)
WAV_PLAYER = "aplay"


def mp3_player_available():
    """The first installed mp3 player, or None — SDK sounds need one."""
    for cmd in MP3_PLAYERS:
        if shutil.which(cmd[0]):
            return cmd[0]
    return None


class AplayMusic:
    """Minimal stand-in for robot_hat's ``Music``, playing via command line tools.

    Only the surface the SunFounder SDK actually uses is implemented:
    ``sound_play`` (blocking) and ``sound_play_threading`` (fire and forget).
    Volume is accepted and ignored: aplay has no per-call volume, and silently
    resetting the mixer would be worse than playing at the system level.
    """

    def __init__(self, device=None, runner=None, timeout=30, log=None):
        self.device = device
        self._run = runner or subprocess.run
        self.timeout = timeout
        # A sound that fails must say so: the first version of this class only
        # stored the reason in last_error, so a missing mp3 player produced a
        # perfectly silent bark that still answered ok:true (issue #24).
        self._log = log if log is not None else _default_log
        self.last_error = None

    def _command(self, path):
        if str(path).lower().endswith(".mp3"):
            return [p + [str(path)] + (["-d"] if p[0] == "sox" else [])
                    for p in MP3_PLAYERS]
        cmd = [WAV_PLAYER]
        if self.device:
            cmd += ["-D", self.device]
        return [cmd + [str(path)]]

    def sound_play(self, path, volume=None):
        """Play a file and wait. Returns True when a player ran."""
        self.last_error = None
        if not os.path.exists(path):
            self.last_error = f"no such sound file: {path}"
            self._log(f"[nox] sound failed — {self.last_error}")
            return False
        errors = []
        for cmd in self._command(path):
            try:
                self._run(cmd, capture_output=True, timeout=self.timeout)
                return True
            except FileNotFoundError:
                errors.append(f"{cmd[0]} not installed")
            except Exception as e:  # noqa: BLE001 - a failed sound must not kill an action
                errors.append(f"{cmd[0]}: {type(e).__name__}: {e}")
        self.last_error = "; ".join(errors)
        hint = ("install one: sudo apt install mpg123"
                if str(path).lower().endswith(".mp3") else
                "check the audio device: aplay -l, then set AUDIODEV")
        self._log(f"[nox] sound failed for {os.path.basename(str(path))} — "
                  f"{self.last_error} ({hint})")
        return False

    def sound_play_threading(self, path, volume=None):
        """Play without blocking — what the SDK's speak() uses mid-action."""
        t = threading.Thread(target=self.sound_play, args=(path, volume), daemon=True)
        t.start()
        return t

    # The SDK occasionally probes these; keep them harmless.
    def music_play(self, path, loops=1, start=0.0, volume=None):
        return self.sound_play(path, volume)

    def music_stop(self):
        return True


def _default_log(message):
    print(message, flush=True)


def audio_capability():
    """What this machine can actually play: {"wav": bool, "mp3": str|None}."""
    return {"wav": bool(shutil.which(WAV_PLAYER)), "mp3": mp3_player_available()}


def ensure_music(dog, device=None, runner=None, log=None):
    """Give ``dog`` a working ``.music`` if the SDK failed to build one.

    Returns {"attached": bool, "reason": str}. ``attached`` False means the
    robot's own sound engine is fine and was left untouched.
    """
    existing = getattr(dog, "music", None)
    if existing is not None:
        return {"attached": False, "reason": "SDK sound engine present"}
    if not _player_available(runner):
        return {"attached": False, "reason": "no aplay found — sound stays unavailable"}
    dog.music = AplayMusic(device=device, runner=runner, log=log)
    result = {"attached": True,
              "reason": f"aplay fallback on {device or 'default device'}"}
    if runner is None:
        # SunFounder's own sounds are .mp3, so aplay alone leaves bark, howling
        # and pant silent while everything reports success (issue #24).
        player = mp3_player_available()
        result["mp3_player"] = player
        if not player:
            result["warning"] = ("no mp3 player installed (ffplay/mpg123/sox) — "
                                 "the SDK's sounds are mp3, so actions will move "
                                 "but stay silent; fix: sudo apt install mpg123")
    return result


def _player_available(runner=None):
    """True when aplay exists. A custom runner implies a test, so assume yes."""
    if runner is not None:
        return True
    from shutil import which
    return bool(which("aplay") or os.path.exists("/usr/bin/aplay"))
