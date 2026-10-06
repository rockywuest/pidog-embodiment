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
import re
import shutil
import subprocess
import threading

# SunFounder ships its sounds as .mp3 in ~/pidog/sounds and aplay cannot play
# those, so a fallback without one of these players is mute. Each entry says how
# that player is told WHICH ALSA device to use — sending audio to the default
# device while the speaker sits on another card is silent in exactly the same
# way as having no player at all (issue #24).
MP3_PLAYERS = (
    # (binary, args before the file, how to name the device)
    ("mpg123", ["-q"], lambda dev: ["-a", dev]),
    ("ffmpeg", ["-loglevel", "quiet", "-i"], lambda dev: ["-f", "alsa", dev]),
    ("sox", [], lambda dev: ["-t", "alsa", dev]),
    # ffplay cannot be pointed at a device on the command line; it follows
    # SDL_AUDIODRIVER/AUDIODEV. Last resort.
    ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"], None),
)
WAV_PLAYER = "aplay"


def mp3_player_available():
    """The first installed mp3 player, or None — SDK sounds need one."""
    for binary, _args, _dev in MP3_PLAYERS:
        if shutil.which(binary):
            return binary
    return None


def list_playback_cards(runner=None):
    """ALSA playback cards from `aplay -l`, as [{"index": int, "id": str}].

    Empty when aplay is missing or finds nothing — which is itself the answer
    to "why is there no sound".
    """
    run = runner or subprocess.run
    try:
        out = run(["aplay", "-l"], capture_output=True, text=True, timeout=5)
    except Exception:  # noqa: BLE001 - diagnostics must not raise
        return []
    cards = []
    for line in (getattr(out, "stdout", "") or "").splitlines():
        m = re.match(r"card (\d+): (\S+)", line.strip())
        if m and not any(c["index"] == int(m.group(1)) for c in cards):
            cards.append({"index": int(m.group(1)), "id": m.group(2)})
    return cards


def device_card(device):
    """The card an ALSA device string points at: int index, str id, or None."""
    m = re.search(r"(?:hw|plughw):(?:CARD=)?([A-Za-z0-9_]+)", str(device or ""))
    if not m:
        return None
    token = m.group(1)
    return int(token) if token.isdigit() else token


def describe_cards(cards):
    """"0:Headphones, 3:DAC" — for log lines and error messages."""
    return ", ".join(f"{c['index']}:{c['id']}" for c in cards) or "none"


# Cards that are almost never the robot's speaker. HDMI in particular is picked
# up by any auto-detection and is usually not even plugged in.
UNLIKELY_SPEAKERS = ("vc4hdmi", "hdmi")
# Names of boards that DO carry a robot speaker. The PiDog's robot_hat registers
# as a Google voiceHAT sound card — the name that taught us this list is not a
# guess we should make silently (issue #24).
LIKELY_SPEAKERS = ("voicehat", "googlevoice", "robot", "hifiberry", "dac",
                   "amp", "usb", "speaker", "headphone")


def pick_card(cards):
    """The card most likely to be the robot's speaker, or None."""
    usable = [c for c in cards
              if not any(k in c["id"].lower() for k in UNLIKELY_SPEAKERS)]
    for keyword in LIKELY_SPEAKERS:
        for card in usable:
            if keyword in card["id"].lower():
                return card
    return usable[0] if usable else (cards[0] if cards else None)


def resolve_device(requested=None, runner=None):
    """Decide which ALSA device to play to.

    Returns {"device": str|None, "usable": bool, "cards": [...], "reason": str}.
    ``device`` None means "use the ALSA default", and that is the default
    behaviour on purpose.

    Two rounds of issue #24 were caused by this function's ancestors guessing:
    first an unconditional fallback to plughw:3,0, which on a robot without a
    card 3 silenced everything and broke the SDK's mixer; then an auto-pick that
    chose card 0 — HDMI, unplugged — while the PiDog's speaker sat on card 2.
    Meanwhile SunFounder's own examples worked, because they set no device at
    all and let ALSA use the default the PiDog installer configured.

    So: an explicit AUDIODEV is honoured when its card exists, AUDIODEV=auto
    asks for the detection, and anything else leaves the default alone.
    """
    cards = list_playback_cards(runner)
    known = {c["index"] for c in cards} | {c["id"] for c in cards}

    if requested and str(requested).lower() != "auto":
        card = device_card(requested)
        if card is None or card in known or not cards:
            # Unparseable or unverifiable: respect an explicit setting.
            return {"device": requested, "usable": True, "cards": cards,
                    "reason": "configured" if card is None else f"card {card} present"}
        return {"device": None, "usable": False, "cards": cards,
                "reason": (f"configured device {requested} points at card {card}, "
                           f"which does not exist (present: {describe_cards(cards)}) "
                           "— falling back to the ALSA default")}

    if not cards:
        return {"device": None, "usable": False, "cards": cards,
                "reason": "aplay lists no playback card — no sound is possible"}

    if str(requested).lower() == "auto":
        chosen = pick_card(cards)
        hdmi_only = all(any(k in c["id"].lower() for k in UNLIKELY_SPEAKERS)
                        for c in cards)
        return {"device": f"plughw:{chosen['index']},0",
                "usable": not hdmi_only,
                "cards": cards,
                "reason": (f"AUDIODEV=auto picked card {chosen['index']}:{chosen['id']}"
                           + (" — but every card looks like an HDMI output, so this "
                              "is probably not the robot's speaker" if hdmi_only else ""))}

    # Nothing requested: the ALSA default is what SunFounder's own tools use, and
    # on a working PiDog install it is already the speaker. Overriding it is how
    # this went wrong twice.
    return {"device": None, "usable": True, "cards": cards,
            "reason": (f"ALSA default (cards: {describe_cards(cards)}; set AUDIODEV "
                       "in body/nox.env to pin one, or AUDIODEV=auto to detect)")}


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
        """Every command worth trying for this file, best first."""
        if str(path).lower().endswith(".mp3"):
            commands = []
            for binary, args, device_args in MP3_PLAYERS:
                cmd = [binary] + list(args) + [str(path)]
                if self.device and device_args:
                    cmd += device_args(self.device)
                elif not self.device and binary == "sox":
                    cmd += ["-d"]  # sox needs an explicit output
                commands.append(cmd)
            return commands
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
                proc = self._run(cmd, capture_output=True, timeout=self.timeout)
                # A player that runs and exits non-zero (wrong device, busy card,
                # unreadable file) played nothing — try the next one (issue #35).
                code = getattr(proc, "returncode", 0)
                if code:
                    errors.append(f"{cmd[0]} exited {code}: {_last_line(getattr(proc, 'stderr', ''))}")
                    continue
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


def _last_line(output):
    """The last non-empty line of a tool's stderr — usually the actual reason."""
    if isinstance(output, bytes):
        output = output.decode("utf-8", "replace")
    lines = [line.strip() for line in (output or "").splitlines() if line.strip()]
    return lines[-1] if lines else "no error output"


# Where piper voices live. Ours first, then SunFounder's: their examples
# download into ~/.piper_models — and when run with sudo, into
# /root/.piper_models, which the service user cannot read (issue #35).
PIPER_VOICE_DIRS = (
    "~/.local/share/piper-voices",
    "~/.piper_models",
    "/root/.piper_models",
)
PREFERRED_PIPER_VOICE = "de_DE-thorsten-high"


def find_piper_voice(configured=None, dirs=PIPER_VOICE_DIRS, preferred=PREFERRED_PIPER_VOICE):
    """Pick the voice /speak uses: {"model": path|None, "reason": str, "voices": [...]}.

    An explicit PIPER_MODEL wins and is never second-guessed. Otherwise any
    installed voice will do — the robot speaking in the wrong voice is a far
    better first experience than "model not found" while SunFounder's own
    examples talk fine on the same robot. A voice needs its .onnx.json too.
    """
    voices, locked = [], []
    for d in dirs:
        d = os.path.expanduser(d)
        try:
            names = sorted(os.listdir(d))
        except PermissionError:
            locked.append(d)
            continue
        except OSError:
            continue
        for name in names:
            path = os.path.join(d, name)
            if (name.endswith(".onnx") and os.path.isfile(path + ".json")
                    and os.access(path, os.R_OK)):
                voices.append(path)

    if configured:
        if os.path.isfile(configured):
            return {"model": configured, "reason": "PIPER_MODEL from body/nox.env",
                    "voices": voices}
        return {"model": None, "voices": voices, "reason": (
            f"PIPER_MODEL={configured} does not exist"
            + (f" — installed voices: {', '.join(voices)}" if voices else "")
            + ". Fix the path in body/nox.env, or remove the line to pick one automatically")}

    if voices:
        chosen = next((v for v in voices if os.path.basename(v) == preferred + ".onnx"),
                      voices[0])
        return {"model": chosen, "voices": voices,
                "reason": f"found {os.path.basename(chosen)} in {os.path.dirname(chosen)}"}

    hint = ("no piper voice found in " + ", ".join(os.path.expanduser(d) for d in dirs)
            + ". Download one (both .onnx and .onnx.json) from "
            "https://huggingface.co/rhasspy/piper-voices into ~/.local/share/piper-voices/")
    if any(d.startswith("/root") for d in locked):
        hint = ("no readable piper voice. SunFounder's examples run with sudo keep "
                "theirs in /root/.piper_models, which this service cannot read — copy "
                "them: sudo cp -r /root/.piper_models ~/ && sudo chown -R $USER: "
                "~/.piper_models — or " + hint[0].lower() + hint[1:])
    return {"model": None, "voices": [], "reason": hint}


def speak_text(text, piper_bin, piper_model, music, wav_path, runner=None, log=None,
               play_lock=None):
    """Text → Piper → wav → ``music.sound_play``. Returns {"ok": bool, ...}.

    The daemon used to run this inline in a fire-and-forget thread: Piper's
    stderr went to the log only on a non-zero exit, a failed player was counted
    as played, and the bridge never even waited for the daemon's answer — so
    /speak said ok:true while the dog stayed silent (issue #35). Every outcome
    is now returned AND logged, so both `"blocking": true` and journalctl tell
    the truth.

    ``play_lock`` guards only the playback — the SDK's mixer is shared with
    cmd_sound — never the synthesis, which can take seconds and would hold up
    every movement. ``music.sound_play`` must block until the sound ends (the
    SDK's does, and so does AplayMusic): the caller deletes the wav afterwards.
    """
    run = runner or subprocess.run
    log = log if log is not None else _default_log

    def fail(error):
        log(f"[nox] speak failed — {error}")
        return {"ok": False, "error": error}

    try:
        # Text goes in on stdin, not through a shell: no quoting to get wrong.
        proc = run([piper_bin, "--model", piper_model, "--output_file", wav_path],
                   input=text, capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        return fail(f"piper not found: {piper_bin} — pip3 install piper-tts, "
                    "or set PIPER_BIN in body/nox.env")
    except subprocess.TimeoutExpired:
        return fail("piper took longer than 60 s")
    if getattr(proc, "returncode", 0):
        return fail(f"piper exited {proc.returncode}: {_last_line(getattr(proc, 'stderr', ''))}")
    if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
        return fail(f"piper produced no audio ({wav_path} missing or empty) — "
                    f"{_last_line(getattr(proc, 'stderr', ''))}")

    if music is None:
        return fail("no sound engine and no aplay — nothing can play the speech")
    try:
        if play_lock is not None:
            with play_lock:
                played = music.sound_play(wav_path)
        else:
            played = music.sound_play(wav_path)
    except Exception as e:  # noqa: BLE001 - report it, the daemon must keep running
        return fail(f"playback raised {type(e).__name__}: {e}")
    # robot_hat's Music.sound_play returns None; only our AplayMusic says False.
    if played is False:
        return {"ok": False,
                "error": f"playback failed: {getattr(music, 'last_error', None) or 'unknown'}"}
    return {"ok": True, "spoke": text, "via": type(music).__name__}


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
