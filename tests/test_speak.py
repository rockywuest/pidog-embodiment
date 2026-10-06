"""/speak said ok:true while the dog stayed silent — issue #35.

bark and howling played on the reporter's robot, /speak did not, and every
layer answered ok:true: the bridge sent the request fire-and-forget and never
read the daemon's reply, the daemon's background thread counted a failed
player as "played", and aplay exiting non-zero counted as success. These tests
pin each layer to the truth.
"""
import json
import urllib.request

import pytest

import body.nox_brain_bridge as bridge
from body.nox_audio import AplayMusic, speak_text
from tests.test_bridge_api import server_port  # noqa: F401 - fixture


class Logger:
    def __init__(self):
        self.lines = []

    def __call__(self, message):
        self.lines.append(message)


class Result:
    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stderr = stderr


class FakePiper:
    """subprocess.run stand-in that writes a wav like piper does."""

    def __init__(self, returncode=0, stderr="", writes=True, missing=False):
        self.returncode, self.stderr = returncode, stderr
        self.writes, self.missing = writes, missing
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        if self.missing:
            raise FileNotFoundError(cmd[0])
        if self.writes:
            with open(cmd[cmd.index("--output_file") + 1], "wb") as f:
                f.write(b"RIFF....WAVE")
        return Result(self.returncode, self.stderr)


class Music:
    def __init__(self, returns=None, raises=None):
        self.returns, self.raises = returns, raises
        self.played = []
        self.last_error = "aplay exited 1: no such device"

    def sound_play(self, path, volume=None):
        self.played.append(path)
        if self.raises:
            raise self.raises
        return self.returns


@pytest.fixture
def wav(tmp_path):
    return str(tmp_path / "speak.wav")


def test_speech_is_synthesised_and_played(wav):
    piper, music = FakePiper(), Music(returns=True)
    r = speak_text("Hallo", "piper", "voice.onnx", music, wav, runner=piper, log=Logger())
    assert r["ok"] is True
    assert music.played == [wav]
    cmd, kwargs = piper.calls[0]
    assert cmd == ["piper", "--model", "voice.onnx", "--output_file", wav]
    # Text on stdin, no shell — quotes and $ cannot break it.
    assert kwargs["input"] == "Hallo" and "shell" not in kwargs


def test_the_sdks_music_returning_none_counts_as_played(wav):
    r = speak_text("Hi", "piper", "v.onnx", Music(returns=None), wav,
                   runner=FakePiper(), log=Logger())
    assert r["ok"] is True


def test_a_failed_player_is_not_reported_as_spoken(wav):
    r = speak_text("Hallo", "piper", "v.onnx", Music(returns=False), wav,
                   runner=FakePiper(), log=Logger())
    assert r["ok"] is False
    assert "no such device" in r["error"]


def test_a_failing_piper_is_reported_with_its_reason(wav):
    log = Logger()
    piper = FakePiper(returncode=1, writes=False,
                      stderr="loading...\nRuntimeError: model file is corrupt\n")
    r = speak_text("Hallo", "piper", "v.onnx", Music(returns=True), wav, runner=piper, log=log)
    assert r["ok"] is False
    assert "model file is corrupt" in r["error"]
    assert any("speak failed" in line for line in log.lines)


def test_piper_that_writes_nothing_is_a_failure(wav):
    music = Music(returns=True)
    r = speak_text("Hallo", "piper", "v.onnx", music, wav,
                   runner=FakePiper(writes=False), log=Logger())
    assert r["ok"] is False and "no audio" in r["error"]
    assert music.played == []


def test_a_missing_piper_says_how_to_install_it(wav):
    r = speak_text("Hallo", "piper", "v.onnx", Music(returns=True), wav,
                   runner=FakePiper(missing=True), log=Logger())
    assert r["ok"] is False and "piper-tts" in r["error"]


def test_a_crashing_sound_engine_is_reported_not_raised(wav):
    r = speak_text("Hallo", "piper", "v.onnx", Music(raises=RuntimeError("mixer gone")), wav,
                   runner=FakePiper(), log=Logger())
    assert r["ok"] is False and "mixer gone" in r["error"]


def test_aplay_exiting_non_zero_is_not_success(tmp_path):
    f = tmp_path / "speak.wav"
    f.write_bytes(b"RIFF")
    log = Logger()

    def aplay(cmd, **kwargs):
        return Result(1, "aplay: main:831: audio open error: No such file or directory")

    music = AplayMusic(device="plughw:2,0", runner=aplay, log=log)
    assert music.sound_play(str(f)) is False
    assert "audio open error" in music.last_error
    assert log.lines


# ─── bridge: the daemon's answer reaches the caller ───

def _post(port, payload):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/speak",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def test_bridge_returns_the_daemons_error(server_port, monkeypatch):  # noqa: F811
    sent = []

    def fake_daemon(payload, timeout=30):
        sent.append(payload)
        return {"ok": False, "error": "piper voice model not found: /x.onnx"}

    monkeypatch.setattr(bridge, "send_to_daemon", fake_daemon)
    body = _post(server_port, {"text": "Hallo"})
    assert body["ok"] is False and "voice model" in body["error"]
    assert sent == [{"cmd": "speak", "text": "Hallo", "wait": False}]


def test_bridge_blocking_asks_the_daemon_to_wait(server_port, monkeypatch):  # noqa: F811
    sent = []

    def fake_daemon(payload, timeout=30):
        sent.append((payload, timeout))
        return {"ok": True, "spoke": "Hallo", "via": "AplayMusic"}

    monkeypatch.setattr(bridge, "send_to_daemon", fake_daemon)
    body = _post(server_port, {"text": "Hallo", "blocking": True})
    assert body["ok"] is True
    assert sent[0][0]["wait"] is True and sent[0][1] >= 60


def test_bridge_unreachable_daemon_is_not_ok(server_port, monkeypatch):  # noqa: F811
    monkeypatch.setattr(bridge, "send_to_daemon",
                        lambda payload, timeout=30: {"error": "[Errno 111] Connection refused"})
    body = _post(server_port, {"text": "Hallo"})
    assert body["ok"] is False
