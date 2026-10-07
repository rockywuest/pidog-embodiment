"""nox_vision warmup on a cold camera — issue #36 follow-up.

The reporter's first /vision result was "timed out": the first photo starts
the camera inside the daemon, which takes longer than the 10 s the vision
engine used to allow. The next cycle — a minute later — worked.
"""
import body.nox_vision as vision


def test_photo_waits_long_enough_for_a_cold_camera(monkeypatch):
    seen = {}

    def fake_cmd(cmd, timeout=10):
        seen["timeout"] = timeout
        return {"photo": __file__}

    monkeypatch.setattr(vision, "_daemon_cmd", fake_cmd)
    path, err = vision.capture_frame()
    assert err is None and path == __file__
    assert seen["timeout"] >= 30


def test_patrol_prompt_is_not_a_numbered_checklist():
    # SmolVLM-256M echoed "1) Any people." back instead of answering.
    assert "1)" not in vision.PROMPTS["patrol"]
