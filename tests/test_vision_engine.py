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


def test_warmup_retries_a_cold_camera_once(monkeypatch):
    answers = iter([(None, "timed out"), ("/tmp/frame.jpg", None)])
    calls = []
    monkeypatch.setattr(vision, "capture_frame", lambda: calls.append(1) or next(answers))
    monkeypatch.setattr(vision.time, "sleep", lambda s: None)
    assert vision.capture_with_retry() == ("/tmp/frame.jpg", None)
    assert len(calls) == 2


def test_warmup_gives_up_after_one_retry(monkeypatch):
    calls = []
    monkeypatch.setattr(vision, "capture_frame", lambda: calls.append(1) or (None, "no camera"))
    monkeypatch.setattr(vision.time, "sleep", lambda s: None)
    assert vision.capture_with_retry() == (None, "no camera")
    assert len(calls) == 2


# ─── patrol: who counts as a person ───

import pytest  # noqa: E402

from body.nox_behavior_engine import mentions_person  # noqa: E402


@pytest.mark.parametrize("desc", [
    "A person is standing near the door.",
    "Two people sit on a sofa, a chair blocks the way.",
    "A child is playing on the floor.",
    "Someone is walking past.",
])
def test_people_in_the_scene_count(desc):
    assert mentions_person(desc)


@pytest.mark.parametrize("desc", [
    "No people are visible. A chair blocks the way.",
    "There are no people in the room.",
    "An empty hallway without any person.",
    "Nobody is there, the path is clear.",
    "A kitchen with no one in it.",
    "There isn't a person here.",
    "A manual on the table.",   # 'man' inside a word
    "",
    None,
])
def test_scenes_without_people_do_not(desc):
    assert not mentions_person(desc)


def test_one_person_among_negations_still_counts():
    assert mentions_person("No cars, but a woman is standing by the window.")
