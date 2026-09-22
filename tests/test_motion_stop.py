"""Draining the SDK's motion buffers — issue #25.

The reporter in #12 stopped the behaviour engine and the dog kept moving, with
227 frames still queued. do_action() only appends to the SDK's action buffers,
so stopping the producer leaves the queue playing and an emergency-stop `lie`
lands behind it. These tests pin the drain.
"""
import pytest

from body.nox_motion import BUFFER_NAMES, buffer_depth, drain


class FakeDog:
    """Minimal stand-in for Pidog: three list buffers and optional stop methods."""

    def __init__(self, legs=0, head=0, tail=0, methods=("body_stop",), broken=False):
        self.legs_action_buffer = [[0] * 8 for _ in range(legs)]
        self.head_action_buffer = [[0] * 3 for _ in range(head)]
        self.tail_action_buffer = [[0] for _ in range(tail)]
        self.calls = []
        self._broken = broken
        for name in methods:
            setattr(self, name, self._make(name))

    def _make(self, name):
        def fn():
            self.calls.append(name)
            if self._broken:
                raise RuntimeError("I2C write failed")
            if name == "body_stop":
                for b in BUFFER_NAMES:
                    getattr(self, b).clear()
            else:
                getattr(self, f"{name.split('_')[0]}_action_buffer").clear()
        return fn


def test_depth_sums_all_three_buffers():
    assert buffer_depth(FakeDog(legs=179, head=40, tail=8)) == 227


def test_depth_survives_missing_buffers():
    class Bare:
        pass
    assert buffer_depth(Bare()) == 0


def test_body_stop_is_preferred_and_reports_the_count():
    dog = FakeDog(legs=200, head=20, tail=7)
    r = drain(dog)
    assert dog.calls == ["body_stop"]
    assert r == {"via": "body_stop", "drained": 227, "remaining": 0}


def test_falls_back_to_per_part_stops():
    dog = FakeDog(legs=10, head=5, tail=2, methods=("legs_stop", "head_stop", "tail_stop"))
    r = drain(dog)
    assert dog.calls == ["legs_stop", "head_stop", "tail_stop"]
    assert r["via"] == "part_stop"
    assert r["drained"] == 17 and r["remaining"] == 0


def test_clears_the_lists_when_the_sdk_offers_no_stop_method():
    dog = FakeDog(legs=12, methods=())
    r = drain(dog)
    assert r["via"] == "clear"
    assert r["drained"] == 12 and r["remaining"] == 0
    assert buffer_depth(dog) == 0


def test_a_raising_stop_method_still_empties_the_queue():
    """An emergency stop must never be blocked by a failing SDK call."""
    dog = FakeDog(legs=50, broken=True)
    r = drain(dog)
    assert "I2C write failed" in r["error"]
    assert r["remaining"] == 0, "the queue must be empty even after the error"
    assert r["drained"] == 50


def test_partial_stop_is_completed_by_clearing():
    dog = FakeDog(legs=4, head=3, tail=2, methods=("legs_stop", "head_stop", "tail_stop"))
    dog.tail_stop = lambda: dog.calls.append("tail_stop")  # clears nothing
    r = drain(dog)
    assert r["via"] == "part_stop+clear"
    assert r["drained"] == 9 and r["remaining"] == 0


def test_empty_queue_is_a_no_op_result():
    dog = FakeDog()
    r = drain(dog)
    assert r["drained"] == 0 and r["remaining"] == 0


@pytest.mark.parametrize("name", BUFFER_NAMES)
def test_each_buffer_is_drained(name):
    dog = FakeDog(legs=3, head=3, tail=3, methods=())
    drain(dog)
    assert len(getattr(dog, name)) == 0
