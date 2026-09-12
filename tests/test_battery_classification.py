"""Battery-reading classification — issue #12.

A PiDog whose pack is unplugged (or whose power switch is off) keeps running
happily from USB-C: daemon up, bridge up, /status green, every motion command
answering ok:true — and the dog never moves, because do_action() only writes
PWM values and there is no power behind them.

The engine used to rewrite any reading below 1.0 V to a full 8.4 V, so that
state was indistinguishable from a healthy battery. These tests pin the
three-way split that makes it visible again.
"""
import pytest

from body.nox_behavior_engine import (
    BATTERY_ABSENT_V,
    BATTERY_ASSUMED_V,
    classify_battery,
)


@pytest.mark.parametrize("raw", [0.0, 0.5, BATTERY_ABSENT_V])
def test_zero_rail_is_flagged_not_reported_as_full(raw):
    voltage, missing = classify_battery(raw)
    assert missing is True, "a 0 V rail must be reported, not masked as full"
    # Charge level is unknown, so the mood engine keeps its neutral stand-in.
    assert voltage == BATTERY_ASSUMED_V


@pytest.mark.parametrize("raw", [None, "error", {}, [], True, False])
def test_unreadable_adc_is_not_a_missing_pack(raw):
    """A broken ADC says nothing about servo power — never guess 'no power'."""
    voltage, missing = classify_battery(raw)
    assert missing is False
    assert voltage == BATTERY_ASSUMED_V


@pytest.mark.parametrize("raw", [1.1, 6.2, 6.79, 7.4, 8.4])
def test_real_measurement_passes_through(raw):
    voltage, missing = classify_battery(raw)
    assert missing is False
    assert voltage == raw, "a real reading must reach the low-battery logic intact"


def test_low_battery_threshold_still_reachable():
    """Regression: masking must not swallow a genuinely weak pack either."""
    voltage, missing = classify_battery(6.5)
    assert voltage < 6.8 and missing is False


@pytest.mark.parametrize("missing", [True, False])
def test_get_state_reports_the_flag_separately_from_low_battery(missing):
    """/status must distinguish "no pack" from "weak pack" — issue #12.

    Built without __init__ so the test needs no robot, no threads and no SDK;
    only the attributes get_state() actually reads are filled in.
    """
    import time as _time

    from body.nox_behavior_engine import BehaviorEngine, BehaviorState

    class _Mood:
        as_dict = staticmethod(lambda: {})
        dominant_mood = staticmethod(lambda: "calm")

    class _Obstacles:
        _last_scan = {}
        _last_scan_ts = 0
        _stuck_count = 0
        forward_clear = staticmethod(lambda: True)

    engine = BehaviorEngine.__new__(BehaviorEngine)
    engine.state = BehaviorState.IDLE
    engine.mood = _Mood()
    engine.low_battery_mode = False
    engine.servo_power_missing = missing
    engine._patrol_enabled = True
    engine.obstacles = _Obstacles()
    engine._state_entered_at = _time.time()

    state = engine.get_state()
    assert state["servo_power_missing"] is missing
    # A missing pack is never a *weak* pack: the two must not be merged.
    assert state["low_battery"] is False
