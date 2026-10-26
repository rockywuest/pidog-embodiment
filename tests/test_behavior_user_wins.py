"""Autonomy must yield to people and must not walk blind (issue #68).

Reported: in patrol "no detection, always wag tail even if command lie".
Causes found in the engine: a state forced by /behavior/start was restored
by the main loop every two seconds, so any user command was undone at once;
forward_clear() treated a missing ultrasonic reading as "clear", so a dead
sensor meant patrolling blind; the vision branch called a method that does
not exist (self._transition) and crashed the loop on every person; every
touch reading wagged the tail, with no cooldown.
"""
import json
import threading
import urllib.request

import pytest

import body.nox_behavior_engine as be
import body.nox_brain_bridge as bridge


class Recorder:
    """Stands in for the daemon; answers scans with a chosen forward reading."""

    def __init__(self, forward=-1):
        self.forward = forward
        self.cmds = []

    def __call__(self, cmd):
        self.cmds.append(dict(cmd))
        if cmd.get("cmd") == "three_way_scan":
            return {"ok": True, "scan": {"forward": self.forward, "left": -1, "right": -1}}
        return {"ok": True}

    def moves(self):
        return [c.get("action") for c in self.cmds if c.get("cmd") == "move"]


def engine(forward=-1):
    rec = Recorder(forward)
    e = be.BehaviorEngine(daemon_send_fn=rec)
    e._read_vision = lambda: None
    return e, rec


# ─── the person wins ───

def test_a_user_command_ends_a_forced_patrol_without_moving():
    e, rec = engine()
    e.force_state("patrol")
    assert e.state == be.BehaviorState.PATROL and e._forced_state is not None
    rec.cmds.clear()
    e.user_command("/action ['lie']")
    assert e.state == be.BehaviorState.IDLE
    assert e._forced_state is None      # else the main loop restores patrol
    assert e.user_hold_active()
    assert rec.moves() == []            # stepping aside must not move the dog


def test_hold_zero_disables_the_feature(monkeypatch):
    monkeypatch.setattr(be, "USER_HOLD_S", 0)
    e, _ = engine()
    e.force_state("patrol")
    e.user_command("lie")
    assert e.state == be.BehaviorState.PATROL and not e.user_hold_active()


def test_an_explicit_behavior_request_ends_the_hold():
    e, _ = engine()
    e.user_command("lie")
    e.force_state("patrol")
    assert not e.user_hold_active() and e.state == be.BehaviorState.PATROL


# ─── touch wags, but not non-stop and not over a command ───

def test_touch_wags_at_most_once_per_cooldown():
    e, rec = engine()
    e._handle_touch("L")
    e._handle_touch("L")
    e._handle_touch("R")
    assert rec.moves().count("wag_tail") == 1


def test_no_touch_wag_while_a_command_holds():
    e, rec = engine()
    e.user_command("lie")
    e._handle_touch("L")
    assert "wag_tail" not in rec.moves()


# ─── patrol never walks blind ───

def test_dead_ultrasonic_stops_patrol_instead_of_walking_blind(capsys):
    e, rec = engine(forward=-1)
    e.force_state("patrol")
    rec.cmds.clear()
    for _ in range(be.BLIND_PATROL_TICKS):
        e._tick_patrol()
    assert "forward" not in rec.moves()
    assert e.state == be.BehaviorState.IDLE and e._forced_state is None
    assert "will not walk blind" in capsys.readouterr().out


def test_after_a_blind_stop_mood_cannot_restart_patrol_but_a_request_can():
    e, _ = engine(forward=-1)
    e.force_state("patrol")
    for _ in range(be.BLIND_PATROL_TICKS):
        e._tick_patrol()
    e._state_entered_at = 0           # dwell satisfied
    for _ in range(3):
        e._request_transition(be.BehaviorState.PATROL)
    assert e.state == be.BehaviorState.IDLE
    e.force_state("patrol")
    assert e.state == be.BehaviorState.PATROL


def test_a_working_sensor_walks():
    e, rec = engine(forward=120.0)
    e.force_state("patrol")
    rec.cmds.clear()
    e._tick_patrol()
    assert "forward" in rec.moves()


def test_one_dropped_echo_does_not_stop_patrol():
    e, rec = engine(forward=-1)
    e.force_state("patrol")
    e._tick_patrol()                  # one blind tick: wait, don't walk
    assert e.state == be.BehaviorState.PATROL
    rec.forward = 120.0
    e._last_patrol_scan = 0           # next tick scans again
    e._tick_patrol()
    assert "forward" in rec.moves() and e._blind_ticks == 0


def test_a_person_in_view_no_longer_crashes_the_patrol():
    e, _ = engine(forward=120.0)
    e.force_state("patrol")
    e._read_vision = lambda: {"description": "a person standing near the sofa"}
    e._state_entered_at = 0           # dwell satisfied
    e._tick_patrol()                  # used to raise AttributeError: _transition
    assert e._candidate_state == be.BehaviorState.INVESTIGATE


# ─── the bridge tells the engine ───

class FakeEngine:
    running = True

    def __init__(self):
        self.calls = []

    def user_command(self, what):
        self.calls.append(what)


@pytest.fixture
def served(monkeypatch):
    fake = FakeEngine()
    monkeypatch.setattr(bridge, "_behavior_engine", fake)
    monkeypatch.setattr(bridge, "send_to_daemon", lambda *a, **k: {"ok": True})
    srv = bridge.ThreadedHTTPServer(("127.0.0.1", 0), bridge.BridgeHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1], fake
    srv.shutdown()


def post(port, path, payload):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


@pytest.mark.parametrize("path,payload", [
    ("/action", {"action": "lie"}),
    ("/combo", {"actions": ["sit"], "speak": ""}),
    ("/move", {"direction": "forward", "distance_cm": 10}),
])
def test_movement_endpoints_put_autonomy_on_hold(served, path, payload):
    port, fake = served
    post(port, path, payload)
    assert len(fake.calls) == 1


def test_speech_alone_does_not_pause_autonomy(served):
    port, fake = served
    post(port, "/combo", {"actions": [], "speak": "bonjour"})
    assert fake.calls == []


def test_the_legacy_engine_without_the_hook_is_fine(monkeypatch):
    class Legacy:
        running = True
    monkeypatch.setattr(bridge, "_behavior_engine", Legacy())
    bridge.note_user_command("/action ['lie']")  # must not raise


def test_a_bored_dog_does_not_set_off_by_itself():
    """Field report: the dog started patrolling unattended, cables still
    plugged in. Mood-driven patrol is opt-in (NOX_AUTO_PATROL)."""
    e, rec = engine(forward=120.0)
    assert e._patrol_enabled is be.AUTO_PATROL is False
    e._state_entered_at = 0
    for _ in range(3):
        e._request_transition(be.BehaviorState.PATROL)
    e.mood.boredom = 1.0
    e._last_behavior_time = 0
    e._tick_idle(10_000)
    assert e.state != be.BehaviorState.PATROL
    assert "forward" not in rec.moves()


def test_an_explicit_request_still_patrols():
    e, rec = engine(forward=120.0)
    assert e.force_state("patrol")["ok"]
    assert e.state == be.BehaviorState.PATROL
    rec.cmds.clear()
    e._tick_patrol()
    assert "forward" in rec.moves()


def test_auto_patrol_can_be_turned_back_on(monkeypatch):
    monkeypatch.setattr(be, "AUTO_PATROL", True)
    e, _ = engine(forward=120.0)
    e._state_entered_at = 0
    for _ in range(3):
        e._request_transition(be.BehaviorState.PATROL)
    assert e.state == be.BehaviorState.PATROL


@pytest.mark.parametrize("leave_to", ["idle", "investigate", "alert"])
def test_a_finished_patrol_is_not_resumed(leave_to):
    """Stuck, touched or a person seen: the patrol ends. The main loop used to
    restore the forced patrol two seconds later — a walk nobody asked for."""
    e, rec = engine(forward=120.0)
    e.force_state("patrol")
    e._do_transition(be.BehaviorState(leave_to))
    rec.cmds.clear()
    e._honor_forced_state()
    assert e.state == be.BehaviorState(leave_to) and e._forced_state is None
    assert "forward" not in rec.moves()


def test_a_forced_rest_is_still_restored():
    e, _ = engine()
    e.force_state("rest")
    e._do_transition(be.BehaviorState.IDLE)
    e._honor_forced_state()
    assert e.state == be.BehaviorState.REST
