"""Live tests for the bridge HTTP handler — no hardware needed.

The bridge talks to the daemon over TCP, so the handler itself can run
anywhere. These tests pin the loud-failure behavior from issues #12/#13:
broken JSON and empty input must produce actionable errors, never a silent ok.
"""
import json
import threading
import urllib.error
import urllib.request

import pytest

import body.nox_brain_bridge as bridge


@pytest.fixture(scope="module")
def server_port():
    srv = bridge.ThreadedHTTPServer(("127.0.0.1", 0), bridge.BridgeHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield port
    srv.shutdown()


def post(port, path, data):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data.encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        r = urllib.request.urlopen(req, timeout=10)
        return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_broken_json_is_loud_400(server_port):
    # Swapped shell quotes mangle the JSON exactly like this (issue #12).
    code, body = post(server_port, "/action", "{action: sit}")
    assert code == 400
    assert not body["ok"]
    assert "not valid JSON" in body["error"]
    assert body["received"] == "{action: sit}"
    assert "single quotes OUTSIDE" in body["hint"]


def test_empty_action_is_loud(server_port):
    code, body = post(server_port, "/action", "{}")
    assert not body["ok"]
    assert body["error"] == "no action given"
    assert "valid_actions" in body


def test_singular_action_is_forwarded(server_port):
    # No daemon is running in CI: the action must still be forwarded and the
    # connection error surfaced per-result — not swallowed.
    code, body = post(server_port, "/action", '{"action": "sit"}')
    assert len(body["results"]) == 1
    assert not body["ok"]  # daemon unreachable -> error propagates


def test_array_actions_forwarded_per_item(server_port):
    code, body = post(server_port, "/action", '{"actions": ["sit", "wag_tail"]}')
    assert len(body["results"]) == 2


def test_body_client_class_against_live_handler(server_port):
    # The class the README and examples use — must exist and speak the API.
    from brain.nox_body_client import BodyClient

    robot = BodyClient("127.0.0.1", server_port, timeout=10)
    result = robot.move("sit")
    # No daemon in CI: the bridge must forward and surface the connection
    # error — any dict response proves request/response plumbing works.
    assert isinstance(result, dict)
    caps = robot.capabilities()
    assert caps.get("ok") and "sit" in caps.get("actions", [])


class _FakeEngine:
    """Behaviour engine stand-in: records that it was asked to stop."""

    def __init__(self):
        self.running = True
        self.stopped = False

    def stop(self):
        self.stopped = True
        self.running = False


def test_behavior_stop_drains_the_motion_queue_before_sleeping(server_port, monkeypatch):
    """Issue #25: stopping the engine must also drop the frames it already
    queued, and do so BEFORE the sleep pose — otherwise the `lie` waits behind
    227 pending frames and the dog keeps moving."""
    sent = []

    def fake_daemon(payload, timeout=10):
        sent.append(payload["cmd"])
        if payload["cmd"] == "stop_motion":
            return {"ok": True, "drained": 227, "remaining": 0, "via": "body_stop"}
        return {"ok": True}

    engine = _FakeEngine()
    monkeypatch.setattr(bridge, "send_to_daemon", fake_daemon)
    monkeypatch.setattr(bridge, "_behavior_engine", engine)

    code, body = post(server_port, "/behavior/stop", "{}")

    assert code == 200
    assert body["ok"] and body["stopped"]
    assert body["motion_frames_dropped"] == 227
    assert engine.stopped
    assert sent == ["stop_motion", "sleep"], "the queue must be cleared first"


def test_behavior_stop_surfaces_a_failed_drain(server_port, monkeypatch):
    def fake_daemon(payload, timeout=10):
        if payload["cmd"] == "stop_motion":
            return {"error": "[Errno 111] Connection refused"}
        return {"ok": True}

    monkeypatch.setattr(bridge, "send_to_daemon", fake_daemon)
    monkeypatch.setattr(bridge, "_behavior_engine", _FakeEngine())

    code, body = post(server_port, "/behavior/stop", "{}")

    assert body["stopped"]
    assert "Connection refused" in body["motion_error"]
    assert "motion_frames_dropped" not in body


def test_sensors_survives_a_dead_i2c_bus_without_the_behavior_engine(server_port, monkeypatch):
    """Regression: the I2C warning added in #22 left the obstacle lines inside
    its own branch, so /sensors raised NameError when the engine was disabled
    (NOX_NO_AUTO=1) — exactly the state the troubleshooting docs ask users to
    inspect."""
    def fake_daemon(payload, timeout=10):
        return {
            "ts": 1.0, "hostname": "cat1", "uptime_s": 10, "disk_free_gb": 1.0,
            "battery_v": "error",
            "i2c": {"responding": False, "verdict": "wrong_address",
                    "error": "robot_hat talks to 0x14 but the MCU answers on 0x15",
                    "hint": "put /usr/sbin on the service PATH"},
        }

    monkeypatch.setattr(bridge, "send_to_daemon", fake_daemon)
    monkeypatch.setattr(bridge, "_behavior_engine", None)

    req = urllib.request.Request(f"http://127.0.0.1:{server_port}/sensors")
    with urllib.request.urlopen(req, timeout=10) as r:
        code, body = r.status, json.loads(r.read())

    assert code == 200
    assert body["i2c"]["verdict"] == "wrong_address"
    assert "unreachable" in body["warning"]
    assert "/usr/sbin" in body["hint"]


def test_sensors_reports_obstacles_when_the_bus_is_healthy(server_port, monkeypatch):
    """The same bug also made obstacle distances appear ONLY while I2C was
    broken. With a healthy bus and a running engine they must be present."""
    class Engine:
        running = True

        def get_state(self):
            return {"state": "patrol", "dominant_mood": "curious",
                    "patrol_enabled": True, "low_battery": False,
                    "obstacles": {"last_scan": {"forward": 42.5}, "scan_age_s": 1.2}}

    def fake_daemon(payload, timeout=10):
        return {"ts": 1.0, "hostname": "cat1", "uptime_s": 10, "disk_free_gb": 1.0,
                "battery_v": 7.6, "i2c": {"responding": True, "verdict": "ok"}}

    monkeypatch.setattr(bridge, "send_to_daemon", fake_daemon)
    monkeypatch.setattr(bridge, "_behavior_engine", Engine())

    with urllib.request.urlopen(
            f"http://127.0.0.1:{server_port}/sensors", timeout=10) as r:
        body = json.loads(r.read())

    assert body["distance"]["forward_cm"] == 42.5
    assert body["distance"]["scan_age_s"] == 1.2
    assert "warning" not in body


def test_sensors_handles_an_unreadable_battery(server_port, monkeypatch):
    """battery_v is the string "error" when the ADC read fails; the percentage
    maths turned that into HTTP 500."""
    def fake_daemon(payload, timeout=10):
        return {"ts": 1.0, "battery_v": "error",
                "battery_error": "RuntimeError: I2C: MCU not responding"}

    monkeypatch.setattr(bridge, "send_to_daemon", fake_daemon)
    monkeypatch.setattr(bridge, "_behavior_engine", None)

    with urllib.request.urlopen(
            f"http://127.0.0.1:{server_port}/sensors", timeout=10) as r:
        assert r.status == 200
        body = json.loads(r.read())

    assert body["battery"]["voltage"] == "error"
    assert body["battery"]["readable"] is False
    assert body["battery"]["percent"] is None
    assert body["battery"]["charging"] is False
