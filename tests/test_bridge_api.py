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
