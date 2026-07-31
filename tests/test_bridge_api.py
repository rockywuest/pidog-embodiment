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
