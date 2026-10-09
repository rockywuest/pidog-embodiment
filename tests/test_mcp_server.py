"""The robot as MCP tools — brain/nox_mcp_server.py.

Protocol handling is hand-rolled (stdio JSON-RPC, zero dependencies), so these
tests pin the handshake, the tool list, dispatch, the image answer for photos,
and that a dead robot comes back as a tool error — never as a crash.
"""
import json

import pytest

import brain.nox_mcp_server as mcp


@pytest.fixture
def fake_bridge(monkeypatch):
    calls = []
    replies = {}

    def bridge(path, payload=None, timeout=30):
        calls.append((path, payload))
        return replies.get(path, {"ok": True, "path": path})

    monkeypatch.setattr(mcp, "bridge", bridge)
    bridge.calls, bridge.replies = calls, replies
    return bridge


def rpc(method, params=None, msg_id=1):
    return mcp.handle({"jsonrpc": "2.0", "id": msg_id, "method": method,
                       "params": params or {}})


# ─── protocol ───

def test_initialize_negotiates_a_known_protocol_version():
    r = rpc("initialize", {"protocolVersion": "2025-03-26",
                           "capabilities": {}, "clientInfo": {"name": "x", "version": "1"}})
    assert r["result"]["protocolVersion"] == "2025-03-26"
    assert r["result"]["capabilities"] == {"tools": {}}
    assert "robot dog" in r["result"]["instructions"]


def test_initialize_with_an_unknown_version_offers_the_newest():
    r = rpc("initialize", {"protocolVersion": "9999-01-01"})
    assert r["result"]["protocolVersion"] == mcp.PROTOCOL_VERSIONS[0]


def test_ping_and_unknown_method_and_notification():
    assert rpc("ping")["result"] == {}
    assert rpc("no/such/method")["error"]["code"] == -32601
    assert mcp.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_tools_list_names_every_tool_with_a_schema():
    tools = rpc("tools/list")["result"]["tools"]
    names = {t["name"] for t in tools}
    assert {"dog_action", "dog_speak", "dog_photo", "dog_vision", "dog_status",
            "dog_sensors", "dog_look_at", "dog_expression", "dog_rgb",
            "dog_behavior"} == names
    for t in tools:
        assert t["description"] and t["inputSchema"]["type"] == "object"


def test_action_enum_matches_the_bridges_valid_actions():
    import re
    src = open("body/nox_brain_bridge.py").read()
    block = re.search(r"VALID_ACTIONS = \[(.*?)\]", src, re.S).group(1)
    bridge_actions = set(re.findall(r'"(\w+)"', block))
    assert set(mcp.ACTIONS) == bridge_actions


# ─── dispatch ───

def test_action_is_posted_with_steps(fake_bridge):
    r = rpc("tools/call", {"name": "dog_action", "arguments": {"action": "sit", "steps": 2}})
    assert fake_bridge.calls == [("/action", {"action": "sit", "steps": 2})]
    assert r["result"]["isError"] is False


def test_speak_waits_for_the_real_outcome(fake_bridge):
    rpc("tools/call", {"name": "dog_speak", "arguments": {"text": "Hallo"}})
    assert fake_bridge.calls == [("/speak", {"text": "Hallo", "blocking": True})]


def test_a_failed_speak_is_a_tool_error(fake_bridge):
    fake_bridge.replies["/speak"] = {"ok": False, "error": "piper voice model not found"}
    r = rpc("tools/call", {"name": "dog_speak", "arguments": {"text": "Hallo"}})
    assert r["result"]["isError"] is True
    assert "voice model" in r["result"]["content"][0]["text"]


def test_photo_returns_an_image_block(fake_bridge):
    fake_bridge.replies["/photo"] = {"ok": True, "photo_b64": "aGVsbG8=", "faces": [{"name": "?"}]}
    r = rpc("tools/call", {"name": "dog_photo", "arguments": {}})
    img, note = r["result"]["content"]
    assert img == {"type": "image", "data": "aGVsbG8=", "mimeType": "image/jpeg"}
    assert "1 face(s)" in note["text"]


def test_photo_without_a_camera_is_a_tool_error(fake_bridge):
    fake_bridge.replies["/photo"] = {"ok": False, "error": "camera init failed"}
    r = rpc("tools/call", {"name": "dog_photo", "arguments": {}})
    assert r["result"]["isError"] is True


def test_behavior_modes_map_to_their_endpoints(fake_bridge):
    for mode, path in [("start", "/behavior/start"), ("stop", "/behavior/stop"),
                       ("emergency_stop", "/emergency_stop")]:
        fake_bridge.calls.clear()
        rpc("tools/call", {"name": "dog_behavior", "arguments": {"mode": mode}})
        assert fake_bridge.calls[0][0] == path


def test_unknown_tool_is_a_jsonrpc_error(fake_bridge):
    assert rpc("tools/call", {"name": "dog_fly"})["error"]["code"] == -32602


def test_missing_required_argument_is_a_tool_error_not_a_crash(fake_bridge):
    r = rpc("tools/call", {"name": "dog_action", "arguments": {}})
    assert r["result"]["isError"] is True


def test_an_unreachable_robot_is_data_not_an_exception(monkeypatch):
    # the real bridge() catches network errors itself — simulate at urlopen level
    def boom(req, timeout=None):
        raise OSError("No route to host")
    monkeypatch.setattr(mcp.urllib.request, "urlopen", boom)
    r = rpc("tools/call", {"name": "dog_status", "arguments": {}})
    assert r["result"]["isError"] is True
    assert "not reachable" in r["result"]["content"][0]["text"]


def test_status_is_trimmed_to_the_essentials(fake_bridge):
    fake_bridge.replies["/status"] = {
        "ok": True, "battery_v": 7.9, "uptime_s": 12, "perception": {"huge": "blob"},
        "known_faces": {}, "sensors": {"last_speak": {"ok": True}},
        "security": {"auth": "off"}}
    r = rpc("tools/call", {"name": "dog_status", "arguments": {}})
    body = json.loads(r["result"]["content"][0]["text"])
    assert body["battery_v"] == 7.9 and body["last_speak"] == {"ok": True}
    assert "perception" not in body


@pytest.mark.parametrize("bad", [[{"jsonrpc": "2.0", "id": 1, "method": "ping"}], "ping", 42, None])
def test_a_batch_or_non_object_is_an_invalid_request_not_a_crash(bad):
    r = mcp.handle(bad)
    assert r["error"]["code"] == -32600
