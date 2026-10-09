"""The hardware-free dog (examples/mock_body.py) — and the MCP server against it.

The mock exists so people without a PiDog can try the project; its API shapes
must therefore match what real clients expect. The best proof: the MCP server
talks to it over real HTTP and every tool works.
"""
import importlib.util
import json
import threading
import urllib.request

import pytest

import brain.nox_mcp_server as mcp

spec = importlib.util.spec_from_file_location("mock_body", "examples/mock_body.py")
mock_body = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mock_body)


@pytest.fixture(scope="module")
def mock_port():
    mock_body.Handler.dog = mock_body.Dog()
    srv = mock_body.Server(("127.0.0.1", 0), mock_body.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


@pytest.fixture
def via_mcp(mock_port, monkeypatch):
    monkeypatch.setattr(mcp, "BASE_URL", f"http://127.0.0.1:{mock_port}")

    def call(tool, arguments=None):
        r = mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                        "params": {"name": tool, "arguments": arguments or {}}})
        return r["result"]
    return call


def test_every_mcp_tool_works_against_the_mock(via_mcp):
    for tool, args in [
        ("dog_status", {}), ("dog_sensors", {}), ("dog_vision", {}),
        ("dog_action", {"action": "sit"}), ("dog_speak", {"text": "Hallo Welt"}),
        ("dog_look_at", {"direction": "left"}), ("dog_expression", {"emotion": "happy"}),
        ("dog_rgb", {"r": 0, "g": 255, "b": 0}),
        ("dog_behavior", {"mode": "start"}), ("dog_behavior", {"mode": "emergency_stop"}),
    ]:
        result = via_mcp(tool, args)
        assert result["isError"] is False, (tool, result)


def test_the_mock_photo_reaches_the_model_as_an_image(via_mcp):
    result = via_mcp("dog_photo")
    assert result["isError"] is False
    img = result["content"][0]
    assert img["type"] == "image" and img["mimeType"] == "image/jpeg"
    import base64
    assert base64.b64decode(img["data"])[:3] == b"\xff\xd8\xff"  # real JPEG magic


def test_unknown_action_is_refused_like_the_real_bridge(via_mcp):
    result = via_mcp("dog_action", {"action": "backflip"})
    assert result["isError"] is True
    assert "Unknown action" in result["content"][0]["text"]


def test_mock_actions_match_the_bridges_list():
    import re
    src = open("body/nox_brain_bridge.py").read()
    block = re.search(r"VALID_ACTIONS = \[(.*?)\]", src, re.S).group(1)
    assert set(mock_body.VALID_ACTIONS) == set(re.findall(r'"(\w+)"', block))


def test_voice_input_says_it_has_no_brain(mock_port):
    req = urllib.request.Request(f"http://127.0.0.1:{mock_port}/voice/input",
                                 data=json.dumps({"text": "Sitz"}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        body = json.loads(r.read())
    assert body["ok"] is False and "brain" in body["error"]


def test_install_sh_parses():
    import subprocess
    assert subprocess.run(["bash", "-n", "install.sh"]).returncode == 0
