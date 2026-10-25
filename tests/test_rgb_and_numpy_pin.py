"""Fresh-install field fixes: a named LED color is honoured, and the camera survives pip.

"color": "cyan" used to be dropped silently and the strip stayed purple while
/rgb said ok. And body/requirements.txt let pip install numpy 2, which breaks
Bookworm's picamera2/simplejpeg ("numpy.dtype size changed") — the camera of
every freshly installed dog was dead.
"""
import re
from pathlib import Path

import pytest

import body.nox_brain_bridge as bridge
from body.nox_security import RGB_MODES, parse_rgb_request
from tests.test_bridge_api import server_port  # noqa: F401 - fixture
from tests.test_bridge_security import post

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("body, rgb", [
    ({"color": "cyan"}, (0, 255, 255)),
    ({"color": "Cyan "}, (0, 255, 255)),
    ({"color": "#00ff80"}, (0, 255, 128)),
    ({"color": "ff0000"}, (255, 0, 0)),
    ({"r": 1, "g": 2, "b": 3}, (1, 2, 3)),
    ({}, (128, 0, 255)),
])
def test_colors_resolve(body, rgb):
    parsed, err = parse_rgb_request(body)
    assert err is None and parsed[:3] == rgb


def test_mode_and_bps_pass_through():
    assert parse_rgb_request({"color": "red", "mode": "boom", "bps": 2})[0] == (255, 0, 0, "boom", 2.0)
    assert parse_rgb_request({"mode": "off"})[0][3] == "off"
    assert parse_rgb_request({"color": "off"})[0][3] == "off"
    assert parse_rgb_request({"color": "black", "mode": "boom"})[0][3] == "boom"


@pytest.mark.parametrize("body, needle", [
    ({"color": "cyan", "style": "breath"}, "style"),
    ({"colour": "cyan"}, "colour"),
    ({"color": "teal-ish"}, "unknown color"),
    ({"color": "cyan", "r": 0}, "not both"),
    ({"color": 5}, "string"),
    ({"mode": "rainbow"}, "unknown mode"),
    ({"r": 999}, "r"),
    ({"color": None}, "string"),
    ("cyan", "JSON object"),
    ([1], "JSON object"),
])
def test_bad_requests_say_why(body, needle):
    parsed, err = parse_rgb_request(body)
    assert parsed is None and needle in err


def test_bridge_refuses_unknown_rgb_fields(server_port, monkeypatch):  # noqa: F811
    sent = []
    monkeypatch.setattr(bridge, "send_to_daemon", lambda p, timeout=30: sent.append(p) or {"ok": True})
    status, body = post(server_port, "/rgb", {"color": "cyan", "style": "breath"})
    assert status == 400 and "style" in body["error"] and sent == []


def test_bridge_sends_the_named_color(server_port, monkeypatch):  # noqa: F811
    sent = []
    monkeypatch.setattr(bridge, "send_to_daemon", lambda p, timeout=30: sent.append(p) or {"ok": True})
    status, _ = post(server_port, "/rgb", {"color": "cyan", "mode": "breath"})
    assert status == 200 and (sent[0]["r"], sent[0]["g"], sent[0]["b"]) == (0, 255, 255)


def test_daemon_and_bridge_agree_on_modes():
    daemon = (ROOT / "body" / "nox_daemon.py").read_text()
    styles = set(re.search(r"_VALID_RGB_STYLES = \{([^}]*)\}", daemon).group(1).replace('"', "").replace(" ", "").split(","))
    assert set(RGB_MODES) == styles | {"off"}


def test_requirements_keep_numpy_below_2():
    req = (ROOT / "body" / "requirements.txt").read_text()
    assert re.search(r"^numpy[^\n]*<2\b", req, re.M)
    assert re.search(r"^opencv-python-headless[^\n]*<4\.12\b", req, re.M)


def test_combo_honours_a_named_color(server_port, monkeypatch):  # noqa: F811
    # brain/nox_voice_brain.py forwards the LLM's own rgb object to /combo
    sent = []
    monkeypatch.setattr(bridge, "send_to_daemon", lambda p, timeout=30: sent.append(p) or {"ok": True})
    status, _ = post(server_port, "/combo", {"rgb": {"color": "cyan"}})
    rgb = [p for p in sent if p.get("cmd") == "rgb"]
    assert status == 200 and (rgb[0]["r"], rgb[0]["g"], rgb[0]["b"]) == (0, 255, 255)
    sent.clear()
    status, body = post(server_port, "/combo", {"rgb": {"colour": "cyan"}, "actions": ["sit"]})
    assert status == 400 and "colour" in body["error"] and sent == []


def test_mcp_rgb_without_a_color_asks_for_one(monkeypatch):
    import brain.nox_mcp_server as mcp
    monkeypatch.setattr(mcp, "bridge", lambda *a, **k: pytest.fail("must not reach the robot"))
    assert mcp.tool_rgb({})["isError"] is True
