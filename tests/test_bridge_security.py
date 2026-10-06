"""Bridge auth, rate limiting and input checks — issue #30.

shared/security.py implemented all of this and nothing imported it, while
the README claimed it as a feature. Now the bridge uses it; these tests pin
what it lets through and what it refuses.
"""
import json
import urllib.error
import urllib.request

import pytest

import body.nox_brain_bridge as bridge
from body.nox_security import (BridgeSecurity, RateLimiter, TokenAuth, validate_head,
                               validate_name, validate_rgb, validate_text)
from tests.test_bridge_api import server_port  # noqa: F401 - fixture


# ─── token ───

def test_no_token_configured_means_open():
    assert TokenAuth("").check({}) is None


def test_token_required_when_configured():
    auth = TokenAuth("s3cret")
    assert "missing" in auth.check({})
    assert auth.check({"Authorization": "Bearer wrong"}) == "invalid token"
    assert auth.check({"Authorization": "Bearer s3cret"}) is None
    assert auth.check({"Authorization": "bearer s3cret"}) is None


# ─── who is local ───

def test_localhost_skips_auth_and_rate_limit():
    sec = BridgeSecurity(token="s3cret", rate_limit=1)
    for _ in range(5):
        assert sec.check("127.0.0.1", {}) is None


def test_a_tunnel_on_the_robot_is_not_local():
    # cloudflared (docs/remote-access.md) forwards outside requests from 127.0.0.1
    sec = BridgeSecurity(token="s3cret", rate_limit=0)
    status, error, extra = sec.check("127.0.0.1", {"Cf-Connecting-Ip": "203.0.113.9"})
    assert status == 401 and extra["WWW-Authenticate"] == "Bearer"


def test_remote_needs_the_token():
    sec = BridgeSecurity(token="s3cret", rate_limit=0)
    assert sec.check("198.51.100.50", {})[0] == 401
    assert sec.check("198.51.100.50", {"Authorization": "Bearer s3cret"}) is None


# ─── rate limit ───

def test_rate_limit_per_address_with_retry_after():
    now = [1000.0]
    limiter = RateLimiter(max_requests=3, window_seconds=60, clock=lambda: now[0])
    assert [limiter.check("a") for _ in range(3)] == [0, 0, 0]
    wait = limiter.check("a")
    assert wait > 0
    assert limiter.check("b") == 0  # another client is unaffected
    now[0] += 61
    assert limiter.check("a") == 0


def test_rate_limit_zero_is_off():
    limiter = RateLimiter(max_requests=0)
    assert all(limiter.check("a") == 0 for _ in range(1000))


def test_rate_limit_answers_429():
    sec = BridgeSecurity(token="", rate_limit=1)
    assert sec.check("198.51.100.2", {}) is None
    status, error, extra = sec.check("198.51.100.2", {})
    assert status == 429 and int(extra["Retry-After"]) >= 1


# ─── values ───

@pytest.mark.parametrize("args", [(256, 0, 0), (-1, 0, 0), ("red", 0, 0), (True, 0, 0)])
def test_bad_rgb_is_refused(args):
    assert validate_rgb(*args)[1]


def test_good_rgb_passes():
    assert validate_rgb(255, 0, 128.0) == ((255, 0, 128, 0.8), None)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1e9, "left"])
def test_bad_head_angle_is_refused(value):
    assert validate_head(yaw=value)[1]


def test_head_angles_within_reach_pass():
    assert validate_head(45, -10, 20) == ((45.0, -10.0, 20.0), None)


def test_text_is_cleaned_and_bounded():
    assert validate_text("  Hallo\x00 Welt \n") == ("Hallo Welt", None)
    assert validate_text("x" * 1001)[1]
    assert validate_text(42)[1]


@pytest.mark.parametrize("name", ["../../etc/x", "a/b", ".hidden", "", "x" * 51, "a..b"])
def test_face_names_cannot_become_paths(name):
    assert validate_name(name)[1]


@pytest.mark.parametrize("name", ["Rocky", "Jean-Luc", "Thio 007", "Zoë", "dr.who"])
def test_ordinary_face_names_pass(name):
    assert validate_name(name) == (name, None)


# ─── through the real handler ───

def post(port, path, data, headers=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_bridge_refuses_a_bad_rgb_before_the_daemon(server_port, monkeypatch):  # noqa: F811
    sent = []
    monkeypatch.setattr(bridge, "send_to_daemon", lambda p, timeout=30: sent.append(p) or {"ok": True})
    status, body = post(server_port, "/rgb", {"r": 999})
    assert status == 400 and "between 0 and 255" in body["error"]
    assert sent == []


def test_bridge_refuses_a_path_as_face_name(server_port):  # noqa: F811
    status, body = post(server_port, "/face/register", {"name": "../../x"})
    assert status == 400


def test_bridge_asks_tunnelled_requests_for_the_token(server_port, monkeypatch):  # noqa: F811
    monkeypatch.setattr(bridge, "security", BridgeSecurity(token="s3cret", rate_limit=0))
    status, body = post(server_port, "/head", {"yaw": 10}, {"X-Forwarded-For": "203.0.113.9"})
    assert status == 401 and body["ok"] is False


def test_status_reports_the_security_settings(server_port, monkeypatch):  # noqa: F811
    monkeypatch.setattr(bridge, "send_to_daemon", lambda p, timeout=30: {})
    with urllib.request.urlopen(f"http://127.0.0.1:{server_port}/status", timeout=10) as r:
        body = json.loads(r.read())
    assert body["security"]["auth"] in ("on", "off")


def test_body_client_sends_the_token(monkeypatch):
    from brain import nox_body_client
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["auth"] = req.get_header("Authorization")
        raise OSError("no robot here")

    monkeypatch.setattr(nox_body_client.urllib.request, "urlopen", fake_urlopen)
    nox_body_client.BodyClient("robot.local", 8888, token="s3cret").status()
    assert seen["auth"] == "Bearer s3cret"
    nox_body_client.BodyClient("robot.local", 8888, token="").status()
    assert seen["auth"] is None
