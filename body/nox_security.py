"""Who may drive the robot, how often, and with what values (issue #30).

The bridge is an HTTP API that moves a physical robot. This code used to sit
in shared/security.py, imported by nothing, while the README advertised it —
so every request was unauthenticated, unthrottled and passed straight through.

Defaults keep existing installs working:

- **Token auth is optional.** Set NOX_API_TOKEN in body/nox.env and every
  request from another machine needs ``Authorization: Bearer <token>``. The
  robot's own services (behavior engine, voice loop) talk to the bridge over
  localhost and are not asked for it — unless the request came through a
  proxy or tunnel on the robot (X-Forwarded-For & co.), which is outside.
- **Rate limiting is on**: NOX_RATE_LIMIT requests per minute per remote
  address (default 600 — ten a second, far above what the brain sends; 0 turns
  it off). Localhost is exempt.
- **Values are checked** before they reach the servos and LEDs, and a bad one is
  a 400 with the reason, not a daemon traceback.

Kept free of SDK imports so it can be tested anywhere.
"""
import hmac
import math
import os
import re
import threading
import time
from collections import defaultdict, deque

LOCAL_ADDRESSES = {"127.0.0.1", "::1", "::ffff:127.0.0.1"}
# A tunnel or reverse proxy on the robot (docs/remote-access.md: cloudflared)
# delivers outside requests from 127.0.0.1 — these headers give it away.
PROXY_HEADERS = ("X-Forwarded-For", "X-Real-IP", "Forwarded", "Cf-Connecting-Ip")
DEFAULT_RATE_LIMIT = 600  # per remote address per minute


class TokenAuth:
    """Bearer token check. Disabled while no token is configured."""

    def __init__(self, token=None):
        self.token = token if token is not None else os.environ.get("NOX_API_TOKEN", "")
        self.token = (self.token or "").strip()
        self.enabled = bool(self.token)

    def check(self, headers):
        """Returns None when allowed, otherwise the reason."""
        if not self.enabled:
            return None
        auth = (headers.get("Authorization") or "").strip()
        if not auth:
            return "missing Authorization header (Bearer token required: NOX_API_TOKEN)"
        provided = auth[7:].strip() if auth.lower().startswith("bearer ") else auth
        if hmac.compare_digest(provided.encode(), self.token.encode()):
            return None
        return "invalid token"


class RateLimiter:
    """Sliding window per client address."""

    def __init__(self, max_requests=DEFAULT_RATE_LIMIT, window_seconds=60, clock=time.monotonic):
        self.max_requests = max_requests
        self.window = window_seconds
        self._clock = clock
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    @property
    def enabled(self):
        return self.max_requests > 0

    def check(self, client):
        """Returns 0 when allowed, otherwise the seconds to wait."""
        if not self.enabled:
            return 0
        now = self._clock()
        with self._lock:
            hits = self._hits[client]
            while hits and hits[0] <= now - self.window:
                hits.popleft()
            if len(hits) >= self.max_requests:
                return max(1, int(hits[0] + self.window - now) + 1)
            hits.append(now)
            # Forget idle clients so the table cannot grow without bound.
            if len(self._hits) > 1024:
                for ip in [ip for ip, h in self._hits.items() if not h]:
                    del self._hits[ip]
            return 0


class BridgeSecurity:
    """What the bridge asks before handling a request."""

    def __init__(self, token=None, rate_limit=None, clock=time.monotonic):
        self.auth = TokenAuth(token)
        if rate_limit is None:
            rate_limit = _int_env("NOX_RATE_LIMIT", DEFAULT_RATE_LIMIT)
        self.limiter = RateLimiter(rate_limit, clock=clock)

    def check(self, client_ip, headers):
        """Returns None when allowed, otherwise (status, error, extra_headers)."""
        if client_ip in LOCAL_ADDRESSES and not any(headers.get(h) for h in PROXY_HEADERS):
            return None
        reason = self.auth.check(headers)
        if reason:
            return 401, reason, {"WWW-Authenticate": "Bearer"}
        wait = self.limiter.check(client_ip)
        if wait:
            return 429, f"rate limited: more than {self.limiter.max_requests} requests a minute, retry in {wait}s", {"Retry-After": str(wait)}
        return None

    def status(self):
        return {"auth": "on" if self.auth.enabled else "off",
                "rate_limit_per_min": self.limiter.max_requests if self.limiter.enabled else "off"}


def _int_env(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        print(f"[bridge] WARNING: {name}={raw!r} is not a number, using {default}", flush=True)
        return default


# ─── Input validation: return (value, None) or (None, error) ───

MAX_TEXT_LENGTH = 1000
MAX_NAME_LENGTH = 50
# Generous bounds: the SDK clamps to the servo range itself. These only stop
# values that are not angles at all (strings, NaN, 1e9).
HEAD_LIMIT = 180.0
_NAME_RE = re.compile(r"^[\w][\w \-.]*$", re.UNICODE)
_CONTROL_CHARS = re.compile(r"[\x00-\x09\x0b-\x1f\x7f]")


def _number(name, value, lo, hi):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, f"{name} must be a number, got {value!r}"
    if not math.isfinite(value) or not lo <= value <= hi:
        return None, f"{name} must be between {lo:g} and {hi:g}, got {value!r}"
    return value, None


def validate_rgb(r, g, b, bps=0.8):
    for name, value in (("r", r), ("g", g), ("b", b)):
        _, err = _number(name, value, 0, 255)
        if err:
            return None, err
    _, err = _number("bps", bps, 0, 10)
    if err:
        return None, err
    return (int(r), int(g), int(b), float(bps)), None


def validate_head(yaw=0, roll=0, pitch=0):
    for name, value in (("yaw", yaw), ("roll", roll), ("pitch", pitch)):
        _, err = _number(name, value, -HEAD_LIMIT, HEAD_LIMIT)
        if err:
            return None, err
    return (float(yaw), float(roll), float(pitch)), None


def validate_text(text, max_len=MAX_TEXT_LENGTH):
    if not isinstance(text, str):
        return None, f"text must be a string, got {type(text).__name__}"
    text = _CONTROL_CHARS.sub("", text).strip()
    if not text:
        return None, "text is empty"
    if len(text) > max_len:
        return None, f"text too long ({len(text)} > {max_len} characters)"
    return text, None


def validate_name(name):
    """A face name becomes part of a file name — no paths, no dots up front."""
    if not isinstance(name, str):
        return None, "name must be a string"
    name = name.strip()
    if not name or len(name) > MAX_NAME_LENGTH:
        return None, f"name must be 1-{MAX_NAME_LENGTH} characters"
    if not _NAME_RE.match(name) or ".." in name:
        return None, "name may contain letters, digits, spaces, '-', '_' and '.' only"
    return name, None
