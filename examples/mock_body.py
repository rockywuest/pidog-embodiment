#!/usr/bin/env python3
"""
mock_body.py — a robot dog made of log lines. No hardware needed.

Serves the body bridge's HTTP API on localhost and prints what the "dog" does,
so you can try the brain, the MCP server, or your own client without owning a
PiDog:

    python3 examples/mock_body.py                 # listens on :8888
    # in another terminal, Claude Code becomes its brain:
    claude mcp add pidog --env PIDOG_HOST=127.0.0.1 -- \
        python3 /path/to/brain/nox_mcp_server.py

Implements the endpoints clients actually use: /status, /sensors, /vision,
/photo, /capabilities, /action, /speak, /look_at, /expression, /rgb,
/voice/input, /behavior/start|stop, /emergency_stop. Same shapes as the real
bridge, one pretend dog behind them. Zero dependencies.
"""
import argparse
import json
import random
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

# 1x1 grey JPEG — just enough for clients that want to "see" something.
TINY_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRof"
    "Hh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAAB"
    "AAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AVN//2Q=="
)

VALID_ACTIONS = [
    "stand", "sit", "lie", "forward", "backward", "turn_left", "turn_right",
    "wag_tail", "bark", "trot", "stretch", "push_up", "howling", "pant", "doze_off",
]
EMOTIONS = ["happy", "sad", "excited", "curious", "alert", "sleepy", "scared",
            "angry", "love", "think"]
ACTION_FACES = {"sit": "🐕 *sits*", "stand": "🐕 *stands up*", "lie": "🐕 *lies down*",
                "wag_tail": "🐕 *wags tail happily*", "bark": "🐕 WOOF! WOOF!",
                "forward": "🐾 *trots forward*", "backward": "🐾 *steps back*",
                "turn_left": "↩️  *turns left*", "turn_right": "↪️  *turns right*",
                "howling": "🐺 Awoooooo!", "stretch": "🧘 *stretches*",
                "push_up": "💪 *does a push-up*", "pant": "😛 *pants*",
                "trot": "🐾 *trots in place*", "doze_off": "😴 *dozes off*"}


class Dog:
    """The whole robot, as state + print statements."""

    def __init__(self):
        self.posture = "stand"
        self.rgb = {"r": 128, "g": 0, "b": 255, "mode": "breath"}
        self.behavior = "off"
        self.started = time.time()

    def log(self, line):
        print(f"  {line}", flush=True)

    def act(self, action, steps=3):
        if action not in VALID_ACTIONS:
            return {"ok": False, "error": f"Unknown action: {action}",
                    "valid": VALID_ACTIONS}
        if action in ("sit", "stand", "lie"):
            self.posture = action
        self.log(ACTION_FACES.get(action, f"🐕 *{action}*"))
        time.sleep(0.2)  # a pretend robot still takes a moment
        return {"ok": True, "action": action, "steps": steps}

    def status(self):
        return {"ok": True, "mock": True, "battery_v": 7.9,
                "uptime_s": int(time.time() - self.started),
                "posture": self.posture,
                "behavior": {"state": self.behavior},
                "security": {"auth": "off", "rate_limit_per_min": "off"},
                "sensors": {"last_speak": None}}

    def sensors(self):
        return {"ok": True, "mock": True, "battery_v": 7.9,
                "distance_cm": round(random.uniform(30, 120), 1),
                "touch": False, "obstacle": False}


class Handler(BaseHTTPRequestHandler):
    dog = None  # set in main()

    def log_message(self, *args):
        pass

    def _json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            return json.loads(self.rfile.read(length)) if length else {}
        except ValueError:
            return {}

    def do_GET(self):
        path = self.path.split("?")[0]
        dog = self.dog
        if path == "/status":
            self._json(dog.status())
        elif path == "/sensors":
            self._json(dog.sensors())
        elif path == "/photo":
            dog.log("📷 *click*")
            if "format=jpeg" in (self.path.partition("?")[2]) or "format=jpg" in self.path:
                img = __import__("base64").b64decode(TINY_JPEG_B64)
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(img)))
                self.end_headers()
                self.wfile.write(img)
                return
            self._json({"ok": True, "mock": True, "photo_b64": TINY_JPEG_B64,
                        "faces": [], "face_count": 0})
        elif path == "/vision":
            self._json({"ok": True, "mock": True, "model": "imagination-1x1",
                        "description": "A cozy room. A human is watching a terminal, "
                                       "looking pleased that no hardware was required.",
                        "age_s": 1.0, "error": None})
        elif path == "/capabilities":
            self._json({"ok": True, "mock": True, "actions": VALID_ACTIONS,
                        "emotions": EMOTIONS})
        else:
            self._json({"ok": False, "error": f"mock body: no GET {path}"}, 404)

    def do_POST(self):
        path = self.path.split("?")[0]
        body = self._read()
        dog = self.dog
        if path == "/action":
            actions = body.get("actions") or ([body["action"]] if body.get("action") else [])
            if not actions:
                self._json({"ok": False, "error": "empty action",
                            "valid_actions": VALID_ACTIONS}, 400)
                return
            # Exactly the real bridge's shape — always {ok, results}, HTTP 200,
            # even for one action (clients built against the mock must not
            # break on the real dog; review finding on #50).
            results = [dog.act(a, body.get("steps", 3)) for a in actions]
            self._json({"ok": all(r["ok"] for r in results), "results": results})
        elif path == "/speak":
            text = (body.get("text") or "").strip()
            if not text:
                self._json({"error": "no text"}, 400)
                return
            dog.log(f'🗣️  "{text}"')
            self._json({"ok": True, "spoke": text, "via": "MockSpeaker",
                        "mock": True})
        elif path == "/look_at":
            target = body.get("direction") or f"yaw={body.get('angle', 0)}"
            dog.log(f"👀 *looks {target}*")
            self._json({"ok": True, "head": body})
        elif path == "/expression":
            emotion = body.get("type", "")
            if emotion not in EMOTIONS:
                self._json({"ok": False, "error": f"unknown expression: {emotion}",
                            "valid": EMOTIONS}, 400)
                return
            dog.log(f"🎭 *looks {emotion}*")
            self._json({"ok": True, "expression": emotion})
        elif path == "/rgb":
            dog.rgb = {k: body.get(k) for k in ("r", "g", "b", "mode")}
            dog.log(f"💡 LEDs → rgb({body.get('r')},{body.get('g')},{body.get('b')}) "
                    f"{body.get('mode', 'breath')}")
            self._json({"ok": True, **dog.rgb})
        elif path == "/voice/input":
            dog.log(f"👂 heard: \"{body.get('text', '')}\"")
            self._json({"ok": False, "queued": True,
                        "error": "mock body has no brain attached — this endpoint "
                                 "needs nox-brain; with MCP, Claude IS the brain"})
        elif path == "/behavior/start":
            dog.behavior = "idle"
            dog.log("🤖 autonomous mode ON (pretending to patrol)")
            self._json({"ok": True, "state": "idle"})
        elif path in ("/behavior/stop", "/emergency_stop"):
            dog.behavior = "off"
            dog.log("🛑 FREEZE" if path == "/emergency_stop" else "🤖 autonomous mode off")
            self._json({"ok": True, "state": "stopped"})
        else:
            self._json({"ok": False, "error": f"mock body: no POST {path}"}, 404)


class Server(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser(description="A robot dog made of log lines.")
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default localhost only)")
    args = ap.parse_args()
    Handler.dog = Dog()
    srv = Server((args.host, args.port), Handler)
    print(f"🐕 Mock PiDog listening on http://{args.host}:{srv.server_address[1]}")
    print("   Point a brain at it:  PIDOG_HOST=127.0.0.1")
    print("   Or Claude Code:       claude mcp add pidog --env PIDOG_HOST=127.0.0.1 "
          "-- python3 brain/nox_mcp_server.py")
    print("   Ctrl+C stops it. The dog will miss you.\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n🐕 *lies down* bye!")


if __name__ == "__main__":
    main()
