#!/usr/bin/env python3
"""
nox_mcp_server.py — the robot as a set of MCP tools.

Model Context Protocol server (stdio transport) in front of the body bridge:
any MCP client — Claude Code, Claude Desktop, OpenClaw, an agent SDK — gets
the dog as tools: take a photo and look at it, speak, move, emote. The LLM on
the client side *is* the brain; nox_voice_brain.py is not needed for this.

Zero dependencies, like the rest of brain/: MCP's stdio transport is
newline-delimited JSON-RPC 2.0, which the standard library covers.

Usage (Claude Code):
    claude mcp add pidog --env PIDOG_HOST=<robot> -- python3 /path/to/brain/nox_mcp_server.py

Config: PIDOG_HOST (default pidog.local), PIDOG_BRIDGE_PORT (8888),
NOX_API_TOKEN (when the bridge has auth on).
"""
import json
import os
import sys
import urllib.error
import urllib.request

PIDOG_HOST = os.environ.get("PIDOG_HOST", "pidog.local")
BRIDGE_PORT = int(os.environ.get("PIDOG_BRIDGE_PORT", "8888"))
BASE_URL = f"http://{PIDOG_HOST}:{BRIDGE_PORT}"
API_TOKEN = os.environ.get("NOX_API_TOKEN", "").strip()

SERVER_INFO = {"name": "pidog-embodiment", "version": "1.0.0"}
PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

# Mirrors the bridge's VALID_ACTIONS / EXPRESSION_MAP / LOOK_DIRECTIONS; the
# bridge rejects anything else loudly, so drift fails visibly, not silently.
ACTIONS = ["stand", "sit", "lie", "forward", "backward", "turn_left", "turn_right",
           "wag_tail", "bark", "trot", "stretch", "push_up", "howling", "pant", "doze_off"]
EMOTIONS = ["happy", "sad", "excited", "curious", "alert", "sleepy", "scared",
            "angry", "love", "think"]
DIRECTIONS = ["left", "right", "up", "down", "center", "forward"]


def bridge(path, payload=None, timeout=30):
    """One HTTP call to the body. Errors come back as data, never raise."""
    req = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        method="POST" if payload is not None else "GET")
    if payload is not None:
        req.add_header("Content-Type", "application/json")
    if API_TOKEN:
        req.add_header("Authorization", f"Bearer {API_TOKEN}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read().decode())
        except ValueError:
            return {"ok": False, "error": f"HTTP {e.code} from the bridge"}
    except Exception as e:
        reason = getattr(e, "reason", e)
        return {"ok": False, "error": (
            f"robot not reachable at {BASE_URL} ({reason}) — is the robot on, "
            "is PIDOG_HOST right, are you on its network/VPN?")}


# ─── Tools ───────────────────────────────────────────────────────────────────

def text(data):
    """A compact text content block."""
    if not isinstance(data, str):
        data = json.dumps(data, ensure_ascii=False)
    return {"type": "text", "text": data}


def result_of(reply, trim=()):
    """Standard tool result: the bridge's answer, isError when it failed."""
    failed = not isinstance(reply, dict) or reply.get("ok") is False or (
        "error" in reply and reply.get("error"))
    if isinstance(reply, dict):
        reply = {k: v for k, v in reply.items() if k not in trim}
    return {"content": [text(reply)], "isError": bool(failed)}


def tool_status(args):
    reply = bridge("/status")
    if isinstance(reply, dict):
        # The full answer nests the whole perception state — keep the essentials.
        slim = {k: reply[k] for k in
                ("ok", "battery_v", "uptime_s", "behavior", "security", "warning", "hint", "error")
                if k in reply}
        sensors = reply.get("sensors") or {}
        slim["last_speak"] = sensors.get("last_speak")
        return result_of(slim or reply)
    return result_of(reply)


def tool_sensors(args):
    return result_of(bridge("/sensors"))


def tool_action(args):
    payload = {"action": args["action"]}
    if args.get("steps"):
        payload["steps"] = int(args["steps"])
    return result_of(bridge("/action", payload, timeout=60))


def tool_speak(args):
    return result_of(bridge("/speak", {"text": args["text"], "blocking": True}, timeout=90))


def tool_photo(args):
    reply = bridge("/photo", timeout=45)
    b64 = isinstance(reply, dict) and reply.get("photo_b64")
    if not b64:
        return result_of(reply)
    faces = reply.get("faces") or []
    note = f"{len(faces)} face(s) detected" if faces else "no faces detected"
    return {"content": [
        {"type": "image", "data": b64, "mimeType": "image/jpeg"},
        text(f"Photo taken ({note}). You are seeing through the robot's camera."),
    ], "isError": False}


def tool_vision(args):
    reply = bridge("/vision")
    if isinstance(reply, dict) and reply.get("ok") and reply.get("description"):
        return {"content": [text(
            f"Local vision model ({reply.get('model', '?')}), "
            f"{reply.get('age_s', '?')}s ago: {reply['description']}")], "isError": False}
    return result_of(reply)


def tool_look_at(args):
    payload = {}
    if args.get("direction"):
        payload["direction"] = args["direction"]
    if args.get("angle") is not None:
        payload["angle"] = args["angle"]
    if args.get("tilt") is not None:
        payload["tilt"] = args["tilt"]
    return result_of(bridge("/look_at", payload or {"direction": "center"}))


def tool_expression(args):
    return result_of(bridge("/expression", {"type": args["emotion"]}, timeout=60))


def tool_rgb(args):
    return result_of(bridge("/rgb", {k: args[k] for k in ("r", "g", "b", "mode", "bps")
                                     if k in args}))


def tool_behavior(args):
    mode = args["mode"]
    path = {"start": "/behavior/start", "stop": "/behavior/stop",
            "emergency_stop": "/emergency_stop"}[mode]
    return result_of(bridge(path, {}))


TOOLS = [
    ("dog_status", "Robot status: battery, uptime, autonomous-behavior state, last speech result.",
     {"type": "object", "properties": {}}, tool_status),
    ("dog_sensors", "Live sensors: ultrasonic distance (cm), touch, battery voltage, obstacle flags.",
     {"type": "object", "properties": {}}, tool_sensors),
    ("dog_action", "Perform a physical action. forward/backward move ~5 cm per step, turns ~15° per step.",
     {"type": "object",
      "properties": {"action": {"type": "string", "enum": ACTIONS},
                     "steps": {"type": "integer", "minimum": 1, "maximum": 10,
                               "description": "Repetitions (default 3)"}},
      "required": ["action"]}, tool_action),
    ("dog_speak", "Speak text aloud through the robot's speaker (local TTS). Waits and reports the real outcome.",
     {"type": "object", "properties": {"text": {"type": "string", "maxLength": 1000}},
      "required": ["text"]}, tool_speak),
    ("dog_photo", "Take a photo with the robot's camera and return the image (plus face count).",
     {"type": "object", "properties": {}}, tool_photo),
    ("dog_vision", "Latest scene description from the robot's own on-device vision model (if installed).",
     {"type": "object", "properties": {}}, tool_vision),
    ("dog_look_at", "Point the robot's head: a direction, or an exact angle (yaw °, +left) and tilt (pitch °, +down).",
     {"type": "object",
      "properties": {"direction": {"type": "string", "enum": DIRECTIONS},
                     "angle": {"type": "number", "minimum": -80, "maximum": 80},
                     "tilt": {"type": "number", "minimum": -30, "maximum": 30}}},
     tool_look_at),
    ("dog_expression", "Show an emotion: coordinated action + LED color + head pose (+ sound).",
     {"type": "object", "properties": {"emotion": {"type": "string", "enum": EMOTIONS}},
      "required": ["emotion"]}, tool_expression),
    ("dog_rgb", "Set the LED strip color and animation.",
     {"type": "object",
      "properties": {"r": {"type": "integer", "minimum": 0, "maximum": 255},
                     "g": {"type": "integer", "minimum": 0, "maximum": 255},
                     "b": {"type": "integer", "minimum": 0, "maximum": 255},
                     "mode": {"type": "string",
                              "enum": ["monochromatic", "breath", "boom", "bark", "speak", "listen", "off"]},
                     "bps": {"type": "number", "minimum": 0, "maximum": 10}},
      "required": ["r", "g", "b"]}, tool_rgb),
    ("dog_behavior", "Autonomous mode: start (idle/patrol/play on its own), stop, or emergency_stop (freeze NOW).",
     {"type": "object", "properties": {"mode": {"type": "string",
                                                "enum": ["start", "stop", "emergency_stop"]}},
      "required": ["mode"]}, tool_behavior),
]

INSTRUCTIONS = (
    "These tools drive a real robot dog (SunFounder PiDog). dog_photo + dog_vision are its "
    "eyes, dog_speak its voice, dog_action its legs. Check dog_sensors before walking "
    "forward; use dog_behavior emergency_stop if anything looks unsafe. Actions move a "
    "physical machine in someone's home — when in doubt, ask the user first.")


# ─── JSON-RPC over stdio ─────────────────────────────────────────────────────

def _error(msg_id, code, message):
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _result(msg_id, result):
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def handle(message):
    """One JSON-RPC message in, one response dict out (None for notifications)."""
    method = message.get("method", "")
    msg_id = message.get("id")
    params = message.get("params") or {}

    if method == "initialize":
        asked = params.get("protocolVersion", "")
        version = asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        return _result(msg_id, {
            "protocolVersion": version,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
            "instructions": INSTRUCTIONS,
        })
    if method == "ping":
        return _result(msg_id, {})
    if method == "tools/list":
        return _result(msg_id, {"tools": [
            {"name": name, "description": desc, "inputSchema": schema}
            for name, desc, schema, _fn in TOOLS]})
    if method == "tools/call":
        name = params.get("name")
        for tool_name, _desc, _schema, fn in TOOLS:
            if tool_name == name:
                try:
                    return _result(msg_id, fn(params.get("arguments") or {}))
                except (KeyError, TypeError, ValueError) as e:
                    return _result(msg_id, {
                        "content": [text(f"bad arguments for {name}: {e}")],
                        "isError": True})
        return _error(msg_id, -32602, f"unknown tool: {name}")
    if method.startswith("notifications/") or msg_id is None:
        return None  # notifications need no answer
    return _error(msg_id, -32601, f"method not found: {method}")


def main():
    print(f"[mcp] pidog MCP server — robot at {BASE_URL}"
          f" (auth {'on' if API_TOKEN else 'off'})", file=sys.stderr, flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            print(json.dumps(_error(None, -32700, "parse error")), flush=True)
            continue
        response = handle(message)
        if response is not None:
            print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
