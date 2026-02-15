#!/usr/bin/env python3
"""
nox_brain_bridge.py — HTTP bridge between Nox's Brain (Pi 5) and Body (Pi 4).

Runs on PiDog. Exposes an HTTP API for:
- Perception data (camera frames, sensor readings, face/object detection)
- Action commands (move, speak, RGB, etc.)
- Face registration and identification
- Voice relay (STT text → brain → TTS response)

Port: 8888
"""

import os
import sys
import json
import time
import base64
import socket
import threading
import traceback
import urllib.request
import signal

# Behavior engine reference (set in main())
_behavior_engine = None

# Memory integration
try:
    import pidog_memory
    _HAS_MEMORY = True
except ImportError:
    _HAS_MEMORY = False
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path

# ─── Configuration ───
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 8888
DAEMON_HOST = "localhost"
DAEMON_PORT = 9999

# Brain callback (push voice to brain instead of waiting for poll)
BRAIN_HOST = os.environ.get("BRAIN_HOST", "192.168.1.18")
BRAIN_CALLBACK_PORT = int(os.environ.get("BRAIN_CALLBACK_PORT", "8889"))
PHOTO_DIR = "/tmp"
FACE_DB_DIR = "/home/pidog/nox_face_db"
PERCEPTION_INTERVAL = 2.0  # seconds between auto-perception cycles
VISION_RESULT_FILE = "/tmp/nox_vision_latest.json"

# Ensure directories exist
os.makedirs(FACE_DB_DIR, exist_ok=True)

# ─── Sprint 4: Expression + Movement Constants ───
EXPRESSION_MAP = {
    "happy":   {"actions": ["wag_tail"], "rgb": {"r": 0, "g": 255, "b": 100, "mode": "breath", "bps": 1.5}, "head": {"yaw": 0, "roll": 0, "pitch": -5}},
    "sad":     {"actions": ["lie"], "rgb": {"r": 0, "g": 0, "b": 128, "mode": "breath", "bps": 0.3}, "head": {"yaw": 0, "roll": 0, "pitch": 15}},
    "excited": {"actions": ["wag_tail", "bark"], "rgb": {"r": 255, "g": 255, "b": 0, "mode": "boom", "bps": 2.0}, "head": {"yaw": 0, "roll": 0, "pitch": -10}, "sound": "woohoo"},
    "curious": {"actions": [], "rgb": {"r": 0, "g": 200, "b": 255, "mode": "breath", "bps": 1.0}, "head": {"yaw": 20, "roll": 5, "pitch": -10}},
    "alert":   {"actions": ["stand"], "rgb": {"r": 255, "g": 150, "b": 0, "mode": "boom", "bps": 1.5}, "head": {"yaw": 0, "roll": 0, "pitch": -5}, "sound": "growl_1"},
    "sleepy":  {"actions": ["doze_off"], "rgb": {"r": 0, "g": 0, "b": 60, "mode": "breath", "bps": 0.2}, "head": {"yaw": 0, "roll": 5, "pitch": 10}},
    "scared":  {"actions": ["lie"], "rgb": {"r": 200, "g": 0, "b": 200, "mode": "boom", "bps": 2.5}, "head": {"yaw": 0, "roll": 0, "pitch": 10}, "sound": "confused_1"},
    "angry":   {"actions": ["bark"], "rgb": {"r": 255, "g": 0, "b": 0, "mode": "boom", "bps": 2.0}, "head": {"yaw": 0, "roll": 0, "pitch": -5}, "sound": "growl_2"},
    "love":    {"actions": ["wag_tail"], "rgb": {"r": 255, "g": 50, "b": 150, "mode": "breath", "bps": 1.0}, "head": {"yaw": 0, "roll": 0, "pitch": -5}},
    "think":   {"actions": [], "rgb": {"r": 128, "g": 0, "b": 255, "mode": "breath", "bps": 0.8}, "head": {"yaw": 15, "roll": -10, "pitch": 10}},
}

LOOK_DIRECTIONS = {
    "left":    {"yaw": 35, "roll": 0, "pitch": 0},
    "right":   {"yaw": -35, "roll": 0, "pitch": 0},
    "up":      {"yaw": 0, "roll": 0, "pitch": -20},
    "down":    {"yaw": 0, "roll": 0, "pitch": 15},
    "center":  {"yaw": 0, "roll": 0, "pitch": 0},
    "forward": {"yaw": 0, "roll": 0, "pitch": 0},
}

STEP_DISTANCE_CM = 5.0   # approximate cm per forward/backward step
STEP_ANGLE_DEG = 15.0    # approximate degrees per turn step

VALID_ACTIONS = [
    "stand", "sit", "lie", "forward", "backward", "turn_left", "turn_right",
    "wag_tail", "bark", "trot", "stretch", "push_up", "howling", "pant", "doze_off",
]

AVAILABLE_SOUNDS = [
    "angry", "confused_1", "confused_2", "confused_3", "growl_1", "growl_2",
    "howling", "pant", "single_bark_1", "single_bark_2", "snoring", "woohoo",
]


# ─── Brain Push (zero-latency voice delivery) ───
def push_to_brain(data, timeout=5):
    """Push voice input directly to brain (no polling delay)."""
    try:
        url = f"http://{BRAIN_HOST}:{BRAIN_CALLBACK_PORT}/voice/push"
        body = json.dumps(data).encode()
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"[bridge] Brain push failed: {e}", flush=True)
        return None


# ─── Daemon Communication ───
def send_to_daemon(cmd_json, timeout=30):
    """Send command to nox_daemon.py via TCP."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((DAEMON_HOST, DAEMON_PORT))
        s.sendall((json.dumps(cmd_json) + "\n").encode())
        resp = s.recv(8192).decode().strip()
        s.close()
        return json.loads(resp) if resp else {"error": "empty response"}
    except Exception as e:
        return {"error": str(e)}


# ─── Perception State ───
class PerceptionState:
    """Shared perception state, updated by background thread."""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.last_update = 0
        self.faces = []          # Detected faces [{x, y, w, h, name, confidence}]
        self.objects = []        # Detected objects [{class, x, y, w, h, score}]
        self.scene_text = ""     # Last scene description from brain
        self.sensors = {}        # Latest sensor readings
        self.last_photo_path = None
        self.last_photo_b64 = None
        self.voice_inbox = []    # Pending voice messages
        self.voice_outbox = []   # Pending voice responses
        self.tts_echo_until = 0  # Timestamp when TTS will finish (for echo suppression)
    
    def update(self, **kwargs):
        with self.lock:
            for k, v in kwargs.items():
                setattr(self, k, v)
            self.last_update = time.time()
    
    def snapshot(self):
        with self.lock:
            return {
                "ts": self.last_update,
                "age_s": round(time.time() - self.last_update, 1) if self.last_update else None,
                "faces": list(self.faces),
                "objects": list(self.objects),
                "scene": self.scene_text,
                "sensors": dict(self.sensors),
                "has_photo": self.last_photo_b64 is not None,
            }


perception = PerceptionState()


# ─── Face Database ───
class FaceDB:
    """Simple face database using stored reference images.
    
    For now: stores reference images per person.
    Later: can store face embeddings when InsightFace is available.
    """
    
    def __init__(self, db_dir):
        self.db_dir = db_dir
        self.faces_file = os.path.join(db_dir, "faces.json")
        self.faces = self._load()
    
    def _load(self):
        if os.path.exists(self.faces_file):
            with open(self.faces_file) as f:
                return json.load(f)
        return {}
    
    def _save(self):
        with open(self.faces_file, "w") as f:
            json.dump(self.faces, f, indent=2)
    
    def register(self, name, image_path):
        """Register a face image for a person."""
        if name not in self.faces:
            self.faces[name] = {"images": [], "registered": time.time()}
        
        # Copy image to face DB directory
        import shutil
        dest = os.path.join(self.db_dir, f"{name}_{len(self.faces[name]['images'])}.jpg")
        shutil.copy2(image_path, dest)
        self.faces[name]["images"].append(dest)
        self._save()
        return {"ok": True, "name": name, "images": len(self.faces[name]["images"])}
    
    def list_known(self):
        return {name: len(data["images"]) for name, data in self.faces.items()}
    
    def get_reference_images(self, name):
        if name in self.faces:
            return self.faces[name]["images"]
        return []


face_db = FaceDB(FACE_DB_DIR)


# ─── Face Recognition Engine (SCRFD + ArcFace) ───
_face_engine = None

def get_face_engine():
    """Lazy-load the SCRFD+ArcFace face engine to save memory until first use."""
    global _face_engine
    if _face_engine is None:
        try:
            from nox_face_recognition import FaceEngine
            _face_engine = FaceEngine(
                model_dir="/home/pidog/models",
                db_dir="/home/pidog/face_db"
            )
            print("[bridge] SCRFD+ArcFace face engine loaded", flush=True)
        except Exception as e:
            print(f"[bridge] Face engine failed, falling back to Haar: {e}", flush=True)
            _face_engine = "fallback"
    return _face_engine


# ─── Camera & Detection (runs on PiDog) ───
def capture_and_detect():
    """Capture a photo and run local detections.

    Uses SCRFD+ArcFace (nox_face_recognition.py) for detection and identification.
    Falls back to Haar Cascade if ONNX models fail to load.

    Returns dict with photo path, face detections, object detections.
    """
    result = {"ts": time.time()}

    # Take photo via daemon (camera startup needs extra time)
    photo_result = send_to_daemon({"cmd": "photo"}, timeout=60)
    if not photo_result.get("ok"):
        result["error"] = f"photo failed: {photo_result.get('error', 'unknown')}"
        return result

    photo_path = photo_result.get("photo", "/tmp/nox_snap.jpg")
    result["photo_path"] = photo_path

    # Read photo as base64
    try:
        with open(photo_path, "rb") as f:
            result["photo_b64"] = base64.b64encode(f.read()).decode()
    except:
        result["photo_b64"] = None

    # Face detection + identification
    try:
        import cv2
        img = cv2.imread(photo_path)
        if img is None:
            result["faces"] = []
            result["face_count"] = 0
            return result

        engine = get_face_engine()
        face_list = []

        if engine and engine != "fallback":
            # Use SCRFD+ArcFace engine (detect + identify)
            faces = engine.identify(img)
            for f in faces:
                bbox = f.get("bbox", [0, 0, 0, 0])
                x1, y1, x2, y2 = [int(v) for v in bbox]
                w, h = x2 - x1, y2 - y1
                face_data = {
                    "x": int(x1 + w // 2),
                    "y": int(y1 + h // 2),
                    "w": w,
                    "h": h,
                    "name": f.get("name", "unknown"),
                    "confidence": round(f.get("confidence", 0.0), 3),
                    "det_score": round(f.get("score", 0.0), 3),
                }
                # Crop face
                face_crop = img[max(0, y1):y2, max(0, x1):x2]
                if face_crop.size > 0:
                    crop_path = f"/tmp/face_crop_{len(face_list)}.jpg"
                    cv2.imwrite(crop_path, face_crop)
                    face_data["crop_path"] = crop_path
                    face_data["crop_b64"] = base64.b64encode(
                        cv2.imencode('.jpg', face_crop)[1]
                    ).decode()
                face_list.append(face_data)
        else:
            # Fallback: Haar Cascade (no identification, only detection)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                '/opt/vilib/haarcascade_frontalface_default.xml'
            )
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face_data = {
                    "x": int(x + w // 2),
                    "y": int(y + h // 2),
                    "w": int(w),
                    "h": int(h),
                    "name": "unknown",
                    "confidence": 0.0,
                }
                face_crop = img[y:y+h, x:x+w]
                crop_path = f"/tmp/face_crop_{len(face_list)}.jpg"
                cv2.imwrite(crop_path, face_crop)
                face_data["crop_path"] = crop_path
                face_data["crop_b64"] = base64.b64encode(
                    cv2.imencode('.jpg', face_crop)[1]
                ).decode()
                face_list.append(face_data)

        result["faces"] = face_list
        result["face_count"] = len(face_list)
    except Exception as e:
        result["face_error"] = str(e)
        result["faces"] = []

    return result


def get_sensor_data():
    """Get current sensor readings from daemon."""
    status = send_to_daemon({"cmd": "status"})
    return status


# ─── HTTP Request Handler ───
class BridgeHandler(BaseHTTPRequestHandler):
    
    def log_message(self, format, *args):
        """Quiet logging."""
        pass
    
    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        if length > 0:
            body = self.rfile.read(length)
            return json.loads(body)
        return {}
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    def do_GET(self):
        path = self.path.split("?")[0]
        
        if path == "/status":
            # System status
            sensors = get_sensor_data()
            state = perception.snapshot()
            resp = {
                "ok": True,
                "sensors": sensors,
                "perception": state,
                "known_faces": face_db.list_known(),
                "uptime_s": sensors.get("uptime_s", 0),
                "battery_v": sensors.get("battery_v", 0),
            }
            if _behavior_engine:
                resp["behavior"] = _behavior_engine.get_state()
            self._send_json(resp)
        
        elif path == "/perception":
            # Current perception state
            self._send_json(perception.snapshot())
        
        elif path == "/photo":
            # Take a photo and return it
            result = capture_and_detect()
            # Update perception state
            perception.update(
                faces=result.get("faces", []),
                last_photo_path=result.get("photo_path"),
                last_photo_b64=result.get("photo_b64"),
            )
            self._send_json(result)
        
        elif path == "/look":
            # Take photo with full analysis — returns everything
            # Capture
            result = capture_and_detect()
            # Sensors
            sensors = get_sensor_data()
            result["sensors"] = sensors
            # Update state
            perception.update(
                faces=result.get("faces", []),
                sensors=sensors,
                last_photo_path=result.get("photo_path"),
                last_photo_b64=result.get("photo_b64"),
            )
            self._send_json(result)
        
        elif path == "/faces":
            # List known faces
            self._send_json({"known_faces": face_db.list_known()})
        
        elif path == "/voice/inbox":
            # Get pending voice messages
            with perception.lock:
                msgs = list(perception.voice_inbox)
                perception.voice_inbox.clear()
            self._send_json({"messages": msgs})

        
        elif path == "/voice/echo_until":
            # Return when TTS echo suppression should end
            self._send_json({"echo_until": perception.tts_echo_until})

        elif path == "/memory/recent":
            # Recent memories for brain sync
            if _HAS_MEMORY:
                limit = 10
                try:
                    qs = self.path.split("?", 1)
                    if len(qs) > 1:
                        for p in qs[1].split("&"):
                            if p.startswith("limit="):
                                limit = int(p.split("=")[1])
                except Exception:
                    pass
                self._send_json({"memories": pidog_memory.get_recent(limit)})
            else:
                self._send_json({"error": "memory not available"}, 503)

        elif path == "/memory/stats":
            if _HAS_MEMORY:
                self._send_json(pidog_memory.stats())
            else:
                self._send_json({"error": "memory not available"}, 503)


        elif path == "/state":
            if _behavior_engine:
                self._send_json(_behavior_engine.get_state())
            else:
                self._send_json({"error": "behavior engine not running"}, 503)

        elif path == "/scan":
            r = send_to_daemon({"cmd": "three_way_scan"})
            self._send_json(r)

        elif path == "/scan/sweep":
            r = send_to_daemon({"cmd": "scan_sweep"})
            self._send_json(r)

        elif path == "/sensors":
            # Aggregated sensor data (Sprint 4)
            raw = send_to_daemon({"cmd": "sensors"})
            if raw.get("error"):
                self._send_json({"ok": False, "error": raw["error"]}, 503)
            else:
                batt_v = raw.get("battery_v", 0)
                batt_pct = max(0, min(100, int((batt_v - 6.0) / (8.4 - 6.0) * 100))) if batt_v > 0 else 0
                result = {
                    "ok": True,
                    "ts": raw.get("ts", time.time()),
                    "battery": {
                        "voltage": batt_v,
                        "percent": batt_pct,
                        "charging": batt_v > 8.35,
                    },
                    "distance": {
                        "forward_cm": raw.get("distance_cm"),
                    },
                    "touch": {
                        "active": raw.get("touch", "N") != "N",
                        "side": {"N": "none", "L": "left", "R": "right", "LS": "slide-left", "RS": "slide-right"}.get(raw.get("touch", "N"), str(raw.get("touch", "N"))),
                    },
                    "sound": raw.get("sound", {}),
                    "imu": raw.get("imu", {}),
                    "system": {
                        "hostname": raw.get("hostname", "pidog"),
                        "uptime_s": raw.get("uptime_s", 0),
                        "disk_free_gb": raw.get("disk_free_gb", 0),
                    },
                }
                if _behavior_engine:
                    be = _behavior_engine.get_state()
                    result["behavior"] = {
                        "state": be.get("state", "unknown"),
                        "mood": be.get("dominant_mood", "unknown"),
                        "patrol_enabled": be.get("patrol_enabled", False),
                        "low_battery": be.get("low_battery", False),
                    }
                    obs = be.get("obstacles", {})
                    scan = obs.get("last_scan", {})
                    if scan.get("forward"):
                        result["distance"]["forward_cm"] = scan["forward"]
                        result["distance"]["scan_age_s"] = obs.get("scan_age_s")
                # Add vision age if available (Sprint 5)
                try:
                    with open(VISION_RESULT_FILE, "r") as vf:
                        vd = json.load(vf)
                    result["vision"] = {
                        "age_s": round(time.time() - vd.get("ts", 0), 1),
                        "description": (vd.get("description") or "")[:100],
                        "error": vd.get("error"),
                    }
                except Exception:
                    result["vision"] = {"age_s": None, "error": "not running"}
                self._send_json(result)

        elif path == "/vision":
            # Vision engine results (Sprint 5)
            try:
                with open(VISION_RESULT_FILE, "r") as vf:
                    vision_data = json.load(vf)
                vision_data["age_s"] = round(time.time() - vision_data.get("ts", 0), 1)
                vision_data["ok"] = True
                self._send_json(vision_data)
            except FileNotFoundError:
                self._send_json({"ok": False, "error": "vision not running (no result file)"}, 503)
            except Exception as e:
                self._send_json({"ok": False, "error": f"vision read error: {e}"}, 500)

        elif path == "/capabilities":
            # Endpoint discovery (Sprint 4)
            self._send_json({
                "ok": True,
                "version": "sprint5",
                "endpoints": {
                    "GET": ["/status", "/perception", "/photo", "/look", "/faces",
                            "/voice/inbox", "/voice/echo_until", "/memory/recent",
                            "/memory/stats", "/state", "/scan", "/scan/sweep",
                            "/sensors", "/capabilities", "/vision"],
                    "POST": ["/action", "/speak", "/command", "/rgb", "/head",
                             "/face/register", "/voice/respond", "/voice/input",
                             "/combo", "/behavior/start", "/behavior/stop",
                             "/emergency_stop", "/move", "/expression", "/look_at"],
                },
                "actions": VALID_ACTIONS,
                "expressions": list(EXPRESSION_MAP.keys()),
                "sounds": AVAILABLE_SOUNDS,
                "rgb_modes": ["monochromatic", "breath", "boom", "bark", "speak", "listen"],
                "look_directions": list(LOOK_DIRECTIONS.keys()),
                "vision": {"model": "SmolVLM-256M-Q8", "prompts": ["patrol", "describe", "obstacles", "people"]},
            })
        else:
            self._send_json({"error": f"unknown path: {path}"}, 404)
    
    def do_POST(self):
        path = self.path.split("?")[0]
        body = self._read_json()
        
        if path == "/action":
            # Execute body action(s)
            actions = body.get("actions", [])
            results = []
            for action in actions:
                if isinstance(action, str):
                    r = send_to_daemon({"cmd": "move", "action": action})
                elif isinstance(action, dict):
                    r = send_to_daemon(action)
                else:
                    r = {"error": f"invalid action: {action}"}
                results.append(r)
            self._send_json({"ok": True, "results": results})
        
        elif path == "/speak":
            # Speak text (non-blocking: respond immediately, speak in background)
            text = body.get("text", "")
            blocking = body.get("blocking", False)
            if text:
                if blocking:
                    r = send_to_daemon({"cmd": "speak", "text": text})
                    self._send_json(r)
                else:
                    # Fire-and-forget in thread
                    t = threading.Thread(
                        target=send_to_daemon,
                        args=({"cmd": "speak", "text": text},),
                        daemon=True
                    )
                    t.start()
                    self._send_json({"ok": True, "spoke": text, "async": True})
            else:
                self._send_json({"error": "no text"}, 400)
        
        elif path == "/command":
            # Raw daemon command
            r = send_to_daemon(body)
            self._send_json(r)
        
        elif path == "/rgb":
            # Set RGB LEDs
            r = send_to_daemon({
                "cmd": "rgb",
                "r": body.get("r", 128),
                "g": body.get("g", 0),
                "b": body.get("b", 255),
                "mode": body.get("mode", "breath"),
                "bps": body.get("bps", 0.8),
            })
            self._send_json(r)
        
        elif path == "/head":
            # Move head
            r = send_to_daemon({
                "cmd": "head",
                "yaw": body.get("yaw", 0),
                "roll": body.get("roll", 0),
                "pitch": body.get("pitch", 0),
            })
            self._send_json(r)
        
        elif path == "/face/register":
            # Register a face: take photo, detect, store embedding
            name = body.get("name", "")
            if not name:
                self._send_json({"error": "name required"}, 400)
                return

            # Take photo
            result = capture_and_detect()
            if result.get("face_count", 0) == 0:
                self._send_json({"error": "no face detected"}, 400)
                return

            # Try SCRFD+ArcFace engine for embedding-based registration
            engine = get_face_engine()
            if engine and engine != "fallback" and engine.recognizer:
                try:
                    photo_path = result.get("photo_path", "/tmp/nox_snap.jpg")
                    reg_result = engine.register(name, photo_path)
                    if reg_result.get("ok"):
                        reg_result["method"] = "embedding"
                        reg_result["known_faces"] = engine.list_known()
                        self._send_json(reg_result)
                        return
                except Exception as e:
                    print(f"[bridge] Embedding registration failed: {e}", flush=True)

            # Fallback: image-based registration via FaceDB
            face = result["faces"][0]
            crop_path = face.get("crop_path")
            if crop_path and os.path.exists(crop_path):
                reg_result = face_db.register(name, crop_path)
                self._send_json(reg_result)
            else:
                self._send_json({"error": "face crop failed"}, 500)
        
        elif path == "/voice/respond":
            # Brain sends response to voice query
            text = body.get("text", "")
            if text:
                with perception.lock:
                    perception.voice_outbox.append({
                        "text": text,
                        "ts": time.time()
                    })
                # Speak it
                send_to_daemon({"cmd": "speak", "text": text})
                self._send_json({"ok": True})
            else:
                self._send_json({"error": "no text"}, 400)
        
        elif path == "/voice/input":
            # Voice loop reports new speech input
            text = body.get("text", "")
            if text:
                msg = {
                    "text": text,
                    "ts": time.time(),
                    "source": "voice"
                }
                # Push to brain immediately (non-blocking)
                threading.Thread(
                    target=push_to_brain,
                    args=(msg,),
                    daemon=True
                ).start()
                # Also store in inbox as fallback
                with perception.lock:
                    perception.voice_inbox.append(msg)
                self._send_json({"ok": True})
            else:
                self._send_json({"error": "no text"}, 400)
        
        elif path == "/combo":
            # Execute a combo of actions with speak
            actions = body.get("actions", [])
            speak_text = body.get("speak", "")
            rgb = body.get("rgb", None)
            head = body.get("head", None)
            
            results = []
            
            # Set RGB if specified
            if rgb:
                r = send_to_daemon({"cmd": "rgb", **rgb})
                results.append(r)
            
            # Move head if specified
            if head:
                r = send_to_daemon({"cmd": "head", **head})
                results.append(r)
            
            # Execute actions
            for action in actions:
                r = send_to_daemon({"cmd": "move", "action": action})
                results.append(r)
            
            # Speak if specified (async to avoid blocking)
            if speak_text:
                t = threading.Thread(
                    target=send_to_daemon,
                    args=({"cmd": "speak", "text": speak_text},),
                    daemon=True
                )
                t.start()
                # Smart echo suppression: estimate when TTS will finish
                # Piper TTS: ~80ms/char for German + 1s buffer
                est_tts_end = time.time() + len(speak_text) * 0.08 + 1.5
                perception.tts_echo_until = est_tts_end
                results.append({"ok": True, "spoke": speak_text, "async": True, "echo_until": est_tts_end})
            
            self._send_json({"ok": True, "results": results})
        
        elif path == "/voice/echo_until":
            # Return when TTS echo suppression should end
            self._send_json({"echo_until": perception.tts_echo_until})

        elif path == "/memory/recent":
            # Recent memories for brain sync
            if _HAS_MEMORY:
                limit = 10
                try:
                    qs = self.path.split("?", 1)
                    if len(qs) > 1:
                        for p in qs[1].split("&"):
                            if p.startswith("limit="):
                                limit = int(p.split("=")[1])
                except Exception:
                    pass
                self._send_json({"memories": pidog_memory.get_recent(limit)})
            else:
                self._send_json({"error": "memory not available"}, 503)

        elif path == "/memory/stats":
            if _HAS_MEMORY:
                self._send_json(pidog_memory.stats())
            else:
                self._send_json({"error": "memory not available"}, 503)


        elif path == "/behavior/start":
            global _behavior_engine
            state_name = body.get("behavior", body.get("state", "patrol"))
            if _behavior_engine and _behavior_engine.running:
                result = _behavior_engine.force_state(state_name)
                self._send_json(result)
            else:
                try:
                    from nox_behavior_engine import BehaviorEngine
                    _behavior_engine = BehaviorEngine(daemon_send_fn=send_to_daemon)
                    _behavior_engine.start()
                    if state_name != "idle":
                        _behavior_engine.force_state(state_name)
                    self._send_json({"ok": True, "state": state_name, "restarted": True})
                except Exception as e:
                    self._send_json({"error": f"engine restart failed: {e}"}, 500)

        elif path == "/behavior/stop":
            if _behavior_engine:
                _behavior_engine.stop()
                send_to_daemon({"cmd": "sleep"})
                result = {"ok": True, "stopped": True}
                self._send_json(result)
            else:
                self._send_json({"error": "behavior engine not running"}, 503)

        elif path == "/emergency_stop":
            r = send_to_daemon({"cmd": "emergency_stop"})
            if _behavior_engine:
                _behavior_engine.stop()
            self._send_json(r)

        elif path == "/move":
            # Higher-level movement: distance_cm or angle_deg (Sprint 4)
            direction = body.get("direction", "forward")
            speed = body.get("speed", 70)
            if direction in ("forward", "backward"):
                dist = body.get("distance_cm", 15)
                steps = max(1, min(30, round(dist / STEP_DISTANCE_CM)))
            elif direction in ("turn_left", "turn_right"):
                angle = body.get("angle_deg", 45)
                steps = max(1, min(20, round(angle / STEP_ANGLE_DEG)))
            else:
                steps = body.get("steps", 3)
            r = send_to_daemon({"cmd": "move", "action": direction, "steps": steps, "speed": speed})
            r["steps_executed"] = steps
            r["direction"] = direction
            self._send_json(r)

        elif path == "/expression":
            # Predefined emotion expressions (Sprint 4)
            expr_type = body.get("type", "")
            if expr_type not in EXPRESSION_MAP:
                self._send_json({"error": f"unknown expression: {expr_type}", "valid": list(EXPRESSION_MAP.keys())}, 400)
            else:
                expr = EXPRESSION_MAP[expr_type]
                results = []
                # Head position
                if "head" in expr:
                    r = send_to_daemon({"cmd": "head", **expr["head"]})
                    results.append(r)
                # RGB LEDs
                if "rgb" in expr:
                    r = send_to_daemon({"cmd": "rgb", **expr["rgb"]})
                    results.append(r)
                # Actions
                for action in expr.get("actions", []):
                    r = send_to_daemon({"cmd": "move", "action": action, "steps": 3, "speed": 80})
                    results.append(r)
                # Sound (fire-and-forget in thread)
                if expr.get("sound"):
                    threading.Thread(
                        target=send_to_daemon,
                        args=({"cmd": "sound", "name": expr["sound"]},),
                        daemon=True
                    ).start()
                # Speak (optional, from request body)
                speak_text = body.get("speak", "")
                if speak_text:
                    threading.Thread(
                        target=send_to_daemon,
                        args=({"cmd": "speak", "text": speak_text},),
                        daemon=True
                    ).start()
                self._send_json({
                    "ok": True,
                    "expression": expr_type,
                    "actions": expr.get("actions", []),
                    "rgb": expr.get("rgb"),
                    "head": expr.get("head"),
                    "sound": expr.get("sound"),
                    "spoke": speak_text or None,
                })

        elif path == "/look_at":
            # Semantic head control (Sprint 4)
            direction = body.get("direction")
            angle = body.get("angle")
            tilt = body.get("tilt")
            if direction and direction in LOOK_DIRECTIONS:
                target = dict(LOOK_DIRECTIONS[direction])
            elif angle is not None:
                target = {"yaw": float(angle), "roll": 0, "pitch": 0}
            else:
                target = {"yaw": 0, "roll": 0, "pitch": 0}
            if tilt is not None:
                target["pitch"] = float(tilt)
            if angle is not None and direction:
                target["yaw"] = float(angle)
            r = send_to_daemon({"cmd": "head", **target})
            r["head"] = target
            self._send_json(r)
        else:
            self._send_json({"error": f"unknown path: {path}"}, 404)


# ─── Threaded HTTP Server ───
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in separate threads."""
    daemon_threads = True


# ─── Main ───
def main():
    print(f"[bridge] Starting Nox Brain Bridge on port {LISTEN_PORT}...", flush=True)

    # Pre-import heavy modules to avoid cold-start timeout on first photo
    try:
        import cv2
        print(f"[bridge] OpenCV {cv2.__version__} pre-loaded", flush=True)
    except ImportError:
        print("[bridge] WARNING: OpenCV not available — face detection disabled", flush=True)
    
    server = ThreadedHTTPServer((LISTEN_HOST, LISTEN_PORT), BridgeHandler)
    server.timeout = 1.0
    
    print(f"[bridge] Listening on http://{LISTEN_HOST}:{LISTEN_PORT}", flush=True)
    print(f"[bridge] Known faces: {face_db.list_known()}", flush=True)
    
    # Start behavior engine (optional — disable with NOX_NO_AUTO=1)
    global _behavior_engine
    if not os.environ.get("NOX_NO_AUTO"):
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from nox_behavior_engine import BehaviorEngine
            _behavior_engine = BehaviorEngine(daemon_send_fn=send_to_daemon)
            _behavior_engine.start()
            print("[bridge] Behavior Engine started (FSM + Patrol + Obstacle Avoidance)", flush=True)
        except Exception as e:
            print(f"[bridge] BehaviorEngine failed, trying v2 fallback: {e}", flush=True)
            try:
                from nox_autonomous_v2 import AutonomousBehavior
                auto = AutonomousBehavior(daemon_send_fn=send_to_daemon)
                auto.start()
                print("[bridge] Autonomous behavior v2 active (fallback)", flush=True)
            except Exception as e2:
                print(f"[bridge] Autonomous behavior skipped: {e2}", flush=True)
    else:
        print("[bridge] Behavior engine disabled (NOX_NO_AUTO=1)", flush=True)
    
    # Memory session start
    if _HAS_MEMORY:
        info = pidog_memory.session_start()
        print(f"[bridge] Memory: {info['total_memories']} memories loaded ({info['core']} core)", flush=True)

    # Graceful shutdown handler
    def _shutdown(signum, frame):
        print(f"[bridge] Signal {signum} received, shutting down...", flush=True)
        if _HAS_MEMORY:
            pidog_memory.session_end()
        server.server_close()
        sys.exit(0)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if _HAS_MEMORY:
            pidog_memory.session_end()
        server.server_close()
        print("[bridge] Stopped.", flush=True)


if __name__ == "__main__":
    main()
