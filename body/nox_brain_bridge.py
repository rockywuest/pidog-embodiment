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
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path

# ─── Configuration ───
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 8888
DAEMON_HOST = "localhost"
DAEMON_PORT = 9999
PHOTO_DIR = "/tmp"
FACE_DB_DIR = "/home/pidog/nox_face_db"
PERCEPTION_INTERVAL = 2.0  # seconds between auto-perception cycles

# Ensure directories exist
os.makedirs(FACE_DB_DIR, exist_ok=True)

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


# ─── Camera & Detection (runs on PiDog) ───
def capture_and_detect():
    """Capture a photo and run local detections.
    
    Returns dict with photo path, face detections, object detections.
    """
    result = {"ts": time.time()}
    
    # Take photo via daemon
    photo_result = send_to_daemon({"cmd": "photo"})
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
    
    # Run face detection using OpenCV (lightweight, already available)
    try:
        import cv2
        img = cv2.imread(photo_path)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Face detection
            face_cascade = cv2.CascadeClassifier(
                '/opt/vilib/haarcascade_frontalface_default.xml'
            )
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            face_list = []
            for (x, y, w, h) in faces:
                face_data = {
                    "x": int(x + w//2),
                    "y": int(y + h//2),
                    "w": int(w),
                    "h": int(h),
                    "name": "unknown",
                    "confidence": 0.0
                }
                
                # Crop face for potential identification
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
            self._send_json({
                "ok": True,
                "sensors": sensors,
                "perception": state,
                "known_faces": face_db.list_known(),
                "uptime_s": sensors.get("uptime_s", 0),
                "battery_v": sensors.get("battery_v", 0),
            })
        
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
            # Register a face: take photo, crop face, store
            name = body.get("name", "")
            if not name:
                self._send_json({"error": "name required"}, 400)
                return
            
            # Take photo
            result = capture_and_detect()
            if result.get("face_count", 0) == 0:
                self._send_json({"error": "no face detected"}, 400)
                return
            
            # Register first detected face
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
                with perception.lock:
                    perception.voice_inbox.append({
                        "text": text,
                        "ts": time.time(),
                        "source": "voice"
                    })
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
                results.append({"ok": True, "spoke": speak_text, "async": True})
            
            self._send_json({"ok": True, "results": results})
        
        else:
            self._send_json({"error": f"unknown path: {path}"}, 404)


# ─── Threaded HTTP Server ───
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in separate threads."""
    daemon_threads = True


# ─── Main ───
def main():
    print(f"[bridge] Starting Nox Brain Bridge on port {LISTEN_PORT}...", flush=True)
    
    server = ThreadedHTTPServer((LISTEN_HOST, LISTEN_PORT), BridgeHandler)
    server.timeout = 1.0
    
    print(f"[bridge] Listening on http://{LISTEN_HOST}:{LISTEN_PORT}", flush=True)
    print(f"[bridge] Known faces: {face_db.list_known()}", flush=True)
    
    # Start autonomous behavior system
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from nox_autonomous import start_autonomous
        auto_state = start_autonomous()
        print("[bridge] Autonomous behavior active", flush=True)
    except Exception as e:
        print(f"[bridge] Autonomous behavior failed to start: {e}", flush=True)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("[bridge] Stopped.", flush=True)


if __name__ == "__main__":
    main()
