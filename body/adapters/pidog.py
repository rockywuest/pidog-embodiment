#!/usr/bin/env python3
"""
pidog.py — SunFounder PiDog adapter.

Implements BodyAdapter for the SunFounder PiDog robot kit.
Communicates with nox_daemon.py via TCP socket (port 9999).
"""

import json
import socket
from typing import Dict, List, Optional
from .base import BodyAdapter


class PiDogAdapter(BodyAdapter):
    """SunFounder PiDog body adapter."""
    
    ACTIONS = [
        "forward", "backward", "turn_left", "turn_right",
        "stand", "sit", "lie", "wag_tail", "bark", "trot",
        "doze_off", "stretch", "push_up", "howling", "pant",
        "nod_lethargy", "shake_head", "nod",
    ]
    
    def __init__(self, daemon_host="localhost", daemon_port=9999):
        self.daemon_host = daemon_host
        self.daemon_port = daemon_port
        self._connected = False
    
    @property
    def body_type(self) -> str:
        return "pidog"
    
    @property
    def capabilities(self) -> List[str]:
        return ["walk", "speak", "camera", "touch", "sound_direction", 
                "imu", "rgb", "head", "battery"]
    
    @property
    def available_actions(self) -> List[str]:
        return self.ACTIONS
    
    def _send(self, cmd: dict, timeout: int = 10) -> dict:
        """Send command to daemon via TCP."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            s.connect((self.daemon_host, self.daemon_port))
            s.sendall((json.dumps(cmd) + "\n").encode())
            resp = s.recv(65536).decode().strip()
            s.close()
            return json.loads(resp) if resp else {}
        except Exception as e:
            return {"error": str(e)}
    
    def connect(self) -> bool:
        result = self._send({"cmd": "status"})
        self._connected = "error" not in result
        return self._connected
    
    def disconnect(self):
        self._connected = False
    
    def move(self, action: str, steps: int = 3, speed: int = 80) -> Dict:
        return self._send({"cmd": "move", "action": action, "steps": steps, "speed": speed})
    
    def stop(self) -> Dict:
        return self._send({"cmd": "move", "action": "stand", "steps": 1, "speed": 100})
    
    def get_sensors(self) -> Dict:
        return self._send({"cmd": "sensors"})
    
    def get_battery(self) -> Dict:
        sensors = self.get_sensors()
        return {
            "voltage": sensors.get("battery_v", 0),
            "percentage": sensors.get("battery_pct", 0),
            "charging": sensors.get("charging", False),
        }
    
    def head(self, yaw=0, roll=0, pitch=0) -> Dict:
        return self._send({"cmd": "head", "yaw": yaw, "roll": roll, "pitch": pitch})
    
    def speak(self, text: str) -> Dict:
        return self._send({"cmd": "speak", "text": text})
    
    def set_rgb(self, r=0, g=0, b=0, mode="solid", bps=1.0) -> Dict:
        return self._send({"cmd": "rgb", "r": r, "g": g, "b": b, "mode": mode, "bps": bps})
    
    def capture_photo(self) -> Optional[bytes]:
        result = self._send({"cmd": "photo"}, timeout=15)
        if "image" in result:
            import base64
            return base64.b64decode(result["image"])
        return None
    
    def get_body_state(self) -> Dict:
        return self._send({"cmd": "body_state"})
    
    def get_imu(self) -> Dict:
        result = self._send({"cmd": "imu"})
        return {
            "pitch": result.get("pitch", 0),
            "roll": result.get("roll", 0),
            "yaw": 0,
        }
    
    def get_touch(self) -> str:
        result = self._send({"cmd": "touch"})
        return result.get("touch", "N")
    
    def get_sound_direction(self) -> Optional[int]:
        result = self._send({"cmd": "ears"})
        if result.get("detected"):
            return result.get("direction")
        return None
    
    def play_sound(self, name: str) -> Dict:
        return self._send({"cmd": "sound", "name": name})
    
    def get_distance(self) -> Optional[float]:
        result = self._send({"cmd": "distance"})
        dist = result.get("distance")
        return float(dist) if dist is not None else None
