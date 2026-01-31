#!/usr/bin/env python3
"""
picar.py — Robot Car adapter (template).

Implements BodyAdapter for a wheeled robot car.
Adapt this to your specific car hardware (SunFounder PiCar-X, etc.)
"""

from typing import Dict, List, Optional
from .base import BodyAdapter


class PiCarAdapter(BodyAdapter):
    """Robot car body adapter (template — customize for your hardware)."""
    
    ACTIONS = [
        "forward", "backward", "turn_left", "turn_right",
        "stop", "speed_up", "slow_down",
        "drift_left", "drift_right",
    ]
    
    def __init__(self, daemon_host="localhost", daemon_port=9998):
        self.daemon_host = daemon_host
        self.daemon_port = daemon_port
        self._connected = False
        self._speed = 50  # Default speed 0-100
    
    @property
    def body_type(self) -> str:
        return "picar"
    
    @property
    def capabilities(self) -> List[str]:
        return ["drive", "camera", "battery", "distance"]
    
    @property
    def available_actions(self) -> List[str]:
        return self.ACTIONS
    
    def connect(self) -> bool:
        # TODO: Connect to your car's control daemon
        # self._connected = True
        return False
    
    def disconnect(self):
        self.stop()
        self._connected = False
    
    def move(self, action: str, steps: int = 3, speed: int = 50) -> Dict:
        """Execute car movement.
        
        For cars, 'steps' maps to duration (steps * 0.3s).
        Speed is 0-100.
        """
        # TODO: Implement for your specific car
        # Example:
        # if action == "forward":
        #     self.car.forward(speed)
        #     time.sleep(steps * 0.3)
        #     self.car.stop()
        return {"error": "not implemented — customize picar.py for your hardware"}
    
    def stop(self) -> Dict:
        # TODO: Stop all motors
        return {"ok": True}
    
    def get_sensors(self) -> Dict:
        # TODO: Read car sensors
        return {
            "battery_v": 0,
            "battery_pct": 0,
            "speed": self._speed,
            "distance_front": None,
        }
    
    def get_battery(self) -> Dict:
        return {
            "voltage": 0,
            "percentage": 0,
            "charging": False,
        }
    
    def head(self, yaw=0, roll=0, pitch=0) -> Dict:
        """For cars, head = camera pan/tilt."""
        # TODO: Move camera servo
        return {"ok": True}
    
    def capture_photo(self) -> Optional[bytes]:
        # TODO: Capture from car camera
        return None
    
    def get_distance(self) -> Optional[float]:
        # TODO: Read ultrasonic sensor
        return None
