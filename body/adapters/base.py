#!/usr/bin/env python3
"""
base.py — Abstract base class for robot body adapters.

Each body type (dog, car, arm, etc.) implements this interface.
The bridge uses the adapter to control whatever hardware is connected.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


class BodyAdapter(ABC):
    """Abstract interface for a robot body."""
    
    @property
    @abstractmethod
    def body_type(self) -> str:
        """Return body type identifier (e.g., 'pidog', 'picar', 'custom')."""
        pass
    
    @property
    @abstractmethod
    def capabilities(self) -> List[str]:
        """Return list of capabilities (e.g., ['walk', 'speak', 'camera', 'touch'])."""
        pass
    
    @property
    @abstractmethod
    def available_actions(self) -> List[str]:
        """Return list of available movement actions."""
        pass
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to hardware. Returns True on success."""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Disconnect from hardware."""
        pass
    
    @abstractmethod
    def move(self, action: str, steps: int = 3, speed: int = 80) -> Dict:
        """Execute a movement action."""
        pass
    
    @abstractmethod
    def stop(self) -> Dict:
        """Emergency stop all movement."""
        pass
    
    @abstractmethod
    def get_sensors(self) -> Dict:
        """Read all sensor values."""
        pass
    
    @abstractmethod
    def get_battery(self) -> Dict:
        """Get battery status: {voltage, percentage, charging}."""
        pass
    
    # ─── Optional capabilities (override if supported) ───
    
    def head(self, yaw: float = 0, roll: float = 0, pitch: float = 0) -> Dict:
        """Move head/camera. Override if body has a head."""
        return {"error": "head not supported"}
    
    def speak(self, text: str) -> Dict:
        """Text-to-speech. Override if body has a speaker."""
        return {"error": "speech not supported"}
    
    def set_rgb(self, r: int, g: int, b: int, mode: str = "solid", bps: float = 1.0) -> Dict:
        """Set LED color. Override if body has LEDs."""
        return {"error": "RGB not supported"}
    
    def capture_photo(self) -> Optional[bytes]:
        """Capture camera image. Returns JPEG bytes or None."""
        return None
    
    def get_body_state(self) -> Dict:
        """Get body posture/state. Override if available."""
        return {"posture": "unknown"}
    
    def get_imu(self) -> Dict:
        """Get IMU data. Override if available."""
        return {"pitch": 0, "roll": 0, "yaw": 0}
    
    def get_touch(self) -> str:
        """Get touch sensor state. Override if available."""
        return "N"  # No touch
    
    def get_sound_direction(self) -> Optional[int]:
        """Get sound direction in degrees. Override if available."""
        return None
    
    def play_sound(self, name: str) -> Dict:
        """Play a sound file. Override if body has a speaker."""
        return {"error": "sound not supported"}
    
    def get_distance(self) -> Optional[float]:
        """Get ultrasonic distance in cm. Override if available."""
        return None
