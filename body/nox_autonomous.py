#!/usr/bin/env python3
"""
nox_autonomous.py — Autonomous behavior loop for Nox's PiDog body.

Runs on PiDog (Pi 4). Continuous loop that:
1. Monitors sensors (touch, sound, battery, IMU)
2. Periodically captures and analyzes camera frames
3. Executes autonomous reactions
4. Reports perception state to brain via bridge

This is the "reflexes and instincts" layer — fast, local reactions
that don't need the brain (Clawdbot) for processing.

Runs as part of the nox-bridge or as separate service.
"""

import os
import sys
import json
import time
import socket
import threading
import traceback

# ─── Config ───
DAEMON_HOST = "localhost"
DAEMON_PORT = 9999
PERCEPTION_INTERVAL = 3.0   # Seconds between camera checks
SENSOR_INTERVAL = 0.5       # Seconds between sensor polls
IDLE_TIMEOUT = 30.0         # Seconds before idle behavior
FACE_TRACK_SPEED = 60       # Head tracking speed

# Battery thresholds
BATTERY_LOW = 6.8
BATTERY_CRITICAL = 6.2
BATTERY_CHARGING = 8.35

# ─── State ───
class BodyState:
    def __init__(self):
        self.lock = threading.Lock()
        self.posture = "unknown"       # sitting, standing, lying
        self.battery_v = 0.0
        self.battery_pct = 0
        self.charging = False
        self.touched = False
        self.touch_side = "none"
        self.sound_detected = False
        self.sound_direction = None
        self.face_detected = False
        self.face_count = 0
        self.face_position = (0, 0)    # Relative to center
        self.last_interaction = time.time()
        self.last_face_time = 0
        self.last_touch_time = 0
        self.last_sound_time = 0
        self.mood = "neutral"          # happy, curious, alert, sleepy, neutral
        self.is_idle = False
        self.battery_warned = False
    
    def update(self, **kwargs):
        with self.lock:
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)
    
    def snapshot(self):
        with self.lock:
            return {k: v for k, v in self.__dict__.items() if k != "lock"}

state = BodyState()


# ─── Daemon Communication ───
def send_cmd(cmd_json, timeout=10):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((DAEMON_HOST, DAEMON_PORT))
        s.sendall((json.dumps(cmd_json) + "\n").encode())
        resp = s.recv(4096).decode().strip()
        s.close()
        return json.loads(resp) if resp else {}
    except Exception as e:
        return {"error": str(e)}


def do_action(action, steps=3, speed=80):
    return send_cmd({"cmd": "move", "action": action, "steps": steps, "speed": speed})


def move_head(yaw=0, roll=0, pitch=0):
    return send_cmd({"cmd": "head", "yaw": yaw, "roll": roll, "pitch": pitch})


def set_rgb(r=128, g=0, b=255, mode="breath", bps=0.8):
    return send_cmd({"cmd": "rgb", "r": r, "g": g, "b": b, "mode": mode, "bps": bps})


def speak(text):
    threading.Thread(target=send_cmd, args=({"cmd": "speak", "text": text},), daemon=True).start()


def play_sound(name):
    threading.Thread(target=send_cmd, args=({"cmd": "sound", "name": name},), daemon=True).start()


# ─── Sensor Monitor ───
def sensor_loop():
    """Continuous sensor monitoring loop."""
    print("[auto] Sensor loop started", flush=True)
    
    while True:
        try:
            # Read all sensors
            result = send_cmd({"cmd": "sensors"})
            if "error" in result:
                time.sleep(SENSOR_INTERVAL * 2)
                continue
            
            # Battery
            batt_v = result.get("battery_v", 0)
            batt_pct = result.get("battery_pct", 0)
            charging = result.get("charging", False)
            
            state.update(
                battery_v=batt_v,
                battery_pct=batt_pct,
                charging=charging,
            )
            
            # Battery warnings
            if batt_v < BATTERY_CRITICAL and not state.battery_warned:
                speak("Achtung! Batterie kritisch! Bitte sofort laden!")
                set_rgb(255, 0, 0, "boom", 3)
                state.update(battery_warned=True)
            elif batt_v < BATTERY_LOW and not state.battery_warned:
                speak("Meine Batterie wird langsam leer.")
                set_rgb(255, 100, 0, "breath", 1)
                state.update(battery_warned=True)
            elif batt_v > BATTERY_LOW + 0.5:
                state.update(battery_warned=False)
            
            # Touch
            touch = result.get("touch", "N")
            if touch != "N":
                state.update(
                    touched=True,
                    touch_side=touch,
                    last_touch_time=time.time(),
                    last_interaction=time.time(),
                )
                
                # React to touch
                if touch in ("L", "R"):
                    # Head pat → happy
                    do_action("wag_tail", steps=3, speed=100)
                    set_rgb(0, 255, 0, "breath", 1.5)
                    state.update(mood="happy")
                elif touch in ("LS", "RS"):
                    # Slide → playful
                    play_sound("woohoo")
                    state.update(mood="happy")
            else:
                state.update(touched=False, touch_side="none")
            
            # Sound Direction
            sound = result.get("sound", {})
            if sound.get("detected"):
                direction = sound.get("direction")
                state.update(
                    sound_detected=True,
                    sound_direction=direction,
                    last_sound_time=time.time(),
                    last_interaction=time.time(),
                )
                
                # Turn head toward sound
                if direction is not None:
                    # Convert 0-360 to yaw angle
                    if direction <= 180:
                        yaw = -min(80, direction)
                    else:
                        yaw = min(80, 360 - direction)
                    move_head(yaw=yaw, roll=0, pitch=0)
                    state.update(mood="curious")
            else:
                state.update(sound_detected=False, sound_direction=None)
            
            # IMU
            imu = result.get("imu", {})
            # Could detect if robot has been picked up, flipped, etc.
            
            # Idle detection
            idle_time = time.time() - state.last_interaction
            if idle_time > IDLE_TIMEOUT and not state.is_idle:
                state.update(is_idle=True, mood="neutral")
            elif idle_time <= IDLE_TIMEOUT:
                state.update(is_idle=False)
            
        except Exception as e:
            print(f"[auto] Sensor error: {e}", flush=True)
        
        time.sleep(SENSOR_INTERVAL)


# ─── Idle Behavior ───
def idle_loop():
    """Periodic idle behaviors when nothing is happening."""
    import random
    
    print("[auto] Idle loop started", flush=True)
    
    IDLE_BEHAVIORS = [
        lambda: move_head(yaw=random.randint(-30, 30), roll=0, pitch=random.randint(-10, 10)),
        lambda: do_action("wag_tail", steps=2, speed=60),
        lambda: set_rgb(random.randint(0, 128), random.randint(0, 128), random.randint(0, 255), "breath", 0.5),
        lambda: None,  # Do nothing
        lambda: None,
    ]
    
    while True:
        if state.is_idle:
            behavior = random.choice(IDLE_BEHAVIORS)
            try:
                behavior()
            except:
                pass
            time.sleep(random.uniform(5, 15))
        else:
            time.sleep(2)


# ─── Face Tracking ───
def track_face(face_x, face_y, img_width=640, img_height=480):
    """Track detected face with head movement.
    
    face_x, face_y: center of detected face in image coordinates
    """
    # Calculate offset from center
    dx = face_x - (img_width / 2)
    dy = face_y - (img_height / 2)
    
    # Convert to head angles (proportional control)
    # Negative dx = face is to the right of image = head should turn right (negative yaw)
    yaw_adjust = -dx * 0.1  # Scale factor
    pitch_adjust = -dy * 0.08
    
    # Clamp
    yaw = max(-80, min(80, yaw_adjust))
    pitch = max(-30, min(30, pitch_adjust))
    
    move_head(yaw=yaw, roll=0, pitch=pitch)


# ─── Main ───
def start_autonomous():
    """Start all autonomous behavior threads."""
    print("[auto] Starting autonomous behavior system...", flush=True)
    
    # Start sensor monitor
    sensor_thread = threading.Thread(target=sensor_loop, daemon=True)
    sensor_thread.start()
    
    # Start idle behavior
    idle_thread = threading.Thread(target=idle_loop, daemon=True)
    idle_thread.start()
    
    print("[auto] Autonomous system running.", flush=True)
    return state


if __name__ == "__main__":
    start_autonomous()
    
    # Keep main thread alive
    try:
        while True:
            snap = state.snapshot()
            print(f"[auto] State: mood={snap['mood']} idle={snap['is_idle']} "
                  f"batt={snap['battery_v']}V touch={snap['touch_side']} "
                  f"sound={snap['sound_direction']}", flush=True)
            time.sleep(5)
    except KeyboardInterrupt:
        print("[auto] Stopped.")
