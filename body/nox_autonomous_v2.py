#!/usr/bin/env python3
"""
nox_autonomous_v2.py — Intelligent autonomous behavior for PiDog.

NOT just reactive (touch → wag). This is adaptive, contextual behavior:
- Idle behaviors that look natural (stretching, looking around, yawning)
- Environmental awareness (sound direction, light changes, proximity)
- Social behavior (different reactions for known vs unknown people)
- Energy management (battery-aware behavior scaling)
- Curiosity (investigate sounds, look at movement)
- Mood system (boredom, excitement, alertness decay over time)

Runs as part of the bridge process on PiDog (Pi 4).
"""

import time
import random
import json
import threading
import math

# ─── Mood System ───
class MoodState:
    """Tracks PiDog's emotional/energy state over time."""
    
    def __init__(self):
        self.energy = 0.8        # 0-1: physical energy
        self.excitement = 0.3    # 0-1: how excited
        self.curiosity = 0.5     # 0-1: how curious about environment
        self.boredom = 0.0       # 0-1: increases without stimulation
        self.alertness = 0.5     # 0-1: how alert to surroundings
        self.social = 0.5        # 0-1: desire for interaction
        self.last_interaction = time.time()
        self.last_movement = time.time()
        self.last_sound_heard = 0
        self.last_person_seen = 0
        self.people_nearby = False
        self.battery_level = 1.0  # 0-1
        
    def update(self, dt_seconds):
        """Natural mood decay/growth over time."""
        dt = dt_seconds / 60.0  # Convert to minutes
        
        # Boredom increases without stimulation
        time_since_interaction = time.time() - self.last_interaction
        if time_since_interaction > 120:  # 2 min without interaction
            self.boredom = min(1.0, self.boredom + 0.02 * dt)
        else:
            self.boredom = max(0.0, self.boredom - 0.05 * dt)
        
        # Excitement decays naturally
        self.excitement = max(0.1, self.excitement - 0.01 * dt)
        
        # Curiosity rises when bored, drops when stimulated
        if self.boredom > 0.5:
            self.curiosity = min(1.0, self.curiosity + 0.03 * dt)
        
        # Alertness follows sounds and people
        if time.time() - self.last_sound_heard < 30:
            self.alertness = min(1.0, self.alertness + 0.05 * dt)
        else:
            self.alertness = max(0.2, self.alertness - 0.02 * dt)
        
        # Social need rises over time without people
        if not self.people_nearby:
            self.social = min(1.0, self.social + 0.01 * dt)
        
        # Energy tied to battery
        self.energy = self.battery_level * 0.8 + 0.2
        
    def on_interaction(self):
        self.last_interaction = time.time()
        self.excitement = min(1.0, self.excitement + 0.3)
        self.boredom = max(0.0, self.boredom - 0.3)
        self.social = max(0.0, self.social - 0.2)
    
    def on_sound(self, direction):
        self.last_sound_heard = time.time()
        self.alertness = min(1.0, self.alertness + 0.2)
        self.curiosity = min(1.0, self.curiosity + 0.1)
    
    def on_touch(self):
        self.on_interaction()
        self.excitement = min(1.0, self.excitement + 0.2)
    
    def on_person_detected(self, known=False):
        self.last_person_seen = time.time()
        self.people_nearby = True
        self.social = max(0.0, self.social - 0.3)
        if known:
            self.excitement = min(1.0, self.excitement + 0.4)
        else:
            self.alertness = min(1.0, self.alertness + 0.3)
    
    def on_person_gone(self):
        self.people_nearby = False
    
    def dominant_mood(self):
        """Return the dominant mood for behavior selection."""
        moods = {
            "bored": self.boredom,
            "excited": self.excitement,
            "curious": self.curiosity,
            "alert": self.alertness,
            "social": self.social if not self.people_nearby else 0,
            "resting": 1.0 - self.energy,
        }
        return max(moods, key=moods.get)


# ─── Behavior Patterns ───
# Each pattern is a list of (action, duration) tuples

IDLE_BEHAVIORS = {
    "look_around": [
        {"head": {"yaw": 30, "pitch": 0}},
        {"wait": 1.5},
        {"head": {"yaw": -30, "pitch": 0}},
        {"wait": 1.5},
        {"head": {"yaw": 0, "pitch": 0}},
    ],
    "curious_sniff": [
        {"head": {"yaw": 0, "pitch": -20}},
        {"wait": 0.5},
        {"head": {"yaw": 10, "pitch": -25}},
        {"wait": 0.8},
        {"head": {"yaw": -10, "pitch": -20}},
        {"wait": 0.8},
        {"head": {"yaw": 0, "pitch": 0}},
    ],
    "stretch": [
        {"action": "stretch"},
        {"wait": 3.0},
        {"action": "stand"},
    ],
    "yawn_settle": [
        {"action": "doze_off"},
        {"wait": 4.0},
        {"action": "stand"},
    ],
    "ear_twitch": [
        {"head": {"yaw": 5, "roll": 10}},
        {"wait": 0.3},
        {"head": {"yaw": -5, "roll": -10}},
        {"wait": 0.3},
        {"head": {"yaw": 0, "roll": 0}},
    ],
    "tail_wag_gentle": [
        {"action": "wag_tail"},
        {"wait": 2.0},
    ],
    "shift_weight": [
        {"head": {"yaw": 0, "pitch": 5, "roll": 8}},
        {"wait": 1.0},
        {"head": {"yaw": 0, "pitch": 0, "roll": -8}},
        {"wait": 1.0},
        {"head": {"yaw": 0, "pitch": 0, "roll": 0}},
    ],
    "pant_happy": [
        {"action": "pant"},
        {"rgb": {"r": 0, "g": 200, "b": 100, "mode": "breath", "bps": 1.2}},
        {"wait": 5.0},
    ],
}

# Which behaviors fit which moods
MOOD_BEHAVIORS = {
    "bored": ["yawn_settle", "look_around", "shift_weight", "stretch"],
    "excited": ["tail_wag_gentle", "pant_happy", "look_around"],
    "curious": ["curious_sniff", "look_around", "ear_twitch"],
    "alert": ["look_around", "ear_twitch"],
    "social": ["tail_wag_gentle", "look_around", "pant_happy"],
    "resting": ["yawn_settle", "shift_weight"],
}

# Mood → RGB mapping
MOOD_RGB = {
    "bored": {"r": 80, "g": 80, "b": 128, "mode": "breath", "bps": 0.3},
    "excited": {"r": 0, "g": 255, "b": 100, "mode": "breath", "bps": 1.5},
    "curious": {"r": 0, "g": 200, "b": 255, "mode": "breath", "bps": 0.8},
    "alert": {"r": 255, "g": 150, "b": 0, "mode": "boom", "bps": 1.0},
    "social": {"r": 255, "g": 50, "b": 200, "mode": "breath", "bps": 1.0},
    "resting": {"r": 0, "g": 0, "b": 60, "mode": "breath", "bps": 0.2},
}


class AutonomousBehavior:
    """Main autonomous behavior controller."""
    
    def __init__(self, daemon_send_fn, bridge_post_fn=None):
        """
        Args:
            daemon_send_fn: function(cmd_dict) → sends to nox_daemon via TCP
            bridge_post_fn: function(path, data) → HTTP POST to bridge (optional, for self-use)
        """
        self.daemon = daemon_send_fn
        self.bridge = bridge_post_fn
        self.mood = MoodState()
        self.running = False
        self._threads = []
        self.last_behavior_time = 0
        self.min_behavior_interval = 15  # seconds between idle behaviors
        self.low_battery_mode = False
        
    def start(self):
        self.running = True
        
        # Sensor monitoring thread
        t1 = threading.Thread(target=self._sensor_loop, daemon=True)
        t1.start()
        self._threads.append(t1)
        
        # Behavior selection thread
        t2 = threading.Thread(target=self._behavior_loop, daemon=True)
        t2.start()
        self._threads.append(t2)
        
        print("[auto-v2] Autonomous behavior v2 started", flush=True)
    
    def stop(self):
        self.running = False
    
    def _sensor_loop(self):
        """Poll sensors and update mood."""
        last_check = 0
        
        while self.running:
            try:
                now = time.time()
                if now - last_check < 2.0:  # Check every 2s
                    time.sleep(0.5)
                    continue
                
                last_check = now
                
                # Get sensor data from daemon
                result = self.daemon({"cmd": "sensors"})
                if not isinstance(result, dict) or result.get("error"):
                    time.sleep(2)
                    continue

                if True:
                    # Touch — daemon returns "N", "L", "R", or "S" (slide)
                    touch = result.get("touch", "N")
                    if isinstance(touch, str) and touch in ("L", "R", "S"):
                        self.mood.on_touch()
                        print(f"[auto-v2] Touch detected: {touch}", flush=True)
                    elif isinstance(touch, dict) and (touch.get("L") or touch.get("R")):
                        self.mood.on_touch()
                        print(f"[auto-v2] Touch detected!", flush=True)

                    # Sound direction
                    sound = result.get("sound", result.get("sound_direction", {}))
                    if isinstance(sound, dict) and sound.get("detected"):
                        direction = sound.get("direction", sound.get("angle", 0))
                        if direction is None:
                            direction = 0
                        self.mood.on_sound(direction)
                        # Turn head toward sound
                        # Map 0-360 to yaw range (-45 to 45)
                        if 0 <= direction <= 180:
                            yaw = min(45, direction * 0.25)
                        else:
                            yaw = max(-45, -(360 - direction) * 0.25)
                        self.daemon({"cmd": "head", "yaw": int(yaw)})
                    
                    # Battery
                    batt_v = result.get("battery_v", 8.4)
                    self.mood.battery_level = max(0, min(1.0, (batt_v - 6.0) / 2.4))
                    
                    if batt_v < 6.8 and not self.low_battery_mode:
                        self.low_battery_mode = True
                        self.daemon({"cmd": "speak", "text": "Meine Batterie wird schwach. Bitte lade mich auf."})
                        self.daemon({"cmd": "rgb", "r": 255, "g": 0, "b": 0, "mode": "boom", "bps": 2})
                        print(f"[auto-v2] LOW BATTERY: {batt_v}V", flush=True)
                    elif batt_v > 7.2:
                        self.low_battery_mode = False
                
                # Update mood with elapsed time
                self.mood.update(2.0)
                
            except Exception as e:
                print(f"[auto-v2] Sensor error: {e}", flush=True)
                time.sleep(5)
    
    def _behavior_loop(self):
        """Select and execute behaviors based on mood."""
        while self.running:
            try:
                now = time.time()
                
                # Respect minimum interval
                if now - self.last_behavior_time < self.min_behavior_interval:
                    time.sleep(1)
                    continue
                
                # In low battery mode, minimal behavior
                if self.low_battery_mode:
                    time.sleep(30)
                    continue
                
                # Get dominant mood
                mood = self.mood.dominant_mood()
                
                # Select random behavior for this mood
                available = MOOD_BEHAVIORS.get(mood, ["look_around"])
                behavior_name = random.choice(available)
                behavior = IDLE_BEHAVIORS.get(behavior_name, [])
                
                if not behavior:
                    time.sleep(5)
                    continue
                
                # Set mood RGB
                rgb = MOOD_RGB.get(mood, MOOD_RGB["curious"])
                self.daemon({"cmd": "rgb", **rgb})
                
                # Execute behavior sequence
                print(f"[auto-v2] Mood: {mood} → Behavior: {behavior_name}", flush=True)
                
                for step in behavior:
                    if not self.running:
                        break
                    
                    if "wait" in step:
                        time.sleep(step["wait"])
                    elif "action" in step:
                        self.daemon({"cmd": "move", "action": step["action"], "steps": 1, "speed": 80})
                    elif "head" in step:
                        self.daemon({"cmd": "head", **step["head"]})
                    elif "rgb" in step:
                        self.daemon({"cmd": "rgb", **step["rgb"]})
                
                self.last_behavior_time = time.time()
                
                # Variable interval based on mood
                if mood == "excited":
                    self.min_behavior_interval = 8
                elif mood == "bored":
                    self.min_behavior_interval = 25
                elif mood == "alert":
                    self.min_behavior_interval = 10
                else:
                    self.min_behavior_interval = 15
                
            except Exception as e:
                print(f"[auto-v2] Behavior error: {e}", flush=True)
                time.sleep(10)


# For standalone testing
if __name__ == "__main__":
    import socket
    
    def send_to_daemon(cmd_json):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(10)
            s.connect(("localhost", 9999))
            s.sendall((json.dumps(cmd_json) + "\n").encode())
            resp = s.recv(4096).decode()
            s.close()
            return json.loads(resp) if resp else {}
        except Exception as e:
            return {"error": str(e)}
    
    auto = AutonomousBehavior(send_to_daemon)
    auto.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        auto.stop()
