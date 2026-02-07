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
- Memory integration: events → drift-memory store → co-occurrence over time

Runs as part of the bridge process on PiDog (Pi 4).
"""

import time
import random
import json
import threading
import math

# Try to import memory — graceful fallback if not available
try:
    import pidog_memory
    _HAS_MEMORY = True
except ImportError:
    _HAS_MEMORY = False
    print("[auto-v2] pidog_memory not available — running without memory", flush=True)


# ─── Mood System ───
class MoodState:
    """Tracks PiDog's emotional/energy state over time."""
    
    def __init__(self):
        self.energy = 0.8
        self.excitement = 0.3
        self.curiosity = 0.5
        self.boredom = 0.0
        self.alertness = 0.5
        self.social = 0.5
        self.last_interaction = time.time()
        self.last_movement = time.time()
        self.last_sound_heard = 0
        self.last_person_seen = 0
        self.people_nearby = False
        self.battery_level = 1.0
        self._prev_dominant = None
        
    def update(self, dt_seconds):
        """Natural mood decay/growth over time."""
        dt = dt_seconds / 60.0
        
        time_since_interaction = time.time() - self.last_interaction
        if time_since_interaction > 120:
            self.boredom = min(1.0, self.boredom + 0.02 * dt)
        else:
            self.boredom = max(0.0, self.boredom - 0.05 * dt)
        
        self.excitement = max(0.1, self.excitement - 0.01 * dt)
        
        if self.boredom > 0.5:
            self.curiosity = min(1.0, self.curiosity + 0.03 * dt)
        
        if time.time() - self.last_sound_heard < 30:
            self.alertness = min(1.0, self.alertness + 0.05 * dt)
        else:
            self.alertness = max(0.2, self.alertness - 0.02 * dt)
        
        if not self.people_nearby:
            self.social = min(1.0, self.social + 0.01 * dt)
        
        self.energy = self.battery_level * 0.8 + 0.2
        
    def as_dict(self):
        """Return mood values as dict (for memory storage)."""
        return {
            "energy": round(self.energy, 2),
            "excitement": round(self.excitement, 2),
            "curiosity": round(self.curiosity, 2),
            "boredom": round(self.boredom, 2),
            "alertness": round(self.alertness, 2),
            "social": round(self.social, 2),
        }
    
    def on_interaction(self):
        self.last_interaction = time.time()
        self.excitement = min(1.0, self.excitement + 0.3)
        self.boredom = max(0.0, self.boredom - 0.3)
        self.social = max(0.0, self.social - 0.2)
    
    def on_sound(self, direction):
        self.last_sound_heard = time.time()
        self.alertness = min(1.0, self.alertness + 0.2)
        self.curiosity = min(1.0, self.curiosity + 0.1)
    
    def on_touch(self, side="unknown"):
        self.on_interaction()
        self.excitement = min(1.0, self.excitement + 0.2)
        # Store touch event in memory
        if _HAS_MEMORY:
            pidog_memory.store_event(
                "touch",
                f"Physical interaction: touched on {side}. Excitement rose to {self.excitement:.1f}",
                mood_state=self.as_dict(),
            )
    
    def on_person_detected(self, known=False, name=None):
        self.last_person_seen = time.time()
        self.people_nearby = True
        self.social = max(0.0, self.social - 0.3)
        if known:
            self.excitement = min(1.0, self.excitement + 0.4)
        else:
            self.alertness = min(1.0, self.alertness + 0.3)
        # Store person event in memory
        if _HAS_MEMORY:
            who = name or ("known person" if known else "unknown person")
            pidog_memory.store_event(
                "person_detected",
                f"Person detected: {who}. {'Excited!' if known else 'Alert.'}",
                mood_state=self.as_dict(),
            )
    
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
        dominant = max(moods, key=moods.get)
        
        # Track mood shifts for memory
        if _HAS_MEMORY and self._prev_dominant and dominant != self._prev_dominant:
            pidog_memory.store_event(
                "mood_shift",
                f"Mood shifted: {self._prev_dominant} -> {dominant}",
                mood_state=self.as_dict(),
            )
        self._prev_dominant = dominant
        return dominant


# ─── Behavior Patterns ───
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

MOOD_BEHAVIORS = {
    "bored": ["yawn_settle", "look_around", "shift_weight", "stretch"],
    "excited": ["tail_wag_gentle", "pant_happy", "look_around"],
    "curious": ["curious_sniff", "look_around", "ear_twitch"],
    "alert": ["look_around", "ear_twitch"],
    "social": ["tail_wag_gentle", "look_around", "pant_happy"],
    "resting": ["yawn_settle", "shift_weight"],
}

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
        self.daemon = daemon_send_fn
        self.bridge = bridge_post_fn
        self.mood = MoodState()
        self.running = False
        self._threads = []
        self.last_behavior_time = 0
        self.min_behavior_interval = 15
        self.low_battery_mode = False
        
    def start(self):
        self.running = True
        
        t1 = threading.Thread(target=self._sensor_loop, daemon=True)
        t1.start()
        self._threads.append(t1)
        
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
                if now - last_check < 2.0:
                    time.sleep(0.5)
                    continue
                
                last_check = now
                
                result = self.daemon({"cmd": "sensors"})
                if not isinstance(result, dict) or result.get("error"):
                    time.sleep(2)
                    continue

                # Touch
                touch = result.get("touch", "N")
                if isinstance(touch, str) and touch in ("L", "R", "S"):
                    self.mood.on_touch(side=touch)
                    print(f"[auto-v2] Touch detected: {touch}", flush=True)
                elif isinstance(touch, dict) and (touch.get("L") or touch.get("R")):
                    side = "L" if touch.get("L") else "R"
                    self.mood.on_touch(side=side)
                    print(f"[auto-v2] Touch detected!", flush=True)

                # Sound direction
                sound = result.get("sound", result.get("sound_direction", {}))
                if isinstance(sound, dict) and sound.get("detected"):
                    direction = sound.get("direction", sound.get("angle", 0))
                    if direction is None:
                        direction = 0
                    self.mood.on_sound(direction)
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
                    if _HAS_MEMORY:
                        pidog_memory.store_event(
                            "battery_low",
                            f"Battery critically low: {batt_v}V. Entering low-power mode.",
                            mood_state=self.mood.as_dict(),
                            sensor_data={"battery_v": batt_v},
                        )
                elif batt_v > 7.2:
                    self.low_battery_mode = False
                
                self.mood.update(2.0)
                
            except Exception as e:
                print(f"[auto-v2] Sensor error: {e}", flush=True)
                time.sleep(5)
    
    def _behavior_loop(self):
        """Select and execute behaviors based on mood."""
        while self.running:
            try:
                now = time.time()
                
                if now - self.last_behavior_time < self.min_behavior_interval:
                    time.sleep(1)
                    continue
                
                if self.low_battery_mode:
                    time.sleep(30)
                    continue
                
                mood = self.mood.dominant_mood()
                available = MOOD_BEHAVIORS.get(mood, ["look_around"])
                behavior_name = random.choice(available)
                behavior = IDLE_BEHAVIORS.get(behavior_name, [])
                
                if not behavior:
                    time.sleep(5)
                    continue
                
                # Set mood RGB
                rgb = MOOD_RGB.get(mood, MOOD_RGB["curious"])
                self.daemon({"cmd": "rgb", **rgb})
                
                print(f"[auto-v2] Mood: {mood} → Behavior: {behavior_name}", flush=True)
                
                # Execute behavior
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
                
                # Store behavior execution in memory (throttled by pidog_memory)
                if _HAS_MEMORY:
                    pidog_memory.store_observation(
                        scene=f"Mood: {mood}, behavior: {behavior_name}",
                        action_taken=behavior_name,
                        sensor_data={"battery_pct": round(self.mood.battery_level * 100)}
                    )
                
                # Variable interval
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
    
    if _HAS_MEMORY:
        pidog_memory.session_start()
    
    auto = AutonomousBehavior(send_to_daemon)
    auto.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        auto.stop()
        if _HAS_MEMORY:
            pidog_memory.session_end()
