#!/usr/bin/env python3
"""
nox_patrol.py — Autonomous patrol behavior for Nox/PiDog.

Runs as a controller that connects to the daemon via socket.
Implements: obstacle avoidance, room exploration, reactive behaviors.

State machine:
  IDLE → EXPLORING → AVOIDING → SCANNING → EXPLORING
  Any state → REACT (on touch/sound) → previous state
"""

import socket
import json
import time
import random
import threading
import sys

DAEMON_SOCK = "/tmp/nox.sock"
MIN_DISTANCE_CM = 30       # Stop if closer than this
CAUTION_DISTANCE_CM = 60   # Slow down if closer than this
SCAN_INTERVAL = 10         # Seconds between environment scans
EXPLORE_STEPS = 3          # Steps per movement command
SPEAK_COOLDOWN = 30        # Min seconds between voice comments

class NoxPatrol:
    def __init__(self):
        self.state = "IDLE"
        self.last_speak = 0
        self.last_scan = 0
        self.distance = -1
        self.touch = "N"
        self.sound_detected = False
        self.sound_direction = None
        self.battery_pct = 100
        self.running = True
        self.explore_count = 0
        self.stuck_count = 0
        self.turn_preference = random.choice(["left", "right"])
        
        # Room memory (simple)
        self.obstacles_hit = 0
        self.turns_made = 0
        self.meters_walked = 0
        
    def daemon_cmd(self, cmd, timeout=10):
        """Send command to daemon via unix socket."""
        try:
            s = socket.socket(socket.AF_UNIX)
            s.settimeout(timeout)
            s.connect(DAEMON_SOCK)
            s.sendall(json.dumps(cmd).encode() + b"\n")
            data = s.recv(8192)
            s.close()
            result = json.loads(data)
            if "error" in result:
                print(f"[patrol] CMD {cmd.get('cmd','?')} error: {result['error']}", flush=True)
            return result
        except socket.timeout:
            print(f"[patrol] CMD {cmd.get('cmd','?')} TIMEOUT after {timeout}s", flush=True)
            return {"error": "timeout"}
        except Exception as e:
            print(f"[patrol] CMD {cmd.get('cmd','?')} exception: {e}", flush=True)
            return {"error": str(e)}
    
    def read_sensors(self):
        """Update sensor readings."""
        data = self.daemon_cmd({"cmd": "sensors"})
        if "error" not in data:
            self.distance = data.get("distance_cm", -1) or -1
            self.touch = data.get("touch", "N")
            self.sound_detected = data.get("sound", {}).get("detected", False)
            self.sound_direction = data.get("sound", {}).get("direction", None)
            self.battery_pct = data.get("battery_pct", 100)
        return data
    
    def move(self, action, steps=2, speed=70):
        """Execute a movement."""
        return self.daemon_cmd({"cmd": "move", "action": action, "steps": steps, "speed": speed})
    
    def head(self, yaw=0, roll=0, pitch=0):
        """Move head."""
        return self.daemon_cmd({"cmd": "head", "yaw": yaw, "roll": roll, "pitch": pitch})
    
    def speak(self, text):
        """Speak if cooldown allows."""
        now = time.time()
        if now - self.last_speak > SPEAK_COOLDOWN:
            self.daemon_cmd({"cmd": "speak", "text": text}, timeout=20)
            self.last_speak = now
    
    def rgb(self, r, g, b, mode="breath", bps=1):
        """Set LEDs."""
        return self.daemon_cmd({"cmd": "rgb", "r": r, "g": g, "b": b, "mode": mode, "bps": bps})
    
    def scan_environment(self):
        """Do a 3-point scan: left, center, right. Return distances."""
        distances = {}
        
        # Look left
        self.head(yaw=-40, pitch=0)
        time.sleep(0.8)
        self.read_sensors()
        distances["left"] = self.distance
        
        # Look center
        self.head(yaw=0, pitch=0)
        time.sleep(0.8)
        self.read_sensors()
        distances["center"] = self.distance
        
        # Look right
        self.head(yaw=40, pitch=0)
        time.sleep(0.8)
        self.read_sensors()
        distances["right"] = self.distance
        
        # Back to center
        self.head(yaw=0, pitch=0)
        
        self.last_scan = time.time()
        return distances
    
    def choose_direction(self, distances):
        """Choose best direction based on scan results."""
        # Filter valid readings
        valid = {k: v for k, v in distances.items() if v > 0}
        if not valid:
            return "backward"  # Can't see anything, back up
        
        # Find direction with most space
        best = max(valid, key=valid.get)
        best_dist = valid[best]
        
        if best_dist < MIN_DISTANCE_CM:
            return "backward"
        elif best == "left":
            return "turn_left"
        elif best == "right":
            return "turn_right"
        else:
            return "forward"
    
    # ─── State handlers ───
    
    def state_idle(self):
        """Starting state — stand up and begin."""
        print("[patrol] Starting up...", flush=True)
        self.move("stand", steps=1, speed=60)
        time.sleep(1)
        self.rgb(0, 255, 128, "breath", 1)
        self.speak("Ich gehe jetzt auf Erkundungstour!")
        time.sleep(2)
        self.state = "SCANNING"
    
    def state_exploring(self):
        """Walk forward, checking distance."""
        self.read_sensors()
        
        # Check battery
        if self.battery_pct < 15:
            self.speak("Meine Batterie ist fast leer. Ich lege mich hin.")
            self.move("lie", steps=1)
            self.rgb(255, 0, 0, "breath", 0.5)
            self.state = "LOW_BATTERY"
            return
        
        # Check for obstacles
        if 0 < self.distance < MIN_DISTANCE_CM:
            print(f"[patrol] Obstacle at {self.distance}cm! Avoiding.", flush=True)
            self.obstacles_hit += 1
            self.state = "AVOIDING"
            return
        
        if 0 < self.distance < CAUTION_DISTANCE_CM:
            # Getting close — slow down and scan
            print(f"[patrol] Caution: {self.distance}cm ahead", flush=True)
            self.rgb(255, 200, 0, "breath", 1.5)
            self.state = "SCANNING"
            return
        
        # Check for sound
        if self.sound_detected and self.sound_direction is not None:
            self.state = "REACT_SOUND"
            return
        
        # Check for touch
        if self.touch != "N":
            self.state = "REACT_TOUCH"
            return
        
        # All clear — walk forward!
        self.rgb(0, 255, 0, "breath", 0.8)
        print(f"[patrol] EXPLORE: dist={self.distance}cm → forward {EXPLORE_STEPS} steps (total: {self.explore_count})", flush=True)
        self.move("forward", steps=EXPLORE_STEPS, speed=70)
        self.explore_count += 1
        self.meters_walked += 0.05 * EXPLORE_STEPS  # Rough estimate
        
        # Periodic scan
        if time.time() - self.last_scan > SCAN_INTERVAL:
            self.state = "SCANNING"
            return
        
        # Occasional head movements (curiosity)
        if random.random() < 0.15:
            yaw = random.randint(-30, 30)
            self.head(yaw=yaw, pitch=random.randint(-10, 10))
            time.sleep(0.5)
            self.head(yaw=0, pitch=0)
        
        # Occasional comments
        if random.random() < 0.05:
            comments = [
                f"Schon {self.explore_count} Schritte gemacht. Die Welt ist groß!",
                f"Alles frei vor mir. {round(self.distance)}cm bis zum nächsten Hindernis.",
                "Das ist spannend. Ich lerne meinen Raum kennen.",
                f"Batterie bei {self.battery_pct} Prozent. Läuft gut.",
                "Ich frage mich was hinter der nächsten Ecke ist.",
            ]
            self.speak(random.choice(comments))
        
        time.sleep(0.5)
    
    def state_scanning(self):
        """Scan environment and decide direction."""
        print("[patrol] Scanning environment...", flush=True)
        self.rgb(0, 100, 255, "boom", 2)
        
        distances = self.scan_environment()
        print(f"[patrol] Scan: L={distances.get('left',-1)}cm C={distances.get('center',-1)}cm R={distances.get('right',-1)}cm", flush=True)
        
        direction = self.choose_direction(distances)
        print(f"[patrol] Best direction: {direction}", flush=True)
        
        if direction == "backward":
            self.speak("Hmm, hier ist es eng. Ich gehe zurück.")
            self.move("backward", steps=3, speed=60)
            time.sleep(1)
            # Random turn
            turn = random.choice(["turn_left", "turn_right"])
            self.move(turn, steps=4, speed=60)
            self.turns_made += 1
        elif direction in ("turn_left", "turn_right"):
            self.move(direction, steps=3, speed=60)
            self.turns_made += 1
        
        time.sleep(0.5)
        self.state = "EXPLORING"
    
    def state_avoiding(self):
        """Obstacle avoidance — back up and find new path."""
        print(f"[patrol] AVOIDING! Distance: {self.distance}cm", flush=True)
        self.rgb(255, 0, 0, "boom", 3)
        
        # Back up
        self.move("backward", steps=3, speed=60)
        time.sleep(1)
        
        # Scan and find new direction
        self.state = "SCANNING"
    
    def state_react_touch(self):
        """React to being touched."""
        print(f"[patrol] Touch detected: {self.touch}", flush=True)
        self.rgb(255, 100, 255, "boom", 2)
        
        if self.touch == "L":
            self.speak("Hey! Jemand streichelt mich links!")
        elif self.touch == "R":
            self.speak("Oh! Rechts gekrault, das ist nett!")
        else:
            self.speak("Jemand hat mich berührt!")
        
        # Happy response
        self.move("wag_tail", steps=5, speed=80)
        time.sleep(2)
        
        self.state = "EXPLORING"
    
    def state_react_sound(self):
        """React to detected sound — turn toward it."""
        raw_dir = self.sound_direction
        print(f"[patrol] Sound detected! Raw direction: {raw_dir}°", flush=True)
        self.rgb(255, 255, 0, "boom", 2)
        
        if raw_dir is not None:
            # Convert 0-360° to -180 to +180° (0=front, negative=left, positive=right)
            direction = raw_dir
            if direction > 180:
                direction = direction - 360  # e.g., 348 → -12, 270 → -90
            
            # Clamp to safe head range (-45 to +45)
            head_yaw = max(-45, min(45, int(direction)))
            print(f"[patrol] Sound: raw={raw_dir}° → normalized={direction}° → head_yaw={head_yaw}°", flush=True)
            
            # Turn head toward sound
            self.head(yaw=head_yaw, pitch=-5)
            time.sleep(1)
            
            self.speak("Ich habe ein Geräusch gehört!")
            
            # Turn body toward sound
            if direction < -20:
                self.move("turn_left", steps=2)
            elif direction > 20:
                self.move("turn_right", steps=2)
        
        time.sleep(1)
        self.head(yaw=0, pitch=0)
        self.state = "EXPLORING"
    
    def state_low_battery(self):
        """Low battery — stay still."""
        time.sleep(10)
    
    # ─── Main loop ───
    
    def run(self):
        """Main patrol loop."""
        print("[patrol] === NOX PATROL STARTED ===", flush=True)
        print(f"[patrol] Min distance: {MIN_DISTANCE_CM}cm, Caution: {CAUTION_DISTANCE_CM}cm", flush=True)
        
        states = {
            "IDLE": self.state_idle,
            "EXPLORING": self.state_exploring,
            "SCANNING": self.state_scanning,
            "AVOIDING": self.state_avoiding,
            "REACT_TOUCH": self.state_react_touch,
            "REACT_SOUND": self.state_react_sound,
            "LOW_BATTERY": self.state_low_battery,
        }
        
        try:
            while self.running:
                handler = states.get(self.state)
                if handler:
                    handler()
                else:
                    print(f"[patrol] Unknown state: {self.state}", flush=True)
                    self.state = "IDLE"
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[patrol] Interrupted, sitting down...", flush=True)
        finally:
            self.speak("Erkundung beendet. Ich lege mich hin.")
            self.move("sit", steps=1, speed=60)
            self.rgb(128, 0, 255, "breath", 0.5)
            stats = {
                "steps": self.explore_count,
                "obstacles": self.obstacles_hit,
                "turns": self.turns_made,
                "est_meters": round(self.meters_walked, 1),
            }
            print(f"[patrol] Stats: {json.dumps(stats)}", flush=True)
            print("[patrol] === NOX PATROL ENDED ===", flush=True)


if __name__ == "__main__":
    patrol = NoxPatrol()
    patrol.run()
