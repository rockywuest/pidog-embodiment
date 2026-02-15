#!/usr/bin/env python3
"""
nox_behavior_engine.py — Nox Behavior Engine (Tier 2)

Replaces nox_autonomous_v2.py with a formal FSM, obstacle avoidance,
patrol mode, and cherry-picked patterns from HoundMind.

States: IDLE → PATROL → INVESTIGATE → ALERT → PLAY → REST
Features:
  - Formal FSM with transition guards (min dwell + confirmation ticks)
  - Head sweep scan for obstacle mapping
  - Obstacle avoidance (3-way scan, best-direction, stuck detection)
  - Patrol mode (walk + scan + avoid)
  - Sound tracking with EMA head
  - Touch reactions
  - Mood system (6 dimensions)
  - Face detection integration
  - Emergency stop on close obstacles
  - Memory integration (optional)
  - Low battery mode

Architecture: Runs as threads within nox_brain_bridge.py process.
Communicates with nox_daemon.py via TCP socket (localhost:9999).

Sprint 3 of Nox Embodiment Upgrade.
"""

import time
import random
import json
import threading
import urllib.request
from enum import Enum
from collections import deque

# Optional memory
try:
    import pidog_memory
    _HAS_MEMORY = True
except ImportError:
    _HAS_MEMORY = False
    print("[behavior] pidog_memory not available — running without memory", flush=True)


# ─── FSM States ───────────────────────────────────────────────────────────────
class BehaviorState(str, Enum):
    IDLE = "idle"
    PATROL = "patrol"
    INVESTIGATE = "investigate"
    ALERT = "alert"
    PLAY = "play"
    REST = "rest"


# ─── Mood System (enhanced from v2) ──────────────────────────────────────────
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
        self.last_sound_heard = 0
        self.last_person_seen = 0
        self.people_nearby = False
        self.battery_level = 1.0
        self._prev_dominant = None

    def update(self, dt_seconds):
        dt = dt_seconds / 60.0
        time_since = time.time() - self.last_interaction
        if time_since > 120:
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
        return {k: round(v, 2) for k, v in {
            "energy": self.energy, "excitement": self.excitement,
            "curiosity": self.curiosity, "boredom": self.boredom,
            "alertness": self.alertness, "social": self.social,
        }.items()}

    def on_interaction(self):
        self.last_interaction = time.time()
        self.excitement = min(1.0, self.excitement + 0.3)
        self.boredom = max(0.0, self.boredom - 0.3)

    def on_sound(self, direction):
        self.last_sound_heard = time.time()
        self.alertness = min(1.0, self.alertness + 0.2)
        self.curiosity = min(1.0, self.curiosity + 0.1)

    def on_touch(self, side="unknown"):
        self.on_interaction()
        self.excitement = min(1.0, self.excitement + 0.2)
        if _HAS_MEMORY:
            pidog_memory.store_event("touch",
                f"Touch on {side}. Excitement: {self.excitement:.1f}",
                mood_state=self.as_dict())

    def on_person_detected(self, known=False, name=None):
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
        moods = {
            "bored": self.boredom, "excited": self.excitement,
            "curious": self.curiosity, "alert": self.alertness,
            "social": self.social if not self.people_nearby else 0,
            "resting": 1.0 - self.energy,
        }
        return max(moods, key=moods.get)

    def suggest_state(self):
        """Map dominant mood to suggested FSM state."""
        mood = self.dominant_mood()
        return {
            "bored": BehaviorState.PATROL,
            "curious": BehaviorState.PATROL,
            "excited": BehaviorState.PLAY,
            "alert": BehaviorState.INVESTIGATE,
            "social": BehaviorState.IDLE,
            "resting": BehaviorState.REST,
        }.get(mood, BehaviorState.IDLE)


# ─── Obstacle Tracker ─────────────────────────────────────────────────────────
class ObstacleTracker:
    """Tracks recent obstacle readings and finds best open direction.
    Cherry-picked from HoundMind ObstacleAvoidanceModule concepts."""

    EMERGENCY_CM = 15       # Emergency stop threshold
    CLOSE_CM = 30           # "Close" obstacle — slow down / avoid
    SAFE_CM = 50            # Safe distance — free to move

    def __init__(self):
        self._last_scan = {}
        self._last_scan_ts = 0
        self._stuck_count = 0
        self._last_stuck_ts = 0
        self._no_go = deque(maxlen=10)  # recent blocked directions

    def update_scan(self, scan_data):
        """Update with scan_sweep or three_way_scan result."""
        self._last_scan = scan_data
        self._last_scan_ts = time.time()

    def has_emergency(self):
        """Check if any direction shows critically close obstacle."""
        for angle, dist in self._last_scan.items():
            if isinstance(dist, (int, float)) and 0 < dist <= self.EMERGENCY_CM:
                return True
        return False

    def forward_clear(self):
        """Check if forward direction is clear."""
        fwd = self._last_scan.get("0", self._last_scan.get("forward", -1))
        if isinstance(fwd, (int, float)) and fwd > 0:
            return fwd > self.SAFE_CM
        return True  # If no reading, assume clear

    def forward_distance(self):
        """Get forward distance in cm, -1 if unknown."""
        fwd = self._last_scan.get("0", self._last_scan.get("forward", -1))
        return fwd if isinstance(fwd, (int, float)) and fwd > 0 else -1

    def best_direction(self):
        """Find the direction with most open space.
        Returns: ('left', 'right', or 'forward') + distance."""
        if not self._last_scan:
            return "forward", -1

        # For three_way_scan results
        if "forward" in self._last_scan:
            fwd = self._last_scan.get("forward", 0)
            left = self._last_scan.get("left", 0)
            right = self._last_scan.get("right", 0)
            fwd = fwd if isinstance(fwd, (int, float)) and fwd > 0 else 0
            left = left if isinstance(left, (int, float)) and left > 0 else 0
            right = right if isinstance(right, (int, float)) and right > 0 else 0

            best = max([("forward", fwd), ("left", left), ("right", right)],
                       key=lambda x: x[1])
            return best

        # For sweep scan results (angle → distance)
        best_angle = 0
        best_dist = 0
        for angle_str, dist in self._last_scan.items():
            try:
                angle = int(angle_str)
                d = float(dist)
            except (ValueError, TypeError):
                continue
            if d > best_dist:
                best_dist = d
                best_angle = angle

        if best_angle > 10:
            return "left", best_dist
        elif best_angle < -10:
            return "right", best_dist
        else:
            return "forward", best_dist

    def record_stuck(self):
        self._stuck_count += 1
        self._last_stuck_ts = time.time()

    def is_repeatedly_stuck(self):
        return self._stuck_count >= 3

    def reset_stuck(self):
        self._stuck_count = 0


# ─── Behavior Patterns ────────────────────────────────────────────────────────
IDLE_BEHAVIORS = {
    "look_around": [
        {"head_ema": {"yaw": 20}}, {"wait": 2.0},
        {"head_ema": {"yaw": -20}}, {"wait": 2.0},
        {"head_ema": {"yaw": 0}},
    ],
    "curious_sniff": [
        {"head_ema": {"yaw": 0, "pitch": -15}}, {"wait": 0.8},
        {"head_ema": {"yaw": 8, "pitch": -18}}, {"wait": 1.0},
        {"head_ema": {"yaw": -8, "pitch": -15}}, {"wait": 1.0},
        {"head_ema": {"yaw": 0, "pitch": 0}},
    ],
    "stretch": [{"action": "stretch"}, {"wait": 3.0}, {"action": "stand"}],
    "yawn_settle": [{"action": "doze_off"}, {"wait": 4.0}, {"action": "stand"}],
    "ear_twitch": [
        {"head_ema": {"yaw": 5, "roll": 10}}, {"wait": 0.3},
        {"head_ema": {"yaw": -5, "roll": -10}}, {"wait": 0.3},
        {"head_ema": {"yaw": 0, "roll": 0}},
    ],
    "tail_wag": [{"action": "wag_tail"}, {"wait": 2.0}],
    "shift_weight": [
        {"head_ema": {"pitch": 5, "roll": 8}}, {"wait": 1.0},
        {"head_ema": {"pitch": 0, "roll": -8}}, {"wait": 1.0},
        {"head_ema": {"roll": 0}},
    ],
    "pant_happy": [
        {"action": "pant"},
        {"rgb": {"r": 0, "g": 200, "b": 100, "mode": "breath", "bps": 1.2}},
        {"wait": 5.0},
    ],
}

STATE_BEHAVIORS = {
    BehaviorState.IDLE: ["look_around", "curious_sniff", "ear_twitch", "shift_weight"],
    BehaviorState.PLAY: ["tail_wag", "pant_happy", "look_around"],
    BehaviorState.REST: ["yawn_settle", "shift_weight"],
    BehaviorState.ALERT: ["look_around", "ear_twitch"],
    BehaviorState.INVESTIGATE: ["curious_sniff", "look_around"],
}

STATE_RGB = {
    BehaviorState.IDLE: {"r": 80, "g": 80, "b": 128, "mode": "breath", "bps": 0.4},
    BehaviorState.PATROL: {"r": 0, "g": 200, "b": 255, "mode": "breath", "bps": 0.8},
    BehaviorState.INVESTIGATE: {"r": 255, "g": 200, "b": 0, "mode": "breath", "bps": 1.0},
    BehaviorState.ALERT: {"r": 255, "g": 150, "b": 0, "mode": "boom", "bps": 1.0},
    BehaviorState.PLAY: {"r": 0, "g": 255, "b": 100, "mode": "breath", "bps": 1.5},
    BehaviorState.REST: {"r": 0, "g": 0, "b": 60, "mode": "breath", "bps": 0.2},
}

# Minimum time in each state before transition (prevents jitter)
STATE_MIN_DWELL_S = {
    BehaviorState.IDLE: 10,
    BehaviorState.PATROL: 20,
    BehaviorState.INVESTIGATE: 8,
    BehaviorState.ALERT: 5,
    BehaviorState.PLAY: 15,
    BehaviorState.REST: 30,
}


# ─── Main Engine ──────────────────────────────────────────────────────────────
# ─── Vision Integration (Sprint 5) ───

VISION_MAX_AGE = 30  # seconds — ignore vision older than this
VISION_RESULT_FILE = "/tmp/nox_vision_latest.json"
class BehaviorEngine:
    """Nox Behavior Engine — formal FSM with obstacle avoidance and patrol."""

    def __init__(self, daemon_send_fn, bridge_post_fn=None):
        # Wrap daemon calls to mark as internal (won't reset idle timer)
        def _internal_send(cmd):
            cmd["_internal"] = True
            return daemon_send_fn(cmd)
        self.daemon = _internal_send
        self._raw_daemon = daemon_send_fn  # Keep unwrapped for bridge use
        self.bridge = bridge_post_fn
        self.mood = MoodState()
        self.obstacles = ObstacleTracker()
        self.running = False
        self._threads = []

        # FSM
        self.state = BehaviorState.IDLE
        self._state_entered_at = time.time()
        self._candidate_state = None
        self._candidate_ticks = 0
        self._transition_confirm_ticks = 2  # need 2 consecutive ticks wanting same state

        # Patrol
        self._patrol_steps = 0
        self._patrol_max_steps = 20  # steps before re-evaluating
        self._patrol_scan_interval = 3.0  # seconds between scans during patrol
        self._last_patrol_scan = 0

        # Behavior timing
        self._last_behavior_time = 0
        self._min_behavior_interval = 20  # seconds between idle behaviors

        # Battery
        self.low_battery_mode = False

        # External control
        self._forced_state = None  # Set by API to force a state
        self._patrol_enabled = True

    # ─── State Management ─────────────────────────────────────────────────
    def _request_transition(self, desired):
        """Request a state transition with guard logic."""
        if desired == self.state:
            self._candidate_state = None
            self._candidate_ticks = 0
            return

        now = time.time()
        dwell = now - self._state_entered_at
        min_dwell = STATE_MIN_DWELL_S.get(self.state, 10)

        # Emergency states bypass guards
        if desired == BehaviorState.ALERT:
            self._do_transition(desired)
            return

        # Dwell check
        if dwell < min_dwell:
            return

        # Confirmation ticks
        if self._candidate_state != desired:
            self._candidate_state = desired
            self._candidate_ticks = 1
            return

        self._candidate_ticks += 1
        if self._candidate_ticks >= self._transition_confirm_ticks:
            self._do_transition(desired)

    def _do_transition(self, new_state):
        old = self.state
        self.state = new_state
        self._state_entered_at = time.time()
        self._candidate_state = None
        self._candidate_ticks = 0
        print(f"[behavior] State: {old.value} → {new_state.value}", flush=True)

        # Set RGB for new state
        rgb = STATE_RGB.get(new_state, STATE_RGB[BehaviorState.IDLE])
        self.daemon({"cmd": "rgb", **rgb})

        # State entry actions
        if new_state == BehaviorState.PATROL:
            self._patrol_steps = 0
            self.daemon({"cmd": "move", "action": "stand", "speed": 60})
        elif new_state == BehaviorState.REST:
            self.daemon({"cmd": "move", "action": "lie", "speed": 50})
        elif new_state == BehaviorState.PLAY:
            self.daemon({"cmd": "move", "action": "wag_tail", "steps": 3, "speed": 80})
        elif new_state == BehaviorState.ALERT:
            self.daemon({"cmd": "move", "action": "stand", "speed": 80})

        if _HAS_MEMORY:
            pidog_memory.store_event("state_change",
                f"State: {old.value} → {new_state.value}",
                mood_state=self.mood.as_dict())

    def force_state(self, state_name):
        """Force a specific state (from API)."""
        try:
            desired = BehaviorState(state_name)
            self._forced_state = desired
            self._do_transition(desired)
            return {"ok": True, "state": desired.value}
        except ValueError:
            return {"ok": False, "error": f"Unknown state: {state_name}",
                    "valid": [s.value for s in BehaviorState]}

    def get_state(self):
        """Return current engine state for API."""
        return {
            "state": self.state.value,
            "mood": self.mood.as_dict(),
            "dominant_mood": self.mood.dominant_mood(),
            "low_battery": self.low_battery_mode,
            "patrol_enabled": self._patrol_enabled,
            "obstacles": {
                "last_scan": self.obstacles._last_scan,
                "scan_age_s": round(time.time() - self.obstacles._last_scan_ts, 1)
                    if self.obstacles._last_scan_ts > 0 else None,
                "forward_clear": self.obstacles.forward_clear(),
                "stuck_count": self.obstacles._stuck_count,
            },
            "uptime_in_state_s": round(time.time() - self._state_entered_at, 1),
        }

    # ─── Thread Entry Points ──────────────────────────────────────────────
    def start(self):
        self.running = True

        t1 = threading.Thread(target=self._sensor_loop, daemon=True)
        t1.start()
        self._threads.append(t1)

        t2 = threading.Thread(target=self._main_loop, daemon=True)
        t2.start()
        self._threads.append(t2)

        t3 = threading.Thread(target=self._face_check_loop, daemon=True)
        t3.start()
        self._threads.append(t3)

        # Set initial RGB
        rgb = STATE_RGB.get(self.state, STATE_RGB[BehaviorState.IDLE])
        self.daemon({"cmd": "rgb", **rgb})

        print("[behavior] Behavior Engine started (FSM + Patrol + Obstacle Avoidance)", flush=True)

    def stop(self):
        self.running = False

    # ─── Sensor Loop (2s interval) ────────────────────────────────────────
    def _sensor_loop(self):
        while self.running:
            try:
                result = self.daemon({"cmd": "sensors"})
                if not isinstance(result, dict) or result.get("error"):
                    time.sleep(2)
                    continue

                # Touch
                touch = result.get("touch", "N")
                if isinstance(touch, str) and touch in ("L", "R", "S"):
                    self.mood.on_touch(side=touch)
                    self._handle_touch(touch)
                elif isinstance(touch, dict) and (touch.get("L") or touch.get("R")):
                    side = "L" if touch.get("L") else "R"
                    self.mood.on_touch(side=side)
                    self._handle_touch(side)

                # Sound direction
                sound = result.get("sound", {})
                if isinstance(sound, dict) and sound.get("detected"):
                    direction = sound.get("direction", 0) or 0
                    self.mood.on_sound(direction)
                    self._handle_sound(direction)

                # Battery
                batt_v = result.get("battery_v", 8.4)
                if isinstance(batt_v, (int, float)) and batt_v < 1.0:
                    batt_v = 8.4
                self.mood.battery_level = max(0, min(1.0, (batt_v - 6.0) / 2.4))

                if batt_v < 6.8 and not self.low_battery_mode:
                    self.low_battery_mode = True
                    self.daemon({"cmd": "speak", "text": "Meine Batterie wird schwach."})
                    self.daemon({"cmd": "rgb", "r": 255, "g": 0, "b": 0, "mode": "boom", "bps": 2})
                    self._request_transition(BehaviorState.REST)
                elif batt_v > 7.2:
                    self.low_battery_mode = False

                self.mood.update(2.0)
                time.sleep(2)

            except Exception as e:
                print(f"[behavior] Sensor error: {e}", flush=True)
                time.sleep(5)

    def _handle_touch(self, side):
        """React to touch based on current state."""
        print(f"[behavior] Touch: {side}", flush=True)
        if self.state == BehaviorState.PATROL:
            # Touch during patrol → stop and acknowledge
            self._request_transition(BehaviorState.IDLE)
        self.daemon({"cmd": "move", "action": "wag_tail", "steps": 3, "speed": 80})

    def _handle_sound(self, direction):
        """React to sound: turn head toward it."""
        if 0 <= direction <= 180:
            yaw = min(40, direction * 0.25)
        else:
            yaw = max(-40, -(360 - direction) * 0.25)
        self.daemon({"cmd": "head_ema", "yaw": int(yaw)})

        # If idle and sound is significant → investigate
        if self.state == BehaviorState.IDLE:
            self._request_transition(BehaviorState.INVESTIGATE)

    # ─── Face Check Loop (2 min interval, daytime only) ───────────────────
    def _face_check_loop(self):
        import datetime
        while self.running:
            try:
                now = datetime.datetime.now()
                if 7 <= now.hour < 22 and not self.low_battery_mode:
                    try:
                        req = urllib.request.Request("http://localhost:8888/look", method="GET")
                        with urllib.request.urlopen(req, timeout=30) as resp:
                            data = json.loads(resp.read().decode())
                            faces = data.get("faces", [])
                            if faces:
                                for face in faces:
                                    name = face.get("name", "unknown")
                                    known = name not in ("unknown", "", None)
                                    self.mood.on_person_detected(known=known, name=name)
                                if any(f.get("name") not in ("unknown", "", None) for f in faces):
                                    self.daemon({"cmd": "move", "action": "wag_tail", "steps": 2})
                                else:
                                    self.daemon({"cmd": "rgb", "r": 255, "g": 150, "b": 0, "mode": "boom", "bps": 1.5})
                            else:
                                if self.mood.people_nearby:
                                    self.mood.on_person_gone()
                    except Exception as e:
                        pass  # Camera/bridge not ready
                time.sleep(120)
            except Exception:
                time.sleep(120)

    # ─── Main Behavior Loop ───────────────────────────────────────────────
    def _main_loop(self):
        """Main FSM tick loop — runs every 2 seconds."""
        while self.running:
            try:
                now = time.time()

                # Honor forced state from API
                if self._forced_state and self.state != self._forced_state:
                    self._do_transition(self._forced_state)
                    self._forced_state = None

                # Low battery → REST
                if self.low_battery_mode:
                    self._request_transition(BehaviorState.REST)
                    time.sleep(10)
                    continue

                # State-specific behavior
                if self.state == BehaviorState.PATROL:
                    self._tick_patrol()
                elif self.state == BehaviorState.INVESTIGATE:
                    self._tick_investigate()
                elif self.state == BehaviorState.IDLE:
                    self._tick_idle(now)
                elif self.state == BehaviorState.PLAY:
                    self._tick_play(now)
                elif self.state == BehaviorState.REST:
                    self._tick_rest()
                elif self.state == BehaviorState.ALERT:
                    self._tick_alert()

                # Mood-based state suggestions (if not forced or patrolling)
                if not self._forced_state and self.state not in (BehaviorState.PATROL, BehaviorState.ALERT):
                    suggested = self.mood.suggest_state()
                    if suggested != self.state:
                        self._request_transition(suggested)

                time.sleep(2)

            except Exception as e:
                print(f"[behavior] Main loop error: {e}", flush=True)
                time.sleep(5)

    # ─── State Tick Functions ─────────────────────────────────────────────

    def _read_vision(self):
        """Read latest vision result. Returns dict or None if stale/missing."""
        try:
            with open(VISION_RESULT_FILE, "r") as f:
                import json as _json
                data = _json.load(f)
            age = time.time() - data.get("ts", 0)
            if age > VISION_MAX_AGE:
                return None
            if data.get("error"):
                return None
            data["age_s"] = round(age, 1)
            return data
        except Exception:
            return None


    def _tick_patrol(self):
        """Patrol: walk forward, scan for obstacles, avoid them."""
        # Vision check (Sprint 5) — react to scene description
        vision_data = self._read_vision()
        if vision_data:
            desc = (vision_data.get('description') or '').lower()
            if any(w in desc for w in ('person', 'people', 'human', 'someone')):
                print(f'[behavior] Vision: person detected → INVESTIGATE', flush=True)
                self._transition('investigate')
                return
            if any(w in desc for w in ('blocked', 'obstacle', 'wall', 'furniture', 'chair')):
                if 'clear' not in desc:
                    print(f'[behavior] Vision: obstacle detected → scan', flush=True)
                    # Don't transition, let ultrasonic handle avoidance

        now = time.time()

        # Periodic scan during patrol
        if now - self._last_patrol_scan >= self._patrol_scan_interval:
            scan_result = self.daemon({"cmd": "three_way_scan"})
            if isinstance(scan_result, dict) and scan_result.get("ok"):
                self.obstacles.update_scan(scan_result.get("scan", {}))
                self._last_patrol_scan = now

                # Emergency stop?
                if self.obstacles.has_emergency():
                    print("[behavior] EMERGENCY: Obstacle too close!", flush=True)
                    self.daemon({"cmd": "emergency_stop"})
                    self.obstacles.record_stuck()
                    time.sleep(2)
                    self._do_avoidance()
                    return

                # Forward blocked?
                if not self.obstacles.forward_clear():
                    print(f"[behavior] Obstacle ahead ({self.obstacles.forward_distance()}cm), avoiding", flush=True)
                    self._do_avoidance()
                    return

        # Forward is clear → walk
        if self._patrol_steps < self._patrol_max_steps:
            self.daemon({"cmd": "move", "action": "forward", "steps": 2, "speed": 70})
            self._patrol_steps += 2
        else:
            # Re-evaluate after max steps
            self._patrol_steps = 0
            # Do a full sweep scan
            sweep = self.daemon({"cmd": "scan_sweep"})
            if isinstance(sweep, dict) and sweep.get("ok"):
                self.obstacles.update_scan(sweep.get("scan", {}))

            # If repeatedly stuck, rest for a bit
            if self.obstacles.is_repeatedly_stuck():
                print("[behavior] Repeatedly stuck, resting", flush=True)
                self.obstacles.reset_stuck()
                self._request_transition(BehaviorState.IDLE)
                return

    def _do_avoidance(self):
        """Obstacle avoidance: find best direction and turn toward it."""
        direction, dist = self.obstacles.best_direction()
        print(f"[behavior] Avoidance: best direction = {direction} ({dist}cm)", flush=True)

        if direction == "left":
            self.daemon({"cmd": "move", "action": "turn_left", "steps": 2, "speed": 70})
        elif direction == "right":
            self.daemon({"cmd": "move", "action": "turn_right", "steps": 2, "speed": 70})
        else:
            # No good direction → backup and try again
            self.daemon({"cmd": "move", "action": "backward", "steps": 2, "speed": 60})
            time.sleep(1)
            # Random turn to get unstuck
            if random.random() > 0.5:
                self.daemon({"cmd": "move", "action": "turn_left", "steps": 3, "speed": 70})
            else:
                self.daemon({"cmd": "move", "action": "turn_right", "steps": 3, "speed": 70})
            self.obstacles.record_stuck()

        if _HAS_MEMORY:
            pidog_memory.store_event("obstacle_avoided",
                f"Avoided obstacle: turned {direction} (dist={dist}cm)",
                mood_state=self.mood.as_dict())

    def _tick_investigate(self):
        """Investigate: look around more actively, maybe move toward sound."""
        dwell = time.time() - self._state_entered_at
        if dwell > 15:
            # Investigation done, return to idle
            self._request_transition(BehaviorState.IDLE)
            return

        # Do a scan
        if dwell < 3:
            self.daemon({"cmd": "head_ema", "yaw": 25, "pitch": -10})
        elif dwell < 6:
            self.daemon({"cmd": "head_ema", "yaw": -25, "pitch": -10})
        elif dwell < 9:
            self.daemon({"cmd": "head_ema", "yaw": 0, "pitch": 0})
        # After scanning, take a photo for analysis
        elif dwell < 10:
            self.daemon({"cmd": "photo"})

    def _tick_idle(self, now):
        """Idle: occasional natural behaviors based on mood."""
        if now - self._last_behavior_time < self._min_behavior_interval:
            return

        # If bored enough and patrol enabled → start patrol
        if self.mood.boredom > 0.6 and self._patrol_enabled:
            self._request_transition(BehaviorState.PATROL)
            return

        # Pick a behavior
        mood = self.mood.dominant_mood()
        available = STATE_BEHAVIORS.get(self.state, ["look_around"])
        behavior_name = random.choice(available)
        behavior = IDLE_BEHAVIORS.get(behavior_name, [])

        if not behavior:
            return

        print(f"[behavior] IDLE ({mood}): {behavior_name}", flush=True)
        self._execute_behavior(behavior)
        self._last_behavior_time = time.time()

        # Update interval based on mood
        if mood == "excited":
            self._min_behavior_interval = 15
        elif mood == "bored":
            self._min_behavior_interval = 25
        elif mood == "alert":
            self._min_behavior_interval = 18
        else:
            self._min_behavior_interval = 20

    def _tick_play(self, now):
        """Play: active, fun behaviors."""
        if now - self._last_behavior_time < 12:
            return

        available = STATE_BEHAVIORS.get(BehaviorState.PLAY, ["tail_wag"])
        behavior_name = random.choice(available)
        behavior = IDLE_BEHAVIORS.get(behavior_name, [])
        if behavior:
            print(f"[behavior] PLAY: {behavior_name}", flush=True)
            self._execute_behavior(behavior)
            self._last_behavior_time = now

        # Play doesn't last forever
        if time.time() - self._state_entered_at > 60:
            self._request_transition(BehaviorState.IDLE)

    def _tick_rest(self):
        """Rest: minimal activity, dimmed LEDs."""
        if time.time() - self._state_entered_at > 120 and not self.low_battery_mode:
            self._request_transition(BehaviorState.IDLE)

    def _tick_alert(self):
        """Alert: heightened awareness, quick to respond."""
        dwell = time.time() - self._state_entered_at
        if dwell > 20:
            self._request_transition(BehaviorState.IDLE)
            return

        # Quick look around
        if dwell < 3:
            self.daemon({"cmd": "head_ema", "yaw": 30})
        elif dwell < 6:
            self.daemon({"cmd": "head_ema", "yaw": -30})
        elif dwell < 8:
            self.daemon({"cmd": "head_ema", "yaw": 0})

    # ─── Behavior Execution ───────────────────────────────────────────────
    def _execute_behavior(self, steps):
        """Execute a behavior sequence (list of steps)."""
        for step in steps:
            if not self.running:
                break
            if "wait" in step:
                time.sleep(step["wait"])
            elif "action" in step:
                self.daemon({"cmd": "move", "action": step["action"], "steps": 1, "speed": 80})
            elif "head_ema" in step:
                self.daemon({"cmd": "head_ema", **step["head_ema"]})
            elif "rgb" in step:
                self.daemon({"cmd": "rgb", **step["rgb"]})

        if _HAS_MEMORY:
            pidog_memory.store_observation(
                scene=f"State: {self.state.value}, mood: {self.mood.dominant_mood()}",
                action_taken=str(steps[0]) if steps else "unknown",
                sensor_data={"battery_pct": round(self.mood.battery_level * 100)})


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

    engine = BehaviorEngine(send_to_daemon)
    engine.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        engine.stop()
        if _HAS_MEMORY:
            pidog_memory.session_end()
