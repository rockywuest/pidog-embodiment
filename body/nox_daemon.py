#!/usr/bin/env python3
"""
nox_daemon.py — Nox's persistent body controller.
Runs as a systemd service. Holds GPIO, exposes a Unix socket API.
Skips broken sensors (ultrasonic, IMU).
"""

import os
import sys
import json
import time
import signal
import socket
import threading
import traceback
from pathlib import Path

# Audio config for HifiBerry DAC
os.environ["SDL_AUDIODRIVER"] = "alsa"
os.environ["AUDIODEV"] = "plughw:4,0"

SOCKET_PATH = "/tmp/nox.sock"
PHOTO_DIR = "/tmp"
PIPER_BIN = "/home/pidog/.local/bin/piper"
PIPER_MODEL = "/home/pidog/.local/share/piper-voices/de_DE-thorsten-high.onnx"
SOUNDS_DIR = "/home/pidog/pidog/sounds"

# ─── Patch: Skip ultrasonic to prevent init hang ───
import pidog.pidog as _pidog_mod
_orig_init = _pidog_mod.Pidog.__init__

def _patched_init(self, *args, **kwargs):
    """Patched init that skips sensory_process_start (ultrasonic hangs)."""
    # Temporarily replace sensory_process_start with a no-op
    _orig_sensory = self.__class__.sensory_process_start
    self.__class__.sensory_process_start = lambda self_: None
    try:
        _orig_init(self, *args, **kwargs)
    finally:
        self.__class__.sensory_process_start = _orig_sensory

_pidog_mod.Pidog.__init__ = _patched_init

from pidog import Pidog

# ─── Global state ───
dog = None
camera_lock = threading.Lock()
dog_lock = threading.Lock()
running = True


def init_dog():
    """Initialize PiDog with broken sensors skipped."""
    global dog
    print("[nox] Initializing PiDog...", flush=True)
    dog = Pidog()
    time.sleep(0.5)
    # Wake up
    dog.do_action('stand', speed=60)
    time.sleep(1)
    dog.rgb_strip.set_mode('breath', [128, 0, 255], bps=0.8)
    print("[nox] PiDog ready. Nox lebt! ⚡", flush=True)


def cmd_status():
    """System status."""
    import shutil
    info = {"hostname": os.uname().nodename, "uptime_s": int(float(open("/proc/uptime").read().split()[0]))}
    total, used, free = shutil.disk_usage("/")
    info["disk_free_gb"] = round(free / (1024**3), 1)
    try:
        with dog_lock:
            info["battery_v"] = round(dog.get_battery_voltage(), 2)
    except:
        info["battery_v"] = "error"
    return info


def cmd_move(action, steps=3, speed=80):
    """Execute a movement action."""
    with dog_lock:
        dog.do_action(action, step_count=int(steps), speed=int(speed))
        time.sleep(1.5)
    return {"ok": True, "action": action}


def cmd_head(yaw=0, roll=0, pitch=0):
    """Move head."""
    with dog_lock:
        dog.head_move([[float(yaw), float(roll), float(pitch)]], immediately=True, speed=80)
        time.sleep(0.5)
    return {"ok": True, "head": [yaw, roll, pitch]}


def cmd_rgb(r=128, g=0, b=255, mode="breath", bps=0.8):
    """Set RGB LEDs."""
    with dog_lock:
        color = [int(r), int(g), int(b)]
        if mode == "off":
            dog.rgb_strip.set_mode('monochromatic', [0, 0, 0])
        else:
            dog.rgb_strip.set_mode(mode, color, bps=float(bps))
    return {"ok": True, "rgb": [r, g, b], "mode": mode}


def cmd_photo(path=None):
    """Take a photo."""
    if path is None:
        path = os.path.join(PHOTO_DIR, "nox_snap.jpg")
    basename = os.path.splitext(os.path.basename(path))[0]
    dirname = os.path.dirname(path) or PHOTO_DIR

    with camera_lock:
        from vilib import Vilib
        Vilib.camera_start(vflip=False, hflip=False)
        time.sleep(1.5)
        Vilib.take_photo(basename, dirname)
        Vilib.camera_close()

    actual = os.path.join(dirname, basename + ".jpg")
    return {"ok": True, "photo": actual}


def cmd_speak(text):
    """TTS via Piper."""
    import subprocess
    wav = "/tmp/nox_speak.wav"
    proc = subprocess.run(
        f'echo "{text}" | {PIPER_BIN} --model {PIPER_MODEL} --output_file {wav}',
        shell=True, capture_output=True, text=True
    )
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr}
    subprocess.run(["aplay", wav], capture_output=True)
    return {"ok": True, "spoke": text}


def cmd_sound(name):
    """Play built-in sound."""
    import subprocess
    for ext in ['', '.wav', '.mp3']:
        path = os.path.join(SOUNDS_DIR, name + ext)
        if os.path.exists(path):
            if path.endswith('.mp3'):
                subprocess.run(["mpg123", "-q", path], capture_output=True)
            else:
                subprocess.run(["aplay", path], capture_output=True)
            return {"ok": True, "sound": name}
    available = os.listdir(SOUNDS_DIR)
    return {"ok": False, "error": f"not found", "available": available}


def cmd_combo(sequence):
    """Run action sequence. Format: action1:steps:speed,action2:steps:speed"""
    results = []
    with dog_lock:
        for part in sequence.split(","):
            parts = part.strip().split(":")
            action = parts[0]
            steps = int(parts[1]) if len(parts) > 1 else 3
            speed = int(parts[2]) if len(parts) > 2 else 80
            dog.do_action(action, step_count=steps, speed=speed)
            time.sleep(1.2)
            results.append(action)
    return {"ok": True, "combo": results}


def cmd_wake():
    """Wake up sequence."""
    with dog_lock:
        dog.do_action('stand', speed=60)
        time.sleep(1)
        dog.do_action('wag_tail', step_count=5, speed=80)
        time.sleep(1)
        dog.rgb_strip.set_mode('breath', [128, 0, 255], bps=0.8)
    return {"ok": True}


def cmd_sleep():
    """Sleep sequence."""
    with dog_lock:
        dog.do_action('lie', speed=50)
        time.sleep(1)
        dog.rgb_strip.set_mode('breath', [0, 0, 80], bps=0.3)
    return {"ok": True}


def cmd_reset():
    """Reset to neutral standing."""
    with dog_lock:
        dog.do_action('stand', speed=60)
        time.sleep(1)
        dog.head_move([[0, 0, 0]], immediately=True, speed=60)
        dog.rgb_strip.set_mode('monochromatic', [0, 0, 0])
    return {"ok": True}


# ─── Sensor Commands ───
def cmd_sensors():
    """Complete sensor readout."""
    result = {"ts": time.time()}
    
    # Battery
    try:
        with dog_lock:
            v = dog.get_battery_voltage()
            result["battery_v"] = round(v, 2)
            result["battery_pct"] = max(0, min(100, round((v - 6.0) / (8.4 - 6.0) * 100)))
            result["charging"] = v > 8.35
    except Exception as e:
        result["battery_error"] = str(e)
    
    # IMU
    try:
        with dog_lock:
            result["imu"] = {
                "pitch": round(dog.pitch, 1) if hasattr(dog, 'pitch') else None,
                "roll": round(dog.roll, 1) if hasattr(dog, 'roll') else None,
            }
    except Exception as e:
        result["imu_error"] = str(e)
    
    # Touch
    try:
        with dog_lock:
            touch = dog.dual_touch.read()
            result["touch"] = touch
    except Exception as e:
        result["touch_error"] = str(e)
    
    # Sound Direction
    try:
        with dog_lock:
            detected = dog.ears.isdetected()
            result["sound"] = {
                "detected": detected,
                "direction": dog.ears.read() if detected else None,
            }
    except Exception as e:
        result["sound_error"] = str(e)
    
    # Ultrasonic
    try:
        dist = dog.read_distance()
        result["distance_cm"] = dist if dist > 0 else None
    except:
        result["distance_cm"] = None
    
    # System
    import shutil
    total, used, free = shutil.disk_usage("/")
    try:
        mem_lines = open("/proc/meminfo").readlines()
        mem_avail = int(mem_lines[2].split()[1]) // 1024
    except:
        mem_avail = 0
    result["system"] = {
        "hostname": os.uname().nodename,
        "uptime_s": int(float(open("/proc/uptime").read().split()[0])),
        "disk_free_gb": round(free / (1024**3), 1),
        "mem_available_mb": mem_avail,
    }
    
    return result


def cmd_body_state():
    """Current body state: servo angles, posture."""
    result = {}
    
    with dog_lock:
        result["leg_angles"] = list(dog.leg_current_angles) if hasattr(dog, 'leg_current_angles') else None
        result["head_angles"] = list(dog.head_current_angles) if hasattr(dog, 'head_current_angles') else None
        result["tail_angles"] = list(dog.tail_current_angles) if hasattr(dog, 'tail_current_angles') else None
        
        # Posture estimation
        try:
            la = dog.leg_current_angles
            if la:
                if all(abs(a) < 50 for a in la):
                    result["posture"] = "lying"
                elif la[4] > 60 and la[6] < -60:
                    result["posture"] = "sitting"
                else:
                    result["posture"] = "standing"
        except:
            result["posture"] = "unknown"
        
        try:
            v = dog.get_battery_voltage()
            result["battery_v"] = round(v, 2)
            result["battery_pct"] = max(0, min(100, round((v - 6.0) / (8.4 - 6.0) * 100)))
            result["charging"] = v > 8.35
        except:
            pass
    
    return result


def cmd_imu():
    """Raw IMU data."""
    with dog_lock:
        return {
            "pitch": round(dog.pitch, 1) if hasattr(dog, 'pitch') else None,
            "roll": round(dog.roll, 1) if hasattr(dog, 'roll') else None,
        }


def cmd_touch():
    """Touch sensor state."""
    with dog_lock:
        touch = dog.dual_touch.read()
        return {
            "touch": touch,
            "touched": touch != "N",
            "side": {"N": "none", "L": "left", "R": "right", "LS": "slide-left", "RS": "slide-right"}.get(touch, touch)
        }


def cmd_ears():
    """Sound direction sensor."""
    with dog_lock:
        detected = dog.ears.isdetected()
        return {
            "detected": detected,
            "direction_deg": dog.ears.read() if detected else None,
        }


def cmd_scan():
    """Scan surroundings by sweeping head."""
    positions = [(-40, 0, 0), (0, 0, 0), (40, 0, 0)]
    with dog_lock:
        for yaw, roll, pitch in positions:
            dog.head_move([[yaw, roll, pitch]], immediately=True, speed=80)
            time.sleep(0.8)
        dog.head_move([[0, 0, 0]], immediately=True, speed=80)
    return {"ok": True, "scanned": ["left", "center", "right"]}


# ─── Command dispatcher ───
COMMANDS = {
    "status": lambda args: cmd_status(),
    "move": lambda args: cmd_move(args.get("action", "stand"), args.get("steps", 3), args.get("speed", 80)),
    "head": lambda args: cmd_head(args.get("yaw", 0), args.get("roll", 0), args.get("pitch", 0)),
    "rgb": lambda args: cmd_rgb(args.get("r", 128), args.get("g", 0), args.get("b", 255), args.get("mode", "breath"), args.get("bps", 0.8)),
    "photo": lambda args: cmd_photo(args.get("path")),
    "speak": lambda args: cmd_speak(args.get("text", "")),
    "sound": lambda args: cmd_sound(args.get("name", "single_bark_1")),
    "combo": lambda args: cmd_combo(args.get("sequence", "stand:1:60")),
    "wake": lambda args: cmd_wake(),
    "sleep": lambda args: cmd_sleep(),
    "reset": lambda args: cmd_reset(),
    "ping": lambda args: {"pong": True, "ts": time.time()},
    "sensors": lambda args: cmd_sensors(),
    "body_state": lambda args: cmd_body_state(),
    "imu": lambda args: cmd_imu(),
    "touch": lambda args: cmd_touch(),
    "ears": lambda args: cmd_ears(),
    "scan": lambda args: cmd_scan(),
}


def handle_client(conn):
    """Handle a single client connection."""
    try:
        data = conn.recv(4096).decode('utf-8').strip()
        if not data:
            return

        try:
            request = json.loads(data)
        except json.JSONDecodeError:
            # Simple text command: "move sit" or "speak Hallo"
            parts = data.split(None, 1)
            cmd_name = parts[0]
            if len(parts) > 1:
                # Try to parse remaining as key=value or just pass as first arg
                request = {"cmd": cmd_name}
                remaining = parts[1]
                # Simple arg parsing
                if cmd_name == "move":
                    request["action"] = remaining.split()[0]
                elif cmd_name == "speak":
                    request["text"] = remaining
                elif cmd_name == "sound":
                    request["name"] = remaining
                elif cmd_name == "rgb":
                    rgb_parts = remaining.split()
                    if len(rgb_parts) >= 3:
                        request["r"], request["g"], request["b"] = int(rgb_parts[0]), int(rgb_parts[1]), int(rgb_parts[2])
                    if len(rgb_parts) >= 4:
                        request["mode"] = rgb_parts[3]
                elif cmd_name == "head":
                    h_parts = remaining.split()
                    if len(h_parts) >= 3:
                        request["yaw"], request["roll"], request["pitch"] = float(h_parts[0]), float(h_parts[1]), float(h_parts[2])
                elif cmd_name == "combo":
                    request["sequence"] = remaining
                elif cmd_name == "photo":
                    request["path"] = remaining
            else:
                request = {"cmd": cmd_name}

        cmd_name = request.get("cmd", request.get("command", ""))
        if cmd_name in COMMANDS:
            result = COMMANDS[cmd_name](request)
            response = json.dumps(result)
        else:
            response = json.dumps({"error": f"unknown command: {cmd_name}", "available": list(COMMANDS.keys())})

        conn.sendall((response + "\n").encode('utf-8'))
    except Exception as e:
        try:
            conn.sendall(json.dumps({"error": str(e)}).encode('utf-8'))
        except:
            pass
        traceback.print_exc()
    finally:
        conn.close()


def socket_server():
    """Unix socket server for receiving commands."""
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    os.chmod(SOCKET_PATH, 0o777)
    server.listen(5)
    server.settimeout(1.0)
    print(f"[nox] Socket server listening on {SOCKET_PATH}", flush=True)

    while running:
        try:
            conn, _ = server.accept()
            threading.Thread(target=handle_client, args=(conn,), daemon=True).start()
        except socket.timeout:
            continue
        except Exception as e:
            if running:
                print(f"[nox] Socket error: {e}", flush=True)

    server.close()
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)


# Also listen on TCP for remote access from Nox's Pi
def tcp_server(port=9999):
    """TCP server for remote commands from Nox's Pi."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', port))
    server.listen(5)
    server.settimeout(1.0)
    print(f"[nox] TCP server listening on port {port}", flush=True)

    while running:
        try:
            conn, addr = server.accept()
            threading.Thread(target=handle_client, args=(conn,), daemon=True).start()
        except socket.timeout:
            continue
        except Exception as e:
            if running:
                print(f"[nox] TCP error: {e}", flush=True)

    server.close()


def shutdown(signum, frame):
    """Clean shutdown."""
    global running
    print(f"\n[nox] Shutting down (signal {signum})...", flush=True)
    running = False
    try:
        with dog_lock:
            dog.rgb_strip.set_mode('monochromatic', [0, 0, 0])
            dog.do_action('lie', speed=50)
            time.sleep(1)
            dog.close()
    except:
        pass
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    init_dog()

    # Start servers
    sock_thread = threading.Thread(target=socket_server, daemon=True)
    sock_thread.start()

    tcp_thread = threading.Thread(target=tcp_server, daemon=True)
    tcp_thread.start()

    print("[nox] All systems go. Waiting for commands...", flush=True)

    # Keep main thread alive
    try:
        while running:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown(signal.SIGINT, None)
