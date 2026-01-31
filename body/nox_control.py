#!/usr/bin/env python3
"""
nox_control.py — Nox's body control interface.
Called via SSH from Clawdbot. Accepts commands as CLI args.

Usage:
  python3 nox_control.py <command> [args...]

Commands:
  status          — Battery, uptime, system info
  photo [path]    — Take a photo, save to path (default: /tmp/nox_snap.jpg)
  move <action>   — Perform movement (forward, backward, turn_left, turn_right, 
                     stand, sit, lie, wag_tail, bark, trot, doze_off, stretch,
                     push_up, howling, pant)
  head <yaw> <roll> <pitch>  — Move head (degrees)
  rgb <r> <g> <b> [mode]     — Set RGB strip (mode: breath|listen|speak|off)
  speak <text>               — TTS via Piper (German)
  sound <name>               — Play built-in sound (bark, growl, howling, etc.)
  distance                   — Read ultrasonic distance
  battery                    — Battery voltage
  wake                       — Wake up sequence
  sleep                      — Go to sleep
  reset                      — Reset to neutral standing position
"""

import sys
import os
import time
import subprocess
import json

PIDOG_DIR = "/home/pidog/pidog"
SOUNDS_DIR = os.path.join(PIDOG_DIR, "sounds")
PIPER_BIN = "/home/pidog/.local/bin/piper"
PIPER_MODEL = "/home/pidog/.local/share/piper-voices/de_DE-thorsten-high.onnx"

sys.path.insert(0, os.path.join(PIDOG_DIR, "gpt_examples"))

def get_dog():
    """Initialize PiDog — cached per process."""
    from pidog import Pidog
    dog = Pidog()
    time.sleep(0.5)
    return dog

def cmd_status():
    import platform
    info = {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "arch": platform.machine(),
    }
    # Battery
    try:
        dog = get_dog()
        info["battery_v"] = round(dog.get_battery_voltage(), 2)
        dog.close()
    except:
        info["battery_v"] = "error"
    # Uptime
    with open("/proc/uptime") as f:
        info["uptime_s"] = int(float(f.read().split()[0]))
    # Memory
    import shutil
    total, used, free = shutil.disk_usage("/")
    info["disk_free_gb"] = round(free / (1024**3), 1)
    
    print(json.dumps(info, indent=2))

def cmd_photo(path="/tmp/nox_snap.jpg"):
    from vilib import Vilib
    dirname = os.path.dirname(path) or "/tmp"
    basename = os.path.splitext(os.path.basename(path))[0]
    Vilib.camera_start(vflip=False, hflip=False)
    time.sleep(1.5)
    Vilib.take_photo(basename, dirname)
    Vilib.camera_close()
    actual = os.path.join(dirname, basename + ".jpg")
    print(f"PHOTO:{actual}")

def cmd_move(action, steps=3, speed=80):
    dog = get_dog()
    try:
        dog.do_action(action, step_count=int(steps), speed=int(speed))
        time.sleep(2)
        print(f"OK:{action}")
    finally:
        dog.close()

def cmd_head(yaw=0, roll=0, pitch=0):
    dog = get_dog()
    try:
        dog.head_move([[float(yaw), float(roll), float(pitch)]], immediately=True, speed=80)
        time.sleep(1)
        print(f"OK:head [{yaw},{roll},{pitch}]")
    finally:
        dog.close()

def cmd_rgb(r=128, g=0, b=255, mode="breath"):
    dog = get_dog()
    try:
        color = [int(r), int(g), int(b)]
        if mode == "off":
            dog.rgb_strip.set_mode('off')
        else:
            bps = {"breath": 0.8, "listen": 1.5, "speak": 2.0}.get(mode, 0.8)
            dog.rgb_strip.set_mode(mode if mode in ['breath','listen','speak'] else 'breath', color, bps=bps)
        time.sleep(0.5)
        print(f"OK:rgb {color} {mode}")
    finally:
        dog.close()

def cmd_speak(text):
    """TTS via Piper → aplay"""
    wav_path = "/tmp/nox_speak.wav"
    proc = subprocess.run(
        f'echo "{text}" | {PIPER_BIN} --model {PIPER_MODEL} --output_file {wav_path}',
        shell=True, capture_output=True, text=True
    )
    if proc.returncode != 0:
        print(f"ERROR:piper {proc.stderr}")
        return
    subprocess.run(["aplay", wav_path], capture_output=True)
    print(f"OK:spoke '{text}'")

def cmd_sound(name):
    """Play built-in sound file."""
    for ext in ['.wav', '.mp3']:
        path = os.path.join(SOUNDS_DIR, name + ext)
        if os.path.exists(path):
            if ext == '.mp3':
                subprocess.run(["mpg123", "-q", path], capture_output=True)
            else:
                subprocess.run(["aplay", path], capture_output=True)
            print(f"OK:sound {name}")
            return
    # Try exact filename
    path = os.path.join(SOUNDS_DIR, name)
    if os.path.exists(path):
        subprocess.run(["aplay", path], capture_output=True)
        print(f"OK:sound {name}")
    else:
        available = os.listdir(SOUNDS_DIR)
        print(f"ERROR:sound not found. Available: {available}")

def cmd_battery():
    dog = get_dog()
    try:
        v = dog.get_battery_voltage()
        print(f"BATTERY:{v:.2f}V")
    finally:
        dog.close()

def cmd_distance():
    dog = get_dog()
    try:
        d = dog.read_distance()
        print(f"DISTANCE:{d}cm")
    finally:
        dog.close()

def cmd_wake():
    dog = get_dog()
    try:
        dog.do_action('stand', speed=60)
        time.sleep(1)
        dog.do_action('wag_tail', step_count=5, speed=80)
        time.sleep(1)
        dog.rgb_strip.set_mode('breath', [128, 0, 255], bps=0.8)
        print("OK:wake")
    finally:
        dog.close()

def cmd_sleep():
    dog = get_dog()
    try:
        dog.do_action('lie', speed=50)
        time.sleep(1)
        dog.rgb_strip.set_mode('breath', [0, 0, 80], bps=0.3)
        time.sleep(1)
        dog.do_action('doze_off', speed=50)
        print("OK:sleep")
    finally:
        dog.close()

def cmd_reset():
    dog = get_dog()
    try:
        dog.do_action('stand', speed=60)
        time.sleep(1)
        dog.head_move([[0, 0, 0]], immediately=True, speed=60)
        dog.rgb_strip.set_mode('off')
        time.sleep(1)
        print("OK:reset")
    finally:
        dog.close()

def cmd_combo(actions_str):
    """Run a sequence of actions. Format: action1:steps:speed,action2:steps:speed,..."""
    dog = get_dog()
    try:
        for part in actions_str.split(","):
            parts = part.strip().split(":")
            action = parts[0]
            steps = int(parts[1]) if len(parts) > 1 else 3
            speed = int(parts[2]) if len(parts) > 2 else 80
            dog.do_action(action, step_count=steps, speed=speed)
            time.sleep(1.5)
        print(f"OK:combo {actions_str}")
    finally:
        dog.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    
    cmd = sys.argv[1]
    args = sys.argv[2:]
    
    commands = {
        "status": lambda: cmd_status(),
        "photo": lambda: cmd_photo(args[0] if args else "/tmp/nox_snap.jpg"),
        "move": lambda: cmd_move(args[0], *(args[1:3])),
        "head": lambda: cmd_head(*args[:3]),
        "rgb": lambda: cmd_rgb(*args[:4]),
        "speak": lambda: cmd_speak(" ".join(args)),
        "sound": lambda: cmd_sound(args[0] if args else "single_bark_1"),
        "battery": lambda: cmd_battery(),
        "distance": lambda: cmd_distance(),
        "wake": lambda: cmd_wake(),
        "sleep": lambda: cmd_sleep(),
        "reset": lambda: cmd_reset(),
        "combo": lambda: cmd_combo(args[0] if args else "stand:1:60"),
    }
    
    if cmd in commands:
        try:
            commands[cmd]()
        except Exception as e:
            print(f"ERROR:{cmd}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Unknown command: {cmd}")
        print(f"Available: {', '.join(commands.keys())}")
        sys.exit(1)
