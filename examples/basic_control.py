#!/usr/bin/env python3
"""Basic robot control example.

Run from the repo root (or the examples/ directory) on the brain machine:
    python3 examples/basic_control.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from brain.nox_body_client import BodyClient

# Connect to your robot (hostname or IP of the body)
robot = BodyClient("pidog.local", 8888)

# Check status
status = robot.status()
print(f"Battery: {status.get('sensors', {}).get('battery_v')}V")
print(f"Behavior state: {status.get('behavior', {}).get('state')}")

# Make it do things
robot.speak("Hallo! Ich bin bereit!")
robot.move("sit")
robot.move("wag_tail")
robot.rgb(0, 255, 0, mode="breath")

# Express emotions (coordinated action + RGB + head + sound)
robot.express("happy")
robot.express("curious")

# Take a photo and save it locally
photo = robot.photo(save_path="/tmp/pidog_photo.jpg")
print(f"Photo saved: {photo.get('saved_to')}")

# Combo: do multiple things at once
robot.combo(
    actions=["stand", "wag_tail"],
    speak="Auf geht's!",
    rgb={"r": 255, "g": 255, "b": 0, "mode": "boom"},
)

# What can this robot do? (endpoints, actions, expressions, sounds)
caps = robot.capabilities()
print(f"Actions: {', '.join(caps.get('actions', []))}")
