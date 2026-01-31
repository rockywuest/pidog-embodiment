#!/usr/bin/env python3
"""Basic robot control example."""

import sys
sys.path.insert(0, '..')
from brain.nox_body_client import BodyClient

# Connect to your robot
robot = BodyClient("pidog.local", 8888)

# Check status
status = robot.status()
print(f"Battery: {status['sensors']['battery_v']}V")
print(f"Posture: {status['body_state']['posture']}")

# Make it do things
robot.speak("Hallo! Ich bin bereit!")
robot.move("sit")
robot.move("wag_tail")
robot.rgb(0, 255, 0, mode="breath")

# Express emotions
robot.express("happy")
robot.express("curious")

# Take a photo
photo = robot.photo()
print(f"Photo: {len(photo.get('image', ''))} bytes")

# Combo: do multiple things at once
robot.combo(
    actions=["stand", "wag_tail"],
    speak="Auf geht's!",
    rgb={"r": 255, "g": 255, "b": 0, "mode": "boom"},
)
