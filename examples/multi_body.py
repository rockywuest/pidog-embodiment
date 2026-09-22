#!/usr/bin/env python3
"""Multi-body control example — same brain, different bodies.

Run from the repo root (or the examples/ directory) on the brain machine:
    python3 examples/multi_body.py
    PIDOG_HOST=mydog.local PICAR_HOST=mycar.local python3 examples/multi_body.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from brain.nox_body_client import BodyClient

# Hostnames come from the environment so this file needs no editing.
PORT = int(os.environ.get("PIDOG_BRIDGE_PORT", 8888))

# Define available bodies
bodies = {
    "dog": BodyClient(os.environ.get("PIDOG_HOST", "pidog.local"), PORT),
    "car": BodyClient(os.environ.get("PICAR_HOST", "picar.local"), PORT),
}

# Choose active body
active = "dog"
robot = bodies[active]

# Switch bodies seamlessly
def switch_body(name):
    global active, robot
    if name in bodies:
        # Say goodbye to current body
        robot.speak(f"Ich wechsle zu {name}")
        
        # Switch
        active = name
        robot = bodies[name]
        
        # Greet from new body
        robot.speak(f"Jetzt bin ich ein {name}!")
        return True
    return False

# Use current body
robot.speak("Ich bin bereit!")
robot.move("sit")

# Switch to car
switch_body("car")
robot.move("forward")

# Switch back to dog
switch_body("dog")
robot.move("wag_tail")
