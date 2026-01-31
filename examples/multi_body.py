#!/usr/bin/env python3
"""Multi-body control example — same brain, different bodies."""

import sys
sys.path.insert(0, '..')
from brain.nox_body_client import BodyClient

# Define available bodies
bodies = {
    "dog": BodyClient("pidog.local", 8888),
    "car": BodyClient("picar.local", 8888),
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
