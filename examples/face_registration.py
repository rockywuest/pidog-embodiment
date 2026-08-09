#!/usr/bin/env python3
"""Face registration and identification example.

Runs on the BODY (the engine lives in body/, next to the camera).
Download the ONNX models first: cd models && ./download_models.sh
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "body"))
from nox_face_recognition import FaceEngine

repo_root = os.path.join(os.path.dirname(__file__), "..")
engine = FaceEngine(os.path.join(repo_root, "models"),
                    os.path.join(repo_root, "face_db"))

# Register a face from an image
result = engine.register("Rocky", "rocky_photo.jpg")
print(f"Registered: {result}")

# Register more samples (improves accuracy)
engine.register("Rocky", "rocky_photo2.jpg")
engine.register("Rocky", "rocky_photo3.jpg")

# Identify faces in a new image
faces = engine.identify("new_photo.jpg")
for face in faces:
    print(f"  {face['name']} (confidence: {face['confidence']:.0%})")
    print(f"  Location: {[int(x) for x in face['bbox']]}")

# List all known faces
print(f"\nKnown faces: {engine.list_known()}")
