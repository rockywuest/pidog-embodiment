#!/usr/bin/env python3
"""Face registration and identification example."""

import sys
sys.path.insert(0, '..')
from brain.nox_face_recognition import FaceEngine

# Initialize (download models first: cd models && ./download_models.sh)
engine = FaceEngine("../models", "../face_db")

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
