#!/usr/bin/env python3
"""
nox_face_recognition.py — Face detection and recognition using ONNX models.

Uses SCRFD for face detection and ArcFace for face recognition.
Designed to run on Raspberry Pi 4 with onnxruntime (CPU).

Can also run on Pi 5 (brain-side) for heavier processing.

Usage:
    # As library
    from nox_face_recognition import FaceEngine
    engine = FaceEngine(model_dir="/path/to/models")
    
    # Detect faces in image
    faces = engine.detect(image_path)
    
    # Register a face
    engine.register("Rocky", image_path)
    
    # Identify faces
    results = engine.identify(image_path)
    
    # As CLI
    python3 nox_face_recognition.py detect photo.jpg
    python3 nox_face_recognition.py register Rocky photo.jpg
    python3 nox_face_recognition.py identify photo.jpg
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path

try:
    import cv2
except ImportError:
    print("OpenCV not available, using PIL fallback")
    cv2 = None

try:
    import onnxruntime as ort
except ImportError:
    print("onnxruntime not available")
    ort = None


class FaceDetector:
    """SCRFD face detector using ONNX runtime."""
    
    def __init__(self, model_path, input_size=(640, 640)):
        if ort is None:
            raise ImportError("onnxruntime required")
        
        self.input_size = input_size
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        # SCRFD detection thresholds
        self.score_threshold = 0.5
        self.nms_threshold = 0.4
        
        # Strides and anchor centers (generated once)
        self.strides = [8, 16, 32]
        self._anchor_centers = {}
    
    def _get_anchor_centers(self, height, width, stride):
        """Generate anchor center points for a given stride."""
        key = (height, width, stride)
        if key not in self._anchor_centers:
            ny = height // stride
            nx = width // stride
            # Each cell has 2 anchors
            centers = []
            for y in range(ny):
                for x in range(nx):
                    centers.append([x * stride, y * stride])
                    centers.append([x * stride, y * stride])
            self._anchor_centers[key] = np.array(centers, dtype=np.float32)
        return self._anchor_centers[key]
    
    def _preprocess(self, img):
        """Preprocess image for SCRFD."""
        h, w = img.shape[:2]
        
        # Resize to input size
        scale_x = self.input_size[1] / w
        scale_y = self.input_size[0] / h
        scale = min(scale_x, scale_y)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(img, (new_w, new_h))
        
        # Pad to input size
        padded = np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # Normalize
        blob = padded.astype(np.float32)
        blob = (blob - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)  # HWC -> CHW
        blob = np.expand_dims(blob, axis=0)
        
        return blob, scale
    
    def _distance2bbox(self, points, distance):
        """Convert distance predictions to bounding boxes.
        
        points: anchor centers [N, 2]
        distance: predictions [N, 4] (left, top, right, bottom distances)
        """
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)
    
    def _distance2kps(self, points, distance):
        """Convert distance predictions to keypoints.
        
        points: anchor centers [N, 2]
        distance: predictions [N, 10] (5 landmarks × 2)
        """
        kps = []
        for i in range(0, 10, 2):
            px = points[:, 0] + distance[:, i]
            py = points[:, 1] + distance[:, i + 1]
            kps.extend([px, py])
        return np.stack(kps, axis=-1)
    
    def detect(self, img):
        """Detect faces in image.
        
        Returns list of dicts: [{bbox: [x1,y1,x2,y2], score: float, kps: [[x,y],...]}]
        """
        if cv2 is None:
            return []
        
        blob, scale = self._preprocess(img)
        
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        
        # SCRFD det_10g outputs (9 tensors):
        #   [0,1,2] = scores for stride 8,16,32
        #   [3,4,5] = bbox distances for stride 8,16,32
        #   [6,7,8] = keypoints for stride 8,16,32
        
        faces = []
        input_h, input_w = self.input_size
        
        for idx, stride in enumerate(self.strides):
            scores = outputs[idx]          # [N, 1]
            bbox_preds = outputs[idx + 3]  # [N, 4]
            kps_preds = outputs[idx + 6]   # [N, 10]
            
            # Get anchor centers
            anchor_centers = self._get_anchor_centers(input_h, input_w, stride)
            
            # Filter by score threshold
            score_vals = scores[:, 0]
            keep = score_vals >= self.score_threshold
            
            if not np.any(keep):
                continue
            
            score_vals = score_vals[keep]
            bbox_preds = bbox_preds[keep] * stride
            kps_preds = kps_preds[keep] * stride
            anchor_centers_filtered = anchor_centers[keep]
            
            # Convert to absolute coordinates
            bboxes = self._distance2bbox(anchor_centers_filtered, bbox_preds)
            kps = self._distance2kps(anchor_centers_filtered, kps_preds)
            
            # Scale back to original image size
            bboxes /= scale
            kps /= scale
            
            for i in range(len(score_vals)):
                keypoints = []
                for j in range(0, 10, 2):
                    keypoints.append([float(kps[i][j]), float(kps[i][j+1])])
                
                faces.append({
                    "bbox": [float(x) for x in bboxes[i]],
                    "score": round(float(score_vals[i]), 3),
                    "keypoints": keypoints,
                })
        
        # NMS
        if len(faces) > 1:
            faces = self._nms(faces)
        
        return faces
    
    def _nms(self, faces, threshold=0.4):
        """Non-maximum suppression."""
        if not faces:
            return []
        
        bboxes = np.array([f["bbox"] for f in faces])
        scores = np.array([f["score"] for f in faces])
        
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]
        
        return [faces[i] for i in keep]


class FaceRecognizer:
    """ArcFace face recognizer using ONNX runtime."""
    
    def __init__(self, model_path, input_size=(112, 112)):
        if ort is None:
            raise ImportError("onnxruntime required")
        
        self.input_size = input_size
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
    
    def get_embedding(self, face_img):
        """Get face embedding vector from aligned face image.
        
        Args:
            face_img: BGR image, ideally 112x112 aligned face crop
        
        Returns:
            512-dim normalized embedding vector
        """
        if cv2 is None:
            return None
        
        # Resize to 112x112
        img = cv2.resize(face_img, self.input_size)
        
        # Normalize
        img = img.astype(np.float32)
        img = (img - 127.5) / 127.5
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, axis=0)
        
        # Run inference
        embedding = self.session.run(None, {self.input_name: img})[0][0]
        
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding


class FaceDB:
    """Face database: stores embeddings for known people."""
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.db_file = os.path.join(db_path, "face_embeddings.json")
        os.makedirs(db_path, exist_ok=True)
        self.faces = self._load()
    
    def _load(self):
        if os.path.exists(self.db_file):
            with open(self.db_file) as f:
                data = json.load(f)
                # Convert lists back to numpy arrays
                for name in data:
                    data[name]["embeddings"] = [
                        np.array(e) for e in data[name]["embeddings"]
                    ]
                return data
        return {}
    
    def _save(self):
        data = {}
        for name, info in self.faces.items():
            data[name] = {
                "embeddings": [e.tolist() for e in info["embeddings"]],
                "registered": info["registered"],
                "image_count": info["image_count"],
            }
        with open(self.db_file, "w") as f:
            json.dump(data, f)
    
    def register(self, name, embedding):
        """Register a face embedding for a person."""
        if name not in self.faces:
            self.faces[name] = {
                "embeddings": [],
                "registered": time.time(),
                "image_count": 0,
            }
        
        self.faces[name]["embeddings"].append(embedding)
        self.faces[name]["image_count"] += 1
        self._save()
        return len(self.faces[name]["embeddings"])
    
    def identify(self, embedding, threshold=0.4):
        """Identify a face by comparing embedding to database.
        
        Returns (name, confidence) or ("unknown", 0.0)
        """
        best_name = "unknown"
        best_score = 0.0
        
        for name, info in self.faces.items():
            for stored_emb in info["embeddings"]:
                # Cosine similarity
                score = float(np.dot(embedding, stored_emb))
                if score > best_score:
                    best_score = score
                    best_name = name
        
        if best_score >= threshold:
            return best_name, round(best_score, 3)
        return "unknown", round(best_score, 3)
    
    def list_known(self):
        return {name: info["image_count"] for name, info in self.faces.items()}
    
    def remove(self, name):
        if name in self.faces:
            del self.faces[name]
            self._save()
            return True
        return False


class FaceEngine:
    """Complete face detection + recognition engine."""
    
    def __init__(self, model_dir, db_dir=None):
        self.model_dir = model_dir
        
        # Initialize detector
        det_path = os.path.join(model_dir, "det_10g.onnx")
        if not os.path.exists(det_path):
            # Fallback to smaller detector
            det_path = os.path.join(model_dir, "det_500m.onnx")
        
        if os.path.exists(det_path):
            print(f"[face] Loading detector: {det_path}")
            self.detector = FaceDetector(det_path)
        else:
            print("[face] No detector model found, using OpenCV Haar cascade")
            self.detector = None
        
        # Initialize recognizer
        rec_path = os.path.join(model_dir, "w600k_r50.onnx")
        if not os.path.exists(rec_path):
            rec_path = os.path.join(model_dir, "w600k_mbf.onnx")
        
        if os.path.exists(rec_path):
            print(f"[face] Loading recognizer: {rec_path}")
            self.recognizer = FaceRecognizer(rec_path)
        else:
            print("[face] No recognizer model found")
            self.recognizer = None
        
        # Face database
        if db_dir is None:
            db_dir = os.path.join(model_dir, "..", "face_db")
        self.db = FaceDB(db_dir)
    
    def _haar_detect(self, img):
        """Fallback detection using OpenCV Haar cascade."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = cascade.detectMultiScale(gray, 1.3, 5)
        
        results = []
        for (x, y, w, h) in faces:
            results.append({
                "bbox": [float(x), float(y), float(x+w), float(y+h)],
                "score": 0.9,  # Haar doesn't give scores
                "keypoints": None,
            })
        return results
    
    def detect(self, img_or_path):
        """Detect faces in image.
        
        Args:
            img_or_path: numpy array (BGR) or path to image file
        
        Returns list of face dicts with bbox, score, crop
        """
        if isinstance(img_or_path, str):
            img = cv2.imread(img_or_path)
        else:
            img = img_or_path
        
        if img is None:
            return []
        
        # Detect
        if self.detector:
            faces = self.detector.detect(img)
        else:
            faces = self._haar_detect(img)
        
        # Add face crops
        h, w = img.shape[:2]
        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face["bbox"]]
            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                face["crop"] = img[y1:y2, x1:x2].copy()
                face["center"] = [int((x1+x2)/2), int((y1+y2)/2)]
                face["size"] = [x2-x1, y2-y1]
            else:
                face["crop"] = None
        
        return faces
    
    def identify(self, img_or_path):
        """Detect and identify all faces in image.
        
        Returns list of face dicts with name and confidence added.
        """
        faces = self.detect(img_or_path)
        
        for face in faces:
            if face.get("crop") is not None and self.recognizer:
                embedding = self.recognizer.get_embedding(face["crop"])
                name, confidence = self.db.identify(embedding)
                face["name"] = name
                face["confidence"] = confidence
                face["embedding"] = embedding
            else:
                face["name"] = "unknown"
                face["confidence"] = 0.0
        
        return faces
    
    def register(self, name, img_or_path):
        """Register a face for identification.
        
        Takes a photo, detects the largest face, and stores its embedding.
        """
        faces = self.detect(img_or_path)
        
        if not faces:
            return {"error": "no face detected"}
        
        # Use the largest face
        largest = max(faces, key=lambda f: f["size"][0] * f["size"][1] if f.get("size") else 0)
        
        if largest.get("crop") is None or self.recognizer is None:
            return {"error": "face crop or recognizer not available"}
        
        embedding = self.recognizer.get_embedding(largest["crop"])
        count = self.db.register(name, embedding)
        
        return {
            "ok": True,
            "name": name,
            "total_embeddings": count,
            "face_size": largest["size"],
        }
    
    def list_known(self):
        return self.db.list_known()


# ─── CLI ───
def main():
    if len(sys.argv) < 2:
        print("""Nox Face Recognition Engine

Usage:
  nox_face_recognition.py detect <image>        Detect faces
  nox_face_recognition.py identify <image>      Detect and identify faces
  nox_face_recognition.py register <name> <img> Register a face
  nox_face_recognition.py list                  List known faces
  nox_face_recognition.py remove <name>         Remove a person
  nox_face_recognition.py benchmark <image>     Benchmark detection speed
""")
        return
    
    cmd = sys.argv[1]
    
    # Find model directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "models")
    db_dir = os.path.join(script_dir, "face_db")
    
    engine = FaceEngine(model_dir, db_dir)
    
    if cmd == "detect":
        img_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/nox_snap.jpg"
        t0 = time.time()
        faces = engine.detect(img_path)
        dt = time.time() - t0
        
        for f in faces:
            if "crop" in f:
                del f["crop"]  # Don't print crop data
        
        print(json.dumps({
            "faces": faces,
            "count": len(faces),
            "time_ms": round(dt * 1000),
        }, indent=2))
    
    elif cmd == "identify":
        img_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/nox_snap.jpg"
        t0 = time.time()
        faces = engine.identify(img_path)
        dt = time.time() - t0
        
        for f in faces:
            if "crop" in f:
                del f["crop"]
            if "embedding" in f:
                del f["embedding"]
        
        print(json.dumps({
            "faces": faces,
            "count": len(faces),
            "time_ms": round(dt * 1000),
        }, indent=2))
    
    elif cmd == "register":
        name = sys.argv[2]
        img_path = sys.argv[3] if len(sys.argv) > 3 else "/tmp/nox_snap.jpg"
        result = engine.register(name, img_path)
        print(json.dumps(result, indent=2))
    
    elif cmd == "list":
        print(json.dumps(engine.list_known(), indent=2))
    
    elif cmd == "remove":
        name = sys.argv[2]
        result = engine.db.remove(name)
        print(json.dumps({"removed": result, "name": name}))
    
    elif cmd == "benchmark":
        img_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/nox_snap.jpg"
        
        # Warm up
        engine.detect(img_path)
        
        times = []
        for _ in range(5):
            t0 = time.time()
            engine.detect(img_path)
            times.append(time.time() - t0)
        
        avg = sum(times) / len(times)
        print(json.dumps({
            "avg_ms": round(avg * 1000),
            "min_ms": round(min(times) * 1000),
            "max_ms": round(max(times) * 1000),
            "fps": round(1.0 / avg, 1) if avg > 0 else 0,
        }, indent=2))
    
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
