#!/bin/bash
# Download ONNX models for face recognition
# SCRFD (detection) + ArcFace (recognition)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "🤖 Downloading face recognition models..."

# SCRFD face detector (16MB)
if [ ! -f "det_10g.onnx" ]; then
    echo "Downloading det_10g.onnx (SCRFD face detector, ~16MB)..."
    wget -q --show-progress -O det_10g.onnx \
        "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/face_detector_scrfd_2.5g.onnx" \
        || wget -q --show-progress -O det_10g.onnx \
        "https://huggingface.co/rockerBOO/scrfd/resolve/main/det_10g.onnx"
    echo "✅ det_10g.onnx"
else
    echo "✅ det_10g.onnx (already exists)"
fi

# ArcFace face recognizer (166MB)
if [ ! -f "w600k_r50.onnx" ]; then
    echo "Downloading w600k_r50.onnx (ArcFace recognizer, ~166MB)..."
    wget -q --show-progress -O w600k_r50.onnx \
        "https://huggingface.co/rockerBOO/arcface/resolve/main/w600k_r50.onnx"
    echo "✅ w600k_r50.onnx"
else
    echo "✅ w600k_r50.onnx (already exists)"
fi

echo ""
echo "🎉 All models downloaded!"
echo "Models directory: $SCRIPT_DIR"
ls -lh *.onnx 2>/dev/null
