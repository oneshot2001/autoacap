#!/bin/bash
# Capture a 60-second benchmark video from the camera
# Usage: ./capture_benchmark.sh <camera_ip> <camera_pass> [output_dir]

set -euo pipefail

CAMERA_IP="${1:?Usage: ./capture_benchmark.sh <camera_ip> <camera_pass> [output_dir]}"
CAMERA_PASS="${2:?Missing camera password}"
OUTPUT_DIR="${3:-$(dirname "$0")/../video}"
DURATION=60

echo "Capturing ${DURATION}s benchmark video from ${CAMERA_IP}..." >&2

# Option 1: RTSP stream capture
ffmpeg -y -rtsp_transport tcp \
    -i "rtsp://root:${CAMERA_PASS}@${CAMERA_IP}/axis-media/media.amp?resolution=640x480&fps=15" \
    -t "${DURATION}" \
    -c:v copy \
    "${OUTPUT_DIR}/benchmark_60s.h264"

echo "Saved to ${OUTPUT_DIR}/benchmark_60s.h264" >&2
echo "" >&2
echo "Next steps:" >&2
echo "  1. Extract key frames for labeling:" >&2
echo "     ffmpeg -i ${OUTPUT_DIR}/benchmark_60s.h264 -vf 'select=not(mod(n\\,15))' -vsync vfr frames/frame_%04d.jpg" >&2
echo "  2. Label with CVAT, Label Studio, or Roboflow" >&2
echo "  3. Export as COCO JSON to benchmark/ground_truth/detections.json" >&2
