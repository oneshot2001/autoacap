#!/bin/bash
# evaluate.sh - AutoACAP Fixed Benchmark Harness
# DO NOT MODIFY - This file is the stable ground truth
#
# Usage: ./evaluate.sh <c|rust> <camera_ip> <camera_pass>
# Example: ./evaluate.sh c 192.168.1.33 mypassword
#
# Outputs a single RESULT: line with tab-separated metrics:
# RESULT: <fps> <mAP> <latency_p95> <memory_mb> <cpu_pct> <binary_kb>

set -euo pipefail

LANG="${1:?Usage: ./evaluate.sh <c|rust> <camera_ip> <camera_pass>}"
CAMERA_IP="${2:?Missing camera IP}"
CAMERA_PASS="${3:?Missing camera password}"
BENCHMARK_DURATION=60  # seconds
WARMUP_DURATION=10     # seconds
APP_NAME="autoacap"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# SSH credentials (separate SSH account on AXIS OS 12)
SSH_USER="alpha"
SSH_PASS="alpha2026"

ssh_cam() {
    sshpass -p "$SSH_PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${SSH_USER}@${CAMERA_IP}" "$@"
}

# VAPIX helper (digest auth)
vapix() {
    curl -s --anyauth -u "root:${CAMERA_PASS}" --connect-timeout 10 "$@"
}

# --- Pre-flight: verify camera is reachable ---
echo "Checking camera at ${CAMERA_IP}..." >&2
if ! vapix "http://${CAMERA_IP}/axis-cgi/param.cgi?action=list&group=root.Properties.System.Soc" >/dev/null 2>&1; then
    echo "ERROR: Camera unreachable at ${CAMERA_IP}" >&2
    echo "RESULT:	0	0.0	0	0	0	0"
    exit 1
fi

# --- Step 1: Build ---
echo "Building ${LANG} variant..." >&2
BUILD_START=$(date +%s)

case $LANG in
    c)
        cd "${SCRIPT_DIR}/src/c"
        docker build -t autoacap-c . >&2 2>&1
        # Extract .eap from Docker image
        CONTAINER_ID=$(docker create autoacap-c)
        docker cp "${CONTAINER_ID}:/opt/app/" /tmp/autoacap-build/ 2>/dev/null || true
        docker rm "${CONTAINER_ID}" >/dev/null 2>&1
        EAP_FILE=$(find /tmp/autoacap-build -name "*.eap" 2>/dev/null | head -1)
        cd "${SCRIPT_DIR}"
        ;;
    rust)
        cd "${SCRIPT_DIR}/src/rust"
        cargo-acap-build 2>&1 >&2
        EAP_FILE=$(find target/acap -name "*.eap" 2>/dev/null | head -1)
        cd "${SCRIPT_DIR}"
        ;;
    *)
        echo "ERROR: Unknown language '${LANG}'. Use 'c' or 'rust'." >&2
        exit 1
        ;;
esac

BUILD_END=$(date +%s)
BUILD_TIME=$((BUILD_END - BUILD_START))
echo "Build completed in ${BUILD_TIME}s" >&2

if [ -z "${EAP_FILE:-}" ] || [ ! -f "${EAP_FILE}" ]; then
    echo "BUILD_FAILED: No .eap file produced" >&2
    echo "RESULT:	0	0.0	0	0	0	0"
    exit 1
fi

BINARY_KB=$(( $(stat -f%z "${EAP_FILE}" 2>/dev/null || stat -c%s "${EAP_FILE}" 2>/dev/null) / 1024 ))
echo "Binary size: ${BINARY_KB} KB" >&2

# --- Step 2: Deploy ---
echo "Deploying to ${CAMERA_IP}..." >&2
"${SCRIPT_DIR}/deploy/deploy_eap.sh" "${EAP_FILE}" "${CAMERA_IP}" "${CAMERA_PASS}"

# --- Step 3: Start ACAP + warmup ---
echo "Starting ACAP and warming up for ${WARMUP_DURATION}s..." >&2
"${SCRIPT_DIR}/deploy/start_app.sh" "${CAMERA_IP}" "${CAMERA_PASS}" "${APP_NAME}"
sleep "${WARMUP_DURATION}"

# --- Step 4: Collect metrics over benchmark duration ---
echo "Benchmarking for ${BENCHMARK_DURATION}s..." >&2
"${SCRIPT_DIR}/benchmark/scripts/collect_metrics.sh" \
    "${CAMERA_IP}" "${CAMERA_PASS}" "${APP_NAME}" "${BENCHMARK_DURATION}" \
    > /tmp/autoacap_metrics.json

# --- Step 5: Compute mAP ---
if [ -f "${SCRIPT_DIR}/benchmark/ground_truth/detections.json" ]; then
    python3 "${SCRIPT_DIR}/benchmark/scripts/compute_map.py" \
        --detections /tmp/autoacap_metrics.json \
        --ground-truth "${SCRIPT_DIR}/benchmark/ground_truth/detections.json" \
        > /tmp/autoacap_accuracy.json
    MAP=$(python3 -c "import json; print(json.load(open('/tmp/autoacap_accuracy.json'))['mAP_50'])")
else
    echo "WARNING: No ground truth file found, using mAP=0.0" >&2
    MAP="0.0"
fi

# --- Step 6: Stop ACAP ---
"${SCRIPT_DIR}/deploy/stop_app.sh" "${CAMERA_IP}" "${CAMERA_PASS}" "${APP_NAME}"

# --- Step 7: Parse metrics and output ---
FPS=$(python3 -c "import json; print(json.load(open('/tmp/autoacap_metrics.json')).get('fps', 0))")
LAT=$(python3 -c "import json; print(json.load(open('/tmp/autoacap_metrics.json')).get('latency_p95_ms', 0))")
MEM=$(python3 -c "import json; print(json.load(open('/tmp/autoacap_metrics.json')).get('memory_rss_mb', 0))")
CPU=$(python3 -c "import json; print(json.load(open('/tmp/autoacap_metrics.json')).get('cpu_percent', 0))")

echo "RESULT:	${FPS}	${MAP}	${LAT}	${MEM}	${CPU}	${BINARY_KB}"
