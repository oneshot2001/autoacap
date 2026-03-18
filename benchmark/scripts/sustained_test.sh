#!/bin/bash
# sustained_test.sh — Run an ACAP variant for an extended period, sampling memory every N seconds
#
# Usage: ./sustained_test.sh <c|rust> <camera_ip> <camera_pass> <duration_minutes> <sample_interval_secs>
# Example: ./sustained_test.sh c 192.168.1.33 pass 10 10
#
# Output: Tab-separated log to stdout (redirect to file)
#   elapsed_s  fps  latency_p95  memory_rss_kb  memory_peak_kb  frame_count  status

set -euo pipefail

LANG="${1:?Usage: ./sustained_test.sh <c|rust> <camera_ip> <camera_pass> <duration_min> <sample_interval_sec>}"
CAMERA_IP="${2:?Missing camera IP}"
CAMERA_PASS="${3:?Missing camera password}"
DURATION_MIN="${4:-10}"
SAMPLE_INTERVAL="${5:-10}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_NAME="autoacap"
SSH_USER="alpha"
SSH_PASS="alpha2026"
TOTAL_SECONDS=$((DURATION_MIN * 60))

ssh_cam() {
    sshpass -p "$SSH_PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${SSH_USER}@${CAMERA_IP}" "$@" 2>/dev/null
}

vapix() {
    curl -s --anyauth -u "root:${CAMERA_PASS}" --connect-timeout 10 "$@"
}

# --- Step 1: Build and deploy ---
echo "Building ${LANG} variant..." >&2
case $LANG in
    c)
        cd "${SCRIPT_DIR}/src/c"
        docker build -t autoacap-c . >&2 2>&1
        CONTAINER_ID=$(docker create autoacap-c)
        rm -rf /tmp/autoacap-build/
        docker cp "${CONTAINER_ID}:/opt/app/" /tmp/autoacap-build/ 2>/dev/null || true
        docker rm "${CONTAINER_ID}" >/dev/null 2>&1
        EAP_FILE=$(find /tmp/autoacap-build -name "*.eap" 2>/dev/null | head -1)
        cd "${SCRIPT_DIR}"
        ;;
    rust)
        cd "${SCRIPT_DIR}/src/rust"
        docker build -t autoacap-rust . >&2 2>&1
        CONTAINER_ID=$(docker create autoacap-rust)
        rm -rf /tmp/autoacap-build/
        docker cp "${CONTAINER_ID}:/opt/app/" /tmp/autoacap-build/ 2>/dev/null || true
        docker rm "${CONTAINER_ID}" >/dev/null 2>&1
        EAP_FILE=$(find /tmp/autoacap-build -name "*.eap" 2>/dev/null | head -1)
        cd "${SCRIPT_DIR}"
        ;;
esac

if [ -z "${EAP_FILE:-}" ] || [ ! -f "${EAP_FILE}" ]; then
    echo "BUILD FAILED" >&2
    exit 1
fi

echo "Deploying to ${CAMERA_IP}..." >&2
"${SCRIPT_DIR}/deploy/deploy_eap.sh" "${EAP_FILE}" "${CAMERA_IP}" "${CAMERA_PASS}"

echo "Starting ACAP..." >&2
"${SCRIPT_DIR}/deploy/start_app.sh" "${CAMERA_IP}" "${CAMERA_PASS}" "${APP_NAME}"

echo "Warming up for 15s..." >&2
sleep 15

# Verify it's running
STATUS=$(vapix "http://${CAMERA_IP}/axis-cgi/applications/list.cgi" 2>/dev/null | grep "autoacap" | grep -o 'Status="[^"]*"' | cut -d'"' -f2)
if [ "$STATUS" != "Running" ]; then
    echo "ACAP not running (status: ${STATUS:-unknown}). Aborting." >&2
    exit 1
fi

# --- Step 2: Sample loop ---
echo "Starting ${DURATION_MIN}-minute sustained test (sampling every ${SAMPLE_INTERVAL}s)..." >&2
echo "elapsed_s	fps	latency_p95	memory_rss_kb	memory_peak_kb	status"

START_TIME=$(date +%s)
SAMPLE=0
CRASHES=0

while true; do
    ELAPSED=$(( $(date +%s) - START_TIME ))
    if [ "$ELAPSED" -ge "$TOTAL_SECONDS" ]; then
        break
    fi

    # Check if ACAP is still running
    STATUS=$(vapix "http://${CAMERA_IP}/axis-cgi/applications/list.cgi" 2>/dev/null | grep "autoacap" | grep -o 'Status="[^"]*"' | cut -d'"' -f2)
    if [ "$STATUS" != "Running" ]; then
        echo "${ELAPSED}	0	0	0	0	CRASHED"
        CRASHES=$((CRASHES + 1))
        echo "CRASH DETECTED at ${ELAPSED}s (crash #${CRASHES})" >&2
        # Restart and continue
        "${SCRIPT_DIR}/deploy/start_app.sh" "${CAMERA_IP}" "${CAMERA_PASS}" "${APP_NAME}" 2>/dev/null
        sleep 5
        continue
    fi

    # Read self-reported metrics from the ACAP
    METRICS=$(ssh_cam "cat /tmp/autoacap_metrics.json 2>/dev/null" || echo '{}')

    FPS=$(echo "$METRICS" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('fps',0))" 2>/dev/null || echo "0")
    RSS=$(echo "$METRICS" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('memory_rss_kb',0))" 2>/dev/null || echo "0")
    PEAK=$(echo "$METRICS" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('memory_peak_kb',0))" 2>/dev/null || echo "0")

    # Compute P95 latency from the latencies array
    LAT_P95=$(echo "$METRICS" | python3 -c "
import json, sys
d = json.load(sys.stdin)
lats = d.get('latencies_ms', [])
if lats:
    lats.sort()
    idx = int(len(lats) * 0.95)
    print(round(lats[min(idx, len(lats)-1)], 1))
else:
    print(0)
" 2>/dev/null || echo "0")

    echo "${ELAPSED}	${FPS}	${LAT_P95}	${RSS}	${PEAK}	ok"

    SAMPLE=$((SAMPLE + 1))
    if [ $((SAMPLE % 6)) -eq 0 ]; then
        echo "  [${ELAPSED}s] FPS=${FPS} RSS=${RSS}KB Peak=${PEAK}KB Crashes=${CRASHES}" >&2
    fi

    sleep "$SAMPLE_INTERVAL"
done

# --- Step 3: Final summary ---
echo "" >&2
echo "=== SUSTAINED TEST COMPLETE ===" >&2
echo "Language: ${LANG}" >&2
echo "Duration: ${DURATION_MIN} minutes" >&2
echo "Samples: ${SAMPLE}" >&2
echo "Crashes: ${CRASHES}" >&2
echo "" >&2

# Stop ACAP
"${SCRIPT_DIR}/deploy/stop_app.sh" "${CAMERA_IP}" "${CAMERA_PASS}" "${APP_NAME}" 2>/dev/null
