#!/bin/bash
# Collect performance metrics from a running ACAP on an Axis camera
# Usage: ./collect_metrics.sh <camera_ip> <camera_pass> <app_name> <duration_seconds>
#
# Uses SSH account 'alpha' for /proc access and VAPIX for API calls.
# Outputs JSON with: fps, latency_p95_ms, memory_rss_mb, cpu_percent
#
# The ACAP app writes its own metrics to /tmp/autoacap_metrics.json on camera.

set -euo pipefail

CAMERA_IP="${1:?Usage: ./collect_metrics.sh <camera_ip> <camera_pass> <app_name> <duration>}"
CAMERA_PASS="${2:?Missing camera password}"
APP_NAME="${3:?Missing app name}"
DURATION="${4:?Missing duration in seconds}"

# SSH credentials (separate SSH account on AXIS OS 12)
SSH_USER="alpha"
SSH_PASS="alpha2026"

ssh_cam() {
    sshpass -p "$SSH_PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${SSH_USER}@${CAMERA_IP}" "$@"
}

# Find the ACAP process PID
PID=$(ssh_cam "ps -e -o pid,comm 2>/dev/null | grep ${APP_NAME} | head -1 | awk '{print \$1}'" 2>/dev/null)
if [ -z "$PID" ]; then
    echo '{"fps": 0, "latency_p95_ms": 0, "memory_rss_mb": 0, "cpu_percent": 0, "error": "process not found"}'
    exit 1
fi

echo "Monitoring PID ${PID} for ${DURATION}s..." >&2

# Collect CPU and memory samples over the benchmark duration
SAMPLE_INTERVAL=5
NUM_SAMPLES=$((DURATION / SAMPLE_INTERVAL))
CPU_SUM=0
MEM_SUM=0
SAMPLE_COUNT=0

for i in $(seq 1 "$NUM_SAMPLES"); do
    STATS=$(ssh_cam "
        # CPU usage from ps
        CPU=\$(ps -p ${PID} -o pcpu= 2>/dev/null | tr -d ' ' || echo '0')
        # Memory RSS in KB
        MEM_KB=\$(cat /proc/${PID}/status 2>/dev/null | grep VmRSS | awk '{print \$2}' || echo '0')
        echo \"\${CPU:-0} \${MEM_KB:-0}\"
    " 2>/dev/null || echo "0 0")

    CPU_VAL=$(echo "$STATS" | awk '{print $1}')
    MEM_KB=$(echo "$STATS" | awk '{print $2}')

    CPU_SUM=$(echo "$CPU_SUM + $CPU_VAL" | bc 2>/dev/null || echo "$CPU_SUM")
    MEM_SUM=$(echo "$MEM_SUM + $MEM_KB" | bc 2>/dev/null || echo "$MEM_SUM")
    SAMPLE_COUNT=$((SAMPLE_COUNT + 1))

    sleep "$SAMPLE_INTERVAL"
done

# Get app-written metrics from camera
APP_METRICS=$(ssh_cam "cat /tmp/autoacap_metrics.json 2>/dev/null" 2>/dev/null || echo '{}')

# Compute averages
if [ "$SAMPLE_COUNT" -gt 0 ]; then
    AVG_CPU=$(echo "scale=1; $CPU_SUM / $SAMPLE_COUNT" | bc 2>/dev/null || echo "0")
    AVG_MEM_KB=$(echo "scale=0; $MEM_SUM / $SAMPLE_COUNT" | bc 2>/dev/null || echo "0")
    AVG_MEM_MB=$(echo "scale=1; $AVG_MEM_KB / 1024" | bc 2>/dev/null || echo "0")
else
    AVG_CPU="0"
    AVG_MEM_MB="0"
fi

# Parse FPS and latency from app-written metrics
FPS=$(echo "$APP_METRICS" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('fps', 0))
except:
    print(0)
" 2>/dev/null || echo "0")

LAT_P95=$(echo "$APP_METRICS" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    lats = d.get('latencies_ms', [])
    if lats:
        lats.sort()
        idx = int(len(lats) * 0.95)
        print(f'{lats[min(idx, len(lats)-1)]:.1f}')
    else:
        print(0)
except:
    print(0)
" 2>/dev/null || echo "0")

# Output JSON
cat <<EOF
{
    "fps": ${FPS},
    "latency_p95_ms": ${LAT_P95},
    "memory_rss_mb": ${AVG_MEM_MB},
    "cpu_percent": ${AVG_CPU}
}
EOF
