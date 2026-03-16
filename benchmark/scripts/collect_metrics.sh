#!/bin/bash
# Collect performance metrics from a running ACAP on an Axis camera
# Usage: ./collect_metrics.sh <camera_ip> <camera_pass> <app_name> <duration_seconds>
#
# Outputs JSON with: fps, latency_p95_ms, memory_rss_mb, cpu_percent
# The ACAP app must write detection results to /tmp/autoacap_detections.json on camera

set -euo pipefail

CAMERA_IP="${1:?Usage: ./collect_metrics.sh <camera_ip> <camera_pass> <app_name> <duration>}"
CAMERA_PASS="${2:?Missing camera password}"
APP_NAME="${3:?Missing app name}"
DURATION="${4:?Missing duration in seconds}"

ssh_cam() {
    sshpass -p "$CAMERA_PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "root@${CAMERA_IP}" "$@"
}

# Find the ACAP process PID
PID=$(ssh_cam "pgrep -f ${APP_NAME}" 2>/dev/null | head -1)
if [ -z "$PID" ]; then
    echo '{"fps": 0, "latency_p95_ms": 0, "memory_rss_mb": 0, "cpu_percent": 0, "error": "process not found"}'
    exit 1
fi

echo "Monitoring PID ${PID} for ${DURATION}s..." >&2

# Collect CPU samples over the benchmark duration
CPU_SAMPLES=()
MEM_SAMPLES=()
SAMPLE_INTERVAL=5
NUM_SAMPLES=$((DURATION / SAMPLE_INTERVAL))

for i in $(seq 1 "$NUM_SAMPLES"); do
    # Get CPU and memory from /proc
    STATS=$(ssh_cam "
        # CPU from /proc/stat (process-level)
        CPU=\$(ps -p ${PID} -o %cpu= 2>/dev/null | tr -d ' ')
        # Memory RSS in KB from /proc/pid/status
        MEM_KB=\$(grep VmRSS /proc/${PID}/status 2>/dev/null | awk '{print \$2}')
        echo \"\${CPU:-0} \${MEM_KB:-0}\"
    " 2>/dev/null)

    CPU_VAL=$(echo "$STATS" | awk '{print $1}')
    MEM_KB=$(echo "$STATS" | awk '{print $2}')
    MEM_MB=$(echo "scale=1; ${MEM_KB:-0} / 1024" | bc)

    CPU_SAMPLES+=("$CPU_VAL")
    MEM_SAMPLES+=("$MEM_MB")

    sleep "$SAMPLE_INTERVAL"
done

# Get FPS and latency from the ACAP's own output (expects app to write metrics)
# The ACAP should write a JSON file with per-frame timing data
APP_METRICS=$(ssh_cam "cat /tmp/autoacap_metrics.json 2>/dev/null" || echo '{}')

# Compute averages
AVG_CPU=$(python3 -c "
samples = [${CPU_SAMPLES[*]// /,}]
print(f'{sum(samples)/len(samples):.1f}' if samples else '0')
" 2>/dev/null || echo "0")

AVG_MEM=$(python3 -c "
samples = [${MEM_SAMPLES[*]// /,}]
print(f'{sum(samples)/len(samples):.1f}' if samples else '0')
" 2>/dev/null || echo "0")

# Parse FPS and latency from app metrics (or use defaults)
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
    "memory_rss_mb": ${AVG_MEM},
    "cpu_percent": ${AVG_CPU}
}
EOF
