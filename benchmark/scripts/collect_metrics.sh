#!/bin/bash
# Collect performance metrics from a running ACAP on an Axis camera
# Usage: ./collect_metrics.sh <camera_ip> <camera_pass> <app_name> <duration_seconds>
#
# Reads the ACAP's self-reported metrics from /tmp/autoacap_metrics.json
# and supplements with CPU/memory from /proc.
# Outputs JSON to stdout.

set -euo pipefail

CAMERA_IP="${1:?Usage: ./collect_metrics.sh <camera_ip> <camera_pass> <app_name> <duration>}"
CAMERA_PASS="${2:?Missing camera password}"
APP_NAME="${3:?Missing app name}"
DURATION="${4:?Missing duration in seconds}"

SSH_USER="alpha"
SSH_PASS="alpha2026"

ssh_cam() {
    sshpass -p "$SSH_PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${SSH_USER}@${CAMERA_IP}" "$@" 2>/dev/null
}

echo "Waiting ${DURATION}s for benchmark data..." >&2
sleep "$DURATION"

# Read the ACAP's self-reported metrics
APP_METRICS=$(ssh_cam "cat /tmp/autoacap_metrics.json 2>/dev/null" || echo '{"fps":0,"latencies_ms":[]}')

# Get process memory and CPU
PID=$(ssh_cam "ps -e -o pid,comm 2>/dev/null | grep ${APP_NAME} | head -1 | awk '{print \$1}'" || echo "")

if [ -n "$PID" ]; then
    PROC_STATS=$(ssh_cam "
        MEM_KB=\$(cat /proc/${PID}/status 2>/dev/null | grep VmRSS | awk '{print \$2}' || echo '0')
        CPU=\$(ps -p ${PID} -o pcpu= 2>/dev/null | tr -d ' ' || echo '0')
        echo \"\${MEM_KB:-0} \${CPU:-0}\"
    " || echo "0 0")
    MEM_KB=$(echo "$PROC_STATS" | awk '{print $1}')
    CPU_PCT=$(echo "$PROC_STATS" | awk '{print $2}')
else
    MEM_KB=0
    CPU_PCT=0
fi

# Compute final metrics with Python (bc not always available)
python3 -c "
import json, sys

app = json.loads('''${APP_METRICS}''')
fps = app.get('fps', 0)
lats = app.get('latencies_ms', [])

if lats:
    lats.sort()
    p95_idx = int(len(lats) * 0.95)
    lat_p95 = lats[min(p95_idx, len(lats) - 1)]
else:
    lat_p95 = 0

mem_mb = round(${MEM_KB} / 1024, 1)
cpu = float('${CPU_PCT}')

result = {
    'fps': fps,
    'latency_p95_ms': round(lat_p95, 1),
    'memory_rss_mb': mem_mb,
    'cpu_percent': cpu
}
print(json.dumps(result, indent=2))
"
