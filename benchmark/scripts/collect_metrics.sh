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

# Get process memory and CPU by scanning /proc (works on BusyBox without awk)
# Use heredoc to avoid shell escaping issues
PROC_STATS=$(ssh_cam sh << 'REMOTE_SCRIPT'
PID=""
for p in /proc/[0-9]*/cmdline; do
    if grep -ql "/usr/local/packages/autoacap" "$p" 2>/dev/null; then
        dir=$(dirname "$p")
        candidate=$(basename "$dir")
        # Verify we can read status (permissions may vary)
        if [ -r "$dir/status" ]; then
            PID=$candidate
            break
        fi
    fi
done
if [ -n "$PID" ]; then
    MEM_KB=$(grep "^VmRSS:" /proc/$PID/status 2>/dev/null | sed 's/[^0-9]//g')
    MEM_PEAK_KB=$(grep "^VmHWM:" /proc/$PID/status 2>/dev/null | sed 's/[^0-9]//g')
    THREADS=$(grep "^Threads:" /proc/$PID/status 2>/dev/null | sed 's/[^0-9]//g')
    # CPU: two stat samples 1s apart, fields 14+15 (utime+stime)
    STAT1=$(cat /proc/$PID/stat 2>/dev/null)
    T1_U=$(echo "$STAT1" | cut -d' ' -f14)
    T1_S=$(echo "$STAT1" | cut -d' ' -f15)
    sleep 1
    STAT2=$(cat /proc/$PID/stat 2>/dev/null)
    T2_U=$(echo "$STAT2" | cut -d' ' -f14)
    T2_S=$(echo "$STAT2" | cut -d' ' -f15)
    HZ=$(getconf CLK_TCK 2>/dev/null || echo 100)
    DELTA=$(( (T2_U + T2_S) - (T1_U + T1_S) ))
    CPU_PCT=$(( DELTA * 100 / HZ ))
    echo "${MEM_KB:-0} ${MEM_PEAK_KB:-0} ${THREADS:-0} ${CPU_PCT:-0}"
else
    echo "0 0 0 0"
fi
REMOTE_SCRIPT
)
MEM_KB=$(echo "$PROC_STATS" | tail -1 | cut -d' ' -f1)
MEM_PEAK_KB=$(echo "$PROC_STATS" | tail -1 | cut -d' ' -f2)
THREADS=$(echo "$PROC_STATS" | tail -1 | cut -d' ' -f3)
CPU_PCT=$(echo "$PROC_STATS" | tail -1 | cut -d' ' -f4)

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

# Self-reported memory from ACAP (in KB)
rss_kb = app.get('memory_rss_kb', 0)
peak_kb = app.get('memory_peak_kb', 0)
mem_mb = round(rss_kb / 1024, 1)
mem_peak_mb = round(peak_kb / 1024, 1)

# CPU from /proc (may be 0 if permissions deny)
cpu = float('${CPU_PCT}')
threads = int('${THREADS}')

result = {
    'fps': fps,
    'latency_p95_ms': round(lat_p95, 1),
    'memory_rss_mb': mem_mb,
    'memory_peak_mb': mem_peak_mb,
    'cpu_percent': cpu,
    'threads': threads
}
print(json.dumps(result, indent=2))
"
