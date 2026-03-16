#!/bin/bash
# Emergency reboot of an Axis camera
# Usage: ./reboot_camera.sh <camera_ip> <camera_pass>

set -euo pipefail

CAMERA_IP="${1:?Usage: ./reboot_camera.sh <camera_ip> <camera_pass>}"
CAMERA_PASS="${2:?Missing camera password}"

echo "Rebooting camera at ${CAMERA_IP}..." >&2

curl -s -u "root:${CAMERA_PASS}" \
    --connect-timeout 10 \
    "http://${CAMERA_IP}/axis-cgi/restart.cgi" >&2

echo "Reboot command sent. Camera will be back in ~90 seconds." >&2
