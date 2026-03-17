#!/bin/bash
# Stop an ACAP application on an Axis camera via VAPIX
# Usage: ./stop_app.sh <camera_ip> <camera_pass> <app_name>

set -euo pipefail

CAMERA_IP="${1:?Usage: ./stop_app.sh <camera_ip> <camera_pass> <app_name>}"
CAMERA_PASS="${2:?Missing camera password}"
APP_NAME="${3:?Missing app name}"

echo "Stopping ${APP_NAME} on ${CAMERA_IP}..." >&2

curl -s --anyauth -u "root:${CAMERA_PASS}" \
    --connect-timeout 10 \
    "http://${CAMERA_IP}/axis-cgi/applications/control.cgi?action=stop&package=${APP_NAME}" >&2

echo "Stopped." >&2
