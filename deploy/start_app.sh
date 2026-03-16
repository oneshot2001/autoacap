#!/bin/bash
# Start an ACAP application on an Axis camera
# Usage: ./start_app.sh <camera_ip> <camera_pass> <app_name>

set -euo pipefail

CAMERA_IP="${1:?Usage: ./start_app.sh <camera_ip> <camera_pass> <app_name>}"
CAMERA_PASS="${2:?Missing camera password}"
APP_NAME="${3:?Missing app name}"

echo "Starting ${APP_NAME} on ${CAMERA_IP}..." >&2

curl -s -u "root:${CAMERA_PASS}" \
    --connect-timeout 10 \
    "http://${CAMERA_IP}/axis-cgi/applications/control.cgi?action=start&package=${APP_NAME}" >&2

echo "Started." >&2
