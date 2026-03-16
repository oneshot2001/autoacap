#!/bin/bash
# Deploy an .eap file to an Axis camera
# Usage: ./deploy_eap.sh <eap_file> <camera_ip> <camera_pass>

set -euo pipefail

EAP_FILE="${1:?Usage: ./deploy_eap.sh <eap_file> <camera_ip> <camera_pass>}"
CAMERA_IP="${2:?Missing camera IP}"
CAMERA_PASS="${3:?Missing camera password}"

echo "Deploying ${EAP_FILE} to ${CAMERA_IP}..." >&2

# Upload .eap via VAPIX application management API
curl -s -u "root:${CAMERA_PASS}" \
    --connect-timeout 10 \
    -F "file=@${EAP_FILE}" \
    "http://${CAMERA_IP}/axis-cgi/applications/upload.cgi" >&2

echo "Deployed successfully." >&2
