#!/bin/bash
# Deploy an .eap file to an Axis camera via VAPIX API
# Usage: ./deploy_eap.sh <eap_file> <camera_ip> <camera_pass>
#
# Uses VAPIX application upload API (digest auth) — works without root SSH

set -euo pipefail

EAP_FILE="${1:?Usage: ./deploy_eap.sh <eap_file> <camera_ip> <camera_pass>}"
CAMERA_IP="${2:?Missing camera IP}"
CAMERA_PASS="${3:?Missing camera password}"

echo "Stopping existing autoacap if running..." >&2
curl -s --anyauth -u "root:${CAMERA_PASS}" \
    --connect-timeout 10 \
    "http://${CAMERA_IP}/axis-cgi/applications/control.cgi?action=stop&package=autoacap" >&2 2>/dev/null || true

echo "Removing existing autoacap if installed..." >&2
curl -s --anyauth -u "root:${CAMERA_PASS}" \
    --connect-timeout 10 \
    "http://${CAMERA_IP}/axis-cgi/applications/control.cgi?action=remove&package=autoacap" >&2 2>/dev/null || true

sleep 2

echo "Deploying ${EAP_FILE} to ${CAMERA_IP} via VAPIX..." >&2

RESPONSE=$(curl -s --anyauth -u "root:${CAMERA_PASS}" \
    --connect-timeout 30 \
    -F "file=@${EAP_FILE}" \
    "http://${CAMERA_IP}/axis-cgi/applications/upload.cgi" 2>&1)

if echo "$RESPONSE" | grep -qi "error"; then
    echo "DEPLOY FAILED: ${RESPONSE}" >&2
    exit 1
fi

echo "Deployed successfully." >&2
