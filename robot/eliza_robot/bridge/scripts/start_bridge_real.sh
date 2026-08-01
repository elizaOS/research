#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
TRACE_LOG_PATH="${TRACE_LOG_PATH:-}"
: "${ELIZA_ROBOT_BRIDGE_AUTH_TOKEN:?set a random 32-to-4096-character visible-ASCII token}"
: "${ELIZA_ROBOT_PHYSICAL_RESOURCE_ID:?set the stable inventory identity for this robot}"
if (( ${#ELIZA_ROBOT_BRIDGE_AUTH_TOKEN} < 32 || ${#ELIZA_ROBOT_BRIDGE_AUTH_TOKEN} > 4096 )) ||
  ! (export LC_ALL=C; [[ "$ELIZA_ROBOT_BRIDGE_AUTH_TOKEN" =~ ^[\!-~]+$ ]]); then
  echo "ELIZA_ROBOT_BRIDGE_AUTH_TOKEN must contain 32-4096 visible ASCII characters" >&2
  exit 64
fi
exec python -m eliza_robot.bridge.server \
  --backend ros_real \
  --host 127.0.0.1 \
  --port 9100 \
  --queue-size 256 \
  --max-commands-per-sec 30 \
  --deadman-timeout-sec 1.0 \
  --trace-log-path "$TRACE_LOG_PATH"
