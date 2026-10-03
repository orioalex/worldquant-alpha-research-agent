#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${ALPHA_AGENT_APP_DIR:-$SCRIPT_DIR}"
PID_FILE="$APP_DIR/alpha-agent.pid"

if [[ ! -f "$PID_FILE" ]]; then
  printf 'Alpha agent is not running.\n'
  exit 0
fi

pid="$(cat "$PID_FILE")"
if kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  printf 'Stopped alpha agent PID %s\n' "$pid"
else
  printf 'PID %s is no longer running.\n' "$pid"
fi
rm -f "$PID_FILE"
