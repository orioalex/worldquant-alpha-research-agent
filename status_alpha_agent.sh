#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${ALPHA_AGENT_APP_DIR:-$SCRIPT_DIR}"
PID_FILE="$APP_DIR/alpha-agent.pid"
LOG_FILE="$APP_DIR/alpha-agent.log"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  printf 'running pid=%s\n' "$(cat "$PID_FILE")"
else
  printf 'stopped\n'
fi
if [[ -f "$LOG_FILE" ]]; then
  tail -n 20 "$LOG_FILE"
fi
