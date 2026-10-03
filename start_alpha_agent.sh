#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${ALPHA_AGENT_APP_DIR:-$SCRIPT_DIR}"
PID_FILE="$APP_DIR/alpha-agent.pid"
LOG_FILE="$APP_DIR/alpha-agent.log"
ENV_FILE="$APP_DIR/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  printf 'Missing %s; copy .env.example to .env and configure WQ credentials.\n' "$ENV_FILE" >&2
  exit 2
fi

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  printf 'Alpha agent already running with PID %s\n' "$(cat "$PID_FILE")"
  exit 0
fi

if ! grep -Eq '^(WQB_COOKIE_HEADER=[^[:space:]]+|WQB_EMAIL=[^[:space:]]+)' "$ENV_FILE" || ! grep -Eq '^(WQB_COOKIE_HEADER=[^[:space:]]+|WQB_PASSWORD=[^[:space:]]+)' "$ENV_FILE"; then
  printf 'Configure WQB_EMAIL/WQB_PASSWORD or WQB_COOKIE_HEADER in %s before starting.\n' "$ENV_FILE" >&2
  exit 2
fi

# Load the configured runtime values before expanding command-line defaults.
# This makes ALPHA_AGENT_SUBMISSION_MODE in .env effective for this wrapper.
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

cd "$APP_DIR"
nohup "$APP_DIR/.venv/bin/python" "$APP_DIR/alpha_research_agent.py" \
  --pretty run \
  --budget "${ALPHA_AGENT_BUDGET:-24}" \
  --max-iterations "${ALPHA_AGENT_MAX_ITERATIONS:-12}" \
  --submission-mode "${ALPHA_AGENT_SUBMISSION_MODE:-disabled}" \
  > "$LOG_FILE" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$PID_FILE"
printf 'Started alpha agent with PID %s; log: %s\n' "$(cat "$PID_FILE")" "$LOG_FILE"
