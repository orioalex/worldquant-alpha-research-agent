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

if [[ "${ALPHA_AGENT_PLANNER_PROVIDER:-heuristic}" == "openai" \
  && "${ALPHA_AGENT_LOCAL_OLLAMA_AUTOSTART:-true}" =~ ^(1|true|yes|on)$ ]]; then
  OLLAMA_LOCAL_BIN="${OLLAMA_LOCAL_BIN:-/tmp2/b12902064/ollama-local/bin/ollama}"
  OLLAMA_LOCAL_RUNNER="${OLLAMA_LOCAL_RUNNER:-/tmp2/b12902064/ollama-local/lib/ollama/llama-server}"
  OLLAMA_LOCAL_HOST="${OLLAMA_LOCAL_HOST:-127.0.0.1:11436}"
  OLLAMA_LOCAL_LOG="${OLLAMA_LOCAL_LOG:-/tmp2/b12902064/ollama-local-11436.log}"
  OLLAMA_LOCAL_PID="${OLLAMA_LOCAL_PID:-/tmp2/b12902064/ollama-local-11436.pid}"
  OLLAMA_MODELS="${OLLAMA_MODELS:-/tmp2/b12902064/.ollama/models}"
  OLLAMA_MODEL_PATH="${OLLAMA_MODEL_PATH:-}"
  OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-8192}"
  OLLAMA_CUDA_VISIBLE_DEVICES="${OLLAMA_CUDA_VISIBLE_DEVICES:-}"
  OLLAMA_LLM_LIBRARY="${OLLAMA_LLM_LIBRARY:-vulkan}"
  GGML_VK_VISIBLE_DEVICES="${GGML_VK_VISIBLE_DEVICES:-0,1,3,4,5}"
  OLLAMA_DIRECT_RUNNER="${ALPHA_AGENT_LOCAL_OLLAMA_DIRECT_RUNNER:-false}"
  ollama_url="http://${OLLAMA_LOCAL_HOST}/v1/models"
  if [[ "$OLLAMA_DIRECT_RUNNER" =~ ^(1|true|yes|on)$ ]]; then
    if ! curl -fsS --max-time 3 "$ollama_url" >/dev/null 2>&1; then
      if [[ ! -x "$OLLAMA_LOCAL_RUNNER" ]]; then
        printf 'Local llama-server runner not found: %s\n' "$OLLAMA_LOCAL_RUNNER" >&2
        exit 2
      fi
      if [[ -z "$OLLAMA_MODEL_PATH" || ! -f "$OLLAMA_MODEL_PATH" ]]; then
        printf 'OLLAMA_MODEL_PATH must point to a GGUF model blob for direct runner mode.\n' >&2
        exit 2
      fi
      runner_env=( "GGML_VK_VISIBLE_DEVICES=$GGML_VK_VISIBLE_DEVICES" )
      if [[ -n "$OLLAMA_CUDA_VISIBLE_DEVICES" ]]; then
        runner_env+=( "CUDA_VISIBLE_DEVICES=$OLLAMA_CUDA_VISIBLE_DEVICES" )
      fi
      nohup env "${runner_env[@]}" "$OLLAMA_LOCAL_RUNNER" \
        --model "$OLLAMA_MODEL_PATH" \
        --host "${OLLAMA_LOCAL_HOST%:*}" --port "${OLLAMA_LOCAL_HOST##*:}" \
        --no-webui --offline -c "$OLLAMA_CONTEXT_LENGTH" -np 1 \
        --log-verbosity 4 --no-log-prefix --no-log-timestamps --no-jinja \
        --chat-template chatml --load-mode "${OLLAMA_LOAD_MODE:-none}" \
        --flash-attn "${OLLAMA_FLASH_ATTENTION:-auto}" -b 512 -ub 512 \
        --split-mode none --main-gpu 0 --context-shift --keep 4 \
        >"$OLLAMA_LOCAL_LOG" 2>&1 < /dev/null &
      printf '%s\n' "$!" > "$OLLAMA_LOCAL_PID"
      for _ in $(seq 1 120); do
        if curl -fsS --max-time 3 "$ollama_url" >/dev/null 2>&1; then
          break
        fi
        sleep 1
      done
      if ! curl -fsS --max-time 3 "$ollama_url" >/dev/null 2>&1; then
        printf 'Local direct llama-server failed to start; see %s\n' "$OLLAMA_LOCAL_LOG" >&2
        exit 2
      fi
      printf 'Started local direct llama-server at %s; log: %s\n' "$OLLAMA_LOCAL_HOST" "$OLLAMA_LOCAL_LOG"
    fi
  elif ! curl -fsS --max-time 3 "http://${OLLAMA_LOCAL_HOST}/api/tags" >/dev/null 2>&1; then
    if [[ ! -x "$OLLAMA_LOCAL_BIN" ]]; then
      printf 'Local Ollama binary not found: %s\n' "$OLLAMA_LOCAL_BIN" >&2
      exit 2
    fi
    if [[ -n "$OLLAMA_CUDA_VISIBLE_DEVICES" ]]; then
      nohup env OLLAMA_HOST="$OLLAMA_LOCAL_HOST" OLLAMA_MODELS="$OLLAMA_MODELS" \
        OLLAMA_CONTEXT_LENGTH="$OLLAMA_CONTEXT_LENGTH" OLLAMA_NUM_PARALLEL=1 \
        OLLAMA_LLM_LIBRARY="$OLLAMA_LLM_LIBRARY" GGML_VK_VISIBLE_DEVICES="$GGML_VK_VISIBLE_DEVICES" \
        CUDA_VISIBLE_DEVICES="$OLLAMA_CUDA_VISIBLE_DEVICES" \
        "$OLLAMA_LOCAL_BIN" serve >"$OLLAMA_LOCAL_LOG" 2>&1 < /dev/null &
    else
      nohup env OLLAMA_HOST="$OLLAMA_LOCAL_HOST" OLLAMA_MODELS="$OLLAMA_MODELS" \
        OLLAMA_CONTEXT_LENGTH="$OLLAMA_CONTEXT_LENGTH" OLLAMA_NUM_PARALLEL=1 \
        OLLAMA_LLM_LIBRARY="$OLLAMA_LLM_LIBRARY" GGML_VK_VISIBLE_DEVICES="$GGML_VK_VISIBLE_DEVICES" \
        "$OLLAMA_LOCAL_BIN" serve >"$OLLAMA_LOCAL_LOG" 2>&1 < /dev/null &
    fi
    printf '%s\n' "$!" > "$OLLAMA_LOCAL_PID"
    for _ in $(seq 1 60); do
      if curl -fsS --max-time 3 "http://${OLLAMA_LOCAL_HOST}/api/tags" >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done
    if ! curl -fsS --max-time 3 "http://${OLLAMA_LOCAL_HOST}/api/tags" >/dev/null 2>&1; then
      printf 'Local Ollama failed to start; see %s\n' "$OLLAMA_LOCAL_LOG" >&2
      exit 2
    fi
    printf 'Started local Ollama at %s; log: %s\n' "$OLLAMA_LOCAL_HOST" "$OLLAMA_LOCAL_LOG"
  fi
fi

if [[ "${ALPHA_AGENT_RENEW_SEED_ON_START:-true}" =~ ^(1|true|yes|on)$ ]]; then
  if command -v shuf >/dev/null 2>&1; then
    ALPHA_AGENT_RANDOM_SEED="$(shuf -i 1-2147483646 -n 1)"
  else
    ALPHA_AGENT_RANDOM_SEED="$(( $(date +%s) ^ $$ ))"
  fi
  export ALPHA_AGENT_RANDOM_SEED
fi

cd "$APP_DIR"
WORKDIR="${ALPHA_AGENT_WORKDIR:-.alpha_agent}"
if [[ "$WORKDIR" == "." || "$WORKDIR" == "/" || "$WORKDIR" == "$APP_DIR" ]]; then
  printf 'Refusing to archive unsafe ALPHA_AGENT_WORKDIR=%s\n' "$WORKDIR" >&2
  exit 2
fi
archived_workdir=""
if [[ "${ALPHA_AGENT_ARCHIVE_WORKDIR_ON_START:-true}" =~ ^(1|true|yes|on)$ ]] && [[ -e "$WORKDIR" ]]; then
  archive_path="${WORKDIR}.archive.$(date +%Y%m%d_%H%M%S)_$$"
  mv -- "$WORKDIR" "$archive_path"
  archived_workdir="$archive_path"
  printf 'Archived previous workdir: %s\n' "$archive_path"
fi
mkdir -p -- "$WORKDIR"
if [[ "${ALPHA_AGENT_PRESERVE_SUBMISSIONS_ON_START:-true}" =~ ^(1|true|yes|on)$ ]] \
  && [[ -n "$archived_workdir" ]] \
  && [[ -f "$archived_workdir/submissions.jsonl" ]]; then
  cp -p -- "$archived_workdir/submissions.jsonl" "$WORKDIR/submissions.jsonl"
  printf 'Preserved submission de-duplication history in: %s/submissions.jsonl\n' "$WORKDIR"
fi
nohup "$APP_DIR/.venv/bin/python" "$APP_DIR/alpha_research_agent.py" \
  --pretty run \
  --budget "${ALPHA_AGENT_BUDGET:-24}" \
  --max-iterations "${ALPHA_AGENT_MAX_ITERATIONS:-12}" \
  --random-seed "${ALPHA_AGENT_RANDOM_SEED:-7}" \
  --submission-mode "${ALPHA_AGENT_SUBMISSION_MODE:-disabled}" \
  > "$LOG_FILE" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$PID_FILE"
printf 'Started alpha agent with PID %s; seed=%s; log: %s\n' "$(cat "$PID_FILE")" "${ALPHA_AGENT_RANDOM_SEED:-7}" "$LOG_FILE"
