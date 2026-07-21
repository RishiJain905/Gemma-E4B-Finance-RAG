#!/usr/bin/env bash
# scripts/serve_model.sh — TraceAlchemy llama-server Wrapper
# Paths come from configs/model.yaml + configs/model.local.yaml
# (copy configs/model.example.yaml to model.local.yaml on first setup).
# See serve_model.ps1 for the Windows/AMD RDNA3 PowerShell version.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON="${PROJECT_DIR}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="python3"
fi

read_config() {
    local json
    if ! json="$("$PYTHON" "$SCRIPT_DIR/resolve_model_paths.py")"; then
        echo "Setup: copy configs/model.example.yaml to configs/model.local.yaml and set your paths." >&2
        exit 1
    fi
    echo "$json"
}

CFG="$(read_config)"
PORT="${2:-$(echo "$CFG" | python3 -c 'import json,sys; print(json.load(sys.stdin)["port"])')}"
LOG_DIR="${HOME}/models/tracealchemy/logs"
PID_FILE="$LOG_DIR/server.pid"
LOG_FILE="$LOG_DIR/server.log"
ACTION="${1:-start}"

SERVER_EXE="$(echo "$CFG" | python3 -c 'import json,sys; print(json.load(sys.stdin)["server_exe"])')"
MAIN_MODEL="$(echo "$CFG" | python3 -c 'import json,sys; print(json.load(sys.stdin)["main_model"])')"
DRAFT_MODEL="$(echo "$CFG" | python3 -c 'import json,sys; print(json.load(sys.stdin)["draft_model"])')"
MAX_CONTEXT="$(echo "$CFG" | python3 -c 'import json,sys; print(json.load(sys.stdin)["max_context"])')"
GPU_LAYERS="$(echo "$CFG" | python3 -c 'import json,sys; print(json.load(sys.stdin)["gpu_layers"])')"
CPU_THREADS="$(echo "$CFG" | python3 -c 'import json,sys; print(json.load(sys.stdin)["cpu_threads"])')"
SPEC_ENABLED="$(echo "$CFG" | python3 -c 'import json,sys; print(json.load(sys.stdin)["speculative_enabled"])')"
DRAFT_BLOCK="$(echo "$CFG" | python3 -c 'import json,sys; print(json.load(sys.stdin)["draft_block_size"])')"
CACHE_KEY="$(echo "$CFG" | python3 -c 'import json,sys; print(json.load(sys.stdin)["cache_type_key"])')"
CACHE_VAL="$(echo "$CFG" | python3 -c 'import json,sys; print(json.load(sys.stdin)["cache_type_value"])')"

ensure_log_dir() {
    mkdir -p "$LOG_DIR"
}

start_server() {
    ensure_log_dir
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "Server is already running (PID $(cat "$PID_FILE"))."
        exit 1
    fi

    if [[ ! -x "$SERVER_EXE" && ! -f "$SERVER_EXE" ]]; then
        echo "ERROR: llama-server not found: $SERVER_EXE" >&2
        echo "Set paths.build_dir and paths.binary in configs/model.local.yaml" >&2
        exit 1
    fi
    if [[ ! -f "$MAIN_MODEL" ]]; then
        echo "ERROR: Main model not found: $MAIN_MODEL" >&2
        exit 1
    fi

    echo "Starting TraceAlchemy on port $PORT..."
    echo "Windows + AMD RDNA3? Use serve_model.ps1 instead."

    # --jinja is REQUIRED for OpenAI-style tool calling: without it llama-server
    # rejects `tools` payloads and the middleware permanently disables tools for
    # the process (_tools_supported kill-switch).
    local -a args=(
        -m "$MAIN_MODEL"
        --host 127.0.0.1
        --port "$PORT"
        -c "$MAX_CONTEXT"
        -ngl "$GPU_LAYERS"
        -t "$CPU_THREADS"
        --jinja
        --embeddings
        --pooling mean
    )

    if [[ "$SPEC_ENABLED" == "True" || "$SPEC_ENABLED" == "true" ]]; then
        args+=(--mtp-head "$DRAFT_MODEL" --spec-type mtp --draft-block-size "$DRAFT_BLOCK")
        args+=(-ctk "$CACHE_KEY" -ctv "$CACHE_VAL")
    else
        args+=(--flash-attn 1 --cont-batching 1)
    fi

    nohup "$SERVER_EXE" "${args[@]}" > "$LOG_FILE" 2>&1 &

    PID=$!
    echo $PID > "$PID_FILE"
    echo "Server started (PID $PID) — http://127.0.0.1:$PORT"

    sleep 2
    if kill -0 $PID 2>/dev/null; then
        echo "   Status: running"
    else
        echo "   Exited early. Check: $LOG_FILE"
        tail -5 "$LOG_FILE"
        exit 1
    fi
}

stop_server() {
    [[ -f "$PID_FILE" ]] && pid=$(cat "$PID_FILE") || pid=""

    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        echo "Stopping (PID $pid)..."
        kill "$pid" 2>/dev/null
        for _ in $(seq 1 10); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        kill -9 "$pid" 2>/dev/null || true
        echo "Stopped."
    else
        echo "Not running. Cleaning up."
    fi
    rm -f "$PID_FILE"
}

status_server() {
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        pid=$(cat "$PID_FILE")
        echo "Running (PID $pid) — http://127.0.0.1:$PORT"
        curl -sf "http://127.0.0.1:${PORT}/v1/completions" \
             -H "Content-Type: application/json" \
             -d '{"prompt": "test", "max_tokens": 1}' > /dev/null 2>&1 \
          && echo "   API: responding" || echo "   API: not responding"
    else
        echo "Not running"
        rm -f "$PID_FILE"
    fi
}

case "${ACTION}" in
    start)  start_server ;;
    stop)   stop_server ;;
    restart) stop_server; sleep 1; start_server ;;
    status) status_server ;;
    *)
        echo "Usage: $0 {start|stop|restart|status} [port]"
        echo "Windows users: use serve_model.ps1"
        exit 1
        ;;
esac
