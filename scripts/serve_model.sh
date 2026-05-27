#!/usr/bin/env bash
# scripts/serve_model.sh — TraceAlchemy llama-server Wrapper
# ⚠ NOTE: This is a reference script for POSIX systems.
# See serve_model.ps1 for the Windows/AMD RDNA3 PowerShell version.

MODEL_PATH="$HOME/models/tracealchemy/TraceAlchemy-Gemma-4-E4B-Finance-IT.gguf"
PORT="${2:-8080}"
LOG_DIR="$HOME/models/tracealchemy/logs"
PID_FILE="$LOG_DIR/server.pid"
LOG_FILE="$LOG_DIR/server.log"

ACTION="${1:-start}"

ensure_log_dir() {
    mkdir -p "$LOG_DIR"
}

start_server() {
    ensure_log_dir
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "❌ Server is already running (PID $(cat "$PID_FILE"))."
        exit 1
    fi

    echo "🚀 Starting TraceAlchemy on port $PORT..."
    echo "ℹ️  Windows + AMD RDNA3? Use serve_model.ps1 instead."

    nohup llama-server \
        -m "$MODEL_PATH" \
        --port "$PORT" \
        --host 127.0.0.1 \
        --ctx-size 32768 \
        --n-gpu-layers 99 \
        --rope-scaling none \
        --flash-attn 1 \
        --cont-batching 1 \
        --embeddings \
        --pooling mean \
        > "$LOG_FILE" 2>&1 &

    PID=$!
    echo $PID > "$PID_FILE"
    echo "✅ Server started (PID $PID) — http://127.0.0.1:$PORT"

    sleep 2
    if kill -0 $PID 2>/dev/null; then
        echo "   Status: running"
    else
        echo "   ⚠️  Exited early. Check: $LOG_FILE"
        tail -5 "$LOG_FILE"
        exit 1
    fi
}

stop_server() {
    [ -f "$PID_FILE" ] && pid=$(cat "$PID_FILE") || pid=""

    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "🛑 Stopping (PID $pid)..."
        kill "$pid" 2>/dev/null
        for i in $(seq 1 10); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        kill -9 "$pid" 2>/dev/null || true
        echo "✅ Stopped."
    else
        echo "ℹ️  Not running. Cleaning up."
    fi
    rm -f "$PID_FILE"
}

status_server() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        pid=$(cat "$PID_FILE")
        echo "✅ Running (PID $pid) — http://127.0.0.1:$PORT"
        curl -sf http://127.0.0.1:"$PORT"/v1/completions \
             -H "Content-Type: application/json" \
             -d '{"prompt": "test", "max_tokens": 1}' > /dev/null 2>&1 \
          && echo "   API: responding ✅" || echo "   API: not responding ❌"
    else
        echo "❌ Not running"
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
