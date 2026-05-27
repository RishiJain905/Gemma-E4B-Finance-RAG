#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────
# TraceAlchemy llama-server Wrapper
# Usage: ./scripts/serve_model.sh [start|stop|restart|status] [port]
# ──────────────────────────────────────────────

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
        echo "❌ Server is already running (PID $(cat "$PID_FILE")). Use 'stop' first or 'restart'."
        exit 1
    fi

    echo "🚀 Starting TraceAlchemy on port $PORT..."

    nohup llama-server \
        -m "$MODEL_PATH" \
        --port "$PORT" \
        --host 127.0.0.1 \
        --ctx-size 32768 \
        --n-gpu-layers 99 \
        --rope-scaling none \
        --flash-attn 1 \
        --parallel 1 \
        --cont-batching 1 \
        > "$LOG_FILE" 2>&1 &

    PID=$!
    echo $PID > "$PID_FILE"

    echo "✅ Server started (PID $PID)"
    echo "   Logs: $LOG_FILE"
    echo "   API:  http://127.0.0.1:$PORT"

    sleep 2
    if kill -0 $PID 2>/dev/null; then
        echo "   Status: running"
    else
        echo "   ⚠️  Server exited prematurely. Check logs:"
        tail -5 "$LOG_FILE"
        exit 1
    fi
}

stop_server() {
    if [ ! -f "$PID_FILE" ]; then
        echo "ℹ️  No PID file found. Server may not be running."
        local pids
        pids=$(lsof -ti :"$PORT" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo "   Found process on port $PORT. Stopping..."
            kill $pids 2>/dev/null || true
            echo "✅ Stopped."
        else
            echo "   No server found on port $PORT."
        fi
        return
    fi

    local pid
    pid=$(cat "$PID_FILE")

    if kill -0 "$pid" 2>/dev/null; then
        echo "🛑 Stopping TraceAlchemy (PID $pid)..."
        kill "$pid"
        for i in $(seq 1 10); do
            if ! kill -0 "$pid" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo "   Force killing..."
            kill -9 "$pid" 2>/dev/null || true
        fi
        echo "✅ Server stopped."
    else
        echo "ℹ️  PID $pid is not running. Cleaning up PID file."
    fi

    rm -f "$PID_FILE"
}

status_server() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        local pid
        pid=$(cat "$PID_FILE")
        local uptime
        uptime=$(ps -o etime= -p "$pid" | xargs)
        echo "✅ TraceAlchemy is running"
        echo "   PID:     $pid"
        echo "   Port:    $PORT"
        echo "   Uptime:  $uptime"
        echo "   Logs:    $LOG_FILE"

        if curl -sf http://127.0.0.1:"$PORT"/v1/completions \
             -H "Content-Type: application/json" \
             -d '{"prompt": "test", "max_tokens": 1}' > /dev/null 2>&1; then
            echo "   API:     responding ✅"
        else
            echo "   API:     not responding ❌"
        fi
    else
        echo "❌ TraceAlchemy is not running"
        rm -f "$PID_FILE"
    fi
}

case "$ACTION" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        stop_server
        sleep 1
        start_server
        ;;
    status)
        status_server
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status} [port]"
        echo ""
        echo "Examples:"
        echo "  $0 start          Start on default port 8080"
        echo "  $0 start 8081     Start on port 8081"
        echo "  $0 status         Check if server is running"
        echo "  $0 stop           Stop the server"
        echo "  $0 restart        Restart the server"
        exit 1
        ;;
esac
