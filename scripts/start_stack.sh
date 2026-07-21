#!/bin/bash
# start_stack.sh — Bring up the full Gemma-E4B-Finance-RAG stack
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

mkdir -p logs

# 1. Check prerequisites
log_info "Checking prerequisites..."
command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1 || { log_error "python is required"; exit 1; }
PYTHON="$(command -v python3 || command -v python)"
command -v llama-server >/dev/null 2>&1 || log_warn "llama-server not found in PATH"

# 2. Activate virtual environment
if [ -d ".venv" ]; then
    # shellcheck disable=SC1091
    source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate 2>/dev/null || log_warn "Could not activate .venv"
elif [ -d "venv" ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
else
    log_warn "No virtual environment found, using system Python"
fi

# 3. Verify dependencies
log_info "Verifying dependencies..."
pip check 2>/dev/null || log_warn "Dependencies may have conflicts"

# 4. Check config files exist
for cfg in configs/storage.yaml configs/middleware.yaml configs/watchlist.yaml; do
    if [ ! -f "$cfg" ]; then
        log_warn "Missing config: $cfg"
    fi
done

# 5. Start llama-server (if not already running)
if ! curl -s http://127.0.0.1:8087/health >/dev/null 2>&1; then
    log_info "Starting llama-server..."
    MODEL_PATH="${TRACEALCHEMY_MODEL:-./models/TraceAlchemy-Gemma-4-E4B-Finance-IT.gguf}"
    if [ -f "$MODEL_PATH" ]; then
        llama-server -m "$MODEL_PATH" \
            --port 8087 \
            --ctx-size 32768 \
            --n-gpu-layers 99 \
            --jinja \
            --embeddings \
            --pooling mean \
            > logs/llama-server.log 2>&1 &
        LLAMA_PID=$!
        log_info "llama-server started (PID: $LLAMA_PID)"
        sleep 5
    else
        log_warn "Model not found at $MODEL_PATH — skipping llama-server (start it manually)"
    fi
else
    log_info "llama-server already running"
fi

# 6. Start FastAPI middleware
log_info "Starting FastAPI middleware..."
uvicorn src.middleware.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info \
    > logs/middleware.log 2>&1 &
MIDDLEWARE_PID=$!
log_info "Middleware started (PID: $MIDDLEWARE_PID)"

# 7. Verify stack is up
sleep 2
log_info "Verifying stack..."
curl -s http://127.0.0.1:8000/health >/dev/null 2>&1 && \
    log_info "✅ Middleware health check passed" || \
    log_warn "Middleware health check failed"

echo ""
log_info "Stack is running!"
echo "  Middleware:  http://127.0.0.1:8000"
echo "  Docs:        http://127.0.0.1:8000/docs"
echo "  Model:       http://127.0.0.1:8087"
echo ""
echo "To stop: ./scripts/stop_stack.sh"
