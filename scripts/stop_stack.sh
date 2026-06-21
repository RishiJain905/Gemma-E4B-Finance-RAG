#!/bin/bash
# stop_stack.sh — Gracefully shut down the RAG stack
echo "Shutting down RAG stack..."
pkill -f "uvicorn src.middleware.app" 2>/dev/null && echo "✅ Middleware stopped" || echo "Middleware not running"
pkill -f "llama-server" 2>/dev/null && echo "✅ llama-server stopped" || echo "llama-server not running"
echo "Done."
