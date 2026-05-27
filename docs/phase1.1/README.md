# Phase 1.1 — Model Serving

**Goal:** Get TraceAlchemy-Gemma-4-E4B-Finance-IT running reliably via `llama-server` with a verified, repeatable setup.

**Status:** ✅ Pre-verified — model is tested and operational. This document records the canonical setup for reproducibility and future debugging.

---

## Task 1.1.1 — Verify llama.cpp Build

Check that your local `llama.cpp` build supports the features required for this project.

```bash
# Check version and build flags
llama-server --version

# Verify required capabilities:
#   - CUDA / Metal / GPU offloading (--n-gpu-layers)
#   - Server mode (llama-server binary exists)
#   - GGUF format support
#   - At least 32K context support
```

**Acceptance criteria:**
- `llama-server` binary exists and runs
- GPU offloading is enabled for your hardware (Mac Mini M4 = Metal)

---

## Task 1.1.2 — Place the GGUF Model

Ensure the TraceAlchemy GGUF is in a stable, known location.

```bash
# Recommended location
mkdir -p ~/models/tracealchemy
mv /path/to/downloaded/TraceAlchemy-Gemma-4-E4B-Finance-IT.gguf ~/models/tracealchemy/
```

**Naming convention:** Use the exact filename from Hugging Face without renaming, so configs and scripts reference a consistent path.

**Acceptance criteria:**
- Model file exists at a fixed path: `~/models/tracealchemy/TraceAlchemy-Gemma-4-E4B-Finance-IT.gguf`
- File size matches expected (~9.6 GB for the full quant)

---

## Task 1.1.3 — Start llama-server with Optimal Config

Launch `llama-server` with parameters tuned for this model and your hardware.

```bash
llama-server \
  -m ~/models/tracealchemy/TraceAlchemy-Gemma-4-E4B-Finance-IT.gguf \
  --port 8080 \
  --host 127.0.0.1 \
  --ctx-size 32768 \
  --n-gpu-layers 99 \
  --rope-scaling none \
  --flash-attn 1 \
  --parallel 1 \
  --cont-batching 1
```

**Parameter rationale:**

| Flag | Value | Why |
|------|-------|-----|
| `--ctx-size` | 32768 | Balances 128K native context with memory. 32K covers full SEC filings comfortably |
| `--n-gpu-layers` | 99 | Offloads all layers to GPU (M4 unified memory). Adjust down if VRAM pressure |
| `--flash-attn` | 1 | Enables memory-efficient attention. Required for long context performance |
| `--cont-batching` | 1 | Allows continuous batching — useful when middleware sends multiple queries |
| `--parallel` | 1 | One concurrent request is fine for single-user use. Increase if needed |

**Acceptance criteria:**
- Server starts without errors
- Logs show: model loaded, GPU layers offloaded, context size confirmed
- Server binds to `127.0.0.1:8080`

---

## Task 1.1.4 — Health Check & Test Completion

Verify the server responds correctly with a simple test prompt.

```bash
# Test basic completion
curl http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the P/E ratio and how is it calculated?",
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Test chat completion (for middleware integration)
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tracealchemy",
    "messages": [
      {"role": "system", "content": "You are a financial analyst."},
      {"role": "user", "content": "Explain the difference between trailing P/E and forward P/E."}
    ],
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

**Acceptance criteria:**
- Both endpoints return valid JSON with a `choices` array containing `text` or `message.content`
- Response time is reasonable (< 30s for short prompts)
- Model output is coherent financial reasoning (not gibberish)
- Token generation is smooth (no stuttering or repetition loops)

---

## Task 1.1.5 — Create a Server Wrapper Script

Build a simple shell script to start/stop the server consistently, so you don't have to remember the flags.

```bash
#!/usr/bin/env bash
# scripts/serve_model.sh
# Start TraceAlchemy llama-server

MODEL_PATH="$HOME/models/tracealchemy/TraceAlchemy-Gemma-4-E4B-Finance-IT.gguf"
PORT=${1:-8080}

llama-server \
  -m "$MODEL_PATH" \
  --port "$PORT" \
  --host 127.0.0.1 \
  --ctx-size 32768 \
  --n-gpu-layers 99 \
  --flash-attn 1 \
  --parallel 1 \
  --cont-batching 1
```

```bash
# Make it executable
chmod +x scripts/serve_model.sh

# Usage
./scripts/serve_model.sh        # starts on :8080
./scripts/serve_model.sh 8081   # starts on :8081
```

---

## Task 1.1.6 — Document Model Parameters

Record the exact model configuration for reference when building the middleware (Phase 1.5). The middleware needs to know:

```
Model ID:       tracealchemy
Endpoint:       http://127.0.0.1:8080/v1/chat/completions
Context window: 32768 tokens
Max output:     2048 tokens (default, adjustable per query)
Temperature:    0.7 (default, adjustable)
Top-P:          0.95
Stop tokens:    none (model handles its own termination)
Supported modes: chat, completion, embedding
```

---

## Task 1.1.7 — Verify Embedding Endpoint (Prerequisite for ChromaDB)

ChromaDB needs an embedding function. Test that `llama-server`'s embedding endpoint works.

```bash
curl http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "NVIDIA reported record datacenter revenue of $26 billion in Q1 2026",
    "model": "tracealchemy"
  }'
```

**Acceptance criteria:**
- Response contains an embedding vector (list of floats)
- Vector dimension is consistent (typically 4096 or 2048 depending on model config)
- This will be used in Phase 1.2 when we connect ChromaDB

---

## 1.1 Completion Checklist

- [x] `llama-server` binary verified
- [x] GGUF model placed at stable path
- [x] Server starts with optimal params
- [x] Completion endpoint returns correct financial responses
- [x] Chat endpoint works for middleware integration
- [x] Wrapper script created
- [x] Model parameters documented
- [x] Embedding endpoint verified

**Phase 1.1 is complete when all acceptance criteria above are met.** Move to **Phase 1.2 — Storage Foundation**.
