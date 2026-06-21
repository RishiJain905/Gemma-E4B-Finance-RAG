"""
scripts/chat.py
Interactive terminal chat client for the Gemma-E4B-Finance-RAG middleware.

One centralized loop:
  - Auto-starts the FastAPI middleware (:8000) if it isn't already running.
  - Sends your questions to POST /query and prints grounded answers + metadata.
  - Slash commands to run ingestion jobs, refresh a ticker, and check health.

Usage:
    python scripts/chat.py                # start middleware (if needed) + chat
    python scripts/chat.py --no-start     # don't auto-start; just connect
    python scripts/chat.py --port 8000

Prerequisite: the model server (llama-server) on :8087 with --embeddings
enabled. The script warns if it is unreachable (answers degrade gracefully).
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_HEALTH_URL = "http://127.0.0.1:8087/health"
SCHEDULER_MODES = {"daily", "hourly", "weekly", "all", "status"}

# Enable ANSI escape processing on Windows consoles.
if os.name == "nt":
    os.system("")


class C:
    R = "\033[0m"; B = "\033[1m"; DIM = "\033[2m"
    CY = "\033[36m"; GR = "\033[32m"; YE = "\033[33m"; RE = "\033[31m"; MA = "\033[35m"


def col(s: str, c: str) -> str:
    return f"{c}{s}{C.R}"


# ── Middleware lifecycle ───────────────────────────────

def middleware_up(base: str) -> bool:
    try:
        return httpx.get(f"{base}/health", timeout=3).status_code == 200
    except Exception:
        return False


def start_middleware(port: int):
    """Launch uvicorn as a subprocess; return the Popen handle once healthy."""
    PROJECT_ROOT.joinpath("logs").mkdir(exist_ok=True)
    log = open(PROJECT_ROOT / "logs" / "middleware.log", "a", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.middleware.app:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=str(PROJECT_ROOT), stdout=log, stderr=log,
    )
    base = f"http://127.0.0.1:{port}"
    print(col(f"Starting middleware on :{port} ...", C.DIM), end="", flush=True)
    for _ in range(40):  # ~40s
        if proc.poll() is not None:
            print(col(" failed (process exited). See logs/middleware.log", C.RE))
            return None
        if middleware_up(base):
            print(col(" ready.", C.GR))
            return proc
        print(col(".", C.DIM), end="", flush=True)
        time.sleep(1)
    print(col(" timed out waiting for /health.", C.RE))
    return proc


def check_model():
    try:
        ok = httpx.get(MODEL_HEALTH_URL, timeout=3).status_code == 200
    except Exception:
        ok = False
    if ok:
        print(col("Model server (:8087) reachable.", C.GR))
    else:
        print(col("Model server (:8087) NOT reachable — answers will run in "
                  "degraded mode (raw retrieved data). Start llama-server with "
                  "--embeddings.", C.YE))


# ── Actions ────────────────────────────────────────────

def do_query(base: str, question: str, ticker, refresh: bool):
    payload = {"question": question, "refresh": refresh}
    if ticker:
        payload["ticker"] = ticker
    try:
        resp = httpx.post(f"{base}/query", json=payload, timeout=240)
    except Exception as e:
        print(col(f"  request failed: {e}", C.RE))
        return
    if resp.status_code != 200:
        print(col(f"  HTTP {resp.status_code}: {resp.text[:200]}", C.RE))
        return

    data = resp.json()
    answer = (data.get("answer") or "").strip()
    print()
    if answer:
        print(col(answer, C.B))
    else:
        print(col("(model returned an empty completion — the pipeline ran but "
                  "the model produced no text; see the prompt/sampling note in "
                  "docs/phase1.8/PHASE1_COMPLETION.md)", C.YE))

    meta = (f"  ticker={data.get('detected_ticker')} "
            f"intent={data.get('detected_intent')} "
            f"facts={data.get('facts_used')} docs={data.get('documents_used')} "
            f"model_available={data.get('model_available')} "
            f"latency={data.get('latency_ms')}ms")
    print(col(meta, C.DIM))

    fresh = data.get("freshness") or {}
    if fresh.get("warning"):
        print(col(f"  ⚠ {fresh['warning']}", C.YE))
    if fresh.get("refreshed_during_query"):
        print(col(f"  ↻ refreshed: {', '.join(fresh['refreshed_during_query'])}", C.CY))
    print()


def do_refresh(base: str, arg: str):
    """Bare/mode arg -> run the scheduler; a ticker -> hit /refresh/{ticker}."""
    arg = (arg or "").strip()
    mode = arg.lower() if arg else "all"

    if not arg or mode in SCHEDULER_MODES:
        run_scheduler(mode)
        return

    ticker = arg.upper()
    print(col(f"Refreshing {ticker} via API ...", C.DIM))
    try:
        resp = httpx.post(f"{base}/refresh/{ticker}", json={}, timeout=600)
        data = resp.json()
        print(col(f"  refreshed: {data.get('refreshed')}", C.GR))
        if data.get("errors"):
            print(col(f"  errors: {data.get('errors')}", C.YE))
        print(col(f"  duration: {data.get('duration_s')}s", C.DIM))
    except Exception as e:
        print(col(f"  refresh failed: {e}", C.RE))


def run_scheduler(mode: str):
    """Run `python -m src.scheduler <mode>` and stream its output."""
    cmd = [sys.executable, "-m", "src.scheduler", mode]
    if mode != "status":
        cmd.append("--force")
    print(col(f"Running scheduler '{mode}' (this can take a few minutes) ...", C.MA))
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            print(col("  " + line.rstrip(), C.DIM))
        proc.wait()
        print(col(f"scheduler '{mode}' finished (exit {proc.returncode}).", C.GR))
    except Exception as e:
        print(col(f"  scheduler failed: {e}", C.RE))


def do_health(base: str):
    try:
        data = httpx.get(f"{base}/health", timeout=10).json()
    except Exception as e:
        print(col(f"  health request failed: {e}", C.RE))
        return
    print(col(f"  status={data.get('status')} model_available={data.get('model_available')}", C.GR))
    fresh = data.get("freshness") or {}
    if fresh:
        print(col("  freshness: " + ", ".join(f"{k}={v}" for k, v in fresh.items()), C.DIM))
    storage = data.get("storage") or {}
    print(col(f"  storage: sqlite={storage.get('sqlite')} chroma={storage.get('chroma')} "
              f"docs={storage.get('chroma_doc_count')}", C.DIM))


HELP = f"""
{C.B}Commands{C.R}
  {C.CY}<just type a question>{C.R}   ask the RAG (POST /query)
  {C.CY}/refresh{C.R}                 run ALL ingestion jobs (scheduler all --force)
  {C.CY}/refresh daily|hourly|weekly|all|status{C.R}   run that scheduler mode
  {C.CY}/refresh NVDA{C.R}            refresh one ticker via the API
  {C.CY}/ticker NVDA{C.R}             pin a ticker override for following questions
  {C.CY}/ticker clear{C.R}            clear the pinned ticker
  {C.CY}/autorefresh on|off{C.R}      toggle auto-refresh of stale data per query
  {C.CY}/health{C.R}                  show middleware health summary
  {C.CY}/help{C.R}                    show this help
  {C.CY}/quit{C.R} or {C.CY}/exit{C.R}            leave (stops the middleware if this script started it)
"""


# ── REPL ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Interactive RAG chat client")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-start", action="store_true",
                        help="Do not auto-start the middleware")
    args = parser.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    started_proc = None

    print(col("\n=== Gemma-E4B-Finance-RAG chat ===", C.B))
    check_model()

    if middleware_up(base):
        print(col(f"Middleware already running on :{args.port}.", C.GR))
    elif args.no_start:
        print(col(f"Middleware not running on :{args.port} and --no-start set. "
                  f"Start it first.", C.RE))
        return
    else:
        started_proc = start_middleware(args.port)
        if not middleware_up(base):
            print(col("Could not reach the middleware; exiting.", C.RE))
            if started_proc:
                started_proc.terminate()
            return

    ticker = None
    autorefresh = False
    print(HELP)

    try:
        while True:
            prefix = f"[{ticker}]" if ticker else ""
            try:
                line = input(col(f"\n{prefix}you> ", C.GR)).strip()
            except EOFError:
                break
            if not line:
                continue

            if line.startswith("/"):
                parts = line[1:].split(maxsplit=1)
                cmd = parts[0].lower()
                rest = parts[1] if len(parts) > 1 else ""

                if cmd in ("quit", "exit", "q"):
                    break
                elif cmd == "help":
                    print(HELP)
                elif cmd == "health":
                    do_health(base)
                elif cmd == "refresh":
                    do_refresh(base, rest)
                elif cmd == "ticker":
                    if rest.lower() in ("", "clear", "none"):
                        ticker = None
                        print(col("  ticker override cleared.", C.DIM))
                    else:
                        ticker = rest.upper()
                        print(col(f"  ticker pinned to {ticker}.", C.DIM))
                elif cmd == "autorefresh":
                    autorefresh = rest.lower() in ("on", "true", "1", "yes")
                    print(col(f"  autorefresh = {autorefresh}", C.DIM))
                else:
                    print(col(f"  unknown command: /{cmd} (try /help)", C.YE))
                continue

            do_query(base, line, ticker, autorefresh)
    except KeyboardInterrupt:
        print()
    finally:
        if started_proc is not None:
            print(col("Stopping middleware (started by this script) ...", C.DIM))
            started_proc.terminate()
            try:
                started_proc.wait(timeout=10)
            except Exception:
                started_proc.kill()
        print(col("Bye.", C.B))


if __name__ == "__main__":
    main()
