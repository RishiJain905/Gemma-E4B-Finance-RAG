"""Verify the TraceAlchemy model and embedding endpoints are reachable.

Run:
    python scripts/test_model_connectivity.py

Exits non-zero if either the chat-completions or embeddings endpoint fails.
"""

import sys

import httpx

LLAMA_URL = "http://127.0.0.1:8087/v1/chat/completions"
EMBED_URL = "http://127.0.0.1:8087/v1/embeddings"


def check_chat() -> bool:
    """Confirm the chat-completions endpoint returns a 200 response."""
    try:
        resp = httpx.post(
            LLAMA_URL,
            json={
                "model": "tracealchemy",
                "messages": [{"role": "user", "content": "Say 'OK' if you can hear me."}],
                "max_tokens": 10,
            },
            timeout=60,
        )
    except Exception as e:  # noqa: BLE001
        print(f"FAIL Model endpoint unreachable: {e}")
        return False

    if resp.status_code != 200:
        print(f"FAIL Model health check failed: HTTP {resp.status_code}")
        return False

    content = resp.json()["choices"][0]["message"]["content"]
    print(f"OK   Model endpoint reachable: {content!r}")
    return True


def check_embedding() -> bool:
    """Confirm the embeddings endpoint returns a valid vector."""
    try:
        resp = httpx.post(
            EMBED_URL,
            json={"model": "tracealchemy", "input": "What is NVIDIA's revenue?"},
            timeout=60,
        )
    except Exception as e:  # noqa: BLE001
        print(f"FAIL Embedding endpoint unreachable: {e}")
        return False

    if resp.status_code != 200:
        print(f"FAIL Embedding health check failed: HTTP {resp.status_code}")
        return False

    embedding = resp.json()["data"][0]["embedding"]
    print(f"OK   Embedding endpoint reachable: {len(embedding)} dimensions")
    return True


def main() -> int:
    print("=== Model connectivity check (port 8087) ===")
    chat_ok = check_chat()
    embed_ok = check_embedding()
    if chat_ok and embed_ok:
        print("\nAll model endpoints OK.")
        return 0
    print("\nOne or more model endpoints failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
