"""src/middleware/graph_api.py
Read-only localhost API for bounded query-trace graph snapshots and SSE deltas.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from .graph_observer import TraceHub
from .models import GraphHealthResponse, GraphTraceSnapshot, GraphTraceSummary

logger = logging.getLogger(__name__)

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _require_loopback(request: Request) -> None:
    """Hide the observer API from non-loopback clients."""
    host = request.client.host if request.client else ""
    if host not in _LOOPBACK_HOSTS:
        raise HTTPException(status_code=404, detail="Not found")


def _sse(event: dict) -> str:
    delta = event["delta"]
    return (
        f"id: {event['event_id']}\n"
        f"event: {delta['operation']}\n"
        f"data: {json.dumps(delta, separators=(',', ':'))}\n\n"
    )


def create_graph_router(
    get_hub: Callable[[], Optional[TraceHub]],
    is_enabled: Callable[[], bool],
) -> APIRouter:
    """Create the read-only graph router around app-owned observer state."""
    router = APIRouter(
        prefix="/graph/api",
        tags=["query-graph"],
        dependencies=[Depends(_require_loopback)],
    )

    def enabled_hub() -> TraceHub:
        hub = get_hub()
        if not is_enabled() or hub is None:
            raise HTTPException(status_code=404, detail="Graph observer disabled")
        return hub

    @router.get("/traces", response_model=list[GraphTraceSummary])
    def traces(limit: int = Query(25, ge=1, le=100)) -> list[dict]:
        return enabled_hub().list_traces(limit)

    @router.get("/traces/{query_id}", response_model=GraphTraceSnapshot)
    def trace(query_id: str) -> dict:
        snapshot = enabled_hub().snapshot(query_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="Trace not found")
        return snapshot

    @router.get("/traces/{query_id}/evidence/{evidence_id}")
    def evidence(query_id: str, evidence_id: str) -> dict:
        item = enabled_hub().evidence(query_id, evidence_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Evidence not found")
        return item

    @router.get("/health", response_model=GraphHealthResponse)
    def health() -> dict:
        return enabled_hub().health()

    @router.get("/events")
    async def events(
        last_sequence: Optional[int] = Query(None),
        once: bool = Query(False),
        last_event_id: Optional[str] = Header(None, alias="Last-Event-ID"),
    ) -> StreamingResponse:
        hub = enabled_hub()
        requested = last_sequence
        if requested is None and last_event_id is not None:
            try:
                requested = int(last_event_id)
            except ValueError:
                requested = -1
        requested = 0 if requested is None else requested
        subscriber = None if once else hub.subscribe()
        reset, replay = hub.events_since(requested)

        async def stream():
            try:
                if reset:
                    yield _sse(hub._reset_event())
                    return
                latest = requested
                for event in replay:
                    latest = max(latest, event["event_id"])
                    yield _sse(event)
                if once:
                    return
                while True:
                    if subscriber.reset_required:
                        yield _sse(hub.next_subscriber_event_nowait(subscriber))
                        return
                    try:
                        event = await asyncio.wait_for(subscriber.queue.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    if event["event_id"] <= latest:
                        continue
                    latest = event["event_id"]
                    yield _sse(event)
            finally:
                if subscriber is not None:
                    hub.unsubscribe(subscriber)

        return StreamingResponse(
            stream(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router
