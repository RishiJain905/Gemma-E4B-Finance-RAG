"""src/middleware/graph_api.py
Read-only localhost API for bounded query-trace graph snapshots and SSE deltas.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from .graph_observer import TraceHub
from .corpus_graph import CorpusGraph, CorpusRevisionChanged
from .graph_models import (
    CorpusFacetFilters,
    CorpusGraphResponse,
    GraphHealthResponse,
    GraphTraceSnapshot,
    GraphTraceSummary,
)

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


def _csv(value: Optional[Any]) -> Optional[list[str]]:
    """Parse a comma-separated explorer filter without accepting raw JSON."""
    if value is None:
        return None
    values = value if isinstance(value, list) else [value]
    return [
        item.strip()
        for value_item in values
        for item in str(value_item).split(",")
        if item.strip()
    ]


def create_graph_router(
    get_hub: Callable[[], Optional[TraceHub]],
    is_enabled: Callable[[], bool],
    get_store: Optional[Callable[[], Any]] = None,
    get_config: Optional[Callable[[], Any]] = None,
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

    corpus_projector: Optional[CorpusGraph] = None
    corpus_store: Any = object()

    def enabled_corpus() -> CorpusGraph:
        """Return the Store-backed corpus projector when the local graph is on."""
        nonlocal corpus_projector, corpus_store
        if not is_enabled() or get_store is None:
            raise HTTPException(status_code=404, detail="Graph observer disabled")
        current_store = get_store()
        if current_store is None:
            raise HTTPException(status_code=404, detail="Graph observer disabled")
        if corpus_projector is None or corpus_store is not current_store:
            cfg = get_config() if get_config is not None else None
            corpus_projector = CorpusGraph(
                current_store,
                page_limit=int(getattr(cfg, "corpus_page_limit", 100)),
                default_page_limit=int(getattr(cfg, "corpus_default_page_limit", 50)),
                element_limit=int(getattr(cfg, "corpus_element_limit", 2000)),
                visible_node_target=int(getattr(cfg, "corpus_visible_node_target", 450)),
                overview_ttl_s=float(getattr(cfg, "corpus_overview_cache_ttl_s", 2.0)),
                id_ttl_s=float(getattr(cfg, "corpus_opaque_id_ttl_s", 300.0)),
                excerpt_bytes=int(getattr(cfg, "corpus_inspector_excerpt_bytes", 1000)),
                metadata_bytes=int(getattr(cfg, "corpus_inspector_metadata_bytes", 4000)),
            )
            corpus_store = current_store
        return corpus_projector

    def corpus_limit(projector: CorpusGraph, limit: Optional[int]) -> int:
        """Use the configured page default and map invalid caps to HTTP 422."""
        value = projector.default_page_limit if limit is None else limit
        try:
            return projector._validate_limit(value)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    def corpus_error(exc: Exception, *, missing_is_404: bool = False) -> None:
        if isinstance(exc, CorpusRevisionChanged):
            raise HTTPException(status_code=409, detail="Corpus updated; reload the explorer") from exc
        if isinstance(exc, KeyError) or missing_is_404:
            raise HTTPException(status_code=404, detail="Corpus node or filing not found") from exc
        raise HTTPException(status_code=422, detail=str(exc)) from exc

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

    @router.get("/corpus/overview", response_model=CorpusGraphResponse)
    def corpus_overview() -> dict:
        """Return cached bounded corpus coverage from authoritative Store reads."""
        try:
            return enabled_corpus().overview()
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 - explorer reads fail as API errors
            corpus_error(exc)
        return {}  # pragma: no cover - corpus_error always raises

    @router.get("/corpus/search", response_model=CorpusGraphResponse)
    def corpus_search(
        q: str = Query("", max_length=256),
        kinds: Optional[list[str]] = Query(None, max_length=256),
        sources: Optional[list[str]] = Query(None, max_length=256),
        ticker: Optional[str] = Query(None, max_length=32),
        facets: CorpusFacetFilters = Depends(),
        limit: Optional[int] = Query(None),
        cursor: Optional[str] = Query(None, max_length=2_048),
    ) -> dict:
        """Search labels/metadata only; this route never invokes embeddings.

        Server-side facet filters narrow the same authoritative metadata; a
        filter change resets paging by changing the opaque cursor scope.
        """
        projector = enabled_corpus()
        page_limit = corpus_limit(projector, limit)
        try:
            return projector.search(
                q=q, kinds=_csv(kinds), sources=_csv(sources), ticker=ticker,
                filters=facets.as_filters(), limit=page_limit, cursor=cursor,
            )
        except (CorpusRevisionChanged, ValueError) as exc:
            corpus_error(exc)
        return {}  # pragma: no cover

    @router.get("/corpus/facets", response_model=CorpusGraphResponse)
    def corpus_facets(
        facets: CorpusFacetFilters = Depends(),
        dimensions: Optional[list[str]] = Query(None, max_length=256),
    ) -> dict:
        """Return bounded facet counts for the current filter set (cached)."""
        try:
            return enabled_corpus().facets(
                filters=facets.as_filters(), dimensions=_csv(dimensions),
            )
        except HTTPException:
            raise
        except (CorpusRevisionChanged, ValueError) as exc:
            corpus_error(exc)
        return {}  # pragma: no cover

    @router.get("/corpus/groups", response_model=CorpusGraphResponse)
    def corpus_groups(
        group_by: str = Query("source_category", max_length=32),
        facets: CorpusFacetFilters = Depends(),
        limit: Optional[int] = Query(None),
        cursor: Optional[str] = Query(None, max_length=2_048),
    ) -> dict:
        """Return one paged aggregate level as drillable, counted nodes."""
        projector = enabled_corpus()
        page_limit = corpus_limit(projector, limit)
        try:
            return projector.groups(
                group_by, filters=facets.as_filters(), limit=page_limit,
                cursor=cursor,
            )
        except (CorpusRevisionChanged, ValueError) as exc:
            corpus_error(exc)
        return {}  # pragma: no cover

    @router.get("/corpus/items/{node_id}", response_model=CorpusGraphResponse)
    def corpus_item(node_id: str) -> dict:
        """Return one bounded corpus-item/event detail with safe provenance."""
        projector = enabled_corpus()
        try:
            result = projector.item_detail(node_id)
        except ValueError as exc:
            corpus_error(exc, missing_is_404=True)
        if result is None:
            raise HTTPException(status_code=404, detail="Corpus item not found")
        return result

    @router.get("/corpus/nodes/{node_id}", response_model=CorpusGraphResponse)
    def corpus_node(node_id: str) -> dict:
        """Return one bounded node detail and at most one safe excerpt."""
        projector = enabled_corpus()
        try:
            result = projector.detail(node_id)
        except ValueError as exc:
            corpus_error(exc, missing_is_404=True)
        if result is None:
            raise HTTPException(status_code=404, detail="Corpus node not found")
        return result

    @router.get("/corpus/nodes/{node_id}/neighbors", response_model=CorpusGraphResponse)
    def corpus_neighbors(
        node_id: str,
        cursor: Optional[str] = Query(None, max_length=2_048),
        limit: Optional[int] = Query(None),
        relations: Optional[list[str]] = Query(None, max_length=512),
    ) -> dict:
        """Expand one corpus node by one bounded revision-aware page."""
        projector = enabled_corpus()
        page_limit = corpus_limit(projector, limit)
        try:
            return projector.neighbors(
                node_id, cursor=cursor, limit=page_limit, relations=_csv(relations),
            )
        except (CorpusRevisionChanged, ValueError) as exc:
            corpus_error(exc, missing_is_404=True)
        return {}  # pragma: no cover

    @router.get("/corpus/filings/{accession}/sections", response_model=CorpusGraphResponse)
    def corpus_filing_sections(
        accession: str,
        cursor: Optional[str] = Query(None, max_length=2_048),
        limit: Optional[int] = Query(None),
    ) -> dict:
        """Return bounded section-family neighbors for one stored filing."""
        projector = enabled_corpus()
        page_limit = corpus_limit(projector, limit)
        try:
            return projector.filing_sections(
                accession, cursor=cursor, limit=page_limit,
            )
        except (CorpusRevisionChanged, KeyError, ValueError) as exc:
            corpus_error(exc, missing_is_404=isinstance(exc, KeyError))
        return {}  # pragma: no cover

    @router.get("/corpus/aggregates", response_model=CorpusGraphResponse)
    def corpus_aggregates(
        group_by: str = Query("source_category", max_length=32),
        source_category: Optional[str] = Query(None, max_length=64),
        source: Optional[str] = Query(None, max_length=64),
        item_type: Optional[str] = Query(None, max_length=64),
        security: Optional[str] = Query(None, max_length=64),
        year: Optional[str] = Query(None, max_length=8),
        month: Optional[str] = Query(None, max_length=8),
        indexing_state: Optional[str] = Query(None, max_length=32),
        limit: Optional[int] = Query(None),
        cursor: Optional[str] = Query(None, max_length=2_048),
    ) -> dict:
        """Return bounded authoritative facet counts; never scans Chroma bodies."""
        projector = enabled_corpus()
        page_limit = corpus_limit(projector, limit)
        try:
            return projector.aggregates(
                group_by,
                filters={
                    "source_category": source_category, "source": source,
                    "item_type": item_type, "security": security,
                    "year": year, "month": month, "indexing_state": indexing_state,
                },
                limit=page_limit, cursor=cursor,
            )
        except (CorpusRevisionChanged, ValueError) as exc:
            corpus_error(exc)
        return {}  # pragma: no cover

    @router.get("/corpus/refresh-status", response_model=CorpusGraphResponse)
    def corpus_refresh_status() -> dict:
        """Return persisted freshness/scheduler state; never refresh or heartbeat."""
        try:
            return enabled_corpus().refresh_status()
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            corpus_error(exc)
        return {}  # pragma: no cover

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
