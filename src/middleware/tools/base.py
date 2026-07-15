"""
src/middleware/tools/base.py
Tool registry and guarded function-call dispatch for middleware model calls.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable
    write: bool = False


@dataclass
class ToolContext:
    allow_write: bool
    max_refreshes: int
    refresh_count: int = 0


REGISTRY: dict[str, Tool] = {}


def tool_result_count(result: object) -> Optional[int]:
    """Return only the bounded row/item count from an executed tool result."""
    if not isinstance(result, dict) or result.get("error"):
        return None
    rows = result.get("results")
    if isinstance(rows, list):
        return len(rows)
    for key in ("fundamentals", "estimates", "price_targets", "guidance"):
        value = result.get(key)
        if isinstance(value, dict):
            return len(value)
    articles = result.get("article_count")
    return articles if isinstance(articles, int) else None


def _trace_tool_started(name: str, subquery_id: Optional[str]) -> float:
    """Emit an actual dispatch start through the request's shared emitter."""
    from ..stream_events import current_emitter

    emitter = current_emitter()
    if emitter is not None:
        emitter.tool_started(name, subquery_id=subquery_id)
    return perf_counter()


def _trace_tool_completed(
    name: str, result: object, started_at: float, subquery_id: Optional[str]
) -> None:
    """Emit completion from the executed result without exposing its body."""
    from ..stream_events import current_emitter

    emitter = current_emitter()
    if emitter is not None:
        failed = isinstance(result, dict) and bool(result.get("error"))
        emitter.tool_completed(
            name,
            "error" if failed else "ok",
            count=tool_result_count(result),
            elapsed_ms=(perf_counter() - started_at) * 1000,
            subquery_id=subquery_id,
        )


def register(tool: Tool):
    REGISTRY[tool.name] = tool


def openai_schema() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in REGISTRY.values()
    ]


def validate_args(schema: dict, args: dict) -> Optional[str]:
    properties = schema.get("properties", {}) or {}
    for key in schema.get("required", []) or []:
        if key not in args:
            return f"missing required argument: {key}"

    for key, value in args.items():
        spec = properties.get(key, {}) or {}
        expected = spec.get("type")
        if expected == "string" and not isinstance(value, str):
            return f"{key} must be string"
        if expected == "number" and (
            not isinstance(value, (int, float)) or isinstance(value, bool)
        ):
            return f"{key} must be number"
        if expected == "integer" and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            return f"{key} must be integer"
        if expected == "boolean" and not isinstance(value, bool):
            return f"{key} must be boolean"
        if expected == "array" and not isinstance(value, list):
            return f"{key} must be array"
        if "enum" in spec and value not in spec["enum"]:
            return f"{key} must be one of {spec['enum']}"
    return None


def dispatch_named_tool(
    name: str,
    arguments: dict,
    store,
    ctx: ToolContext,
    *,
    subquery_id: Optional[str] = None,
) -> tuple[dict, str, dict]:
    """Validate and dispatch one already-resolved tool call.

    This is THE single validation/dispatch implementation for the middleware:
    both the model tool loop (via :func:`dispatch_tool_traced`) and the
    deterministic router (2.2.3.2) go through here, so schema validation, the
    write guard, the refresh cap, and structured logging are shared. ``name`` is
    a registry tool name and ``arguments`` is an already-parsed argument dict
    (no model JSON payload). Returns ``(result, name, validated_args)`` where
    ``validated_args`` is the schema-filtered subset actually passed to the
    handler. Never raises for a handler failure — it is returned as
    ``{"error": ...}`` so callers can fall back to the normal retrieval lane.
    """
    trace_start = _trace_tool_started(name, subquery_id)

    def finish(result: dict, resolved_name: str, args: dict) -> tuple[dict, str, dict]:
        _trace_tool_completed(resolved_name or name, result, trace_start, subquery_id)
        return result, resolved_name, args

    tool = REGISTRY.get(name)
    if not tool:
        return finish({"error": f"unknown tool: {name}"}, name, {})

    if not isinstance(arguments, dict):
        return finish({"error": "tool arguments must be an object", "tool": name}, name, {})

    properties = tool.parameters.get("properties", {}) or {}
    args = {k: v for k, v in arguments.items() if k in properties}
    error = validate_args(tool.parameters, args)
    if error:
        return finish({"error": error, "tool": name}, name, args)

    if tool.write and not ctx.allow_write:
        return finish({"error": "write tools disabled"}, name, args)
    if tool.write:
        refresh_cost = 1
        if name == "refresh_data":
            requested_sources = args.get("sources")
            if isinstance(requested_sources, list) and requested_sources:
                refresh_cost = len({
                    str(source).strip().lower()
                    for source in requested_sources
                    if str(source).strip()
                })
            else:
                try:
                    report = store.get_freshness_report(
                        str(args.get("ticker") or "").strip().upper()
                    )
                    refresh_cost = sum(
                        1
                        for status in report.get("sources", {}).values()
                        if status.get("status") in {"stale", "never_fetched"}
                    )
                except Exception:  # noqa: BLE001 - the guard must fail closed
                    refresh_cost = ctx.max_refreshes - ctx.refresh_count
                refresh_cost = max(refresh_cost, 1)
        if ctx.refresh_count + refresh_cost > ctx.max_refreshes:
            return finish({"error": "max refreshes exceeded", "tool": name}, name, args)
        ctx.refresh_count += refresh_cost

    start = perf_counter()
    try:
        result = tool.handler(store, **args)
    except Exception as e:  # noqa: BLE001
        logger.exception("Tool handler failed: %s", name)
        return finish({"error": str(e), "tool": name}, name, args)
    finally:
        duration_ms = round((perf_counter() - start) * 1000, 1)
        logger.info("Tool call %s args=%s duration_ms=%s", name, args, duration_ms)
    return finish(result, name, args)


def dispatch_tool(call: dict, store, ctx: ToolContext) -> dict:
    result, _name, _args = dispatch_tool_traced(call, store, ctx)
    return result


def dispatch_tool_traced(call: dict, store, ctx: ToolContext) -> tuple[dict, str, dict]:
    """Parse a model tool-call payload, then delegate to
    :func:`dispatch_named_tool`.

    Extracts the tool name and JSON ``arguments`` from the OpenAI-style function
    call, then hands validation and dispatch to the shared implementation. Also
    returns the tool name and validated (schema-filtered) arguments so the model
    tool loop can populate the evidence trace (2.2.1.2) with exactly what a tool
    call resolved to. ``dispatch_tool`` is a thin wrapper for callers that only
    need the result.
    """
    try:
        function = call.get("function", {}) or {}
        name = function.get("name", "")
        try:
            raw_args = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError as e:
            return {"error": f"invalid tool arguments JSON: {e}"}, name, {}
        if not isinstance(raw_args, dict):
            return {"error": "tool arguments must be an object"}, name, {}
        return dispatch_named_tool(name, raw_args, store, ctx)
    except Exception as e:  # noqa: BLE001
        logger.exception("Tool dispatch failed")
        return {"error": str(e)}, "", {}
