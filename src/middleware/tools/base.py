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


def dispatch_tool(call: dict, store, ctx: ToolContext) -> dict:
    try:
        function = call.get("function", {}) or {}
        name = function.get("name", "")
        tool = REGISTRY.get(name)
        if not tool:
            return {"error": f"unknown tool: {name}"}

        try:
            raw_args = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError as e:
            return {"error": f"invalid tool arguments JSON: {e}"}
        if not isinstance(raw_args, dict):
            return {"error": "tool arguments must be an object"}

        properties = tool.parameters.get("properties", {}) or {}
        args = {k: v for k, v in raw_args.items() if k in properties}
        error = validate_args(tool.parameters, args)
        if error:
            return {"error": error, "tool": name}

        if tool.write and not ctx.allow_write:
            return {"error": "write tools disabled"}
        if tool.write:
            if ctx.refresh_count + 1 > ctx.max_refreshes:
                return {"error": "max refreshes exceeded", "tool": name}
            ctx.refresh_count += 1

        start = perf_counter()
        try:
            result = tool.handler(store, **args)
        except Exception as e:  # noqa: BLE001
            logger.exception("Tool handler failed: %s", name)
            return {"error": str(e), "tool": name}
        finally:
            duration_ms = round((perf_counter() - start) * 1000, 1)
            logger.info("Tool call %s args=%s duration_ms=%s", name, args, duration_ms)
        return result
    except Exception as e:  # noqa: BLE001
        logger.exception("Tool dispatch failed")
        return {"error": str(e)}
