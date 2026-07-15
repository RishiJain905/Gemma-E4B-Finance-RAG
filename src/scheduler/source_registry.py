"""src/scheduler/source_registry.py
Validated data-driven source definitions for scheduler selection and status.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Optional

import yaml

logger = logging.getLogger(__name__)

VALID_SCOPES = frozenset({"universe", "broad", "deep", "sector", "global"})
VALID_RUN_MODES = frozenset({"daily", "hourly", "weekly", "all"})
VALID_CADENCES = frozenset({"daily", "hourly", "weekly"})
VALID_CURSOR_KINDS = frozenset(
    {"date", "timestamp", "page_token", "daily_index", "none"}
)


@dataclass(frozen=True)
class SourceSpec:
    """One validated scheduler source definition and its availability state."""

    name: str
    capability_group: str
    enabled: bool
    required_env: Optional[str]
    scope: str
    cadence: str
    run_modes: tuple[str, ...]
    priority: int
    dependencies: tuple[str, ...]
    cursor_kind: str
    overlap: Optional[str]
    requests_per_minute: int
    requests_per_day: int
    requests_per_run: int
    batch_size: int
    max_work_items_per_run: int
    retry_policy: str
    ttl_key: str
    ttl_hours: int
    status: str = "enabled"
    disabled_reason: Optional[str] = None

    @property
    def is_available(self) -> bool:
        """Return whether this source may be selected for work."""
        return self.enabled and self.status == "enabled"

    def __getitem__(self, key: str) -> object:
        """Preserve the legacy scheduler config lookup during one transition."""
        if key == "weight":
            return self.priority
        if key == "ttl_key":
            return self.ttl_key
        raise KeyError(key)


class SourceRegistry:
    """Load, validate, select, and report independently isolated source specs."""

    DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "sources.yaml"

    def __init__(self, specs: Mapping[str, SourceSpec], *, version: str = "1") -> None:
        self.sources = dict(specs)
        self.version = str(version)

    @classmethod
    def load(
        cls,
        path: Optional[Path] = None,
        *,
        environ: Optional[Mapping[str, str]] = None,
    ) -> "SourceRegistry":
        """Load source YAML while converting entry-level errors to disabled states."""
        config_path = Path(path) if path is not None else cls.DEFAULT_PATH
        with config_path.open(encoding="utf-8") as config_file:
            loaded = yaml.safe_load(config_file) or {}
        if not isinstance(loaded, dict):
            raise ValueError("source registry configuration must be a mapping")
        raw_sources = loaded.get("sources")
        if not isinstance(raw_sources, dict) or not raw_sources:
            raise ValueError("source registry must contain a non-empty sources mapping")

        environment = os.environ if environ is None else environ
        specs: dict[str, SourceSpec] = {}
        for raw_name, raw_config in raw_sources.items():
            name = str(raw_name or "").strip()
            try:
                specs[name] = cls._parse_spec(name, raw_config, environment)
            except (TypeError, ValueError) as exc:
                logger.error("Invalid scheduler source %s: %s", name or "<empty>", exc)
                specs[name] = cls._invalid_spec(name, raw_config, str(exc))

        known = set(specs)
        for name, spec in tuple(specs.items()):
            unknown = sorted(set(spec.dependencies) - known)
            if unknown and spec.status != "invalid_configuration":
                reason = f"unknown dependencies: {', '.join(unknown)}"
                logger.error("Invalid scheduler source %s: %s", name, reason)
                specs[name] = replace(
                    spec,
                    enabled=False,
                    status="invalid_configuration",
                    disabled_reason=reason,
                )
        return cls(specs, version=str(loaded.get("version") or "1"))

    @staticmethod
    def _parse_spec(
        name: str,
        raw_config: object,
        environ: Mapping[str, str],
    ) -> SourceSpec:
        if not name:
            raise ValueError("source name is required")
        if not isinstance(raw_config, dict):
            raise ValueError("configuration must be a mapping")

        required = (
            "capability_group",
            "enabled",
            "scope",
            "cadence",
            "run_modes",
            "priority",
            "dependencies",
            "cursor_kind",
            "requests_per_minute",
            "requests_per_day",
            "requests_per_run",
            "batch_size",
            "max_work_items_per_run",
            "retry_policy",
            "ttl_key",
            "ttl_hours",
        )
        missing = [key for key in required if key not in raw_config]
        if missing:
            raise ValueError(f"missing fields: {', '.join(missing)}")

        scope = str(raw_config["scope"] or "").strip()
        if scope not in VALID_SCOPES:
            raise ValueError(f"scope must be one of {', '.join(sorted(VALID_SCOPES))}")
        cursor_kind = str(raw_config["cursor_kind"] or "").strip()
        if cursor_kind not in VALID_CURSOR_KINDS:
            raise ValueError(
                f"cursor_kind must be one of {', '.join(sorted(VALID_CURSOR_KINDS))}"
            )
        run_modes = SourceRegistry._string_tuple(raw_config["run_modes"], "run_modes")
        invalid_modes = sorted(set(run_modes) - VALID_RUN_MODES)
        if invalid_modes:
            raise ValueError(f"invalid run_modes: {', '.join(invalid_modes)}")
        dependencies = SourceRegistry._string_tuple(
            raw_config["dependencies"], "dependencies", allow_empty=True
        )

        positive_fields = (
            "requests_per_minute",
            "requests_per_day",
            "requests_per_run",
            "batch_size",
            "max_work_items_per_run",
            "ttl_hours",
        )
        numbers = {field: int(raw_config[field]) for field in positive_fields}
        if any(value <= 0 for value in numbers.values()):
            raise ValueError(f"{', '.join(positive_fields)} must be positive integers")
        request_limits = (
            numbers["requests_per_minute"],
            numbers["requests_per_day"],
            numbers["requests_per_run"],
        )
        if numbers["batch_size"] > min(request_limits):
            raise ValueError("batch_size must not exceed any request limit")
        priority = int(raw_config["priority"])
        if priority < 0:
            raise ValueError("priority must be a non-negative integer")

        text_fields = ("capability_group", "cadence", "retry_policy", "ttl_key")
        texts = {field: str(raw_config[field] or "").strip() for field in text_fields}
        empty = [field for field, value in texts.items() if not value]
        if empty:
            raise ValueError(f"empty fields: {', '.join(empty)}")
        cadence = texts["cadence"]
        if cadence not in VALID_CADENCES:
            raise ValueError(
                f"cadence must be one of {', '.join(sorted(VALID_CADENCES))}"
            )
        if cadence not in run_modes:
            raise ValueError("cadence must be included in run_modes")

        enabled = raw_config["enabled"]
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")
        required_env = str(raw_config.get("required_env") or "").strip() or None
        status = "enabled"
        disabled_reason = None
        if not enabled:
            status = "configured_disabled"
            disabled_reason = "disabled by configuration"
        elif required_env and not str(environ.get(required_env, "")).strip():
            status = "disabled_missing_key"
            disabled_reason = f"missing {required_env}"

        overlap = raw_config.get("overlap")
        overlap_text = str(overlap).strip() if overlap is not None else None
        if cursor_kind in {"none", "page_token"} and overlap_text is not None:
            raise ValueError(f"overlap must be null for {cursor_kind} cursors")
        if overlap_text is not None:
            pattern = r"\d+d" if cursor_kind in {"date", "daily_index"} else r"\d+[dhm]"
            if re.fullmatch(pattern, overlap_text) is None:
                raise ValueError(f"invalid overlap for {cursor_kind} cursor")
        return SourceSpec(
            name=name,
            capability_group=texts["capability_group"],
            enabled=enabled,
            required_env=required_env,
            scope=scope,
            cadence=texts["cadence"],
            run_modes=run_modes,
            priority=priority,
            dependencies=dependencies,
            cursor_kind=cursor_kind,
            overlap=overlap_text,
            requests_per_minute=numbers["requests_per_minute"],
            requests_per_day=numbers["requests_per_day"],
            requests_per_run=numbers["requests_per_run"],
            batch_size=numbers["batch_size"],
            max_work_items_per_run=numbers["max_work_items_per_run"],
            retry_policy=texts["retry_policy"],
            ttl_key=texts["ttl_key"],
            ttl_hours=numbers["ttl_hours"],
            status=status,
            disabled_reason=disabled_reason,
        )

    @staticmethod
    def _string_tuple(
        value: object,
        field: str,
        *,
        allow_empty: bool = False,
    ) -> tuple[str, ...]:
        if not isinstance(value, list):
            raise ValueError(f"{field} must be a list")
        items = tuple(str(item or "").strip() for item in value)
        if any(not item for item in items) or (not items and not allow_empty):
            raise ValueError(f"{field} must contain non-empty values")
        return items

    @staticmethod
    def _invalid_spec(name: str, raw_config: object, reason: str) -> SourceSpec:
        raw = raw_config if isinstance(raw_config, dict) else {}
        raw_modes = raw.get("run_modes")
        run_modes = (
            tuple(
                str(mode)
                for mode in raw_modes
                if str(mode) in VALID_RUN_MODES
            )
            if isinstance(raw_modes, list)
            else ("all",)
        )
        try:
            priority = int(raw.get("priority") or 0)
        except (TypeError, ValueError):
            priority = 0
        return SourceSpec(
            name=name,
            capability_group=str(raw.get("capability_group") or "invalid"),
            enabled=False,
            required_env=None,
            scope=str(raw.get("scope") or "invalid"),
            cadence=str(raw.get("cadence") or "invalid"),
            run_modes=run_modes,
            priority=priority,
            dependencies=(),
            cursor_kind=str(raw.get("cursor_kind") or "none"),
            overlap=None,
            requests_per_minute=1,
            requests_per_day=1,
            requests_per_run=1,
            batch_size=1,
            max_work_items_per_run=1,
            retry_policy=str(raw.get("retry_policy") or "invalid"),
            ttl_key=str(raw.get("ttl_key") or "invalid"),
            ttl_hours=1,
            status="invalid_configuration",
            disabled_reason=reason,
        )

    def get(self, name: str) -> SourceSpec:
        """Return one named source specification."""
        try:
            return self.sources[name]
        except KeyError as exc:
            raise ValueError(f"unknown source: {name}") from exc

    def select(
        self,
        mode: str,
        *,
        source: Optional[str] = None,
        scope: Optional[str] = None,
        include_disabled: bool = False,
    ) -> list[SourceSpec]:
        """Select sources for one run mode and optional exact source/scope filters."""
        if mode not in VALID_RUN_MODES:
            raise ValueError(f"invalid run mode: {mode}")
        if scope is not None and scope not in VALID_SCOPES:
            raise ValueError(f"invalid source scope: {scope}")
        if source is not None and source not in self.sources:
            raise ValueError(f"unknown source: {source}")
        selected = [
            spec
            for spec in self.sources.values()
            if mode in spec.run_modes
            and (source is None or spec.name == source)
            and (scope is None or spec.scope == scope)
            and (include_disabled or spec.is_available)
        ]
        return sorted(selected, key=lambda item: (item.priority, item.name))

    def status(self) -> dict[str, dict]:
        """Return a visible, non-secret registry status for every configured source."""
        return {
            name: {
                "status": spec.status,
                "reason": spec.disabled_reason,
                "enabled": spec.enabled,
                "scope": spec.scope,
                "cadence": spec.cadence,
                "run_modes": list(spec.run_modes),
                "cursor_kind": spec.cursor_kind,
                "required_env": spec.required_env,
                "configured": spec.required_env is None or spec.status != "disabled_missing_key",
            }
            for name, spec in sorted(
                self.sources.items(), key=lambda item: (item[1].priority, item[0])
            )
        }
