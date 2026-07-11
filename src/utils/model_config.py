"""
src/utils/model_config.py
Load TraceAlchemy model settings for serve scripts and tooling.

Resolution order (each step deep-merges over the previous):
  1. ``configs/model.yaml`` — committed defaults (no machine-specific paths)
  2. ``configs/model.local.yaml`` — optional, gitignored overrides
  3. ``${ENV_VAR}`` substitution on string values (env wins over file values)

Copy ``configs/model.example.yaml`` to ``configs/model.local.yaml`` and set
``paths`` (and any other overrides) for your machine.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

import yaml

_CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"
_BASE_PATH = _CONFIG_DIR / "model.yaml"
_LOCAL_PATH = _CONFIG_DIR / "model.local.yaml"

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_env(value: Any) -> Any:
    if isinstance(value, str):
        def _replace(match: re.Match[str]) -> str:
            env_key = match.group(1)
            return os.environ.get(env_key, match.group(0))

        return _ENV_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


def load_model_config(
    base_path: Optional[Path] = None,
    local_path: Optional[Path] = None,
) -> dict:
    """Return the merged ``model`` section from YAML config files."""
    base_path = base_path or _BASE_PATH
    local_path = local_path or _LOCAL_PATH

    if not base_path.exists():
        raise FileNotFoundError(
            f"Model config not found: {base_path}. "
            "Expected configs/model.yaml in the project root."
        )

    with open(base_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if local_path.exists():
        with open(local_path, encoding="utf-8") as f:
            local_data = yaml.safe_load(f) or {}
        data = _deep_merge(data, local_data)

    data = _resolve_env(data)
    model = data.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"Expected top-level 'model' mapping in {base_path}")
    return model


def _require_path(value: str, label: str) -> str:
    if not value or "${" in value:
        raise ValueError(
            f"Missing {label}. Copy configs/model.example.yaml to "
            f"configs/model.local.yaml and set paths.{label}, or export the "
            f"matching environment variable."
        )
    return value


def get_serve_settings(
    base_path: Optional[Path] = None,
    local_path: Optional[Path] = None,
) -> dict:
    """Flatten model config into the fields used by serve_model scripts."""
    model = load_model_config(base_path=base_path, local_path=local_path)
    paths = model.get("paths") or {}
    spec = model.get("speculative_decoding") or {}
    cache = model.get("cache") or {}

    build_dir = _require_path(str(paths.get("build_dir", "")), "build_dir")
    main_model = _require_path(str(paths.get("main_model", "")), "main_model")
    binary = str(paths.get("binary") or "bin/llama-server.exe")

    draft_model = ""
    if spec.get("enabled", False):
        draft_model = _require_path(
            str(spec.get("draft_model_path", "")),
            "speculative_decoding.draft_model_path",
        )

    server_exe = str(Path(build_dir) / binary)

    return {
        "host": str(model.get("host") or "127.0.0.1"),
        "port": int(model.get("port") or 8087),
        "build_dir": build_dir,
        "server_exe": server_exe,
        "main_model": main_model,
        "draft_model": draft_model,
        "speculative_enabled": bool(spec.get("enabled", False)),
        "draft_block_size": int(spec.get("draft_block_size") or 3),
        "max_context": int(model.get("max_context") or 131072),
        "gpu_layers": int(model.get("gpu_layers") or 99),
        "cpu_threads": int(model.get("cpu_threads") or 8),
        "cache_type_key": str(cache.get("type_key") or "q8_0"),
        "cache_type_value": str(cache.get("type_value") or "turbo4"),
        "platform": str(model.get("platform") or "windows"),
        "gpu_arch": str(model.get("gpu_arch") or ""),
        "quantization": str(model.get("quantization") or "Q8_0"),
    }
