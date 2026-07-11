"""Tests for src/utils/model_config.py."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.utils.model_config import get_serve_settings, load_model_config


def _write_yaml(path, data):
    path.write_text(yaml.dump(data), encoding="utf-8")


def test_load_model_config_merges_local_over_base(tmp_path):
    base = tmp_path / "model.yaml"
    local = tmp_path / "model.local.yaml"
    _write_yaml(base, {
        "model": {
            "port": 8087,
            "paths": {"main_model": "${MAIN_MODEL_PATH}", "build_dir": "${LLAMA_BUILD_DIR}"},
            "speculative_decoding": {"enabled": False, "draft_model_path": ""},
        }
    })
    _write_yaml(local, {
        "model": {
            "port": 9090,
            "paths": {"main_model": "/models/main.gguf", "build_dir": "/opt/llama/build"},
        }
    })

    cfg = load_model_config(base_path=base, local_path=local)
    assert cfg["port"] == 9090
    assert cfg["paths"]["main_model"] == "/models/main.gguf"
    assert cfg["paths"]["build_dir"] == "/opt/llama/build"


def test_env_substitution_overrides_placeholder(tmp_path, monkeypatch):
    base = tmp_path / "model.yaml"
    _write_yaml(base, {
        "model": {
            "paths": {
                "main_model": "${MAIN_MODEL_PATH}",
                "build_dir": "${LLAMA_BUILD_DIR}",
                "binary": "bin/llama-server",
            },
            "speculative_decoding": {"enabled": False},
        }
    })
    monkeypatch.setenv("MAIN_MODEL_PATH", "/env/main.gguf")
    monkeypatch.setenv("LLAMA_BUILD_DIR", "/env/build")

    cfg = load_model_config(base_path=base, local_path=tmp_path / "missing.yaml")
    assert cfg["paths"]["main_model"] == "/env/main.gguf"
    assert cfg["paths"]["build_dir"] == "/env/build"


def test_get_serve_settings_requires_resolved_paths(tmp_path):
    base = tmp_path / "model.yaml"
    _write_yaml(base, {
        "model": {
            "paths": {"main_model": "${MAIN_MODEL_PATH}", "build_dir": "${LLAMA_BUILD_DIR}"},
            "speculative_decoding": {"enabled": True, "draft_model_path": "${DRAFT_MODEL_PATH}"},
        }
    })

    with pytest.raises(ValueError, match="Missing build_dir"):
        get_serve_settings(base_path=base, local_path=tmp_path / "missing.yaml")


def test_get_serve_settings_flattened(tmp_path):
    base = tmp_path / "model.yaml"
    _write_yaml(base, {
        "model": {
            "host": "127.0.0.1",
            "port": 8087,
            "max_context": 32768,
            "gpu_layers": 40,
            "cpu_threads": 4,
            "cache": {"type_key": "q8_0", "type_value": "f16"},
            "paths": {
                "main_model": "/models/main.gguf",
                "build_dir": "/opt/llama/build",
                "binary": "bin/llama-server.exe",
            },
            "speculative_decoding": {
                "enabled": True,
                "draft_model_path": "/models/draft.gguf",
                "draft_block_size": 3,
            },
        }
    })

    settings = get_serve_settings(base_path=base, local_path=tmp_path / "missing.yaml")
    assert settings["main_model"] == "/models/main.gguf"
    assert settings["server_exe"] == str(
        Path("/opt/llama/build") / "bin/llama-server.exe"
    )
    assert settings["draft_model"] == "/models/draft.gguf"
    assert settings["speculative_enabled"] is True
    assert settings["max_context"] == 32768
