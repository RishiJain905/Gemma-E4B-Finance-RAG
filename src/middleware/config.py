"""
src/middleware/config.py
Middleware configuration — loaded from configs/ or environment.
"""

from pathlib import Path
from typing import Optional

import yaml


class MiddlewareConfig:
    """Configuration for the FastAPI middleware layer."""

    def __init__(self, config_path: Optional[Path] = None):
        config_path = config_path or (
            Path(__file__).parent.parent.parent / "configs/middleware.yaml"
        )

        self.llama_endpoint: str = "http://127.0.0.1:8087/v1/chat/completions"
        self.embedding_endpoint: str = "http://127.0.0.1:8087/v1/embeddings"
        self.model_name: str = "tracealchemy"
        self.default_temperature: float = 0.3
        self.max_tokens: int = 2048
        self.top_k_documents: int = 5
        self.top_k_facts: int = 10
        self.enable_citations: bool = True

        if config_path.exists():
            self._load_from_file(config_path)

    def _load_from_file(self, path: Path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
