"""
src/utils
Shared utilities for the ingestion pipeline — resilience primitives
(retry, circuit breaker, dead-letter queue) and structured logging.
"""

from .resilience import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    DeadLetterQueue,
    retry_with_backoff,
)
from .logging import IngestionLogger

__all__ = [
    "retry_with_backoff",
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "DeadLetterQueue",
    "IngestionLogger",
]
