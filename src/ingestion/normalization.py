"""src/ingestion/normalization.py
Deterministic normalization and deduplication keys for provider records.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


NORMALIZATION_VERSION = "1"
SYNDICATED_NEWS_WINDOW_HOURS = 6

_TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "source",
}
_MULTIPLE_SLASHES = re.compile(r"/{2,}")
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def normalize_canonical_url(url: str | None) -> str | None:
    """Return a stable HTTP(S) URL while preserving content-identifying params."""
    if url is None:
        return None
    value = url.strip()
    if not value:
        return None

    parts = urlsplit(value)
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("canonical URL must be an absolute HTTP(S) URL")

    host = parts.hostname.lower().encode("idna").decode("ascii")
    port = parts.port
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        host = f"{host}:{port}"

    path = _MULTIPLE_SLASHES.sub("/", parts.path or "/")
    if path != "/":
        path = path.rstrip("/")

    query = []
    for key, query_value in parse_qsl(parts.query, keep_blank_values=True):
        normalized_key = key.lower()
        if normalized_key.startswith("utm_") or normalized_key in _TRACKING_QUERY_KEYS:
            continue
        query.append((key, query_value))
    query.sort(key=lambda pair: (pair[0], pair[1]))

    return urlunsplit((scheme, host, path, urlencode(query, doseq=True), ""))


def content_hash(content: str) -> str:
    """Hash exact Unicode content after only newline and NFC normalization."""
    if not isinstance(content, str):
        raise TypeError("content must be a string")
    normalized = unicodedata.normalize("NFC", content.replace("\r\n", "\n").replace("\r", "\n"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalize_headline(headline: str) -> str:
    """Conservatively normalize a headline without fuzzy or semantic matching."""
    if not isinstance(headline, str):
        raise TypeError("headline must be a string")
    punctuation_spaced = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in headline
    )
    ascii_text = (
        unicodedata.normalize("NFKD", punctuation_spaced)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    return _NON_ALPHANUMERIC.sub(" ", ascii_text.lower()).strip()


def parse_timestamp(value: str, field_name: str = "timestamp") -> datetime:
    """Parse an ISO-8601 date or timestamp and return an aware UTC datetime."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty ISO-8601 value")
    candidate = value.strip()
    try:
        parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def syndicated_news_key(title: str, published_at: str) -> str:
    """Build an exact normalized-headline key in a six-hour UTC window."""
    normalized = normalize_headline(title)
    if not normalized:
        raise ValueError("title must contain alphanumeric characters")
    published = parse_timestamp(published_at, "published_at")
    bucket_hour = published.hour - (published.hour % SYNDICATED_NEWS_WINDOW_HOURS)
    bucket = published.replace(hour=bucket_hour, minute=0, second=0, microsecond=0)
    material = f"{normalized}\n{bucket.isoformat()}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()
