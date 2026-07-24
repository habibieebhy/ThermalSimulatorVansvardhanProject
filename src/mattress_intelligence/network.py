"""Shared HTTP transport helpers for resilient provider requests."""

from __future__ import annotations

from http.client import IncompleteRead, RemoteDisconnected
from urllib.error import HTTPError, URLError


RETRYABLE_TRANSPORT_ERRORS = (
    URLError,
    TimeoutError,
    IncompleteRead,
    RemoteDisconnected,
    ConnectionError,
)


def http_error_detail(exc: HTTPError, *, limit: int = 1_500) -> str:
    """Read an HTTP error body without allowing a truncated body to mask the status code."""

    try:
        body = exc.read()
    except IncompleteRead as read_error:
        body = read_error.partial
    except Exception:
        body = b""
    return body.decode("utf-8", errors="replace")[:limit]
