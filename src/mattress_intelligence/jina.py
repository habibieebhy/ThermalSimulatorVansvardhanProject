"""Jina Search and Reader clients used for discovery and reader fallback."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class JinaError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class JinaSearchResult:
    url: str
    title: str | None
    content: str


@dataclass(frozen=True, slots=True)
class JinaReaderResult:
    url: str
    title: str | None
    content: str
    published_at: str | None = None


class _JinaClient:
    def __init__(self, api_key: str | None, timeout_seconds: float = 60.0, max_retries: int = 3) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/plain;q=0.9",
            "User-Agent": "brixta-mattress-intelligence/1.3",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get(self, url: str, extra_headers: dict[str, str] | None = None) -> object:
        headers = self._headers()
        headers.update(extra_headers or {})
        for attempt in range(self.max_retries + 1):
            request = Request(url, method="GET", headers=headers)
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    body = response.read().decode("utf-8", errors="replace")
                    content_type = response.headers.get("Content-Type", "")
                    if "json" in content_type.casefold() or body.lstrip().startswith(("{", "[")):
                        return json.loads(body)
                    return body
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:1_000]
                if (exc.code == 429 or 500 <= exc.code < 600) and attempt < self.max_retries:
                    time.sleep(min(2**attempt, 20.0))
                    continue
                raise JinaError(f"Jina HTTP {exc.code}: {detail}") from exc
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 20.0))
                    continue
                raise JinaError(f"Jina request failed: {exc}") from exc
        raise JinaError("Jina request failed after retries.")


class JinaSearchClient(_JinaClient):
    """Search through s.jina.ai and normalize JSON or markdown results."""

    endpoint = "https://s.jina.ai/"

    @staticmethod
    def _parse_markdown(text: str) -> list[JinaSearchResult]:
        results: list[JinaSearchResult] = []
        # Reader-style search output commonly contains Title/URL/Description blocks.
        blocks = re.split(r"\n(?=Title:|\[\d+\])", text)
        for block in blocks:
            url_match = re.search(r"(?:URL Source|URL):\s*(https?://\S+)", block)
            if not url_match:
                link_match = re.search(r"\[[^\]]+\]\((https?://[^)]+)\)", block)
                if not link_match:
                    continue
                url = link_match.group(1)
            else:
                url = url_match.group(1).rstrip(".,;)")
            title_match = re.search(r"Title:\s*(.+)", block)
            if not title_match:
                title_match = re.search(r"\[([^\]]+)\]\(https?://", block)
            title = title_match.group(1).strip() if title_match else None
            results.append(JinaSearchResult(url=url, title=title, content=block.strip()))
        return results

    def search(self, query: str, *, limit: int = 10) -> list[JinaSearchResult]:
        url = f"{self.endpoint}{quote(query, safe='')}"
        payload = self._get(
            url,
            {
                "X-Respond-With": "markdown",
                "X-Retain-Images": "none",
                "X-Timeout": str(int(self.timeout_seconds)),
            },
        )
        if isinstance(payload, str):
            return self._parse_markdown(payload)[:limit]
        raw_items: list[object]
        if isinstance(payload, list):
            raw_items = payload
        elif isinstance(payload, dict):
            data = payload.get("data")
            raw_items = data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        else:
            raw_items = []
        results: list[JinaSearchResult] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            candidate_url = str(item.get("url") or item.get("sourceURL") or "").strip()
            if not candidate_url.startswith(("http://", "https://")):
                continue
            results.append(
                JinaSearchResult(
                    url=candidate_url,
                    title=(str(item.get("title")).strip() if item.get("title") else None),
                    content=str(item.get("content") or item.get("description") or ""),
                )
            )
        return results[:limit]

    def check_connection(self) -> dict[str, object]:
        results = self.search("BRIXTA mattress intelligence connectivity check", limit=3)
        return {"provider": "jina", "results": len(results)}


class JinaReaderClient(_JinaClient):
    """Convert a public URL into clean markdown with generated image captions."""

    endpoint = "https://r.jina.ai/"

    def read(self, target_url: str, *, no_cache: bool = False) -> JinaReaderResult:
        url = f"{self.endpoint}{quote(target_url, safe=':/?&=%#')}"
        headers = {
            "X-With-Generated-Alt": "true",
            "X-With-Images-Summary": "true",
            "X-With-Links-Summary": "true",
            "X-Respond-With": "markdown",
            "X-Timeout": str(int(self.timeout_seconds)),
            "X-Use-Final-Url": "true",
        }
        if no_cache:
            headers["X-No-Cache"] = "true"
        payload = self._get(url, headers)
        if isinstance(payload, str):
            title_match = re.search(r"^Title:\s*(.+)$", payload, re.MULTILINE)
            url_match = re.search(r"^URL Source:\s*(https?://\S+)$", payload, re.MULTILINE)
            return JinaReaderResult(
                url=url_match.group(1).strip() if url_match else target_url,
                title=title_match.group(1).strip() if title_match else None,
                content=payload,
            )
        if not isinstance(payload, dict):
            raise JinaError("Jina Reader returned an unsupported response shape.")
        raw_data = payload.get("data")

        if isinstance(raw_data, dict):
            data = raw_data
        else:
            data = payload
        return JinaReaderResult(
            url=str(data.get("url") or target_url),
            title=(str(data.get("title")).strip() if data.get("title") else None),
            content=str(data.get("content") or data.get("markdown") or ""),
            published_at=(
                str(data.get("publishedTime") or data.get("timestamp"))
                if data.get("publishedTime") or data.get("timestamp")
                else None
            ),
        )
