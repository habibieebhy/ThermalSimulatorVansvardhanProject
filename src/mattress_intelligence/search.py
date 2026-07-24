"""Replaceable web-search providers for evidence discovery."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .firecrawl import FirecrawlClient, FirecrawlError
from .jina import JinaError, JinaSearchClient
from .llm import LLMProvider, discovery_queries
from .network import RETRYABLE_TRANSPORT_ERRORS, http_error_detail


class SearchError(RuntimeError):
    pass


class DisabledSearchProvider:
    name = "none"

    def __init__(self) -> None:
        self.discovery_log: list[dict[str, object]] = []

    def discover_urls(self, *args, **kwargs) -> list[str]:
        self.discovery_log.clear()
        return []

    def check_connection(self) -> dict:
        return {"provider": "none"}


def _official_host(url: str, official_domain: str) -> bool:
    candidate = (urlsplit(url).hostname or "").removeprefix("www.")
    official = (urlsplit(official_domain).hostname or official_domain).removeprefix("www.")
    return candidate == official or candidate.endswith(f".{official}")


def _discovery_score(url: str, title: str | None, content: str, official_domain: str) -> float:
    text = f"{url} {title or ''} {content[:2_000]}".casefold()
    score = 0.0
    if _official_host(url, official_domain):
        score += 0.35
    if any(token in text for token in ("/products/", "/product/", "mattress")):
        score += 0.25
    if any(token in text for token in ("catalogue", "catalog", "brochure", ".pdf")):
        score += 0.22
    if any(token in text for token in ("layer", "density", "construction", "specification")):
        score += 0.12
    if any(token in text for token in ("mattress in ", "store locator", "/blogs/", "/blog/")):
        score -= 0.30
    return max(0.0, min(1.0, score))


def _record_results(
    *,
    log: list[dict[str, object]],
    query_number: int,
    query: str,
    source: str,
    results: list[tuple[str, str | None, str]],
    official_domain: str,
    seen: set[str],
    accepted: list[str],
) -> None:
    for rank, (url, title, content) in enumerate(results, start=1):
        score = _discovery_score(url, title, content, official_domain)
        keep = score >= 0.18 or _official_host(url, official_domain)
        log.append(
            {
                "query_number": query_number,
                "query": query,
                "rank": rank,
                "url": url,
                "title": title,
                "score": round(score, 4),
                "snippet": content[:1_000],
                "source": source,
                "accepted": keep,
                "note": "URL discovery only; final product admission occurs after document/image recognition.",
            }
        )
        if keep and url not in seen:
            seen.add(url)
            accepted.append(url)


@dataclass(slots=True)
class JinaSearchProvider:
    api_key: str | None
    max_search_queries: int = 6
    max_results: int = 10
    timeout_seconds: float = 60.0
    name: str = "jina"
    discovery_log: list[dict[str, object]] = field(default_factory=list)

    def discover_urls(
        self,
        company_name: str,
        official_domain: str,
        market: str,
        brand_aliases: list[str] | None = None,
        custom_queries: list[str] | None = None,
    ) -> list[str]:
        client = JinaSearchClient(self.api_key, timeout_seconds=self.timeout_seconds)
        queries = discovery_queries(company_name, official_domain, market, brand_aliases, custom_queries)[
            : self.max_search_queries
        ]
        self.discovery_log.clear()
        accepted: list[str] = []
        seen: set[str] = set()
        for query_number, query in enumerate(queries, start=1):
            try:
                results = client.search(query, limit=self.max_results)
            except JinaError as exc:
                self.discovery_log.append(
                    {
                        "query_number": query_number,
                        "query": query,
                        "source": "jina_search",
                        "accepted": False,
                        "note": str(exc),
                    }
                )
                continue
            _record_results(
                log=self.discovery_log,
                query_number=query_number,
                query=query,
                source="jina_search",
                results=[(item.url, item.title, item.content) for item in results],
                official_domain=official_domain,
                seen=seen,
                accepted=accepted,
            )
        return accepted

    def check_connection(self) -> dict:
        return JinaSearchClient(self.api_key, timeout_seconds=self.timeout_seconds).check_connection()


@dataclass(slots=True)
class FirecrawlSearchProvider:
    api_key: str
    max_search_queries: int = 6
    max_results: int = 10
    timeout_seconds: float = 60.0
    name: str = "firecrawl"
    discovery_log: list[dict[str, object]] = field(default_factory=list)

    def discover_urls(
        self,
        company_name: str,
        official_domain: str,
        market: str,
        brand_aliases: list[str] | None = None,
        custom_queries: list[str] | None = None,
    ) -> list[str]:
        client = FirecrawlClient(self.api_key, timeout_seconds=self.timeout_seconds)
        queries = discovery_queries(company_name, official_domain, market, brand_aliases, custom_queries)[
            : self.max_search_queries
        ]
        self.discovery_log.clear()
        accepted: list[str] = []
        seen: set[str] = set()
        for query_number, query in enumerate(queries, start=1):
            try:
                results = client.search(query, limit=self.max_results, location=market)
            except FirecrawlError as exc:
                self.discovery_log.append(
                    {
                        "query_number": query_number,
                        "query": query,
                        "source": "firecrawl_search",
                        "accepted": False,
                        "note": str(exc),
                    }
                )
                continue
            _record_results(
                log=self.discovery_log,
                query_number=query_number,
                query=query,
                source="firecrawl_search",
                results=[(item.url, item.title, item.description) for item in results],
                official_domain=official_domain,
                seen=seen,
                accepted=accepted,
            )
        return accepted

    def check_connection(self) -> dict:
        return FirecrawlClient(self.api_key, timeout_seconds=self.timeout_seconds).check_connection()


@dataclass(slots=True)
class ServicesSearchProvider:
    """Firecrawl-first discovery with Jina used only as a true fallback."""

    jina_api_key: str | None
    firecrawl_api_key: str | None
    max_search_queries: int = 6
    max_results: int = 10
    timeout_seconds: float = 60.0
    name: str = "services"
    discovery_log: list[dict[str, object]] = field(default_factory=list)

    def discover_urls(self, *args, **kwargs) -> list[str]:
        self.discovery_log.clear()

        if self.firecrawl_api_key:
            firecrawl = FirecrawlSearchProvider(
                self.firecrawl_api_key,
                max_search_queries=self.max_search_queries,
                max_results=self.max_results,
                timeout_seconds=self.timeout_seconds,
            )
            firecrawl_urls = firecrawl.discover_urls(*args, **kwargs)
            self.discovery_log.extend(firecrawl.discovery_log)
            if firecrawl_urls:
                return list(dict.fromkeys(firecrawl_urls))

        if self.jina_api_key is not None or not self.firecrawl_api_key:
            jina = JinaSearchProvider(
                self.jina_api_key,
                max_search_queries=self.max_search_queries,
                max_results=self.max_results,
                timeout_seconds=self.timeout_seconds,
            )
            jina_urls = jina.discover_urls(*args, **kwargs)
            self.discovery_log.extend(jina.discovery_log)
            if jina_urls:
                return list(dict.fromkeys(jina_urls))

        raise SearchError("No Firecrawl or Jina search results were available.")

    def check_connection(self) -> dict:
        status: dict[str, object] = {"provider": "services"}
        if self.firecrawl_api_key:
            status["firecrawl"] = FirecrawlClient(
                self.firecrawl_api_key, timeout_seconds=self.timeout_seconds
            ).check_connection()
            status["jina"] = "fallback_not_checked"
            return status
        status["jina"] = JinaSearchClient(
            self.jina_api_key, timeout_seconds=self.timeout_seconds
        ).check_connection()
        return status


@dataclass(slots=True)
class TavilySearchProvider:
    api_key: str
    max_search_queries: int = 6
    max_results: int = 10
    timeout_seconds: float = 60.0
    max_retries: int = 3
    name: str = "tavily"
    discovery_log: list[dict[str, object]] = field(default_factory=list)
    endpoint: str = "https://api.tavily.com/search"
    usage_endpoint: str = "https://api.tavily.com/usage"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Project-ID": "brixta-mattress-intelligence",
        }

    def _post(self, payload: dict) -> dict:
        for attempt in range(self.max_retries + 1):
            request = Request(
                self.endpoint,
                data=json.dumps(payload).encode("utf-8"),
                method="POST",
                headers=self._headers(),
            )
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                detail = http_error_detail(exc, limit=1_000)
                if (exc.code == 429 or 500 <= exc.code < 600) and attempt < self.max_retries:
                    time.sleep(min(2**attempt, 20.0))
                    continue
                raise SearchError(f"Tavily HTTP {exc.code}: {detail}") from exc
            except RETRYABLE_TRANSPORT_ERRORS + (json.JSONDecodeError,) as exc:
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 20.0))
                    continue
                raise SearchError(f"Tavily request failed: {exc}") from exc
        raise SearchError("Tavily request failed after retries.")

    def discover_urls(
        self,
        company_name: str,
        official_domain: str,
        market: str,
        brand_aliases: list[str] | None = None,
        custom_queries: list[str] | None = None,
    ) -> list[str]:
        queries = discovery_queries(company_name, official_domain, market, brand_aliases, custom_queries)[
            : self.max_search_queries
        ]
        self.discovery_log.clear()
        urls: list[str] = []
        for query_number, query in enumerate(queries, start=1):
            response = self._post(
                {
                    "query": query,
                    "topic": "general",
                    "search_depth": "basic",
                    "max_results": self.max_results,
                    "include_answer": False,
                    "include_raw_content": False,
                    "include_images": False,
                    "auto_parameters": False,
                }
            )
            credits = (response.get("usage") or {}).get("credits")
            request_id = response.get("request_id")
            for rank, result in enumerate(response.get("results") or [], start=1):
                url = str(result.get("url") or "").strip().rstrip(".,;")
                if not url:
                    continue
                urls.append(url)
                self.discovery_log.append(
                    {
                        "query_number": query_number,
                        "query": query,
                        "rank": rank,
                        "url": url,
                        "title": result.get("title"),
                        "score": result.get("score"),
                        "snippet": result.get("content"),
                        "source": "tavily_search",
                        "credits": credits,
                        "request_id": request_id,
                        "accepted": True,
                    }
                )
        return list(dict.fromkeys(urls))

    def check_connection(self) -> dict:
        request = Request(self.usage_endpoint, method="GET", headers=self._headers())
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = http_error_detail(exc, limit=1_000)
            raise SearchError(f"Tavily usage check HTTP {exc.code}: {detail}") from exc
        except RETRYABLE_TRANSPORT_ERRORS + (json.JSONDecodeError,) as exc:
            raise SearchError(f"Tavily usage check failed: {exc}") from exc


def build_search_provider(
    provider: str,
    tavily_api_key: str | None,
    jina_api_key: str | None,
    firecrawl_api_key: str | None,
    max_search_queries: int,
    max_results: int,
    llm_provider: LLMProvider,
    *,
    timeout_seconds: float = 60.0,
):
    normalized = provider.strip().casefold()
    common = {
        "max_search_queries": max(1, min(max_search_queries, 12)),
        "max_results": max(1, min(max_results, 50)),
        "timeout_seconds": timeout_seconds,
    }
    if normalized in {"", "none", "disabled"}:
        return DisabledSearchProvider()
    if normalized in {"services", "jina_firecrawl", "fallback"}:
        return ServicesSearchProvider(jina_api_key, firecrawl_api_key, **common)
    if normalized == "jina":
        return JinaSearchProvider(jina_api_key, **common)
    if normalized == "firecrawl":
        if not firecrawl_api_key:
            raise ValueError("FIRECRAWL_API_KEY is required when search provider is firecrawl.")
        return FirecrawlSearchProvider(firecrawl_api_key, **common)
    if normalized == "tavily":
        if not tavily_api_key:
            raise ValueError("TAVILY_API_KEY is required when search provider is tavily.")
        return TavilySearchProvider(tavily_api_key, **common)
    if normalized in {"openai", "gemini"}:
        if llm_provider.name != normalized:
            key_name = "OPENAI_API_KEY" if normalized == "openai" else "GEMINI_API_KEY"
            raise ValueError(
                f"{normalized.title()} search requires MATTRESS_INTEL_LLM_PROVIDER={normalized} and {key_name}."
            )
        return llm_provider
    raise ValueError(
        f"Unsupported search provider: {provider}. Use none, services, jina, firecrawl, tavily, openai, or gemini."
    )
