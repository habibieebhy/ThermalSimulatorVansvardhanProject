"""Replaceable web-search providers for evidence discovery."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .llm import LLMProvider, discovery_queries


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


@dataclass(slots=True)
class TavilySearchProvider:
    """No-SDK Tavily adapter retained as a URL-discovery alternative."""

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
                detail = exc.read().decode("utf-8", errors="replace")[:1_000]
                if (exc.code == 429 or 500 <= exc.code < 600) and attempt < self.max_retries:
                    time.sleep(min(2**attempt, 20.0))
                    continue
                raise SearchError(f"Tavily HTTP {exc.code}: {detail}") from exc
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
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
        queries = discovery_queries(
            company_name,
            official_domain,
            market,
            brand_aliases,
            custom_queries,
        )[: self.max_search_queries]
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
                        "note": (
                            "Tavily discovered this URL. Final product admission occurs only "
                            "after deterministic and optional LLM document recognition."
                        ),
                    }
                )
        return list(dict.fromkeys(urls))

    def check_connection(self) -> dict:
        request = Request(self.usage_endpoint, method="GET", headers=self._headers())
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1_000]
            raise SearchError(f"Tavily usage check HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise SearchError(f"Tavily usage check failed: {exc}") from exc


def build_search_provider(
    provider: str,
    tavily_api_key: str | None,
    max_search_queries: int,
    llm_provider: LLMProvider,
):
    """Build a source-discovery provider.

    OpenAI/Gemini providers may use their native search tools, but their role remains URL
    discovery and source classification. They do not perform downstream inference.
    """

    normalized = provider.strip().casefold()
    if normalized in {"", "none", "disabled"}:
        return DisabledSearchProvider()
    if normalized == "tavily":
        if not tavily_api_key:
            raise ValueError("TAVILY_API_KEY is required when search provider is tavily.")
        return TavilySearchProvider(
            api_key=tavily_api_key,
            max_search_queries=max(1, min(max_search_queries, 12)),
        )
    if normalized in {"openai", "gemini"}:
        if llm_provider.name != normalized:
            key_name = "OPENAI_API_KEY" if normalized == "openai" else "GEMINI_API_KEY"
            raise ValueError(
                f"{normalized.title()} search requires MATTRESS_INTEL_LLM_PROVIDER={normalized} "
                f"and {key_name}."
            )
        return llm_provider
    raise ValueError(
        f"Unsupported search provider: {provider}. Use none, tavily, openai, or gemini."
    )
