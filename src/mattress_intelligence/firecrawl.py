"""Firecrawl v2 adapter for search, rendered page capture, and image manifests."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class FirecrawlError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FirecrawlImage:
    url: str
    alt_text: str | None = None
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True, slots=True)
class FirecrawlPage:
    url: str
    title: str | None
    markdown: str
    html: str
    raw_html: str
    images: tuple[FirecrawlImage, ...]
    links: tuple[str, ...]
    scrape_id: str | None = None


@dataclass(frozen=True, slots=True)
class FirecrawlSearchResult:
    url: str
    title: str | None
    description: str


class FirecrawlClient:
    base_endpoint = "https://api.firecrawl.dev/v2"

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 60.0,
        wait_ms: int = 1500,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.wait_ms = wait_ms
        self.max_retries = max_retries

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "brixta-mattress-intelligence/1.3",
        }

    def _post(
        self,
        path: str,
        payload: dict[str, object],
    ) -> dict[str, Any]:
        endpoint = f"{self.base_endpoint}/{path.lstrip('/')}"

        for attempt in range(self.max_retries + 1):
            request = Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                method="POST",
                headers=self._headers(),
            )

            try:
                with urlopen(
                    request,
                    timeout=self.timeout_seconds + 10,
                ) as response:
                    value: object = json.loads(
                        response.read().decode("utf-8")
                    )

                    if not isinstance(value, dict):
                        raise FirecrawlError(
                            "Firecrawl returned a non-object response."
                        )

                    return value

            except HTTPError as exc:
                detail = exc.read().decode(
                    "utf-8",
                    errors="replace",
                )[:1_500]

                if (
                    exc.code == 429
                    or 500 <= exc.code < 600
                ) and attempt < self.max_retries:
                    time.sleep(min(2**attempt, 20.0))
                    continue

                raise FirecrawlError(
                    f"Firecrawl HTTP {exc.code}: {detail}"
                ) from exc

            except (
                URLError,
                TimeoutError,
                json.JSONDecodeError,
            ) as exc:
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 20.0))
                    continue

                raise FirecrawlError(
                    f"Firecrawl request failed: {exc}"
                ) from exc

        raise FirecrawlError(
            "Firecrawl request failed after retries."
        )

    @staticmethod
    def _images(raw: object) -> tuple[FirecrawlImage, ...]:
        if not isinstance(raw, list):
            return ()

        images: list[FirecrawlImage] = []

        for item in raw:
            if isinstance(item, str):
                url = item.strip()

                if url.startswith(("http://", "https://")):
                    images.append(FirecrawlImage(url=url))

                continue

            if not isinstance(item, dict):
                continue

            url = str(
                item.get("url")
                or item.get("src")
                or ""
            ).strip()

            if not url.startswith(("http://", "https://")):
                continue

            width = item.get("width")
            height = item.get("height")

            images.append(
                FirecrawlImage(
                    url=url,
                    alt_text=(
                        str(
                            item.get("alt")
                            or item.get("altText")
                            or ""
                        ).strip()
                        or None
                    ),
                    width=(
                        int(width)
                        if isinstance(width, (int, float))
                        else None
                    ),
                    height=(
                        int(height)
                        if isinstance(height, (int, float))
                        else None
                    ),
                )
            )

        return tuple(images)

    def scrape(self, url: str) -> FirecrawlPage:
        payload = self._post(
            "scrape",
            {
                "url": url,
                "formats": [
                    "markdown",
                    "html",
                    "rawHtml",
                    "links",
                    "images",
                ],
                "onlyMainContent": False,
                "removeBase64Images": True,
                "blockAds": True,
                "storeInCache": True,
                "waitFor": self.wait_ms,
                "timeout": int(self.timeout_seconds * 1_000),
            },
        )

        if not payload.get("success", True):
            raise FirecrawlError(
                str(
                    payload.get("error")
                    or "Firecrawl scrape failed."
                )
            )

        raw_data = payload.get("data")

        data: dict[str, Any]
        if isinstance(raw_data, dict):
            data = raw_data
        else:
            data = payload

        raw_metadata = data.get("metadata")

        metadata: dict[str, Any]
        if isinstance(raw_metadata, dict):
            metadata = raw_metadata
        else:
            metadata = {}

        raw_links = data.get("links")

        links_list: list[object]
        if isinstance(raw_links, list):
            links_list = raw_links
        else:
            links_list = []

        links = tuple(
            link
            for item in links_list
            if (
                link := str(item).strip()
            ).startswith(("http://", "https://"))
        )

        title_value = metadata.get("title")
        scrape_id_value = metadata.get("scrapeId")

        return FirecrawlPage(
            url=str(
                metadata.get("url")
                or metadata.get("sourceURL")
                or url
            ),
            title=(
                str(title_value).strip()
                if title_value
                else None
            ),
            markdown=str(data.get("markdown") or ""),
            html=str(data.get("html") or ""),
            raw_html=str(data.get("rawHtml") or ""),
            images=self._images(data.get("images")),
            links=links,
            scrape_id=(
                str(scrape_id_value)
                if scrape_id_value
                else None
            ),
        )

    def scrape_for_assets(self, url: str) -> FirecrawlPage:
        return self.scrape(url)

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        location: str | None = None,
    ) -> list[FirecrawlSearchResult]:
        body: dict[str, object] = {
            "query": query,
            "limit": limit,
        }

        if location:
            body["location"] = location

        payload = self._post("search", body)

        if not payload.get("success", True):
            raise FirecrawlError(
                str(
                    payload.get("error")
                    or "Firecrawl search failed."
                )
            )

        raw_data = payload.get("data")

        raw_results: list[object]

        if isinstance(raw_data, dict):
            web_results = raw_data.get("web")

            if isinstance(web_results, list):
                raw_results = web_results
            else:
                raw_results = []

        elif isinstance(raw_data, list):
            raw_results = raw_data

        else:
            raw_results = []

        results: list[FirecrawlSearchResult] = []

        for item in raw_results:
            if not isinstance(item, dict):
                continue

            result_url = str(
                item.get("url") or ""
            ).strip()

            if not result_url.startswith(
                ("http://", "https://")
            ):
                continue

            title_value = item.get("title")

            results.append(
                FirecrawlSearchResult(
                    url=result_url,
                    title=(
                        str(title_value).strip()
                        if title_value
                        else None
                    ),
                    description=str(
                        item.get("description")
                        or item.get("markdown")
                        or ""
                    ),
                )
            )

        return results

    def check_connection(self) -> dict[str, object]:
        results = self.search(
            "BRIXTA mattress intelligence",
            limit=1,
        )

        return {
            "provider": "firecrawl",
            "results": len(results),
        }