"""Polite, bounded catalogue discovery with sitemap-first crawling."""

from __future__ import annotations

import gzip
import heapq
import hashlib
import io
import mimetypes
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser

import lxml.etree as etree
import lxml.html as html
from pypdf import PdfReader

from .normalization import canonicalize_url, clean_text
from .settings import Settings

if TYPE_CHECKING:
    from playwright.sync_api import Browser, Playwright


PRODUCT_HINTS = (
    "/products/",
    "/product/",
    "/mattress/",
    "/mattresses/",
)
REJECT_HINTS = (
    "/account",
    "/cart",
    "/checkout",
    "/login",
    "/privacy",
    "/terms",
    "/wishlist",
    "/search",
    "/collections/bedsheets",
    "/collections/recliners",
    "/collections/pillows",
    "/collections/chairs",
    "/collections/sofas",
)
NON_PRODUCT_PATH_RE = re.compile(
    r"/(?:collections?|pages?)/(?:mattress-in-[^/]+|mattress-for-[^/]+|"
    r"mattress-store-in-[^/]+|mattress-category-products|store-locator)(?:/|$)",
    re.IGNORECASE,
)
PRODUCT_DETAIL_PATH_RE = re.compile(r"/(?:products?|shop|mattress(?:es)?)/[^/?#]+", re.IGNORECASE)



class FetchError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FetchedDocument:
    url: str
    status: int
    content_type: str
    body: bytes
    retrieved_at_epoch: float
    artifact_path: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.body).hexdigest()

    @property
    def text(self) -> str:
        charset = "utf-8"
        match = re.search(r"charset=([\w-]+)", self.content_type, re.IGNORECASE)
        if match:
            charset = match.group(1)
        return self.body.decode(charset, errors="replace")

    @property
    def is_html(self) -> bool:
        return "html" in self.content_type.casefold() or self.url.casefold().endswith(
            (".htm", ".html")
        )

    @property
    def is_pdf(self) -> bool:
        return "pdf" in self.content_type.casefold() or self.url.casefold().endswith(".pdf")

    def extracted_text(self, max_characters: int = 200_000) -> str:
        """Extract bounded text from HTML, PDF, or plain-text research artifacts."""

        if self.is_html:
            try:
                tree = html.fromstring(self.body, base_url=self.url)
                for node in tree.xpath("//script|//style|//noscript|//svg"):
                    parent = node.getparent()
                    if parent is not None:
                        parent.remove(node)
                return clean_text(" ".join(tree.xpath("//body//text()")))[:max_characters]
            except (etree.ParserError, ValueError):
                return ""
        if self.is_pdf:
            try:
                reader = PdfReader(io.BytesIO(self.body))
                chunks: list[str] = []
                length = 0
                for page_number, page in enumerate(reader.pages, start=1):
                    text = clean_text(page.extract_text() or "")
                    if not text:
                        continue
                    chunk = f"[PDF page {page_number}] {text}"
                    chunks.append(chunk)
                    length += len(chunk)
                    if length >= max_characters:
                        break
                return "\n".join(chunks)[:max_characters]
            except Exception as exc:  # pypdf exposes several parser-specific exception classes.
                raise FetchError(f"PDF text extraction failed: {exc}") from exc
        if self.content_type.casefold().startswith("text/"):
            return clean_text(self.text)[:max_characters]
        return ""


@dataclass(slots=True)
class CrawlReport:
    documents: list[FetchedDocument] = field(default_factory=list)
    discovered_urls: set[str] = field(default_factory=set)
    failed_urls: dict[str, str] = field(default_factory=dict)
    blocked_urls: set[str] = field(default_factory=set)
    sitemap_urls: set[str] = field(default_factory=set)
    crawl_log: list[dict[str, object]] = field(default_factory=list)


class ArtifactStore:
    """Content-addressed local artifact storage compatible with later MinIO upload."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, body: bytes, content_type: str, url: str) -> Path:
        digest = hashlib.sha256(body).hexdigest()
        extension = mimetypes.guess_extension(content_type.split(";", 1)[0].strip()) or ""
        if not extension:
            suffix = Path(urlsplit(url).path).suffix
            extension = suffix if len(suffix) <= 8 else ""
        target = self.root / digest[:2] / f"{digest}{extension}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(body)
        return target


class HttpFetcher:
    """Small standard-library fetcher with size limits and robots.txt support."""

    def __init__(self, settings: Settings, respect_robots_txt: bool = True) -> None:
        self.settings = settings
        self.respect_robots_txt = respect_robots_txt
        self.artifacts = ArtifactStore(settings.artifact_dir)
        self._robots: dict[str, RobotFileParser] = {}
        self._last_request_at: dict[str, float] = {}

    def _origin(self, url: str) -> str:
        split = urlsplit(url)
        return f"{split.scheme}://{split.netloc}"

    def _load_robots(self, url: str) -> RobotFileParser:
        origin = self._origin(url)
        if origin in self._robots:
            return self._robots[origin]
        robots_url = f"{origin}/robots.txt"
        parser = RobotFileParser(robots_url)
        try:
            request = Request(
                robots_url,
                headers={"User-Agent": self.settings.user_agent, "Accept": "text/plain,*/*;q=0.1"},
            )
            with urlopen(request, timeout=self.settings.request_timeout_seconds) as response:
                body = response.read(1_000_000).decode("utf-8", errors="replace")
            parser.parse(body.splitlines())
        except (HTTPError, URLError, TimeoutError, ValueError):
            parser.parse([])
        self._robots[origin] = parser
        return parser

    def allowed(self, url: str) -> bool:
        if not self.respect_robots_txt:
            return True
        return self._load_robots(url).can_fetch(self.settings.user_agent, url)

    def _rate_limit(self, url: str) -> None:
        host = urlsplit(url).netloc
        elapsed = time.monotonic() - self._last_request_at.get(host, 0.0)
        remaining = self.settings.request_delay_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_at[host] = time.monotonic()

    def fetch(self, url: str) -> FetchedDocument:
        canonical_url = canonicalize_url(url)
        if not self.allowed(canonical_url):
            raise FetchError("Blocked by robots.txt")
        self._rate_limit(canonical_url)
        request = Request(
            canonical_url,
            headers={
                "User-Agent": self.settings.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/pdf,application/xml;q=0.9,*/*;q=0.5",
                "Accept-Encoding": "identity",
            },
        )
        try:
            with urlopen(request, timeout=self.settings.request_timeout_seconds) as response:
                status = int(getattr(response, "status", 200))
                final_url = canonicalize_url(response.geturl())
                content_type = response.headers.get_content_type()
                charset = response.headers.get_content_charset()
                if charset:
                    content_type = f"{content_type}; charset={charset}"
                body = response.read(self.settings.max_download_bytes + 1)
        except HTTPError as exc:
            raise FetchError(f"HTTP {exc.code}") from exc
        except (URLError, TimeoutError, ValueError) as exc:
            raise FetchError(str(exc)) from exc
        if len(body) > self.settings.max_download_bytes:
            raise FetchError(
                f"Response exceeds MATTRESS_INTEL_MAX_DOWNLOAD_BYTES={self.settings.max_download_bytes}"
            )
        path = self.artifacts.put(body, content_type, final_url)
        return FetchedDocument(
            url=final_url,
            status=status,
            content_type=content_type,
            body=body,
            retrieved_at_epoch=time.time(),
            artifact_path=str(path),
        )

    def robots_sitemaps(self, base_url: str) -> list[str]:
        parser = self._load_robots(base_url)
        return [canonicalize_url(url) for url in parser.site_maps() or []]

    def close(self) -> None:
        """Release optional fetcher resources; standard HTTP has none."""


class HybridBrowserFetcher(HttpFetcher):
    """Render JavaScript shells with Playwright only when the HTTP page is empty."""

    def __init__(self, settings: Settings, respect_robots_txt: bool = True) -> None:
        super().__init__(settings, respect_robots_txt)
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None

    @staticmethod
    def _needs_browser(document: FetchedDocument) -> bool:
        if not document.is_html:
            return False
        lowered = document.text.casefold()
        visible_text = clean_text(re.sub(r"<[^>]+>", " ", lowered))
        shell_marker = any(
            marker in lowered
            for marker in ('id="__next"', "id='__next'", 'id="root"', 'id="app"')
        )
        has_product_json = '"@type":"product"' in lowered.replace(" ", "")
        return shell_marker and len(visible_text) < 800 and not has_product_json

    def _ensure_browser(self) -> Browser:
        if self._browser is not None:
            return self._browser
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise FetchError(
                "JavaScript rendering requested but Playwright is not installed. "
                "Install the crawl extra and Chromium."
            ) from exc
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=True)
        self._playwright = playwright
        self._browser = browser
        return browser

    def fetch(self, url: str) -> FetchedDocument:
        document = super().fetch(url)
        if not self._needs_browser(document):
            return document
        browser = self._ensure_browser()
        page = browser.new_page(user_agent=self.settings.user_agent)
        try:
            response = page.goto(
                document.url,
                wait_until="networkidle",
                timeout=int(self.settings.request_timeout_seconds * 1_000),
            )
            rendered = page.content().encode("utf-8")
            if len(rendered) > self.settings.max_download_bytes:
                raise FetchError("Rendered page exceeds MATTRESS_INTEL_MAX_DOWNLOAD_BYTES.")
            path = self.artifacts.put(rendered, "text/html; charset=utf-8", document.url)
            return FetchedDocument(
                url=document.url,
                status=int(response.status if response else document.status),
                content_type="text/html; charset=utf-8",
                body=rendered,
                retrieved_at_epoch=time.time(),
                artifact_path=str(path),
            )
        finally:
            page.close()

    def close(self) -> None:
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None


def _same_company_host(url: str, base_url: str) -> bool:
    candidate = (urlsplit(url).hostname or "").removeprefix("www.")
    base = (urlsplit(base_url).hostname or "").removeprefix("www.")
    return candidate == base or candidate.endswith(f".{base}")


def _url_priority(url: str) -> int:
    """Score URLs for evidence value while protecting the crawl budget from SEO pages."""

    lowered = url.casefold()
    path = urlsplit(url).path.casefold()
    if any(hint in lowered for hint in REJECT_HINTS) or NON_PRODUCT_PATH_RE.search(path):
        return -100
    if "/blogs/" in path or "/blog/" in path:
        return -20
    if PRODUCT_DETAIL_PATH_RE.search(path):
        score = 90
    elif path.rstrip("/") in {"/collections/mattress", "/collections/mattresses"}:
        score = 24  # useful catalogue hub, never treated as a product by extraction
    elif "/collections/" in path:
        score = 4
    else:
        score = sum(10 for hint in PRODUCT_HINTS if hint in path)
    if lowered.endswith((".pdf", ".pdf?download=1")):
        score += 70
    if any(word in lowered for word in ("catalog", "catalogue", "brochure", "specification")):
        score += 25
    if any(word in lowered for word in ("patent", "teardown", "cut-open", "cutopen")):
        score += 20
    return score


def _html_links(document: FetchedDocument) -> list[str]:
    if not document.is_html:
        return []
    try:
        tree = html.fromstring(document.body, base_url=document.url)
    except (etree.ParserError, ValueError):
        return []
    links: list[str] = []
    for raw in tree.xpath("//a[@href]/@href | //link[@href]/@href"):
        if not isinstance(raw, str) or raw.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        try:
            links.append(canonicalize_url(raw, document.url))
        except ValueError:
            continue
    return links


def _sitemap_locations(document: FetchedDocument) -> tuple[list[str], bool]:
    raw = document.body
    if document.url.casefold().endswith(".gz") or "gzip" in document.content_type.casefold():
        try:
            raw = gzip.decompress(raw)
        except OSError:
            return [], False
    try:
        root = etree.fromstring(raw, parser=etree.XMLParser(resolve_entities=False, no_network=True))
    except etree.XMLSyntaxError:
        return [], False
    local_name = etree.QName(root).localname.casefold()
    locations = [clean_text(str(value)) for value in root.xpath("//*[local-name()='loc']/text()")]
    return [url for url in locations if url], local_name == "sitemapindex"


class CatalogueCrawler:
    """Priority-based, bounded official-site crawler with a complete decision log."""

    def __init__(self, fetcher: HttpFetcher) -> None:
        self.fetcher = fetcher

    def _discover_sitemaps(
        self,
        base_url: str,
        report: CrawlReport,
        limit: int = 25,
    ) -> set[str]:
        seeds = self.fetcher.robots_sitemaps(base_url)
        seeds.append(canonicalize_url("/sitemap.xml", base_url))
        pending = deque(dict.fromkeys(seeds))
        visited: set[str] = set()
        page_urls: set[str] = set()
        while pending and len(visited) < limit:
            sitemap_url = pending.popleft()
            if sitemap_url in visited:
                continue
            visited.add(sitemap_url)
            try:
                document = self.fetcher.fetch(sitemap_url)
            except FetchError as exc:
                report.crawl_log.append(
                    {
                        "stage": "sitemap",
                        "action": "failed",
                        "url": sitemap_url,
                        "reason": str(exc),
                    }
                )
                continue
            locations, is_index = _sitemap_locations(document)
            report.crawl_log.append(
                {
                    "stage": "sitemap",
                    "action": "parsed",
                    "url": sitemap_url,
                    "locations": len(locations),
                    "is_index": is_index,
                }
            )
            for location in locations:
                if not _same_company_host(location, base_url):
                    continue
                canonical = canonicalize_url(location)
                if is_index or canonical.casefold().endswith((".xml", ".xml.gz")):
                    pending.append(canonical)
                else:
                    page_urls.add(canonical)
        return page_urls

    def crawl(
        self,
        base_url: str,
        max_pages: int,
        extra_urls: list[str] | None = None,
        *,
        max_depth: int = 4,
    ) -> CrawlReport:
        if not urlsplit(base_url).scheme:
            base_url = f"https://{base_url}"
        base_url = canonicalize_url(base_url)
        report = CrawlReport()
        sitemap_pages = self._discover_sitemaps(base_url, report)
        report.sitemap_urls.update(sitemap_pages)
        report.discovered_urls.update(sitemap_pages)

        pending: list[tuple[int, int, int, str, str | None, str]] = []
        queued: set[str] = set()
        sequence = 0

        def enqueue(
            url: str,
            *,
            depth: int,
            parent_url: str | None,
            reason: str,
            force_priority: int | None = None,
        ) -> None:
            nonlocal sequence
            canonical = canonicalize_url(url, parent_url)
            if canonical in queued:
                return
            priority = force_priority if force_priority is not None else _url_priority(canonical)
            if priority < 0:
                report.crawl_log.append(
                    {
                        "stage": "crawl",
                        "action": "rejected",
                        "url": canonical,
                        "parent_url": parent_url,
                        "depth": depth,
                        "priority": priority,
                        "reason": "URL matched a rejected path",
                    }
                )
                return
            queued.add(canonical)
            report.discovered_urls.add(canonical)
            sequence += 1
            heapq.heappush(
                pending,
                (-priority, sequence, depth, canonical, parent_url, reason),
            )
            if len(report.crawl_log) < 100_000:
                report.crawl_log.append(
                    {
                        "stage": "crawl",
                        "action": "queued",
                        "url": canonical,
                        "parent_url": parent_url,
                        "depth": depth,
                        "priority": priority,
                        "reason": reason,
                    }
                )

        enqueue(base_url, depth=0, parent_url=None, reason="official homepage", force_priority=100)
        for sitemap_url in sorted(sitemap_pages, key=lambda url: (-_url_priority(url), url)):
            enqueue(sitemap_url, depth=0, parent_url=None, reason="official sitemap")
        for url in extra_urls or []:
            if _same_company_host(url, base_url):
                enqueue(url, depth=0, parent_url=None, reason="same-domain seed/search result")

        while pending and len(report.documents) < max_pages:
            negative_priority, _, depth, url, parent_url, reason = heapq.heappop(pending)
            priority = -negative_priority
            if depth > max_depth:
                report.crawl_log.append(
                    {
                        "stage": "crawl",
                        "action": "skipped",
                        "url": url,
                        "parent_url": parent_url,
                        "depth": depth,
                        "priority": priority,
                        "reason": f"maximum crawl depth {max_depth} exceeded",
                    }
                )
                continue
            try:
                document = self.fetcher.fetch(url)
            except FetchError as exc:
                if "robots.txt" in str(exc):
                    report.blocked_urls.add(url)
                    action = "blocked"
                else:
                    report.failed_urls[url] = str(exc)
                    action = "failed"
                report.crawl_log.append(
                    {
                        "stage": "crawl",
                        "action": action,
                        "url": url,
                        "parent_url": parent_url,
                        "depth": depth,
                        "priority": priority,
                        "reason": str(exc),
                    }
                )
                continue

            report.documents.append(document)
            report.crawl_log.append(
                {
                    "stage": "crawl",
                    "action": "fetched",
                    "url": document.url,
                    "parent_url": parent_url,
                    "depth": depth,
                    "priority": priority,
                    "reason": reason,
                    "status": document.status,
                    "content_type": document.content_type,
                    "bytes": len(document.body),
                    "artifact_path": document.artifact_path,
                }
            )

            if depth >= max_depth:
                continue
            for link in _html_links(document):
                if not _same_company_host(link, base_url):
                    continue
                link_priority = _url_priority(link)
                early_exploration = len(report.documents) < max(10, max_pages // 5)
                if link_priority > 0 or early_exploration:
                    enqueue(
                        link,
                        depth=depth + 1,
                        parent_url=document.url,
                        reason=(
                            "product/catalogue URL hint"
                            if link_priority > 0
                            else "bounded early-site exploration"
                        ),
                    )

        return report

