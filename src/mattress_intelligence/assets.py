"""Image/PDF asset discovery and GPT vision transcription."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import lxml.etree as etree
import lxml.html as html
from PIL import Image

from .crawler import FetchedDocument
from .firecrawl import FirecrawlClient, FirecrawlError
from .llm import LLMError, LLMProvider
from .materials import MaterialLibrary
from .models import (
    AssetKind,
    AssetRecord,
    ClaimStatus,
    CompanyResearchRequest,
    EvidenceObservation,
    EvidenceRef,
    LayerRecord,
    ProductRecord,
    SourceRecord,
    stable_id,
)
from .normalization import canonicalize_url, clean_text
from .object_store import ObjectStore
from .settings import Settings


_LAYER_KEYWORDS = (
    "layer",
    "layers",
    "construction",
    "cross-section",
    "crosssection",
    "inside",
    "cutaway",
    "cut-open",
    "foam",
    "smartgrid",
    "spring",
    "latex",
    "technology",
    "specification",
    "diagram",
)
_REJECT_KEYWORDS = (
    "logo",
    "icon",
    "payment",
    "rating",
    "stars",
    "social",
    "avatar",
    "placeholder",
    "spinner",
    "tracking",
    "pixel",
    "favicon",
)
_MARKDOWN_IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<url>https?://[^)\s]+)")
_TECHNICAL_IMAGE_TERMS = (
    "layer",
    "construction",
    "inside",
    "cutaway",
    "cross section",
    "cross-section",
    "specification",
    "technology",
    "memory foam",
    "latex",
    "pocketed coil",
    "pocket coil",
    "spring",
    "density",
    "thickness",
    "comfort",
    "support",
    "pressure relief",
    "temperature management",
)
_HIGH_VALUE_ASSET_TYPES = {
    "layer_diagram",
    "cutaway_or_cross_section",
    "catalogue_page",
    "specification_table",
    "law_or_manufacturer_label",
    "teardown_frame",
}


@dataclass(frozen=True, slots=True)
class ImageCandidate:
    url: str
    alt_text: str | None
    discovery_method: str
    width: int | None = None
    height: int | None = None


@dataclass(slots=True)
class AssetProcessingResult:
    assets: list[AssetRecord] = field(default_factory=list)
    products: list[ProductRecord] = field(default_factory=list)
    observations: list[EvidenceObservation] = field(default_factory=list)
    log: list[dict[str, object]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    visual_followup_urls: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _PreparedAsset:
    asset: AssetRecord
    candidate: ImageCandidate
    source: SourceRecord
    page_context: str
    body: bytes
    content_type: str
    ocr_text: str


def _srcset_urls(value: str) -> list[str]:
    return [part.strip().split()[0] for part in value.split(",") if part.strip()]


def _html_candidates(document: FetchedDocument) -> list[ImageCandidate]:
    if not document.is_html:
        return []
    try:
        tree = html.fromstring(document.body, base_url=document.url)
    except (ValueError, etree.ParserError):
        return []
    results: list[ImageCandidate] = []
    for node in tree.xpath("//img | //source"):
        alt = clean_text(node.get("alt") or node.get("title") or "") or None
        width = node.get("width")
        height = node.get("height")
        width_value = int(width) if width and str(width).isdigit() else None
        height_value = int(height) if height and str(height).isdigit() else None
        raw_urls: list[str] = []
        for attribute in ("src", "data-src", "data-original", "data-lazy-src"):
            if node.get(attribute):
                raw_urls.append(str(node.get(attribute)))
        for attribute in ("srcset", "data-srcset"):
            if node.get(attribute):
                raw_urls.extend(_srcset_urls(str(node.get(attribute))))
        for raw in raw_urls:
            if raw.startswith("data:"):
                continue
            try:
                url = canonicalize_url(raw, document.url)
            except ValueError:
                continue
            results.append(
                ImageCandidate(
                    url=url,
                    alt_text=alt,
                    discovery_method="html",
                    width=width_value,
                    height=height_value,
                )
            )
    for raw in tree.xpath(
        "//meta[@property='og:image']/@content | //meta[@name='twitter:image']/@content"
    ):
        try:
            results.append(
                ImageCandidate(
                    url=canonicalize_url(str(raw), document.url),
                    alt_text="Open Graph product image",
                    discovery_method="html_meta",
                )
            )
        except ValueError:
            continue
    return results


def _markdown_candidates(document: FetchedDocument) -> list[ImageCandidate]:
    if "markdown" not in document.content_type.casefold():
        return []
    return [
        ImageCandidate(
            url=match.group("url"),
            alt_text=clean_text(match.group("alt")) or None,
            discovery_method="jina_reader",
        )
        for match in _MARKDOWN_IMAGE_RE.finditer(document.text)
    ]


def _candidate_score(candidate: ImageCandidate) -> float:
    text = f"{candidate.url} {candidate.alt_text or ''}".casefold()
    score = 0.15
    score += min(0.58, 0.10 * sum(token in text for token in _LAYER_KEYWORDS))
    score -= min(0.60, 0.18 * sum(token in text for token in _REJECT_KEYWORDS))
    if candidate.width and candidate.height:
        if candidate.width >= 800 and candidate.height >= 500:
            score += 0.15
        if candidate.width <= 120 or candidate.height <= 120:
            score -= 0.40
    if any(text.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")):
        score += 0.05
    return max(0.0, min(1.0, score))


def _recover_original_image_url(url: str) -> str:
    """Prefer the original CDN image rather than a small resized derivative."""

    try:
        parsed = urlsplit(url)
    except ValueError:
        return url
    path = re.sub(
        r"_(?:\d+x\d+|\d+x|x\d+)(?:_crop_[^./?]+)?(?=\.[A-Za-z0-9]{2,5}$)",
        "",
        parsed.path,
        flags=re.IGNORECASE,
    )
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.casefold() not in {"width", "height", "w", "h", "crop"}
    ]
    return urlunsplit((parsed.scheme, parsed.netloc, path, urlencode(query), parsed.fragment))


def _perceptual_hash(body: bytes) -> str | None:
    """Return a 64-bit difference hash for near-duplicate technical images."""

    try:
        with Image.open(io.BytesIO(body)) as image:
            grayscale = image.convert("L").resize((9, 8))
            pixels = grayscale.tobytes()
    except Exception:
        return None
    bits = 0
    for row in range(8):
        for column in range(8):
            left = pixels[row * 9 + column]
            right = pixels[row * 9 + column + 1]
            bits = (bits << 1) | int(left > right)
    return f"{bits:016x}"


def _hash_distance(left: str | None, right: str | None) -> int:
    if not left or not right:
        return 65
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError:
        return 65


def _technical_text_score(text: str) -> float:
    folded = text.casefold()
    hits = sum(term in folded for term in _TECHNICAL_IMAGE_TERMS)
    return min(0.45, hits * 0.055)


def _ocr_similarity(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z0-9]+", left.casefold()))
    right_tokens = set(re.findall(r"[a-z0-9]+", right.casefold()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _vision_priority(
    *,
    base_score: float,
    candidate: ImageCandidate,
    ocr_text: str,
    width: int | None,
    height: int | None,
) -> float:
    text = f"{candidate.url} {candidate.alt_text or ''} {ocr_text}"
    score = base_score + _technical_text_score(text)
    if candidate.discovery_method == "pdf_page":
        score += 0.18
    if width and height:
        if width >= 1200 and height >= 700:
            score += 0.10
        if width <= 300 or height <= 180:
            score -= 0.28
    if any(term in text.casefold() for term in ("bedroom", "lifestyle", "banner", "hero")):
        score -= 0.15
    return max(0.0, min(1.0, score))


def _content_type(response, url: str) -> str:
    content_type = response.headers.get_content_type()
    if content_type == "application/octet-stream":
        suffix = Path(url.split("?", 1)[0]).suffix.casefold()
        content_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(suffix, content_type)
    return content_type


def _prepare_for_vision(body: bytes, content_type: str) -> tuple[bytes, str]:
    """Bound pixel dimensions and convert uncommon image formats to JPEG/PNG."""

    try:
        with Image.open(io.BytesIO(body)) as image:
            image.load()
            if image.width > 2200 or image.height > 2200:
                image.thumbnail((2200, 2200))
            if image.mode not in {"RGB", "L"}:
                background = Image.new("RGB", image.size, "white")
                if image.mode == "RGBA":
                    background.paste(image, mask=image.getchannel("A"))
                else:
                    background.paste(image.convert("RGB"))
                image = background
            elif image.mode == "L":
                image = image.convert("RGB")
            output = io.BytesIO()
            if content_type == "image/png" and len(body) < 4_000_000:
                image.save(output, format="PNG", optimize=True)
                return output.getvalue(), "image/png"
            image.save(output, format="JPEG", quality=88, optimize=True)
            return output.getvalue(), "image/jpeg"
    except Exception:
        return body, content_type


def _run_local_ocr(body: bytes, settings: Settings) -> tuple[str, str | None, float | None]:
    """Run optional Tesseract OCR. Failure never blocks acquisition."""

    if not settings.local_ocr_enabled:
        return "", None, None
    try:
        import pytesseract
    except ImportError:
        return "", None, None
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
    try:
        with Image.open(io.BytesIO(body)) as image:
            image.load()
            prepared = image.convert("RGB")
            data = pytesseract.image_to_data(
                prepared, output_type=pytesseract.Output.DICT, config="--psm 6"
            )
    except Exception:
        return "", None, None
    words: list[str] = []
    confidences: list[float] = []
    for text, confidence in zip(data.get("text", []), data.get("conf", [])):
        cleaned = clean_text(str(text))
        if not cleaned:
            continue
        words.append(cleaned)
        try:
            value = float(confidence)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            confidences.append(value / 100.0)
    output = clean_text(" ".join(words))
    confidence = sum(confidences) / len(confidences) if confidences else None
    return output, "tesseract", confidence


class AssetPipeline:
    def __init__(
        self,
        settings: Settings,
        object_store: ObjectStore,
        materials: MaterialLibrary,
        llm: LLMProvider,
    ) -> None:
        self.settings = settings
        self.object_store = object_store
        self.materials = materials
        self.llm = llm
        self.firecrawl = (
            FirecrawlClient(
                settings.firecrawl_api_key,
                timeout_seconds=settings.firecrawl_timeout_seconds,
                wait_ms=settings.firecrawl_wait_ms,
            )
            if settings.firecrawl_enabled and settings.firecrawl_api_key
            else None
        )

    def _download(self, candidate: ImageCandidate) -> tuple[bytes, str, int | None, int | None]:
        request = Request(
            candidate.url,
            headers={
                "User-Agent": self.settings.user_agent,
                "Accept": "image/avif,image/webp,image/png,image/jpeg,image/*;q=0.8,*/*;q=0.2",
            },
        )
        try:
            with urlopen(request, timeout=self.settings.request_timeout_seconds) as response:
                body = response.read(self.settings.maximum_image_bytes + 1)
                if len(body) > self.settings.maximum_image_bytes:
                    raise ValueError("image exceeds configured byte limit")
                content_type = _content_type(response, candidate.url)
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            raise RuntimeError(str(exc)) from exc
        if not content_type.startswith("image/"):
            raise RuntimeError(f"not an image: {content_type}")
        width = candidate.width
        height = candidate.height
        try:
            with Image.open(io.BytesIO(body)) as image:
                width, height = image.size
        except Exception:
            pass
        return body, content_type, width, height

    def _pdf_page_candidates(
        self,
        document: FetchedDocument,
        request: CompanyResearchRequest,
    ) -> list[tuple[ImageCandidate, bytes, str]]:
        try:
            import fitz
        except ImportError:
            return []
        try:
            pdf = fitz.open(stream=document.body, filetype="pdf")
        except Exception:
            return []
        results: list[tuple[ImageCandidate, bytes, str]] = []
        page_limit = min(len(pdf), request.max_pdf_pages, self.settings.max_pdf_pages_per_document)
        try:
            for page_index in range(page_limit):
                page = pdf.load_page(page_index)
                pixmap = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
                body = pixmap.tobytes("png")
                synthetic_url = f"{document.url}#page={page_index + 1}"
                results.append(
                    (
                        ImageCandidate(
                            url=synthetic_url,
                            alt_text=f"Catalogue PDF page {page_index + 1}",
                            discovery_method="pdf_page",
                            width=pixmap.width,
                            height=pixmap.height,
                        ),
                        body,
                        "image/png",
                    )
                )
        finally:
            pdf.close()
        return results

    @staticmethod
    def _manifest_candidates(document: FetchedDocument) -> list[ImageCandidate]:
        candidates: list[ImageCandidate] = []
        for item in document.asset_manifest:
            url = str(item.get("url") or "").strip()
            if not url.startswith(("http://", "https://")):
                continue
            width = item.get("width")
            height = item.get("height")
            candidates.append(
                ImageCandidate(
                    url=url,
                    alt_text=clean_text(str(item.get("alt_text") or "")) or None,
                    discovery_method=str(item.get("discovery_method") or document.capture_method),
                    width=int(width) if isinstance(width, (int, float)) else None,
                    height=int(height) if isinstance(height, (int, float)) else None,
                )
            )
        return candidates

    def _firecrawl_candidates(self, document: FetchedDocument) -> list[ImageCandidate]:
        if document.asset_manifest:
            return []
        if self.firecrawl is None or not document.url.startswith(("http://", "https://")):
            return []
        try:
            page = self.firecrawl.scrape_for_assets(document.url)
        except FirecrawlError:
            return []
        return [
            ImageCandidate(
                url=item.url,
                alt_text=item.alt_text,
                discovery_method="firecrawl",
                width=item.width,
                height=item.height,
            )
            for item in page.images
        ]

    @staticmethod
    def _dedupe_candidates(candidates: list[ImageCandidate]) -> list[ImageCandidate]:
        by_url: dict[str, ImageCandidate] = {}
        expanded: list[ImageCandidate] = []
        for candidate in candidates:
            expanded.append(candidate)
            if candidate.url.startswith(("http://", "https://")):
                original = _recover_original_image_url(candidate.url)
                if original != candidate.url:
                    expanded.append(
                        ImageCandidate(
                            url=original,
                            alt_text=candidate.alt_text,
                            discovery_method=f"{candidate.discovery_method}_original",
                            width=None,
                            height=None,
                        )
                    )
        for candidate in expanded:
            try:
                key = canonicalize_url(candidate.url)
            except ValueError:
                key = candidate.url
            previous = by_url.get(key)
            if previous is None or _candidate_score(candidate) > _candidate_score(previous):
                by_url[key] = candidate
        return sorted(by_url.values(), key=_candidate_score, reverse=True)

    def process(
        self,
        documents: list[FetchedDocument],
        sources: list[SourceRecord],
        request: CompanyResearchRequest,
        *,
        discover_followups: bool = True,
        vision_limit: int | None = None,
    ) -> AssetProcessingResult:
        result = AssetProcessingResult()
        source_by_url = {source.url: source for source in sources}
        source_by_hash = {source.content_sha256: source for source in sources}
        configured_limit = min(request.max_vision_assets, self.settings.max_vision_assets_per_run)
        vision_budget = configured_limit if vision_limit is None else min(configured_limit, vision_limit)
        prepared_assets: list[_PreparedAsset] = []

        for document in documents:
            source = source_by_url.get(document.url) or source_by_hash.get(document.sha256)
            if source is None:
                continue
            page_context = document.extracted_text(max_characters=14_000)
            prepared_candidates: list[tuple[ImageCandidate, bytes | None, str | None]] = []
            if document.is_pdf:
                prepared_candidates.extend(self._pdf_page_candidates(document, request))
            else:
                candidates = self._dedupe_candidates(
                    [
                        *self._manifest_candidates(document),
                        *_html_candidates(document),
                        *_markdown_candidates(document),
                        *self._firecrawl_candidates(document),
                    ]
                )[: min(request.max_assets_per_document, self.settings.max_assets_per_document)]
                prepared_candidates.extend((candidate, None, None) for candidate in candidates)

            for candidate, supplied_body, supplied_type in prepared_candidates:
                score = 0.72 if candidate.discovery_method == "pdf_page" else _candidate_score(candidate)
                if score < self.settings.minimum_asset_score:
                    result.log.append(
                        {
                            "source_id": source.source_id,
                            "page_url": document.url,
                            "asset_url": candidate.url,
                            "action": "rejected",
                            "reason": "asset relevance score below threshold",
                            "score": round(score, 4),
                        }
                    )
                    continue
                try:
                    if supplied_body is None:
                        body, content_type, width, height = self._download(candidate)
                    else:
                        body = supplied_body
                        content_type = supplied_type or "image/png"
                        width, height = candidate.width, candidate.height
                except RuntimeError as exc:
                    result.log.append(
                        {
                            "source_id": source.source_id,
                            "page_url": document.url,
                            "asset_url": candidate.url,
                            "action": "failed",
                            "reason": str(exc),
                            "score": round(score, 4),
                        }
                    )
                    continue

                stored = self.object_store.put_bytes(
                    body,
                    content_type=content_type,
                    source_url=candidate.url,
                    namespace="assets",
                )
                asset_id = stable_id("ast", source.source_id, stored.sha256)
                ocr_text, ocr_engine, ocr_confidence = _run_local_ocr(body, self.settings)
                priority = _vision_priority(
                    base_score=score,
                    candidate=candidate,
                    ocr_text=ocr_text,
                    width=width,
                    height=height,
                )
                asset = AssetRecord(
                    asset_id=asset_id,
                    source_id=source.source_id,
                    company_id=request.company_id,
                    page_url=document.url,
                    asset_url=candidate.url,
                    kind=(
                        AssetKind.CATALOGUE_PAGE
                        if candidate.discovery_method == "pdf_page"
                        else AssetKind.OTHER_IMAGE
                    ),
                    discovery_method=candidate.discovery_method,
                    alt_text=candidate.alt_text,
                    width=width,
                    height=height,
                    content_type=content_type,
                    content_sha256=stored.sha256,
                    local_path=stored.local_path,
                    object_uri=stored.object_uri,
                    relevance_score=score,
                    perceptual_hash=_perceptual_hash(body),
                    vision_priority=priority,
                    ocr_text=ocr_text or None,
                    ocr_engine=ocr_engine,
                    ocr_confidence=ocr_confidence,
                )
                prepared_assets.append(
                    _PreparedAsset(
                        asset=asset,
                        candidate=candidate,
                        source=source,
                        page_context=page_context,
                        body=body,
                        content_type=content_type,
                        ocr_text=ocr_text,
                    )
                )

                if ocr_text:
                    result.observations.append(
                        EvidenceObservation(
                            observation_id=stable_id("obs", asset.asset_id, "ocr_text", ocr_text[:160]),
                            source_id=source.source_id,
                            asset_id=asset.asset_id,
                            company_id=request.company_id,
                            document_url=source.url,
                            product_name_hint=None,
                            field_path="asset.ocr_text",
                            value=ocr_text,
                            method="ocr",
                            locator=f"asset {asset.asset_id}",
                            excerpt=ocr_text[:1000],
                            confidence=ocr_confidence or 0.55,
                        )
                    )
                    for mention, material_id in self.materials.find_material_mentions(ocr_text):
                        result.observations.append(
                            EvidenceObservation(
                                observation_id=stable_id(
                                    "obs", asset.asset_id, "ocr_material", material_id
                                ),
                                source_id=source.source_id,
                                asset_id=asset.asset_id,
                                company_id=request.company_id,
                                document_url=source.url,
                                product_name_hint=None,
                                field_path="material.mention",
                                value=mention,
                                normalized_material=material_id,
                                method="ocr",
                                locator=f"asset {asset.asset_id}",
                                excerpt=ocr_text[:1000],
                                confidence=ocr_confidence or 0.55,
                            )
                        )

        # Global ranking prevents early lifestyle images from consuming the complete vision budget.
        prepared_assets.sort(key=lambda item: item.asset.vision_priority, reverse=True)
        deduped: list[_PreparedAsset] = []
        for item in prepared_assets:
            duplicate_of: _PreparedAsset | None = None
            for previous in deduped:
                exact = item.asset.content_sha256 == previous.asset.content_sha256
                distance = _hash_distance(
                    item.asset.perceptual_hash,
                    previous.asset.perceptual_hash,
                )
                same_original = _recover_original_image_url(
                    item.asset.asset_url
                ) == _recover_original_image_url(previous.asset.asset_url)
                near = distance <= 2 or (
                    distance <= self.settings.visual_near_duplicate_distance
                    and _ocr_similarity(item.ocr_text, previous.ocr_text) >= 0.85
                )
                if near and item.asset.width and item.asset.height and previous.asset.width and previous.asset.height:
                    left_ratio = item.asset.width / item.asset.height
                    right_ratio = previous.asset.width / previous.asset.height
                    near = abs(left_ratio - right_ratio) <= 0.12
                if exact or same_original or near:
                    duplicate_of = previous
                    break
            if duplicate_of is not None:
                result.log.append(
                    {
                        "source_id": item.source.source_id,
                        "page_url": item.asset.page_url,
                        "asset_url": item.asset.asset_url,
                        "action": "near_duplicate",
                        "reason": f"visually duplicates {duplicate_of.asset.asset_id}",
                        "score": round(item.asset.vision_priority, 4),
                        "perceptual_hash": item.asset.perceptual_hash,
                    }
                )
                continue
            deduped.append(item)

        result.assets = [item.asset for item in deduped]
        analyzable = [
            item
            for item in deduped
            if item.asset.vision_priority >= self.settings.visual_min_priority
        ]
        if (
            request.analyze_assets_with_vision
            and self.llm.name == "openai"
            and self.settings.visual_forensics_enabled
            and vision_budget > 0
        ):
            for item in analyzable[:vision_budget]:
                vision_body, vision_type = _prepare_for_vision(item.body, item.content_type)
                try:
                    vision = self.llm.recognize_image(
                        image_bytes=vision_body,
                        content_type=vision_type,
                        source_url=item.asset.page_url,
                        page_context=item.page_context,
                    )
                    asset_type = str(vision.get("asset_type") or "other")
                    verified = False
                    if (
                        self.settings.visual_second_pass_enabled
                        and asset_type in _HIGH_VALUE_ASSET_TYPES
                    ):
                        vision = self.llm.verify_image_analysis(
                            image_bytes=vision_body,
                            content_type=vision_type,
                            source_url=item.asset.page_url,
                            page_context=item.page_context,
                            first_pass=vision,
                        )
                        verified = True
                except LLMError as exc:
                    result.warnings.append(
                        f"Vision recognition failed for {item.asset.asset_url}: {exc}"
                    )
                    continue

                confidence = float(vision.get("confidence") or 0.0)
                kind_map = {
                    "layer_diagram": AssetKind.LAYER_DIAGRAM,
                    "cutaway_or_cross_section": AssetKind.LAYER_DIAGRAM,
                    "catalogue_page": AssetKind.CATALOGUE_PAGE,
                    "specification_table": AssetKind.SPECIFICATION_TABLE,
                    "law_or_manufacturer_label": AssetKind.SPECIFICATION_TABLE,
                    "teardown_frame": AssetKind.LAYER_DIAGRAM,
                    "product_image": AssetKind.PRODUCT_IMAGE,
                }
                item.asset.kind = kind_map.get(str(vision.get("asset_type")), item.asset.kind)
                item.asset.vision_provider = self.llm.name
                item.asset.vision_model = getattr(self.llm, "model", None)
                item.asset.vision_confidence = confidence
                item.asset.vision_payload = vision
                item.asset.vision_verified = verified
                item.asset.vision_search_queries = list(
                    dict.fromkeys(
                        str(query).strip()
                        for query in vision.get("forensic_search_queries") or []
                        if str(query).strip()
                    )
                )
                self._vision_records(
                    vision=vision,
                    asset=item.asset,
                    source=item.source,
                    request=request,
                    result=result,
                )

        if discover_followups and self.settings.visual_followup_enabled and self.llm.name == "openai":
            analyses = [
                asset.vision_payload
                for asset in result.assets
                if asset.vision_payload
                and str(asset.vision_payload.get("asset_type")) in _HIGH_VALUE_ASSET_TYPES
            ]
            if analyses:
                try:
                    result.visual_followup_urls = self.llm.discover_visual_evidence(
                        company_name=request.company_name,
                        official_domain=request.official_domain,
                        market=request.market,
                        analyses=analyses,
                        limit=self.settings.visual_followup_max_pages,
                    )
                except LLMError as exc:
                    result.warnings.append(f"Visual corroboration search failed: {exc}")
                else:
                    for asset in result.assets:
                        if asset.vision_payload and asset.vision_search_queries:
                            asset.vision_followup_urls = list(result.visual_followup_urls)

        for item in deduped:
            result.log.append(
                {
                    "source_id": item.source.source_id,
                    "page_url": item.asset.page_url,
                    "asset_id": item.asset.asset_id,
                    "asset_url": item.asset.asset_url,
                    "action": "stored",
                    "discovery_method": item.candidate.discovery_method,
                    "kind": item.asset.kind,
                    "score": round(item.asset.relevance_score, 4),
                    "vision_priority": round(item.asset.vision_priority, 4),
                    "perceptual_hash": item.asset.perceptual_hash,
                    "vision_analyzed": item.asset.vision_payload is not None,
                    "vision_verified": item.asset.vision_verified,
                    "ocr_engine": item.asset.ocr_engine,
                    "ocr_characters": len(item.asset.ocr_text or ""),
                    "local_path": item.asset.local_path,
                    "object_uri": item.asset.object_uri,
                }
            )
        return result

    def _vision_records(
        self,
        *,
        vision: dict,
        asset: AssetRecord,
        source: SourceRecord,
        request: CompanyResearchRequest,
        result: AssetProcessingResult,
    ) -> None:
        confidence = float(vision.get("confidence") or 0.0)
        summary = clean_text(str(vision.get("diagram_summary") or ""))
        visible_text = clean_text(str(vision.get("visible_text") or ""))
        if summary:
            result.observations.append(
                EvidenceObservation(
                    observation_id=stable_id("obs", asset.asset_id, "diagram_summary", summary),
                    source_id=source.source_id,
                    asset_id=asset.asset_id,
                    company_id=request.company_id,
                    document_url=source.url,
                    field_path="visual.diagram_summary",
                    value=summary,
                    method="vision",
                    locator=f"asset {asset.asset_id}",
                    excerpt=summary[:1000],
                    confidence=confidence,
                )
            )
        if visible_text:
            result.observations.append(
                EvidenceObservation(
                    observation_id=stable_id("obs", asset.asset_id, "vision_text", visible_text[:200]),
                    source_id=source.source_id,
                    asset_id=asset.asset_id,
                    company_id=request.company_id,
                    document_url=source.url,
                    field_path="visual.visible_text",
                    value=visible_text,
                    method="vision",
                    locator=f"asset {asset.asset_id}",
                    excerpt=visible_text[:1000],
                    confidence=confidence,
                )
            )
        for term in vision.get("technology_terms") or []:
            cleaned_term = clean_text(str(term))
            if not cleaned_term:
                continue
            result.observations.append(
                EvidenceObservation(
                    observation_id=stable_id("obs", asset.asset_id, "technology", cleaned_term),
                    source_id=source.source_id,
                    asset_id=asset.asset_id,
                    company_id=request.company_id,
                    document_url=source.url,
                    field_path="visual.technology_term",
                    value=cleaned_term,
                    normalized_material=self.materials.normalize(cleaned_term),
                    method="vision",
                    locator=f"asset {asset.asset_id}",
                    excerpt=cleaned_term,
                    confidence=confidence,
                )
            )

        for raw_product in vision.get("products") or []:
            if not isinstance(raw_product, dict):
                continue
            name = clean_text(str(raw_product.get("name") or ""))
            specific = bool(raw_product.get("is_specific_model"))
            layers: list[LayerRecord] = []
            for index, raw_layer in enumerate(raw_product.get("layers") or [], start=1):
                if not isinstance(raw_layer, dict):
                    continue
                position = int(raw_layer.get("position") or index)
                visible_label = clean_text(str(raw_layer.get("visible_label") or ""))
                marketing_name = clean_text(
                    str(raw_layer.get("marketing_name") or visible_label or "")
                )
                generic_class = clean_text(str(raw_layer.get("generic_material_class") or "unknown"))
                normalized = clean_text(str(raw_layer.get("normalized_material") or ""))
                normalized = self.materials.normalize(normalized or marketing_name or generic_class)
                evidence_status = str(raw_layer.get("evidence_status") or "unknown")
                assignment_scope = str(raw_layer.get("assignment_scope") or "ambiguous")
                layer_confidence = max(
                    0.0,
                    min(1.0, float(raw_layer.get("confidence") or confidence)),
                )
                region = raw_layer.get("region")
                locator = (
                    f"asset {asset.asset_id}; visual position {position}; "
                    f"scope={assignment_scope}; region={region}"
                )
                excerpt = clean_text(
                    str(raw_layer.get("callout_text") or visible_label or marketing_name)
                )
                thickness = raw_layer.get("thickness_mm")
                density = raw_layer.get("density_kg_m3")
                explicit_measurement = evidence_status == "observed_measurement"
                admitted_component = (
                    bool(marketing_name)
                    and evidence_status in {"observed_label", "observed_measurement"}
                    and assignment_scope in {"exact_layer", "layer_zone"}
                )
                if admitted_component:
                    evidence = EvidenceRef(
                        source_id=source.source_id,
                        asset_id=asset.asset_id,
                        locator=locator,
                        excerpt=excerpt or marketing_name,
                        reliability=source.reliability,
                    )
                    layers.append(
                        LayerRecord(
                            position=position,
                            marketing_name=marketing_name,
                            normalized_material=normalized,
                            thickness_mm=(
                                float(thickness)
                                if explicit_measurement and thickness is not None and float(thickness) > 0
                                else None
                            ),
                            density_kg_m3=(
                                float(density)
                                if explicit_measurement and density is not None and float(density) > 0
                                else None
                            ),
                            thickness_status=(
                                ClaimStatus.OBSERVED
                                if explicit_measurement and thickness is not None
                                else ClaimStatus.UNKNOWN
                            ),
                            density_status=(
                                ClaimStatus.OBSERVED
                                if explicit_measurement and density is not None
                                else ClaimStatus.UNKNOWN
                            ),
                            evidence=[evidence],
                        )
                    )

                visual_value = {
                    "position": position,
                    "marketing_name": marketing_name or None,
                    "generic_material_class": generic_class,
                    "normalized_material": normalized,
                    "visible_label": visible_label or None,
                    "callout_text": raw_layer.get("callout_text"),
                    "assignment_scope": assignment_scope,
                    "evidence_status": evidence_status,
                    "region": region,
                    "thickness_mm": thickness if explicit_measurement else None,
                    "density_kg_m3": density if explicit_measurement else None,
                }
                result.observations.append(
                    EvidenceObservation(
                        observation_id=stable_id(
                            "obs", asset.asset_id, name or "unresolved", "visual_layer", position
                        ),
                        source_id=source.source_id,
                        asset_id=asset.asset_id,
                        company_id=request.company_id,
                        document_url=source.url,
                        product_name_hint=name or None,
                        field_path=f"visual.layer[{position}]",
                        value=visual_value,
                        normalized_material=normalized,
                        method="vision",
                        locator=locator,
                        excerpt=excerpt[:1000] if excerpt else None,
                        confidence=layer_confidence,
                    )
                )

            if specific and name and confidence >= self.settings.vision_recognition_threshold:
                total_thickness = raw_product.get("total_thickness_mm")
                product_weight = raw_product.get("product_weight_kg")
                price = raw_product.get("price")
                product = ProductRecord(
                    company_id=request.company_id,
                    company_name=request.company_name,
                    brand=clean_text(str(raw_product.get("brand") or request.company_name)),
                    name=name,
                    family=clean_text(str(raw_product.get("family") or "")) or None,
                    canonical_url=source.url,
                    description=clean_text(str(raw_product.get("description") or "")),
                    firmness=clean_text(str(raw_product.get("firmness") or "")) or None,
                    total_thickness_mm=(
                        float(total_thickness)
                        if total_thickness is not None and float(total_thickness) > 0
                        else None
                    ),
                    product_weight_kg=(
                        float(product_weight)
                        if product_weight is not None and float(product_weight) > 0
                        else None
                    ),
                    price=float(price) if price is not None and float(price) >= 0 else None,
                    currency=clean_text(str(raw_product.get("currency") or "")) or None,
                    layers=layers,
                    source_ids=[source.source_id],
                    tags=[
                        "vision_observed",
                        f"asset:{asset.asset_id}",
                        f"visual_type:{vision.get('asset_type')}",
                    ],
                    extraction_method="vision",
                    extraction_confidence=confidence,
                )
                result.products.append(product)

        for index, region in enumerate(vision.get("unassigned_regions") or [], start=1):
            if not isinstance(region, dict):
                continue
            visual_description = clean_text(str(region.get("visual_description") or ""))
            if not visual_description:
                continue
            result.observations.append(
                EvidenceObservation(
                    observation_id=stable_id(
                        "obs", asset.asset_id, "unassigned_region", index, visual_description
                    ),
                    source_id=source.source_id,
                    asset_id=asset.asset_id,
                    company_id=request.company_id,
                    document_url=source.url,
                    field_path=f"visual.unassigned_region[{index}]",
                    value={
                        "position": region.get("position") or index,
                        "visual_description": visual_description,
                        "generic_material_class": region.get("generic_material_class") or "unknown",
                        "region": region.get("region"),
                    },
                    normalized_material=self.materials.normalize(visual_description),
                    method="vision",
                    locator=f"asset {asset.asset_id}; unassigned visual region {index}",
                    excerpt=visual_description[:1000],
                    confidence=max(
                        0.0,
                        min(1.0, float(region.get("confidence") or confidence)),
                    ),
                )
            )
