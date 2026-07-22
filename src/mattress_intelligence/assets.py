"""Image/PDF asset discovery and GPT vision transcription."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
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
        for candidate in candidates:
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
    ) -> AssetProcessingResult:
        result = AssetProcessingResult()
        source_by_url = {source.url: source for source in sources}
        source_by_hash = {source.content_sha256: source for source in sources}
        vision_remaining = min(request.max_vision_assets, self.settings.max_vision_assets_per_run)

        for document in documents:
            source = source_by_url.get(document.url) or source_by_hash.get(document.sha256)
            if source is None:
                continue
            page_context = document.extracted_text(max_characters=12_000)
            prepared: list[tuple[ImageCandidate, bytes | None, str | None]] = []
            if document.is_pdf:
                prepared.extend(self._pdf_page_candidates(document, request))
            else:
                candidates = self._dedupe_candidates(
                    [
                        *self._manifest_candidates(document),
                        *_html_candidates(document),
                        *_markdown_candidates(document),
                        *self._firecrawl_candidates(document),
                    ]
                )[: min(request.max_assets_per_document, self.settings.max_assets_per_document)]
                prepared.extend((candidate, None, None) for candidate in candidates)

            for candidate, supplied_body, supplied_type in prepared:
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
                )

                ocr_text, ocr_engine, ocr_confidence = _run_local_ocr(body, self.settings)
                asset.ocr_text = ocr_text or None
                asset.ocr_engine = ocr_engine
                asset.ocr_confidence = ocr_confidence
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
                                observation_id=stable_id("obs", asset.asset_id, "ocr_material", material_id),
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

                should_analyze = (
                    request.analyze_assets_with_vision
                    and self.llm.name == "openai"
                    and vision_remaining > 0
                    and (
                        score >= self.settings.vision_recognition_threshold
                        or document.is_pdf
                        or len(ocr_text) < self.settings.ocr_min_characters
                    )
                )
                if should_analyze:
                    vision_body, vision_type = _prepare_for_vision(body, content_type)
                    try:
                        vision = self.llm.recognize_image(
                            image_bytes=vision_body,
                            content_type=vision_type,
                            source_url=document.url,
                            page_context=page_context,
                        )
                    except LLMError as exc:
                        result.warnings.append(
                            f"Vision recognition failed for {candidate.url}: {exc}"
                        )
                    else:
                        vision_remaining -= 1
                        confidence = float(vision.get("confidence") or 0.0)
                        kind_map = {
                            "layer_diagram": AssetKind.LAYER_DIAGRAM,
                            "catalogue_page": AssetKind.CATALOGUE_PAGE,
                            "specification_table": AssetKind.SPECIFICATION_TABLE,
                            "product_image": AssetKind.PRODUCT_IMAGE,
                        }
                        asset.kind = kind_map.get(str(vision.get("asset_type")), asset.kind)
                        asset.vision_provider = self.llm.name
                        asset.vision_model = getattr(self.llm, "model", None)
                        asset.vision_confidence = confidence
                        asset.vision_payload = vision
                        self._vision_records(
                            vision=vision,
                            asset=asset,
                            source=source,
                            request=request,
                            result=result,
                        )

                result.assets.append(asset)
                result.log.append(
                    {
                        "source_id": source.source_id,
                        "page_url": document.url,
                        "asset_id": asset.asset_id,
                        "asset_url": candidate.url,
                        "action": "stored",
                        "discovery_method": candidate.discovery_method,
                        "kind": asset.kind,
                        "score": round(score, 4),
                        "vision_analyzed": asset.vision_payload is not None,
                        "ocr_engine": asset.ocr_engine,
                        "ocr_characters": len(asset.ocr_text or ""),
                        "local_path": asset.local_path,
                        "object_uri": asset.object_uri,
                    }
                )

        # Deduplicate byte-identical images while preserving the strongest record.
        deduped: dict[str, AssetRecord] = {}
        for asset in result.assets:
            previous = deduped.get(asset.content_sha256)
            if previous is None or asset.relevance_score > previous.relevance_score:
                deduped[asset.content_sha256] = asset
        result.assets = list(deduped.values())
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
        for raw_product in vision.get("products") or []:
            if not isinstance(raw_product, dict):
                continue
            name = clean_text(str(raw_product.get("name") or ""))
            specific = bool(raw_product.get("is_specific_model"))
            layers: list[LayerRecord] = []
            for index, raw_layer in enumerate(raw_product.get("layers") or [], start=1):
                if not isinstance(raw_layer, dict):
                    continue
                marketing_name = clean_text(
                    str(raw_layer.get("marketing_name") or raw_layer.get("visible_label") or "")
                )
                if not marketing_name:
                    continue
                normalized = clean_text(str(raw_layer.get("normalized_material") or ""))
                normalized = self.materials.normalize(normalized or marketing_name)
                evidence = EvidenceRef(
                    source_id=source.source_id,
                    asset_id=asset.asset_id,
                    locator=f"image asset {asset.asset_id}; visible layer {index}",
                    excerpt=clean_text(str(raw_layer.get("visible_label") or marketing_name)),
                    reliability=source.reliability,
                )
                thickness = raw_layer.get("thickness_mm")
                density = raw_layer.get("density_kg_m3")
                layers.append(
                    LayerRecord(
                        position=int(raw_layer.get("position") or index),
                        marketing_name=marketing_name,
                        normalized_material=normalized,
                        thickness_mm=float(thickness) if thickness else None,
                        density_kg_m3=float(density) if density else None,
                        thickness_status=(
                            ClaimStatus.OBSERVED if thickness else ClaimStatus.UNKNOWN
                        ),
                        density_status=(ClaimStatus.OBSERVED if density else ClaimStatus.UNKNOWN),
                        evidence=[evidence],
                    )
                )
                for field_path, value, unit in (
                    (f"layer[{index}].marketing_name", marketing_name, None),
                    (f"layer[{index}].thickness_mm", thickness, "mm"),
                    (f"layer[{index}].density_kg_m3", density, "kg/m3"),
                ):
                    if value is None:
                        continue
                    result.observations.append(
                        EvidenceObservation(
                            observation_id=stable_id(
                                "obs", asset.asset_id, name or "unresolved", field_path, value
                            ),
                            source_id=source.source_id,
                            asset_id=asset.asset_id,
                            company_id=request.company_id,
                            document_url=source.url,
                            product_name_hint=name or None,
                            field_path=field_path,
                            value=value,
                            unit=unit,
                            normalized_material=normalized,
                            method="vision",
                            locator=f"asset {asset.asset_id}",
                            excerpt=evidence.excerpt,
                            confidence=confidence,
                        )
                    )

            if specific and name and confidence >= self.settings.vision_recognition_threshold:
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
                        float(raw_product["total_thickness_mm"])
                        if raw_product.get("total_thickness_mm")
                        else None
                    ),
                    product_weight_kg=(
                        float(raw_product["product_weight_kg"])
                        if raw_product.get("product_weight_kg")
                        else None
                    ),
                    price=float(raw_product["price"]) if raw_product.get("price") else None,
                    currency=clean_text(str(raw_product.get("currency") or "")) or None,
                    layers=layers,
                    source_ids=[source.source_id],
                    tags=["vision_observed", f"asset:{asset.asset_id}"],
                    extraction_method="vision",
                    extraction_confidence=confidence,
                )
                result.products.append(product)
