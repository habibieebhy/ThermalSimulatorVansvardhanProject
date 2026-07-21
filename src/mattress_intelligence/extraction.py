"""Deterministic-first product, observation, and evidence extraction."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import lxml.etree as etree
import lxml.html as html

from .crawler import FetchedDocument
from .llm import LLMError, LLMProvider
from .materials import MaterialLibrary
from .models import (
    ClaimRecord,
    ClaimStatus,
    CompanyResearchRequest,
    EvidenceObservation,
    EvidenceRef,
    LayerRecord,
    ProductRecord,
    SourceKind,
    SourceRecord,
    VariantRecord,
    stable_id,
)
from .normalization import clean_text, length_to_mm, parse_density_kg_m3, parse_first_thickness_mm


_PRICE_RE = re.compile(
    r"(?P<currency>₹|INR\s*|Rs\.?\s*)(?P<value>\d{2,3}(?:[,.]\d{2,3})*(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
_WEIGHT_RE = re.compile(
    r"(?:net\s+|gross\s+|product\s+)?weight\s*(?:is|:|-)?\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>kg|kilograms?|kgs?)\b",
    re.IGNORECASE,
)
_WARRANTY_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>years?|yrs?)\s+(?:limited\s+)?warranty\b",
    re.IGNORECASE,
)
_DIMENSION_RE = re.compile(
    r"(?P<a>\d+(?:\.\d+)?)\s*[x×]\s*(?P<b>\d+(?:\.\d+)?)"
    r"(?:\s*[x×]\s*(?P<c>\d+(?:\.\d+)?))?\s*"
    r"(?P<unit>mm|cm|inches|inch|in|\")",
    re.IGNORECASE,
)
_DENSITY_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?:kg\s*/\s*m(?:\^?3|³)|kgm(?:\^?3|³)|kg\s+per\s+cubic\s+met(?:er|re))",
    re.IGNORECASE,
)
_THICKNESS_PATTERNS = (
    re.compile(
        r"(?:total\s+)?(?:thickness|height|depth)\s*(?:is|:|-)?\s*"
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|inches|inch|in|\")",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|inches|inch|in|\")\s+"
        r"(?:thick\s+)?(?:mattress|layer|foam|latex|core|pad|sheet)\b",
        re.IGNORECASE,
    ),
)
_FIRMNESS_LABELS = (
    "extra firm",
    "very firm",
    "medium firm",
    "medium-firm",
    "medium soft",
    "medium-soft",
    "plush",
    "soft",
    "medium",
    "firm",
)
_FIRMNESS_RE = re.compile(
    r"(?<![a-z])(?:extra firm|very firm|medium firm|medium-firm|medium soft|medium-soft|plush|soft|medium|firm)(?![a-z])",
    re.IGNORECASE,
)
_PRODUCT_NAME_RE = re.compile(
    r"\b(?P<name>[A-Z][A-Za-z0-9&+™®'’().\-/ ]{2,80}?\bMattress)\b"
)

_NON_PRODUCT_PATH_RE = re.compile(
    r"/(?:collections?|pages?)/(?:mattress-in-|mattress-for-|mattress-store-in-|"
    r"mattress-category|recliners?|bedsheets?|pillows?|chairs?|sofas?|stores?)(?:/|$)",
    re.IGNORECASE,
)
_GENERIC_PRODUCT_NAME_RE = re.compile(
    r"^(?:mattresses?|mattress category products?|best mattresses?|"
    r"mattress (?:in|for|near)\b|buy mattresses? online\b|mattress store\b)",
    re.IGNORECASE,
)
_PRODUCT_DETAIL_PATH_RE = re.compile(r"/(?:products?|shop|mattress(?:es)?)/[^/?#]+", re.IGNORECASE)


def _is_non_product_url(url: str) -> bool:
    path = urlsplit(url).path.casefold()
    return bool(_NON_PRODUCT_PATH_RE.search(path)) or any(
        token in path
        for token in (
            "/blogs/",
            "/blog/",
            "/pages/store",
            "/pages/mattress-store",
            "/collections/all",
        )
    )


def _is_specific_product_name(name: str | None) -> bool:
    cleaned = clean_text(name or "")
    if not cleaned or _GENERIC_PRODUCT_NAME_RE.search(cleaned):
        return False
    lowered = cleaned.casefold()
    if lowered in {"product", "products", "mattress", "mattresses", "shop"}:
        return False
    return len(cleaned.split()) >= 2 or any(char.isdigit() for char in cleaned)


def _explicit_total_thickness_mm(text: str, product_name: str | None = None) -> float | None:
    """Extract product thickness only from explicit thickness/height/depth context.

    This intentionally ignores free-floating expressions such as "5 inch mattress" that often
    occur in navigation, recommendations, size filters, and comparison copy.
    """

    scope = clean_text(text)[:30_000]
    for match in _THICKNESS_PATTERNS[0].finditer(scope):
        try:
            value = length_to_mm(float(match.group("value")), match.group("unit"))
        except (TypeError, ValueError):
            continue
        if 30 <= value <= 600:
            return value
    if product_name:
        escaped = re.escape(clean_text(product_name))
        named = re.search(
            rf"{escaped}.{{0,100}}?(?P<value>\d+(?:\.\d+)?)\s*"
            rf"(?P<unit>mm|cm|inches|inch|in|\")\s*(?:thick|height|depth)",
            scope,
            re.IGNORECASE,
        )
        if named:
            value = length_to_mm(float(named.group("value")), named.group("unit"))
            if 30 <= value <= 600:
                return value
    return None


def _iter_json_objects(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_json_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_json_objects(child)


def _json_ld_objects(tree: html.HtmlElement) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    query = (
        "//script[contains(translate(@type,'ABCDEFGHIJKLMNOPQRSTUVWXYZ',"
        "'abcdefghijklmnopqrstuvwxyz'),'ld+json')]/text()"
    )
    for text in tree.xpath(query):
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            continue
        objects.extend(_iter_json_objects(parsed))
    return objects


def _type_matches(value: Any, expected: str) -> bool:
    if isinstance(value, str):
        return value.casefold() == expected.casefold()
    if isinstance(value, list):
        return any(_type_matches(item, expected) for item in value)
    return False


def _text_content(tree: html.HtmlElement) -> str:
    for node in tree.xpath("//script|//style|//noscript|//svg"):
        parent = node.getparent()
        if parent is not None:
            parent.remove(node)
    return clean_text(" ".join(tree.xpath("//body//text()")))[:200_000]


def _meta(tree: html.HtmlElement, *names: str) -> str | None:
    lowered = [name.casefold() for name in names]
    for element in tree.xpath("//meta[@content]"):
        key = (element.get("property") or element.get("name") or "").casefold()
        if key in lowered:
            value = clean_text(element.get("content"))
            if value:
                return value
    return None


def _source_kind(document: FetchedDocument, official_domain: str) -> tuple[SourceKind, bool, float]:
    parsed_official = urlsplit(
        official_domain if urlsplit(official_domain).scheme else f"https://{official_domain}"
    )
    official_host = (parsed_official.hostname or "").casefold().removeprefix("www.")
    source_host = (urlsplit(document.url).hostname or "").casefold().removeprefix("www.")
    official = source_host == official_host or source_host.endswith(f".{official_host}")
    lowered = document.url.casefold()
    if official and lowered.endswith(".pdf"):
        return SourceKind.OFFICIAL_CATALOGUE, True, 0.95
    if official and _PRODUCT_DETAIL_PATH_RE.search(urlsplit(document.url).path) and not _is_non_product_url(document.url):
        return SourceKind.OFFICIAL_PRODUCT, True, 0.95
    if official:
        return SourceKind.OFFICIAL_OTHER, True, 0.85
    if "patent" in lowered:
        return SourceKind.PATENT, False, 0.85
    if any(token in lowered for token in ("review", "retailer", "store", "shop")):
        return SourceKind.RETAILER, False, 0.60
    if any(token in lowered for token in ("teardown", "cut-open", "cutopen")):
        return SourceKind.TEARDOWN, False, 0.80
    return SourceKind.OTHER, False, 0.50


def source_from_document(
    document: FetchedDocument,
    request: CompanyResearchRequest,
    title: str | None = None,
) -> SourceRecord:
    kind, official, reliability = _source_kind(document, request.official_domain)
    return SourceRecord(
        source_id=stable_id("src", document.url, document.sha256),
        company_id=request.company_id,
        url=document.url,
        title=title,
        kind=kind,
        is_official=official,
        reliability=reliability,
        retrieved_at=datetime.fromtimestamp(document.retrieved_at_epoch, tz=timezone.utc),
        content_sha256=document.sha256,
        artifact_path=document.artifact_path,
        http_status=document.status,
        content_type=document.content_type,
    )


def _context(text: str, start: int, end: int, radius: int = 150) -> str:
    return clean_text(text[max(0, start - radius) : min(len(text), end + radius)])


def _url_name_hint(url: str) -> str | None:
    stem = Path(unquote(urlsplit(url).path)).stem
    candidate = clean_text(re.sub(r"[-_]+", " ", stem))
    if not candidate or candidate.casefold() in {"index", "product", "products", "catalogue", "catalog"}:
        return None
    return candidate.title()


def _first_price(text: str) -> tuple[float | None, str | None]:
    match = _PRICE_RE.search(text)
    if not match:
        return None, None
    raw = match.group("value").replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return None, None
    token = match.group("currency").strip().casefold()
    currency = "INR" if token in {"₹", "inr", "rs", "rs."} else token.upper()
    return value, currency


def _first_weight_kg(text: str) -> float | None:
    match = _WEIGHT_RE.search(text)
    return float(match.group("value")) if match else None


def _first_firmness(text: str) -> str | None:
    match = _FIRMNESS_RE.search(text)
    return match.group(0).replace("-", " ").title() if match else None


def _variants_from_text(text: str, source_id: str, limit: int = 30) -> list[VariantRecord]:
    variants: list[VariantRecord] = []
    seen: set[tuple[int, int, int | None]] = set()
    for match in _DIMENSION_RE.finditer(text):
        unit = match.group("unit")
        values = [float(match.group("a")), float(match.group("b"))]
        third = float(match.group("c")) if match.group("c") else None
        converted = [int(round(length_to_mm(value, unit))) for value in values]
        thickness = int(round(length_to_mm(third, unit))) if third is not None else None
        width, length = sorted(converted)
        key = (width, length, thickness)
        if key in seen:
            continue
        seen.add(key)
        variants.append(
            VariantRecord(
                size_name=clean_text(match.group(0)),
                width_mm=width,
                length_mm=length,
                thickness_mm=thickness,
                source_ids=[source_id],
            )
        )
        if len(variants) >= limit:
            break
    return variants


class ProductExtractor:
    """Extract explicit facts with strict product admission and optional LLM recognition."""

    def __init__(
        self,
        materials: MaterialLibrary,
        llm: LLMProvider,
        recognition_threshold: float = 0.68,
    ) -> None:
        self.materials = materials
        self.llm = llm
        self.recognition_threshold = recognition_threshold
        self.warnings: list[str] = []
        self.recognition_log: list[dict[str, Any]] = []

    def extract(
        self,
        document: FetchedDocument,
        request: CompanyResearchRequest,
    ) -> tuple[ProductRecord | None, SourceRecord]:
        products, source = self.extract_many(document, request)
        return (products[0] if products else None), source

    def extract_many(
        self,
        document: FetchedDocument,
        request: CompanyResearchRequest,
    ) -> tuple[list[ProductRecord], SourceRecord]:
        products, source, _ = self.extract_document(document, request)
        return products, source

    def extract_document(
        self,
        document: FetchedDocument,
        request: CompanyResearchRequest,
    ) -> tuple[list[ProductRecord], SourceRecord, list[EvidenceObservation]]:
        """Return admitted products, source metadata, and atomic evidence observations."""

        if not document.is_html:
            title = _url_name_hint(document.url)
            source = source_from_document(document, request, title)
            try:
                document_text = document.extracted_text()
            except Exception as exc:
                self.warnings.append(f"Text extraction failed for {document.url}: {exc}")
                self._record_recognition(
                    source,
                    document_type="other",
                    confidence=0.0,
                    deterministic_count=0,
                    llm_count=0,
                    admitted_count=0,
                    rejection_reason=str(exc),
                    document_warnings=[],
                )
                return [], source, []
            observations = self._observations_from_text(
                document_text,
                source,
                request,
                product_name_hint=title,
                locator="document text",
            )
            deterministic = self._from_text_document(document_text, source, request)
            llm_products, recognition = self._llm_recognition(
                document.url,
                document_text,
                source,
                request,
                allow_llm=bool(document_text),
            )
            admitted = self._merge_product_lists(deterministic, llm_products)
            self._record_recognition(
                source,
                document_type=str(
                    recognition.get("document_type")
                    or ("catalogue" if document.is_pdf else "other")
                ),
                confidence=float(recognition.get("recognition_confidence") or 0.0),
                deterministic_count=len(deterministic),
                llm_count=len(llm_products),
                admitted_count=len(admitted),
                rejection_reason=recognition.get("rejection_reason"),
                document_warnings=list(recognition.get("document_warnings") or []),
            )
            return admitted, source, observations

        try:
            tree = html.fromstring(document.body, base_url=document.url)
        except (etree.ParserError, ValueError):
            source = source_from_document(document, request)
            self._record_recognition(
                source,
                document_type="other",
                confidence=0.0,
                deterministic_count=0,
                llm_count=0,
                admitted_count=0,
                rejection_reason="HTML parsing failed.",
                document_warnings=[],
            )
            return [], source, []

        title = clean_text(" ".join(tree.xpath("//title/text()"))) or None
        h1 = clean_text(" ".join(tree.xpath("//h1[1]//text()"))) or None
        source = source_from_document(document, request, title)
        all_json_ld = _json_ld_objects(tree)
        product_objects = [
            obj for obj in all_json_ld if _type_matches(obj.get("@type"), "Product")
        ]
        page_text = _text_content(tree)

        json_ld_products = [
            product
            for obj in product_objects
            if (product := self._from_json_ld(obj, source, request)) is not None
        ]
        heuristic_product = self._from_html(tree, page_text, source, request)
        deterministic = list(json_ld_products)
        if heuristic_product is not None:
            deterministic = self._merge_product_lists(deterministic, [heuristic_product])

        observations = self._observations_from_text(
            page_text,
            source,
            request,
            product_name_hint=h1 or title,
            locator="page text",
        )
        observations.extend(
            self._observations_from_html(tree, product_objects, source, request, h1 or title)
        )
        observations = self._deduplicate_observations(observations)

        non_product_url = _is_non_product_url(document.url)
        allow_llm = (
            self.llm.name != "none"
            and not non_product_url
            and (bool(json_ld_products) or self._looks_relevant(document.url, page_text, request))
        )
        llm_products, recognition = self._llm_recognition(
            document.url,
            page_text,
            source,
            request,
            allow_llm=allow_llm,
        )

        document_type = str(
            recognition.get("document_type")
            or self._deterministic_document_type(document.url, bool(json_ld_products))
        )
        recognition_confidence = float(recognition.get("recognition_confidence") or 0.0)
        llm_admits_document = (
            not allow_llm
            or (
                bool(recognition.get("is_product_bearing"))
                and recognition_confidence >= self.recognition_threshold
            )
        )

        # Explicit Product JSON-LD is retained. Heuristic records require a strict product-detail
        # URL, and an active recognizer may veto them. This prevents collection/location pages
        # from becoming products while preserving atomic observations from those pages.
        admitted_deterministic = list(json_ld_products)
        if heuristic_product is not None and llm_admits_document:
            admitted_deterministic = self._merge_product_lists(
                admitted_deterministic, [heuristic_product]
            )
        admitted = self._merge_product_lists(admitted_deterministic, llm_products)

        rejection_reason = recognition.get("rejection_reason")
        if non_product_url and not json_ld_products:
            rejection_reason = rejection_reason or "Rejected by deterministic non-product URL gate."
        elif heuristic_product is not None and not llm_admits_document:
            rejection_reason = rejection_reason or "LLM recognition vetoed heuristic product admission."

        self._record_recognition(
            source,
            document_type=document_type,
            confidence=recognition_confidence,
            deterministic_count=len(deterministic),
            llm_count=len(llm_products),
            admitted_count=len(admitted),
            rejection_reason=rejection_reason,
            document_warnings=list(recognition.get("document_warnings") or []),
        )
        return admitted, source, observations

    def _llm_recognition(
        self,
        url: str,
        text: str,
        source: SourceRecord,
        request: CompanyResearchRequest,
        *,
        allow_llm: bool,
    ) -> tuple[list[ProductRecord], dict[str, Any]]:
        if not text or self.llm.name == "none" or not allow_llm:
            return [], {
                "document_type": None,
                "is_product_bearing": False,
                "recognition_confidence": 0.0,
                "rejection_reason": (
                    "LLM disabled."
                    if self.llm.name == "none"
                    else "LLM skipped by deterministic relevance/admission gate."
                ),
                "document_warnings": [],
            }
        try:
            recognition = self.llm.recognize_document(url, text)
        except LLMError as exc:
            self.warnings.append(f"LLM recognition failed for {url}: {exc}")
            return [], {
                "document_type": "other",
                "is_product_bearing": False,
                "recognition_confidence": 0.0,
                "rejection_reason": str(exc),
                "document_warnings": [],
            }
        products = [
            product
            for raw in recognition.get("products") or []
            if (product := self._from_llm(raw, source, request)) is not None
        ]
        return products, recognition

    def _record_recognition(
        self,
        source: SourceRecord,
        *,
        document_type: str,
        confidence: float,
        deterministic_count: int,
        llm_count: int,
        admitted_count: int,
        rejection_reason: Any,
        document_warnings: list[Any],
    ) -> None:
        self.recognition_log.append(
            {
                "source_id": source.source_id,
                "url": source.url,
                "provider": self.llm.name,
                "model": getattr(self.llm, "model", None),
                "document_type": document_type,
                "recognition_confidence": round(max(0.0, min(1.0, confidence)), 4),
                "deterministic_product_count": deterministic_count,
                "llm_product_count": llm_count,
                "admitted_product_count": admitted_count,
                "accepted": admitted_count > 0,
                "rejection_reason": clean_text(str(rejection_reason or "")) or None,
                "warnings": " | ".join(clean_text(str(item)) for item in document_warnings if item),
            }
        )

    @staticmethod
    def _deterministic_document_type(url: str, has_product_json_ld: bool) -> str:
        lowered = url.casefold()
        if _is_non_product_url(url):
            if "mattress-in-" in lowered:
                return "location_page"
            if "store" in lowered:
                return "store_page"
            if "/blog" in lowered:
                return "blog_or_guide"
            return "collection"
        if has_product_json_ld or _PRODUCT_DETAIL_PATH_RE.search(urlsplit(url).path):
            return "product_detail"
        if lowered.endswith(".pdf"):
            return "catalogue"
        return "other"

    @staticmethod
    def _looks_like_product(url: str, text: str) -> bool:
        if _is_non_product_url(url):
            return False
        path = urlsplit(url).path
        if not _PRODUCT_DETAIL_PATH_RE.search(path):
            return False
        lowered_text = text[:20_000].casefold()
        mattress_signal = "mattress" in lowered_text
        commerce_signal = any(
            token in lowered_text
            for token in (
                "add to cart",
                "buy now",
                "sku",
                "price",
                "choose size",
                "select size",
                "warranty",
                "product details",
            )
        )
        return mattress_signal and commerce_signal

    @classmethod
    def _looks_relevant(
        cls,
        url: str,
        text: str,
        request: CompanyResearchRequest,
    ) -> bool:
        if cls._looks_like_product(url, text):
            return True
        lowered = text[:50_000].casefold()
        company_terms = [request.company_name, *request.brand_aliases]
        company_signal = any(term.casefold() in lowered for term in company_terms if term.strip())
        mattress_signal = "mattress" in lowered
        evidence_signal = any(
            token in lowered
            for token in (
                "density",
                "kg/m",
                "layer",
                "foam",
                "latex",
                "thickness",
                "firmness",
                "construction",
                "specification",
                "warranty",
            )
        )
        return mattress_signal and company_signal and evidence_signal

    @staticmethod
    def _brand(value: Any, fallback: str) -> str:
        if isinstance(value, dict):
            value = value.get("name")
        return clean_text(str(value or fallback))

    @staticmethod
    def _offers(value: Any) -> tuple[float | None, str | None]:
        offers = value[0] if isinstance(value, list) and value else value
        if not isinstance(offers, dict):
            return None, None
        raw_price = offers.get("price") or offers.get("lowPrice")
        try:
            price = float(str(raw_price).replace(",", "")) if raw_price is not None else None
        except ValueError:
            price = None
        currency = clean_text(str(offers.get("priceCurrency") or "")) or None
        return price, currency

    def _from_json_ld(
        self,
        obj: dict[str, Any],
        source: SourceRecord,
        request: CompanyResearchRequest,
    ) -> ProductRecord | None:
        name = clean_text(str(obj.get("name") or ""))
        if not _is_specific_product_name(name):
            return None
        description = clean_text(str(obj.get("description") or ""))
        combined = f"{name} {description}"
        if "mattress" not in combined.casefold() and "mattress" not in source.url.casefold():
            return None
        brand = self._brand(obj.get("brand"), request.company_name)
        price, currency = self._offers(obj.get("offers"))
        weight_value = obj.get("weight")
        weight_text = clean_text(str(weight_value or ""))
        weight = _first_weight_kg(f"weight {weight_text}") if weight_text else None
        return ProductRecord(
            company_id=request.company_id,
            company_name=request.company_name,
            brand=brand,
            name=name,
            family=clean_text(str(obj.get("category") or "")) or None,
            canonical_url=clean_text(str(obj.get("url") or source.url)),
            description=description,
            firmness=_first_firmness(combined),
            total_thickness_mm=parse_first_thickness_mm(combined),
            product_weight_kg=weight,
            price=price,
            currency=currency,
            layers=self._layers_from_text(combined, source),
            variants=_variants_from_text(combined, source.source_id),
            source_ids=[source.source_id],
            extraction_method="json_ld",
            extraction_confidence=0.90,
        )

    def _from_html(
        self,
        tree: html.HtmlElement,
        page_text: str,
        source: SourceRecord,
        request: CompanyResearchRequest,
    ) -> ProductRecord | None:
        if not self._looks_like_product(source.url, page_text):
            return None
        name = clean_text(" ".join(tree.xpath("//h1[1]//text()")))
        if not name:
            name = _meta(tree, "og:title", "twitter:title") or ""
        if not _is_specific_product_name(name):
            return None
        description = _meta(tree, "description", "og:description") or ""
        brand = _meta(tree, "product:brand", "brand") or request.company_name
        price_text = _meta(tree, "product:price:amount", "og:price:amount")
        try:
            price = float(price_text.replace(",", "")) if price_text else None
        except ValueError:
            price = None
        currency = _meta(tree, "product:price:currency", "og:price:currency")
        if price is None:
            price, parsed_currency = _first_price(page_text[:30_000])
            currency = currency or parsed_currency

        # Product-level measurements are extracted from a bounded scope rather than the entire
        # page, which may contain navigation, recommendations, filters, and unrelated products.
        main_text = clean_text(" ".join(tree.xpath("//main//text()")))[:40_000]
        product_scope = clean_text(f"{name} {description} {main_text or page_text[:20_000]}")
        return ProductRecord(
            company_id=request.company_id,
            company_name=request.company_name,
            brand=brand,
            name=name,
            canonical_url=source.url,
            description=description or product_scope[:3_000],
            firmness=_first_firmness(product_scope),
            total_thickness_mm=_explicit_total_thickness_mm(product_scope, name),
            product_weight_kg=_first_weight_kg(product_scope),
            layers=self._layers_from_text(product_scope, source),
            variants=_variants_from_text(product_scope, source.source_id),
            price=price,
            currency=currency,
            source_ids=[source.source_id],
            extraction_method="heuristic",
            extraction_confidence=0.72,
        )

    def _from_text_document(
        self,
        text: str,
        source: SourceRecord,
        request: CompanyResearchRequest,
    ) -> list[ProductRecord]:
        if not text or not self._looks_relevant(source.url, text, request):
            return []
        segments = [
            clean_text(item)
            for item in re.split(r"(?=\[PDF page \d+\])", text)
            if clean_text(item)
        ] or [clean_text(text)]
        products: list[ProductRecord] = []
        seen_names: set[str] = set()
        for segment in segments[:500]:
            candidates: list[str] = []
            for match in _PRODUCT_NAME_RE.finditer(segment):
                name = clean_text(match.group("name"))
                words = name.split()
                if (
                    2 <= len(words) <= 10
                    and name.casefold() not in seen_names
                    and _is_specific_product_name(name)
                ):
                    candidates.append(name)
            if not candidates and len(segments) == 1:
                hint = source.title or _url_name_hint(source.url)
                if (
                    hint
                    and _is_specific_product_name(hint)
                    and "mattress" in f"{hint} {segment[:500]}".casefold()
                ):
                    candidates.append(hint)
            for name in candidates[:20]:
                seen_names.add(name.casefold())
                products.append(
                    ProductRecord(
                        company_id=request.company_id,
                        company_name=request.company_name,
                        brand=request.company_name,
                        name=name,
                        canonical_url=source.url,
                        description=segment[:3_000],
                        firmness=_first_firmness(segment),
                        total_thickness_mm=_explicit_total_thickness_mm(segment, name),
                        product_weight_kg=_first_weight_kg(segment),
                        price=_first_price(segment)[0],
                        currency=_first_price(segment)[1],
                        layers=self._layers_from_text(segment, source),
                        variants=_variants_from_text(segment, source.source_id),
                        source_ids=[source.source_id],
                        extraction_method="heuristic",
                        extraction_confidence=0.48,
                    )
                )
        return products

    def _layers_from_text(self, text: str, source: SourceRecord) -> list[LayerRecord]:
        candidates: list[tuple[int, str, str, float | None, float | None, str]] = []
        seen: set[tuple[str, float | None, float | None]] = set()
        for alias, material_id, start, end in self.materials.iter_material_mentions(
            text, max_matches=300
        ):
            excerpt = _context(text, start, end, radius=180)
            local_thickness = parse_first_thickness_mm(excerpt)
            local_density = parse_density_kg_m3(excerpt)
            key = (material_id, local_thickness, local_density)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                (start, clean_text(alias).title(), material_id, local_thickness, local_density, excerpt)
            )
        layers: list[LayerRecord] = []
        for index, (_, alias, material_id, thickness, density, excerpt) in enumerate(
            sorted(candidates, key=lambda item: item[0])[:20], start=1
        ):
            layers.append(
                LayerRecord(
                    position=index,
                    marketing_name=alias,
                    normalized_material=material_id,
                    thickness_mm=thickness,
                    density_kg_m3=density,
                    thickness_status=ClaimStatus.OBSERVED if thickness else ClaimStatus.UNKNOWN,
                    density_status=ClaimStatus.OBSERVED if density else ClaimStatus.UNKNOWN,
                    evidence=[
                        EvidenceRef(
                            source_id=source.source_id,
                            locator="deterministic text window",
                            excerpt=excerpt,
                            reliability=source.reliability,
                        )
                    ],
                )
            )
        return layers

    def _observations_from_text(
        self,
        text: str,
        source: SourceRecord,
        request: CompanyResearchRequest,
        *,
        product_name_hint: str | None,
        locator: str,
    ) -> list[EvidenceObservation]:
        observations: list[EvidenceObservation] = []

        def add(
            field_path: str,
            value: Any,
            *,
            unit: str | None = None,
            material: str | None = None,
            method: str = "regex",
            start: int = 0,
            end: int = 0,
            excerpt: str | None = None,
            confidence: float = 0.70,
            row_locator: str | None = None,
        ) -> None:
            context = excerpt or _context(text, start, end)
            observations.append(
                EvidenceObservation(
                    observation_id=stable_id(
                        "obs", source.source_id, field_path, value, context[:250]
                    ),
                    source_id=source.source_id,
                    company_id=request.company_id,
                    document_url=source.url,
                    product_name_hint=product_name_hint,
                    field_path=field_path,
                    value=value,
                    unit=unit,
                    normalized_material=material,
                    method=method,  # type: ignore[arg-type]
                    locator=row_locator or locator,
                    excerpt=context or None,
                    confidence=min(1.0, confidence * source.reliability / 0.95),
                )
            )

        for alias, material_id, start, end in self.materials.iter_material_mentions(
            text, max_matches=500
        ):
            add(
                "material.mention",
                clean_text(alias),
                material=material_id,
                method="material_dictionary",
                start=start,
                end=end,
                confidence=0.88,
            )

        for match in _DENSITY_RE.finditer(text):
            add(
                "measurement.density_kg_m3",
                float(match.group("value")),
                unit="kg/m3",
                start=match.start(),
                end=match.end(),
                confidence=0.92,
            )

        for pattern in _THICKNESS_PATTERNS:
            for match in pattern.finditer(text):
                value_mm = round(
                    length_to_mm(float(match.group("value")), match.group("unit")), 2
                )
                add(
                    "measurement.thickness_mm",
                    value_mm,
                    unit="mm",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.88,
                )

        for match in _DIMENSION_RE.finditer(text):
            values = [float(match.group("a")), float(match.group("b"))]
            if match.group("c"):
                values.append(float(match.group("c")))
            converted = [
                round(length_to_mm(value, match.group("unit")), 2) for value in values
            ]
            add(
                "variant.dimensions_mm",
                converted,
                unit="mm",
                start=match.start(),
                end=match.end(),
                confidence=0.85,
            )

        for match in _WEIGHT_RE.finditer(text):
            add(
                "measurement.weight_kg",
                float(match.group("value")),
                unit="kg",
                start=match.start(),
                end=match.end(),
                confidence=0.90,
            )

        for match in _PRICE_RE.finditer(text):
            raw_value = match.group("value").replace(",", "")
            try:
                value = float(raw_value)
            except ValueError:
                continue
            add(
                "commercial.price",
                value,
                unit="INR",
                start=match.start(),
                end=match.end(),
                confidence=0.82,
            )

        for match in _FIRMNESS_RE.finditer(text):
            add(
                "product.firmness",
                match.group(0).replace("-", " ").casefold(),
                start=match.start(),
                end=match.end(),
                confidence=0.72,
            )

        for match in _WARRANTY_RE.finditer(text):
            add(
                "commercial.warranty_years",
                float(match.group("value")),
                unit="years",
                start=match.start(),
                end=match.end(),
                confidence=0.90,
            )

        return self._deduplicate_observations(observations)[:2_000]

    def _observations_from_html(
        self,
        tree: html.HtmlElement,
        product_objects: list[dict[str, Any]],
        source: SourceRecord,
        request: CompanyResearchRequest,
        product_name_hint: str | None,
    ) -> list[EvidenceObservation]:
        observations: list[EvidenceObservation] = []

        def direct(
            field_path: str,
            value: Any,
            method: str,
            confidence: float,
            excerpt: str | None = None,
        ) -> None:
            if value in (None, "", []):
                return
            observations.append(
                EvidenceObservation(
                    observation_id=stable_id("obs", source.source_id, field_path, value),
                    source_id=source.source_id,
                    company_id=request.company_id,
                    document_url=source.url,
                    product_name_hint=product_name_hint,
                    field_path=field_path,
                    value=value,
                    method=method,  # type: ignore[arg-type]
                    locator="HTML structured data",
                    excerpt=clean_text(excerpt)[:1_000] if excerpt else None,
                    confidence=min(1.0, confidence * source.reliability / 0.95),
                )
            )

        title = clean_text(" ".join(tree.xpath("//title/text()")))
        h1 = clean_text(" ".join(tree.xpath("//h1[1]//text()")))
        direct("document.title", title, "meta", 0.85)
        direct("document.h1", h1, "meta", 0.88)
        direct("document.description", _meta(tree, "description", "og:description"), "meta", 0.80)

        for obj in product_objects[:200]:
            direct("product.name", clean_text(str(obj.get("name") or "")), "json_ld", 0.95)
            direct(
                "product.brand",
                self._brand(obj.get("brand"), request.company_name),
                "json_ld",
                0.95,
            )
            price, currency = self._offers(obj.get("offers"))
            if price is not None:
                observation = EvidenceObservation(
                    observation_id=stable_id("obs", source.source_id, "commercial.price", price),
                    source_id=source.source_id,
                    company_id=request.company_id,
                    document_url=source.url,
                    product_name_hint=clean_text(str(obj.get("name") or "")) or product_name_hint,
                    field_path="commercial.price",
                    value=price,
                    unit=currency,
                    method="json_ld",
                    locator="Product.offers",
                    confidence=source.reliability,
                )
                observations.append(observation)

        for row_number, row in enumerate(tree.xpath("//table//tr"), start=1):
            cells = [clean_text(" ".join(cell.xpath(".//text()"))) for cell in row.xpath("./th|./td")]
            cells = [cell for cell in cells if cell]
            if len(cells) < 2:
                continue
            row_text = " | ".join(cells)
            direct("table.row", row_text, "table", 0.82, row_text)
            observations.extend(
                self._observations_from_text(
                    row_text,
                    source,
                    request,
                    product_name_hint=product_name_hint,
                    locator=f"HTML table row {row_number}",
                )
            )
            if row_number >= 500:
                break
        return self._deduplicate_observations(observations)[:2_000]

    @staticmethod
    def _deduplicate_observations(
        observations: list[EvidenceObservation],
    ) -> list[EvidenceObservation]:
        deduplicated: dict[str, EvidenceObservation] = {}
        for observation in observations:
            current = deduplicated.get(observation.observation_id)
            if current is None or observation.confidence > current.confidence:
                deduplicated[observation.observation_id] = observation
        return list(deduplicated.values())

    def _from_llm(
        self,
        result: dict[str, Any],
        source: SourceRecord,
        request: CompanyResearchRequest,
    ) -> ProductRecord | None:
        name = clean_text(result.get("name"))
        if (
            not result.get("is_mattress_product")
            or not result.get("is_specific_model", True)
            or not _is_specific_product_name(name)
        ):
            return None
        layers: list[LayerRecord] = []
        for index, raw in enumerate(result.get("layers") or [], start=1):
            material = self.materials.normalize(
                raw.get("normalized_material") or raw.get("marketing_name")
            )
            thickness = raw.get("thickness_mm")
            density = raw.get("density_kg_m3")
            layers.append(
                LayerRecord(
                    position=int(raw.get("position") or index),
                    marketing_name=clean_text(raw.get("marketing_name")) or material,
                    normalized_material=material,
                    thickness_mm=thickness,
                    density_kg_m3=density,
                    thickness_status=ClaimStatus.OBSERVED if thickness else ClaimStatus.UNKNOWN,
                    density_status=ClaimStatus.OBSERVED if density else ClaimStatus.UNKNOWN,
                    evidence=[
                        EvidenceRef(
                            source_id=source.source_id,
                            locator="LLM-extracted page excerpt",
                            excerpt=clean_text(raw.get("evidence_excerpt")) or None,
                            reliability=source.reliability,
                        )
                    ],
                )
            )
        self.warnings.extend(str(item) for item in result.get("warnings") or [])
        return ProductRecord(
            company_id=request.company_id,
            company_name=request.company_name,
            brand=clean_text(result.get("brand")) or request.company_name,
            name=name,
            family=clean_text(result.get("family")) or None,
            canonical_url=source.url,
            description=clean_text(result.get("description")),
            firmness=clean_text(result.get("firmness")) or None,
            total_thickness_mm=result.get("total_thickness_mm"),
            product_weight_kg=result.get("product_weight_kg"),
            price=result.get("price"),
            currency=clean_text(result.get("currency")) or None,
            layers=layers,
            source_ids=[source.source_id],
            extraction_method="llm",
            extraction_confidence=0.75,
        )

    def _merge_product_lists(
        self,
        primary: list[ProductRecord],
        secondary: list[ProductRecord],
    ) -> list[ProductRecord]:
        merged = [item.model_copy(deep=True) for item in primary]
        for incoming in secondary:
            match_index = next(
                (
                    index
                    for index, current in enumerate(merged)
                    if current.name.casefold() == incoming.name.casefold()
                ),
                None,
            )
            if match_index is None:
                merged.append(incoming)
            else:
                merged[match_index] = self._merge(merged[match_index], incoming)
        return merged

    @staticmethod
    def _merge(primary: ProductRecord, secondary: ProductRecord) -> ProductRecord:
        layers = [layer.model_copy(deep=True) for layer in primary.layers]
        primary_by_material = {layer.normalized_material: layer for layer in layers}
        for layer in secondary.layers:
            existing = primary_by_material.get(layer.normalized_material)
            if existing is None:
                copy = layer.model_copy(deep=True)
                copy.position = len(layers) + 1
                layers.append(copy)
                primary_by_material[copy.normalized_material] = copy
                continue
            if existing.thickness_mm is None and layer.thickness_mm is not None:
                existing.thickness_mm = layer.thickness_mm
                existing.thickness_status = layer.thickness_status
            if existing.density_kg_m3 is None and layer.density_kg_m3 is not None:
                existing.density_kg_m3 = layer.density_kg_m3
                existing.density_status = layer.density_status
            existing.evidence.extend(
                evidence for evidence in layer.evidence if evidence not in existing.evidence
            )
        return ProductRecord(
            **{
                **primary.model_dump(
                    exclude={
                        "layers",
                        "variants",
                        "source_ids",
                        "extraction_method",
                        "extraction_confidence",
                    }
                ),
                "description": primary.description or secondary.description,
                "firmness": primary.firmness or secondary.firmness,
                "total_thickness_mm": primary.total_thickness_mm or secondary.total_thickness_mm,
                "product_weight_kg": primary.product_weight_kg or secondary.product_weight_kg,
                "price": primary.price if primary.price is not None else secondary.price,
                "currency": primary.currency or secondary.currency,
                "layers": layers,
                "variants": list(
                    {
                        variant.model_dump_json(): variant
                        for variant in [*primary.variants, *secondary.variants]
                    }.values()
                ),
                "source_ids": list(dict.fromkeys(primary.source_ids + secondary.source_ids)),
                "extraction_method": "merged",
                "extraction_confidence": max(
                    primary.extraction_confidence, secondary.extraction_confidence
                ),
            }
        )


def claims_from_product(product: ProductRecord) -> list[ClaimRecord]:
    claims: list[ClaimRecord] = []
    scalar_fields = (
        ("total_thickness_mm", product.total_thickness_mm, "mm"),
        ("product_weight_kg", product.product_weight_kg, "kg"),
        ("price", product.price, product.currency),
        ("firmness", product.firmness, None),
    )
    for field_path, value, unit in scalar_fields:
        if value is None:
            continue
        claims.append(
            ClaimRecord(
                claim_id=stable_id("clm", product.product_id, field_path, value),
                product_id=str(product.product_id),
                field_path=field_path,
                value=value,
                unit=unit,
                status=ClaimStatus.OBSERVED,
                confidence=product.extraction_confidence,
                evidence=[EvidenceRef(source_id=source_id) for source_id in product.source_ids],
                method=product.extraction_method,
            )
        )
    for layer in product.layers:
        for field_name, value, unit, status in (
            ("thickness_mm", layer.thickness_mm, "mm", layer.thickness_status),
            ("density_kg_m3", layer.density_kg_m3, "kg/m3", layer.density_status),
        ):
            if value is None:
                continue
            field_path = f"layers.{layer.position}.{field_name}"
            claims.append(
                ClaimRecord(
                    claim_id=stable_id("clm", product.product_id, field_path, value),
                    product_id=str(product.product_id),
                    field_path=field_path,
                    value=value,
                    unit=unit,
                    status=status,
                    confidence=max((item.reliability for item in layer.evidence), default=0.5),
                    evidence=layer.evidence,
                    method=product.extraction_method,
                )
            )
    return claims
