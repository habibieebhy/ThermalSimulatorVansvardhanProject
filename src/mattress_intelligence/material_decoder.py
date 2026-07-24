"""Trademark-material decoding from technical diagrams and corroborating documents."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import ValidationError

from .llm import LLMError, LLMProvider
from .models import (
    AssetRecord,
    DensityEvidenceGrade,
    DensityEvidenceStatus,
    MaterialEvidenceScope,
    MaterialIdentityStatus,
    ProductRecord,
    SourceKind,
    TrademarkMaterialRecord,
    stable_id,
)

_GENERIC_LABELS = {
    "comfort foam",
    "support foam",
    "support core",
    "base foam",
    "foam",
    "memory foam",
    "latex",
    "latex foam",
    "pocketed coil",
    "pocket coil",
    "spring unit",
    "quilted layer",
    "quilted comfort layer",
    "cover",
    "fabric",
    "support system",
    "mattress",
}


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").replace("_x0000_", " ").split()).strip()


def _term_key(value: str) -> str:
    cleaned = re.sub(r"[™®©]", "", value).casefold()
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    return " ".join(cleaned.split())


def _product_key(value: str | None) -> str:
    return _term_key(value or "")


def _is_useful_material_name(value: str, generic_class: str | None = None) -> bool:
    key = _term_key(value)
    if len(key) < 4 or key in _GENERIC_LABELS:
        return False
    if any(marker in value for marker in ("™", "®", "©")):
        return True
    if generic_class and key != _term_key(generic_class):
        return True
    words = value.split()
    return len(words) >= 2 and any(character.isupper() for character in value[1:])


@dataclass(slots=True)
class MaterialCandidate:
    candidate_key: str
    trademark_name: str
    product_id: str | None
    product_name: str | None
    family: str | None
    diagram_asset_id: str | None
    diagram_region: dict[str, float] | None
    source_page_url: str | None
    source_image_url: str | None
    visible_label: str | None
    callout_text: str | None
    diagram_summary: str | None
    current_generic_class: str | None
    current_normalized_material: str | None
    assignment_scope: str | None
    evidence_status: str | None
    position: int | None
    visible_thickness_mm: float | None
    visible_density_kg_m3: float | None
    visual_confidence: float
    search_queries: list[str] = field(default_factory=list)
    crop_path: str | None = None

    def as_prompt_dict(self) -> dict[str, Any]:
        return {
            "candidate_key": self.candidate_key,
            "trademark_name": self.trademark_name,
            "product_id": self.product_id,
            "product_name": self.product_name,
            "family": self.family,
            "diagram_asset_id": self.diagram_asset_id,
            "diagram_region": self.diagram_region,
            "source_page_url": self.source_page_url,
            "source_image_url": self.source_image_url,
            "visible_label": self.visible_label,
            "callout_text": self.callout_text,
            "diagram_summary": self.diagram_summary,
            "current_generic_class": self.current_generic_class,
            "current_normalized_material": self.current_normalized_material,
            "assignment_scope": self.assignment_scope,
            "evidence_status": self.evidence_status,
            "position": self.position,
            "visible_thickness_mm": self.visible_thickness_mm,
            "visible_density_kg_m3": self.visible_density_kg_m3,
            "visual_confidence": self.visual_confidence,
            "search_queries": self.search_queries,
        }


@dataclass(slots=True)
class MaterialDecoderResult:
    records: list[TrademarkMaterialRecord] = field(default_factory=list)
    discovery_urls: list[str] = field(default_factory=list)
    discovery_rows: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class TrademarkMaterialDecoder:
    """Turn visual marketing labels into evidence-scoped industrial material identities."""

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    @staticmethod
    def collect_candidates(
        assets: list[AssetRecord],
        products: list[ProductRecord],
        *,
        limit: int,
    ) -> list[MaterialCandidate]:
        product_by_name = {_product_key(item.name): item for item in products}
        candidates: dict[str, MaterialCandidate] = {}

        for asset in sorted(assets, key=lambda item: item.vision_priority, reverse=True):
            payload = asset.vision_payload or {}
            diagram_summary = _clean_text(payload.get("diagram_summary")) or None
            asset_queries = [
                _clean_text(query)
                for query in asset.vision_search_queries
                if _clean_text(query)
            ]
            for visual_product in payload.get("products") or []:
                product_name = _clean_text(visual_product.get("name")) or None
                family = _clean_text(visual_product.get("family")) or None
                matched = product_by_name.get(_product_key(product_name))
                product_id = str(matched.product_id) if matched and matched.product_id else None
                if matched is not None:
                    product_name = matched.name
                    family = matched.family or family

                for layer in visual_product.get("layers") or []:
                    trademark_name = _clean_text(
                        layer.get("visible_label") or layer.get("marketing_name")
                    )
                    generic_class = _clean_text(layer.get("generic_material_class")) or None
                    if not _is_useful_material_name(trademark_name, generic_class):
                        continue
                    key = stable_id(
                        "matcand",
                        trademark_name,
                        product_name or family or asset.company_id,
                    )
                    candidate = MaterialCandidate(
                        candidate_key=key,
                        trademark_name=trademark_name,
                        product_id=product_id,
                        product_name=product_name,
                        family=family,
                        diagram_asset_id=asset.asset_id,
                        diagram_region=layer.get("region"),
                        source_page_url=asset.page_url,
                        source_image_url=asset.asset_url,
                        visible_label=_clean_text(layer.get("visible_label")) or None,
                        callout_text=_clean_text(layer.get("callout_text")) or None,
                        diagram_summary=diagram_summary,
                        current_generic_class=generic_class,
                        current_normalized_material=(
                            _clean_text(layer.get("normalized_material")) or None
                        ),
                        assignment_scope=_clean_text(layer.get("assignment_scope")) or None,
                        evidence_status=_clean_text(layer.get("evidence_status")) or None,
                        position=layer.get("position"),
                        visible_thickness_mm=layer.get("thickness_mm"),
                        visible_density_kg_m3=layer.get("density_kg_m3"),
                        visual_confidence=float(layer.get("confidence") or 0.0),
                        search_queries=list(asset_queries),
                    )
                    previous = candidates.get(key)
                    if previous is None or candidate.visual_confidence > previous.visual_confidence:
                        candidates[key] = candidate

            existing_terms = {_term_key(item.trademark_name) for item in candidates.values()}
            for raw_term in payload.get("technology_terms") or []:
                trademark_name = _clean_text(raw_term)
                if not _is_useful_material_name(trademark_name):
                    continue
                if _term_key(trademark_name) in existing_terms:
                    continue
                key = stable_id("matcand", trademark_name, asset.company_id)
                candidates.setdefault(
                    key,
                    MaterialCandidate(
                        candidate_key=key,
                        trademark_name=trademark_name,
                        product_id=None,
                        product_name=None,
                        family=None,
                        diagram_asset_id=asset.asset_id,
                        diagram_region=None,
                        source_page_url=asset.page_url,
                        source_image_url=asset.asset_url,
                        visible_label=trademark_name,
                        callout_text=None,
                        diagram_summary=diagram_summary,
                        current_generic_class=None,
                        current_normalized_material=None,
                        assignment_scope="technology_term",
                        evidence_status="observed_label",
                        position=None,
                        visible_thickness_mm=None,
                        visible_density_kg_m3=None,
                        visual_confidence=float(asset.vision_confidence or 0.0),
                        search_queries=list(asset_queries),
                    ),
                )

        ordered = sorted(
            candidates.values(),
            key=lambda item: (
                item.visible_density_kg_m3 is not None,
                item.visible_thickness_mm is not None,
                item.visual_confidence,
            ),
            reverse=True,
        )
        return ordered[: max(0, limit)]

    @staticmethod
    def create_crops(
        candidates: list[MaterialCandidate],
        assets: list[AssetRecord],
        output_dir: Path,
    ) -> None:
        asset_by_id = {asset.asset_id: asset for asset in assets}
        crop_dir = output_dir / "material_diagrams"
        crop_dir.mkdir(parents=True, exist_ok=True)

        for candidate in candidates:
            if not candidate.diagram_asset_id or not candidate.diagram_region:
                continue
            asset = asset_by_id.get(candidate.diagram_asset_id)
            if asset is None or not asset.local_path:
                continue
            source_path = Path(asset.local_path)
            if not source_path.is_file():
                continue
            region = candidate.diagram_region
            try:
                with Image.open(source_path) as image:
                    width, height = image.size
                    x = min(1.0, max(0.0, float(region.get("x", 0.0))))
                    y = min(1.0, max(0.0, float(region.get("y", 0.0))))
                    region_width = min(1.0 - x, max(0.0, float(region.get("width", 0.0))))
                    region_height = min(1.0 - y, max(0.0, float(region.get("height", 0.0))))
                    if region_width <= 0 or region_height <= 0:
                        continue
                    left = max(0, int(round(x * width)))
                    top = max(0, int(round(y * height)))
                    right = min(width, max(left + 1, int(round((x + region_width) * width))))
                    bottom = min(height, max(top + 1, int(round((y + region_height) * height))))
                    crop = image.convert("RGB").crop((left, top, right, bottom))
                    destination = crop_dir / f"{candidate.candidate_key}.jpg"
                    crop.save(destination, format="JPEG", quality=92, optimize=True)
                    candidate.crop_path = str(destination)
            except (OSError, ValueError):
                continue

    def discover(
        self,
        *,
        company_name: str,
        official_domain: str,
        market: str,
        candidates: list[MaterialCandidate],
        max_urls: int,
    ) -> MaterialDecoderResult:
        result = MaterialDecoderResult()
        if not candidates:
            return result
        try:
            discovery_rows = self.llm.discover_material_evidence(
                company_name=company_name,
                official_domain=official_domain,
                market=market,
                candidates=[item.as_prompt_dict() for item in candidates],
                limit=max_urls,
            )
        except LLMError as exc:
            result.warnings.append(f"Trademark-material evidence discovery failed: {exc}")
            return result

        result.discovery_rows = discovery_rows
        result.discovery_urls = list(
            dict.fromkeys(
                str(item.get("url") or "").strip()
                for item in discovery_rows
                if str(item.get("url") or "").startswith(("http://", "https://"))
            )
        )[:max_urls]
        return result

    def adjudicate(
        self,
        *,
        company_id: str,
        company_name: str,
        official_domain: str,
        market: str,
        candidates: list[MaterialCandidate],
        evidence_documents: list[dict[str, Any]],
    ) -> MaterialDecoderResult:
        result = MaterialDecoderResult()
        candidate_by_key = {item.candidate_key: item for item in candidates}
        try:
            raw_records = self.llm.decode_trademark_materials(
                company_name=company_name,
                official_domain=official_domain,
                market=market,
                candidates=[item.as_prompt_dict() for item in candidates],
                evidence_documents=evidence_documents,
            )
        except LLMError as exc:
            result.warnings.append(f"Trademark-material adjudication failed: {exc}")
            raw_records = []

        allowed_urls = {
            str(document.get("url") or "").strip()
            for document in evidence_documents
            if str(document.get("url") or "").strip()
        }
        for candidate in candidates:
            if candidate.source_page_url:
                allowed_urls.add(candidate.source_page_url)
            if candidate.source_image_url:
                allowed_urls.add(candidate.source_image_url)

        returned_keys: set[str] = set()
        for raw in raw_records:
            key = str(raw.get("candidate_key") or "")
            candidate = candidate_by_key.get(key)
            if candidate is None:
                continue
            returned_keys.add(key)
            density = raw.pop("density", {}) or {}
            evidence_sources = [
                item
                for item in (raw.pop("evidence_sources", []) or [])
                if str(item.get("url") or "").strip() in allowed_urls
            ]
            identity_sources = [
                item for item in evidence_sources if bool(item.get("supports_identity"))
            ]
            density_sources = [
                item
                for item in evidence_sources
                if bool(item.get("supports_density"))
                and (
                    item.get("density_min_kg_m3") is not None
                    or item.get("density_max_kg_m3") is not None
                )
            ]

            identity_status = str(raw.get("identity_status") or "unresolved")
            exact_identity_sources = [
                item
                for item in identity_sources
                if str(item.get("evidence_scope") or "")
                in {"exact_variant", "exact_product", "product_family"}
            ]
            official_identity_sources = [
                item
                for item in exact_identity_sources
                if str(item.get("source_kind") or "").startswith("official")
            ]
            if identity_status == "verified_primary" and not official_identity_sources:
                identity_status = "probable" if identity_sources else "unresolved"
            if identity_status == "verified_corroborated":
                independent_identity_urls = {
                    str(item.get("url") or "") for item in exact_identity_sources
                }
                if len(independent_identity_urls) < 2:
                    identity_status = "probable" if identity_sources else "unresolved"

            density_status = str(density.get("status") or "unknown")
            density_grade = str(density.get("grade") or "unknown")
            if not density_sources:
                density_status = "unknown"
                density_grade = "unknown"
                density["minimum_kg_m3"] = None
                density["maximum_kg_m3"] = None
                density["representative_kg_m3"] = None
                density["confidence"] = 0.0
                density["basis"] = "No fetched source supplied a numeric density claim."
            else:
                exact_density_sources = [
                    item
                    for item in density_sources
                    if str(item.get("evidence_scope") or "")
                    in {"exact_variant", "exact_product", "product_family"}
                ]
                official_exact_density = [
                    item
                    for item in exact_density_sources
                    if str(item.get("source_kind") or "").startswith("official")
                ]
                teardown_density = [
                    item
                    for item in density_sources
                    if str(item.get("source_kind") or "") == "teardown"
                ]
                independent_exact_urls = {
                    str(item.get("url") or "") for item in exact_density_sources
                }
                if density_grade == "A_manufacturer_exact" and not official_exact_density:
                    density_grade = "D_same_technology"
                    density_status = "provisional_range"
                elif density_grade == "B_corroborated_exact" and len(independent_exact_urls) < 2:
                    density_grade = (
                        "C_measured_teardown" if teardown_density else "D_same_technology"
                    )
                    density_status = (
                        "corroborated_range" if teardown_density else "provisional_range"
                    )
                elif density_grade == "C_measured_teardown" and not teardown_density:
                    density_grade = "D_same_technology"
                    density_status = "provisional_range"
                elif density_grade == "E_generic_category":
                    density_status = "generic_comparison_only"

            payload = {
                "material_id": stable_id(
                    "mat",
                    company_id,
                    candidate.trademark_name,
                    candidate.product_name or candidate.family or "technology",
                ),
                "company_id": company_id,
                "product_id": candidate.product_id,
                "product_name": candidate.product_name,
                "family": candidate.family,
                "market": market,
                "trademark_name": candidate.trademark_name,
                "diagram_asset_id": candidate.diagram_asset_id,
                "diagram_region": candidate.diagram_region,
                "diagram_crop_path": candidate.crop_path,
                "visible_description": raw.get("visible_description") or (
                    candidate.callout_text
                    or candidate.diagram_summary
                    or f"A labelled mattress component named {candidate.trademark_name}."
                ),
                "generic_material_class": raw.get("generic_material_class") or "unresolved",
                "generic_material_name": raw.get("generic_material_name") or "Unresolved material",
                "actual_material_description": raw.get("actual_material_description")
                or "The public evidence collected in this session did not establish a more specific identity.",
                "base_polymer": raw.get("base_polymer"),
                "additives_or_structure": raw.get("additives_or_structure") or [],
                "probable_functions": raw.get("probable_functions") or [],
                "stack_position": raw.get("stack_position"),
                "identity_status": identity_status,
                "identity_confidence": raw.get("identity_confidence") or 0.0,
                "evidence_scope": raw.get("evidence_scope") or MaterialEvidenceScope.TECHNOLOGY,
                "density_status": density_status,
                "density_grade": density_grade,
                "density_min_kg_m3": density.get("minimum_kg_m3"),
                "density_max_kg_m3": density.get("maximum_kg_m3"),
                "density_representative_kg_m3": density.get("representative_kg_m3"),
                "density_confidence": density.get("confidence") or 0.0,
                "density_basis": density.get("basis") or "No defensible numeric density evidence was found.",
                "evidence_sources": evidence_sources,
                "search_queries": raw.get("search_queries_used") or candidate.search_queries,
                "contradictions": raw.get("contradictions") or [],
                "unknowns": raw.get("unknowns") or [],
                "conclusion": raw.get("conclusion")
                or "The material identity remains unresolved from public evidence.",
            }
            try:
                result.records.append(TrademarkMaterialRecord.model_validate(payload))
            except ValidationError as exc:
                result.warnings.append(
                    f"Rejected invalid trademark-material result for {candidate.trademark_name}: {exc}"
                )

        for candidate in candidates:
            if candidate.candidate_key in returned_keys:
                continue
            result.records.append(self._fallback_record(company_id, market, candidate))
        return result

    @staticmethod
    def _fallback_record(
        company_id: str,
        market: str,
        candidate: MaterialCandidate,
    ) -> TrademarkMaterialRecord:
        generic_class = candidate.current_generic_class or "unresolved"
        generic_name = generic_class.replace("_", " ").title()
        density_status = DensityEvidenceStatus.UNKNOWN
        density_grade = DensityEvidenceGrade.UNKNOWN
        density_value = candidate.visible_density_kg_m3
        if density_value is not None:
            density_status = DensityEvidenceStatus.VERIFIED_EXACT
            density_grade = DensityEvidenceGrade.A_MANUFACTURER_EXACT
        return TrademarkMaterialRecord(
            material_id=stable_id(
                "mat",
                company_id,
                candidate.trademark_name,
                candidate.product_name or candidate.family or "technology",
            ),
            company_id=company_id,
            product_id=candidate.product_id,
            product_name=candidate.product_name,
            family=candidate.family,
            market=market,
            trademark_name=candidate.trademark_name,
            diagram_asset_id=candidate.diagram_asset_id,
            diagram_region=candidate.diagram_region,
            diagram_crop_path=candidate.crop_path,
            visible_description=(
                candidate.callout_text
                or candidate.diagram_summary
                or f"A labelled mattress component named {candidate.trademark_name}."
            ),
            generic_material_class=generic_class,
            generic_material_name=generic_name,
            actual_material_description=(
                "Only the manufacturer-facing name and broad visual class were established. "
                "The exact formulation was not verified."
            ),
            stack_position=(
                f"Visible diagram position {candidate.position}"
                if candidate.position is not None
                else None
            ),
            identity_status=(
                MaterialIdentityStatus.PROBABLE
                if generic_class != "unresolved"
                else MaterialIdentityStatus.UNRESOLVED
            ),
            identity_confidence=min(1.0, candidate.visual_confidence),
            evidence_scope=MaterialEvidenceScope.TECHNOLOGY,
            density_status=density_status,
            density_grade=density_grade,
            density_min_kg_m3=density_value,
            density_max_kg_m3=density_value,
            density_representative_kg_m3=density_value,
            density_confidence=min(1.0, candidate.visual_confidence) if density_value else 0.0,
            density_basis=(
                "Density was visibly printed in the source diagram."
                if density_value
                else "No defensible numeric density evidence was found."
            ),
            search_queries=candidate.search_queries,
            unknowns=["Exact formulation", "Supplier grade", "ILD or hardness"],
            conclusion=(
                f"{candidate.trademark_name} maps provisionally to {generic_name}; "
                "numeric density remains unverified."
            ),
        )
