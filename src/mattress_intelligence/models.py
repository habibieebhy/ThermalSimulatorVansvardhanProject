"""Canonical data contracts used by every pipeline stage."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def stable_id(prefix: str, *parts: object) -> str:
    normalized = "|".join(str(part).strip().casefold() for part in parts)
    return f"{prefix}_{sha256(normalized.encode('utf-8')).hexdigest()[:16]}"


class SourceKind(StrEnum):
    OFFICIAL_PRODUCT = "official_product"
    OFFICIAL_CATALOGUE = "official_catalogue"
    OFFICIAL_OTHER = "official_other"
    RETAILER = "retailer"
    PATENT = "patent"
    TEARDOWN = "teardown"
    OTHER = "other"


class AssetKind(StrEnum):
    PRODUCT_IMAGE = "product_image"
    LAYER_DIAGRAM = "layer_diagram"
    CATALOGUE_PAGE = "catalogue_page"
    SPECIFICATION_TABLE = "specification_table"
    SCREENSHOT = "screenshot"
    API_PAYLOAD = "api_payload"
    OTHER_IMAGE = "other_image"


class ClaimStatus(StrEnum):
    OBSERVED = "observed"
    DERIVED = "derived"
    INFERRED = "inferred"
    UNKNOWN = "unknown"
    CONTRADICTED = "contradicted"


class MaterialEvidenceScope(StrEnum):
    EXACT_VARIANT = "exact_variant"
    EXACT_PRODUCT = "exact_product"
    PRODUCT_FAMILY = "product_family"
    TECHNOLOGY = "technology"
    MANUFACTURER = "manufacturer"
    COMPARABLE_MATERIAL = "comparable_material"
    GENERIC_CATEGORY = "generic_category"


class MaterialIdentityStatus(StrEnum):
    VERIFIED_PRIMARY = "verified_primary"
    VERIFIED_CORROBORATED = "verified_corroborated"
    PROBABLE = "probable"
    UNRESOLVED = "unresolved"


class DensityEvidenceGrade(StrEnum):
    A_MANUFACTURER_EXACT = "A_manufacturer_exact"
    B_CORROBORATED_EXACT = "B_corroborated_exact"
    C_MEASURED_TEARDOWN = "C_measured_teardown"
    D_SAME_TECHNOLOGY = "D_same_technology"
    E_GENERIC_CATEGORY = "E_generic_category"
    UNKNOWN = "unknown"


class DensityEvidenceStatus(StrEnum):
    VERIFIED_EXACT = "verified_exact"
    CORROBORATED_RANGE = "corroborated_range"
    PROVISIONAL_RANGE = "provisional_range"
    GENERIC_COMPARISON_ONLY = "generic_comparison_only"
    UNKNOWN = "unknown"


class SourceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    company_id: str
    url: str
    title: str | None = None
    kind: SourceKind = SourceKind.OTHER
    is_official: bool = False
    reliability: float = Field(default=0.5, ge=0.0, le=1.0)
    retrieved_at: datetime = Field(default_factory=utc_now)
    content_sha256: str
    artifact_path: str | None = None
    object_uri: str | None = None
    capture_method: str = "http"
    http_status: int = 200
    content_type: str = "text/html"


class AssetRecord(BaseModel):
    """Image/PDF-page asset with immutable source and storage provenance."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str
    source_id: str
    company_id: str
    page_url: str
    asset_url: str
    kind: AssetKind = AssetKind.OTHER_IMAGE
    discovery_method: str
    alt_text: str | None = None
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    content_type: str = "application/octet-stream"
    content_sha256: str
    local_path: str | None = None
    object_uri: str | None = None
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    perceptual_hash: str | None = None
    vision_priority: float = Field(default=0.0, ge=0.0, le=1.0)
    vision_provider: str | None = None
    vision_model: str | None = None
    vision_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    vision_payload: dict[str, Any] | None = None
    vision_verified: bool = False
    vision_search_queries: list[str] = Field(default_factory=list)
    vision_followup_urls: list[str] = Field(default_factory=list)
    ocr_engine: str | None = None
    ocr_text: str | None = None
    ocr_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    asset_id: str | None = None
    locator: str | None = None
    excerpt: str | None = Field(default=None, max_length=1_000)
    reliability: float = Field(default=0.5, ge=0.0, le=1.0)


class MaterialEvidenceSource(BaseModel):
    """One source supporting or contradicting a trademark-material conclusion."""

    model_config = ConfigDict(extra="forbid")

    url: str
    title: str | None = None
    source_kind: SourceKind = SourceKind.OTHER
    evidence_scope: MaterialEvidenceScope = MaterialEvidenceScope.TECHNOLOGY
    supports_identity: bool = False
    supports_density: bool = False
    excerpt: str | None = Field(default=None, max_length=1_500)
    density_min_kg_m3: float | None = Field(default=None, gt=0)
    density_max_kg_m3: float | None = Field(default=None, gt=0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def order_density_range(self) -> "MaterialEvidenceSource":
        if (
            self.density_min_kg_m3 is not None
            and self.density_max_kg_m3 is not None
            and self.density_min_kg_m3 > self.density_max_kg_m3
        ):
            self.density_min_kg_m3, self.density_max_kg_m3 = (
                self.density_max_kg_m3,
                self.density_min_kg_m3,
            )
        return self


class TrademarkMaterialRecord(BaseModel):
    """Evidence-backed decoding of one proprietary mattress material name."""

    model_config = ConfigDict(extra="forbid")

    material_id: str
    company_id: str
    product_id: str | None = None
    product_name: str | None = None
    family: str | None = None
    market: str
    trademark_name: str
    diagram_asset_id: str | None = None
    diagram_region: dict[str, float] | None = None
    diagram_crop_path: str | None = None
    visible_description: str
    generic_material_class: str
    generic_material_name: str
    actual_material_description: str
    base_polymer: str | None = None
    additives_or_structure: list[str] = Field(default_factory=list)
    probable_functions: list[str] = Field(default_factory=list)
    stack_position: str | None = None
    identity_status: MaterialIdentityStatus = MaterialIdentityStatus.UNRESOLVED
    identity_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_scope: MaterialEvidenceScope = MaterialEvidenceScope.TECHNOLOGY
    density_status: DensityEvidenceStatus = DensityEvidenceStatus.UNKNOWN
    density_grade: DensityEvidenceGrade = DensityEvidenceGrade.UNKNOWN
    density_min_kg_m3: float | None = Field(default=None, gt=0)
    density_max_kg_m3: float | None = Field(default=None, gt=0)
    density_representative_kg_m3: float | None = Field(default=None, gt=0)
    density_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    density_basis: str
    evidence_sources: list[MaterialEvidenceSource] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    conclusion: str

    @field_validator(
        "trademark_name",
        "visible_description",
        "generic_material_class",
        "generic_material_name",
        "actual_material_description",
        "density_basis",
        "conclusion",
    )
    @classmethod
    def required_material_text(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Trademark-material text fields cannot be empty.")
        return cleaned

    @model_validator(mode="after")
    def validate_density_claim(self) -> "TrademarkMaterialRecord":
        if (
            self.density_min_kg_m3 is not None
            and self.density_max_kg_m3 is not None
            and self.density_min_kg_m3 > self.density_max_kg_m3
        ):
            self.density_min_kg_m3, self.density_max_kg_m3 = (
                self.density_max_kg_m3,
                self.density_min_kg_m3,
            )
        if self.density_status == DensityEvidenceStatus.UNKNOWN:
            self.density_min_kg_m3 = None
            self.density_max_kg_m3 = None
            self.density_representative_kg_m3 = None
            self.density_confidence = 0.0
        if self.density_grade == DensityEvidenceGrade.E_GENERIC_CATEGORY:
            self.density_status = DensityEvidenceStatus.GENERIC_COMPARISON_ONLY
        return self


class LayerRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layer_id: str | None = None
    position: int = Field(ge=1)
    marketing_name: str
    normalized_material: str
    thickness_mm: float | None = Field(default=None, gt=0)
    density_kg_m3: float | None = Field(default=None, gt=0)
    thickness_status: ClaimStatus = ClaimStatus.UNKNOWN
    density_status: ClaimStatus = ClaimStatus.UNKNOWN
    evidence: list[EvidenceRef] = Field(default_factory=list)

    @field_validator("marketing_name", "normalized_material")
    @classmethod
    def non_empty(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Layer names cannot be empty.")
        return cleaned


class VariantRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variant_id: str | None = None
    sku: str | None = None
    size_name: str | None = None
    width_mm: float | None = Field(default=None, gt=0)
    length_mm: float | None = Field(default=None, gt=0)
    thickness_mm: float | None = Field(default=None, gt=0)
    weight_kg: float | None = Field(default=None, gt=0)
    price: float | None = Field(default=None, ge=0)
    currency: str | None = None
    source_ids: list[str] = Field(default_factory=list)


class ProductRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: str | None = None
    company_id: str
    company_name: str
    brand: str
    name: str
    family: str | None = None
    canonical_url: str | None = None
    description: str = ""
    firmness: str | None = None
    total_thickness_mm: float | None = Field(default=None, gt=0)
    product_weight_kg: float | None = Field(default=None, gt=0)
    price: float | None = Field(default=None, ge=0)
    currency: str | None = None
    layers: list[LayerRecord] = Field(default_factory=list)
    variants: list[VariantRecord] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    extraction_method: Literal[
        "json_ld", "heuristic", "llm", "vision", "imported", "merged"
    ] = "heuristic"
    extraction_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reviewed: bool = False

    @field_validator("company_name", "brand", "name")
    @classmethod
    def required_text(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Company, brand, and product name cannot be empty.")
        return cleaned

    @model_validator(mode="after")
    def assign_ids(self) -> "ProductRecord":
        if self.product_id is None:
            self.product_id = stable_id("prd", self.company_id, self.brand, self.name)
        for layer in self.layers:
            if layer.layer_id is None:
                layer.layer_id = stable_id("lyr", self.product_id, layer.position, layer.marketing_name)
        for index, variant in enumerate(self.variants, start=1):
            if variant.variant_id is None:
                variant.variant_id = stable_id(
                    "var", self.product_id, variant.sku or variant.size_name or index
                )
        return self

    @property
    def searchable_text(self) -> str:
        layer_text = " ".join(
            f"{layer.marketing_name} {layer.normalized_material}" for layer in self.layers
        )
        return " ".join(
            filter(
                None,
                [self.brand, self.name, self.family, self.description, self.firmness, layer_text],
            )
        )


class ClaimRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    product_id: str
    field_path: str
    value: Any = None
    unit: str | None = None
    status: ClaimStatus
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    method: str


class EvidenceObservation(BaseModel):
    """Atomic fact captured without requiring a complete product match."""

    model_config = ConfigDict(extra="forbid")

    observation_id: str
    source_id: str
    asset_id: str | None = None
    company_id: str
    document_url: str
    product_name_hint: str | None = None
    field_path: str
    value: Any = None
    unit: str | None = None
    normalized_material: str | None = None
    method: Literal[
        "json_ld",
        "meta",
        "regex",
        "table",
        "material_dictionary",
        "url",
        "jina_reader",
        "firecrawl",
        "vision",
        "pdf_page",
        "ocr",
        "network_json",
    ]
    locator: str | None = None
    excerpt: str | None = Field(default=None, max_length=1_000)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class CandidateLayer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position: int
    material: str
    marketing_name: str
    thickness_mm: int = Field(gt=0)
    density_kg_m3: int = Field(gt=0)
    conductivity_w_mk: float = Field(gt=0)
    specific_heat_j_kgk: float = Field(gt=0)

class SimulationScreeningResult(BaseModel):
    """Comparative passive thermal screening metrics for one configuration."""

    model_config = ConfigDict(extra="forbid")

    configuration_id: str

    thermal_resistance_m2k_w: float = Field(ge=0.0)
    areal_heat_capacity_kj_m2k: float = Field(ge=0.0)

    estimated_final_interface_temperature_c: float
    comfort_zone_minutes: float = Field(ge=0.0)
    peak_interface_temperature_c: float

    screening_only: bool = True


class ConfigurationCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    configuration_id: str
    product_id: str
    rank: int = 0
    layers: list[CandidateLayer]
    total_thickness_mm: int
    estimated_weight_kg: float | None = None
    posterior_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_score: float = Field(default=0.0, ge=0.0, le=100.0)
    evidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)


class CatalogueCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    discovered_urls: int = 0
    fetched_urls: int = 0
    failed_urls: int = 0
    product_pages: int = 0
    unique_products: int = 0
    variants: int = 0
    assets: int = 0
    vision_assets: int = 0
    official_source_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    estimated_coverage_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    limitations: list[str] = Field(default_factory=list)


class CompanyResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_name: str
    official_domain: str
    market: str = "India"
    brand_aliases: list[str] = Field(default_factory=list)
    seed_urls: list[str] = Field(default_factory=list)
    custom_search_queries: list[str] = Field(default_factory=list)
    include_external_evidence: bool = False
    use_search_grounding: bool = False
    discover_assets: bool = True
    analyze_assets_with_vision: bool = True
    max_pages: int = Field(default=100, ge=1, le=10_000)
    max_external_pages: int = Field(default=25, ge=0, le=2_000)
    max_crawl_depth: int = Field(default=4, ge=0, le=20)
    max_assets_per_document: int = Field(default=30, ge=0, le=500)
    max_vision_assets: int = Field(default=80, ge=0, le=5_000)
    max_pdf_pages: int = Field(default=100, ge=1, le=2_000)
    max_configurations_per_product: int = Field(default=10, ge=1, le=100)
    respect_robots_txt: bool = True

    @property
    def company_id(self) -> str:
        return stable_id("cmp", self.company_name, self.market)


class ResearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    request: CompanyResearchRequest
    started_at: datetime
    completed_at: datetime
    products: list[ProductRecord]
    sources: list[SourceRecord]
    assets: list[AssetRecord] = Field(default_factory=list)
    trademark_materials: list[TrademarkMaterialRecord] = Field(default_factory=list)
    claims: list[ClaimRecord]
    observations: list[EvidenceObservation] = Field(default_factory=list)
    configurations: list[ConfigurationCandidate]
    similarity_matches: list[dict[str, Any]] = Field(default_factory=list)
    discovery_log: list[dict[str, Any]] = Field(default_factory=list)
    crawl_log: list[dict[str, Any]] = Field(default_factory=list)
    acquisition_log: list[dict[str, Any]] = Field(default_factory=list)
    recognition_log: list[dict[str, Any]] = Field(default_factory=list)
    graph_edges: list[dict[str, Any]]
    coverage: CatalogueCoverage
    warnings: list[str] = Field(default_factory=list)
    excel_path: str | None = None
