"""Optional evidence-recognition providers.

LLMs may discover sources, classify documents, and extract explicit published facts.
They never generate configurations, Bayesian posteriors, graph conclusions, or confidence scores.
"""

from __future__ import annotations

import base64
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .network import RETRYABLE_TRANSPORT_ERRORS, http_error_detail


class LLMError(RuntimeError):
    pass


DOCUMENT_TYPES = (
    "product_detail",
    "retailer_product",
    "catalogue",
    "collection",
    "location_page",
    "store_page",
    "blog_or_guide",
    "patent",
    "teardown",
    "other",
)

SOURCE_TYPES = (
    "official_product",
    "official_catalogue",
    "official_collection",
    "retailer_product",
    "patent",
    "teardown",
    "technical_article",
    "archive",
    "other",
)

PRODUCT_SCHEMA = {
    "type": "object",
    "properties": {
        "is_mattress_product": {"type": "boolean"},
        "is_specific_model": {"type": "boolean"},
        "brand": {"type": ["string", "null"]},
        "name": {"type": ["string", "null"]},
        "family": {"type": ["string", "null"]},
        "description": {"type": ["string", "null"]},
        "firmness": {"type": ["string", "null"]},
        "total_thickness_mm": {"type": ["number", "null"]},
        "product_weight_kg": {"type": ["number", "null"]},
        "price": {"type": ["number", "null"]},
        "currency": {"type": ["string", "null"]},
        "product_evidence_excerpt": {"type": ["string", "null"]},
        "layers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "position": {"type": "integer"},
                    "marketing_name": {"type": "string"},
                    "normalized_material": {"type": "string"},
                    "thickness_mm": {"type": ["number", "null"]},
                    "density_kg_m3": {"type": ["number", "null"]},
                    "evidence_excerpt": {"type": ["string", "null"]},
                },
                "required": [
                    "position",
                    "marketing_name",
                    "normalized_material",
                    "thickness_mm",
                    "density_kg_m3",
                    "evidence_excerpt",
                ],
                "additionalProperties": False,
            },
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "is_mattress_product",
        "is_specific_model",
        "brand",
        "name",
        "family",
        "description",
        "firmness",
        "total_thickness_mm",
        "product_weight_kg",
        "price",
        "currency",
        "product_evidence_excerpt",
        "layers",
        "warnings",
    ],
    "additionalProperties": False,
}

DOCUMENT_RECOGNITION_SCHEMA = {
    "type": "object",
    "properties": {
        "document_type": {"type": "string", "enum": list(DOCUMENT_TYPES)},
        "is_product_bearing": {"type": "boolean"},
        "recognition_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rejection_reason": {"type": ["string", "null"]},
        "products": {"type": "array", "items": PRODUCT_SCHEMA},
        "document_warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "document_type",
        "is_product_bearing",
        "recognition_confidence",
        "rejection_reason",
        "products",
        "document_warnings",
    ],
    "additionalProperties": False,
}

# Backward-compatible name used by older tests/integrations.
DOCUMENT_EXTRACTION_SCHEMA = DOCUMENT_RECOGNITION_SCHEMA

VISION_REGION_SCHEMA = {
    "type": "object",
    "properties": {
        "x": {"type": "number", "minimum": 0, "maximum": 1},
        "y": {"type": "number", "minimum": 0, "maximum": 1},
        "width": {"type": "number", "minimum": 0, "maximum": 1},
        "height": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["x", "y", "width", "height"],
    "additionalProperties": False,
}

VISION_LAYER_SCHEMA = {
    "type": "object",
    "properties": {
        "position": {"type": "integer", "minimum": 1},
        "marketing_name": {"type": "string"},
        "normalized_material": {"type": ["string", "null"]},
        "generic_material_class": {
            "type": "string",
            "enum": [
                "textile_or_cover",
                "polyurethane_foam",
                "memory_foam",
                "latex_foam",
                "gel_foam",
                "fiber_or_batting",
                "microcoil",
                "pocket_coil",
                "spring_unit",
                "base_foam",
                "other",
                "unknown",
            ],
        },
        "thickness_mm": {"type": ["number", "null"]},
        "density_kg_m3": {"type": ["number", "null"]},
        "visible_label": {"type": "string"},
        "callout_text": {"type": ["string", "null"]},
        "assignment_scope": {
            "type": "string",
            "enum": ["exact_layer", "layer_zone", "whole_product", "ambiguous"],
        },
        "evidence_status": {
            "type": "string",
            "enum": [
                "observed_label",
                "observed_measurement",
                "visually_classified",
                "unknown",
            ],
        },
        "region": {"anyOf": [VISION_REGION_SCHEMA, {"type": "null"}]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "position",
        "marketing_name",
        "normalized_material",
        "generic_material_class",
        "thickness_mm",
        "density_kg_m3",
        "visible_label",
        "callout_text",
        "assignment_scope",
        "evidence_status",
        "region",
        "confidence",
    ],
    "additionalProperties": False,
}

VISION_UNASSIGNED_REGION_SCHEMA = {
    "type": "object",
    "properties": {
        "position": {"type": "integer", "minimum": 1},
        "visual_description": {"type": "string"},
        "generic_material_class": {
            "type": "string",
            "enum": [
                "textile_or_cover",
                "polyurethane_foam",
                "memory_foam",
                "latex_foam",
                "gel_foam",
                "fiber_or_batting",
                "microcoil",
                "pocket_coil",
                "spring_unit",
                "base_foam",
                "other",
                "unknown",
            ],
        },
        "region": {"anyOf": [VISION_REGION_SCHEMA, {"type": "null"}]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "position",
        "visual_description",
        "generic_material_class",
        "region",
        "confidence",
    ],
    "additionalProperties": False,
}

VISION_EVIDENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "is_relevant": {"type": "boolean"},
        "asset_type": {
            "type": "string",
            "enum": [
                "layer_diagram",
                "cutaway_or_cross_section",
                "catalogue_page",
                "specification_table",
                "law_or_manufacturer_label",
                "teardown_frame",
                "product_image",
                "marketing_or_lifestyle_image",
                "other",
                "irrelevant",
            ],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "diagram_summary": {"type": ["string", "null"]},
        "visible_text": {"type": ["string", "null"]},
        "products": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "is_specific_model": {"type": "boolean"},
                    "brand": {"type": ["string", "null"]},
                    "name": {"type": ["string", "null"]},
                    "family": {"type": ["string", "null"]},
                    "description": {"type": ["string", "null"]},
                    "firmness": {"type": ["string", "null"]},
                    "total_thickness_mm": {"type": ["number", "null"]},
                    "product_weight_kg": {"type": ["number", "null"]},
                    "price": {"type": ["number", "null"]},
                    "currency": {"type": ["string", "null"]},
                    "visible_text": {"type": ["string", "null"]},
                    "layers": {"type": "array", "items": VISION_LAYER_SCHEMA},
                },
                "required": [
                    "is_specific_model",
                    "brand",
                    "name",
                    "family",
                    "description",
                    "firmness",
                    "total_thickness_mm",
                    "product_weight_kg",
                    "price",
                    "currency",
                    "visible_text",
                    "layers",
                ],
                "additionalProperties": False,
            },
        },
        "unassigned_regions": {
            "type": "array",
            "items": VISION_UNASSIGNED_REGION_SCHEMA,
        },
        "technology_terms": {"type": "array", "items": {"type": "string"}},
        "forensic_search_queries": {"type": "array", "items": {"type": "string"}},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "is_relevant",
        "asset_type",
        "confidence",
        "diagram_summary",
        "visible_text",
        "products",
        "unassigned_regions",
        "technology_terms",
        "forensic_search_queries",
        "warnings",
    ],
    "additionalProperties": False,
}


SEARCH_DISCOVERY_SCHEMA = {
    "type": "object",
    "properties": {
        "queries_used": {"type": "array", "items": {"type": "string"}},
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "title": {"type": ["string", "null"]},
                    "source_type": {"type": "string", "enum": list(SOURCE_TYPES)},
                    "product_name": {"type": ["string", "null"]},
                    "is_official": {"type": "boolean"},
                    "product_likelihood": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence_value": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"},
                },
                "required": [
                    "url",
                    "title",
                    "source_type",
                    "product_name",
                    "is_official",
                    "product_likelihood",
                    "evidence_value",
                    "reason",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["queries_used", "results"],
    "additionalProperties": False,
}


MATERIAL_EVIDENCE_SCOPES = (
    "exact_variant",
    "exact_product",
    "product_family",
    "technology",
    "manufacturer",
    "comparable_material",
    "generic_category",
)

MATERIAL_SOURCE_KINDS = (
    "official_product",
    "official_catalogue",
    "official_other",
    "retailer",
    "patent",
    "teardown",
    "other",
)

MATERIAL_EVIDENCE_DISCOVERY_SCHEMA = {
    "type": "object",
    "properties": {
        "queries_used": {"type": "array", "items": {"type": "string"}},
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_key": {"type": "string"},
                    "trademark_name": {"type": "string"},
                    "query": {"type": "string"},
                    "url": {"type": "string"},
                    "title": {"type": ["string", "null"]},
                    "source_kind": {"type": "string", "enum": list(MATERIAL_SOURCE_KINDS)},
                    "evidence_scope": {
                        "type": "string",
                        "enum": list(MATERIAL_EVIDENCE_SCOPES),
                    },
                    "relevance": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"},
                },
                "required": [
                    "candidate_key",
                    "trademark_name",
                    "query",
                    "url",
                    "title",
                    "source_kind",
                    "evidence_scope",
                    "relevance",
                    "reason",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["queries_used", "results"],
    "additionalProperties": False,
}

MATERIAL_DENSITY_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": [
                "verified_exact",
                "corroborated_range",
                "provisional_range",
                "generic_comparison_only",
                "unknown",
            ],
        },
        "grade": {
            "type": "string",
            "enum": [
                "A_manufacturer_exact",
                "B_corroborated_exact",
                "C_measured_teardown",
                "D_same_technology",
                "E_generic_category",
                "unknown",
            ],
        },
        "minimum_kg_m3": {"type": ["number", "null"]},
        "maximum_kg_m3": {"type": ["number", "null"]},
        "representative_kg_m3": {"type": ["number", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "basis": {"type": "string"},
    },
    "required": [
        "status",
        "grade",
        "minimum_kg_m3",
        "maximum_kg_m3",
        "representative_kg_m3",
        "confidence",
        "basis",
    ],
    "additionalProperties": False,
}

MATERIAL_EVIDENCE_SOURCE_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {"type": "string"},
        "title": {"type": ["string", "null"]},
        "source_kind": {"type": "string", "enum": list(MATERIAL_SOURCE_KINDS)},
        "evidence_scope": {"type": "string", "enum": list(MATERIAL_EVIDENCE_SCOPES)},
        "supports_identity": {"type": "boolean"},
        "supports_density": {"type": "boolean"},
        "excerpt": {"type": ["string", "null"]},
        "density_min_kg_m3": {"type": ["number", "null"]},
        "density_max_kg_m3": {"type": ["number", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "url",
        "title",
        "source_kind",
        "evidence_scope",
        "supports_identity",
        "supports_density",
        "excerpt",
        "density_min_kg_m3",
        "density_max_kg_m3",
        "confidence",
    ],
    "additionalProperties": False,
}

MATERIAL_DECODER_SCHEMA = {
    "type": "object",
    "properties": {
        "materials": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_key": {"type": "string"},
                    "visible_description": {"type": "string"},
                    "generic_material_class": {
                        "type": "string",
                        "enum": [
                            "polyurethane_foam",
                            "viscoelastic_memory_foam",
                            "gel_infused_memory_foam",
                            "latex_foam",
                            "fiber_or_batting",
                            "textile_or_cover",
                            "phase_change_material",
                            "microcoil",
                            "pocket_coil",
                            "spring_unit",
                            "adhesive_or_bonding",
                            "other",
                            "unresolved",
                        ],
                    },
                    "generic_material_name": {"type": "string"},
                    "actual_material_description": {"type": "string"},
                    "base_polymer": {"type": ["string", "null"]},
                    "additives_or_structure": {"type": "array", "items": {"type": "string"}},
                    "probable_functions": {"type": "array", "items": {"type": "string"}},
                    "stack_position": {"type": ["string", "null"]},
                    "identity_status": {
                        "type": "string",
                        "enum": [
                            "verified_primary",
                            "verified_corroborated",
                            "probable",
                            "unresolved",
                        ],
                    },
                    "identity_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence_scope": {
                        "type": "string",
                        "enum": list(MATERIAL_EVIDENCE_SCOPES),
                    },
                    "density": MATERIAL_DENSITY_SCHEMA,
                    "evidence_sources": {
                        "type": "array",
                        "items": MATERIAL_EVIDENCE_SOURCE_SCHEMA,
                    },
                    "search_queries_used": {"type": "array", "items": {"type": "string"}},
                    "contradictions": {"type": "array", "items": {"type": "string"}},
                    "unknowns": {"type": "array", "items": {"type": "string"}},
                    "conclusion": {"type": "string"},
                },
                "required": [
                    "candidate_key",
                    "visible_description",
                    "generic_material_class",
                    "generic_material_name",
                    "actual_material_description",
                    "base_polymer",
                    "additives_or_structure",
                    "probable_functions",
                    "stack_position",
                    "identity_status",
                    "identity_confidence",
                    "evidence_scope",
                    "density",
                    "evidence_sources",
                    "search_queries_used",
                    "contradictions",
                    "unknowns",
                    "conclusion",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["materials"],
    "additionalProperties": False,
}


def discovery_queries(
    company_name: str,
    official_domain: str,
    market: str,
    brand_aliases: list[str] | None = None,
    custom_queries: list[str] | None = None,
) -> list[str]:
    """Return complementary searches instead of trusting one broad query."""

    names = " OR ".join(f'"{name}"' for name in [company_name, *(brand_aliases or [])])
    built_in = [
        f'{names} exact mattress product models official catalogue {market} site:{official_domain}',
        f'{names} site:{official_domain} inurl:products mattress',
        f'{names} mattress layers foam density thickness construction specifications',
        f'{names} mattress catalogue brochure filetype:pdf',
        f'{names} mattress discontinued model archive retailer specification {market}',
        f'{names} mattress patent trademark material technology construction',
    ]
    return [*(query.strip() for query in custom_queries or [] if query.strip()), *built_in]


def _json_schema_format(name: str, schema: dict) -> dict:
    return {
        "type": "json_schema",
        "name": name,
        "strict": True,
        "schema": schema,
    }


def _extract_json_text(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise LLMError("Provider returned text that was not valid JSON.") from exc
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as nested:
            raise LLMError("Provider returned invalid structured JSON.") from nested
    if not isinstance(value, dict):
        raise LLMError("Provider structured response must be a JSON object.")
    return value


class LLMProvider(ABC):
    name: str
    discovery_log: list[dict[str, object]]

    @abstractmethod
    def discover_urls(
        self,
        company_name: str,
        official_domain: str,
        market: str,
        brand_aliases: list[str] | None = None,
        custom_queries: list[str] | None = None,
    ) -> list[str]:
        """Return candidate public evidence URLs."""

    @abstractmethod
    def extract_product(self, url: str, page_text: str) -> dict | None:
        """Extract explicit product evidence without performing inference."""

    def extract_products(self, url: str, page_text: str) -> list[dict]:
        product = self.extract_product(url, page_text)
        return [product] if product else []

    def recognize_document(self, url: str, page_text: str) -> dict:
        products = self.extract_products(url, page_text)
        return {
            "document_type": "other",
            "is_product_bearing": bool(products),
            "recognition_confidence": 0.5 if products else 0.0,
            "rejection_reason": None if products else "No specific product was extracted.",
            "products": products,
            "document_warnings": [],
        }

    def recognize_image(
        self,
        *,
        image_bytes: bytes,
        content_type: str,
        source_url: str,
        page_context: str = "",
    ) -> dict:
        raise LLMError(f"Image recognition is unavailable for provider {self.name}.")

    def verify_image_analysis(
        self,
        *,
        image_bytes: bytes,
        content_type: str,
        source_url: str,
        page_context: str,
        first_pass: dict,
    ) -> dict:
        return first_pass

    def discover_visual_evidence(
        self,
        *,
        company_name: str,
        official_domain: str,
        market: str,
        analyses: list[dict],
        limit: int = 8,
    ) -> list[str]:
        return []

    def discover_material_evidence(
        self,
        *,
        company_name: str,
        official_domain: str,
        market: str,
        candidates: list[dict],
        limit: int = 24,
    ) -> list[dict]:
        return []

    def decode_trademark_materials(
        self,
        *,
        company_name: str,
        official_domain: str,
        market: str,
        candidates: list[dict],
        evidence_documents: list[dict],
    ) -> list[dict]:
        return []

    def check_connection(self) -> dict:
        raise LLMError(f"Connection checks are unavailable for provider {self.name}.")


class DisabledLLMProvider(LLMProvider):
    name = "none"

    def __init__(self) -> None:
        self.discovery_log: list[dict[str, object]] = []

    def discover_urls(
        self,
        company_name: str,
        official_domain: str,
        market: str,
        brand_aliases: list[str] | None = None,
        custom_queries: list[str] | None = None,
    ) -> list[str]:
        return []

    def extract_product(self, url: str, page_text: str) -> dict | None:
        return None


@dataclass(slots=True)
class OpenAIProvider(LLMProvider):
    """OpenAI Responses API adapter for search and explicit product recognition only."""

    api_key: str
    model: str = "gpt-5-nano"
    timeout_seconds: float = 90.0
    max_search_queries: int = 6
    max_retries: int = 3
    min_product_likelihood: float = 0.58
    min_evidence_value: float = 0.62
    name: str = "openai"
    discovery_log: list[dict[str, object]] = field(default_factory=list)

    endpoint: str = "https://api.openai.com/v1/responses"
    models_endpoint: str = "https://api.openai.com/v1/models"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "brixta-mattress-intelligence/1.6",
        }

    def _request(self, payload: dict) -> dict:
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
                detail = http_error_detail(exc, limit=2_000)
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if retryable and attempt < self.max_retries:
                    retry_after = exc.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
                    time.sleep(min(delay, 20.0))
                    continue
                raise LLMError(f"OpenAI HTTP {exc.code}: {detail}") from exc
            except RETRYABLE_TRANSPORT_ERRORS + (json.JSONDecodeError,) as exc:
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 20.0))
                    continue
                raise LLMError(f"OpenAI request failed: {exc}") from exc
        raise LLMError("OpenAI request failed after retries.")

    @staticmethod
    def _response_text(response: dict) -> str:
        direct = response.get("output_text")
        if isinstance(direct, str) and direct.strip():
            return direct
        chunks: list[str] = []
        for item in response.get("output") or []:
            if item.get("type") != "message":
                continue
            for content in item.get("content") or []:
                if content.get("type") in {"output_text", "text"} and content.get("text"):
                    chunks.append(str(content["text"]))
        if not chunks:
            error = response.get("error") or response.get("incomplete_details") or {}
            raise LLMError(f"OpenAI returned no output text: {error}")
        return "\n".join(chunks)

    def _structured_request(
        self,
        *,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: dict,
        use_web_search: bool,
        reasoning_effort: str | None = None,
    ) -> dict:
        payload: dict = {
            "model": self.model,
            "instructions": instructions,
            "input": input_text,
            "text": {"format": _json_schema_format(schema_name, schema)},
            "store": False,
        }
        if reasoning_effort is not None:
            payload["reasoning"] = {"effort": reasoning_effort}
        if use_web_search:
            payload["tools"] = [{"type": "web_search"}]
            payload["tool_choice"] = "auto"
        return _extract_json_text(self._response_text(self._request(payload)))

    def check_connection(self) -> dict:
        model_url = f"{self.models_endpoint}/{quote(self.model, safe='-_.')}"
        request = Request(model_url, method="GET", headers=self._headers())
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = http_error_detail(exc, limit=2_000)
            raise LLMError(f"OpenAI model check HTTP {exc.code}: {detail}") from exc
        except RETRYABLE_TRANSPORT_ERRORS + (json.JSONDecodeError,) as exc:
            raise LLMError(f"OpenAI model check failed: {exc}") from exc
        return {
            "name": payload.get("id", self.model),
            "owned_by": payload.get("owned_by"),
            "object": payload.get("object"),
        }

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
        aliases = ", ".join(brand_aliases or []) or "none"
        result_limit = min(80, max(15, self.max_search_queries * 10))
        input_text = f"""
Company: {company_name}
Official domain: {official_domain}
Market: {market}
Known aliases: {aliases}
Search angles to execute or improve:
{chr(10).join(f'- {query}' for query in queries)}

Use web search and return at most {result_limit} high-value public evidence URLs.
Prioritise exact single-model product pages, official catalogues, retailer product pages,
archived model pages, patents, and teardowns. A page titled "Mattress in Delhi",
"Mattress for Pregnancy", a store locator, generic collection, category page, or broad blog
is not a product detail page. Keep a collection page only when it is valuable for discovering
links, and score its product_likelihood low. Do not infer any hidden construction values.
""".strip()
        payload = self._structured_request(
            instructions=(
                "You are BRIXTA's source-discovery worker. Your only job is to find and classify "
                "public evidence URLs. Use web search. Do not perform mattress construction analysis. "
                "Do not invent product names or technical facts."
            ),
            input_text=input_text,
            schema_name="mattress_source_discovery",
            schema=SEARCH_DISCOVERY_SCHEMA,
            use_web_search=True,
            reasoning_effort="low",
        )
        self.discovery_log.clear()
        accepted: list[str] = []
        seen: set[str] = set()
        queries_used = payload.get("queries_used") or queries
        for rank, raw in enumerate(payload.get("results") or [], start=1):
            url = str(raw.get("url") or "").strip().rstrip(".,;")
            if not url or not url.startswith(("http://", "https://")):
                continue
            product_likelihood = float(raw.get("product_likelihood") or 0.0)
            evidence_value = float(raw.get("evidence_value") or 0.0)
            source_type = str(raw.get("source_type") or "other")
            accepted_flag = (
                product_likelihood >= self.min_product_likelihood
                or evidence_value >= self.min_evidence_value
                or source_type in {"official_catalogue", "patent", "teardown", "archive"}
            )
            self.discovery_log.append(
                {
                    "query_number": None,
                    "query": " | ".join(str(item) for item in queries_used),
                    "rank": rank,
                    "url": url,
                    "title": raw.get("title"),
                    "score": round((0.6 * product_likelihood) + (0.4 * evidence_value), 4),
                    "product_likelihood": product_likelihood,
                    "evidence_value": evidence_value,
                    "source_type": source_type,
                    "product_name": raw.get("product_name"),
                    "is_official": bool(raw.get("is_official")),
                    "reason": raw.get("reason"),
                    "accepted": accepted_flag,
                    "source": "openai_web_search",
                    "model": self.model,
                }
            )
            if accepted_flag and url not in seen:
                seen.add(url)
                accepted.append(url)
        return accepted

    def recognize_document(self, url: str, page_text: str) -> dict:
        input_text = f"""
SOURCE URL: {url}

DOCUMENT TEXT:
{page_text[:70_000]}
""".strip()
        payload = self._structured_request(
            instructions=(
                "Classify one captured mattress-related document and extract only explicitly "
                "published product evidence. This is recognition and extraction only. Never estimate "
                "hidden density, thickness, weight, price, layer order, material composition, or "
                "configuration. A specific product must have an exact model name. Generic collection, "
                "location, city, store, category, pregnancy-use, and broad guide pages must not become "
                "product records. Catalogues may contain multiple exact models. Preserve short evidence "
                "excerpts for every non-null technical layer claim."
            ),
            input_text=input_text,
            schema_name="mattress_document_recognition",
            schema=DOCUMENT_RECOGNITION_SCHEMA,
            use_web_search=False,
            reasoning_effort=None,
        )
        admitted: list[dict] = []
        for item in payload.get("products") or []:
            if item.get("is_mattress_product") and item.get("is_specific_model") and item.get("name"):
                admitted.append(item)
        payload["products"] = admitted
        payload["is_product_bearing"] = bool(admitted)
        return payload

    def _image_structured_request(
        self,
        *,
        image_bytes: bytes,
        content_type: str,
        instructions: str,
        prompt: str,
        schema_name: str,
        schema: dict,
        reasoning_effort: str = "medium",
    ) -> dict:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{content_type.split(';', 1)[0]};base64,{encoded}"
        payload: dict = {
            "model": self.model,
            "instructions": instructions,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url, "detail": "high"},
                    ],
                }
            ],
            "text": {"format": _json_schema_format(schema_name, schema)},
            "reasoning": {"effort": reasoning_effort},
            "store": False,
        }
        return _extract_json_text(self._response_text(self._request(payload)))

    def recognize_image(
        self,
        *,
        image_bytes: bytes,
        content_type: str,
        source_url: str,
        page_context: str = "",
    ) -> dict:
        prompt = f"""
SOURCE PAGE OR CATALOGUE: {source_url}
PAGE CONTEXT (may be incomplete):
{page_context[:12_000]}

Perform a forensic reading of this mattress-related image.

1. Classify whether it is a technical layer diagram, cutaway/cross-section, catalogue page,
   specification table, label, teardown frame, ordinary product image, or irrelevant image.
2. Transcribe all visible product, technology, material, measurement, density, coil, and
   construction text exactly enough to search for it later.
3. For an exploded stack or cutaway, enumerate every visible region from top to bottom.
4. Link printed labels and callout lines to a precise layer only when the visual connection is
   clear. Otherwise use assignment_scope=layer_zone or ambiguous.
5. A colour or texture alone may support a broad visual class, never an exact chemistry,
   density, ILD, gauge, or thickness.
6. Numeric thickness and density may be returned only when visibly printed or measured in the
   image. Never estimate them from apparent scale.
7. Distinguish observed labels from visual classifications. Unlabelled slabs belong in
   unassigned_regions rather than being silently named.
8. Return normalized 0-to-1 bounding boxes when practical: x, y, width, height.
9. Generate focused forensic web-search queries using exact model names, proprietary technology
   terms, brochure/catalogue/specification/patent/teardown keywords, and quoted phrases.
10. Never turn a total mattress height option into an individual layer thickness.
""".strip()
        return self._image_structured_request(
            image_bytes=image_bytes,
            content_type=content_type,
            instructions=(
                "You are BRIXTA's forensic visual-evidence analyst. Extract and classify what the "
                "image actually supports. Preserve ambiguity. Unsupported values must remain null."
            ),
            prompt=prompt,
            schema_name="mattress_forensic_visual_evidence",
            schema=VISION_EVIDENCE_SCHEMA,
            reasoning_effort="medium",
        )

    def verify_image_analysis(
        self,
        *,
        image_bytes: bytes,
        content_type: str,
        source_url: str,
        page_context: str,
        first_pass: dict,
    ) -> dict:
        prompt = f"""
SOURCE PAGE OR CATALOGUE: {source_url}
PAGE CONTEXT:
{page_context[:8_000]}

FIRST-PASS ANALYSIS:
{json.dumps(first_pass, ensure_ascii=False)[:24_000]}

Audit the first-pass result against the image. Return a corrected complete result using the same
schema. Delete unsupported claims, fix layer order, preserve labels verbatim, and lower confidence
where a callout targets a zone rather than one exact slab. Thickness, density, ILD, coil gauge, and
chemistry must remain null unless explicitly printed or visibly measured. Keep useful exact search
queries for corroborating brochures, archived pages, dealer specifications, patents, labels, and
teardowns.
""".strip()
        return self._image_structured_request(
            image_bytes=image_bytes,
            content_type=content_type,
            instructions=(
                "You are the second-pass evidence auditor. Prefer false negatives over false "
                "technical claims. Do not preserve a first-pass claim merely because it was supplied."
            ),
            prompt=prompt,
            schema_name="mattress_forensic_visual_audit",
            schema=VISION_EVIDENCE_SCHEMA,
            reasoning_effort="medium",
        )

    def discover_visual_evidence(
        self,
        *,
        company_name: str,
        official_domain: str,
        market: str,
        analyses: list[dict],
        limit: int = 8,
    ) -> list[str]:
        terms: list[str] = []
        queries: list[str] = []
        product_names: list[str] = []
        for analysis in analyses:
            terms.extend(str(item).strip() for item in analysis.get("technology_terms") or [])
            queries.extend(
                str(item).strip() for item in analysis.get("forensic_search_queries") or []
            )
            for product in analysis.get("products") or []:
                name = str(product.get("name") or "").strip()
                if name:
                    product_names.append(name)
        terms = list(dict.fromkeys(item for item in terms if item))[:24]
        queries = list(dict.fromkeys(item for item in queries if item))[:24]
        product_names = list(dict.fromkeys(product_names))[:12]
        if not terms and not queries and not product_names:
            return []

        prompt = f"""
Company: {company_name}
Official domain: {official_domain}
Market: {market}
Product/model names read from technical images: {', '.join(product_names) or 'none'}
Proprietary technology/material terms: {', '.join(terms) or 'none'}
Forensic queries proposed by visual analysis:
{chr(10).join(f'- {query}' for query in queries) or '- none'}

Use web search to find at most {max(1, limit)} public URLs that can corroborate or clarify the
same image/model/technology. Prioritise manufacturer catalogues, high-resolution copies of the
same diagram, dealer technical PDFs, archived product pages, patents, law/manufacturer labels,
and exact-model teardowns. Reject generic sleep blogs, location pages, and unrelated models.
Do not claim that a visually similar foam is chemically identical. Return URLs only when the
model name, proprietary text, diagram identity, SKU, or a strong technical relationship makes
it useful evidence.
""".strip()
        payload = self._structured_request(
            instructions=(
                "You are BRIXTA's visual-evidence corroboration search worker. Find public source "
                "documents; do not infer hidden mattress specifications."
            ),
            input_text=prompt,
            schema_name="mattress_visual_followup_discovery",
            schema=SEARCH_DISCOVERY_SCHEMA,
            use_web_search=True,
            reasoning_effort="low",
        )
        accepted: list[str] = []
        seen: set[str] = set()
        for rank, raw in enumerate(payload.get("results") or [], start=1):
            url = str(raw.get("url") or "").strip().rstrip(".,;")
            if not url.startswith(("http://", "https://")) or url in seen:
                continue
            evidence_value = float(raw.get("evidence_value") or 0.0)
            source_type = str(raw.get("source_type") or "other")
            accept = evidence_value >= 0.58 or source_type in {
                "official_catalogue",
                "patent",
                "teardown",
                "archive",
                "retailer_product",
            }
            self.discovery_log.append(
                {
                    "query_number": None,
                    "query": " | ".join(str(item) for item in payload.get("queries_used") or queries),
                    "rank": rank,
                    "url": url,
                    "title": raw.get("title"),
                    "score": evidence_value,
                    "product_likelihood": raw.get("product_likelihood"),
                    "evidence_value": evidence_value,
                    "source_type": source_type,
                    "product_name": raw.get("product_name"),
                    "is_official": bool(raw.get("is_official")),
                    "reason": raw.get("reason"),
                    "accepted": accept,
                    "source": "openai_visual_followup",
                    "model": self.model,
                }
            )
            if accept:
                seen.add(url)
                accepted.append(url)
                if len(accepted) >= limit:
                    break
        return accepted

    def discover_material_evidence(
        self,
        *,
        company_name: str,
        official_domain: str,
        market: str,
        candidates: list[dict],
        limit: int = 24,
    ) -> list[dict]:
        if not candidates or limit <= 0:
            return []
        compact_candidates = [
            {
                "candidate_key": item.get("candidate_key"),
                "trademark_name": item.get("trademark_name"),
                "product_name": item.get("product_name"),
                "family": item.get("family"),
                "visible_label": item.get("visible_label"),
                "callout_text": item.get("callout_text"),
                "diagram_summary": item.get("diagram_summary"),
                "current_generic_class": item.get("current_generic_class"),
                "source_page_url": item.get("source_page_url"),
                "search_queries": (item.get("search_queries") or [])[:8],
            }
            for item in candidates[:40]
        ]
        prompt = f"""
Company: {company_name}
Official domain: {official_domain}
Market: {market}

TRADEMARKED OR PROPRIETARY MATERIAL CANDIDATES:
{json.dumps(compact_candidates, ensure_ascii=False, indent=2)[:48_000]}

Use web search to find at most {limit} high-value public documents that help determine what these
named materials actually are and, where publicly disclosed, their numeric density in kg/m³.
Search exact quoted names, product/model combinations, manufacturer catalogues, dealer training
sheets, technical PDFs, archived pages, patents, teardowns, labels, and procurement documents.

Priorities:
1. Exact manufacturer or official technical statements.
2. Exact-product or exact-family dealer technical documents.
3. Independent teardowns or measured specifications.
4. Same proprietary technology used in another identified product.
5. Generic material-category documents only as comparison evidence.

Do not treat marketing adjectives such as high-density as a number. Do not convert lb/ft³ unless
the source explicitly supplies the value and context. Do not assign a source from a different
market or model generation to an exact product; classify its scope honestly. Every URL must be a
public result actually found during web search. Return the candidate_key that each source supports.
""".strip()
        payload = self._structured_request(
            instructions=(
                "You are BRIXTA's trademark-material source investigator. Find public evidence that "
                "decodes proprietary mattress material names. Discovery only: do not manufacture a "
                "density or composition and do not treat generic ranges as product specifications."
            ),
            input_text=prompt,
            schema_name="mattress_material_evidence_discovery",
            schema=MATERIAL_EVIDENCE_DISCOVERY_SCHEMA,
            use_web_search=True,
            reasoning_effort="medium",
        )
        accepted: list[dict] = []
        seen: set[tuple[str, str]] = set()
        valid_keys = {str(item.get("candidate_key")) for item in compact_candidates}
        for raw in payload.get("results") or []:
            key = str(raw.get("candidate_key") or "")
            url = str(raw.get("url") or "").strip().rstrip(".,;")
            relevance = float(raw.get("relevance") or 0.0)
            if key not in valid_keys or not url.startswith(("http://", "https://")):
                continue
            fingerprint = (key, url)
            if fingerprint in seen or relevance < 0.5:
                continue
            seen.add(fingerprint)
            row = dict(raw)
            row["url"] = url
            accepted.append(row)
            self.discovery_log.append(
                {
                    "query_number": None,
                    "query": raw.get("query"),
                    "rank": len(accepted),
                    "url": url,
                    "title": raw.get("title"),
                    "score": relevance,
                    "source_type": raw.get("source_kind"),
                    "product_name": raw.get("trademark_name"),
                    "is_official": str(raw.get("source_kind") or "").startswith("official"),
                    "reason": raw.get("reason"),
                    "accepted": True,
                    "source": "openai_material_decoder_search",
                    "model": self.model,
                    "candidate_key": key,
                    "evidence_scope": raw.get("evidence_scope"),
                }
            )
            if len(accepted) >= limit:
                break
        return accepted

    def decode_trademark_materials(
        self,
        *,
        company_name: str,
        official_domain: str,
        market: str,
        candidates: list[dict],
        evidence_documents: list[dict],
    ) -> list[dict]:
        if not candidates:
            return []
        compact_documents = []
        remaining = 72_000
        for document in evidence_documents:
            if remaining <= 0:
                break
            text = str(document.get("text") or "")[:12_000]
            row = {
                "url": document.get("url"),
                "title": document.get("title"),
                "source_kind": document.get("source_kind"),
                "candidate_keys": document.get("candidate_keys") or [],
                "text": text,
            }
            compact_documents.append(row)
            remaining -= len(text)
        prompt = f"""
Company: {company_name}
Official domain: {official_domain}
Market: {market}

VISUAL MATERIAL CANDIDATES:
{json.dumps(candidates[:40], ensure_ascii=False, indent=2)[:48_000]}

FETCHED CORROBORATING DOCUMENTS:
{json.dumps(compact_documents, ensure_ascii=False, indent=2)[:80_000]}

For every candidate_key, produce one final trademark-material dossier.

The business question is: what industrial material is hidden behind the proprietary name, and is
there defensible density evidence? Compare exact wording, claimed function, base material,
additives, structure, stack position, product family, market, and model generation. Strip away
marketing language but do not overstate the conclusion.

Rules:
- A diagram label is evidence that the named technology appears, not proof of unseen chemistry.
- "High density", "premium", "HD", colour, pore appearance, or visual thickness are never numeric density.
- A manufacturer exact numeric value is grade A.
- Two independent exact technical sources agreeing is grade B.
- A physically measured teardown is grade C.
- The same proprietary technology in a different product/market is grade D and provisional only.
- A generic material-category range is grade E, comparison only, never a product specification.
- If no defensible value exists, density status must be unknown and all numeric density fields null.
- Patent evidence normally proves technology-level possibilities, not exact commercial SKU use.
- Preserve contradictions and unresolved formulation details.
- Every evidence source must be one of the fetched document URLs or the candidate's original source.
- Return vivid but technical physical/functional descriptions, not promotional copy.
""".strip()
        payload = self._structured_request(
            instructions=(
                "You are BRIXTA's trademark-material adjudicator. Use only the supplied visual facts "
                "and fetched public documents. Prefer an explicit unknown over a plausible invention. "
                "Produce concise evidence-backed conclusions suitable for engineering comparison."
            ),
            input_text=prompt,
            schema_name="mattress_trademark_material_decoder",
            schema=MATERIAL_DECODER_SCHEMA,
            use_web_search=False,
            reasoning_effort="medium",
        )
        return list(payload.get("materials") or [])

    def extract_product(self, url: str, page_text: str) -> dict | None:
        products = self.extract_products(url, page_text)
        return products[0] if products else None

    def extract_products(self, url: str, page_text: str) -> list[dict]:
        return list(self.recognize_document(url, page_text).get("products") or [])


@dataclass(slots=True)
class GeminiProvider(LLMProvider):
    """Dependency-free Gemini REST adapter retained as an optional extraction provider."""

    api_key: str
    model: str = "gemini-3.5-flash"
    timeout_seconds: float = 60.0
    max_search_queries: int = 6
    max_retries: int = 3
    name: str = "gemini"
    discovery_log: list[dict[str, object]] = field(default_factory=list)

    @property
    def endpoint(self) -> str:
        model = quote(self.model, safe="-_.")
        return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    @property
    def model_endpoint(self) -> str:
        model = quote(self.model, safe="-_.")
        return f"https://generativelanguage.googleapis.com/v1beta/models/{model}"

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-goog-api-key": self.api_key,
            "x-goog-api-client": "brixta-mattress-intelligence/1.6",
        }

    def _request(self, payload: dict) -> dict:
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
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if retryable and attempt < self.max_retries:
                    retry_after = exc.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
                    time.sleep(min(delay, 20.0))
                    continue
                raise LLMError(f"Gemini HTTP {exc.code}: {detail}") from exc
            except RETRYABLE_TRANSPORT_ERRORS + (json.JSONDecodeError,) as exc:
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 20.0))
                    continue
                raise LLMError(f"Gemini request failed: {exc}") from exc
        raise LLMError("Gemini request failed after retries.")

    def check_connection(self) -> dict:
        request = Request(self.model_endpoint, method="GET", headers=self._headers())
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = http_error_detail(exc, limit=1_000)
            raise LLMError(f"Gemini model check HTTP {exc.code}: {detail}") from exc
        except RETRYABLE_TRANSPORT_ERRORS + (json.JSONDecodeError,) as exc:
            raise LLMError(f"Gemini model check failed: {exc}") from exc
        return {
            "name": payload.get("name", self.model),
            "display_name": payload.get("displayName"),
            "input_token_limit": payload.get("inputTokenLimit"),
            "output_token_limit": payload.get("outputTokenLimit"),
            "methods": payload.get("supportedGenerationMethods", []),
        }

    @staticmethod
    def _response_text(response: dict) -> str:
        candidates = response.get("candidates") or []
        if not candidates:
            raise LLMError(f"Gemini returned no candidates: {response.get('promptFeedback', {})}")
        parts = candidates[0].get("content", {}).get("parts", [])
        return "\n".join(str(part.get("text", "")) for part in parts if part.get("text"))

    def discover_urls(
        self,
        company_name: str,
        official_domain: str,
        market: str,
        brand_aliases: list[str] | None = None,
        custom_queries: list[str] | None = None,
    ) -> list[str]:
        urls: list[str] = []
        self.discovery_log.clear()
        queries = discovery_queries(
            company_name,
            official_domain,
            market,
            brand_aliases,
            custom_queries,
        )[: self.max_search_queries]
        for query_index, query in enumerate(queries, start=1):
            prompt = f"""
Use Google Search to find public evidence for this exact research query:

{query}

Return sources, not guesses. Prefer exact product pages and primary catalogues. Keep useful
retailer product pages, archived pages, patents, and teardowns. Do not estimate hidden mattress
properties or recommend a construction.
""".strip()
            response = self._request(
                {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "tools": [{"google_search": {}}],
                    "generationConfig": {"temperature": 0.0, "maxOutputTokens": 4_096},
                }
            )
            candidate = (response.get("candidates") or [{}])[0]
            metadata = candidate.get("groundingMetadata", {})
            query_urls: list[str] = []
            for chunk in metadata.get("groundingChunks", []):
                web = chunk.get("web", {})
                uri = web.get("uri")
                if uri:
                    cleaned = str(uri).rstrip(".,;")
                    query_urls.append(cleaned)
                    self.discovery_log.append(
                        {
                            "query_number": query_index,
                            "query": query,
                            "url": cleaned,
                            "title": web.get("title"),
                            "source": "gemini_grounding_metadata",
                        }
                    )
            text = self._response_text(response)
            for raw_url in re.findall(r"https?://[^\s)\]}>\"']+", text):
                cleaned = raw_url.rstrip(".,;")
                query_urls.append(cleaned)
                self.discovery_log.append(
                    {
                        "query_number": query_index,
                        "query": query,
                        "url": cleaned,
                        "title": None,
                        "source": "gemini_grounded_response_text",
                    }
                )
            urls.extend(query_urls)
        return list(dict.fromkeys(url.rstrip(".,;") for url in urls))

    def recognize_document(self, url: str, page_text: str) -> dict:
        prompt = f"""
Classify and extract explicit evidence from one captured document. Return exact mattress models
only. Generic collection, location, store, city, category, and broad guide pages must not become
product records. Never estimate density, thickness, weight, price, layer order, or materials.
Use null when absent and preserve short evidence excerpts.

SOURCE URL: {url}

PAGE TEXT:
{page_text[:60_000]}
""".strip()
        response = self._request(
            {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.0,
                    "responseMimeType": "application/json",
                    "responseJsonSchema": DOCUMENT_RECOGNITION_SCHEMA,
                },
            }
        )
        result = _extract_json_text(self._response_text(response))
        admitted = [
            item
            for item in result.get("products", [])
            if item.get("is_mattress_product")
            and item.get("is_specific_model", True)
            and item.get("name")
        ]
        result["products"] = admitted
        result["is_product_bearing"] = bool(admitted)
        return result

    def extract_product(self, url: str, page_text: str) -> dict | None:
        products = self.extract_products(url, page_text)
        return products[0] if products else None

    def extract_products(self, url: str, page_text: str) -> list[dict]:
        return list(self.recognize_document(url, page_text).get("products") or [])


def build_llm_provider(
    provider: str,
    gemini_api_key: str | None,
    gemini_model: str,
    max_search_queries: int = 6,
    *,
    openai_api_key: str | None = None,
    openai_model: str = "gpt-5-nano",
    timeout_seconds: float = 90.0,
) -> LLMProvider:
    normalized = provider.strip().casefold()
    if normalized in {"", "none", "disabled", "heuristic"}:
        return DisabledLLMProvider()
    if normalized == "openai":
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when MATTRESS_INTEL_LLM_PROVIDER=openai.")
        return OpenAIProvider(
            api_key=openai_api_key,
            model=openai_model,
            max_search_queries=max(1, min(max_search_queries, 12)),
            timeout_seconds=timeout_seconds,
        )
    if normalized == "gemini":
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when MATTRESS_INTEL_LLM_PROVIDER=gemini.")
        return GeminiProvider(
            api_key=gemini_api_key,
            model=gemini_model,
            max_search_queries=max(1, min(max_search_queries, 12)),
            timeout_seconds=timeout_seconds,
        )
    raise ValueError(
        f"Unsupported LLM provider: {provider}. Supported values: none, openai, gemini."
    )
