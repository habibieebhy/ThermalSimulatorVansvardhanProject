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
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


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

VISION_EVIDENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "is_relevant": {"type": "boolean"},
        "asset_type": {
            "type": "string",
            "enum": [
                "layer_diagram",
                "catalogue_page",
                "specification_table",
                "product_image",
                "other",
            ],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
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
                    "layers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "position": {"type": "integer"},
                                "marketing_name": {"type": "string"},
                                "normalized_material": {"type": ["string", "null"]},
                                "thickness_mm": {"type": ["number", "null"]},
                                "density_kg_m3": {"type": ["number", "null"]},
                                "visible_label": {"type": "string"},
                            },
                            "required": [
                                "position",
                                "marketing_name",
                                "normalized_material",
                                "thickness_mm",
                                "density_kg_m3",
                                "visible_label",
                            ],
                            "additionalProperties": False,
                        },
                    },
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
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["is_relevant", "asset_type", "confidence", "products", "warnings"],
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
            "User-Agent": "brixta-mattress-intelligence/1.3",
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
                detail = exc.read().decode("utf-8", errors="replace")[:2_000]
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if retryable and attempt < self.max_retries:
                    retry_after = exc.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
                    time.sleep(min(delay, 20.0))
                    continue
                raise LLMError(f"OpenAI HTTP {exc.code}: {detail}") from exc
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
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
            detail = exc.read().decode("utf-8", errors="replace")[:2_000]
            raise LLMError(f"OpenAI model check HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
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

    def recognize_image(
        self,
        *,
        image_bytes: bytes,
        content_type: str,
        source_url: str,
        page_context: str = "",
    ) -> dict:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{content_type.split(';', 1)[0]};base64,{encoded}"
        prompt = f"""
SOURCE PAGE OR CATALOGUE: {source_url}
PAGE CONTEXT (may be incomplete):
{page_context[:8_000]}

Read only text and construction information visibly present in this image. Identify exact
mattress models when the model name is visible. For layer diagrams, preserve the visible
top-to-bottom order and exact marketing labels. Convert explicit lengths to millimetres and
explicit density values to kg/m^3. Never infer hidden materials, chemistry, density, thickness,
weight, price, or product identity. Use null when a value is not visibly present.
""".strip()
        payload = {
            "model": self.model,
            "instructions": (
                "You are an evidence transcription worker. The output is observed image evidence, "
                "not mattress construction analysis. Do not guess or complete missing labels."
            ),
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url, "detail": "high"},
                    ],
                }
            ],
            "text": {
                "format": _json_schema_format(
                    "mattress_image_evidence", VISION_EVIDENCE_SCHEMA
                )
            },
            "store": False,
        }
        return _extract_json_text(self._response_text(self._request(payload)))

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
            "x-goog-api-client": "brixta-mattress-intelligence/1.3",
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
                detail = exc.read().decode("utf-8", errors="replace")[:1_000]
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if retryable and attempt < self.max_retries:
                    retry_after = exc.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
                    time.sleep(min(delay, 20.0))
                    continue
                raise LLMError(f"Gemini HTTP {exc.code}: {detail}") from exc
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
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
            detail = exc.read().decode("utf-8", errors="replace")[:1_000]
            raise LLMError(f"Gemini model check HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
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
