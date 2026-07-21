"""Logical knowledge-graph construction and traversal."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from .models import (
    ClaimRecord,
    ConfigurationCandidate,
    EvidenceObservation,
    ProductRecord,
    SourceRecord,
    stable_id,
)
from .normalization import normalized_name


def edge(source: str, relation: str, target: str, **properties: Any) -> dict[str, Any]:
    return {
        "edge_id": stable_id("edg", source, relation, target),
        "source_node": source,
        "relation": relation,
        "target_node": target,
        "properties": properties,
    }


class KnowledgeGraph:
    """Portable edge list for evidence, similarity, and inferred configurations."""

    def __init__(self, edges: list[dict[str, Any]] | None = None) -> None:
        self.edges = edges or []

    @classmethod
    def build(
        cls,
        products: list[ProductRecord],
        sources: list[SourceRecord],
        claims: list[ClaimRecord],
        configurations: list[ConfigurationCandidate],
        *,
        observations: list[EvidenceObservation] | None = None,
        similarity_matches: list[dict[str, Any]] | None = None,
    ) -> "KnowledgeGraph":
        result: list[dict[str, Any]] = []
        source_by_id = {source.source_id: source for source in sources}
        products_by_normalized_name = {
            normalized_name(product.name): str(product.product_id) for product in products
        }

        for product in products:
            product_id = str(product.product_id)
            result.append(edge(product.company_id, "HAS_PRODUCT", product_id))
            brand_id = stable_id("brd", product.company_id, product.brand)
            result.append(edge(product.company_id, "HAS_BRAND", brand_id, name=product.brand))
            result.append(edge(brand_id, "SELLS", product_id))
            for layer in product.layers:
                material_id = f"mat_{layer.normalized_material}"
                result.append(
                    edge(
                        product_id,
                        "HAS_LAYER",
                        str(layer.layer_id),
                        position=layer.position,
                        marketing_name=layer.marketing_name,
                    )
                )
                result.append(edge(str(layer.layer_id), "NORMALIZES_TO", material_id))
                for evidence_ref in layer.evidence:
                    if evidence_ref.source_id in source_by_id:
                        result.append(
                            edge(str(layer.layer_id), "SUPPORTED_BY", evidence_ref.source_id)
                        )
            for source_id in product.source_ids:
                if source_id in source_by_id:
                    result.append(edge(product_id, "SUPPORTED_BY", source_id))

        for claim in claims:
            result.append(edge(claim.product_id, "HAS_CLAIM", claim.claim_id, field=claim.field_path))
            for evidence_ref in claim.evidence:
                result.append(edge(claim.claim_id, "SUPPORTED_BY", evidence_ref.source_id))

        for observation in observations or []:
            result.append(
                edge(
                    observation.source_id,
                    "HAS_OBSERVATION",
                    observation.observation_id,
                    field=observation.field_path,
                    method=observation.method,
                    confidence=observation.confidence,
                )
            )
            if observation.normalized_material:
                result.append(
                    edge(
                        observation.observation_id,
                        "NORMALIZES_TO",
                        f"mat_{observation.normalized_material}",
                    )
                )
            if observation.product_name_hint:
                product_id = products_by_normalized_name.get(
                    normalized_name(observation.product_name_hint)
                )
                if product_id:
                    result.append(edge(product_id, "HAS_OBSERVATION", observation.observation_id))

        for item in similarity_matches or []:
            product_id = str(item.get("product_id") or "")
            similar_id = str(item.get("similar_product_id") or "")
            if product_id and similar_id:
                result.append(
                    edge(
                        product_id,
                        "SIMILAR_TO",
                        similar_id,
                        cosine_similarity=item.get("cosine_similarity"),
                        reference_scope=item.get("reference_scope"),
                    )
                )

        for candidate in configurations:
            result.append(
                edge(
                    candidate.product_id,
                    "HAS_POSSIBLE_CONFIGURATION",
                    candidate.configuration_id,
                    rank=candidate.rank,
                    posterior=candidate.posterior_probability,
                )
            )

        deduplicated = {item["edge_id"]: item for item in result}
        return cls(list(deduplicated.values()))

    def traverse(
        self,
        start_node: str,
        max_depth: int = 3,
        relations: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        adjacency: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in self.edges:
            if relations is None or item["relation"] in relations:
                adjacency[item["source_node"]].append(item)
        visited = {start_node}
        pending = deque([(start_node, 0)])
        found: list[dict[str, Any]] = []
        while pending:
            node, depth = pending.popleft()
            if depth >= max_depth:
                continue
            for item in adjacency[node]:
                found.append(item)
                target = item["target_node"]
                if target not in visited:
                    visited.add(target)
                    pending.append((target, depth + 1))
        return found
