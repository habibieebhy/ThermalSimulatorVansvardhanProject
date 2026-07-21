"""Conservative product entity resolution and duplicate merging."""

from __future__ import annotations

from .models import LayerRecord, ProductRecord
from .normalization import name_similarity, normalized_name


def _choose(primary, secondary):
    return primary if primary not in (None, "", []) else secondary


def _merge_layers(left: list[LayerRecord], right: list[LayerRecord]) -> list[LayerRecord]:
    merged = [layer.model_copy(deep=True) for layer in left]
    by_key = {(layer.position, layer.normalized_material): layer for layer in merged}
    for incoming in right:
        key = (incoming.position, incoming.normalized_material)
        current = by_key.get(key)
        if current is None:
            copy = incoming.model_copy(deep=True)
            if any(layer.position == copy.position for layer in merged):
                copy.position = max((layer.position for layer in merged), default=0) + 1
            merged.append(copy)
            by_key[(copy.position, copy.normalized_material)] = copy
            continue
        current.marketing_name = _choose(current.marketing_name, incoming.marketing_name)
        current.thickness_mm = _choose(current.thickness_mm, incoming.thickness_mm)
        current.density_kg_m3 = _choose(current.density_kg_m3, incoming.density_kg_m3)
        if current.thickness_mm == incoming.thickness_mm:
            current.thickness_status = incoming.thickness_status
        if current.density_kg_m3 == incoming.density_kg_m3:
            current.density_status = incoming.density_status
        existing_evidence = {item.model_dump_json() for item in current.evidence}
        current.evidence.extend(
            item for item in incoming.evidence if item.model_dump_json() not in existing_evidence
        )
    return sorted(merged, key=lambda layer: layer.position)


def merge_products(left: ProductRecord, right: ProductRecord) -> ProductRecord:
    primary, secondary = (
        (left, right)
        if left.extraction_confidence >= right.extraction_confidence
        else (right, left)
    )
    payload = primary.model_dump(exclude={"layers", "variants", "source_ids"})
    for field in (
        "family",
        "canonical_url",
        "description",
        "firmness",
        "total_thickness_mm",
        "product_weight_kg",
        "price",
        "currency",
    ):
        payload[field] = _choose(getattr(primary, field), getattr(secondary, field))
    variants = {variant.variant_id: variant for variant in primary.variants}
    variants.update({variant.variant_id: variant for variant in secondary.variants})
    payload.update(
        layers=_merge_layers(primary.layers, secondary.layers),
        variants=list(variants.values()),
        source_ids=list(dict.fromkeys(primary.source_ids + secondary.source_ids)),
        tags=list(dict.fromkeys(primary.tags + secondary.tags)),
        extraction_method="merged",
        extraction_confidence=max(primary.extraction_confidence, secondary.extraction_confidence),
    )
    return ProductRecord(**payload)


class ProductEntityResolver:
    """Merge only high-similarity records from the same brand and company."""

    def __init__(self, similarity_threshold: float = 0.93) -> None:
        self.similarity_threshold = similarity_threshold

    def resolve(self, products: list[ProductRecord]) -> list[ProductRecord]:
        resolved: list[ProductRecord] = []
        for product in products:
            match_index: int | None = None
            for index, existing in enumerate(resolved):
                if existing.company_id != product.company_id:
                    continue
                if normalized_name(existing.brand) != normalized_name(product.brand):
                    continue
                if normalized_name(existing.name) == normalized_name(product.name):
                    match_index = index
                    break
                if name_similarity(existing.name, product.name) >= self.similarity_threshold:
                    match_index = index
                    break
            if match_index is None:
                resolved.append(product.model_copy(deep=True))
            else:
                resolved[match_index] = merge_products(resolved[match_index], product)
        return resolved

