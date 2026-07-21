"""Nearest-product evidence for candidate priors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import ProductRecord


@dataclass(frozen=True, slots=True)
class SimilarProduct:
    product_id: str
    name: str
    score: float
    density_evidence: dict[str, float]


class ProductSimilarityIndex:
    """Standalone TF-IDF/cosine product-similarity index."""

    def __init__(self, products: list[ProductRecord]) -> None:
        self.products = products
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        if products:
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                max_features=20_000,
                sublinear_tf=True,
            )
            self.matrix = self.vectorizer.fit_transform(
                product.searchable_text for product in products
            )

    def nearest(self, product: ProductRecord, limit: int = 5) -> list[SimilarProduct]:
        if self.vectorizer is None or self.matrix is None:
            return []
        query = self.vectorizer.transform([product.searchable_text])
        scores = cosine_similarity(query, self.matrix)[0]
        order = np.argsort(scores)[::-1]
        results: list[SimilarProduct] = []
        for index in order:
            reference = self.products[int(index)]
            if reference.product_id == product.product_id:
                continue
            score = float(scores[int(index)])
            if score <= 0:
                continue
            evidence = {
                f"{layer.normalized_material}:{int(round(layer.density_kg_m3))}": score
                for layer in reference.layers
                if layer.density_kg_m3 is not None
            }
            results.append(
                SimilarProduct(
                    product_id=str(reference.product_id),
                    name=f"{reference.brand} {reference.name}",
                    score=score,
                    density_evidence=evidence,
                )
            )
            if len(results) >= limit:
                break
        return results


def density_support(neighbors: list[SimilarProduct]) -> dict[str, float]:
    support: dict[str, float] = {}
    total = 0.0
    for neighbor in neighbors:
        for key, score in neighbor.density_evidence.items():
            weight = score
            support[key] = support.get(key, 0.0) + weight
            total += weight
    if total > 0:
        support = {key: value / total for key, value in support.items()}
    return support
