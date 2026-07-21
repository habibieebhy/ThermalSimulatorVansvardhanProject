"""Transparent Bayesian candidate ranking and provisional confidence scoring."""

from __future__ import annotations

import math

import numpy as np

from .materials import MaterialLibrary
from .models import ConfigurationCandidate, ProductRecord
from .normalization import firmness_score
from .similarity import SimilarProduct, density_support


def _gaussian_log_likelihood(observed: float, predicted: float, sigma: float) -> float:
    sigma = max(sigma, 1e-9)
    z = (observed - predicted) / sigma
    return -0.5 * z * z - math.log(sigma * math.sqrt(2.0 * math.pi))


class BayesianCandidateRanker:
    """Rank constraint survivors using explicit priors and interpretable likelihoods."""

    def __init__(self, materials: MaterialLibrary) -> None:
        self.materials = materials

    def _density_log_prior(self, candidate: ConfigurationCandidate) -> float:
        score = 0.0
        for layer in candidate.layers:
            spec = self.materials.get(layer.material)
            prior_by_density = dict(zip(spec.densities_kg_m3, spec.density_prior, strict=True))
            score += math.log(max(prior_by_density.get(layer.density_kg_m3, 0.01), 1e-8))
        return score

    def _predicted_firmness(self, candidate: ConfigurationCandidate) -> float:
        total = max(1, candidate.total_thickness_mm)
        return sum(
            self.materials.get(layer.material).firmness_index * layer.thickness_mm / total
            for layer in candidate.layers
        )

    def _evidence_score(self, product: ProductRecord, neighbors: list[SimilarProduct]) -> float:
        observed = 0.0
        possible = 4.0 + 2.0 * max(1, len(product.layers))
        observed += float(product.total_thickness_mm is not None)
        observed += float(product.product_weight_kg is not None)
        observed += float(product.price is not None)
        observed += float(product.firmness is not None)
        for layer in product.layers:
            observed += float(layer.thickness_mm is not None)
            observed += float(layer.density_kg_m3 is not None)
        source_quality = min(1.0, 0.35 + 0.15 * len(product.source_ids))
        similarity_quality = max((item.score for item in neighbors), default=0.0)
        return min(1.0, 0.65 * (observed / possible) + 0.2 * source_quality + 0.15 * similarity_quality)

    def rank(
        self,
        product: ProductRecord,
        candidates: list[ConfigurationCandidate],
        neighbors: list[SimilarProduct] | None = None,
        limit: int = 10,
    ) -> list[ConfigurationCandidate]:
        if not candidates:
            return []
        neighbors = neighbors or []
        support = density_support(neighbors)
        target_firmness = firmness_score(product.firmness)
        observed_weight = product.product_weight_kg or next(
            (variant.weight_kg for variant in product.variants if variant.weight_kg), None
        )
        evidence_score = self._evidence_score(product, neighbors)

        log_scores: list[float] = []
        reasons_by_candidate: list[list[str]] = []
        for candidate in candidates:
            log_score = self._density_log_prior(candidate)
            reasons = ["Material-specific density priors applied"]
            if observed_weight is not None and candidate.estimated_weight_kg is not None:
                sigma = max(1.5, observed_weight * 0.12)
                log_score += _gaussian_log_likelihood(
                    observed_weight, candidate.estimated_weight_kg, sigma
                )
                reasons.append("Calculated layer mass compared with observed product weight")
            if target_firmness is not None:
                predicted_firmness = self._predicted_firmness(candidate)
                log_score += _gaussian_log_likelihood(target_firmness, predicted_firmness, 0.18)
                reasons.append("Material stack compared with stated firmness")
            similarity_log_bonus = 0.0
            matches = 0
            for layer in candidate.layers:
                key = f"{layer.material}:{layer.density_kg_m3}"
                if support.get(key, 0.0) > 0:
                    similarity_log_bonus += math.log1p(4.0 * support[key])
                    matches += 1
            log_score += similarity_log_bonus
            if matches:
                reasons.append(f"{matches} density choice(s) supported by similar products")
            log_scores.append(log_score)
            reasons_by_candidate.append(reasons)

        shifted = np.asarray(log_scores, dtype=float) - float(np.max(log_scores))
        weights = np.exp(np.clip(shifted, -700.0, 0.0))
        probabilities = weights / float(np.sum(weights))
        order = np.argsort(probabilities)[::-1][:limit]

        ranked: list[ConfigurationCandidate] = []
        for rank, index in enumerate(order, start=1):
            candidate = candidates[int(index)].model_copy(deep=True)
            posterior = float(probabilities[int(index)])
            candidate.rank = rank
            candidate.posterior_probability = posterior
            candidate.evidence_score = evidence_score
            candidate.confidence_score = round(
                100.0 * min(0.99, posterior * (0.55 + 0.45 * evidence_score)), 2
            )
            candidate.reasons = reasons_by_candidate[int(index)]
            if product.total_thickness_mm is None:
                candidate.contradictions.append("Total thickness is a research baseline, not observed.")
            if not product.layers:
                candidate.contradictions.append("Layer pattern is generic because layer order is undisclosed.")
            ranked.append(candidate)
        return ranked

