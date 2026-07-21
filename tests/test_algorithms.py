from __future__ import annotations

import unittest

from mattress_intelligence.configurations import ConfigurationGenerator
from mattress_intelligence.inference import BayesianCandidateRanker
from mattress_intelligence.materials import MaterialLibrary
from mattress_intelligence.models import LayerRecord, ProductRecord, VariantRecord
from mattress_intelligence.similarity import ProductSimilarityIndex


def product() -> ProductRecord:
    return ProductRecord(
        company_id="cmp_test",
        company_name="Example",
        brand="Example",
        name="Example Medium Firm Mattress",
        description="Memory comfort with HR support",
        firmness="Medium Firm",
        total_thickness_mm=200,
        product_weight_kg=19.0,
        layers=[
            LayerRecord(
                position=1,
                marketing_name="Memory foam",
                normalized_material="memory_foam",
                thickness_mm=50,
            ),
            LayerRecord(
                position=2,
                marketing_name="HR foam",
                normalized_material="hr_foam",
                thickness_mm=150,
            ),
        ],
        variants=[
            VariantRecord(width_mm=1520, length_mm=1980, thickness_mm=200, weight_kg=19.0)
        ],
    )


class AlgorithmTests(unittest.TestCase):
    def setUp(self) -> None:
        self.materials = MaterialLibrary.load()

    def test_constraint_generator_preserves_total_thickness(self) -> None:
        report = ConfigurationGenerator(self.materials).generate(product(), max_candidates=10)
        self.assertTrue(report.candidates)
        for candidate in report.candidates:
            self.assertEqual(sum(layer.thickness_mm for layer in candidate.layers), 200)

    def test_bayesian_ranker_orders_and_bounds_probabilities(self) -> None:
        target = product()
        candidates = ConfigurationGenerator(self.materials).generate(
            target, max_candidates=10
        ).candidates
        ranked = BayesianCandidateRanker(self.materials).rank(target, candidates, limit=10)
        self.assertTrue(ranked)
        self.assertEqual([item.rank for item in ranked], list(range(1, len(ranked) + 1)))
        self.assertTrue(all(0.0 <= item.posterior_probability <= 1.0 for item in ranked))
        self.assertTrue(
            all(
                ranked[index].posterior_probability >= ranked[index + 1].posterior_probability
                for index in range(len(ranked) - 1)
            )
        )

    def test_similarity_uses_density_evidence(self) -> None:
        reference = product().model_copy(deep=True)
        reference.name = "Reference HR Mattress"
        reference.product_id = "prd_reference"
        reference.layers[1].density_kg_m3 = 32
        neighbors = ProductSimilarityIndex([reference]).nearest(product())
        self.assertEqual(neighbors[0].product_id, "prd_reference")
        self.assertGreater(neighbors[0].score, 0)
        self.assertIn("hr_foam:32", neighbors[0].density_evidence)


if __name__ == "__main__":
    unittest.main()

