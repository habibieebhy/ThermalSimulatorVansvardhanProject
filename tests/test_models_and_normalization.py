from __future__ import annotations

import unittest
import os

from mattress_intelligence.models import LayerRecord, ProductRecord
from mattress_intelligence.settings import Settings
from mattress_intelligence.normalization import canonicalize_url, parse_density_kg_m3, parse_first_thickness_mm


class NormalizationTests(unittest.TestCase):
    def test_settings_read_environment_when_instantiated(self) -> None:
        previous = os.environ.get("GEMINI_MODEL")
        os.environ["GEMINI_MODEL"] = "gemini-test-model"
        try:
            self.assertEqual(Settings().gemini_model, "gemini-test-model")
        finally:
            if previous is None:
                os.environ.pop("GEMINI_MODEL", None)
            else:
                os.environ["GEMINI_MODEL"] = previous

    def test_url_tracking_parameters_are_removed(self) -> None:
        self.assertEqual(
            canonicalize_url("https://EXAMPLE.com/products/bed/?utm_source=x&size=queen#buy"),
            "https://example.com/products/bed?size=queen",
        )

    def test_engineering_units_are_parsed(self) -> None:
        self.assertEqual(parse_first_thickness_mm("Total thickness: 8 inches"), 203.2)
        self.assertEqual(parse_density_kg_m3("HR foam density 32 kg/m³"), 32.0)

    def test_product_and_child_ids_are_deterministic(self) -> None:
        left = ProductRecord(
            company_id="cmp_1",
            company_name="Example",
            brand="Example",
            name="Alpha Mattress",
            layers=[
                LayerRecord(
                    position=1,
                    marketing_name="HR Foam",
                    normalized_material="hr_foam",
                )
            ],
        )

        right = ProductRecord(
            company_id="cmp_1",
            company_name="Example",
            brand="Example",
            name="Alpha Mattress",
            layers=[
                LayerRecord(
                    position=1,
                    marketing_name="HR Foam",
                    normalized_material="hr_foam",
                )
            ],
        )

        self.assertEqual(left.product_id, right.product_id)
        self.assertEqual(left.layers[0].layer_id, right.layers[0].layer_id)

if __name__ == "__main__":
    unittest.main()
