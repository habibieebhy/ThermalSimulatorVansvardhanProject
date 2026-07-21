from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from mattress_intelligence.crawler import FetchedDocument
from mattress_intelligence.extraction import ProductExtractor
from mattress_intelligence.llm import DisabledLLMProvider
from mattress_intelligence.materials import MaterialLibrary
from mattress_intelligence.models import CompanyResearchRequest


HTML = b"""
<!doctype html>
<html><head><title>Alpha Mattress</title>
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Product",
  "name": "Alpha Mattress",
  "brand": {"@type": "Brand", "name": "Example Sleep"},
  "description": "A 200 mm mattress with 40 mm memory foam at 50 kg/m3 and HR foam.",
  "offers": {"price": "19999", "priceCurrency": "INR"}
}
</script></head><body><h1>Alpha Mattress</h1></body></html>
"""


class ExtractionTests(unittest.TestCase):
    def test_json_ld_product_is_extracted_without_llm(self) -> None:
        document = FetchedDocument(
            url="https://example.com/mattress/alpha",
            status=200,
            content_type="text/html; charset=utf-8",
            body=HTML,
            retrieved_at_epoch=time.time(),
            artifact_path="artifact.html",
        )
        request = CompanyResearchRequest(
            company_name="Example Sleep", official_domain="https://example.com"
        )
        extractor = ProductExtractor(MaterialLibrary.load(), DisabledLLMProvider())
        product, source = extractor.extract(document, request)
        self.assertIsNotNone(product)
        assert product is not None
        self.assertEqual(product.name, "Alpha Mattress")
        self.assertEqual(product.total_thickness_mm, 200.0)
        self.assertEqual(product.price, 19999.0)
        self.assertEqual(source.url, document.url)
        self.assertTrue(any(layer.normalized_material == "memory_foam" for layer in product.layers))


    def test_deterministic_document_observations_without_llm(self) -> None:
        document = FetchedDocument(
            url="https://example.com/mattress/alpha-pro",
            status=200,
            content_type="text/html; charset=utf-8",
            body=b"""
            <html><head><title>Alpha Pro Mattress</title></head><body>
            <h1>Alpha Pro Mattress</h1>
            <table><tr><th>Construction</th><td>50 mm memory foam at 55 kg/m3</td></tr></table>
            <p>Total thickness: 200 mm. Weight: 19 kg. Medium firm. Rs. 24,999.
            Size 72 x 60 x 8 inches. 10 years warranty.</p>
            </body></html>
            """,
            retrieved_at_epoch=time.time(),
            artifact_path="artifact.html",
        )
        request = CompanyResearchRequest(
            company_name="Example", official_domain="https://example.com"
        )
        extractor = ProductExtractor(MaterialLibrary.load(), DisabledLLMProvider())
        products, _, observations = extractor.extract_document(document, request)
        self.assertEqual(len(products), 1)
        fields = {item.field_path for item in observations}
        self.assertIn("material.mention", fields)
        self.assertIn("measurement.density_kg_m3", fields)
        self.assertIn("measurement.thickness_mm", fields)
        self.assertIn("measurement.weight_kg", fields)
        self.assertIn("commercial.price", fields)
        self.assertIn("variant.dimensions_mm", fields)
        self.assertIn("commercial.warranty_years", fields)
        self.assertTrue(products[0].variants)

    def test_location_collection_page_is_not_admitted_as_product(self) -> None:
        document = FetchedDocument(
            url="https://example.com/collections/mattress-in-delhi",
            status=200,
            content_type="text/html; charset=utf-8",
            body=b"""
            <html><head><title>Mattress in Delhi</title></head><body>
            <h1>Mattress in Delhi</h1>
            <p>Buy mattresses in Delhi. Choose a 5 inch mattress with memory foam,
            latex, 10 year warranty, and great prices.</p>
            </body></html>
            """,
            retrieved_at_epoch=time.time(),
            artifact_path="location.html",
        )
        request = CompanyResearchRequest(
            company_name="Example", official_domain="https://example.com"
        )
        extractor = ProductExtractor(MaterialLibrary.load(), DisabledLLMProvider())
        products, _, observations = extractor.extract_document(document, request)
        self.assertEqual(products, [])
        self.assertTrue(observations)  # evidence can remain without becoming a product
        self.assertFalse(extractor.recognition_log[-1]["accepted"])

    def test_product_page_does_not_promote_unscoped_five_inch_copy(self) -> None:
        document = FetchedDocument(
            url="https://example.com/products/alpha-pro-mattress",
            status=200,
            content_type="text/html; charset=utf-8",
            body=b"""
            <html><head><title>Alpha Pro Mattress</title>
            <meta name="description" content="Alpha Pro orthopedic mattress."></head><body>
            <h1>Alpha Pro Mattress</h1><main><p>Add to cart. 10 year warranty.</p></main>
            <footer>Read our guide to choosing a 5 inch mattress.</footer>
            </body></html>
            """,
            retrieved_at_epoch=time.time(),
            artifact_path="product.html",
        )
        request = CompanyResearchRequest(
            company_name="Example", official_domain="https://example.com"
        )
        product, _ = ProductExtractor(MaterialLibrary.load(), DisabledLLMProvider()).extract(
            document, request
        )
        self.assertIsNotNone(product)
        assert product is not None
        self.assertIsNone(product.total_thickness_mm)

    def test_external_article_can_supply_explicit_evidence_through_llm(self) -> None:
        class ArticleLLM(DisabledLLMProvider):
            name = "test"

            def extract_products(self, url: str, page_text: str) -> list[dict]:
                return [
                    {
                        "is_mattress_product": True,
                        "brand": "Example Sleep",
                        "name": "Archive Pro",
                        "family": None,
                        "description": "The article states the construction.",
                        "firmness": "medium firm",
                        "total_thickness_mm": 200,
                        "product_weight_kg": None,
                        "price": None,
                        "currency": None,
                        "layers": [
                            {
                                "position": 1,
                                "marketing_name": "HR foam",
                                "normalized_material": "hr_foam",
                                "thickness_mm": 150,
                                "density_kg_m3": 32,
                                "evidence_excerpt": "150 mm HR foam at 32 kg/m3",
                            }
                        ],
                        "warnings": [],
                    }
                ]

        article = b"""
        <html><head><title>Archived construction notes</title></head><body>
        Example Sleep Archive Pro mattress specification: 150 mm HR foam at 32 kg/m3,
        with a total thickness of 200 mm and medium firmness.
        </body></html>
        """
        document = FetchedDocument(
            url="https://independent.example/articles/archive-pro-review",
            status=200,
            content_type="text/html; charset=utf-8",
            body=article,
            retrieved_at_epoch=time.time(),
            artifact_path="article.html",
        )
        request = CompanyResearchRequest(
            company_name="Example Sleep", official_domain="https://example.com"
        )
        products, source = ProductExtractor(MaterialLibrary.load(), ArticleLLM()).extract_many(
            document, request
        )
        self.assertEqual(len(products), 1)
        self.assertEqual(products[0].layers[0].density_kg_m3, 32)
        self.assertFalse(source.is_official)


if __name__ == "__main__":
    unittest.main()
