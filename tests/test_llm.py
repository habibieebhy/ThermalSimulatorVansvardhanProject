from __future__ import annotations

import unittest

from mattress_intelligence.llm import GeminiProvider, OpenAIProvider, discovery_queries


class GeminiCollectionTests(unittest.TestCase):
    def test_search_plan_includes_aliases_and_custom_queries_first(self) -> None:
        queries = discovery_queries(
            "Example Bedding",
            "example.test",
            "India",
            ["Example Sleep"],
            ["custom archive query"],
        )
        self.assertEqual(queries[0], "custom archive query")
        self.assertTrue(any('"Example Sleep"' in query for query in queries[1:]))
        self.assertTrue(any("filetype:pdf" in query for query in queries))

    def test_grounded_discovery_records_query_provenance_and_deduplicates(self) -> None:
        class FakeGemini(GeminiProvider):
            def _request(self, payload: dict) -> dict:
                return {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "Evidence https://example.test/catalogue.pdf"}]
                            },
                            "groundingMetadata": {
                                "groundingChunks": [
                                    {
                                        "web": {
                                            "uri": "https://example.test/catalogue.pdf",
                                            "title": "Catalogue",
                                        }
                                    }
                                ]
                            },
                        }
                    ]
                }

        provider = FakeGemini(api_key="not-used", max_search_queries=2)
        urls = provider.discover_urls("Example Bedding", "example.test", "India")
        self.assertEqual(urls, ["https://example.test/catalogue.pdf"])
        self.assertEqual(len(provider.discovery_log or []), 4)
        self.assertEqual((provider.discovery_log or [])[0]["query_number"], 1)


class OpenAIRecognitionTests(unittest.TestCase):
    def test_openai_search_filters_generic_pages_and_keeps_product_evidence(self) -> None:
        class FakeOpenAI(OpenAIProvider):
            def _structured_request(self, **kwargs) -> dict:
                if kwargs["schema_name"] == "mattress_source_discovery":
                    return {
                        "queries_used": ["exact product search"],
                        "results": [
                            {
                                "url": "https://example.test/collections/mattress-in-delhi",
                                "title": "Mattress in Delhi",
                                "source_type": "official_collection",
                                "product_name": None,
                                "is_official": True,
                                "product_likelihood": 0.08,
                                "evidence_value": 0.12,
                                "reason": "Location SEO page",
                            },
                            {
                                "url": "https://example.test/products/alpha-pro-mattress",
                                "title": "Alpha Pro Mattress",
                                "source_type": "official_product",
                                "product_name": "Alpha Pro Mattress",
                                "is_official": True,
                                "product_likelihood": 0.98,
                                "evidence_value": 0.91,
                                "reason": "Exact product detail page",
                            },
                        ],
                    }
                return {
                    "document_type": "product_detail",
                    "is_product_bearing": True,
                    "recognition_confidence": 0.96,
                    "rejection_reason": None,
                    "products": [],
                    "document_warnings": [],
                }

        provider = FakeOpenAI(api_key="not-used", max_search_queries=2)
        urls = provider.discover_urls("Example", "example.test", "India")
        self.assertEqual(urls, ["https://example.test/products/alpha-pro-mattress"])
        self.assertFalse(provider.discovery_log[0]["accepted"])
        self.assertTrue(provider.discovery_log[1]["accepted"])

    def test_openai_recognition_rejects_non_specific_products(self) -> None:
        class FakeOpenAI(OpenAIProvider):
            def _structured_request(self, **kwargs) -> dict:
                return {
                    "document_type": "collection",
                    "is_product_bearing": True,
                    "recognition_confidence": 0.92,
                    "rejection_reason": "Generic catalogue hub",
                    "products": [
                        {
                            "is_mattress_product": True,
                            "is_specific_model": False,
                            "brand": "Example",
                            "name": "Mattresses",
                            "family": None,
                            "description": None,
                            "firmness": None,
                            "total_thickness_mm": None,
                            "product_weight_kg": None,
                            "price": None,
                            "currency": None,
                            "product_evidence_excerpt": None,
                            "layers": [],
                            "warnings": [],
                        }
                    ],
                    "document_warnings": [],
                }

        result = FakeOpenAI(api_key="not-used").recognize_document(
            "https://example.test/collections/mattress", "Mattresses"
        )
        self.assertEqual(result["products"], [])
        self.assertFalse(result["is_product_bearing"])


if __name__ == "__main__":
    unittest.main()
