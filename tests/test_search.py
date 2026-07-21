from __future__ import annotations

import unittest

from mattress_intelligence.search import TavilySearchProvider


class TavilySearchTests(unittest.TestCase):
    def test_basic_search_records_sources_and_credit_usage(self) -> None:
        class FakeTavily(TavilySearchProvider):
            def _post(self, payload: dict) -> dict:
                self.assert_basic(payload)
                return {
                    "results": [
                        {
                            "title": "Old catalogue",
                            "url": "https://example.test/catalogue.pdf",
                            "content": "A relevant catalogue",
                            "score": 0.91,
                        }
                    ],
                    "usage": {"credits": 1},
                    "request_id": "request-test",
                }

            @staticmethod
            def assert_basic(payload: dict) -> None:
                if payload["search_depth"] != "basic" or payload["auto_parameters"]:
                    raise AssertionError("Search must remain on predictable one-credit settings")

        provider = FakeTavily(api_key="not-used", max_search_queries=2)
        urls = provider.discover_urls("Example Sleep", "example.test", "India")
        self.assertEqual(urls, ["https://example.test/catalogue.pdf"])
        self.assertEqual(len(provider.discovery_log or []), 2)
        self.assertTrue(all(item["credits"] == 1 for item in provider.discovery_log or []))


if __name__ == "__main__":
    unittest.main()
