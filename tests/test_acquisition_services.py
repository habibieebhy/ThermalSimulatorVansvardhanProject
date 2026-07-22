from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mattress_intelligence.firecrawl import FirecrawlClient
from mattress_intelligence.jina import JinaSearchClient
from mattress_intelligence.object_store import LocalObjectStore


class AcquisitionServiceTests(unittest.TestCase):
    def test_jina_markdown_search_is_normalized(self) -> None:
        class FakeJina(JinaSearchClient):
            def _get(self, url: str, extra_headers=None):
                return """Title: Alpha Mattress\nURL Source: https://example.test/products/alpha\nPublished Time: x\nMarkdown Content:\nLayer construction and density details."""

        results = FakeJina(None).search("alpha mattress", limit=5)
        self.assertEqual(results[0].url, "https://example.test/products/alpha")
        self.assertEqual(results[0].title, "Alpha Mattress")

    def test_firecrawl_search_and_asset_manifest_are_normalized(self) -> None:
        class FakeFirecrawl(FirecrawlClient):
            def _post(self, path: str, payload: dict) -> dict:
                if path == "search":
                    return {
                        "success": True,
                        "data": {"web": [{"url": "https://example.test/p", "title": "P", "description": "Mattress"}]},
                    }
                return {
                    "success": True,
                    "data": {
                        "rawHtml": "<html><body>Product</body></html>",
                        "images": [{"url": "https://example.test/layers.png", "alt": "layer diagram", "width": 1200, "height": 800}],
                        "links": ["https://example.test/products/alpha"],
                        "metadata": {"url": "https://example.test/p", "title": "P"},
                    },
                }

        client = FakeFirecrawl("unused")
        self.assertEqual(client.search("mattress")[0].url, "https://example.test/p")
        page = client.scrape("https://example.test/p")
        self.assertEqual(page.images[0].alt_text, "layer diagram")
        self.assertIn("Product", page.raw_html)

    def test_local_object_store_is_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = LocalObjectStore(Path(temp))
            left = store.put_bytes(b"same", content_type="text/plain", source_url="urn:a", namespace="docs")
            right = store.put_bytes(b"same", content_type="text/plain", source_url="urn:b", namespace="docs")
            self.assertEqual(left.sha256, right.sha256)
            self.assertEqual(left.local_path, right.local_path)
            self.assertEqual(store.get_bytes(local_path=left.local_path, object_uri=None), b"same")


if __name__ == "__main__":
    unittest.main()
