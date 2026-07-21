from __future__ import annotations

import time
import unittest

from mattress_intelligence.crawler import CatalogueCrawler, FetchedDocument, _url_priority


class FakeFetcher:
    def robots_sitemaps(self, base_url: str) -> list[str]:
        return []

    def fetch(self, url: str) -> FetchedDocument:
        if url.endswith("/sitemap.xml"):
            body = b"""<?xml version='1.0'?><urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>
            <url><loc>https://example.com/products/alpha-mattress</loc></url>
            <url><loc>https://example.com/privacy</loc></url>
            </urlset>"""
            content_type = "application/xml"
        elif url == "https://example.com/":
            body = b"""<html><body>
            <a href='/about'>About</a>
            <a href='/products/alpha-mattress'>Alpha</a>
            <a href='/checkout'>Checkout</a>
            </body></html>"""
            content_type = "text/html"
        elif url == "https://example.com/products/alpha-mattress":
            body = b"<html><body><h1>Alpha Mattress</h1></body></html>"
            content_type = "text/html"
        else:
            body = b"<html><body>Other</body></html>"
            content_type = "text/html"
        return FetchedDocument(
            url=url,
            status=200,
            content_type=content_type,
            body=body,
            retrieved_at_epoch=time.time(),
            artifact_path=f"artifact-{len(body)}",
        )

    def close(self) -> None:
        return None


class CrawlerTests(unittest.TestCase):
    def test_sitemap_first_priority_crawl_and_rejection_log(self) -> None:
        report = CatalogueCrawler(FakeFetcher()).crawl(
            "https://example.com",
            max_pages=2,
            max_depth=2,
        )
        fetched = [document.url for document in report.documents]
        self.assertEqual(fetched[0], "https://example.com/")
        self.assertIn("https://example.com/products/alpha-mattress", fetched)
        self.assertNotIn("https://example.com/privacy", fetched)
        self.assertTrue(
            any(
                item.get("action") == "rejected" and item.get("url") == "https://example.com/privacy"
                for item in report.crawl_log
            )
        )
        self.assertTrue(any(item.get("action") == "fetched" for item in report.crawl_log))

    def test_product_detail_outscores_and_location_pages_are_rejected(self) -> None:
        self.assertGreater(
            _url_priority("https://example.test/products/alpha-mattress"),
            _url_priority("https://example.test/collections/mattress"),
        )
        self.assertLess(
            _url_priority("https://example.test/collections/mattress-in-delhi"),
            0,
        )


if __name__ == "__main__":
    unittest.main()
