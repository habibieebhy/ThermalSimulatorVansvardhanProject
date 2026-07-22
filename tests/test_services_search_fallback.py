from mattress_intelligence.search import (
    FirecrawlSearchProvider,
    JinaSearchProvider,
    ServicesSearchProvider,
)


def test_services_provider_stops_after_successful_firecrawl(monkeypatch) -> None:
    calls = []

    def firecrawl_discover(self, *args, **kwargs):
        calls.append("firecrawl")
        self.discovery_log.append({"source": "firecrawl_search"})
        return ["https://example.com/products/a"]

    def jina_discover(self, *args, **kwargs):
        calls.append("jina")
        return ["https://example.com/products/b"]

    monkeypatch.setattr(FirecrawlSearchProvider, "discover_urls", firecrawl_discover)
    monkeypatch.setattr(JinaSearchProvider, "discover_urls", jina_discover)

    provider = ServicesSearchProvider("jina-key", "firecrawl-key")
    urls = provider.discover_urls("Company", "https://example.com", "India")

    assert urls == ["https://example.com/products/a"]
    assert calls == ["firecrawl"]


def test_services_provider_falls_back_to_jina_when_firecrawl_is_empty(monkeypatch) -> None:
    calls = []

    def firecrawl_discover(self, *args, **kwargs):
        calls.append("firecrawl")
        return []

    def jina_discover(self, *args, **kwargs):
        calls.append("jina")
        return ["https://example.com/products/b"]

    monkeypatch.setattr(FirecrawlSearchProvider, "discover_urls", firecrawl_discover)
    monkeypatch.setattr(JinaSearchProvider, "discover_urls", jina_discover)

    provider = ServicesSearchProvider("jina-key", "firecrawl-key")
    urls = provider.discover_urls("Company", "https://example.com", "India")

    assert urls == ["https://example.com/products/b"]
    assert calls == ["firecrawl", "jina"]
