"""Pure helpers used by the Streamlit research control plane."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from .models import CompanyResearchRequest
from .settings import Settings


def normalize_official_domain(value: str) -> str:
    """Return a canonical HTTPS origin or raise a user-facing validation error."""

    cleaned = value.strip()
    if not cleaned:
        raise ValueError("Enter the company's official website.")
    if "://" not in cleaned:
        cleaned = f"https://{cleaned}"
    parsed = urlsplit(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Enter a valid official website, for example https://example.com.")
    netloc = parsed.netloc
    return urlunsplit((parsed.scheme, netloc, "", "", "")).rstrip("/")


def build_research_request(
    company_name: str,
    official_domain: str,
    settings: Settings,
) -> CompanyResearchRequest:
    """Build the full, balanced research request used by the one-click UI."""

    company = " ".join(company_name.split())
    if not company:
        raise ValueError("Enter the company name.")
    domain = normalize_official_domain(official_domain)
    return CompanyResearchRequest(
        company_name=company,
        official_domain=domain,
        market=settings.default_market,
        include_external_evidence=True,
        use_search_grounding=settings.search_provider not in {"", "none", "disabled"},
        discover_assets=True,
        analyze_assets_with_vision=settings.llm_provider == "openai" and bool(settings.openai_api_key),
        max_pages=settings.ui_max_pages,
        max_external_pages=settings.ui_max_external_pages,
        max_crawl_depth=settings.ui_max_crawl_depth,
        max_assets_per_document=settings.ui_max_assets_per_document,
        max_vision_assets=settings.ui_max_vision_assets,
        max_pdf_pages=settings.ui_max_pdf_pages,
        max_configurations_per_product=settings.ui_max_configurations_per_product,
    )


def worker_settings_overrides(settings: Settings) -> dict[str, object]:
    """Select the working acquisition path without transmitting any credentials."""

    provider = settings.search_provider
    jina_reader_enabled = settings.jina_reader_enabled

    # Firecrawl is already configured and verified in the intended local stack. Prefer it for
    # discovery and capture so an invalid or unfunded Jina account cannot stall the entire job.
    if settings.firecrawl_api_key:
        provider = "firecrawl"
        jina_reader_enabled = False

    return {
        "search_provider": provider,
        "jina_reader_enabled": jina_reader_enabled,
        "search_queries": min(settings.search_queries, 4),
        "capture_strategy": "services_first",
    }
