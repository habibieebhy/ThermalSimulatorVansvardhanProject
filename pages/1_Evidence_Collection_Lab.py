"""Streamlit control plane for evidence collection and deterministic inference review."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mattress_intelligence.models import CompanyResearchRequest, ResearchResult
from mattress_intelligence.pipeline import MattressIntelligencePipeline
from mattress_intelligence.settings import Settings


st.title("BRIXTA Mattress Product Intelligence")
st.caption(
    "OpenAI may search, classify documents, and extract explicit facts. Knowledge graphs, "
    "similarity, constraints, Bayesian ranking, and confidence are deterministic code paths."
)

configured = Settings()

with st.sidebar:
    st.header("Research job")
    mode = st.radio(
        "Input mode",
        ["Live collection only", "Live collection + algorithms", "Offline demo"],
        index=0,
    )
    company = st.text_input("Company", "The Sleep Company")
    domain = st.text_input("Official website", "https://thesleepcompany.in")
    market = st.text_input("Market", "India")
    aliases_text = st.text_input(
        "Brand aliases",
        help="Comma-separated old names, spellings, sub-brands, or legal company names.",
    )
    seed_urls_text = st.text_area(
        "Known source URLs",
        help="One official, archive, retailer, patent, teardown, or catalogue URL per line.",
    )
    custom_queries_text = st.text_area(
        "Custom search queries",
        help="One precise query per line. These run before built-in product/evidence searches.",
    )
    max_pages = st.number_input("Maximum official pages", 1, 10_000, 100)
    max_external_pages = st.number_input(
        "Maximum external pages",
        0,
        2_000,
        25,
        help="Exact external results are fetched once; external sites are never recursively crawled.",
    )
    max_crawl_depth = st.slider("Maximum official-site crawl depth", 0, 12, 4)
    max_configurations = st.slider("Configurations per product", 1, 30, 10)
    st.divider()

    recognition_provider = st.selectbox(
        "Search and product-recognition provider",
        ["openai", "none", "gemini"],
        index=0 if configured.llm_provider in {"", "none", "openai"} else 2,
        help=(
            "The model is restricted to source discovery, page classification, and extraction "
            "of explicitly published facts. It is not used for inference analysis."
        ),
    )
    openai_key = st.text_input(
        "OpenAI API key",
        type="password",
        disabled=recognition_provider != "openai",
        help="Leave blank to use OPENAI_API_KEY from the private .env file.",
    )
    openai_model = st.text_input(
        "OpenAI model",
        configured.openai_model,
        disabled=recognition_provider != "openai",
    )
    gemini_key = st.text_input(
        "Gemini API key",
        type="password",
        disabled=recognition_provider != "gemini",
    )
    gemini_model = st.text_input(
        "Gemini model",
        configured.gemini_model,
        disabled=recognition_provider != "gemini",
    )

    search_provider = st.selectbox(
        "Web search provider",
        ["openai", "tavily", "none", "gemini"],
        index=0,
        help=(
            "OpenAI search returns URLs already classified by product likelihood and evidence value. "
            "Tavily remains available as a broader URL finder."
        ),
    )
    tavily_key = st.text_input(
        "Tavily API key",
        type="password",
        disabled=search_provider != "tavily",
        help="Leave blank to use TAVILY_API_KEY from .env.",
    )
    search_queries = st.slider(
        "Search angles",
        1,
        12,
        configured.search_queries,
        disabled=search_provider == "none",
    )
    recognition_threshold = st.slider(
        "Product admission threshold",
        0.50,
        0.95,
        float(configured.product_recognition_threshold),
        0.01,
        help="Higher values admit fewer heuristic/LLM-recognized documents as products.",
    )
    use_search = st.checkbox(
        "Use web search discovery",
        value=True,
        disabled=search_provider == "none",
    )
    external = st.checkbox("Fetch external evidence", value=True, disabled=not use_search)
    run_clicked = st.button("Start data collection", type="primary", use_container_width=True)


def run_pipeline() -> ResearchResult:
    offline = mode == "Offline demo"
    base = Settings()
    settings = replace(
        base,
        llm_provider="none" if offline else recognition_provider,
        openai_api_key=openai_key or base.openai_api_key,
        openai_model=openai_model,
        gemini_api_key=gemini_key or base.gemini_api_key,
        gemini_model=gemini_model,
        gemini_search_queries=search_queries,
        product_recognition_threshold=recognition_threshold,
        search_provider="none" if offline else search_provider,
        tavily_api_key=tavily_key or base.tavily_api_key,
    )
    pipeline = MattressIntelligencePipeline(settings)
    request = CompanyResearchRequest(
        company_name="Sleepwell Demo" if offline else company,
        official_domain="https://example.invalid/sleepwell" if offline else domain,
        market=market,
        brand_aliases=[item.strip() for item in aliases_text.split(",") if item.strip()],
        seed_urls=[item.strip() for item in seed_urls_text.splitlines() if item.strip()],
        custom_search_queries=[
            item.strip() for item in custom_queries_text.splitlines() if item.strip()
        ],
        max_pages=int(max_pages),
        max_external_pages=int(max_external_pages),
        max_crawl_depth=max_crawl_depth,
        max_configurations_per_product=max_configurations,
        use_search_grounding=use_search,
        include_external_evidence=external,
    )
    if offline:
        return pipeline.import_catalogue(
            request,
            Path("examples/demo_catalogue.json"),
            Path("outputs/demo_mattress_intelligence.xlsx"),
        )
    if mode == "Live collection only":
        return pipeline.collect(request)
    return pipeline.research(request)


if run_clicked:
    try:
        with st.status("Running evidence pipeline…", expanded=True) as status:
            st.write("Searching and classifying candidate evidence URLs")
            st.write("Parsing robots.txt and official sitemaps")
            st.write("Priority-crawling bounded official pages")
            st.write("Preserving raw HTML, PDF, XML, and text artifacts")
            st.write("Recognizing exact product documents and extracting explicit facts")
            result = run_pipeline()
            if mode in {"Live collection + algorithms", "Offline demo"}:
                st.write("Running deterministic similarity, constraints, Bayesian ranking, and graph assembly")
            st.write("Writing Excel and SQLite outputs")
            status.update(label="Research complete", state="complete")
        st.session_state["result_json"] = result.model_dump_json()
    except Exception as exc:
        st.exception(exc)

if "result_json" not in st.session_state:
    st.info(
        "Use OpenAI for search/product recognition, or choose None for a fully deterministic run. "
        "The API key stays in your local .env or password field."
    )
    st.stop()

result = ResearchResult.model_validate_json(st.session_state["result_json"])
metrics = st.columns(6)
metrics[0].metric("Products", len(result.products))
metrics[1].metric("Variants", sum(len(product.variants) for product in result.products))
metrics[2].metric("Sources", len(result.sources))
metrics[3].metric("Observations", len(result.observations))
metrics[4].metric("Configurations", len(result.configurations))
metrics[5].metric("Coverage", f"{result.coverage.estimated_coverage_percent:.1f}%")

tabs = st.tabs(
    [
        "Products",
        "Layers",
        "Observations",
        "Discovery",
        "Recognition",
        "Crawl log",
        "Evidence",
        "Configurations",
        "Warnings",
    ]
)
with tabs[0]:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "product_id": product.product_id,
                    "brand": product.brand,
                    "product": product.name,
                    "family": product.family,
                    "thickness_mm": product.total_thickness_mm,
                    "weight_kg": product.product_weight_kg,
                    "price": product.price,
                    "firmness": product.firmness,
                    "layers": len(product.layers),
                    "variants": len(product.variants),
                    "method": product.extraction_method,
                    "confidence": product.extraction_confidence,
                }
                for product in result.products
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
with tabs[1]:
    st.dataframe(
        pd.DataFrame(
            [
                {"product": product.name, **layer.model_dump(exclude={"evidence"})}
                for product in result.products
                for layer in product.layers
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
with tabs[2]:
    st.caption(
        "Atomic matches from all captured documents. They remain evidence rows until product and "
        "context admission checks succeed."
    )
    st.dataframe(
        pd.DataFrame([item.model_dump() for item in result.observations]),
        use_container_width=True,
        hide_index=True,
    )
with tabs[3]:
    st.dataframe(pd.DataFrame(result.discovery_log), use_container_width=True, hide_index=True)
with tabs[4]:
    st.caption("Every page-recognition decision, including rejected collection/location pages.")
    st.dataframe(pd.DataFrame(result.recognition_log), use_container_width=True, hide_index=True)
with tabs[5]:
    st.dataframe(pd.DataFrame(result.crawl_log), use_container_width=True, hide_index=True)
with tabs[6]:
    st.dataframe(
        pd.DataFrame([source.model_dump() for source in result.sources]),
        use_container_width=True,
        hide_index=True,
    )
with tabs[7]:
    if not result.products:
        st.info("No products were admitted. Review Recognition, Observations, Discovery, and Crawl log.")
    else:
        selected_product = st.selectbox(
            "Product",
            options=[str(product.product_id) for product in result.products],
            format_func=lambda product_id: next(
                product.name for product in result.products if product.product_id == product_id
            ),
        )
        candidates = [
            item for item in result.configurations if item.product_id == selected_product
        ]
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "rank": item.rank,
                        "configuration_id": item.configuration_id,
                        "posterior": item.posterior_probability,
                        "confidence": item.confidence_score,
                        "estimated_weight_kg": item.estimated_weight_kg,
                        "stack": " / ".join(
                            f"{layer.material} {layer.thickness_mm}mm {layer.density_kg_m3}kg/m³"
                            for layer in item.layers
                        ),
                        "why": " | ".join(item.reasons),
                        "contradictions": " | ".join(item.contradictions),
                    }
                    for item in candidates
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
with tabs[8]:
    for warning in result.warnings or ["No pipeline warnings."]:
        st.write(f"- {warning}")
    for limitation in result.coverage.limitations:
        st.write(f"- {limitation}")

if result.excel_path and Path(result.excel_path).exists():
    st.download_button(
        "Download complete Excel workbook",
        data=Path(result.excel_path).read_bytes(),
        file_name=Path(result.excel_path).name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )
