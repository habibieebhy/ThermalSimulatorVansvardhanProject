"""One-click Streamlit control plane for background mattress product research."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from .models import ProductRecord, ResearchResult, SourceRecord
from .settings import Settings
from .storage import build_repository
from .ui_helpers import build_research_request, worker_settings_overrides


_STAGE_LABELS = {
    "queued": "Queued for the worker",
    "initializing": "Initializing services and storage",
    "discovering": "Discovering official, catalogue, and external evidence URLs",
    "crawling": "Capturing official pages, catalogues, and product pages",
    "extracting": "Extracting products, prices, variants, and explicit specifications",
    "assets": "Downloading images and catalogue pages; running OCR and vision",
    "resolving": "Resolving duplicate product records and evidence",
    "analyzing": "Generating evidence-ranked construction configurations",
    "exporting": "Saving the complete run and Excel workbook",
    "completed": "Research complete",
}

_STAGE_PROGRESS = {
    "queued": 3,
    "initializing": 7,
    "discovering": 15,
    "crawling": 35,
    "extracting": 58,
    "assets": 72,
    "resolving": 82,
    "analyzing": 90,
    "exporting": 97,
    "completed": 100,
}


def _safe_dataframe(rows: list[dict[str, Any]], *, empty_message: str) -> None:
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption(empty_message)


def _product_summary(product: ProductRecord) -> dict[str, object]:
    return {
        "Product": product.name,
        "Family": product.family,
        "Firmness": product.firmness,
        "Thickness (mm)": product.total_thickness_mm,
        "Price": product.price,
        "Currency": product.currency,
        "Layers": len(product.layers),
        "Variants": len(product.variants),
        "Confidence": round(product.extraction_confidence, 3),
        "URL": product.canonical_url,
    }


def _source_label(source_by_id: dict[str, SourceRecord], source_id: str) -> str:
    """Return a stable human-readable label for an observation source."""
    source = source_by_id.get(source_id)
    if source is None:
        return source_id
    return source.title or source.url


def _render_product(product: ProductRecord, *, expanded: bool) -> None:
    label_bits = [product.name]
    if product.price is not None:
        label_bits.append(f"{product.currency or ''} {product.price:,.0f}".strip())
    with st.expander(" · ".join(label_bits), expanded=expanded):
        facts = st.columns(4)
        facts[0].metric("Firmness", product.firmness or "Not stated")
        facts[1].metric(
            "Thickness",
            f"{product.total_thickness_mm:g} mm" if product.total_thickness_mm else "Not stated",
        )
        facts[2].metric("Observed layers", len(product.layers))
        facts[3].metric("Variants", len(product.variants))

        if product.description:
            st.write(product.description)
        if product.canonical_url:
            st.link_button("Open official product page", product.canonical_url)

        st.markdown("**Observed construction**")
        _safe_dataframe(
            [
                {
                    "Position": layer.position,
                    "Layer / marketing name": layer.marketing_name,
                    "Normalized material": layer.normalized_material,
                    "Thickness (mm)": layer.thickness_mm,
                    "Thickness status": layer.thickness_status,
                    "Density (kg/m³)": layer.density_kg_m3,
                    "Density status": layer.density_status,
                    "Evidence refs": len(layer.evidence),
                }
                for layer in product.layers
            ],
            empty_message="No explicit layer stack was admitted from the collected evidence.",
        )

        st.markdown("**Variants and prices**")
        _safe_dataframe(
            [variant.model_dump() for variant in product.variants],
            empty_message="No structured variants were found on the captured pages.",
        )


def _render_result(result: ResearchResult) -> None:
    st.success(
        f"Completed: {len(result.products)} products, "
        f"{len(result.assets)} evidence assets, and {len(result.configurations)} ranked configurations."
    )

    metrics = st.columns(6)
    values = (
        ("Products", len(result.products)),
        ("Variants", sum(len(product.variants) for product in result.products)),
        ("Sources", len(result.sources)),
        ("Images / assets", len(result.assets)),
        ("Configurations", len(result.configurations)),
        ("Coverage", f"{result.coverage.estimated_coverage_percent:.1f}%"),
    )
    for column, (label, value) in zip(metrics, values):
        column.metric(label, value)

    product_tab, config_tab, evidence_tab, image_tab, technical_tab = st.tabs(
        ["Products", "Construction configurations", "Evidence", "Images", "Technical details"]
    )

    with product_tab:
        if not result.products:
            st.warning(
                "No product records passed the evidence admission rules. Open Technical details to "
                "inspect captured URLs and recognition decisions."
            )
        else:
            _safe_dataframe(
                [_product_summary(product) for product in result.products],
                empty_message="No products found.",
            )
            for index, product in enumerate(result.products):
                _render_product(product, expanded=index == 0)

    with config_tab:
        if not result.configurations:
            st.info(
                "No ranked configurations were generated. This usually means the run found no admitted "
                "products or insufficient layer/thickness evidence."
            )
        else:
            product_names = {str(product.product_id): product.name for product in result.products}
            _safe_dataframe(
                [
                    {
                        "Product": product_names.get(item.product_id, item.product_id),
                        "Rank": item.rank,
                        "Posterior probability": item.posterior_probability,
                        "Confidence score": item.confidence_score,
                        "Evidence score": item.evidence_score,
                        "Total thickness (mm)": item.total_thickness_mm,
                        "Estimated weight (kg)": item.estimated_weight_kg,
                        "Layer stack": " → ".join(
                            f"{layer.marketing_name} ({layer.thickness_mm} mm, {layer.density_kg_m3} kg/m³)"
                            for layer in item.layers
                        ),
                        "Reasons": " | ".join(item.reasons),
                        "Contradictions": " | ".join(item.contradictions),
                    }
                    for item in result.configurations
                ],
                empty_message="No configurations generated.",
            )

    with evidence_tab:
        source_by_id = {source.source_id: source for source in result.sources}
        _safe_dataframe(
            [
                {
                    "Title": source.title,
                    "Type": source.kind,
                    "Official": source.is_official,
                    "Reliability": source.reliability,
                    "Capture": source.capture_method,
                    "URL": source.url,
                }
                for source in result.sources
            ],
            empty_message="No evidence sources were persisted.",
        )
        with st.expander("Atomic observations"):
            _safe_dataframe(
                [
                    {
                        "Source": _source_label(source_by_id, item.source_id),
                        "Product hint": item.product_name_hint,
                        "Field": item.field_path,
                        "Value": item.value,
                        "Material": item.normalized_material,
                        "Method": item.method,
                        "Confidence": item.confidence,
                        "URL": item.document_url,
                    }
                    for item in result.observations
                ],
                empty_message="No atomic observations were stored.",
            )

    with image_tab:
        visible_assets = sorted(result.assets, key=lambda item: item.relevance_score, reverse=True)
        if not visible_assets:
            st.info("No image or catalogue-page assets were retained.")
        for asset in visible_assets[:24]:
            with st.container(border=True):
                columns = st.columns([1, 2])
                path = Path(asset.local_path) if asset.local_path else None
                if path and path.exists() and asset.content_type.startswith("image/"):
                    columns[0].image(str(path), use_container_width=True)
                else:
                    columns[0].write(asset.kind)
                columns[1].write(
                    {
                        "kind": asset.kind,
                        "page": asset.page_url,
                        "asset_url": asset.asset_url,
                        "relevance": asset.relevance_score,
                        "ocr_engine": asset.ocr_engine,
                        "ocr_text": asset.ocr_text,
                        "vision_provider": asset.vision_provider,
                        "vision_confidence": asset.vision_confidence,
                    }
                )

    with technical_tab:
        if result.warnings:
            st.markdown("**Warnings**")
            for warning in result.warnings:
                st.write(f"- {warning}")
        _safe_dataframe(result.discovery_log, empty_message="No discovery log entries.")
        with st.expander("Crawl log"):
            _safe_dataframe(result.crawl_log, empty_message="No crawl log entries.")
        with st.expander("Recognition log"):
            _safe_dataframe(result.recognition_log, empty_message="No recognition log entries.")
        with st.expander("Asset acquisition log"):
            _safe_dataframe(result.acquisition_log, empty_message="No acquisition log entries.")

    if result.excel_path and Path(result.excel_path).exists():
        st.download_button(
            "Download complete Excel workbook",
            data=Path(result.excel_path).read_bytes(),
            file_name=Path(result.excel_path).name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )


def _worker_is_available() -> bool:
    from .celery_app import celery_app

    return bool(celery_app.control.ping(timeout=2.0))


def _submit_job(company: str, website: str, settings: Settings) -> None:
    from .tasks import enqueue_research

    if not settings.celery_enabled:
        raise RuntimeError("Celery is disabled. Set CELERY_ENABLED=true in .env.")
    if settings.celery_always_eager:
        # The UI explicitly requests a background job. Override contradictory local eager mode in
        # this producer process so the task is actually sent to Redis/Celery.
        from .celery_app import celery_app

        celery_app.conf.task_always_eager = False
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing from .env.")
    if not settings.firecrawl_api_key and settings.search_provider == "firecrawl":
        raise RuntimeError("FIRECRAWL_API_KEY is missing from .env.")
    if not _worker_is_available():
        raise RuntimeError(
            "No Celery worker responded. Start the worker, wait for 'ready', and click Start again."
        )

    request = build_research_request(company, website, settings)
    task = enqueue_research(
        request,
        settings_overrides=worker_settings_overrides(settings),
    )
    st.session_state["active_task_id"] = task.id
    st.session_state["active_company"] = request.company_name
    st.session_state.pop("result_json", None)
    st.session_state.pop("job_error", None)


def _poll_active_job(settings: Settings) -> None:
    from celery.result import AsyncResult

    from .celery_app import celery_app

    task_id = st.session_state.get("active_task_id")
    if not task_id:
        return

    task = AsyncResult(task_id, app=celery_app)
    state = task.state
    info = task.info if isinstance(task.info, dict) else {}
    stage = str(info.get("stage") or ("queued" if state == "PENDING" else state.casefold()))
    label = _STAGE_LABELS.get(stage, f"Worker state: {state}")
    progress = _STAGE_PROGRESS.get(stage, 5 if state == "PENDING" else 50)

    with st.status(label, expanded=True, state="running"):
        st.progress(progress, text=f"{progress}% · {label}")
        if info.get("current") is not None and info.get("total") is not None:
            st.write(f"Processed {info['current']} of {info['total']} items")
        st.caption(f"Background task: {task_id}")

    if task.successful():
        summary = task.result if isinstance(task.result, dict) else {}
        run_id = summary.get("run_id")
        if not run_id:
            st.session_state["job_error"] = "Worker completed without returning a persisted run ID."
        else:
            result = build_repository(settings).load(str(run_id))
            st.session_state["result_json"] = result.model_dump_json()
        st.session_state.pop("active_task_id", None)
        st.rerun()

    if task.failed():
        st.session_state["job_error"] = str(task.result)
        st.session_state.pop("active_task_id", None)
        st.rerun()

    time.sleep(2.0)
    st.rerun()


def render_app(*, configure_page: bool = True) -> None:
    if configure_page:
        st.set_page_config(
            page_title="BRIXTA Mattress Product Intelligence",
            page_icon="🔎",
            layout="wide",
            initial_sidebar_state="collapsed",
        )

    settings = Settings()
    st.title("BRIXTA Mattress Product Intelligence")
    st.caption(
        "Enter a company and its official website. The background worker discovers products, "
        "captures pages and images, extracts specifications, and ranks likely mattress constructions."
    )

    with st.form("research_job", clear_on_submit=False):
        company_column, website_column = st.columns([1, 2])
        company = company_column.text_input("Company", value="The Sleep Company")
        website = website_column.text_input(
            "Official website", value="https://thesleepcompany.in"
        )
        submitted = st.form_submit_button(
            "Start product research",
            type="primary",
            use_container_width=True,
            disabled=bool(st.session_state.get("active_task_id")),
        )

    st.caption(
        f"Automatic run: up to {settings.ui_max_pages} official pages, "
        f"{settings.ui_max_external_pages} external evidence pages, images/OCR/vision, "
        "product resolution, configuration ranking, and Excel export."
    )

    if submitted:
        try:
            _submit_job(company, website, settings)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    if st.session_state.get("job_error"):
        st.error(st.session_state["job_error"])

    if st.session_state.get("active_task_id"):
        _poll_active_job(settings)
        return

    if "result_json" in st.session_state:
        _render_result(ResearchResult.model_validate_json(st.session_state["result_json"]))
    else:
        st.info(
            "Ready. Click Start product research; the job will run in Celery and results "
            "will appear here."
        )