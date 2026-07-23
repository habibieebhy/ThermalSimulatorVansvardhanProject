"""Multi-session Streamlit control plane for background mattress product research."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import streamlit as st

from .exporter import (
    primary_table_frame,
    table_csv_bytes,
    table_excel_bytes,
    table_json_bytes,
)
from .jobs import (
    ACTIVE_JOB_STATUSES,
    ResearchJob,
    ResearchJobStore,
    build_job_output_dir,
)
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
    "exporting": "Saving the complete run and downloads",
    "completed": "Research complete",
    "failed": "Research failed",
}


def _safe_dataframe(rows: list[dict[str, Any]], *, empty_message: str) -> None:
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption(empty_message)


def _source_label(source_by_id: dict[str, SourceRecord], source_id: str) -> str:
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


def _filtered_primary_table(result: ResearchResult) -> pd.DataFrame:
    frame = primary_table_frame(result)
    if frame.empty:
        return frame

    filter_columns = st.columns([2, 1, 1])
    query = filter_columns[0].text_input(
        "Search the table",
        placeholder="Product, family, firmness, material...",
        key=f"product_search_{result.run_id}",
    ).strip()
    family_options = sorted(
        {str(value) for value in frame["Family"].dropna().tolist() if str(value).strip()}
    )
    firmness_options = sorted(
        {str(value) for value in frame["Firmness"].dropna().tolist() if str(value).strip()}
    )
    selected_families = filter_columns[1].multiselect(
        "Family",
        family_options,
        key=f"product_family_{result.run_id}",
    )
    selected_firmness = filter_columns[2].multiselect(
        "Firmness",
        firmness_options,
        key=f"product_firmness_{result.run_id}",
    )

    filtered = frame.copy()
    if query:
        searchable = filtered.fillna("").astype(str).agg(" ".join, axis=1)
        filtered = filtered[searchable.str.contains(query, case=False, regex=False)]
    if selected_families:
        filtered = filtered[filtered["Family"].astype(str).isin(selected_families)]
    if selected_firmness:
        filtered = filtered[filtered["Firmness"].astype(str).isin(selected_firmness)]
    return filtered.reset_index(drop=True)


def _render_primary_table(result: ResearchResult) -> None:
    st.markdown("### Download-ready product table")
    st.caption(
        "The downloads below are generated from this exact displayed dataset, including the active filters."
    )
    frame = _filtered_primary_table(result)
    st.dataframe(frame, use_container_width=True, hide_index=True)

    safe_company = "-".join(result.request.company_name.casefold().split()) or "company"
    download_columns = st.columns(3)
    download_columns[0].download_button(
        "Download displayed table · CSV",
        data=table_csv_bytes(frame),
        file_name=f"{safe_company}_products.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )
    download_columns[1].download_button(
        "Download displayed table · Excel",
        data=table_excel_bytes(frame),
        file_name=f"{safe_company}_products.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    download_columns[2].download_button(
        "Download displayed table · JSON",
        data=table_json_bytes(frame),
        file_name=f"{safe_company}_products.json",
        mime="application/json",
        use_container_width=True,
    )


def _render_result(result: ResearchResult, job: ResearchJob | None = None) -> None:
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
            _render_primary_table(result)
            st.markdown("### Product details")
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
        if job is not None:
            st.write(
                {
                    "session_id": job.job_id,
                    "task_id": job.task_id,
                    "output_directory": job.output_dir,
                    "submitted_at": job.submitted_at.isoformat(),
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                }
            )
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

    excel_path = Path(job.excel_path) if job and job.excel_path else Path(result.excel_path or "")
    if excel_path.is_file():
        st.download_button(
            "Download complete multi-sheet research workbook",
            data=excel_path.read_bytes(),
            file_name=excel_path.name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


def _broker_is_available() -> bool:
    from .celery_app import celery_app

    try:
        with celery_app.connection_for_write() as connection:
            connection.ensure_connection(max_retries=1)
        return True
    except Exception:
        return False


def _submit_job(
    company: str,
    website: str,
    settings: Settings,
    store: ResearchJobStore,
) -> ResearchJob:
    from .tasks import enqueue_research

    if not settings.celery_enabled:
        raise RuntimeError("Celery is disabled. Set CELERY_ENABLED=true in .env.")
    if settings.celery_always_eager:
        from .celery_app import celery_app

        celery_app.conf.task_always_eager = False
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing from .env.")
    if not settings.firecrawl_api_key and settings.search_provider == "firecrawl":
        raise RuntimeError("FIRECRAWL_API_KEY is missing from .env.")
    if not _broker_is_available():
        raise RuntimeError("Redis/Celery broker is unavailable. Start Redis and retry.")

    request = build_research_request(company, website, settings)
    job_id = uuid4().hex
    output_dir = build_job_output_dir(settings, request.company_name, job_id)
    job = store.create(request, job_id=job_id, output_dir=output_dir)
    try:
        task = enqueue_research(
            request,
            output_path=output_dir / "complete_research.xlsx",
            settings_overrides=worker_settings_overrides(settings),
            job_id=job_id,
        )
        if task.id != job_id:
            job = store.update(job_id, task_id=task.id)
    except Exception as exc:
        store.mark_failed(job_id, f"The job could not be submitted to Celery: {exc}")
        raise

    st.session_state["selected_job_id"] = job.job_id
    st.session_state["job_selector"] = job.job_id
    st.session_state.pop("job_error", None)
    return job


def _reconcile_celery_state(job: ResearchJob, store: ResearchJobStore) -> ResearchJob:
    """Repair the durable ledger from Celery if a worker died between its final writes."""

    from celery.result import AsyncResult

    from .celery_app import celery_app

    task = AsyncResult(job.task_id, app=celery_app)
    try:
        state = task.state
        info = task.info if isinstance(task.info, dict) else {}
    except Exception:
        return job

    if job.status == "queued" and state in {"STARTED", "PROGRESS"}:
        stage = str(info.get("stage") or "initializing")
        job = store.mark_running(
            job.job_id,
            task_id=job.task_id,
            stage=stage,
            message=str(info.get("message") or _STAGE_LABELS.get(stage, stage)),
        )
    elif job.status in ACTIVE_JOB_STATUSES and task.successful():
        summary = task.result if isinstance(task.result, dict) else {}
        run_id = str(summary.get("run_id") or "")
        if run_id:
            job = store.mark_completed(
                job.job_id,
                run_id=run_id,
                summary=summary,
                excel_path=str(summary.get("excel_path") or "") or None,
                table_csv_path=str(summary.get("table_csv_path") or "") or None,
                table_json_path=str(summary.get("table_json_path") or "") or None,
                result_json_path=str(summary.get("result_json_path") or "") or None,
            )
    elif job.status in ACTIVE_JOB_STATUSES and task.failed():
        job = store.mark_failed(job.job_id, str(task.result))

    if job.is_terminal and job.backend_cleared_at is None:
        try:
            task.forget()
            job = store.mark_backend_cleared(job.job_id)
        except Exception:
            pass
    return job


def _load_job_result(job: ResearchJob, settings: Settings) -> ResearchResult:
    if job.run_id:
        try:
            return build_repository(settings).load(job.run_id)
        except (KeyError, OSError, ValueError):
            pass
    if job.result_json_path:
        path = Path(job.result_json_path)
        if path.is_file():
            return ResearchResult.model_validate_json(path.read_text(encoding="utf-8"))
    raise RuntimeError("The session completed, but its persisted research result could not be loaded.")


def _history_frame(jobs: list[ResearchJob]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for job in jobs:
        summary = job.summary
        rows.append(
            {
                "Session": job.job_id[:8],
                "Company": job.company_name,
                "Status": job.status.upper(),
                "Stage": _STAGE_LABELS.get(job.stage, job.stage.replace("_", " ").title()),
                "Progress": f"{job.progress}%",
                "Products": summary.get("products", ""),
                "Submitted (UTC)": job.submitted_at.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    return pd.DataFrame(rows)


def _clear_session_view() -> None:
    st.session_state["selected_job_id"] = None
    st.session_state["job_selector"] = ""


def _render_job_history(store: ResearchJobStore, settings: Settings) -> list[ResearchJob]:
    jobs = store.list(limit=settings.ui_history_limit)
    st.markdown("### Research sessions")
    st.caption(
        "Each company has an isolated task ID, output directory, progress record, and downloadable "
        "result. Submitting another company switches the view without overwriting earlier sessions."
    )
    if not jobs:
        st.info("No research sessions have been submitted yet.")
        return jobs

    st.dataframe(_history_frame(jobs), use_container_width=True, hide_index=True)
    labels = {
        job.job_id: (
            f"{job.company_name} · {job.status.upper()} · "
            f"{job.submitted_at.strftime('%Y-%m-%d %H:%M UTC')} · {job.job_id[:8]}"
        )
        for job in jobs
    }
    valid_ids = {job.job_id for job in jobs}
    if "selected_job_id" not in st.session_state:
        st.session_state["selected_job_id"] = jobs[0].job_id
    selected = st.session_state.get("selected_job_id")
    if selected is not None and selected not in valid_ids:
        selected = None
        st.session_state["selected_job_id"] = None

    valid_selector_values = {"", *valid_ids}
    if st.session_state.get("job_selector") not in valid_selector_values:
        st.session_state["job_selector"] = selected or ""

    selector_column, clear_column = st.columns([4, 1])
    selected_job_id = selector_column.selectbox(
        "Open a session",
        options=["", *[job.job_id for job in jobs]],
        format_func=lambda job_id: "No session selected" if not job_id else labels[job_id],
        key="job_selector",
    )
    normalized_selection = selected_job_id or None
    if normalized_selection != st.session_state.get("selected_job_id"):
        st.session_state["selected_job_id"] = normalized_selection
        st.rerun()
    clear_column.button(
        "Clear current view",
        use_container_width=True,
        on_click=_clear_session_view,
    )
    return jobs


def _render_selected_job(
    job: ResearchJob,
    jobs: list[ResearchJob],
    store: ResearchJobStore,
    settings: Settings,
) -> None:
    previous_state = (job.status, job.stage, job.progress, job.updated_at)
    job = _reconcile_celery_state(job, store)
    current_state = (job.status, job.stage, job.progress, job.updated_at)
    if current_state != previous_state:
        st.rerun()
    st.divider()
    st.subheader(job.company_name)
    st.caption(f"{job.official_domain} · session {job.job_id} · task {job.task_id}")

    if job.status in ACTIVE_JOB_STATUSES:
        label = _STAGE_LABELS.get(job.stage, job.message or job.stage)
        with st.status(label, expanded=True, state="running"):
            st.progress(job.progress, text=f"{job.progress}% · {label}")
            if job.message and job.message != label:
                st.write(job.message)
            running_others = [
                item for item in jobs if item.job_id != job.job_id and item.status == "running"
            ]
            if job.status == "queued" and running_others:
                st.info(
                    f"Queued behind {len(running_others)} running session(s). A solo worker processes "
                    "one company at a time; this session will start automatically."
                )
            elif job.status == "queued":
                age_seconds = (datetime.now(timezone.utc) - job.submitted_at).total_seconds()
                if age_seconds >= settings.ui_queue_warning_seconds:
                    st.warning(
                        "The broker accepted this task, but no worker has recorded a start yet. Check the "
                        "Celery terminal for a 'ready' worker. This is a queued state, not an old result."
                    )
            st.caption(f"Output directory: {job.output_dir}")
        time.sleep(2.0)
        st.rerun()

    if job.status == "failed":
        st.error(job.error or "The research session failed without a recorded error message.")
        st.caption(f"This failure is isolated to session {job.job_id}; previous company results are unchanged.")
        return

    if job.status == "completed":
        try:
            _render_result(_load_job_result(job, settings), job)
        except Exception as exc:
            st.error(str(exc))
        return

    st.info(f"Session status: {job.status}")


def render_app(*, configure_page: bool = True) -> None:
    if configure_page:
        st.set_page_config(
            page_title="BRIXTA Mattress Product Intelligence",
            page_icon="🔎",
            layout="wide",
            initial_sidebar_state="collapsed",
        )

    settings = Settings()
    settings.ensure_directories()
    store = ResearchJobStore.from_settings(settings)

    st.title("BRIXTA Mattress Product Intelligence")
    st.caption(
        "Run one company after another. Every submission is an isolated Celery session with its own "
        "status, output directory, result table, and complete Excel workbook."
    )

    with st.form("research_job", clear_on_submit=True):
        company_column, website_column = st.columns([1, 2])
        company = company_column.text_input("Company", placeholder="Sleepwell")
        website = website_column.text_input(
            "Official website",
            placeholder="https://www.sleepwellproducts.com",
        )
        submitted = st.form_submit_button(
            "Start product research",
            type="primary",
            use_container_width=True,
        )

    st.caption(
        f"Automatic run: up to {settings.ui_max_pages} official pages, "
        f"{settings.ui_max_external_pages} external evidence pages, images/OCR/vision, "
        "product resolution, configuration ranking, and isolated exports. You may queue another company "
        "while one is running."
    )

    if submitted:
        try:
            _submit_job(company, website, settings, store)
            st.rerun()
        except Exception as exc:
            st.session_state["job_error"] = str(exc)

    if st.session_state.get("job_error"):
        st.error(st.session_state["job_error"])

    jobs = _render_job_history(store, settings)
    selected_job_id = st.session_state.get("selected_job_id")
    selected_job = next((job for job in jobs if job.job_id == selected_job_id), None)
    if selected_job is not None:
        _render_selected_job(selected_job, jobs, store, settings)
    elif jobs:
        st.info("Select a research session above to inspect its status or results.")
    else:
        st.info("Ready. Submit a company to create the first isolated research session.")

    selected_is_active = selected_job is not None and selected_job.status in ACTIVE_JOB_STATUSES
    if not selected_is_active and any(job.status in ACTIVE_JOB_STATUSES for job in jobs):
        time.sleep(2.0)
        st.rerun()
