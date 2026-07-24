"""Celery tasks with durable state, execution leases, and compact result payloads."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from celery import Task
from celery.exceptions import Ignore

from .celery_app import celery_app
from .exporter import export_primary_artifacts
from .jobs import ResearchJob, ResearchJobStore
from .models import CompanyResearchRequest
from .pipeline import MattressIntelligencePipeline
from .settings import Settings


_ALLOWED_SETTINGS_OVERRIDES = {
    "search_provider",
    "jina_reader_enabled",
    "jina_reader_on_thin_page",
    "search_queries",
    "capture_strategy",
}

_STAGE_MESSAGES = {
    "initializing": "Initializing services and storage",
    "discovering": "Discovering official, catalogue, and external evidence URLs",
    "crawling": "Capturing official pages, catalogues, and product pages",
    "extracting": "Extracting products, prices, variants, and explicit specifications",
    "assets": "Downloading images and catalogue pages; running OCR and forensic vision",
    "visual_followup": "Searching and capturing diagram-matched brochures, archives, patents, and teardowns",
    "material_decoding": "Reading proprietary material names and preparing evidence searches",
    "material_evidence": "Crawling technical documents for material identity and density evidence",
    "material_adjudication": "Decoding trademarked materials and grading density evidence",
    "resolving": "Resolving duplicate product records and evidence",
    "analyzing": "Generating evidence-ranked construction configurations",
    "exporting": "Saving the run and downloadable artifacts",
    "completed": "Research complete",
}


def _summary(result: Any) -> dict[str, object]:
    return {
        "run_id": result.run_id,
        "company": result.request.company_name,
        "products": len(result.products),
        "variants": sum(len(product.variants) for product in result.products),
        "sources": len(result.sources),
        "assets": len(result.assets),
        "trademark_materials": len(result.trademark_materials),
        "materials_with_density_evidence": sum(
            1 for item in result.trademark_materials if str(item.density_status) != "unknown"
        ),
        "observations": len(result.observations),
        "configurations": len(result.configurations),
        "coverage_percent": result.coverage.estimated_coverage_percent,
        "excel_path": result.excel_path,
        "warnings": len(result.warnings),
    }


def _completed_job_payload(job: ResearchJob) -> dict[str, object]:
    payload: dict[str, object] = dict(job.summary)
    payload.setdefault("run_id", job.run_id or "")
    payload.setdefault("company", job.company_name)
    payload.setdefault("excel_path", job.excel_path)
    payload.setdefault("table_csv_path", job.table_csv_path)
    payload.setdefault("table_json_path", job.table_json_path)
    payload.setdefault("result_json_path", job.result_json_path)
    payload["job_id"] = job.job_id
    payload["duplicate_delivery_ignored"] = True
    return payload


def _settings_with_overrides(overrides: dict[str, Any] | None) -> Settings:
    settings = Settings()
    if not overrides:
        return settings
    unknown = set(overrides) - _ALLOWED_SETTINGS_OVERRIDES
    if unknown:
        raise ValueError(f"Unsupported task settings override(s): {', '.join(sorted(unknown))}")
    return replace(settings, **overrides)


def _friendly_error(exc: BaseException) -> str:
    detail = str(exc).strip() or exc.__class__.__name__
    lowered = detail.casefold()
    quota_markers = (
        "quota",
        "credit",
        "insufficient balance",
        "payment required",
        "http 402",
        "plan limit",
    )
    if "firecrawl" in lowered and any(marker in lowered for marker in quota_markers):
        return (
            "Firecrawl quota or credits appear to be exhausted. The session was stopped without "
            "overwriting previous company results. Recharge Firecrawl and retry this session. "
            f"Provider detail: {detail}"
        )
    if (
        "incompleteread" in lowered
        or ("bytes read" in lowered and "more expected" in lowered)
        or "remote end closed connection" in lowered
    ):
        return (
            "An upstream HTTP service closed the response before sending the complete body. "
            "The session was stopped cleanly and previous company results remain unchanged. "
            "Retry the session; repeated failures should be investigated using the Celery traceback. "
            f"Transport detail: {detail}"
        )
    if "soft time limit" in lowered or "timelimit" in lowered:
        return f"The worker time limit was reached. Provider detail: {detail}"
    if "connection refused" in lowered and "redis" in lowered:
        return f"Redis is unavailable. Start Redis and the Celery worker, then retry. Detail: {detail}"
    return detail


def _job_store(settings: Settings, job_id: str | None) -> ResearchJobStore | None:
    return ResearchJobStore.from_settings(settings) if job_id else None


def _progress_updater(
    task: Any,
    store: ResearchJobStore | None,
    job_id: str | None,
    execution_token: str | None,
):
    def update(stage: str, **metadata: object) -> None:
        default_message = _STAGE_MESSAGES.get(stage, stage.replace("_", " ").title())
        message = str(metadata.get("message") or default_message)
        current = metadata.get("current")
        total = metadata.get("total")
        if current is not None and total is not None:
            message = f"{message} ({current}/{total})"
        task.update_state(state="PROGRESS", meta={"stage": stage, "message": message, **metadata})
        if store is not None and job_id is not None:
            store.mark_progress(
                job_id,
                stage,
                message=message,
                execution_token=execution_token,
            )

    return update


def _run_pipeline_task(
    task: Any,
    *,
    mode: str,
    request_payload: dict[str, Any],
    output_path: str | None,
    settings_overrides: dict[str, Any] | None,
    job_id: str | None,
) -> dict[str, object]:
    settings = _settings_with_overrides(settings_overrides)
    store = _job_store(settings, job_id)
    task_id = str(task.request.id or job_id or "")
    execution_token: str | None = None

    if store is not None and job_id is not None:
        # Repair any ledger damaged by a pre-v1.6.1 duplicate before deciding whether to run.
        existing = store.recover_completed_from_artifacts(job_id)
        if existing.status == "completed":
            return _completed_job_payload(existing)

        execution_token = uuid4().hex
        claimed, disposition = store.claim_for_execution(
            job_id,
            task_id=task_id,
            execution_token=execution_token,
            stale_after_seconds=settings.celery_job_lease_stale_seconds,
            message=_STAGE_MESSAGES["initializing"],
        )
        if disposition == "completed":
            return _completed_job_payload(claimed)
        if disposition in {"already_running", "terminal"}:
            # Acknowledge this delivery without changing the shared task backend state.
            raise Ignore()

    task.update_state(
        state="PROGRESS",
        meta={"stage": "initializing", "message": _STAGE_MESSAGES["initializing"]},
    )

    try:
        pipeline = MattressIntelligencePipeline(settings)
        request = CompanyResearchRequest.model_validate(request_payload)
        callback = _progress_updater(task, store, job_id, execution_token)
        destination = Path(output_path) if output_path else None
        if mode == "collect":
            result = pipeline.collect(request, destination, progress_callback=callback)
        else:
            result = pipeline.research(request, destination, progress_callback=callback)

        summary = _summary(result)
        artifact_dir = destination.parent if destination is not None else settings.output_dir / result.run_id
        artifacts = export_primary_artifacts(result, artifact_dir)
        summary.update(artifacts)
        summary["job_id"] = job_id or task_id

        if store is not None and job_id is not None:
            completed = store.mark_completed(
                job_id,
                run_id=result.run_id,
                summary=summary,
                excel_path=result.excel_path,
                table_csv_path=artifacts["table_csv_path"],
                table_json_path=artifacts["table_json_path"],
                result_json_path=artifacts["result_json_path"],
                execution_token=execution_token,
            )
            if completed.status == "completed":
                return _completed_job_payload(completed) | {
                    "duplicate_delivery_ignored": False,
                }
        return summary
    except Ignore:
        raise
    except Exception as exc:
        error = _friendly_error(exc)
        if store is not None and job_id is not None:
            store.mark_failed(job_id, error, execution_token=execution_token)
        raise RuntimeError(error) from exc


@celery_app.task(name="mattress_intelligence.collect", bind=True)
def run_collection_task(
    self,
    request_payload: dict[str, Any],
    output_path: str | None = None,
    settings_overrides: dict[str, Any] | None = None,
    job_id: str | None = None,
) -> dict[str, object]:
    return _run_pipeline_task(
        self,
        mode="collect",
        request_payload=request_payload,
        output_path=output_path,
        settings_overrides=settings_overrides,
        job_id=job_id,
    )


@celery_app.task(name="mattress_intelligence.research", bind=True)
def run_research_task(
    self,
    request_payload: dict[str, Any],
    output_path: str | None = None,
    settings_overrides: dict[str, Any] | None = None,
    job_id: str | None = None,
) -> dict[str, object]:
    return _run_pipeline_task(
        self,
        mode="research",
        request_payload=request_payload,
        output_path=output_path,
        settings_overrides=settings_overrides,
        job_id=job_id,
    )


# Tell the type checker these decorated callables are Celery Tasks.
collection_task = cast(Task, run_collection_task)
research_task = cast(Task, run_research_task)


def enqueue_collection(
    request: CompanyResearchRequest,
    output_path: Path | None = None,
    *,
    settings_overrides: dict[str, Any] | None = None,
    job_id: str | None = None,
) -> Any:
    task_id = job_id or uuid4().hex
    return collection_task.apply_async(
        args=[
            request.model_dump(mode="json"),
            str(output_path) if output_path else None,
            settings_overrides,
            job_id,
        ],
        task_id=task_id,
    )


def enqueue_research(
    request: CompanyResearchRequest,
    output_path: Path | None = None,
    *,
    settings_overrides: dict[str, Any] | None = None,
    job_id: str | None = None,
) -> Any:
    task_id = job_id or uuid4().hex
    return research_task.apply_async(
        args=[
            request.model_dump(mode="json"),
            str(output_path) if output_path else None,
            settings_overrides,
            job_id,
        ],
        task_id=task_id,
    )
