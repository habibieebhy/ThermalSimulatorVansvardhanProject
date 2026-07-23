"""Celery tasks with durable job state and compact Redis result payloads."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from celery import Task

from .celery_app import celery_app
from .exporter import export_primary_artifacts
from .jobs import ResearchJobStore
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
    "assets": "Downloading images and catalogue pages; running OCR and vision",
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
        "observations": len(result.observations),
        "configurations": len(result.configurations),
        "coverage_percent": result.coverage.estimated_coverage_percent,
        "excel_path": result.excel_path,
        "warnings": len(result.warnings),
    }


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
    if "soft time limit" in lowered or "timelimit" in lowered:
        return f"The worker time limit was reached. Provider detail: {detail}"
    if "connection refused" in lowered and "redis" in lowered:
        return f"Redis is unavailable. Start Redis and the Celery worker, then retry. Detail: {detail}"
    return detail


def _job_store(settings: Settings, job_id: str | None) -> ResearchJobStore | None:
    return ResearchJobStore.from_settings(settings) if job_id else None


def _safe_job_update(callback: Any, *args: object, **kwargs: object) -> None:
    try:
        callback(*args, **kwargs)
    except Exception:
        # Celery task execution must not be lost merely because the auxiliary job ledger is unavailable.
        return


def _progress_updater(task: Any, store: ResearchJobStore | None, job_id: str | None):
    def update(stage: str, **metadata: object) -> None:
        default_message = _STAGE_MESSAGES.get(stage, stage.replace("_", " ").title())
        message = str(metadata.get("message") or default_message)
        current = metadata.get("current")
        total = metadata.get("total")
        if current is not None and total is not None:
            message = f"{message} ({current}/{total})"
        task.update_state(state="PROGRESS", meta={"stage": stage, "message": message, **metadata})
        if store is not None and job_id is not None:
            _safe_job_update(store.mark_progress, job_id, stage, message=message)

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

    if store is not None and job_id is not None:
        _safe_job_update(
            store.mark_running,
            job_id,
            task_id=task_id,
            stage="initializing",
            message=_STAGE_MESSAGES["initializing"],
        )

    task.update_state(
        state="PROGRESS",
        meta={"stage": "initializing", "message": _STAGE_MESSAGES["initializing"]},
    )

    try:
        pipeline = MattressIntelligencePipeline(settings)
        request = CompanyResearchRequest.model_validate(request_payload)
        callback = _progress_updater(task, store, job_id)
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
            _safe_job_update(
                store.mark_completed,
                job_id,
                run_id=result.run_id,
                summary=summary,
                excel_path=result.excel_path,
                table_csv_path=artifacts["table_csv_path"],
                table_json_path=artifacts["table_json_path"],
                result_json_path=artifacts["result_json_path"],
            )
        return summary
    except Exception as exc:
        error = _friendly_error(exc)
        if store is not None and job_id is not None:
            _safe_job_update(store.mark_failed, job_id, error)
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
