"""Celery tasks. Full evidence is persisted; Redis receives compact result summaries."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from celery import Task

from .celery_app import celery_app
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


def _progress_updater(task: Any):
    def update(stage: str, **metadata: object) -> None:
        task.update_state(state="PROGRESS", meta={"stage": stage, **metadata})

    return update


@celery_app.task(name="mattress_intelligence.collect", bind=True)
def run_collection_task(
    self,
    request_payload: dict[str, Any],
    output_path: str | None = None,
    settings_overrides: dict[str, Any] | None = None,
) -> dict[str, object]:
    self.update_state(state="PROGRESS", meta={"stage": "initializing"})
    pipeline = MattressIntelligencePipeline(_settings_with_overrides(settings_overrides))
    request = CompanyResearchRequest.model_validate(request_payload)
    result = pipeline.collect(
        request,
        Path(output_path) if output_path else None,
        progress_callback=_progress_updater(self),
    )
    return _summary(result)


@celery_app.task(name="mattress_intelligence.research", bind=True)
def run_research_task(
    self,
    request_payload: dict[str, Any],
    output_path: str | None = None,
    settings_overrides: dict[str, Any] | None = None,
) -> dict[str, object]:
    self.update_state(state="PROGRESS", meta={"stage": "initializing"})
    pipeline = MattressIntelligencePipeline(_settings_with_overrides(settings_overrides))
    request = CompanyResearchRequest.model_validate(request_payload)
    result = pipeline.research(
        request,
        Path(output_path) if output_path else None,
        progress_callback=_progress_updater(self),
    )
    return _summary(result)


# Tell the type checker these decorated callables are Celery Tasks.
collection_task = cast(Task, run_collection_task)
research_task = cast(Task, run_research_task)


def enqueue_collection(
    request: CompanyResearchRequest,
    output_path: Path | None = None,
    *,
    settings_overrides: dict[str, Any] | None = None,
) -> Any:
    return collection_task.delay(
        request.model_dump(mode="json"),
        str(output_path) if output_path else None,
        settings_overrides,
    )


def enqueue_research(
    request: CompanyResearchRequest,
    output_path: Path | None = None,
    *,
    settings_overrides: dict[str, Any] | None = None,
) -> Any:
    return research_task.delay(
        request.model_dump(mode="json"),
        str(output_path) if output_path else None,
        settings_overrides,
    )
