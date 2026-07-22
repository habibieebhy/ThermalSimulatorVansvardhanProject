"""FastAPI service with synchronous and Celery-backed job endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .models import CompanyResearchRequest, ResearchResult
from .pipeline import MattressIntelligencePipeline
from .settings import Settings


def create_app() -> FastAPI:
    settings = Settings()
    app = FastAPI(title="BRIXTA Mattress Intelligence API", version="1.3.0")
    pipeline = MattressIntelligencePipeline(settings)

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "recognition_provider": pipeline.llm.name,
            "search_provider": pipeline.search_provider.name,
            "capture_strategy": settings.capture_strategy,
            "database": "postgres/neon" if settings.postgres_enabled else "sqlite",
            "object_store": "minio" if settings.object_storage_enabled else "local",
            "celery_enabled": settings.celery_enabled,
            "deterministic_analysis": True,
            "llm_downstream_analysis": False,
        }

    @app.get("/v1/runs")
    def list_runs() -> list[dict]:
        return pipeline.repository.list_runs()

    @app.get("/v1/runs/{run_id}", response_model=ResearchResult)
    def get_run(run_id: str) -> ResearchResult:
        try:
            return pipeline.repository.load(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/runs/{run_id}/excel")
    def get_excel(run_id: str) -> FileResponse:
        try:
            result = pipeline.repository.load(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        path = Path(result.excel_path or "")
        if not path.exists():
            raise HTTPException(status_code=404, detail="Excel artifact is unavailable on this API node.")
        return FileResponse(
            path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=path.name,
        )

    @app.post("/v1/research", response_model=ResearchResult)
    def research(request: CompanyResearchRequest) -> ResearchResult:
        return pipeline.research(request)

    @app.post("/v1/collect", response_model=ResearchResult)
    def collect(request: CompanyResearchRequest) -> ResearchResult:
        return pipeline.collect(request)

    @app.post("/v1/jobs/collect", status_code=202)
    def enqueue_collect(request: CompanyResearchRequest) -> dict[str, str]:
        if not settings.celery_enabled:
            raise HTTPException(status_code=503, detail="Celery is disabled. Set CELERY_ENABLED=true.")
        from .tasks import enqueue_collection

        task = enqueue_collection(request)
        return {"task_id": task.id, "state": task.state}

    @app.post("/v1/jobs/research", status_code=202)
    def enqueue_research_job(request: CompanyResearchRequest) -> dict[str, str]:
        if not settings.celery_enabled:
            raise HTTPException(status_code=503, detail="Celery is disabled. Set CELERY_ENABLED=true.")
        from .tasks import enqueue_research

        task = enqueue_research(request)
        return {"task_id": task.id, "state": task.state}

    @app.get("/v1/jobs/{task_id}")
    def job_status(task_id: str) -> dict[str, object]:
        if not settings.celery_enabled:
            raise HTTPException(status_code=503, detail="Celery is disabled. Set CELERY_ENABLED=true.")
        from celery.result import AsyncResult
        from .celery_app import celery_app

        result = AsyncResult(task_id, app=celery_app)
        payload: dict[str, object] = {"task_id": task_id, "state": result.state}
        if result.successful():
            payload["result"] = result.result
        elif result.failed():
            payload["error"] = str(result.result)
        elif isinstance(result.info, dict):
            payload["progress"] = result.info
        return payload

    return app


app = create_app()
