"""FastAPI service with synchronous runs and durable Celery research sessions."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .jobs import ResearchJobStore, build_job_output_dir
from .models import CompanyResearchRequest, ResearchResult
from .pipeline import MattressIntelligencePipeline
from .settings import Settings


def create_app() -> FastAPI:
    settings = Settings()
    settings.ensure_directories()
    app = FastAPI(title="BRIXTA Mattress Intelligence API", version="1.6.1")
    pipeline = MattressIntelligencePipeline(settings)
    jobs = ResearchJobStore.from_settings(settings)

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "recognition_provider": pipeline.llm.name,
            "search_provider": pipeline.search_provider.name,
            "capture_strategy": settings.capture_strategy,
            "database": "postgres/neon" if settings.postgres_enabled else "sqlite",
            "job_ledger": str(settings.job_database_path),
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
        if not path.is_file():
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

    def enqueue(request: CompanyResearchRequest, *, mode: str) -> dict[str, object]:
        if not settings.celery_enabled:
            raise HTTPException(status_code=503, detail="Celery is disabled. Set CELERY_ENABLED=true.")
        from .tasks import enqueue_collection, enqueue_research

        job_id = uuid4().hex
        output_dir = build_job_output_dir(settings, request.company_name, job_id)
        job = jobs.create(request, job_id=job_id, output_dir=output_dir)
        output_path = output_dir / "complete_research.xlsx"
        try:
            if mode == "collect":
                enqueue_collection(request, output_path, job_id=job_id)
            else:
                enqueue_research(request, output_path, job_id=job_id)
        except Exception as exc:
            jobs.mark_failed(job_id, f"The job could not be submitted to Celery: {exc}")
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return job.model_dump()

    @app.post("/v1/jobs/collect", status_code=202)
    def enqueue_collect(request: CompanyResearchRequest) -> dict[str, object]:
        return enqueue(request, mode="collect")

    @app.post("/v1/jobs/research", status_code=202)
    def enqueue_research_job(request: CompanyResearchRequest) -> dict[str, object]:
        return enqueue(request, mode="research")

    @app.get("/v1/jobs")
    def list_jobs(limit: int = 50) -> list[dict[str, object]]:
        return [job.model_dump() for job in jobs.list(limit=max(1, min(limit, 500)))]

    @app.get("/v1/jobs/{job_id}")
    def job_status(job_id: str) -> dict[str, object]:
        try:
            return jobs.get(job_id).model_dump()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app


app = create_app()
