"""Optional standalone FastAPI service wrapper."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .models import CompanyResearchRequest, ResearchResult
from .pipeline import MattressIntelligencePipeline


def create_app() -> FastAPI:
    app = FastAPI(title="BRIXTA R&D API", version="1.2.0")
    pipeline = MattressIntelligencePipeline()

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "recognition_provider": pipeline.llm.name,
            "search_provider": pipeline.search_provider.name,
            "deterministic_extraction": True,
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
            raise HTTPException(status_code=404, detail="Excel artifact is unavailable.")
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
        """Collect and extract evidence without running engineering inference."""

        return pipeline.collect(request)

    return app


app = create_app()
