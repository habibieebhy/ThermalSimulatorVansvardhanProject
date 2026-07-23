from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from mattress_intelligence.exporter import (
    export_primary_artifacts,
    primary_table_frame,
    table_csv_bytes,
    table_excel_bytes,
    table_json_bytes,
)
from mattress_intelligence.jobs import ResearchJobStore
from mattress_intelligence.models import (
    CatalogueCoverage,
    CompanyResearchRequest,
    ProductRecord,
    ResearchResult,
)


def _request(company: str = "Example Sleep") -> CompanyResearchRequest:
    return CompanyResearchRequest(
        company_name=company,
        official_domain="https://example.com",
    )


def _result() -> ResearchResult:
    request = _request()
    now = datetime.now(timezone.utc)
    return ResearchResult(
        run_id="run_test",
        request=request,
        started_at=now,
        completed_at=now,
        products=[
            ProductRecord(
                company_id=request.company_id,
                company_name=request.company_name,
                brand=request.company_name,
                name="Cloud Mattress",
                family="Cloud",
                firmness="Medium",
                total_thickness_mm=200,
                price=19999,
                currency="INR",
                extraction_confidence=0.87654,
            )
        ],
        sources=[],
        claims=[],
        configurations=[],
        graph_edges=[],
        coverage=CatalogueCoverage(unique_products=1, product_pages=1),
    )


def test_job_store_keeps_company_sessions_isolated() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        store = ResearchJobStore(root / "research_jobs.sqlite3")
        first = store.create(_request("Company One"), output_dir=root / "one")
        second = store.create(_request("Company Two"), output_dir=root / "two")

        assert first.job_id != second.job_id
        assert first.task_id == first.job_id
        assert second.task_id == second.job_id
        assert first.output_dir != second.output_dir

        store.mark_running(first.job_id, task_id=first.task_id, stage="crawling")
        store.mark_progress(first.job_id, "extracting", message="Extracting products")
        completed = store.mark_completed(
            first.job_id,
            run_id="run_one",
            summary={"products": 7},
            excel_path=str(root / "one" / "complete_research.xlsx"),
            table_csv_path=str(root / "one" / "displayed_products.csv"),
            table_json_path=str(root / "one" / "displayed_products.json"),
            result_json_path=str(root / "one" / "research_result.json"),
        )

        assert completed.status == "completed"
        assert completed.progress == 100
        assert completed.summary["products"] == 7
        assert store.get(second.job_id).status == "queued"
        assert {item.company_name for item in store.list()} == {"Company One", "Company Two"}


def test_primary_table_downloads_use_the_same_columns_and_rows() -> None:
    result = _result()
    frame = primary_table_frame(result)

    assert list(frame.columns) == [
        "Product",
        "Family",
        "Firmness",
        "Thickness (mm)",
        "Price",
        "Currency",
        "Layers",
        "Variants",
        "Confidence",
        "URL",
    ]
    assert frame.iloc[0]["Product"] == "Cloud Mattress"
    assert frame.iloc[0]["Confidence"] == 0.877
    assert b"Cloud Mattress" in table_csv_bytes(frame)
    assert json.loads(table_json_bytes(frame))[0]["Product"] == "Cloud Mattress"
    assert len(table_excel_bytes(frame)) > 1_000

    with tempfile.TemporaryDirectory() as temporary:
        paths = export_primary_artifacts(result, Path(temporary))
        assert Path(paths["table_csv_path"]).is_file()
        assert Path(paths["table_json_path"]).is_file()
        assert Path(paths["result_json_path"]).is_file()
