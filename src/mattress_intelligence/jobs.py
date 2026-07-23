"""Durable research-job registry shared by Streamlit, API, and Celery workers."""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import CompanyResearchRequest
from .settings import Settings


TERMINAL_JOB_STATUSES = frozenset({"completed", "failed", "cancelled"})
ACTIVE_JOB_STATUSES = frozenset({"queued", "running"})

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
    "failed": 100,
}

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS research_jobs (
    job_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL UNIQUE,
    company_name TEXT NOT NULL,
    official_domain TEXT NOT NULL,
    request_json TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    progress INTEGER NOT NULL,
    message TEXT,
    submitted_at TEXT NOT NULL,
    started_at TEXT,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    run_id TEXT,
    output_dir TEXT NOT NULL,
    excel_path TEXT,
    table_csv_path TEXT,
    table_json_path TEXT,
    result_json_path TEXT,
    error TEXT,
    summary_json TEXT,
    backend_cleared_at TEXT
);

CREATE INDEX IF NOT EXISTS research_jobs_status_idx
    ON research_jobs(status, submitted_at DESC);
CREATE INDEX IF NOT EXISTS research_jobs_company_idx
    ON research_jobs(company_name, submitted_at DESC);
"""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return normalized or "company"


def build_job_output_dir(settings: Settings, company_name: str, job_id: str) -> Path:
    stamp = utc_now().strftime("%Y%m%d_%H%M%S")
    return settings.output_dir / "sessions" / f"{stamp}_{_slug(company_name)}_{job_id[:8]}"


@dataclass(frozen=True, slots=True)
class ResearchJob:
    job_id: str
    task_id: str
    company_name: str
    official_domain: str
    request_json: str
    status: str
    stage: str
    progress: int
    message: str | None
    submitted_at: datetime
    started_at: datetime | None
    updated_at: datetime
    completed_at: datetime | None
    run_id: str | None
    output_dir: str
    excel_path: str | None
    table_csv_path: str | None
    table_json_path: str | None
    result_json_path: str | None
    error: str | None
    summary_json: str | None
    backend_cleared_at: datetime | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ResearchJob":
        return cls(
            job_id=str(row["job_id"]),
            task_id=str(row["task_id"]),
            company_name=str(row["company_name"]),
            official_domain=str(row["official_domain"]),
            request_json=str(row["request_json"]),
            status=str(row["status"]),
            stage=str(row["stage"]),
            progress=int(row["progress"]),
            message=str(row["message"]) if row["message"] is not None else None,
            submitted_at=_parse_datetime(row["submitted_at"]) or utc_now(),
            started_at=_parse_datetime(row["started_at"]),
            updated_at=_parse_datetime(row["updated_at"]) or utc_now(),
            completed_at=_parse_datetime(row["completed_at"]),
            run_id=str(row["run_id"]) if row["run_id"] is not None else None,
            output_dir=str(row["output_dir"]),
            excel_path=str(row["excel_path"]) if row["excel_path"] is not None else None,
            table_csv_path=(
                str(row["table_csv_path"]) if row["table_csv_path"] is not None else None
            ),
            table_json_path=(
                str(row["table_json_path"]) if row["table_json_path"] is not None else None
            ),
            result_json_path=(
                str(row["result_json_path"]) if row["result_json_path"] is not None else None
            ),
            error=str(row["error"]) if row["error"] is not None else None,
            summary_json=(
                str(row["summary_json"]) if row["summary_json"] is not None else None
            ),
            backend_cleared_at=_parse_datetime(row["backend_cleared_at"]),
        )

    @property
    def request(self) -> CompanyResearchRequest:
        return CompanyResearchRequest.model_validate_json(self.request_json)

    @property
    def summary(self) -> dict[str, Any]:
        if not self.summary_json:
            return {}
        value = json.loads(self.summary_json)
        return value if isinstance(value, dict) else {}

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_JOB_STATUSES

    def model_dump(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "submitted_at",
            "started_at",
            "updated_at",
            "completed_at",
            "backend_cleared_at",
        ):
            payload[key] = _iso(payload[key])
        payload["summary"] = self.summary
        payload.pop("summary_json", None)
        payload.pop("request_json", None)
        return payload


class ResearchJobStore:
    """Small SQLite job ledger that removes Celery-result-backend ambiguity."""

    _UPDATABLE_COLUMNS = frozenset(
        {
            "task_id",
            "status",
            "stage",
            "progress",
            "message",
            "started_at",
            "completed_at",
            "run_id",
            "excel_path",
            "table_csv_path",
            "table_json_path",
            "result_json_path",
            "error",
            "summary_json",
            "backend_cleared_at",
        }
    )

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as connection:
            connection.executescript(_SCHEMA)

    @classmethod
    def from_settings(cls, settings: Settings) -> "ResearchJobStore":
        return cls(settings.job_database_path)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def create(
        self,
        request: CompanyResearchRequest,
        *,
        job_id: str | None = None,
        output_dir: Path | None = None,
    ) -> ResearchJob:
        resolved_job_id = job_id or uuid4().hex
        destination = output_dir or self.path.parent / "sessions" / resolved_job_id
        destination.mkdir(parents=True, exist_ok=True)
        now = utc_now()
        payload = (
            resolved_job_id,
            resolved_job_id,
            request.company_name,
            request.official_domain,
            request.model_dump_json(),
            "queued",
            "queued",
            _STAGE_PROGRESS["queued"],
            "Waiting for a Celery worker",
            now.isoformat(),
            now.isoformat(),
            str(destination),
        )
        with closing(self.connect()) as connection:
            connection.execute(
                """INSERT INTO research_jobs (
                    job_id, task_id, company_name, official_domain, request_json,
                    status, stage, progress, message, submitted_at, updated_at, output_dir
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                payload,
            )
            connection.commit()
        return self.get(resolved_job_id)

    def get(self, job_id: str) -> ResearchJob:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT * FROM research_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown job_id: {job_id}")
        return ResearchJob.from_row(row)

    def list(self, *, limit: int = 50, include_archived: bool = True) -> list[ResearchJob]:
        del include_archived  # Reserved for a future non-destructive archive view.
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """SELECT * FROM research_jobs
                ORDER BY submitted_at DESC LIMIT ?""",
                (max(1, limit),),
            ).fetchall()
        return [ResearchJob.from_row(row) for row in rows]

    def update(self, job_id: str, **fields: object) -> ResearchJob:
        if not fields:
            return self.get(job_id)
        unknown = set(fields) - self._UPDATABLE_COLUMNS
        if unknown:
            raise ValueError(f"Unsupported job field(s): {', '.join(sorted(unknown))}")

        values = dict(fields)
        values["updated_at"] = utc_now().isoformat()
        assignments: list[str] = []
        parameters: list[object] = []
        for key, value in values.items():
            assignments.append(f"{key} = ?")
            if isinstance(value, datetime):
                parameters.append(value.isoformat())
            elif key == "summary_json" and isinstance(value, dict):
                parameters.append(json.dumps(value, separators=(",", ":"), default=str))
            else:
                parameters.append(value)
        parameters.append(job_id)
        with closing(self.connect()) as connection:
            cursor = connection.execute(
                f"UPDATE research_jobs SET {', '.join(assignments)} WHERE job_id = ?",
                parameters,
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Unknown job_id: {job_id}")
            connection.commit()
        return self.get(job_id)

    def mark_running(
        self,
        job_id: str,
        *,
        task_id: str,
        stage: str = "initializing",
        message: str | None = None,
    ) -> ResearchJob:
        return self.update(
            job_id,
            task_id=task_id,
            status="running",
            stage=stage,
            progress=_STAGE_PROGRESS.get(stage, 5),
            message=message or "Worker started",
            started_at=utc_now(),
            error=None,
        )

    def mark_progress(
        self,
        job_id: str,
        stage: str,
        *,
        message: str | None = None,
        progress: int | None = None,
    ) -> ResearchJob:
        return self.update(
            job_id,
            status="running",
            stage=stage,
            progress=max(0, min(100, progress if progress is not None else _STAGE_PROGRESS.get(stage, 50))),
            message=message,
        )

    def mark_completed(
        self,
        job_id: str,
        *,
        run_id: str,
        summary: dict[str, object],
        excel_path: str | None,
        table_csv_path: str | None,
        table_json_path: str | None,
        result_json_path: str | None,
    ) -> ResearchJob:
        return self.update(
            job_id,
            status="completed",
            stage="completed",
            progress=100,
            message="Research complete",
            completed_at=utc_now(),
            run_id=run_id,
            excel_path=excel_path,
            table_csv_path=table_csv_path,
            table_json_path=table_json_path,
            result_json_path=result_json_path,
            error=None,
            summary_json=summary,
        )

    def mark_failed(self, job_id: str, error: str) -> ResearchJob:
        return self.update(
            job_id,
            status="failed",
            stage="failed",
            progress=100,
            message="Research failed",
            completed_at=utc_now(),
            error=error,
        )

    def mark_backend_cleared(self, job_id: str) -> ResearchJob:
        return self.update(job_id, backend_cleared_at=utc_now())

    def delete(self, job_id: str, *, remove_output: bool = False, output_root: Path | None = None) -> None:
        job = self.get(job_id)
        if remove_output:
            destination = Path(job.output_dir).resolve()
            if output_root is None:
                raise ValueError("output_root is required when remove_output=True")
            root = output_root.resolve()
            if destination != root and root in destination.parents and destination.exists():
                shutil.rmtree(destination)
        with closing(self.connect()) as connection:
            connection.execute("DELETE FROM research_jobs WHERE job_id = ?", (job_id,))
            connection.commit()
