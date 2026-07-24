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
from typing import Any, Literal
from uuid import uuid4

from .models import CompanyResearchRequest, ResearchResult
from .settings import Settings


TERMINAL_JOB_STATUSES = frozenset({"completed", "failed", "cancelled"})
ACTIVE_JOB_STATUSES = frozenset({"queued", "running"})

_STAGE_PROGRESS = {
    "queued": 3,
    "initializing": 7,
    "discovering": 15,
    "crawling": 35,
    "extracting": 58,
    "assets": 70,
    "visual_followup": 76,
    "material_decoding": 81,
    "material_evidence": 84,
    "material_adjudication": 88,
    "resolving": 89,
    "analyzing": 92,
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
    backend_cleared_at TEXT,
    execution_token TEXT,
    heartbeat_at TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS research_jobs_status_idx
    ON research_jobs(status, submitted_at DESC);
CREATE INDEX IF NOT EXISTS research_jobs_company_idx
    ON research_jobs(company_name, submitted_at DESC);
"""

_REQUIRED_COLUMNS: dict[str, str] = {
    "execution_token": "TEXT",
    "heartbeat_at": "TEXT",
    "attempt_count": "INTEGER NOT NULL DEFAULT 0",
}


ClaimDisposition = Literal[
    "claimed",
    "already_running",
    "completed",
    "terminal",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return normalized or "company"


def build_job_output_dir(settings: Settings, company_name: str, job_id: str) -> Path:
    stamp = utc_now().strftime("%Y%m%d_%H%M%S")
    return settings.output_dir / "sessions" / f"{stamp}_{_slug(company_name)}_{job_id[:8]}"


def _result_summary(result: ResearchResult) -> dict[str, object]:
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
    execution_token: str | None
    heartbeat_at: datetime | None
    attempt_count: int

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
            execution_token=(
                str(row["execution_token"]) if row["execution_token"] is not None else None
            ),
            heartbeat_at=_parse_datetime(row["heartbeat_at"]),
            attempt_count=int(row["attempt_count"] or 0),
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
            "heartbeat_at",
        ):
            payload[key] = _iso(payload[key])
        payload["summary"] = self.summary
        payload.pop("summary_json", None)
        payload.pop("request_json", None)
        payload.pop("execution_token", None)
        return payload


class ResearchJobStore:
    """SQLite job ledger with immutable terminal state and duplicate-execution leases."""

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
            "execution_token",
            "heartbeat_at",
            "attempt_count",
        }
    )

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as connection:
            connection.executescript(_SCHEMA)
            self._migrate_schema(connection)
            connection.commit()

    @staticmethod
    def _migrate_schema(connection: sqlite3.Connection) -> None:
        existing = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(research_jobs)").fetchall()
        }
        for column, declaration in _REQUIRED_COLUMNS.items():
            if column not in existing:
                connection.execute(
                    f"ALTER TABLE research_jobs ADD COLUMN {column} {declaration}"
                )

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

    def claim_for_execution(
        self,
        job_id: str,
        *,
        task_id: str,
        execution_token: str,
        stale_after_seconds: int,
        message: str = "Initializing services and storage",
    ) -> tuple[ResearchJob, ClaimDisposition]:
        """Atomically claim one delivery while rejecting live duplicates.

        A stale lease may be reclaimed after a worker crash. A completed job is never
        reopened, and a concurrent redelivery cannot acquire the same job.
        """

        now = utc_now()
        with closing(self.connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM research_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError(f"Unknown job_id: {job_id}")
            current = ResearchJob.from_row(row)

            if current.status == "completed":
                connection.commit()
                return current, "completed"
            if current.status in {"failed", "cancelled"}:
                connection.commit()
                return current, "terminal"

            lease_time = current.heartbeat_at or current.updated_at or current.started_at
            lease_is_stale = (
                current.status == "running"
                and current.execution_token is not None
                and lease_time is not None
                and (now - lease_time).total_seconds() >= max(60, stale_after_seconds)
            )
            may_claim = (
                current.status == "queued"
                or current.execution_token is None
                or lease_is_stale
            )
            if not may_claim:
                connection.commit()
                return current, "already_running"

            connection.execute(
                """UPDATE research_jobs
                SET task_id = ?, status = 'running', stage = 'initializing', progress = ?,
                    message = ?, started_at = COALESCE(started_at, ?), updated_at = ?,
                    error = NULL, execution_token = ?, heartbeat_at = ?,
                    attempt_count = COALESCE(attempt_count, 0) + 1
                WHERE job_id = ?""",
                (
                    task_id,
                    _STAGE_PROGRESS["initializing"],
                    message,
                    now.isoformat(),
                    now.isoformat(),
                    execution_token,
                    now.isoformat(),
                    job_id,
                ),
            )
            connection.commit()
        return self.get(job_id), "claimed"

    def mark_running(
        self,
        job_id: str,
        *,
        task_id: str,
        stage: str = "initializing",
        message: str | None = None,
    ) -> ResearchJob:
        """Reflect Celery STARTED state without reopening a terminal job."""

        now = utc_now().isoformat()
        with closing(self.connect()) as connection:
            connection.execute(
                """UPDATE research_jobs
                SET task_id = ?, status = 'running', stage = ?, progress = ?, message = ?,
                    started_at = COALESCE(started_at, ?), updated_at = ?, error = NULL,
                    heartbeat_at = COALESCE(heartbeat_at, ?)
                WHERE job_id = ? AND status IN ('queued', 'running')""",
                (
                    task_id,
                    stage,
                    _STAGE_PROGRESS.get(stage, 5),
                    message or "Worker started",
                    now,
                    now,
                    now,
                    job_id,
                ),
            )
            connection.commit()
        return self.get(job_id)

    def mark_progress(
        self,
        job_id: str,
        stage: str,
        *,
        message: str | None = None,
        progress: int | None = None,
        execution_token: str | None = None,
    ) -> ResearchJob:
        resolved_progress = max(
            0,
            min(100, progress if progress is not None else _STAGE_PROGRESS.get(stage, 50)),
        )
        now = utc_now().isoformat()
        token_clause = "" if execution_token is None else " AND execution_token = ?"
        parameters: list[object] = [
            stage,
            resolved_progress,
            message,
            now,
            now,
            job_id,
        ]
        if execution_token is not None:
            parameters.append(execution_token)
        with closing(self.connect()) as connection:
            connection.execute(
                """UPDATE research_jobs
                SET status = 'running', stage = ?, progress = ?, message = ?,
                    updated_at = ?, heartbeat_at = ?
                WHERE job_id = ? AND status IN ('queued', 'running')"""
                + token_clause,
                parameters,
            )
            connection.commit()
        return self.get(job_id)

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
        execution_token: str | None = None,
    ) -> ResearchJob:
        now = utc_now().isoformat()
        token_clause = "" if execution_token is None else " AND execution_token = ?"
        parameters: list[object] = [
            now,
            now,
            run_id,
            excel_path,
            table_csv_path,
            table_json_path,
            result_json_path,
            json.dumps(summary, separators=(",", ":"), default=str),
            job_id,
        ]
        if execution_token is not None:
            parameters.append(execution_token)
        with closing(self.connect()) as connection:
            connection.execute(
                """UPDATE research_jobs
                SET status = 'completed', stage = 'completed', progress = 100,
                    message = 'Research complete', completed_at = ?, updated_at = ?,
                    run_id = ?, excel_path = ?, table_csv_path = ?, table_json_path = ?,
                    result_json_path = ?, error = NULL, summary_json = ?,
                    execution_token = NULL, heartbeat_at = NULL
                WHERE job_id = ? AND status IN ('queued', 'running')"""
                + token_clause,
                parameters,
            )
            connection.commit()
        return self.get(job_id)

    def mark_failed(
        self,
        job_id: str,
        error: str,
        *,
        execution_token: str | None = None,
    ) -> ResearchJob:
        now = utc_now().isoformat()
        token_clause = "" if execution_token is None else " AND execution_token = ?"
        parameters: list[object] = [now, now, error, job_id]
        if execution_token is not None:
            parameters.append(execution_token)
        with closing(self.connect()) as connection:
            connection.execute(
                """UPDATE research_jobs
                SET status = 'failed', stage = 'failed', progress = 100,
                    message = 'Research failed', completed_at = ?, updated_at = ?, error = ?,
                    execution_token = NULL, heartbeat_at = NULL
                WHERE job_id = ? AND status IN ('queued', 'running')"""
                + token_clause,
                parameters,
            )
            connection.commit()
        return self.get(job_id)

    def recover_completed_from_artifacts(self, job_id: str) -> ResearchJob:
        """Restore a completed UI session from its validated durable result JSON.

        This repairs ledgers already damaged by an older duplicate delivery. It does not
        infer completion from an Excel file alone; the complete Pydantic result must parse.
        """

        job = self.get(job_id)
        if job.status == "completed":
            return job

        candidates: list[Path] = []
        if job.result_json_path:
            candidates.append(Path(job.result_json_path))
        candidates.append(Path(job.output_dir) / "research_result.json")

        result_path = next((path for path in candidates if path.is_file()), None)
        if result_path is None:
            return job
        try:
            result = ResearchResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return job

        output_dir = Path(job.output_dir)
        excel_candidate = Path(result.excel_path or "")
        if not excel_candidate.is_file():
            excel_candidate = output_dir / "complete_research.xlsx"
        table_csv = output_dir / "displayed_products.csv"
        table_json = output_dir / "displayed_products.json"

        summary = {**job.summary, **_result_summary(result)}
        summary["run_id"] = result.run_id
        summary["job_id"] = job.job_id
        summary["excel_path"] = str(excel_candidate) if excel_candidate.is_file() else result.excel_path
        material_artifacts = {
            "material_csv_path": output_dir / "trademark_materials.csv",
            "material_json_path": output_dir / "trademark_materials.json",
            "material_excel_path": output_dir / "trademark_materials.xlsx",
        }
        for key, path in material_artifacts.items():
            if path.is_file():
                summary[key] = str(path)

        completed_at = result.completed_at.isoformat()
        with closing(self.connect()) as connection:
            connection.execute(
                """UPDATE research_jobs
                SET status = 'completed', stage = 'completed', progress = 100,
                    message = 'Research complete (restored from durable artifacts)',
                    completed_at = ?, updated_at = ?, run_id = ?, excel_path = ?,
                    table_csv_path = ?, table_json_path = ?, result_json_path = ?,
                    error = NULL, summary_json = ?, execution_token = NULL, heartbeat_at = NULL
                WHERE job_id = ?""",
                (
                    completed_at,
                    utc_now().isoformat(),
                    result.run_id,
                    str(excel_candidate) if excel_candidate.is_file() else result.excel_path,
                    str(table_csv) if table_csv.is_file() else job.table_csv_path,
                    str(table_json) if table_json.is_file() else job.table_json_path,
                    str(result_path),
                    json.dumps(summary, separators=(",", ":"), default=str),
                    job_id,
                ),
            )
            connection.commit()
        return self.get(job_id)

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
