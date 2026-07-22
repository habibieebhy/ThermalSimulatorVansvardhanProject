"""SQLite or PostgreSQL/Neon persistence for research runs and evidence."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from .models import ProductRecord, ResearchResult
from .settings import Settings


SQLITE_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS research_runs (
    run_id TEXT PRIMARY KEY,
    company_id TEXT NOT NULL,
    company_name TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT NOT NULL,
    run_id TEXT NOT NULL REFERENCES research_runs(run_id) ON DELETE CASCADE,
    company_id TEXT NOT NULL,
    url TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (source_id, run_id)
);
CREATE TABLE IF NOT EXISTS assets (
    asset_id TEXT NOT NULL,
    run_id TEXT NOT NULL REFERENCES research_runs(run_id) ON DELETE CASCADE,
    source_id TEXT NOT NULL,
    company_id TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (asset_id, run_id)
);
CREATE TABLE IF NOT EXISTS products (
    product_id TEXT NOT NULL,
    run_id TEXT NOT NULL REFERENCES research_runs(run_id) ON DELETE CASCADE,
    company_id TEXT NOT NULL,
    name TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (product_id, run_id)
);
CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT NOT NULL,
    run_id TEXT NOT NULL REFERENCES research_runs(run_id) ON DELETE CASCADE,
    product_id TEXT NOT NULL,
    field_path TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (claim_id, run_id)
);
CREATE TABLE IF NOT EXISTS observations (
    observation_id TEXT NOT NULL,
    run_id TEXT NOT NULL REFERENCES research_runs(run_id) ON DELETE CASCADE,
    source_id TEXT NOT NULL,
    asset_id TEXT,
    field_path TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (observation_id, run_id)
);
CREATE TABLE IF NOT EXISTS configurations (
    configuration_id TEXT NOT NULL,
    run_id TEXT NOT NULL REFERENCES research_runs(run_id) ON DELETE CASCADE,
    product_id TEXT NOT NULL,
    rank INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (configuration_id, run_id)
);
CREATE TABLE IF NOT EXISTS graph_edges (
    edge_id TEXT NOT NULL,
    run_id TEXT NOT NULL REFERENCES research_runs(run_id) ON DELETE CASCADE,
    source_node TEXT NOT NULL,
    relation TEXT NOT NULL,
    target_node TEXT NOT NULL,
    properties_json TEXT NOT NULL,
    PRIMARY KEY (edge_id, run_id)
);
CREATE INDEX IF NOT EXISTS products_company_idx ON products(company_id);
CREATE INDEX IF NOT EXISTS claims_product_idx ON claims(product_id);
CREATE INDEX IF NOT EXISTS observations_source_idx ON observations(source_id, field_path);
CREATE INDEX IF NOT EXISTS observations_asset_idx ON observations(asset_id);
CREATE INDEX IF NOT EXISTS assets_source_idx ON assets(source_id);
CREATE INDEX IF NOT EXISTS assets_hash_idx ON assets(content_sha256);
CREATE INDEX IF NOT EXISTS edges_source_idx ON graph_edges(source_node, relation);
"""


POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_runs (
    run_id TEXT PRIMARY KEY,
    company_id TEXT NOT NULL,
    company_name TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ NOT NULL,
    payload_json JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT NOT NULL,
    run_id TEXT NOT NULL REFERENCES research_runs(run_id) ON DELETE CASCADE,
    company_id TEXT NOT NULL,
    url TEXT NOT NULL,
    payload_json JSONB NOT NULL,
    PRIMARY KEY (source_id, run_id)
);
CREATE TABLE IF NOT EXISTS assets (
    asset_id TEXT NOT NULL,
    run_id TEXT NOT NULL REFERENCES research_runs(run_id) ON DELETE CASCADE,
    source_id TEXT NOT NULL,
    company_id TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    payload_json JSONB NOT NULL,
    PRIMARY KEY (asset_id, run_id)
);
CREATE TABLE IF NOT EXISTS products (
    product_id TEXT NOT NULL,
    run_id TEXT NOT NULL REFERENCES research_runs(run_id) ON DELETE CASCADE,
    company_id TEXT NOT NULL,
    name TEXT NOT NULL,
    payload_json JSONB NOT NULL,
    PRIMARY KEY (product_id, run_id)
);
CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT NOT NULL,
    run_id TEXT NOT NULL REFERENCES research_runs(run_id) ON DELETE CASCADE,
    product_id TEXT NOT NULL,
    field_path TEXT NOT NULL,
    payload_json JSONB NOT NULL,
    PRIMARY KEY (claim_id, run_id)
);
CREATE TABLE IF NOT EXISTS observations (
    observation_id TEXT NOT NULL,
    run_id TEXT NOT NULL REFERENCES research_runs(run_id) ON DELETE CASCADE,
    source_id TEXT NOT NULL,
    asset_id TEXT,
    field_path TEXT NOT NULL,
    payload_json JSONB NOT NULL,
    PRIMARY KEY (observation_id, run_id)
);
CREATE TABLE IF NOT EXISTS configurations (
    configuration_id TEXT NOT NULL,
    run_id TEXT NOT NULL REFERENCES research_runs(run_id) ON DELETE CASCADE,
    product_id TEXT NOT NULL,
    rank INTEGER NOT NULL,
    payload_json JSONB NOT NULL,
    PRIMARY KEY (configuration_id, run_id)
);
CREATE TABLE IF NOT EXISTS graph_edges (
    edge_id TEXT NOT NULL,
    run_id TEXT NOT NULL REFERENCES research_runs(run_id) ON DELETE CASCADE,
    source_node TEXT NOT NULL,
    relation TEXT NOT NULL,
    target_node TEXT NOT NULL,
    properties_json JSONB NOT NULL,
    PRIMARY KEY (edge_id, run_id)
);
CREATE INDEX IF NOT EXISTS products_company_idx ON products(company_id);
CREATE INDEX IF NOT EXISTS claims_product_idx ON claims(product_id);
CREATE INDEX IF NOT EXISTS observations_source_idx ON observations(source_id, field_path);
CREATE INDEX IF NOT EXISTS observations_asset_idx ON observations(asset_id);
CREATE INDEX IF NOT EXISTS assets_source_idx ON assets(source_id);
CREATE INDEX IF NOT EXISTS assets_hash_idx ON assets(content_sha256);
CREATE INDEX IF NOT EXISTS edges_source_idx ON graph_edges(source_node, relation);
"""


def _json(value: BaseModel | dict) -> str:
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    return json.dumps(value, separators=(",", ":"), default=str)


def _migrate_payload(payload: dict) -> dict:
    payload.pop("simulations", None)
    request = payload.get("request")
    if isinstance(request, dict):
        request.pop("simulate_top_configurations", None)
        request.setdefault("discover_assets", True)
        request.setdefault("analyze_assets_with_vision", True)
        request.setdefault("max_assets_per_document", 30)
        request.setdefault("max_vision_assets", 80)
        request.setdefault("max_pdf_pages", 100)
    payload.setdefault("assets", [])
    payload.setdefault("observations", [])
    payload.setdefault("crawl_log", [])
    payload.setdefault("acquisition_log", [])
    payload.setdefault("recognition_log", [])
    coverage = payload.get("coverage")
    if isinstance(coverage, dict):
        coverage.setdefault("assets", len(payload["assets"]))
        coverage.setdefault(
            "vision_assets",
            sum(1 for item in payload["assets"] if item.get("vision_payload")),
        )
    return payload


class Repository(Protocol):
    def save(self, result: ResearchResult) -> None: ...
    def load(self, run_id: str) -> ResearchResult: ...
    def load_products(self, exclude_run_id: str | None = None) -> list[ProductRecord]: ...
    def list_runs(self) -> list[dict]: ...
    def check_connection(self) -> dict[str, object]: ...


class SQLiteRepository:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as connection:
            connection.executescript(SQLITE_SCHEMA)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def save(self, result: ResearchResult) -> None:
        with closing(self.connect()) as connection:
            connection.execute("BEGIN")
            connection.execute(
                """INSERT OR REPLACE INTO research_runs
                (run_id, company_id, company_name, started_at, completed_at, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    result.run_id,
                    result.request.company_id,
                    result.request.company_name,
                    result.started_at.isoformat(),
                    result.completed_at.isoformat(),
                    _json(result),
                ),
            )
            for table in (
                "sources", "assets", "products", "claims", "observations",
                "configurations", "graph_edges",
            ):
                connection.execute(f"DELETE FROM {table} WHERE run_id = ?", (result.run_id,))
            connection.executemany(
                "INSERT INTO sources VALUES (?, ?, ?, ?, ?)",
                [(x.source_id, result.run_id, x.company_id, x.url, _json(x)) for x in result.sources],
            )
            connection.executemany(
                "INSERT INTO assets VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (x.asset_id, result.run_id, x.source_id, x.company_id, x.content_sha256, _json(x))
                    for x in result.assets
                ],
            )
            connection.executemany(
                "INSERT INTO products VALUES (?, ?, ?, ?, ?)",
                [
                    (x.product_id, result.run_id, x.company_id, x.name, _json(x))
                    for x in result.products
                ],
            )
            connection.executemany(
                "INSERT INTO claims VALUES (?, ?, ?, ?, ?)",
                [(x.claim_id, result.run_id, x.product_id, x.field_path, _json(x)) for x in result.claims],
            )
            connection.executemany(
                "INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (x.observation_id, result.run_id, x.source_id, x.asset_id, x.field_path, _json(x))
                    for x in result.observations
                ],
            )
            connection.executemany(
                "INSERT INTO configurations VALUES (?, ?, ?, ?, ?)",
                [
                    (x.configuration_id, result.run_id, x.product_id, x.rank, _json(x))
                    for x in result.configurations
                ],
            )
            connection.executemany(
                "INSERT INTO graph_edges VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        x["edge_id"], result.run_id, x["source_node"], x["relation"],
                        x["target_node"], _json(x.get("properties", {})),
                    )
                    for x in result.graph_edges
                ],
            )
            connection.commit()

    def load(self, run_id: str) -> ResearchResult:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT payload_json FROM research_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        return ResearchResult.model_validate(_migrate_payload(json.loads(row["payload_json"])))

    def load_products(self, exclude_run_id: str | None = None) -> list[ProductRecord]:
        query = "SELECT payload_json FROM products"
        parameters: tuple[str, ...] = ()
        if exclude_run_id is not None:
            query += " WHERE run_id != ?"
            parameters = (exclude_run_id,)
        with closing(self.connect()) as connection:
            rows = connection.execute(query, parameters).fetchall()
        products: dict[str, ProductRecord] = {}
        for row in rows:
            product = ProductRecord.model_validate_json(row["payload_json"])
            products[str(product.product_id)] = product
        return list(products.values())

    def list_runs(self) -> list[dict]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """SELECT run_id, company_id, company_name, started_at, completed_at
                FROM research_runs ORDER BY completed_at DESC"""
            ).fetchall()
        return [dict(row) for row in rows]

    def check_connection(self) -> dict[str, object]:
        with closing(self.connect()) as connection:
            value = connection.execute("SELECT 1").fetchone()[0]
        return {"backend": "sqlite", "path": str(self.path), "ok": value == 1}


class PostgresRepository:
    """PostgreSQL persistence; suitable for Neon pooled and direct connection URLs."""

    def __init__(self, database_url: str, direct_url: str | None = None) -> None:
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "DATABASE_URL is configured but psycopg is not installed. Install psycopg[binary]."
            ) from exc
        self.psycopg = psycopg
        self.database_url = database_url
        self.schema_url = direct_url or database_url
        with self.psycopg.connect(self.schema_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(POSTGRES_SCHEMA)
            connection.commit()

    def _connect(self):
        return self.psycopg.connect(self.database_url)

    @staticmethod
    def _jsonb(value: BaseModel | dict) -> str:
        return _json(value)

    def save(self, result: ResearchResult) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO research_runs
                    (run_id, company_id, company_name, started_at, completed_at, payload_json)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (run_id) DO UPDATE SET
                    company_id=EXCLUDED.company_id, company_name=EXCLUDED.company_name,
                    started_at=EXCLUDED.started_at, completed_at=EXCLUDED.completed_at,
                    payload_json=EXCLUDED.payload_json""",
                    (
                        result.run_id, result.request.company_id, result.request.company_name,
                        result.started_at, result.completed_at, _json(result),
                    ),
                )
                for table in (
                    "sources", "assets", "products", "claims", "observations",
                    "configurations", "graph_edges",
                ):
                    cursor.execute(f"DELETE FROM {table} WHERE run_id = %s", (result.run_id,))
                cursor.executemany(
                    "INSERT INTO sources VALUES (%s,%s,%s,%s,%s::jsonb)",
                    [(x.source_id, result.run_id, x.company_id, x.url, _json(x)) for x in result.sources],
                )
                cursor.executemany(
                    "INSERT INTO assets VALUES (%s,%s,%s,%s,%s,%s::jsonb)",
                    [
                        (x.asset_id, result.run_id, x.source_id, x.company_id, x.content_sha256, _json(x))
                        for x in result.assets
                    ],
                )
                cursor.executemany(
                    "INSERT INTO products VALUES (%s,%s,%s,%s,%s::jsonb)",
                    [(x.product_id, result.run_id, x.company_id, x.name, _json(x)) for x in result.products],
                )
                cursor.executemany(
                    "INSERT INTO claims VALUES (%s,%s,%s,%s,%s::jsonb)",
                    [(x.claim_id, result.run_id, x.product_id, x.field_path, _json(x)) for x in result.claims],
                )
                cursor.executemany(
                    "INSERT INTO observations VALUES (%s,%s,%s,%s,%s,%s::jsonb)",
                    [
                        (x.observation_id, result.run_id, x.source_id, x.asset_id, x.field_path, _json(x))
                        for x in result.observations
                    ],
                )
                cursor.executemany(
                    "INSERT INTO configurations VALUES (%s,%s,%s,%s,%s::jsonb)",
                    [
                        (x.configuration_id, result.run_id, x.product_id, x.rank, _json(x))
                        for x in result.configurations
                    ],
                )
                cursor.executemany(
                    "INSERT INTO graph_edges VALUES (%s,%s,%s,%s,%s,%s::jsonb)",
                    [
                        (
                            x["edge_id"], result.run_id, x["source_node"], x["relation"],
                            x["target_node"], _json(x.get("properties", {})),
                        )
                        for x in result.graph_edges
                    ],
                )
            connection.commit()

    def load(self, run_id: str) -> ResearchResult:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT payload_json FROM research_runs WHERE run_id = %s", (run_id,))
                row = cursor.fetchone()
        if row is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        payload = row[0]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return ResearchResult.model_validate(_migrate_payload(dict(payload)))

    def load_products(self, exclude_run_id: str | None = None) -> list[ProductRecord]:
        query = "SELECT payload_json FROM products"
        params: tuple[str, ...] = ()
        if exclude_run_id is not None:
            query += " WHERE run_id != %s"
            params = (exclude_run_id,)
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                rows = cursor.fetchall()
        products: dict[str, ProductRecord] = {}
        for row in rows:
            payload = row[0]
            if isinstance(payload, str):
                payload = json.loads(payload)
            product = ProductRecord.model_validate(payload)
            products[str(product.product_id)] = product
        return list(products.values())

    def list_runs(self) -> list[dict]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT run_id, company_id, company_name, started_at, completed_at
                    FROM research_runs ORDER BY completed_at DESC"""
                )
                rows = cursor.fetchall()
        return [
            {
                "run_id": row[0], "company_id": row[1], "company_name": row[2],
                "started_at": row[3].isoformat(), "completed_at": row[4].isoformat(),
            }
            for row in rows
        ]

    def check_connection(self) -> dict[str, object]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT current_database(), version()")
                row = cursor.fetchone()
        if row is None:
            raise RuntimeError("DB retuned no rows")
        return {"backend": "postgresql", "database": row[0], "version": row[1], "ok": True}


def build_repository(settings: Settings) -> Repository:
    if settings.postgres_enabled:
        return PostgresRepository(settings.database_url or "", settings.database_direct_url)
    return SQLiteRepository(settings.database_path)
