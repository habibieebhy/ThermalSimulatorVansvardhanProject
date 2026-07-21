"""SQLite persistence for standalone use; schema maps directly to PostgreSQL."""

from __future__ import annotations

import json
from contextlib import closing
import sqlite3
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel

from .models import ProductRecord, ResearchResult


SCHEMA = """
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
    source_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES research_runs(run_id) ON DELETE CASCADE,
    company_id TEXT NOT NULL,
    url TEXT NOT NULL,
    payload_json TEXT NOT NULL
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
CREATE INDEX IF NOT EXISTS edges_source_idx ON graph_edges(source_node, relation);
"""


def _json(value: BaseModel | dict) -> str:
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    return json.dumps(value, separators=(",", ":"), default=str)


class SQLiteRepository:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as connection:
            connection.executescript(SCHEMA)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def save(self, result: ResearchResult) -> None:
        with closing(self.connect()) as connection:
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
            connection.executemany(
                "INSERT OR REPLACE INTO sources VALUES (?, ?, ?, ?, ?)",
                [
                    (source.source_id, result.run_id, source.company_id, source.url, _json(source))
                    for source in result.sources
                ],
            )
            connection.executemany(
                "INSERT OR REPLACE INTO products VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        product.product_id,
                        result.run_id,
                        product.company_id,
                        product.name,
                        _json(product),
                    )
                    for product in result.products
                ],
            )
            connection.executemany(
                "INSERT OR REPLACE INTO claims VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        claim.claim_id,
                        result.run_id,
                        claim.product_id,
                        claim.field_path,
                        _json(claim),
                    )
                    for claim in result.claims
                ],
            )
            connection.executemany(
                "INSERT OR REPLACE INTO observations VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        observation.observation_id,
                        result.run_id,
                        observation.source_id,
                        observation.field_path,
                        _json(observation),
                    )
                    for observation in result.observations
                ],
            )
            connection.executemany(
                "INSERT OR REPLACE INTO configurations VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        candidate.configuration_id,
                        result.run_id,
                        candidate.product_id,
                        candidate.rank,
                        _json(candidate),
                    )
                    for candidate in result.configurations
                ],
            )
            connection.executemany(
                "INSERT OR REPLACE INTO graph_edges VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        item["edge_id"],
                        result.run_id,
                        item["source_node"],
                        item["relation"],
                        item["target_node"],
                        _json(item.get("properties", {})),
                    )
                    for item in result.graph_edges
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
        payload = json.loads(row["payload_json"])
        # Migrate version 1.0 payloads that contained the removed thermal-screening fields.
        payload.pop("simulations", None)
        request = payload.get("request")
        if isinstance(request, dict):
            request.pop("simulate_top_configurations", None)
        payload.setdefault("observations", [])
        payload.setdefault("crawl_log", [])
        payload.setdefault("recognition_log", [])
        return ResearchResult.model_validate(payload)


    def load_products(self, exclude_run_id: str | None = None) -> list[ProductRecord]:
        """Load the accumulated structured product corpus for cross-run similarity."""

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

