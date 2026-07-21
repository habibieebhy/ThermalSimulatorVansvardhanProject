"""Environment-backed runtime settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # Source-tree tests can run before optional environment helpers are installed.
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw in (None, "") else float(raw)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw in (None, "") else int(raw)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    """Application configuration with conservative crawler and recognition defaults."""

    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("MATTRESS_INTEL_DATA_DIR", "data"))
    )
    output_dir: Path = field(
        default_factory=lambda: Path(os.getenv("MATTRESS_INTEL_OUTPUT_DIR", "outputs"))
    )
    artifact_dir: Path = field(
        default_factory=lambda: Path(os.getenv("MATTRESS_INTEL_ARTIFACT_DIR", "artifacts"))
    )
    database_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("MATTRESS_INTEL_DATABASE_PATH", "data/mattress_intelligence.sqlite3")
        )
    )

    # LLMs are restricted to source discovery, document recognition, and explicit extraction.
    llm_provider: str = field(
        default_factory=lambda: os.getenv("MATTRESS_INTEL_LLM_PROVIDER", "none").strip().lower()
    )
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY") or None)
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-5.4-nano")
    )
    gemini_api_key: str | None = field(default_factory=lambda: os.getenv("GEMINI_API_KEY") or None)
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    )

    # Retain the historical field name for source compatibility; it is provider-neutral now.
    gemini_search_queries: int = field(
        default_factory=lambda: _int_env(
            "MATTRESS_INTEL_SEARCH_QUERIES",
            _int_env("MATTRESS_INTEL_GEMINI_SEARCH_QUERIES", 6),
        )
    )
    product_recognition_threshold: float = field(
        default_factory=lambda: _float_env(
            "MATTRESS_INTEL_PRODUCT_RECOGNITION_THRESHOLD", 0.68
        )
    )

    search_provider: str = field(
        default_factory=lambda: os.getenv("MATTRESS_INTEL_SEARCH_PROVIDER", "none").strip().lower()
    )
    tavily_api_key: str | None = field(default_factory=lambda: os.getenv("TAVILY_API_KEY") or None)

    user_agent: str = field(
        default_factory=lambda: os.getenv(
            "MATTRESS_INTEL_USER_AGENT",
            "BRIXTA-Mattress-Intelligence/1.2 (+https://example.com/research-policy)",
        )
    )
    request_timeout_seconds: float = field(
        default_factory=lambda: _float_env("MATTRESS_INTEL_REQUEST_TIMEOUT_SECONDS", 30.0)
    )
    request_delay_seconds: float = field(
        default_factory=lambda: _float_env("MATTRESS_INTEL_REQUEST_DELAY_SECONDS", 1.0)
    )
    max_download_bytes: int = field(
        default_factory=lambda: _int_env("MATTRESS_INTEL_MAX_DOWNLOAD_BYTES", 15_000_000)
    )
    render_javascript: bool = field(
        default_factory=lambda: _bool_env("MATTRESS_INTEL_RENDER_JAVASCRIPT", False)
    )

    @property
    def search_queries(self) -> int:
        """Provider-neutral alias for the retained configuration field."""

        return self.gemini_search_queries

    def ensure_directories(self) -> None:
        for directory in (self.data_dir, self.output_dir, self.artifact_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
