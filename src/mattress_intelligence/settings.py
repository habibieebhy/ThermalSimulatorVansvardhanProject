"""Environment-backed runtime settings for acquisition, storage, and workers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
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
    """Application configuration with explicit local fallbacks."""

    data_dir: Path = field(default_factory=lambda: Path(os.getenv("MATTRESS_INTEL_DATA_DIR", "data")))
    output_dir: Path = field(default_factory=lambda: Path(os.getenv("MATTRESS_INTEL_OUTPUT_DIR", "outputs")))
    artifact_dir: Path = field(default_factory=lambda: Path(os.getenv("MATTRESS_INTEL_ARTIFACT_DIR", "artifacts")))
    database_path: Path = field(
        default_factory=lambda: Path(os.getenv("MATTRESS_INTEL_DATABASE_PATH", "data/mattress_intelligence.sqlite3"))
    )
    database_url: str | None = field(default_factory=lambda: os.getenv("DATABASE_URL") or None)
    database_direct_url: str | None = field(default_factory=lambda: os.getenv("DATABASE_DIRECT_URL") or None)

    # One-click UI defaults. These remain environment-tunable without exposing controls in the UI.
    default_market: str = field(
        default_factory=lambda: os.getenv("MATTRESS_INTEL_DEFAULT_MARKET", "India").strip() or "India"
    )
    ui_max_pages: int = field(
        default_factory=lambda: _int_env("MATTRESS_INTEL_UI_MAX_PAGES", 30)
    )
    ui_max_external_pages: int = field(
        default_factory=lambda: _int_env("MATTRESS_INTEL_UI_MAX_EXTERNAL_PAGES", 5)
    )
    ui_max_crawl_depth: int = field(
        default_factory=lambda: _int_env("MATTRESS_INTEL_UI_MAX_CRAWL_DEPTH", 2)
    )
    ui_max_assets_per_document: int = field(
        default_factory=lambda: _int_env("MATTRESS_INTEL_UI_MAX_ASSETS_PER_DOCUMENT", 8)
    )
    ui_max_vision_assets: int = field(
        default_factory=lambda: _int_env("MATTRESS_INTEL_UI_MAX_VISION_ASSETS", 8)
    )
    ui_max_pdf_pages: int = field(
        default_factory=lambda: _int_env("MATTRESS_INTEL_UI_MAX_PDF_PAGES", 40)
    )
    ui_max_configurations_per_product: int = field(
        default_factory=lambda: _int_env("MATTRESS_INTEL_UI_MAX_CONFIGURATIONS", 8)
    )

    # GPT is limited to source/document/image recognition and explicit transcription.
    llm_provider: str = field(
        default_factory=lambda: os.getenv("MATTRESS_INTEL_LLM_PROVIDER", "openai").strip().lower()
    )
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY") or None)
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-5-nano"))
    gemini_api_key: str | None = field(default_factory=lambda: os.getenv("GEMINI_API_KEY") or None)
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"))
    product_recognition_threshold: float = field(
        default_factory=lambda: _float_env("MATTRESS_INTEL_PRODUCT_RECOGNITION_THRESHOLD", 0.68)
    )
    vision_recognition_threshold: float = field(
        default_factory=lambda: _float_env("MATTRESS_INTEL_VISION_RECOGNITION_THRESHOLD", 0.62)
    )

    # Discovery defaults to Jina, with Firecrawl fallback.
    search_provider: str = field(
        default_factory=lambda: os.getenv("MATTRESS_INTEL_SEARCH_PROVIDER", "services").strip().lower()
    )
    search_queries: int = field(default_factory=lambda: _int_env("MATTRESS_INTEL_SEARCH_QUERIES", 6))
    search_results_per_query: int = field(
        default_factory=lambda: _int_env("MATTRESS_INTEL_SEARCH_RESULTS_PER_QUERY", 10)
    )
    tavily_api_key: str | None = field(default_factory=lambda: os.getenv("TAVILY_API_KEY") or None)
    jina_api_key: str | None = field(default_factory=lambda: os.getenv("JINA_API_KEY") or None)
    jina_reader_enabled: bool = field(
        default_factory=lambda: _bool_env("MATTRESS_INTEL_JINA_READER_ENABLED", True)
    )
    jina_reader_on_thin_page: bool = field(
        default_factory=lambda: _bool_env("MATTRESS_INTEL_JINA_READER_ON_THIN_PAGE", True)
    )
    jina_reader_min_characters: int = field(
        default_factory=lambda: _int_env("MATTRESS_INTEL_JINA_READER_MIN_CHARACTERS", 800)
    )
    jina_timeout_seconds: float = field(
        default_factory=lambda: _float_env("MATTRESS_INTEL_JINA_TIMEOUT_SECONDS", 60.0)
    )

    firecrawl_api_key: str | None = field(default_factory=lambda: os.getenv("FIRECRAWL_API_KEY") or None)
    firecrawl_enabled: bool = field(
        default_factory=lambda: _bool_env("MATTRESS_INTEL_FIRECRAWL_ENABLED", True)
    )
    firecrawl_timeout_seconds: float = field(
        default_factory=lambda: _float_env("MATTRESS_INTEL_FIRECRAWL_TIMEOUT_SECONDS", 60.0)
    )
    firecrawl_wait_ms: int = field(
        default_factory=lambda: _int_env("MATTRESS_INTEL_FIRECRAWL_WAIT_MS", 1500)
    )

    # services_first = Firecrawl -> Jina Reader -> local crawler/Playwright.
    # local_first = local crawler/Playwright -> Jina Reader -> Firecrawl.
    capture_strategy: str = field(
        default_factory=lambda: os.getenv("MATTRESS_INTEL_CAPTURE_STRATEGY", "services_first").strip().lower()
    )

    discover_assets: bool = field(default_factory=lambda: _bool_env("MATTRESS_INTEL_DISCOVER_ASSETS", True))
    max_assets_per_document: int = field(
        default_factory=lambda: _int_env("MATTRESS_INTEL_MAX_ASSETS_PER_DOCUMENT", 30)
    )
    max_vision_assets_per_run: int = field(
        default_factory=lambda: _int_env("MATTRESS_INTEL_MAX_VISION_ASSETS_PER_RUN", 80)
    )
    max_pdf_pages_per_document: int = field(
        default_factory=lambda: _int_env("MATTRESS_INTEL_MAX_PDF_PAGES", 100)
    )
    minimum_asset_score: float = field(
        default_factory=lambda: _float_env("MATTRESS_INTEL_MINIMUM_ASSET_SCORE", 0.28)
    )
    maximum_image_bytes: int = field(
        default_factory=lambda: _int_env("MATTRESS_INTEL_MAX_IMAGE_BYTES", 12_000_000)
    )

    # Local OCR happens before GPT vision. Missing Tesseract degrades gracefully.
    local_ocr_enabled: bool = field(
        default_factory=lambda: _bool_env("MATTRESS_INTEL_LOCAL_OCR_ENABLED", True)
    )
    tesseract_cmd: str | None = field(default_factory=lambda: os.getenv("TESSERACT_CMD") or None)
    ocr_min_characters: int = field(
        default_factory=lambda: _int_env("MATTRESS_INTEL_OCR_MIN_CHARACTERS", 24)
    )

    # S3-compatible object storage. Local artifacts remain a cache/fallback.
    minio_endpoint: str | None = field(default_factory=lambda: os.getenv("MINIO_ENDPOINT") or None)
    minio_access_key: str | None = field(default_factory=lambda: os.getenv("MINIO_ACCESS_KEY") or None)
    minio_secret_key: str | None = field(default_factory=lambda: os.getenv("MINIO_SECRET_KEY") or None)
    minio_bucket: str = field(default_factory=lambda: os.getenv("MINIO_BUCKET", "mattress-intelligence"))
    minio_secure: bool = field(default_factory=lambda: _bool_env("MINIO_SECURE", False))
    minio_region: str | None = field(default_factory=lambda: os.getenv("MINIO_REGION") or None)

    # Redis is the Celery broker/backend; task payloads stay small, artifacts go to MinIO.
    celery_enabled: bool = field(default_factory=lambda: _bool_env("CELERY_ENABLED", False))
    celery_always_eager: bool = field(default_factory=lambda: _bool_env("CELERY_ALWAYS_EAGER", False))
    celery_broker_url: str = field(
        default_factory=lambda: os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    )
    celery_result_backend: str = field(
        default_factory=lambda: os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
    )
    celery_task_time_limit_seconds: int = field(
        default_factory=lambda: _int_env("CELERY_TASK_TIME_LIMIT_SECONDS", 7200)
    )
    celery_wait_timeout_seconds: int = field(
        default_factory=lambda: _int_env("CELERY_WAIT_TIMEOUT_SECONDS", 7200)
    )

    user_agent: str = field(
        default_factory=lambda: os.getenv(
            "MATTRESS_INTEL_USER_AGENT", "BRIXTA-Mattress-Intelligence/1.3 (+evidence-research)"
        )
    )
    request_timeout_seconds: float = field(
        default_factory=lambda: _float_env("MATTRESS_INTEL_REQUEST_TIMEOUT_SECONDS", 30.0)
    )
    request_delay_seconds: float = field(
        default_factory=lambda: _float_env("MATTRESS_INTEL_REQUEST_DELAY_SECONDS", 1.0)
    )
    max_download_bytes: int = field(
        default_factory=lambda: _int_env("MATTRESS_INTEL_MAX_DOWNLOAD_BYTES", 25_000_000)
    )
    render_javascript: bool = field(
        default_factory=lambda: _bool_env("MATTRESS_INTEL_RENDER_JAVASCRIPT", True)
    )

    @property
    def object_storage_enabled(self) -> bool:
        return bool(self.minio_endpoint and self.minio_access_key and self.minio_secret_key)

    @property
    def postgres_enabled(self) -> bool:
        return bool(self.database_url)

    def ensure_directories(self) -> None:
        for directory in (self.data_dir, self.output_dir, self.artifact_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
