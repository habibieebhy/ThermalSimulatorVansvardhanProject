"""Command-line interface for local and Celery-backed runs."""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

from .exporter import export_excel
from .firecrawl import FirecrawlClient
from .jina import JinaReaderClient, JinaSearchClient
from .models import CompanyResearchRequest
from .object_store import build_object_store
from .pipeline import MattressIntelligencePipeline
from .settings import Settings
from .storage import build_repository


LLM_CHOICES = ("none", "openai", "gemini")
SEARCH_CHOICES = ("none", "services", "jina", "firecrawl", "openai", "tavily", "gemini")


def _load_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _request_from_args(args: argparse.Namespace) -> CompanyResearchRequest:
    return CompanyResearchRequest(
        company_name=args.company,
        official_domain=args.domain,
        market=args.market,
        brand_aliases=getattr(args, "alias", []) or [],
        seed_urls=getattr(args, "seed_url", []) or [],
        custom_search_queries=getattr(args, "query", []) or [],
        include_external_evidence=getattr(args, "external", False),
        use_search_grounding=getattr(args, "search", False),
        discover_assets=not getattr(args, "no_assets", False),
        analyze_assets_with_vision=not getattr(args, "no_vision", False),
        max_pages=getattr(args, "max_pages", 100),
        max_external_pages=getattr(args, "max_external_pages", 25),
        max_crawl_depth=getattr(args, "max_depth", 4),
        max_assets_per_document=getattr(args, "max_assets_per_document", 30),
        max_vision_assets=getattr(args, "max_vision_assets", 80),
        max_pdf_pages=getattr(args, "max_pdf_pages", 100),
        max_configurations_per_product=getattr(args, "max_configurations", 10),
    )


def _settings(args: argparse.Namespace) -> Settings:
    settings = Settings()
    if getattr(args, "llm", None):
        settings = replace(settings, llm_provider=args.llm)
    if getattr(args, "search_provider", None):
        settings = replace(settings, search_provider=args.search_provider)
    if getattr(args, "capture_strategy", None):
        settings = replace(settings, capture_strategy=args.capture_strategy)
    return settings


def _pipeline(args: argparse.Namespace) -> MattressIntelligencePipeline:
    return MattressIntelligencePipeline(_settings(args))


def _print_result(result) -> None:
    print("\nBRIXTA Mattress Intelligence run completed")
    print(f"Run ID:          {result.run_id}")
    print(f"Company:         {result.request.company_name}")
    print(f"Products:        {len(result.products)}")
    print(f"Variants:        {sum(len(product.variants) for product in result.products)}")
    print(f"Sources:         {len(result.sources)}")
    print(f"Assets:          {len(result.assets)}")
    print(f"Observations:    {len(result.observations)}")
    print(f"Configurations:  {len(result.configurations)}")
    print(f"Coverage:        {result.coverage.estimated_coverage_percent:.1f}% (estimated)")
    print(f"Excel:           {result.excel_path}")
    if result.warnings:
        print(f"Warnings:        {len(result.warnings)} (see Run Metadata sheet)")


def _add_common_live_arguments(parser: argparse.ArgumentParser, *, default_pages: int) -> None:
    parser.add_argument("--company", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--market", default="India")
    parser.add_argument("--alias", action="append", default=[])
    parser.add_argument("--seed-url", action="append", default=[])
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--max-pages", type=int, default=default_pages)
    parser.add_argument("--max-external-pages", type=int, default=25)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-assets-per-document", type=int, default=30)
    parser.add_argument("--max-vision-assets", type=int, default=80)
    parser.add_argument("--max-pdf-pages", type=int, default=100)
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--external", action="store_true")
    parser.add_argument("--no-assets", action="store_true")
    parser.add_argument("--no-vision", action="store_true")
    parser.add_argument("--enqueue", action="store_true", help="Submit to Celery instead of running locally.")
    parser.add_argument("--llm", choices=LLM_CHOICES, default="openai")
    parser.add_argument("--search-provider", choices=SEARCH_CHOICES, default="services")
    parser.add_argument("--capture-strategy", choices=("services_first", "local_first"), default=None)
    parser.add_argument("--output", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mattress-lab",
        description=(
            "Jina/Firecrawl acquisition, Playwright fallback, local OCR and GPT transcription, "
            "followed by deterministic graph, similarity, constraints, and Bayesian ranking."
        ),
    )
    subs = parser.add_subparsers(dest="command", required=True)

    demo = subs.add_parser("demo")
    demo.add_argument("--output", type=Path, default=Path("outputs/demo_mattress_intelligence.xlsx"))
    demo.add_argument("--input", type=Path, default=Path("examples/demo_catalogue.json"))
    demo.add_argument("--llm", choices=("none",), default="none")
    demo.add_argument("--search-provider", choices=("none",), default="none")

    research = subs.add_parser("research")
    _add_common_live_arguments(research, default_pages=100)
    research.add_argument("--max-configurations", type=int, default=10)

    collect = subs.add_parser("collect")
    _add_common_live_arguments(collect, default_pages=100)

    imported = subs.add_parser("import-json")
    imported.add_argument("--company", required=True)
    imported.add_argument("--domain", required=True)
    imported.add_argument("--market", default="India")
    imported.add_argument("--input", required=True, type=Path)
    imported.add_argument("--output", type=Path)
    imported.add_argument("--max-configurations", type=int, default=10)
    imported.add_argument("--llm", choices=("none",), default="none")
    imported.add_argument("--search-provider", choices=("none",), default="none")
    imported.add_argument("--collection-only", action="store_true")

    subs.add_parser("runs")
    export = subs.add_parser("export")
    export.add_argument("--run-id", required=True)
    export.add_argument("--output", required=True, type=Path)

    for name in ("doctor", "openai-check", "jina-check", "firecrawl-check", "database-check", "storage-check", "worker-check"):
        sub = subs.add_parser(name)
        sub.add_argument("--llm", choices=LLM_CHOICES, default=None)
        sub.add_argument("--search-provider", choices=SEARCH_CHOICES, default=None)

    status = subs.add_parser("job-status")
    status.add_argument("--task-id", required=True)
    return parser


def _enqueue(command: str, request: CompanyResearchRequest, output: Path | None) -> int:
    from .celery_app import celery_app
    from .tasks import enqueue_collection, enqueue_research

    # --enqueue must always mean broker-backed execution, even when an old .env still contains
    # CELERY_ALWAYS_EAGER=true.
    celery_app.conf.task_always_eager = False
    task = enqueue_collection(request, output) if command == "collect" else enqueue_research(request, output)
    print(f"Task ID: {task.id}", flush=True)
    print("State:   submitted", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    _load_env()
    args = build_parser().parse_args(argv)
    settings = _settings(args)
    try:
        if args.command == "job-status":
            from celery.result import AsyncResult
            from .celery_app import celery_app

            result = AsyncResult(args.task_id, app=celery_app)
            print(f"Task ID: {args.task_id}")
            print(f"State:   {result.state}")
            if result.ready():
                print(result.result)
            elif result.info:
                print(result.info)
            return 0

        if args.command == "jina-check":
            print(JinaSearchClient(settings.jina_api_key, settings.jina_timeout_seconds).check_connection())
            print(JinaReaderClient(settings.jina_api_key, settings.jina_timeout_seconds).read("https://example.com").url)
            return 0
        if args.command == "firecrawl-check":
            if not settings.firecrawl_api_key:
                raise ValueError("FIRECRAWL_API_KEY is not configured.")
            print(FirecrawlClient(settings.firecrawl_api_key, timeout_seconds=settings.firecrawl_timeout_seconds).check_connection())
            return 0
        if args.command == "database-check":
            print(build_repository(settings).check_connection())
            return 0
        if args.command == "storage-check":
            store = build_object_store(settings)
            stored = store.put_bytes(b"BRIXTA storage check", content_type="text/plain", source_url="urn:brixta:check", namespace="checks")
            print({"local_path": stored.local_path, "object_uri": stored.object_uri, "sha256": stored.sha256})
            return 0
        if args.command == "worker-check":
            from .celery_app import celery_app

            replies = celery_app.control.ping(timeout=3.0)
            print(replies or "No Celery workers responded.")
            return 0

        pipeline = MattressIntelligencePipeline(settings)
        if args.command == "doctor":
            print("Core configuration: OK")
            print(f"Database: {'Postgres/Neon' if settings.postgres_enabled else settings.database_path}")
            print(f"Object storage: {'MinIO' if settings.object_storage_enabled else 'local artifacts'}")
            print(f"Recognition provider: {pipeline.llm.name}")
            print(f"Search provider: {pipeline.search_provider.name}")
            print(f"Capture strategy: {settings.capture_strategy}")
            print(f"Local OCR: {'enabled' if settings.local_ocr_enabled else 'disabled'}")
            print(f"Celery: {'enabled' if settings.celery_enabled else 'disabled'}")
            print("LLM downstream analysis: disabled by architecture")
            return 0
        if args.command == "openai-check":
            print(pipeline.llm.check_connection())
            return 0
        if args.command == "demo":
            request = CompanyResearchRequest(company_name="Sleepwell Demo", official_domain="https://example.invalid/sleepwell")
            _print_result(pipeline.import_catalogue(request, args.input, args.output))
            return 0
        if args.command in {"research", "collect"}:
            request = _request_from_args(args)
            if args.enqueue:
                if not settings.celery_enabled:
                    raise ValueError("--enqueue requires CELERY_ENABLED=true and a running Redis/Celery worker.")
                return _enqueue(args.command, request, args.output)
            result = pipeline.collect(request, args.output) if args.command == "collect" else pipeline.research(request, args.output)
            _print_result(result)
            if args.command == "collect":
                print("Inference:       skipped")
            return 0
        if args.command == "import-json":
            request = CompanyResearchRequest(
                company_name=args.company,
                official_domain=args.domain,
                market=args.market,
                max_configurations_per_product=args.max_configurations,
            )
            _print_result(pipeline.import_catalogue(request, args.input, args.output, analyze=not args.collection_only))
            return 0
        if args.command == "runs":
            runs = pipeline.repository.list_runs()
            if not runs:
                print("No persisted research runs.")
            for item in runs:
                print(f"{item['run_id']}  {item['company_name']}  {item['completed_at']}")
            return 0
        if args.command == "export":
            export_excel(pipeline.repository.load(args.run_id), args.output)
            print(args.output)
            return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
