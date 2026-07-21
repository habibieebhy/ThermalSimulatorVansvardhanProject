"""Command-line interface."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

from .exporter import export_excel
from .models import CompanyResearchRequest
from .pipeline import MattressIntelligencePipeline
from .settings import Settings


LLM_CHOICES = ("none", "openai", "gemini")
SEARCH_CHOICES = ("none", "openai", "tavily", "gemini")


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
        max_pages=getattr(args, "max_pages", 100),
        max_external_pages=getattr(args, "max_external_pages", 25),
        max_crawl_depth=getattr(args, "max_depth", 4),
        max_configurations_per_product=getattr(args, "max_configurations", 10),
    )


def _pipeline(args: argparse.Namespace) -> MattressIntelligencePipeline:
    settings = Settings()
    provider = getattr(args, "llm", None)
    if provider:
        settings = replace(settings, llm_provider=provider)
    search_provider = getattr(args, "search_provider", None)
    if search_provider:
        settings = replace(settings, search_provider=search_provider)
    return MattressIntelligencePipeline(settings)


def _print_result(result) -> None:
    print()
    print("BRIXTA Mattress Intelligence run completed")
    print(f"Run ID:          {result.run_id}")
    print(f"Company:         {result.request.company_name}")
    print(f"Products:        {len(result.products)}")
    print(f"Variants:        {sum(len(product.variants) for product in result.products)}")
    print(f"Sources:         {len(result.sources)}")
    print(f"Observations:    {len(result.observations)}")
    print(f"Recognitions:    {len(result.recognition_log)}")
    print(f"Configurations:  {len(result.configurations)}")
    print(f"Coverage:        {result.coverage.estimated_coverage_percent:.1f}% (estimated)")
    print(f"Excel:           {result.excel_path}")
    if result.warnings:
        print(f"Warnings:        {len(result.warnings)} (see Run Metadata sheet)")


def _add_common_live_arguments(parser: argparse.ArgumentParser, *, default_pages: int) -> None:
    parser.add_argument("--company", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--market", default="India")
    parser.add_argument("--alias", action="append", default=[], help="Repeatable alias.")
    parser.add_argument("--seed-url", action="append", default=[], help="Repeatable known URL.")
    parser.add_argument("--query", action="append", default=[], help="Repeatable custom query.")
    parser.add_argument("--max-pages", type=int, default=default_pages)
    parser.add_argument("--max-external-pages", type=int, default=25)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument(
        "--search",
        action="store_true",
        help="Use the selected provider for URL discovery and source classification.",
    )
    parser.add_argument(
        "--external",
        action="store_true",
        help="Fetch exact external search/seed URLs without recursively crawling them.",
    )
    parser.add_argument(
        "--llm",
        choices=LLM_CHOICES,
        default="openai",
        help="Used only for search/recognition/extraction, never downstream analysis.",
    )
    parser.add_argument(
        "--search-provider",
        choices=SEARCH_CHOICES,
        default="openai",
    )
    parser.add_argument("--output", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mattress-lab",
        description=(
            "Evidence-first crawling with optional OpenAI search/product recognition, followed "
            "by deterministic graph, similarity, constraints, and Bayesian ranking."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run the offline deterministic demo.")
    demo.add_argument("--output", type=Path, default=Path("outputs/demo_mattress_intelligence.xlsx"))
    demo.add_argument("--input", type=Path, default=Path("examples/demo_catalogue.json"))
    demo.add_argument("--llm", choices=("none",), default="none")
    demo.add_argument("--search-provider", choices=("none",), default="none")

    research = subparsers.add_parser(
        "research",
        help="Collect evidence, then run similarity/constraint/Bayesian algorithms.",
    )
    _add_common_live_arguments(research, default_pages=100)
    research.add_argument("--max-configurations", type=int, default=10)

    collect = subparsers.add_parser(
        "collect",
        help="Collect and recognize products without hidden-value inference.",
    )
    _add_common_live_arguments(collect, default_pages=100)

    imported = subparsers.add_parser("import-json", help="Import structured catalogue JSON.")
    imported.add_argument("--company", required=True)
    imported.add_argument("--domain", required=True)
    imported.add_argument("--market", default="India")
    imported.add_argument("--input", required=True, type=Path)
    imported.add_argument("--output", type=Path)
    imported.add_argument("--max-configurations", type=int, default=10)
    imported.add_argument("--llm", choices=("none",), default="none")
    imported.add_argument("--search-provider", choices=("none",), default="none")
    imported.add_argument("--collection-only", action="store_true")

    for command, help_text in (
        ("runs", "List persisted runs."),
        ("export", "Regenerate Excel for a persisted run."),
    ):
        sub = subparsers.add_parser(command, help=help_text)
        if command == "export":
            sub.add_argument("--run-id", required=True)
            sub.add_argument("--output", required=True, type=Path)
        sub.add_argument("--llm", choices=("none",), default="none")
        sub.add_argument("--search-provider", choices=("none",), default="none")

    doctor = subparsers.add_parser("doctor", help="Validate configuration and integrations.")
    doctor.add_argument("--llm", choices=LLM_CHOICES, default=None)
    doctor.add_argument("--search-provider", choices=SEARCH_CHOICES, default=None)

    openai = subparsers.add_parser("openai-check", help="Verify OpenAI key/model access.")
    openai.add_argument("--llm", choices=("openai",), default="openai")
    openai.add_argument("--search-provider", choices=("none",), default="none")

    gemini = subparsers.add_parser("gemini-check", help="Verify Gemini key/model access.")
    gemini.add_argument("--llm", choices=("gemini",), default="gemini")
    gemini.add_argument("--search-provider", choices=("none",), default="none")

    tavily = subparsers.add_parser("tavily-check", help="Verify Tavily key and usage.")
    tavily.add_argument("--llm", choices=("none",), default="none")
    tavily.add_argument("--search-provider", choices=("tavily",), default="tavily")
    return parser


def main(argv: list[str] | None = None) -> int:
    _load_env()
    args = build_parser().parse_args(argv)
    try:
        pipeline = _pipeline(args)
        if args.command == "demo":
            request = CompanyResearchRequest(
                company_name="Sleepwell Demo",
                official_domain="https://example.invalid/sleepwell",
                market="India",
                max_configurations_per_product=10,
            )
            _print_result(pipeline.import_catalogue(request, args.input, args.output))
            return 0
        if args.command == "research":
            _print_result(pipeline.research(_request_from_args(args), args.output))
            return 0
        if args.command == "collect":
            result = pipeline.collect(_request_from_args(args), args.output)
            _print_result(result)
            print("Inference:       skipped")
            print(f"Search hits:     {len(result.discovery_log)}")
            print(f"Crawl events:    {len(result.crawl_log)}")
            return 0
        if args.command == "import-json":
            _print_result(
                pipeline.import_catalogue(
                    _request_from_args(args),
                    args.input,
                    args.output,
                    analyze=not args.collection_only,
                )
            )
            return 0
        if args.command == "runs":
            runs = pipeline.repository.list_runs()
            print("No persisted research runs." if not runs else "")
            for item in runs:
                print(f"{item['run_id']}  {item['company_name']}  {item['completed_at']}")
            return 0
        if args.command == "export":
            export_excel(pipeline.repository.load(args.run_id), args.output)
            print(args.output)
            return 0
        if args.command == "doctor":
            print("Core configuration: OK")
            print(f"Database: {pipeline.settings.database_path}")
            print(f"Recognition provider: {pipeline.llm.name}")
            print(f"Search provider: {pipeline.search_provider.name}")
            print("LLM downstream analysis: disabled by architecture")
            print("Deterministic graph/similarity/constraints/Bayesian ranking: enabled")
            return 0
        if args.command in {"openai-check", "gemini-check"}:
            details = pipeline.llm.check_connection()
            print(f"{pipeline.llm.name.title()} connection: OK")
            for key, value in details.items():
                print(f"{key}: {value}")
            return 0
        if args.command == "tavily-check":
            details = pipeline.search_provider.check_connection()
            print("Tavily connection: OK")
            print(details)
            return 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
