"""End-to-end orchestration for evidence collection and engineering inference."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .configurations import ConfigurationGenerator
from .crawler import CatalogueCrawler, FetchError, HttpFetcher, HybridBrowserFetcher
from .entities import ProductEntityResolver
from .exporter import export_excel
from .extraction import ProductExtractor, claims_from_product
from .graph import KnowledgeGraph
from .inference import BayesianCandidateRanker
from .llm import LLMError, build_llm_provider
from .materials import MaterialLibrary
from .models import (
    CatalogueCoverage,
    CompanyResearchRequest,
    EvidenceObservation,
    ProductRecord,
    ResearchResult,
    SourceKind,
    SourceRecord,
    stable_id,
)
from .settings import Settings
from .search import SearchError, build_search_provider
from .similarity import ProductSimilarityIndex
from .storage import SQLiteRepository


class MattressIntelligencePipeline:
    """Evidence-first service whose decision algorithms do not depend on an LLM."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.settings.ensure_directories()
        self.materials = MaterialLibrary.load()
        self.llm = build_llm_provider(
            self.settings.llm_provider,
            self.settings.gemini_api_key,
            self.settings.gemini_model,
            self.settings.search_queries,
            openai_api_key=self.settings.openai_api_key,
            openai_model=self.settings.openai_model,
            timeout_seconds=self.settings.request_timeout_seconds,
        )
        self.search_provider = build_search_provider(
            self.settings.search_provider,
            self.settings.tavily_api_key,
            self.settings.search_queries,
            self.llm,
        )
        self.repository = SQLiteRepository(self.settings.database_path)

    def research(
        self,
        request: CompanyResearchRequest,
        output_path: Path | None = None,
        *,
        analyze: bool = True,
    ) -> ResearchResult:
        started_at = datetime.now(timezone.utc)
        run_id = stable_id("run", request.company_id, started_at.isoformat())
        warnings: list[str] = []
        search_urls: list[str] = []

        if request.use_search_grounding:
            if self.search_provider.name == "none":
                warnings.append("Web-search discovery requested but no search provider is configured.")
            else:
                try:
                    search_urls = self.search_provider.discover_urls(
                        request.company_name,
                        request.official_domain,
                        request.market,
                        request.brand_aliases,
                        request.custom_search_queries,
                    )
                except (LLMError, SearchError) as exc:
                    warnings.append(f"Source discovery failed: {exc}")

        fetcher_class = HybridBrowserFetcher if self.settings.render_javascript else HttpFetcher
        fetcher = fetcher_class(self.settings, respect_robots_txt=request.respect_robots_txt)
        crawler = CatalogueCrawler(fetcher)
        try:
            collection_urls = list(dict.fromkeys([*request.seed_urls, *search_urls]))
            report = crawler.crawl(
                request.official_domain,
                request.max_pages,
                collection_urls,
                max_depth=request.max_crawl_depth,
            )
            if request.include_external_evidence:
                remaining = request.max_external_pages
                for url in collection_urls:
                    if remaining <= 0:
                        break
                    if any(document.url == url for document in report.documents):
                        continue
                    report.discovered_urls.add(url)
                    try:
                        document = fetcher.fetch(url)
                        report.documents.append(document)
                        remaining -= 1
                        report.crawl_log.append(
                            {
                                "stage": "external",
                                "action": "fetched",
                                "url": document.url,
                                "status": document.status,
                                "content_type": document.content_type,
                                "bytes": len(document.body),
                                "artifact_path": document.artifact_path,
                                "reason": "exact external search/seed URL; no recursive external crawl",
                            }
                        )
                    except FetchError as exc:
                        report.failed_urls[url] = str(exc)
                        report.crawl_log.append(
                            {
                                "stage": "external",
                                "action": "failed",
                                "url": url,
                                "reason": str(exc),
                            }
                        )
        finally:
            fetcher.close()

        extractor = ProductExtractor(
            self.materials,
            self.llm,
            recognition_threshold=self.settings.product_recognition_threshold,
        )
        raw_products: list[ProductRecord] = []
        sources: list[SourceRecord] = []
        observations: list[EvidenceObservation] = []
        for document in report.documents:
            extracted_products, source, document_observations = extractor.extract_document(
                document, request
            )
            sources.append(source)
            raw_products.extend(extracted_products)
            observations.extend(document_observations)
        warnings.extend(extractor.warnings)
        if self.llm.name == "none":
            warnings.append(
                "Deterministic-only extraction active: JSON-LD, metadata, HTML tables, "
                "regular expressions, material dictionaries, PDF text, and entity resolution were used."
            )
        products = ProductEntityResolver().resolve(raw_products)

        return self._analyze_and_export(
            request=request,
            run_id=run_id,
            started_at=started_at,
            products=products,
            sources=sources,
            observations=observations,
            warnings=warnings,
            discovered_urls=len(report.discovered_urls),
            fetched_urls=len(report.documents),
            failed_urls=len(report.failed_urls) + len(report.blocked_urls),
            sitemap_count=len(report.sitemap_urls),
            output_path=output_path,
            discovery_log=list(getattr(self.search_provider, "discovery_log", []) or []),
            crawl_log=report.crawl_log,
            recognition_log=extractor.recognition_log,
            run_analysis=analyze,
        )

    def collect(
        self,
        request: CompanyResearchRequest,
        output_path: Path | None = None,
    ) -> ResearchResult:
        """Capture and structure evidence without generating hidden configurations."""

        return self.research(request, output_path, analyze=False)

    def import_catalogue(
        self,
        request: CompanyResearchRequest,
        input_path: Path,
        output_path: Path | None = None,
        *,
        analyze: bool = True,
    ) -> ResearchResult:
        started_at = datetime.now(timezone.utc)
        run_id = stable_id("run", request.company_id, started_at.isoformat())
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        raw_products = payload.get("products", payload) if isinstance(payload, dict) else payload
        if not isinstance(raw_products, list):
            raise ValueError("Imported JSON must be a list or an object containing a products list.")
        products: list[ProductRecord] = []
        sources: list[SourceRecord] = []
        for index, raw in enumerate(raw_products, start=1):
            product_payload = dict(raw)
            product_payload.update(
                company_id=request.company_id,
                company_name=request.company_name,
                brand=product_payload.get("brand") or request.company_name,
                extraction_method="imported",
            )
            product = ProductRecord.model_validate(product_payload)
            source_url = product.canonical_url or f"import://{input_path.name}/product/{index}"
            source_id = stable_id("src", source_url, input_path)
            source = SourceRecord(
                source_id=source_id,
                company_id=request.company_id,
                url=source_url,
                title=product.name,
                kind=SourceKind.OFFICIAL_PRODUCT,
                is_official=True,
                reliability=0.95,
                content_sha256=stable_id("sha", input_path, index).removeprefix("sha_"),
                artifact_path=str(input_path),
                content_type="application/json",
            )
            product.source_ids = list(dict.fromkeys(product.source_ids + [source_id]))
            for layer in product.layers:
                if not layer.evidence:
                    from .models import EvidenceRef

                    layer.evidence.append(EvidenceRef(source_id=source_id, reliability=0.95))
            products.append(product)
            sources.append(source)
        products = ProductEntityResolver().resolve(products)
        return self._analyze_and_export(
            request=request,
            run_id=run_id,
            started_at=started_at,
            products=products,
            sources=sources,
            observations=[],
            warnings=["Catalogue was imported from structured JSON; web coverage was not measured."],
            discovered_urls=len(sources),
            fetched_urls=len(sources),
            failed_urls=0,
            sitemap_count=0,
            output_path=output_path,
            discovery_log=[],
            crawl_log=[],
            recognition_log=[],
            run_analysis=analyze,
        )

    def _analyze_and_export(
        self,
        *,
        request: CompanyResearchRequest,
        run_id: str,
        started_at: datetime,
        products: list[ProductRecord],
        sources: list[SourceRecord],
        observations: list[EvidenceObservation],
        warnings: list[str],
        discovered_urls: int,
        fetched_urls: int,
        failed_urls: int,
        sitemap_count: int,
        output_path: Path | None,
        discovery_log: list[dict],
        crawl_log: list[dict],
        recognition_log: list[dict],
        run_analysis: bool,
    ) -> ResearchResult:
        claims = [claim for product in products for claim in claims_from_product(product)]
        configurations = []
        similarities: list[dict] = []

        if run_analysis:
            historical_products = self.repository.load_products()
            reference_by_id = {
                str(product.product_id): product for product in historical_products
            }
            reference_by_id.update({str(product.product_id): product for product in products})
            reference_products = list(reference_by_id.values())
            similarity_index = ProductSimilarityIndex(reference_products)
            current_ids = {str(product.product_id) for product in products}
            generator = ConfigurationGenerator(self.materials)
            ranker = BayesianCandidateRanker(self.materials)

            for product in products:
                neighbors = similarity_index.nearest(product)
                for neighbor_rank, neighbor in enumerate(neighbors, start=1):
                    similarities.append(
                        {
                            "product_id": product.product_id,
                            "product": product.name,
                            "neighbor_rank": neighbor_rank,
                            "similar_product_id": neighbor.product_id,
                            "similar_product": neighbor.name,
                            "cosine_similarity": neighbor.score,
                            "density_evidence": neighbor.density_evidence,
                            "reference_scope": (
                                "current_run"
                                if neighbor.product_id in current_ids
                                else "historical_corpus"
                            ),
                        }
                    )
                generation = generator.generate(
                    product,
                    max_candidates=request.max_configurations_per_product,
                )
                warnings.extend(f"{product.name}: {warning}" for warning in generation.warnings)
                ranked = ranker.rank(
                    product,
                    generation.candidates,
                    neighbors=neighbors,
                    limit=request.max_configurations_per_product,
                )
                configurations.extend(ranked)
        else:
            warnings.append(
                "Collection-only run: similarity, constraints, Bayesian ranking, and "
                "configuration generation were skipped."
            )

        graph = KnowledgeGraph.build(
            products,
            sources,
            claims,
            configurations,
            observations=observations,
            similarity_matches=similarities,
        )
        official_ratio = (
            sum(1 for source in sources if source.is_official) / len(sources) if sources else 0.0
        )
        reachable = fetched_urls + failed_urls
        retrieval_ratio = fetched_urls / reachable if reachable else 0.0
        sitemap_bonus = 1.0 if sitemap_count else 0.5
        estimated_coverage = 100.0 * (
            0.55 * retrieval_ratio + 0.25 * official_ratio + 0.20 * sitemap_bonus
        )
        limitations = []
        if sitemap_count == 0:
            limitations.append("No product sitemap was discovered; catalogue completeness is uncertain.")
        if failed_urls:
            limitations.append(f"{failed_urls} URL(s) failed or were blocked.")
        limitations.append("Unpublished, regional, or discontinued products may be absent.")
        coverage = CatalogueCoverage(
            discovered_urls=discovered_urls,
            fetched_urls=fetched_urls,
            failed_urls=failed_urls,
            product_pages=len(products),
            unique_products=len(products),
            variants=sum(len(product.variants) for product in products),
            official_source_ratio=official_ratio,
            estimated_coverage_percent=round(min(100.0, estimated_coverage), 2),
            limitations=limitations,
        )
        completed_at = datetime.now(timezone.utc)
        destination = output_path or self.settings.output_dir / f"{run_id}.xlsx"
        result = ResearchResult(
            run_id=run_id,
            request=request,
            started_at=started_at,
            completed_at=completed_at,
            products=products,
            sources=sources,
            claims=claims,
            observations=observations,
            configurations=configurations,
            similarity_matches=similarities,
            discovery_log=discovery_log,
            crawl_log=crawl_log,
            recognition_log=recognition_log,
            graph_edges=graph.edges,
            coverage=coverage,
            warnings=list(dict.fromkeys(warnings)),
            excel_path=str(destination),
        )
        export_excel(result, destination)
        self.repository.save(result)
        return result
