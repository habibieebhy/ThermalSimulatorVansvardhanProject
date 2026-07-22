"""End-to-end orchestration for acquisition, evidence extraction, and deterministic inference."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable

from .assets import AssetPipeline
from .configurations import ConfigurationGenerator
from .crawler import (
    CatalogueCrawler,
    EvidenceFetcher,
    FetchError,
    HttpFetcher,
    HybridBrowserFetcher,
)
from .entities import ProductEntityResolver
from .exporter import export_excel
from .extraction import ProductExtractor, claims_from_product
from .graph import KnowledgeGraph
from .inference import BayesianCandidateRanker
from .firecrawl import FirecrawlClient
from .jina import JinaReaderClient
from .llm import LLMError, build_llm_provider
from .materials import MaterialLibrary
from .models import (
    AssetRecord,
    CatalogueCoverage,
    CompanyResearchRequest,
    EvidenceObservation,
    EvidenceRef,
    ProductRecord,
    ResearchResult,
    SourceKind,
    SourceRecord,
    stable_id,
)
from .search import SearchError, build_search_provider
from .settings import Settings
from .similarity import ProductSimilarityIndex
from .storage import build_repository


class MattressIntelligencePipeline:
    """LLMs transcribe explicit evidence; algorithms alone produce inference."""

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
            self.settings.jina_api_key,
            self.settings.firecrawl_api_key,
            self.settings.search_queries,
            self.settings.search_results_per_query,
            self.llm,
            timeout_seconds=self.settings.jina_timeout_seconds,
        )
        self.repository = build_repository(self.settings)

    def _build_fetchers(
        self, request: CompanyResearchRequest
    ) -> tuple[HttpFetcher, EvidenceFetcher]:
        fetcher_class = HybridBrowserFetcher if self.settings.render_javascript else HttpFetcher
        primary = fetcher_class(
            self.settings,
            respect_robots_txt=request.respect_robots_txt,
        )
        reader = None
        if self.settings.jina_reader_enabled:
            reader = JinaReaderClient(
                self.settings.jina_api_key,
                timeout_seconds=self.settings.jina_timeout_seconds,
            )
        firecrawl = None
        if self.settings.firecrawl_enabled and self.settings.firecrawl_api_key:
            firecrawl = FirecrawlClient(
                self.settings.firecrawl_api_key,
                timeout_seconds=self.settings.firecrawl_timeout_seconds,
                wait_ms=self.settings.firecrawl_wait_ms,
            )
        return primary, EvidenceFetcher(primary, self.settings, reader, firecrawl)

    def research(
        self,
        request: CompanyResearchRequest,
        output_path: Path | None = None,
        *,
        analyze: bool = True,
        progress_callback: Callable[..., None] | None = None,
    ) -> ResearchResult:
        started_at = datetime.now(timezone.utc)
        run_id = stable_id("run", request.company_id, started_at.isoformat())
        warnings: list[str] = []
        search_urls: list[str] = []

        def progress(stage: str, **metadata: object) -> None:
            if progress_callback is not None:
                progress_callback(stage, **metadata)

        progress("initializing")
        if request.use_search_grounding:
            progress("discovering")
            if self.search_provider.name == "none":
                warnings.append("Web discovery requested but no search provider is configured.")
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

        progress("crawling")
        primary, fetcher = self._build_fetchers(request)
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
                known_urls = {document.url for document in report.documents}
                for url in collection_urls:
                    if remaining <= 0:
                        break
                    if url in known_urls:
                        continue
                    report.discovered_urls.add(url)
                    try:
                        document = fetcher.fetch(url)
                        report.documents.append(document)
                        known_urls.add(document.url)
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
                                "object_uri": document.object_uri,
                                "capture_method": document.capture_method,
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

        progress("extracting", current=0, total=len(report.documents))
        extractor = ProductExtractor(
            self.materials,
            self.llm,
            recognition_threshold=self.settings.product_recognition_threshold,
        )
        raw_products: list[ProductRecord] = []
        sources: list[SourceRecord] = []
        observations: list[EvidenceObservation] = []
        for document_index, document in enumerate(report.documents, start=1):
            progress("extracting", current=document_index, total=len(report.documents))
            extracted_products, source, document_observations = extractor.extract_document(
                document, request
            )
            sources.append(source)
            raw_products.extend(extracted_products)
            observations.extend(document_observations)
            for network_item in document.network_manifest:
                if network_item.get("resource_type") not in {"xhr", "fetch"}:
                    continue
                endpoint = str(network_item.get("url") or "")
                excerpt = str(network_item.get("body_excerpt") or "")
                observations.append(
                    EvidenceObservation(
                        observation_id=stable_id("obs", source.source_id, "network_json", endpoint),
                        source_id=source.source_id,
                        company_id=request.company_id,
                        document_url=document.url,
                        field_path="network_json.endpoint",
                        value=endpoint,
                        method="network_json",
                        locator=str(network_item.get("object_uri") or network_item.get("artifact_path") or endpoint),
                        excerpt=excerpt[:1000] or None,
                        confidence=0.72 if excerpt else 0.55,
                    )
                )
                if excerpt:
                    for mention, material_id in self.materials.find_material_mentions(excerpt):
                        observations.append(
                            EvidenceObservation(
                                observation_id=stable_id("obs", source.source_id, endpoint, material_id),
                                source_id=source.source_id,
                                company_id=request.company_id,
                                document_url=document.url,
                                field_path="network_json.material_mention",
                                value=mention,
                                normalized_material=material_id,
                                method="network_json",
                                locator=endpoint,
                                excerpt=excerpt[:1000],
                                confidence=0.70,
                            )
                        )
        warnings.extend(extractor.warnings)

        assets: list[AssetRecord] = []
        acquisition_log: list[dict] = []
        if request.discover_assets and self.settings.discover_assets:
            progress("assets")
            asset_result = AssetPipeline(
                self.settings,
                primary.object_store,
                self.materials,
                self.llm,
            ).process(report.documents, sources, request)
            assets = asset_result.assets
            raw_products.extend(asset_result.products)
            observations.extend(asset_result.observations)
            acquisition_log.extend(asset_result.log)
            warnings.extend(asset_result.warnings)
        else:
            warnings.append("Asset discovery was disabled for this run.")

        if self.llm.name == "none":
            warnings.append(
                "Deterministic-only extraction active: JSON-LD, metadata, HTML tables, "
                "regular expressions, material dictionaries, PDF text, and entity resolution were used."
            )
        progress("resolving")
        products = ProductEntityResolver().resolve(raw_products)

        return self._analyze_and_export(
            request=request,
            run_id=run_id,
            started_at=started_at,
            products=products,
            sources=sources,
            assets=assets,
            observations=observations,
            warnings=warnings,
            discovered_urls=len(report.discovered_urls),
            fetched_urls=len(report.documents),
            failed_urls=len(report.failed_urls) + len(report.blocked_urls),
            sitemap_count=len(report.sitemap_urls),
            output_path=output_path,
            discovery_log=list(getattr(self.search_provider, "discovery_log", []) or []),
            crawl_log=report.crawl_log,
            acquisition_log=acquisition_log,
            recognition_log=extractor.recognition_log,
            run_analysis=analyze,
            progress_callback=progress_callback,
        )

    def collect(
        self,
        request: CompanyResearchRequest,
        output_path: Path | None = None,
        *,
        progress_callback: Callable[..., None] | None = None,
    ) -> ResearchResult:
        return self.research(
            request,
            output_path,
            analyze=False,
            progress_callback=progress_callback,
        )

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
            source_id = stable_id("src", run_id, index, input_path)
            source = SourceRecord(
                source_id=source_id,
                company_id=request.company_id,
                url=f"file://{input_path.resolve()}#product={index}",
                title=product.name,
                kind=SourceKind.OFFICIAL_CATALOGUE,
                is_official=True,
                reliability=0.95,
                content_sha256=stable_id("sha", input_path, index),
                artifact_path=str(input_path),
                capture_method="import_json",
                content_type="application/json",
            )
            product.source_ids = list(dict.fromkeys(product.source_ids + [source_id]))
            for layer in product.layers:
                if not layer.evidence:
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
            assets=[],
            observations=[],
            warnings=["Catalogue was imported from structured JSON; web coverage was not measured."],
            discovered_urls=len(sources),
            fetched_urls=len(sources),
            failed_urls=0,
            sitemap_count=0,
            output_path=output_path,
            discovery_log=[],
            crawl_log=[],
            acquisition_log=[],
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
        assets: list[AssetRecord],
        observations: list[EvidenceObservation],
        warnings: list[str],
        discovered_urls: int,
        fetched_urls: int,
        failed_urls: int,
        sitemap_count: int,
        output_path: Path | None,
        discovery_log: list[dict],
        crawl_log: list[dict],
        acquisition_log: list[dict],
        recognition_log: list[dict],
        run_analysis: bool,
        progress_callback: Callable[..., None] | None = None,
    ) -> ResearchResult:
        def progress(stage: str, **metadata: object) -> None:
            if progress_callback is not None:
                progress_callback(stage, **metadata)

        progress("analyzing")
        claims = [claim for product in products for claim in claims_from_product(product)]
        configurations = []
        similarities: list[dict] = []

        if run_analysis:
            historical_products = self.repository.load_products(exclude_run_id=run_id)
            reference_by_id = {str(product.product_id): product for product in historical_products}
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
                                "current_run" if neighbor.product_id in current_ids else "historical_corpus"
                            ),
                        }
                    )
                generation = generator.generate(
                    product,
                    max_candidates=request.max_configurations_per_product,
                )
                warnings.extend(f"{product.name}: {warning}" for warning in generation.warnings)
                configurations.extend(
                    ranker.rank(
                        product,
                        generation.candidates,
                        neighbors=neighbors,
                        limit=request.max_configurations_per_product,
                    )
                )
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
            assets=assets,
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
            assets=len(assets),
            vision_assets=sum(1 for asset in assets if asset.vision_payload is not None),
            official_source_ratio=official_ratio,
            estimated_coverage_percent=round(min(100.0, estimated_coverage), 2),
            limitations=limitations,
        )
        progress("exporting")
        completed_at = datetime.now(timezone.utc)
        destination = output_path or self.settings.output_dir / f"{run_id}.xlsx"
        result = ResearchResult(
            run_id=run_id,
            request=request,
            started_at=started_at,
            completed_at=completed_at,
            products=products,
            sources=sources,
            assets=assets,
            claims=claims,
            observations=observations,
            configurations=configurations,
            similarity_matches=similarities,
            discovery_log=discovery_log,
            crawl_log=crawl_log,
            acquisition_log=acquisition_log,
            recognition_log=recognition_log,
            graph_edges=graph.edges,
            coverage=coverage,
            warnings=list(dict.fromkeys(warnings)),
            excel_path=str(destination),
        )
        export_excel(result, destination)
        self.repository.save(result)
        progress("completed")
        return result
