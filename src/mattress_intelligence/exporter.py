"""Analyst-friendly multi-sheet Excel export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
from xlsxwriter.workbook import Workbook
from xlsxwriter.worksheet import Worksheet

from .models import ResearchResult


def _flat(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return value


def _product_rows(result: ResearchResult) -> list[dict[str, Any]]:
    return [
        {
            "product_id": product.product_id,
            "company": product.company_name,
            "brand": product.brand,
            "product": product.name,
            "family": product.family,
            "canonical_url": product.canonical_url,
            "firmness": product.firmness,
            "total_thickness_mm": product.total_thickness_mm,
            "product_weight_kg": product.product_weight_kg,
            "price": product.price,
            "currency": product.currency,
            "layer_count": len(product.layers),
            "variant_count": len(product.variants),
            "source_count": len(product.source_ids),
            "extraction_method": product.extraction_method,
            "extraction_confidence": product.extraction_confidence,
            "reviewed": product.reviewed,
            "description": product.description,
        }
        for product in result.products
    ]


def _variant_rows(result: ResearchResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for product in result.products:
        for variant in product.variants:
            rows.append(
                {
                    "product_id": product.product_id,
                    "product": product.name,
                    **variant.model_dump(),
                    "source_ids": ", ".join(variant.source_ids),
                }
            )
    return rows


def _layer_rows(result: ResearchResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for product in result.products:
        for layer in product.layers:
            rows.append(
                {
                    "product_id": product.product_id,
                    "product": product.name,
                    "layer_id": layer.layer_id,
                    "position": layer.position,
                    "marketing_name": layer.marketing_name,
                    "normalized_material": layer.normalized_material,
                    "thickness_mm": layer.thickness_mm,
                    "thickness_status": layer.thickness_status,
                    "density_kg_m3": layer.density_kg_m3,
                    "density_status": layer.density_status,
                    "evidence_source_ids": ", ".join(
                        item.source_id for item in layer.evidence
                    ),
                    "evidence_asset_ids": ", ".join(
                        item.asset_id for item in layer.evidence if item.asset_id
                    ),
                    "evidence_excerpts": " | ".join(
                        item.excerpt or "" for item in layer.evidence if item.excerpt
                    ),
                }
            )
    return rows


def _configuration_rows(result: ResearchResult) -> list[dict[str, Any]]:
    product_names = {str(product.product_id): product.name for product in result.products}
    return [
        {
            "configuration_id": item.configuration_id,
            "product_id": item.product_id,
            "product": product_names.get(item.product_id),
            "rank": item.rank,
            "total_thickness_mm": item.total_thickness_mm,
            "estimated_weight_kg": item.estimated_weight_kg,
            "posterior_probability": item.posterior_probability,
            "confidence_score": item.confidence_score,
            "evidence_score": item.evidence_score,
            "reasons": " | ".join(item.reasons),
            "contradictions": " | ".join(item.contradictions),
        }
        for item in result.configurations
    ]


def _configuration_layer_rows(result: ResearchResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in result.configurations:
        for layer in candidate.layers:
            rows.append({"configuration_id": candidate.configuration_id, **layer.model_dump()})
    return rows


def _asset_rows(result: ResearchResult) -> list[dict[str, Any]]:
    return [
        {
            **asset.model_dump(exclude={"vision_payload"}),
            "vision_payload": _flat(asset.vision_payload),
        }
        for asset in result.assets
    ]


def _observation_rows(result: ResearchResult) -> list[dict[str, Any]]:
    return [
        {
            **item.model_dump(exclude={"value"}),
            "value": _flat(item.value),
        }
        for item in result.observations
    ]


def _review_rows(result: ResearchResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for product in result.products:
        missing = []
        if product.total_thickness_mm is None:
            missing.append("total_thickness_mm")
        if not product.layers:
            missing.append("layers")
        if any(layer.density_kg_m3 is None for layer in product.layers):
            missing.append("one_or_more_layer_densities")
        if product.extraction_confidence < 0.7 or missing:
            rows.append(
                {
                    "entity_type": "product",
                    "entity_id": product.product_id,
                    "product": product.name,
                    "priority": "high" if product.extraction_confidence < 0.6 else "normal",
                    "reason": ", ".join(missing) or "low extraction confidence",
                    "confidence": product.extraction_confidence,
                }
            )
    for candidate in result.configurations:
        if candidate.rank == 1 and candidate.confidence_score < 60:
            rows.append(
                {
                    "entity_type": "configuration",
                    "entity_id": candidate.configuration_id,
                    "product": candidate.product_id,
                    "priority": "high",
                    "reason": "Top configuration has low provisional confidence",
                    "confidence": candidate.confidence_score / 100.0,
                }
            )
    return rows


def _frames(result: ResearchResult) -> dict[str, pd.DataFrame]:
    company_row = {
        "company_id": result.request.company_id,
        "company": result.request.company_name,
        "official_domain": result.request.official_domain,
        "market": result.request.market,
        **result.coverage.model_dump(),
        "limitations": " | ".join(result.coverage.limitations),
    }
    source_rows = [
        {**source.model_dump(), "retrieved_at": source.retrieved_at.isoformat()}
        for source in result.sources
    ]
    claim_rows = [
        {
            **claim.model_dump(exclude={"evidence"}),
            "value": _flat(claim.value),
            "evidence": _flat([item.model_dump() for item in claim.evidence]),
        }
        for claim in result.claims
    ]
    graph_rows = [
        {**item, "properties": _flat(item.get("properties", {}))}
        for item in result.graph_edges
    ]
    metadata = [
        {"key": "run_id", "value": result.run_id},
        {"key": "started_at", "value": result.started_at.isoformat()},
        {"key": "completed_at", "value": result.completed_at.isoformat()},
        {"key": "web_search_enabled", "value": result.request.use_search_grounding},
        {"key": "assets", "value": len(result.assets)},
        {"key": "vision_assets", "value": sum(1 for item in result.assets if item.vision_payload)},
        {"key": "deterministic_observations", "value": len(result.observations)},
        {"key": "recognition_events", "value": len(result.recognition_log)},
        {
            "key": "admitted_product_documents",
            "value": sum(1 for item in result.recognition_log if item.get("accepted")),
        },
        {"key": "warnings", "value": " | ".join(result.warnings)},
        {
            "key": "confidence_notice",
            "value": "Configuration scores are provisional until calibrated against verified constructions.",
        },
        {
            "key": "observation_notice",
            "value": "Evidence Observations are deterministic text matches; review context before treating them as product-level facts.",
        },
    ]
    return {
        "Companies": pd.DataFrame([company_row]),
        "Products": pd.DataFrame(_product_rows(result)),
        "Variants": pd.DataFrame(_variant_rows(result)),
        "Layers": pd.DataFrame(_layer_rows(result)),
        "Assets": pd.DataFrame(_asset_rows(result)),
        "Evidence Observations": pd.DataFrame(_observation_rows(result)),
        "Observed Claims": pd.DataFrame(claim_rows),
        "Configurations": pd.DataFrame(_configuration_rows(result)),
        "Config Layers": pd.DataFrame(_configuration_layer_rows(result)),
        "Similar Products": pd.DataFrame(result.similarity_matches),
        "Discovery Log": pd.DataFrame(result.discovery_log),
        "Crawl Log": pd.DataFrame(result.crawl_log),
        "Acquisition Log": pd.DataFrame(result.acquisition_log),
        "Recognition Log": pd.DataFrame(result.recognition_log),
        "Evidence Sources": pd.DataFrame(source_rows),
        "Graph Edges": pd.DataFrame(graph_rows),
        "Review Queue": pd.DataFrame(_review_rows(result)),
        "Run Metadata": pd.DataFrame(metadata),
    }


def export_excel(result: ResearchResult, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = _frames(result)
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        workbook = cast(Workbook, writer.book)
        header = workbook.add_format(
            {
                "bold": True,
                "font_color": "#FFFFFF",
                "bg_color": "#18212F",
                "border": 0,
                "text_wrap": True,
                "valign": "vcenter",
            }
        )
        warning = workbook.add_format({"bg_color": "#FFF2CC", "font_color": "#7F6000"})
        percentage = workbook.add_format({"num_format": "0.0%"})
        decimal = workbook.add_format({"num_format": "0.00"})

        for sheet_name, frame in frames.items():
            safe_frame = frame if not frame.empty else pd.DataFrame({"message": ["No records"]})
            safe_frame.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
                startrow=1,
                header=False,
            )
            worksheet = cast(Worksheet, writer.sheets[sheet_name])
            rows, columns = safe_frame.shape
            worksheet.add_table(
                0,
                0,
                rows,
                columns - 1,
                {
                    "columns": [
                        {"header": str(column), "header_format": header}
                        for column in safe_frame.columns
                    ],
                    "style": "Table Style Medium 2",
                },
            )
            worksheet.freeze_panes(1, 0)
            for index, column in enumerate(safe_frame.columns):
                sample_lengths = [len(str(column))]
                sample_lengths.extend(
                    len(str(value)) for value in safe_frame[column].head(200)
                )
                width = min(70, max(12, max(sample_lengths, default=12) + 2))
                cell_format = None
                if "probability" in str(column) or str(column) == "evidence_score":
                    cell_format = percentage
                elif any(token in str(column) for token in ("confidence", "weight")):
                    cell_format = decimal
                worksheet.set_column(index, index, width, cell_format)
            if sheet_name == "Review Queue" and rows:
                worksheet.conditional_format(
                    1,
                    0,
                    rows,
                    columns - 1,
                    {"type": "no_blanks", "format": warning},
                )

        dashboard = workbook.add_worksheet("Dashboard")
        dashboard.set_tab_color("#FF4B55")
        title_format = workbook.add_format(
            {"bold": True, "font_size": 20, "font_color": "#18212F"}
        )
        metric_format = workbook.add_format(
            {"bold": True, "font_size": 14, "bg_color": "#E8EEF6", "border": 1}
        )
        dashboard.write(1, 1, "Mattress Intelligence", title_format)
        dashboard.write(3, 1, "Company", metric_format)
        dashboard.write(3, 2, result.request.company_name)
        dashboard.write(4, 1, "Products", metric_format)
        dashboard.write(4, 2, len(result.products))
        dashboard.write(5, 1, "Variants", metric_format)
        dashboard.write(5, 2, sum(len(product.variants) for product in result.products))
        dashboard.write(6, 1, "Evidence sources", metric_format)
        dashboard.write(6, 2, len(result.sources))
        dashboard.write(7, 1, "Evidence observations", metric_format)
        dashboard.write(7, 2, len(result.observations))
        dashboard.write(8, 1, "Captured assets", metric_format)
        dashboard.write(8, 2, len(result.assets))
        dashboard.write(9, 1, "Vision-analyzed assets", metric_format)
        dashboard.write(9, 2, sum(1 for item in result.assets if item.vision_payload))
        dashboard.write(10, 1, "Recognized product documents", metric_format)
        dashboard.write(10, 2, sum(1 for item in result.recognition_log if item.get("accepted")))
        dashboard.write(11, 1, "Estimated catalogue coverage", metric_format)
        dashboard.write(
            11,
            2,
            result.coverage.estimated_coverage_percent / 100.0,
            percentage,
        )
        dashboard.write(12, 1, "Candidate configurations", metric_format)
        dashboard.write(12, 2, len(result.configurations))
        dashboard.write(
            14,
            1,
            "Important: observations require context review; inferred values are hypotheses.",
            warning,
        )
        dashboard.set_column(1, 1, 38)
        dashboard.set_column(2, 2, 26)
        dashboard.activate()
    return output_path
