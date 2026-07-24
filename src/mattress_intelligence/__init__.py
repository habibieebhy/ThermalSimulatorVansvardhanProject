"""Standalone mattress catalogue intelligence and configuration generation."""

from .models import (
    CompanyResearchRequest,
    ProductRecord,
    ResearchResult,
    TrademarkMaterialRecord,
)
from .pipeline import MattressIntelligencePipeline

__all__ = [
    "CompanyResearchRequest",
    "MattressIntelligencePipeline",
    "ProductRecord",
    "ResearchResult",
    "TrademarkMaterialRecord",
]

__version__ = "1.6.1"
