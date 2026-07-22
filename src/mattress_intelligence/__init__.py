"""Standalone mattress catalogue intelligence and configuration generation."""

from .models import CompanyResearchRequest, ProductRecord, ResearchResult
from .pipeline import MattressIntelligencePipeline

__all__ = [
    "CompanyResearchRequest",
    "MattressIntelligencePipeline",
    "ProductRecord",
    "ResearchResult",
]

__version__ = "1.3.0"
