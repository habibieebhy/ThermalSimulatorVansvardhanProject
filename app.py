"""Streamlit entry point for BRIXTA Mattress Product Intelligence."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mattress_intelligence.streamlit_app import render_app


render_app()
