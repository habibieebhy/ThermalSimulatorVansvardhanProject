"""Convenience runner for a fresh checkout without package installation."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mattress_thermal import run_mattress_simulation  # noqa: E402


if __name__ == "__main__":
    run_mattress_simulation(
        output_path=PROJECT_ROOT / "outputs" / "mattress_investor_dashboard.png",
        csv_path=PROJECT_ROOT / "outputs" / "mattress_simulation.csv",
        show=False,
        print_report=True,
    )

