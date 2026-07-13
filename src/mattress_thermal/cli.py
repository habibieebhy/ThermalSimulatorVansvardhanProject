"""Command-line interface for the mattress thermal simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .simulation import run_mattress_simulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate five mattress thermal architectures over a six-hour "
            "sleep cycle and render an investor-ready dashboard."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/mattress_investor_dashboard.png"),
        help="Dashboard image path (default: %(default)s).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path for exporting all time-series values as CSV.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the Matplotlib window after saving the dashboard.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_mattress_simulation(
        output_path=args.output,
        csv_path=args.csv,
        show=args.show,
        print_report=True,
    )


if __name__ == "__main__":
    main()

