"""CLI: python -m src.train --config configs/experiment.yaml [--quick]."""
from __future__ import annotations

import argparse

from src.experiments.run import run_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate graph-based network intrusion detectors.")
    parser.add_argument("--config", default="configs/experiment.yaml", help="YAML experiment configuration")
    parser.add_argument("--quick", action="store_true", help="One seed and a small benign cap: a fast smoke run")
    args = parser.parse_args()
    run_experiments(args.config, quick=args.quick)


if __name__ == "__main__":
    main()
