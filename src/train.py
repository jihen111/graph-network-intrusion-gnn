"""CLI: python -m src.train --config configs/experiment.yaml [--sample 40000]."""
from __future__ import annotations

import argparse

from src.experiments.run import run_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Train graph-based network intrusion detectors.")
    parser.add_argument("--config", default="configs/experiment.yaml", help="YAML experiment configuration")
    parser.add_argument("--sample", type=int, default=None, help="Override the class-stratified flow sample size; 0 means use the full dataset")
    args = parser.parse_args()
    override = args.sample if args.sample and args.sample > 0 else (None if args.sample is None else 0)
    run_experiments(args.config, sample_override=override)


if __name__ == "__main__":
    main()
