"""Run the whole project end to end: data, experiments, notebooks, report, tests, ZIP.

Usage: python run_all.py [--quick]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def step(title: str, cmd: list[str]) -> None:
    t = time.time()
    print(f"\n=== {title} ===", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"--- {title}: {time.time() - t:.0f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="fast smoke run (1 seed, small benign cap)")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    py = sys.executable
    started = time.time()
    step("1/6 Download CIC-IDS2017 (skipped if already present)", [py, "scripts/download_data.py"])
    step("2/6 Experiments", [py, "-m", "src.train", "--config", args.config] + (["--quick"] if args.quick else []))
    step("3/6 Notebooks", [py, "scripts/create_notebooks.py"])
    for nb in ("notebooks/01_eda.ipynb", "notebooks/02_graph_analysis.ipynb"):
        step(f"    execute {nb}", [py, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                                    "--ExecutePreprocessor.timeout=3600", nb])
    step("4/6 README and SUMMARY from results", [py, "scripts/write_report.py"])
    step("5/6 Unit tests", [py, "-m", "unittest", "discover", "-s", "tests", "-v"])
    print("\n=== 6/6 Package ===", flush=True)
    out = ROOT.parent / "graph-network-intrusion-gnn-v2"
    staging = ROOT.parent / "_package" / "graph-network-intrusion-gnn"
    shutil.rmtree(staging.parent, ignore_errors=True)
    shutil.copytree(ROOT, staging, ignore=shutil.ignore_patterns("raw", "*.zip", "__pycache__", ".ipynb_checkpoints", "smoke"))
    archive = shutil.make_archive(str(out), "zip", staging.parent)
    shutil.rmtree(staging.parent, ignore_errors=True)
    print(f"Done in {(time.time() - started) / 60:.0f} min. Archive: {archive}")


if __name__ == "__main__":
    main()
