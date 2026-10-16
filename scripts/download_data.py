"""Download a public mirror of CIC-IDS2017 GeneratedLabelledFlows."""
from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

import requests

URL = "https://huggingface.co/datasets/bencorn/CICIDS2017/resolve/main/csvs/GeneratedLabelledFlows.zip?download=true"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", default="data/raw/cic-generated-labelled-flows")
    parser.add_argument("--force", action="store_true", help="Redownload even if CSV files already exist")
    args = parser.parse_args()
    destination = Path(args.destination)
    if destination.exists() and any(destination.rglob("*.csv")) and not args.force:
        print(f"CSV files already exist under {destination}; nothing to do.")
        return
    archive = destination.parent / "cic-generated-labelled-flows.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading public mirror: {URL}")
    response = requests.get(URL, stream=True, timeout=(20, 300))
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    received = 0
    with archive.open("wb") as handle:
        for chunk in response.iter_content(4 * 1024 * 1024):
            if chunk:
                handle.write(chunk)
                received += len(chunk)
                if received and (received % (50 * 1024 * 1024) < 4 * 1024 * 1024):
                    print(f"Downloaded {received / 1e6:.0f} MB" + (f" / {total / 1e6:.0f} MB" if total else ""), flush=True)
    if not zipfile.is_zipfile(archive):
        raise RuntimeError("The response is not a valid ZIP archive; check the source URL and retry.")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(destination)
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
    if not names:
        raise RuntimeError("The archive contained no CSV files.")
    print(f"Extracted {len(names)} labelled-flow CSV files to {destination}")


if __name__ == "__main__":
    main()
