"""Write a tiny synthetic CIC-like dataset for smoke-testing the pipeline only.

The rows are random and carry no meaning: never report metrics from them.
Usage: python scripts/make_smoke_data.py --out data/smoke
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DAYS = {"Monday": (3, None), "Tuesday": (4, "FTP-Patator"), "Wednesday": (5, "DoS Hulk"),
        "Thursday": (6, "Web Attack � Brute Force"), "Friday": (7, "PortScan")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/smoke")
    parser.add_argument("--flows-per-day", type=int, default=6000)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hosts = [f"192.168.10.{i}" for i in range(2, 40)]
    for day, (dom, attack) in DAYS.items():
        n = args.flows_per_day
        minutes = rng.uniform(9 * 60, 17 * 60, n)
        label = np.array(["BENIGN"] * n, dtype=object)
        src = rng.choice(hosts, n)
        dst = rng.choice(hosts, n)
        dport = rng.choice([80, 443, 53, 22], n)
        if attack:
            hit = (minutes > 10 * 60) & (minutes < 10 * 60 + 40) | (minutes > 14 * 60) & (minutes < 14 * 60 + 40)
            label[hit] = attack
            src[hit] = "172.16.0.1"
            if attack == "PortScan":
                dport[hit] = rng.integers(1, 5000, hit.sum())
        hh = (minutes // 60).astype(int)
        hh12 = np.where(hh > 12, hh - 12, hh)  # mimic the CIC 12-hour clock
        ts = [f"{dom}/7/2017 {h}:{int(m % 60):02d}" for h, m in zip(hh12, minutes)]
        df = pd.DataFrame({
            "Flow ID": "x", " Source IP": src, " Source Port": rng.integers(1024, 65535, n),
            " Destination IP": dst, " Destination Port": dport, " Protocol": 6, " Timestamp": ts,
            " Flow Duration": rng.exponential(1e5, n) * (1 + 5 * (label != "BENIGN")),
            " Total Fwd Packets": rng.poisson(5, n), " Total Backward Packets": rng.poisson(4, n),
            "Total Length of Fwd Packets": rng.exponential(500, n), " Flow Bytes/s": rng.exponential(1e4, n),
            " SYN Flag Count": (label != "BENIGN").astype(int), " Label": label,
        })
        df.to_csv(out / f"{day}-WorkingHours.pcap_ISCX.csv", index=False)
    print(f"Wrote synthetic smoke CSVs to {out}")


if __name__ == "__main__":
    main()
