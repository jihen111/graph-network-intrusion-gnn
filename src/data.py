"""Load CIC-IDS2017 labelled flows, repair timestamps, sample by window and split.

Two evaluation protocols are provided:

* Protocol A ("unseen attacks"): train on Monday-Wednesday, validate on
  Thursday, test on Friday. Friday's attacks (Bot, PortScan, DDoS) never appear
  in training, so this measures generalisation to new attack types.
* Protocol B ("seen attacks"): every time window is labelled by its dominant
  attack type, and the windows of each type are split chronologically
  60/20/20. Every attack type with enough windows appears in train, validation
  and test, while whole windows never cross a split boundary.
"""
from __future__ import annotations

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

DAY_NAMES = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday"}
WORK_START_HOUR = 8  # CIC-IDS2017 captures run from about 08:55 to 17:00 local time


def find_column(frame: pd.DataFrame, candidates: tuple[str, ...], required: bool = True) -> str | None:
    lookup = {str(c).strip().lstrip("﻿").lower(): str(c) for c in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    if required:
        raise ValueError(f"Missing required column; tried {candidates}. Available: {list(frame.columns)}")
    return None


def _binary_labels(series: pd.Series) -> np.ndarray:
    """0 = benign, 1 = attack, -1 = missing label (dropped, never counted as attack)."""
    tokens = series.astype("string").str.strip().str.lower()
    empty = tokens.isna() | tokens.isin({"", "nan", "none", "null"})
    benign = tokens.isin({"0", "benign", "normal", "false"})
    result = np.full(len(series), -1, dtype=np.int64)
    valid = (~empty).to_numpy(dtype=bool)
    benign_values = benign.to_numpy(dtype=bool)
    result[valid & benign_values] = 0
    result[valid & ~benign_values] = 1
    return result


def normalize_attack_name(name: str) -> str:
    """'Web Attack � Brute Force' -> 'Web Attack - Brute Force'; 'BENIGN' -> 'Benign'."""
    text = re.sub(r"[^\x20-\x7e]+", "-", str(name)).strip()
    text = re.sub(r"\s*-\s*", " - ", text) if "Web Attack" in text else text
    return "Benign" if text.lower() in {"benign", "normal", "0"} else text


def repair_cic_timestamps(raw: pd.Series) -> pd.Series:
    """Parse CIC timestamps (day/month/year) and fix the 12-hour clock.

    Several CIC-IDS2017 CSVs write afternoon times on a 12-hour clock without
    an AM/PM marker (e.g. '7/7/2017 3:56' for 15:56). Captures only cover
    working hours, so any hour before 08:00 is shifted by 12 hours. The fix is
    checked against the documented attack schedule in notebook 01.
    """
    text = raw.astype(str).str.strip()
    parsed = pd.to_datetime(text, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text[missing], format="%d/%m/%Y %H:%M", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text[missing], dayfirst=True, errors="coerce", format="mixed")
    early = parsed.dt.hour < WORK_START_HOUR
    parsed = parsed.where(~early, parsed + pd.Timedelta(hours=12))
    return parsed


def load_cic(path: str | Path) -> pd.DataFrame:
    """Read every labelled-flow CSV under `path` into one frame (float32 features)."""
    path = Path(path)
    files = sorted(path.rglob("*.csv")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No CSV files found in {path}. Run python scripts/download_data.py first.")
    parts = []
    for file in files:
        df = pd.read_csv(file, low_memory=False, encoding="cp1252")
        df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
        df = df.loc[:, ~pd.Index(df.columns).duplicated()]
        for col in df.columns:
            if col in {"Flow ID", "Source IP", "Destination IP", "Timestamp", "Label"}:
                continue
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
        df["_file"] = file.name
        parts.append(df.drop(columns=["Flow ID"], errors="ignore"))
    frame = pd.concat(parts, ignore_index=True, sort=False)
    del parts
    target = _binary_labels(frame["Label"])
    keep = target >= 0
    if (~keep).any():
        warnings.warn(f"Dropping {(~keep).sum()} rows with missing labels.")
    frame = frame.loc[keep].reset_index(drop=True)
    frame["_target"] = target[keep]
    frame["_attack_type"] = frame["Label"].map(normalize_attack_name)
    ts = repair_cic_timestamps(frame["Timestamp"])
    bad = ts.isna().to_numpy()
    if bad.any():
        warnings.warn(f"Dropping {bad.sum()} rows with unparseable timestamps.")
        frame, ts = frame.loc[~bad].reset_index(drop=True), ts.loc[~bad].reset_index(drop=True)
    frame["_datetime"] = ts.to_numpy()
    frame["_time_seconds"] = ((ts - pd.Timestamp("1970-01-01")).dt.total_seconds()).astype(np.int64).to_numpy()  # unit-safe
    frame["_day"] = ts.dt.dayofweek.map(DAY_NAMES).fillna("Other").to_numpy()
    frame["_time_source"] = "Timestamp (12h clock repaired)"
    return frame.sort_values("_time_seconds", kind="stable").reset_index(drop=True)


def add_windows(frame: pd.DataFrame, window_seconds: int) -> pd.DataFrame:
    frame["_window"] = (frame["_time_seconds"] // window_seconds).astype(np.int64)
    return frame


def window_benign_cap(frame: pd.DataFrame, cap: int | None, seed: int = 17) -> pd.DataFrame:
    """Topology-preserving reduction: keep every attack flow and every window.

    Inside each window at most `cap` benign flows are kept (random), so benign
    traffic is thinned but attack structure (scans, DDoS stars) is complete.
    `cap=None` keeps everything.
    """
    if not cap:
        return frame.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    keys = rng.random(len(frame))
    benign = frame["_target"].to_numpy() == 0
    order = pd.DataFrame({"w": frame["_window"].to_numpy(), "k": keys})
    rank = order.where(pd.Series(benign)).groupby("w")["k"].rank(method="first")
    keep = (~benign) | (rank.to_numpy() <= cap)
    return frame.loc[keep].sort_values("_time_seconds", kind="stable").reset_index(drop=True)


def split_protocol_a(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unseen attacks: Mon-Wed train, Thursday validation, Friday test."""
    day = frame["_day"].to_numpy()
    train = np.flatnonzero(np.isin(day, ["Monday", "Tuesday", "Wednesday"]))
    val = np.flatnonzero(day == "Thursday")
    test = np.flatnonzero(day == "Friday")
    return train, val, test


def window_labels(frame: pd.DataFrame) -> pd.Series:
    """Dominant attack type of each window ('Benign' when it has no attack flow)."""
    attacks = frame.loc[frame["_target"] == 1, ["_window", "_attack_type"]]
    dominant = attacks.groupby("_window")["_attack_type"].agg(lambda s: s.value_counts().index[0])
    windows = pd.Series(np.unique(frame["_window"]))
    return windows.map(dominant).fillna("Benign").set_axis(windows.to_numpy())


def split_protocol_b(frame: pd.DataFrame, train_fraction: float = 0.6, validation_fraction: float = 0.2):
    """Seen attacks: split the windows of each dominant type chronologically 60/20/20."""
    labels = window_labels(frame)
    assignment: dict[int, str] = {}
    for kind, windows in labels.groupby(labels):
        ordered = np.sort(windows.index.to_numpy())
        n = len(ordered)
        if n < 3:  # too rare to split: keep in train so the type is at least learned
            for w in ordered:
                assignment[int(w)] = "train"
            continue
        n_train = max(1, int(round(n * train_fraction)))
        n_val = max(1, int(round(n * validation_fraction)))
        n_train = min(n_train, n - 2)
        n_val = min(n_val, n - n_train - 1)
        for i, w in enumerate(ordered):
            assignment[int(w)] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    part = frame["_window"].map(assignment).to_numpy()
    return (np.flatnonzero(part == "train"), np.flatnonzero(part == "val"), np.flatnonzero(part == "test"))


def attack_counts(frame: pd.DataFrame, indices: np.ndarray) -> dict[str, int]:
    counts = frame.iloc[indices]["_attack_type"].value_counts()
    return {str(k): int(v) for k, v in counts.items()}
