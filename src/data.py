"""Dataset loading, label normalization, stratified sampling and ordered splits."""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd


def find_column(frame: pd.DataFrame, candidates: tuple[str, ...], required: bool = True) -> str | None:
    lookup = {str(c).strip().lstrip("\ufeff").lower(): str(c) for c in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    if required:
        raise ValueError(f"Missing required column; tried {candidates}. Available: {list(frame.columns)}")
    return None


def _binary_labels(series: pd.Series) -> np.ndarray:
    tokens = series.astype("string").str.strip().str.lower()
    empty = tokens.isna() | tokens.isin({"", "nan", "none", "null"})
    benign = tokens.isin({"0", "benign", "normal", "false"})
    result = np.full(len(series), -1, dtype=np.int64)
    valid = (~empty).to_numpy(dtype=bool)
    benign_values = benign.to_numpy(dtype=bool)
    result[valid & benign_values] = 0
    result[valid & ~benign_values] = 1
    return result


def _read_csv_directory(path: Path, sample: int | None, seed: int) -> pd.DataFrame:
    files = sorted(path.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {path}")
    if not sample or sample <= 0:
        return pd.concat([pd.read_csv(f, low_memory=False, encoding="cp1252") for f in files], ignore_index=True, sort=False)

    # First pass reads labels only; the second keeps a uniform, class-stratified reservoir.
    label_counts = np.zeros(2, dtype=np.int64)
    label_by_file: dict[Path, str] = {}
    for file in files:
        header = pd.read_csv(file, nrows=0, encoding="cp1252")
        label_col = find_column(header, ("Label", "binary_label", "class", "target"))
        label_by_file[file] = label_col
        for chunk in pd.read_csv(file, usecols=[label_col], chunksize=200_000, low_memory=False, encoding="cp1252"):
            y = _binary_labels(chunk[label_col])
            label_counts += np.bincount(y[y >= 0], minlength=2)
    total = int(label_counts.sum())
    take = min(int(sample), total)
    raw_quota = take * label_counts / max(total, 1)
    quotas = np.floor(raw_quota).astype(int)
    for cls in np.argsort(-(raw_quota - quotas))[: take - int(quotas.sum())]:
        quotas[cls] += 1

    rng = np.random.default_rng(seed)
    reservoirs: dict[int, pd.DataFrame] = {0: pd.DataFrame(), 1: pd.DataFrame()}
    source_offset = 0
    for file in files:
        for chunk in pd.read_csv(file, chunksize=100_000, low_memory=False, encoding="cp1252"):
            label_col = label_by_file[file]
            y_all = _binary_labels(chunk[label_col])
            source_rows = np.arange(source_offset, source_offset + len(chunk), dtype=np.int64)
            source_offset += len(chunk)
            valid = y_all >= 0
            if not valid.all():
                chunk = chunk.loc[valid].copy()
                y = y_all[valid]
                source_rows = source_rows[valid]
            else:
                y = y_all
            chunk["_target"] = y
            chunk["_source_row"] = source_rows
            for cls in (0, 1):
                quota = int(quotas[cls])
                if quota == 0:
                    continue
                part = chunk.loc[y == cls].copy()
                if part.empty:
                    continue
                part["_sampling_key"] = rng.random(len(part))
                candidates = pd.concat([reservoirs[cls], part], ignore_index=True, sort=False)
                reservoirs[cls] = candidates.nsmallest(quota, "_sampling_key")
    sampled = pd.concat([reservoirs[0], reservoirs[1]], ignore_index=True, sort=False)
    return sampled.drop(columns=["_sampling_key"], errors="ignore")


def _read_dataset(path: Path, sample: int | None, seed: int) -> pd.DataFrame:
    if path.is_dir():
        return _read_csv_directory(path, sample, seed)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False, encoding="cp1252")
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def load_flows(path: str | Path, sample: int | None = None, seed: int = 17) -> pd.DataFrame:
    """Load flow files; sample by binary class and restore chronological order."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Run python scripts/download_data.py first.")
    frame = _read_dataset(path, sample if path.is_dir() else None, seed)
    frame.columns = [str(c).strip().lstrip("\ufeff") for c in frame.columns]
    label_col = find_column(frame, ("Label", "binary_label", "class", "target"))
    target = _binary_labels(frame[label_col])
    labeled = target >= 0
    if not labeled.all():
        warnings.warn(f"Dropping {(~labeled).sum()} rows with missing/empty labels.")
        frame = frame.loc[labeled].copy()
        target = target[labeled]
    labels = frame[label_col].astype(str).str.strip()
    benign_tokens = {"0", "benign", "normal", "false"}
    frame["_target"] = target
    attack_col = find_column(frame, ("Attack", "attack_cat", "attack_type"), required=False)
    attacks = frame[attack_col].fillna("Unknown").astype(str).str.strip() if attack_col else labels
    frame["_attack_type"] = attacks.where(~attacks.str.lower().isin(benign_tokens), "Benign")

    if not path.is_dir() and sample is not None and sample > 0 and sample < len(frame):
        # Stratified uniform sampling on binary labels, followed by temporal sorting.
        rng = np.random.default_rng(seed)
        chosen: list[int] = []
        target_values = frame["_target"].to_numpy()
        for cls in np.unique(target_values):
            positions = np.flatnonzero(target_values == cls)
            quota = int(round(sample * len(positions) / len(frame)))
            chosen.extend(rng.choice(positions, size=min(quota, len(positions)), replace=False).tolist())
        selected = set(chosen)
        if len(chosen) < sample:
            remaining = np.setdiff1d(np.arange(len(frame)), np.fromiter(selected, dtype=np.int64))
            chosen.extend(rng.choice(remaining, size=sample - len(chosen), replace=False).tolist())
        elif len(chosen) > sample:
            chosen = rng.choice(chosen, size=sample, replace=False).tolist()
        frame = frame.iloc[chosen].copy()

    ts_col = find_column(frame, ("Timestamp", "ts", "flow_start", "flow_start_time", "FLOW_START_MILLISECONDS", "Stime"), required=False)
    if ts_col:
        numeric_ts = pd.to_numeric(frame[ts_col], errors="coerce")
        if numeric_ts.notna().mean() > 0.9:
            frame["_order"] = numeric_ts
            values = numeric_ts.to_numpy(dtype=float)
            magnitude = float(np.nanmedian(np.abs(values)))
            unit_divisor = 1_000_000.0 if magnitude > 1e14 else (1_000.0 if magnitude > 1e11 else 1.0)
            frame["_time_seconds"] = numeric_ts / unit_divisor
        else:
            parsed = pd.to_datetime(frame[ts_col].astype(str).str.strip(), errors="coerce", utc=True, dayfirst=True, format="mixed")
            if parsed.notna().mean() > 0.9:
                valid_time = parsed.notna()
                if not valid_time.all():
                    warnings.warn(f"Dropping {(~valid_time).sum()} rows with invalid {ts_col!r} timestamps.")
                    frame = frame.loc[valid_time].copy()
                    parsed = parsed.loc[valid_time]
                parsed_ns = parsed.astype("datetime64[ns, UTC]")
                time_ns = parsed_ns.astype("int64")
                frame["_order"] = time_ns
                frame["_time_seconds"] = time_ns // 1_000_000_000
            else:
                warnings.warn(f"Could not reliably parse {ts_col!r}; using source row order.")
                frame["_order"] = frame["_source_row"] if "_source_row" in frame else np.arange(len(frame))
                frame["_time_seconds"] = frame["_order"].astype(float)
                frame["_time_source"] = "source_row_order"
        if "_time_source" not in frame:
            frame["_time_source"] = ts_col
    else:
        warnings.warn("No timestamp found: ordered windows use source row order, not wall-clock time.")
        frame["_order"] = frame["_source_row"] if "_source_row" in frame else np.arange(len(frame))
        frame["_time_seconds"] = frame["_order"].astype(float)
        frame["_time_source"] = "source_row_order"
    return frame.sort_values("_order", kind="stable").reset_index(drop=True)


def temporal_indices(n_rows: int, train_fraction: float = 0.6, validation_fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return contiguous train/validation/test indices in chronological order."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must be positive and sum to less than one.")
    train_end = int(n_rows * train_fraction)
    val_end = int(n_rows * (train_fraction + validation_fraction))
    if train_end == 0 or val_end == train_end or val_end >= n_rows:
        raise ValueError("Not enough rows for a temporal split.")
    return np.arange(0, train_end), np.arange(train_end, val_end), np.arange(val_end, n_rows)


def temporal_window_indices(frame: pd.DataFrame, window_seconds: int = 3600, train_fraction: float = 0.6, validation_fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split whole timestamp windows chronologically so no graph crosses a split.

    If a real timestamp is unavailable, fall back to contiguous row splits; the
    data loader marks that case explicitly as a source-order proxy.
    """
    source = str(frame["_time_source"].iloc[0]) if "_time_source" in frame and len(frame) else "source_row_order"
    if source == "source_row_order" or "_time_seconds" not in frame:
        return temporal_indices(len(frame), train_fraction, validation_fraction)
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    seconds = frame["_time_seconds"].to_numpy()
    if np.issubdtype(seconds.dtype, np.integer):
        buckets = np.floor_divide(seconds, window_seconds).astype(np.int64)
    else:
        buckets = np.floor(seconds.astype(float) / window_seconds).astype(np.int64)
    unique = np.unique(buckets)
    if len(unique) < 3:
        raise ValueError("At least three timestamp windows are required for a temporal split.")
    train_end = max(1, min(len(unique) - 2, int(len(unique) * train_fraction)))
    val_end = max(train_end + 1, min(len(unique) - 1, int(len(unique) * (train_fraction + validation_fraction))))
    train_buckets, val_buckets, test_buckets = unique[:train_end], unique[train_end:val_end], unique[val_end:]
    train = np.flatnonzero(np.isin(buckets, train_buckets))
    val = np.flatnonzero(np.isin(buckets, val_buckets))
    test = np.flatnonzero(np.isin(buckets, test_buckets))
    return train, val, test
