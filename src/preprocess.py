"""Leakage-aware feature preparation for network flows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


ID_HINTS = ("ipv4_src_addr", "ipv4_dst_addr", "src_ip", "dst_ip", "srcip", "dstip", "source_ip", "destination_ip", "src_addr", "dst_addr", "flow_id", "timestamp", "stime", "ltime")
PORT_HINTS = {"l4_src_port", "l4_dst_port", "src_port", "dst_port", "source_port", "destination_port", "source port", "destination port", "sport", "dsport"}
CAT_HINTS = ("protocol", "proto", "l7_proto", "tcp_flags", "flag")
HEAVY_HINTS = ("bytes", "pkts", "packets", "duration", "flows", "count")


def _port_bucket(values: pd.Series) -> pd.Series:
    """Map ports to conventional service/client port ranges, not raw integers."""
    port = pd.to_numeric(values, errors="coerce")
    return pd.cut(port, bins=[-1, 1023, 49151, 65535], labels=["well_known", "registered", "ephemeral"]).astype("object").fillna("unknown")


def _log_nonnegative(values: np.ndarray) -> np.ndarray:
    """Apply log1p to heavy-tailed non-negative counters after median imputation."""
    return np.log1p(np.clip(values, a_min=0, a_max=None))


def _lower_name(name: object) -> str:
    return str(name).strip().lower()


def identifier_columns(columns: Iterable[object]) -> list[str]:
    """Return network identifiers and target-like fields that must not be features."""
    drop = []
    for col in columns:
        name = _lower_name(col)
        normalized = name.replace(" ", "_")
        if name.startswith("_") or name in {"label", "attack", "class", "target", "binary_label", "id"} or any(h in normalized for h in ID_HINTS) or name in PORT_HINTS:
            drop.append(str(col))
    return drop


@dataclass
class FeaturePreprocessor:
    """Fit imputers, encoders and scalers on training flows only.

    Raw source/destination IPs and ports are deliberately excluded from the
    feature matrix. Ports can enter as coarse categories in engineered mode.
    """
    engineered: bool = True

    def __post_init__(self) -> None:
        self.transformer: ColumnTransformer | None = None
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.output_names: list[str] = []
        self._is_fitted = False

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        numeric_cols = out.select_dtypes(include=[np.number]).columns
        if len(numeric_cols):
            out.loc[:, numeric_cols] = out.loc[:, numeric_cols].replace([np.inf, -np.inf], np.nan)
        if self.engineered:
            for col in out.columns:
                name = _lower_name(col)
                if name in PORT_HINTS:
                    side = "src" if ("src" in name or "source" in name or name == "sport") else "dst"
                    out[f"{side}_port_bucket"] = _port_bucket(out[col])
        return out

    def fit(self, train_frame: pd.DataFrame) -> "FeaturePreprocessor":
        work = self._prepare(train_frame)
        excluded = set(identifier_columns(work.columns))
        # Generated coarse port fields are intentional features; raw ports remain excluded.
        excluded -= {c for c in work.columns if c.endswith("_port_bucket")}
        feature_cols = [str(c) for c in work.columns if str(c) not in excluded]
        cats: list[str] = []
        nums: list[str] = []
        for col in feature_cols:
            name = _lower_name(col)
            series = work[col]
            forced_cat = any(h in name for h in CAT_HINTS) or name.endswith("_port_bucket")
            if forced_cat or not pd.api.types.is_numeric_dtype(series.dtype):
                cats.append(col)
            else:
                nums.append(col)
        self.numeric_columns, self.categorical_columns = nums, cats
        transformers = []
        if nums:
            heavy = [c for c in nums if self.engineered and any(h in _lower_name(c) for h in HEAVY_HINTS)]
            ordinary = [c for c in nums if c not in heavy]
            if ordinary:
                transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), ordinary))
            if heavy:
                transformers.append(("heavy", Pipeline([("imputer", SimpleImputer(strategy="median")), ("log1p", FunctionTransformer(_log_nonnegative, feature_names_out="one-to-one")), ("scale", StandardScaler())]), heavy))
        if cats:
            transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32))]), cats))
        if not transformers:
            raise ValueError("No usable flow features found after excluding identifiers and targets.")
        self.transformer = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
        self.transformer.fit(work)
        self.output_names = list(self.transformer.get_feature_names_out())
        self._is_fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted or self.transformer is None:
            raise RuntimeError("Call fit on training flows before transform.")
        work = self._prepare(frame)
        return np.asarray(self.transformer.transform(work), dtype=np.float32)

    def fit_transform(self, train_frame: pd.DataFrame) -> np.ndarray:
        return self.fit(train_frame).transform(train_frame)
