"""Host-level graph windows: nodes are IPs, directed edges are observed flows.

Each fixed time window (5 minutes by default) becomes one graph. Flow features
are edge attributes. Node features have two parts:

* structural features computed from the window's topology (degree, number of
  distinct peers, number of distinct destination ports contacted), which is
  where scanning or DDoS patterns show up;
* the mean of the incident flows' features (an aggregate, not a copy of any
  single flow).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import torch

from src.data import find_column

N_STRUCT = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STRUCT_NAMES = ["log_out_degree", "log_in_degree", "log_distinct_destinations",
                "log_distinct_sources", "log_distinct_dst_ports", "log_distinct_src_ports_seen"]


@dataclass
class FlowGraph:
    """A single time window and its flow-level edge labels."""
    edge_index: torch.Tensor       # [2, 2E]: E original arcs, then E reversed arcs (message passing only)
    edge_x: torch.Tensor           # [E, F] one feature row per original flow
    y: torch.Tensor                # [E] binary label per flow
    struct_x: torch.Tensor         # [N, 6] structural node features
    node_x: torch.Tensor           # [N, 6 + F] structural + mean incident flow features
    src_hosts: list[str]
    dst_hosts: list[str]
    row_indices: np.ndarray        # positions in the frame passed to build_graph_windows
    window_id: int
    dst_port: np.ndarray | None = None
    src_port: np.ndarray | None = None

    def to(self, device) -> "FlowGraph":
        """Copy of the graph with its tensors on `device` (no-op on CPU)."""
        return replace(self, edge_index=self.edge_index.to(device), edge_x=self.edge_x.to(device), y=self.y.to(device),
                       struct_x=self.struct_x.to(device), node_x=self.node_x.to(device))


def _ip_columns(frame: pd.DataFrame) -> tuple[str, str]:
    src = find_column(frame, ("Source IP", "IPV4_SRC_ADDR", "src_ip", "source_ip", "srcip"))
    dst = find_column(frame, ("Destination IP", "IPV4_DST_ADDR", "dst_ip", "destination_ip", "dstip"))
    return src, dst


def _distinct_count(owner: np.ndarray, value: np.ndarray, n: int) -> np.ndarray:
    if len(owner) == 0:
        return np.zeros(n)
    pairs = np.unique(np.stack([owner, value], axis=1), axis=0)
    return np.bincount(pairs[:, 0], minlength=n).astype(float)


def structural_features(src_idx, dst_idx, dst_port, src_port, n_nodes) -> np.ndarray:
    out_deg = np.bincount(src_idx, minlength=n_nodes)
    in_deg = np.bincount(dst_idx, minlength=n_nodes)
    feats = np.stack([
        out_deg, in_deg,
        _distinct_count(src_idx, dst_idx, n_nodes),
        _distinct_count(dst_idx, src_idx, n_nodes),
        _distinct_count(src_idx, dst_port, n_nodes),
        _distinct_count(dst_idx, src_port, n_nodes),
    ], axis=1)
    return np.log1p(feats).astype(np.float32)


def build_graph_windows(frame: pd.DataFrame, edge_features: np.ndarray, window_col: str = "_window",
                        window_size: int | None = None) -> list[FlowGraph]:
    """Build one graph per value of `window_col` (or per `window_size` rows if absent)."""
    if len(frame) != len(edge_features):
        raise ValueError("Rows and edge feature matrix must have equal lengths.")
    src_col, dst_col = _ip_columns(frame)
    dport_col = find_column(frame, ("Destination Port", "dst_port", "L4_DST_PORT"), required=False)
    sport_col = find_column(frame, ("Source Port", "src_port", "L4_SRC_PORT"), required=False)
    labels = frame["_target"].to_numpy(dtype=np.int64)
    src_all = frame[src_col].astype(str).to_numpy()
    dst_all = frame[dst_col].astype(str).to_numpy()
    dport_all = frame[dport_col].fillna(-1).to_numpy(dtype=np.int64) if dport_col else np.zeros(len(frame), dtype=np.int64)
    sport_all = frame[sport_col].fillna(-1).to_numpy(dtype=np.int64) if sport_col else np.zeros(len(frame), dtype=np.int64)
    if window_col in frame:
        win = frame[window_col].to_numpy()
        order = np.argsort(win, kind="stable")
        bounds = np.flatnonzero(np.diff(win[order])) + 1
        groups = [(int(win[g[0]]), g) for g in np.split(order, bounds) if len(g)]
    else:
        size = window_size or 2000
        groups = [(i, np.arange(s, min(s + size, len(frame)))) for i, s in enumerate(range(0, len(frame), size))]

    graphs: list[FlowGraph] = []
    for window_id, pos in groups:
        src, dst = src_all[pos], dst_all[pos]
        hosts, inverse = np.unique(np.concatenate([src, dst]), return_inverse=True)
        src_idx, dst_idx = inverse[: len(pos)], inverse[len(pos):]
        n = len(hosts)
        e_x = np.asarray(edge_features[pos], dtype=np.float32)
        struct = structural_features(src_idx, dst_idx, dport_all[pos], sport_all[pos], n)
        sums = np.zeros((n, e_x.shape[1]), dtype=np.float64)
        np.add.at(sums, src_idx, e_x)
        np.add.at(sums, dst_idx, e_x)
        counts = np.bincount(src_idx, minlength=n) + np.bincount(dst_idx, minlength=n)
        mean_x = (sums / np.maximum(counts, 1)[:, None]).astype(np.float32)
        edge_index = np.vstack((np.concatenate([src_idx, dst_idx]), np.concatenate([dst_idx, src_idx])))
        graphs.append(FlowGraph(
            edge_index=torch.as_tensor(edge_index, dtype=torch.long),
            edge_x=torch.as_tensor(e_x),
            y=torch.as_tensor(labels[pos].copy()),
            struct_x=torch.as_tensor(struct),
            node_x=torch.as_tensor(np.concatenate([struct, mean_x], axis=1)),
            src_hosts=src.tolist(), dst_hosts=dst.tolist(),
            row_indices=pos.copy(), window_id=int(window_id),
            dst_port=dport_all[pos].copy(), src_port=sport_all[pos].copy(),
        ))
    return graphs


def flow_order(graphs: list[FlowGraph]) -> np.ndarray:
    """Frame positions in the order in which predictions are concatenated."""
    return np.concatenate([g.row_indices for g in graphs]) if graphs else np.array([], dtype=int)
