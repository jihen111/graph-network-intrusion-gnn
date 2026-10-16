"""Host-level graph windows: nodes are IPs and directed edges are observed flows."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src.data import find_column


@dataclass
class FlowGraph:
    """A single ordered graph window and its flow-level edge labels."""
    edge_index: torch.Tensor       # [2, 2E], bidirectional for neighborhood aggregation
    edge_x: torch.Tensor           # [E, F], one feature row per original flow
    y: torch.Tensor                # [E], one binary label per original flow
    src_hosts: list[str]
    dst_hosts: list[str]
    node_x: torch.Tensor           # [N, F], incident-edge mean (same schema as edge_x)
    row_indices: np.ndarray
    window_id: int


def _ip_columns(frame: pd.DataFrame) -> tuple[str, str]:
    src = find_column(frame, ("IPV4_SRC_ADDR", "src_ip", "source_ip", "src_addr", "Source IP", "srcip"))
    dst = find_column(frame, ("IPV4_DST_ADDR", "dst_ip", "destination_ip", "dst_addr", "Destination IP", "dstip"))
    return src, dst


def build_graph_windows(
    frame: pd.DataFrame,
    edge_features: np.ndarray,
    window_size: int = 2000,
    window_seconds: int | None = 3600,
) -> list[FlowGraph]:
    """Build separate chronological graphs by fixed UTC intervals when possible.

    Host-level nodes preserve communication topology while avoiding a node
    explosion. IP:port endpoints could model service/client roles more
    precisely, but fragment hosts into sparse nodes; ports are instead kept
    as coarse categories in the edge feature vector. If no real timestamp is
    supplied, ordered fixed-flow chunks are used only as a documented proxy.
    """
    if len(frame) != len(edge_features):
        raise ValueError("Rows and edge feature matrix must have equal lengths.")
    if window_size < 1:
        raise ValueError("window_size must be positive")
    src_col, dst_col = _ip_columns(frame)
    labels = frame["_target"].to_numpy(dtype=np.int64, copy=True)
    source = str(frame["_time_source"].iloc[0]) if "_time_source" in frame and len(frame) else "source_row_order"
    use_clock = window_seconds is not None and "_time_seconds" in frame and source != "source_row_order"
    if use_clock:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        times = frame["_time_seconds"].to_numpy(dtype=float)
        if not np.isfinite(times).all():
            raise ValueError("Timestamp contains non-finite values; clean it before graph construction.")
        buckets = np.floor(times / window_seconds).astype(np.int64)
        groups = [(int(bucket), np.flatnonzero(buckets == bucket)) for bucket in np.unique(buckets)]
    else:
        groups = [(idx, np.arange(start, min(start + window_size, len(frame)), dtype=np.int64))
                  for idx, start in enumerate(range(0, len(frame), window_size))]

    graphs: list[FlowGraph] = []
    for window_id, positions in groups:
        part = frame.iloc[positions]
        src = part[src_col].fillna("unknown-src").astype(str).tolist()
        dst = part[dst_col].fillna("unknown-dst").astype(str).tolist()
        hosts = sorted(set(src) | set(dst))
        mapping = {host: idx for idx, host in enumerate(hosts)}
        src_idx = np.fromiter((mapping[x] for x in src), dtype=np.int64)
        dst_idx = np.fromiter((mapping[x] for x in dst), dtype=np.int64)
        e_x = np.array(edge_features[positions], dtype=np.float32, copy=True)
        # Each flow is classified once; the reversed arcs only support message passing.
        edge_index = np.vstack((np.concatenate([src_idx, dst_idx]), np.concatenate([dst_idx, src_idx])))
        node_x = np.zeros((len(hosts), e_x.shape[1]), dtype=np.float32)
        counts = np.zeros(len(hosts), dtype=np.float32)
        for node_idx, flow_idx in zip(np.concatenate([src_idx, dst_idx]), np.tile(np.arange(len(e_x)), 2)):
            node_x[node_idx] += e_x[flow_idx]
            counts[node_idx] += 1
        node_x /= np.maximum(counts[:, None], 1.0)
        graphs.append(FlowGraph(
            edge_index=torch.as_tensor(edge_index, dtype=torch.long),
            edge_x=torch.as_tensor(e_x, dtype=torch.float32),
            y=torch.as_tensor(labels[positions].copy(), dtype=torch.long),
            src_hosts=src,
            dst_hosts=dst,
            node_x=torch.as_tensor(node_x, dtype=torch.float32),
            row_indices=positions.copy(),
            window_id=int(window_id),
        ))
    return graphs
