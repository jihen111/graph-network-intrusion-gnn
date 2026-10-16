"""Benign-only structural graph autoencoder for flow anomaly scoring."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import SAGEConv

from src.build_graph import DEVICE, FlowGraph, N_STRUCT
from src.models.gnn import set_seed


class EdgeFeatureAutoencoder(nn.Module):
    """Reconstruct a flow's attributes from its endpoints' structural embeddings only."""
    def __init__(self, edge_dim: int, hidden: int = 64):
        super().__init__()
        self.conv1 = SAGEConv(N_STRUCT, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.decoder = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, edge_dim))

    def forward(self, graph: FlowGraph) -> torch.Tensor:
        # Never feed the target edge features (or node aggregates of them) to
        # the reconstruction network: that would permit a direct copy shortcut.
        edge_count = len(graph.y)
        src = graph.edge_index[0, :edge_count]
        dst = graph.edge_index[1, :edge_count]
        x = graph.struct_x  # degrees, distinct peers and ports: topology only
        h = torch.relu(self.conv1(x, graph.edge_index))
        h = torch.relu(self.conv2(h, graph.edge_index))
        return self.decoder(torch.cat([h[src], h[dst]], dim=1))


def _benign_structure(graph: FlowGraph, benign_idx: torch.Tensor) -> torch.Tensor:
    """Recompute structural node features from benign flows only (no attack leakage)."""
    from src.build_graph import structural_features
    idx = benign_idx.numpy()
    n = graph.struct_x.shape[0]
    src = graph.edge_index[0, benign_idx].numpy()
    dst = graph.edge_index[1, benign_idx].numpy()
    dport = graph.dst_port[idx] if graph.dst_port is not None else np.zeros(len(idx), dtype=np.int64)
    sport = graph.src_port[idx] if graph.src_port is not None else np.zeros(len(idx), dtype=np.int64)
    return torch.as_tensor(structural_features(src, dst, dport, sport, n))


def _benign_only_graph(graph: FlowGraph) -> FlowGraph | None:
    """Remove every attack edge before the autoencoder learns graph structure."""
    benign_idx = torch.nonzero(graph.y == 0, as_tuple=False).flatten()
    if not len(benign_idx):
        return None
    n_edges = len(graph.y)
    msg_idx = torch.cat([benign_idx, benign_idx + n_edges])
    return FlowGraph(
        edge_index=graph.edge_index[:, msg_idx],
        edge_x=graph.edge_x[benign_idx],
        y=graph.y[benign_idx],
        src_hosts=[graph.src_hosts[i] for i in benign_idx.tolist()],
        dst_hosts=[graph.dst_hosts[i] for i in benign_idx.tolist()],
        struct_x=_benign_structure(graph, benign_idx),
        node_x=torch.zeros_like(graph.node_x),
        row_indices=graph.row_indices[benign_idx.cpu().numpy()],
        window_id=graph.window_id,
        dst_port=graph.dst_port[benign_idx.numpy()] if graph.dst_port is not None else None,
        src_port=graph.src_port[benign_idx.numpy()] if graph.src_port is not None else None,
    )


def fit_benign_autoencoder(
    train_graphs: Iterable[FlowGraph], edge_dim: int, seed: int,
    epochs: int = 20, learning_rate: float = 0.003, weight_decay: float = 1e-4, hidden: int = 64,
    validation_graphs: Iterable[FlowGraph] | None = None, patience: int = 3, min_delta: float = 1e-5,
) -> EdgeFeatureAutoencoder:
    """Fit benign-only subgraphs and stop on validation-benign reconstruction loss."""
    set_seed(seed)
    graphs = [subgraph for graph in train_graphs if (subgraph := _benign_only_graph(graph)) is not None]
    if not graphs:
        raise ValueError("No benign training edges available for autoencoder fitting.")
    val_graphs = [subgraph for graph in validation_graphs or [] if (subgraph := _benign_only_graph(graph)) is not None]
    if val_graphs and patience < 1:
        raise ValueError("Early-stopping patience must be at least one.")
    model = EdgeFeatureAutoencoder(edge_dim, hidden).to(DEVICE)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_loss, best_state, stale_epochs, trained_epochs = float("inf"), None, 0, 0
    for epoch in range(epochs):
        model.train()
        for k in rng.permutation(len(graphs)):
            graph = graphs[k].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(graph)
            loss = nn.functional.mse_loss(reconstruction, graph.edge_x)
            loss.backward()
            optimizer.step()
        trained_epochs = epoch + 1
        if val_graphs:
            model.eval()
            total_loss, element_count = 0.0, 0
            with torch.no_grad():
                for graph in val_graphs:
                    graph = graph.to(DEVICE)
                    reconstruction = model(graph)
                    total_loss += nn.functional.mse_loss(reconstruction, graph.edge_x, reduction="sum").item()
                    element_count += graph.edge_x.numel()
            val_loss = total_loss / max(element_count, 1)
            if val_loss < best_loss - min_delta:
                best_loss = val_loss
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.epochs_trained = trained_epochs
    return model


def reconstruction_scores(model: EdgeFeatureAutoencoder, graphs: Iterable[FlowGraph]) -> np.ndarray:
    """Per-flow mean squared reconstruction error; larger values are more anomalous."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(DEVICE)
            err = (model(graph) - graph.edge_x).pow(2).mean(dim=1)
            outputs.append(err.cpu().numpy())
    return np.concatenate(outputs) if outputs else np.array([], dtype=float)


def threshold_by_validation_f1(y_val: np.ndarray, scores: np.ndarray) -> float:
    """Select a threshold on validation labels only; test labels are never consulted."""
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_val, scores)
    if not len(thresholds):
        return float(np.median(scores))
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])
