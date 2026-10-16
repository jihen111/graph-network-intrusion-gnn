"""Edge-aware GraphSAGE models for flow-level intrusion classification."""
from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import torch
from torch import nn

from src.build_graph import DEVICE, FlowGraph, N_STRUCT


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))


class EdgeSAGEConv(nn.Module):
    """GraphSAGE mean aggregation with flow attributes in each message."""
    def __init__(self, in_channels: int, edge_channels: int, out_channels: int):
        super().__init__()
        self.message = nn.Linear(in_channels + edge_channels, out_channels)
        self.update = nn.Linear(in_channels + out_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        source, destination = edge_index
        messages = torch.relu(self.message(torch.cat([x[source], edge_attr], dim=1)))
        aggregate = x.new_zeros((x.shape[0], messages.shape[1]))
        aggregate.index_add_(0, destination, messages)
        degree = torch.bincount(destination, minlength=x.shape[0]).to(dtype=x.dtype).clamp_min(1).unsqueeze(1)
        aggregate = aggregate / degree
        return torch.relu(self.update(torch.cat([x, aggregate], dim=1)))


class EdgeGraphSAGE(nn.Module):
    """Classify a directed flow from edge-aware host embeddings and its attributes.

    With `use_flow_features=False` (ablation) the model only sees the graph
    structure: structural node features, and no flow attributes at all.
    """
    def __init__(self, flow_dim: int, hidden: int = 64, dropout: float = 0.2, use_flow_features: bool = True):
        super().__init__()
        self.use_flow_features = use_flow_features
        node_dim = N_STRUCT + (flow_dim if use_flow_features else 0)
        edge_dim = flow_dim if use_flow_features else 0
        self.conv1 = EdgeSAGEConv(node_dim, edge_dim, hidden)
        self.conv2 = EdgeSAGEConv(hidden, edge_dim, hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2 + edge_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2)
        )

    def forward(self, graph: FlowGraph) -> torch.Tensor:
        n_flows = len(graph.y)
        if self.use_flow_features:
            x, edge_x = graph.node_x, graph.edge_x
        else:
            x, edge_x = graph.struct_x, graph.edge_x.new_zeros((n_flows, 0))
        # edge_index holds E original arcs then E reversed arcs; both carry the flow's attributes.
        message_edge_x = torch.cat([edge_x, edge_x], dim=0)
        h = self.dropout(self.conv1(x, graph.edge_index, message_edge_x))
        h = self.dropout(self.conv2(h, graph.edge_index, message_edge_x))
        src, dst = graph.edge_index[0, :n_flows], graph.edge_index[1, :n_flows]
        return self.classifier(torch.cat([h[src], h[dst], edge_x], dim=1))


def _labeled_mask(y: torch.Tensor, fraction: float, rng: np.random.Generator) -> torch.Tensor:
    """Sample a class-stratified subset of training labels, preserving rare attacks."""
    if fraction >= 1:
        return torch.ones(len(y), dtype=torch.bool)
    mask = np.zeros(len(y), dtype=bool)
    y_np = y.cpu().numpy()
    for cls in np.unique(y_np):
        idx = np.flatnonzero(y_np == cls)
        k = max(1, int(round(len(idx) * fraction)))
        mask[rng.choice(idx, size=min(k, len(idx)), replace=False)] = True
    return torch.as_tensor(mask)


def fit_supervised(
    train_graphs: Iterable[FlowGraph], flow_dim: int, seed: int,
    epochs: int = 20, learning_rate: float = 0.003, weight_decay: float = 1e-4,
    hidden: int = 64, label_fraction: float = 1.0, use_flow_features: bool = True,
    validation_graphs: Iterable[FlowGraph] | None = None, patience: int = 3, min_delta: float = 1e-4,
) -> EdgeGraphSAGE:
    """Train on selected labels and restore the best validation-loss checkpoint."""
    set_seed(seed)
    graphs = list(train_graphs)
    if not graphs:
        raise ValueError("No training graph windows supplied.")
    rng = np.random.default_rng(seed)
    # Choose one global stratified annotation budget once. Resampling masks each
    # epoch would quietly expose more labels than the requested fraction.
    all_y = torch.cat([g.y for g in graphs])
    global_mask = _labeled_mask(all_y, label_fraction, rng)
    masks, offset = [], 0
    for graph in graphs:
        masks.append(global_mask[offset:offset + len(graph.y)])
        offset += len(graph.y)
    observed_y = torch.cat([graph.y[mask] for graph, mask in zip(graphs, masks)])
    counts = torch.bincount(observed_y, minlength=2).float().clamp_min(1)
    weights = counts.sum() / (2.0 * counts)
    val_graphs = list(validation_graphs) if validation_graphs is not None else []
    if val_graphs and patience < 1:
        raise ValueError("Early-stopping patience must be at least one.")
    model = EdgeGraphSAGE(flow_dim, hidden=hidden, use_flow_features=use_flow_features).to(DEVICE)
    weights = weights.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_loss, best_state, stale_epochs, trained_epochs = float("inf"), None, 0, 0
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(graphs))  # shuffle window order every epoch
        for k in order:
            graph, mask = graphs[k].to(DEVICE), masks[k].to(DEVICE)
            if not mask.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            logits = model(graph)
            loss = nn.functional.cross_entropy(logits[mask], graph.y[mask], weight=weights)
            loss.backward()
            optimizer.step()
        trained_epochs = epoch + 1
        if val_graphs:
            model.eval()
            total_loss, edge_count = 0.0, 0
            with torch.no_grad():
                for graph in val_graphs:
                    graph = graph.to(DEVICE)
                    logits = model(graph)
                    total_loss += nn.functional.cross_entropy(logits, graph.y, weight=weights, reduction="sum").item()
                    edge_count += len(graph.y)
            val_loss = total_loss / max(edge_count, 1)
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


def predict_supervised(model: EdgeGraphSAGE, graphs: Iterable[FlowGraph]) -> np.ndarray:
    """Return attack probabilities for each original flow in supplied windows."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for graph in graphs:
            outputs.append(torch.softmax(model(graph.to(DEVICE)), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(outputs) if outputs else np.array([], dtype=float)
