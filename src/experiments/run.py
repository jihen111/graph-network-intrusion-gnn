"""Run the end-to-end CPU experiments and save real metrics and figures."""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import yaml

from src.build_graph import FlowGraph, build_graph_windows
from src.data import load_flows, temporal_window_indices
from src.metrics import classification_metrics
from src.models.autoencoder import fit_benign_autoencoder, reconstruction_scores, threshold_by_validation_f1
from src.models.baseline import make_baseline
from src.models.gnn import fit_supervised, predict_supervised
from src.preprocess import FeaturePreprocessor


def _write_json(path: Path, obj) -> None:
    def safe(value):
        if isinstance(value, dict):
            return {str(k): safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(v) for v in value]
        if isinstance(value, (np.integer,)): return int(value)
        if isinstance(value, (np.floating, float)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, np.ndarray): return safe(value.tolist())
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe(obj), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _labels(graphs: list[FlowGraph]) -> np.ndarray:
    return np.concatenate([g.y.numpy() for g in graphs]) if graphs else np.array([], dtype=int)


def _graph_sets(frame: pd.DataFrame, features: np.ndarray, indices: np.ndarray, window_size: int, window_seconds: int):
    subset = frame.iloc[indices].reset_index(drop=True)
    return build_graph_windows(subset, features[indices], window_size, window_seconds)


def _train_endpoints(frame, train_idx, val_idx, test_idx, sample, seed, window_size, window_seconds):
    # Fitting transforms on train only prevents validation/test statistics leaking into features.
    prep = FeaturePreprocessor(engineered=True).fit(frame.iloc[train_idx])
    X = prep.transform(frame)
    return prep, X, (
        _graph_sets(frame, X, train_idx, window_size, window_seconds),
        _graph_sets(frame, X, val_idx, window_size, window_seconds),
        _graph_sets(frame, X, test_idx, window_size, window_seconds),
    )


def _attack_types(frame: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    return frame.iloc[indices]["_attack_type"].astype(str).to_numpy()


def _safe_positive_probability(model, X: np.ndarray) -> np.ndarray:
    if 1 not in model.classes_:
        return np.zeros(len(X), dtype=float)
    return model.predict_proba(X)[:, list(model.classes_).index(1)]


def _summarize(records: list[dict]) -> dict:
    numeric = [k for k, v in records[0].items() if isinstance(v, (int, float, np.number)) and k != "seed"] if records else []
    return {key: {"mean": float(np.nanmean([r[key] for r in records])), "std": float(np.nanstd([r[key] for r in records], ddof=0))} for key in numeric}


def _flow_graph(frame: pd.DataFrame, labels: np.ndarray, selector: np.ndarray | None = None) -> nx.DiGraph:
    src_col, dst_col = _ip_columns_local(frame)
    g = nx.DiGraph()
    indices = np.flatnonzero(selector) if selector is not None else np.arange(len(frame))
    for i in indices:
        g.add_edge(str(frame.iloc[i][src_col]), str(frame.iloc[i][dst_col]))
    return g


def _ip_columns_local(frame: pd.DataFrame) -> tuple[str, str]:
    from src.data import find_column
    return find_column(frame, ("Source IP", "IPV4_SRC_ADDR", "src_ip", "source_ip")), find_column(frame, ("Destination IP", "IPV4_DST_ADDR", "dst_ip", "destination_ip"))


def _graph_statistics(graphs: list[FlowGraph]) -> dict:
    rows = []
    degree_values = {"Benign": [], "Attack": []}
    fanouts = {"Benign": [], "Attack": []}
    for graph in graphs:
        for label, name in ((0, "Benign"), (1, "Attack")):
            ids = np.flatnonzero(graph.y.numpy() == label)
            g = nx.DiGraph()
            g.add_edges_from((graph.src_hosts[i], graph.dst_hosts[i]) for i in ids)
            if len(g) == 0:
                continue
            degree_values[name].extend(dict(g.degree()).values())
            source_fanouts = [degree for _, degree in g.out_degree() if degree > 0]
            fanouts[name].extend(source_fanouts)
            rows.append({
                "class": name,
                "window_id": int(graph.window_id),
                "flows": int(len(ids)),
                "nodes": int(g.number_of_nodes()),
                "edges": int(g.number_of_edges()),
                "components": int(nx.number_weakly_connected_components(g)),
                "density": float(nx.density(g)),
                "mean_unique_destinations_per_source": float(np.mean(source_fanouts)),
            })
    summary = {}
    for name in ("Benign", "Attack"):
        selected = [r for r in rows if r["class"] == name]
        summary[name] = {
            "windows": len(selected),
            "mean_flows": float(np.mean([r["flows"] for r in selected])) if selected else 0.0,
            "mean_nodes": float(np.mean([r["nodes"] for r in selected])) if selected else 0.0,
            "mean_edges": float(np.mean([r["edges"] for r in selected])) if selected else 0.0,
            "mean_components": float(np.mean([r["components"] for r in selected])) if selected else 0.0,
            "mean_density": float(np.mean([r["density"] for r in selected])) if selected else 0.0,
            "mean_fanout": float(np.mean(fanouts[name])) if fanouts[name] else 0.0,
        }
    return {"per_window": rows, "summary": summary, "degree_values": degree_values, "fanout_values": fanouts}


def _save_graph_figures(graph_data: dict, test_graphs: list[FlowGraph], test_prob: np.ndarray, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"Benign": "#277da1", "Attack": "#f94144"}
    for name in ("Benign", "Attack"):
        values = graph_data["degree_values"][name]
        if values:
            axes[0].hist(values, bins=min(25, max(5, len(set(values)))), density=True, alpha=0.65, label=name, color=colors[name])
        fanout = graph_data["fanout_values"][name]
        if fanout:
            axes[1].hist(fanout, bins=min(25, max(5, len(set(fanout)))), density=True, alpha=0.65, label=name, color=colors[name])
    axes[0].set(title="Degree distribution", xlabel="Degree", ylabel="Density")
    axes[1].set(title="Unique destinations per source", xlabel="Out-degree / fan-out", ylabel="Density")
    for ax in axes: ax.legend(frameon=False); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(out / "graph_comparison.png", dpi=160); plt.close(fig)

    remaining = np.asarray(test_prob, dtype=float).copy()
    candidates = []
    cursor = 0
    for g in test_graphs:
        scores = remaining[cursor:cursor + len(g.y)]
        cursor += len(g.y)
        flagged = np.flatnonzero(scores >= 0.5)
        candidates.append((len(flagged), g, flagged, scores))
    count, graph, flagged, scores = max(candidates, key=lambda x: x[0]) if candidates else (0, None, [], np.array([]))
    fig, ax = plt.subplots(figsize=(8, 6))
    if graph is None or count == 0:
        ax.text(0.5, 0.5, "No test-window edge crossed the 0.5 decision threshold.", ha="center", va="center", wrap=True)
        ax.set_axis_off()
    else:
        G = nx.DiGraph()
        for i in flagged:
            G.add_edge(graph.src_hosts[i], graph.dst_hosts[i], score=float(scores[i]))
        if G.number_of_nodes() > 60:
            keep = sorted(G.degree, key=lambda x: x[1], reverse=True)[:60]
            G = G.subgraph([n for n, _ in keep]).copy()
        pos = nx.spring_layout(G, seed=11, k=0.6)
        nx.draw_networkx(G, pos, ax=ax, node_size=320, font_size=7, arrows=True, width=0.8, node_color="#f4a261", edge_color="#555555")
        ax.set_title(f"Model-flagged test subgraph — window {graph.window_id} ({G.number_of_edges()} displayed edges)")
        ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "attack_subgraph.png", dpi=160); plt.close(fig)


def _plot_label_efficiency(records: list[dict], out: Path) -> None:
    frame = pd.DataFrame(records)
    summary = frame.groupby("label_fraction")["f1"].agg(["mean", "std"]).reset_index()
    summary["std"] = summary["std"].fillna(0)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    # F1 is bounded; asymmetric clipped errors avoid implying impossible values.
    lower = np.minimum(summary["std"].to_numpy(), summary["mean"].to_numpy())
    upper = np.minimum(summary["std"].to_numpy(), 1.0 - summary["mean"].to_numpy())
    ax.errorbar(summary["label_fraction"] * 100, summary["mean"], yerr=np.vstack([lower, upper]), marker="o", capsize=3, color="#277da1")
    ax.set(xscale="log", xticks=[1, 5, 10, 25, 100], xticklabels=["1", "5", "10", "25", "100"], xlabel="Training labels used (%)", ylabel="Test F1", title="Label-efficiency study (mean ± std, 3 seeds)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def run_experiments(config_path: str | Path, sample_override: int | None = None) -> dict:
    started = time.time()
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    data_cfg, split_cfg, model_cfg, exp_cfg = cfg["data"], cfg["split"], cfg["model"], cfg["experiments"]
    sample = sample_override if sample_override is not None else data_cfg.get("sample")
    out = Path(exp_cfg.get("output_dir", "results")); out.mkdir(parents=True, exist_ok=True)
    print(f"Loading {data_cfg['path']} (sample={sample})", flush=True)
    frame = load_flows(data_cfg["path"], sample=sample, seed=int(data_cfg.get("sample_seed", 17)))
    window_size = int(data_cfg.get("window_size", 2000))
    window_seconds = int(data_cfg.get("window_seconds", 3600))
    train_idx, val_idx, test_idx = temporal_window_indices(frame, window_seconds, float(split_cfg["train_fraction"]), float(split_cfg["validation_fraction"]))
    y_train = frame.iloc[train_idx]["_target"].to_numpy(dtype=int)
    y_val = frame.iloc[val_idx]["_target"].to_numpy(dtype=int)
    y_test = frame.iloc[test_idx]["_target"].to_numpy(dtype=int)
    attack_test = _attack_types(frame, test_idx)
    seeds = [int(s) for s in exp_cfg["seeds"]]
    early_stopping_patience = int(model_cfg.get("early_stopping_patience", 2))
    prep, X, (train_graphs, val_graphs, test_graphs) = _train_endpoints(frame, train_idx, val_idx, test_idx, sample, seeds[0], window_size, window_seconds)
    print(f"Loaded {len(frame):,} flows; train/val/test={len(train_idx):,}/{len(val_idx):,}/{len(test_idx):,}; features={X.shape[1]}; windows={len(train_graphs)}/{len(val_graphs)}/{len(test_graphs)}", flush=True)
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        print("Warning: a temporal split contains a single binary class; some metrics may be undefined.", flush=True)

    main: dict[str, list[dict]] = {"Random Forest (flow-only)": [], "E-GraphSAGE": [], "Benign Graph Autoencoder": []}
    test_prob_first_seed = None
    for seed in seeds:
        print(f"Main models, seed={seed}", flush=True)
        baseline = make_baseline(seed, int(exp_cfg.get("max_trees", 120)))
        baseline.fit(X[train_idx], y_train)
        p_rf = _safe_positive_probability(baseline, X[test_idx])
        main["Random Forest (flow-only)"].append({"seed": seed, **classification_metrics(y_test, p_rf, attack_test)})

        gnn = fit_supervised(train_graphs, X.shape[1], X.shape[1], seed, int(model_cfg["epochs"]), float(model_cfg["learning_rate"]), float(model_cfg["weight_decay"]), int(model_cfg["hidden_channels"]), validation_graphs=val_graphs, patience=early_stopping_patience)
        p_gnn = predict_supervised(gnn, test_graphs)
        main["E-GraphSAGE"].append({"seed": seed, **classification_metrics(y_test, p_gnn, attack_test)})
        if seed == seeds[0]: test_prob_first_seed = p_gnn

        ae = fit_benign_autoencoder(train_graphs, X.shape[1], X.shape[1], seed, int(model_cfg["autoencoder_epochs"]), float(model_cfg["learning_rate"]), float(model_cfg["weight_decay"]), int(model_cfg["hidden_channels"]), validation_graphs=val_graphs, patience=early_stopping_patience)
        val_scores = reconstruction_scores(ae, val_graphs)
        threshold = threshold_by_validation_f1(y_val, val_scores)
        test_scores = reconstruction_scores(ae, test_graphs)
        main["Benign Graph Autoencoder"].append({"seed": seed, **classification_metrics(y_test, test_scores, attack_test, threshold)})

    aggregate = {name: _summarize(records) for name, records in main.items()}
    # Label-efficiency: only training labels are reduced; val/test are unchanged.
    label_records = []
    for fraction in [float(x) for x in exp_cfg["label_fractions"]]:
        for seed in seeds:
            print(f"Label-efficiency fraction={fraction:g}, seed={seed}", flush=True)
            model = fit_supervised(train_graphs, X.shape[1], X.shape[1], seed, int(model_cfg["epochs"]), float(model_cfg["learning_rate"]), float(model_cfg["weight_decay"]), int(model_cfg["hidden_channels"]), label_fraction=fraction, validation_graphs=val_graphs, patience=early_stopping_patience)
            probs = predict_supervised(model, test_graphs)
            m = classification_metrics(y_test, probs, attack_test)
            label_records.append({"label_fraction": fraction, "seed": seed, **m})
    _write_json(out / "label_efficiency.json", label_records)
    _plot_label_efficiency(label_records, out / "label_efficiency.png")

    # Preprocessing and edge-feature ablation; every scaler is refit on train only.
    ablation_records = []
    if bool(exp_cfg.get("run_ablation", True)):
        for engineered in (True, False):
            ablation_prep = FeaturePreprocessor(engineered=engineered).fit(frame.iloc[train_idx])
            Xa = ablation_prep.transform(frame)
            tr_graphs = _graph_sets(frame, Xa, train_idx, window_size, window_seconds)
            va_graphs = _graph_sets(frame, Xa, val_idx, window_size, window_seconds)
            te_graphs = _graph_sets(frame, Xa, test_idx, window_size, window_seconds)
            for include_edge in (True, False):
                for seed in seeds:
                    print(f"Ablation engineered={engineered}, edge_features={include_edge}, seed={seed}", flush=True)
                    model = fit_supervised(tr_graphs, Xa.shape[1], Xa.shape[1], seed, int(model_cfg["epochs"]), float(model_cfg["learning_rate"]), float(model_cfg["weight_decay"]), int(model_cfg["hidden_channels"]), include_edge_features=include_edge, validation_graphs=va_graphs, patience=early_stopping_patience)
                    probs = predict_supervised(model, te_graphs, include_edge_features=include_edge)
                    record = {"features": "engineered" if engineered else "raw-scaled", "edge_features": include_edge, "seed": seed, **classification_metrics(y_test, probs, attack_test)}
                    ablation_records.append(record)
    _write_json(out / "ablation.json", ablation_records)

    graph_data = _graph_statistics(test_graphs)
    graph_json = {"per_window": graph_data["per_window"], "summary": graph_data["summary"], "split": "test only"}
    _write_json(out / "graph_analysis.json", graph_json)
    _save_graph_figures(graph_data, test_graphs, test_prob_first_seed if test_prob_first_seed is not None else np.zeros(len(y_test)), out)

    result = {
        "dataset": {"path": str(data_cfg["path"]), "rows_loaded": int(len(frame)), "sample_requested": sample, "sample_seed": int(data_cfg.get("sample_seed", 17)), "timestamp_source": str(frame["_time_source"].iloc[0]), "features_after_encoding": int(X.shape[1]), "raw_columns": [str(c) for c in frame.columns if not str(c).startswith("_")]},
        "split": {"method": f"contiguous chronological split by whole {window_seconds}-second windows after timestamp sorting", "train_fraction": float(split_cfg["train_fraction"]), "validation_fraction": float(split_cfg["validation_fraction"]), "counts": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)}, "positive_counts": {"train": int(y_train.sum()), "validation": int(y_val.sum()), "test": int(y_test.sum())}},
        "main_models_per_seed": main,
        "main_models_mean_std": aggregate,
        "label_efficiency_file": str(out / "label_efficiency.json"),
        "ablation_file": str(out / "ablation.json"),
        "graph_analysis_file": str(out / "graph_analysis.json"),
        "runtime_seconds": float(time.time() - started),
        "device": "cpu",
    }
    _write_json(out / "metrics.json", result)
    print(f"Saved real test metrics to {out / 'metrics.json'} in {result['runtime_seconds']:.1f}s", flush=True)
    return result
