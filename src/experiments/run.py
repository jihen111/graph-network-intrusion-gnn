"""End-to-end experiments on CIC-IDS2017: two protocols, three models, studies, figures.

Every number written to results/ comes from this script. Thresholds are chosen
on validation data only; test labels are used once, for the final metrics.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import yaml

from src.build_graph import build_graph_windows, flow_order
from src.data import (add_windows, attack_counts, load_cic, split_protocol_a, split_protocol_b,
                      window_benign_cap)
from src.metrics import classification_metrics, evaluate_scores, threshold_max_f1
from src.models.autoencoder import fit_benign_autoencoder, reconstruction_scores
from src.models.baseline import make_baseline
from src.models.gnn import fit_supervised, predict_supervised
from src.preprocess import FeaturePreprocessor

# Validated categorical palette (dataviz reference palette, light mode)
C_RF, C_GNN, C_AE = "#2a78d6", "#eb6834", "#1baf7a"
C_BENIGN, C_ATTACK, C_GRID, C_TEXT = "#2a78d6", "#e34948", "#dce1e8", "#52514e"
MODELS = ["Random Forest (flow-only)", "E-GraphSAGE", "Graph Autoencoder (benign-only)"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json(path: Path, obj) -> None:
    def safe(v):
        if isinstance(v, dict):
            return {str(k): safe(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [safe(x) for x in v]
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, (np.floating, float)):
            return float(v) if np.isfinite(v) else None
        if isinstance(v, np.ndarray):
            return safe(v.tolist())
        return v
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe(obj), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def style(ax):
    ax.grid(color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_GRID)
    ax.tick_params(colors=C_TEXT)


# ---------------------------------------------------------------- preparation
def prepare(frame, splits, engineered=True):
    """Fit preprocessing on train only, then build one graph per window for each split."""
    train_idx = splits[0]
    prep = FeaturePreprocessor(engineered=engineered).fit(frame.iloc[train_idx])
    X = prep.transform(frame)
    graphs, order = {}, {}
    for name, idx in zip(("train", "val", "test"), splits):
        sub = frame.iloc[idx].reset_index(drop=True)
        graphs[name] = build_graph_windows(sub, X[idx])
        order[name] = idx[flow_order(graphs[name])]  # frame positions in prediction order
        del sub
    return X, graphs, order, prep.output_names


def labels_of(frame, order):
    return frame["_target"].to_numpy()[order], frame["_attack_type"].to_numpy()[order]


def summarize(records: list[dict], point: str) -> dict:
    keys = ["precision", "recall", "f1", "false_positive_rate", "pr_auc"]
    return {k: {"mean": float(np.mean([r[point][k] for r in records])),
                "std": float(np.std([r[point][k] for r in records]))} for k in keys}


def per_attack_mean(records: list[dict], point: str) -> dict:
    names = records[0][point]["detection_rate_per_attack"].keys()
    return {n: {"mean": float(np.mean([r[point]["detection_rate_per_attack"][n]["detection_rate"] for r in records])),
                "support": records[0][point]["detection_rate_per_attack"][n]["support"]} for n in names}


# ---------------------------------------------------------------- one protocol
def run_protocol(name, frame, splits, cfg, seeds, keep_scores=False):
    m, e = cfg["model"], cfg["experiments"]
    t0 = time.time()
    X, graphs, order, feat_names = prepare(frame, splits)
    y_tr, _ = labels_of(frame, order["train"])
    y_va, a_va = labels_of(frame, order["val"])
    y_te, a_te = labels_of(frame, order["test"])
    log(f"{name}: flows train/val/test = {len(y_tr):,}/{len(y_va):,}/{len(y_te):,}; "
        f"windows = {len(graphs['train'])}/{len(graphs['val'])}/{len(graphs['test'])}; features = {X.shape[1]}")
    records = {k: [] for k in MODELS}
    kept, timings = {}, {k: [] for k in MODELS}
    store = {"y_val": y_va.astype(np.int8), "y_test": y_te.astype(np.int8), "attack_test": a_te.astype(str)}
    for seed in seeds:
        t = time.time()
        rf = make_baseline(seed, int(e["rf_trees"]), float(e["rf_max_samples"]))
        rf.fit(X[order["train"]], y_tr)
        s_va, s_te = rf.predict_proba(X[order["val"]])[:, 1], rf.predict_proba(X[order["test"]])[:, 1]
        records[MODELS[0]].append({"seed": seed, **evaluate_scores(y_va, s_va, y_te, s_te, a_te)})
        store[f"rf_{seed}_val"], store[f"rf_{seed}_test"] = s_va.astype(np.float32), s_te.astype(np.float32)
        timings[MODELS[0]].append(time.time() - t)
        if keep_scores and seed == seeds[0]:
            kept[MODELS[0]] = s_te
        log(f"{name} seed {seed}: RF done ({time.time() - t:.0f}s)")

        t = time.time()
        gnn = fit_supervised(graphs["train"], X.shape[1], seed, int(m["epochs"]), float(m["learning_rate"]),
                             float(m["weight_decay"]), int(m["hidden_channels"]),
                             validation_graphs=graphs["val"], patience=int(m["early_stopping_patience"]))
        s_va, s_te = predict_supervised(gnn, graphs["val"]), predict_supervised(gnn, graphs["test"])
        records[MODELS[1]].append({"seed": seed, "epochs": gnn.epochs_trained, **evaluate_scores(y_va, s_va, y_te, s_te, a_te)})
        store[f"gnn_{seed}_val"], store[f"gnn_{seed}_test"] = s_va.astype(np.float32), s_te.astype(np.float32)
        timings[MODELS[1]].append(time.time() - t)
        if keep_scores and seed == seeds[0]:
            kept[MODELS[1]] = s_te
            kept["gnn_threshold"] = threshold_max_f1(y_va, s_va)
        log(f"{name} seed {seed}: E-GraphSAGE done ({time.time() - t:.0f}s, {gnn.epochs_trained} epochs)")

        t = time.time()
        ae = fit_benign_autoencoder(graphs["train"], X.shape[1], seed, int(m["autoencoder_epochs"]),
                                    float(m["learning_rate"]), float(m["weight_decay"]), int(m["hidden_channels"]),
                                    validation_graphs=graphs["val"], patience=int(m["early_stopping_patience"]))
        s_va, s_te = reconstruction_scores(ae, graphs["val"]), reconstruction_scores(ae, graphs["test"])
        ev = evaluate_scores(y_va, s_va, y_te, s_te, a_te)
        store[f"ae_{seed}_val"], store[f"ae_{seed}_test"] = s_va.astype(np.float32), s_te.astype(np.float32)
        ev.pop("fixed_0.5")  # reconstruction errors are not probabilities
        records[MODELS[2]].append({"seed": seed, "epochs": ae.epochs_trained, **ev})
        timings[MODELS[2]].append(time.time() - t)
        if keep_scores and seed == seeds[0]:
            kept[MODELS[2]] = s_te
        log(f"{name} seed {seed}: autoencoder done ({time.time() - t:.0f}s, {ae.epochs_trained} epochs)")
    result = {
        "splits": {
            part: {"flows": int(len(idx)), "attacks": int(frame["_target"].to_numpy()[idx].sum()),
                   "windows": int(len(graphs[part])), "attack_types": attack_counts(frame, idx)}
            for part, idx in zip(("train", "val", "test"), splits)},
        "features": int(X.shape[1]),
        "per_seed": records,
        "summary": {k: {p: summarize(v, p) for p in v[0] if p not in {"seed", "epochs"}} for k, v in records.items()},
        "detection_rate_per_attack": {k: {p: per_attack_mean(v, p) for p in ("val_f1_threshold", "val_fpr_1pct_threshold")}
                                      for k, v in records.items()},
        "seconds_per_seed": {k: float(np.mean(v)) for k, v in timings.items()},
        "runtime_seconds": float(time.time() - t0),
    }
    out = Path(cfg["experiments"].get("output_dir", "results"))
    np.savez_compressed(out / f"scores_{name.lower().replace(' ', '_')}.npz", **store)  # lets thresholds be re-derived without retraining
    extra = {"graphs": graphs, "order": order, "X": X, "scores": kept, "y_test": y_te} if keep_scores else None
    return result, extra


# ---------------------------------------------------------------- studies
def label_efficiency(frame, graphs, order, X, cfg, seeds):
    m = cfg["model"]
    y_va, _ = labels_of(frame, order["val"])
    y_te, a_te = labels_of(frame, order["test"])
    rows = []
    for fraction in [float(f) for f in cfg["experiments"]["label_fractions"]]:
        for seed in seeds:
            model = fit_supervised(graphs["train"], X.shape[1], seed, int(m["epochs"]), float(m["learning_rate"]),
                                   float(m["weight_decay"]), int(m["hidden_channels"]), label_fraction=fraction,
                                   validation_graphs=graphs["val"], patience=int(m["early_stopping_patience"]))
            s_va, s_te = predict_supervised(model, graphs["val"]), predict_supervised(model, graphs["test"])
            ev = evaluate_scores(y_va, s_va, y_te, s_te, a_te)
            rows.append({"label_fraction": fraction, "seed": seed, "f1": ev["val_f1_threshold"]["f1"],
                         "pr_auc": ev["val_f1_threshold"]["pr_auc"], "recall_at_1pct_fpr": ev["val_fpr_1pct_threshold"]["recall"]})
            log(f"label efficiency {fraction:g} seed {seed}: F1 {rows[-1]['f1']:.3f}")
    return rows


def ablation(frame, splits, cfg, seeds):
    m = cfg["model"]
    rows = []
    for engineered in (True, False):
        X, graphs, order, _ = prepare(frame, splits, engineered=engineered)
        y_va, _ = labels_of(frame, order["val"])
        y_te, a_te = labels_of(frame, order["test"])
        for use_flow in (True, False):
            if not engineered and not use_flow:
                continue  # structure-only model does not use flow features at all
            for seed in seeds:
                model = fit_supervised(graphs["train"], X.shape[1], seed, int(m["epochs"]), float(m["learning_rate"]),
                                       float(m["weight_decay"]), int(m["hidden_channels"]), use_flow_features=use_flow,
                                       validation_graphs=graphs["val"], patience=int(m["early_stopping_patience"]))
                s_va, s_te = predict_supervised(model, graphs["val"]), predict_supervised(model, graphs["test"])
                ev = evaluate_scores(y_va, s_va, y_te, s_te, a_te)
                label = ("engineered features" if engineered else "raw scaled features") if use_flow else "graph structure only"
                rows.append({"variant": label, "seed": seed, "f1": ev["val_f1_threshold"]["f1"],
                             "pr_auc": ev["val_f1_threshold"]["pr_auc"], "recall_at_1pct_fpr": ev["val_fpr_1pct_threshold"]["recall"]})
                log(f"ablation {label} seed {seed}: F1 {rows[-1]['f1']:.3f}")
        del X, graphs
        gc.collect()
    return rows


# ---------------------------------------------------------------- figures
def fig_timeline(frame, out):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    fig, axes = plt.subplots(len(days), 1, figsize=(11, 10), sharex=True)
    for ax, day in zip(axes, days):
        d = frame[frame["_day"] == day]
        hours = d["_datetime"].dt.hour + d["_datetime"].dt.minute / 60
        bins = np.arange(8, 18.01, 5 / 60)
        ax.hist(hours[d["_target"] == 0], bins=bins, color=C_BENIGN, alpha=0.55, label="Benign")
        ax.hist(hours[d["_target"] == 1], bins=bins, color=C_ATTACK, alpha=0.9, label="Attack")
        ax.set_yscale("log")
        for kind in d.loc[d["_target"] == 1, "_attack_type"].value_counts().index[:4]:
            h = hours[d["_attack_type"] == kind]
            ax.text(h.median(), ax.get_ylim()[1] * 0.5 if ax.get_ylim()[1] > 1 else 1, kind, fontsize=8, color=C_TEXT, ha="center")
        ax.set_ylabel(day, color=C_TEXT)
        style(ax)
    axes[0].legend(frameon=False, loc="upper right")
    axes[-1].set_xlabel("Hour of day (local time, after 12-hour clock repair)", color=C_TEXT)
    fig.suptitle("Flows per 5 minutes (log scale)", color=C_TEXT)
    fig.tight_layout()
    fig.savefig(out / "timeline.png", dpi=150)
    plt.close(fig)


def fig_models(results, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (proto, title) in zip(axes, (("protocol_a", "Protocol A: unseen attacks"), ("protocol_b", "Protocol B: seen attacks"))):
        s = results[proto]["summary"]
        metrics = [("f1", "F1"), ("recall", "Recall"), ("precision", "Precision")]
        x = np.arange(len(metrics))
        for i, (model, color) in enumerate(zip(MODELS, (C_RF, C_GNN, C_AE))):
            vals = [s[model]["val_f1_threshold"][k]["mean"] for k, _ in metrics]
            errs = [s[model]["val_f1_threshold"][k]["std"] for k, _ in metrics]
            ax.bar(x + (i - 1) * 0.27, vals, 0.25, yerr=errs, color=color, label=model, capsize=2, edgecolor="white", linewidth=1)
        ax.set_xticks(x, [lab for _, lab in metrics])
        ax.set_ylim(0, 1.05)
        ax.set_title(title, color=C_TEXT)
        style(ax)
    axes[0].set_ylabel("Test score (threshold chosen on validation)", color=C_TEXT)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "model_comparison.png", dpi=150)
    plt.close(fig)


def fig_label_efficiency(rows, out):
    df = pd.DataFrame(rows).groupby("label_fraction")[["f1", "recall_at_1pct_fpr"]].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    for col, color, lab in (("f1", C_GNN, "F1 (validation-tuned threshold)"), ("recall_at_1pct_fpr", C_RF, "Recall at 1 % FPR")):
        ax.errorbar(df["label_fraction"] * 100, df[(col, "mean")], yerr=df[(col, "std")].fillna(0), marker="o",
                    capsize=3, color=color, label=lab, linewidth=2)
    ax.set(xscale="log", ylim=(0, 1.02))
    ax.set_xticks([1, 5, 10, 25, 100], ["1", "5", "10", "25", "100"])
    ax.set_xlabel("Training labels used (%)", color=C_TEXT)
    ax.set_ylabel("Test score (E-GraphSAGE, protocol B)", color=C_TEXT)
    ax.legend(frameon=False, loc="lower right")
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "label_efficiency.png", dpi=150)
    plt.close(fig)


def graph_statistics(graphs, attack_names):
    """Per source host and window: distinct destinations and distinct destination ports."""
    rows = []
    for g in graphs:
        y = g.y.numpy()
        src = np.array(g.src_hosts)
        dst = np.array(g.dst_hosts)
        for label in (0, 1):
            ids = np.flatnonzero(y == label)
            if not len(ids):
                continue
            df = pd.DataFrame({"src": src[ids], "dst": dst[ids], "port": g.dst_port[ids]})
            agg = df.groupby("src").agg(destinations=("dst", "nunique"), ports=("port", "nunique"), flows=("dst", "size"))
            agg["class"] = "Attack" if label else "Benign"
            agg["window"] = g.window_id
            rows.append(agg.reset_index())
    stats = pd.concat(rows, ignore_index=True)
    summary = stats.groupby("class")[["destinations", "ports", "flows"]].agg(["mean", "median", "max"])
    return stats, {c: {f"{a}_{b}": float(summary.loc[c, (a, b)]) for a, b in summary.columns} for c in summary.index}


def fig_graph_comparison(stats, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, title in ((axes[0], "destinations", "Distinct destination hosts per source"),
                           (axes[1], "ports", "Distinct destination ports per source")):
        data = [stats.loc[stats["class"] == c, col].to_numpy() for c in ("Benign", "Attack")]
        bins = np.logspace(0, np.log10(max(2, max(d.max() for d in data if len(d)))) + 0.05, 30)
        for d, c, color in zip(data, ("Benign", "Attack"), (C_BENIGN, C_ATTACK)):
            ax.hist(d, bins=bins, density=True, alpha=0.55 if c == "Benign" else 0.75, color=color, label=c)
        ax.set(xscale="log", yscale="log")
        ax.set_title(title, color=C_TEXT)
        ax.set_xlabel("Count per source host and 5-minute window", color=C_TEXT)
        style(ax)
    axes[0].set_ylabel("Density", color=C_TEXT)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "graph_comparison.png", dpi=150)
    plt.close(fig)


def _flagged_windows(graphs, attack, scores, threshold):
    cursor = 0
    for g in graphs:
        n = len(g.y)
        yield g, scores[cursor:cursor + n], attack[cursor:cursor + n]
        cursor += n


def fig_attack_subgraph(frame, graphs, order, scores, threshold, out):
    """Two flagged test subgraphs: the attack involving the most hosts, and a port scan."""
    attack = frame["_attack_type"].to_numpy()[order]
    multi, scan = None, None
    for g, s, a in _flagged_windows(graphs, attack, scores, threshold):
        hit = np.flatnonzero((s >= threshold) & (a != "Benign"))
        if not len(hit):
            continue
        hosts = len({g.src_hosts[i] for i in hit} | {g.dst_hosts[i] for i in hit})
        if multi is None or hosts > multi[0]:
            multi = (hosts, g, s, a)
        scan_hits = hit[a[hit] == "PortScan"]
        if len(scan_hits) and (scan is None or len(set(g.dst_port[scan_hits])) > scan[0]):
            scan = (len(set(g.dst_port[scan_hits])), g, s, a)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    info = {}
    ax = axes[0]
    if multi:
        _, g, s, a = multi
        flagged = np.flatnonzero(s >= threshold)
        G = nx.DiGraph()
        for i in flagged:
            G.add_edge(g.src_hosts[i], g.dst_hosts[i], bad=G.get_edge_data(g.src_hosts[i], g.dst_hosts[i], {"bad": False})["bad"] or a[i] != "Benign")
        if len(G) > 60:
            G = G.subgraph([n for n, _ in sorted(G.degree, key=lambda x: -x[1])[:60]]).copy()
        pos = nx.spring_layout(G, seed=7, k=1.5 / np.sqrt(max(len(G), 1)))
        bad_nodes = {n for u, v, d in G.edges(data=True) if d["bad"] for n in (u, v)}
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=[C_ATTACK if d["bad"] else C_BENIGN for _, _, d in G.edges(data=True)],
                               width=1.4, arrows=True, arrowsize=10, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=[C_ATTACK if n in bad_nodes else "#8a93a1" for n in G],
                               node_size=[90 + 25 * G.degree(n) for n in G], edgecolors="white", linewidths=1)
        hub = max(G.degree, key=lambda x: x[1])[0]
        shown_labels = bad_nodes if len(bad_nodes) <= 15 else {hub}
        nx.draw_networkx_labels(G, pos, labels={n: n for n in shown_labels}, font_size=8, ax=ax)
        kind = pd.Series(a[flagged][a[flagged] != "Benign"]).value_counts().index[0]
        n_true = int((a[flagged] != "Benign").sum())
        ax.set_title(f"{kind}: {len(bad_nodes)} hosts involved in one 5-minute test window\n"
                     f"{len(flagged):,} flagged flows, {n_true:,} truly malicious (red), {len(flagged) - n_true:,} false alarms (blue)",
                     color=C_TEXT, fontsize=10)
        info["most_hosts"] = {"window": int(g.window_id), "attack_type": kind, "hosts": len(bad_nodes),
                              "flagged_flows": int(len(flagged)), "truly_malicious": n_true}
    ax.set_axis_off()
    ax = axes[1]
    if scan:
        _, g, s, a = scan
        hit = np.flatnonzero((s >= threshold) & (a == "PortScan"))
        src, dst = g.src_hosts[hit[0]], g.dst_hosts[hit[0]]
        ports = np.unique(g.dst_port[hit])
        shown = np.sort(np.random.default_rng(0).choice(ports, size=min(80, len(ports)), replace=False))
        G = nx.Graph()
        G.add_node(src)
        for p in shown:
            G.add_edge(src, f":{p}")
        pos = nx.spring_layout(G, seed=3, k=0.35)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=C_ATTACK, width=0.8, alpha=0.7)
        nx.draw_networkx_nodes(G, pos, nodelist=[src], node_color=C_ATTACK, node_size=500, ax=ax, edgecolors="white")
        nx.draw_networkx_nodes(G, pos, nodelist=[n for n in G if n != src], node_color="#8a93a1", node_size=40, ax=ax)
        nx.draw_networkx_labels(G, pos, labels={src: src}, font_size=9, ax=ax)
        ax.set_title(f"PortScan: {src} → {dst} on {len(ports):,} distinct ports in 5 minutes\n"
                     f"({len(hit):,} flagged flows; {len(shown)} ports drawn)", color=C_TEXT, fontsize=10)
        info["port_scan"] = {"window": int(g.window_id), "source": src, "target": dst, "distinct_ports": int(len(ports)),
                             "flagged_flows": int(len(hit))}
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out / "attack_subgraph.png", dpi=150)
    plt.close(fig)
    return info


# ---------------------------------------------------------------- main
def run_experiments(config_path: str | Path, quick: bool = False) -> dict:
    started = time.time()
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    d, e = cfg["data"], cfg["experiments"]
    seeds = [int(s) for s in e["seeds"]][: (1 if quick else None)]
    out = Path(e.get("output_dir", "results"))
    out.mkdir(parents=True, exist_ok=True)
    log(f"Loading {d['path']}")
    full = load_cic(d["path"])
    full = add_windows(full, int(d["window_seconds"]))
    n_full, n_full_attack = len(full), int(full["_target"].sum())
    full_attack_counts = full["_attack_type"].value_counts().to_dict()
    fig_timeline(full, out)
    schedule = (full[full["_target"] == 1].groupby(["_day", "_attack_type"])["_datetime"]
                .agg(["min", "max", "size"]).reset_index())
    schedule["min"], schedule["max"] = schedule["min"].dt.strftime("%H:%M"), schedule["max"].dt.strftime("%H:%M")
    frame = window_benign_cap(full, d.get("benign_cap_per_window"), int(d.get("sample_seed", 17)))
    del full
    gc.collect()
    if quick:
        frame = window_benign_cap(frame, 300, 17)
    log(f"Flows used: {len(frame):,} of {n_full:,} (all {n_full_attack:,} attack flows kept)")

    results = {}
    results["protocol_a"], _ = run_protocol("Protocol A", frame, split_protocol_a(frame), cfg, seeds)
    gc.collect()
    splits_b = split_protocol_b(frame, float(cfg["split"]["train_fraction"]), float(cfg["split"]["validation_fraction"]))
    results["protocol_b"], extra = run_protocol("Protocol B", frame, splits_b, cfg, seeds, keep_scores=True)
    fig_models(results, out)

    stats, graph_summary = graph_statistics(extra["graphs"]["test"], None)
    fig_graph_comparison(stats, out)
    subgraph = fig_attack_subgraph(frame, extra["graphs"]["test"], extra["order"]["test"],
                                   extra["scores"]["E-GraphSAGE"], extra["scores"]["gnn_threshold"], out)
    write_json(out / "graph_analysis.json", {"split": "protocol B test windows", "per_source_summary": graph_summary,
                                             "attack_subgraph": subgraph})

    le_rows = label_efficiency(frame, extra["graphs"], extra["order"], extra["X"], cfg, seeds)
    write_json(out / "label_efficiency.json", le_rows)
    fig_label_efficiency(le_rows, out)
    del extra
    gc.collect()

    ab_rows = ablation(frame, splits_b, cfg, seeds) if e.get("run_ablation", True) else []
    write_json(out / "ablation.json", ab_rows)

    metrics = {
        "dataset": {"source": "CIC-IDS2017 GeneratedLabelledFlows (TrafficLabelling CSVs)", "flows_total": n_full,
                    "attack_flows_total": n_full_attack, "flows_used": int(len(frame)),
                    "benign_cap_per_window": d.get("benign_cap_per_window"), "window_seconds": int(d["window_seconds"]),
                    "attack_types_total": {str(k): int(v) for k, v in full_attack_counts.items()},
                    "attack_schedule_after_repair": schedule.to_dict(orient="records")},
        "seeds": seeds,
        **results,
        "runtime_seconds": float(time.time() - started),
        "quick_mode": quick,
    }
    write_json(out / "metrics.json", metrics)
    log(f"Done in {metrics['runtime_seconds'] / 60:.1f} min; results in {out}/")
    return metrics
