"""Write README.md and SUMMARY.md from results/*.json so every number comes from the run.

Usage: python scripts/write_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("results")
MODELS = ["Random Forest (flow-only)", "E-GraphSAGE", "Graph Autoencoder (benign-only)"]
SHORT = {"Random Forest (flow-only)": "Random Forest", "E-GraphSAGE": "E-GraphSAGE", "Graph Autoencoder (benign-only)": "Graph autoencoder"}


def load(name):
    return json.loads((R / name).read_text(encoding="utf-8"))


def ms(d, k, digits=3):
    return f"{d[k]['mean']:.{digits}f} ± {d[k]['std']:.{digits}f}"


def pct(x):
    return f"{100 * x:.1f} %"


def fr(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def main_table(proto):
    rows = ["| Model | Precision | Recall | F1 | False-positive rate | PR-AUC | Recall at 1 % FPR |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for m in MODELS:
        s = proto["summary"][m]
        v, f = s["val_f1_threshold"], s["val_fpr_1pct_threshold"]
        rows.append(f"| {m} | {ms(v, 'precision')} | {ms(v, 'recall')} | **{ms(v, 'f1')}** | {ms(v, 'false_positive_rate', 4)} | "
                    f"{ms(v, 'pr_auc')} | {ms(f, 'recall')} (FPR {f['false_positive_rate']['mean']:.4f}) |")
    return "\n".join(rows)


def attack_table(proto, point="val_fpr_1pct_threshold"):
    det = {m: proto["detection_rate_per_attack"][m][point] for m in MODELS}
    names = sorted(det[MODELS[0]], key=lambda n: -det[MODELS[0]][n]["support"])
    rows = ["| Attack type (test flows) | " + " | ".join(SHORT[m] for m in MODELS) + " |",
            "|---|" + "---:|" * len(MODELS)]
    for n in names:
        rows.append(f"| {n} ({det[MODELS[0]][n]['support']:,}) | " + " | ".join(pct(det[m][n]["mean"]) for m in MODELS) + " |")
    return "\n".join(rows)


def split_table(metrics):
    rows = ["| Protocol | Split | Flows | Attack flows | Windows | Attack types |", "|---|---|---:|---:|---:|---|"]
    for key, name in (("protocol_a", "A – unseen attacks"), ("protocol_b", "B – seen attacks")):
        for part, s in metrics[key]["splits"].items():
            types = ", ".join(k for k in s["attack_types"] if k != "Benign") or "–"
            rows.append(f"| {name} | {part} | {s['flows']:,} | {s['attacks']:,} | {s['windows']} | {types} |")
    return "\n".join(rows)


def best(proto, point="val_f1_threshold", key="f1"):
    vals = {m: proto["summary"][m][point][key]["mean"] for m in MODELS}
    return max(vals, key=vals.get), vals


def main():
    metrics = load("metrics.json")
    le = pd.DataFrame(load("label_efficiency.json"))
    ab = pd.DataFrame(load("ablation.json"))
    ga = load("graph_analysis.json")
    A, B = metrics["protocol_a"], metrics["protocol_b"]
    d = metrics["dataset"]
    best_a, va = best(A)
    best_b, vb = best(B)
    rec_a = {m: A["summary"][m]["val_fpr_1pct_threshold"]["recall"]["mean"] for m in MODELS}
    fpr_a = {m: A["summary"][m]["val_fpr_1pct_threshold"]["false_positive_rate"]["mean"] for m in MODELS}
    rec_b = {m: B["summary"][m]["val_fpr_1pct_threshold"]["recall"]["mean"] for m in MODELS}
    prev_a = A["splits"]["test"]["attacks"] / A["splits"]["test"]["flows"]
    det_a = A["detection_rate_per_attack"]["E-GraphSAGE"]["val_fpr_1pct_threshold"]
    det_a_txt = ", ".join(f"{k} {pct(v['mean'])}" for k, v in sorted(det_a.items(), key=lambda kv: -kv[1]["support"]))
    le_s = le.groupby("label_fraction")[["f1", "recall_at_1pct_fpr"]].agg(["mean", "std"])
    ab_s = ab.groupby("variant")[["f1", "pr_auc", "recall_at_1pct_fpr"]].agg(["mean", "std"])
    lo, hi = float(le_s.index.min()), float(le_s.index.max())
    mid = 0.1 if 0.1 in le_s.index else lo
    sub = ga.get("attack_subgraph", {})
    src = ga["per_source_summary"]
    nat_txt = (f", but only {src['Attack']['destinations_mean']:.1f} destination host on average: in CIC-IDS2017 external attacks reach the "
               "victims through one NAT gateway (172.16.0.1), so attacks show up as many ports between few hosts."
               if src["Attack"]["destinations_mean"] < 2 else f", and {src['Attack']['destinations_mean']:.1f} distinct destination hosts.")
    ae_fpr = A["summary"]["Graph Autoencoder (benign-only)"]["val_f1_threshold"]["false_positive_rate"]["mean"]
    ae_txt = (f"* **Caveat on the autoencoder:** its F1 in protocol A ({va['Graph Autoencoder (benign-only)']:.3f}) is inflated by the test day's "
              f"attack prevalence ({pct(prev_a)} of Friday's flows are attacks) and comes with a {pct(ae_fpr)} false-positive rate. "
              "Recall at a usable false-positive budget is the fairer comparison.\n" if prev_a > 0.3 and ae_fpr > 0.05 else "")
    minutes = metrics["runtime_seconds"] / 60
    seeds = ", ".join(map(str, metrics["seeds"]))

    gnn_vs_rf_b = vb["E-GraphSAGE"] - vb["Random Forest (flow-only)"]
    finding_b = ("E-GraphSAGE beats the flow-only Random Forest" if gnn_vs_rf_b > 0.005 else
                 "E-GraphSAGE and the flow-only Random Forest are within 0.005 F1" if abs(gnn_vs_rf_b) <= 0.005 else
                 "the flow-only Random Forest beats E-GraphSAGE")

    le_rows = "\n".join(f"| {100 * f:g} % | {r[('f1', 'mean')]:.3f} ± {r[('f1', 'std')]:.3f} | "
                        f"{r[('recall_at_1pct_fpr', 'mean')]:.3f} ± {r[('recall_at_1pct_fpr', 'std')]:.3f} |"
                        for f, r in le_s.iterrows())
    ab_rows = "\n".join(f"| {v} | {r[('f1', 'mean')]:.3f} ± {r[('f1', 'std')]:.3f} | {r[('pr_auc', 'mean')]:.3f} | "
                        f"{r[('recall_at_1pct_fpr', 'mean')]:.3f} |" for v, r in ab_s.iterrows())
    sched = pd.DataFrame(d["attack_schedule_after_repair"])
    sched_rows = "\n".join(f"| {r['_day']} | {r['_attack_type']} | {r['min']}–{r['max']} | {int(r['size']):,} |" for _, r in sched.iterrows())

    readme = f"""# Graph-based Network Intrusion Detection with GNNs

Detecting attacks in network traffic by modelling it as a graph. Hosts are nodes, flows are edges, and each 5-minute window of traffic becomes one graph. The project compares three detectors on **CIC-IDS2017** (five days of traffic from a lab enterprise network, {d['flows_total']:,} labelled flows):

* a **Random Forest** that sees each flow on its own (no graph),
* **E-GraphSAGE**, a supervised graph neural network that classifies each flow (edge) from its own features and its endpoints' neighbourhoods,
* a **graph autoencoder** trained on benign traffic only, which flags flows it cannot reconstruct (no attack labels needed).

It also measures how much labelled data the GNN needs, which inputs matter (ablation), and how attack traffic differs structurally from benign traffic.

## Key results

All numbers below are produced by `python -m src.train` and stored in `results/`. Mean ± standard deviation over seeds {seeds}. Each model's alert threshold is chosen on the validation split, then applied once to the test split.

* **Unseen attacks (protocol A, test = Friday's Bot, PortScan and DDoS):** with the alert threshold set for 1 % false positives on validation, E-GraphSAGE catches **{pct(rec_a['E-GraphSAGE'])}** of the attack flows ({det_a_txt}; test false-positive rate {pct(fpr_a['E-GraphSAGE'])}), against {pct(rec_a['Random Forest (flow-only)'])} for the flow-only Random Forest ({pct(fpr_a['Random Forest (flow-only)'])}) and {pct(rec_a['Graph Autoencoder (benign-only)'])} for the benign-only autoencoder ({pct(fpr_a['Graph Autoencoder (benign-only)'])}).
* **Seen attacks (protocol B):** F1 is {vb['Random Forest (flow-only)']:.3f} for the Random Forest, {vb['E-GraphSAGE']:.3f} for E-GraphSAGE and {vb['Graph Autoencoder (benign-only)']:.3f} for the autoencoder (validation-tuned threshold). At 1 % false positives, recall is {pct(rec_b['Random Forest (flow-only)'])} and {pct(rec_b['E-GraphSAGE'])} respectively.
* **Few labels are enough on seen attacks:** with {100 * lo:g} % of the training labels E-GraphSAGE reaches F1 {le_s.loc[lo, ('f1', 'mean')]:.3f}, against {le_s.loc[hi, ('f1', 'mean')]:.3f} with {100 * hi:g} %.
* **Structure carries the signal:** a GNN that sees only graph structure (degrees, distinct peers and ports, no flow statistics) reaches PR-AUC {ab_s.loc['graph structure only', ('pr_auc', 'mean')]:.3f} on protocol B. In test windows an attacking source contacts on average {src['Attack']['ports_mean']:.1f} distinct destination ports, against {src['Benign']['ports_mean']:.1f} for a benign source{nat_txt}
{ae_txt}
![Model comparison](results/model_comparison.png)

## Data and evaluation protocols

* **Data:** CIC-IDS2017 `GeneratedLabelledFlows` CSVs (they keep source/destination IPs, ports and timestamps, which graph construction needs). Source: [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html), public mirror on [Hugging Face](https://huggingface.co/datasets/bencorn/CICIDS2017).
* **Timestamps:** several CSVs write afternoon times on a 12-hour clock without AM/PM. Hours before 08:00 are shifted by 12 hours; the repaired attack times match the schedule published by the dataset authors (table below, and notebook 01).
* **Topology-preserving reduction:** every attack flow and every window is kept; inside each 5-minute window at most {d['benign_cap_per_window']:,} benign flows are kept at random. {d['flows_used']:,} of {d['flows_total']:,} flows are used (all {d['attack_flows_total']:,} attack flows). Unlike random flow sampling, this keeps scans and DDoS stars intact.
* **Protocol A, unseen attacks:** train Monday–Wednesday, validate Thursday, test Friday. Friday's attacks (Bot, PortScan, DDoS) never appear in training, so this measures generalisation to new attack types (a zero-day setting).
* **Protocol B, seen attacks:** each window gets its dominant attack type, and the windows of each type are split chronologically 60/20/20. Every attack type appears in train, validation and test; windows never cross splits. Neighbouring windows of one attack episode can be similar, so protocol B is the optimistic setting.
* **No leakage:** IPs, raw ports, timestamps and labels are never model inputs (IPs only define the graph; ports enter as coarse service/client buckets). Preprocessing is fitted on training flows only.

{split_table(metrics)}

Attack times after the 12-hour clock repair:

| Day | Attack | First – last flow | Flows |
|---|---|---|---:|
{sched_rows}

![Traffic timeline](results/timeline.png)

## Results

### Protocol A – unseen attacks (test = Friday)

{main_table(A)}

Share of each attack type's test flows flagged, with the threshold set for 1 % validation false positives:

{attack_table(A)}

### Protocol B – seen attacks

{main_table(B)}

{attack_table(B)}

### Label efficiency (E-GraphSAGE, protocol B)

| Training labels used | Test F1 | Recall at 1 % FPR |
|---:|---:|---:|
{le_rows}

![Label efficiency](results/label_efficiency.png)

### Ablation (E-GraphSAGE, protocol B)

| Variant | Test F1 | PR-AUC | Recall at 1 % FPR |
|---|---:|---:|---:|
{ab_rows}

"Graph structure only" sees no flow statistics at all: only degrees, distinct peers and distinct ports per host.

### Benign vs attack graphs, and an extracted attack subgraph

| Source hosts per 5-min window | Distinct destinations (mean / max) | Distinct destination ports (mean / max) | Flows (mean) |
|---|---:|---:|---:|
| Benign | {src['Benign']['destinations_mean']:.1f} / {src['Benign']['destinations_max']:.0f} | {src['Benign']['ports_mean']:.1f} / {src['Benign']['ports_max']:.0f} | {src['Benign']['flows_mean']:.1f} |
| Attack | {src['Attack']['destinations_mean']:.1f} / {src['Attack']['destinations_max']:.0f} | {src['Attack']['ports_mean']:.1f} / {src['Attack']['ports_max']:.0f} | {src['Attack']['flows_mean']:.1f} |

![Graph comparison](results/graph_comparison.png)

""" + (f"""The figure below shows flows flagged by E-GraphSAGE in two protocol B test windows. Left: **{sub['most_hosts']['attack_type']}**, the attack involving the most hosts ({sub['most_hosts']['hosts']} hosts; {sub['most_hosts']['truly_malicious']:,} of {sub['most_hosts']['flagged_flows']:,} flagged flows truly malicious). Right: a port scan from `{sub['port_scan']['source']}` to `{sub['port_scan']['target']}` on {sub['port_scan']['distinct_ports']:,} distinct ports within 5 minutes.

![Attack subgraphs](results/attack_subgraph.png)
""" if sub.get("most_hosts") and sub.get("port_scan") else "") + f"""
## From flows to provenance graphs and lateral movement

This project works on network flows. Moving to system-log provenance graphs (processes, files, sockets as typed nodes; read/write/fork/connect as typed edges) calls for heterogeneous or temporal GNNs (R-GCN, TGN) instead of the homogeneous GraphSAGE used here. Lateral movement is a path across several hosts and windows, so per-edge detections must be joined: link anomalous edges that share a host and follow each other in time, then score the chain. Evaluating that needs path-level labels (for example LANL or DARPA OpTC) and splits by time and by host. Notebook 02 develops this.

## Limitations

* CIC-IDS2017 is a 2017 lab capture with known labelling artefacts; results do not transfer as-is to a real network.
* Flows, not system logs: no provenance information.
* Benign traffic is thinned inside each window ({d['benign_cap_per_window']:,} flows max), so benign graphs are sparser than in reality; attack traffic is complete.
* Protocol A has a single test day with three attack types; protocol B is optimistic because neighbouring windows of one attack are correlated.
* The autoencoder is a simple design (it reconstructs flow features from structural embeddings); stronger unsupervised graph models exist.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/download_data.py              # CIC-IDS2017 GeneratedLabelledFlows (~270 MB)
python -m src.train --config configs/experiment.yaml
python scripts/create_notebooks.py && jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
python scripts/write_report.py               # regenerates this README from results/
python -m unittest discover -s tests -v
```

Or run everything at once: `python run_all.py` (also used by `RUN_ON_COLAB.ipynb`). The full run took {minutes:.0f} minutes.

## Repository map

* `src/data.py` – loading, timestamp repair, topology-preserving reduction, protocols A and B
* `src/preprocess.py` – train-only imputation, log transform, scaling, categorical and port encoding
* `src/build_graph.py` – one host graph per window, structural node features
* `src/models/` – Random Forest baseline, E-GraphSAGE, benign-only graph autoencoder
* `src/metrics.py` – validation-chosen thresholds, test metrics, per-attack detection rate
* `src/experiments/run.py` – all experiments and figures
* `notebooks/` – data exploration and timestamp check; graph analysis and lateral-movement discussion
* `DECISIONS.md` – design choices; `LEARNING.md` – interview questions and answers

## References

* Sharafaldin, Lashkari, Ghorbani (2018). *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.* ICISSP.
* Lo, Layeghy, Sarhan, Gallagher, Portmann (2022). *E-GraphSAGE: A Graph Neural Network based Intrusion Detection System for IoT.* NOMS.
* Hamilton, Ying, Leskovec (2017). *Inductive Representation Learning on Large Graphs.* NeurIPS.
"""
    Path("README.md").write_text(readme, encoding="utf-8")

    summary = f"""# Run summary (version 2)

**Run:** {minutes:.0f} minutes, seeds {seeds}, {d['flows_used']:,} flows used of {d['flows_total']:,} (all {d['attack_flows_total']:,} attack flows), 5-minute graph windows.

## What changed since version 1 and why

| Problem in v1 | Fix in v2 |
|---|---|
| 40,000 flows sampled at random over 5 days left about 600 flows per hourly graph, so scans and DDoS patterns disappeared. | Every attack flow and every 5-minute window is kept; only benign flows are thinned inside each window. |
| The chronological 60/20/20 split put PortScan and DDoS only in the test set, so supervised models scored 0 on them by design. | Two protocols: A (unseen attacks, by day) and B (seen attacks, per-type chronological split of windows). |
| Random Forest and E-GraphSAGE used a fixed 0.5 threshold while the autoencoder's was tuned. | Every threshold is chosen on validation (max F1), plus a 1 % false-positive operating point. |
| Afternoon timestamps on a 12-hour clock were not checked. | Repaired and checked against the published attack schedule. |
| Node features were only incident-flow averages. | Six structural node features (degrees, distinct peers, distinct ports) added; they expose scanning. |
| Notebooks had no outputs. | Notebooks are executed with outputs. |

## Results (test split, threshold chosen on validation)

| | Protocol A F1 (unseen) | Protocol B F1 (seen) |
|---|---:|---:|
""" + "\n".join(f"| {m} | {A['summary'][m]['val_f1_threshold']['f1']['mean']:.3f} | {B['summary'][m]['val_f1_threshold']['f1']['mean']:.3f} |" for m in MODELS) + f"""

Label efficiency (protocol B, E-GraphSAGE): F1 {le_s.loc[lo, ('f1', 'mean')]:.3f} with {100 * lo:g} % of labels, {le_s.loc[mid, ('f1', 'mean')]:.3f} with {100 * mid:g} %, {le_s.loc[hi, ('f1', 'mean')]:.3f} with {100 * hi:g} %.

Time per seed (seconds): """ + ", ".join(f"{SHORT[m]} {B['seconds_per_seed'][m]:.0f}" for m in MODELS) + f""" (protocol B).

## Lignes CV (français, chiffres réels)

1. **Détection d'intrusions réseau par GNN** – pipeline PyTorch Geometric sur CIC-IDS2017 ({fr(d['flows_used'] / 1e6, 1)} M de flux, un graphe d'hôtes par fenêtre de 5 min, attributs structurels) : E-GraphSAGE, autoencodeur de graphes non supervisé et Random Forest de référence.
2. Sur des attaques jamais vues à l'entraînement, E-GraphSAGE détecte {fr(100 * rec_a['E-GraphSAGE'], 0)} % des flux malveillants (seuil réglé pour 1 % de faux positifs en validation ; {fr(100 * fpr_a['E-GraphSAGE'], 1)} % en test) contre {fr(100 * rec_a['Random Forest (flow-only)'], 0)} % pour le Random Forest ; F1 {fr(le_s.loc[lo, ('f1', 'mean')])} avec seulement {100 * lo:g} % des labels ; extraction de sous-graphes d'attaque (NetworkX).
"""
    Path("SUMMARY.md").write_text(summary, encoding="utf-8")
    print("Wrote README.md and SUMMARY.md from results/")


if __name__ == "__main__":
    main()
