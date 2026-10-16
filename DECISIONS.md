# Design decisions (version 2)

## Dataset

* **CIC-IDS2017 `GeneratedLabelledFlows`.** It keeps source/destination IPs, ports and timestamps, which are needed to build host graphs and time windows. NF-UNSW-NB15-v2 and the CIC `MachineLearningCSV` files were rejected in v1 because they lack endpoints and timestamps.
* **Labels:** rows with an empty label are dropped, never counted as attacks. Attack names are normalised (the CSVs use a non-ASCII dash in `Web Attack – …`).

## Timestamps

* The CSVs write dates as day/month/year. Several files write afternoon times on a 12-hour clock without AM/PM (`7/7/2017 3:56` is 15:56). Captures only cover working hours (about 08:55–17:00), so any hour before 08:00 is shifted by 12 hours.
* The repair is checked, not assumed: `results/metrics.json` stores the first and last time of every attack type after repair, and notebook 01 compares them with the schedule published by the dataset authors.
* Times are local lab time; no timezone conversion is applied (none is needed for windows or splits).

## Graph construction

* **One graph per 5-minute window.** Short enough for attack structure to stand out, long enough to contain many flows. Windows never cross a split.
* **Nodes are hosts (IPs), edges are flows.** Endpoint nodes (IP:port) would split each host into many low-degree nodes and hide fan-out; ports are kept as coarse buckets in the flow features and as distinct-port counts in the node features.
* **Node features = 6 structural features + mean incident flow features.** Structural features: log out-degree, in-degree, distinct destinations, distinct sources, distinct destination ports contacted, distinct source ports seen. They are computed from the window's topology and expose scanning (many ports) and DDoS (many sources to one target).
* **Message passing uses both directions** of every flow; each flow is classified once from [source embedding ‖ destination embedding ‖ flow features].

## Data reduction (topology-preserving)

* v1 sampled 40,000 flows at random over five days, which left about 600 flows per hourly graph and destroyed scan and DDoS patterns.
* v2 keeps **every attack flow and every window**, and keeps at most `benign_cap_per_window` benign flows per window (random). Attack structure is complete; benign graphs are thinner than reality, which is stated as a limitation.

## Evaluation protocols

* **Protocol A, unseen attacks:** train Monday–Wednesday, validate Thursday, test Friday. Friday's Bot, PortScan and DDoS never appear in training. Supervised models are expected to struggle; this is the zero-day setting where the benign-only autoencoder should help.
* **Protocol B, seen attacks:** each window is labelled by its dominant attack type; the windows of each type are split chronologically 60/20/20, so every type appears in train, validation and test. Types with fewer than 3 windows stay in train. Neighbouring windows of one attack episode are correlated, so B is the optimistic setting.
* Label efficiency and the ablation run on protocol B, where supervised learning is meaningful.

## Thresholds and metrics

* Every model's threshold is chosen on **validation** data: an F1 threshold, and the threshold giving a 1 % validation false-positive rate. Test labels are used once.
* **F1 threshold rule (changed after the first full run).** The first full run used the single F1 arg-max on validation. For E-GraphSAGE the validation F1 curve is flat near its maximum, and the arg-max sometimes landed on an extreme value (0.9994) that flagged only a third of the test attacks although the same model reached F1 0.99 at 0.5 and PR-AUC 0.99. The rule is now the middle of the score range whose validation F1 is within 0.01 of the best. The rule uses validation data only; it was changed because of a visible selection artefact, and the 1 % FPR operating point, which this change does not affect, is reported alongside.
* **Recall at 1 % false positives is the headline metric** for unseen attacks: Friday's test day is about 60 % attack traffic, which inflates F1 for a detector with many false alarms.
* Validation and test scores of every model and seed are saved in `results/scores_*.npz`, so operating points can be re-derived without retraining.
* The fixed 0.5 threshold is still stored for the supervised models, for comparison with v1.
* Reported: precision, recall, F1, false-positive rate, PR-AUC, recall at 1 % FPR, and per attack type the share of flows flagged (detection rate).

## Models

* **Random Forest (flow-only):** 100 trees, balanced class weights, each tree on 30 % of the training flows (speed on CPU).
* **E-GraphSAGE:** two edge-aware GraphSAGE layers (mean aggregation, flow features inside each message), hidden size 64, class-weighted cross-entropy, up to 20 epochs with early stopping on validation loss (patience 3), best checkpoint restored. Window order is shuffled every epoch.
* **Graph autoencoder (benign-only):** trained on benign edges only; endpoint embeddings come from the structural node features (recomputed without attack edges), and the decoder reconstructs the flow's features. The target flow's own features are never an input, so it cannot copy them. Anomaly score = reconstruction error.
* Seeds 11, 23 and 37 for every model; mean ± standard deviation reported.

## Compute

* Everything runs on CPU; a GPU is used automatically when available (`RUN_ON_COLAB.ipynb`).
* `python -m src.train --quick` runs one seed with a small benign cap, for smoke tests only.
* `scripts/make_smoke_data.py` writes a tiny synthetic CIC-like dataset used only to test that the pipeline runs; no metric from it is ever reported.

## Figures (after the full run)

* `model_comparison.png` shows the operating point a security team would use (threshold set for 1 % false positives on validation), not F1: on Friday about 60 % of flows are attacks, which rewards a detector that raises many false alarms.
* `attack_subgraph.png` shows the port scan only. The "most hosts involved" panel always found 2 hosts, because every external attack in CIC-IDS2017 crosses one NAT gateway; the code now draws that panel only when an attack involves at least 3 hosts. Both figures were regenerated from the saved results without retraining.
