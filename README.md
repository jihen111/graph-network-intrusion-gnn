# Graph-based Network Intrusion Detection with GNNs

Detecting attacks in network traffic by modelling it as a graph: hosts are nodes, flows are edges, and each 5-minute window becomes one graph. Three detectors are compared on CIC-IDS2017: a flow-only Random Forest, a supervised E-GraphSAGE edge classifier, and a graph autoencoder trained on benign traffic only. Two protocols are evaluated: attacks seen in training (B) and attacks never seen in training (A).

**Results are not in this README yet.** Run the pipeline once; `scripts/write_report.py` then rewrites this file with the real numbers from `results/`.

```bash
python run_all.py          # downloads the data, runs everything, writes README/SUMMARY, packages a ZIP
```

On Google Colab, open `RUN_ON_COLAB.ipynb` and choose *Run all*. Design choices are in `DECISIONS.md`, interview notes in `LEARNING.md`.
