"""Generate the EDA and graph-analysis notebooks (run them with nbconvert --execute)."""
from pathlib import Path
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
META = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}
Path("notebooks").mkdir(exist_ok=True)

SETUP = """import os, sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
import yaml, numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Image, display
cfg = yaml.safe_load(Path('configs/experiment.yaml').read_text())
metrics = json.loads(Path('results/metrics.json').read_text())"""

eda = nbf.v4.new_notebook(metadata=META)
eda.cells = [
    md("""# 01 — CIC-IDS2017 flows: exploration and timestamp check

CIC-IDS2017 records five working days (3–7 July 2017) of a small enterprise network. Each row is a network flow with its endpoints, a timestamp, about 80 statistics and a label. This notebook checks the data before modelling, with a focus on the timestamps, because the graph windows and both evaluation protocols depend on them."""),
    code(SETUP + """
from src.data import load_cic, find_column
flows = load_cic(cfg['data']['path'])
print(f"{len(flows):,} labelled flows, {int(flows['_target'].sum()):,} attacks ({flows['_target'].mean():.1%})")"""),
    md("## Attack types per day\n\nEach attack type happens on a single day. This is why a purely chronological split (protocol A) tests attacks that were never seen in training."),
    code("""table = pd.crosstab(flows['_attack_type'], flows['_day']).reindex(columns=['Monday','Tuesday','Wednesday','Thursday','Friday'], fill_value=0)
display(table.sort_values(list(table.columns), ascending=False))"""),
    md("""## Timestamp repair

Several CSV files write afternoon times on a 12-hour clock without AM/PM (for example `7/7/2017 3:56` for 15:56). The loader shifts every hour before 08:00 by 12 hours, since captures only cover working hours. The table below compares the repaired attack times with the schedule published by the dataset authors; matching windows confirm the repair."""),
    code("""documented = {
 ('Tuesday','FTP-Patator'):'09:20–10:20', ('Tuesday','SSH-Patator'):'14:00–15:00',
 ('Wednesday','DoS slowloris'):'09:47–10:10', ('Wednesday','DoS Slowhttptest'):'10:14–10:35', ('Wednesday','DoS Hulk'):'10:43–11:00',
 ('Wednesday','DoS GoldenEye'):'11:10–11:23', ('Wednesday','Heartbleed'):'15:12–15:32',
 ('Thursday','Web Attack - Brute Force'):'09:20–10:00', ('Thursday','Web Attack - XSS'):'10:15–10:35', ('Thursday','Web Attack - Sql Injection'):'10:40–10:42',
 ('Thursday','Infiltration'):'14:19–15:45', ('Friday','Bot'):'10:02–11:02', ('Friday','PortScan'):'13:55–14:35', ('Friday','DDoS'):'15:56–16:16'}
sched = pd.DataFrame(metrics['dataset']['attack_schedule_after_repair'])
sched['documented (CIC)'] = [documented.get((d, a), '–') for d, a in zip(sched['_day'], sched['_attack_type'])]
sched = sched.rename(columns={'_day':'day','_attack_type':'attack','min':'first flow','max':'last flow','size':'flows'})
display(sched)
display(Image(filename='results/timeline.png'))"""),
    md("## Top talkers\n\nIP addresses are used only to build the graph (hosts become nodes); they never enter the model as numeric features."),
    code("""display(flows['Source IP'].value_counts().head(8).rename('outgoing flows').to_frame())
display(flows.loc[flows['_target']==1, 'Source IP'].value_counts().head(5).rename('outgoing attack flows').to_frame())"""),
    md("## Heavy-tailed features\n\nCounters such as bytes, packets and durations span many orders of magnitude. The preprocessing applies `log1p` to them, then standard scaling, fitted on training flows only."),
    code("""cols = ['Flow Duration', 'Total Fwd Packets', 'Total Length of Fwd Packets', 'Flow Bytes/s']
fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
for ax, c in zip(axes, cols):
    v = flows[c].replace([np.inf, -np.inf], np.nan).dropna().clip(lower=0)
    ax.hist(np.log1p(v), bins=60, color='#2a78d6'); ax.set_title(f'log1p({c})', fontsize=9); ax.set_yscale('log')
plt.tight_layout()"""),
    md("## Splits used by the experiments"),
    code("""rows = []
for proto in ('protocol_a', 'protocol_b'):
    for part, s in metrics[proto]['splits'].items():
        rows.append({'protocol': proto, 'split': part, 'flows': s['flows'], 'attacks': s['attacks'], 'windows': s['windows'],
                     'attack types': ', '.join(k for k in s['attack_types'] if k != 'Benign')})
display(pd.DataFrame(rows))"""),
]
nbf.write(eda, "notebooks/01_eda.ipynb")

graph = nbf.v4.new_notebook(metadata=META)
graph.cells = [
    md("""# 02 — Graph analysis: benign vs attack traffic

Each 5-minute window becomes a directed graph: hosts are nodes and flows are edges carrying the flow features. This notebook builds one real window graph, compares benign and attack sources on graph measures, and shows the subgraph of flows flagged by E-GraphSAGE."""),
    code(SETUP + """
import networkx as nx
from src.data import load_cic, add_windows
from src.build_graph import build_graph_windows, STRUCT_NAMES
flows = add_windows(load_cic(cfg['data']['path']), cfg['data']['window_seconds'])
scan = flows[(flows['_attack_type'] == 'PortScan')]
w = scan['_window'].value_counts().index[0]
window = flows[flows['_window'] == w].reset_index(drop=True)
g = build_graph_windows(window, np.zeros((len(window), 1), dtype=np.float32))[0]
print(f"Window starting {window['_datetime'].min():%A %H:%M}: {len(window):,} flows, {g.struct_x.shape[0]} hosts, "
      f"{int(g.y.sum()):,} PortScan flows")"""),
    md("## Structural node features in that window\n\nThese six features are computed from the graph only. A scanner stands out immediately: it contacts many distinct ports."),
    code("""hosts = sorted(set(g.src_hosts) | set(g.dst_hosts))
feats = pd.DataFrame(np.expm1(g.struct_x.numpy()).round().astype(int), index=hosts, columns=[n.replace('log_', '') for n in STRUCT_NAMES])
display(feats.sort_values('distinct_dst_ports', ascending=False).head(8))"""),
    md("## Benign vs attack sources (protocol B test windows)"),
    code("""ga = json.loads(Path('results/graph_analysis.json').read_text())
display(pd.DataFrame(ga['per_source_summary']).T.round(1))
display(Image(filename='results/graph_comparison.png'))"""),
    md("## Attack subgraph flagged by the model\n\nFlows scored above the validation-tuned threshold in one test window. Red edges are truly malicious flows, blue edges are false alarms. The red node is the most connected host."),
    code("""print(json.dumps(ga['attack_subgraph'], indent=1))
display(Image(filename='results/attack_subgraph.png'))"""),
    md("""## From this project to lateral movement

Lateral movement is an attacker moving from one internal machine to the next. The same building blocks apply: hosts as nodes, timestamped host-to-host events (authentications, remote sessions, process launches) as edges, and short time windows. Two changes are needed. First, provenance graphs from system logs have typed nodes (process, file, socket, user) and typed edges (read, write, fork, connect), so a heterogeneous or temporal GNN (for example R-GCN or TGN) fits better than the homogeneous GraphSAGE used here. Second, an attack is a path across several nodes and windows, so detections must be joined: link anomalous edges that share a host and follow each other in time, and score the resulting chains. Evaluating that needs path-level labels (for example the LANL or DARPA OpTC datasets) and splits by time and by host."""),
]
nbf.write(graph, "notebooks/02_graph_analysis.ipynb")
print("Wrote notebooks/01_eda.ipynb and notebooks/02_graph_analysis.ipynb")
