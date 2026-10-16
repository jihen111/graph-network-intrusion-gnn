"""Generate clean, reproducible EDA and graph-analysis Jupyter notebooks."""
from pathlib import Path
import nbformat as nbf


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


Path("notebooks").mkdir(exist_ok=True)

eda = nbf.v4.new_notebook()
eda.cells = [
    md("""# 01 — Flow dataset EDA

This notebook inspects the downloaded, labelled flow dataset before modelling. The stratified sample is deterministic. Identifiers are retained here for graph construction and talker analysis, but are excluded from numeric model features in `src/preprocess.py`."""),
    code("""from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from src.data import load_flows, find_column

cfg = yaml.safe_load(Path('configs/experiment.yaml').read_text())
flows = load_flows(cfg['data']['path'], sample=cfg['data'].get('sample'), seed=cfg['data'].get('sample_seed', 17))
print(f\"{len(flows):,} sampled rows; time source: {flows['_time_source'].iloc[0]}\")
display(flows.head())"""),
    md("""## Binary balance and attack categories

Because attack prevalence is low, report both counts and proportions. The project samples proportionally by the binary label so the CPU subset does not artificially balance away the operational class imbalance."""),
    code("""counts = flows['_attack_type'].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
flows['_target'].value_counts().sort_index().rename(index={0:'Benign', 1:'Attack'}).plot.bar(ax=axes[0], color=['#277da1','#f94144'])
axes[0].set(title='Binary class counts', ylabel='Flows')
counts.head(12).sort_values().plot.barh(ax=axes[1], color='#577590')
axes[1].set(title='Most frequent attack labels', xlabel='Flows')
plt.tight_layout()"""),
    md("""## Top talkers

The source and destination addresses are used only to construct host-level graph structure. They are not passed as raw numeric model features."""),
    code("""src_col = find_column(flows, ('Source IP','srcip','src_ip','IPV4_SRC_ADDR'), required=False)
dst_col = find_column(flows, ('Destination IP','dstip','dst_ip','IPV4_DST_ADDR'), required=False)
if src_col and dst_col:
    display(flows[src_col].value_counts().head(10).rename('outgoing_flows').to_frame())
    display(flows[dst_col].value_counts().head(10).rename('incoming_flows').to_frame())
else:
    print('This dataset file has no endpoint identifiers; graph modelling must use a different source file.')"""),
    md("""## Feature distributions

Flow counters and durations are typically heavy-tailed. The engineered pipeline applies `log1p` to non-negative count-like features before standardization; all fitted transforms are learned only on the training period."""),
    code("""numeric_cols = flows.select_dtypes(include='number').columns.tolist()
selected = [c for c in numeric_cols if any(k in c.lower() for k in ('byte','packet','duration','pkts','sbytes','dbytes'))][:4]
if selected:
    flows[selected].replace([float('inf'), -float('inf')], np.nan).hist(bins=40, figsize=(12, 7), log=True)
    plt.suptitle('Raw heavy-tailed flow features (log count axis)')
    plt.tight_layout()"""),
    md("""## Temporal coverage and split

The main scripts parse CIC timestamps as day/month/year UTC and group the sorted flows into whole one-hour windows. These windows are assigned chronologically 60/20/20 to train/validation/test. Validation selects the autoencoder threshold; only the final period is used for reported test metrics."""),
    code("""if '_order' in flows:
    print('Ordered range:', pd.to_datetime([flows['_order'].min(), flows['_order'].max()], unit='ns', utc=True))
    display(pd.crosstab(flows['_time_source'], flows['_target'], normalize='columns'))
else:
    print('No usable timestamp was found; any order-based split would be a proxy only.')"""),
]
eda.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}}
nbf.write(eda, "notebooks/01_eda.ipynb")

graph = nbf.v4.new_notebook()
graph.cells = [
    md("""# 02 — Graph analysis: benign vs attack flows

Nodes represent hosts and directed edges represent flows. Each ordered one-hour window is kept as a separate graph. Flow features are edge attributes; node features are incident-edge aggregates. This notebook inspects the test-window graphs and the model-flagged subgraph."""),
    code("""from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from IPython.display import Image, display
from src.data import load_flows, temporal_window_indices
from src.preprocess import FeaturePreprocessor
from src.build_graph import build_graph_windows

cfg = yaml.safe_load(Path('configs/experiment.yaml').read_text())
flows = load_flows(cfg['data']['path'], sample=cfg['data'].get('sample'), seed=cfg['data'].get('sample_seed',17))
window_seconds = cfg['data'].get('window_seconds', 3600)
train_idx, val_idx, test_idx = temporal_window_indices(flows, window_seconds, cfg['split']['train_fraction'], cfg['split']['validation_fraction'])
prep = FeaturePreprocessor(engineered=True).fit(flows.iloc[train_idx])
X = prep.transform(flows)
test_frame = flows.iloc[test_idx].reset_index(drop=True)
test_graphs = build_graph_windows(test_frame, X[test_idx], cfg['data']['window_size'], window_seconds)
print(f'{len(test_graphs)} test graph windows')"""),
    md("""## Graph measures

For each class-specific induced subgraph, compare degree distributions, fan-out (unique destinations per source), connected components and density. These are descriptive test-set graph statistics, not evidence of generalization beyond this dataset."""),
    code("""def class_graph_stats(graph, label):
    ids = np.flatnonzero(graph.y.numpy() == label)
    G = nx.DiGraph()
    G.add_edges_from((graph.src_hosts[i], graph.dst_hosts[i]) for i in ids)
    if not G:
        return None
    source_fanouts=[degree for _,degree in G.out_degree() if degree>0]
    return {'flows':len(ids), 'nodes':G.number_of_nodes(), 'edges':G.number_of_edges(),
            'mean_fanout':float(np.mean(source_fanouts)),
            'weak_components':nx.number_weakly_connected_components(G), 'density':nx.density(G)}
rows=[]
for w, graph in enumerate(test_graphs):
    for y, name in [(0,'Benign'),(1,'Attack')]:
        stats=class_graph_stats(graph,y)
        if stats: rows.append({'window':w,'class':name,**stats})
stats=pd.DataFrame(rows)
display(stats.groupby('class').mean(numeric_only=True).T)
fig, axes = plt.subplots(1,2,figsize=(11,4))
for y,name,color in [(0,'Benign','#277da1'),(1,'Attack','#f94144')]:
    deg=[]; fan=[]
    for graph in test_graphs:
        ids=np.flatnonzero(graph.y.numpy()==y)
        G=nx.DiGraph(); G.add_edges_from((graph.src_hosts[i],graph.dst_hosts[i]) for i in ids)
        deg.extend(dict(G.degree()).values()); fan.extend(v for _,v in G.out_degree() if v>0)
    if deg: axes[0].hist(deg,bins=20,density=True,alpha=.55,label=name,color=color)
    if fan: axes[1].hist(fan,bins=20,density=True,alpha=.55,label=name,color=color)
axes[0].set(title='Degree distribution',xlabel='Degree',ylabel='Density')
axes[1].set(title='Unique destinations per source',xlabel='Fan-out',ylabel='Density')
for ax in axes: ax.legend(); ax.grid(alpha=.2)
plt.tight_layout()"""),
    md("""## Extracted attack subgraph

The experiment script selects a test window and shows the directed edges whose E-GraphSAGE attack probability is at least 0.5. A dense high-fan-out source can be consistent with scanning, but that interpretation is not a ground-truth attack-path claim."""),
    code("""image_path = Path('results/attack_subgraph.png')
if image_path.exists():
    display(Image(filename=str(image_path)))
else:
    print('Run python -m src.train --config configs/experiment.yaml to generate the flagged subgraph.')"""),
    md("""## From this project to lateral movement

For lateral-movement detection, replace flow edges with timestamped authentication or host-to-host events. Keep hosts as nodes, and encode event type, account role, result, and time gap as edge attributes. A temporal model can score events within short windows, while an event store links unusual transitions across windows into candidate paths. On a provenance graph, process, file, user, and host entities become typed nodes and system events become typed edges. Evaluation then needs path-level labels and careful separation by time and enterprise, not only per-edge flow labels. Public LANL authentication data could support a separate follow-up, subject to its own access and label limitations."""),
]
graph.metadata = eda.metadata.copy()
nbf.write(graph, "notebooks/02_graph_analysis.ipynb")
print("Wrote notebooks/01_eda.ipynb and notebooks/02_graph_analysis.ipynb")
