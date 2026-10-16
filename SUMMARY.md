# Run summary (version 2)

**Run:** 22 minutes, seeds 11, 23, 37, 1,492,235 flows used of 2,830,743 (all 557,646 attack flows), 5-minute graph windows.

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
| Random Forest (flow-only) | 0.604 | 0.979 |
| E-GraphSAGE | 0.843 | 0.977 |
| Graph Autoencoder (benign-only) | 0.879 | 0.564 |

Label efficiency (protocol B, E-GraphSAGE): F1 0.989 with 1 % of labels, 0.975 with 10 %, 0.987 with 100 %.

Time per seed (seconds): Random Forest 38, E-GraphSAGE 27, Graph autoencoder 19 (protocol B).

## Lignes CV (français, chiffres réels)

1. **Détection d'intrusions réseau par GNN** – pipeline PyTorch Geometric sur CIC-IDS2017 (1,5 M de flux, un graphe d'hôtes par fenêtre de 5 min, attributs structurels) : E-GraphSAGE, autoencodeur de graphes non supervisé et Random Forest de référence.
2. Sur des attaques jamais vues à l'entraînement, E-GraphSAGE détecte 97 % des flux malveillants (seuil réglé pour 1 % de faux positifs en validation ; 2,7 % en test) contre 31 % pour le Random Forest ; F1 0,99 avec seulement 1 % des labels ; extraction de sous-graphes d'attaque (NetworkX).
