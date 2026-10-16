# Interview preparation

## 1. What is message passing in a GNN?
Each node aggregates information from its neighbors, combines that summary with its own representation, and updates its embedding. Repeating this process lets a node representation reflect progressively larger neighborhoods.

## 2. Why use GraphSAGE here?
GraphSAGE learns a neighborhood aggregation function and can produce embeddings for nodes not seen during training. It is compact and practical for a CPU-scale prototype. This project uses one host graph per 5-minute window, rather than one giant graph.

## 3. What makes this an edge-classification problem?
The object being classified is a network flow. The model first computes source and destination host embeddings, then classifies the concatenation of those embeddings and that flow's edge features.

## 4. Why model hosts as nodes instead of IP:port pairs?
Host nodes preserve communication relationships and make fan-out interpretable. IP:port nodes are more service-specific, but can split one host into many low-degree nodes. This project keeps coarse port buckets as edge features, and counts distinct ports per host as a structural node feature.

## 5. Why are flow features edge features?
Bytes, packet counts, duration, protocol, and flags describe a communication event. Assigning them to the edge keeps event attributes separate from host identity and lets the classifier combine them with endpoint embeddings.

## 6. Why compare against a classical model?
A random forest on the same flow features is the no-graph reference. It tests whether neighborhood context adds value beyond ordinary flow statistics. A GNN is not automatically better and should be judged against this baseline.

## 7. Why use a temporal split rather than a random flow split?
Flows close in time can share attack episodes and network context. A random split can put near-duplicate or same-episode flows into both train and test. A chronological split better approximates deployment on later traffic.

## 8. How do you avoid feature leakage?
Source/destination IPs, raw ports, timestamps, flow IDs, labels, and attack categories are not model inputs. Scalers, imputers, and one-hot encoders are fitted on the training period only. IPs are used only to build graph topology.

## 9. Why log-transform some counters?
Bytes, packets, durations, and counts can be strongly right-skewed. `log1p` compresses their extreme values while retaining zero as zero. Scaling then puts numeric features on comparable ranges.

## 10. How do you handle class imbalance?
The supervised GNN uses inverse-frequency weighted cross-entropy. The baseline uses balanced class weights. Precision, recall, F1, PR-AUC, false-positive rate, and class-specific results make the trade-offs visible.

## 11. What does the label-efficiency experiment measure?
It retrains the supervised GNN with 1%, 5%, 10%, 25%, and 100% of the training-edge labels. One global class-stratified subset is selected and held fixed across every epoch; validation and test data remain unchanged. Class weights use only the selected labels. It measures sensitivity to annotation budget, not zero-shot detection.

## 12. Why include a benign-only graph autoencoder?
It uses endpoint embeddings from graph topology (degrees, distinct peers and ports) to reconstruct edge-feature vectors, and fits on benign training flows only. Keeping the target edge features out of the encoder avoids a trivial self-copy shortcut. High reconstruction error is treated as anomalous; a threshold is selected on validation data, not on the test set.

## 13. Why report PR-AUC?
With rare attacks, a model can obtain a high ROC-AUC while producing too many false alarms. Precision-recall summaries focus on the positive attack class and are more informative under imbalance.

## 14. What does a high-fan-out subgraph suggest?
One source contacting many destinations can be consistent with scanning or discovery behavior. It is a useful analyst clue, not proof of an attack; legitimate inventory or monitoring can look similar.

## 15. How would you extend this to provenance graphs and lateral movement?
Use typed nodes for hosts, users, processes, and files, and typed edges for logins, process launches, or file access. Score events over time, then link unusual host/account transitions into candidate paths. A system such as LANL authentication logs could support a separate study, but would need careful labels and time-aware evaluation.

## 16. Why does random flow sampling break a graph model?
A graph model learns from who talks to whom. If you keep 1 flow in 70 at random, a port scan of 1,000 flows becomes about 14 flows, and a DDoS star loses most of its sources: the patterns the GNN should see are gone. Version 2 keeps every attack flow and every time window, and only thins benign flows inside each window.

## 17. What is threshold calibration and why did it matter here?
A classifier outputs a score; you still have to choose the score above which you raise an alert. In version 1 the supervised models ranked attacks well (high PR-AUC) but used 0.5, which was far from optimal, so recall looked terrible. The fix is to pick the threshold on validation data (for example the F1-maximising one, or the one giving 1 % false positives) and apply it unchanged to test.

## 18. What is the difference between protocol A and protocol B?
Protocol A trains on Monday–Wednesday and tests on Friday, whose attacks were never seen: it measures generalisation to new attack types (zero-day). Protocol B splits each attack type's windows over time so every type is in train and test: it measures how well a known attack is recognised. Supervised models do much better on B; an anomaly detector trained only on benign traffic is the natural tool for A.

## 19. What are the structural node features and why add them?
Per host and window: out-degree, in-degree, distinct destinations, distinct sources, distinct destination ports, distinct source ports. They are computed from the graph alone. A scanner contacts many distinct ports; a DDoS target receives flows from many sources. They let the model, and the ablation "graph structure only", use topology directly.

## 20. Why recall at 1 % false-positive rate?
In a security operations centre, analysts can only review so many alerts. Fixing the false-positive budget (here 1 % of benign flows, set on validation) and reporting how many attacks are caught is closer to how a detector is actually used than F1 alone.
