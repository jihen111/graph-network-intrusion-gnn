# Design decisions

## Dataset selection

- I first downloaded the public [NF-UNSW-NB15-v2 mirror](https://www.kaggle.com/datasets/dhoogla/nfunswnb15v2). Its 1,986,745 rows have 43 fields, including `Label` and `Attack`, but the distributed table contains no source/destination host identifiers and no event timestamp. A host-flow graph or honest chronological split cannot be built from those columns. I did not invent endpoints or times.
- I also inspected the CIC-IDS2017 MachineLearningCSV mirror; it is a feature-only table and likewise omits endpoint identifiers and timestamps.
- I therefore selected the CIC-IDS2017 **GeneratedLabelledFlows** archive, which preserves flow IDs, source/destination IPs and ports, timestamps, flow features, and labels. The [official CIC portal](https://www.unb.ca/cic/datasets/ids-2017.html) documents the labelled flows; this repository downloads the public [Hugging Face mirror](https://huggingface.co/datasets/bencorn/CICIDS2017). Only labelled flow CSVs are used; packet captures are not downloaded.
- This is an explicit scope adjustment: the experiment uses CIC-IDS2017 labelled flows rather than claiming the NF-UNSW-NB15-v2 feature-only parquet contains graph endpoints.
- The public CSV mirror includes a small set of rows with both empty timestamps and empty labels. The loader excludes these as unlabelled records before class-stratified sampling; they are never miscounted as attacks.

## Graph definition

- Nodes are IP hosts; each original flow is a directed edge from source to destination. Message passing uses the reverse arc as well, but each flow label is scored once.
- Host-level nodes preserve communication fan-out across ports. IP:port endpoints would capture service/client roles more precisely but fragment hosts into many sparse nodes. Ports are retained as coarse well-known/registered/ephemeral edge-feature categories, never as raw numeric model inputs.
- Flow features are edge attributes and are included in both directions of each EdgeSAGE message. Node attributes are the mean of incident edge features in the current one-hour window. The no-edge-feature ablation uses degree-based node features and zeros the edge vector.
- The benign-only autoencoder intentionally uses only endpoint in/out-degree to produce node embeddings, then reconstructs the flow attributes. It never receives the target edge's feature vector as input, preventing an easy self-copy shortcut.

## Time and splits

- The CIC `Timestamp` field uses day/month/year and is parsed by assigning UTC to the naive source time (the archive supplies no timezone offset). One graph covers one UTC-aligned hour; whole hourly windows are assigned chronologically to 60/20/20 train/validation/test partitions, so no graph crosses a split boundary.
- If a future dataset has no usable timestamp, the loader warns and uses source-row order only as a proxy. Such a proxy must not be described as wall-clock temporal evaluation.
- All imputers, encoders and scalers are fitted on training flows only. The validation period selects the benign autoencoder threshold; the test period is used only for final metrics.

## Compute and reproducibility

- The default is a stratified 40,000-flow sample so three-seed comparisons and label/feature ablations finish on CPU. `--sample 0` requests all available labelled flows; larger values can be supplied with `--sample N`.
- Both graph models use a maximum of eight epochs and patience-two early stopping on validation loss (benign validation reconstruction loss for the autoencoder); the best validation checkpoint is restored. The test split is not used for stopping.
- Sampling is deterministic with seed 17 and preserves the overall benign/attack proportions. Model seeds are fixed separately at 11, 23 and 37.
- Label-efficiency masks are drawn once over all training edges, stratified by binary class, then held fixed across epochs. Class weights are calculated from the selected labels only.
- The original dataset files remain under the Git-ignored `data/raw/` directory and are not committed or included in the source archive.
