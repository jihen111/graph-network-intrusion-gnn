# Dataset files

Raw data is not committed. `python scripts/download_data.py` downloads CIC-IDS2017 `GeneratedLabelledFlows.zip` (about 270 MB) from the public Hugging Face mirror [bencorn/CICIDS2017](https://huggingface.co/datasets/bencorn/CICIDS2017) into `data/raw/cic-generated-labelled-flows/`. These CSVs keep source/destination IPs, ports and timestamps, which graph construction needs. Dataset description: [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html).

`scripts/make_smoke_data.py` writes a tiny synthetic dataset to `data/smoke/` for testing that the code runs. Never report metrics from it.
