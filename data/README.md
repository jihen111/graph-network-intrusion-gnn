# Dataset files

The raw archives are intentionally excluded from Git. Run `python scripts/download_data.py` to obtain the public mirror of CIC-IDS2017 `GeneratedLabelledFlows.zip` into `data/raw/cic-generated-labelled-flows/`.

The source project and field description are on the [official CIC-IDS2017 dataset page](https://www.unb.ca/cic/datasets/ids-2017.html). The public archive mirror used by this project is [bencorn/CICIDS2017 on Hugging Face](https://huggingface.co/datasets/bencorn/CICIDS2017), file `csvs/GeneratedLabelledFlows.zip`. It contains the labelled flow CSVs, not the multi-gigabyte PCAP files.

The preferred NF-UNSW-NB15-v2 parquet was also tested during project setup. The downloaded feature-only table does not include endpoints or timestamps, so it cannot support this repository's host-graph and chronological-split requirements. The rationale is recorded in `DECISIONS.md`.
