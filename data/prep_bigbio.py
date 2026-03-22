import pandas as pd
import json
from pathlib import Path

# paths
base_path = Path("./data_splits_stratified/6-2-2_all_classes_enriched")
splits = ["train", "val", "test"]

for split in splits:
    df = pd.read_csv(base_path / f"{split}.csv")

    records = []
    for _, row in df.iterrows():
        record = {
            "document_id": str(row["pmid"]),
            "title": row["title"],
            "abstract": row["abstract"],
            "label": row["accepted_label"],
        }
        records.append(record)

    # save as JSON (list of dicts)
    if split=="val":
        split = "dev"
    with open(base_path / f"{split}.json", "w") as f:
        json.dump(records, f, indent=2)

    print(f"Saved {split}.json with {len(records)} records")