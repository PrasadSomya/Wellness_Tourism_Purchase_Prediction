import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from sklearn.model_selection import train_test_split

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME", "somya1607")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO_ID", f"{HF_USERNAME}/tourism-wellness-data")
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "tourism.csv"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SCHEMA_PATH = DATA_DIR / "data_schema.json"

def clean_tourism_data(dataframe):
    cleaned = dataframe.copy().drop(columns=[c for c in ["Unnamed: 0", "CustomerID"] if c in dataframe.columns])
    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = cleaned[column].apply(lambda value: value.strip() if isinstance(value, str) else value).replace("", pd.NA)
    if "Gender" in cleaned.columns:
        cleaned["Gender"] = cleaned["Gender"].replace({"Fe Male": "Female", "FeMale": "Female"})
    cleaned["ProdTaken"] = pd.to_numeric(cleaned["ProdTaken"], errors="coerce")
    cleaned = cleaned.dropna(subset=["ProdTaken"])
    cleaned["ProdTaken"] = cleaned["ProdTaken"].astype(int)
    if set(cleaned["ProdTaken"].unique()) - {0, 1}:
        raise ValueError("ProdTaken contains invalid target values.")
    return cleaned.drop_duplicates().reset_index(drop=True)

def distribution(series):
    return {str(int(k)): float(v) for k, v in series.value_counts(normalize=True).sort_index().items()}

def main():
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is required. Add it to GitHub Actions Secrets.")
    try:
        raw_file = hf_hub_download(repo_id=HF_DATASET_REPO, filename="tourism.csv", repo_type="dataset", token=HF_TOKEN)
        dataframe = pd.read_csv(raw_file)
    except Exception as exc:
        if not RAW_DATA_PATH.exists():
            raise FileNotFoundError(f"Unable to obtain tourism.csv: {exc}") from exc
        dataframe = pd.read_csv(RAW_DATA_PATH)
    cleaned = clean_tourism_data(dataframe)
    train_data, test_data = train_test_split(cleaned, test_size=0.20, random_state=42, stratify=cleaned["ProdTaken"])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_data.to_csv(TRAIN_PATH, index=False)
    test_data.to_csv(TEST_PATH, index=False)
    features = cleaned.drop(columns=["ProdTaken"])
    schema = {"target": "ProdTaken", "feature_columns": features.columns.tolist(), "numeric_columns": features.select_dtypes(include=[np.number]).columns.tolist(), "categorical_columns": features.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist(), "train_rows": int(train_data.shape[0]), "test_rows": int(test_data.shape[0]), "target_distribution_train": distribution(train_data["ProdTaken"]), "target_distribution_test": distribution(test_data["ProdTaken"])}
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    api = HfApi(token=HF_TOKEN)
    for path in [TRAIN_PATH, TEST_PATH, SCHEMA_PATH]:
        api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name, repo_id=HF_DATASET_REPO, repo_type="dataset", commit_message=f"Prepare {path.name}")
    print("Data preparation complete.")

if __name__ == "__main__":
    main()
