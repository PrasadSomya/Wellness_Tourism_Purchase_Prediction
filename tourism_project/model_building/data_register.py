import os
from pathlib import Path
import pandas as pd
from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME", "somya1607")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO_ID", f"{HF_USERNAME}/tourism-wellness-data")
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "tourism.csv"
README_PATH = DATA_DIR / "README.md"

REQUIRED_COLUMNS = {"CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "Occupation", "Gender", "NumberOfPersonVisiting", "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips", "Passport", "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome", "PitchSatisfactionScore", "ProductPitched", "NumberOfFollowups", "DurationOfPitch"}

def main():
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is required. Add it to GitHub Actions Secrets.")
    if not RAW_DATA_PATH.exists() or not README_PATH.exists():
        raise FileNotFoundError("tourism.csv or data README.md is missing from tourism_project/data.")
    dataframe = pd.read_csv(RAW_DATA_PATH)
    missing = sorted(REQUIRED_COLUMNS - set(dataframe.columns))
    if missing:
        raise ValueError(f"Raw dataset is missing required columns: {missing}")
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_DATASET_REPO, repo_type="dataset", exist_ok=True, private=False)
    api.upload_file(path_or_fileobj=str(RAW_DATA_PATH), path_in_repo="tourism.csv", repo_id=HF_DATASET_REPO, repo_type="dataset", commit_message="Register raw tourism dataset")
    api.upload_file(path_or_fileobj=str(README_PATH), path_in_repo="README.md", repo_id=HF_DATASET_REPO, repo_type="dataset", commit_message="Add dataset README")
    print(f"Dataset registered: https://huggingface.co/datasets/{HF_DATASET_REPO}")

if __name__ == "__main__":
    main()
