import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, ConfigDict

app = FastAPI(title="Visit With Us Wellness Tourism API", version="1.0.0")
APP_DIR = Path(__file__).resolve().parent
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO_ID", "somya1607/tourism-wellness-model")
MODEL_FILENAME = "wellness_tourism_model.joblib"
METADATA_FILENAME = "model_metadata.json"
FEATURE_SCHEMA_FILENAME = "feature_schema.json"
model = None
model_metadata: dict[str, Any] = {}
feature_schema: dict[str, Any] = {}
model_load_error = None

def resolve_artifact(filename: str) -> Path:
    local_path = APP_DIR / filename
    if local_path.exists():
        return local_path
    return Path(hf_hub_download(repo_id=HF_MODEL_REPO, filename=filename, token=os.getenv("HF_TOKEN") or None))

def load_artifacts() -> None:
    global model, model_metadata, feature_schema, model_load_error
    try:
        model = joblib.load(resolve_artifact(MODEL_FILENAME))
        model_metadata = json.loads(resolve_artifact(METADATA_FILENAME).read_text(encoding="utf-8"))
        feature_schema = json.loads(resolve_artifact(FEATURE_SCHEMA_FILENAME).read_text(encoding="utf-8"))
        model_load_error = None
    except Exception as exc:
        model = None
        model_metadata = {}
        feature_schema = {}
        model_load_error = f"{type(exc).__name__}: {exc}"

load_artifacts()

class CustomerData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    Age: float
    TypeofContact: str
    CityTier: int
    DurationOfPitch: float
    Occupation: str
    Gender: str
    NumberOfPersonVisiting: int
    NumberOfFollowups: int
    ProductPitched: str
    PreferredPropertyStar: int
    MaritalStatus: str
    NumberOfTrips: int
    Passport: int
    PitchSatisfactionScore: int
    OwnCar: int
    NumberOfChildrenVisiting: int
    Designation: str
    MonthlyIncome: float

@app.get("/")
def root():
    return {"service": "Visit With Us Wellness Tourism API", "status": "running", "model_loaded": model is not None}

@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_repo": HF_MODEL_REPO,
        "model_name": model_metadata.get("best_model_name"),
        "error": model_load_error,
    }

@app.post("/predict")
def predict(data: CustomerData):
    if model is None:
        raise HTTPException(status_code=503, detail={"message": "Model is not loaded.", "error": model_load_error})
    payload = data.model_dump()
    feature_order = feature_schema.get("feature_order", list(payload))
    missing = [feature for feature in feature_order if feature not in payload]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing model features: {missing}")
    frame = pd.DataFrame([{feature: payload[feature] for feature in feature_order}])
    try:
        probability = float(model.predict_proba(frame)[0, 1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(exc).__name__}: {exc}") from exc
    threshold = float(model_metadata.get("recommended_threshold", 0.50))
    prediction = int(probability >= threshold)
    return {
        "prediction": prediction,
        "predicted_class": prediction,
        "purchase_probability": probability,
        "decision_threshold": threshold,
        "model_name": model_metadata.get("best_model_name"),
    }
