import json
import os
from pathlib import Path
import joblib
import mlflow
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME", "somya1607")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO_ID", f"{HF_USERNAME}/tourism-wellness-data")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO_ID", f"{HF_USERNAME}/tourism-wellness-model")
PROJECT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "wellness_tourism_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
FEATURE_SCHEMA_PATH = MODELS_DIR / "feature_schema.json"
EXPERIMENT_PATH = REPORTS_DIR / "experiment_results.csv"
THRESHOLD_PATH = REPORTS_DIR / "threshold_analysis.csv"
REPORT_PATH = REPORTS_DIR / "classification_report.txt"
MLFLOW_DB_PATH = PROJECT_DIR / "mlflow.db"
MLFLOW_ARTIFACTS_DIR = PROJECT_DIR / "mlartifacts"
MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.resolve().as_posix()}"

def evaluate(y_true, probabilities, threshold):
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    return {"threshold": float(threshold), "accuracy": float(accuracy_score(y_true, predictions)), "precision": float(precision_score(y_true, predictions, zero_division=0)), "recall": float(recall_score(y_true, predictions, zero_division=0)), "f1": float(f1_score(y_true, predictions, zero_division=0)), "roc_auc": float(roc_auc_score(y_true, probabilities)), "average_precision": float(average_precision_score(y_true, probabilities)), "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist()}

def safe_mode(series):
    values = series.dropna().astype(str)
    return "" if values.empty else str(values.mode().iloc[0])

def main():
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is required. Add it to GitHub Actions Secrets.")
    train_file = hf_hub_download(repo_id=HF_DATASET_REPO, filename="train.csv", repo_type="dataset", token=HF_TOKEN)
    test_file = hf_hub_download(repo_id=HF_DATASET_REPO, filename="test.csv", repo_type="dataset", token=HF_TOKEN)
    train_data, test_data = pd.read_csv(train_file), pd.read_csv(test_file)
    X_train, y_train = train_data.drop(columns=["ProdTaken"]), train_data["ProdTaken"].astype(int)
    X_test, y_test = test_data.drop(columns=["ProdTaken"]), test_data["ProdTaken"].astype(int)
    numeric = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X_train.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    preprocessor = ColumnTransformer([("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric), ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical)])
    candidates = {
        "Decision Tree": (DecisionTreeClassifier(random_state=42, class_weight="balanced"), {"classifier__max_depth": [4, 6, None], "classifier__min_samples_leaf": [1, 3]}),
        "Random Forest": (RandomForestClassifier(random_state=42, class_weight="balanced", n_estimators=100, n_jobs=1), {"classifier__max_depth": [8, None], "classifier__min_samples_leaf": [1, 3]}),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=42), {"classifier__n_estimators": [80, 120], "classifier__learning_rate": [0.05, 0.10], "classifier__max_depth": [3]}),
        "AdaBoost": (AdaBoostClassifier(random_state=42), {"classifier__n_estimators": [50, 100], "classifier__learning_rate": [0.05, 0.10]}),
    }
    experiment_name = "wellness-tourism-model-experiments"
    configured_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if not configured_tracking_uri or configured_tracking_uri.lower().startswith("file:"):
        tracking_uri = DEFAULT_MLFLOW_TRACKING_URI
    else:
        tracking_uri = configured_tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    if mlflow.active_run() is not None:
        mlflow.end_run()
    existing_experiment = mlflow.get_experiment_by_name(experiment_name)
    if existing_experiment is None:
        mlflow.create_experiment(name=experiment_name, artifact_location=MLFLOW_ARTIFACTS_DIR.resolve().as_uri())
    mlflow.set_experiment(experiment_name)
    print("MLflow tracking URI:", mlflow.get_tracking_uri())
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    results, best_model, best_result = [], None, None
    for name, (classifier, params) in candidates.items():
        search = GridSearchCV(Pipeline([("preprocessor", preprocessor), ("classifier", classifier)]), params, scoring={"f1": "f1", "precision": "precision", "recall": "recall", "roc_auc": "roc_auc"}, refit="f1", cv=cv, n_jobs=1, error_score="raise")
        search.fit(X_train, y_train)
        idx = int(search.best_index_)
        result = {"model_name": name, "best_cv_f1_score": float(search.best_score_), "best_cv_precision": float(search.cv_results_["mean_test_precision"][idx]), "best_cv_recall": float(search.cv_results_["mean_test_recall"][idx]), "best_cv_roc_auc": float(search.cv_results_["mean_test_roc_auc"][idx]), "best_params": search.best_params_}
        results.append(result)
        with mlflow.start_run(run_name=name):
            mlflow.log_params({k: str(v) for k, v in result["best_params"].items()})
            mlflow.log_metrics({"cv_f1": result["best_cv_f1_score"], "cv_precision": result["best_cv_precision"], "cv_recall": result["best_cv_recall"], "cv_roc_auc": result["best_cv_roc_auc"]})
        if best_result is None or result["best_cv_f1_score"] > best_result["best_cv_f1_score"]:
            best_result, best_model = result, search.best_estimator_
    pd.DataFrame(results).sort_values("best_cv_f1_score", ascending=False).to_csv(EXPERIMENT_PATH, index=False)
    oof = cross_val_predict(clone(best_model), X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    threshold_rows = []
    for threshold in np.arange(0.10, 0.91, 0.05):
        threshold = round(float(threshold), 2)
        metrics = evaluate(y_train, oof, threshold)
        threshold_rows.append({"threshold": threshold, "accuracy": metrics["accuracy"], "precision": metrics["precision"], "recall": metrics["recall"], "f1": metrics["f1"], "contact_rate": float((oof >= threshold).mean())})
    threshold_frame = pd.DataFrame(threshold_rows)
    threshold_frame.to_csv(THRESHOLD_PATH, index=False)
    max_f1 = threshold_frame["f1"].max()
    choices = threshold_frame[np.isclose(threshold_frame["f1"], max_f1)].copy()
    choices["distance"] = (choices["threshold"] - 0.5).abs()
    recommended = float(choices.sort_values(["distance", "precision"], ascending=[True, False]).iloc[0]["threshold"])
    best_model.fit(X_train, y_train)
    probabilities = best_model.predict_proba(X_test)[:, 1]
    final_metrics = evaluate(y_test, probabilities, recommended)
    predictions = (probabilities >= recommended).astype(int)
    REPORT_PATH.write_text(classification_report(y_test, predictions, labels=[0, 1], zero_division=0), encoding="utf-8")
    joblib.dump(best_model, MODEL_PATH)
    schema = {"target": "ProdTaken", "feature_order": X_train.columns.tolist(), "numeric_features": numeric, "categorical_features": categorical, "numeric_defaults": {c: float(X_train[c].median()) for c in numeric}, "categorical_defaults": {c: safe_mode(X_train[c]) for c in categorical}, "categorical_values": {c: sorted(X_train[c].dropna().astype(str).unique().tolist()) for c in categorical}}
    FEATURE_SCHEMA_PATH.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    metadata = {"project": "Visit with Us - Wellness Tourism Package Purchase Prediction", "best_model_name": best_result["model_name"], "selection_metric": "cross_validated_f1", "best_cross_validation_result": best_result, "default_threshold": 0.50, "recommended_threshold": recommended, "metrics_at_recommended_threshold": final_metrics}
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (MODELS_DIR / "README.md").write_text(f"---\nlicense: mit\nlibrary_name: scikit-learn\n---\n\n# Wellness Tourism Package Purchase Prediction Model\n\nBest model: {best_result['model_name']}\n\nRecommended threshold: {recommended}\n", encoding="utf-8")
    with mlflow.start_run(run_name="selected-best-model"):
        mlflow.log_metrics({"test_accuracy": final_metrics["accuracy"], "test_precision": final_metrics["precision"], "test_recall": final_metrics["recall"], "test_f1": final_metrics["f1"], "test_roc_auc": final_metrics["roc_auc"]})
        for artifact in [MODEL_PATH, METADATA_PATH, FEATURE_SCHEMA_PATH, EXPERIMENT_PATH, THRESHOLD_PATH, REPORT_PATH]:
            mlflow.log_artifact(str(artifact))
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_MODEL_REPO, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(folder_path=str(MODELS_DIR), repo_id=HF_MODEL_REPO, repo_type="model", commit_message="Register best wellness tourism model")
    for report in [EXPERIMENT_PATH, THRESHOLD_PATH, REPORT_PATH]:
        api.upload_file(path_or_fileobj=str(report), path_in_repo=report.name, repo_id=HF_MODEL_REPO, repo_type="model", commit_message=f"Upload {report.name}")
    print("Best model:", best_result["model_name"])
    print("Final metrics:", json.dumps(final_metrics, indent=2))

if __name__ == "__main__":
    main()
