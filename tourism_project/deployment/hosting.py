import os
import sys
import time
import traceback
from pathlib import Path

import requests
from huggingface_hub import CommitOperationAdd, HfApi

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL_REPO = os.getenv(
    "HF_MODEL_REPO_ID",
    "somya1607/tourism-wellness-model",
)
HF_FRONTEND_SPACE = os.getenv(
    "HF_FRONTEND_SPACE_REPO_ID",
    "somya1607/visit-with-us-frontend",
)
HF_BACKEND_SPACE = os.getenv(
    "HF_BACKEND_SPACE_REPO_ID",
    "somya1607/visit-with-us-backend",
)
BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "https://somya1607-visit-with-us-backend.hf.space",
).rstrip("/")
FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "https://somya1607-visit-with-us-frontend.hf.space",
).rstrip("/")

PROJECT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_FOLDER = PROJECT_DIR / "deployment" / "frontend"
BACKEND_FOLDER = PROJECT_DIR / "deployment" / "backend"
REQUIRED_FILES = ("app.py", "requirements.txt", "Dockerfile", "README.md")
OPTIONAL_BACKEND_ARTIFACTS = (
    "wellness_tourism_model.joblib",
    "model_metadata.json",
    "feature_schema.json",
)


def validate_folder(folder: Path, required_files: tuple[str, ...]) -> None:
    if not folder.is_dir():
        raise FileNotFoundError(f"Deployment folder does not exist: {folder}")

    missing = [name for name in required_files if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing deployment files in {folder}: {missing}")

    for name in required_files:
        if (folder / name).stat().st_size == 0:
            raise RuntimeError(f"Deployment file is empty: {folder / name}")

    print(f"Validated deployment folder: {folder}")
    for name in required_files:
        print(f"  - {name}")


def require_existing_space(api: HfApi, repo_id: str) -> None:
    if not api.repo_exists(repo_id=repo_id, repo_type="space"):
        raise RuntimeError(
            f"Required existing Space was not found: {repo_id}. "
            "This script intentionally does not create a paid Docker Space."
        )
    print(f"Using existing Space: {repo_id}")


def add_space_variable_compatible(
    api: HfApi,
    repo_id: str,
    key: str,
    value: str,
    description: str,
) -> None:
    try:
        api.add_space_variable(
            repo_id=repo_id,
            key=key,
            value=value,
            description=description,
        )
    except TypeError:
        api.add_space_variable(repo_id=repo_id, key=key, value=value)


def deploy_files_atomically(
    api: HfApi,
    repo_id: str,
    folder: Path,
    filenames: list[str],
    commit_message: str,
) -> None:
    require_existing_space(api, repo_id)

    operations = [
        CommitOperationAdd(
            path_in_repo=filename,
            path_or_fileobj=str(folder / filename),
        )
        for filename in filenames
    ]

    # Never delete all files from a live Space. The previous wildcard deletion
    # caused the 503 and "No application file" state.
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="space",
        operations=operations,
        commit_message=commit_message,
    )
    print(f"Committed deployment files: {getattr(commit, 'commit_url', commit)}")

    repo_files = set(api.list_repo_files(repo_id=repo_id, repo_type="space"))
    missing_after_upload = [name for name in REQUIRED_FILES if name not in repo_files]
    if missing_after_upload:
        raise RuntimeError(
            f"Space upload verification failed for {repo_id}. "
            f"Missing files: {missing_after_upload}"
        )

    print(f"Verified required files in {repo_id}:")
    for name in REQUIRED_FILES:
        print(f"  - {name}")


def request_restart(api: HfApi, repo_id: str) -> None:
    try:
        api.restart_space(repo_id=repo_id)
        print(f"Restart requested: {repo_id}")
    except Exception as exc:
        print(f"Restart request skipped for {repo_id}: {type(exc).__name__}: {exc}")


def wait_for_endpoint(url: str, label: str, attempts: int = 12) -> None:
    last_message = "No response received."
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, timeout=20)
            last_message = f"HTTP {response.status_code}: {response.text[:300]}"
            if response.status_code == 200:
                print(f"{label} is responding: {url}")
                print(response.text[:500])
                return
        except requests.RequestException as exc:
            last_message = f"{type(exc).__name__}: {exc}"

        print(f"Waiting for {label} build/startup ({attempt}/{attempts})...")
        time.sleep(10)

    print(
        f"{label} files were deployed, but the endpoint is still building or in error. "
        f"Last result: {last_message}"
    )


def main() -> None:
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is required. Add a Hugging Face write token to "
            "Colab Secrets or GitHub Actions Secrets."
        )

    validate_folder(BACKEND_FOLDER, REQUIRED_FILES)
    validate_folder(FRONTEND_FOLDER, REQUIRED_FILES)

    api = HfApi(token=HF_TOKEN)
    identity = api.whoami()
    authenticated_user = identity.get("name") or identity.get("fullname") or "authenticated user"
    print(f"Authenticated with Hugging Face as: {authenticated_user}")

    backend_files = list(REQUIRED_FILES)
    backend_files.extend(
        name for name in OPTIONAL_BACKEND_ARTIFACTS if (BACKEND_FOLDER / name).is_file()
    )

    deploy_files_atomically(
        api=api,
        repo_id=HF_BACKEND_SPACE,
        folder=BACKEND_FOLDER,
        filenames=backend_files,
        commit_message="Restore Wellness Tourism FastAPI backend files",
    )
    add_space_variable_compatible(
        api=api,
        repo_id=HF_BACKEND_SPACE,
        key="HF_MODEL_REPO_ID",
        value=HF_MODEL_REPO,
        description="Registered Wellness Tourism model repository",
    )

    deploy_files_atomically(
        api=api,
        repo_id=HF_FRONTEND_SPACE,
        folder=FRONTEND_FOLDER,
        filenames=list(REQUIRED_FILES),
        commit_message="Restore Wellness Tourism Streamlit frontend files",
    )
    add_space_variable_compatible(
        api=api,
        repo_id=HF_FRONTEND_SPACE,
        key="BACKEND_URL",
        value=BACKEND_URL,
        description="Visit With Us FastAPI backend URL",
    )

    request_restart(api, HF_BACKEND_SPACE)
    request_restart(api, HF_FRONTEND_SPACE)

    wait_for_endpoint(f"{BACKEND_URL}/health", "Backend")
    wait_for_endpoint(f"{FRONTEND_URL}/_stcore/health", "Frontend")

    print("\nHugging Face deployment files restored successfully.")
    print("Backend Space:", f"https://huggingface.co/spaces/{HF_BACKEND_SPACE}")
    print("Frontend Space:", f"https://huggingface.co/spaces/{HF_FRONTEND_SPACE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nDEPLOYMENT ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise
