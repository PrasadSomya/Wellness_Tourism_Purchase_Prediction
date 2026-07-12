import os
import sys
import time
import traceback
from pathlib import Path

from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME", "somya1607")
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

PROJECT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_FOLDER = PROJECT_DIR / "deployment" / "frontend"
BACKEND_FOLDER = PROJECT_DIR / "deployment" / "backend"


def validate_folder(folder: Path, required_files: list[str]) -> None:
    if not folder.exists():
        raise FileNotFoundError(f"Deployment folder does not exist: {folder}")

    missing = [
        filename
        for filename in required_files
        if not (folder / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing deployment files in {folder}: {missing}"
        )

    print(f"Validated deployment folder: {folder}")
    for filename in required_files:
        print(f"  - {filename}")


def space_exists(api: HfApi, repo_id: str) -> bool:
    try:
        return bool(api.repo_exists(repo_id=repo_id, repo_type="space"))
    except Exception as exc:
        print(
            f"Unable to check Space {repo_id}: "
            f"{type(exc).__name__}: {exc}"
        )
        return False


def ensure_existing_space(
    api: HfApi,
    repo_id: str,
) -> str:
    # Deploy only to the learner's existing Spaces. Do not create or rename.
    if space_exists(api, repo_id):
        print(f"Using existing Space: {repo_id}")
        return repo_id

    raise RuntimeError(
        f"Required existing Space was not found: {repo_id}. "
        "Confirm the repository name and ensure HF_TOKEN has write access. "
        "This deployment intentionally does not create or rename Spaces."
    )


def deploy_space(
    api: HfApi,
    repo_id: str,
    folder: Path,
    commit_message: str,
) -> str:
    repo_id = ensure_existing_space(
        api=api,
        repo_id=repo_id,
    )

    print(f"Uploading {folder} to {repo_id}")
    commit = api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="space",
        commit_message=commit_message,
    )
    print(f"Upload completed: {getattr(commit, 'commit_url', commit)}")
    return repo_id


def main() -> None:
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is required. Add a Hugging Face write token "
            "to Colab Secrets or GitHub Actions Secrets."
        )

    required = ["app.py", "requirements.txt", "Dockerfile", "README.md"]
    validate_folder(BACKEND_FOLDER, required)
    validate_folder(FRONTEND_FOLDER, required)

    api = HfApi(token=HF_TOKEN)

    identity = api.whoami()
    authenticated_user = (
        identity.get("name")
        or identity.get("fullname")
        or "authenticated user"
    )
    print(f"Authenticated with Hugging Face as: {authenticated_user}")

    backend_repo_id = deploy_space(
        api=api,
        repo_id=HF_BACKEND_SPACE,
        folder=BACKEND_FOLDER,
        commit_message="Deploy Wellness Tourism FastAPI backend",
    )

    api.add_space_variable(
        repo_id=backend_repo_id,
        key="HF_MODEL_REPO_ID",
        value=HF_MODEL_REPO,
        description="Registered Wellness Tourism model repository",
    )

    frontend_repo_id = deploy_space(
        api=api,
        repo_id=HF_FRONTEND_SPACE,
        folder=FRONTEND_FOLDER,
        commit_message="Deploy Wellness Tourism Streamlit frontend",
    )

    api.add_space_variable(
        repo_id=frontend_repo_id,
        key="BACKEND_URL",
        value=BACKEND_URL,
        description="Visit With Us FastAPI backend URL",
    )

    for repo_id in (backend_repo_id, frontend_repo_id):
        try:
            api.restart_space(repo_id=repo_id)
            print(f"Restart requested: {repo_id}")
        except Exception as exc:
            print(
                f"Space restart skipped for {repo_id}: "
                f"{type(exc).__name__}: {exc}"
            )

    print(
        "\nBackend deployed:",
        f"https://huggingface.co/spaces/{backend_repo_id}",
    )
    print(
        "Frontend deployed:",
        f"https://huggingface.co/spaces/{frontend_repo_id}",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"\nDEPLOYMENT ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc()
        raise
