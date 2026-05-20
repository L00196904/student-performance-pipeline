from pathlib import Path

from src.gcp_utils import (
    GCPStorageManager
)


def download_artifacts():

    storage_manager = (
        GCPStorageManager()
    )

    # Create directories
    Path(
        "artifacts/model"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    Path(
        "artifacts/processed"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    # Download model
    storage_manager.download_file(
        "model/model.pkl",
        "artifacts/model/model.pkl"
    )

    # Download preprocessor
    storage_manager.download_file(
        "processed/preprocessor.pkl",
        "artifacts/processed/preprocessor.pkl"
    )

    print(
        "Artifacts downloaded successfully"
    )


if __name__ == "__main__":

    download_artifacts()