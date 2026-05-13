import os

from src.gcp_utils import (
    GCPStorageManager
)


def upload_artifacts():
    storage_manager = (
        GCPStorageManager()
    )

    artifacts = [

        (
            "artifacts/model/model.pkl",
            "model/model.pkl"
        ),

        (
            "artifacts/processed/preprocessor.pkl",
            "processed/preprocessor.pkl"
        ),

        (
            "artifacts/metrics/metrics.json",
            "metrics/metrics.json"
        )
    ]

    for source, destination in artifacts:

        # Check if file exists
        if os.path.exists(source):

            storage_manager.upload_file(
                source,
                destination
            )
            print(
                f"Uploaded {source} to {destination}"
            )
        else:

            print(
                f"File not found: {source}"
            )

if __name__ == "__main__":
    upload_artifacts()