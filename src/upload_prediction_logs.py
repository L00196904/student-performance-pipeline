from src.gcp_utils import (
    GCPStorageManager
)


def upload_prediction_logs():

    storage_manager = (
        GCPStorageManager()
    )

    storage_manager.upload_file(
        "artifacts/predictions/predictions.jsonl",
        "production_logs/predictions.jsonl"
    )

    print(
        "Prediction logs uploaded successfully"
    )


if __name__ == "__main__":

    upload_prediction_logs()