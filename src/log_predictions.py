import json
import os

from datetime import datetime

LOG_FILE = (
    "artifacts/predictions/predictions.jsonl"
)

TRIGGER_THRESHOLD = 10


def trigger_pipeline():

    try:

        from src.prepare_monitoring_data import (
            prepare_monitoring_data
        )

        from src.upload_prediction_logs import (
            upload_prediction_logs
        )

        from src.drift_detection import (
            run_drift_detection
        )

        from src.retraining_pipeline import (
            run_pipeline
        )

        print(
            "\nMonitoring threshold reached"
        )

        prepare_monitoring_data()

        upload_prediction_logs()

        run_drift_detection()

        run_pipeline()

    except Exception as error:

        print(
            f"\nAutomation failed: "
            f"{error}"
        )


def log_prediction(
    features,
    prediction,
    actual=None
):

    os.makedirs(
        "artifacts/predictions",
        exist_ok=True
    )

    record = {

        "timestamp":
        datetime.utcnow().isoformat(),

        "features":
        features,

        "prediction":
        float(prediction),

        "actual":
        actual
    }

    with open(
        LOG_FILE,
        "a"
    ) as file:

        file.write(
            json.dumps(record)
            + "\n"
        )

    with open(
        LOG_FILE,
        "r"
    ) as file:

        prediction_count = sum(
            1 for _ in file
        )

    print(
        f"Prediction Count: "
        f"{prediction_count}"
    )

    if prediction_count >= TRIGGER_THRESHOLD:

        trigger_pipeline()