import json
import os
from datetime import datetime

LOG_FILE = (
    "artifacts/predictions/predictions.jsonl"
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