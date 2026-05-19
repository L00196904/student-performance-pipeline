import json

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

LOG_FILE = (
    "artifacts/predictions/predictions.jsonl"
)

OUTPUT_FILE = (
    "artifacts/monitoring/performance_metrics.json"
)

MINIMUM_LABELS = 20


def monitor_performance():

    y_true = []

    y_pred = []

    with open(
        LOG_FILE,
        "r"
    ) as file:

        for line in file:

            row = json.loads(line)

            actual = row.get(
                "actual"
            )

            prediction = row.get(
                "prediction"
            )

            if actual is not None:

                y_true.append(
                    actual
                )

                y_pred.append(
                    prediction
                )

    if len(y_true) < MINIMUM_LABELS:

        print(
            f"Only {len(y_true)} labelled records found"
        )

        return None

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_true,
        y_pred
    )

    metrics = {

        "labelled_records":
        len(y_true),

        "MAE":
        float(mae),

        "MSE":
        float(mse),

        "RMSE":
        float(rmse),

        "R2_SCORE":
        float(r2)
    }

    with open(
        OUTPUT_FILE,
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print(
        "\n===== PRODUCTION PERFORMANCE ====="
    )

    for key, value in metrics.items():

        print(
            f"{key}: {value}"
        )

    print(
        "=================================="
    )

    return metrics


if __name__ == "__main__":

    monitor_performance()