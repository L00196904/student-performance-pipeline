import json

DRIFT_THRESHOLD = 0.30

MINIMUM_RECORDS = 100

MINIMUM_R2 = 0.85

DRIFT_FILE = (
    "artifacts/drift/drift_metrics.json"
)

PERFORMANCE_FILE = (
    "artifacts/monitoring/performance_metrics.json"
)

PREDICTIONS_FILE = (
    "artifacts/predictions/predictions.jsonl"
)


def get_drift_share():

    with open(
        DRIFT_FILE,
        "r"
    ) as file:

        report = json.load(file)

    for metric in report["metrics"]:

        if (
            "DriftedColumnsCount"
            in metric["metric_name"]
        ):

            return (
                metric["value"]["share"]
            )

    return 0.0


def get_prediction_count():

    count = 0

    with open(
        PREDICTIONS_FILE,
        "r"
    ) as file:

        for _ in file:

            count += 1

    return count


def get_production_r2():

    try:

        with open(
            PERFORMANCE_FILE,
            "r"
        ) as file:

            metrics = json.load(
                file
            )

        return metrics.get(
            "R2_SCORE",
            None
        )

    except Exception:

        return None


def check_retraining_needed():

    drift_share = (
        get_drift_share()
    )

    prediction_count = (
        get_prediction_count()
    )

    production_r2 = (
        get_production_r2()
    )

    print(
        f"Drift Share: "
        f"{drift_share}"
    )

    print(
        f"Prediction Count: "
        f"{prediction_count}"
    )

    print(
        f"Production R2: "
        f"{production_r2}"
    )

    if prediction_count < MINIMUM_RECORDS:

        print(
            "\nNot enough production data"
        )

        return False

    if drift_share >= DRIFT_THRESHOLD:

        print(
            "\nRetraining triggered "
            "by drift detection"
        )

        return True

    if (
        production_r2 is not None
        and
        production_r2 < MINIMUM_R2
    ):

        print(
            "\nRetraining triggered "
            "by accuracy degradation"
        )

        return True

    print(
        "\nNo retraining required"
    )

    return False


if __name__ == "__main__":

    check_retraining_needed()