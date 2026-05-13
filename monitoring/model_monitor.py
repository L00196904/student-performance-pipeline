import json
import sys


# Threshold Configuration
R2_DROP_THRESHOLD = 0.10
RMSE_INCREASE_THRESHOLD = 3.0



# Load Baseline Metrics
with open(
    "monitoring/baseline_metrics.json",
    "r"
) as baseline_file:

    baseline_metrics = json.load(
        baseline_file
    )



# Load Current Metrics
with open(
    "artifacts/metrics/metrics.json",
    "r"
) as current_file:

    current_metrics = json.load(
        current_file
    )


baseline_r2 = (
    baseline_metrics["R2_SCORE"]
)

baseline_rmse = (
    baseline_metrics["RMSE"]
)

current_r2 = (
    current_metrics["R2_SCORE"]
)

current_rmse = (
    current_metrics["RMSE"]
)


print(
    f"Baseline R2: {baseline_r2}"
)

print(
    f"Current R2: {current_r2}"
)

print(
    f"Baseline RMSE: {baseline_rmse}"
)

print(
    f"Current RMSE: {current_rmse}"
)



# Threshold Logic
r2_drop = (
    baseline_r2 - current_r2
)

rmse_increase = (
    current_rmse - baseline_rmse
)


if (
    r2_drop > R2_DROP_THRESHOLD
    or
    rmse_increase > RMSE_INCREASE_THRESHOLD
):

    print(
        "Model drift detected."
    )

    print(
        "Retraining required."
    )

    sys.exit(1)

else:
    print(
        "Model performance acceptable."
    )

    sys.exit(0)