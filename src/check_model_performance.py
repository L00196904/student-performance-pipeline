import json
import sys


THRESHOLD_R2 = 0.85
THRESHOLD_RMSE = 10


with open(
    "artifacts/metrics/metrics.json",
    "r"
) as file:

    metrics = json.load(file)


r2_score = metrics["R2_SCORE"]
rmse = metrics["RMSE"]


print(f"Current R2 Score: {r2_score}")
print(f"Current RMSE: {rmse}")


# Retraining Conditions
if (
    r2_score < THRESHOLD_R2
    or rmse > THRESHOLD_RMSE
):

    print(
        "Performance degraded. "
        "Triggering retraining pipeline."
    )

    sys.exit(1)

else:

    print(
        "Model performance acceptable."
    )

    sys.exit(0)