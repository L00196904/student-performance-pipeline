from pathlib import Path
import json
import pandas as pd

INPUT_FILE = (
    "artifacts/predictions/predictions.jsonl"
)

OUTPUT_FILE = (
    "artifacts/monitoring/current_data.csv"
)


def prepare_monitoring_data():

    records = []

    with open(
        INPUT_FILE,
        "r"
    ) as file:

        for line in file:

            row = json.loads(line)

            features = row["features"]

            records.append(features)

    df = pd.DataFrame(records)

    # create parent directory
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Monitoring dataset saved "
        f"to {OUTPUT_FILE}"
    )


if __name__ == "__main__":

    prepare_monitoring_data()