from pathlib import Path
import json
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset

REFERENCE_DATA = (
    "artifacts/monitoring/reference_data.csv"
)

CURRENT_DATA = (
    "artifacts/monitoring/current_data.csv"
)

OUTPUT_JSON = (
    "artifacts/drift/drift_metrics.json"
)


def run_drift_detection():

    # Load datasets
    reference_data = pd.read_csv(
        REFERENCE_DATA
    )

    current_data = pd.read_csv(
        CURRENT_DATA
    )

    # Ensure matching columns
    common_columns = list(
        set(reference_data.columns)
        .intersection(
            set(current_data.columns)
        )
    )

    reference_data = (
        reference_data[common_columns]
    )

    current_data = (
        current_data[common_columns]
    )

    print("Reference columns:")
    print(
        reference_data.columns.tolist()
    )

    print("\nCurrent columns:")
    print(
        current_data.columns.tolist()
    )

    # Create output directory
    Path(
        OUTPUT_JSON
    ).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Run Evidently report
    report = Report(
        metrics=[
            DataDriftPreset()
        ]
    )

    snapshot = report.run(
        reference_data=reference_data,
        current_data=current_data
    )

    print("\nReport generated")

    print(
        f"Snapshot type: {type(snapshot)}"
    )

    # Show available methods
    print("\nAvailable methods:")
    print(dir(snapshot))

    # Try exporting report
    report_data = None

    try:
        report_data = (
            snapshot.model_dump()
        )

        print(
            "\nUsing model_dump()"
        )

    except Exception as e:

        print(
            f"\nmodel_dump() failed: {e}"
        )

        try:

            report_data = (
                snapshot.dict()
            )

            print(
                "Using dict()"
            )

        except Exception as e:

            print(
                f"dict() failed: {e}"
            )

            try:

                report_data = (
                    json.loads(
                        snapshot.json()
                    )
                )

                print(
                    "Using json()"
                )

            except Exception as e:

                print(
                    f"json() failed: {e}"
                )

    if report_data is not None:

        with open(
            OUTPUT_JSON,
            "w"
        ) as file:

            json.dump(
                report_data,
                file,
                indent=4
            )

        print(
            f"\nReport saved to "
            f"{OUTPUT_JSON}"
        )

    else:

        print(
            "\nCould not export "
            "Snapshot data."
        )

    drift_share = 0.0

    for metric in report_data["metrics"]:

        if (
            "DriftedColumnsCount"
            in metric["metric_name"]
        ):

            drift_share = (
                metric["value"]["share"]
            )

    print(
        f"\nDrift Share: "
        f"{drift_share}"
    )

    return drift_share > 0.30




if __name__ == "__main__":

    run_drift_detection()