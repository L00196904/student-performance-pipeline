import json

PREDICTIONS_FILE = (
    "artifacts/predictions/predictions.jsonl"
)

MINIMUM_RECORDS = 100


def check_data_volume():

    count = 0

    with open(
        PREDICTIONS_FILE,
        "r"
    ) as file:

        for _ in file:

            count += 1

    print(
        f"Production records: "
        f"{count}"
    )

    if count >= MINIMUM_RECORDS:

        print(
            "Enough data available"
        )

        return True

    print(
        "Not enough data"
    )

    return False


if __name__ == "__main__":

    check_data_volume()