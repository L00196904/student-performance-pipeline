import shutil
import joblib
import pandas as pd

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from src.utils import (
    read_yaml,
    setup_logging
)

from src.upload_artifacts import (
    upload_artifacts
)

logging = setup_logging()

config = read_yaml(
    "config.yaml"
)


class ModelComparison:

    def __init__(self):

        self.test_data_path = (
            config["artifacts"]["test_data_path"]
        )

        self.production_model_path = (
            config["training"]["model_path"]
        )

        self.candidate_model_path = (
            "artifacts/model/candidate_model.pkl"
        )

    def load_test_data(self):

        logging.info(
            "Loading test dataset"
        )

        test_df = pd.read_csv(
            self.test_data_path
        )

        X_test = test_df.drop(
            columns=["target"]
        )

        y_test = test_df["target"]

        return (
            X_test,
            y_test
        )

    def evaluate_model(
        self,
        model,
        X_test,
        y_test
    ):

        predictions = model.predict(
            X_test
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        mse = mean_squared_error(
            y_test,
            predictions
        )

        rmse = mse ** 0.5

        r2 = r2_score(
            y_test,
            predictions
        )

        return {

            "MAE": mae,

            "MSE": mse,

            "RMSE": rmse,

            "R2_SCORE": r2
        }

    def promote_candidate(self):

        shutil.copyfile(

            self.candidate_model_path,

            self.production_model_path
        )

        logging.info(
            "Candidate promoted "
            "to production"
        )

    def run(self):

        logging.info(
            "Starting model comparison"
        )

        X_test, y_test = (
            self.load_test_data()
        )

        production_model = (
            joblib.load(
                self.production_model_path
            )
        )

        candidate_model = (
            joblib.load(
                self.candidate_model_path
            )
        )

        production_metrics = (
            self.evaluate_model(
                production_model,
                X_test,
                y_test
            )
        )

        candidate_metrics = (
            self.evaluate_model(
                candidate_model,
                X_test,
                y_test
            )
        )

        print(
            "\n===== CHAMPION MODEL ====="
        )

        for key, value in (
            production_metrics.items()
        ):

            print(
                f"{key}: {value}"
            )

        print(
            "\n===== CHALLENGER MODEL ====="
        )

        for key, value in (
            candidate_metrics.items()
        ):

            print(
                f"{key}: {value}"
            )

        champion_r2 = (
            production_metrics[
                "R2_SCORE"
            ]
        )

        challenger_r2 = (
            candidate_metrics[
                "R2_SCORE"
            ]
        )

        print(
            "\n===== DECISION ====="
        )

        if challenger_r2 > champion_r2:

            print(
                "Challenger wins"
            )

            self.promote_candidate()

            upload_artifacts()

            print(
                "New production model "
                "uploaded to GCP"
            )

            print(
                "=====================\n"
            )

            return True

        print(
            "Champion retained"
        )

        print(
            "=====================\n"
        )

        return False


if __name__ == "__main__":

    comparison = (
        ModelComparison()
    )

    comparison.run()