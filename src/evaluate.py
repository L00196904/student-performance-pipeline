import os
import joblib
import pandas as pd
import json

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.utils import (
    create_directory,
    read_yaml,
    setup_logging
)


logging = setup_logging()


config = read_yaml("config.yaml")


class ModelEvaluation:
    """
    Standalone model evaluation pipeline.
    """

    def __init__(self):

        self.test_data_path = (
            config["artifacts"]["test_data_path"]
        )

        self.model_path = (
            config["prediction"]["model_path"]
        )

        self.metrics_path = (
            config["evaluation"]["metrics_path"]
        )


    def load_test_data(self):

        logging.info(
            "Loading test dataset"
        )

        test_df = pd.read_csv(
            self.test_data_path
        )

        logging.info(
            "Test dataset loaded successfully"
        )

        return test_df


    def load_model(self):

        logging.info(
            "Loading trained model"
        )

        model = joblib.load(
            self.model_path
        )

        logging.info(
            "Model loaded successfully"
        )

        return model


    def evaluate(
        self,
        model,
        test_df
    ):

        logging.info(
            "Starting model evaluation"
        )

        X_test = test_df.drop(
            columns=["target"]
        )

        y_test = test_df["target"]

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

        metrics = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2_SCORE": r2
        }

        logging.info(
            f"Evaluation Metrics: {metrics}"
        )

        return metrics


    def save_metrics(self, metrics):

        logging.info(
            "Saving evaluation metrics"
        )

        metrics_dir = os.path.dirname(
            self.metrics_path
        )

        create_directory(metrics_dir)

        with open(
            self.metrics_path,
            "w"
        ) as file:

            for key, value in metrics.items():

                file.write(
                    f"{key}: {value}\n"
                )
        
            # Save JSON file
        json_metrics_path = (
            "artifacts/metrics/metrics.json"
        )

        with open(
            json_metrics_path,
            "w"
        ) as json_file:

            json.dump(
                metrics,
                json_file,
                indent=4
            )

        logging.info(
            f"Metrics saved at: "
            f"{self.metrics_path}"
        )


    def run(self):

        logging.info(
            "Starting evaluation pipeline"
        )

        # Load test data
        test_df = self.load_test_data()

        # Load model
        model = self.load_model()

        # Evaluate
        metrics = self.evaluate(
            model,
            test_df
        )

        # Save metrics
        self.save_metrics(metrics)

        logging.info(
            "Evaluation pipeline completed"
        )

        print("\n========== EVALUATION METRICS ==========")

        for key, value in metrics.items():
            print(f"{key}: {value}")

        print("========================================\n")

        return metrics


if __name__ == "__main__":

    evaluator = ModelEvaluation()

    evaluator.run()