import os
import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
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

from src.upload_artifacts import (
    upload_artifacts
)


logging = setup_logging()

config = read_yaml("config.yaml")


class ModelTraining:
    """
    End-to-End Model Training Pipeline
    """

    def __init__(self):

        # Processed datasets
        self.train_data_path = (
            config["artifacts"]["train_data_path"]
        )

        self.test_data_path = (
            config["artifacts"]["test_data_path"]
        )

        # Model path
        self.model_path = (
            config["training"]["model_path"]
        )

        # Metrics path
        self.metrics_path = (
            "artifacts/metrics/metrics.json"
        )

        # Hyperparameters
        self.random_state = (
            config["training"]["random_state"]
        )

        self.n_estimators = (
            config["training"]["n_estimators"]
        )

        self.max_depth = (
            config["training"]["max_depth"]
        )


    # Load Processed Data
    def load_data(self):

        logging.info(
            "Loading processed datasets"
        )

        train_df = pd.read_csv(
            self.train_data_path
        )

        test_df = pd.read_csv(
            self.test_data_path
        )

        logging.info(
            "Datasets loaded successfully"
        )

        logging.info(
            f"Train dataset shape: {train_df.shape}"
        )

        logging.info(
            f"Test dataset shape: {test_df.shape}"
        )

        return train_df, test_df


    # Split Features and Target
    def split_features_target(
        self,
        train_df,
        test_df
    ):

        logging.info(
            "Splitting features and target"
        )

        X_train = train_df.drop(
            columns=["target"]
        )

        y_train = train_df["target"]

        X_test = test_df.drop(
            columns=["target"]
        )

        y_test = test_df["target"]

        return (
            X_train,
            X_test,
            y_train,
            y_test
        )


    # Train Model
    def train_model(
        self,
        X_train,
        y_train
    ):

        logging.info(
            "Training Random Forest model"
        )

        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )

        model.fit(
            X_train,
            y_train
        )

        logging.info(
            "Model training completed"
        )

        return model


    # Evaluate Model
    def evaluate_model(
        self,
        model,
        X_test,
        y_test
    ):

        logging.info(
            "Evaluating model"
        )

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
            "MAE": float(mae),
            "MSE": float(mse),
            "RMSE": float(rmse),
            "R2_SCORE": float(r2)
        }

        logging.info(
            f"Evaluation Metrics: {metrics}"
        )

        return metrics


    # Save Model
    def save_model(
        self,
        model
    ):

        logging.info(
            "Saving trained model"
        )

        model_dir = os.path.dirname(
            self.model_path
        )

        create_directory(
            model_dir
        )

        joblib.dump(
            model,
            self.model_path
        )

        logging.info(
            f"Model saved at: {self.model_path}"
        )


    # Save Metrics
    def save_metrics(
        self,
        metrics
    ):

        logging.info(
            "Saving evaluation metrics"
        )

        metrics_dir = os.path.dirname(
            self.metrics_path
        )

        create_directory(
            metrics_dir
        )

        with open(
            self.metrics_path,
            "w"
        ) as json_file:

            json.dump(
                metrics,
                json_file,
                indent=4
            )

        logging.info(
            f"Metrics saved at: {self.metrics_path}"
        )


    # Execute Pipeline
    def run(self):

        logging.info(
            "Starting model training pipeline"
        )

        # Load datasets
        train_df, test_df = (
            self.load_data()
        )

        # Split features/target
        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = self.split_features_target(
            train_df,
            test_df
        )

        # Train model
        model = self.train_model(
            X_train,
            y_train
        )

        # Evaluate model
        metrics = self.evaluate_model(
            model,
            X_test,
            y_test
        )

        # Save model locally
        self.save_model(model)

        # Save metrics locally
        self.save_metrics(metrics)

        # Upload artifacts to GCP
        upload_artifacts()

        logging.info(
            "Training pipeline completed"
        )

        print("\n========== MODEL METRICS ==========")

        for key, value in metrics.items():

            print(f"{key}: {value}")

        print("===================================\n")

        return model, metrics


if __name__ == "__main__":
    trainer = ModelTraining()
    trainer.run()