import os
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

logging = setup_logging()

config = read_yaml("config.yaml")


class ModelRetraining:
    """
    Retraining pipeline that creates a
    challenger model.
    """

    def __init__(self):

        self.train_data_path = (
            config["artifacts"]["train_data_path"]
        )

        self.test_data_path = (
            config["artifacts"]["test_data_path"]
        )

        self.candidate_model_path = (
            "artifacts/model/candidate_model.pkl"
        )

        self.random_state = (
            config["training"]["random_state"]
        )

        self.n_estimators = (
            config["training"]["n_estimators"]
        )

        self.max_depth = (
            config["training"]["max_depth"]
        )

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

        return train_df, test_df

    def split_features_target(
        self,
        train_df,
        test_df
    ):

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

    def train_model(
        self,
        X_train,
        y_train
    ):

        logging.info(
            "Training challenger model"
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

        return model

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

        metrics = {

            "MAE": mae,

            "MSE": mse,

            "RMSE": rmse,

            "R2_SCORE": r2
        }

        logging.info(
            f"Candidate metrics: "
            f"{metrics}"
        )

        return metrics

    def save_candidate_model(
        self,
        model
    ):

        model_dir = os.path.dirname(
            self.candidate_model_path
        )

        create_directory(
            model_dir
        )

        joblib.dump(
            model,
            self.candidate_model_path
        )

        logging.info(
            f"Candidate model saved at "
            f"{self.candidate_model_path}"
        )

    def run(self):

        logging.info(
            "Starting retraining pipeline"
        )

        train_df, test_df = (
            self.load_data()
        )

        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = self.split_features_target(
            train_df,
            test_df
        )

        model = self.train_model(
            X_train,
            y_train
        )

        metrics = self.evaluate_model(
            model,
            X_test,
            y_test
        )

        self.save_candidate_model(
            model
        )

        print(
            "\n========== CANDIDATE MODEL =========="
        )

        for key, value in metrics.items():

            print(
                f"{key}: {value}"
            )

        print(
            "=====================================\n"
        )

        return model, metrics


if __name__ == "__main__":

    retrainer = (
        ModelRetraining()
    )

    retrainer.run()