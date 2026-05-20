import joblib
import pandas as pd

from src.utils import (
    read_yaml,
    setup_logging
)

from src.log_predictions import (
    log_prediction
)

logging = setup_logging()

config = read_yaml("config.yaml")


class PredictionPipeline:
    """
    Deployment-ready prediction pipeline.
    """

    def __init__(self):

        self.model_path = (
            config["prediction"]["model_path"]
        )

        self.preprocessor_path = (
            config["prediction"]["preprocessor_path"]
        )

    def load_model(self):
        """
        Load trained model.
        """

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

    def load_preprocessor(self):
        """
        Load fitted preprocessor.
        """

        logging.info(
            "Loading preprocessor"
        )

        preprocessor = joblib.load(
            self.preprocessor_path
        )

        logging.info(
            "Preprocessor loaded successfully"
        )

        return preprocessor

    def predict(
        self,
        input_data
    ):
        """
        Generate prediction and log it.
        """

        logging.info(
            "Starting prediction pipeline"
        )

        # Load model artifacts
        model = self.load_model()

        preprocessor = self.load_preprocessor()

        # Convert request to dataframe
        input_df = pd.DataFrame(
            [input_data]
        )

        # Apply preprocessing
        processed_data = (
            preprocessor.transform(
                input_df
            )
        )

        # Generate prediction
        prediction = model.predict(
            processed_data
        )[0]

        # Log prediction
        log_prediction(
            features=input_data,
            prediction=prediction
        )

        logging.info(
            f"Prediction generated: {prediction}"
        )

        return prediction


if __name__ == "__main__":

    sample_input = {
        "hours_studied": 7,
        "previous_scores": 85,
        "extracurricular_activities": "Yes",
        "sleep_hours": 7,
        "sample_question_papers_practiced": 5
    }

    predictor = PredictionPipeline()

    prediction = predictor.predict(
        sample_input
    )

    print(
        f"\nPredicted Score: "
        f"{prediction:.2f}\n"
    )