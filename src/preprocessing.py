import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from src.utils import (
    create_directory,
    read_yaml,
    setup_logging
)

# Setup logger
logging = setup_logging()

# Load configuration
config = read_yaml("config.yaml")


class DataPreprocessing:
    """
    Data preprocessing pipeline.
    """

    def __init__(self):

        self.raw_data_path = config["artifacts"]["raw_data_path"]

        self.processed_dir = config["artifacts"]["processed_dir"]

        self.train_data_path = config["artifacts"]["train_data_path"]

        self.test_data_path = config["artifacts"]["test_data_path"]

        self.preprocessor_path = config["artifacts"]["preprocessor_path"]

        self.target_column = config["preprocessing"]["target_column"]

        self.test_size = config["preprocessing"]["test_size"]

        self.random_state = config["preprocessing"]["random_state"]

    def load_data(self):
        """
        Load raw dataset.
        """

        logging.info("Loading raw dataset")

        df = pd.read_csv(self.raw_data_path)

        logging.info(f"Dataset shape: {df.shape}")

        return df

    def identify_columns(self, df):
        """
        Separate numerical and categorical columns.
        """

        X = df.drop(columns=[self.target_column])

        numerical_columns = X.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        categorical_columns = X.select_dtypes(
            include=["object"]
        ).columns.tolist()

        logging.info(
            f"Numerical columns: {numerical_columns}"
        )

        logging.info(
            f"Categorical columns: {categorical_columns}"
        )

        return numerical_columns, categorical_columns

    def create_preprocessor(
        self,
        numerical_columns,
        categorical_columns
    ):
        """
        Create sklearn preprocessing pipeline.
        """

        logging.info("Creating preprocessing pipeline")

        # Numerical pipeline
        numerical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        # Categorical pipeline
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]
        )

        # Combine pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    numerical_pipeline,
                    numerical_columns
                ),
                (
                    "cat",
                    categorical_pipeline,
                    categorical_columns
                )
            ]
        )

        logging.info(
            "Preprocessing pipeline created successfully"
        )

        return preprocessor

    def split_data(self, df):
        """
        Split dataset into train and test sets.
        """

        logging.info("Splitting dataset")

        X = df.drop(columns=[self.target_column])

        y = df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        logging.info(
            f"Train shape: {X_train.shape}"
        )

        logging.info(
            f"Test shape: {X_test.shape}"
        )

        return X_train, X_test, y_train, y_test

    def save_processed_data(
        self,
        X_train_processed,
        X_test_processed,
        y_train,
        y_test
    ):
        """
        Save processed train and test datasets.
        """

        logging.info("Saving processed datasets")

        create_directory(self.processed_dir)

        train_df = pd.DataFrame(X_train_processed)

        train_df["target"] = y_train.reset_index(drop=True)

        test_df = pd.DataFrame(X_test_processed)

        test_df["target"] = y_test.reset_index(drop=True)

        train_df.to_csv(
            self.train_data_path,
            index=False
        )

        test_df.to_csv(
            self.test_data_path,
            index=False
        )

        logging.info(
            "Processed datasets saved successfully"
        )

    def save_preprocessor(self, preprocessor):
        """
        Save preprocessing object.
        """

        logging.info("Saving preprocessor object")

        joblib.dump(
            preprocessor,
            self.preprocessor_path
        )

        logging.info(
            f"Preprocessor saved at: "
            f"{self.preprocessor_path}"
        )

    def run(self):
        """
        Execute preprocessing pipeline.
        """

        logging.info(
            "Starting preprocessing pipeline"
        )

        # Load dataset
        df = self.load_data()

        # Identify columns
        numerical_columns, categorical_columns = (
            self.identify_columns(df)
        )

        # Create preprocessor
        preprocessor = self.create_preprocessor(
            numerical_columns,
            categorical_columns
        )

        # Split dataset
        X_train, X_test, y_train, y_test = (
            self.split_data(df)
        )

        # Fit preprocessor
        logging.info("Fitting preprocessor")

        X_train_processed = preprocessor.fit_transform(
            X_train
        )

        X_test_processed = preprocessor.transform(
            X_test
        )

        # Save processed datasets
        self.save_processed_data(
            X_train_processed,
            X_test_processed,
            y_train,
            y_test
        )

        # Save preprocessor
        self.save_preprocessor(preprocessor)

        logging.info(
            "Preprocessing pipeline completed"
        )

        return (
            X_train_processed,
            X_test_processed,
            y_train,
            y_test
        )


if __name__ == "__main__":
    preprocessing = DataPreprocessing()
    preprocessing.run()