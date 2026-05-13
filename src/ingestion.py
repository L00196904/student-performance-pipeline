import os
import shutil
import pandas as pd

from src.utils import (
    create_directory,
    read_yaml,
    setup_logging
)

# Setup logger
logging = setup_logging()

# Read configuration
config = read_yaml("config.yaml")


class DataIngestion:
    """
    Data Ingestion Pipeline
    """

    def __init__(self):

        self.source_file = config["data_source"]["source_file"]

        self.raw_data_path = config["artifacts"]["raw_data_path"]

        self.allowed_extensions = config["ingestion"]["allowed_extensions"]

    def validate_file(self):
        """
        Validate dataset existence and extension.
        """

        logging.info("Starting file validation")

        # Check file existence
        if not os.path.exists(self.source_file):
            raise FileNotFoundError(
                f"Dataset not found at {self.source_file}"
            )

        # Validate extension
        file_extension = os.path.splitext(self.source_file)[1]

        if file_extension not in self.allowed_extensions:
            raise ValueError(
                f"Invalid file format: {file_extension}"
            )

        logging.info("File validation successful")

    def load_dataset(self):
        """
        Load dataset into pandas dataframe.
        """

        logging.info("Loading dataset")

        df = pd.read_csv(self.source_file)

        logging.info(f"Dataset loaded successfully")

        logging.info(f"Dataset shape: {df.shape}")

        return df

    def save_raw_data(self):
        """
        Save raw dataset into artifacts folder.
        """

        logging.info("Saving raw dataset")

        artifact_dir = os.path.dirname(self.raw_data_path)

        create_directory(artifact_dir)

        shutil.copy(
            self.source_file,
            self.raw_data_path
        )

        logging.info(
            f"Raw dataset saved at: {self.raw_data_path}"
        )

    def run(self):
        """
        Execute complete ingestion pipeline.
        """

        logging.info("Starting data ingestion pipeline")

        self.validate_file()

        df = self.load_dataset()

        self.save_raw_data()

        logging.info("Data ingestion pipeline completed")

        return df


if __name__ == "__main__":

    ingestion = DataIngestion()

    dataframe = ingestion.run()

    print(dataframe.head())