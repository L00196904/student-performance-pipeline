import os
import yaml
import logging
from datetime import datetime


def create_directory(path: str):
    """
    Create directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def read_yaml(path_to_yaml: str):
    """
    Read YAML configuration file.
    """
    with open(path_to_yaml, "r") as yaml_file:
        content = yaml.safe_load(yaml_file)

    return content


def setup_logging(log_dir="logs"):
    """
    Configure logging.
    """

    create_directory(log_dir)

    log_file = os.path.join(
        log_dir,
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    return logging