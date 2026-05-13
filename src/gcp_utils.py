import os

from google.cloud import storage


class GCPStorageManager:
    """
    Google Cloud Storage utility.
    """

    def __init__(self):

        os.environ[
            "GOOGLE_APPLICATION_CREDENTIALS"
        ] = (
            "credentials/service-account.json"
        )

        self.bucket_name = (
            "student-performance-mlops-artifacts"
        )

        self.client = storage.Client()

        self.bucket = self.client.bucket(
            self.bucket_name
        )

    # Upload File
    def upload_file(
        self,
        source_file_path,
        destination_blob_name
    ):

        blob = self.bucket.blob(
            destination_blob_name
        )

        blob.upload_from_filename(
            source_file_path
        )

        print(
            f"Uploaded "
            f"{source_file_path} "
            f"to "
            f"{destination_blob_name}"
        )