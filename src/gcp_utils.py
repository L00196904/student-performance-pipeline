from google.cloud import storage


class GCPStorageManager:
    """
    Google Cloud Storage utility class.
    """

    def __init__(self):

        self.bucket_name = (
            "student-performance-mlops-artifacts"
        )

        self.client = storage.Client()

        self.bucket = self.client.bucket(
            self.bucket_name
        )

    def upload_file(
        self,
        source_file_path,
        destination_blob_name
    ):
        """
        Upload a local file to GCS.
        """

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

    def download_file(
        self,
        source_blob_name,
        destination_file_path
    ):
        """
        Download a file from GCS.
        """

        blob = self.bucket.blob(
            source_blob_name
        )

        blob.download_to_filename(
            destination_file_path
        )

        print(
            f"Downloaded "
            f"{source_blob_name} "
            f"to "
            f"{destination_file_path}"
        )

    def file_exists(
        self,
        blob_name
    ):
        """
        Check whether a blob exists.
        """

        blob = self.bucket.blob(
            blob_name
        )

        return blob.exists()

    def list_files(
        self,
        prefix=None
    ):
        """
        List files in bucket.
        """

        blobs = self.client.list_blobs(
            self.bucket_name,
            prefix=prefix
        )

        return [
            blob.name
            for blob in blobs
        ]