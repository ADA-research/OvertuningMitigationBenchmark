import os
import logging
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging to show warnings and above
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GCSManager:
    def __init__(self, key_file_path='google-cloud-phd-storage-key.json'):
        """
        Attempts to authenticate with Google Cloud Storage.
        If it fails, it logs a warning and sets the client to None,
        allowing the rest of the program to continue.
        """
        self.client = None
        self.bucket_name = "overtuningbenchmark-prd"

        # 1. Check if file exists
        if not os.path.exists(key_file_path):
            logging.warning(f"GCS Auth File '{key_file_path}' not found. GCS features will be DISABLED.")
            return

        # 2. Attempt Authentication
        try:
            self.client = storage.Client.from_service_account_json(key_file_path)
            logging.info(f"GCS Authentication successful using {key_file_path}.")
        except Exception as e:
            logging.warning(f"GCS Authentication failed: {e}. GCS features will be DISABLED.")
            self.client = None

    def upload_blob(self, source_file_name, destination_blob_name):
        """
        Uploads a file to the bucket.
        Fails silently (with a warning) if auth failed or upload errors occur.
        """
        # Check if we are authenticated
        if self.client is None:
            logging.warning("Skipping upload: Not authenticated to GCS.")
            return

        try:
            bucket = self.client.bucket(self.bucket_name)
            blob = bucket.blob(destination_blob_name)
            blob.upload_from_filename(source_file_name)

            logging.info(f"Successfully uploaded: {source_file_name} -> gs://{self.bucket_name}/{destination_blob_name}")

        except Exception as e:
            logging.warning(f"Failed to upload '{source_file_name}': {e}. Continuing execution...")

    def download_blob(self, source_blob_name, destination_file_name):
        """
        Downloads a blob from the bucket.
        Fails silently (with a warning) if auth failed or download errors occur.
        """
        # Check if we are authenticated
        if self.client is None:
            logging.warning("Skipping download: Not authenticated to GCS.")
            return

        try:
            bucket = self.client.bucket(self.bucket_name)
            blob = bucket.blob(source_blob_name)
            blob.download_to_filename(destination_file_name)

            logging.info(f"Successfully downloaded: gs://{self.bucket_name}/{source_blob_name} -> {destination_file_name}")

        except Exception as e:
            logging.warning(f"Failed to download '{source_blob_name}': {e}. Continuing execution...")

    def download_experiment_results(self, experiment_name: str, n_workers: int = None):
        """
        Downloads matching experiment files (task_config.yaml and history.csv) in parallel.

        Args:
            experiment_name: Name of the top-level experiment folder in the bucket.
            n_workers: Number of threads to use. If None, defaults to os.cpu_count() or 4.
        """
        if self.client is None:
            logging.warning("Skipping download: Not authenticated to GCS.")
            return

        if n_workers is None:
            n_workers = os.cpu_count() or 4

        bucket = self.client.bucket(self.bucket_name)
        prefix = f"{experiment_name}/"
        blobs_iter = self.client.list_blobs(self.bucket_name, prefix=prefix)

        # Collect matching blobs first
        to_download = []  # list of (blob, local_path)
        for blob in blobs_iter:
            if blob.name.endswith("task_config.yaml") or blob.name.endswith("history.csv"):
                rel_path = blob.name
                local_path = os.path.join("results", rel_path)
                to_download.append((blob, local_path))

        if not to_download:
            logging.info(f"No matching files found in gs://{self.bucket_name}/{prefix}")
            return

        # Ensure directories exist (do this before threading to avoid races)
        for _blob, local_path in to_download:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

        def _download_one(blob, local_path):
            try:
                print(f"Downloading: {blob.name} → {local_path}")
                if not os.path.exists(local_path):  # Avoid re-downloading if file already exists
                    blob.download_to_filename(local_path)
                logging.info(f"Downloaded: gs://{self.bucket_name}/{blob.name} -> {local_path}")
            except Exception as e:
                logging.warning(f"Failed to download '{blob.name}': {e}. Continuing execution...")

        # Use ThreadPool for IO-bound downloads
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_download_one, b, p) for b, p in to_download]
            for fut in as_completed(futures):
                # results() just to propagate unexpected exceptions (they are handled inside _download_one)
                try:
                    fut.result()
                except Exception as e:
                    logging.warning(f"Unexpected failure while downloading: {e}")

        logging.info("Download complete.")


    def list_experiment_results(self, experiment_name: str):
        if self.client is None:
            return []

        blobs_iter = self.client.list_blobs(self.bucket_name, prefix=experiment_name)

        return [blob.name for blob in blobs_iter]




if __name__ == "__main__":
    gcs_manager = GCSManager()
    gcs_manager.download_experiment_results("RealMLP_363696_20260417_203153", n_workers=8)