import os
from typing import List

from google.cloud import storage


def download_documents(
    bucket_name: str,
    prefix: str = "extracted-text",
    file_format: List[str] = ["txt", "pdf"],
    destination: str = "./data",
) -> None:
    """Downloads a set of documents from a bucket.

    :param bucket_name: str
        name of the bucket to download the documents from.
    :param prefix: str, defaults to "extracted-text".
        prefix to filter to the documents on, such as the name of the folder where they are located.
    :param file_format: List[str], defaults to ["txt", "pdf"].
        format(s) of the files to dowload.
    :param destination: str, defaults to "./data".
        path where to locally store the downloaded documents.
    """
    assert os.path.exists(destination), f"Location {destination} does not exist."

    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name=bucket_name)
    for blob in bucket.list_blobs():
        if blob.name.startswith(prefix) and blob.name.split(".")[-1] in file_format:
            file_name = blob.name.split("/")[-1]
            blob.download_to_filename(f"{destination}/{file_name}")
