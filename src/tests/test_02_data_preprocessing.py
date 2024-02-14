import tempfile

import pytest

from ..data_preprocessing import download_documents
from ..utils import fetch_files


@pytest.mark.dependency(name="load_documents", depends=["fetch_files"], scope="session")
def test_load_documents():
    prefix_list = ["raw-pdf/82_2020_217.pdf", "extracted-text/PMC"]
    expected_file_list = [["82_2020_217.pdf"], ["PMC8325057.txt", "PMC8198544.txt"]]
    for prefix, expected_file in zip(prefix_list, expected_file_list):
        with tempfile.TemporaryDirectory() as temp_dir:
            download_documents(bucket_name="llm-technical-test-data", prefix=prefix, destination=temp_dir)
            assert sorted(fetch_files(temp_dir)) == sorted(expected_file)
