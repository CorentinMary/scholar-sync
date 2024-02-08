import os

import pytest

from ..data_preprocessing import download_documents


@pytest.fixture
def test_documents():
    return [
        {
            "bucket": "llm-technical-test-data",
            "prefix": "raw_pdf/82_2020_217.pdf",
            "file_format": ["pdf"],
            "destination": "./src/tests/test_data",
        },
        {
            "bucket": "llm-technical-test-data",
            "prefix": "extracted_text/PMC",
            "file_format": ["txt"],
            "destination": "./src/tests/test_data",
        },
    ]


def test_load_documents(test_documents):
    for documents in test_documents:
        download_documents(**documents)
        assert len(os.listdir("./src/tests/test_data"))
        # emptying the test_data/ folder to make sure there is no residual document at next execution
        for file in os.listdir("./src/tests/test_data"):
            os.remove(file)
