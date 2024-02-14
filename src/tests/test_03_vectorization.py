from itertools import product

import pytest
from dotenv import load_dotenv

from ..engine.vectorization import EMBEDDING_DICT, LOADER_DICT, SPLITTER_DICT, Vectorizer

load_dotenv()


@pytest.fixture
def temp_folder(tmpdir):
    # Create temporary directory with some sample documents
    documents = ["doc1.txt", "doc2.txt", "doc3.txt"]
    for doc in documents:
        tmpdir.join(doc).write("Sample document content.")
    return str(tmpdir)


@pytest.mark.dependency(name="vectorizer", scope="session")
def test_vectorizer(temp_folder):
    for loader_name, splitter_name, embedding_name in product(LOADER_DICT, SPLITTER_DICT, EMBEDDING_DICT):
        vectorizer = Vectorizer(
            document_folder=temp_folder,
            loader_name=loader_name,
            splitter_name=splitter_name,
            embedding_name=embedding_name,
            chunk_size=1000,
            chunk_overlap=100,
        )
        vectorizer.create_vectorstore()
        assert hasattr(vectorizer, "vectorstore")
