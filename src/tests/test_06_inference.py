from itertools import product

import pytest
from dotenv import load_dotenv

from ..inference import RETRIEVER_DICT, SUMMARIZER_DICT, InferencePipeline

load_dotenv()


@pytest.fixture
def temp_folder(tmpdir):
    # Create temporary directory with some sample documents
    documents = ["doc1.txt", "doc2.txt", "doc3.txt"]
    for doc in documents:
        tmpdir.join(doc).write("Sample document content.")
    return str(tmpdir)


@pytest.mark.dependency(
    name="inference",
    depends=["vectorizer", "similarity_retriever", "dummy_summarizer", "prompt_summarizer"],
    scope="session",
)
def test_inference(temp_folder):
    for retriever_name, summarizer_name in product(RETRIEVER_DICT, SUMMARIZER_DICT):
        pipeline = InferencePipeline(
            document_folder=temp_folder, retriever_name=retriever_name, summarizer_name=summarizer_name
        )
        input_text = "Some input text about vaccines."
        output = pipeline.predict(input_text=input_text, n_sim_docs=2)
        assert len(output) == 2
        assert all(set(doc.keys()) == set(["title", "similarity", "summary"]) for doc in output)
        assert len(set([doc["title"] for doc in output])) == 2
