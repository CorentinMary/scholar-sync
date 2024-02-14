import pytest
from dotenv import load_dotenv

from ..engine.retrieval import RandomRetriever, SimilarityRetriever
from ..engine.vectorization import Vectorizer

load_dotenv()


@pytest.fixture
def temp_folder(tmpdir):
    # Create temporary directory with some sample documents
    documents = ["doc1.txt", "doc2.txt", "doc3.txt"]
    for doc in documents:
        tmpdir.join(doc).write("Sample document content.")
    return str(tmpdir)


@pytest.mark.dependency(name="random_retriever", depends=["fetch_files", "get_title_from_path"], scope="session")
def test_random_retriever(temp_folder):
    input_text = "Sample input text."
    retriever = RandomRetriever(temp_folder)
    doc_title_list = [doc_name.split(".")[0] for doc_name in retriever.document_list]
    sim_docs = retriever.get_similar_docs(input_text, n_sim_docs=3)

    assert len(sim_docs) == 3
    assert set([doc["title"] for doc in sim_docs]).issubset(set(doc_title_list))


@pytest.mark.dependency(
    name="similarity_retriever", depends=["fetch_files", "get_title_from_path", "vectorizer"], scope="session"
)
def test_similarity_retriever_get_similar_docs(temp_folder):
    input_text = "Sample input text."
    vectorizer = Vectorizer(document_folder=temp_folder)
    retriever = SimilarityRetriever(document_folder=temp_folder, vectorizer=vectorizer)
    sim_docs = retriever.get_similar_docs(input_text, n_sim_docs=2)
    assert len(sim_docs) == 2
    assert all(set(doc.keys()) == set(["title", "content", "similarity"]) for doc in sim_docs)
    assert len(set([doc["title"] for doc in sim_docs])) == 2
    assert all(doc["similarity"] >= 0 and doc["similarity"] <= 1 for doc in sim_docs)
