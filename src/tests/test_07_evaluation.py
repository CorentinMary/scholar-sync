import tempfile

import pytest

from ..data_preprocessing import download_documents
from ..evaluation import evaluate, generate_dataset, generate_mixture_input, retrieval_accuracy


@pytest.mark.dependency(name="gen_input", scope="session")
def test_generate_mixture_input():
    assert generate_mixture_input([]) == ""

    text_list = ["First text. With two sentences.", "Second text. With two sentences as well."]
    paragraph = generate_mixture_input(text_list, n_sample_sentences=1)
    sentences = paragraph.split(". ")
    assert len(sentences) == 2
    assert all(sentence in text_list[0] or sentence in text_list[1] for sentence in sentences)

    text_list = [
        "First text. With two sentences.",
        "Second text. Not with two sentences this time. Rather three sentences.",
    ]
    paragraph = generate_mixture_input(text_list, n_sample_sentences=3)
    sentences = paragraph.split(". ")
    assert len(sentences) == 5


@pytest.mark.dependency(name="retrieval_dataset", depends=["load_documents", "gen_input"], scope="session")
def test_generate_dataset():
    with tempfile.TemporaryDirectory() as temp_dir:
        download_documents(bucket_name="llm-technical-test-data", destination=temp_dir)
        dataset = generate_dataset(document_folder=temp_dir, size=100, n_sample_docs=3)
        assert len(dataset) == 100
        assert all(set(item.keys()) == set(["input_text", "expected_output"]) for item in dataset)
        assert all(len(item["expected_output"]) == 3 for item in dataset)


@pytest.mark.dependency(name="accuracy", scope="session")
def test_retrieval_accuracy():
    y_pred = [{"title": "Document A"}, {"title": "Document B"}]
    y_true = [{"title": "Document X"}, {"title": "Document Y"}]
    assert retrieval_accuracy(y_pred, y_true) == 0.0

    y_pred = [{"title": "Document A"}, {"title": "Document B"}]
    y_true = [{"title": "Document A"}, {"title": "Document B"}]
    assert retrieval_accuracy(y_pred, y_true) == 1.0

    y_pred = [{"title": "Document A"}, {"title": "Document B"}, {"title": "Document C"}]
    y_true = [{"title": "Document A"}, {"title": "Document C"}, {"title": "Document D"}]
    assert retrieval_accuracy(y_pred, y_true) == 2 / 3


@pytest.mark.dependency(name="evaluate", depends=["accuracy"], scope="session")
def test_evaluate():
    y_pred = [
        {"title": "Document A", "summary": "This is the predicted summary of Document A", "similarity": 0.90},
        {"title": "Document B", "summary": "This is the predicted summary of Document B", "similarity": 0.78},
    ]
    y_true = [
        {"title": "Document A", "summary": "This is the real summary of Document A"},
        {"title": "Document X", "summary": "This is the real summary of Document X"},
    ]
    score = evaluate(y_pred, y_true)
    assert all(key in score.keys() for key in ["similarity", "accuracy", "rouge_score"])
    assert all(isinstance(value, float) for value in score.values())
    assert all(value >= 0 and value <= 1 for value in score.values())
