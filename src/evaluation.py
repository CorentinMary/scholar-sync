# Functions to evaluate the retrieval and summarization tasks
from typing import List

import numpy as np
from nltk.tokenize import sent_tokenize
from numpy.random import choice, shuffle
from rouge_score import rouge_scorer

from .utils import fetch_files, get_abstract, get_title_from_path

AVAILABLE_METRICS = ["similarity", "accuracy", "rouge_score"]


def retrieval_accuracy(y_pred: List[dict], y_true: List[dict]) -> float:
    """Computes the retrieval accuracy of a single prediction defined as the ratio of the number of correct documents
    retrieved over the number of documents retrieved.

    :param y_pred: List[dict]
        documents retrieved as dictionaries with a 'title' key.
    :param y_true: List[dict]
        expected documents as dictionaries with a 'title' key.
    :return: accuracy score
    """
    assert len(y_pred), "Prediction should not be empty."
    assert len(y_true), "Ground truth should not be empty."
    assert len(y_pred) == len(y_true), "Prediction and expected output shoud have the same length."
    assert all("title" in doc.keys() for doc in y_pred), "Each element of prediction should have a 'title' key."
    assert all("title" in doc.keys() for doc in y_true), "Each element of ground truth should have a 'title' key."

    n_docs = len(y_pred)
    pred_title_list = [doc["title"] for doc in y_pred]
    true_title_list = [doc["title"] for doc in y_true]
    # the number of documents correctly predicted is the size of the intersection between pred_title_list and
    # true_title_list
    accuracy = len(set(pred_title_list).intersection(set(true_title_list))) / n_docs

    return accuracy


def evaluate(
    y_pred: List[dict], y_true: List[dict], metrics: List[str] = ["similarity", "accuracy", "rouge_score"]
) -> dict:
    """Evaluates a prediction with a given set of metrics.

    :param y_pred: List[dict]
        documents retrieved as dictionaries with a 'title' key.
    :param y_true: List[dict]
        expected documents as dictionaries with a 'title' key.
    :param metrics: List[str], defaults to ["similarity", "accuracy", "rouge_score"]
        metrics to compute for evaluation. Should be included in AVAILABLE_METRICS.
    """
    assert all(metric in AVAILABLE_METRICS for metric in metrics), f"metrics should be included in {AVAILABLE_METRICS}"

    score = {}

    # the similarity metric is returned by the retriever and corresponds to the cosine similarity between the
    # embeddings of the input text and the embeddings of the document chunk retrieved
    if "similarity" in metrics:
        assert all("similarity" in doc.keys() for doc in y_pred), "similarity key missing from prediction."
        score["similarity"] = np.mean([doc["similarity"] for doc in y_pred])

    # the accuracy metric is the ratio of the number of correct documents retrieved over the number of documents
    # retrieved.
    if "accuracy" in metrics:
        score["accuracy"] = retrieval_accuracy(y_pred, y_true)

    # the rouge_score metric is the ROUGE score (ROUGE-1) of the predicted summary w.r.t. the true
    # summary
    if "rouge_score" in metrics:
        assert all("summary" in doc.keys() for doc in y_pred), "summary key missing from prediction."
        assert all("summary" in doc.keys() for doc in y_true), "summary key missing from ground truth."
        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
        score["rouge_score"] = np.mean(
            [
                scorer.score(pred_doc["summary"], true_doc["summary"])["rouge1"].fmeasure
                for (pred_doc, true_doc) in zip(y_pred, y_true)
            ]
        )

    return score


def generate_mixture_input(text_list: List[str], n_sample_sentences: int = 3) -> str:
    """Generates a paragraph by randomly sampling sentences from a list of texts.

    :param text_list: List[str]
        list of texts to sample from.
    :param n_sample_sentences: int, defaults to 3
        number of sentences to sample from each text.
    :return: paragraph created.
    """
    # Splitting the texts into sentences
    sentence_list = [sent_tokenize(text) for text in text_list]
    # Sampling at most n_sentence from each text
    paragraph = []
    for text_sentence in sentence_list:
        paragraph += list(choice(text_sentence, size=min(n_sample_sentences, len(text_sentence)), replace=False))
    # Shuffling the sentences sampled for additional randomness
    shuffle(paragraph)

    return " ".join(paragraph)


def generate_dataset(
    document_folder: str = "./data", size: int = 100, n_sample_docs: int = 3, n_sample_sentences: int = 2
) -> List[dict]:
    """Creates an evaluation dataset by randomly sampling documents.
    The input paragraph is created by taking a random sample of sentences from the sampled documents abstracts.
    The expected output is a dictionary with the title of the sampled documents and their abstracts.

    :param document_folder: str, defaults to "./data"
        storage folder of the documents to sample.
    :param size: int, defaults to 100
        Dataset size.
    :param n_sample_doc: int, defaults to 3
        number of documents to sample for each element of the dataset.
    :param n_sample_sentences: int, defaults to 2
        number of sentences to sample from each text.
    :return: list of dictionaries containing the input paragraph generated and the expected output.
    """
    document_list = fetch_files(document_folder)
    assert (
        len(document_list) >= n_sample_docs
    ), f"Cannot sample {n_sample_docs} document(s) from folder containing \
        {len(document_list)} document(s)."

    text_list = [open(f"{document_folder}/{doc}", "r").read() for doc in document_list]
    abstract_list = [get_abstract(text) for text in text_list]
    dataset = []
    for _ in range(size):
        # draw n_sample_docs random indices
        sample_idx = choice(range(len(document_list)), size=n_sample_docs, replace=False)
        # generate a random input paragraph by sampling sentences from the abstracts
        input_text = generate_mixture_input(
            text_list=[abstract_list[idx] for idx in sample_idx], n_sample_sentences=n_sample_sentences
        )
        dataset.append(
            {
                "input_text": input_text,
                "expected_output": [
                    {"title": get_title_from_path(document_list[idx]), "summary": abstract_list[idx]}
                    for idx in sample_idx
                ],
            }
        )

    return dataset
