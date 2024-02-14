# Classes for the documents retrieval task

import os
from typing import List

import numpy as np

from ..utils import fetch_files, get_title_from_path
from .vectorization import Vectorizer


class Retriever:
    """Parent class for retrievers."""

    def __init__(self, document_folder: str):
        """
        :param document_folder: str.
            storage location of documents to retrieve from.
        """
        assert os.path.exists(document_folder), f"Location {document_folder} does not exist."
        self.document_folder = document_folder
        self.document_list = fetch_files(self.document_folder)

    @property
    def n_docs(self) -> int:
        return len(self.document_list)


class RandomRetriever(Retriever):
    """Test retriever which returns random documents."""

    def __init__(self, document_folder: str):
        """
        :param document_folder: str.
            storage location of documents to retrieve.
        """
        super(RandomRetriever, self).__init__(document_folder=document_folder)

    def get_similar_docs(self, input_text: str, n_sim_docs: int = 3) -> List[dict]:
        """Returns a random list of documents names from the folder provided.

        :param input_text: str.
            (!) Not used, here for method consistency.
        :param n_sim_docs: int, defaults to 3.
            number of documents to retrieve.
        :return: list of retrieved documents.
        """
        _ = input_text
        random_idx = np.random.choice([i for i in range(self.n_docs)], n_sim_docs, replace=False)
        sim_file_list = [self.document_list[idx] for idx in random_idx]
        sim_docs = [
            {"title": get_title_from_path(file), "content": open(f"{self.document_folder}/{file}", "r").read()}
            for file in sim_file_list
        ]

        return sim_docs


class SimilarityRetriever(Retriever):
    """Retriever based on embeddings similarity."""

    def __init__(self, document_folder: str, vectorizer: Vectorizer):
        """
        :param document_folder: str
            storage location of documents to retrieve.
        :param vectorizer: Vectorizer
            Vectorizer object for similarity search.
        """
        super(SimilarityRetriever, self).__init__(document_folder=document_folder)
        assert (
            vectorizer.document_folder == self.document_folder
        ), "document_folder from vectorizer and retriever should be identical."
        self.vectorizer = vectorizer
        # Create the vectorstore if it does not already exist
        if not hasattr(self.vectorizer, "vectorstore"):
            self.vectorizer.create_vectorstore()

    def get_similar_docs(self, input_text: str, n_sim_docs: int = 3) -> List[dict]:
        """Returns a list of documents most similar to the input text.

        :param input_text: str.
            text to find similar documents to.
        :param n_sim_docs: int, defaults to 3.
            number of documents to retrieve.
        :return: list of retrieved documents.
        """
        # To retrieve a list of n_sim_docs different documents, we retrieve one document at a time and iterate by
        # excluding at each step the documents already retrieved.
        # NB: we do the retrieval iteratively because doing it only once can lead to several chunks coming from the
        # same document.
        sim_docs = []
        for _ in range(n_sim_docs):
            search_kwargs = {}
            if sim_docs:  # adding a filter to exclude already retrieved document(s)
                search_kwargs["filter"] = {"source": {"$nin": [doc["source"] for doc in sim_docs]}}
            # Get the chunk most similar to input_text
            sim_chunk, sim_score = self.vectorizer.vectorstore.similarity_search_with_relevance_scores(
                query=input_text, k=1, **search_kwargs
            )[0]
            # Get the source document from the chunk
            sim_docs.append({"source": sim_chunk.metadata["source"], "similarity": sim_score})
        # Return a dictionary containing the document title, its content and the similarity score
        output = [
            {
                "title": get_title_from_path(doc["source"]),
                "content": open(f"{doc['source']}", "r").read(),
                "similarity": doc["similarity"],
            }
            for doc in sim_docs
        ]

        return output
