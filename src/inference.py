# inference pipeline for retrieval and summarization
from typing import List

from .engine.retrieval import SimilarityRetriever
from .engine.summarization import DummySummarizer, PromptSummarizer
from .engine.vectorization import Vectorizer

RETRIEVER_DICT = {"similarity": SimilarityRetriever}
SUMMARIZER_DICT = {"dummy": DummySummarizer, "prompt": PromptSummarizer}


class InferencePipeline:
    """Pipeline to run vectorization, retrieval and summarization steps."""

    def __init__(
        self,
        document_folder: str,
        retriever_name: str,
        summarizer_name: str,
        vectorizer_kwargs: dict = {},
        summarizer_kwargs: dict = {},
    ) -> None:
        """
        :param document_folder: str
            storage location of documents to use.
        :param retriever_name: str
            name of the retrieval engine to use. Should be one of RETRIEVER_DICT's keys.
        :param summarizer_name: str
            name of the summarizer engine to use. Should be one of SUMMARIZER_DICT's keys.
        """
        assert (
            retriever_name in RETRIEVER_DICT.keys()
        ), f"retriever_name not recognised. Use one of {list(RETRIEVER_DICT.keys())} instead."
        assert (
            summarizer_name in SUMMARIZER_DICT.keys()
        ), f"summarizer_name not recognised. Use one of {list(SUMMARIZER_DICT.keys())} instead."
        self.document_folder = document_folder
        self.vectorizer = Vectorizer(document_folder=self.document_folder, **vectorizer_kwargs)
        self.retriever = RETRIEVER_DICT[retriever_name](
            document_folder=self.document_folder, vectorizer=self.vectorizer
        )
        self.summarizer = SUMMARIZER_DICT[summarizer_name](**summarizer_kwargs)

    def predict(self, input_text: str, n_sim_docs: int = 3) -> List[dict]:
        """_summary_
        :param input_text: str.
            text to find similar documents to.
        :param n_sim_papers: int, defaults to 3.
            number of similar documents to retrieve.
        :return: list of dictionaries containing the retrieved documents titles and summaries.
        """
        sim_docs = self.retriever.get_similar_docs(input_text=input_text, n_sim_docs=n_sim_docs)
        summaries = self.summarizer.summarize([doc["content"] for doc in sim_docs])
        output = [
            {"title": doc["title"], "similarity": doc["similarity"], "summary": summary}
            for doc, summary in zip(sim_docs, summaries)
        ]

        return output
