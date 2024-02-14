# Classes for the summarization task

from typing import List

from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI

from ..utils import get_abstract
from .prompt import MAP_PROMPT, REDUCE_PROMPT

CHAT_MODEL_DICT = {"openai": ChatOpenAI}
MAX_CONTEXT_SIZE = 4097


class Summarizer:
    """Parent class for summarizers."""

    def __init__(self, max_summary_tokens: int) -> None:
        """
        :param max_summary_tokens: int.
            Maximum number of tokens in the summary.
        """
        self.max_summary_tokens = max_summary_tokens


class DummySummarizer(Summarizer):
    """Baseline summarizer which returns the abstract section of a paper."""

    def __init__(
        self, max_summary_tokens: int = 200, sep: str = "----", abstract_section_name: List[str] = ["ABSTRACT"]
    ):
        """
        :param max_summary_tokens: int, defaults to 200
            Maximum number of tokens in the summary.
        :param sep: int, defaults to "----".
            delimiter for text sections.
        :param abstract_section_name: List[str], defaults to ["ABSTRACT"].
            section title(s) for abstract sections.
        """
        super(DummySummarizer, self).__init__(max_summary_tokens=max_summary_tokens)
        self.sep = sep
        self.abstract_section_name = abstract_section_name

    def summarize(self, text_list: List[str]) -> List[str]:
        """Provides a list of summaries from a list of texts by extracting the abstract section.

        :param text_list: List[str].
            list of texts to summarize.
        :return: list of summaries.
        """
        self.text_list = text_list
        self.summary_list = [
            " ".join(
                get_abstract(text, sep=self.sep, abstract_section_name=self.abstract_section_name).split(" ")[
                    : self.max_summary_tokens
                ]
            )
            for text in self.text_list
        ]

        return self.summary_list


class PromptSummarizer(Summarizer):
    """LLM based summarizer using langchain's MapReduce approach."""

    def __init__(
        self,
        model_name: str = "openai",
        model_kwargs: dict = {},
        max_summary_tokens: int = 200,
        map_prompt: str = MAP_PROMPT,
        reduce_prompt: str = REDUCE_PROMPT,
    ) -> None:
        """
        :param model_name: str, defaults to "openai
            name of the chat model to use for prompting.
        :param model_kwargs: dict, defaults to {}
            additional arguments for the chat model.
        :param max_summary_tokens: int, defaults to 200
            maximum number of tokens in the summary.
        :param map_prompt: str, defaults to MAP_PROMPT
            prompt to use for the map step.
        :param reduce_prompt: str, defaults to REDUCE_PROMPT
            prompt to use for the reduce step.
        """
        super(PromptSummarizer, self).__init__(max_summary_tokens=max_summary_tokens)

        assert (
            model_name in CHAT_MODEL_DICT.keys()
        ), f"model_name not recognised. Use one of {list(CHAT_MODEL_DICT.keys())} instead."

        self.max_summary_tokens = max_summary_tokens
        self.llm = CHAT_MODEL_DICT[model_name](max_tokens=self.max_summary_tokens, **model_kwargs)

        # build the MapReduce chain
        self.map_prompt = PromptTemplate.from_template(map_prompt)
        map_chain = LLMChain(llm=self.llm, prompt=self.map_prompt)
        self.reduce_prompt = PromptTemplate.from_template(reduce_prompt)
        reduce_chain = LLMChain(llm=self.llm, prompt=self.reduce_prompt)
        combine_documents_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_variable_name="docs")
        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,
            token_max=MAX_CONTEXT_SIZE,
        )
        self.map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="docs",
            return_intermediate_steps=False,
        )

        # build the text splitter
        # the chunk size to split documents for MapReduce should be small enough so that the number of tokens in the
        # prompt, the chunk and the summary does not exceed the context size
        max_prompt_tokens = max(self.llm.get_num_tokens(map_prompt), self.llm.get_num_tokens(reduce_prompt))
        self.chunk_size = MAX_CONTEXT_SIZE - max_prompt_tokens - self.max_summary_tokens
        self.text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=self.chunk_size, chunk_overlap=0)

    def summarize(self, text_list: List[str]) -> List[str]:
        """Provides a list of summaries from a list of texts by prompting a LLM.
        For each text we use langchain's MapReduce chain to avoid exceeding the LLM's context size.

        :param text_list: List[str]
            list of texts to summarize.
        :return: list of summaries.
        """
        self.summary_list = []
        for text in text_list:
            # split the text into several chunks
            docs = [Document(page_content=chunk) for chunk in self.text_splitter.split_text(text)]
            # apply the MapReduce chain to create a final summary of all chunks
            summary = self.map_reduce_chain.run(docs)
            self.summary_list.append(summary)

        return self.summary_list
