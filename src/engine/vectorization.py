from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

LOADER_DICT = {
    "directory": DirectoryLoader,
}
SPLITTER_DICT = {
    "recursive": RecursiveCharacterTextSplitter,
}
EMBEDDING_DICT = {
    "openai": OpenAIEmbeddings,
}


class Vectorizer:
    """Langchain based documents vectorizer."""

    def __init__(
        self,
        document_folder: str,
        loader_name: str = "directory",
        splitter_name: str = "recursive",
        embedding_name: str = "openai",
        chunk_size: int = 8000,
        chunk_overlap: int = 200,
    ):
        """
        :param document_folder: str.
            storage location of documents to vectorize.
        :param loader_name: str, defaults to "directory".
            name of the doocument loader object to use. Should be one of LOADER_DICT's keys.
        :param splitter_name: str, defaults to "recursive"
            name of the text splitter object to use. Should be one of SPLITTER_DICT's keys.
        :param embedding_name: str, defaults to "openai"
            name of the embedding object to use. Should be one of EMBEDDING_DICT's keys.
        :param chunk_size: int, defaults to 8000.
            maximum character size of the chunks to split texts into.
        :param chunk_overlap: int, defaults to 200.
            number of characters that can overlap when splitting texts.
        """
        assert (
            loader_name in LOADER_DICT.keys()
        ), f"loader_name not recognised. Use one of {list(LOADER_DICT.keys())} instead."
        assert (
            splitter_name in SPLITTER_DICT.keys()
        ), f"splitter_name not recognised. Use one of {list(SPLITTER_DICT.keys())} instead."
        assert (
            embedding_name in EMBEDDING_DICT.keys()
        ), f"embedding_name not recognised. Use one of {list(EMBEDDING_DICT.keys())} instead."
        self.document_folder = document_folder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.loader = LOADER_DICT[loader_name](self.document_folder)
        self.splitter = SPLITTER_DICT[splitter_name](chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.embedding = EMBEDDING_DICT[embedding_name]()

    def create_vectorstore(self) -> None:
        """Creates a vectorstore from the documents in the given folder."""
        # Loading the documents from the folder
        self.docs = self.loader.load()
        # Splitting texts into chunks of reasonable size
        self.splits = self.splitter.split_documents(self.docs)
        # Creating the vector database from chunks' embeddings
        self.vectorstore = Chroma.from_documents(documents=self.splits, embedding=self.embedding)
