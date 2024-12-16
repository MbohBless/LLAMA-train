import re
from typing import List
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDocument
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import os


class LangChainDocumentProcessor:
    def __init__(self, documents: List[LangchainDocument]):
        self.documents = documents

    def chunk_by_characters(self, chunk_size: int) -> List[LangchainDocument]:
        splitter = CharacterTextSplitter(chunk_size=chunk_size)
        chunks = []
        for doc in self.documents:
            chunks.extend(splitter.split_text(doc.page_content))
        return chunks

    def chunk_by_words(self, chunk_size: int) -> List[LangchainDocument]:
        splitter = CharacterTextSplitter(chunk_size=chunk_size, separator=' ')
        chunks = []
        for doc in self.documents:
            chunks.extend(splitter.split_text(doc.page_content))
        return chunks

    def chunk_by_sentences(self, chunk_size: int) -> List[LangchainDocument]:
        splitter = SentenceTextSplitter(chunk_size=chunk_size)
        chunks = []
        for doc in self.documents:
            chunks.extend(splitter.split_text(doc.page_content))
        return chunks

    def recursive_chunk(self, chunk_size: int, min_chunk_size: int) -> List[LangchainDocument]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=min_chunk_size//4
        )
        chunks = []
        for doc in self.documents:
            chunks.extend(splitter.split_text(doc.page_content))
        chonk_documents = [LangchainDocument(
            page_content=chunk) for chunk in chunks]
        return chonk_documents

    def semantic_chunking(self, model: str = "openai") -> List[LangchainDocument]:
        if model == "openai":
            text_splitter = SemanticChunker(
                OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    base_url=os.getenv("OPENAI_API_URL"),
                ),
                breakpoint_threshold_type="percentile",
            )
        else:
            model_name = "BAAI/bge-large-en-v1.5"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': True}
            text_splitter = SemanticChunker(
                HuggingFaceBgeEmbeddings(
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs,
                )
            )
        document_content = [doc.page_content for doc in self.documents]
        chunks = text_splitter.create_documents(document_content)
        return chunks
