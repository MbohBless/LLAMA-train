from abc import ABC, abstractmethod
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from typing import List
from langchain_chroma import Chroma
import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


class VectorizeData(ABC):
    @abstractmethod
    def vectorize_data(self, data: List[Document]):
        pass

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def get_vector_store(self):
        pass


class OpenAIVectorizeData(VectorizeData):
    def __init__(self, name, collection_name: str = 'banking_collection', persisitent_dir: str = "./chroma_langchain_db"):
        self.name = name
        self.collection_name = collection_name
        self.persistent_dir = persisitent_dir
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            base_url=os.getenv("OPENAI_API_URL"),
        )
        self.vector_store = Chroma(collection_name=self.collection_name,
                                   persist_directory=self.persistent_dir,
                                   embedding_function=embeddings)

    def vectorize_data(self, data: List[Document]):
        self.vector_store.add_documents(data)
        return self.vector_store

    def get_name(self):
        return self.name

    def get_vector_store(self):
        return self.vector_store

class HuggingFaceVectorizeData(VectorizeData):
    def __init__(self, name, collection_name: str = 'banking_collection', persisitent_dir: str = "./chroma_langchain_db"):
        self.name = name
        self.collection_name = collection_name
        self.persistent_dir = persisitent_dir
        embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = Chroma(collection_name=self.collection_name,
                                   persist_directory=self.persistent_dir,
                                   embedding_function=embeddings)

    def vectorize_data(self, data: List[Document]):
        self.vector_store.add_documents(data)
        return self.vector_store

    def get_name(self):
        return self.name

    def get_vector_store(self):
        return self.vector_store