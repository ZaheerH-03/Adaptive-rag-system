from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

class Chunker(ABC):
    @abstractmethod
    def chunk(self, text):
        pass

class Loader(ABC):
    @abstractmethod
    def load(self, path):
        pass

class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: List[str]):
        pass

class Index(ABC):
    @abstractmethod
    def add(
        self,
        documents: List[str],
        embeddings,
        metadatas: List[Dict],
        ids: List[str],
    ):
        pass

    @abstractmethod
    def query(
        self,
        query_embedding,
        top_k: int,
    ):
        pass