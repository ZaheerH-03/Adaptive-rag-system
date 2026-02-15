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

    @abstractmethod
    def get(self, ids: List[str]) -> List[Dict]:
        """Retrieve documents by ID (required for RAPTOR)"""
        pass

class Retriever(ABC):
    """
    Abstract base class for any retrieval strategy.
    It takes a query string and returns a list of relevant content (chunks/nodes).
    """
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        pass

class Generator(ABC):
    """
    Abstract base class for the LLM response generation.
    It takes a prompt (string) and returns the generated text.
    """
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass
