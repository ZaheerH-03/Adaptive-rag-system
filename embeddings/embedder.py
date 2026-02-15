from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from core.interfaces import Embedder
from config import EMBED_MODEL_NAME

class HuggingFaceEmbedder(Embedder):
    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Returns a list of vectors (lists of floats).
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_numpy(self, texts: List[str]) -> np.ndarray:
        """
        Returns numpy array (useful for math/chunking).
        """
        return self.model.encode(texts, convert_to_numpy=True)
