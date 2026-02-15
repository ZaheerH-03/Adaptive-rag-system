from typing import List, Dict, Any
from core.interfaces import Retriever, Embedder
import chromadb

class StandardRetriever(Retriever):
    def __init__(self, collection: chromadb.Collection, embed_model: Embedder):
        self.collection = collection
        self.embed_model = embed_model

    def _embed_query(self, q: str) -> List[float]:
        # Interface returns List[List[float]]
        return self.embed_model.embed([q])[0]

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q_emb = self._embed_query(query)
        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
        )
        
        # Check if we got results
        if not results["documents"] or not results["documents"][0]:
            return []

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        ids = results["ids"][0]
        distances = results["distances"][0]

        retrieved_items = []
        for i in range(len(docs)):
            item = {
                "text": docs[i],
                "metadata": metas[i],
                "id": ids[i],
                "score": distances[i] # In Chroma, lower distance = better match usually
            }
            retrieved_items.append(item)
            
        return retrieved_items
