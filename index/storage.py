import hashlib
from typing import List, Dict, Any
import chromadb
from core.interfaces import Index
from config import CHROMA_DIR

class ChromaIndex(Index):
    def __init__(self, collection_name: str = "notes"):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(collection_name)

    def add(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        ids: List[str],
    ):
        """
        Adds documents to the index. 
        IDs MUST be provided (we use the stable UUIDs from the Chunk objects).
        """
        if not ids:
             raise ValueError("IDs must be provided for ChromaIndex.add()! use chunk.chunk_id")

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        print(f"Added {len(documents)} documents to Chroma.")

    def query(
        self,
        query_embedding: List[float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        
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
                "score": distances[i]
            }
            retrieved_items.append(item)
            
        return retrieved_items

    def get(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch specific documents entirely by ID.
        Used by RAPTOR to retrieve children chunks for summarization.
        """
        results = self.collection.get(ids=ids, include=["documents", "metadatas", "embeddings"])
        
        # Chroma .get() returns lists directly (not nested if we ask for specific IDs? Wait, let's verify format)
        # Actually .get() structure is: {'ids': [], 'embeddings': [], 'metadatas': [], 'documents': []}
        
        fetched = []
        # Zip safely
        found_ids = results["ids"]
        found_docs = results["documents"]
        found_metas = results["metadatas"]
        found_embs = results["embeddings"]
        
        for i in range(len(found_ids)):
            fetched.append({
                "id": found_ids[i],
                "text": found_docs[i],
                "metadata": found_metas[i],
                "embedding": found_embs[i]
            })
        return fetched
