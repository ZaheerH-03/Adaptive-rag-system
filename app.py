from embeddings.embedder import HuggingFaceEmbedder
import chromadb

from pipeline.engine import RAGEngine
from retrieval.retriever import StandardRetriever
from generation.llm_wrapper import Llama32Local
from config import CHROMA_DIR, EMBED_MODEL_NAME

if __name__ == "__main__":
    print("--- Initializing RAG Components ---")
    
    # 1. Setup Models & DB (Infrastructure)
    embed_model = HuggingFaceEmbedder(model_name=EMBED_MODEL_NAME)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_collection("notes")
    
    # 2. Instantiate Concrete Strategies
    print("Building Retriever...")
    retriever = StandardRetriever(collection, embed_model)
    
    print("Building Generator (LLM)...")
    generator = Llama32Local()
    
    # 3. Inject into Engine
    print("Assembling Engine...")
    engine = RAGEngine(retriever=retriever, generator=generator)
    
    # 4. Run Query
    q = "Tell me about magpie sensing and their solution"
    print(f"\nQUERY: {q}\n")
    
    res = engine.answer(q)
    
    print("ANSWER:\n", res["answer"])
    print("\nSOURCES:")
    for s in res["sources"]:
        print(s)
