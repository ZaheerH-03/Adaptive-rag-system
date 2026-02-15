from embeddings.embedder import HuggingFaceEmbedder
import chromadb
from retrieval.retriever import StandardRetriever
from config import CHROMA_DIR, EMBED_MODEL_NAME

def main():
    print("--- Testing Retrieval Module ---")
    
    # 1. Setup Models & DB
    print(f"Loading Embedding Model: {EMBED_MODEL_NAME}")
    embed_model = HuggingFaceEmbedder(model_name=EMBED_MODEL_NAME)
    
    print(f"Connecting to ChromaDB at: {CHROMA_DIR}")
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_collection("notes")
    
    # 2. Instantiate Retriever
    retriever = StandardRetriever(collection, embed_model)
    
    # 3. Define Query
    q = "Explain the different types of Organizational Behaviour models with Organizational relevance ?" 
    # Feel free to change this!
    
    print(f"\nQUERY: '{q}'\n")
    print("-" * 50)
    
    # 4. Retrieve
    results = retriever.retrieve(q, top_k=5)
    
    # 5. Inspect Results
    if not results:
        print("No documents found!")
        return

    for i, item in enumerate(results, start=1):
        print(f"Result #{i}")
        print(f"Score (Distance): {item['score']:.4f} (Lower is better)")
        print(f"ID: {item['id']}")
        
        meta = item['metadata']
        print(f"Source: {meta.get('filename')} | Page: {meta.get('page_num')} | Slide: {meta.get('slide_num')}")
        
        # Print a snippet of the text
        text_preview = item['text'][:200].replace("\n", " ") + "..."
        print(f"Content: {text_preview}")
        print("-" * 50)

if __name__ == "__main__":
    main()
