from flask import Flask, request, jsonify
import chromadb
from sentence_transformers import SentenceTransformer

from embeddings.embedder import HuggingFaceEmbedder
from retrieval.retriever import StandardRetriever
from generation.llm_wrapper import Llama32Local
from pipeline.engine import RAGEngine
from config import CHROMA_DIR, EMBED_MODEL_NAME

app = Flask(__name__)

# Global RAG Engine
rag_engine = None

def get_engine():
    global rag_engine
    if rag_engine is None:
        print("--- Initializing RAG Components ---")
        
        # 1. Setup Models & DB (Infrastructure)
        print(f"Loading Embedding Model: {EMBED_MODEL_NAME}")
        embed_model = HuggingFaceEmbedder(model_name=EMBED_MODEL_NAME)
        
        print(f"Connecting to ChromaDB at: {CHROMA_DIR}")
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = chroma_client.get_collection("notes")
        
        # 2. Instantiate Concrete Strategies
        print("Building Retriever...")
        retriever = StandardRetriever(collection, embed_model)
        
        print("Building Generator (LLM)...")
        # Initialize Llama32Local here. This loads the model into memory.
        # This is where we could swap for an LM Studio Client if needed.
        generator = Llama32Local()
        
        # 3. Inject into Engine
        print("Assembling Engine...")
        rag_engine = RAGEngine(retriever=retriever, generator=generator)
    
    return rag_engine

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "running"})

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    question = data.get('question')
    
    if not question:
        return jsonify({"error": "Missing 'question' field"}), 400
        
    engine = get_engine()
    
    try:
        response = engine.answer(question)
        return jsonify(response)
    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Ensure engine is initialized before starting server to catch errors early
    get_engine()
    print("--- Starting RAG API Server ---")
    app.run(host='0.0.0.0', port=5000)
