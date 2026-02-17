from flask import Flask, request, jsonify
from flask_cors import CORS
import chromadb

from embeddings.embedder import HuggingFaceEmbedder
from retrieval.retriever import StandardRetriever
from generation.llm_wrapper import Llama32Local
from pipeline.engine import RAGEngine
from config import CHROMA_DIR, EMBED_MODEL_NAME

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (needed if UI runs on different port)

# ── Global RAG Engine (singleton) ──────────────────────────────────
rag_engine = None

def get_engine():
    global rag_engine
    if rag_engine is not None:
        return rag_engine

    print("═══ Initializing RAG Components ═══")

    # 1. Embedding model
    print(f"  → Embedding Model: {EMBED_MODEL_NAME}")
    embed_model = HuggingFaceEmbedder(model_name=EMBED_MODEL_NAME)

    # 2. ChromaDB
    print(f"  → ChromaDB: {CHROMA_DIR}")
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_collection("notes")

    # 3. Retriever
    print("  → Building Retriever...")
    retriever = StandardRetriever(collection, embed_model)

    # 4. Generator (LLM)
    print("  → Loading LLM...")
    generator = Llama32Local()

    # 5. Assemble engine
    print("  → Assembling Engine...")
    rag_engine = RAGEngine(retriever=retriever, generator=generator)
    print("═══ RAG Engine Ready ═══\n")
    return rag_engine


# ── Routes ─────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running"})


@app.route("/query", methods=["POST"])
def query():
    data = request.json
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Missing 'question' field"}), 400

    engine = get_engine()

    try:
        result = engine.answer(question)
        return jsonify(result)
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500


# ── Main ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    get_engine()                       # warm-up on startup
    print("Starting RAG API  →  http://localhost:5000")
    app.run(host="0.0.0.0", port=5000)
