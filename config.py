# config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Data paths
RAW_DIR = BASE_DIR / "data" / "raw"
CHROMA_DIR = BASE_DIR / "data" / "chroma"

# Embedding model (for semantic chunking + retrieval)
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Semantic chunking hyperparams
SEM_SIM_THRESHOLD = 0.7   # lower => bigger chunks, higher => more splits
SEM_MIN_CHARS = 200
SEM_MAX_CHARS = 900

# Retrieval
TOP_K = 5

# LLM model (HuggingFace)
LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
