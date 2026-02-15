# index/build_index.py
import os
from pathlib import Path
from typing import List, Dict

from sentence_transformers import SentenceTransformer

from config import (
    RAW_DIR,
    EMBED_MODEL_NAME,
    SEM_SIM_THRESHOLD,
    SEM_MIN_CHARS,
    SEM_MAX_CHARS,
)
from ingest.loaders import load_units_for_file
from ingest.semantic_chunking import semantic_chunk_units
from index.storage import ChromaIndex
from embeddings.embedder import HuggingFaceEmbedder


def main():
    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    # embed_model = SentenceTransformer(EMBED_MODEL_NAME) -> OLD
    embed_model = HuggingFaceEmbedder(EMBED_MODEL_NAME) # -> NEW

    # Use our new Class!
    index = ChromaIndex(collection_name="notes")

    all_documents: List[str] = []
    all_metadatas: List[Dict] = []
    all_ids: List[str] = []
    
    for root, _, files in os.walk(RAW_DIR):
        for fname in files:
            path = Path(root) / fname
            units = load_units_for_file(path)
            if not units:
                print(f"Skipping unsupported/empty: {path}")
                continue

            print(f"Processing {path} ({len(units)} units)")

            chunks = semantic_chunk_units(
                units,
                embed_model=embed_model,
                sim_threshold=SEM_SIM_THRESHOLD,
                min_chars=SEM_MIN_CHARS,
                max_chars=SEM_MAX_CHARS,
            )

            file_chunk_idx = 0  # index within this file

            for chunk in chunks:
                if not chunk.text.strip():
                    continue

                # augment metadata with per-file chunk index
                meta = dict(chunk.metadata)  # copy to avoid mutating original
                meta["chunk_idx"] = file_chunk_idx
                file_chunk_idx += 1

                all_documents.append(chunk.text)
                all_metadatas.append(meta)
                all_ids.append(chunk.chunk_id) # USE THE EXISTING UUID!

    if not all_documents:
        print("No chunks to index.")
        return

    print(f"Total chunks: {len(all_documents)}")
    print("Embedding chunks...")
    embeddings = embed_model.encode(all_documents, show_progress_bar=True)

    print("Adding to Chroma Index (with stable IDs)...")
    index.add(
        documents=all_documents,
        embeddings=embeddings.tolist(),
        metadatas=all_metadatas,
        ids=all_ids  # Passing the UUIDs from chunks
    )

    print("Index built successfully!")


if __name__ == "__main__":
    main()
