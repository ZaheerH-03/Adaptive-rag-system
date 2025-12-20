# ingest/build_index.py
import os
from pathlib import Path
from typing import List, Dict

from sentence_transformers import SentenceTransformer
import chromadb

from config import (
    RAW_DIR,
    CHROMA_DIR,
    EMBED_MODEL_NAME,
    SEM_SIM_THRESHOLD,
    SEM_MIN_CHARS,
    SEM_MAX_CHARS,
)
from .loaders import load_units_for_file
from .semantic_chunking import semantic_chunk_units


def main():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_or_create_collection("notes")

    all_documents: List[str] = []
    all_metadatas: List[Dict] = []
    all_ids: List[str] = []

    global_chunk_counter = 0  # id uniqueness across all files

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

            for chunk_text, meta in chunks:
                if not chunk_text.strip():
                    continue

                # augment metadata with per-file chunk index
                meta = dict(meta)  # copy to avoid mutating original
                meta["chunk_idx"] = file_chunk_idx
                file_chunk_idx += 1

                global_chunk_counter += 1
                all_documents.append(chunk_text)
                all_metadatas.append(meta)
                all_ids.append(f"{fname}_chunk_{global_chunk_counter}")

    if not all_documents:
        print("No chunks to index.")
        return

    print(f"Total chunks: {len(all_documents)}")
    print("Embedding chunks...")
    embeddings = embed_model.encode(all_documents, show_progress_bar=True)

    print("Adding to Chroma collection...")
    collection.add(
        documents=all_documents,
        embeddings=embeddings.tolist(),
        metadatas=all_metadatas,
        ids=all_ids,
    )

    print("Index built and stored at:", CHROMA_DIR)


if __name__ == "__main__":
    main()
