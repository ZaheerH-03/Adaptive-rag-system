# ingest/semantic_chunking.py
from typing import List, Dict, Tuple
import numpy as np
import re
from sentence_transformers import SentenceTransformer

from .schema import DocUnit


def split_into_sentences(text: str) -> List[str]:
    # Simple sentence splitter; can replace with nltk/spacy later
    raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences


def semantic_chunks_for_unit(
    unit: DocUnit,
    embed_model: SentenceTransformer,
    sim_threshold: float,
    min_chars: int,
    max_chars: int,
) -> List[Tuple[str, Dict]]:
    sentences = split_into_sentences(unit.text)
    if not sentences:
        return []

    sent_embeddings = embed_model.encode(sentences, convert_to_numpy=True)

    chunks: List[Tuple[str, Dict]] = []
    current_sentences: List[str] = [sentences[0]]
    current_embs: List[np.ndarray] = [sent_embeddings[0]]
    current_len = len(sentences[0])

    def flush_chunk():
        nonlocal current_sentences, current_embs, current_len
        if not current_sentences:
            return
        chunk_text = " ".join(current_sentences).strip()
        if not chunk_text:
            current_sentences, current_embs, current_len = [], [], 0
            return
        meta = unit.to_metadata()
        chunks.append((chunk_text, meta))
        current_sentences, current_embs, current_len = [], [], 0

    for i in range(1, len(sentences)):
        s = sentences[i]
        e = sent_embeddings[i]
        prev_e = current_embs[-1]

        # cosine similarity
        sim = float(
            np.dot(e, prev_e) / (np.linalg.norm(e) * np.linalg.norm(prev_e) + 1e-8)
        )

        hard_limit = (current_len + len(s)) > max_chars
        topic_shift = sim < sim_threshold and current_len > min_chars

        if hard_limit or topic_shift:
            flush_chunk()

        current_sentences.append(s)
        current_embs.append(e)
        current_len += len(s)

    flush_chunk()
    return chunks


def semantic_chunk_units(
    units: List[DocUnit],
    embed_model: SentenceTransformer,
    sim_threshold: float,
    min_chars: int,
    max_chars: int,
) -> List[Tuple[str, Dict]]:
    all_chunks: List[Tuple[str, Dict]] = []
    for unit in units:
        unit_chunks = semantic_chunks_for_unit(
            unit,
            embed_model=embed_model,
            sim_threshold=sim_threshold,
            min_chars=min_chars,
            max_chars=max_chars,
        )
        all_chunks.extend(unit_chunks)
    return all_chunks
