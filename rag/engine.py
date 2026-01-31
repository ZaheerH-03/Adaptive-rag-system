# rag/engine.py
from typing import List, Dict, Any, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, EMBED_MODEL_NAME, TOP_K
from .llm_wrapper import Llama32Local


class RAGEngine:
    def __init__(self):
        # Embedding model (same as for indexing)
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)

        # Chroma client (persistent)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_collection("notes")

        # Local Llama
        self.llm = Llama32Local()

    # ---------- 1. Embed query ----------

    def embed_query(self, q: str) -> List[float]:
        return self.embed_model.encode([q])[0].tolist()

    # ---------- 2. Retrieve with distances ----------

    def retrieve(
        self,
        q: str,
        top_k: int = TOP_K,
    ) -> Tuple[List[str], List[Dict[str, Any]], List[str], List[float]]:
        q_emb = self.embed_query(q)
        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
        )
        docs = results["documents"][0]
        metas_raw = results["metadatas"][0]
        ids = results["ids"][0]
        distances = results["distances"][0]  # smaller = closer in Chroma

        # Normalize metadata to plain dicts
        metas: List[Dict[str, Any]] = [dict(m) for m in metas_raw]

        return docs, metas, ids, distances

    # ---------- 3. Build prompt ----------

    def build_prompt(
        self,
        question: str,
        docs: List[str],
        metas: List[Dict[str, Any]],
    ) -> str:
        ctx_blocks = []
        for i, (d, m) in enumerate(zip(docs, metas), start=1):
            label = f"Source {i}: {m.get('filename', 'unknown')}"
            if m.get("page_num"):
                label += f" (page {m['page_num']})"
            if m.get("slide_num"):
                label += f" (slide {m['slide_num']})"
            if m.get("section_title"):
                label += f" – {m['section_title']}"
            ctx_blocks.append(f"[{label}]\n{d}")

        context = "\n\n".join(ctx_blocks)

        prompt = f"""
You are a professional tutor helping students prepare for exams using their own class notes stored in SOURCES.
Your job is to explain concepts clearly and accurately, only using information present in sources.

First, read all the sources fully.
Extract important definitions, properties, formulas, examples, and implications.

Your answer must be structured clearly and well-explained:

Answer Format:
1. Short definition in 2-3 lines
2. Explanation in simple terms
3. A detailed explanation for showcase all vital topics and subtopics.
3. Why this matters / when it is used
4. Optional: provide short example IF it is explicitly present in the sources

Rules:
- You may rewrite content in simpler language.
- DO NOT add any new facts that are not clearly present in the provided sources.
- DO NOT invent external references, organizations, history, or examples.
- Use [Source 1], [Source 2], etc whenever referring to information.
- If the answer cannot be found in the provided sources, reply:

"I cannot answer this from the notes. Please re-check the material or ask your faculty, and do not repeat the prompt in your replies."

SOURCES:
{context}

QUESTION:
{question}

ANSWER (refer to sources like [Source 1], [Source 2]):
"""
        return prompt.strip()

    # ---------- 4. Main answer method ----------

    def answer(
        self,
        question: str,
        top_k: int = TOP_K,
        max_new_tokens: int = 512,
    ) -> Dict[str, Any]:
        docs, metas, ids, distances = self.retrieve(question, top_k=top_k)

        if not docs:
            return {
                "answer": "I couldn't find anything at all in the indexed notes.",
                "sources": [],
                "debug": {"distances": []},
            }

        # Simple "confidence" check: if best distance is too large, bail out
        best_dist = distances[0]  # chroma uses L2 or IP distance; smaller is better
        # you may need to tune this threshold empirically:
        THRESHOLD = 2.0  # start with 1.0, then adjust

        if best_dist > THRESHOLD:
            return {
                "answer": "I cannot answer this from the notes. It seems unrelated to the indexed material.",
                "sources": [],
                "debug": {"distances": distances},
            }

        prompt = self.build_prompt(question, docs, metas)
        ans_text = self.llm.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
        )

        sources = []
        for i, m in enumerate(metas, start=1):
            sources.append(
                {
                    "label": f"Source {i}",
                    "filename": m.get("filename"),
                    "page_num": m.get("page_num"),
                    "slide_num": m.get("slide_num"),
                    "section_title": m.get("section_title"),
                }
            )

        return {
            "answer": ans_text,
            "sources": sources,
            "debug": {"distances": distances},
        }