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
        SYSTEM ROLE:
        You are an exam tutor. Answer questions ONLY using the provided sources.

        STRICT RULES:
        - Do NOT repeat the question, instructions, or sources.
        - Do NOT explain your reasoning process.
        - Do NOT mention the word "source" except in citations like [Source 1].
        - You MAY rephrase and summarize the information in your own words.
        - Do NOT copy sentences verbatim unless necessary.
        - When the question asks for "types", explicitly list and briefly explain each type using the sources.

        - If the answer is not present in the sources, reply EXACTLY with:
        "I cannot answer this from the notes."

        SOURCES (read-only):
        <<<
        {context}
        >>>

        QUESTION:
        {question}

        FINAL ANSWER:
        """

        return prompt.strip()

    # ---------- 4. Main answer method ----------
    def clean_answer(text: str) -> str:
        banned_starts = [
            "You are",
            "SYSTEM ROLE",
            "STRICT RULES",
            "SOURCES",
            "QUESTION"
        ]
        for b in banned_starts:
            if text.strip().startswith(b):
                text = text.split("\n", 1)[-1]
        return text.strip()

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
            temperature=0.6,
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