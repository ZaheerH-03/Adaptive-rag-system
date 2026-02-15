from typing import List, Dict, Any, Tuple

from core.interfaces import Retriever, Generator
from config import TOP_K

class RAGEngine:
    def __init__(self, retriever: Retriever, generator: Generator):
        """
        Initializes the RAGEngine with a retriever and a generator.
        This is Dependency Injection: we pass the components in,
        rather than creating them inside.
        """
        self.retriever = retriever
        self.generator = generator

    def retrieve(self, q: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """
        Delegates retrieval to the injected retriever instance.
        """
        return self.retriever.retrieve(q, top_k)

    def build_prompt(
        self,
        question: str,
        retrieved_items: List[Dict[str, Any]],
    ) -> str:
        ctx_blocks = []
        # retrieved_items is a list of dicts: {"text": ..., "metadata": ...}
        for i, item in enumerate(retrieved_items, start=1):
            text = item["text"]
            m = item["metadata"]
            
            label = f"Source {i}: {m.get('filename', 'unknown')}"
            if m.get("page_num"):
                label += f" (page {m['page_num']})"
            if m.get("slide_num"):
                label += f" (slide {m['slide_num']})"
            if m.get("section_title"):
                label += f" – {m['section_title']}"
            ctx_blocks.append(f"[{label}]\n{text}")

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

    def clean_answer(self, text: str) -> str:
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
        results = self.retrieve(question, top_k=top_k)

        if not results:
            return {
                "answer": "I couldn't find anything at all in the indexed notes.",
                "sources": [],
                "debug": {"distances": []},
            }

        # Simple "confidence" check: if best score (distance) is too large, bail out
        # Note: StandardRetriever returns 'score' which is distance in Chroma (lower is better)
        best_dist = results[0]["score"]
        THRESHOLD = 2.0 

        if best_dist > THRESHOLD:
            return {
                "answer": "I cannot answer this from the notes. It seems unrelated to the indexed material.",
                "sources": [],
                "debug": {"distances": [r["score"] for r in results]},
            }

        prompt = self.build_prompt(question, results)
        
        # Generator interface call
        ans_text = self.generator.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
        )
        
        sources = []
        for i, item in enumerate(results, start=1):
            m = item["metadata"]
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
            "debug": {"distances": [r["score"] for r in results]},
        }
