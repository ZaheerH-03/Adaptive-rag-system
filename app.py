from rag.engine import RAGEngine

if __name__ == "__main__":
    engine = RAGEngine()
    q = "Tell me about Magpie sensing and their solution"
    res = engine.answer(q)
    print("ANSWER:\n", res["answer"])
    print("\nSOURCES:")
    for s in res["sources"]:
        print(s)
