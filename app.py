from rag.engine import RAGEngine

if __name__ == "__main__":
    engine = RAGEngine()
    q = "what are the epicycles of data analytics"
    retrieved = engine.retrieve(q,5)
    print(retrieved)
    res = engine.answer(q)
    print("ANSWER:\n", res["answer"])
    print("\nSOURCES:")
    for s in res["sources"]:
        print(s)
