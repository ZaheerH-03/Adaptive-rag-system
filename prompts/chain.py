from hf_llm import llm
from langchain_core.output_parsers import StrOutputParser

def rag_answer(question: str, context: str, prompt_func):
    """
    Runs RAG pipeline with selected prompting strategy.

    question: user question
    context: retrieved context
    prompt_func: function returning PromptTemplate
    """

    prompt = prompt_func()

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response
