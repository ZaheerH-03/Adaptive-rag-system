from langchain_core.prompts import PromptTemplate

def zero_shot_prompt():
    template = """
Provide the answer in two parts:

Overview:
A clear and concise definition of the concept (about 80–120 words).

Details:
A deeper explanation expanding on the overview using relevant information.
Provide examples if applicable to illustrate the concept.

Conclusion: 
A concluding sentence or call to action

Follow this structure consistently.

Context:
{context}

Question:
{question}

Answer:
"""
    return PromptTemplate.from_template(template)
