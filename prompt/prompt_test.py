
import os
from pathlib import Path
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

PROMPT_DIR = Path(r'C:\documents_dump\4-1_p\adaptive rag\Adaptive-rag-system\prompt')
print("Prompt directory:", PROMPT_DIR)

client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    token=HF_TOKEN
)


def load_prompt(path):
    return (PROMPT_DIR / path).read_text(encoding="utf-8")


def build_prompt(strategy, context, question):
    base = load_prompt("base.txt")
    strat = load_prompt(f"strategies/{strategy}.txt")

    return f"{strat}\n\n{base}".format(
        context=context,
        question=question
    )
    

def llm(prompt: str,
        max_tokens: int = 400,
        temperature: float = 0.2) -> str:
    
    messages = []
    
    
    messages.append({
        "role": "user",
        "content": prompt
    })
    
    response = client.chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    return response.choices[0].message["content"].strip()


STRATEGY = "zero_shot"   # zero_shot | one_shot | few_shot | cot

context = "Data analysis and data analytics are closely related disciplines focused on extracting meaningful insights from data to support decision-making. Data analysis refers to the systematic process of inspecting, cleaning, transforming, and interpreting data to identify patterns, trends, and relationships. It typically involves statistical techniques, exploratory analysis, and visualization to understand what has happened or why it happened within a dataset. In contrast, data analytics is a broader concept that encompasses data analysis along with the tools, technologies, and processes used to manage and apply data-driven insights at scale. Data analytics includes descriptive, diagnostic, predictive, and prescriptive approaches that help organizations anticipate outcomes and optimize actions. While data analysis is often a component task performed on specific datasets, data analytics represents the end-to-end practice of leveraging data for strategic and operational value. Together, they enable organizations to transform raw data into actionable knowledge, improve performance, and make informed decisions across domains such as business, healthcare, finance, and technology."


#context = "Reliability is the probability that a system will perform its intended function under stated conditions. For a system where all nodes must work for the system to be operational, the total reliability is the product of each node's probability of working (P_working). If a node has a failure probability of F, then P_working = 1 − F."


#context="Scalability measures a system's ability to handle an increasing amount of work by adding resources (processors). Actual Speedup is calculated as S = T_old / T_new, where T is execution time. Ideal Speedup is the ratio of processors used in the new state compared to the old state (e.g., if processors increase from 4 to 8, Ideal Speedup is 8/4 = 2). A system is considered scalable if the actual speedup is close to the ideal speedup.In practice, a system is considered scalable if the actual speedup reaches at least 75% of the ideal speedup."

user_q=input("Enter your question: ")


prompt = build_prompt(
    STRATEGY,
    context=context,
    question=user_q
)

response = llm(prompt)

print(response)