from prompts.base import build_base_prompt

def cot_prompt(context: str, question: str) -> str:
    strategy = """
Use step-by-step reasoning based only on the provided information to derive the answer.

Provide the response in this structure:

Step-by-Step Reasoning:
Explain the logical or mathematical steps needed to arrive at the answer using only the given information.

Final Answer:
State the final result clearly and concisely.

Example:

Context:
Amdahl's Law determines the theoretical speedup of a task at fixed workload that can be expected of a system whose resources are improved. The formula is S(N) = 1 / ((P / N) + S), where P is the parallel fraction, S is the serial fraction (S = 1 − P), and N is the number of processors.

Question:
A system has a 90% parallel fraction (P = 0.9). Calculate the speedup if we use 10 processors (N = 10).

Answer:
Step-by-Step Reasoning:
Identify variables: The parallel fraction P is 0.9. The serial fraction S is 1 − 0.9 = 0.1. The number of processors N is 10.
Set up the formula: S(N) = 1 / ((P / N) + S).
Substitute values: S(10) = 1 / ((0.9 / 10) + 0.1).
Compute division: 0.9 / 10 = 0.09.
Sum denominator: 0.09 + 0.1 = 0.19.
Final division: 1 / 0.19 ≈ 5.26.

Final Answer:
With 10 processors, the theoretical speedup is approximately 5.26×.
"""
    return build_base_prompt(context, question, strategy)
