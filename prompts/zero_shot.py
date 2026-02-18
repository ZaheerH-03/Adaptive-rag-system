from prompts.base import build_base_prompt

def zero_shot_prompt(context: str, question: str) -> str:
    strategy = """
Provide the answer in two parts:

Overview:
A clear and concise definition of the concept (about 80–120 words).

Details:
A deeper explanation expanding on the overview using relevant information.
Provide examples if applicable to illustrate the concept.

Conclusion:
A concluding sentence or call to action
"""

    return build_base_prompt(context, question, strategy)