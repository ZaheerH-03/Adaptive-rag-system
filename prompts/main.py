from chain import rag_answer
from few_shot import few_shot_prompt
from zero_shot import zero_shot_prompt
from one_shot import one_shot_prompt
from cot import cot_prompt

context = "Data analysis and data analytics are closely related disciplines focused on extracting meaningful insights from data to support decision-making. Data analysis refers to the systematic process of inspecting, cleaning, transforming, and interpreting data to identify patterns, trends, and relationships. It typically involves statistical techniques, exploratory analysis, and visualization to understand what has happened or why it happened within a dataset. In contrast, data analytics is a broader concept that encompasses data analysis along with the tools, technologies, and processes used to manage and apply data-driven insights at scale. Data analytics includes descriptive, diagnostic, predictive, and prescriptive approaches that help organizations anticipate outcomes and optimize actions. While data analysis is often a component task performed on specific datasets, data analytics represents the end-to-end practice of leveraging data for strategic and operational value. Together, they enable organizations to transform raw data into actionable knowledge, improve performance, and make informed decisions across domains such as business, healthcare, finance, and technology."
question = input("Enter question: ")

response = rag_answer(
    question=question,
    context=context,
    prompt_func=zero_shot_prompt
)

print(response)