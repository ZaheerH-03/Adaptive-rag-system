import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=HF_TOKEN,
    task="conversational",
    temperature=0.2,
    max_new_tokens=400,
)

llm = ChatHuggingFace(llm=endpoint)
