from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=r"C:\Users\hp\OneDrive\Desktop\Python Files_VS Code\Langchain_models\2.ChatModels\.env")

# print("Token loaded:", os.getenv("HUGGINGFACEHUB_API_TOKEN") is not None)

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational",
    max_new_tokens=100,
)

chat_model = ChatHuggingFace(llm=llm)

response = chat_model.invoke("what are the main topics to study in Gen AI?")
print(response.content)