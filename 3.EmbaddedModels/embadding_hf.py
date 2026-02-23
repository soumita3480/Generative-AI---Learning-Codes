from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os

# Load token
load_dotenv(dotenv_path=r"C:\Users\hp\OneDrive\Desktop\Python Files_VS Code\Langchain_models\2.ChatModels\.env")

print("Token loaded:", os.getenv("HUGGINGFACEHUB_API_TOKEN") is not None)

# Create API-based embeddings
embedding = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

text = "what is the capital of Delhi"

result = embedding.embed_query(text)

print("Embedding vector length:", len(result),'\n')
print(result)