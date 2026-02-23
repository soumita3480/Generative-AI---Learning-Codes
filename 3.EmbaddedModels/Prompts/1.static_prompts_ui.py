from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
import os

# Load token
load_dotenv(dotenv_path=r"C:\Users\hp\OneDrive\Desktop\Python Files_VS Code\Langchain_models\2.ChatModels\.env")
# print("Token loaded:", os.getenv("HUGGINGFACEHUB_API_TOKEN") is not None)

st.header("Chat Bot - Demo")

user_input=st.text_input("write ur prompt")
text="what is the caiptal of india"

llm=HuggingFaceEndpoint(repo_id='mistralai/Mistral-7B-Instruct-v0.2',
    task="conversational",
    max_new_tokens=50)

chat_model=ChatHuggingFace(llm=llm)

if st.button("Submit Your Prompt"):
    result=chat_model.invoke(user_input)
    print(result.content)
    st.write(result.content)

