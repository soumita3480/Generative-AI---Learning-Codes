from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate,load_prompt # load_prompt required for dynamic prompt
import streamlit as st
import os

load_dotenv(dotenv_path=r"C:\Users\hp\OneDrive\Desktop\Python Files_VS Code\Langchain_models\2.ChatModels\.env")

llm=HuggingFaceEndpoint(repo_id='mistralai/Mistral-7B-Instruct-v0.2',
        task="conversational"
        # max_completion_token=100)
)

model=ChatHuggingFace(llm=llm)

st.header("Dynamic Prompts- No Json")

capital_input=st.selectbox("what is the captal of india",['','Delhi','Kolkata','Mumbai'])
continent_input=st.selectbox("india is located in?",['','Asia','USA','Africa'])

# #############static prompt ############
# capital_input='Delhi'
# continent_input='Asia'
# template = PromptTemplate(
#     template="""Write a summary on "{capital_input}" which is in the continent of "{continent_input}".
# You can add geographical location details and describe it in a crisp form within 700 words and dont 
# truncate the paragraph instead complete the sentence even if its less that 700.
# """,
#     input_variables=["capital_input", "continent_input"]
# )


# ############# Dynamic prompt ############
template=load_prompt('template.json')
prompt=template.invoke({
    'capital_input':capital_input,
    'continent_input':continent_input
})

if st.button("Click me"):
    result=model.invoke(prompt)
    st.write(result.content)
# print(result.content)
