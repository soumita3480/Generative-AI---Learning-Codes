from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate,load_prompt # load_prompt required for dynamic prompt
import streamlit as st
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
import os

load_dotenv(dotenv_path=r"C:\Users\hp\OneDrive\Desktop\Python Files_VS Code\Langchain_models\2.ChatModels\.env")

llm=HuggingFaceEndpoint(model='Qwen/Qwen3-Coder-Next',
        task="conversational"
        # max_completion_token=100)
)

model=ChatHuggingFace(llm=llm)

st.header("Dynamic Prompts- No Json")

capital_input=st.selectbox("what is the captal of india",['','Delhi','Kolkata','Mumbai'])
continent_input=st.selectbox("india is located in?",['','Asia','USA','Africa'])


# ############# Dynamic prompt ############
template=load_prompt('template.json')

chain=template | model #chain is used to pipepine all the process to be invoked instead of invoking them separately
# prompt=template.invoke({
#     'capital_input':capital_input,
#     'continent_input':continent_input
# })

messeges=[SystemMessage=content("I am a good model.. try me !!"),
          HumanMessage=content("what is the capital of delhi")]
if st.button("Click me"):
    result=chain.invoke(
        {
    'capital_input':capital_input,
    'continent_input':continent_input
})
    # result=model.invoke(prompt)
    st.write(result.content)

