from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate,load_prompt # load_prompt required for dynamic prompt
# import streamlit as st
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
import os

load_dotenv(dotenv_path=r"C:\Users\hp\OneDrive\Desktop\Python Files_VS Code\Langchain_models\2.ChatModels\.env")

llm=HuggingFaceEndpoint(model='Qwen/Qwen3-Coder-Next',
        task="conversational"
        # max_completion_token=100)
)

model=ChatHuggingFace(llm=llm)


# capital_input=st.selectbox("what is the captal of india",['','Delhi','Kolkata','Mumbai'])
# continent_input=st.selectbox("india is located in?",['','Asia','USA','Africa'])


# ############# Dynamic prompt ############
template=load_prompt('template.json')
messeges=[] 

# chain=template | model #chain is used to pipepine all the process to be invoked instead of invoking them separately
# while True:
#     user_input=input('give your prompt:')
#     # messeges=[SystemMessage(content="I am a good model.. try me !!"), 
#     #         HumanMessage(content=user_input)
#     #         ]
#     messeges.append(user_input)  # we have to send latest prompt with chat history and append the result also
#     result=model.invoke(messeges)
#     messeges.append(result)
    
#     if user_input=='Exit':
#         break
    
#     print(result.content)


##### but AI might forget which ans was given by AI or it was my question, so we ahve to specify who has sent which ans/quesiton.. so to do that we use AImesseges, systemmesseges and humanmesseges
messeges=[SystemMessage(content="I am a good model.. try me !!")
            ]
while True:
    user_input=input('give your prompt:')
    
    messeges.append(HumanMessage(content=user_input))  # we have to send latest prompt with chat history and append the result also
    result=model.invoke(messeges)
    messeges.append(AIMessage(content=result.content))
    
    if user_input=='Exit':
        break
    
    print('AI:',result.content)
