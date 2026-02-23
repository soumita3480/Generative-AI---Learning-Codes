from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


load_dotenv(dotenv_path=r"2.ChatModels\.env")

####### model 1 #######
llm1=HuggingFaceEndpoint(repo_id='Qwen/Qwen3-Coder-Next',
        task="conversational")

model1 =ChatHuggingFace(llm=llm1)

####### model 2 #######
llm2 = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational",
    max_new_tokens=3000,
)

model2= ChatHuggingFace(llm=llm2)


parser= StrOutputParser() 

template1=PromptTemplate(
    template="give a detailed summary on thr topic -{topic}",
    input_variables=['topic']
)

template2=PromptTemplate(
    template="generate notes from the given text -{text}",
    input_variables=['text']
)

template3=PromptTemplate(
    template="generate quiz from the given text -{text}",
    input_variables=['text']
)

template4=PromptTemplate(
    template="merge the given text - {notes} \n {quiz}",
    input_variables=['notes','quiz']
)

prompt1= template1 | model1 | parser
prompt2=RunnableParallel({
    'notes': template2 | model1 | parser,
    'quiz': template3 | model2 | parser
})

chain= prompt1 | prompt2 | template4 | model2 | parser



print(chain.invoke({'topic':'stock market of walls street'}))

chain.get_graph().print_ascii()