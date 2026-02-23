from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv(dotenv_path=r"2.ChatModels\.env")

llm=HuggingFaceEndpoint(repo_id='Qwen/Qwen3-Coder-Next',
        task="conversational")

model =ChatHuggingFace(llm=llm)

parser= StrOutputParser() 

prompt1=PromptTemplate(
    template="give a detailed summary on thr topic -{topic}",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template="give top 5 points from the given information -{text}",
    input_variables=['text']
)

chain = prompt1 | model | parser | prompt2 | model | parser

print(chain.invoke({'topic':'stock market of walls street'}))

chain.get_graph().print_ascii()