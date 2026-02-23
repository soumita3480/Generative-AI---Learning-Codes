from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate   # want to make a dynamic template

load_dotenv(dotenv_path=r"2.ChatModels\.env")

llm=HuggingFaceEndpoint(repo_id='Qwen/Qwen3-Coder-Next',
        task="conversational")

model = ChatHuggingFace(llm=llm)

template1= PromptTemplate(
    template ="write a summary on -{topic}",
    input_variables=['topic']
)

prompt1=template1.invoke({'topic':'black hole'})
result1=model.invoke(prompt1)

template2=PromptTemplate(
    template="convert this summary into 5 line text - {text}",
    input_variables=['text']
)


prompt2=template2.invoke({'text':result1.content})
result=model.invoke(prompt2)
print(result.content)
# print(result1.content)