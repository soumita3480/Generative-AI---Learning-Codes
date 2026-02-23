from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate   # want to make a dynamic template
from langchain_core.output_parsers import StrOutputParser

load_dotenv(dotenv_path=r"2.ChatModels\.env")

llm=HuggingFaceEndpoint(repo_id='Qwen/Qwen3-Coder-Next',
        task="conversational")

model = ChatHuggingFace(llm=llm)

parser=StrOutputParser()

template1= PromptTemplate(
    template ="write a summary on -{topic}",
    input_variables=['topic']
)

template2=PromptTemplate(
    template="convert this summary into 5 line text - {text}",
    input_variables=['text']
)

chain = template1 | model | parser| template2 | model | parser  # if we use string parser here, we no more need to use result.content just to extract the main data and not the metadata

result=chain.invoke({'topic':'Black Hole'})
print(result)

