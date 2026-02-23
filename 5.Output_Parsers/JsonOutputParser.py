from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate   # want to make a dynamic template
from langchain_core.output_parsers import JsonOutputParser

load_dotenv(dotenv_path=r"2.ChatModels\.env")

llm=HuggingFaceEndpoint(repo_id='Qwen/Qwen3-Coder-Next',
        task="conversational")

model = ChatHuggingFace(llm=llm)

parser=JsonOutputParser()

template1= PromptTemplate(
    template ="Give me name, age, gender of a fictional person , \n {format_information}",
    input_variables=[],
    partial_variables={'format_information':parser.get_format_instructions()}
)

chain = template1 | model |parser

result = chain.invoke({})
print(result)

# prompt=template1.invoke({})
# result=model.invoke(prompt)
# print(result.content)
# print(parser.parse(result.content))  # we can use .invoke instead of .parser as well




