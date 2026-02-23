from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate   # want to make a dynamic template
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from pydantic import BaseModel,Field   # using pydantic model calss to define structure, datetype (all it validates in rultime), and description og the i/p variables


load_dotenv(dotenv_path=r"2.ChatModels\.env")

llm=HuggingFaceEndpoint(repo_id='Qwen/Qwen3-Coder-Next',
        task="conversational")

model = ChatHuggingFace(llm=llm)

# defining scehma

class Person(BaseModel):
    name: str = Field(description='its the persons name')
    age: int = Field(gt=18, description='age of the person')
    

parser=PydanticOutputParser(pydantic_object=Person)



template1= PromptTemplate(
    template ="Give me name, age of a fictional person , \n {format_information}",
    input_variables=[],
    partial_variables={'format_information':parser.get_format_instructions()}
)

print(template1.invoke({}))  # the prompt
chain = template1 | model |parser

result = chain.invoke({})
print(result)





