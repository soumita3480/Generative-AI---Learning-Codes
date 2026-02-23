from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import Field, BaseModel
from typing import Literal

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

# schema

class Sentiment (BaseModel):
    Sentiment: Literal['Positive', 'Negative']= Field(description='the output should be either positive or negative')


parser= PydanticOutputParser(pydantic_object=Sentiment)
parser1=StrOutputParser()

prompt1= PromptTemplate(
    template='analyze the sentiment and give the response in positive or negative, {feedback}\n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

prompt2=PromptTemplate(
    template='give response for positive review, {feedback}',
    input_variables=['feedback']
)

prompt3=PromptTemplate(
    template='give response for negative review, {feedback}',
    input_variables=['feedback']
)

sequential_runnable = prompt1 | model1 | parser

conditional_runnables=RunnableBranch(
    (lambda x: x.Sentiment=='Positive', prompt2 | model1),
    (lambda x: x.Sentiment=='Negative', prompt3 | model1),
    RunnableLambda(lambda x: 'could not find statement')
)


chain = sequential_runnable | conditional_runnables | parser1

print(chain.invoke({'feedback':'the built quality of the phone is very poor'}))
# print(chain.invoke({'feedback':'the built quality of the phone is very good'}))

# print(prompt1)

chain.get_graph().print_ascii()