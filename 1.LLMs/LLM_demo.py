from langchain_openai import OpenAI  # langchain_openai is an interagation library using which langain understands how to interatct with openai llms using api
from dotenv import load_dotenv # dotenv function is used to identify and import secret key/ env variables stroed in .env files


load_dotenv() # invoked the function

llm= OpenAI(model='gpt-3.5-turbo-instruct')

result=llm.invoke("what is the capital of india?")

print(result)
