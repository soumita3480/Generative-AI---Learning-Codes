from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chatai=ChatOpenAI(model='gpt-4',max_completion_token=10) # can limit no of tokens in ans as the subscription is a count of tokens asked for.. so we can use our subscription wisely

result = chatai.invoke("what is the capital of india?")

print(result.content)