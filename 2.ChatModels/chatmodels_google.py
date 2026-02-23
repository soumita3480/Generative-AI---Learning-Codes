from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv load_dotenv

load_dotenv()  # we got the api key from .env file by calling this funciton
model=ChatGoogleGenerativeAI(model='gemini-1.5-pro')

result=model.invoke("what is the capital of india?")

print(result.content)