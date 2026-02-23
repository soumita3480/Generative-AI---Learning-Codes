from langchain_anthropic import ChatAnthropic
from dotenv load_dotenv

load_dotenv()

model=ChatAnthropic(mdoel='claude-3-5-sonnet-20241022')

result=model.invoke("what is the capital of india")
print(result.content)