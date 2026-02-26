from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser

# imported all env variables
load_dotenv(dotenv_path=r'3.EmbaddedModels\.env')

# model (task specific runnable)
llm=HuggingFaceEndpoint(repo_id='Qwen/Qwen3-Coder-Next',
        task="conversational")

model =ChatHuggingFace(llm=llm)

# parser
parser=StrOutputParser()

# prompt (task specific runnable)
prompt=PromptTemplate(
    template="give me a joke on this topic -{topic}",
    input_variables=['topic']
)

# chain
chain=RunnableSequence(prompt,model,parser)  # prompt, model, parser these are task specific runnables joined with runnable primitives

result=chain.invoke({'topic':'cricket'})

print(result)