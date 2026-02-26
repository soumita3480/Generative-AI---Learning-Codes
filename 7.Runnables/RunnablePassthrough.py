from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough
from langchain_core.prompts import PromptTemplate

load_dotenv(dotenv_path=r"3.EmbaddedModels\.env")


# model (task specific runnable)
llm=HuggingFaceEndpoint(repo_id='Qwen/Qwen3-Coder-Next',
        task="conversational")

model =ChatHuggingFace(llm=llm)
parser=StrOutputParser()

# prompt1

prompt1=PromptTemplate(
    template="generate a joke on topic - {topic}",
    input_variables=['topic']
)

# prompt2

prompt2=PromptTemplate(
    template="Give explanation of this given joke -> {explanation}",
    input_variables=['explanation']
)

runnable_joke= RunnableSequence(prompt1,model,parser)

runnable_parallel= RunnableParallel({
    'explanation':RunnableSequence(prompt2,model,parser),
    'joke':RunnablePassthrough()
})

chain=RunnableSequence(runnable_joke,runnable_parallel)

print(chain.invoke({'topic':'cricket'}))