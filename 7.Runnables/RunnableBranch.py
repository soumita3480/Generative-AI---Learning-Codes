from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableBranch,RunnableLambda
from langchain_core.prompts import PromptTemplate

load_dotenv(dotenv_path=r"3.EmbaddedModels\.env")


# model (task specific runnable)
llm=HuggingFaceEndpoint(repo_id='Qwen/Qwen3-Coder-Next',
        task="conversational")

model =ChatHuggingFace(llm=llm)
parser=StrOutputParser()

# prompt1

prompt1=PromptTemplate(
    template="give me an essay on - {topic}",
    input_variables=['topic']
)


# prompt2

prompt2=PromptTemplate(
    template="summarize the topic within 500 words - {summarize}",
    input_variables=['summarize']
)

runnable_joke= RunnableSequence(prompt1,model,parser)

# in runnable ranch we have to have 'else' condition with all the 'if's
runnable_branch= RunnableBranch(
    (lambda x: len(x.split())<100,RunnableSequence(prompt2,model,parser)),
    # (RunnablePassthrough())
    RunnableLambda(lambda x:'its more than 100 words')
)

chain=RunnableSequence(runnable_joke,runnable_branch)

print(chain.invoke({'topic':'genAI'}))