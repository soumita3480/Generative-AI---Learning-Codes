from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda
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


runnable_joke= RunnableSequence(prompt1,model,parser)

def joke(word_count):
    word=0
    for words in word_count.split():
        if words:
            word+=1
    return word

runnable_parallel= RunnableParallel({
    'count_of_words_joke':RunnableLambda(joke),
    'joke':RunnablePassthrough()
})

chain=RunnableSequence(runnable_joke,runnable_parallel)

result=chain.invoke({'topic':'cricket'})

print(f"count of words are {result['count_of_words_joke']} and \n, joke is {result['joke']}")