from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

# -------------------------------------------------
# 1️⃣ Load Environment Variables (.env file)
# -------------------------------------------------
load_dotenv()   # Make sure your HUGGINGFACEHUB_API_TOKEN is inside .env

# -------------------------------------------------
# 2️⃣ Create HuggingFace Chat Model
# -------------------------------------------------
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational",
    max_new_tokens=200,
)

chat_model = ChatHuggingFace(llm=llm)

# -------------------------------------------------
# 3️⃣ Create ChatPromptTemplate
# -------------------------------------------------
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

# -------------------------------------------------
# 4️⃣ Create Chain (Modern LangChain Style)
# -------------------------------------------------
chain = chat_template | chat_model

# -------------------------------------------------
# 5️⃣ Maintain Chat History
# -------------------------------------------------
chat_history = [
    HumanMessage(content="what is LLM"),
    AIMessage(content="LLM is large language model")
]


print("Type 'exit' to stop.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    # Invoke chain
    response = chain.invoke({
        "chat_history": chat_history,
        "query": user_input
    })

    # Print AI response
    print("AI:", response.content)

    # Update history properly
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(response)

    print('###### bot####',chat_history)
