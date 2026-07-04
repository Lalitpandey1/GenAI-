from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(max_completion_tokens=100)

 
chat_history = [SystemMessage(content="You are a helpful assistant.")]

while True:
    user_input = input("You : ")
    if user_input == 'exit':
        break
    chat_history.append(HumanMessage(content=user_input))
    result = model.invoke(chat_history)
    print(result.content)
    chat_history.append(AIMessage(content=result.content))

print(result.content)