from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv

load_dotenv()
chat_history = []

model = ChatOpenAI(max_completion_tokens=100)

while True:
    user_input = input("You : ")
    if user_input == 'exit':
        break
    chat_history.append(user_input )
    result = model.invoke(chat_history)
    print(result.content)
    chat_history.append(result.content)
    
print(result.content)