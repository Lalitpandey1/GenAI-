from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(max_completion_tokens=100)

while True:
    user_input = input("You : ")
    if user_input == 'exit':
        break
    result = model.invoke(user_input)
    print("AI: ",result.content)