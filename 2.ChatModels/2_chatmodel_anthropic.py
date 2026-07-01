from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

llm = ChatAnthropic(
    model="claude-3.0")

result = llm.invoke("What is the capital of India?")

print(result.content)