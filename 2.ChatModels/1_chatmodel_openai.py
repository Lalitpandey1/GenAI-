from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import httpx

load_dotenv()

client = httpx.Client(verify=False)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    http_client=client
)

result = llm.invoke("What is the capital of India?")
print(result.content)
