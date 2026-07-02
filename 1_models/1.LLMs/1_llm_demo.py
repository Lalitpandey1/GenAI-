from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
llm = OpenAI(model='gpt-4o-mini')    # this method is used to create an object of llm and it's configuration

result = llm.invoke("What is the capital on india")    #this method is used to communicate with the llms
print(result.content)