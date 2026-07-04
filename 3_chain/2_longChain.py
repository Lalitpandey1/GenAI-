from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

prompt1 = PromptTemplate(
    template='generate the detailed report on {topic}',input_variables=['topic'])

prompt2 = PromptTemplate(
    template='generate 5 pointer summary {text}', input_variables=['text'])

model = ChatOpenAI()

parser = StrOutputParser()

chain = prompt1 | model | parser | RunnableLambda(lambda x: {'text': x}) | prompt2 | model | parser
result = chain.invoke({'topic':'Cancer in India'})
print(result)
