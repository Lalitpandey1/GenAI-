# from langchain_community.document_loaders import TextLoader
# loader = TextLoader("example.txt", encoding = 'utf-8')
# docs = loader.load()
# print(docs)
# # print(type(docs))
# # print(docs[0].page_content)
# # print(docs[0].metadata)
# # print(type(docs[0]))

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

model = ChatOpenAI()
parser = StrOutputParser()
prompt = PromptTemplate(template='Write a poem on topic - {topic}', input_variables=['topic'])
chain = prompt | model | parser
result = chain.invoke({'topic' : 'AI'})
print(result)