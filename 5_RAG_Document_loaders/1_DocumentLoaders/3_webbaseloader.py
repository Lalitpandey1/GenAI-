from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI

url = "https://www.flipkart.com"
loader = WebBaseLoader(url)
docs = loader.load()

docs = loader.lazy_load()

for doc in docs:
    print(doc.metadata)