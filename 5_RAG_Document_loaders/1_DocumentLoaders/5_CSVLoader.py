from langchain_community.document_loaders import CSVLoader
from langchain_openai import ChatOpenAI
# creates document for each row in the CSV file, with the column names as metadata
loader = CSVLoader(file_path="orderbook.csv", encoding="utf-8")
data = loader.load()
print(len(data))
print(data[0].page_content)