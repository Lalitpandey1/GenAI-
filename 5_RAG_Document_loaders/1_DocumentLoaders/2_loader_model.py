from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
loader = TextLoader("example.txt", encoding = 'utf-8')
docs = loader.load()
load_dotenv()
prompt = PromptTemplate(
    template='Write a summary for the following poem -\n{poem}',input_variables=['poem']
)
model = ChatOpenAI()
parser = StrOutputParser()

chain = prompt | model | parser
print(chain.invoke({'poem':docs[0].page_content}))