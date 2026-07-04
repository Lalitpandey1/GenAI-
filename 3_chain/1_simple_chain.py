from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()


prompt = PromptTemplate(
    template='Generate 5 intresting fact about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI()

parser = StrOutputParser()

chain = model | prompt | parser

result = chain.invoke({'topic':'Python programming language'})
print(result)