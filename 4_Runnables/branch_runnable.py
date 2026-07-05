from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableBranch, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()
parser = StrOutputParser()
prompt1 = PromptTemplate(template='Write a detailed report on topic - {topic}',input_variables=['topic'])
prompt2 = PromptTemplate(template='Write a quick summary on topic - {text}',input_variables=['text'])

report_gen_chain = RunnableSequence(prompt1, model, parser)

branch_Chain = RunnableBranch(
    (lambda x: len(x.split())>300, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()

)
