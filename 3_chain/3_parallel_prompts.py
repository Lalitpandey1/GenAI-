from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

# Models
model1 = ChatOpenAI()
model2 = ChatOpenAI()
parser = StrOutputParser()

# Prompts
prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

# Individual parallel branches
notes_branch = prompt1 | model1 | parser
quiz_branch = prompt2 | model2 | parser

# Complete Chain
chain = (
    # Step 1: Split the input 'text' and run both branches in parallel
    RunnableParallel({
        "notes": notes_branch,
        "quiz": quiz_branch
    })
    # Step 2: Since RunnableParallel outputs a dict with {'notes': ..., 'quiz': ...},
    # it perfectly matches the exact schema prompt3 expects!
    | prompt3
    | model1  # You can use either model here to merge
    | parser
)

# Invoke the parallel chain
sample_text = "LangChain is a framework designed to simplify the creation of applications using LLMs..."
result = chain.invoke({"text": sample_text})
print(result)