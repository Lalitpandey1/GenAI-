from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()


# Part 1 -> Retrival of sentement 
class Feedback(BaseModel):
    sentiment : Literal['Positive', 'Negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)
pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the feedback from the user into positive or negative.\n{format_instructions}\nFeedback: {feedback}', 
    input_variables=['feedback'], 
    partial_variables={'format_instructions': pydantic_parser.get_format_instructions()}
)

classifier_chain = prompt1 | model | pydantic_parser

# --- Part 2: Branching Responses Setup ---

prompt2 = PromptTemplate(template='If the feedback is positive, give a response to the user thanking them for their feedback \n {feedback}', input_variables=['feedback'])

prompt3 = PromptTemplate(template='Write the appropriate response based on negative feedback.\n {feedback}', input_variables=['sentiment', 'feedback'])


branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'Positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'Negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback': 'this is a best phone yet screen'})
print(result)