from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from bs4 import BeautifulSoup
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
url = 'https://www.flipkart.com/lenovo-loq-intel-core-i7-13th-gen-13645hx-16-gb-1-tb-ssd-windows-11-home-8-gb-graphics-nvidia-geforce-rtx-5060-15irx10-gaming-laptop/p/itm5dd600329a32e?pid=  COMHMZBNQXSHZVNV&lid=LSTCOMHMZBNQXSHZVNVNQXQVZ&marketplace=FLIPKART&cmpid=content_computer_22567620189_g_8965229628_gmc_pla&tgi=sem,1,G,11214002,g,search,,753186311945,,,,m,,mobile,,,,,&entryMethod=22567620189&cmpid=content_22567620189_gmc_pla&gad_source=1&gad_campaignid=22567620189&gclid=CjwKCAjwmJjSBhB-EiwAkZgxi60jclzICEV73IyKIgQxMaUEwW9kWoeDDZ6vNfGKnMJDFZ-04skWyRoCH8cQAvD_BwE'
loader  = WebBaseLoader(url)
docs = loader.load()
model = ChatOpenAI()
prompt = PromptTemplate(
    template='Answer the user query based on the -\n{q} from the following product description:\n{product_description}',input_variables=['q', 'product_description']
)
parser = StrOutputParser()
chain = prompt | model | parser
result = chain.invoke({'q': 'What is GPU of this product', 'product_description': docs[0].page_content})
print(result)