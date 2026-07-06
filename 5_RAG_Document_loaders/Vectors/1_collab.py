# !pip install langchain chromadb openai tiktoken pypdf langchain_openai langchain_community langchain-chroma
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

doc1 = Document(page_content='A modern-day batting legend, Kohli’s unyielding drive, flawless technique, and chasing mastery have shattered historic records, cementing his legacy as one of cricket most dominant and fiercely passionate icons.',
    metadata = {"team":"Royal Challengers Bangalore"}
)
doc2 = Document(page_content='Renowned for his effortless elegance, record-breaking double centuries, and tactical brilliance, the powerhouse opening batter and inspirational leader has anchored India top order with sheer destructive grace.',
    metadata = {"team":"Mumbai Indian"}
)
doc3 = Document(page_content='The ultimate master of calm, Dhoni redefined wicketkeeping and finishing, steering Indian cricket to historic global triumphs with his legendary intuition, lightning reflexes, and unmatched leadership composure.',
    metadata = {"team":"Chennai Super Kings"}
)
doc4 = Document(page_content='Weaponizing a unique action, deadly yorkers, and elite tactical intelligence, the premier pace spearhead routinely dismantles world-class batting lineups across all formats as India’s ultimate match-winner.',
    metadata = {"team":"Mumbai Indian"}
)
doc5 = Document(page_content='A premier three-dimensional asset, "Sir" Jadeja turns games single-handedly through electric, athletic fielding, deceptive, high-accuracy left-arm spin, and vital, clutch batting contributions in high-pressure moments.',
                metadata = {"team":"Chennai Super Kings"}
)

docs = [doc1,doc2,doc3,doc4,doc5]

vector_store = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory='chroma_db',
    collection_name='sample_ipl'
)

vector_store.add_documents(docs)

# View Documents
vector_store.get(include=['embeddings','documents','metadatas'])

print(vector_store.similarity_search(query="Who is the best captain in IPL?", k=1 ))