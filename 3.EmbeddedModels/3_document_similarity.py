from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=300)

documets = [
    "Delhi is the capital of India",
    "Mumbai is the financial capital of India",
    "Bangalore is the IT hub of India"]

query = "What is the capital of India?"

query_embedding = embedding.embed_query(query)
document_embeddings = embedding.embed_documents(documets)

scores = cosine_similarity([query_embedding], document_embeddings)[0]

index,score = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[0]

print(query)

print(documets[index])
print("Similarity Score:", score)