from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS 
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.2)

video_id = "Rni7Fz7208c"

try: 
    # 1. Instantiate the class instance
    api = YouTubeTranscriptApi()
    
    # 2. Call .fetch() and chain .to_raw_data() to get the list of dicts
    transcript_list = api.fetch(video_id, languages=['en']).to_raw_data()
    
    # 3. Flatten it into plain text
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    
    print("Transcript loaded successfully!\n")
    # print(transcript[:500] + "...") # Verifying the first 500 characters
    
except TranscriptsDisabled:
    print("No Transcript available for this video")

# Split the transcript into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.create_documents([transcript])

# print(f"Number of chunks created: {len(chunks)}")  # Check how many chunks were created


# Create a FAISS vector store from the chunks
   # first creating embeddings for the chunks
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    #store in the vector store
vector_store = FAISS.from_documents(chunks,embeddings)
# print(vector_store.index_to_docstore_id)

# Part 2 : Create Retrivals
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={"k":3})
# print(retriever)
# print(vector_store.get_by_ids(['51e7fe46-d98f-4907-9629-9e00965a4843','318e3f40-bf09-4d7e-ab41-f64bd33cc80f']))

# Part 3 Augmentation 
prompt = PromptTemplate(template='You are a helpful agent.Answer only from the provided context, if not available say I don\'t know question {context}\n\nQuestion: {question}', input_variables=['context', 'question'])

question = "What is the main topic of the video?"
retrieved_docs    = retriever.invoke(question)
# print(retrieved_docs)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
# print(context_text)

final_prompt = prompt.invoke({"context": context_text, "question": question})
answer = llm.invoke(final_prompt)
# print(answer.content)


# Step 4 - Generation
answer = llm.invoke(final_prompt)
# print(answer.content)

# Building a Chain
def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

# print(parallel_chain.invoke('What is next goal of Elon Musk?'))
parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

result = main_chain.invoke('Can you summarize the video')

print(result)