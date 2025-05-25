# imports
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# load restaurant reviews from csv -> df
df = pd.read_csv("realistic_restaurant_reviews.csv")
# initialize embeddings model; 1536-dim, optimized for semantic search (text -> vector reps) 
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# vector persistent vector db location 
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    # lists to store docs and ids for each doc
    documents = []
    ids = []
    
    # loop through each review row from the df
    for i, row in df.iterrows():
        # create a document object with 
        # 1) title & review [concatenated for semantic understanding] 
        # 2) rating & date
        # 3) unique id 
        document = Document(
            page_content = row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)
        
# initialize chroma vector store (persistent vector database)
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

# if this is first run, add all documents to the vector store
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
# create retriever that returns top 5 most similar documents 
# cosine similarity used
# top 5 similar docs returned
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)
