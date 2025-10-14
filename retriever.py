# retriever.py
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize embeddings model (must match ingest.py)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load persisted ChromaDB
vectordb = Chroma(
    persist_directory="vector_store/",
    embedding_function=embedding_model
)

# Retriever for similarity search
retriever = vectordb.as_retriever(search_kwargs={"k": 5})