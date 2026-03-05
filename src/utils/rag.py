import os
import shutil
from typing import List, Optional
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_embeddings():
    """Initialize and return DashScope embeddings."""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Warning: DASHSCOPE_API_KEY not found in environment variables.")
        return None
    return DashScopeEmbeddings(model="text-embedding-v1")

def build_vector_store_from_file(file_path: str, persist_directory: str):
    """
    Builds a Chroma vector store from a text or PDF file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Select loader based on extension
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding='utf-8')
    
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    embeddings = get_embeddings()
    if embeddings is None:
        raise ValueError("Embeddings could not be initialized. Check API Key.")

    # Create and persist vector store
    # Note: Chroma automatically persists when created with persist_directory
    db = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    return db

def delete_vector_store(persist_directory: str):
    """
    Deletes a vector store by removing its directory.
    """
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        return True
    return False

def get_retriever(persist_directory: str):
    """
    Returns a retriever from the existing vector store.
    """
    if not os.path.exists(persist_directory):
        return None
        
    embeddings = get_embeddings()
    if embeddings is None:
        return None
        
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": 3})

if __name__ == "__main__":
    # Test
    pass
