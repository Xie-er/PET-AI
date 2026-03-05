import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Embeddings
# If DASHSCOPE_API_KEY is not set, this might fail unless handled.
def get_embeddings():
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Warning: DASHSCOPE_API_KEY not found in environment variables.")
        return None
    return DashScopeEmbeddings(model="text-embedding-v1")

def build_vector_store(file_path: str, persist_directory: str = "./chroma_db"):
    """
    Builds a Chroma vector store from a text file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    embeddings = get_embeddings()
    if embeddings is None:
        raise ValueError("Embeddings could not be initialized. Check API Key.")

    # Create and persist vector store
    db = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    return db

def get_retriever(persist_directory: str = "./chroma_db"):
    """
    Returns a retriever from the existing vector store.
    """
    embeddings = get_embeddings()
    if embeddings is None:
        return None
        
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": 2})

if __name__ == "__main__":
    # Test building the vector store
    try:
        print("Building Vector Store...")
        build_vector_store("data/pet_knowledge.txt")
        print("Vector Store built successfully.")
    except Exception as e:
        print(f"Error building vector store: {e}")
