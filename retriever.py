from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DIR = "./chroma_db"


def load_retriever(k: int = 3):
    """Load ChromaDB and return a retriever."""
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})