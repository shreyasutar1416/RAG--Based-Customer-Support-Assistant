from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

CHROMA_DIR = "./chroma_db"
PDF_PATH = "data/Customer-Service-Handbook.pdf"


def ingest_pdf():
    print("📄 Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print("✂️  Chunking...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    print("🔢 Embedding and storing in ChromaDB...")
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=CHROMA_DIR
    )
    print(f"✅ Ingested {len(chunks)} chunks into ChromaDB at '{CHROMA_DIR}'")
    return vectorstore


if __name__ == "__main__":
    ingest_pdf()