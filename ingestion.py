"""
Document Ingestion Pipeline for RAG Customer Support Assistant.

Steps:
    1. Load PDF using PyPDFLoader
    2. Split into chunks using RecursiveCharacterTextSplitter
    3. Generate embeddings using HuggingFaceEmbeddings
    4. Store in ChromaDB vector store
"""

import logging
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHROMA_DIR, PDF_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def ingest_pdf(
    pdf_path: Path = PDF_PATH,
    chroma_dir: Path = CHROMA_DIR,
    embedding_model: str = EMBEDDING_MODEL,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    collection_name: str = "customer_support"
) -> Chroma:
    """
    Ingest a PDF into ChromaDB vector store.

    Args:
        pdf_path: Path to the PDF file.
        chroma_dir: Directory to persist ChromaDB.
        embedding_model: HuggingFace model name for embeddings.
        chunk_size: Maximum chunk size.
        chunk_overlap: Overlap between chunks.
        collection_name: ChromaDB collection name.

    Returns:
        Chroma vector store instance.
    """
    if not pdf_path.exists():
        logger.error(f"PDF not found at: {pdf_path}")
        sys.exit(1)

    logger.info(f"📄 Loading PDF: {pdf_path.name}")
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    logger.info(f"   Loaded {len(documents)} pages")

    logger.info(f"✂️  Chunking (size={chunk_size}, overlap={chunk_overlap})")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"   Split into {len(chunks)} chunks")

    logger.info(f"🔎 Generating embeddings with '{embedding_model}'")
    embedding = HuggingFaceEmbeddings(model_name=embedding_model)

    logger.info(f"💾 Storing in ChromaDB at: {chroma_dir}")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=str(chroma_dir),
        collection_name=collection_name
    )

    logger.info(f"✅ Ingested {len(chunks)} chunks into collection '{collection_name}'")
    return vectorstore


def get_vectorstore(
    chroma_dir: Path = CHROMA_DIR,
    embedding_model: str = EMBEDDING_MODEL,
    collection_name: str = "customer_support"
) -> Chroma:
    """
    Load an existing ChromaDB vector store.
    """
    embedding = HuggingFaceEmbeddings(model_name=embedding_model)
    return Chroma(
        persist_directory=str(chroma_dir),
        embedding_function=embedding,
        collection_name=collection_name
    )


if __name__ == "__main__":
    ingest_pdf()

