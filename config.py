"""
Centralized configuration for RAG Customer Support Assistant.
Uses pathlib for OS-agnostic path handling.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Suppress excessive third-party logging before anything else loads
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).parent.resolve()
CHROMA_DIR = BASE_DIR / "chroma_db"
DATA_DIR = BASE_DIR / "data"
PDF_PATH = DATA_DIR / "Customer-Service-Handbook.pdf"

# Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# LLM Backend Selection: "ollama" | "huggingface" | "mock"
LLM_BACKEND = os.getenv("LLM_BACKEND", "huggingface")

# Ollama Settings (only used if LLM_BACKEND="ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")

# HuggingFace Settings (only used if LLM_BACKEND="huggingface")
# Using a small model that works on CPU: TinyLlama or similar
HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
HF_MAX_LENGTH = int(os.getenv("HF_MAX_LENGTH", "512"))
HF_TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0.3"))

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Retrieval Configuration
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "4"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.7"))

# HITL Configuration
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))

# ChromaDB
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "customer_support")

# Ensure directories exist
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

