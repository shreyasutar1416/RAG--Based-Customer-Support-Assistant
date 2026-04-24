"""
Retrieval Module for RAG Customer Support Assistant.

Provides context-aware retrieval from ChromaDB with relevance scoring,
re-ranking support, and fallback handling for low-confidence queries.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    RETRIEVER_K,
    RELEVANCE_THRESHOLD,
    COLLECTION_NAME
)

logger = logging.getLogger(__name__)


class SupportRetriever:
    """
    Advanced retriever for customer support knowledge base.
    
    Features:
        - Semantic search via ChromaDB
        - Relevance score filtering
        - MMR-based diversity re-ranking
        - Query expansion for better coverage
        - Fallback handling for no-relevant-context scenarios
    """

    def __init__(
        self,
        chroma_dir: Path = CHROMA_DIR,
        embedding_model: str = EMBEDDING_MODEL,
        collection_name: str = "customer_support",
        k: int = RETRIEVER_K,
        relevance_threshold: float = RELEVANCE_THRESHOLD,
        use_mmr: bool = False,
        mmr_lambda: float = 0.5
    ):
        self.k = k
        self.relevance_threshold = relevance_threshold
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda

        logger.info(f"🔎 Initializing retriever (k={k}, threshold={relevance_threshold})")
        embedding = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = Chroma(
            persist_directory=str(chroma_dir),
            embedding_function=embedding,
            collection_name=collection_name
        )

    def _expand_query(self, query: str) -> str:
        """
        Simple query expansion for better retrieval.
        Adds common customer support synonyms.
        """
        expansions = {
            "refund": "refund money back return payment",
            "cancel": "cancel stop terminate end subscription",
            "password": "password login credentials reset forgot",
            "shipping": "shipping delivery tracking package order",
            "support": "support help assistance contact customer service",
            "billing": "billing payment invoice charge subscription",
            "error": "error issue problem bug failure not working",
            "account": "account profile user settings preferences"
        }
        
        expanded_terms = []
        query_lower = query.lower()
        for keyword, expansion in expansions.items():
            if keyword in query_lower:
                expanded_terms.append(expansion)
        
        if expanded_terms:
            return f"{query} {' '.join(expanded_terms)}"
        return query

    def retrieve(self, query: str, expand: bool = True) -> Tuple[List[Document], float]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: User query string.
            expand: Whether to apply query expansion.

        Returns:
            Tuple of (list of relevant documents, average relevance score).
        """
        search_query = self._expand_query(query) if expand else query
        logger.info(f"🔍 Retrieving for: '{query[:80]}...'")

        if self.use_mmr:
            results = self.vectorstore.max_marginal_relevance_search(
                search_query,
                k=self.k,
                lambda_mult=self.mmr_lambda
            )
        else:
            results_with_scores = self.vectorstore.similarity_search_with_relevance_scores(
                search_query,
                k=self.k * 2  # Retrieve more for filtering
            )
            results = self._filter_by_relevance(results_with_scores)

        avg_score = self._calculate_average_score(results)
        logger.info(f"   Retrieved {len(results)} docs (avg relevance: {avg_score:.3f})")
        
        return results, avg_score

    def _filter_by_relevance(
        self,
        results_with_scores: List[Tuple[Document, float]]
    ) -> List[Document]:
        """Filter results by relevance threshold.
        
        ChromaDB may return negative cosine distances; normalize to 0-1.
        """
        if not results_with_scores:
            return []

        # Normalize scores: shift minimum to 0, scale max to 1
        scores = [score for _, score in results_with_scores]
        min_score, max_score = min(scores), max(scores)
        score_range = max_score - min_score if max_score != min_score else 1.0

        normalized = []
        for (doc, score) in results_with_scores:
            norm_score = (score - min_score) / score_range
            doc.metadata["relevance_score"] = round(norm_score, 3)
            normalized.append((doc, norm_score))

        filtered = [doc for doc, score in normalized if score >= self.relevance_threshold]

        if not filtered:
            logger.warning("   No docs passed relevance threshold, using top result")
            return [normalized[0][0]]
        return filtered[:self.k]

    def _calculate_average_score(self, documents: List[Document]) -> float:
        """Calculate average relevance score from document metadata."""
        if not documents:
            return 0.0
        scores = [
            doc.metadata.get("relevance_score", 0.5)
            for doc in documents
        ]
        return sum(scores) / len(scores)

    def get_context_string(self, documents: List[Document]) -> str:
        """Convert retrieved documents into a formatted context string."""
        if not documents:
            return "No relevant information found in the knowledge base."
        
        contexts = []
        for i, doc in enumerate(documents, 1):
            page = doc.metadata.get("page", "?")
            source = doc.metadata.get("source", "unknown")
            contexts.append(
                f"[Document {i} | Page {page} | Source: {source}]\n{doc.page_content.strip()}"
            )
        return "\n\n---\n\n".join(contexts)

    def is_context_sufficient(self, documents: List[Document], avg_score: float) -> bool:
        """
        Determine if retrieved context is sufficient to answer the query.
        
        Returns False if:
            - No documents retrieved
            - Average relevance below threshold
            - Documents are too short (potential low-quality match)
        """
        if not documents:
            return False
        if avg_score < self.relevance_threshold * 0.8:
            return False
        total_content = sum(len(doc.page_content) for doc in documents)
        if total_content < 200:  # Minimum context length
            return False
        return True


def create_retriever(**kwargs) -> SupportRetriever:
    """Factory function to create a SupportRetriever instance."""
    return SupportRetriever(**kwargs)


if __name__ == "__main__":
    # Simple test
    retriever = create_retriever()
    docs, score = retriever.retrieve("How do I request a refund?")
    print(f"\nRetrieved {len(docs)} documents (score: {score:.3f})")
    print("=" * 60)
    print(retriever.get_context_string(docs))

