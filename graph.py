"""
LangGraph Workflow for RAG Customer Support Assistant.

Graph Architecture:
    START → retrieve → generate → route_decision
                              ↓ (needs_escalation=True)
                         hitl_escalation → finalize
                              ↓ (needs_escalation=False)
                         finalize → END

Supports multiple LLM backends:
    - HuggingFace local pipeline (default - no external server)
    - Ollama / OpenAI-compatible API (optional)
    - Mock fallback (extractive answers when no LLM available)
"""

import logging
import os
from typing import TypedDict, Optional, Literal

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage

from config import (
    LLM_BACKEND,
    LLM_MODEL, LLM_BASE_URL, LLM_API_KEY,
    HF_LLM_MODEL, HF_MAX_LENGTH, HF_TEMPERATURE,
    CONFIDENCE_THRESHOLD
)
from retriever import SupportRetriever, create_retriever
from hitl import HITLEscalationManager, EscalationReason, create_hitl_manager

logger = logging.getLogger(__name__)


# =============================================================================
# STATE SCHEMA
# =============================================================================

class GraphState(TypedDict):
    query: str
    documents: list
    context: str
    answer: str
    confidence: float
    needs_escalation: bool
    escalation_reason: Optional[str]
    human_response: Optional[str]
    final_output: str
    metadata: dict


# =============================================================================
# LLM INITIALIZATION (Multi-Backend)
# =============================================================================

_llm_instance = None
_hf_pipeline = None


def _get_llm_huggingface():
    """Initialize HuggingFace local pipeline (default - no server needed)."""
    global _hf_pipeline
    if _hf_pipeline is not None:
        return _hf_pipeline

    try:
        from transformers import pipeline, AutoTokenizer
        import torch

        logger.info(f"🤖 Loading HuggingFace model: {HF_LLM_MODEL}")
        logger.info("   This may take a few minutes on first run (downloading model)...")

        tokenizer = AutoTokenizer.from_pretrained(HF_LLM_MODEL)

        _hf_pipeline = pipeline(
            "text-generation",
            model=HF_LLM_MODEL,
            tokenizer=tokenizer,
            dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            max_length=HF_MAX_LENGTH,
            temperature=HF_TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        logger.info("✅ HuggingFace model loaded successfully")
        return _hf_pipeline

    except Exception as e:
        logger.warning(f"⚠️  HuggingFace load failed: {e}")
        return None


def _get_llm_ollama():
    """Initialize Ollama / OpenAI-compatible client."""
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    try:
        from langchain_openai import ChatOpenAI

        _llm_instance = ChatOpenAI(
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            temperature=0.3,
            max_tokens=1024
        )
        # Test connectivity
        _llm_instance.invoke([HumanMessage(content="Hi")])
        logger.info(f"✅ Ollama LLM connected: {LLM_MODEL}")
        return _llm_instance

    except Exception as e:
        logger.warning(f"⚠️  Ollama unavailable ({e})")
        return None


def _get_llm():
    """
    Initialize LLM based on configured backend.
    Falls back gracefully if primary backend fails.
    """
    if LLM_BACKEND == "huggingface":
        llm = _get_llm_huggingface()
        if llm is not None:
            return ("huggingface", llm)
        logger.info("Falling back to Ollama...")
        llm = _get_llm_ollama()
        if llm is not None:
            return ("ollama", llm)

    elif LLM_BACKEND == "ollama":
        llm = _get_llm_ollama()
        if llm is not None:
            return ("ollama", llm)
        logger.info("Falling back to HuggingFace...")
        llm = _get_llm_huggingface()
        if llm is not None:
            return ("huggingface", llm)

    logger.warning("⚠️  No LLM available. Using fallback extractive generation.")
    return ("mock", None)


def _generate_with_hf(pipeline, system_prompt: str, user_prompt: str) -> str:
    """Generate text using HuggingFace pipeline."""
    try:
        # Format for chat models
        messages = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{user_prompt}</s>\n<|assistant|>\n"

        outputs = pipeline(
            messages,
            max_new_tokens=256,
            return_full_text=False
        )
        generated = outputs[0]["generated_text"].strip()

        # Clean up any trailing prompts
        if "<|" in generated:
            generated = generated.split("<|")[0].strip()

        return generated

    except Exception as e:
        logger.error(f"HuggingFace generation failed: {e}")
        return ""


def _generate_with_ollama(llm, system_prompt: str, user_prompt: str) -> str:
    """Generate text using Ollama / OpenAI-compatible API."""
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        logger.error(f"Ollama generation failed: {e}")
        return ""


# =============================================================================
# NODE: RETRIEVE
# =============================================================================

def retrieve_node(state: GraphState) -> dict:
    """Retrieve relevant documents from the vector store."""
    query = state["query"]
    logger.info(f"📚 [Node: retrieve] Query: '{query[:60]}...'")

    retriever = create_retriever()
    docs, avg_score = retriever.retrieve(query, expand=True)
    context = retriever.get_context_string(docs)
    has_context = retriever.is_context_sufficient(docs, avg_score)

    return {
        "documents": docs,
        "context": context,
        "metadata": {
            **state.get("metadata", {}),
            "retrieval_score": round(avg_score, 3),
            "num_docs": len(docs),
            "has_context": has_context
        }
    }


# =============================================================================
# NODE: GENERATE
# =============================================================================

def generate_node(state: GraphState) -> dict:
    """Generate an answer using the LLM and retrieved context."""
    query = state["query"]
    context = state["context"]
    has_context = state["metadata"].get("has_context", False)

    logger.info("🤖 [Node: generate] Drafting answer...")

    system_prompt = (
        "You are a helpful customer support assistant. "
        "Answer the user's question using ONLY the provided context. "
        "If the context does not contain enough information, say so clearly. "
        "Be concise, professional, and friendly. "
        "Cite document numbers when referencing specific information."
    )

    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    backend, llm = _get_llm()
    answer = ""

    if backend == "huggingface" and llm is not None:
        answer = _generate_with_hf(llm, system_prompt, user_prompt)
    elif backend == "ollama" and llm is not None:
        answer = _generate_with_ollama(llm, system_prompt, user_prompt)

    if not answer:
        answer = _fallback_generate(query, context, has_context)
        confidence = 0.4 if not has_context else 0.6
        generation_method = "fallback"
    else:
        confidence = _estimate_confidence(answer, has_context)
        generation_method = backend

    return {
        "answer": answer,
        "confidence": confidence,
        "metadata": {
            **state.get("metadata", {}),
            "generation_method": generation_method
        }
    }


def _fallback_generate(query: str, context: str, has_context: bool) -> str:
    """Fallback answer generation when LLM is unavailable."""
    if not has_context or "No relevant information" in context:
        return (
            "I'm sorry, but I don't have enough information in my knowledge base "
            "to answer your question accurately. Could you please rephrase or provide "
            "more details? Alternatively, I can connect you with a human agent."
        )

    parts = context.split("\n\n---\n\n")
    if parts:
        excerpt = parts[0].replace("[Document 1 |", "According to our documentation [")
        return f"{excerpt}\n\nI hope this helps! Let me know if you need more details."
    return context[:800]


def _estimate_confidence(answer: str, has_context: bool) -> float:
    """Heuristic confidence estimation based on answer characteristics."""
    if not has_context:
        return 0.3

    score = 0.6

    if len(answer) > 100 and len(answer) < 800:
        score += 0.1

    if "[" in answer and "]" in answer:
        score += 0.1

    uncertainty_words = ["not sure", "don't know", "insufficient", "cannot", "unable"]
    if any(w in answer.lower() for w in uncertainty_words):
        score -= 0.2

    return max(0.0, min(1.0, score))


# =============================================================================
# NODE: ROUTE DECISION (Conditional Logic)
# =============================================================================

def route_decision(state: GraphState) -> Literal["hitl", "finalize"]:
    """Determine whether to escalate to human review or finalize directly."""
    query = state["query"]
    confidence = state["confidence"]
    has_context = state["metadata"].get("has_context", False)
    answer = state["answer"]

    logger.info("🔀 [Node: route_decision] Evaluating routing...")

    hitl = create_hitl_manager(confidence_threshold=CONFIDENCE_THRESHOLD)
    should_escalate, reason = hitl.should_escalate(query, confidence, has_context, answer)

    if should_escalate:
        logger.info(f"   → Route to HITL ({reason.value if reason else 'unknown'})")
        return "hitl"
    else:
        logger.info("   → Route to finalize (direct answer)")
        return "finalize"


# =============================================================================
# NODE: HITL ESCALATION
# =============================================================================

def hitl_node(state: GraphState) -> dict:
    """Handle human-in-the-loop escalation via CLI interaction."""
    query = state["query"]
    answer = state["answer"]
    context = state["context"]

    logger.info("👤 [Node: hitl_escalation] Awaiting human review...")

    hitl = create_hitl_manager(confidence_threshold=CONFIDENCE_THRESHOLD)

    _, reason = hitl.should_escalate(query, state["confidence"], True, answer)
    if reason is None:
        reason = EscalationReason.LOW_CONFIDENCE

    decision = hitl.request_human_review(
        query=query,
        reason=reason,
        draft_answer=answer,
        context=context
    )

    ticket = hitl.create_escalation_ticket(query, reason, decision)

    human_response = decision.human_response if decision.human_response else None

    return {
        "needs_escalation": True,
        "escalation_reason": reason.value,
        "human_response": human_response,
        "metadata": {
            **state.get("metadata", {}),
            "ticket_id": ticket["ticket_id"],
            "hitl_status": decision.status.value,
            "moderator_notes": decision.moderator_notes
        }
    }


# =============================================================================
# NODE: FINALIZE
# =============================================================================

def finalize_node(state: GraphState) -> dict:
    """Produce the final output delivered to the user."""
    logger.info("📤 [Node: finalize] Preparing final output...")

    if state.get("human_response"):
        final_output = state["human_response"]
    else:
        final_output = state["answer"]

    metadata = state.get("metadata", {})
    if metadata.get("num_docs", 0) > 0:
        sources = []
        for i, doc in enumerate(state["documents"], 1):
            page = doc.metadata.get("page", "?")
            sources.append(f"[{i}] Page {page}")
        if sources:
            final_output += f"\n\n📚 Sources: {', '.join(sources)}"

    return {
        "final_output": final_output
    }


# =============================================================================
# GRAPH COMPILATION
# =============================================================================

def build_graph():
    """Build and compile the LangGraph state machine."""
    builder = StateGraph(GraphState)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("hitl", hitl_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_conditional_edges(
        "generate",
        route_decision,
        {
            "hitl": "hitl",
            "finalize": "finalize"
        }
    )
    builder.add_edge("hitl", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


graph = build_graph()


def run_support_pipeline(query: str) -> dict:
    """Execute the complete support pipeline for a user query."""
    logger.info("=" * 60)
    logger.info("🚀 Starting Support Pipeline")
    logger.info(f"Query: {query}")
    logger.info("=" * 60)

    initial_state: GraphState = {
        "query": query,
        "documents": [],
        "context": "",
        "answer": "",
        "confidence": 0.0,
        "needs_escalation": False,
        "escalation_reason": None,
        "human_response": None,
        "final_output": "",
        "metadata": {}
    }

    final_state = graph.invoke(initial_state)

    logger.info("=" * 60)
    logger.info("✅ Pipeline Complete")
    logger.info("=" * 60)

    return final_state


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_support_pipeline("How do I reset my password?")
    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    print(result["final_output"])
    print("=" * 60)
    print("Metadata:", result["metadata"])

