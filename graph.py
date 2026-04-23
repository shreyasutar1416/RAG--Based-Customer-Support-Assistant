import os
from typing import TypedDict
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from langgraph.constants import END

from retriever import load_retriever
from hitl import should_escalate, human_handoff

load_dotenv()

# ─────────────────────────────────────────────
# State Definition
# ─────────────────────────────────────────────
class SupportState(TypedDict):
    query: str          # user's input
    context: list       # retrieved chunks
    response: str       # LLM generated answer
    escalated: bool     # whether HITL was triggered
    final_answer: str   # what gets shown to user


# ─────────────────────────────────────────────
# LLM + Retriever
# ─────────────────────────────────────────────
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

retriever = load_retriever(k=3)


# ─────────────────────────────────────────────
# Node 1: Retrieve
# ─────────────────────────────────────────────
def retrieve_node(state: SupportState) -> SupportState:
    """Fetches top-k relevant chunks from ChromaDB."""
    docs = retriever.invoke(state["query"])
    context = [doc.page_content for doc in docs]
    return {**state, "context": context}


# ─────────────────────────────────────────────
# Node 2: Generate
# ─────────────────────────────────────────────
def generate_node(state: SupportState) -> SupportState:
    """Generates answer from context using Groq LLM."""
    context_text = "\n\n".join(state["context"])

    prompt = f"""You are a helpful customer support assistant.
Use ONLY the context below to answer the question.
If the answer is not available in the context, say "I don't know."

Context:
{context_text}

Question: {state["query"]}

Answer:"""

    result = llm.invoke([HumanMessage(content=prompt)])
    response = result.content

    return {**state, "response": response}


# ─────────────────────────────────────────────
# Node 3: Route (HITL check)
# ─────────────────────────────────────────────
def route_node(state: SupportState) -> SupportState:
    """Decides: send to output or escalate to human."""
    escalate = should_escalate(
        query=state["query"],
        response=state["response"],
        context_docs=state["context"]
    )

    if escalate:
        human_reply = human_handoff(state["query"])
        return {
            **state,
            "escalated": True,
            "final_answer": human_reply
        }

    return {
        **state,
        "escalated": False,
        "final_answer": state["response"]
    }


# ─────────────────────────────────────────────
# Build Graph
# ─────────────────────────────────────────────
def build_graph():
    workflow = StateGraph(SupportState)

    # Register nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("route", route_node)

    # Define flow
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "route")
    workflow.add_edge("route", END)

    return workflow.compile()