ESCALATION_KEYWORDS = [
    "complaint", "issue", "problem", "broken",
    "refund", "damaged", "not working", "angry"
]

UNCERTAINTY_PHRASES = [
    "i don't know", "i'm not sure", "cannot find",
    "no information", "not mentioned", "unclear"
]


def should_escalate(query: str, response: str, context_docs: list) -> bool:
    """
    Decides whether to escalate to a human agent.

    Escalation criteria:
    1. Query contains complaint/issue keywords
    2. No relevant context was retrieved
    3. LLM response signals uncertainty
    4. Response is too short (likely a non-answer)
    """
    # Keyword-based routing
    if any(word in query.lower() for word in ESCALATION_KEYWORDS):
        return True

    # No context retrieved
    if not context_docs:
        return True

    # LLM uncertainty
    if any(phrase in response.lower() for phrase in UNCERTAINTY_PHRASES):
        return True

    # Too short to be useful
    if len(response.strip()) < 30:
        return True

    return False


def human_handoff(query: str) -> str:
    """Simulate human agent taking over (CLI input for demo purposes)."""
    print("\n" + "="*50)
    print("🚨 HITL ESCALATION TRIGGERED")
    print(f"📋 Customer Query: {query}")
    print("="*50)
    human_response = input("👤 Human Agent — Enter your response: ")
    return human_response