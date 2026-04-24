"""
Human-in-the-Loop (HITL) Module for RAG Customer Support Assistant.

Handles escalation scenarios where the AI is uncertain or the query
requires human judgment. Provides approval/rejection workflows and
integrates human responses back into the system.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class EscalationReason(Enum):
    """Reasons for escalating to human review."""
    LOW_CONFIDENCE = "low_confidence"
    MISSING_CONTEXT = "missing_context"
    COMPLEX_QUERY = "complex_query"
    SENSITIVE_DATA = "sensitive_data"
    POLICY_VIOLATION = "policy_violation"
    CUSTOMER_REQUEST = "customer_request"


class HITLStatus(Enum):
    """Status of HITL intervention."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    TIMEOUT = "timeout"


@dataclass
class HITLDecision:
    """Represents a human decision on an escalated query."""
    status: HITLStatus
    human_response: Optional[str] = None
    moderator_notes: Optional[str] = None
    timestamp: Optional[str] = field(default=None)


class HITLEscalationManager:
    """
    Manages human-in-the-loop escalation for customer support.
    
    Triggers escalation when:
        - Confidence score is below threshold
        - No relevant context found
        - Query involves sensitive operations (refunds, account deletion)
        - Complex multi-step requests
        - Explicit customer request for human
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        auto_approve_short_queries: bool = True
    ):
        self.confidence_threshold = confidence_threshold
        self.auto_approve_short_queries = auto_approve_short_queries
        self.escalation_history: list = []

    def should_escalate(
        self,
        query: str,
        confidence: float,
        has_context: bool,
        answer: Optional[str] = None
    ) -> tuple[bool, Optional[EscalationReason]]:
        """
        Determine if a query should be escalated to human review.

        Returns:
            Tuple of (should_escalate, reason).
        """
        # Check for explicit human request
        human_keywords = [
            "speak to human", "talk to agent", "human please",
            "representative", "supervisor", "real person"
        ]
        query_lower = query.lower()
        if any(kw in query_lower for kw in human_keywords):
            return True, EscalationReason.CUSTOMER_REQUEST

        # Check for sensitive operations
        sensitive_keywords = [
            "delete account", "close account", "gdpr", "data deletion",
            "legal", "lawsuit", "attorney", "fraud"
        ]
        if any(kw in query_lower for kw in sensitive_keywords):
            return True, EscalationReason.SENSITIVE_DATA

        # Check for missing context
        if not has_context:
            return True, EscalationReason.MISSING_CONTEXT

        # Check for low confidence
        if confidence < self.confidence_threshold:
            return True, EscalationReason.LOW_CONFIDENCE

        # Check for complex queries (multiple questions)
        question_marks = query.count("?")
        if question_marks > 2 or " and " in query_lower.split("?")[-1]:
            return True, EscalationReason.COMPLEX_QUERY

        # Check for potential policy violations in generated answer
        if answer and self._check_policy_violations(answer):
            return True, EscalationReason.POLICY_VIOLATION

        return False, None

    def _check_policy_violations(self, answer: str) -> bool:
        """Check if answer contains policy violations or unsafe content."""
        unsafe_patterns = [
            "password is", "ssn is", "credit card number",
            "send money to", "click this link to verify"
        ]
        answer_lower = answer.lower()
        return any(pattern in answer_lower for pattern in unsafe_patterns)

    def request_human_review(
        self,
        query: str,
        reason: EscalationReason,
        draft_answer: Optional[str] = None,
        context: Optional[str] = None
    ) -> HITLDecision:
        """
        Present escalation to human moderator and capture decision.
        
        In production, this would send to a ticketing system or
        live agent dashboard. Here we simulate via CLI.
        """
        logger.info(f"🚨 ESCALATION triggered: {reason.value}")
        
        print("\n" + "=" * 70)
        print("🚨 HUMAN-IN-THE-LOOP ESCALATION")
        print("=" * 70)
        print(f"Reason: {reason.value.replace('_', ' ').title()}")
        print(f"\nQuery: {query}")
        
        if context:
            print(f"\nRetrieved Context:\n{context[:500]}...")
        
        if draft_answer:
            print(f"\nDraft AI Answer:\n{draft_answer}")
        
        print("\n" + "-" * 70)
        print("Options:")
        print("  [1] Approve - Send draft answer to customer")
        print("  [2] Reject - Escalate to human agent")
        print("  [3] Modify - Edit and send custom response")
        print("-" * 70)

        while True:
            try:
                choice = input("Select option (1/2/3): ").strip()
                
                if choice == "1":
                    return HITLDecision(
                        status=HITLStatus.APPROVED,
                        human_response=draft_answer,
                        moderator_notes="Approved by moderator"
                    )
                
                elif choice == "2":
                    notes = input("Moderator notes (optional): ").strip()
                    return HITLDecision(
                        status=HITLStatus.REJECTED,
                        moderator_notes=notes or "Escalated to human agent"
                    )
                
                elif choice == "3":
                    print("\nEnter custom response (press Enter twice to finish):")
                    lines = []
                    while True:
                        line = input()
                        if line == "" and lines and lines[-1] == "":
                            break
                        lines.append(line)
                    custom_response = "\n".join(lines).strip()
                    
                    return HITLDecision(
                        status=HITLStatus.MODIFIED,
                        human_response=custom_response,
                        moderator_notes="Modified by moderator"
                    )
                
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
            
            except (KeyboardInterrupt, EOFError):
                print("\nEscalation timed out / cancelled.")
                return HITLDecision(
                    status=HITLStatus.TIMEOUT,
                    moderator_notes="User cancelled / timeout"
                )

    def create_escalation_ticket(
        self,
        query: str,
        reason: EscalationReason,
        decision: HITLDecision
    ) -> dict:
        """
        Create an escalation ticket for record-keeping.
        In production, this would integrate with a ticketing system.
        """
        import datetime
        
        ticket = {
            "ticket_id": f"ESC-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "reason": reason.value,
            "decision_status": decision.status.value,
            "human_response": decision.human_response,
            "moderator_notes": decision.moderator_notes
        }
        self.escalation_history.append(ticket)
        logger.info(f"📝 Created escalation ticket: {ticket['ticket_id']}")
        return ticket

    def get_escalation_stats(self) -> dict:
        """Get statistics on escalations."""
        if not self.escalation_history:
            return {"total": 0}
        
        from collections import Counter
        status_counts = Counter(t["decision_status"] for t in self.escalation_history)
        reason_counts = Counter(t["reason"] for t in self.escalation_history)
        
        return {
            "total": len(self.escalation_history),
            "by_status": dict(status_counts),
            "by_reason": dict(reason_counts)
        }


def create_hitl_manager(**kwargs) -> HITLEscalationManager:
    """Factory function to create HITL manager."""
    return HITLEscalationManager(**kwargs)

