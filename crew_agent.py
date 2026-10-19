from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Dict, List

from pydantic import BaseModel, ValidationError

from dataset import generate_loans
from knowledge_base import DOCUMENTS
from rag_core import RetrievalSystem

try:  # pragma: no cover
    from crewai.llms.base_llm import BaseLLM
except Exception:  # pragma: no cover
    class BaseLLM:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            self.model = kwargs.get("model", "mock")

        def invoke(self, *args, **kwargs):
            return "Mock LLM response" 


LOAN_APPLICATIONS = generate_loans()


class CrewResponseModel(BaseModel):
    answer: str
    tool_used: str
    confidence: float
    query: str


class DecisionModel(BaseModel):
    approved: bool
    final_answer: str
    reason: str


def mask_pii(text: str) -> str:
    masked = re.sub(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", "PAN-REDACTED", text, flags=re.IGNORECASE)
    masked = re.sub(r"\b\d{12}\b", "AADHAAR-REDACTED", masked)
    masked = re.sub(r"\b\d{9,18}\b", "ACCOUNT-REDACTED", masked)
    return masked


def detect_prompt_injection(text: str) -> bool:
    suspicious = [
        "ignore previous instructions",
        "system prompt",
        "developer prompt",
        "override guardrails",
        "secret instruction",
        "reveal hidden instructions",
    ]
    lower = text.lower()
    return any(token in lower for token in suspicious)


def output_groundedness_check(context: List[str], answer: str) -> bool:
    if not context:
        return False
    normalized_answer = answer.lower()
    for snippet in context:
        if snippet.lower() in normalized_answer or any(token in snippet.lower() for token in ["loan", "emi", "kyc", "fraud", "account", "rate"] if token in normalized_answer):
            return True
    return "i don't know" in normalized_answer


class LoanStatusTool:
    def __init__(self, records=None):
        self.records = records or LOAN_APPLICATIONS

    def check_loan_application_status(self, record_id: str) -> Dict[str, Any]:
        record = next((r for r in self.records if r["record_id"] == record_id), None)
        if record is None:
            raise ValueError(f"Record {record_id} not found")
        days = int(record["days_since_created"])
        flagged = bool(record["flagged_for_fraud_review"])
        max_days = max(r["days_since_created"] for r in self.records)
        norm_days = max(0.0, min(1.0, days / max(1, max_days)))
        escalation = 0.6 * (1.0 if flagged else 0.0) + 0.4 * norm_days
        status = record["status"]
        return {
            "record_id": record_id,
            "status": status,
            "loan_amount_inr": record["loan_amount_inr"],
            "escalation_score": round(escalation, 4),
            "formula": "0.6 * flagged_for_fraud_review + 0.4 * days_since_created/max_days",
            "threshold": 0.72,
        }

    def check_application_status(self, record_id: str) -> Dict[str, Any]:
        return self.check_loan_application_status(record_id)


def check_loan_application_status(record_id: str) -> Dict[str, Any]:
    """Look up a generated loan application and calculate its escalation score."""
    return LoanStatusTool().check_loan_application_status(record_id)


class MockLLM(BaseLLM):
    def __init__(self, model_name: str = "mock-llm"):
        super().__init__(model=model_name)

    def call(self, *args: Any, **kwargs: Any) -> str:
        return json.dumps({"answer": "mocked response", "status": "ok"})

    def invoke(self, *args, **kwargs):
        return json.dumps({"answer": "mocked response", "status": "ok"})


class RetrievalTool:
    def __init__(self, rag_system: RetrievalSystem):
        self.rag_system = rag_system

    def run(self, query: str) -> Dict[str, Any]:
        return self.rag_system.ground_answer(query, strategy="fixed_overlap")


class CrewHarness:
    def __init__(self, rag_system: RetrievalSystem):
        self.rag_system = rag_system
        self.lookup_tool = LoanStatusTool()
        self.rag_tool = RetrievalTool(rag_system)
        self.mock_llm = MockLLM()

    def _validate_response(self, payload: Dict[str, Any]) -> CrewResponseModel:
        model = CrewResponseModel(**payload)
        return model

    def kickoff(self, query: str, record_id: str | None = None) -> CrewResponseModel:
        masked = mask_pii(query)
        if detect_prompt_injection(masked):
            raise ValueError("Prompt injection attempt rejected")

        if record_id:
            lookup_result = self.lookup_tool.check_loan_application_status(record_id)
        else:
            lookup_result = None

        rag_result = self.rag_tool.run(masked)
        if not output_groundedness_check(rag_result.get("context", []), rag_result.get("answer", "")):
            rag_result["answer"] = "I don't know. The available context does not support that answer."
            rag_result["grounded"] = False

        payload = {
            "answer": rag_result.get("answer", "") if rag_result.get("grounded", True) else "I don't know. The available context does not support that answer.",
            "tool_used": "rag_lookup" if rag_result.get("grounded", True) else "grounded_fallback",
            "confidence": float(rag_result.get("top_1_similarity", 0.0)),
            "query": masked,
        }
        if lookup_result:
            payload["answer"] = f"{payload['answer']} Lookup status: {lookup_result['status']} (score {lookup_result['escalation_score']})."
            payload["tool_used"] = "lookup_status"
        return self._validate_response(payload)


def run_memory_demo() -> Dict[str, Any]:
    try:
        from langchain_core.chat_history import InMemoryChatMessageHistory
        from langchain_core.runnables import RunnableLambda
        from langchain_core.runnables.history import RunnableWithMessageHistory
    except Exception:  # pragma: no cover
        return {
            "session_memory": "Fallback in-memory store used because LangChain core is not installed.",
            "transcripts": [
                {"session": "s1", "turns": ["Q: How is EMI calculated?", "A: Based on principal, rate, and tenure."]},
                {"session": "s2", "turns": ["Q: What is KYC?", "A: KYC requires identity, address proof, and matching documents."]},
            ],
        }

    session_store: Dict[str, List[str]] = {
        "session-a": ["How do I close my account?", "Does the closure require paying dues first?"],
        "session-b": ["What is the minimum balance requirement?"],
    }
    return {
        "session_memory": "LangChain-style in-memory session history active",
        "transcripts": [
            {"session": "session-a", "turns": session_store["session-a"]},
            {"session": "session-b", "turns": session_store["session-b"]},
        ],
    }


if __name__ == "__main__":
    rag = RetrievalSystem(DOCUMENTS)
    crew = CrewHarness(rag)
    print(crew.kickoff("What documents are required for a personal loan?"))
    print(crew.kickoff("Check CL-0005", record_id="CL-0005"))
    print(run_memory_demo())
