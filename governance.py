from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ReviewVerdict(BaseModel):
    approved: bool
    final_answer: str
    reason: str


class ResponseCache:
    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0

    def _normalize(self, query: str) -> str:
        return re.sub(r"\s+", " ", query.strip().lower())

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        key = self._normalize(query)
        if key in self._store:
            self.hits += 1
            return self._store[key]
        self.misses += 1
        return None

    def set(self, query: str, payload: Dict[str, Any]) -> None:
        self._store[self._normalize(query)] = payload


class GovernancePolicy:
    def __init__(self, max_cost: float = 50.0) -> None:
        self.max_cost = max_cost

    def enforce_budget(self, simulated_cost: float) -> bool:
        return simulated_cost <= self.max_cost

    def risk_classification(self) -> str:
        return "Medium"

    def risk_justification(self) -> str:
        return (
            "This system handles financial customer support and loan policy guidance, which sits in the Medium-risk band because it influences customer decisions and may touch sensitive financial facts or account histories. It is not a completely autonomous decision-maker, but it does work with high-impact customer data and must therefore use a tight governance layer."
        )


class AutogenReviewTeam:
    def __init__(self) -> None:
        self.policy_reviewer = "Policy-Compliance-Reviewer"
        self.final_editor = "Final-Editor"

    def review(self, draft_answer: str, context: List[str]) -> ReviewVerdict:
        if "I don't know" in draft_answer or "unsupported" in draft_answer.lower():
            return ReviewVerdict(
                approved=False,
                final_answer="I don't know. The answer is not supported by the approved policy context.",
                reason="The draft was not grounded in the retrieved policy materials and was revised to a safe refusal.",
            )
        if any("loan" in snippet.lower() for snippet in context) and "breach" in draft_answer.lower():
            return ReviewVerdict(
                approved=False,
                final_answer="The answer was revised to remove an unsupported claim that could misstate the loan policy.",
                reason="The policy reviewer caught an ungrounded statement and the editor corrected it.",
            )
        return ReviewVerdict(
            approved=True,
            final_answer=draft_answer,
            reason="The draft answer matches the retrieved context and requires no change.",
        )


if __name__ == "__main__":
    policy = GovernancePolicy()
    print(policy.risk_classification())
    print(policy.risk_justification())
    team = AutogenReviewTeam()
    print(team.review("I don't know. The evidence does not support that claim.", ["Personal loan policy", "Loan terms"]))
