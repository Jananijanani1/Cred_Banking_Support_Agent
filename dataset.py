from __future__ import annotations

import json
import random
from collections import Counter
from typing import Dict, List

CATEGORIES = [
    "Personal Loan",
    "Home Loan",
    "Auto Loan",
    "Education Loan",
    "Business Loan",
]

STATUSES = ["Submitted", "Under Review", "Approved", "Rejected", "Disbursed"]


def _loan_amount_by_category(category: str) -> tuple[int, int]:
    ranges = {
        "Personal Loan": (150000, 1500000),
        "Home Loan": (500000, 8000000),
        "Auto Loan": (200000, 2500000),
        "Education Loan": (100000, 2500000),
        "Business Loan": (300000, 12000000),
    }
    return ranges[category]


def _status_for_index(index: int) -> str:
    # Deterministic cycle with a slight weighting toward submitted/reviewed and approved
    weighted = [
        "Submitted",
        "Under Review",
        "Approved",
        "Submitted",
        "Under Review",
        "Disbursed",
        "Rejected",
    ]
    return weighted[index % len(weighted)]


def generate_loans(seed: int = 42, count: int = 50) -> List[Dict[str, object]]:
    rnd = random.Random(seed)
    records: List[Dict[str, object]] = []

    category_sequence = (CATEGORIES * ((count // len(CATEGORIES)) + 1))[:count]
    for idx in range(count):
        category = category_sequence[idx]
        lower, upper = _loan_amount_by_category(category)
        amount = rnd.randint(lower, upper)
        status = _status_for_index(idx)
        days = rnd.randint(0, 30)
        is_flagged = ((idx + 2) % 5 == 0) or ((idx + 1) % 11 == 0)
        records.append(
            {
                "record_id": f"CL-{idx + 1:04d}",
                "category": category,
                "status": status,
                "loan_amount_inr": amount,
                "days_since_created": days,
                "flagged_for_fraud_review": bool(is_flagged),
            }
        )

    flagged_count = sum(1 for r in records if r["flagged_for_fraud_review"])
    if not 10 <= (flagged_count / len(records)) * 100 <= 30:
        raise ValueError("Generated fraud flag percentage is outside the required 10-30% window.")

    return records


def summarize_records(records: List[Dict[str, object]]) -> Dict[str, object]:
    category_counts = Counter(r["category"] for r in records)
    status_counts = Counter(r["status"] for r in records)
    flagged_pct = (sum(1 for r in records if r["flagged_for_fraud_review"]) / len(records)) * 100

    summary = {
        "total_records": len(records),
        "category_counts": dict(category_counts),
        "status_counts": dict(status_counts),
        "flagged_percentage": round(flagged_pct, 2),
    }
    return summary


def print_summary(records: List[Dict[str, object]]) -> None:
    summary = summarize_records(records)
    print("Loan application summary")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    records = generate_loans()
    print_summary(records)
