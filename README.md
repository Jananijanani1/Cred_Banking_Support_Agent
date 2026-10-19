# Cred (Banking & FinTech) Support Agent

I completed the Cred (Banking & FinTech) track. This repository contains a production-minded loan-policy support agent built around a deterministic dataset, a policy knowledge base, a dual-collection retrieval layer, a CrewAI-style orchestration with lookup and guardrails, an evaluation harness, and a FastAPI deployment.

The project runs in a zero-network, MOCK_LLM mode by design. The telemetry setting is enforced before crew execution:

`CREWAI_DISABLE_TELEMETRY=true`

## Dataset design choices

- Seed: `42`
- Categories used: Personal Loan, Home Loan, Auto Loan, Education Loan, Business Loan
- Status values used: Submitted, Under Review, Approved, Rejected, Disbursed
- Total records generated: `50`
- Fraud-flag percentage: `26.0%` (within the required 10-30% band)
- Loan ranges reflect realistic Indian lending sizes and were deliberately chosen so the synthetic data remains plausible without hand-editing individual records.
  - Personal Loan: INR 1.5 lakh to INR 15 lakh
  - Home Loan: INR 5 lakh to INR 80 lakh
  - Auto Loan: INR 2 lakh to INR 25 lakh
  - Education Loan: INR 1 lakh to INR 25 lakh
  - Business Loan: INR 3 lakh to INR 120 lakh

The dataset generator prints the category/status counts and validates the fraud-review percentage before finishing.

## Knowledge base coverage

The repository includes 15 policy documents covering the required topics:

- loan eligibility criteria by loan type
- EMI calculation rules
- credit-card fee structure
- KYC document requirements
- fraud-dispute resolution
- account-closure process
- interest-rate slabs
- prepayment-penalty rules
- minimum-balance requirements
- credit-score impact factors
- joint-account rules
- NRI-account eligibility

## Grounded generation threshold

The offline retrieval layer was calibrated with in-scope and out-of-scope prompts to separate relevant policy queries from unrelated ones.

Representative observed top-1 similarities:

- In-scope: `0.2121`, `0.1644`, `0.3904`, `0.1826`, `0.2561`, `0.1826`
- Out-of-scope: `0.0`, `0.0`
- Chosen threshold: `0.10`

The `RetrievalSystem` measures those calibration queries at startup and uses the measured threshold, rather than a separate hard-coded value. In-scope queries pass when matching context is present, while unrelated prompts trigger the fallback answer. Grounded answers use the retrieved passage that best matches the query terms, avoiding truncation of relevant evidence by unrelated passages.

## Task 13 evaluation

Run the 15-query, zero-API-key evaluation on Python 3.14.3 with:

`py -3.14 evaluation.py`

The script was verified on Python 3.14.3 and also runs in the project's existing virtual environment. It evaluates validated `CrewHarness` responses under `MOCK_LLM=true`. Its test set covers personal- and home-loan eligibility and each of the other 11 required knowledge-base topics, plus an out-of-scope weather question and a prompt-injection/privacy edge case. For every case, the judge prompt includes the query, required reference facts, retrieved context, and crew answer. The deterministic MOCK_LLM judge applies the prompt's 1-5 rubric using reference-fact coverage, answer-to-context support, refusal behavior, and checks for sensitive data or unsupported guarantees; it makes no external model calls.

The console report prints the exact query and all four named scores for each case, followed by the arithmetic mean of **Accuracy**, **Grounding**, **Completeness**, and **Safety** across all 15 cases. A fresh run of `py -3.14 evaluation.py` produced the results below. These are the deterministic MOCK_LLM judge's rubric scores, not ratings from an independent human or external model:

| # | Evaluation topic | Accuracy | Grounding | Completeness | Safety |
|---:|---|---:|---:|---:|---:|
| 1 | Personal-loan eligibility | 5 | 5 | 5 | 5 |
| 2 | Home-loan eligibility | 5 | 5 | 5 | 5 |
| 3 | EMI calculation | 5 | 5 | 5 | 5 |
| 4 | Credit-card fees | 5 | 5 | 5 | 5 |
| 5 | KYC documents | 5 | 5 | 5 | 5 |
| 6 | Fraud-dispute resolution | 5 | 5 | 5 | 5 |
| 7 | Account closure | 5 | 5 | 5 | 5 |
| 8 | Interest-rate slabs | 5 | 5 | 5 | 5 |
| 9 | Prepayment penalties | 5 | 5 | 5 | 5 |
| 10 | Minimum balance | 5 | 5 | 5 | 5 |
| 11 | Credit-score factors | 5 | 5 | 5 | 5 |
| 12 | Joint-account rules | 5 | 5 | 5 | 5 |
| 13 | NRI-account eligibility | 5 | 5 | 5 | 5 |
| 14 | Out-of-scope weather request | 5 | 5 | 5 | 5 |
| 15 | Prompt-injection and privacy edge case | 5 | 5 | 5 | 5 |
| **Average** | **All 15 queries** | **5.00** | **5.00** | **5.00** | **5.00** |

## Project structure

- `dataset.py` — deterministic generator for the loan-application dataset
- `knowledge_base.py` — knowledge-base documents and required topics
- `rag_core.py` — chunking, retrieval, scoring, threshold calibration, and evaluation
- `evaluation.py` — 15-query Accuracy, Grounding, Completeness, and Safety evaluation
- `crew_agent.py` — `check_loan_application_status(record_id)`, lookup/retrieval tools, guardrails, and orchestration demo
- `governance.py` — review stage, response cache, and policy governance
- `api.py` — FastAPI deployment with the required endpoints
- `requirements.txt` — dependencies for the project

## ChromaDB storage

The retrieval collections are stored locally in `chroma_db/` by default. Set
`CHROMA_DB_PATH` to use a different directory. The database uses the project's
local embedding model, so indexing and queries do not need to download an
embedding model.

## How to run

1. Create and activate a virtual environment:

   `python -m venv .venv`
   `.\.venv\Scripts\activate`
   `pip install -r requirements.txt`

2. Generate and inspect the dataset:

   `python dataset.py`

3. Run the retrieval and evaluation harness:

   `python rag_core.py`

4. Run the 15-query quality/safety evaluation and four-score summary on Python 3.14.3:

   `py -3.14 evaluation.py`

5. Run the crew orchestration demo and memory examples:

   `python crew_agent.py`

6. Start the API server:

   `uvicorn api:app --host 0.0.0.0 --port 8000`

7. Example HTTP calls:

   `curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"query":"What are the home-loan eligibility checks?"}'`

   `curl -X POST http://localhost:8000/add-document -H "Content-Type: application/json" -d '{"title":"Account closure","content":"Account closure requires dues settlement and a final balance check before closure is approved."}'`

8. WebSocket chat endpoint:

   `ws://localhost:8000/ws`

## Guardrails and governance

- Input-side PII masking for PAN, Aadhaar, and account-number patterns
- Prompt-injection detection for override-style attempts
- Output-side groundedness refusal when the retrieved context is insufficient
- Least-autonomy enforcement: only the lookup agent is wired to the loan-status function
- Response caching keyed by normalized query text
- Structured review passes and in-process conversation memory

**Risk classification: High.** Under the brief's Low/Medium/High scheme, customer-support guidance alone would fall under Medium, but this system also processes financial data: loan-application records, application statuses, amounts, and fraud-review indicators. Those records could influence financial servicing or escalation decisions if used operationally, so the High-risk financial-data category applies. The repository uses synthetic records and its policy answers are intended as support, not binding lending decisions; these safeguards reduce exposure but do not change the system's risk classification.

## Notes

- The project is designed to run without external API keys in MOCK_LLM mode.
- Retrieval and orchestration use local offline logic so that the workflow remains deterministic and zero-network.
- Request logs apply the same PII masking used in the runtime request path before writing structured JSON lines.
