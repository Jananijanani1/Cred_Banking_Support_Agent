from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("CHROMA_TELEMETRY", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import chromadb
import numpy as np

from dataset import generate_loans
from knowledge_base import DOCUMENTS


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _overlap_tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[A-Za-z0-9]+", text.lower()))
    normalized: set[str] = set()
    for token in tokens:
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and not token.endswith(("ss", "us", "is")) and len(token) > 3:
            token = token[:-1]
        normalized.add(token)
    return normalized


@dataclass
class Chunk:
    doc_id: str
    doc_title: str
    text: str
    chunk_index: int = 0


STOP_WORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is",
    "are", "it", "this", "that", "as", "by", "be", "if", "from", "at", "into",
    "we", "you", "your", "our", "their", "they", "them", "when", "what", "how",
    "who", "why", "where", "does", "do", "did", "not", "can", "should", "may",
    "will", "about", "have", "has", "had", "been", "more", "most", "then", "than",
    "also", "each", "such", "all", "any", "some", "there", "its", "his", "her",
}


class SimpleEmbeddingModel:
    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {}

    def _ensure_vocab(self, texts: Iterable[str], allow_new: bool = True) -> None:
        tokens: set[str] = set()
        for text in texts:
            for token in re.findall(r"[A-Za-z0-9]+", text.lower()):
                if token not in STOP_WORDS and len(token) > 2:
                    tokens.add(token)
        for token in sorted(tokens):
            if allow_new or token in self.vocab or not self.vocab:
                self.vocab.setdefault(token, len(self.vocab))

    def encode(self, texts: Iterable[str], allow_new: bool = False) -> np.ndarray:
        text_list = list(texts)
        if not self.vocab:
            self._ensure_vocab(text_list, allow_new=True)
        elif allow_new:
            self._ensure_vocab(text_list, allow_new=True)
        vectors = []
        for text in text_list:
            token_counts: Dict[str, int] = {}
            for token in re.findall(r"[A-Za-z0-9]+", text.lower()):
                if token in STOP_WORDS or len(token) <= 2:
                    continue
                token_counts[token] = token_counts.get(token, 0) + 1
            vector = np.zeros(len(self.vocab), dtype=float)
            for token, count in token_counts.items():
                if token in self.vocab:
                    vector[self.vocab[token]] = count
            norm = np.linalg.norm(vector)
            if norm:
                vector = vector / norm
            vectors.append(vector)
        return np.vstack(vectors)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def sentence_chunks(documents: List[Dict[str, str]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for doc in documents:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", doc["content"]) if s.strip()]
        for i, sentence in enumerate(sentences):
            chunks.append(Chunk(doc_id=doc["id"], doc_title=doc["title"], text=sentence, chunk_index=i))
    return chunks


def fixed_overlap_chunks(documents: List[Dict[str, str]], size: int = 120, overlap: int = 30) -> List[Chunk]:
    chunks: List[Chunk] = []
    for doc in documents:
        words = doc["content"].split()
        step = size - overlap
        for i in range(0, len(words), step):
            snippet = " ".join(words[i:i + size])
            if not snippet:
                continue
            chunks.append(Chunk(doc_id=doc["id"], doc_title=doc["title"], text=snippet, chunk_index=i // step))
            if i + size >= len(words):
                break
    return chunks


class MemoryCollection:
    def __init__(self, name: str) -> None:
        self.name = name
        self._data: List[Dict[str, object]] = []
        self.model = SimpleEmbeddingModel()

    def upsert(self, documents: List[Dict[str, object]]) -> None:
        self._data = documents
        if self._data:
            self.model = SimpleEmbeddingModel()
            self.model.encode([item["text"] for item in self._data], allow_new=True)

    def query(self, query: str, k: int = 3) -> List[Dict[str, object]]:
        if not self._data:
            return []
        texts = [item["text"] for item in self._data]
        vectors = self.model.encode(texts, allow_new=False)
        q_vec = self.model.encode([query], allow_new=False)[0]
        scores = [(i, cosine_similarity(q_vec, v)) for i, v in enumerate(vectors)]
        sorted_scores = sorted(scores, key=lambda item: item[1], reverse=True)[:k]
        results = []
        for idx, score in sorted_scores:
            item = dict(self._data[idx])
            item["score"] = round(score, 4)
            results.append(item)
        return results


class ChromaCollectionAdapter:
    def __init__(self, name: str, client: chromadb.ClientAPI) -> None:
        self.name = name
        self.client = client
        self.collection = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
        self.model = SimpleEmbeddingModel()

    def upsert(self, documents: List[Dict[str, object]]) -> None:
        ids = [str(item["id"]) for item in documents]
        texts = [str(item["text"]) for item in documents]
        embeddings = self.model.encode(texts, allow_new=True)

        if self.collection.count():
            existing = self.collection.get(limit=1, include=["embeddings"])
            if len(existing["embeddings"][0]) != embeddings.shape[1]:
                self.client.delete_collection(self.name)
                self.collection = self.client.create_collection(
                    name=self.name,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=None,
                )

        existing_ids = self.collection.get()["ids"]
        stale_ids = set(existing_ids) - set(ids)
        if stale_ids:
            self.collection.delete(ids=list(stale_ids))

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[{"doc_id": item["doc_id"], "doc_title": item["doc_title"]} for item in documents],
        )

    def query(self, query: str, k: int = 3, include_metadata: bool = True) -> List[Dict[str, object]]:
        query_embedding = self.model.encode([query], allow_new=False)[0].tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=k)
        hits = []
        for idx, text in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][idx]
            distance = results["distances"][0][idx]
            hits.append({"id": results["ids"][0][idx], "text": text, "doc_id": meta.get("doc_id"), "doc_title": meta.get("doc_title"), "score": 1.0 - distance})
        return hits


class RetrievalSystem:
    def __init__(self, documents: List[Dict[str, str]]):
        self.documents = documents
        self.fixed_chunks = fixed_overlap_chunks(documents)
        self.sentence_chunks = sentence_chunks(documents)
        db_path = os.environ.get(
            "CHROMA_DB_PATH",
            os.path.join(os.path.dirname(__file__), "chroma_db"),
        )
        client = chromadb.PersistentClient(path=db_path)
        self.collections = {
            "fixed_overlap": ChromaCollectionAdapter("fixed_overlap", client),
            "sentence": ChromaCollectionAdapter("sentence", client),
        }
        self._load_documents()
        self.threshold = 0.0
        self.calibration = self.measure_threshold()
        self.threshold = float(self.calibration["threshold"])

    def _load_documents(self) -> None:
        for name, chunks in [("fixed_overlap", self.fixed_chunks), ("sentence", self.sentence_chunks)]:
            items = [{
                "id": f"{name}-{idx}",
                "doc_id": chunk.doc_id,
                "doc_title": chunk.doc_title,
                "text": chunk.text,
            } for idx, chunk in enumerate(chunks)]
            self.collections[name].upsert(items)

    def retrieve(self, query: str, strategy: str = "fixed_overlap", top_k: int = 3) -> List[Dict[str, object]]:
        return self.collections[strategy].query(query, k=top_k)

    def ground_answer(self, query: str, strategy: str = "fixed_overlap") -> Dict[str, object]:
        hits = self.retrieve(query, strategy=strategy, top_k=3)
        if not hits:
            return {"answer": "I don't know. I could not find supporting context in the approved knowledge base.", "context": [], "top_1_similarity": 0.0, "grounded": False}
        true_scores = [float(hit.get("score", 0.0)) for hit in hits]
        top_1_similarity = max(true_scores) if true_scores else 0.0
        if top_1_similarity < self.threshold:
            return {"answer": "I don't know. The information requested is not supported by the available policy context.", "context": [hit["text"] for hit in hits], "top_1_similarity": top_1_similarity, "grounded": False}

        context = "\n".join(hit["text"] for hit in hits)
        query_terms = _overlap_tokens(query) - STOP_WORDS
        answer_hit = max(
            hits,
            key=lambda hit: (
                len(
                    query_terms
                    & _overlap_tokens(
                        f"{hit.get('doc_title', '')} {hit['text']}"
                    )
                ),
                float(hit.get("score", 0.0)),
            ),
        )
        answer = "Based on the available policy context: " + str(answer_hit["text"])
        return {"answer": answer, "context": [hit["text"] for hit in hits], "top_1_similarity": top_1_similarity, "grounded": True}

    def measure_threshold(self) -> Dict[str, object]:
        in_scope_queries = [
            "What documents do I need for a personal loan application?",
            "How is EMI calculated for a 7 year home loan?",
            "Explain the minimum balance rules for this account.",
            "What are the requirements for KYC when opening a loan?",
            "What is the process if a credit-card payment is disputed?",
            "What is the penalty if I prepay my personal loan early?",
        ]
        out_of_scope_queries = [
            "What is a weather forecast for Delhi next week?",
            "Write a sonnet about a moonlit forest.",
        ]
        in_scores = []
        for query in in_scope_queries:
            hits = self.retrieve(query, strategy="fixed_overlap", top_k=1)
            sim = float(hits[0]["score"]) if hits else 0.0
            in_scores.append(sim)
        out_scores = []
        for query in out_of_scope_queries:
            hits = self.retrieve(query, strategy="fixed_overlap", top_k=1)
            sim = float(hits[0]["score"]) if hits else 0.0
            out_scores.append(sim)

        # Choose a threshold between the observed in-scope and out-of-scope clusters.
        threshold = float((min(in_scores) + max(out_scores)) / 2.0)
        threshold = min(0.9, max(0.1, threshold))
        return {"in_scope": [round(v, 4) for v in in_scores], "out_of_scope": [round(v, 4) for v in out_scores], "threshold": round(threshold, 4)}

    def evaluate_precision_recall(self, queries: List[Tuple[str, List[str]]]) -> Dict[str, object]:
        results = {}
        for strategy in ["fixed_overlap", "sentence"]:
            per_query = []
            for question, relevant_ids in queries:
                hits = self.retrieve(question, strategy=strategy, top_k=5)
                retrieved_ids = []
                for hit in hits:
                    doc_id = str(hit.get("doc_id") or "")
                    if doc_id:
                        retrieved_ids.append(doc_id)
                retrieved_ids = list(dict.fromkeys(retrieved_ids))
                relevant = list(dict.fromkeys(relevant_ids))
                tp = len(set(retrieved_ids).intersection(relevant))
                precision = tp / len(retrieved_ids) if retrieved_ids else 0.0
                recall = tp / len(relevant) if relevant else 0.0
                per_query.append({"question": question, "precision": round(precision, 4), "recall": round(recall, 4), "retrieved": retrieved_ids, "relevant": relevant})
            results[strategy] = per_query
        return results


def build_demo_queries() -> List[Tuple[str, List[str]]]:
    return [
        ("What are the standard requirements for a personal loan?", ["eligibility_personal_loan"]),
        ("How is EMI calculated and which variables matter?", ["emi_rules"]),
        ("What KYC documents are required for a bank account?", ["kyc_documents"]),
        ("How is a fraud dispute resolved for a suspicious transaction?", ["fraud_dispute"]),
        ("What is the process for closing an account and settling dues?", ["account_closure"]),
    ]


if __name__ == "__main__":
    system = RetrievalSystem(DOCUMENTS)
    metrics = system.measure_threshold()
    print("Threshold metrics:", metrics)
    demo_queries = build_demo_queries()
    print("Evaluation:", system.evaluate_precision_recall(demo_queries))
    print("Sample answer:", system.ground_answer("What are the rules for EMI calculation?", strategy="fixed_overlap"))
    print("Flagged fraud record count:", sum(1 for r in generate_loans() if r["flagged_for_fraud_review"]))
