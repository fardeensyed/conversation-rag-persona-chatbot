from __future__ import annotations

import math
from collections import Counter, defaultdict

from .utils import tokenize


class TfidfIndex:
    def __init__(self) -> None:
        self.documents: list[dict] = []
        self.idf: dict[str, float] = {}
        self.vectors: list[dict[str, float]] = []
        self.norms: list[float] = []

    @staticmethod
    def _term_counts(text: str) -> Counter[str]:
        return Counter(tokenize(text))

    def fit(self, documents: list[dict]) -> "TfidfIndex":
        self.documents = documents
        df: Counter[str] = Counter()
        term_counts = []
        for doc in documents:
            counts = self._term_counts(doc["text"])
            term_counts.append(counts)
            df.update(counts.keys())

        total = max(len(documents), 1)
        self.idf = {term: math.log((1 + total) / (1 + freq)) + 1 for term, freq in df.items()}
        self.vectors = [self._to_vector(counts) for counts in term_counts]
        self.norms = [self._norm(vector) for vector in self.vectors]
        return self

    def _to_vector(self, counts: Counter[str]) -> dict[str, float]:
        if not counts:
            return {}
        max_count = max(counts.values())
        return {term: (count / max_count) * self.idf.get(term, 1.0) for term, count in counts.items()}

    @staticmethod
    def _norm(vector: dict[str, float]) -> float:
        return math.sqrt(sum(value * value for value in vector.values()))

    @staticmethod
    def _cosine(a: dict[str, float], a_norm: float, b: dict[str, float], b_norm: float) -> float:
        if not a_norm or not b_norm:
            return 0.0
        if len(a) > len(b):
            a, b = b, a
        dot = sum(value * b.get(term, 0.0) for term, value in a.items())
        return dot / (a_norm * b_norm)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        q_counts = self._term_counts(query)
        q_vec = self._to_vector(q_counts)
        q_norm = self._norm(q_vec)
        scored = []
        for idx, vector in enumerate(self.vectors):
            score = self._cosine(q_vec, q_norm, vector, self.norms[idx])
            if score > 0:
                result = dict(self.documents[idx])
                result["score"] = round(score, 4)
                scored.append(result)
        return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]

    def to_dict(self) -> dict:
        return {
            "documents": self.documents,
            "idf": self.idf,
            "vectors": self.vectors,
            "norms": self.norms,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TfidfIndex":
        index = cls()
        index.documents = payload["documents"]
        index.idf = payload["idf"]
        index.vectors = payload["vectors"]
        index.norms = payload["norms"]
        return index


def cosine_for_texts(window_texts: list[str], new_text: str) -> float:
    texts = window_texts + [new_text]
    counts = [Counter(tokenize(text)) for text in texts]
    vocab = sorted({term for counter in counts for term in counter})
    if not vocab:
        return 1.0
    df = defaultdict(int)
    for counter in counts:
        for term in counter:
            df[term] += 1
    total = len(texts)
    idf = {term: math.log((1 + total) / (1 + df[term])) + 1 for term in vocab}

    def vec(counter: Counter[str]) -> dict[str, float]:
        max_count = max(counter.values()) if counter else 1
        return {term: (counter[term] / max_count) * idf[term] for term in counter}

    vectors = [vec(counter) for counter in counts]
    window_vec = defaultdict(float)
    for vector in vectors[:-1]:
        for term, value in vector.items():
            window_vec[term] += value / max(len(vectors) - 1, 1)
    return TfidfIndex._cosine(dict(window_vec), TfidfIndex._norm(window_vec), vectors[-1], TfidfIndex._norm(vectors[-1]))
