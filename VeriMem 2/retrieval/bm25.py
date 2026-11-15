"""BM25 over a fixed document set. Pure Python, no torch -- runs on the Mac.

Used in two places: ranking evidence passages against a claim (the plain RAG baseline),
and ranking past trajectories against a new one (retrieval-augmented SFT, until the FAISS
store lands in P2.6). Both want the same scoring, so it lives here rather than twice.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence

_WORD = re.compile(r"[a-z0-9']+")

# Common words carry no ranking signal and swamp the score on short queries.
STOPWORDS: frozenset[str] = frozenset(
    """a an and are as at be been by for from has have in is it its of on or that the
    to was were will with this these those there their they he she his her not no had
    but if then than so such can could would should may might must do does did done""".split()
)


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS and len(w) > 1]


class BM25Index:
    """Okapi BM25. Documents are (id, text) pairs; ids are returned by `search`."""

    def __init__(
        self,
        documents: Iterable[tuple[str, str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.ids: list[str] = []
        self._docs: list[Counter[str]] = []
        self._lengths: list[int] = []
        for doc_id, text in documents:
            counts = Counter(tokenize(text))
            self.ids.append(doc_id)
            self._docs.append(counts)
            self._lengths.append(sum(counts.values()))

        self.k1, self.b = k1, b
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        df: Counter[str] = Counter()
        for doc in self._docs:
            df.update(doc.keys())
        n = len(self._docs)
        self._idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def __len__(self) -> int:
        return len(self.ids)

    def score(self, query_terms: Sequence[str], index: int) -> float:
        doc, length = self._docs[index], self._lengths[index]
        if not length:
            return 0.0
        norm = self.k1 * (1 - self.b + self.b * length / (self._avg_len or 1))
        return sum(
            self._idf.get(term, 0.0) * (doc[term] * (self.k1 + 1)) / (doc[term] + norm)
            for term in set(query_terms)
            if term in doc
        )

    def search(
        self,
        query: str,
        k: int = 5,
        exclude: Callable[[str], bool] | None = None,
        min_score: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Top-k (id, score), best first. Ties break on insertion order, for determinism."""
        if k <= 0 or not self.ids:
            return []
        terms = tokenize(query)
        scored = [
            (self.score(terms, i), i)
            for i in range(len(self.ids))
            if exclude is None or not exclude(self.ids[i])
        ]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [(self.ids[i], s) for s, i in scored[:k] if s > min_score]
