"""Neighbour lookup for building retrieval-augmented SFT data.

P2.6 builds the real FAISS store over sentence-transformer embeddings. Until then this
lexical retriever lets the training-data pipeline be written, tested and smoke-trained
without torch, so it runs on the Mac. Swap in the FAISS store through the same
`Retriever` signature once it exists -- nothing downstream changes.

Rule 5: a trajectory is never retrieved into its own training example. Exclusion is by
id, by dataset claim id, and by exact claim text, because the same claim can appear
under more than one trajectory id (e.g. a re-run, or a poisoned twin).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from experience.format import Trajectory
from retrieval.bm25 import BM25Index

Retriever = Callable[[Trajectory, int], list[Trajectory]]


def is_self(query: Trajectory, candidate: Trajectory) -> bool:
    """True when `candidate` must be excluded from `query`'s retrieval (rule 5)."""
    if candidate.id == query.id:
        return True
    if query.claim_id is not None and candidate.claim_id == query.claim_id:
        return True
    return candidate.claim.strip().lower() == query.claim.strip().lower()


class LexicalRetriever:
    """BM25 over claim text. A stand-in for the FAISS store (P2.6)."""

    def __init__(self, pool: Sequence[Trajectory]) -> None:
        self.pool = list(pool)
        self._by_id = {t.id: t for t in self.pool}
        self._index = BM25Index((t.id, t.claim) for t in self.pool)

    def retrieve(self, query: Trajectory, k: int) -> list[Trajectory]:
        """Top-k neighbours of `query`, with `query` itself excluded (rule 5)."""
        if k <= 0:
            return []
        hits = self._index.search(
            query.claim,
            k=k,
            exclude=lambda doc_id: is_self(query, self._by_id[doc_id]),
        )
        return [self._by_id[doc_id] for doc_id, _ in hits]
