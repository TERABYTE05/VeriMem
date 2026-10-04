"""Evidence retrieval over the AVeriTeC knowledge store (P0.11, P1.5).

Benchmark runs retrieve from the knowledge store, never from live search (rule 5): the
store is what AVeriTeC scraped for each claim, so results are reproducible and no
fact-checking site can leak the label.

The store ships as scraped pages per claim, each a list of extracted sentences. Those
sentences are grouped into passages of a few sentences, which is the unit the evaluator
grades and the verifier reads -- a single sentence is usually too short to support a
verdict, and a whole page is too long for the context budget.

Two layouts are accepted, because the released dumps differ: a directory of
`<claim_id>.json` / `.jsonl` files, or one JSONL file whose lines carry `claim_id`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Protocol

from experience.format import Evidence
from retrieval.blocklist import is_blocked
from retrieval.bm25 import BM25Index

SENTENCES_PER_PASSAGE = 4


class EvidenceSource(Protocol):
    """What a baseline or the agent needs from an evidence provider."""

    def retrieve(self, claim_id: str, query: str, k: int) -> list[Evidence]: ...


def _passages(sentences: Sequence[str], size: int = SENTENCES_PER_PASSAGE) -> Iterator[str]:
    buf: list[str] = []
    for sentence in sentences:
        sentence = (sentence or "").strip()
        if not sentence:
            continue
        buf.append(sentence)
        if len(buf) >= size:
            yield " ".join(buf)
            buf = []
    if buf:
        yield " ".join(buf)


def _documents(entry: dict[str, Any]) -> Iterator[tuple[str, str]]:
    """(url, passage) pairs from one scraped page."""
    url = entry.get("url") or ""
    text = entry.get("url2text") or entry.get("text") or entry.get("sentences") or []
    if isinstance(text, str):
        text = [text]
    yield from ((url, p) for p in _passages(text))


class KnowledgeStore:
    """Lazily loads and indexes one claim's scraped pages, then BM25-ranks passages."""

    def __init__(self, root: Path | str, apply_blocklist: bool = True) -> None:
        self.root = Path(root)
        self.apply_blocklist = apply_blocklist
        self._cache: dict[str, tuple[BM25Index, dict[str, tuple[str, str]]]] = {}
        self._flat: dict[str, list[dict[str, Any]]] | None = None

    # --- loading -------------------------------------------------------------------

    def _load_flat_file(self) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        if self.root.is_file():
            with self.root.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    grouped.setdefault(str(entry.get("claim_id", "")), []).append(entry)
        return grouped

    def _entries_for(self, claim_id: str) -> list[dict[str, Any]]:
        if self.root.is_file():
            if self._flat is None:
                self._flat = self._load_flat_file()
            return self._flat.get(str(claim_id), [])
        for name in (f"{claim_id}.jsonl", f"{claim_id}.json"):
            path = self.root / name
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                return []
            if text[0] == "[":
                return json.loads(text)
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        return []

    def _index_for(self, claim_id: str):
        if claim_id in self._cache:
            return self._cache[claim_id]
        passages: dict[str, tuple[str, str]] = {}
        for entry in self._entries_for(claim_id):
            for url, passage in _documents(entry):
                if self.apply_blocklist and is_blocked(url):
                    continue  # rule 5: fact-checking domains never reach the model
                passages[f"p{len(passages)}"] = (url, passage)
        index = BM25Index((pid, text) for pid, (_, text) in passages.items())
        self._cache[claim_id] = (index, passages)
        return self._cache[claim_id]

    # --- retrieval -----------------------------------------------------------------

    def has(self, claim_id: str) -> bool:
        return bool(self._entries_for(str(claim_id)))

    def retrieve(self, claim_id: str, query: str, k: int = 5) -> list[Evidence]:
        index, passages = self._index_for(str(claim_id))
        if not len(index):
            return []
        return [
            Evidence(
                url=passages[pid][0],
                text=passages[pid][1],
                grade=round(score, 4),
                retrieved_by="knowledge_store",
            )
            for pid, score in index.search(query, k=k)
        ]


class GoldEvidenceSource:
    """ORACLE upper bound: hands back the annotator's own evidence.

    This is not retrieval and must never be reported as a baseline -- the annotation was
    written knowing the verdict. It exists to separate "the retriever is bad" from "the
    verifier is bad" when a baseline scores poorly.
    """

    def __init__(self, claims: Iterable[Any]) -> None:
        self._by_id = {str(c.id): c for c in claims}

    def retrieve(self, claim_id: str, query: str, k: int = 5) -> list[Evidence]:
        claim = self._by_id.get(str(claim_id))
        if claim is None:
            return []
        return [
            Evidence(url=url, text=text, retrieved_by="gold_annotation")
            for url, text in claim.gold_evidence()[:k]
        ]
