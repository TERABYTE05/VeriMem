"""Load AVeriTeC claims and normalise their labels (P0.8, P0.9).

AVeriTeC's label strings are not the four verdicts the rest of this project uses, so the
mapping happens exactly once, here, and `Claim.gold_label` is always one of
`experience.format.VERDICTS`. Anything downstream comparing against a raw AVeriTeC string
is a bug.

Each AVeriTeC entry carries gold `questions` with answers and source URLs. That is the
annotation, not retrieval output -- using it as evidence is an oracle upper bound, never
a baseline (rule 5). `gold_evidence()` exists for that upper-bound row and is labelled
accordingly wherever it is called.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experience.format import VERDICTS

# AVeriTeC's four labels -> ours. Lowercased keys, so casing drift does not matter.
LABEL_MAP: dict[str, str] = {
    "supported": "Supported",
    "refuted": "Refuted",
    "not enough evidence": "Not enough evidence",
    "conflicting evidence/cherrypicking": "Conflicting evidence",
    # seen in some dumps and in FEVER-style exports
    "nei": "Not enough evidence",
    "conflicting evidence": "Conflicting evidence",
    "cherrypicking": "Conflicting evidence",
}


def normalise_label(raw: str | None) -> str | None:
    if raw is None:
        return None
    key = " ".join(str(raw).strip().lower().split())
    if key in LABEL_MAP:
        return LABEL_MAP[key]
    for verdict in VERDICTS:
        if key == verdict.lower():
            return verdict
    raise ValueError(
        f"Unknown AVeriTeC label {raw!r}. Add it to eval.datasets.LABEL_MAP rather than "
        "letting it through -- an unmapped label silently breaks every metric."
    )


@dataclass
class Claim:
    id: str
    claim: str
    gold_label: str | None = None
    claim_date: str | None = None
    speaker: str | None = None
    questions: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def gold_evidence(self) -> list[tuple[str, str]]:
        """(source_url, answer_text) from the annotation. Oracle only -- see module docstring."""
        out = []
        for q in self.questions:
            for answer in q.get("answers", []) or []:
                text = (answer.get("answer") or "").strip()
                if not text:
                    continue
                question = (q.get("question") or "").strip()
                combined = f"{question} {text}".strip() if question else text
                out.append((answer.get("source_url") or "", combined))
        return out


def _entries(path: Path) -> Iterator[dict[str, Any]]:
    """AVeriTeC ships a JSON array; some mirrors ship JSONL. Accept both."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return
    if text[0] == "[":
        yield from json.loads(text)
        return
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: not valid JSON - {exc}") from exc


def load_averitec(path: Path | str, split: str = "dev") -> list[Claim]:
    """Read an AVeriTeC split. Claim ids are positional, matching the knowledge store."""
    path = Path(path)
    claims = []
    for i, entry in enumerate(_entries(path)):
        claims.append(
            Claim(
                id=str(entry.get("claim_id", i)),
                claim=entry.get("claim", "").strip(),
                gold_label=normalise_label(entry.get("label")),
                claim_date=entry.get("claim_date"),
                speaker=entry.get("speaker"),
                questions=entry.get("questions") or [],
                metadata={
                    "split": split,
                    "source": str(path),
                    "justification": entry.get("justification"),
                },
            )
        )
    return claims


def sample(claims: Sequence[Claim], n: int | None, seed: int = 13) -> list[Claim]:
    """Seeded sample (rule 6). Returns everything, in order, when n is None or too large."""
    if n is None or n >= len(claims):
        return list(claims)
    picked = random.Random(seed).sample(range(len(claims)), n)
    return [claims[i] for i in sorted(picked)]


def label_counts(claims: Sequence[Claim]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in claims:
        key = c.gold_label or "(unlabelled)"
        counts[key] = counts.get(key, 0) + 1
    return counts
