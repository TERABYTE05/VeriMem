"""An off-the-shelf cross-encoder, as the baseline the trained evaluator must beat (P1.4).

The T5 evaluator is the heart of both the CRAG reproduction and C2's gate, and it was
fine-tuned for the job. That is only worth doing if it beats a pretrained relevance model
anyone could download, so this runs one on the same held-out pairs and reports the same
two numbers. The project's Oct 27 decision point said: if the trained evaluator does not
beat this, report both and use the better one downstream.

**The comparison has to be scrupulous**, because it is one the project has an interest in
winning:

* the same dev pairs, the same labels, no re-splitting;
* the same metrics computed the same way -- accuracy at the natural threshold, and AUC as
  the probability a random positive outranks a random negative, ties counting half;
* scores mapped so "threshold 0" means the same thing for both models. The T5 emits
  P(yes) mapped to [-1, 1], where 0 is P(yes) = 0.5. A cross-encoder emits an unbounded
  logit, so `sigmoid(logit) * 2 - 1` puts its boundary in the same place. AUC is
  rank-based and unaffected by any monotone mapping, but accuracy is not, and reading a
  raw logit against 0 would be a different and more flattering threshold.

`CrossEncoderScorer` implements the same `RelevanceScorer` protocol as the T5, so if the
baseline wins it drops into the pipeline without touching a caller.

    python -m evaluator.cross_encoder --data data/evaluator/relevance_dev.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Trained on MS MARCO relevance, small enough for a laptop CPU, and the usual default
# when someone reaches for "a relevance cross-encoder" without fine-tuning anything.
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class CrossEncoderScorer:
    """A pretrained cross-encoder behind the project's `RelevanceScorer` interface."""

    model_name: str = DEFAULT_MODEL
    batch_size: int = 32
    _model: Any = field(default=None, repr=False)

    def _load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        from core.device import get_device  # noqa: PLC0415

        self._model = CrossEncoder(self.model_name, device=str(get_device()))

    def score(self, claim: str, passages: Sequence[str]) -> list[float]:
        """Relevance in [-1, 1], one per passage. Same contract as `T5Scorer`."""
        if not passages:
            return []
        self._load()
        logits = self._model.predict(
            [(claim, p) for p in passages], batch_size=self.batch_size, show_progress_bar=False
        )
        return [2.0 / (1.0 + math.exp(-float(x))) - 1.0 for x in logits]


def relevance_metrics(scores: Sequence[float], labels: Sequence[int]) -> dict[str, Any]:
    """Accuracy at threshold 0 and AUC -- identical in definition to `evaluator.train`."""
    pos = [s for s, lab in zip(scores, labels, strict=True) if lab == 1]
    neg = [s for s, lab in zip(scores, labels, strict=True) if lab == 0]
    correct = sum(1 for s, lab in zip(scores, labels, strict=True) if (s > 0) == (lab == 1))

    if pos and neg:
        wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
        auc = wins / (len(pos) * len(neg))
    else:
        auc = float("nan")

    return {
        "accuracy": round(correct / len(labels), 4),
        "auc": round(auc, 4),
        "mean_score_positive": round(sum(pos) / len(pos), 4) if pos else float("nan"),
        "mean_score_negative": round(sum(neg) / len(neg), 4) if neg else float("nan"),
        "n": len(labels),
        "n_positive": len(pos),
    }


def bootstrap_auc(
    scores: Sequence[float],
    labels: Sequence[int],
    n_resamples: int = 2000,
    seed: int = 13,
) -> tuple[float, float]:
    """A 95% interval for AUC, resampling pairs -- the project's standard for any
    reported number, so the evaluator comparison is held to it too."""
    import random  # noqa: PLC0415

    rng = random.Random(seed)
    n = len(labels)
    draws = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        s = [scores[i] for i in idx]
        lab = [labels[i] for i in idx]
        pos = [v for v, x in zip(s, lab, strict=True) if x == 1]
        neg = [v for v, x in zip(s, lab, strict=True) if x == 0]
        if not pos or not neg:
            continue
        wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
        draws.append(wins / (len(pos) * len(neg)))
    draws.sort()
    return draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]


def evaluate(data: Path, model_name: str = DEFAULT_MODEL, limit: int | None = None) -> dict:
    rows = [json.loads(ln) for ln in data.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if limit:
        rows = rows[:limit]
    if not rows:
        raise SystemExit(f"no pairs in {data}")

    scorer = CrossEncoderScorer(model_name)
    scores: list[float] = []
    labels: list[int] = []

    # Grouped by claim so each call shares one claim, which is how the scorer is used in
    # the pipeline and keeps the batching comparable.
    by_claim: dict[str, list[dict]] = {}
    for r in rows:
        by_claim.setdefault(r["claim"], []).append(r)

    for i, (claim, group) in enumerate(by_claim.items(), start=1):
        scores += scorer.score(claim, [g["passage"] for g in group])
        labels += [int(g["label"]) for g in group]
        if i % 50 == 0 or i == len(by_claim):
            print(f"  [{i}/{len(by_claim)} claims]", flush=True)

    lo, hi = bootstrap_auc(scores, labels)
    return {
        "model": model_name,
        "data": str(data),
        **relevance_metrics(scores, labels),
        "auc_ci": [round(lo, 4), round(hi, 4)],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m evaluator.cross_encoder")
    p.add_argument("--data", type=Path, default=Path("data/evaluator/relevance_dev.jsonl"))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--compare-to",
        type=Path,
        default=None,
        help="a trained evaluator's metrics.json, to print the comparison and the verdict",
    )
    p.add_argument("--out", type=Path, default=None, help="write the metrics here")
    args = p.parse_args(argv)

    if not args.data.exists():
        raise SystemExit(f"{args.data} not found")

    print(f"[verimem] {args.model} on {args.data}")
    result = evaluate(args.data, args.model, args.limit)
    ci = result.get("auc_ci", [float("nan"), float("nan")])
    print(
        f"\n  accuracy {result['accuracy']:.4f}   AUC {result['auc']:.4f} "
        f"[{ci[0]:.4f}, {ci[1]:.4f}]   n {result['n']}"
    )
    print(
        f"  mean score: positive {result['mean_score_positive']:+.4f}  "
        f"negative {result['mean_score_negative']:+.4f}"
    )

    if args.compare_to and args.compare_to.exists():
        trained = json.loads(args.compare_to.read_text(encoding="utf-8"))
        # evaluator.train nests its held-out numbers under "dev"; a flat file is also
        # accepted so this works against either shape rather than silently printing None.
        held_out = trained.get("dev") if isinstance(trained.get("dev"), dict) else trained
        t_auc, t_acc = held_out.get("auc"), held_out.get("accuracy")
        print(f"\n  trained T5   accuracy {t_acc}   AUC {t_auc}")
        if t_auc is not None:
            delta = t_auc - result["auc"]
            winner = "the trained evaluator" if delta > 0 else "the off-the-shelf baseline"
            print(f"  difference   AUC {delta:+.4f} in favour of {winner}")
            print(
                "\n  "
                + (
                    "Fine-tuning was worth it; keep the T5 downstream."
                    if delta > 0
                    else "The baseline matches or beats the trained evaluator. The Oct 27 "
                    "decision point applies: report both, and use the better one."
                )
            )
        result["trained_auc"] = t_auc
        result["trained_accuracy"] = t_acc

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n[verimem] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
