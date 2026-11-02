"""Categorise the failures, and print worked traces (P4.5).

Two things the report needs and a results table cannot give: *why* the system is wrong,
and what a single run actually looks like end to end.

**The taxonomy is built from measurable signals, not from reading the cases.** The plan
said "hand-label 50 failures", and eyeballing fifty trajectories would produce categories
shaped by whoever did the eyeballing -- and would cover a tenth of the set. Every category
here is a predicate over the trajectory's own record (its verdict, the gold label, its
evidence grades, the branch the router chose), so it is reproducible, it covers all 500,
and a reader can check any case against the stated rule.

What that cannot capture is whether a passage *really* supports a verdict, which is a
judgement. So the worked traces are printed in full for a human to read, and the report
quotes them rather than asserting a conclusion about them.

    python -m eval.failure_analysis --run results/2026-10-28_g1-rerun --system crag
    python -m eval.failure_analysis --run results/2026-10-28_g1-rerun --traces 3
"""

from __future__ import annotations

import argparse
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from experience.format import Trajectory, read_jsonl

UNCERTAIN = ("Conflicting evidence", "Not enough evidence")
DEFINITE = ("Supported", "Refuted")

# A grade below this means the evaluator judged even the best retrieved strip to be
# weakly related to the claim -- the retrieval failed before the verifier saw anything.
WEAK_EVIDENCE = 0.3


@dataclass
class Failure:
    trajectory: Trajectory
    category: str
    detail: str


def categorise(t: Trajectory) -> Failure | None:
    """One failure category per wrong trajectory, from signals in its own record.

    Order matters: the first matching rule wins, and they are ordered so the more
    specific diagnosis is reached first. Each rule is a statement a reader can verify
    against the trajectory.
    """
    gold, said = t.gold_label, t.verdict
    if not gold or gold == said:
        return None

    best = max((e.grade for e in t.evidence), default=-1.0)
    grade = t.metadata.get("grade")

    if not t.evidence:
        return Failure(t, "no evidence retrieved", "retrieval returned nothing at all")

    if said == "Not enough evidence" and gold in DEFINITE:
        return Failure(
            t,
            "over-abstention",
            f"answered 'not enough evidence' where the gold label is definite; "
            f"best strip graded {best:+.2f}",
        )

    if said in UNCERTAIN and gold in DEFINITE:
        return Failure(t, "hedged on a decidable claim", f"said {said!r}; best strip {best:+.2f}")

    if gold in UNCERTAIN and said in DEFINITE:
        return Failure(
            t,
            "false confidence",
            f"committed to {said!r} where the gold label is {gold!r}",
        )

    if {gold, said} == set(DEFINITE):
        if best < WEAK_EVIDENCE:
            return Failure(
                t,
                "polarity flip on weak evidence",
                f"{gold} -> {said} with the best strip graded only {best:+.2f}",
            )
        return Failure(
            t,
            "polarity flip despite relevant evidence",
            f"{gold} -> {said}; best strip graded {best:+.2f}"
            + (f", router chose {t.action}" if t.action else ""),
        )

    return Failure(t, "other", f"{gold} -> {said}; grade {grade}")


def summarise(trajectories: list[Trajectory]) -> None:
    labelled = [t for t in trajectories if t.gold_label]
    failures = [f for f in (categorise(t) for t in labelled) if f]
    if not labelled:
        raise SystemExit("no labelled trajectories in this run")

    print(
        f"\n  {len(labelled)} labelled claims, {len(failures)} wrong "
        f"({len(failures) / len(labelled):.1%})\n"
    )

    counts = Counter(f.category for f in failures)
    print(f"  {'failure mode':<42}{'n':>5}{'of errors':>11}{'of all':>9}")
    print("  " + "-" * 65)
    for cat, n in counts.most_common():
        print(f"  {cat:<42}{n:>5}{n / len(failures):>11.1%}{n / len(labelled):>9.1%}")

    # The confusion the taxonomy is built from, so a reader can audit the rules.
    print("\n  gold -> predicted, for the errors:")
    pairs = Counter((f.trajectory.gold_label, f.trajectory.verdict) for f in failures)
    for (gold, said), n in pairs.most_common(8):
        print(f"    {gold:<22} -> {said:<22} {n:>4}")

    weak = [
        f
        for f in failures
        if max((e.grade for e in f.trajectory.evidence), default=-1) < WEAK_EVIDENCE
    ]
    print(
        f"\n  {len(weak)} of {len(failures)} errors ({len(weak) / len(failures):.0%}) had no "
        f"strip graded above {WEAK_EVIDENCE:+.1f}:\n  retrieval, not the verifier, is what "
        "failed in those."
    )

    by_action = Counter(f.trajectory.action or "none" for f in failures)
    all_action = Counter(t.action or "none" for t in labelled)
    print("\n  error rate by the branch the router chose:")
    for action, total in all_action.most_common():
        bad = by_action.get(action, 0)
        print(f"    {action:<12}{bad:>4}/{total:<5} = {bad / total:.1%}")


def traces(trajectories: list[Trajectory], n: int) -> None:
    """Worked traces: one correct, one over-abstention, one polarity flip.

    Chosen by rule rather than by hand, so they are not cherry-picked -- the first case
    of each kind in claim-id order.
    """
    labelled = [t for t in trajectories if t.gold_label]
    picks: list[tuple[str, Trajectory]] = []

    correct = [t for t in labelled if t.correct and len(t.evidence) >= 3]
    if correct:
        picks.append(("a correct verification", correct[0]))

    for want in ("over-abstention", "polarity flip despite relevant evidence", "false confidence"):
        hits = [f.trajectory for f in (categorise(t) for t in labelled) if f and f.category == want]
        if hits:
            picks.append((want, hits[0]))

    for title, t in picks[:n]:
        print("\n" + "=" * 78)
        print(f"  {title.upper()}      claim {t.claim_id}")
        print("=" * 78)
        print("\n  CLAIM")
        print(textwrap.fill(t.claim, 74, initial_indent="    ", subsequent_indent="    "))
        print(f"\n  GOLD      {t.gold_label}")
        print(f"  VERDICT   {t.verdict}")
        print(f"  ROUTER    grade {t.metadata.get('grade')} -> {t.action}")
        print("\n  REASONING")
        print(
            textwrap.fill(
                t.rationale or "(none)", 74, initial_indent="    ", subsequent_indent="    "
            )
        )
        print(
            f"\n  EVIDENCE ({len(t.evidence)} strips, "
            f"{sum(len(e.text) for e in t.evidence)} characters)"
        )
        for e in t.evidence[:5]:
            print(f"    {e.grade:+.3f}  [{e.domain}]")
            print(
                textwrap.fill(
                    e.text[:200], 70, initial_indent="        ", subsequent_indent="        "
                )
            )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.failure_analysis")
    p.add_argument("--run", type=Path, required=True, help="a results folder")
    p.add_argument("--system", default="crag")
    p.add_argument("--traces", type=int, default=0, help="also print this many worked traces")
    args = p.parse_args(argv)

    path = args.run / f"{args.system}_trajectories.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found. Trajectory files are written by eval.run_baselines.")

    trajectories = read_jsonl(path)
    print(f"[verimem] {args.system} in {args.run.name}")
    summarise(trajectories)
    if args.traces:
        traces(trajectories, args.traces)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
