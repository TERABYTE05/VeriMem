"""Hand-check the relevance labels and report the noise rate (P1.2).

The training labels were assigned automatically -- positives are passages from the page
the fact-checker actually cited, negatives are BM25-selected passages from other pages --
so the evaluator's 0.728 accuracy is only meaningful against some estimate of how often
that automatic label is simply wrong. This presents 100 sampled pairs one at a time, takes
a y/n judgement, and only then reveals the key.

**A human has to do this.** Asking a language model to judge the pairs would measure one
automatic system agreeing with another's labelling, which is not label noise; the
disagreement that matters is with a person reading the passage. So this is a prompt-and-
record tool, not an annotator.

**The key stays hidden until the end**, because knowing the intended answer makes the
judgement worthless -- it is far too easy to talk yourself into whichever label is shown.

Answers are written after every item, so stopping halfway loses nothing and re-running
resumes.

    python -m evaluator.label_check              # annotate (resumes)
    python -m evaluator.label_check --report     # score what is answered so far
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

REVIEW = Path("data/evaluator/label_review.md")
ANSWERS = Path("data/evaluator/label_review_answers.json")

ITEM = re.compile(r"^### (\d+)\.\s", re.M)
KEY_ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*(relevant|not relevant)\s*\|\s*(\w+)\s*\|", re.M)


@dataclass
class Item:
    number: int
    claim: str
    passage: str


def parse(path: Path) -> tuple[list[Item], dict[int, tuple[str, str]]]:
    """The 100 pairs, and the key kept separate so the annotating path never sees it."""
    text = path.read_text(encoding="utf-8")

    key = {int(n): (verdict, kind) for n, verdict, kind in KEY_ROW.findall(text)}

    body = text.split("| 1 |")[0] if "| 1 |" in text else text
    starts = [(int(m.group(1)), m.start()) for m in ITEM.finditer(body)]
    items: list[Item] = []
    for i, (number, start) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(body)
        block = body[start:end]
        claim = re.search(r"\*\*Claim:\*\*\s*(.+?)(?:\n\n|\Z)", block, re.S)
        passage = re.search(r"\*\*Passage:\*\*\s*(.+?)(?:\n\n|\Z)", block, re.S)
        items.append(
            Item(
                number=number,
                claim=" ".join(claim.group(1).split()) if claim else "",
                passage=" ".join(passage.group(1).split()) if passage else "",
            )
        )
    return items, key


def load_answers() -> dict[int, str]:
    if not ANSWERS.exists():
        return {}
    return {int(k): v for k, v in json.loads(ANSWERS.read_text(encoding="utf-8")).items()}


def save_answers(answers: dict[int, str]) -> None:
    ANSWERS.parent.mkdir(parents=True, exist_ok=True)
    ANSWERS.write_text(json.dumps({str(k): v for k, v in sorted(answers.items())}, indent=2))


def report(items: list[Item], key: dict[int, tuple[str, str]], answers: dict[int, str]) -> None:
    """Disagreement overall and by kind -- the breakdown is the useful part."""
    answered = {n: a for n, a in answers.items() if a in ("y", "n") and n in key}
    if not answered:
        print("Nothing answered yet.")
        return

    by_kind: dict[str, list[bool]] = {}
    disagreements: list[tuple[int, str, str, str]] = []
    for n, a in sorted(answered.items()):
        verdict, kind = key[n]
        auto_relevant = verdict == "relevant"
        human_relevant = a == "y"
        agree = auto_relevant == human_relevant
        by_kind.setdefault(kind, []).append(agree)
        if not agree:
            disagreements.append(
                (n, kind, verdict, "relevant" if human_relevant else "not relevant")
            )

    total = sum(len(v) for v in by_kind.values())
    wrong = sum(1 for v in by_kind.values() for ok in v if not ok)
    print(f"\n  answered           {total} of {len(items)}")
    print(f"  disagreements      {wrong}")
    print(f"  LABEL NOISE RATE   {wrong / total:.3f}\n")

    print(f"  {'kind':<16}{'n':>5}{'disagree':>10}{'rate':>8}")
    for kind in sorted(by_kind):
        v = by_kind[kind]
        bad = sum(1 for ok in v if not ok)
        print(f"  {kind:<16}{len(v):>5}{bad:>10}{bad / len(v):>8.3f}")

    # Hard negatives are where the sampling expected mislabels, so call it out directly.
    hard = by_kind.get("hard_negative")
    if hard:
        bad = sum(1 for ok in hard if not ok)
        print(
            f"\n  Of {len(hard)} hard negatives, {bad} were judged genuinely relevant"
            f" ({bad / len(hard):.1%}).\n  Those are passages the builder called negative"
            " that a reader would accept as evidence."
        )

    if disagreements:
        print("\n  disagreements (number, kind, automatic, human):")
        for n, kind, auto, human in disagreements[:25]:
            print(f"    {n:>4}  {kind:<15} auto={auto:<14} human={human}")

    print(
        f"\n  Put {wrong / total:.3f} in PROGRESS.md -> Results as the label noise rate,"
        f" over {total} hand-checked pairs."
    )


def annotate(items: list[Item], answers: dict[int, str]) -> dict[int, str]:
    todo = [it for it in items if answers.get(it.number) not in ("y", "n")]
    if not todo:
        print("All 100 already answered. Use --report to score them.")
        return answers

    print(f"\n{len(todo)} left of {len(items)}.  y = relevant,  n = not relevant,")
    print("s = skip,  q = save and stop.  The key stays hidden until you finish.\n")

    for i, it in enumerate(todo, start=1):
        print("=" * 78)
        print(f"  [{i}/{len(todo)}]   item {it.number}")
        print("\n  CLAIM")
        print(textwrap.fill(it.claim, 74, initial_indent="    ", subsequent_indent="    "))
        print("\n  PASSAGE")
        print(textwrap.fill(it.passage, 74, initial_indent="    ", subsequent_indent="    "))
        print()

        while True:
            try:
                reply = input("  relevant to judging this claim? [y/n/s/q] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  stopped; answers saved.")
                save_answers(answers)
                return answers
            if reply in ("y", "n", "s", "q"):
                break
            print("  please answer y, n, s or q")

        if reply == "q":
            save_answers(answers)
            print(f"\n  saved {len(answers)} answers to {ANSWERS}. Re-run to continue.")
            return answers
        if reply != "s":
            answers[it.number] = reply
            save_answers(answers)

    save_answers(answers)
    print("\n  done.")
    return answers


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m evaluator.label_check")
    p.add_argument("--review", type=Path, default=REVIEW)
    p.add_argument("--report", action="store_true", help="score what is answered, do not annotate")
    args = p.parse_args(argv)

    if not args.review.exists():
        raise SystemExit(
            f"{args.review} not found. Build it with `python -m evaluator.build_data`."
        )

    items, key = parse(args.review)
    if len(items) != len(key):
        print(
            f"[verimem] WARNING: parsed {len(items)} items but {len(key)} key rows. "
            "The report will only score items present in both."
        )
    answers = load_answers()

    if args.report:
        report(items, key, answers)
        return 0

    answers = annotate(items, answers)
    if sum(1 for a in answers.values() if a in ("y", "n")) == len(items):
        report(items, key, answers)
    else:
        print("  run with --report once you have finished, or to see progress so far.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
