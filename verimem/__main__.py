"""One command that runs VeriMem on a claim (P3.1, G3).

    python -m verimem "The Kochi metro opened to the public in 2017."

The project's stated output is a verdict **plus an evidence trail**: the sources, their
relevance grades, the trust in each publisher, and the past verifications it drew on. A
command that printed only the verdict would be hiding the part the contributions are
about, so the trail is the default and `--quiet` is what suppresses it.

Every contribution is on by default, because the default should be the system the paper
describes. `--no-memory`, `--no-gate` and `--no-trust` switch them off individually, which
makes the ablations demonstrable from the command line rather than only from a results
table -- useful in a viva, where "show me what C2 actually does" is the obvious question.

Running this needs the evidence pool and the trained evaluator. With neither, it still
runs and says loudly what it is missing rather than producing a confident verdict from a
lexical stub.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from agent.crag import crag
from core.config import RunConfig
from core.llm import LLMClient
from core.paths import DATA_DIR, RESULTS_DIR
from eval.datasets import Claim, load_averitec
from evaluator.score import load_scorer
from experience.format import Trajectory, read_jsonl
from experience.store import BetaTrust, ExperienceStore
from retrieval.blocklist import domain_of
from retrieval.store import open_store

DEFAULT_POOL = DATA_DIR / "averitec" / "pool_dev.jsonl"
DEFAULT_TRAJECTORIES = DATA_DIR / "trajectories" / "trajectories.jsonl"


def find_evaluator(explicit: Path | None = None) -> Path | None:
    """The trained relevance evaluator, by convention from a results folder."""
    if explicit:
        return explicit if explicit.exists() else None
    hits = sorted(Path(RESULTS_DIR).glob("*_t5-evaluator/model/config.json"))
    return hits[-1].parent if hits else None


def _bar(value: float, width: int = 12, lo: float = -1.0, hi: float = 1.0) -> str:
    """A small gauge, so a column of grades is scannable without reading each number."""
    frac = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    filled = round(frac * width)
    return "█" * filled + "·" * (width - filled)


def render(trajectory: Trajectory, trust: BetaTrust | None, cfg: RunConfig) -> str:
    out: list[str] = []
    out.append("")
    out.append(f"  CLAIM     {trajectory.claim}")
    out.append(f"  VERDICT   {trajectory.verdict.upper()}")
    if trajectory.gold_label:
        mark = "correct" if trajectory.correct else "WRONG"
        out.append(f"  GOLD      {trajectory.gold_label}  ({mark})")
    out.append(f"  WHY       {trajectory.rationale}")

    flags = [
        n
        for n, on in (
            ("experience", cfg.experience_retrieval),
            ("gate", cfg.gate_experience),
            ("trust", cfg.source_trust),
        )
        if on
    ]
    out.append(f"  CONFIG    {', '.join(flags) or 'plain CRAG (all contributions off)'}")
    grade = trajectory.metadata.get("grade")
    action = trajectory.action
    if grade is not None:
        out.append(f"  ROUTING   grade {grade:+.3f} -> {action}")

    out.append("")
    out.append("  EVIDENCE")
    if not trajectory.evidence:
        out.append("    (none retrieved)")
    for e in trajectory.evidence[:8]:
        dom = domain_of(e.url) or "?"
        t = f"  trust {trust.trust(dom):.2f}" if trust is not None else ""
        out.append(f"    {_bar(e.grade)} {e.grade:+.3f}  [{dom}]{t}")
        out.append(f"      {e.text[:140]}")

    used = trajectory.metadata.get("retrieved_ids") or []
    out.append("")
    out.append(f"  PAST VERIFICATIONS DRAWN ON  ({len(used)})")
    if not used:
        reason = (
            "experience retrieval is off"
            if not cfg.experience_retrieval
            else "nothing survived retrieval or the filters"
        )
        out.append(f"    (none - {reason})")
    for cid in used:
        out.append(f"    {cid}")

    if trajectory.metadata.get("trust"):
        ts = trajectory.metadata["trust"]
        out.append("")
        out.append(
            f"  TRUST FILTER  kept {ts.get('strips_kept')} of "
            f"{ts.get('strips_considered')} strips "
            f"(dropped {ts.get('strips_dropped_by_trust')} below {cfg.trust_floor})"
        )

    out.append("")
    out.append("  STEPS")
    for s in trajectory.steps:
        out.append(f"    {s.action:<12} {s.detail}")
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m verimem",
        description="Fact-check a claim and show the evidence trail.",
    )
    p.add_argument("claim", nargs="?", help="the claim to verify")
    p.add_argument("--claim-id", default=None, help="an AVeriTeC dev id, to score against gold")
    p.add_argument("--store", type=Path, default=DEFAULT_POOL, help="evidence pool")
    p.add_argument("--trajectories", type=Path, default=DEFAULT_TRAJECTORIES)
    p.add_argument("--evaluator", type=Path, default=None)
    p.add_argument("--data", type=Path, default=DATA_DIR / "averitec" / "dev.json")
    p.add_argument("--no-memory", dest="memory", action="store_false", help="turn off ExpRAG")
    p.add_argument("--no-gate", dest="gate", action="store_false", help="turn off C2")
    p.add_argument("--no-trust", dest="trust", action="store_false", help="turn off C3")
    p.add_argument("--json", action="store_true", help="machine-readable trajectory")
    p.add_argument("--quiet", action="store_true", help="verdict only")
    args = p.parse_args(argv)

    if not args.claim and not args.claim_id:
        p.error("give a claim, or --claim-id to take one from the dev split")

    claim: Claim
    if args.claim_id:
        if not args.data.exists():
            raise SystemExit(f"{args.data} not found, so --claim-id cannot be resolved")
        matches = [
            c for c in load_averitec(args.data, args.data.stem) if str(c.id) == args.claim_id
        ]
        if not matches:
            raise SystemExit(f"no claim with id {args.claim_id} in {args.data}")
        claim = matches[0]
    else:
        claim = Claim(id="cli", claim=args.claim, gold_label=None)

    evaluator = find_evaluator(args.evaluator)
    if evaluator is None:
        print(
            "[verimem] WARNING: no trained evaluator found under results/. Relevance "
            "grading falls back to a lexical stub, so the grades below are word overlap, "
            "not the trained model. Train one with `python -m evaluator.train`.",
            file=sys.stderr,
        )
    if not args.store.exists():
        raise SystemExit(
            f"no evidence pool at {args.store}. VeriMem retrieves from the AVeriTeC "
            "archive pool; pass --store, or see the README for how to build it."
        )

    cfg = RunConfig(
        split="cli",
        experience_retrieval=args.memory,
        gate_experience=args.memory and args.gate,
        source_trust=args.trust,
    )

    llm = LLMClient()
    cfg.api_model, cfg.api_base_url = llm.model, llm.base_url
    scorer = load_scorer(evaluator)
    source = open_store(args.store)

    memory = None
    trust: BetaTrust | None = None
    if args.memory or args.trust:
        if args.trajectories.exists():
            past = read_jsonl(args.trajectories)
            trust = BetaTrust()
            for t in past:
                trust.observe(t)
            if args.memory:
                memory = ExperienceStore(
                    past,
                    scorer=scorer,
                    trust=trust,
                    gate_threshold=cfg.gate_threshold,
                    trust_floor=cfg.trust_floor,
                )
        else:
            print(
                f"[verimem] no trajectory store at {args.trajectories}; running without "
                "experience and with uniform source trust.",
                file=sys.stderr,
            )
            cfg = cfg.with_(experience_retrieval=False, gate_experience=False)

    trajectory = crag(
        claim,
        llm,
        source,
        scorer,
        cfg,
        memory=memory,
        trust=trust if args.trust else None,
    )

    if args.json:
        print(json.dumps(trajectory.to_dict(), indent=2, ensure_ascii=False))
    elif args.quiet:
        print(trajectory.verdict)
    else:
        print(render(trajectory, trust if args.trust else None, cfg))
    return 0


def _entry() -> Any:
    return main()


if __name__ == "__main__":
    raise SystemExit(main())
