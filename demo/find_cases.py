"""Find claims that make a good live demonstration (P4.7 support).

A demo case has to do three things in sequence, or it shows nothing:

1. **clean, defences on** -> the correct verdict, so there is something to break;
2. **poisoned, defences off** -> the adversary's verdict, so the attack visibly lands;
3. **poisoned, defences on** -> not the adversary's verdict, so the defence visibly holds.

Most claims fail (1) --- the system is right on about half of dev --- so the search prunes
there before spending anything on the attack runs. Cases are ranked with the full
three-step ones first, since those are the only ones worth putting in front of an
examiner.

Calls are cached, so a second run over the same claims is instant and the shortlist is
stable.

    python -m demo.find_cases --limit 40
    python -m demo.find_cases --limit 40 --adaptive
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from attacks.memory import FLIP
from core.paths import DATA_DIR
from demo.app import Backend
from eval.datasets import Claim, sample

OUT = Path("demo/demo_cases.json")


@dataclass
class Case:
    claim_id: str
    claim: str
    gold: str
    wanted: str
    clean: str
    attacked_undefended: str
    attacked_defended: str

    @property
    def lands(self) -> bool:
        return self.attacked_undefended == self.wanted

    @property
    def defended(self) -> bool:
        return self.attacked_defended != self.wanted

    @property
    def rank(self) -> int:
        """Full story first, then attacks that land but are not stopped."""
        if self.lands and self.defended:
            return 0
        if self.lands:
            return 1
        return 2


def search(backend: Backend, claims: list[Claim], rate: float, adaptive: bool) -> list[Case]:
    cases: list[Case] = []
    checked = correct = 0

    for claim in claims:
        if not claim.gold_label or claim.gold_label not in FLIP:
            continue
        checked += 1

        # Step 1, and the prune: no point attacking a claim already answered wrong.
        clean = backend.run(claim, memory=True, gate=True, trust=True, rate=0.0, adaptive=False)
        if clean.verdict != claim.gold_label:
            continue
        correct += 1

        undef = backend.run(
            claim, memory=True, gate=False, trust=False, rate=rate, adaptive=adaptive
        )
        defended = backend.run(
            claim, memory=True, gate=True, trust=True, rate=rate, adaptive=adaptive
        )
        case = Case(
            claim_id=str(claim.id),
            claim=claim.claim,
            gold=claim.gold_label,
            wanted=FLIP[claim.gold_label],
            clean=clean.verdict,
            attacked_undefended=undef.verdict,
            attacked_defended=defended.verdict,
        )
        cases.append(case)
        flag = "LANDS" if case.lands else "     "
        print(f"  [{checked}] {flag} {claim.id:<5} {claim.claim[:58]}", flush=True)

    print(f"\n  {checked} claims checked, {correct} answered correctly when clean")
    return sorted(cases, key=lambda c: (c.rank, c.claim_id))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m demo.find_cases")
    p.add_argument("--limit", type=int, default=40, help="claims to scan")
    p.add_argument("--rate", type=float, default=0.2)
    p.add_argument("--adaptive", action="store_true", help="use the adaptive adversary")
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args(argv)

    backend = Backend()
    from eval.datasets import load_averitec

    every = load_averitec(DATA_DIR / "averitec" / "dev.json", "dev")
    claims = sample(every, args.limit, args.seed)
    print(
        f"[verimem] scanning {len(claims)} claims at {args.rate:.0%} poisoning, "
        f"{'adaptive' if args.adaptive else 'naive'} adversary\n"
    )

    cases = search(backend, claims, args.rate, args.adaptive)
    good = [c for c in cases if c.lands and c.defended]
    partial = [c for c in cases if c.lands and not c.defended]

    print(f"\n{'=' * 74}")
    print(f"  {len(good)} full demo cases (correct -> attack lands -> defence holds)")
    print(f"  {len(partial)} where the attack lands and the defence does NOT hold")
    print("=" * 74)

    for c in good[:10]:
        print(f"\n  claim {c.claim_id}  (gold {c.gold}, adversary wants {c.wanted})")
        print(f"    {c.claim[:96]}")
        print(
            f"    clean {c.clean}  |  undefended {c.attacked_undefended}"
            f"  |  defended {c.attacked_defended}"
        )

    if partial:
        print("\n  --- attack lands, defence does not stop it ---")
        for c in partial[:5]:
            print(f"  claim {c.claim_id}: {c.claim[:70]}")
            print(f"    defended verdict {c.attacked_defended} == wanted {c.wanted}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps([asdict(c) for c in cases], indent=2), encoding="utf-8")
    print(f"\n[verimem] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
