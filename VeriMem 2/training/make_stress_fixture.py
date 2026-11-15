"""Generate a worst-case-length fixture for the VRAM headroom test.

The committed smoke fixture runs at roughly 500 tokens per example. The real runs go up
to `max_seq_length` (4,096), and training memory scales with sequence length, so a
comfortable smoke test does not prove the real run fits in 8 GB.

This writes trajectories whose SFT examples land right at the cap. The output is several
MB of repetitive filler, so it is generated locally into the gitignored `data/` rather
than committed.

    python -m training.make_stress_fixture
    python -m training.sft --data data/stress_trajectories.jsonl \
        --name headroom --mode retrieval --limit 20 --max-steps 10
"""

from __future__ import annotations

import random

from core.config import RunConfig
from core.paths import DATA_DIR
from experience.format import Evidence, Step, Trajectory, write_jsonl
from training.data import build_dataset

SENTENCES = [
    "The committee's final report sets out the timeline in considerable detail across several annexes.",
    "Independent auditors reviewed the underlying figures and published a reconciliation the following quarter.",
    "Contemporary press coverage described the announcement as significant but noted several open questions.",
    "A subsequent correction adjusted the headline number downward without changing the overall conclusion.",
    "Records held by the national archive give a date that differs from the one cited in the claim.",
    "The organisation's own statement confirms the decision but does not specify when it took effect.",
]

VERDICTS = ("Supported", "Refuted", "Conflicting evidence", "Not enough evidence")


def build(n: int = 24, seed: int = 29) -> list[Trajectory]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        evidence = [
            Evidence(
                url=f"https://source{j}.example.org/doc/{i}",
                title=f"Long source document {i}-{j}",
                text=" ".join(rng.choice(SENTENCES) for _ in range(120)),
                grade=round(rng.uniform(-0.9, 0.95), 2),
            )
            for j in range(6)
        ]
        out.append(
            Trajectory(
                claim=f"Worst-case length claim number {i} about a policy decision and its date",
                verdict=rng.choice(VERDICTS),
                rationale=" ".join(rng.choice(SENTENCES) for _ in range(6)),
                claim_id=f"stress-{i:03d}",
                evidence=evidence,
                steps=[Step("retrieve", "top-6"), Step("verify", "verdict produced")],
                source="synthetic",
                dataset="synthetic-stress",
                seed=seed,
                metadata={"note": "Worst-case-length fixture: VRAM headroom test only."},
            )
        )
    return out


def main() -> int:
    trajectories = build()
    path = write_jsonl(trajectories, DATA_DIR / "stress_trajectories.jsonl")
    cfg = RunConfig()
    print(f"wrote {len(trajectories)} trajectories -> {path}")
    for mode in ("plain", "retrieval"):
        _, stats = build_dataset(trajectories, mode=mode, cfg=cfg)
        print(
            f"  {mode:10s} mean {stats.mean_tokens:6.0f} tokens, "
            f"max {stats.max_tokens} (cap {cfg.max_seq_length})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
