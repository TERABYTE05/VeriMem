# P4.5 — Where VeriMem's errors come from

CRAG on the full dev split, 500 labelled claims, **227 wrong (45.4%)**. Categories are
predicates over each trajectory's own record — its verdict, the gold label, its evidence
grades, the branch the router chose — so every case can be checked against the stated
rule, and the analysis covers all 500 rather than a hand-read sample of 50.

    python -m eval.failure_analysis --run results/2026-10-28_g1-rerun --traces 3

## The failure modes

| mode | n | of errors | of all claims |
|---|---|---|---|
| **over-abstention** | **120** | **52.9%** | 24.0% |
| false confidence | 46 | 20.3% | 9.2% |
| polarity flip despite relevant evidence | 44 | 19.4% | 8.8% |
| hedged on a decidable claim | 8 | 3.5% | 1.6% |
| other | 7 | 3.1% | 1.4% |
| polarity flip on weak evidence | 2 | 0.9% | 0.4% |

**More than half of all errors are the system answering "not enough evidence" when the
gold label is definite.** The two largest gold→predicted confusions are
`Refuted → Not enough evidence` (75) and `Supported → Not enough evidence` (45).

## Retrieval is almost never the culprit

**Only 11 of 227 errors (5%) had no strip graded above +0.3.** In the other 95%, the
evaluator judged the retrieved evidence relevant and the verifier still got it wrong. The
error is concentrated in the verification step, not in finding documents.

That is worth stating precisely because it contradicts the intuitive diagnosis. "The
system is wrong because retrieval is bad" is false here: retrieval is fine, and the
verifier declines to commit.

## Error rate by the branch the router chose

| branch | errors | rate |
|---|---|---|
| correct (refine only) | 126/295 | 42.7% |
| ambiguous (refine + rewrite) | 54/122 | 44.3% |
| **incorrect (rewrite + search)** | **47/83** | **56.6%** |

The rewrite branch performs worst. Some of that is selection — it is chosen when the grade
is low, so it gets the hard claims — but it means the branch intended to *rescue* a poor
retrieval does not. The gap between branches is smaller than the gap between CRAG and
getting these claims right at all.

## What the traces show

`worked_traces.txt` has three in full, selected by rule (first of each kind in claim-id
order) rather than chosen by hand.

The over-abstention trace is the mechanism in a single case. Claim 4 — *"Matt Gaetz was
part of a company that had to pay 75 million in hospice fraud"*, gold **Refuted** — was
graded **0.9825** and routed `correct`: the evaluator judged the retrieved evidence highly
relevant. The verifier answered **"Not enough evidence"**, reasoning that the evidence gave
no information about Gaetz specifically.

Both are right about their own input. The passages were about the right topic, which is
what the evaluator scores; the five surviving *sentences* did not carry the specific fact,
which is what the verifier needed. That is the refinement cost measured elsewhere
(3,750 → 1,281 characters of evidence) appearing as one wrong answer.

## What this implies

The single highest-value change this analysis points at is **passing refined strips
alongside more of their surrounding passage, rather than instead of it** — keeping the
evaluator's ranking while restoring the context the verifier needs to commit. That is a
change to `retrieval/refine.py` and the recompose step, not to any model.

It is not attempted here: it would alter every number in the report with five days to code
freeze. Recorded as the first thing to try next.
