# 2026-11-02_null-analysis — what is actually behind the two nulls

Both reproductions produced a null on macro-F1. Neither is an absence of effect; each is
two effects cancelling, and in both cases the mechanism is measurable from data already
on disk. Analysis only — no new model calls.

---

## 1. CRAG: refinement starves the verifier, and it abstains

The reported null is `crag − plain_rag = +0.009 [−0.032, +0.052]` on macro-F1. Underneath:

| | plain RAG | CRAG |
|---|---|---|
| evidence items given to the verifier | 5 | 5 |
| mean characters per item | 750 | **256** |
| **total characters of evidence** | **3,750** | **1,281** |

Decompose-then-recompose keeps the best five *sentences* where plain RAG passes the best
five *passages*, so the verifier sees **66% less text**. It responds by abstaining:

| predicted | gold | no retrieval | plain RAG | CRAG |
|---|---|---|---|---|
| Refuted | 305 | 348 | 309 | 263 |
| Supported | 122 | 57 | 115 | 82 |
| Conflicting evidence | 38 | 0 | 19 | 10 |
| **Not enough evidence** | **35** | 95 | 57 | **145** |

CRAG answers "Not enough evidence" **145 times against a gold count of 35**. 31.0% of its
predictions are an uncertain class, against 14.6% in gold — and plain RAG sits at 15.2%,
almost exactly calibrated.

The consequences separate cleanly, which is why macro-F1 hid them:

* **accuracy falls, significantly**: `crag − plain_rag = −0.054 [−0.096, −0.012]`
* per-class F1 moves the other way on the rare classes (`Conflicting +0.042`,
  `Not enough evidence +0.081`) and against CRAG on the common ones (`Refuted −0.034`,
  `Supported −0.051`); none individually significant, but the directions are consistent
* macro-F1 weights all four equally, so the gains and losses cancel to +0.009

**The transfer finding.** CRAG's refinement was designed for question answering, where a
short answer span is sufficient and surrounding context is noise. Fact-checking is not
that task: the verdict depends on weighing how much the evidence establishes, and
"Not enough evidence" is a *valid answer* the model can retreat to. Cutting the evidence
by two thirds makes that retreat the easy option. The refinement step is the part of CRAG
that transfers worst, and the symptom is over-abstention rather than error.

---

## 2. ExpRAG: the mechanism fired, but there was nothing to retrieve

The reported null is `retrieval-aug SFT − plain SFT = +0.002 [−0.021, +0.026]` — a tight
interval, so a measured null rather than an underpowered test. Training was configured
correctly: all 1,500 examples carried retrieved context (`n_with_retrieval: 1500`).

The question is what that context contained. Measuring the retrieved neighbours'
similarity to their own claim, as a fraction of the claim's self-score:

| | value |
|---|---|
| median | **0.203** |
| 75th percentile | 0.275 |
| 90th percentile | 0.419 |
| share above 0.30 | **21.0%** |
| share above 0.50 | 7.3% |

So in roughly **four training examples out of five, the two "relevant past verifications"
placed in the context were only weakly related to the claim**. The model was trained to
attend to a context window that usually held nothing worth attending to, so it learned to
ignore it — and at inference, retrieval then adds +0.002.

**The scale finding.** ExpRAG's premise is that an agent's own history contains precedents
for new cases. That requires a store dense enough for precedents to exist. AVeriTeC's
claims span politics, health and economics across many countries and years, and 1,500
trajectories over that spread do not contain near-neighbours for most claims. The
published setting is narrower.

This also says where it *would* work, which is the useful form of a negative result: a
narrower domain, or a store one or two orders of magnitude larger. Note that retrieval
does help the **prompted** student (+0.041), where the model has not been trained to
discount the context — consistent with the same explanation.

---

## Reproduce

Both analyses read only `results/2026-10-28_g1-rerun/*_trajectories.jsonl` and
`data/trajectories/*.jsonl`. No endpoint, no GPU.
