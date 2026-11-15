# VeriMem — slides

14 slides. Written in Marp format (`marp-cli`, or the Marp VS Code extension) so they
render to PDF or HTML from this file; the content ports to Google Slides by hand if
preferred. Figures come from `report/figures/`, so regenerate them first with
`python -m report.make_figures`.

Speaker notes are the `<!-- -->` blocks — they carry the number you should say aloud and
the question each slide invites.

---
marp: true
theme: default
paginate: true
---

# VeriMem
## Poisoning the memory of a fact-checking agent

Teesha Ramchandani · IIT Bhilai · NLP course project

<!--
One sentence: we built a fact-checking agent that remembers its own past work, then
attacked it through that memory. Don't preview the findings here.
-->

---

## The setup

**Input:** a claim.
**Output:** a verdict — Supported / Refuted / Conflicting evidence / Not enough evidence —
**plus an evidence trail**: sources, relevance grades, trust scores, and the past
verifications it drew on.

Two papers reproduced from scratch and transferred to fact-checking:

- **CRAG** — a fine-tuned evaluator grades evidence; two thresholds route
- **ExpRAG** — a store of past trajectories feeds fine-tuning

<!--
The evidence trail is not decoration — it is what makes the attack visible, and it is what
you demo later.
-->

---

## The question

An agent that stores its own past work has a new attack surface: **the memory.**

An adversary who can write to it shapes how future claims are judged —
without touching the model, the retriever, or the evidence corpus.

Every record they write is a **syntactically perfect trajectory.**
No format check can find them.

<!--
This is the pitch. Pause here. The point is that the attack is semantic, not structural.
-->

---

## Pipeline

```
claim → retrieve (BM25) → grade (Flan-T5) → route
      → retrieve experience  [C2 gate]
      → verify (Qwen2.5)     [C3 trust]
      → store trajectory
```

Four flags: `experience_retrieval`, `gate_experience`, `source_trust`, `verifier`.

With all flags off, the pipeline reproduces the Phase 1 baseline **byte for byte** —
asserted by a regression test.

<!--
If asked "how do you know your baseline didn't drift": the test passes the full machinery
to a flags-off run and asserts identical model calls. It caught a real violation.
-->

---

## H1 — experience retrieval is the vulnerability

![w:800](figures/memory_poisoning.png)

**0.220 → 0.430** attack success as poisoning rises to 20%.
No-memory has no store to write to: flat at **0.160**.

<!--
Also worth saying: plain retrieval costs 0.06 attack success at ZERO poisoning, because
trajectory accuracy is only 0.59. Clean experience is often wrong experience.
-->

---

## H2 — both defences work…

**C2 — evaluator gating.** Does this trajectory's evidence support *its own verdict*?

*(Our first version asked "is it relevant to the new claim?" — and was defeated. A poisoned
record is written about the target claim, so it IS relevant. Relevance is not truthfulness.)*

**C3 — Bayesian source trust.** Beta(α, β) per domain, trust = α/(α+β).

Prior is Beta(1, 3), not Beta(1, 1) — a stranger starts at 0.25, below the floor.

<!--
The C2 story is a good one to tell: the first specification failed for an interesting
reason, and fixing it required changing the question, not the threshold.
-->

---

## …but which one works **inverts** with the adversary

![w:780](figures/adversary_inversion.png)

| at 20% poisoning | naive | adaptive |
|---|---|---|
| undefended | 0.430 | 0.460 |
| + gating (C2) | 0.320 | **0.310** |
| + trust (C3) | **0.230** | 0.460 ← defeated |

<!--
THE slide. Each defence fails against the adversary the other stops. A study with one
adversary would have concluded either alone suffices.
-->

---

## Why: what trust actually learned

![w:800](figures/trust_divergence.png)

Misinformation domains: **0.380**. Everything else: **0.375**.

`twitter.com` 0.76 · `nature.com` 0.29 · `cbo.gov` 0.23

<!--
No credibility model produces that ordering. C3 credits domains by whether the VERDICT was
right, so it learns "citing this coincided with being correct". Claims evidenced by a CBO
report are hard; claims evidenced by a tweet are easy.

So C3 is a NOVELTY detector, not a credibility detector. That is why it stopped the naive
adversary (unseen domains, below the floor) and why the adaptive one defeated it.
-->

---

## The second surface is worse

![w:800](figures/source_poisoning.png)

Fabricated **pages**: attack success **0.690** (vs memory poisoning's 0.430),
and macro-F1 collapses 0.374 → 0.178.

Evaluator grades fabricated evidence **0.667** vs real **0.740**.

<!--
This is the measured version of "relevance is not truthfulness". The fakes ARE relevant and
the evaluator is right to say so, so gating has no signal. Against the adaptive adversary
here we have no defence at all: 0.620 vs 0.630 undefended.
-->

---

## Propagation: a clean null

Does a poisoned verdict contaminate the *next* claim?

**0 of 28 flipped targets propagated.**

The channel existed — A's trajectory was retrieved in **67 of 77** pairs under both
conditions, and **0 of 77** when absent.

<!--
Three conditions per pair differing ONLY in A's verdict, so a change would be attributable.
With 28 trials, say "below roughly 10%", not "zero". Likely because the neighbour is
recognisably a different claim at median similarity 0.23.
-->

---

## Reproductions: two honest negatives

**CRAG** — evaluator reproduces (+0.161 AUC over an off-the-shelf cross-encoder), routing
reproduces. **Refinement does not transfer**: +0.009 [−0.031, +0.051].

**ExpRAG** — fine-tuning works (accuracy 0.263 → 0.667).
**The central claim does not replicate**: +0.002 [−0.021, +0.026].

<!--
Do not apologise for these. A tight interval around zero is a stronger statement than a
marginal win. And we found the mechanism for both — next slide.
-->

---

## …and we found why

**CRAG:** refinement hands the verifier **66% less text** (3,750 → 1,281 chars).
It abstains: "Not enough evidence" **145 times against a gold count of 35**.

**ExpRAG:** the retrieved training neighbours are **median 0.203** similarity;
only 21% exceed 0.30. The model learned to ignore a context that held nothing.

<!--
This is what turns a null into a finding. CRAG's refinement was built for QA where a short
span suffices; fact-checking gives the model a valid retreat into "not enough evidence".
ExpRAG needs a store dense enough for precedents to exist — 1,500 trajectories over
AVeriTeC's spread doesn't have them.
-->

---

## What we'd want a reader to check

- **The evaluator has saturated its labels.** Hand-checked noise rate **0.289**; a perfect
  classifier scores ~0.711 against labels that noisy. Ours scores 0.728 — *above* the
  ceiling, so its errors correlate with the labeller's.
- **Over-abstention is 52.9% of all errors.** Only 5% of errors had no relevant evidence —
  the verifier fails, not retrieval.
- Single seed; every number carries a bootstrap interval.

<!--
Volunteering the limitations is the strongest move in the viva. The label-noise ceiling in
particular is the thing an examiner would otherwise find.
-->

---

## Conclusion

An agent that remembers its own work can be attacked through that memory, and the
attack scales with how much of the store the adversary controls.

**Two defences each look sufficient alone. Neither is.**

Source trust detects *novelty*, not *unreliability* — which is why it stops an attacker
using unknown domains, and why it offers nothing against one citing the agent's own
favourites.

<!--
End on the inversion. If there's time, run the demo: `python -m verimem "<claim>"` and then
the same claim with `--no-gate` to show C2 doing its job live.
-->
