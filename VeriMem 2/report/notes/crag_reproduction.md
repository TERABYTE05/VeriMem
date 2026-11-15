# P1.10 — Reproducing CRAG, and transferring it to fact-checking

**CRAG** (Yan et al., arXiv 2401.15884) adds a fine-tuned retrieval evaluator to RAG. The
evaluator grades retrieved documents; two thresholds turn that grade into one of three
actions — *Correct* → refine the retrieved text, *Incorrect* → rewrite the query and search
again, *Ambiguous* → both. Refinement is decompose-then-recompose: split passages into
strips, score each, keep the best.

We reproduced all of it from scratch and applied it to fact-checking on AVeriTeC.

## What reproduced

**The evaluator.** Flan-T5-base fine-tuned on relevance pairs reaches **AUC 0.811 /
accuracy 0.728** on 751 held-out pairs, against **AUC 0.650 [0.611, 0.686] / accuracy
0.573** for an off-the-shelf `ms-marco-MiniLM` cross-encoder. Fine-tuning gains **+0.161
AUC**.

The *reason* it gains is more specific than the headline. The baseline's accuracy (0.573)
barely clears the majority floor (0.531) while its AUC shows real ranking ability, and its
mean score on genuinely relevant passages is **−0.296**: it ranks adequately and decides
in the wrong place. MS MARCO relevance — *does this passage answer this query* — is not our
question. Since CRAG reads two thresholds off this score, a misplaced boundary routes
almost everything down one branch however well the model ranks. Fine-tuning moved the
boundary, not only the ordering.

**Three-way routing beats any fixed branch.** Tuned on 150 dev claims, the thresholds
(0.75 / 0.0) beat the best single fixed action by 0.044 macro-F1, where the guessed ±0.6
beat it by only 0.004. Routing is doing work; the thresholds have to be fitted to do it.

## What did not transfer: refinement

The headline comparison is a null. On the full dev split (500 claims), paired bootstrap:

| comparison | difference | 95% CI | |
|---|---|---|---|
| plain RAG − no retrieval | +0.029 | [−0.016, +0.078] | ns |
| CRAG − plain RAG | **+0.009** | [−0.031, +0.051] | ns |
| CRAG − no retrieval | +0.038 | [−0.005, +0.085] | ns |

No pairwise difference excludes zero, so we do **not** claim CRAG beats plain RAG on
macro-F1. (An earlier single run recorded 0.414 vs 0.346 and was read as a 0.068 win; it
was not. Macro-F1 here gives a quarter of its weight to a 38-instance class on which every
system scores near zero.)

The null is two effects cancelling, and the mechanism is measurable:

| | plain RAG | CRAG |
|---|---|---|
| evidence items given to the verifier | 5 | 5 |
| **total characters of evidence** | **3,750** | **1,281** |

Refinement keeps the best five *sentences* where plain RAG passes the best five
*passages*, so the verifier sees **66% less text** — and it abstains. CRAG answers
"Not enough evidence" **145 times against a gold count of 35**; 31.0% of its predictions
are an uncertain class against 14.6% in gold, while plain RAG sits at 15.2%, almost
exactly calibrated.

So accuracy falls significantly (**−0.054 [−0.096, −0.012]** against plain RAG) while the
two rare classes gain (Conflicting +0.042, Not enough evidence +0.081, neither individually
significant). Macro-F1 weights all four classes equally, so the two cancel.

**This is the transfer finding.** Refinement was designed for question answering, where a
short answer span suffices and surrounding context is noise. Fact-checking is a different
task in a way that matters: the verdict depends on how much the evidence *establishes*, and
"not enough evidence" is a valid label the model can retreat into. Thinning the evidence by
two thirds makes that retreat the easy option. Refinement is the component of CRAG that
transfers worst, and its failure mode is over-abstention rather than error.

## Our gain against the paper's gain

CRAG reports consistent gains across four QA benchmarks. We find no significant macro-F1
gain on AVeriTeC and a significant accuracy *loss*, with the evaluator and the router both
reproducing successfully and refinement identified as the cause. We read this as a
task-transfer limit rather than a failed reproduction: the components work, and one of them
is wrong for this task.

## Caveats a reader should hold

* **The evaluator's 0.728 is at the ceiling its labels can measure.** A hand-check of 95
  pairs puts label noise at 0.289 (dev-weighted); a classifier perfect against truth would
  score ~0.711 against labels that noisy. Scoring 0.728 means its errors correlate with the
  labeller's — it learned the labelling function, including where that function is wrong.
  The cross-encoder comparison is correspondingly narrowed: part of the +0.161 is a better
  match *to this labelling*, and that cannot be separated from relevance without clean
  labels. The calibration finding stands independently.
* **The thresholds were fitted against that same noisy score.**
* Single seed; every number carries a bootstrap interval over the claim set instead.
* Evidence strips were not deduplicated when these runs were made; the repeats cost 5.2% of
  evidence slots. Fixed afterwards and the runs were not repeated.
