# P2.12 — Reproducing ExpRAG: which findings held

**ExpRAG** (Palmeira Ferraz et al., arXiv 2603.18272) keeps a store of the agent's past
trajectories and uses it twice: the store feeds LoRA fine-tuning, and retrieval-augmented
SFT puts *k* retrieved trajectories into the training context so the model learns to use
precedent. We reproduced both from scratch and applied them to fact-checking.

Setting: Qwen2.5-3B-Instruct, QLoRA (r=16, α=32, 4-bit base), 1,500 trajectories generated
by our CRAG pipeline on AVeriTeC train, evaluated on 300 dev claims. Teacher stays
Qwen2.5-7B for query rewriting; only the final verdict moves to the student.

## The four-row table

| row | adapter | retrieval at inference | macro-F1 | 95% CI | accuracy |
|---|---|---|---|---|---|
| prompted 3B | none | off | 0.243 | [0.192, 0.294] | 0.263 |
| prompted 3B + retrieval | none | **on** | 0.285 | [0.236, 0.332] | 0.333 |
| LoRA SFT | gold targets | off | 0.329 | [0.298, 0.358] | 0.667 |
| retrieval-augmented SFT | gold targets | **on** | 0.332 | [0.299, 0.361] | 0.673 |

## What held

**Fine-tuning on the agent's own trajectories works, and substantially.** Accuracy rises
0.263 → 0.667 and macro-F1 by +0.086. This is the part of ExpRAG that transfers cleanly:
a 3B student trained on 1,500 of its own verifications becomes a usable verifier.

**Retrieval at inference helps a model that was not trained with it** (+0.041, prompted
rows). Past verifications in the context are useful information.

## What did not hold: the central claim

ExpRAG's argument is that training *with* retrieved trajectories teaches the model to use
them, so retrieval-augmented SFT should beat plain SFT by more than retrieval alone buys.
Decomposed against the prompted baseline:

```
retrieval alone   +0.041
training alone    +0.086
both together     +0.088
super-additive    -0.039      (needs > 0 to support ExpRAG)
```

Paired bootstrap on the same 300 claims: **retrieval-aug SFT − plain SFT = +0.002
[−0.021, +0.026]**. The interval is tight, so this is a *measured null*, not an
underpowered test.

The prompted-plus-retrieval row is what makes this falsifiable, and it is the row easiest
to omit. Without it, "retrieval-augmented SFT is the best row" is true and uninformative —
it cannot be distinguished from retrieval merely being available at inference.

## Why it did not hold

Training was configured correctly: all 1,500 examples carried retrieved context
(`n_with_retrieval: 1500`), k≤2, each neighbour capped at 1,000 tokens, each example's own
trajectory excluded from its retrieval. The mechanism fired. The problem is what it
retrieved.

Relative similarity of retrieved neighbours, as a fraction of each claim's own BM25
self-score:

| | value |
|---|---|
| median | **0.203** |
| 75th percentile | 0.275 |
| 90th percentile | 0.419 |
| share above 0.30 | **21.0%** |
| share above 0.50 | 7.3% |

In roughly **four training examples out of five, the two "relevant past verifications"
were only weakly related to the claim**. The model was trained to attend to a context
window that usually held nothing worth attending to, so it learned to discount it — and at
inference, retrieval then adds +0.002. That retrieval still helps the *prompted* student
fits the same explanation: that model was never trained to ignore the context.

**This is a scale finding.** ExpRAG's premise is that an agent's history contains
precedents for new cases, which requires a store dense enough for precedents to exist.
AVeriTeC's claims span politics, health and economics across many countries and years, and
1,500 trajectories over that spread simply do not contain near-neighbours for most claims.
The published setting is narrower. The result predicts where the method *would* work: a
narrower domain, or a store one to two orders of magnitude larger.

## A finding of our own: train toward the gold label

ExpRAG trains on trajectories as the agent produced them. Keeping only the agent's
*correct* verifications starved the minority classes — of 885 usable examples, **5** were
"Conflicting evidence". Those are precisely the claims the agent gets wrong, so filtering
removes exactly the cases the model most needs.

Training toward the claim's **gold label** instead keeps every labelled trajectory (the
(claim, evidence) pair is still real work; only the answer is corrected) and restores the
distribution: Conflicting 5 → 106, Not enough evidence 24 → 130, 78% Refuted → 57%.

| | gold targets | filtered | difference | 95% CI |
|---|---|---|---|---|
| LoRA SFT | 0.329 | 0.302 | **+0.027** | [+0.004, +0.052] |
| retrieval-aug SFT | 0.332 | 0.300 | **+0.032** | [+0.009, +0.056] |

Both significant. This is a deliberate deviation from ExpRAG and it helps.

## The limitation it did not fix

Despite 106 and 130 training examples, **both fine-tuned models score 0.00 F1 on
Conflicting evidence and Not enough evidence** — while the *prompted* model manages 0.13
and 0.14. Fine-tuning collapsed the output onto Supported/Refuted, which is why accuracy
reaches 0.667 while macro-F1 only reaches 0.33.

So the gold-target fix improved the data without surviving training. At 7% of examples and
two epochs, the model still learns to ignore the rare classes. Reported as a limitation:
the honest reading is that fine-tuning bought accuracy by abandoning the uncertain labels,
and a deployment that must distinguish "conflicting" from "refuted" would not be served by
this adapter.

## Caveats

* 300 dev claims, single seed; every number carries a bootstrap interval.
* Evaluated on an AVeriTeC dev subset, **not** a held-out-*topic* split — that split was
  planned (P0.9) and never built, and the table says what was actually done.
* The trajectory store is 1,500 examples because that is what one Kaggle session produced;
  the similarity finding above is partly a statement about that budget.
