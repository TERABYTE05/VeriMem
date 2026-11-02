# 2026-11-02_cross-encoder-baseline — P1.4, the comparison the evaluator had to win

`cross-encoder/ms-marco-MiniLM-L-6-v2`, off the shelf, on the same 751 held-out relevance
pairs as the trained evaluator. Same labels, same split, same metric definitions.

| | accuracy | AUC | 95% CI |
|---|---|---|---|
| off-the-shelf cross-encoder | 0.5726 | 0.6498 | [0.611, 0.686] |
| **fine-tuned Flan-T5-base** | **0.7284** | **0.8107** | — |
| difference | +0.156 | **+0.161** | far outside the baseline's interval |

**Fine-tuning was worth it**, decisively. The trained evaluator's 0.811 sits well above the
top of the baseline's confidence interval, so this is not a close call. The Oct 27 decision
point — "if the trained evaluator does not beat the cross-encoder, report both and use the
better one" — resolves in favour of keeping the T5 downstream.

## The more interesting half

The baseline's accuracy is **0.573 against a majority-class floor of 0.531** (399 of 751
pairs are positive). It is barely better than answering "relevant" every time, while its
AUC of 0.650 shows it does carry real ranking signal.

Those two facts together say the problem is **calibration, not ranking**: mean score on
positives is **−0.296**, so the model rates even genuinely relevant passages as
irrelevant. MS MARCO relevance — does this passage answer this query — is not the same
question as "does this passage bear on judging this claim", and the pretrained decision
boundary lands in the wrong place for ours.

That matters for CRAG specifically. The router reads two thresholds off this score, so a
model whose boundary is misplaced would send almost every claim down the same branch
regardless of how well it ranks. Fine-tuning moved the boundary, not just the ordering.

## Reproduce

```bash
python -m evaluator.cross_encoder \
    --compare-to results/2026-10-26_t5-evaluator/metrics.json \
    --out results/2026-11-02_cross-encoder-baseline/metrics.json
```
