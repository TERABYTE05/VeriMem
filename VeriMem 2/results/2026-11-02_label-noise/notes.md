# 2026-11-02_label-noise — P1.2, and what it does to the evaluator's numbers

95 of 100 sampled pairs hand-checked by a human with the automatic label hidden
(`python -m evaluator.label_check`). 5 skipped.

## Disagreement with the automatic label

| kind | disagree | rate | 95% CI | share of dev |
|---|---|---|---|---|
| positive | 9/25 | **0.360** | [0.160, 0.560] | 53% |
| hard negative | 17/60 | 0.283 | [0.167, 0.400] | 35% |
| easy negative | 0/10 | 0.000 | — | 12% |
| **raw (sample-weighted)** | 26/95 | **0.274** | [0.189, 0.368] | — |
| **dev-weighted** | — | **0.289** | — | — |

**Report the dev-weighted figure, not 0.274.** The sample deliberately over-sampled hard
negatives — 63% of the sample against 35% of dev — because that is where mislabels were
expected. It therefore *under*-weights positives, which turned out to be the noisiest
class, so the raw rate understates the noise actually present in the evaluation set.

## What it means for the evaluator

A classifier that is perfect against the truth would score about **0.711** accuracy
against labels this noisy. The trained evaluator scored **0.728** — about 0.017 *above*
that ceiling.

Scoring above the ceiling is not an impossibility; it is a diagnosis. It means the
evaluator's errors are **correlated with the labeller's errors**, which is exactly what
training on those labels produces. The model has learned the labelling function, including
the parts of it that are wrong.

So `accuracy 0.728` cannot be read as "right about relevance 73% of the time". It reads as
"agrees 73% of the time with a labelling that is itself about 29% wrong", and the
evaluator has **saturated what these labels are able to measure**. A better evaluator
would not score higher here; it would score lower.

## What this does and does not undermine

**Weakened — the cross-encoder comparison.** The off-the-shelf model lost by 0.161 AUC,
but part of that gap is the fine-tuned model having learned this labeller's idiosyncrasies
rather than relevance as such. The two cannot be separated without clean labels, so the
honest claim is narrower: fine-tuning produces a scorer better matched *to this labelling*,
and the calibration finding (the baseline's boundary is in the wrong place, mean score on
positives −0.296) stands independently.

**Weakened — the threshold tuning (P1.7).** CRAG's two thresholds were fitted against this
score, so they are tuned on a noisy signal.

**Not affected — C1, the headline contribution.** Attack success is measured against the
claims' gold verdicts, not against relevance labels. Memory poisoning, source poisoning,
the defence ablations and the adaptive adversary are all unaffected by this.

## Why the positives are the worst class

Positives are passages from the page the fact-checker actually cited, ranked within that
page by BM25. A cited page contains a great deal of material, only some of which bears on
the claim — so "appeared on a cited page" is a weak proxy for "is evidence for this claim",
and a third of the time it is simply wrong. The earlier fix (ranking within the page rather
than taking its first three passages) improved this without solving it.

Fixing it properly would mean a relevance judgement per passage, which is the labelling
problem itself. Recorded as a limitation rather than solved.
