# 2026-11-02_trust-divergence — P3.5, and C3 does not do what it was designed to do

Beta trust fitted on all 1,500 trajectories (954 domains acquire a track record).
Figure: `report/figures/trust_divergence.png`.

## C3 does not separate misinformation domains from legitimate ones

| | n | mean trust | below the 0.3 floor |
|---|---|---|---|
| on AVeriTeC's misinformation list | 20 | 0.380 | 6/20 = 0.300 [0.100, 0.500] |
| every other publisher | 934 | 0.375 | 251/934 = 0.269 [0.241, 0.298] |
| difference | | **+0.005** | intervals overlap entirely |

The misinformation domains score *marginally higher*. There is no separation to speak of.

## What it learned instead

Among the 64 domains with at least ten observations behind them, the ordering is the
diagnosis:

| most trusted | | least trusted | |
|---|---|---|---|
| twitter.com | 0.76 (85 obs) | brandsouthafrica.com | 0.17 (14) |
| theguardian.com | 0.70 (26) | standardmedia.co.ke | 0.20 (11) |
| ncbi.nlm.nih.gov | 0.67 (42) | **cbo.gov** | 0.23 (18) |
| forbes.com | 0.67 (17) | cnn.com | 0.25 (16) |
| britannica.com | 0.65 (13) | politico.com | 0.25 (12) |
| bbc.com | 0.63 (42) | **nature.com** | 0.29 (10) |

**Twitter is the single most trusted domain. Nature and the Congressional Budget Office
sit at the floor.** No credibility model would produce that ordering, and C3's does,
because it is not a credibility model. `BetaTrust.observe` credits or blames every domain
in a trajectory according to whether the *verdict* was correct, so what accumulates is
"citing this domain coincided with getting the answer right". Claims whose evidence is a
CBO report or a Nature paper are hard claims; claims whose evidence is a tweet are often
easy ones — someone said something and it is straightforwardly false.

## What this does to the C1 conclusions

It makes H2's result **narrower and more precise**, and it is consistent with everything
already measured:

* **Against the naive adversary C3 was decisive** (attack success flat at 0.230, zero
  poisoned records kept) — but now demonstrably *not* because it recognised
  misinformation. The fabricated domains were simply **unseen**, and Beta(1, 3) places an
  unseen domain at 0.25, below the floor. C3 is a **novelty detector**, not a credibility
  detector.
* **Against the adaptive adversary C3 failed completely** (0.460, identical to undefended,
  every poisoned record kept). That is exactly what a novelty detector does when the
  adversary stops being novel.
* The two results together are one mechanism rather than two findings.

The honest claim for the report is therefore: *source trust defends against an attacker
using sources the system has not seen before, which is a real and common case, and offers
no protection against one that cites sources it already accepts. It is not a measure of
whether a publisher tells the truth.*

## What would fix it

Trust would have to be credited per *piece of evidence* on whether that evidence supported
the verdict, not per domain on whether the verdict was right — which needs an evidence-level
judgement the pipeline does not currently make. Recorded as future work, not attempted:
changing the trust signal would invalidate every C1 number with five days to freeze.
