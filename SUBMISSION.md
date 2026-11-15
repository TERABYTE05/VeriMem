# VeriMem — submission

**Poisoning the Memory of a Retrieval-Augmented Fact-Checking Agent**
NLP course project, Indian Institute of Technology Bhilai.

| | |
|---|---|
| Teesha Ramchandani | 12342260 |
| Rahul Raj | 12341680 |
| Rajeev Kumar | 12341700 |
| Durgesh Nama | 12340720 |
| Yogesh | 12342420 |

## The report

**`report/verimem_tai.pdf`** — 6 pages, IEEE Transactions on AI format, 4 figures.

## What this is

A fact-checking agent that keeps a memory of its own past verifications, and a study of
what an adversary gains by writing to that memory. It reproduces two published systems
from scratch — CRAG and ExpRAG — and attacks the result at two surfaces: the agent's
memory, and the evidence corpus it searches.

Headline finding: two defences each look sufficient alone and neither is. Evaluator
gating and Bayesian source trust fail against *different* adversaries, so a study using
only one would have concluded that either alone suffices.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or: pip install torch transformers openai pyyaml python-dotenv
python -m pytest                          # 514 tests, no GPU or network needed
```

One command verifies a claim and prints the full evidence trail:

```bash
python -m verimem "The Kochi metro opened to the public in 2017."
python -m verimem --claim-id 0 --no-gate   # ablate a defence live
```

The interactive demonstration, which is the quickest way to see the result:

```bash
pip install gradio
python -m demo.app
```

Pick a ★ claim, set poisoning to 20%, and untick the two defences. `README.md` has one
command per experiment.

## Layout

| Path | Contents |
|---|---|
| `agent/` | the CRAG pipeline and the configuration flags |
| `retrieval/` | BM25 retrieval, strip refinement, source tagging, domain blocklist |
| `evaluator/` | the T5 relevance evaluator: data building, training, thresholds, baseline |
| `experience/` | trajectory format, the experience store, gating and trust filters |
| `attacks/` | memory poisoning, source poisoning, propagation pairs (generators only) |
| `training/` | LoRA and retrieval-augmented fine-tuning, and serving an adapter |
| `eval/` | metrics, bootstrap intervals, the experiment runners, failure analysis |
| `verimem/` | `python -m verimem "<claim>"` — the single entry point |
| `demo/` | the Gradio demonstration |
| `report/` | the paper, its sources, and the figure generator |
| `results/` | one folder per experiment: config, metrics, notes, per-claim outputs |
| `tests/` | 514 tests |
| `handoff/` | a self-contained bundle for running an experiment on another machine |
| `notebooks/` | the Kaggle entry points used for the GPU runs |

## What is deliberately not in this archive

- **`data/`** — AVeriTeC is a third-party dataset and the rebuilt evidence pool is 85 MB.
  `data/trajectories/` and `data/evaluator/` *are* included, because those are outputs of
  this project rather than redistributed data. The AVeriTeC splits are public; `README.md`
  describes how the pool is rebuilt from the archived pages in the gold annotations.
- **`results/*_t5-evaluator/model/`** — the trained evaluator is 947 MB. Its metrics and
  training notes are included; `python -m evaluator.train` rebuilds it in about 25 minutes
  on a consumer GPU.
- **`cache/`** — about 1 GB of hashed model-call records. Everything re-derives from the
  code; the cache only makes a re-run fast.
- **`.venv/`**, `__pycache__/`.

No fabricated content is included. The attack generators are here; everything they
produce is written to a gitignored directory, assembled in memory for a run, and
discarded (see the ethics statement in the report).

## Where each number in the report comes from

Every figure and table traces to a folder under `results/`, and the figures are generated
from those folders rather than drawn:

```bash
python -m report.make_figures                                 # the four figures
python -m eval.report_tables --run results/2026-10-28_g1-rerun  # tables with intervals
```

Supporting write-ups, which the report draws on:

| File | Covers |
|---|---|
| `report/notes/crag_reproduction.md` | what transferred from CRAG and what did not |
| `report/notes/exprag_reproduction.md` | why ExpRAG's central claim did not replicate |
| `report/notes/failure_analysis.md` | the error taxonomy over all 500 dev claims |
| `report/notes/worked_traces.txt` | three traces, selected by rule rather than by hand |
| `results/2026-11-02_trust-divergence/notes.md` | what the trust model actually learned |
| `results/2026-11-02_label-noise/notes.md` | the label-noise ceiling on the evaluator |
