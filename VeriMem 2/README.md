# VeriMem

A fact-checking agent that grades its evidence and learns from its own past verifications.

**In:** a claim. **Out:** a verdict (Supported / Refuted / Conflicting evidence / Not enough
evidence) with an evidence trail — sources, relevance grades, source trust scores, and the past
verifications it drew on.

It combines two papers, both reproduced from scratch and transferred to fact-checking: **CRAG**
(a fine-tuned T5 evaluator grades retrieved evidence and routes to refine / rewrite+search / both)
and **ExpRAG** (a store of past trajectories feeds retrieval-augmented LoRA fine-tuning). On top of
those we add evaluator-gated experience retrieval, Bayesian per-domain source trust, and a
memory-poisoning study.

NLP course project, IIT Bhilai. The schedule and task breakdown are kept outside this
repo; ask Teesha for the current plan.

## Setup

```bash
make setup          # python3.11 venv + CPU requirements + .env from the template
$EDITOR .env        # API key, search key, spend cap
make test
```

Python 3.11. The Mac is a dev machine only. GPU work runs on the two laptops under Linux or WSL2
(`pip install -r requirements-gpu.txt`), with Kaggle/Colab as backup:

| Machine | Role |
|---|---|
| Mac M3, 8 GB | code, agent graph, caching, search calls, FAISS, tests on <= 10 claims, report, demo from cache |
| GPU-A: RTX 4070, 8 GB VRAM | LoRA/QLoRA on Qwen2.5-3B, bulk evaluation of the fine-tuned verifier |
| GPU-B: RTX 4050, 6 GB VRAM | Flan-T5 evaluator training, embeddings, baselines, prompted 3B in 4-bit |

The teacher/judge is one hosted Qwen model over an OpenAI-compatible endpoint, fixed in P0.13.

## Layout

| Path | What lives there |
|---|---|
| `core/` | paths, device selection, run config, the hashed cache |
| `retrieval/` | web search, page fetch, strip refinement, source tagging, domain blocklist |
| `evaluator/` | T5 relevance evaluator: data building, training, thresholds |
| `experience/` | trajectory format, BM25 trajectory store, gating and trust filters |
| `training/` | LoRA SFT, retrieval-augmented SFT, and serving an adapter (GPU only) |
| `agent/` | the CRAG pipeline and the config flags |
| `attacks/` | memory + source poisoning sets (fixed seed) and propagation pairs |
| `eval/` | metrics, bootstrap intervals, experiment runners |
| `verimem/` | the single entry point: `python -m verimem "<claim>"` |
| `report/` | figures, generated from `results/` |
| `handoff/` | self-contained bundles for running an experiment on another machine |
| `demo/` | Gradio UI (not built yet) |
| `cache/`, `data/`, `results/` | gitignored working dirs; `results/` keeps config + metrics |

Two of these differ from the original plan, deliberately and for reasons worth stating:

- **`agent/` is plain Python, not LangGraph.** The pipeline is a fixed sequence with a
  single branch — the router choosing refine, rewrite, or both — and a graph framework
  would add a dependency and an abstraction layer without removing either. The shared
  state the plan wanted exists as `CragResult` and `Trajectory`.
- **`experience/` retrieves with BM25, not FAISS over sentence embeddings.** BM25 is
  already shared with passage retrieval, needs no model on an 8 GB machine, and is
  deterministic — which matters when the poisoning study compares runs that differ only
  in the store's contents. `Retriever` is a protocol, so FAISS can be dropped in without
  touching callers. Nothing in C1–C3 depends on which retriever is underneath.

## Caching

Every LLM call, search and page fetch goes through `core.cache`, keyed by a SHA-256 of its inputs.
A repeat call with identical inputs must never hit the network.

```python
from core.cache import get_cache

cache = get_cache("search")
hits = cache.get_or_compute({"q": claim, "k": 5}, lambda: provider.search(claim, k=5))
```

`make cache-stats` shows how many records each namespace holds; `core.cache.STATS` reports hits,
misses and hit rate for the current process.


## Running things

One command per experiment. All of them write `results/<date>_<name>/` with a
`config.yaml` (including the verifier prompt's fingerprint), `metrics.json` and notes.

**Fact-check one claim** — the whole system, with the evidence trail:

```bash
python -m verimem "The Kochi metro opened to the public in 2017."
python -m verimem --claim-id 192            # take a claim from the dev split, score it against gold
python -m verimem --claim-id 192 --no-gate  # ablate C2 live
```

**The baselines and the CRAG table** (G1):

```bash
python -m eval.run_baselines --data data/averitec/dev.json \
    --store data/averitec/pool_dev.jsonl \
    --evaluator results/<date>_t5-evaluator/model \
    --system all --workers 8
```

**The ExpRAG four-row table** (G2) — add `--verifier-adapter` for a fine-tuned row,
`--verifier-base` for the prompted student, `--experience-retrieval` for the retrieval rows.

**Memory poisoning** (C1, H1/H2) — `--adversary adaptive` is the one that defeats trust:

```bash
python -m eval.run_poisoning --data data/averitec/dev.json \
    --store data/averitec/pool_dev.jsonl \
    --trajectories data/trajectories/trajectories.jsonl \
    --evaluator results/<date>_t5-evaluator/model \
    --targets 100 --adversary naive --workers 8
```

**Source poisoning** and **propagation**:

```bash
python -m eval.run_source_poisoning  ... --pages 0 1 3 5
python -m eval.run_propagation       ... --rate 0.2 --per-target 1
```

**Tables and figures for the report:**

```bash
python -m eval.report_tables --run results/<date>_g1-rerun     # with bootstrap intervals
python -m report.make_figures                                   # PNGs into report/figures/
```

**Training** (GPU only):

```bash
python -m evaluator.train --data data/evaluator --name t5-evaluator
python -m training.sft --data data/trajectories/trajectories.jsonl \
    data/trajectories/trajectories_failed.jsonl \
    --name lora-sft-gold --mode plain --gold-targets
```

Every runner takes `--dry-run` to report what it would do without calling anything, and
`--workers N` to run claims concurrently — almost all the wall-clock is spent waiting on
the endpoint, so 8 workers is roughly a 4x speedup. Add `OLLAMA_NUM_PARALLEL=8` on the
server too, or the requests simply queue inside it.

## Reproducing on another machine

`handoff/c1-part2/` is a worked example: a README, a setup script, a preflight that runs
two claims end to end before committing hours, and a run script that packages its results.
`scripts/kaggle_run.py` pushes a notebook to Kaggle and reads its log back.
