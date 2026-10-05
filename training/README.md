# Training the VeriMem verifier

**The real 3B runs go on GPU-A (RTX 4070, 8 GB).** GPU-B (RTX 4050, 6 GB) is for the
Flan-T5 evaluator, embeddings and baselines — it can run the short smoke test, but it does
not have the memory for 3B training at full sequence length. Linux or WSL2. Never the Mac.

Output is a **LoRA adapter** (tens of MB), not a merged model, so it is small enough to
attach to a GitHub release or push to the Hub.

**Measured on GPU-A, 4 Oct 2026:** Unsloth, 4,096-token examples, **5,318 MiB peak of
8,188** — it fits, with about 35% headroom. The OOM ladder below should not be needed.

## 1. One-time setup (P0.12)

```bash
git clone https://github.com/TERABYTE05/VeriMem.git && cd VeriMem
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-gpu.txt       # pulls torch, peft, trl, bitsandbytes, unsloth, vllm
nvidia-smi                                # confirm the GPU and driver
```

**Unsloth matters more than it looks.** Qwen2.5 has a ~152k vocabulary, so at long
sequence lengths the cross-entropy logits tensor dominates memory: at 4,096 tokens it is
roughly 2.4 GB, and about double that once the backward pass needs its gradient. Unsloth's
fused cross-entropy never materialises that tensor. With `--backend peft` the 3B
retrieval-augmented run is not expected to fit in 8 GB at all; with Unsloth it should fit
comfortably. If Unsloth will not install, say so before falling back to `--backend peft` —
the fallback is fine for short-sequence smoke tests, not for the real runs.

## 2. Smoke test — do this first (P2.8)

This runs against a **synthetic fixture committed to the repo**, so it works before any
real trajectories exist. It proves CUDA, 4-bit loading, LoRA and checkpointing all work.
50 examples, 10 steps, a few minutes.

```bash
python -m training.sft --data training/fixtures/sample_trajectories.jsonl \
    --name smoke --smoke
```

Expected: a `results/<date>_smoke/adapter/` directory containing `adapter_model.safetensors`.
The model it produces is worthless — the fixture is synthetic nonsense. Only the fact
that it ran matters.

Check the data pipeline alone, with no GPU and no download:

```bash
python -m training.sft --data training/fixtures/sample_trajectories.jsonl \
    --name check --mode retrieval --dry-run
```

## 2b. Headroom test — do this too

The smoke test above runs at roughly **500 tokens** per example. The real runs go up to
**4,096**. Memory scales with sequence length, so a comfortable smoke test does not prove
the real run fits. This fixture is built to sit right at the cap (~4,070 tokens):

```bash
python -m training.make_stress_fixture      # writes data/stress_trajectories.jsonl
python -m training.sft --data data/stress_trajectories.jsonl \
    --name headroom --mode retrieval --limit 20 --max-steps 10
```

Watch peak VRAM in another terminal while it runs:

```bash
watch -n 1 nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

**This number is the one that decides whether the real run fits.** Report it. On GPU-A it
came back at 5,318 MiB with Unsloth — if yours is far above that, something differs.

If it OOMs, find the ceiling rather than guessing — one command:

```bash
for S in 1024 2048 3072 4096; do
  echo "=== seq $S ==="
  python -m training.sft --data data/stress_trajectories.jsonl \
      --name "headroom-$S" --mode retrieval --limit 8 --max-steps 4 \
      --max-seq-length $S 2>&1 | tail -3
done
```

Report the largest `$S` that completes. That number sets `max_seq_length` for the real
runs, and the dataset builder re-budgets evidence and retrieved context to fit it.

## 3. The real runs (P2.9, P2.10)

Both need `data/trajectories.jsonl` from the trajectory run (P2.4). Same command, one
flag apart — that flag is the entire G2 comparison, so do not change anything else
between them.

```bash
# P2.9  plain LoRA SFT
python -m training.sft --data data/trajectories.jsonl --name lora-sft --mode plain

# P2.10 retrieval-augmented SFT (k<=2 past trajectories in context, self excluded)
python -m training.sft --data data/trajectories.jsonl --name rag-sft --mode retrieval
```

Resume a killed run — checkpoints land every 200 steps:

```bash
python -m training.sft --data data/trajectories.jsonl --name lora-sft --mode plain --resume
```

### If you run out of VRAM

In this order, and **say which one you used in `notes.md`** — these change the numbers:

1. `--max-seq-length 3072` (cheapest; the dataset builder re-budgets automatically)
2. `--grad-accum 32` with batch 1 (same effective batch, less peak memory)
3. `--lora-r 8`
4. On GPU-B only: `--model Qwen/Qwen2.5-1.5B-Instruct` — this is the planned fallback if
   3B will not fit, not a free choice. Flag it to Teesha before using it: it changes which
   model the whole results table is about.

### Useful flags

| Flag | Default | Why you would change it |
|---|---|---|
| `--mode plain\|retrieval` | `plain` | the P2.9 vs P2.10 comparison |
| `--keep-incorrect` | off | by default only verifications that matched gold are trained on |
| `--limit N` | all | quick partial run |
| `--backend peft\|unsloth` | auto | if Unsloth will not install |
| `--no-4bit` | 4-bit on | only if you have far more VRAM than we do |
| `--dry-run` | off | build the dataset and stop |

## 4. Sending the adapter back

Every run writes `results/<date>_<name>/` with `config.yaml`, `metrics.json`,
`dataset_stats.json`, `notes.md` and `adapter/`. **Commit the small files, not the
adapter** — `.gitignore` already blocks `*.safetensors` under `results/`.

```bash
# fill in notes.md first: VRAM used, wall-clock, any flag you changed
git add results/<date>_<name>/config.yaml results/<date>_<name>/metrics.json \
        results/<date>_<name>/notes.md results/<date>_<name>/dataset_stats.json
git commit -m "P2.9 plain LoRA SFT run" && git push
```

Then send the adapter itself one of these ways:

```bash
# preferred: Hugging Face Hub (private repo)
huggingface-cli upload <user>/verimem-lora-sft results/<date>_<name>/adapter

# or attach the zipped adapter to a GitHub release
cd results/<date>_<name> && zip -r adapter.zip adapter
gh release create lora-sft-v1 adapter.zip -n "P2.9 plain LoRA SFT adapter"
```

## 5. Running it on the Mac

The adapter is useless without its base model, and the Mac has 8 GB. What works:

```bash
pip install mlx-lm
# fuse the adapter into the base, then quantise to 4-bit for the Mac
mlx_lm.fuse --model Qwen/Qwen2.5-3B-Instruct --adapter-path ./adapter --save-path ./verimem-3b
mlx_lm.convert --hf-path ./verimem-3b --mlx-path ./verimem-3b-4bit -q
mlx_lm.generate --model ./verimem-3b-4bit --prompt "$(cat a_claim.txt)" --max-tokens 128
```

**Keep the context short.** A 4-bit 3B is roughly 2 GB of weights, which fits, but the
full retrieval-augmented prompt is up to 4,096 tokens, and a context that long on 8 GB of
unified memory will swap. So:

- **single claims, short prompts, demo and debugging** — fine on the Mac;
- **bulk evaluation (P2.11, P4.1, P4.3)** — GPU-A, never the Mac;
- **the Gradio demo (P4.7)** — runs from cached results by design, so it needs no model
  on the Mac at all.

Budget ~6 GB of disk for the fused model and allow several minutes for the fuse step.
