# 2026-10-04_smoke

Mode **plain** · base `Qwen/Qwen2.5-3B-Instruct` · LoRA r=16, alpha=32
seq<=4096 · batch 1 x accum 16 · seed 13

- Trajectories: `training/fixtures/sample_trajectories.jsonl`
- Examples: 48 (2 skipped as incorrect verifications)
- With retrieved context: 0 · trimmed to fit: 0
- Mean ~516.4 tokens, max ~703
- Adapter: `/home/rajeev-kumar/Desktop/NLP/VeriMem/results/2026-10-04_smoke/adapter`

Fill in after evaluating: held-out-topic macro-F1, in-distribution macro-F1, and how this
compares with the previous run.
