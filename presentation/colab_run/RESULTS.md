# Colab run, 2026-11-11: E1 at 70% and E2 on the seven modern models

42 jobs from `experiments/talk.txt` (7 dense LOD runs for E2, 35 runs for E1), all finished.

## Setup
- Google Colab Pro, one NVIDIA L4 (22 GB), torch 2.14.1+cu130, transformers 5.22.1, lm-eval 0.4.13.
- 3.43 hours of job time in total.
- Calibration: 128 x 2048 tokens from C4, seed 0. OWL with M = 5, lambda = 0.08.
- 38 jobs ran at commit 6b20b31. SparseGPT on Qwen2.5-7B and Llama-3.1-8B ran out of memory on the L4, so those 4 jobs ran at commit b7dbc15, which keeps the calibration activations in CPU RAM. The computation is the same.
- An A100 was requested for those 4 jobs but Colab did not assign one.

## WikiText-2 perplexity at 70% sparsity

dL = log PPL(uniform) - log PPL(OWL). Positive means OWL helps.

| Model | Params | Dense | Wanda | Wanda + OWL | dL | SparseGPT | SparseGPT + OWL | dL |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 0.49 B | 13.08 | 395.26 | 500.15 | -0.24 | 181.48 | 215.97 | -0.17 |
| Qwen2.5-1.5B | 1.54 B | 9.27 | 822.72 | 250.22 | +1.19 | 80.74 | 47.97 | +0.52 |
| Qwen2.5-3B | 3.09 B | 8.03 | 161.40 | 89.13 | +0.59 | 31.73 | 34.49 | -0.08 |
| Qwen2.5-7B | 7.62 B | 6.85 | 74.70 | 48.70 | +0.43 | 20.99 | 19.78 | +0.06 |
| Llama-3.2-1B | 1.24 B | 9.76 | 389.02 | 415.10 | -0.06 | 129.67 | 134.93 | -0.04 |
| Llama-3.2-3B | 3.21 B | 7.82 | 143.47 | 123.41 | +0.15 | 67.64 | 58.08 | +0.15 |
| Llama-3.1-8B | 8.03 B | 6.24 | 109.22 | 79.03 | +0.32 | 39.83 | 33.64 | +0.17 |

## What the numbers show
- With Wanda, OWL helps on 5 of 7 models. It hurts on the smallest model of each family (Qwen2.5-0.5B and Llama-3.2-1B).
- With SparseGPT the effect is smaller. OWL helps on 4 of 7 models and hurts on Qwen2.5-0.5B, Qwen2.5-3B and Llama-3.2-1B.
- The gain does not grow as models shrink, which is what the paper reports for LLaMA-1. In Llama-3.x it grows with size. In Qwen2.5 it peaks at 1.5B and is negative at 0.5B.
- One calibration seed per setting, so small differences (for example -0.04 and +0.06) may be noise. The extra seeds in the full E1 list would settle that.

## E2: does the dense LOD predict the gain (n = 7, exploratory)
Spearman correlation between the unevenness of the dense LOD at M = 5 and dL at 70%.

| Method | LOD measure | rho | bootstrap 95% CI |
|---|---|---|---|
| Wanda | CV across blocks | +0.79 | -0.06 to +1.00 |
| Wanda | U (outer blocks minus middle) | +0.86 | +0.40 to +1.00 |
| SparseGPT | CV across blocks | +0.21 | -0.96 to +1.00 |
| SparseGPT | U | +0.18 | -0.88 to +1.00 |

The three paper models are not in this yet, so the proposal's ten-model version is still open.

## Per-job metrics

From `results/summary.csv` (`python scripts/collect.py`). Times are in minutes, memory is peak GPU allocation in GB.
LOD columns are for M = 5: sum over blocks (percent), CV across blocks, and U (outer blocks minus middle).

| Group | Model | Method | Alloc | Sparsity | PPL | LOD sum | LOD CV | LOD U | Alloc s | Prune s | Min | GPU GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e1 | Llama-3.1-8B | dense | - | 0.0000 | 6.24 | - | - | - | - | - | 2.1 | 17.41 |
| e1 | Llama-3.1-8B | sparsegpt | owl | 0.7000 | 33.64 | 41.05 | 0.234 | -0.098 | 97 | 1005 | 20.5 | 20.27 |
| e1 | Llama-3.1-8B | sparsegpt | uniform | 0.7000 | 39.83 | - | - | - | 0 | 1004 | 19.4 | 18.25 |
| e1 | Llama-3.1-8B | wanda | owl | 0.6999 | 79.03 | 41.05 | 0.234 | -0.098 | 98 | 180 | 6.7 | 20.99 |
| e1 | Llama-3.1-8B | wanda | uniform | 0.7000 | 109.22 | - | - | - | 0 | 180 | 5.0 | 20.99 |
| e1 | Llama-3.2-1B | dense | - | 0.0000 | 9.76 | - | - | - | - | - | 0.6 | 4.76 |
| e1 | Llama-3.2-1B | sparsegpt | owl | 0.7000 | 134.93 | 25.63 | 0.125 | -0.178 | 16 | 169 | 3.7 | 5.37 |
| e1 | Llama-3.2-1B | sparsegpt | uniform | 0.7000 | 129.67 | - | - | - | 0 | 170 | 3.5 | 5.37 |
| e1 | Llama-3.2-1B | wanda | owl | 0.6998 | 415.10 | 25.63 | 0.125 | -0.178 | 16 | 28 | 1.4 | 4.89 |
| e1 | Llama-3.2-1B | wanda | uniform | 0.6998 | 389.02 | - | - | - | 0 | 29 | 1.1 | 4.89 |
| e1 | Llama-3.2-3B | dense | - | 0.0000 | 7.82 | - | - | - | - | - | 1.2 | 8.44 |
| e1 | Llama-3.2-3B | sparsegpt | owl | 0.7000 | 58.08 | 37.41 | 0.199 | -0.122 | 43 | 404 | 8.6 | 10.09 |
| e1 | Llama-3.2-3B | sparsegpt | uniform | 0.7000 | 67.64 | - | - | - | 0 | 402 | 7.8 | 10.09 |
| e1 | Llama-3.2-3B | wanda | owl | 0.6999 | 123.41 | 37.41 | 0.199 | -0.122 | 43 | 78 | 3.1 | 9.86 |
| e1 | Llama-3.2-3B | wanda | uniform | 0.6999 | 143.47 | - | - | - | 0 | 79 | 2.5 | 9.86 |
| e1 | Qwen2.5-0.5B | dense | - | 0.0000 | 13.08 | - | - | - | - | - | 0.5 | 3.83 |
| e1 | Qwen2.5-0.5B | sparsegpt | owl | 0.7000 | 215.97 | 30.46 | 0.111 | -0.130 | 8 | 96 | 2.3 | 3.83 |
| e1 | Qwen2.5-0.5B | sparsegpt | uniform | 0.7000 | 181.48 | - | - | - | 0 | 96 | 2.1 | 3.83 |
| e1 | Qwen2.5-0.5B | wanda | owl | 0.6995 | 500.15 | 30.46 | 0.111 | -0.130 | 8 | 14 | 0.9 | 3.83 |
| e1 | Qwen2.5-0.5B | wanda | uniform | 0.6998 | 395.26 | - | - | - | 0 | 14 | 0.8 | 3.83 |
| e1 | Qwen2.5-1.5B | dense | - | 0.0000 | 9.27 | - | - | - | - | - | 0.8 | 5.78 |
| e1 | Qwen2.5-1.5B | sparsegpt | owl | 0.7000 | 47.97 | 24.79 | 0.227 | 0.171 | 23 | 284 | 5.9 | 5.78 |
| e1 | Qwen2.5-1.5B | sparsegpt | uniform | 0.7000 | 80.74 | - | - | - | 0 | 288 | 5.6 | 5.79 |
| e1 | Qwen2.5-1.5B | wanda | owl | 0.6998 | 250.22 | 24.79 | 0.227 | 0.171 | 23 | 41 | 1.8 | 5.78 |
| e1 | Qwen2.5-1.5B | wanda | uniform | 0.6999 | 822.72 | - | - | - | 0 | 42 | 1.5 | 5.78 |
| e1 | Qwen2.5-3B | dense | - | 0.0000 | 8.03 | - | - | - | - | - | 1.2 | 8.66 |
| e1 | Qwen2.5-3B | sparsegpt | owl | 0.7000 | 34.49 | 37.05 | 0.675 | 0.731 | 43 | 529 | 10.7 | 9.65 |
| e1 | Qwen2.5-3B | sparsegpt | uniform | 0.7000 | 31.73 | - | - | - | 0 | 525 | 9.9 | 9.65 |
| e1 | Qwen2.5-3B | wanda | owl | 0.6998 | 89.13 | 37.05 | 0.675 | 0.731 | 43 | 77 | 3.1 | 8.66 |
| e1 | Qwen2.5-3B | wanda | uniform | 0.6998 | 161.40 | - | - | - | 0 | 78 | 2.4 | 8.66 |
| e1 | Qwen2.5-7B | dense | - | 0.0000 | 6.85 | - | - | - | - | - | 2.2 | 17.09 |
| e1 | Qwen2.5-7B | sparsegpt | owl | 0.7000 | 19.78 | 25.35 | 0.439 | 0.421 | 85 | 1145 | 22.7 | 19.79 |
| e1 | Qwen2.5-7B | sparsegpt | uniform | 0.7000 | 20.99 | - | - | - | 0 | 1150 | 21.7 | 19.79 |
| e1 | Qwen2.5-7B | wanda | owl | 0.6999 | 48.70 | 25.35 | 0.439 | 0.421 | 87 | 158 | 6.0 | 20.04 |
| e1 | Qwen2.5-7B | wanda | uniform | 0.6998 | 74.70 | - | - | - | 0 | 159 | 4.6 | 20.04 |
| e2 | Llama-3.1-8B | dense | - | 0.0000 | - | 41.05 | 0.234 | -0.098 | - | - | 3.6 | 20.27 |
| e2 | Llama-3.2-1B | dense | - | 0.0000 | - | 25.63 | 0.125 | -0.178 | - | - | 0.8 | 4.68 |
| e2 | Llama-3.2-3B | dense | - | 0.0000 | - | 37.41 | 0.199 | -0.122 | - | - | 1.5 | 9.58 |
| e2 | Qwen2.5-0.5B | dense | - | 0.0000 | - | 30.46 | 0.111 | -0.130 | - | - | 0.9 | 1.92 |
| e2 | Qwen2.5-1.5B | dense | - | 0.0000 | - | 24.79 | 0.227 | 0.171 | - | - | 0.9 | 4.68 |
| e2 | Qwen2.5-3B | dense | - | 0.0000 | - | 37.05 | 0.675 | 0.731 | - | - | 1.4 | 8.23 |
| e2 | Qwen2.5-7B | dense | - | 0.0000 | - | 25.35 | 0.439 | 0.421 | - | - | 2.9 | 19.13 |

Total: 42 jobs, 206 minutes (3.43 hours). Peak GPU memory 20.99 GB.

## Files in this folder
- `run_all_executed_colab_L4.ipynb`: the notebook as it ran on Colab, with all outputs and job logs
- `e1_owl_gain_1.png`: dL against model size, per method
- `e2_lod_per_block_1.png`: dense LOD per block for the seven models
- `e2_lod_vs_gain_1.png`, `e2_lod_vs_gain_2.png`: LOD unevenness against dL for Wanda and SparseGPT
- `e1_owl_gain_table.html`, `e2_lod_per_block_table.html`: the tables the notebook printed
- `colab_qwen_run.png`, `colab_e1_table.png`: screenshots of the Colab session

Raw results are in `results/e1` and `results/e2`. All metrics in one place: `results/summary.csv` (one row per job), `results/owl_gain.csv` (uniform vs OWL and dL per model and method), `results/blocks.csv` (per-block sparsity, OWL target and LOD). Per-job logs are in `logs/` on the machine that pulled them (not committed).

## Still to run for the talk
The three paper models (LLaMA-7B, OPT-6.7B, LLaMA-13B): 42 jobs for R1, R2, R4 and their E2 runs. LLaMA-13B needs a GPU with about 31 GB.
