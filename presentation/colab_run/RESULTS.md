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

## Files in this folder
- `run_all_executed_colab_L4.ipynb`: the notebook as it ran on Colab, with all outputs and job logs
- `e1_owl_gain_1.png`: dL against model size, per method
- `e2_lod_per_block_1.png`: dense LOD per block for the seven models
- `e2_lod_vs_gain_1.png`, `e2_lod_vs_gain_2.png`: LOD unevenness against dL for Wanda and SparseGPT
- `e1_owl_gain_table.html`, `e2_lod_per_block_table.html`: the tables the notebook printed
- `colab_qwen_run.png`, `colab_e1_table.png`: screenshots of the Colab session

Raw results are in `results/e1` and `results/e2`. Per-job logs are in `logs/` on the machine that pulled them (not committed).

## Still to run for the talk
The three paper models (LLaMA-7B, OPT-6.7B, LLaMA-13B): 42 jobs for R1, R2, R4 and their E2 runs. LLaMA-13B needs a GPU with about 31 GB.
