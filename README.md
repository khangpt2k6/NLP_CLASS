# NLP_CLASS

CAI 5307 (Fall 2026) course project: reproduce and extend OWL, "Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret Sauce for Pruning LLMs to High Sparsity" (Yin et al., ICML 2024, [arXiv 2310.05175](https://arxiv.org/abs/2310.05175), [original code](https://github.com/luuyin/OWL)).

Authors: Tuan Khang Phan, Xuan Gia Han Nguyen (University of South Florida).

## Files
- [proposal/proposal.pdf](proposal/proposal.pdf): project proposal (ACL template, 2 pages + references)
- [owlrepro/](owlrepro/): our port of OWL, Wanda, SparseGPT and magnitude pruning
- [scripts/make_jobs.py](scripts/make_jobs.py): writes the job lists in [experiments/](experiments/)
- [slurm/run_array.sbatch](slurm/run_array.sbatch): runs a job list as a SLURM array on GAIVI
- [scripts/collect.py](scripts/collect.py): gathers `results/*/*.json` into `results/summary.csv`
- [docs/gpu_resources.md](docs/gpu_resources.md): where to get GPUs
- [tests/](tests/): unit tests on tiny random models (CPU, no downloads)

## Setup
```
git clone https://github.com/khangpt2k6/NLP_CLASS.git && cd NLP_CLASS
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # pick the CUDA build that matches the node
pip install -r requirements.txt
huggingface-cli login     # only needed for meta-llama/Llama-3.x (request access on the model page first)
python -m pytest tests -q
```

## One run
```
python -m owlrepro.run --model huggyllama/llama-7b --method wanda --alloc owl \
    --sparsity 0.7 --M 5 --lam 0.08 --zeroshot all --out results/r1/llama7b_wanda_owl_s70_M5_l0.08.json
```
- `--method`: `dense`, `magnitude`, `wanda`, `sparsegpt`
- `--alloc`: `uniform`, `owl`, `owl_inv`, `owl_shuffle`, `er`, `er_plus`, `global` (`owl_shuffle` randomly permutes OWL's block sparsities, seeded by `--seed`)
- `--eval`: any of `wikitext2,c4,viwiki`
- `--lod 5,7`: also log the layerwise outlier distribution before and after pruning
- `--calib viwiki`: calibrate on Vietnamese Wikipedia instead of C4 (E4)
- `--save_model DIR`: also save the pruned checkpoint (for the authors' zero-shot scripts, see below)

Each run writes one JSON file with the arguments, library versions, block sparsities, OWL ratios, perplexities and zero-shot accuracies. If the file exists the run is skipped.

Quick check on a laptop CPU (about 10 minutes, tiny model, few samples):
```
python -m owlrepro.run --model HuggingFaceTB/SmolLM2-135M --method wanda --alloc owl --sparsity 0.5 \
    --nsamples 4 --seqlen 256 --eval wikitext2,c4 --max_eval_windows 16 --lod 5 --out results/smoke/test.json
```

## All experiments on GAIVI
```
python scripts/make_jobs.py          # writes experiments/{r1,r2,r4,e1,e2,e3,e4}.txt
mkdir -p logs
N=$(wc -l < experiments/r1.txt)
sbatch --array=0-$((N-1))%4 slurm/run_array.sbatch experiments/r1.txt
python scripts/collect.py r1
```
Check the GPU type and partition names with `sinfo` first. 7B and 8B models need a 24 GB or larger GPU (A100, L40S, A6000). LLaMA-13B needs about 31 GB. Jobs on the general partition can be preempted, so resubmit the same command if some jobs died.

| Group | What | Jobs |
|---|---|---|
| r1 | Table 3 at 70% on LLaMA-7B, OPT-6.7B, LLaMA-13B (zero-shot of Table 4 on LLaMA-7B) | 18 |
| r2 | Table 8 / Figure 2: 40% to 80% sweep, six allocations, LLaMA-7B + Wanda | 30 |
| r4 | Table 1: LOD before and after pruning | 14 |
| e1 | OWL vs uniform on Qwen2.5 0.5B to 7B and Llama-3.x 1B to 8B, shuffled-OWL control, extra calibration seeds | 120 |
| e2 | Dense LOD of all ten models for M in 3, 5, 7, 10 | 10 |
| e3 | M and lambda grid on Qwen2.5-1.5B and Llama-3.2-1B | 40 |
| e4 | Vietnamese vs English calibration | 18 |

## Kaggle (models up to 3B)
In a notebook with GPU T4 x2 and internet on:
```
!git clone https://github.com/khangpt2k6/NLP_CLASS.git
%cd NLP_CLASS
!pip install -q -r requirements.txt
!python -m owlrepro.run --model Qwen/Qwen2.5-1.5B --method wanda --alloc owl --sparsity 0.7 --dtype fp16 --out results/e1/qwen2.5-1.5b_wanda_owl_s70_M5_l0.08.json
```
T4 has no fast bf16, so pass `--dtype fp16`. Copy `results/` out of `/kaggle/working` before the session ends.

## Status
- 2026-10-17: 28 unit tests pass (torch 2.11, transformers 5.5). CPU smoke runs of Wanda+OWL and SparseGPT+OWL on SmolLM2-135M work end to end, including C4 calibration, WikiText-2/C4 perplexity, LOD logging and lm-eval 0.4.13 zero-shot.
- Next: GAIVI account, then check dense LLaMA-7B WikiText-2 perplexity is 5.68 before running r1.

## Zero-shot exactly as in Table 4
Our `--zeroshot` uses lm-eval 0.4, whose prompts and metrics differ from the authors' pipeline, so it will not match Table 4 number for number. For R3, save the pruned model and run the authors' scripts (in their repo, `zero_shot_benchmark/`) in a separate environment with `lm-eval==0.3.0`:
```
python -m owlrepro.run --model huggyllama/llama-7b --method wanda --alloc owl --sparsity 0.7 --M 5 --lam 0.08 \
    --save_model ckpt/llama7b_wanda_owl_s70 --out results/r3/llama7b_wanda_owl_s70.json
git clone https://github.com/luuyin/OWL.git && cd OWL/zero_shot_benchmark
bash scripts/data_genration/run.sh                                 # writes the request files
# set model=../../ckpt/llama7b_wanda_owl_s70 in scripts/benchmark/run.sh, then
bash scripts/benchmark/run.sh && bash scripts/benchmark/eval.sh
```
Table 4 does not say whether it uses acc or acc_norm, so keep both.

## Differences from the original code
- LLaMA-1 and OPT load the slow tokenizer (`use_fast=False`), as in the original `main.py`. Newer models use the fast one.
- The original calls each block as `layer(x, attention_mask, position_ids)`, which breaks on `transformers` 4.48 and later. We record the arguments the model itself passes to every block (forward pre-hooks) and replay them. A test checks that this gives the same hidden states as a normal forward on LLaMA, Qwen2 and OPT.
- Tokenizers load with `AutoTokenizer`. Model blocks are found for LLaMA, Qwen2, Mistral and OPT.
- The paper says each block's sparsity stays in [S - lambda, S + lambda]. The code (and our port) min-max scales the outlier ratios to a band of width 2 lambda and then shifts the mean to S, so with a skewed LOD some blocks leave that interval. We keep the code's behaviour and log the per-block targets.
- WikiText-2 perplexity uses the test split, as in the code (the paper calls it validation).
- Zero-shot accuracy uses lm-eval-harness 0.4 on the in-memory model. The paper used a pipeline copied from H2O with lm-eval 0.3.0, so absolute accuracies can differ. We report our own dense numbers next to every pruned one.
- Global, ER and ER-plus are not in the released code. Global uses one threshold on Wanda scores of the dense model (estimated from 20M sampled scores) and prunes each matrix without row balancing. ER-plus keeps the last linear layer of the last block dense.
- Default dtype is fp16 for LLaMA-1, Llama-2 and OPT (as in the paper) and bf16 for newer models when the GPU supports it.
