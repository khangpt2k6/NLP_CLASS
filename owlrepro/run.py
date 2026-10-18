"""One experiment = load a model, prune it, evaluate it, write one JSON file.

Example:
    python -m owlrepro.run --model huggyllama/llama-7b --method wanda --alloc owl \
        --sparsity 0.7 --M 5 --lam 0.08 --out results/r1/llama7b_wanda_owl_70.json

If the output file already exists the job is skipped, so a preempted SLURM array can simply be
resubmitted.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch

from . import data
from .evaluate import ZEROSHOT_TASKS, perplexity, zeroshot
from .models import block_sparsity, load_model, load_tokenizer, total_sparsity
from .prune import ALLOCS, METHODS, outlier_pass, prune, summarize_lod


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--method", default="wanda", choices=("dense",) + METHODS)
    p.add_argument("--alloc", default="uniform", choices=ALLOCS)
    p.add_argument("--sparsity", type=float, default=0.7)
    p.add_argument("--M", type=float, default=5.0, help="outlier threshold (Hyper_m in the original code)")
    p.add_argument("--lam", type=float, default=0.08, help="lambda (Lamda in the original code)")
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0, help="calibration sampling seed")
    p.add_argument("--shuffle_seed", type=int, default=0, help="permutation seed for --alloc owl_shuffle")
    p.add_argument("--calib", default="c4", choices=("c4", "viwiki"))
    p.add_argument("--dtype", default="auto", choices=("auto", "fp16", "bf16", "fp32"))
    p.add_argument("--eval", default="wikitext2", help="comma list of wikitext2,c4,viwiki or 'none'")
    p.add_argument("--max_eval_windows", type=int, default=None, help="only for smoke tests")
    p.add_argument("--zeroshot", default="", help="'all' for the 7 paper tasks, or a comma list")
    p.add_argument("--zs_batch_size", type=int, default=8)
    p.add_argument("--zs_limit", type=int, default=None, help="only for smoke tests")
    p.add_argument("--lod", default="", help="comma list of M values: log LOD of the dense and pruned model")
    p.add_argument("--save_model", default="", help="also save the pruned checkpoint here (for the original zero-shot scripts)")
    p.add_argument("--out", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def env_info() -> dict:
    import transformers

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "host": platform.node(),
    }
    try:
        import lm_eval

        info["lm_eval"] = lm_eval.__version__
    except Exception:
        pass
    return info


def main(argv=None) -> dict | None:
    args = parse_args(argv)
    if os.path.exists(args.out) and not args.overwrite:
        print(f"skip, {args.out} exists")
        return None
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    result = {"args": vars(args), "env": env_info(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    t0 = time.time()
    tok = load_tokenizer(args.model)
    model = load_model(args.model, args.dtype, args.seqlen)
    result["dtype"] = str(next(model.parameters()).dtype)
    result["load_seconds"] = time.time() - t0

    need_calib = args.method in ("wanda", "sparsegpt") or args.alloc != "uniform" or args.lod
    calib = data.calibration(args.calib, tok, args.nsamples, args.seqlen, args.seed) if need_calib else None
    lod_ms = [float(m) for m in args.lod.split(",") if m]

    if lod_ms:
        dense = outlier_pass(model, calib, lod_ms)
        result["lod_dense"] = {str(M): v for M, v in dense["lod"].items()}
        result["lod_dense_summary"] = {str(M): summarize_lod(v) for M, v in dense["lod"].items()}
        result["score_means_dense"] = dense["means"]

    if args.method != "dense":
        result["prune"] = prune(model, calib, args.method, args.alloc, args.sparsity, args.M, args.lam, args.shuffle_seed)
    result["sparsity"] = total_sparsity(model)
    result["block_sparsity"] = block_sparsity(model)

    if lod_ms and args.method != "dense":
        # Empirical Study II: count outliers after pruning against the dense model's mean score
        after = outlier_pass(model, calib, lod_ms, ref_means=dense["means"])
        result["lod_pruned"] = {str(M): v for M, v in after["lod"].items()}
        result["lod_pruned_summary"] = {str(M): summarize_lod(v) for M, v in after["lod"].items()}
    del calib

    result["ppl"] = {}
    for name in [e for e in args.eval.split(",") if e and e != "none"]:
        t = time.time()
        ids = data.eval_tokens(name, tok, args.seqlen)
        result["ppl"][name] = perplexity(model, ids, args.seqlen, args.max_eval_windows)
        print(f"{name} ppl {result['ppl'][name]:.3f} ({time.time() - t:.0f}s)")

    if args.zeroshot:
        tasks = ZEROSHOT_TASKS if args.zeroshot == "all" else args.zeroshot.split(",")
        result["zeroshot"] = zeroshot(model, tok, tasks, args.zs_batch_size, args.zs_limit)
        print(f"zero-shot mean acc {result['zeroshot']['mean_acc']}")

    if args.save_model:
        model.save_pretrained(args.save_model)
        tok.save_pretrained(args.save_model)
        result["saved_model"] = args.save_model

    result["total_seconds"] = time.time() - t0
    if torch.cuda.is_available():
        result["max_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=1)
    os.replace(tmp, args.out)
    print(f"sparsity {result['sparsity']:.4f}, wrote {args.out}")
    return result


if __name__ == "__main__":
    main(sys.argv[1:])
