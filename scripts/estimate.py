"""Rough run-time estimate for the job lists in experiments/.

    python scripts/estimate.py                         # every group, jobs that have no result yet, on an A100
    python scripts/estimate.py talk --gpu L4           # one group on another GPU
    python scripts/estimate.py talk --gpu L4 --models llama7b opt6.7b
    python scripts/estimate.py e1 --all                # count finished jobs too

The numbers come from a simple cost model, not from measurements: minutes on an A100 that grow with the
parameter count, times a speed factor for the GPU. Treat them as a guide to within a factor of two.
Once three or more results from the same GPU exist in results/, the estimate is rescaled by the median
of measured / estimated time of those jobs, so it gets better as runs finish.
"""
import argparse
import glob
import json
import os
import re
import shlex
import statistics

# parameter counts in billions (Hugging Face safetensors metadata, OPT from its config)
PARAMS_B = {
    "huggyllama/llama-7b": 6.74, "huggyllama/llama-13b": 13.02, "facebook/opt-6.7b": 6.66,
    "Qwen/Qwen2.5-0.5B": 0.49, "Qwen/Qwen2.5-1.5B": 1.54, "Qwen/Qwen2.5-3B": 3.09, "Qwen/Qwen2.5-7B": 7.62,
    "meta-llama/Llama-3.2-1B": 1.24, "meta-llama/Llama-3.2-3B": 3.21, "meta-llama/Llama-3.1-8B": 8.03,
}
SHORT = {
    "huggyllama/llama-7b": "llama7b", "huggyllama/llama-13b": "llama13b", "facebook/opt-6.7b": "opt6.7b",
    "Qwen/Qwen2.5-0.5B": "qwen2.5-0.5b", "Qwen/Qwen2.5-1.5B": "qwen2.5-1.5b", "Qwen/Qwen2.5-3B": "qwen2.5-3b",
    "Qwen/Qwen2.5-7B": "qwen2.5-7b", "meta-llama/Llama-3.2-1B": "llama3.2-1b",
    "meta-llama/Llama-3.2-3B": "llama3.2-3b", "meta-llama/Llama-3.1-8B": "llama3.1-8b",
}
# how many times slower than an A100 for fp16 inference (guesses, the rescaling above corrects them)
GPU_SLOWDOWN = {"H100": 0.6, "A100": 1.0, "4090": 1.0, "L40S": 1.1, "L40": 1.2, "3090": 1.5, "A6000": 1.5,
                "A40": 1.6, "V100": 2.2, "L4": 3.0, "T4": 4.5, "P100": 5.0}
UNKNOWN_GPU_SLOWDOWN = 1.5
GROUPS = ["talk", "r1", "r2", "r4", "e1", "e2", "e3"]

# A100 minutes per billion parameters
PASS = 0.12        # one forward of the 128 x 2048 calibration set through every block
EVAL = {"wikitext2": 0.12, "c4": 0.18, "viwiki": 0.15}
SPARSEGPT_SOLVE = 0.6
ZEROSHOT = 2.2     # the seven lm-eval tasks
DOWNLOAD = 0.57    # first download of the weights at about 60 MB/s


def gpu_slowdown(name: str) -> float:
    tokens = re.split(r"[^A-Z0-9]+", (name or "").upper())
    for key, factor in GPU_SLOWDOWN.items():
        if key in tokens:
            return factor
    return UNKNOWN_GPU_SLOWDOWN


def job_args(line: str) -> dict:
    parts = shlex.split(line)
    return dict(zip(parts[::2], parts[1::2]))


def job_minutes(a: dict, slowdown: float = 1.0) -> float:
    """Estimated minutes of one job. `a` maps flags such as "--model" to their values."""
    p = PARAMS_B.get(a["--model"], 7.0)
    method, alloc = a.get("--method", "wanda"), a.get("--alloc", "uniform")
    lod = bool(a.get("--lod"))
    size = (int(a.get("--nsamples", 128)) * int(a.get("--seqlen", 2048))) / (128 * 2048)

    passes, gpu = 0.0, 0.0
    if lod:
        passes += 1                      # LOD of the dense model
    if method != "dense":
        if alloc in ("owl", "owl_inv", "owl_shuffle", "global"):
            passes += 1                  # outlier pass that sets the block sparsities
        if method == "wanda":
            passes += 2                  # each block runs before and after it is pruned
            gpu += 0.05 * p
        elif method == "sparsegpt":
            passes += 3                  # as Wanda, and the Hessian hooks cost more than input norms
            gpu += SPARSEGPT_SOLVE * p
        else:
            gpu += 0.03 * p              # magnitude: one threshold per matrix
        if lod:
            passes += 1                  # LOD of the pruned model
    gpu += passes * PASS * p * size

    evals = [e for e in a.get("--eval", "wikitext2").split(",") if e and e != "none"]
    gpu += sum(EVAL.get(e, 0.15) for e in evals) * p

    fixed = 0.5 + 0.15 * p               # start Python, load the weights from disk
    if passes:
        fixed += 0.7                     # calibration data
    if evals:
        fixed += 0.3
    if a.get("--zeroshot"):
        gpu += ZEROSHOT * p
        fixed += 3.0                     # task download and tokenisation
    if a.get("--save_model"):
        fixed += 0.3 * p
    return fixed + gpu * slowdown


def is_done(a: dict, root: str = ".") -> bool:
    """A job is done when its JSON exists and holds zero-shot results if the job asks for them."""
    path = os.path.join(root, a["--out"])
    if not os.path.exists(path):
        return False
    if not a.get("--zeroshot"):
        return True
    try:
        return bool(json.load(open(path, encoding="utf-8")).get("zeroshot"))
    except (OSError, ValueError):
        return False


def measured_scale(gpu_name: str, root: str = ".") -> tuple[float, int]:
    """Median of measured / estimated minutes over finished jobs on this GPU: (scale, number of jobs)."""
    if not gpu_name:
        return 1.0, 0
    slow = gpu_slowdown(gpu_name)
    ratios = []
    for path in glob.glob(os.path.join(root, "results", "*", "*.json")):
        try:
            r = json.load(open(path, encoding="utf-8"))
            args, gpu, secs = r["args"], r["env"].get("gpu") or "", r["total_seconds"]
        except (OSError, ValueError, KeyError):
            continue
        if args.get("model") not in PARAMS_B or gpu_slowdown(gpu) != slow or gpu_name.upper() not in gpu.upper():
            continue
        a = {f"--{k}": str(v) for k, v in args.items() if v not in (None, "", False)}
        ratios.append(secs / 60 / job_minutes(a, slow))
    if len(ratios) < 3:
        return 1.0, len(ratios)
    return min(5.0, max(0.2, statistics.median(ratios))), len(ratios)


def estimate(lines, gpu_name: str = "A100", root: str = ".") -> dict:
    """Total for a list of job lines: minutes of compute, minutes of first-time downloads, per-job minutes."""
    slow = gpu_slowdown(gpu_name)
    scale, n = measured_scale(gpu_name, root)
    per_job = [job_minutes(job_args(l), slow) * scale for l in lines]
    models = {job_args(l)["--model"] for l in lines}
    download = sum(DOWNLOAD * PARAMS_B.get(m, 7.0) for m in models)
    return {"minutes": sum(per_job), "download_minutes": download, "per_job": per_job, "scale": scale,
            "calibrated_on": n, "slowdown": slow}


def group_lines(group: str, root: str = ".") -> list[str]:
    path = os.path.join(root, "experiments", f"{group}.txt")
    return [l.strip() for l in open(path, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("groups", nargs="*", default=GROUPS)
    ap.add_argument("--gpu", default="A100", help="GPU name, for example A100, L4, T4, L40S, 4090")
    ap.add_argument("--models", nargs="*", help="short names from make_jobs.py, for example llama7b qwen2.5-3b")
    ap.add_argument("--all", action="store_true", help="count jobs that already have a result too")
    args = ap.parse_args()

    slow = gpu_slowdown(args.gpu)
    known = any(k in re.split(r"[^A-Z0-9]+", args.gpu.upper()) for k in GPU_SLOWDOWN)
    print(f"GPU {args.gpu}: {slow:g}x the time of an A100" + ("" if known else " (unknown GPU, default factor)"))
    print(f"{'group':6} {'jobs':>5} {'to run':>7} {'hours':>7} {'download min':>13}")
    for g in args.groups:
        lines = group_lines(g)
        if args.models:
            lines = [l for l in lines if SHORT.get(job_args(l)["--model"]) in args.models]
        todo = lines if args.all else [l for l in lines if not is_done(job_args(l))]
        e = estimate(todo, args.gpu)
        print(f"{g:6} {len(lines):5d} {len(todo):7d} {e['minutes'] / 60:7.1f} {e['download_minutes']:13.0f}")
    if e["calibrated_on"] >= 3:
        print(f"rescaled by {e['scale']:.2f} from {e['calibrated_on']} finished jobs on this GPU")
    else:
        print("not calibrated yet: no finished jobs on this GPU, so expect up to 2x off")
    print("talk is a subset of the other groups, so do not add it to them. Downloads happen once per machine,")
    print("or once per Colab session.")


if __name__ == "__main__":
    main()
