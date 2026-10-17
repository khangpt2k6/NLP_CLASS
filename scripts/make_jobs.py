"""Write one job file per experiment group (R1 to R4, E1 to E4) into experiments/.

Each line holds the arguments of one `python -m owlrepro.run` call. Run:
    python scripts/make_jobs.py
"""
import os

# short name -> (HF id, M, lambda). M and lambda for LLaMA-1 and OPT are from Table 17 of the paper.
MODELS = {
    "llama7b": ("huggyllama/llama-7b", 5, 0.08),
    "llama13b": ("huggyllama/llama-13b", 7, 0.08),
    "opt6.7b": ("facebook/opt-6.7b", 10, 0.08),
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B", 5, 0.08),
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B", 5, 0.08),
    "qwen2.5-3b": ("Qwen/Qwen2.5-3B", 5, 0.08),
    "qwen2.5-7b": ("Qwen/Qwen2.5-7B", 5, 0.08),
    "llama3.2-1b": ("meta-llama/Llama-3.2-1B", 5, 0.08),
    "llama3.2-3b": ("meta-llama/Llama-3.2-3B", 5, 0.08),
    "llama3.1-8b": ("meta-llama/Llama-3.1-8B", 5, 0.08),
}
NEW_MODELS = ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b", "llama3.2-1b", "llama3.2-3b", "llama3.1-8b"]


def job(group, short, method, alloc="uniform", s=0.7, M=None, lam=None, calib="c4",
        evals="wikitext2", zs=False, lod="", tag=""):
    name, M0, lam0 = MODELS[short]
    M = M0 if M is None else M
    lam = lam0 if lam is None else lam
    parts = [short, method]
    if method != "dense":
        parts += [alloc, f"s{int(round(s * 100))}"]
        if alloc in ("owl", "owl_inv"):
            parts += [f"M{M:g}", f"l{lam:g}"]
    if calib != "c4":
        parts.append(calib)
    if tag:
        parts.append(tag)
    out = f"results/{group}/{'_'.join(parts)}.json"
    args = [f"--model {name}", f"--method {method}", f"--alloc {alloc}", f"--sparsity {s}",
            f"--M {M:g}", f"--lam {lam:g}", f"--calib {calib}", f"--eval {evals}"]
    if zs:
        args.append("--zeroshot all")
    if lod:
        args.append(f"--lod {lod}")
    args.append(f"--out {out}")
    return " ".join(args)


def r1():
    jobs = []
    for short in ["llama7b", "opt6.7b", "llama13b"]:
        zs = short == "llama7b"  # R3 is the zero-shot part of R1 on LLaMA-7B
        jobs.append(job("r1", short, "dense", zs=zs))
        jobs.append(job("r1", short, "magnitude", zs=zs))
        for method in ["wanda", "sparsegpt"]:
            for alloc in ["uniform", "owl"]:
                jobs.append(job("r1", short, method, alloc, zs=zs))
    return jobs


def r2():
    return [job("r2", "llama7b", "wanda", alloc, s)
            for s in [0.4, 0.5, 0.6, 0.7, 0.8]
            for alloc in ["uniform", "owl", "global", "er", "er_plus", "owl_inv"]]


def r4():
    jobs = []
    for short in ["llama7b", "llama13b"]:
        jobs.append(job("r4", short, "dense", lod="5,7"))
        for s in [0.6, 0.7]:
            for method in ["magnitude", "wanda", "sparsegpt"]:
                jobs.append(job("r4", short, method, "uniform", s, lod="5,7", tag="lod"))
    return jobs


def e1():
    jobs = []
    for short in NEW_MODELS:
        jobs.append(job("e1", short, "dense", zs=True))
        for s in [0.5, 0.6, 0.7]:
            for method in ["wanda", "sparsegpt"]:
                for alloc in ["uniform", "owl"]:
                    jobs.append(job("e1", short, method, alloc, s, zs=(s == 0.7)))
    return jobs


def e2():
    return [job("e2", short, "dense", lod="3,5,7,10", evals="none") for short in MODELS]


def e3():
    return [job("e3", short, "wanda", "owl", 0.7, M=M, lam=lam, evals="c4,wikitext2")
            for short in ["qwen2.5-1.5b", "llama3.2-1b"]
            for M in [3, 5, 7, 10]
            for lam in [0.02, 0.05, 0.08, 0.1, 0.2]]


def e4():
    jobs = []
    for short in ["qwen2.5-1.5b", "qwen2.5-7b"]:
        jobs.append(job("e4", short, "dense", evals="wikitext2,viwiki"))
        for calib in ["c4", "viwiki"]:
            for s in [0.6, 0.7]:
                for alloc in ["uniform", "owl"]:
                    jobs.append(job("e4", short, "wanda", alloc, s, calib=calib, evals="wikitext2,viwiki", lod="5"))
    return jobs


def main():
    os.makedirs("experiments", exist_ok=True)
    groups = {"r1": r1(), "r2": r2(), "r4": r4(), "e1": e1(), "e2": e2(), "e3": e3(), "e4": e4()}
    total = 0
    for g, jobs in groups.items():
        with open(f"experiments/{g}.txt", "w", newline="\n") as f:
            f.write("\n".join(jobs) + "\n")
        print(f"{g}: {len(jobs)} jobs")
        total += len(jobs)
    print(f"total {total}")


if __name__ == "__main__":
    main()
