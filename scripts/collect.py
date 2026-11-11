"""Gather results/*/*.json into CSV files and print a short table.

    python scripts/collect.py            # all groups (results/smoke is skipped)
    python scripts/collect.py r1 e1      # some groups

Writes three files into results/:
    summary.csv   one row per job: settings, perplexity, zero-shot, LOD summary, time, GPU memory
    owl_gain.csv  one row per model x method x setting: dense, uniform and OWL perplexity and dL
    blocks.csv    one row per job x block: actual sparsity, OWL target, LOD per M
"""
import csv
import glob
import json
import math
import os
import sys

SKIP_GROUPS = ("smoke",)
LOD_MS = ("3.0", "5.0", "7.0", "10.0")
COLS = ["group", "job", "model", "method", "alloc", "sparsity_target", "sparsity", "M", "lam", "seed",
        "calib", "nsamples", "seqlen", "dtype", "ppl_wikitext2", "ppl_c4", "ppl_viwiki", "zs_mean_acc",
        "lod_M", "lod_sum", "lod_mean", "lod_cv", "lod_u", "load_seconds", "alloc_seconds",
        "prune_seconds", "minutes", "max_gpu_mem_gb", "gpu", "torch", "transformers", "started"]
GAIN_COLS = ["group", "model", "method", "sparsity_target", "M", "lam", "seed", "calib",
             "ppl_dense", "ppl_uniform", "ppl_owl", "dL"]
BLOCK_COLS = ["group", "job", "model", "method", "alloc", "block", "sparsity", "block_target"] + \
             [f"lod_M{m.split('.')[0]}" for m in LOD_MS]


def rnd(x, n):
    return None if x is None else round(x, n)


def lod_per_m(r) -> dict:
    """Per-block LOD lists by M. Dense LOD if the job logged it, else the LOD OWL used."""
    if r.get("lod_dense"):
        return {str(float(m)): v for m, v in r["lod_dense"].items()}
    lod = (r.get("prune") or {}).get("lod")
    return {str(float(r["args"]["M"])): lod} if lod else {}


def lod_summary(r) -> dict:
    """LOD summary at the job's M, falling back to the first M that was logged."""
    summ = r.get("lod_dense_summary") or {}
    m = str(float(r["args"]["M"]))
    if m not in summ and summ:
        m = next(iter(summ))
    if m in summ:
        return {"lod_M": float(m), **{f"lod_{k}": summ[m][k] for k in ("sum", "mean", "cv", "u")}}
    lod = (r.get("prune") or {}).get("lod")
    if not lod:
        return {}
    mean = sum(lod) / len(lod)
    sd = math.sqrt(sum((x - mean) ** 2 for x in lod) / len(lod))
    k = max(1, int(round(0.2 * len(lod))))
    u = (sum(lod[:k] + lod[-k:]) / (2 * k) - sum(lod[k:-k]) / len(lod[k:-k])) if len(lod) > 2 * k else 0.0
    return {"lod_M": float(m), "lod_sum": sum(lod), "lod_mean": mean, "lod_cv": sd / mean if mean > 0 else 0.0,
            "lod_u": u}


def row(path, r):
    a = r["args"]
    zs = r.get("zeroshot") or {}
    pr = r.get("prune") or {}
    dense = a["method"] == "dense"
    out = {
        "group": os.path.basename(os.path.dirname(path)),
        "job": os.path.splitext(os.path.basename(path))[0],
        "model": a["model"],
        "method": a["method"],
        "alloc": "" if dense else a["alloc"],
        "sparsity_target": 0.0 if dense else a["sparsity"],
        "sparsity": round(r["sparsity"], 4),
        "M": a["M"],
        "lam": a["lam"],
        "seed": a["seed"],
        "calib": a["calib"],
        "nsamples": a["nsamples"],
        "seqlen": a["seqlen"],
        "dtype": r.get("dtype"),
        "ppl_wikitext2": r["ppl"].get("wikitext2"),
        "ppl_c4": r["ppl"].get("c4"),
        "ppl_viwiki": r["ppl"].get("viwiki"),
        "zs_mean_acc": zs.get("mean_acc"),
        "load_seconds": rnd(r.get("load_seconds"), 1),
        "alloc_seconds": rnd(pr.get("alloc_seconds"), 1),
        "prune_seconds": rnd(pr.get("prune_seconds"), 1),
        "minutes": round(r["total_seconds"] / 60, 1),
        "max_gpu_mem_gb": rnd(r.get("max_gpu_mem_gb"), 2),
        "gpu": r["env"].get("gpu"),
        "torch": r["env"].get("torch"),
        "transformers": r["env"].get("transformers"),
        "started": r.get("started"),
    }
    out.update({k: rnd(v, 4) for k, v in lod_summary(r).items()})
    return out


def block_rows(base, r):
    lods = lod_per_m(r)
    target = (r.get("prune") or {}).get("block_target")
    for i, s in enumerate(r["block_sparsity"]):
        b = {k: base[k] for k in ("group", "job", "model", "method", "alloc")}
        b.update(block=i, sparsity=round(s, 4), block_target=rnd(target[i], 4) if target else None)
        for m in LOD_MS:
            b[f"lod_M{m.split('.')[0]}"] = rnd(lods[m][i], 4) if m in lods else None
        yield b


def gain_rows(rows):
    """Pair uniform and OWL runs of the same setting and add the dense perplexity of that model."""
    dense = {(r["group"], r["model"]): r["ppl_wikitext2"] for r in rows if r["method"] == "dense"}
    pairs = {}
    for r in rows:
        if r["alloc"] not in ("uniform", "owl"):
            continue
        # M and lambda only matter for OWL, so a uniform run pairs with every OWL setting
        key = (r["group"], r["model"], r["method"], r["sparsity_target"], r["seed"], r["calib"])
        pairs.setdefault(key, {"uniform": None, "owl": []})
        if r["alloc"] == "uniform":
            pairs[key]["uniform"] = r["ppl_wikitext2"]
        else:
            pairs[key]["owl"].append(r)
    out = []
    for (group, model, method, s, seed, calib), p in sorted(pairs.items()):
        for o in p["owl"]:
            uni, owl = p["uniform"], o["ppl_wikitext2"]
            out.append({
                "group": group, "model": model, "method": method, "sparsity_target": s, "M": o["M"],
                "lam": o["lam"], "seed": seed, "calib": calib,
                "ppl_dense": rnd(dense.get((group, model)), 3), "ppl_uniform": rnd(uni, 3),
                "ppl_owl": rnd(owl, 3),
                "dL": round(math.log(uni) - math.log(owl), 4) if uni and owl else None,
            })
    return out


def write(name, cols, rows):
    with open(os.path.join("results", name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main(groups):
    pattern = [f"results/{g}/*.json" for g in groups] if groups else ["results/*/*.json"]
    paths = sorted(p for pat in pattern for p in glob.glob(pat))
    if not groups:
        paths = [p for p in paths if os.path.basename(os.path.dirname(p)) not in SKIP_GROUPS]
    rows, blocks = [], []
    for p in paths:
        with open(p) as f:
            r = json.load(f)
        rows.append(row(p, r))
        blocks.extend(block_rows(rows[-1], r))
    gains = gain_rows(rows)
    os.makedirs("results", exist_ok=True)
    write("summary.csv", COLS, rows)
    write("owl_gain.csv", GAIN_COLS, gains)
    write("blocks.csv", BLOCK_COLS, blocks)
    for r in rows:
        ppl = r["ppl_wikitext2"]
        ppl = f"{ppl:9.2f}" if ppl is not None else "        -"
        print(f"{r['group']:5} {r['model'].split('/')[-1]:16} {r['method']:9} {r['alloc']:8} "
              f"s={r['sparsity']:.3f} wiki={ppl} zs={r['zs_mean_acc']} {r['minutes']:5.1f} min")
    print(f"{len(rows)} results -> results/summary.csv, {len(gains)} pairs -> results/owl_gain.csv, "
          f"{len(blocks)} block rows -> results/blocks.csv")


if __name__ == "__main__":
    main(sys.argv[1:])
