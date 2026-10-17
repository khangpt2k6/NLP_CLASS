"""Gather results/*/*.json into one CSV (results/summary.csv) and print a short table.

    python scripts/collect.py            # all groups
    python scripts/collect.py r1 e1      # some groups
"""
import csv
import glob
import json
import os
import sys

COLS = ["group", "model", "method", "alloc", "sparsity_target", "sparsity", "M", "lam", "calib",
        "ppl_wikitext2", "ppl_c4", "ppl_viwiki", "zs_mean_acc", "lod_dense_cv", "minutes", "gpu"]


def row(path):
    r = json.load(open(path))
    a = r["args"]
    zs = r.get("zeroshot") or {}
    lod = r.get("lod_dense_summary") or {}
    lod_cv = next(iter(lod.values()))["cv"] if lod else None
    return {
        "group": os.path.basename(os.path.dirname(path)),
        "model": a["model"],
        "method": a["method"],
        "alloc": a["alloc"] if a["method"] != "dense" else "",
        "sparsity_target": a["sparsity"] if a["method"] != "dense" else 0.0,
        "sparsity": round(r["sparsity"], 4),
        "M": a["M"],
        "lam": a["lam"],
        "calib": a["calib"],
        "ppl_wikitext2": r["ppl"].get("wikitext2"),
        "ppl_c4": r["ppl"].get("c4"),
        "ppl_viwiki": r["ppl"].get("viwiki"),
        "zs_mean_acc": zs.get("mean_acc"),
        "lod_dense_cv": lod_cv,
        "minutes": round(r["total_seconds"] / 60, 1),
        "gpu": r["env"].get("gpu"),
    }


def main(groups):
    pattern = [f"results/{g}/*.json" for g in groups] if groups else ["results/*/*.json"]
    paths = sorted(p for pat in pattern for p in glob.glob(pat))
    rows = [row(p) for p in paths]
    os.makedirs("results", exist_ok=True)
    with open("results/summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        ppl = r["ppl_wikitext2"]
        ppl = f"{ppl:9.2f}" if ppl is not None else "        -"
        print(f"{r['group']:5} {r['model'].split('/')[-1]:16} {r['method']:9} {r['alloc']:8} "
              f"s={r['sparsity']:.3f} wiki={ppl} zs={r['zs_mean_acc']}")
    print(f"{len(rows)} results -> results/summary.csv")


if __name__ == "__main__":
    main(sys.argv[1:])
