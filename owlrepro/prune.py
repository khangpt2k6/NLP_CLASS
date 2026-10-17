"""Magnitude, Wanda and SparseGPT pruning with a per-layer sparsity allocation."""
from __future__ import annotations

import time

import numpy as np
import torch

from .alloc import block_outlier_ratio, er_sparsity, global_threshold, owl_block_sparsity
from .calib import CalibRunner, InputNorm
from .models import find_linears, get_layers
from .sparsegpt import SparseGPT

METHODS = ("magnitude", "wanda", "sparsegpt")
ALLOCS = ("uniform", "owl", "owl_inv", "er", "er_plus", "global")
# total number of scores sampled to estimate the global threshold
GLOBAL_SAMPLE = 20_000_000


@torch.no_grad()
def outlier_pass(model, calib: torch.Tensor, ms, ref_means=None, global_sample: bool = False) -> dict:
    """Forward the calibration set block by block without pruning and collect outlier statistics.

    Returns per-block LOD (percent) for every M in ms, the per-block mean Wanda score, and
    optionally a random sample of scores per linear layer for the global allocation.
    """
    runner = CalibRunner(model, calib)
    layers = runner.layers
    lod = {float(M): [] for M in ms}
    means = []
    samples = {}
    n_params = sum(l.weight.numel() for layer in layers for l in find_linears(layer).values())
    frac = min(1.0, GLOBAL_SAMPLE / n_params)
    gen = torch.Generator().manual_seed(0)

    for i, layer in enumerate(layers):
        subset = find_linears(layer)
        stats = {n: InputNorm(m) for n, m in subset.items()}
        handles = [m.register_forward_hook(stats[n].hook) for n, m in subset.items()]
        runner.run_block(i)
        for h in handles:
            h.remove()

        metrics = [stats[n].wanda_metric() for n in subset]
        ratios, mean = block_outlier_ratio(metrics, list(lod), None if ref_means is None else ref_means[i])
        for M, r in ratios.items():
            lod[M].append(r)
        means.append(mean)
        if global_sample:
            for n, m in zip(subset, metrics):
                k = max(1, int(m.numel() * frac))
                idx = torch.randint(m.numel(), (k,), generator=gen).to(m.device)
                samples[(i, n)] = m.flatten()[idx].cpu()
        del metrics, stats
        runner.advance()
    return {"lod": lod, "means": means, "samples": samples}


def _threshold_mask(metric: torch.Tensor, ratio: float) -> torch.Tensor:
    """Matrix-level mask, same as the magnitude baseline: prune scores <= sorted[int(n * ratio)]."""
    flat = metric.flatten()
    k = int(flat.numel() * ratio)
    if k >= flat.numel():
        return torch.ones_like(metric, dtype=torch.bool)
    thresh = torch.kthvalue(flat.float(), k + 1).values
    return metric <= thresh


def _row_mask(metric: torch.Tensor, ratio: float) -> torch.Tensor:
    """Wanda's output-balanced mask: the same fraction is pruned in every output row."""
    mask = torch.zeros_like(metric, dtype=torch.bool)
    idx = torch.sort(metric, dim=-1, stable=True)[1][:, : int(metric.shape[1] * ratio)]
    mask.scatter_(1, idx, True)
    return mask


@torch.no_grad()
def apply_pruning(model, calib: torch.Tensor | None, method: str, ratio_of, matrix_level: bool = False) -> None:
    """Prune every linear layer in every block. ratio_of(block, name) gives its sparsity."""
    if method == "magnitude":
        for i, layer in enumerate(get_layers(model)):
            for n, lin in find_linears(layer).items():
                r = ratio_of(i, n)
                if r > 0:
                    lin.weight.data[_threshold_mask(lin.weight.data.abs(), r)] = 0
        return

    stat_cls = InputNorm if method == "wanda" else SparseGPT
    runner = CalibRunner(model, calib)
    for i, layer in enumerate(runner.layers):
        subset = find_linears(layer)
        stats = {n: stat_cls(m) for n, m in subset.items()}
        handles = [m.register_forward_hook(stats[n].hook) for n, m in subset.items()]
        runner.run_block(i)
        for h in handles:
            h.remove()

        for n, lin in subset.items():
            r = ratio_of(i, n)
            if r <= 0:
                continue
            if method == "wanda":
                metric = stats[n].wanda_metric()
                mask = _threshold_mask(metric, r) if matrix_level else _row_mask(metric, r)
                lin.weight.data[mask] = 0
                del metric, mask
            else:
                stats[n].fasterprune(r)
        del stats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # outputs of the pruned block feed the next one
        runner.run_block(i)
        runner.advance()


def linear_shapes(model) -> dict:
    return {
        (i, n): tuple(lin.weight.shape)
        for i, layer in enumerate(get_layers(model))
        for n, lin in find_linears(layer).items()
    }


@torch.no_grad()
def prune(model, calib, method: str, alloc: str, sparsity: float, M: float = 5.0, lam: float = 0.08) -> dict:
    """Run one pruning job and return what we want to log about the allocation."""
    assert method in METHODS and alloc in ALLOCS
    info: dict = {"method": method, "alloc": alloc, "target_sparsity": sparsity}
    t0 = time.time()
    matrix_level = False

    if alloc == "uniform":
        ratio_of = lambda i, n: sparsity
    elif alloc in ("owl", "owl_inv"):
        stats = outlier_pass(model, calib, [M])
        lod = stats["lod"][float(M)]
        block = owl_block_sparsity(lod, sparsity, lam, inverse=(alloc == "owl_inv"))
        info.update(lod=lod, block_target=block.tolist())
        ratio_of = lambda i, n: float(block[i])
    elif alloc in ("er", "er_plus"):
        shapes = linear_shapes(model)
        keep = (list(shapes)[-1],) if alloc == "er_plus" else ()
        per = er_sparsity(shapes, sparsity, keep_dense=keep)
        info["linear_target"] = {f"{i}.{n}": v for (i, n), v in per.items()}
        ratio_of = lambda i, n: per[(i, n)]
    else:  # global threshold on Wanda scores of the dense model, no per-row balancing
        stats = outlier_pass(model, calib, [M], global_sample=True)
        thr = global_threshold(list(stats["samples"].values()), sparsity)
        per = {k: (v <= thr).float().mean().item() for k, v in stats["samples"].items()}
        info["linear_target"] = {f"{i}.{n}": v for (i, n), v in per.items()}
        ratio_of = lambda i, n: per[(i, n)]
        matrix_level = True
    info["alloc_seconds"] = time.time() - t0

    t1 = time.time()
    apply_pruning(model, calib, method, ratio_of, matrix_level=matrix_level)
    info["prune_seconds"] = time.time() - t1
    return info


def summarize_lod(lod: list[float]) -> dict:
    a = np.asarray(lod)
    return {"sum": float(a.sum()), "mean": float(a.mean()), "cv": float(a.std() / a.mean()) if a.mean() > 0 else 0.0}
