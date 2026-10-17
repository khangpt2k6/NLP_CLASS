"""Layerwise sparsity allocations: OWL, OWL-inverse, ER, ER-plus and the outlier statistic (LOD)."""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def block_outlier_ratio(metrics: list[torch.Tensor], ms: list[float], ref_mean: float | None = None):
    """D^l in Eq. 1, in percent, for several M at once.

    Matches check_outlier_mean in the original code: the threshold is M times the mean score over
    all linear layers of the block. ref_mean lets us reuse the dense model's mean after pruning
    (Empirical Study II), so zeros do not shift the threshold.
    """
    total = sum(m.numel() for m in metrics)
    mean = ref_mean if ref_mean is not None else sum(m.double().sum().item() for m in metrics) / total
    ratios = {}
    for M in ms:
        thr = mean * M
        count = sum((m > thr).sum().item() for m in metrics)
        ratios[M] = 100.0 * count / total
    return ratios, mean


def owl_block_sparsity(lod, sparsity: float, lam: float, inverse: bool = False) -> np.ndarray:
    """Per-block sparsity from the outlier ratios, as in prune_wanda_outlier.

    Ratios are min-max scaled to [0, 2*lam] and shifted so the mean density is 1 - sparsity, which
    puts every block in [S - lam, S + lam]. inverse=True gives OWL-inverse (uses 1 - LOD).
    """
    d = np.asarray(lod, dtype=np.float64)
    if inverse:
        d = -d
    span = d.max() - d.min()
    if span < 1e-12:
        return np.full_like(d, sparsity)
    dens = (d - d.min()) / span * lam * 2
    dens = dens - dens.mean() + (1 - sparsity)
    s = 1 - dens
    # the original code clamps non-positive ratios to 0.01
    return np.where(s <= 0, 0.01, s)


def er_sparsity(shapes: dict, sparsity: float, keep_dense: tuple = ()) -> dict:
    """Erdos-Renyi allocation over linear layers (Mocanu et al., 2018), solved like ERK.

    density_l = eps * (n_in + n_out) / (n_in * n_out), with eps chosen so the total number of kept
    weights matches 1 - sparsity. Layers whose density would exceed 1 are made dense and eps is
    solved again. keep_dense forces some layers dense first (used for ER-plus).
    """
    sizes = {k: r * c for k, (r, c) in shapes.items()}
    target = (1 - sparsity) * sum(sizes.values())
    dense = set(keep_dense)
    while True:
        budget = target - sum(sizes[k] for k in dense)
        raw = {k: (r + c) / (r * c) for k, (r, c) in shapes.items() if k not in dense}
        eps = budget / sum(raw[k] * sizes[k] for k in raw)
        over = {k for k in raw if eps * raw[k] > 1}
        if not over:
            break
        dense |= over
    out = {k: 0.0 for k in dense}
    out.update({k: 1 - eps * raw[k] for k in raw})
    return out


def global_threshold(samples: list[torch.Tensor], sparsity: float) -> float:
    """Global score threshold from a uniform random sample of all scores."""
    allv = torch.cat([s.flatten().float().cpu() for s in samples])
    k = max(1, min(allv.numel(), int(round(allv.numel() * sparsity))))
    return torch.kthvalue(allv, k).values.item()
