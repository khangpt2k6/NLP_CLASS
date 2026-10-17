import numpy as np
import torch

from owlrepro.alloc import block_outlier_ratio, er_sparsity, global_threshold, owl_block_sparsity


def test_owl_mean_and_range():
    lod = [5.0, 1.0, 0.5, 0.4, 0.3, 0.6, 2.0]
    s = owl_block_sparsity(lod, 0.7, 0.08)
    assert abs(s.mean() - 0.7) < 1e-9
    # the code scales to a band of width 2*lam around the mean; with skewed LOD it is not
    # centred on S, so blocks can leave [S - lam, S + lam] (the paper says they stay inside)
    assert abs((s.max() - s.min()) - 0.16) < 1e-9
    assert s.min() < 0.7 - 0.08
    # more outliers -> lower sparsity
    assert s[0] == s.min() and s[4] == s.max()


def test_owl_inverse_flips_order():
    lod = [5.0, 1.0, 0.3]
    s = owl_block_sparsity(lod, 0.6, 0.05)
    si = owl_block_sparsity(lod, 0.6, 0.05, inverse=True)
    assert np.argmin(s) == np.argmax(si)
    assert abs(si.mean() - 0.6) < 1e-9


def test_owl_flat_lod_is_uniform():
    s = owl_block_sparsity([1.0, 1.0, 1.0], 0.5, 0.1)
    assert np.allclose(s, 0.5)


def test_matches_original_formula():
    # copy of the adjustment in prune_wanda_outlier
    lod, S, lam = np.array([3.1, 0.7, 0.2, 0.25, 1.4]), 0.7, 0.08
    r = (lod - lod.min()) * (1 / (lod.max() - lod.min()) * lam * 2)
    r = r - np.mean(r) + (1 - S)
    assert np.allclose(owl_block_sparsity(lod, S, lam), 1 - r)


def test_er_hits_target_and_caps():
    shapes = {"a": (64, 64), "b": (256, 64), "c": (8, 8)}
    per = er_sparsity(shapes, 0.7)
    sizes = {k: r * c for k, (r, c) in shapes.items()}
    kept = sum((1 - per[k]) * sizes[k] for k in shapes)
    assert abs(kept - 0.3 * sum(sizes.values())) < 1e-6
    assert all(0 <= v <= 1 for v in per.values())
    # small layer gets lower sparsity than big ones
    assert per["c"] < per["a"]


def test_er_plus_keeps_layer_dense():
    shapes = {"a": (64, 64), "b": (64, 64), "last": (64, 64)}
    per = er_sparsity(shapes, 0.6, keep_dense=("last",))
    assert per["last"] == 0.0
    assert abs(per["a"] - 0.9) < 1e-9


def test_outlier_ratio():
    m = torch.ones(10, 10)
    m[0, 0] = 1000.0
    ratios, mean = block_outlier_ratio([m], [5.0])
    assert abs(ratios[5.0] - 1.0) < 1e-9  # 1 of 100 weights, in percent
    ratios2, _ = block_outlier_ratio([m], [5.0], ref_mean=0.5)
    assert ratios2[5.0] == 1.0


def test_global_threshold():
    vals = [torch.arange(100.0), torch.arange(100.0, 200.0)]
    thr = global_threshold(vals, 0.25)
    assert thr == 49.0
