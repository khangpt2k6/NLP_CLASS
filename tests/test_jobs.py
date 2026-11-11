"""Checks on the job lists and the run-time estimate (no models, no downloads)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import estimate  # noqa: E402
import make_jobs  # noqa: E402
from owlrepro.run import already_done  # noqa: E402


def outs(lines):
    return [estimate.job_args(l)["--out"] for l in lines]


def test_talk_is_a_ppl_only_subset_of_the_full_groups():
    talk = make_jobs.talk()
    full = make_jobs.r1() + make_jobs.r2() + make_jobs.r4() + make_jobs.e1() + make_jobs.e2()
    assert len(talk) == 84
    assert len(set(outs(talk))) == len(talk)
    assert set(outs(talk)) <= set(outs(full))
    assert not any("--zeroshot" in l for l in talk)
    # apart from zero-shot, a talk job is the same command as its twin
    twin = {estimate.job_args(l)["--out"]: l for l in full}
    for l in talk:
        assert twin[estimate.job_args(l)["--out"]].replace(" --zeroshot all", "") == l


def test_every_job_model_has_a_parameter_count():
    for hf, _, _ in make_jobs.MODELS.values():
        assert hf in estimate.PARAMS_B and hf in estimate.SHORT


def test_estimate_orders_jobs_sensibly():
    def mins(short, method, alloc="uniform", gpu="A100", **kw):
        line = make_jobs.job("x", short, method, alloc, **kw)
        return estimate.job_minutes(estimate.job_args(line), estimate.gpu_slowdown(gpu))

    assert mins("llama7b", "dense") < mins("llama7b", "wanda") < mins("llama7b", "wanda", "owl")
    assert mins("llama7b", "wanda", "owl") < mins("llama7b", "sparsegpt", "owl")
    assert mins("qwen2.5-0.5b", "wanda") < mins("llama7b", "wanda") < mins("llama13b", "wanda")
    assert mins("llama7b", "wanda") < mins("llama7b", "wanda", zs=True)
    assert mins("llama7b", "wanda") < mins("llama7b", "wanda", gpu="L4") < mins("llama7b", "wanda", gpu="T4")


def test_gpu_names():
    assert estimate.gpu_slowdown("NVIDIA A100-SXM4-40GB") == 1.0
    assert estimate.gpu_slowdown("Tesla T4") == 4.5
    assert estimate.gpu_slowdown("NVIDIA L4") == 3.0
    assert estimate.gpu_slowdown("NVIDIA L40S") == 1.1
    assert estimate.gpu_slowdown("some new card") == estimate.UNKNOWN_GPU_SLOWDOWN


def test_ppl_only_result_does_not_count_for_a_zeroshot_job(tmp_path):
    out = tmp_path / "r.json"
    assert not already_done(str(out))
    out.write_text(json.dumps({"ppl": {"wikitext2": 10.0}}))
    assert already_done(str(out))
    assert not already_done(str(out), "all")
    assert not estimate.is_done({"--out": str(out), "--zeroshot": "all"})
    out.write_text(json.dumps({"ppl": {"wikitext2": 10.0}, "zeroshot": {"mean_acc": 0.5}}))
    assert already_done(str(out), "all")
    assert estimate.is_done({"--out": str(out), "--zeroshot": "all"})
    assert estimate.is_done({"--out": str(out)})
