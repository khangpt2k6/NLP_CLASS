"""End-to-end checks on tiny randomly initialised models (no downloads, CPU only)."""
import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, OPTConfig, OPTForCausalLM, Qwen2Config, Qwen2ForCausalLM

from owlrepro.calib import CalibRunner
from owlrepro.evaluate import perplexity
from owlrepro.models import block_sparsity, find_linears, get_layers, total_sparsity
from owlrepro.prune import outlier_pass, prune

VOCAB, HID, LAYERS, SEQ, N = 128, 64, 4, 32, 4


def tiny(kind):
    torch.manual_seed(0)
    if kind == "llama":
        cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=HID, intermediate_size=128, num_hidden_layers=LAYERS,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
        model = LlamaForCausalLM(cfg)
    elif kind == "qwen2":
        cfg = Qwen2Config(vocab_size=VOCAB, hidden_size=HID, intermediate_size=128, num_hidden_layers=LAYERS,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
        model = Qwen2ForCausalLM(cfg)
    else:
        cfg = OPTConfig(vocab_size=VOCAB, hidden_size=HID, ffn_dim=128, num_hidden_layers=LAYERS,
                        num_attention_heads=4, max_position_embeddings=128, word_embed_proj_dim=HID)
        model = OPTForCausalLM(cfg)
    model.eval()
    model.config.use_cache = False
    model.seqlen = SEQ
    return model


def calib():
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, VOCAB, (N, SEQ), generator=g)


@pytest.mark.parametrize("kind", ["llama", "qwen2", "opt"])
def test_runner_matches_full_forward(kind):
    # replaying the recorded block kwargs must reproduce the model's own hidden states
    model = tiny(kind)
    ids = calib()
    runner = CalibRunner(model, ids)
    for i in range(len(runner.layers)):
        runner.run_block(i)
        runner.advance()
    with torch.no_grad():
        ref = model(ids, output_hidden_states=True, use_cache=False).hidden_states[-1]
    inner = model.model if kind != "opt" else model.model.decoder
    final_norm = inner.norm if kind != "opt" else inner.final_layer_norm
    with torch.no_grad():
        got = final_norm(runner.inps)
    assert torch.allclose(got, ref, atol=1e-4)


@pytest.mark.parametrize("kind", ["llama", "qwen2", "opt"])
@pytest.mark.parametrize("method", ["magnitude", "wanda", "sparsegpt"])
def test_uniform_sparsity(kind, method):
    model = tiny(kind)
    prune(model, calib(), method, "uniform", 0.5)
    for s in block_sparsity(model):
        assert abs(s - 0.5) < 0.03


@pytest.mark.parametrize("method", ["wanda", "sparsegpt"])
def test_owl_follows_allocation(method):
    model = tiny("llama")
    info = prune(model, calib(), method, "owl", 0.6, M=3, lam=0.1)
    target = np.array(info["block_target"])
    assert abs(target.mean() - 0.6) < 1e-6
    got = np.array(block_sparsity(model))
    assert np.abs(got - target).max() < 0.03
    assert abs(total_sparsity(model) - 0.6) < 0.03


@pytest.mark.parametrize("alloc", ["er", "er_plus", "global", "owl_inv"])
def test_other_allocs_hit_target(alloc):
    model = tiny("llama")
    prune(model, calib(), "wanda", alloc, 0.6, M=3, lam=0.1)
    assert abs(total_sparsity(model) - 0.6) < 0.04


def test_er_plus_last_linear_dense():
    model = tiny("llama")
    prune(model, calib(), "wanda", "er_plus", 0.6)
    last = list(find_linears(get_layers(model)[-1]).values())[-1]
    assert (last.weight == 0).float().mean().item() < 0.01


def test_lod_and_ppl_run():
    model = tiny("llama")
    ids = calib()
    st = outlier_pass(model, ids, [3.0, 5.0])
    assert len(st["lod"][3.0]) == LAYERS
    assert all(a >= b for a, b in zip(st["lod"][3.0], st["lod"][5.0]))
    ppl = perplexity(model, ids.reshape(1, -1), SEQ)
    assert np.isfinite(ppl)
