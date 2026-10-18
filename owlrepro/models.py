from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# models the paper ran in fp16; everything newer defaults to bf16 when the GPU has it
_FP16_HINTS = ("huggyllama", "llama-7b", "llama-13b", "llama-30b", "llama-65b", "opt-", "llama-2")


def pick_dtype(name: str, dtype: str) -> torch.dtype:
    table = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    if dtype != "auto":
        return table[dtype]
    if not torch.cuda.is_available():
        return torch.float32
    if any(h in name.lower() for h in _FP16_HINTS):
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_model(name: str, dtype: str = "auto", seqlen: int = 2048, device_map: str | None = "auto"):
    torch_dtype = pick_dtype(name, dtype)
    if not torch.cuda.is_available():
        device_map = None
    kwargs = dict(low_cpu_mem_usage=True, device_map=device_map)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch_dtype, **kwargs)
    except TypeError:
        # transformers < 4.56 only knows torch_dtype
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch_dtype, **kwargs)
    model.eval()
    model.config.use_cache = False
    model.seqlen = seqlen
    return model


def load_tokenizer(name: str):
    # the original main.py loads LLaMA-1 and OPT with the slow tokenizer (use_fast=False); the fast
    # one can tokenize WikiText-2 differently, which changes perplexity
    paper_model = any(h in name.lower() for h in _FP16_HINTS)
    tok = AutoTokenizer.from_pretrained(name, use_fast=not paper_model)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    return tok


def get_layers(model) -> nn.ModuleList:
    """Return the list of Transformer blocks for the families we use (LLaMA, Qwen2, Mistral, OPT)."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner.layers
    if inner is not None and hasattr(inner, "decoder") and hasattr(inner.decoder, "layers"):
        return inner.decoder.layers
    raise ValueError(f"don't know where the blocks are in {type(model).__name__}")


def find_linears(module: nn.Module, prefix: str = "") -> dict[str, nn.Linear]:
    """All nn.Linear modules inside a block, in registration order (same as the original find_layers)."""
    if type(module) is nn.Linear:
        return {prefix: module}
    out = {}
    for name, child in module.named_children():
        out.update(find_linears(child, f"{prefix}.{name}" if prefix else name))
    return out


def input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def module_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


@torch.no_grad()
def block_sparsity(model) -> list[float]:
    out = []
    for layer in get_layers(model):
        zeros = total = 0
        for lin in find_linears(layer).values():
            zeros += (lin.weight == 0).sum().item()
            total += lin.weight.numel()
        out.append(zeros / total)
    return out


@torch.no_grad()
def total_sparsity(model) -> float:
    zeros = total = 0
    for layer in get_layers(model):
        for lin in find_linears(layer).values():
            zeros += (lin.weight == 0).sum().item()
            total += lin.weight.numel()
    return zeros / total
