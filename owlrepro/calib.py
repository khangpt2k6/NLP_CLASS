"""Block-by-block forward over the calibration set.

The original OWL/Wanda code calls each block as layer(x, attention_mask=..., position_ids=...).
That breaks on transformers >= 4.48, where blocks also need position_embeddings, and on models
whose blocks get different masks. Here we record whatever arguments the model itself passes to
each block (with forward pre-hooks) and replay them, so the same loop works on 4.4x and 5.x.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .models import get_layers, input_device, module_device


class _Stop(Exception):
    pass


def _to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(_to(o, device) for o in obj)
    if isinstance(obj, list):
        return [_to(o, device) for o in obj]
    if isinstance(obj, dict):
        return {k: _to(v, device) for k, v in obj.items()}
    return obj


def _hidden(args, kwargs):
    return args[0] if args else kwargs["hidden_states"]


class CalibRunner:
    """Holds the hidden states that enter the current block, for every calibration sample."""

    @torch.no_grad()
    def __init__(self, model, calib_ids: torch.Tensor):
        self.model = model
        self.layers = get_layers(model)
        dev = input_device(model)
        n, seqlen = calib_ids.shape
        dtype = next(model.parameters()).dtype
        self.inps = torch.zeros((n, seqlen, model.config.hidden_size), dtype=dtype, device=dev)
        count = [0]

        def grab_input(module, args, kwargs):
            self.inps[count[0]] = _hidden(args, kwargs)[0]
            count[0] += 1
            raise _Stop

        h = self.layers[0].register_forward_pre_hook(grab_input, with_kwargs=True)
        for j in range(n):
            try:
                model(calib_ids[j:j + 1].to(dev), use_cache=False)
            except _Stop:
                pass
        h.remove()

        # one more forward with sample 0 to record the extra arguments of every block
        self.block_args: list[tuple] = [()] * len(self.layers)
        self.block_kwargs: list[dict] = [{}] * len(self.layers)
        last = len(self.layers) - 1

        def make_hook(k):
            def grab_kwargs(module, args, kwargs):
                kw = dict(kwargs)
                kw.pop("hidden_states", None)
                self.block_args[k] = tuple(args[1:])
                self.block_kwargs[k] = kw
                if k == last:
                    raise _Stop
            return grab_kwargs

        hs = [layer.register_forward_pre_hook(make_hook(k), with_kwargs=True) for k, layer in enumerate(self.layers)]
        try:
            model(calib_ids[0:1].to(dev), use_cache=False)
        except _Stop:
            pass
        for h in hs:
            h.remove()

        self.outs = torch.zeros_like(self.inps)

    @property
    def nsamples(self) -> int:
        return self.inps.shape[0]

    @torch.no_grad()
    def run_block(self, i: int) -> None:
        """Forward all samples through block i, writing into self.outs."""
        layer = self.layers[i]
        dev = module_device(layer)
        if self.inps.device != dev:
            self.inps, self.outs = self.inps.to(dev), self.outs.to(dev)
        args = _to(self.block_args[i], dev)
        kwargs = _to(self.block_kwargs[i], dev)
        for j in range(self.nsamples):
            out = layer(self.inps[j:j + 1], *args, **kwargs)
            if isinstance(out, (tuple, list)):
                out = out[0]
            self.outs[j] = out[0]

    def advance(self) -> None:
        self.inps, self.outs = self.outs, self.inps


class InputNorm:
    """Running mean of squared input norms per column, i.e. scaler_row in the Wanda code."""

    def __init__(self, layer: nn.Linear):
        self.layer = layer
        self.scaler_row = torch.zeros(layer.weight.shape[1], device=layer.weight.device)
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor) -> None:
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        inp = inp.reshape(-1, inp.shape[-1]).t().float()
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples

    def hook(self, module, inp, out):
        self.add_batch(inp[0].data)

    def wanda_metric(self) -> torch.Tensor:
        return self.layer.weight.data.abs().float() * torch.sqrt(self.scaler_row).reshape(1, -1)
