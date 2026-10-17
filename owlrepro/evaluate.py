from __future__ import annotations

import torch
import torch.nn as nn

from .models import input_device

ZEROSHOT_TASKS = ["boolq", "rte", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]


@torch.no_grad()
def perplexity(model, ids: torch.Tensor, seqlen: int, max_windows: int | None = None) -> float:
    """Same as eval_ppl_wikitext in the OWL code: non-overlapping windows, batch 1."""
    dev = input_device(model)
    n = ids.numel() // seqlen
    if max_windows:
        n = min(n, max_windows)
    loss_fct = nn.CrossEntropyLoss()
    nlls = []
    for i in range(n):
        inp = ids[:, i * seqlen:(i + 1) * seqlen].to(dev)
        logits = model(inp, use_cache=False).logits.float()
        loss = loss_fct(logits[:, :-1, :].reshape(-1, logits.size(-1)), inp[:, 1:].reshape(-1))
        nlls.append(loss * seqlen)
    return torch.exp(torch.stack(nlls).sum() / (n * seqlen)).item()


def zeroshot(model, tokenizer, tasks=None, batch_size: int = 8, limit: int | None = None) -> dict:
    """lm-eval-harness 0.4.x, run on the in-memory model so no checkpoint has to be saved."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    tasks = tasks or ZEROSHOT_TASKS
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    res = lm_eval.simple_evaluate(model=lm, tasks=tasks, num_fewshot=0, limit=limit, log_samples=False)
    out = {}
    for task, r in res["results"].items():
        out[task] = {k.split(",")[0]: v for k, v in r.items() if k.startswith(("acc,", "acc_norm,"))}
    accs = [out[t]["acc"] for t in tasks if t in out and "acc" in out[t]]
    out["mean_acc"] = sum(accs) / len(accs) if accs else None
    return out
