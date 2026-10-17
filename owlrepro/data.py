"""Calibration and evaluation data. Sampling follows the SparseGPT/Wanda/OWL loaders."""
from __future__ import annotations

import random

import torch
from datasets import load_dataset

C4_TRAIN = "en/c4-train.00000-of-01024.json.gz"
C4_VAL = "en/c4-validation.00000-of-00008.json.gz"
VIWIKI = ("wikimedia/wikipedia", "20231101.vi")
# first VIWIKI_CALIB_DOCS articles are the calibration pool, the next VIWIKI_EVAL_DOCS are held out
VIWIKI_CALIB_DOCS = 20000
VIWIKI_EVAL_DOCS = 1000


def _sample_windows(get_doc, n_docs: int, tokenizer, nsamples: int, seqlen: int, seed: int) -> torch.Tensor:
    # same procedure as get_c4 in the original code: random doc longer than seqlen, then a random window
    random.seed(seed)
    out = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, n_docs - 1)
            ids = tokenizer(get_doc(i), return_tensors="pt").input_ids
            if ids.shape[1] > seqlen:
                break
        j = random.randint(0, ids.shape[1] - seqlen - 1)
        out.append(ids[:, j:j + seqlen])
    return torch.cat(out, dim=0)


def c4_calibration(tokenizer, nsamples: int = 128, seqlen: int = 2048, seed: int = 0) -> torch.Tensor:
    data = load_dataset("allenai/c4", data_files={"train": C4_TRAIN}, split="train")
    return _sample_windows(lambda i: data[i]["text"], len(data), tokenizer, nsamples, seqlen, seed)


def c4_validation(tokenizer, seqlen: int = 2048) -> torch.Tensor:
    data = load_dataset("allenai/c4", data_files={"validation": C4_VAL}, split="validation")
    ids = tokenizer(" ".join(data[:1100]["text"]), return_tensors="pt").input_ids
    return ids[:, : 256 * seqlen]


def wikitext2_test(tokenizer) -> torch.Tensor:
    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return tokenizer("\n\n".join(data["text"]), return_tensors="pt").input_ids


def _viwiki_docs(start: int, count: int) -> list[str]:
    stream = load_dataset(*VIWIKI, split="train", streaming=True)
    docs = []
    for k, row in enumerate(stream):
        if k < start:
            continue
        if k >= start + count:
            break
        docs.append(row["text"])
    return docs


def viwiki_calibration(tokenizer, nsamples: int = 128, seqlen: int = 2048, seed: int = 0) -> torch.Tensor:
    docs = _viwiki_docs(0, VIWIKI_CALIB_DOCS)
    return _sample_windows(lambda i: docs[i], len(docs), tokenizer, nsamples, seqlen, seed)


def viwiki_test(tokenizer, seqlen: int = 2048, max_windows: int = 128) -> torch.Tensor:
    docs = _viwiki_docs(VIWIKI_CALIB_DOCS, VIWIKI_EVAL_DOCS)
    ids = tokenizer("\n\n".join(docs), return_tensors="pt").input_ids
    return ids[:, : max_windows * seqlen]


def calibration(name: str, tokenizer, nsamples: int, seqlen: int, seed: int) -> torch.Tensor:
    if name == "c4":
        return c4_calibration(tokenizer, nsamples, seqlen, seed)
    if name == "viwiki":
        return viwiki_calibration(tokenizer, nsamples, seqlen, seed)
    raise ValueError(f"unknown calibration set {name}")


def eval_tokens(name: str, tokenizer, seqlen: int) -> torch.Tensor:
    if name == "wikitext2":
        return wikitext2_test(tokenizer)
    if name == "c4":
        return c4_validation(tokenizer, seqlen)
    if name == "viwiki":
        return viwiki_test(tokenizer, seqlen)
    raise ValueError(f"unknown eval set {name}")
