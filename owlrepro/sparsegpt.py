# SparseGPT, ported from the OWL repo (which copies IST-DASLab/sparsegpt@f5c2500).
# Only change: no torch.cuda.synchronize() when running on CPU.
import math

import torch
import torch.nn as nn

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class SparseGPT:
    def __init__(self, layer: nn.Linear):
        self.layer = layer
        self.dev = layer.weight.device
        self.rows, self.columns = layer.weight.shape
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor) -> None:
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        inp = inp.reshape(-1, inp.shape[-1]).t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def hook(self, module, inp, out):
        self.add_batch(inp[0].data)

    @torch.no_grad()
    def fasterprune(self, sparsity: float, blocksize: int = 128, percdamp: float = 0.01) -> None:
        W = self.layer.weight.data.clone().float()
        H = self.H
        self.H = None
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        Hinv = torch.linalg.cholesky(H, upper=True)

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
            thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
            mask1 = tmp <= thresh

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                q = w.clone()
                q[mask1[:, i]] = 0
                Q1[:, i] = q
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if W.is_cuda:
            torch.cuda.synchronize()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.dtype)
