"""Optimizers whose hyperparameters survive a change of model width.

The default recipe of this package is Adam with a cosine schedule, and it has a
known weakness: its learning rate has to be retuned when the model gets wider,
and if it is not, a wider model can score *worse* than a narrower one at the
same budget. That is an artifact of the optimizer's parametrization rather than
a statement about capacity, and it is easy to mistake for one.

Muon \\citep{jordan2024} attacks exactly that. Momentum is orthogonalized before
it is applied -- the update direction is the closest orthogonal matrix to the
momentum, computed by a Newton--Schulz iteration -- so the step size is set by
the geometry of the weight matrix rather than by per-entry gradient scales, and
the same learning rate transfers across widths far better. It applies only to
the 2-D hidden weights; biases, norms, and embeddings keep Adam, as in the
reference implementation.
"""
from __future__ import annotations

import torch


def _orthogonalize(G: torch.Tensor, steps: int = 5, eps: float = 1e-7):
    """Newton--Schulz quintic iteration: the closest orthogonal matrix to G.

    Runs in bfloat16 -- the iteration only needs the singular values pushed
    toward one, and the coefficients below are the standard quintic tuned for
    fast convergence from a spectrally normalized start.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon on 2-D parameters, Adam on everything else.

    ``lr`` is the Muon learning rate; ``adam_lr`` governs the fallback group
    (biases, normalizations, embeddings and any parameter that is not a plain
    matrix). Both default to values that work unchanged across the width ladder
    of this package.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5,
                 adam_lr: float = 3e-4, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        params = list(params)
        muon_p = [p for p in params if p.ndim == 2 and min(p.shape) > 1]
        adam_p = [p for p in params if not (p.ndim == 2 and min(p.shape) > 1)]
        groups = [dict(params=muon_p, use_muon=True, lr=lr, momentum=momentum,
                       nesterov=nesterov, ns_steps=ns_steps,
                       weight_decay=weight_decay),
                  dict(params=adam_p, use_muon=False, lr=adam_lr, betas=betas,
                       eps=eps, weight_decay=weight_decay)]
        super().__init__(groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for g in self.param_groups:
            if g["use_muon"]:
                for p in g["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if "m" not in st:
                        st["m"] = torch.zeros_like(p)
                    m = st["m"]
                    m.mul_(g["momentum"]).add_(p.grad)
                    d = p.grad.add(m, alpha=g["momentum"]) if g["nesterov"] else m
                    o = _orthogonalize(d, g["ns_steps"])
                    # keep the update's scale comparable across shapes
                    scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                    if g["weight_decay"]:
                        p.mul_(1 - g["lr"] * g["weight_decay"])
                    p.add_(o, alpha=-g["lr"] * scale)
            else:
                for p in g["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if "step" not in st:
                        st["step"] = 0
                        st["exp_avg"] = torch.zeros_like(p)
                        st["exp_avg_sq"] = torch.zeros_like(p)
                    st["step"] += 1
                    b1, b2 = g["betas"]
                    st["exp_avg"].mul_(b1).add_(p.grad, alpha=1 - b1)
                    st["exp_avg_sq"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                    bc1 = 1 - b1 ** st["step"]
                    bc2 = 1 - b2 ** st["step"]
                    denom = (st["exp_avg_sq"] / bc2).sqrt_().add_(g["eps"])
                    if g["weight_decay"]:
                        p.mul_(1 - g["lr"] * g["weight_decay"])
                    p.addcdiv_(st["exp_avg"] / bc1, denom, value=-g["lr"])
        return loss
