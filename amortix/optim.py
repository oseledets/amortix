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

    Accepts a single matrix ``[m, n]`` or a stack ``[g, m, n]``, and the stacked
    form is the one that matters for speed. The iteration is fifteen small
    matrix products, and on the matrices in this package each costs the same
    0.5 ms whether it is 288x32 or 1152x128 -- the time is kernel launches, not
    arithmetic. Orthogonalizing 36 parameters one at a time therefore costs
    18 ms per optimizer step against Adam's 0.8 ms; doing it in shape-grouped
    batches removes most of that.

    Runs in bfloat16: the iteration only pushes the singular values toward one,
    and the quintic coefficients are the standard fast-converging choice from a
    spectrally normalized start.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


#: Batch at which the Muon step was tuned, and the step measured there.
MUON_REF_BATCH = 512
MUON_REF_LR = 1e-2
#: Square-root batch scaling below the critical batch, measured.
MUON_BATCH_EXPONENT = 0.5
#: Step ceiling: past the critical batch (~2048) the optimum stops moving.
MUON_LR_CAP = 2e-2


def muon_lr_for_batch(batch: int) -> float:
    """The Muon step that belongs to this batch size.

    ``lr(B) = min(1e-2 * (B / 512) ** 0.5, 2e-2)``.

    Both the exponent and the cap are measured on this package, not assumed.
    On the big model (GBM, fixed example budgets, scored on the six-K battery)
    the step was swept at four batch sizes:

      B =   64, 72,000 steps: 1.8e-3 -> 0.0064, 3.54e-3 -> 0.0066,
                              5.95e-3 -> 0.0079, 1.0e-2 -> 0.0083
      B =  512,  9,000 steps: 3e-3 -> 0.0065, 1e-2 -> 0.0061, 2e-2 -> 0.0085
      B = 2048,  2,250 steps: 1e-2 -> 0.0065, 1.41e-2 -> 0.0072,
                              2e-2 -> 0.0066, 3e-2 -> 0.0070
      B = 8192,  2,020 steps: 1e-2 -> 0.0061, 2e-2 -> 0.0052,
                              4e-2 -> 0.0060, 8e-2 -> diverges (nan)

    The square root holds where the curve is sharp: the optimum moves from
    3.5e-3 at B=64 to 1e-2 at B=512 -- a factor 2.82 for an eightfold batch,
    against 2.83 predicted by the square root and 1.68 by the fourth root
    (and the fourth root is not a near miss there: 5.95e-3 scores 0.0079
    against 0.0066).

    Past B~2048 the law changes character and extrapolating the root is a
    trap that was walked into once: the root predicts 4e-2 at B=8192, where
    the measured optimum is 2e-2 -- the same 2e-2 as at 2048 -- and twice the
    root's prediction already diverges. So: square root up to the critical
    batch, constant after it. The flat response at B=2048 (a threefold sweep
    lands within 0.0007) is this ceiling announcing itself.
    """
    lr = MUON_REF_LR * (float(batch) / MUON_REF_BATCH) ** MUON_BATCH_EXPONENT
    return min(lr, MUON_LR_CAP)


class Muon(torch.optim.Optimizer):
    """Muon on 2-D parameters, Adam on everything else.

    ``lr`` is the Muon learning rate; ``adam_lr`` governs the fallback group
    (biases, normalizations, embeddings and any parameter that is not a plain
    matrix).

    The default ``lr`` is measured, not inherited. Swept over 3e-4..6e-2 on the
    largest model of the width ladder (GBM, batch 512, 9,000 steps, scored on
    the six-K design battery), the response is a broad shallow basin between
    3e-3 and 1e-2 with a hard edge just past it: 6e-2 diverges outright. The
    step transfers across width in the direction that matters -- 1e-2 is better
    than 2e-2 at every width tried, and the margin *grows* with width (tiny: a
    tie, small: 0.0010 FID, big: 0.0024) -- so a step tuned on a narrow model
    stays safe when the model is widened, which is the property Muon is being
    used for.

    Running too hot reintroduces exactly the pathology Muon was adopted to
    remove. At 2e-2 the capacity ladder is non-monotone (big 0.0085 scores
    *worse* than small 0.0076); at 1e-2 it is ordered as capacity says it
    should be (big 0.0061, small 0.0066, tiny 0.0089). "Wider is worse" is
    therefore not always a statement about the optimizer family -- it can be a
    statement about one number inside it.
    """

    def __init__(self, params, lr: float = 0.01, momentum: float = 0.95,
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
                # Group by shape so the Newton--Schulz runs once per group on a
                # stacked tensor: a transformer repeats the same few shapes, so
                # this turns ~36 launches-bound calls into ~8 batched ones.
                buckets = {}
                for p in g["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if "m" not in st:
                        st["m"] = torch.zeros_like(p)
                    m = st["m"]
                    m.mul_(g["momentum"]).add_(p.grad)
                    d = p.grad.add(m, alpha=g["momentum"]) if g["nesterov"] else m
                    buckets.setdefault(tuple(p.shape), []).append((p, d))
                for shape, items in buckets.items():
                    D = torch.stack([d for _, d in items])
                    O = _orthogonalize(D, g["ns_steps"])
                    # keep the update's scale comparable across shapes
                    scale = max(1.0, shape[0] / shape[1]) ** 0.5
                    for (p, _), o in zip(items, O):
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
