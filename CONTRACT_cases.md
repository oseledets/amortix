# amortix — contract for new gallery cases

You are adding ONE parameter-recovery case to the `amortix` package (standalone
PyTorch, Python 3.9). Read this fully. The core engine already exists and works
(OU + SEIRD). You only WRITE TWO NEW FILES and must not touch anything else.

## Repo root
`/Users/ioseledets/work/amortix` — run everything from here.

## Core API (already implemented; import, do not modify)

```python
from amortix import (SDEProblem, ODEProblem, PathObserver, TimeSeriesObserver,
                     Channel, FlowPosterior)
from amortix.prior import BoxUniform
```

**BoxUniform(low=[...], high=[...], names=[...])** — uniform prior.
`.sample(n, generator)->[n,d]`, `.dim`, `.names`, `.low`,`.high` (float tensors).

**SDEProblem** subclass — set in `__init__`: `self.prior` (BoxUniform),
`self.state_dim` (int S), `self.observer` (PathObserver), optionally
`self.corr_chol` ([S,S] lower-Cholesky tensor for correlated Brownian noise).
Implement:
- `drift(self, x, m) -> [B,S]`  (x:[B,S], m:[B,d])
- `diffusion(self, x, m) -> [B,S]`  (diagonal noise; one std per component)
- `x0_sampler(self, m, generator=None) -> [B,S]`

**PathObserver(dt_sim, n_steps, channels, n_paths=1, obs_dims=(0,))**
`channels=[Channel(every, count, label)]`. Token feature dim is 6:
`[t/horizon, x, dx, dx^2, log10(dt_obs), component_id]`. Use a fast channel
(every=1) to expose diffusion and a slow channel (every>>1) to expose drift.
`obs_dims` selects which state components are observed (e.g. (0,1) for both
prey & predator). `n_tokens` = n_paths * sum(counts) * len(obs_dims). Use n_paths=1.

**ODEProblem** subclass — set `self.prior`, `self.state_dim`,
`self.observer` (TimeSeriesObserver). Implement:
- `rhs(self, x, m, t) -> [B,S]`  (t is a python float)
- `x0(self, m) -> [B,S]`
**TimeSeriesObserver(dt_sim, n_steps, obs_steps=[int...], obs_indices=[int...], noise_std)**
Token feature dim 3: `[t/horizon, value(+noise), channel]`. `obs_indices` = which
state components are measured; `obs_steps` = integer step indices on the fine grid.

**FlowPosterior(problem, dim_model=64, n_head=4, n_layer=3, hidden=256, depth=3)**
- `.fit(n_train, epochs, batch=256, lr=3e-4, verbose=True)` → self
- `.sample(tokens, n, n_steps=60, seed=0) -> [n, d]` denormalized params;
  `tokens` is [T,F] or [1,T,F] for ONE observation.

**problem.simulate(n, generator)** → `(m [n,d], tokens [n,T,F])`
**problem.observe(m, generator)** → `(tokens [n,T,F], traj [n, n_steps+1, S])`
(for SDE traj is the raw path; for ODE traj is the clean solution, tokens noisy)

## Module contract — file `amortix/problems/<name>.py`

Must define exactly:
- `class <Name>(SDEProblem)` or `(ODEProblem)` — the problem.
- `def make()` → a fresh problem instance.
- `SOTA_NAME: str` — name of the classical baseline (e.g. "exact MLE",
  "SINDy / Kramers–Moyal", "nonlinear least squares").
- `def sota(tokens, traj, prob) -> np.ndarray` of shape `[d]`, aligned to
  `prob.prior.names`. `tokens` is a numpy [T,F] for ONE instance, `traj` the
  matching numpy [n_steps+1, S] path/solution. This is the **SOTA / classical
  comparison** — implement the standard estimator for this model (closed-form
  MLE where it exists; otherwise least-squares / method-of-moments /
  Kramers–Moyal). Make it a genuinely fair, competent baseline.
- Keep `from __future__ import annotations` at the top. No `X | Y` runtime unions.

## Example contract — file `examples/<name>_recovery.py`

Self-contained, runnable from repo root. Skeleton:

```python
from __future__ import annotations
import os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from amortix import FlowPosterior
from amortix.problems.<name> import make, sota, SOTA_NAME

def main():
    prob = make(); names = prob.prior.names
    rng = (prob.prior.high - prob.prior.low).numpy()
    post = FlowPosterior(prob).fit(n_train=4000, epochs=12)   # small for validation
    gen = torch.Generator().manual_seed(123)
    K, NP = 30, 600
    m_true = prob.prior.sample(K, generator=gen)
    tokens, traj = prob.observe(m_true, generator=gen)
    amort = np.zeros((K, prob.prior.dim)); std = np.zeros_like(amort)
    lo = np.zeros_like(amort); hi = np.zeros_like(amort); base = np.zeros_like(amort)
    for i in range(K):
        s = post.sample(tokens[i], n=NP, seed=i).numpy()
        amort[i]=s.mean(0); std[i]=s.std(0)
        lo[i]=np.quantile(s,0.05,0); hi[i]=np.quantile(s,0.95,0)
        base[i]=sota(tokens[i].numpy(), traj[i].numpy(), prob)
    mt = m_true.numpy()
    a_err=(np.abs(amort-mt)/rng*100).mean(0)
    b_err=(np.abs(base-mt)/rng*100).mean(0)
    pstd=(std/rng*100).mean(0)
    cov=(((mt>=lo)&(mt<=hi)).mean(0))*100
    print(f"\n{'param':>10} | {'amort':>7} | {'post.std':>8} | {SOTA_NAME[:14]:>14} | {'cov90':>6}")
    for j,nm in enumerate(names):
        print(f"{nm:>10} | {a_err[j]:6.2f}% | {pstd[j]:7.2f}% | {b_err[j]:13.2f}% | {cov[j]:5.0f}%")
    print(f"{'ALL':>10} | {a_err.mean():6.2f}% | {pstd.mean():7.2f}% | {b_err.mean():13.2f}% | {cov.mean():5.0f}%")

if __name__ == "__main__":
    main()
```

## Rules (strict)
- Create ONLY `amortix/problems/<name>.py` and `examples/<name>_recovery.py`.
- DO NOT edit any existing file: not `amortix/__init__.py`, not
  `amortix/problems/__init__.py`, not `baselines.py`, `sde.py`, `ode.py`,
  `README.md`, `USECASES.md`, or other cases. Registration is done centrally later.
- Choose dt_sim / n_steps / priors so the dynamics are well-posed and the
  simulator is stable (avoid blow-ups; clamp positivity where needed, e.g. CIR/GBM
  use max(x,0) under the sqrt / keep positive).
- VALIDATE: run `python examples/<name>_recovery.py` from the repo root until it
  runs cleanly and the numbers are sane (finite errors; coverage roughly 80–97%;
  amortized at least competitive with the SOTA baseline). Iterate on bugs.
- Report back: the final printed table, the two file paths, one line on the SOTA
  baseline you implemented, and any caveat (e.g. a poorly-identified parameter).
```
