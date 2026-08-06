# Adding a case to the gallery

A "case" is one parameter-recovery problem: a prior, a numerical simulator, an
observation scheme, and a classical baseline to compare against. Adding one is a
single file plus one line of registration; the benchmarks, the CLI and the test
suite then pick it up automatically.

## 1. The core API you build on

```python
from amortix import SDEProblem, ODEProblem, PathObserver, TimeSeriesObserver, Channel
from amortix.prior import BoxUniform
```

**`BoxUniform(low, high, names)`** — uniform prior on a box.
`.sample(n, generator) -> [n, d]`, `.dim`, `.names`, `.low`, `.high`.

**`SDEProblem`** — for `dX = drift(X,m) dt + diffusion(X,m) dW`. In `__init__`
set `self.prior`, `self.state_dim` (S), `self.observer`, and optionally
`self.corr_chol` (an `[S,S]` lower-Cholesky for correlated Brownian noise).
Implement:
- `drift(self, x, m) -> [B,S]` and `diffusion(self, x, m) -> [B,S]` (diagonal noise)
- `x0_sampler(self, m, generator=None) -> [B,S]`

**`ODEProblem`** — for `dX/dt = rhs(X,m,t)`. Set `self.prior`, `self.state_dim`,
`self.observer`; implement `rhs(self, x, m, t) -> [B,S]` and `x0(self, m) -> [B,S]`.

**Observers** turn a trajectory into a permutation-invariant token set:
- `PathObserver(dt_sim, n_steps, channels, n_paths=1, obs_dims=(0,))` with
  `channels=[Channel(every, count, label)]`. Token = `[t/horizon, x, Δx, Δx²,
  log₁₀ dt_obs, component_id]`. Use a **fast** channel (`every=1`) to expose the
  diffusion and a **slow** one (`every ≫ 1`) to expose the drift — this
  multi-resolution view is what lets one network identify both.
- `TimeSeriesObserver(dt_sim, n_steps, obs_steps, obs_indices, noise_std)`.
  Token = `[t/horizon, value(+noise), channel]`.

## 2. The module contract — `amortix/problems/<name>.py`

Define exactly:

| symbol | meaning |
|---|---|
| `class <Name>(SDEProblem \| ODEProblem)` | the problem |
| `make()` | returns a fresh instance |
| `SOTA_NAME: str` | name of the classical baseline, e.g. `"exact MLE"` |
| `sota(tokens, traj, prob) -> np.ndarray [d]` | the classical estimate, aligned to `prob.prior.names` |

`sota` receives numpy arrays for **one** instance: `tokens [T,F]` and the matching
`traj [n_steps+1, S]`. Implement the estimator a practitioner would actually use —
closed-form MLE where one exists, otherwise least squares, method of moments or
Kramers–Moyal. A weak baseline makes the comparison meaningless.

Start the file with `from __future__ import annotations` (the package supports
Python 3.9, so no `X | Y` runtime unions).

## 3. Register it

Add the class import and the module name to `GALLERY` in
`amortix/problems/__init__.py`. That is all — `examples/recover.py`,
`examples/gallery.py`, `examples/calib_gallery.py`, the `amortix` CLI and the
smoke tests all iterate over `GALLERY`.

## 4. Check it

```bash
uv run pytest -q                        # contract tests cover the new case automatically
uv run python examples/recover.py <name>   # accuracy vs the classical baseline
uv run amortix sbc <name>                  # strict calibration (SBC)
```

Sanity targets: finite errors, `cov90` near 90%, and the amortized posterior at
least competitive with the baseline. If SBC fails, read
[`CALIBRATION.md`](CALIBRATION.md) first — a marginal budget, correlated or
weakly-identified parameters are known and documented causes.

## 5. Numerical hygiene

Choose `dt_sim`, `n_steps` and prior ranges so the simulator is stable across the
*whole* prior box — clamp positivity where the model requires it (`x.clamp_min`
for GBM/CIR-type multiplicative or square-root dynamics), and keep confining
terms (e.g. a negative leading cubic coefficient) inside the prior so paths
cannot blow up. A silently diverging simulator produces non-finite tokens, which
the smoke tests will catch.
