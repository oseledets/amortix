# amortix

**Amortized parameter recovery for dynamical systems via flow matching.**

One method, many recovery problems: give a *prior* and a *simulator*, get a fast
*amortized posterior* over the parameters. A transformer set-encoder conditions a
Conditional Flow Matching velocity field; inference for any new dataset is a single
ODE solve (milliseconds), with calibrated uncertainty.

The engine is the method of **Sherki, Oseledets, Muravleva — *Bayesian Inverse
Problems Meet Flow Matching* ([arXiv:2503.01375](https://arxiv.org/abs/2503.01375))**,
packaged for reuse. The same recipe was applied by colleagues to bioprocess model
calibration ([arXiv:2604.22496](https://arxiv.org/abs/2604.22496)). `amortix`
turns that recipe into a library + a curated gallery of recovery problems, with a
first-class focus on **SDE recovery**, where the likelihood is intractable and
amortized simulation-based inference shines most.

---

## Setup

Reproducible environment via [uv](https://docs.astral.sh/uv/) (Python pinned to
3.11 in `.python-version`, deps locked in `uv.lock`):

```bash
uv sync --extra plot            # creates .venv, installs torch/numpy/scipy/matplotlib + amortix (editable)
uv run amortix cases                    # the CLI is installed with the package
```

(`uv run` auto-uses the project venv; or activate `.venv/bin/activate`.)

**As a dependency / standalone tool.** `amortix` is a normal installable package
(hatchling build backend, console script `amortix`):

```bash
uv add amortix                        # as a dependency of another uv project
uv tool install amortix               # as a standalone CLI on your PATH
uvx --from amortix amortix cases      # run once without installing
uv build                              # produce dist/*.whl + *.tar.gz  (uv publish to release)
```

Until it is on PyPI, point those at the repo or a built wheel, e.g.
`uv tool install /path/to/amortix` or `uvx --from ./dist/amortix-0.1.0-py3-none-any.whl amortix cases`.

## Quickstart

```bash
uv run amortix recover ou                  # train + benchmark one case vs exact MLE
uv run amortix sbc ou                      # strict calibration check (SBC)
uv run amortix gallery                     # all 8 cases vs their classical baselines
```

```python
from amortix import OrnsteinUhlenbeck, FlowPosterior

prob = OrnsteinUhlenbeck()                 # dX = theta(mu - X)dt + sigma dW
post = FlowPosterior(prob).fit(n_train=12000, epochs=30)

tokens, _ = prob.observe(true_params)      # one observed dataset -> token set
samples = post.sample(tokens, n=2000)      # posterior over (theta, mu, sigma), ~180 ms
```

### Result on Ornstein–Uhlenbeck (validation case)

OU has an *exact* closed-form MLE, so we can check the amortized posterior against
a near-optimal classical estimator. Mean abs error as % of prior range, over 120
held-out datasets:

| param | amortized | exact MLE | 90% coverage |
|-------|----------:|----------:|-------------:|
| theta | 16.6%     | 24.5%     | 91%          |
| mu    | 1.4%      | 7.8%      | 100%         |
| sigma | 5.5%      | 2.1%      | 88%          |
| **all** | **7.8%** | **11.5%** | **93%**     |

The amortized posterior is **competitive with the exact MLE** (better overall —
it is more robust to sampling resolution for the drift, since it consumes a
multi-resolution view). The point is not to beat MLE on OU — it is that *the same
code transfers to SDEs with no tractable likelihood*, where MLE is unavailable.
(`uv run python examples/recover.py ou --plot`)

*Coverage ≈ 90% here looks reassuring, but note the caveat below: coverage alone
is a weak calibration test, and OU's μ/σ do **not** pass strict SBC at this
budget. Numbers measured on the original engine.*

---

## Why SDE recovery is the sweet spot

For a deterministic ODE the likelihood is often tractable, so classical methods
compete. For an SDE `dX = a(X,m)dt + b(X,m)dW` the likelihood is intractable —
this is exactly where amortized simulation-based inference wins. Two design points
make it work:

- **Drift vs diffusion live on different timescales.** Diffusion is read from the
  quadratic variation (high-frequency increments); drift / mean-reversion from the
  long horizon. `PathObserver` emits **multiple observation channels** (a fast
  short window + a slow long window); each token carries its resolution, so one
  network identifies both. *(This is visible above: amortix beats fine-resolution
  MLE on drift `theta, mu`, MLE wins slightly on `sigma`.)*
- **Sufficient-statistic features per token:** `[t, x, dx, dx², log dt]` — `dx²`
  is the per-step quadratic-variation cue.

---

## Architecture

```
Problem            = prior + simulator + observation spec     (the contract)
  └ SDEProblem     = drift + diffusion + PathObserver         (auto-builds simulator)
FlowPosterior      = SetTransformer encoder + CFM velocity net + ODE sampler
```

| file | role |
|------|------|
| `prior.py`    | `BoxUniform`: sampling + normalization to the flow's base space |
| `sde.py`      | Euler–Maruyama (vector state + correlated noise), `Channel`/`PathObserver`, `SDEProblem` base |
| `ode.py`      | batched RK4, `TimeSeriesObserver`, `ODEProblem` base |
| `encoder.py`  | `SetTransformer`: RoPE attention, ReLU² FFN, RMSNorm, masked mean-pool |
| `flow.py`     | `FlowPosterior`: CFM training + RK4 ODE posterior sampling |
| `baselines.py`| classical baselines (OU MLE, SEIRD nonlinear LS) |
| `problems/`   | the use-case gallery (8 cases, each with a SOTA baseline) |

Swap points: `SetTransformer` ↔ DeepSet/MLP encoder; Euler–Maruyama ↔ `torchsde`
for stiff/multi-D systems; the velocity net / probability path are isolated.

---

## Use-case gallery

Each case is a self-contained `Problem` (simulator + prior + SOTA baseline). Run
the whole benchmark with `python examples/gallery.py`; full results +
interpretation in [`GALLERY_RESULTS.md`](GALLERY_RESULTS.md). Under a uniform
budget the amortized posterior **wins/ties on accuracy in 5/8 cases**, with
~119 ms amortized inference per dataset.

> ⚠️ **These accuracy numbers are from the original engine** (mean-pool encoder,
> affine normalization, standard base, `concat` conditioning) and have **not been
> re-measured** since the calibration work changed the defaults to probit
> normalization + attention pooling + data-dependent base + `xattn` conditioning.
> Treat them as indicative, not current — re-run `examples/gallery.py` to refresh.
>
> On **calibration**, an earlier claim of "calibrated in 8/8" was based only on
> `cov90 ∈ [80,97]%`, which is too weak a test. Strict SBC says otherwise:
> **17/29 parameters pass** rank-uniformity. Coverage is usable everywhere
> (cov50 45–63%, cov90 86–92%), but strict calibration on correlated /
> multimodal / weakly-identified parameters is **not** solved. See
> [`CALIBRATION.md`](CALIBRATION.md) and [`METHOD.md`](METHOD.md) §9.

| case | type | params | SOTA baseline | amort vs SOTA | status |
|------|------|--------|---------------|--------------|--------|
| `OrnsteinUhlenbeck` | SDE-1D | θ, μ, σ | exact MLE | 12.2% vs 13.3% | ✅ |
| `GeometricBrownianMotion` | SDE-1D (finance) | μ, σ | exact MLE | 11.7% vs 11.7% | ✅ |
| `CIR` | SDE-1D (rates) | a, b, σ | Euler pseudo-MLE | 13.9% vs 11.0% | ✅ |
| `DoubleWell` | SDE-1D (bistable) | θ₁, θ₂, σ | Kramers–Moyal | 14.6% vs 20.7% | ✅ |
| `StochasticLotkaVolterra` | SDE-2D (ecology) | α, β, δ, γ | deterministic NLS | 16.3% vs 8.1% | ✅ |
| `PolynomialDriftSDE` | SDE-1D (nonparam drift) | c₀…c₃, σ | SINDy / Kramers–Moyal | **18.0% vs 32.4%** | ✅ |
| `FitzHughNagumo` | ODE-2D (neuron) | a, b, ε, I | nonlinear LS | 13.4% vs 19.7% | ✅ |
| `SEIRD` | ODE-5D (epidemic) | 5 rates | nonlinear LS | 21.7% vs 24.6% | ✅ |
| 2D Darcy flow | PDE | 16D (KL) | — | — | 📋 |
| itaconic-acid kinetics | ODE | kinetic | regression | — | 📋 |

See [`USECASES.md`](USECASES.md) for a cited catalog (~22) of real-world
applications this engine targets (gravitational waves, neuroscience, cosmology,
finance, epidemiology) and positioning vs `sbi` / BayesFlow.

---

## Positioning (honest)

The closest existing tools are **BayesFlow** (amortized inference with summary
networks, now with flow-matching backbones) and **sbi**. `amortix`'s niche is not
the engine but: (1) a curated comp-math / engineering problem gallery wired to
real numerical solvers, (2) a benchmark harness against classical methods
(MLE / MCMC / regression), (3) dedicated SDE-recovery tooling. This is the
"catalog + comparison" line of the numerical-methods archaeology effort.

## Roadmap

1. ✅ SDE recovery seed (OU) end-to-end, validated vs exact MLE.
2. ✅ SDE gallery: GBM, CIR, double-well, stochastic Lotka–Volterra.
3. ✅ Nonparametric drift recovery (basis coeffs) → SINDy-SDE bridge.
4. ✅ Ported the paper's disease-dynamics case (SEIRD) + neuroscience ODE (FHN).
5. ✅ Unified `gallery.py` benchmark harness vs classical SOTA (see `GALLERY_RESULTS.md`).
6. Heston stochastic-volatility (2D correlated SDE) — core already supports `corr_chol`.
7. Diagnostics module: SBC, posterior-predictive, coverage as reusable API.
8. 2D Darcy flow (PDE: FD solver + KL expansion) — last paper case.
9. MCMC baseline + a convergence-budget run for a publication-grade table.
```
