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

## Technical report

The full technical report --- architecture, evaluation methodology, the
14-system benchmark suite with validated reference posteriors, and measured
costs --- is in the repository:
**[report/techreport.pdf](report/techreport.pdf)** (source: `report/techreport.tex`).

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
uv run amortix gallery                     # all 9 fixed-design cases vs classical baselines
```

```python
from amortix import OrnsteinUhlenbeck, FlowPosterior

prob = OrnsteinUhlenbeck()                 # dX = theta(mu - X)dt + sigma dW
post = FlowPosterior(prob).fit(n_train=12000, epochs=30)

tokens, _ = prob.observe(true_params)      # one observed dataset -> token set
samples = post.sample(tokens, n=2000)      # posterior over (theta, mu, sigma), ~180 ms
```

### Result on Ornstein–Uhlenbeck — WITHDRAWN

This section used to publish a table showing the amortized posterior beating the
exact MLE on OU (mu at 1.4% of the prior range). **Those numbers were an artifact
and have been withdrawn.** Two audits established why:

* The OU simulator started every path at `X0 = mu` *exactly*, so mu could be read
  straight off the first observation instead of being inferred. The network found
  the leak; the likelihood baseline, which ignores the initial condition, could
  not. `X0` is now drawn from the stationary law.
* An exact information floor for this problem (the Euler discretization makes OU
  an exact AR(1), so the Bayes posterior given the observed times is computable)
  puts the best achievable error at **mu 7.29%**. The published 1.4% was 5.2x
  below the optimum — arithmetically impossible, which is exactly the signature
  of the leak.

The same audit found the shipped "exact MLE" baseline is neither exact nor
near-optimal here: it carries a +31% small-sample bias in theta.

New numbers will be published only alongside the controls described below.

---

## Runnable gallery

Small, self-contained scripts in [`examples/gallery/`](examples/gallery/) --- each
prints what it demonstrates and runs in minutes:

| script | shows |
|---|---|
| [`01_quickstart_gbm.py`](examples/gallery/01_quickstart_gbm.py) | train + sample + check against an exact posterior |
| [`02_any_design_pk.py`](examples/gallery/02_any_design_pk.py) | one network, any number of observation points |
| [`03_exact_reference_cir.py`](examples/gallery/03_exact_reference_cir.py) | frozen evaluation sets, validated references, resolution floors |
| [`04_custom_problem.py`](examples/gallery/04_custom_problem.py) | add your own system in ~25 lines |

## Variable observation designs (design amortization)

Beyond fixed observation grids, `amortix` trains a single network for
**p(m | any K observation points)** — arbitrary times (and sensors), from a
handful of points to near-continuous monitoring:

```python
from amortix import FlowPosterior, DESIGN_ZOO

prob = DESIGN_ZOO["pk"]()                    # pharmacokinetics: irregular blood draws
post = FlowPosterior(prob)                   # embed/rope resolve by a verified class rule
post.fit(n_train=20000, steps=12000, retokenize=prob.make_retokenizer())
samples = post.sample_batch([tokens_for_your_6_points], n=2000)
```

The design zoo (`amortix.problems.design_zoo`) ships Heston, Merton
jump-diffusion, Hénon–Heiles, Hodgkin–Huxley, pharmacokinetics and a
Fisher–KPP reaction–diffusion PDE, each with exact-likelihood factories for
reference probes where tractable. The canonical training recipe (fresh
designs every optimizer step + the mix-K law + budget per the measured price
curves), the embedding class rule (Markov-observed → set-conditioned pairs,
otherwise bare points), and the full experimental chronicle are in
[CALIBRATION.md](CALIBRATION.md). What to compare against — three tiers of
baselines from prior-only controls to validated reference posteriors — is in
[BASELINES.md](BASELINES.md); a compact English summary of the whole study is
in [report/techreport.pdf](report/techreport.pdf).

## Reading any number in this repo

Every comparison must be read against two floors, or it means nothing:

| reference | what it is |
|---|---|
| **prior-only** | ignore the data, predict the prior mean — exactly **25.00%** of the range for a uniform prior. A parameter scored near this is *prior-limited*: where the likelihood is flat the correct posterior **is** the prior, so error-to-truth is capped there and stops measuring method quality. It is not evidence of a defect. |
| **ridge control** | degree-2 ridge on ~20 summary statistics — the cheapest serious attempt. Neural machinery that does not beat this earns nothing. |

A third caveat applies to the *classical* column: several `sota` estimators
consume the **full fine path** (500-1000 increments) while the network sees only
its token set (73-122 points), a 5-9x information advantage. Restricted to the
observed points every one of them lands on the Cramer-Rao floor, and the headline
"classical wins on sigma" (5.5% vs 2.1%) becomes a 1.17x tie. Conversely the
baselines were not clipped to the prior box the network cannot leave. Both
distortions are documented in `results/CRITIC_baselines.md`; clipping is now
applied, information parity is not yet.

**Error-to-truth is a proxy, and it breaks where the posterior is wide.** The
quantity we actually want is the posterior itself, so the primary metric is the
distance to the *true* posterior — exactly computable on `linear_gaussian`
(`examples/vs_exact.py`), approximated by MCMC where an exact likelihood exists
(`examples/vs_mcmc.py`), and checked by SBC everywhere else. A wide posterior is a
legitimate answer; a posterior of the *wrong width* is the error.

`uv run python examples/scoreboard.py` reports both controls next to the amortized result.
Coverage-based calibration claims are especially treacherous: a posterior that
simply returns the prior passes `cov90 in [80,97]%`, so that criterion alone can
never show a method learned anything. Use SBC (`examples/calib_gallery.py`).

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
| `problems/`   | the use-case gallery (9 fixed-design cases + the 6-case design zoo) |

Swap points: `SetTransformer` ↔ DeepSet/MLP encoder; Euler–Maruyama ↔ `torchsde`
for stiff/multi-D systems; the velocity net / probability path are isolated.

---

## Use-case gallery

Nine cases, each a self-contained `Problem` (simulator + prior + classical
baseline). Re-measured on the fixed engine — 40 000 simulations, 12 000 optimizer
steps, 100 held-out datasets per case. Mean absolute error of the posterior mean,
as % of the prior range; full table and caveats in
[`GALLERY_RESULTS.md`](GALLERY_RESULTS.md).

| case | prior-only | ridge control | **amortized** | classical |
|---|---|---|---|---|
| linear_gaussian (exact posterior known) | 24.19% | 18.24% | **4.25%** | 4.19% ← Bayes optimum |
| ou | 24.84% | 11.12% | **8.90%** | 8.45% |
| seir | 24.83% | 19.97% | **14.24%** | 24.04% |
| gbm | 24.84% | 10.04% | 10.48% | 11.23% |
| cir | 24.14% | 11.06% | **8.27%** | 8.50% |
| double_well | 24.14% | 14.02% | **11.01%** | 11.76% |
| stoch_lv | 24.19% | 11.18% | **4.95%** | 9.83% |
| fhn | 24.19% | 15.50% | **10.10%** | 11.57% |
| sindy_sde | 24.83% | 16.83% | 16.90% | 28.46% |

Beats the ridge control in 7/9 (the other two are ties) and the classical
estimator in 7/9. On the one case where the exact posterior is computable we land
within 1.4% of the Bayes optimum, and the sampled posterior matches the exact one
to 1.6× the Monte-Carlo floor.

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
