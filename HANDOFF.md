# amortix — handoff

**What it is.** A library for *amortized* Bayesian parameter recovery in
dynamical systems: give it a prior + a simulator (ODE/SDE), it trains a
conditional-flow-matching posterior once, then infers parameters for any new
dataset in ~100 ms with uncertainty. Engine = arXiv:2503.01375
(Sherki–Oseledets–Muravleva), packaged for reuse, with an SDE-first gallery and
an honest amortized-vs-classical benchmark harness.

## Start here (5 minutes)

```bash
git clone <this repo> && cd amortix
uv sync --extra plot          # Python 3.11 pinned, deps locked in uv.lock
uv run pytest -q              # 23 smoke tests, ~20 s — verifies the install
uv run amortix recover ou     # first real result (~2 min CPU)
```

If `uv` is missing: `curl -LsSf https://astral.sh/uv/install.sh | sh`.

## Read in this order

| doc | what it answers |
|---|---|
| [`METHOD.md`](METHOD.md) | how the method works, end to end, as implemented |
| [`CALIBRATION.md`](CALIBRATION.md) | the calibration investigation — **read before trusting any posterior** |
| [`USECASES.md`](USECASES.md) | ~22 cited real-world applications + positioning vs `sbi`/BayesFlow |
| [`GALLERY_RESULTS.md`](GALLERY_RESULTS.md) | accuracy vs classical baselines (⚠️ stale, see banner) |
| [`ADDING_CASES.md`](ADDING_CASES.md) | the contract to follow when adding a new case |

## Code map

```
amortix/
  prior.py        BoxUniform + probit normalization (prior -> N(0,I) exactly)
  sde.py          Euler-Maruyama (vector state, correlated noise), PathObserver, SDEProblem
  ode.py          batched RK4, TimeSeriesObserver, ODEProblem
  encoder.py      SetTransformer (RoPE, ReLU^2, RMSNorm) -> token memory + pooling
  flow.py         FlowPosterior: CFM training, data-dependent base, RK4 sampler,
                  concat | xattn conditioning
  diagnostics.py  SBC, coverage curves, calibration report + plots
  baselines.py    classical estimators (OU MLE, SEIRD nonlinear LS)
  problems/       8 gallery cases, each exposing make() / sota() / SOTA_NAME
examples/         runnable studies; results land in results/
tests/            smoke tests (contract + install)
```

## Status — what is solid, what is not

**Solid**
- **Verified against a known answer.** On `linear_gaussian` the posterior is
  analytic; we reproduce it to 1.6× the Monte-Carlo sampling floor and land within
  1.4% of the Bayes optimum. The CFM core was separately checked against the
  closed-form optimal velocity field (5.6–8.5% agreement, correlation 0.871 vs an
  exact 0.850).
- **Accuracy**: beats the ridge control in 7/9 cases (2 ties) and the classical
  estimator in 7/9 — see [`GALLERY_RESULTS.md`](GALLERY_RESULTS.md).
- **Calibration**: 29/32 parameters pass strict SBC, mean calibration error 1.3pp,
  coverage 48–52% / 89–92% against nominal 50/90 — see
  [`CALIBRATION.md`](CALIBRATION.md).
- An MCMC gold standard exists for the cases with a tractable likelihood
  (`amortix/mcmc.py`), and every comparison reports its Monte-Carlo floor.
- Reproducible environment (`uv.lock`, `.python-version`), 37 tests.

**Not solid — open work**
1. **Dependence structure is under-shrunk.** Correlations come out right in sign
   and structure but attenuated 10–40% (on the exact-posterior testbed: −0.640
   against a true −0.710, 0.349 against 0.415). This single defect is what the
   three remaining SBC failures are — `stoch_lv` alpha (which enters as a product
   with beta), `linear_gaussian` m2/m4, and `gbm` sigma — and it is also the 1.6×
   gap to the Monte-Carlo floor. Everything else on the calibration board passes.
2. **Baselines are not information-matched.** For the SDE cases the classical
   estimators read the full 500–1000-point path while the network sees 73–122
   tokens, a 5–9× advantage. Restricted to the observed points they all land on
   the Cramér–Rao floor. Our wins there are over an advantaged opponent, and so is
   the single loss on OU. See `results/CRITIC_baselines.md`.
3. **CPU-bound.** 623 ms/optimizer-step single-threaded for a 483K-parameter model
   on 74 tokens — dispatch overhead, not arithmetic. Batch 256 costs the same per
   step as 64, so at a step-denominated budget the larger batch is free variance
   reduction (re-measure on an idle machine before changing the default).
   `torch.compile` and a GPU path are both untried.
4. **No CI, not on PyPI.**
5. **Not ported:** 2D Darcy flow (the paper's PDE case), Heston (the core already
   supports correlated noise via `corr_chol`), the bioprocess case from
   arXiv:2604.22496.

**How to read any number here.** Never quote an accuracy figure without the
prior-only control (25% by construction) and the ridge control beside it, and
never quote a calibration figure without an accuracy one: a posterior that returns
the prior passes SBC *and* passes `cov90 in [80,97]%`. Budget is denominated in
optimizer steps, never epochs, and convergence is judged by the distance to a
reference posterior, never by the loss — most of the CFM loss is irreducible
variance and it can rise while the posterior improves.

## Reproducing the headline runs

```bash
uv run python examples/gallery.py                 # accuracy vs classical SOTA, all cases
uv run python examples/calib_gallery.py --n_train 50000 --epochs 60 \
    --n_sims 500 --n_post 200 --out results/CALIB_GALLERY_XATTN.md   # SBC canon (slow)
uv run python examples/test_conditioning.py double_well \
    --modes concat xattn --seeds 0 1 2 --n_train 20000 --epochs 50    # controlled A/B
uv run python examples/diagnostics_demo.py ou     # SBC report + plot for one case
```

Long runs are CPU-bound and take tens of minutes; every study writes its results
to `results/` as markdown + JSON so nothing is lost to a closed terminal.

## Adding a new case

Follow [`ADDING_CASES.md`](ADDING_CASES.md): write
`amortix/problems/<name>.py` exposing a `Problem` subclass plus `make()`,
`SOTA_NAME` and `sota(tokens, traj, prob)`, register it in
`amortix/problems/__init__.py` (`GALLERY`). It is then automatically covered by
`examples/recover.py`, the gallery benchmarks and the CLI.
`uv run pytest -q` will then exercise it automatically.
