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
uv run python examples/ou_recovery.py    # first real result (~2.5 min CPU)
```

If `uv` is missing: `curl -LsSf https://astral.sh/uv/install.sh | sh`.

## Read in this order

| doc | what it answers |
|---|---|
| [`METHOD.md`](METHOD.md) | how the method works, end to end, as implemented |
| [`CALIBRATION.md`](CALIBRATION.md) | the calibration investigation — **read before trusting any posterior** |
| [`USECASES.md`](USECASES.md) | ~22 cited real-world applications + positioning vs `sbi`/BayesFlow |
| [`GALLERY_RESULTS.md`](GALLERY_RESULTS.md) | accuracy vs classical baselines (⚠️ stale, see banner) |
| [`CONTRACT_cases.md`](CONTRACT_cases.md) | the contract to follow when adding a new case |

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
- Core engine works across 8 cases (SDE 1D/2D, ODE 2D/5D, nonparametric drift).
- Every case ships a competent classical baseline for honest comparison.
- Reproducible environment (`uv.lock`, `.python-version`), smoke-tested.
- The method wins clearly where the target is drift / structure / an intractable
  likelihood — most strikingly nonparametric drift discovery (SINDy-SDE:
  18.0% vs 32.4% for classical SINDy/Kramers–Moyal).

**Not solid — open work**
1. **Strict calibration on hard posteriors.** SBC: 17/29 parameters pass.
   Failures cluster on correlated (stoch-LV α/β), multimodal (double-well θ₂) and
   weakly-identified (CIR a/b, FHN I) parameters. Coverage is usable
   (cov50 45–63%, cov90 86–92%) — it is rank-uniformity that fails.
   *Leading suspect:* the data-dependent base is a **diagonal** Gaussian and
   cannot seed posterior correlations. First thing to try: low-rank / correlated
   / mixture base.
2. **`GALLERY_RESULTS.md` accuracy numbers are stale** — measured on the old
   engine (mean-pool, affine norm, standard base, concat). Re-run
   `examples/gallery.py` to refresh before quoting them anywhere.
3. **No CI, no packaging to PyPI**, single-author code, CPU-only (no GPU path
   exercised).
4. **Not ported yet:** 2D Darcy flow (the paper's PDE case), Heston (the core
   already supports correlated noise via `corr_chol`), the bioprocess case from
   arXiv:2604.22496.

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

Follow [`CONTRACT_cases.md`](CONTRACT_cases.md): write
`amortix/problems/<name>.py` exposing a `Problem` subclass plus `make()`,
`SOTA_NAME` and `sota(tokens, traj, prob)`, register it in
`amortix/problems/__init__.py` (`GALLERY`), and add `examples/<name>_recovery.py`.
`uv run pytest -q` will then exercise it automatically.
