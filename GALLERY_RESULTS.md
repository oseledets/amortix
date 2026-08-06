# amortix gallery — amortized flow-matching posterior vs classical SOTA

> ⚠️ **Stale relative to the current engine.** These numbers were measured on
> 2026-06-26 with mean-pool encoding, affine normalization, a standard N(0,I)
> base and `concat` conditioning. The defaults have since changed (probit
> normalization, attention pooling, data-dependent base, `xattn` conditioning)
> and the **accuracy** table below has not been re-measured — re-run
> `uv run python examples/gallery.py` to refresh it.
>
> The **calibration** claims in this file ("well-calibrated in 8/8", based on
> `cov90 ∈ [80,97]%`) were later shown to be too weak a test: strict SBC gives
> 17/29 parameters passing. See [`CALIBRATION.md`](CALIBRATION.md).

Reproduce: `python examples/gallery.py` (uniform budget: 6000 sims, 20 epochs,
40 held-out test datasets per case, 600 posterior draws). All errors are mean
absolute error as **% of the prior range**; `post.std` is the posterior standard
deviation in the same units; `cov90` is the empirical coverage of the central
90% posterior interval (calibrated ≈ 90%).

| case | kind | dim | tokens | amort | post.std | SOTA | baseline | cov90 | winner |
|---|---|---|---|---|---|---|---|---|---|
| OU | SDE-1D (validation) | 3 | 74 | **12.2%** | 13.6% | 13.3% | exact MLE | 89% | amort |
| SEIRD | ODE-5D (epidemiology) | 5 | 40 | **21.7%** | 26.2% | 24.6% | nonlinear LS | 92% | amort |
| GBM | SDE-1D (finance) | 2 | 100 | 11.7% | 13.6% | 11.7% | exact MLE | 90% | tie |
| CIR | SDE-1D (rates/vol) | 3 | 74 | 13.9% | 17.8% | **11.0%** | Euler pseudo-MLE | 97% | SOTA |
| Double-well | SDE-1D (bistable) | 3 | 116 | **14.6%** | 16.3% | 20.7% | Kramers–Moyal LS | 87% | amort |
| Stoch. Lotka–Volterra | SDE-2D (ecology) | 4 | 180 | 16.3% | 20.8% | **8.1%** | deterministic NLS | 90% | SOTA |
| FitzHugh–Nagumo | ODE-2D (neuroscience) | 4 | 25 | **13.4%** | 19.6% | 19.7% | nonlinear LS | 94% | amort |
| **SINDy-SDE** | SDE-1D (nonparam drift) | 5 | 125 | **18.0%** | 20.1% | 32.4% | SINDy / Kramers–Moyal | 89% | amort |

**Headline:** amortized wins/ties on point accuracy in **5/8** cases and is
**well-calibrated in 8/8** (cov90 in 87–97%). Training all eight ≈ 608 s on CPU;
amortized inference ≈ **119 ms / dataset** (then reusable for any new dataset).

## What the pattern means (honest read)

- **amortix wins where the target is drift / structure / an intractable
  likelihood.** SEIRD epidemic rates, the double-well drift shape (+6 pp over
  Kramers–Moyal), FitzHugh–Nagumo (+6 pp over NLS), and especially the
  **nonparametric drift discovery (SINDy-SDE): 18.0% vs 32.4%, ~1.8× better** —
  the classical least-squares library fit degrades badly on the higher-order
  coefficients (c1/c2/c3 errors 38–56%) where amortix exploits the prior.
- **Classical wins only the easy, closed-form parameter.** The diffusion level
  σ is estimated near-exactly by quadratic variation from the full fine path
  (GBM 1.8%, CIR 1.6%, double-well 2.2%), because the classical estimator sees
  all 500 fine increments whereas amortix sees only the 50-token fast channel.
  This is a **data-budget artifact, not a method limit** — wider fast channels
  or more paths close it (see `examples/ou_data_axis.py`).
- **Low-noise regimes favor deterministic fitting.** Stochastic Lotka–Volterra
  uses small multiplicative noise (s=0.05), so deterministic NLS fits the
  near-clean path almost exactly; amortix trades a little point accuracy for a
  full calibrated posterior + instant inference.
- **Calibration holds everywhere** (err ≈ post.std, cov90 ≈ 90%): the amortized
  posterior reports honest uncertainty in every case, which the single-point
  classical baselines cannot.

## Per-case parameter detail
See the bottom of `python examples/gallery.py` output for per-parameter
`amort% / post.std% / SOTA%`. Weakly-identified parameters (OU θ, CIR a, SEIRD
α, SINDy c3) correctly show the widest posteriors.
