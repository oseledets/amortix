# amortix gallery — the honest scoreboard

Re-measured on the fixed engine. **The accuracy numbers below predate the
universal learnable embedding** (`embed="auto"` → wbasis on all SDE cases; see
[`CALIBRATION.md`](CALIBRATION.md)) — they were measured with the plain linear
embedding and, for `gbm`, before its sigma centre bias (+0.48 posterior-sd)
was removed. Calibration is fully re-verified on the new defaults (31/32);
accuracy has not been re-measured since and likely understates the current
engine on `gbm`, `sindy_sde`, and `stoch_lv`. Reproduce with

```bash
uv run python examples/scoreboard.py --n_train 40000 --steps 12000 --n_test 100 --n_post 400
```

40 000 simulations, 12 000 optimizer steps, 100 held-out datasets × 400 posterior
draws per case. Errors are mean absolute error of the posterior mean, as **% of
the prior range**.

Every number must be read against the two controls, or it means nothing:

* **prior-only** — ignore the data, predict the prior mean. Exactly 25% by
  construction for a uniform prior.
* **ridge** — a degree-2 ridge on ~20 hand-written summary statistics; a closed-form
  convex estimator, the cheapest serious attempt. Neural machinery that does not
  beat this earns nothing.

| case | prior-only | ridge | **amortized** | classical | verdict |
|---|---|---|---|---|---|
| linear_gaussian | 24.19% | 18.24% | **4.25%** | 4.19% | beats ridge |
| ou | 24.84% | 11.12% | **8.90%** | 8.45% | beats ridge |
| seir | 24.83% | 19.97% | **14.24%** | 24.04% | beats ridge |
| gbm | 24.84% | 10.04% | 10.48% | 11.23% | tie |
| cir | 24.14% | 11.06% | **8.27%** | 8.50% | beats ridge |
| double_well | 24.14% | 14.02% | **11.01%** | 11.76% | beats ridge |
| stoch_lv | 24.19% | 11.18% | **4.95%** | 9.83% | beats ridge |
| fhn | 24.19% | 15.50% | **10.10%** | 11.57% | beats ridge |
| sindy_sde | 24.83% | 16.83% | 16.90% | 28.46% | tie |

**Beats the ridge control in 7/9; the two remaining are ties** (10.48 vs 10.04 and
16.90 vs 16.83). **Beats the classical estimator in 7/9.** The two exceptions are
the right ones: on `linear_gaussian` the "classical" column *is* the Bayes optimum,
and we sit 1.4% above it; on `ou` the classical estimator reads 500 increments to
our 74 tokens.

## What the aggregate hides

**`stoch_lv` is the clearest win**: 4.95% against 9.83% for the classical fit, with
posterior contraction of 3.6–5.8× on all four parameters. This is the case whose
correlated parameters used to fail SBC, before the velocity field was fixed.

**Where we lose, it is one parameter, not the method.** On `gbm` we *win* on the
diffusion (σ 3.79% vs the ridge's 5.31%) and lose on the notoriously hard drift
(μ 17.18% vs 14.77%). On `sindy_sde` we win σ (5.17% vs 6.04%), c₂ is honestly
prior-limited (contraction 1.04×, the ridge cannot beat the prior there either),
and the aggregate is dragged down by c₃ — flagged **WIDTH WRONG**: the ridge
locates it while our posterior stays near the prior. That is a real defect and the
one place on this board still worth attacking.

## Against the previous numbers

Every case improved over the numbers this file used to publish:

| case | old | new |
|---|---|---|
| ou | 12.2% | 8.90% |
| seir | 21.7% | 14.24% |
| gbm | 11.7% | 10.48% |
| cir | 13.9% | 8.27% |
| double_well | 14.6% | 11.01% |
| stoch_lv | 16.3% | **4.95%** |
| fhn | 13.4% | 10.10% |
| sindy_sde | 18.0% | 16.90% |

Not a like-for-like comparison, and deliberately so: the old numbers were measured
on a broken engine (a velocity field factorized across parameters, a degenerate
time embedding, ~88–460 optimizer steps) *and* on problems that leaked — OU handed
the network μ through `X0 = μ`, CIR handed it `b`, and `stoch_lv`'s observer showed
only 40% of the horizon its baseline was fitted on. Those are fixed; OU is now the
canonical two-parameter process. See [`CALIBRATION.md`](CALIBRATION.md) and
`results/CRITIC_*.md`.

## The caveat that still stands

For the SDE cases the classical estimators consume the **full fine path**
(500–1000 increments) while the network sees only its token set (73–122 points), a
5–9× information advantage. Restricted to the observed points every one of them
lands on the Cramér–Rao floor. So our wins there are over an advantaged opponent —
and so is the single loss on `ou`. Information parity is not yet implemented; see
`results/CRITIC_baselines.md`.

Accuracy is only half the claim. For calibration — whether the posterior *widths*
are right, not just its centre — see [`CALIBRATION.md`](CALIBRATION.md).
