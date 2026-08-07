# Closing the optimization gap on `linear_gaussian`

`linear_gaussian` is the only case whose posterior is known exactly, so "how far
is the sampled posterior from the truth" is a number, not a self-consistency
check. Every number below is that number:

    metric = mean over held-out datasets of
             amortix.metrics.energy_distance_scaled(our 400 draws,
                                                    exact 400 draws, prior range)

**Monte-Carlo floor of the estimator** — two *independent* exact draws of the SAME
posterior, 400 vs 400: **0.00075 ± 0.00009** on the 12-dataset monitoring set,
**0.00062** on the 32-dataset held-out set (the task statement quotes 0.0011 for
its own reference set; the floor depends on which datasets are drawn). Nothing
can score below this — it is the noise of the estimator itself.

## The disease and the diagnosis

At a constant learning rate the metric stops improving after ~2000 steps and then
oscillates. Reproduced here at 6000 steps, both seeds, checked every ~600 steps:

    seed 0  0.3527 0.0506 0.0210 0.0136 0.0155 0.0107 0.0104 0.0243 0.0147 0.0108 0.0161
    seed 1  0.2840 0.0492 0.0314 0.0193 0.0204 0.0223 0.0137 0.0153 0.0129 0.0094 0.0119

That is the signature of SGD on a noisy objective: the iterates do not converge
to a point, they random-walk in a ball around the optimum whose radius is
`~ lr x gradient noise`. The CFM gradient is almost all noise (~78% of
`||v - (z1-z0)||^2` is the irreducible `Var(z1-z0 | z_t)`), so the ball is wide
and any single iterate is a lottery ticket. Three things shrink it, and they are
exactly what the training loop was missing:

1. **average the iterates** (EMA) — cancels the walk without changing the walk;
2. **shrink the step size** (cosine decay) — contracts the ball to a point;
3. **shrink the gradient noise itself** (several CFM pairs per encoded batch).

All three work. The first is the biggest, the third is the one that keeps paying.

## Protocol

- `linear_gaussian`, `n_train=20000`, `batch=64`, `lr=3e-4`, Adam, xattn+data base.
- 12 held-out reference datasets, 400 draws each, sampled `midpoint/20`; the
  sampler seed is fixed, so between checks only the model moves (common random
  numbers). A separate **32-dataset set, never used to choose anything**, is used
  for the final verification.
- **2 seeds per configuration** except where `n=1` (second seed dropped when the
  machine was saturated; those rows are marked and none of them is a winner).
- `final` = last monitored check; `tail2` = mean of the last two checks; both
  averaged over seeds. `sd` = spread of `tail2` across seeds — that is the
  yardstick for "inside seed noise" (typically 0.0002-0.0016).
- caveat: the runs made before the `fit` fix in §6 have their last check up to
  ~1500 steps short of the true end, which *under*-states the decaying schedules
  (they settle in exactly that last stretch). The §4 verification evaluates the
  finished model directly and is unaffected.
- caveat, second: `amortix/encoder.py` (a different workstream) gained a feature
  normalization worth ~17% on this same metric partway through this campaign.
  **§1, §2 and the 4000/12000-step k=1..16 rows were all measured before that
  change; §4 and the k=64 / k=16@12000 / 24000-step rows after it.** Every
  comparison drawn below is *within* one of those two groups — in particular the
  headline baseline-vs-recommended pair in §4 was trained and evaluated back to
  back on the same code — but do not compare a §1 number against a §4 number and
  attribute the difference to the optimizer.

## 1. One knob at a time, 4000 steps

| configuration                                | final | tail2 | sd | n |
|----------------------------------------------|-------|-------|----|---|
| **baseline** — constant lr, no EMA, no clip, k=1 | 0.01784 | 0.01896 | 0.0007 | 2 |
| + gradient clipping 1.0                      | 0.02010 | 0.01777 | 0.0021 | 2 |
| + `k_pairs=4`                                | 0.01490 | 0.01375 | — | 1 |
| + cosine decay + warmup 300                  | 0.00754 | 0.00925 | 0.0005 | 2 |
| + **EMA 0.999**                              | 0.00438 | 0.00524 | 0.0007 | 2 |

EMA alone is a **4.1x** cut, cosine alone **2.4x**, `k_pairs=4` alone 1.2x,
gradient clipping nothing at all (its difference from the baseline is inside the
seed spread). And EMA does not merely lower the number: the curve stops
oscillating and becomes monotone (checks at steps 1248/2184/3120/4056) —

    baseline  0.0584  0.0197  0.0176  0.0189   seed 0  |  0.0588 0.0209 0.0226 0.0168  seed 1
    + EMA     0.0326  0.0086  0.0053  0.0039   seed 0  |  0.0353 0.0112 0.0069 0.0049  seed 1

monotone on both seeds, and the last check is the best one on both. That is the
real result: after averaging, the "plateau" turns out not to be a plateau at all
— the model was still improving underneath the noise, and the oscillation was
hiding it.

## 2. Tuning EMA, learning rate, schedule

| configuration (4000 steps unless noted)                   | final | tail2 | sd | n |
|-----------------------------------------------------------|-------|-------|----|---|
| EMA 0.99, constant lr                                     | 0.00447 | 0.00514 | 0.0003 | 2 |
| EMA 0.999, constant lr                                    | 0.00438 | 0.00524 | 0.0007 | 2 |
| EMA 0.9995, constant lr                                   | 0.00502 | 0.00571 | — | 1 |
| EMA 0.999 + cosine                                        | 0.00841 | 0.00914 | — | 1 |
| EMA 0.999 + cosine, `lr=1e-3`                             | 0.00444 | 0.00473 | 0.0005 | 2 |
| EMA 0.999, constant `lr=1e-3`                             | 0.00611 | 0.00600 | 0.0016 | 2 |
| — 12000 steps —                                           |         |         |    |   |
| EMA 0.999, constant lr                                    | 0.00397 | 0.00390 | 0.0002 | 2 |
| EMA 0.999 + cosine                                        | 0.00362 | 0.00357 | 0.0002 | 2 |

Three things to read here.

**The EMA horizon is not delicate**: 0.99 and 0.999 are indistinguishable,
0.9995 (a 2000-step horizon, half the run) is slightly worse. 0.999 it is.

**Cosine and EMA look like they conflict at 4000 steps** (0.0084 vs 0.0044 for
EMA alone) — but that is not a conflict, it is a budget effect. Decaying the step
size halves the distance travelled in weight space, and at 4000 steps the model
is still *under-trained*, so it cannot afford that. Give it either a bigger step
(`lr=1e-3`: back to 0.0044) or more steps, and the effect reverses: at 12000
steps cosine+EMA (0.00362) beats constant+EMA (0.00397), and more importantly it
*settles*. At a constant rate the best check is still well below the last one
even with EMA (seed 0: 0.00307 at step 4000 against 0.00418 at the end — the run
gets *worse* with more training and the endpoint is a lottery); with the decay
the last check is the best one or within 8% of it on every seed.

**Raising the learning rate is not a substitute.** `lr=1e-3` at a constant rate
is worse than `3e-4` (0.0061 vs 0.0044).

## 3. Variance reduction in the objective: `k_pairs`

`k_pairs = k` draws k independent `(t, z0)` pairs against **one** encoder pass:
the observation memory is tiled, not re-encoded, so all k pairs push gradients
into the same encoding and the noisiest term of the gradient is averaged k times.

| k (cosine + warmup + EMA 0.999)     | steps | final | tail2 | sd | n | enc |
|-------------------------------------|-------|-------|-------|----|---|-----|
| 1                                   | 4000  | 0.00841 | 0.00914 | — | 1 | old |
| 4                                   | 4000  | 0.00441 | 0.00464 | 0.0001 | 2 | old |
| 16                                  | 4000  | 0.00253 | 0.00261 | 0.0002 | 2 | old |
| 1                                   | 12000 | 0.00362 | 0.00357 | 0.0002 | 2 | old |
| 4                                   | 12000 | 0.00259 | 0.00278 | 0.0001 | 2 | old |
| 64                                  | 4000  | 0.00219 | 0.00226 | 0.0002 | 2 | new |
| 16                                  | 12000 | 0.00238 | 0.00250 | 0.0001 | 2 | new |
| 4                                   | 24000 | 0.00265 | 0.00235 | 0.0005 | 2 | new |

(`enc` = before/after the concurrent encoder change; compare only within a group.)

k = 1 -> 4 -> 16 at a fixed 4000 steps, all on the same code:
**0.0084 -> 0.0044 -> 0.0025**, a clean monotone 3.3x from nothing but averaging
the target noise. k=64 (0.0022) and k=16 at 12000 steps (0.0024) sit on the newer
encoder, so their small edge over k=16/k=4 cannot be credited to k — the fair
reading is that **k saturates by 16**. Cost per optimizer step
(measured, batch 64, 6-token observations):

    k= 1: 56.6 ms     k= 4: 108.5 ms (1.9x)     k=16: 252 ms (4.5x)     k=64: 881 ms (15.6x)

i.e. pairs 2..k cost ~23% of the first one each — cheap, but on this problem *not*
free: with only 6 observation tokens the encoder is not the dominant cost (fit:
~43 ms fixed + ~13 ms per pair). Per unit of compute k=16 at 4000 steps (0.00253)
and k=4 at 12000 steps (0.00259) land in the same place, and both beat k=1 at
12000 steps (0.00362) at comparable cost. So the honest statement is: **k buys a
3x reduction in the number of optimizer steps needed for a given quality, and a
modest gain at equal wallclock** — and on problems whose observations are large
(the SDE cases carry hundreds of tokens, where the encoder really does dominate),
the same k is much closer to free.

Note the last row: with k=4, doubling the budget from 12000 to 24000 steps buys
**nothing** (0.00265 against 0.00259 at 12000 — and the 24000-step run had the
*better* encoder, so if anything this understates the flatness; its seed spread
also triples). Together with k=64 flattening at ~0.0022, this says the optimizer
is no longer what limits this model.

Stratifying the k time points (`t_strat`, one `t` per `[j/k,(j+1)/k)` bin instead
of k independent uniforms) is **neutral**: 0.00253 with vs 0.00258 without
(k=16, 4000 steps, 2 seeds, sd 0.0002-0.0004) — and those two runs straddle the
encoder change, so at best this is "no effect large enough to see".

## 4. Held-out verification and where the residue comes from

The tables above chose the configuration, so the final number is taken on a
**fresh 32-dataset reference set that took part in no decision** (seed 4242).
All eleven runs in this section were trained after the encoder change, with the
same `encoder.py` / `metrics.py` / problem definition; the only thing that varies
between them is what §5 says it is. Each trained model is then re-sampled with
several ODE solvers, to separate optimization error from discretization error:

| 12000 steps, 32 held-out datasets   | baseline (3 runs) | **recommended (8 runs)** |
|-------------------------------------|---------|---------|
| midpoint / 20 steps (default)       | 0.01141 | **0.00226 ± 0.00048** |
| midpoint / 40 steps                 | 0.01144 | 0.00226 |
| RK4 / 50 steps                      | 0.01141 | 0.00227 |
| Euler / 20 steps                    | 0.01119 | 0.00241 |
| re-draw with another sampler seed   | 0.01088 | 0.00225 |
| **Monte-Carlo floor of the metric** | 0.00062 | 0.00062 |

The 8 "recommended" runs are 2 seeds x 4 variants that differ only in how the
gradient is bounded (clip 1.0 / 0.5 / 10 / none-with-lr-halved). They are pooled
deliberately: their means (0.00195 / 0.00229 / 0.00248 / 0.00232) are ordered
*inside* the per-run spread (0.00165-0.00300), so treating any of them as better
than another would be chasing seed noise. Pooled: **0.00226 ± 0.00048**.

**5.0x better than the baseline at the same 12000 steps, and 3.6x above the
floor** (the task's own reference point: 0.0087 at 15000 steps -> 0.0023, a 3.8x
cut, measured on a stricter reference set).

Refining the ODE solver changes nothing (midpoint/20 = RK4/50 to within 0.5%), so
the residual 3.6x is **not** discretization. Neither is it budget (24000 steps is
no better than 12000) nor remaining gradient noise (k=64 flattens at the same
place). It is the velocity field itself — a model-capacity / model-form question
(`dim_model=64`, 3 blocks, and a *diagonal* Gaussian base that cannot seed the
strong posterior correlations this design matrix produces, so the flow has to
manufacture all of them). That is the next workstream, not this one.

## 5. What did not help — and one trap

- **gradient clipping**: nothing, at either end. Alone at 4000 steps 0.02010 vs
  0.01784 (inside noise); inside the final recipe, clip 1.0 / 0.5 / 10 / none all
  land in one band (see §4). But the *threshold* is a trap worth recording:
  measured gradient norms are median 1.5-3.1, p90 4.3-6.4, max 8.1 across
  `linear_gaussian` / `ou` / `cir` / `stoch_lv`, so the customary `clip=1.0`
  fires on **73-100% of steps** — that is not a safety net, it is an undeclared
  2-3x cut of the learning rate that would quietly eat any future `lr` change.
  The default is therefore 10.0: above every observed norm, still catches a
  genuine blow-up.
- **a larger learning rate** (1e-3), with or without the schedule.
- **stratified `t`** — neutral (kept, defaults to on; it cannot cost anything).
- **EMA with too long a horizon** (0.9995 at a 4000-step budget).
- **more budget**: 24000 steps is not better than 12000.

## 6. Recommended settings (now the defaults in `amortix/flow.py`)

```python
post.fit(n_train=20000, steps=12000, batch=64, lr=3e-4,
         schedule="cosine", warmup=300,     # decay to 0; the run settles
         ema_decay=0.999,                   # Polyak average; evaluated AND shipped
         k_pairs=4, t_strat=True,           # 4 CFM pairs per encoder pass
         grad_clip=10.0,                    # above the working norm: a net, not an lr cut
         betas=(0.9, 0.999), eps=1e-8)      # unchanged
```

`fit` now (a) always measures the final step, not the last multiple of
`monitor_every` — with a decaying schedule the last stretch is exactly where the
model settles; (b) writes the EMA weights into the module at the end, so the
model that was measured is the model that is shipped; (c) caps the warmup at
`total_steps // 10`, so a 300-step warmup cannot swallow a short budget.

Use `k_pairs=16` when you can afford 4.5x per step and want the same quality in
3x fewer steps; do not bother with 64. Turn EMA off (`ema_decay=None`) only to
reproduce the old behaviour.

**Reproduction.** `uv run python examples/train_monitored.py --steps 12000` now
trains with these defaults and prints the monitored curve; the "before" is the
same run with `fit(..., schedule="constant", warmup=0, ema_decay=None,
grad_clip=None, k_pairs=1)`. The reference sets used above are
`prob.prior.sample(n)` / `prob.observe(...)` under `torch.Generator().manual_seed(77)`
(12 monitoring datasets) and `manual_seed(4242)` (32 verification datasets), with
`exact_posterior(y[i], prob, n=400, seed=i)` as the truth and
`post.sample_batch(tok, n=400, seed=0)` as ours. Expect the baseline to sit at
0.010-0.018 and oscillate, and the defaults to descend monotonically to
0.0017-0.0030 per run (0.0023 on average over 8 runs) on the 32-dataset set.

## Summary

| | metric | vs floor |
|---|---|---|
| Monte-Carlo floor (32-dataset set)         | 0.00062 | 1.0x |
| **recommended, 12000 steps** (8 runs)      | **0.00226 ± 0.00048** | 3.6x |
| baseline, 12000 steps (3 runs)             | 0.01141 | 18.4x |
| baseline, 15000 steps (task's measurement) | 0.0087  | ~8x (its own set, floor 0.0011) |

The oscillation is gone (the curve is monotone and its last point is its best),
the distance to the exact posterior is **5x smaller at the same budget**, and
what remains between us and the estimator floor no longer responds to anything
the optimizer can do — it responds to the model.
