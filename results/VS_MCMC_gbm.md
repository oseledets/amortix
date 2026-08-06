# gbm — amortized CFM posterior vs MCMC gold standard

Per held-out dataset, the amortized posterior is compared with MCMC samples from the exact posterior conditioned on **the same data the network sees** (the observer's subsample of the path).

```
================================================================================================
GBM: amortized CFM posterior vs MCMC gold standard
================================================================================================
params ['mu', 'sigma'] | tokens 100 | path 500 steps @ dt=0.01
budget n_train=12000 epochs=40 | 12 held-out datasets | 2000 amortized draws vs 20000 MCMC draws

reference: exact GBM likelihood, data=observed (95 of 501 path points), scheme=exact
MCMC quality: max split-Rhat 1.004 | min ESS 1626 / 20000 | mean acceptance 0.26

   param | mean diff | |bias|/std | std ratio |  W1/std | std amort |  std MCMC
-------------------------------------------------------------------------------
      mu |     2.16% |       0.11 |     1.082 |   0.141 |    23.40% |    21.66%
   sigma |     1.10% |       0.18 |     1.033 |   0.219 |     5.84% |     5.64%
-------------------------------------------------------------------------------
     ALL |     1.63% |       0.14 |     1.058 |   0.180 |    14.62% |    13.65%
MC floor |     0.31% |       0.02 |     1.001 |   0.035 |  (two independent MCMC runs of the SAME posterior)

correlation-matrix difference (mean |d corr| over off-diagonals): 0.057   (MC floor 0.031)
per pair (mean over datasets)  -- a pair whose amortized posterior is degenerate makes its correlation meaningless:
    corr(mu,sigma): amortized +0.041 | MCMC +0.059 | diff 0.018 | MC floor 0.012
worst-dataset W1/std per param: mu:0.33  sigma:0.55
targets: mean diff -> 0, |bias|/std -> 0, std ratio -> 1.000, W1/std -> 0, corr diff -> 0;
the MC floor row is what those metrics read when BOTH sample sets are exact -- that is the resolution limit.
(std ratio > 1 = too wide/under-confident, < 1 = over-confident; mean diff and stds are % of prior range)

anchor -- RMSE of the posterior mean to the true parameter (% of prior range):
  amortized mu:26.82%  sigma:6.28%   (mean 16.55%)
  MCMC      mu:26.62%  sigma:5.70%   (mean 16.16%)

timing: train 1891s once | amortized 798 ms/dataset (2000 draws) | MCMC 873 ms/dataset (20000 draws)
(MCMC is cheap here precisely because the likelihood is closed-form -- these two cases exist to
 validate the machinery, not to be beaten on speed. The amortization pays off where no such likelihood exists.)
```
