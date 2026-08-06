# ou — amortized CFM posterior vs MCMC gold standard

Per held-out dataset, the amortized posterior is compared with MCMC samples from the exact posterior conditioned on **the same data the network sees** (the observer's subsample of the path).

```
================================================================================================
OU: amortized CFM posterior vs MCMC gold standard
================================================================================================
params ['theta', 'mu', 'sigma'] | tokens 74 | path 500 steps @ dt=0.02
budget n_train=12000 epochs=40 | 12 held-out datasets | 2000 amortized draws vs 20000 MCMC draws

reference: exact OU likelihood, data=observed (73 of 501 path points), scheme=exact
MCMC quality: max split-Rhat 1.008 | min ESS 893 / 20000 | mean acceptance 0.25

   param | mean diff | |bias|/std | std ratio |  W1/std | std amort |  std MCMC
-------------------------------------------------------------------------------
   theta |     5.90% |       0.27 |     0.882 |   0.285 |    19.26% |    21.76%
      mu |     8.21% |       0.80 |     0.073 |   1.087 |     0.68% |    10.23%
   sigma |     2.09% |       0.31 |     1.004 |   0.352 |     6.11% |     6.34%
-------------------------------------------------------------------------------
     ALL |     5.40% |       0.46 |     0.653 |   0.575 |     8.68% |    12.78%
MC floor |     0.43% |       0.03 |     1.011 |   0.045 |  (two independent MCMC runs of the SAME posterior)

correlation-matrix difference (mean |d corr| over off-diagonals): 0.251   (MC floor 0.020)
per pair (mean over datasets)  -- a pair whose amortized posterior is degenerate makes its correlation meaningless:
    corr(theta,mu): amortized +0.038 | MCMC -0.030 | diff 0.068 | MC floor 0.002
    corr(theta,sigma): amortized +0.121 | MCMC +0.286 | diff 0.165 | MC floor 0.006
    corr(mu,sigma): amortized -0.418 | MCMC -0.004 | diff 0.414 | MC floor 0.011
worst-dataset W1/std per param: theta:1.04  mu:2.24  sigma:0.78
targets: mean diff -> 0, |bias|/std -> 0, std ratio -> 1.000, W1/std -> 0, corr diff -> 0;
the MC floor row is what those metrics read when BOTH sample sets are exact -- that is the resolution limit.
(std ratio > 1 = too wide/under-confident, < 1 = over-confident; mean diff and stds are % of prior range)

anchor -- RMSE of the posterior mean to the true parameter (% of prior range):
  amortized theta:26.65%  mu:1.91%  sigma:7.05%   (mean 11.87%)
  MCMC      theta:26.26%  mu:11.22%  sigma:6.33%   (mean 14.60%)

note -- OU initial condition: the simulator sets X_0 = mu exactly, so the data pins mu.
The conditional likelihood above deliberately ignores that (as the closed-form MLE does),
so the reference posterior for mu is artificially broad. Measured:
  amortized |E[mu] - X_0| = 1.81% of prior range, amortized std(mu) = 0.68%, MCMC std(mu) = 10.23%
  reference B (mu pinned at X_0 -- the exact posterior of the actual generative model):
     theta: mean diff 4.70%  std ratio 0.985  W1/std 0.235
     sigma: mean diff 2.13%  std ratio 1.007  W1/std 0.356

timing: train 1651s once | amortized 3161 ms/dataset (2000 draws) | MCMC 1701 ms/dataset (20000 draws)
(MCMC is cheap here precisely because the likelihood is closed-form -- these two cases exist to
 validate the machinery, not to be beaten on speed. The amortization pays off where no such likelihood exists.)
```
