"""Quickstart: recover GBM parameters and check against the exact posterior.

Trains a tiny design-amortized posterior for geometric Brownian motion
(a few minutes on a laptop CPU), then compares its samples for one observation
set against the exact conjugate posterior on the same points.

Run:  python examples/gallery/01_quickstart_gbm.py
"""
import torch

from amortix.evaluation import fid, model_of_size
from amortix.problems.design_basic import GBMDesign, gbm_exact_from_points

prob = GBMDesign()
post = model_of_size(prob, "tiny")
post.fit(n_train=3000, steps=1200, batch=256,
         retokenize=prob.make_retokenizer(), verbose=True)

# one fresh observation set: 20 points at random times
gen = torch.Generator().manual_seed(1)
m_true = prob.prior.sample(1, gen)
raw = prob.trajectories(m_true, gen)
tidx, cidx = prob.sample_design(gen, 20)
tokens = prob.tokens_for(raw[0], tidx, cidx, gen)

draws = post.sample(tokens, n=2000)                  # milliseconds
exact = gbm_exact_from_points(prob, raw[0, :, 0], tidx, n_samples=2000)

print(f"\ntrue parameters : {m_true[0].tolist()}")
print(f"posterior mean  : {draws.mean(0).tolist()}")
print(f"FID vs exact    : {fid(draws.numpy(), exact):.4f} "
      f"(estimator floor at n=2000 is ~0.004)")
