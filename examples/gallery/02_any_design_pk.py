"""Design amortization: one network answers any number of blood draws.

Pharmacokinetics (Bateman model): train once, then query the SAME network at
6, 20, and 50 irregular sampling times. The posterior tightens as the design
densifies -- no retraining, no fixed grid. A few minutes on GPU, longer on CPU.

Run:  python examples/gallery/02_any_design_pk.py
"""
import torch

from amortix.evaluation import model_of_size
from amortix.problems.design_zoo import DESIGN_ZOO

prob = DESIGN_ZOO["pk"]()
post = model_of_size(prob, "small")
post.fit(n_train=20000, steps=6000, batch=256,
         retokenize=prob.make_retokenizer(), verbose=True)

gen = torch.Generator().manual_seed(7)
m_true = prob.prior.sample(1, gen)
raw = prob.trajectories(m_true, gen)
print(f"\ntrue (ka, ke, V): {m_true[0].tolist()}")
for K in (6, 20, 50):
    tidx, cidx = prob.sample_design(gen, K)
    tokens = prob.tokens_for(raw[0], tidx, cidx, gen)
    d = post.sample(tokens, n=2000)
    lo, hi = d.quantile(0.05, 0), d.quantile(0.95, 0)
    inside = bool(((m_true[0] >= lo) & (m_true[0] <= hi)).all())
    print(f"K={K:>3}: posterior sd {d.std(0).tolist()}"
          f"   truth in 90% intervals: {inside}")
