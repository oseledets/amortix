"""The evaluation instrument: one FID, one evaluation-set format, one entry point.

Everything the report measures against a reference posterior goes through this
module, so that numbers computed by different scripts, at different times, and
on different machines stay comparable and re-scorable.

Three rules are enforced here:

  * one implementation of the discrepancy (:func:`fid`), never copied;
  * an evaluation set is an artifact, not code -- built once, validated on
    the spot, written to disk with the provenance needed to trust it later
    (:class:`EvalSet`), and loaded by everything downstream;
  * an evaluation set carries its own floor. Two independent reference draws
    are stored, so every reported number can be quoted against the resolution
    of the instrument that produced it. :func:`build_eval_set` refuses to
    save a set whose two chains disagree by more than ``max_discrepancy``
    posterior standard deviations -- such a set would measure its own sampler
    rather than the model.

Typical use::

    es = build_eval_set(problem, "merton", K=20, n_sets=32,
                        path="merton_K20.npz")
    res = evaluate(posterior, es)           # seconds, no retraining
    print(res["fid_median"], res["null_median"])
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.linalg import sqrtm


# --------------------------------------------------------------- discrepancy
def fid(a: np.ndarray, r: np.ndarray, scale: np.ndarray = None) -> float:
    """Squared Frechet distance between Gaussian fits of two sample sets.

    ``a`` and ``r`` are ``[n, d]`` arrays; ``r`` is the reference. Coordinates
    are divided by ``scale`` (default: the reference's per-parameter standard
    deviation), which makes the number dimensionless and comparable across
    parameters with different physical units. Pass ``scale=prior_range`` to
    measure absolute error instead -- the two normalizations answer different
    questions and the report quotes both.

    Zero for a perfect match. Note the estimator is positive even for two
    draws from the same distribution, at order d/n; :func:`evaluate` reports
    that floor alongside the value.
    """
    a = np.asarray(a, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    s = (r.std(0) if scale is None else np.asarray(scale, dtype=np.float64))
    s = np.where(s > 1e-12, s, 1.0)
    a, r = a / s, r / s
    ca, cr = np.cov(a.T), np.cov(r.T)
    cm = sqrtm(cr @ ca)
    if np.iscomplexobj(cm):
        cm = cm.real
    return float(((a.mean(0) - r.mean(0)) ** 2).sum()
                 + np.trace(ca + cr - 2 * cm))


# ------------------------------------------------------------- model loading
#: The named model sizes used throughout the report. Width x transformer blocks.
SIZES = {"pico": (8, 2), "nano": (16, 2), "tiny": (32, 2),
         "small": (64, 3), "big": (128, 4)}


def model_of_size(problem, size: str = "small"):
    """A posterior of one of the named sizes (``SIZES``)."""
    from . import FlowPosterior
    dim, depth = SIZES[size]
    return FlowPosterior(problem, dim_model=dim, depth=depth)


def load_posterior(problem, path: str, device=None):
    """Load a checkpoint without being told its size, onto the fast device.

    The width and depth are read off the state dict, so a checkpoint can never
    be loaded into a differently shaped model -- a mistake that is otherwise
    silent right up to a wall of shape errors, and that costs a training run
    when the checkpoint is the only copy.

    ``device`` defaults to CUDA when it is available. That default is not
    cosmetic: ``sample_batch`` takes its device from the model's parameters, so
    a checkpoint left on the CPU is scored on the CPU, and the scoring is the
    expensive half of a sweep -- on the largest model a full sweep over the
    design sizes takes minutes on the GPU against hours on the CPU. A
    CPU-resident model is silent about this; it simply runs.
    """
    from . import FlowPosterior
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    sd = torch.load(path, map_location="cpu")
    dim = sd["encoder.embed.proj.weight"].shape[0]
    depth = 1 + max(int(k.split(".")[2]) for k in sd
                    if k.startswith("velocity.blocks."))
    post = FlowPosterior(problem, dim_model=dim, depth=depth)
    post.load_state_dict(sd)
    post.eval()
    return post.to(device)


# ----------------------------------------------------------- evaluation set
@dataclass
class EvalSet:
    """A frozen evaluation set: observation instances with validated references.

    Fields are plain arrays so the whole thing round-trips through one
    ``.npz``: ``tokens``/``mask`` are what the model is conditioned on,
    ``chain_a`` is the reference, ``chain_b`` an independent second draw used
    only to report the floor, and ``meta`` records how it was made.
    """

    tokens: np.ndarray
    mask: np.ndarray
    chain_a: np.ndarray
    chain_b: np.ndarray
    m_true: np.ndarray
    meta: dict = field(default_factory=dict)

    # --- persistence ---
    def save(self, path: str) -> "EvalSet":
        np.savez(path, tokens=self.tokens, mask=self.mask, chain_a=self.chain_a,
                 chain_b=self.chain_b, m_true=self.m_true,
                 meta=json.dumps(self.meta))
        return self

    @classmethod
    def load(cls, path: str) -> "EvalSet":
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["meta"])) if "meta" in d else {}
        return cls(d["tokens"], d["mask"], d["chain_a"], d["chain_b"],
                   d["m_true"], meta)

    # --- properties the report quotes ---
    @property
    def discrepancy(self) -> np.ndarray:
        """Per-set, per-parameter |mean_a - mean_b| in posterior-sd units."""
        sd = 0.5 * (self.chain_a.std(1) + self.chain_b.std(1))
        return np.abs(self.chain_a.mean(1) - self.chain_b.mean(1)) / np.maximum(sd, 1e-12)

    @property
    def floor(self) -> float:
        """Median FID between the two independent reference draws: the
        smallest difference this evaluation set can resolve."""
        return float(np.median([fid(self.chain_b[i], self.chain_a[i])
                                for i in range(len(self.chain_a))]))

    def __repr__(self) -> str:
        m = self.meta
        return (f"EvalSet({m.get('system','?')}, K={m.get('K','?')}, "
                f"{len(self.chain_a)} sets, chain={m.get('n_chain','?')}, "
                f"floor={self.floor:.4f}, "
                f"max inter-chain {self.discrepancy.max():.3f} sd)")


# Legacy alias (class). The paper's term is "evaluation set"; scripts written
# against the old name keep running until updated, then this goes.
Battery = EvalSet


def provenance() -> dict:
    """What produced a number: package version, source fingerprint, git state.

    Recorded in every result, so that any reported number can be traced to the
    exact source state that produced it -- version drift between hosts is
    otherwise invisible in the outputs.
    """
    import hashlib
    import subprocess
    import amortix

    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for fn in sorted(os.listdir(here)):
        if fn.endswith(".py"):
            with open(os.path.join(here, fn), "rb") as f:
                h.update(f.read())
    try:
        rev = subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "-C", here, "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5)
        git = (rev.stdout.strip() or "?") + ("+dirty" if dirty.stdout.strip() else "")
    except Exception:
        git = "?"
    return dict(version=getattr(amortix, "__version__", "?"),
                source_sha256=h.hexdigest()[:12], git=git)


def floor_at(battery: EvalSet, n_draw: int) -> float:
    """The evaluation set's resolution at a given number of model draws.

    ``EvalSet.floor`` quotes the resolution at the reference chain length; a
    monitor that draws fewer samples is substantially coarser. Choosing the
    best checkpoint on a monitor whose floor sits above the model's own error
    means choosing the deepest noise dip rather than a better model, so check
    the floor at the sample size actually used for monitoring, not at the
    set's nominal one.
    """
    nb = battery.chain_b.shape[1]
    idx = (np.linspace(0, nb - 1, min(n_draw, nb)).astype(int)
           if n_draw < nb else slice(None))
    return float(np.median([fid(battery.chain_b[i][idx], battery.chain_a[i])
                            for i in range(len(battery.chain_a))]))


def evaluate(post, battery: EvalSet, n_draw: int = 4000, seed: int = 0,
             prior_range: np.ndarray = None) -> dict:
    """Score a trained posterior against an evaluation set.

    ``n_draw`` is the number of samples drawn from the model; the reference
    side is whatever the evaluation set stores (4,000 draws by default). Both
    sample sizes enter the estimator's noise, so the floor is computed with the
    same left-hand sample size as the measurement.

    Returns the median and mean FID over the set's observation instances, the
    set's own floor, their ratio (below ~2 the comparison is not resolved),
    the per-set values, and -- when ``prior_range`` is given -- the absolute
    posterior-mean error in prior-range units, which does not depend on how
    narrow the target posterior is.
    """
    tokens = torch.as_tensor(battery.tokens)
    mask = torch.as_tensor(battery.mask) if battery.mask is not None else None
    kw = {} if mask is None else {"mask": mask}
    smp = post.sample_batch(tokens, n=n_draw, seed=seed, **kw).numpy()
    a = battery.chain_a
    vals = np.array([fid(smp[i], a[i]) for i in range(len(a))])
    # The floor must be measured in the SAME estimator configuration as the
    # value: n draws on the left, the full reference on the right. Comparing an
    # n-vs-n floor against an n-vs-N measurement understates the offset,
    # because the estimator's noise has two terms (one per sample set) and only
    # the model's shrinks with n.
    nb = battery.chain_b.shape[1]
    idx = (np.linspace(0, nb - 1, min(n_draw, nb)).astype(int)
           if n_draw < nb else slice(None))
    nulls = np.array([fid(battery.chain_b[i][idx], a[i]) for i in range(len(a))])
    out = dict(fid_median=float(np.median(vals)), fid_mean=float(vals.mean()),
               null_median=float(np.median(nulls)),
               ratio=float(np.median(vals) / max(np.median(nulls), 1e-12)),
               n_sets=len(a), n_draw=n_draw, vals=[float(v) for v in vals],
               battery=battery.meta, provenance=provenance())
    if prior_range is not None:
        pr = np.asarray(prior_range, dtype=np.float64)
        err = np.array([np.abs(smp[i].mean(0) - a[i].mean(0)) / pr
                        for i in range(len(a))])
        out["abs_error_prior_units"] = np.median(err, 0).tolist()
    return out


def report(res: dict) -> str:
    """One line, with the floor attached -- a number without its resolution is
    not a measurement."""
    warn = "" if res["ratio"] >= 2 else "   [AT THE RESOLUTION LIMIT]"
    return (f"FID {res['fid_median']:.4f} (mean {res['fid_mean']:.4f}), "
            f"battery floor {res['null_median']:.4f}, "
            f"ratio {res['ratio']:.1f}, {res['n_sets']} sets{warn}")


# ------------------------------------------------- reference draws, one place
def _logpost(problem, name, raw_i, tidx, cidx, tokens_i):
    """The per-system likelihood, assembled once. Adding a system means adding
    a branch here, not a new script."""
    from .problems.design_zoo import (merton_logpost_factory,
                                      pk_logpost_factory, kpp_logpost_factory,
                                      fhn_logpost_factory)
    from .problems.design_basic import ou_logpost_factory, cir_logpost_factory

    y = tokens_i[:, 1].numpy().astype(np.float64)
    t_obs = tidx.numpy() * problem.observer.dt_sim
    lo = problem.prior.low.numpy().astype(np.float64)
    hi = problem.prior.high.numpy().astype(np.float64)
    if name == "ou_rd":
        return ou_logpost_factory(problem, raw_i[:, 0].numpy(), tidx.numpy()), lo, hi
    if name == "merton":
        vals = np.concatenate([[1.0], y])
        tt = np.concatenate([[0.0], t_obs])
        r = np.diff(np.log(np.maximum(vals, 1e-9)))
        return merton_logpost_factory(r, np.diff(tt), lo, hi), lo, hi
    if name == "pk":
        return pk_logpost_factory(t_obs, y, dose=problem.DOSE,
                                  logsd=problem.LOGSD), lo, hi
    if name == "kpp":
        return kpp_logpost_factory(problem, tidx, cidx, y), lo, hi
    if name == "fhn":
        return fhn_logpost_factory(problem, tidx.numpy(), y), lo, hi
    if name == "lv":
        from .problems.design_basic import lv_logpost_factory
        return lv_logpost_factory(problem, raw_i.numpy(), tidx.numpy(),
                                  cidx.numpy()), lo, hi
    if name == "heston":
        # amortix.hestonlik, not a continuous-time transition: the simulator
        # integrates the variance by clamped Euler, and past the Feller line
        # the two processes are not the same model (see that module).
        from .hestonlik import heston_logpost_factory
        return heston_logpost_factory(problem, tidx.numpy(), y), lo, hi
    if name == "hodgkin_huxley":
        from .problems.design_zoo import hh_logpost_factory
        return hh_logpost_factory(problem, tidx.numpy(), y), lo, hi
    if name == "seir":
        from .problems.design_zoo import seir_logpost_factory
        return seir_logpost_factory(problem, tidx.numpy(), cidx.numpy(), y), lo, hi
    if name == "poly":
        from .problems.design_basic import poly_logpost_factory
        return poly_logpost_factory(problem, raw_i[:, 0].numpy(),
                                    tidx.numpy()), lo, hi
    if name == "dw":
        from .problems.design_basic import dw_logpost_factory
        return dw_logpost_factory(problem, raw_i[:, 0].numpy(),
                                  tidx.numpy()), lo, hi
    if name == "cir":
        return cir_logpost_factory(problem, raw_i[:, 0].numpy(),
                                   tidx.numpy()), lo, hi
    raise KeyError(f"no reference likelihood registered for {name!r}")


def reference_draw(problem, name, raw_i, tidx, cidx, tokens_i, n_chain, seed,
                   keep=4000):
    """One reference sample set. Exact where a closed form exists, otherwise a
    thinned adaptive-Metropolis chain of ``n_chain`` draws."""
    from .mcmc import metropolis
    from .problems.design_basic import gbm_exact_from_points

    if name == "gbm_rd":
        return gbm_exact_from_points(problem, raw_i[:, 0].numpy(),
                                     tidx.numpy(), n_samples=keep, seed=seed)
    lp, lo, hi = _logpost(problem, name, raw_i, tidx, cidx, tokens_i)
    c, _ = metropolis(lp, 0.5 * (lo + hi), n_samples=n_chain,
                      prior_low=lo, prior_high=hi, seed=seed)
    c = np.asarray(c)
    return c[np.linspace(0, len(c) - 1, keep).astype(int)]


def problem_for(name: str):
    """Reconstruct a benchmark problem from its registry name. Needed so that
    reference draws can run in worker processes, which cannot inherit a live
    problem object."""
    from .problems.design_zoo import DESIGN_ZOO
    from .problems.design_basic import GBMDesign, OUDesign, CIRDesign
    from .problems.design_basic import (DoubleWellDesign, PolyDriftDesign,
                                        LotkaVolterraDesign, LinGaussDesign)
    cls = ({"gbm_rd": GBMDesign, "ou_rd": OUDesign, "cir": CIRDesign,
            "dw": DoubleWellDesign, "poly": PolyDriftDesign,
            "lv": LotkaVolterraDesign,
            "lingauss": LinGaussDesign}.get(name)
           or DESIGN_ZOO[name])
    return cls()


def _draw_worker(args):
    """Top-level so it pickles: one reference draw in its own process."""
    import torch as _t
    name, raw_i, tidx, cidx, tok, n_chain, seed = args
    _t.set_num_threads(1)
    return reference_draw(problem_for(name), name, _t.as_tensor(raw_i),
                          _t.as_tensor(tidx), _t.as_tensor(cidx),
                          _t.as_tensor(tok), n_chain, seed)


def build_eval_set(problem, name, K, n_sets=32, n_chain=200000, seed=4243,
                  path=None, max_discrepancy=0.25, workers=None):
    """Build, validate, and (optionally) save an evaluation set.

    Raises if the two independent reference draws disagree by more than
    ``max_discrepancy`` posterior standard deviations on any instance: such an
    evaluation set measures its own sampler rather than the model, and saving
    it would let that number reach a table.
    """
    from concurrent.futures import ProcessPoolExecutor

    gen = torch.Generator().manual_seed(seed)
    m_true = problem.prior.sample(n_sets, gen)
    raw = problem.trajectories(m_true, generator=gen)
    kmax = problem.observer.k_max
    tokens = torch.zeros(n_sets, kmax, 6)
    mask = torch.zeros(n_sets, kmax, dtype=torch.bool)
    designs = []
    for i in range(n_sets):
        tidx, cidx = problem.sample_design(gen, K)
        if problem.observer.n_channels == 1:
            tidx = torch.unique(tidx)
            cidx = torch.zeros_like(tidx)
        # Multi-channel systems keep the (time, channel) pairs their design law
        # produced: collapsing everything to channel 0 can build sets that
        # never observe one of the channels the network trained on, and the
        # evaluation would then measure a design the model never sees.
        tk = problem.tokens_for(raw[i], tidx, cidx, gen)
        tokens[i, :tk.shape[0]] = tk
        mask[i, :tk.shape[0]] = True
        designs.append((tidx, cidx, tk))

    def job(i, s):
        tidx, cidx, tk = designs[i]
        return (name, raw[i].numpy(), tidx.numpy(), cidx.numpy(), tk.numpy(),
                n_chain, s)

    jobs_a = [job(i, 100 + i) for i in range(n_sets)]
    jobs_b = [job(i, 77000 + i) for i in range(n_sets)]
    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            A = np.stack(list(ex.map(_draw_worker, jobs_a)))
            B = np.stack(list(ex.map(_draw_worker, jobs_b)))
    else:
        A = np.stack([_draw_worker(j) for j in jobs_a])
        B = np.stack([_draw_worker(j) for j in jobs_b])

    bat = Battery(tokens.numpy(), mask.numpy(), A, B, m_true.numpy(),
                  dict(system=name, K=K, n_sets=n_sets, n_chain=n_chain,
                       seed=seed))
    worst = float(bat.discrepancy.max())
    if worst > max_discrepancy:
        raise RuntimeError(
            f"battery rejected: two independent references disagree by "
            f"{worst:.2f} posterior sd (limit {max_discrepancy}). Lengthen the "
            f"chains or report this system under the calibration screen.")
    bat.meta["max_inter_chain_sd"] = round(worst, 4)
    bat.meta["floor"] = round(bat.floor, 5)
    if path:
        bat.save(path)
    return bat


# Legacy alias (builder) -- must follow the definition it points at.
build_battery = build_eval_set
