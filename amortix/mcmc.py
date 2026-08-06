"""Gold-standard MCMC reference posteriors for the tractable-likelihood cases.

Why this module exists
----------------------
SBC (`amortix.diagnostics`) checks *self-consistency averaged over the prior*: it
can say "this posterior is mis-calibrated" but never "for THIS dataset it is
shifted / too wide / correlated wrongly". For the two gallery cases whose
transition density is available in closed form -- Ornstein-Uhlenbeck and
Geometric Brownian Motion -- we can compute the *true* posterior for one
concrete dataset by MCMC and diff the amortized posterior against it, parameter
by parameter. That is the decisive reference.

What counts as "the data" (read this before interpreting any number)
--------------------------------------------------------------------
The network never sees the fine integration grid. `PathObserver` keeps only the
values on a union of channel index sets: for OU **73 of 501** path points, for
GBM **95 of 501**. So the posterior the amortized model is approximating is

    p(m | X_{i_0}, X_{i_1}, ..., X_{i_K}),   i_k = observed step indices

not p(m | whole fine path). Because the path is Markov, the subsampled
likelihood is again a product of transition densities -- now over the
*non-uniform* gaps dt_k = (i_{k+1} - i_k) * dt_sim. That is what
``data="observed"`` (the default of :func:`posterior_samples`) computes.

``data="full"`` conditions on the entire fine path. It is strictly sharper --
measured std(observed)/std(full) is 2.73 for OU sigma and 2.54 for GBM sigma
(drift parameters barely move: ~1.1) -- and is the right thing for the MLE
self-check below, but it is **not** a fair reference for the amortized
posterior: against it even a perfectly calibrated amortized posterior would look
2.7x over-dispersed on sigma.

Both likelihoods condition on the first observed point (they are conditional
likelihoods, exactly like the closed-form estimators in `amortix.baselines`).
OU is the canonical centred process dX = -theta X dt + sigma dW with X_0 drawn
from the stationary law N(0, sigma^2 / (2 theta)); the likelihood below is the
conditional one (given X_0), which is what the closed-form MLE also uses.
Use ``fixed=`` to build the reference posterior that does use it.

Discretization
--------------
Trajectories are produced by Euler-Maruyama, while `scheme="exact"` uses the
exact continuous-time OU transition. For OU the two differ at O(theta*dt)
(<= 6% in transition variance at the top of the prior box), so
`scheme="euler"` provides the transition density of the EM chain itself -- exact
for the process that actually generated the data. :func:`self_check` measures
how much the reference posterior moves between the two.

Everything here is pure numpy; the cost is dominated by O(n_obs) per proposal.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Sequence

import numpy as np

_LOG_2PI = math.log(2.0 * math.pi)

TRACTABLE_CASES = ("ou", "gbm")


# ---------------------------------------------------------------------------
# exact likelihoods
# ---------------------------------------------------------------------------
def _gaps(dt, n: int) -> np.ndarray:
    """Broadcast `dt` to a vector of n per-transition time gaps."""
    g = np.asarray(dt, dtype=np.float64)
    if g.ndim == 0:
        g = np.full(n, float(g))
    g = g.reshape(-1)
    if g.size != n:
        raise ValueError(f"dt must be a scalar or {n} per-transition gaps, got {g.size}")
    if np.any(g <= 0):
        raise ValueError("all time gaps must be > 0")
    return g


def _gauss_ll(y: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    """Sum of independent Gaussian log-densities."""
    if np.any(var <= 0) or not np.all(np.isfinite(var)):
        return -np.inf
    ll = -0.5 * (_LOG_2PI + np.log(var) + (y - mean) ** 2 / var)
    s = float(np.sum(ll))
    return s if np.isfinite(s) else -np.inf


def _as_series(path) -> np.ndarray:
    a = np.asarray(path, dtype=np.float64)
    return a.reshape(-1) if a.ndim == 1 else a.reshape(a.shape[0], -1)[:, 0]


def log_likelihood_ou(path, m, dt, scheme: str = "exact", dt_fine: float = None) -> float:
    """Exact log-likelihood of an OU path,  dX = -theta X dt + sigma dW (centred).

    The OU transition is Gaussian in closed form for *any* gap:

        X_{k+1} | X_k ~ N( X_k rho ,  s2 ),
        rho = exp(-theta dt),   s2 = sigma^2 / (2 theta) * (1 - rho^2)

    Args:
        path: observed values [n+1] (or [n+1, S]; component 0 is used).
        m:    (theta, sigma).
        dt:   scalar gap, or an array of n per-transition gaps (for a
              non-uniformly subsampled path).
        scheme: "exact" -> continuous-time OU transition (above).
                "euler" -> transition of the Euler-Maruyama chain that actually
                generated the data: over k fine steps of size `dt_fine`,
                rho = (1 - theta dt_fine)^k and
                s2 = sigma^2 dt_fine (1 - rho^2) / (1 - (1 - theta dt_fine)^2).
        dt_fine: the EM integration step (required for scheme="euler").

    Returns the conditional log-likelihood given the first point (a float;
    -inf outside the support theta > 0, sigma > 0).
    """
    x = _as_series(path)
    if x.size < 2:
        return 0.0
    p = np.asarray(m, dtype=np.float64).reshape(-1)
    if p.size != 2 or not np.all(np.isfinite(p)):
        return -np.inf
    theta, sigma = float(p[0]), float(p[1])
    if theta <= 0.0 or sigma <= 0.0:
        return -np.inf

    g = _gaps(dt, x.size - 1)
    if scheme == "exact":
        rho = np.exp(-theta * g)
        s2 = sigma ** 2 / (2.0 * theta) * (1.0 - rho ** 2)
    elif scheme == "euler":
        if dt_fine is None:
            raise ValueError("scheme='euler' needs dt_fine (the EM integration step)")
        a = 1.0 - theta * float(dt_fine)
        if abs(a) >= 1.0:                      # EM chain non-contracting -> no stationary law
            return -np.inf
        k = np.rint(g / float(dt_fine)).astype(np.int64)
        if np.any(k < 1):
            raise ValueError("gaps must be integer multiples of dt_fine for scheme='euler'")
        rho = a ** k
        s2 = sigma ** 2 * float(dt_fine) * (1.0 - rho ** 2) / (1.0 - a ** 2)
    else:
        raise ValueError(f"unknown scheme {scheme!r} (use 'exact' or 'euler')")

    mean = x[:-1] * rho
    return _gauss_ll(x[1:], mean, s2)


def log_likelihood_gbm(path, m, dt) -> float:
    """Exact log-likelihood of a GBM path,  dS = mu S dt + sigma S dW.

    Log-returns over a gap dt are independent Gaussians:

        r_k = log(S_{k+1} / S_k) ~ N( (mu - sigma^2/2) dt ,  sigma^2 dt )

    Args:
        path: observed prices [n+1] (or [n+1, S]; component 0 is used), all > 0.
        m:    (mu, sigma).
        dt:   scalar gap or an array of n per-transition gaps.
    """
    s = _as_series(path)
    if s.size < 2:
        return 0.0
    p = np.asarray(m, dtype=np.float64).reshape(-1)
    if p.size != 2 or not np.all(np.isfinite(p)):
        return -np.inf
    mu, sigma = float(p[0]), float(p[1])
    if sigma <= 0.0:
        return -np.inf
    if np.any(s <= 0.0):
        return -np.inf

    r = np.diff(np.log(s))
    g = _gaps(dt, r.size)
    mean = (mu - 0.5 * sigma ** 2) * g
    var = sigma ** 2 * g
    return _gauss_ll(r, mean, var)


# ---------------------------------------------------------------------------
# sampler
# ---------------------------------------------------------------------------
def metropolis(log_post: Callable[[np.ndarray], float], x0, n_samples: int,
               step: float = 0.05, prior_low=None, prior_high=None, seed: int = 0,
               burn_in: int = None, thin: int = 1, target_accept: float = 0.25,
               adapt_every: int = 50):
    """Adaptive random-walk Metropolis on a box-constrained space.

    Proposal: x' = x + step * span * N(0, I), with span = prior_high - prior_low,
    so a single scalar `step` means "this fraction of each prior range". Anything
    outside the box is rejected (the Gaussian proposal is symmetric, so rejecting
    out-of-box moves samples the box-truncated target exactly).

    Two things are adapted during burn-in, and both matter here:

    * `step`, in blocks of `adapt_every`, towards `target_accept` with a decaying
      Robbins-Monro gain;
    * the proposal *shape* (Haario-style adaptive Metropolis): once enough
      burn-in states are collected the isotropic-in-prior-units proposal is
      replaced by ``chol(Cov(trace))``, rescaled to the optimal 2.38/sqrt(d).
      Posterior scales in these problems differ by an order of magnitude between
      parameters (for OU sigma is ~20x sharper than theta in prior units) and are
      correlated, so an isotropic proposal is forced down to the narrowest
      direction and the wide ones barely move.

    Adaptation is frozen for the whole sampling phase, so the retained chain
    comes from a genuine time-homogeneous Markov kernel.

    Args:
        log_post: callable x[d] -> log posterior density (unnormalized). Any
            non-finite value is treated as -inf (an impossible point).
        x0: [d] starting point; must have finite log_post (a box-interior
            fallback start is searched for if it does not).
        n_samples: number of retained samples.
        step: initial proposal scale, as a fraction of the prior range.
        prior_low, prior_high: [d] box bounds (None = unbounded).
        burn_in: adaptation+warmup iterations (default max(500, n_samples // 2)).
        thin: keep every `thin`-th post-burn-in state.

    Returns:
        (samples [n_samples, d], acceptance_rate) -- the acceptance rate is
        measured on the *sampling* phase only.
    """
    x = np.array(x0, dtype=np.float64).reshape(-1)
    d = x.size
    low = np.full(d, -np.inf) if prior_low is None else np.asarray(prior_low, dtype=np.float64).reshape(-1)
    high = np.full(d, np.inf) if prior_high is None else np.asarray(prior_high, dtype=np.float64).reshape(-1)
    if low.size != d or high.size != d:
        raise ValueError("prior_low / prior_high must match the dimension of x0")
    width = high - low
    span = np.where(np.isfinite(width), width, 1.0)
    if np.any(span <= 0):
        raise ValueError("prior_high must be strictly greater than prior_low")

    rng = np.random.default_rng(seed)
    n_samples = int(n_samples)
    thin = max(1, int(thin))
    if burn_in is None:
        burn_in = max(500, n_samples // 2)
    burn_in = int(burn_in)

    def lp(v: np.ndarray) -> float:
        if np.any(v < low) or np.any(v > high):
            return -np.inf
        val = log_post(v)
        val = float(val) if val is not None else -np.inf
        return val if np.isfinite(val) else -np.inf

    cur = lp(x)
    if not np.isfinite(cur):                     # rescue a bad start
        if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
            raise ValueError("x0 has zero posterior density and the box is unbounded")
        for _ in range(1000):
            cand = low + rng.random(d) * width
            val = lp(cand)
            if np.isfinite(val):
                x, cur = cand, val
                break
        else:
            raise RuntimeError("could not find a starting point with finite log-posterior")

    samples = np.empty((n_samples, d), dtype=np.float64)
    kept = 0
    acc_block = 0
    n_block = 0
    block = 0
    acc_sample = 0
    total = burn_in + n_samples * thin
    step = float(step)

    # Haario-style shape adaptation: running moments of the burn-in trace,
    # collected after an initial quarter so the pre-convergence transient is
    # not baked into the covariance estimate.
    chol = None                                  # None -> isotropic in prior units
    collect_from = burn_in // 4
    shape_from = burn_in // 2
    s1 = np.zeros(d)
    s2 = np.zeros((d, d))
    n_mom = 0

    for it in range(total):
        eps = rng.standard_normal(d)
        prop = x + step * (span * eps if chol is None else chol @ eps)
        lpp = lp(prop)
        accepted = False
        if lpp > -np.inf and (lpp >= cur or math.log(rng.random()) < lpp - cur):
            x, cur, accepted = prop, lpp, True

        if it < burn_in:
            acc_block += accepted
            n_block += 1
            if it >= collect_from:
                s1 += x
                s2 += np.outer(x, x)
                n_mom += 1
            if n_block == adapt_every:
                block += 1
                rate = acc_block / n_block
                step *= math.exp((rate - target_accept) / math.sqrt(block))
                step = min(max(step, 1e-8), 10.0)
                acc_block, n_block = 0, 0
                if it >= shape_from and n_mom > max(20, 10 * d):
                    mean = s1 / n_mom
                    cov = (s2 - n_mom * np.outer(mean, mean)) / (n_mom - 1)
                    cov = 0.5 * (cov + cov.T)
                    jitter = 1e-10 * float(np.trace(cov)) / d + 1e-300
                    try:
                        c = np.linalg.cholesky(cov + jitter * np.eye(d))
                    except np.linalg.LinAlgError:
                        c = None
                    if c is not None and np.all(np.isfinite(c)):
                        if chol is None:         # first switch: reset to the optimal scale
                            step = 2.38 / math.sqrt(d)
                        chol = c
        else:
            acc_sample += accepted
            if (it - burn_in) % thin == 0:
                samples[kept] = x
                kept += 1

    acc_rate = acc_sample / max(1, n_samples * thin)
    return samples[:kept], float(acc_rate)


# ---------------------------------------------------------------------------
# convergence diagnostics (so "gold standard" is a measured claim)
# ---------------------------------------------------------------------------
def split_rhat(chains: np.ndarray) -> np.ndarray:
    """Split-Rhat per dimension. chains [C, N, d] -> [d]. Target: < 1.01."""
    c = np.asarray(chains, dtype=np.float64)
    C, N, d = c.shape
    h = N // 2
    if h < 2:
        return np.full(d, np.nan)
    s = np.concatenate([c[:, :h], c[:, h:2 * h]], axis=0)      # [2C, h, d]
    M = s.shape[0]
    means = s.mean(axis=1)                                     # [2C, d]
    varis = s.var(axis=1, ddof=1)                              # [2C, d]
    W = varis.mean(axis=0)
    B = h * means.var(axis=0, ddof=1) if M > 1 else np.zeros(d)
    var_hat = (h - 1) / h * W + B / h
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(np.where(W > 0, var_hat / W, np.nan))


def effective_sample_size(chains: np.ndarray) -> np.ndarray:
    """ESS per dimension via FFT autocorrelations + Geyer's initial positive
    sequence. chains [C, N, d] -> [d]."""
    c = np.asarray(chains, dtype=np.float64)
    C, N, d = c.shape
    out = np.empty(d)
    nfft = 1 << (2 * N - 1).bit_length()
    for j in range(d):
        acov = np.zeros(N)
        for i in range(C):
            y = c[i, :, j] - c[i, :, j].mean()
            f = np.fft.rfft(y, n=nfft)
            a = np.fft.irfft(f * np.conjugate(f), n=nfft)[:N]
            acov += a / N
        acov /= C
        if acov[0] <= 0:
            out[j] = float(C * N)
            continue
        rho = acov / acov[0]
        # Geyer: sum consecutive pairs while positive and monotone
        total, k = 0.0, 1
        prev = np.inf
        while k + 1 < N:
            pair = rho[k] + rho[k + 1]
            if pair <= 0:
                break
            pair = min(pair, prev)
            total += pair
            prev = pair
            k += 2
        tau = max(1.0, 1.0 + 2.0 * total)
        out[j] = C * N / tau
    return out


# ---------------------------------------------------------------------------
# problem-aware wrapper
# ---------------------------------------------------------------------------
def observed_indices(prob) -> np.ndarray:
    """Step indices of the path the observer actually exposes to the network.

    `PathObserver` keeps, per channel, the values at indices
    ``clip(arange(count + 1) * every, max=n_steps)``. Their union is the whole
    of the network's information about the trajectory (every token feature --
    x, dx, dx^2, t, log dt, component id -- is a function of those values).
    """
    obs = prob.observer
    idx: set = set()
    for ch in obs.channels:
        a = np.minimum(np.arange(ch.count + 1) * ch.every, obs.n_steps)
        idx.update(int(v) for v in a)
    return np.array(sorted(idx), dtype=np.int64)


def _resolve_fixed(prob, fixed) -> np.ndarray:
    """-> [d] array with NaN where a parameter is free."""
    d = prob.prior.dim
    out = np.full(d, np.nan)
    if fixed is None:
        return out
    if isinstance(fixed, dict):
        for k, v in fixed.items():
            if k not in prob.prior.names:
                raise ValueError(f"unknown parameter {k!r}; names are {prob.prior.names}")
            out[prob.prior.names.index(k)] = float(v)
        return out
    arr = np.asarray(fixed, dtype=np.float64).reshape(-1)
    if arr.size != d:
        raise ValueError(f"`fixed` must have {d} entries (NaN = free)")
    return arr


def posterior_samples(prob, traj, case: str, n_samples: int = 4000, seed: int = 0,
                      data: str = "observed", scheme: str = "exact",
                      n_chains: int = 4, burn_in: int = None, thin: int = 1,
                      step: float = 0.05, fixed=None, return_info: bool = False):
    """MCMC samples from the true posterior of one dataset.

    Picks the exact likelihood by `case`, adds the uniform (box) log-prior taken
    from ``prob.prior``, and runs :func:`metropolis` in `n_chains` chains started
    from dispersed points (chain 0 at the box centre, the rest uniform in the box).

    Args:
        prob: the Problem instance (supplies prior box and the observer).
        traj: the raw trajectory of ONE dataset, [n_steps+1] or [n_steps+1, S].
        case: "ou" or "gbm" (the cases with a tractable likelihood).
        n_samples: total retained samples across all chains.
        data: "observed" (default) conditions on exactly the points the observer
            exposes -- the matched reference for the amortized posterior;
            "full" conditions on the whole fine path (sharper; use for the MLE
            self-check, not for scoring the amortized posterior).
        scheme: OU only -- "exact" (continuous-time transition) or "euler"
            (transition of the Euler-Maruyama chain that generated the data).
        fixed: optionally pin parameters to known values -- a dict
            {name: value} or a [d] array with NaN for free parameters. Pinned
            coordinates are held constant and not sampled.
        return_info: also return a dict with acceptance rates, split-Rhat, ESS
            and the conditioning actually used.

    Returns:
        samples [n_samples, d] (in the original parameter units), or
        (samples, info) if `return_info`.
    """
    key = str(case).lower()
    if key not in TRACTABLE_CASES:
        raise ValueError(
            f"no tractable likelihood for case {case!r}. Exact MCMC reference posteriors "
            f"exist only for {list(TRACTABLE_CASES)} (closed-form Gaussian transition "
            f"densities). For the other gallery cases the likelihood of a partially "
            f"observed path is an intractable integral over the unobserved path -- "
            f"use simulation-based calibration (amortix.diagnostics.run_sbc) instead."
        )
    obs = prob.observer
    if getattr(obs, "n_paths", 1) != 1:
        raise ValueError(
            f"observer.n_paths = {obs.n_paths}: the network conditions on several "
            "replicate paths but `observe` returns only the last one, so the "
            "likelihood cannot be matched to the data. Build the problem with n_paths=1."
        )

    arr = np.asarray(traj, dtype=np.float64)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError("pass the trajectory of a single dataset, not a batch")
        arr = arr[0]
    x = _as_series(arr)
    dt_sim = float(obs.dt_sim)

    if data == "observed":
        idx = observed_indices(prob)
        values = x[idx]
        gaps = np.diff(idx).astype(np.float64) * dt_sim
    elif data == "full":
        values = x
        gaps = np.full(values.size - 1, dt_sim)
    else:
        raise ValueError(f"unknown data={data!r} (use 'observed' or 'full')")

    if key == "ou":
        def loglik(v):
            return log_likelihood_ou(values, v, gaps, scheme=scheme, dt_fine=dt_sim)
    else:
        if scheme != "exact":
            raise ValueError("scheme='euler' is implemented for OU only")

        def loglik(v):
            return log_likelihood_gbm(values, v, gaps)

    low = prob.prior.low.numpy().astype(np.float64)
    high = prob.prior.high.numpy().astype(np.float64)
    d = low.size
    pin = _resolve_fixed(prob, fixed)
    free = np.isnan(pin)
    if not free.any():
        raise ValueError("all parameters are pinned -- nothing to sample")

    full = np.where(free, 0.0, pin)

    def log_post(v_free):
        full[free] = v_free
        return loglik(full)                      # box prior is uniform -> constant

    n_chains = max(1, int(n_chains))
    per = int(math.ceil(n_samples / n_chains))
    if burn_in is None:
        # generous by default: the shape adaptation only switches on halfway
        # through burn-in, and these posteriors are far narrower than the prior
        # box the chains start from.
        burn_in = max(2000, per)
    rng = np.random.default_rng(seed)
    chains, accs = [], []
    for c in range(n_chains):
        start = 0.5 * (low + high) if c == 0 else low + rng.random(d) * (high - low)
        s, a = metropolis(log_post, start[free], per, step=step,
                          prior_low=low[free], prior_high=high[free],
                          seed=int(rng.integers(0, 2 ** 31 - 1)),
                          burn_in=burn_in, thin=thin)
        chains.append(s)
        accs.append(a)

    stacked = np.stack(chains, axis=0)                       # [C, per, d_free]
    flat = stacked.reshape(-1, stacked.shape[-1])[:n_samples]
    out = np.empty((flat.shape[0], d))
    out[:, free] = flat
    out[:, ~free] = pin[~free]

    if not return_info:
        return out
    rhat = np.full(d, np.nan)
    ess = np.full(d, np.nan)
    rhat[free] = split_rhat(stacked)
    ess[free] = effective_sample_size(stacked)
    info = dict(acceptance=float(np.mean(accs)), acceptance_all=[float(a) for a in accs],
                rhat=rhat, ess=ess, n_obs=int(values.size), n_chains=n_chains,
                data=data, scheme=scheme, fixed=pin,
                gap_range=(float(gaps.min()), float(gaps.max())))
    return out, info


# ---------------------------------------------------------------------------
# self-check: MCMC must reproduce the closed-form MLE on a long path
# ---------------------------------------------------------------------------
def simulate_ou_exact(theta, mu, sigma, x0, dt, n_steps, seed=0) -> np.ndarray:
    """Sample an OU path from its EXACT transition law (no discretization error)."""
    rng = np.random.default_rng(seed)
    rho = math.exp(-theta * dt)
    s = math.sqrt(sigma ** 2 / (2 * theta) * (1 - rho ** 2))
    x = np.empty(n_steps + 1)
    x[0] = x0
    eps = rng.standard_normal(n_steps)
    for i in range(n_steps):
        x[i + 1] = mu + (x[i] - mu) * rho + s * eps[i]
    return x


def simulate_gbm_exact(mu, sigma, s0, dt, n_steps, seed=0) -> np.ndarray:
    """Sample a GBM path from its exact lognormal transition law."""
    rng = np.random.default_rng(seed)
    r = (mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * rng.standard_normal(n_steps)
    return s0 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))


def self_check(verbose: bool = True, strict: bool = True) -> Dict:
    """Validate the sampler + likelihoods against the closed-form estimators.

    On a long path the posterior concentrates on the MLE, so the MCMC posterior
    mean must sit within roughly one posterior std of `amortix.baselines.ou_mle`
    (resp. the GBM closed-form MLE). Also reports split-Rhat, acceptance, and how
    far the OU reference posterior moves when the exact continuous transition is
    swapped for the Euler-Maruyama one on EM-generated data.
    """
    from .baselines import ou_mle

    out: Dict = {}
    lines = []

    def say(s=""):
        lines.append(s)
        if verbose:
            print(s)

    # --- OU -----------------------------------------------------------------
    theta, mu, sigma, dt, n = 1.2, 0.2, 0.8, 0.02, 20000
    low = np.array([0.3, -1.0, 0.2])
    high = np.array([3.0, 1.0, 1.5])
    path = simulate_ou_exact(theta, mu, sigma, mu, dt, n, seed=7)
    mle = ou_mle(path, dt)
    mle_v = np.array([mle["theta"], mle["mu"], mle["sigma"]])

    chains = []
    for c in range(4):
        s, acc = metropolis(lambda v: log_likelihood_ou(path, v, dt),
                            0.5 * (low + high), 5000, step=0.02,
                            prior_low=low, prior_high=high, seed=100 + c, burn_in=3000)
        chains.append(s)
    st = np.stack(chains)
    smp = st.reshape(-1, 3)
    mean, std = smp.mean(0), smp.std(0)
    rh = split_rhat(st)
    z = np.abs(mean - mle_v) / std

    say(f"OU self-check: exact-law path, n={n} steps, dt={dt}, "
        f"truth theta={theta} mu={mu} sigma={sigma}")
    say(f"{'param':>7} | {'MLE':>9} | {'MCMC mean':>9} | {'MCMC std':>9} | "
        f"{'|diff|/std':>10} | {'Rhat':>6}")
    for j, nm in enumerate(["theta", "mu", "sigma"]):
        say(f"{nm:>7} | {mle_v[j]:9.4f} | {mean[j]:9.4f} | {std[j]:9.4f} | "
            f"{z[j]:10.2f} | {rh[j]:6.3f}")
    say(f"acceptance {acc:.2f} | sigma rel-std {std[2]/mean[2]*100:.2f}% "
        f"(Fisher prediction 1/sqrt(2n) = {100/math.sqrt(2*n):.2f}%)")
    out["ou"] = dict(mle=mle_v.tolist(), mean=mean.tolist(), std=std.tolist(),
                     z=z.tolist(), rhat=rh.tolist(), acceptance=float(acc))

    # --- GBM ----------------------------------------------------------------
    gmu, gsig, gdt, gn = 0.15, 0.35, 0.01, 20000
    glow, ghigh = np.array([-0.20, 0.10]), np.array([0.40, 0.60])
    gpath = simulate_gbm_exact(gmu, gsig, 1.0, gdt, gn, seed=11)
    r = np.diff(np.log(gpath))
    s2 = r.var(ddof=1) / gdt
    gmle = np.array([r.mean() / gdt + s2 / 2, math.sqrt(s2)])

    gchains = []
    for c in range(4):
        s, gacc = metropolis(lambda v: log_likelihood_gbm(gpath, v, gdt),
                             0.5 * (glow + ghigh), 5000, step=0.05,
                             prior_low=glow, prior_high=ghigh, seed=200 + c, burn_in=3000)
        gchains.append(s)
    gst = np.stack(gchains)
    gsmp = gst.reshape(-1, 2)
    gmean, gstd = gsmp.mean(0), gsmp.std(0)
    grh = split_rhat(gst)
    gz = np.abs(gmean - gmle) / gstd

    say()
    say(f"GBM self-check: exact-law path, n={gn} steps, dt={gdt}, truth mu={gmu} sigma={gsig}")
    say(f"{'param':>7} | {'MLE':>9} | {'MCMC mean':>9} | {'MCMC std':>9} | "
        f"{'|diff|/std':>10} | {'Rhat':>6}")
    for j, nm in enumerate(["mu", "sigma"]):
        say(f"{nm:>7} | {gmle[j]:9.4f} | {gmean[j]:9.4f} | {gstd[j]:9.4f} | "
            f"{gz[j]:10.2f} | {grh[j]:6.3f}")
    say(f"acceptance {gacc:.2f}")
    out["gbm"] = dict(mle=gmle.tolist(), mean=gmean.tolist(), std=gstd.tolist(),
                      z=gz.tolist(), rhat=grh.tolist(), acceptance=float(gacc))

    # --- discretization sensitivity on the real (Euler-Maruyama) simulator ---
    import torch
    from .problems import ou as ou_mod
    prob = ou_mod.make()
    g = torch.Generator().manual_seed(3)
    m_true = prob.prior.sample(1, generator=g)
    _, traj = prob.observe(m_true, generator=g)
    ex = posterior_samples(prob, traj[0], "ou", n_samples=8000, seed=5, scheme="exact")
    eu = posterior_samples(prob, traj[0], "ou", n_samples=8000, seed=5, scheme="euler")
    rngp = (high - low)
    shift = np.abs(ex.mean(0) - eu.mean(0)) / rngp * 100
    ratio = ex.std(0) / eu.std(0)
    say()
    say("discretization sensitivity of the REFERENCE itself (EM-simulated OU path, "
        "observed subsample):")
    say(f"{'param':>7} | {'mean shift exact vs euler':>26} | {'std ratio':>9} | "
        f"{'shift / post.std':>16}")
    for j, nm in enumerate(["theta", "mu", "sigma"]):
        say(f"{nm:>7} | {shift[j]:25.3f}% | {ratio[j]:9.3f} | "
            f"{abs(ex.mean(0)[j]-eu.mean(0)[j])/eu.std(0)[j]:16.3f}")
    out["scheme_sensitivity"] = dict(mean_shift_pct=shift.tolist(), std_ratio=ratio.tolist())

    ok = bool(np.all(z < 1.0) and np.all(gz < 1.0)
              and np.all(rh < 1.02) and np.all(grh < 1.02))
    out["passed"] = ok
    say()
    say(f"self-check {'PASSED' if ok else 'FAILED'} "
        f"(criterion: |MCMC mean - MLE| < 1 posterior std, split-Rhat < 1.02)")
    out["report"] = "\n".join(lines)
    if strict and not ok:
        raise AssertionError("mcmc self-check failed:\n" + out["report"])
    return out


if __name__ == "__main__":       # uv run python -m amortix.mcmc
    self_check()
