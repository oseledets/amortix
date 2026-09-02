"""Smoke tests: every gallery case must build, simulate, train, sample, and have
a working classical baseline. Fast by design (tiny budgets) -- this verifies the
install and the Problem contract, not statistical quality.

    uv run pytest -q
"""
from __future__ import annotations

import importlib
import math

import numpy as np
import pytest
import torch

from amortix import FlowPosterior
from amortix.problems import GALLERY


@pytest.fixture(scope="module")
def gen():
    return torch.Generator().manual_seed(0)


def test_gallery_is_populated():
    assert len(GALLERY) == 9


@pytest.mark.parametrize("name", GALLERY)
def test_case_contract(name, gen):
    """make() / simulate / observe / sota must satisfy the documented contract."""
    mod = importlib.import_module(f"amortix.problems.{name}")
    prob = mod.make()
    d = prob.prior.dim

    assert isinstance(mod.SOTA_NAME, str) and mod.SOTA_NAME
    assert len(prob.prior.names) == d

    m, tokens = prob.simulate(8, generator=gen)
    assert m.shape == (8, d)
    assert tokens.shape[0] == 8 and tokens.shape[2] == prob.observer.N_FEATURES
    assert torch.isfinite(tokens).all(), "simulator produced non-finite tokens"

    tk, traj = prob.observe(prob.prior.sample(4, generator=gen), generator=gen)
    assert torch.isfinite(traj).all(), "simulator produced non-finite trajectory"

    est = np.asarray(mod.sota(tk[0].numpy(), traj[0].numpy(), prob))
    assert est.shape == (d,), f"sota must return one value per parameter"
    assert np.all(np.isfinite(est)), "sota returned non-finite values"


@pytest.mark.parametrize("name", GALLERY)
def test_train_and_sample(name):
    """A tiny fit must run and produce in-prior posterior samples."""
    mod = importlib.import_module(f"amortix.problems.{name}")
    prob = mod.make()
    post = FlowPosterior(prob).fit(n_train=128, epochs=1, verbose=False)
    _, tokens = prob.simulate(2)
    s = post.sample(tokens[0], n=32, n_steps=10)
    assert s.shape == (32, prob.prior.dim)
    assert torch.isfinite(s).all()
    # probit denormalization must keep every sample inside the prior box
    assert (s >= prob.prior.low - 1e-4).all() and (s <= prob.prior.high + 1e-4).all()


@pytest.mark.parametrize("conditioning", ["concat", "xattn"])
def test_both_conditionings_run(conditioning):
    from amortix import OrnsteinUhlenbeck
    prob = OrnsteinUhlenbeck()
    post = FlowPosterior(prob, conditioning=conditioning)
    assert post.conditioning == conditioning
    post.fit(n_train=128, epochs=1, verbose=False)
    s = post.sample(prob.simulate(2)[1][0], n=16, n_steps=10)
    assert s.shape == (16, prob.prior.dim)


def test_xattn_is_default():
    """The calibration study settled on dense cross-attention conditioning."""
    from amortix import OrnsteinUhlenbeck
    assert FlowPosterior(OrnsteinUhlenbeck()).conditioning == "xattn"


def test_prior_probit_roundtrip():
    """probit normalize/denormalize must round-trip and map the prior to N(0,1)."""
    from amortix import OrnsteinUhlenbeck
    prior = OrnsteinUhlenbeck().prior
    m = prior.sample(20000, generator=torch.Generator().manual_seed(0))
    z = prior.normalize(m)
    assert torch.allclose(prior.denormalize(z), m, atol=1e-4)
    assert z.mean().abs() < 0.05, "normalized prior should be zero-mean"
    assert abs(float(z.std()) - 1.0) < 0.05, "normalized prior should be unit-variance"


def test_diagnostics_runs():
    """SBC harness must produce ranks, coverage and uniformity p-values."""
    from amortix import OrnsteinUhlenbeck
    from amortix.diagnostics import run_sbc, coverage_from_ranks, sbc_uniformity
    prob = OrnsteinUhlenbeck()
    post = FlowPosterior(prob).fit(n_train=128, epochs=1, verbose=False)
    d = prob.prior.dim
    res = run_sbc(post, prob, n_sims=12, n_post=20)
    assert res["ranks"].shape == (12, d)
    assert (res["ranks"] >= 0).all() and (res["ranks"] <= 20).all()
    cov = coverage_from_ranks(res["ranks"], 20, (0.5, 0.9))
    assert set(cov) == {0.5, 0.9} and cov[0.9].shape == (d,)
    p = sbc_uniformity(res["ranks"], 20)
    assert p.shape == (d,) and np.all((p >= 0) & (p <= 1))


def test_sde_vector_state_and_correlated_noise():
    """Vector-state Euler-Maruyama with a correlation Cholesky must run."""
    from amortix import euler_maruyama
    chol = torch.linalg.cholesky(torch.tensor([[1.0, -0.5], [-0.5, 1.0]]))
    m = torch.ones(4, 1)
    traj = euler_maruyama(lambda x, mm: -x, lambda x, mm: torch.ones_like(x),
                          torch.zeros(4, 2), m, dt=0.01, n_steps=20, corr_chol=chol)
    assert traj.shape == (4, 21, 2) and torch.isfinite(traj).all()


def test_sample_batch_matches_single():
    """Batched sampling must agree with the single-dataset path."""
    from amortix import OrnsteinUhlenbeck
    prob = OrnsteinUhlenbeck()
    post = FlowPosterior(prob).fit(n_train=128, epochs=1, verbose=False)
    tk, _ = prob.observe(prob.prior.sample(3, generator=torch.Generator().manual_seed(1)))
    batch = post.sample_batch(tk, n=32, seed=5, n_steps=10)
    single = post.sample_batch(tk[:1], n=32, seed=5, n_steps=10)
    assert batch.shape == (3, 32, prob.prior.dim)
    assert torch.allclose(batch[0], single[0], atol=1e-5)


@pytest.mark.parametrize("solver", ["euler", "midpoint", "rk4"])
def test_solvers_agree(solver):
    """All ODE solvers must produce finite, in-prior samples."""
    from amortix import OrnsteinUhlenbeck
    prob = OrnsteinUhlenbeck()
    post = FlowPosterior(prob).fit(n_train=128, epochs=1, verbose=False)
    s = post.sample(prob.simulate(2)[1][0], n=32, n_steps=10, solver=solver)
    assert torch.isfinite(s).all()
    assert (s >= prob.prior.low - 1e-4).all() and (s <= prob.prior.high + 1e-4).all()


def test_cli_cases_runs(capsys):
    """The console-script entry point must work."""
    from amortix.cli import main
    main(["cases"])
    out = capsys.readouterr().out
    for name in GALLERY:
        assert name in out


def test_base_head_gets_no_gradient_from_cfm():
    """The base must be trained by its own NLL only.

    If z0 were not detached, the CFM loss could be reduced by dragging the base
    onto the target -- paying the base to absorb the flow's job. With
    base_weight=0 the base head must therefore receive exactly zero gradient.
    """
    from amortix import OrnsteinUhlenbeck
    prob = OrnsteinUhlenbeck()
    post = FlowPosterior(prob, base="data")
    post.fit(n_train=256, epochs=1, base_weight=0.0, verbose=False)
    grads = [p.grad for p in post.base_head.parameters() if p.grad is not None]
    assert grads, "base head should still take part in the graph"
    assert all(float(g.abs().max()) == 0.0 for g in grads), \
        "CFM loss leaked into the base head (z0 not detached)"


def test_velocity_couples_parameters():
    """The velocity field must NOT be factorized across parameters.

    An ODE whose velocity for parameter i depends only on z_i maps independent
    coordinates to independent coordinates, so it can never turn an independent
    base into a correlated posterior -- no matter the budget. This regression
    guards the parameter self-attention that makes the field coupled.
    """
    from amortix.problems.linear_gaussian import make
    prob = make()
    d = prob.prior.dim
    post = FlowPosterior(prob, conditioning="xattn")
    post.fit(n_train=128, epochs=1, verbose=False)
    memory = post.encoder.encode(prob.simulate(1)[1])
    cache = post.velocity.encode_memory(memory)
    z = torch.zeros(1, d, requires_grad=True)
    v = post.velocity(z, torch.full((1,), 0.5), cache)
    J = torch.stack([torch.autograd.grad(v[0, i], z, retain_graph=True)[0][0]
                     for i in range(d)])
    off = (J - torch.diag(torch.diag(J))).abs().max()
    assert float(off) > 1e-8, "velocity is factorized across parameters"


def test_posterior_samples_are_independent():
    """Parameter self-attention must act within a sample, never across samples."""
    from amortix.problems.linear_gaussian import make
    prob = make()
    post = FlowPosterior(prob).fit(n_train=128, epochs=1, verbose=False)
    tk, _ = prob.observe(prob.prior.sample(1, generator=torch.Generator().manual_seed(0)))
    a = post.sample_batch(tk, n=8, seed=3, n_steps=10)
    b = post.sample_batch(tk, n=64, seed=3, n_steps=10)
    # the first 8 draws must not change when 56 more are drawn alongside them
    assert torch.allclose(a[0], b[0, :8], atol=1e-5), "samples interact with each other"


def test_base_nll_does_not_train_the_encoder():
    """The shared encoder must be shaped by the flow objective, not by the base.

    The base NLL is a far stronger gradient signal than the CFM term; if it
    reaches the encoder, the token memory becomes a good linear readout for a
    diagonal Gaussian rather than information for the velocity field.
    """
    from amortix import OrnsteinUhlenbeck
    prob = OrnsteinUhlenbeck()
    post = FlowPosterior(prob, base="data")
    m, tokens = prob.simulate(64)
    z1 = prob.prior.normalize(m)
    ctx = post.encoder.pool(post.encoder.encode(tokens))
    nll, _, _ = post.base_head.nll(z1, ctx.detach())
    nll.backward()
    enc_grads = [p.grad for p in post.encoder.parameters() if p.grad is not None]
    assert all(float(g.abs().max()) == 0.0 for g in enc_grads), \
        "base NLL leaked into the encoder"


def test_padding_is_invisible():
    """A masked-out observation slot must not affect the posterior at all.

    The encoder honoured the mask but the velocity's cross-attention did not, so
    padded slots leaked into the field; and the RoPE table was capped at the
    observer's token count, so anything longer simply crashed. Both are guarded
    here, including a length beyond the original cap.
    """
    from amortix import OrnsteinUhlenbeck
    prob = OrnsteinUhlenbeck()
    post = FlowPosterior(prob).fit(n_train=128, epochs=1, verbose=False)
    tk, _ = prob.observe(prob.prior.sample(1, generator=torch.Generator().manual_seed(0)))
    T = tk.shape[1]
    pad = torch.randn(1, 30, tk.shape[2]) * 5.0
    tkp = torch.cat([tk, pad], dim=1)
    mask = torch.zeros(1, T + 30, dtype=torch.bool)
    mask[:, :T] = True
    a = post.sample_batch(tk, n=32, seed=3, n_steps=8)
    b = post.sample_batch(tkp, n=32, seed=3, n_steps=8, mask=mask)
    assert torch.allclose(a, b, atol=1e-6), "padded slots leak into the posterior"


def test_variable_length_inputs():
    """Observation sets of different sizes can be scored in one call."""
    from amortix import OrnsteinUhlenbeck
    from amortix.flow import pack_tokens
    prob = OrnsteinUhlenbeck()
    post = FlowPosterior(prob).fit(n_train=128, epochs=1, verbose=False)
    tk, _ = prob.observe(prob.prior.sample(1, generator=torch.Generator().manual_seed(0)))
    sets = [tk[0][:30], tk[0][:74], tk[0][:55]]
    packed, mask = pack_tokens(sets)
    assert packed.shape[1] == 74 and mask.sum() == 30 + 74 + 55
    out = post.sample_batch(sets, n=16, seed=1, n_steps=8)
    assert out.shape == (3, 16, prob.prior.dim) and torch.isfinite(out).all()


def test_plain_base_is_default():
    """The data-dependent base is off by default: it measurably hurts.

    On the exact-posterior testbed (24 held-out datasets, 12000 steps) the plain
    N(0,I) source scores 0.00140 against 0.00245 for the learned Gaussian base --
    1.75x better and simpler. The base also cannot be fixed naively: making its
    pooling trainable improves the predicted centre 4.4x and makes the posterior
    2.6x worse, because a well-placed base is a moving source for the CFM
    regression.
    """
    from amortix import OrnsteinUhlenbeck
    post = FlowPosterior(OrnsteinUhlenbeck())
    assert post.base == "standard" and post.base_head is None


def test_wdiff_embed_trains_and_samples():
    """embed='wdiff': learnable warped-increment embedding is wired end-to-end.

    The warp parameter must exist, receive gradient, and the posterior must
    sample finite. Warped-increment embeddings are the universal alternative
    to hand-crafting a log-coordinate observer: on raw-price GBM they take
    sigma's SBC from p=0.002 to p=0.876 (wdiff) / p=0.283 (wbasis) at
    production budget (see WarpDiffEmbed's docstring).
    """
    from amortix import OrnsteinUhlenbeck
    prob = OrnsteinUhlenbeck()
    post = FlowPosterior(prob, dim_model=32, depth=2, embed="wdiff")
    assert post.encoder.in_norm is None
    assert any(n.endswith("embed.raw_s") for n, _ in post.named_parameters())
    post.fit(n_train=128, epochs=1, batch=32, verbose=False)
    g = post.encoder.embed.raw_s.grad
    assert g is not None and torch.isfinite(g).all()
    _, tok = prob.simulate(2)
    out = post.sample_batch(tok, n=8, seed=0, n_steps=8)
    assert out.shape == (2, 8, prob.prior.dim) and torch.isfinite(out).all()


def test_auto_embed_resolution():
    """embed='auto' (the default): universal learnable warped-increment
    embedding (wbasis) for PathObserver problems, plain linear elsewhere.

    The package's design position is universal architectures over manual
    feature engineering: no problem ships a hand-crafted observer transform;
    scale handling lives in the learnable embedding. The MonotoneWarp's
    octave coefficients must exist and receive gradient.
    """
    import torch.nn as nn
    from amortix import OrnsteinUhlenbeck
    from amortix.encoder import WarpDiffEmbed
    from amortix.problems.linear_gaussian import make as make_lg

    post = FlowPosterior(OrnsteinUhlenbeck(), dim_model=32, depth=2)
    assert isinstance(post.encoder.embed, WarpDiffEmbed)
    assert post.encoder.embed.kind == "basis"
    post.fit(n_train=128, epochs=1, batch=32, verbose=False)
    g = post.encoder.embed.warp.raw_c.grad
    assert g is not None and torch.isfinite(g).all()

    post_lg = FlowPosterior(make_lg(), dim_model=32, depth=2)
    assert isinstance(post_lg.encoder.embed, nn.Linear)


def test_wpair_trope_variable_designs():
    """embed='wpair' + rope='time': the variable-design configuration.

    A problem may return (m, tokens, mask) from simulate() -- every dataset
    its own K -- and the posterior must sample finite for small designs.
    NOTE: wpair's consecutive-difference features fix the input contract to
    "sorted by t"; order invariance is a property of pointwise embeddings
    with rope='time', not of wpair.
    """
    import torch as th
    from amortix import OrnsteinUhlenbeck

    class OUVarDesign(OrnsteinUhlenbeck):
        def simulate(self, n, generator=None):
            m = self.prior.sample(n, generator)
            traj = self.simulate_paths(m, generator)
            K = 24
            tokens = th.zeros(n, K, 6)
            mask = th.zeros(n, K, dtype=th.bool)
            g = generator or th.Generator()
            for i in range(n):
                k = int(th.randint(4, K + 1, (1,), generator=g))
                idx = th.randperm(self.observer.n_steps, generator=g)[:k].add(1).sort().values
                tokens[i, :k, 0] = idx.float() * self.observer.dt_sim / self.observer.horizon
                tokens[i, :k, 1] = traj[i, idx, 0]
                mask[i, :k] = True
            return m, tokens, mask

    prob = OUVarDesign()
    post = FlowPosterior(prob, dim_model=32, depth=2, embed="wpair", rope="time")
    post.fit(n_train=96, epochs=1, batch=32, verbose=False)
    gen = torch.Generator().manual_seed(3)
    m = prob.prior.sample(1, generator=gen)
    traj = prob.simulate_paths(m, generator=gen)
    idx = torch.randperm(prob.observer.n_steps, generator=gen)[:9].add(1).sort().values
    t = idx.float() * prob.observer.dt_sim / prob.observer.horizon
    x = traj[0, idx, 0]
    z = torch.zeros_like(x)
    tok = torch.stack([t, x, z, z, z, z], dim=-1)
    out = post.sample_batch([tok], n=16, seed=0, n_steps=8)
    assert out.shape == (1, 16, prob.prior.dim) and torch.isfinite(out).all()


def test_design_zoo_end_to_end():
    """Every design-zoo case simulates finite trajectories, trains one step
    under the canonical fresh-design recipe with package defaults
    (embed/rope by the class rule), and samples finite posteriors for a
    small variable-length design."""
    from amortix.problems.design_zoo import DESIGN_ZOO
    from amortix.encoder import PointEmbed, SetCondPairEmbed

    for name, C in DESIGN_ZOO.items():
        prob = C()
        m, raw = prob.simulate(4)
        assert torch.isfinite(raw).all(), name
        post = FlowPosterior(prob, dim_model=32, depth=2)
        # one universal default for every design problem; the pair embedding
        # is opt-in only
        assert isinstance(post.encoder.embed, PointEmbed), name
        assert isinstance(
            FlowPosterior(prob, dim_model=32, depth=2,
                          embed="wfilm").encoder.embed,
            SetCondPairEmbed), name
        post.fit(n_train=32, epochs=1, batch=16, verbose=False,
                 retokenize=prob.make_retokenizer())
        gen = torch.Generator().manual_seed(0)
        k = prob.k_min + 4
        tidx, cidx = prob.sample_design(gen, k)
        tok = prob.tokens_for(raw[0], tidx, cidx, gen)
        out = post.sample_batch([tok], n=8, seed=0, n_steps=6)
        assert out.shape == (1, 8, prob.prior.dim), name
        assert torch.isfinite(out).all(), name


def test_tokens_from_data_matches_tokens_for():
    """tokens_from_data on grid-aligned readings must reproduce tokens_for.

    GBMDesign has no observation noise, so the two tokenizers see identical
    values; the resulting [K, 6] tensors must agree feature by feature.
    """
    from amortix import tokens_from_data
    from amortix.problems.design_basic import GBMDesign

    prob = GBMDesign()
    gen = torch.Generator().manual_seed(4)
    m = prob.prior.sample(1, gen)
    raw = prob.trajectories(m, gen)
    tidx, cidx = prob.sample_design(gen, 17)
    ref = prob.tokens_for(raw[0], tidx, cidx, gen)

    times = tidx.float() * prob.observer.dt_sim
    values = raw[0, tidx, 0]
    tok = tokens_from_data(prob, times, values, channels=cidx)
    assert tok.shape == ref.shape and tok.dtype == torch.float32
    assert torch.allclose(tok, ref)
    # channel default: all zeros, matching the single-channel design draw
    assert torch.allclose(tokens_from_data(prob, times, values), ref)


def test_design_token_layout_is_enforced():
    """DesignObserver tokens that depart from the documented layout are
    rejected at sample() and at the first batch of fit(); padded rows are
    not inspected.

    The bare-point embedding reads slots 0, 1, 4 and 5 only. A second
    series packed into slot 2 is therefore dropped without any error unless
    the layout is checked at the entry points; the error must name the
    slot and the supported constructors.
    """
    from amortix.problems.design_basic import LotkaVolterraDesign

    prob = LotkaVolterraDesign()
    obs = prob.observer
    post = FlowPosterior(prob, dim_model=32, depth=2)
    gen = torch.Generator().manual_seed(0)
    _, raw = prob.simulate(1, generator=gen)
    tidx, cidx = prob.sample_design(gen, 12)
    tok = prob.tokens_for(raw[0], tidx, cidx, gen)
    assert post.sample(tok, n=4, n_steps=2).shape == (4, prob.prior.dim)

    # a second data channel in the reserved slot 2
    bad = tok.clone()
    bad[:, 2] = raw[0, tidx, 1]
    with pytest.raises(ValueError, match="slot 2") as info:
        post.sample(bad, n=4, n_steps=2)
    assert "tokens_for" in str(info.value)
    assert "tokens_from_data" in str(info.value)
    # the batch entry point and the variable-length list input, same check
    with pytest.raises(ValueError, match="slot 2"):
        post.sample_batch(bad[None], n=4, n_steps=2)
    with pytest.raises(ValueError, match="slot 2"):
        post.sample_batch([tok, bad], n=4, n_steps=2)

    def rejected(edit, slot):
        t = tok.clone()
        edit(t)
        with pytest.raises(ValueError, match=f"slot {slot}"):
            post.sample(t, n=4, n_steps=2)

    rejected(lambda t: t.__setitem__((3, 3), 1.0), 3)          # reserved
    rejected(lambda t: t.__setitem__((0, 0), 1.5), 0)          # beyond horizon
    rejected(lambda t: t.__setitem__((0, 0), -0.1), 0)         # before zero
    rejected(lambda t: t.__setitem__((5, 1), float("nan")), 1) # non-finite value
    rejected(lambda t: t.__setitem__((slice(None), 4), 1.0), 4)   # K metadata != 12
    rejected(lambda t: t.__setitem__((slice(None), 4), 1.5), 4)   # out of [0, 1]
    rejected(lambda t: t.__setitem__((slice(None), 5), 2.0), 5)   # channel >= n_channels
    rejected(lambda t: t.__setitem__((slice(None), 5), 0.5), 5)   # non-integer channel
    rejected(lambda t: t.__setitem__((0, 5), -1.0), 5)            # negative channel

    # padded rows are skipped: arbitrary content under mask=False is fine
    padded = torch.zeros(1, obs.k_max, 6)
    padded[0, :12] = tok
    padded[0, 12:, 2] = 7.0
    mask = torch.zeros(1, obs.k_max, dtype=torch.bool)
    mask[0, :12] = True
    out = post.sample_batch(padded, n=4, n_steps=2, mask=mask)
    assert out.shape == (1, 4, prob.prior.dim) and torch.isfinite(out).all()
    # zero padding with the mask forgotten: the design-size entry no longer
    # matches the number of rows the set would be read with
    padded[0, 12:, 2] = 0.0
    with pytest.raises(ValueError, match="slot 4"):
        post.sample_batch(padded, n=4, n_steps=2)
    # wrong feature count
    with pytest.raises(ValueError, match="shape"):
        post.sample(tok[:, :5], n=4, n_steps=2)

    # fit(): a retokenizer that writes slot 2 is rejected on the first batch
    base = prob.make_retokenizer()

    def retok(raw_b, g):
        t, m = base(raw_b, g)
        t = t.clone()
        t[..., 2] = 1.0
        return t, m

    with pytest.raises(ValueError, match="slot 2"):
        post.fit(n_train=32, epochs=1, batch=16, verbose=False, retokenize=retok)
    # the conforming retokenizer trains
    post.fit(n_train=32, epochs=1, batch=16, verbose=False,
             retokenize=base)


def test_tokens_from_data_multi_channel():
    """Two channels through tokens_from_data: one reading per row, channel
    id in slot 5, reserved slots zero; the set passes the layout check and
    samples. Channel ids outside [0, n_channels) or non-integer are
    rejected by the constructor itself.
    """
    from amortix import tokens_from_data
    from amortix.problems.design_basic import LotkaVolterraDesign

    prob = LotkaVolterraDesign()
    obs = prob.observer
    post = FlowPosterior(prob, dim_model=32, depth=2)
    gen = torch.Generator().manual_seed(1)
    _, raw = prob.simulate(1, generator=gen)
    tidx = torch.tensor([50, 100, 150, 200, 250, 300]).repeat_interleave(2)
    channels = [0, 1] * 6
    times = tidx.float() * obs.dt_sim
    values = raw[0, tidx, torch.tensor(channels)]

    tok = tokens_from_data(prob, times, values, channels=channels)
    assert tok.shape == (12, 6)
    assert torch.equal(tok[:, 5], torch.tensor(channels, dtype=torch.float32))
    assert (tok[:, 2:4] == 0).all()
    assert torch.allclose(tok[:, 4], torch.full((12,), math.log(12) / math.log(obs.k_max)))
    out = post.sample(tok, n=4, n_steps=2)
    assert out.shape == (4, prob.prior.dim) and torch.isfinite(out).all()
    # numpy channel ids and a tensor of ids are accepted alike
    assert torch.equal(tokens_from_data(prob, times, values, channels=np.array(channels)), tok)
    assert torch.equal(tokens_from_data(prob, times, values, channels=torch.tensor(channels)), tok)

    with pytest.raises(ValueError, match="channel"):
        tokens_from_data(prob, times, values, channels=[0, 2] * 6)
    with pytest.raises(ValueError, match="channel"):
        tokens_from_data(prob, times, values, channels=[-1, 0] * 6)
    with pytest.raises(ValueError, match="channel"):
        tokens_from_data(prob, times, values, channels=[0, 0.5] * 6)
    # times beyond the horizon and designs larger than k_max are layout errors too
    with pytest.raises(ValueError, match="slot 0"):
        tokens_from_data(prob, times + obs.horizon, values, channels=channels)
    big = obs.k_max + 1
    with pytest.raises(ValueError, match="more than observer.k_max"):
        tokens_from_data(prob, torch.linspace(0, obs.horizon, big),
                         torch.ones(big))


def test_design_sbc_runs():
    """sbc_design produces finite p-values on a tiny run."""
    import numpy as np
    from amortix.designs import sbc_design
    from amortix.problems.design_zoo import PharmacoKineticsDesign

    prob = PharmacoKineticsDesign()
    post = FlowPosterior(prob, dim_model=32, depth=2)
    post.fit(n_train=32, epochs=1, batch=16, verbose=False,
             retokenize=prob.make_retokenizer())
    p = sbc_design(post, prob, n_sims=24, n_post=20, seed=0)
    assert p.shape == (3,) and np.isfinite(p).all()


@pytest.mark.parametrize("embed, cls_name", [
    ("auto", "PointEmbed"),          # the design default (bare-point tokens)
    ("wfilm", "SetCondPairEmbed"),   # the opt-in set-conditioned pair embedding
    ("wpair", "WarpPairEmbed"),
])
def test_load_posterior_roundtrip_design(tmp_path, embed, cls_name):
    """load_posterior rebuilds the embedding a checkpoint was trained with.

    The default embedding depends on the problem class and has changed over
    time, so a loader that assumes the current default silently puts an
    older checkpoint into the wrong architecture. The mode, width and depths
    are read off the state dict; the reloaded model must be the same class
    and reproduce the original's draws exactly.
    """
    from amortix import encoder
    from amortix.evaluation import load_posterior
    from amortix.problems.design_basic import OUDesign

    prob = OUDesign()
    post = FlowPosterior(prob, dim_model=32, depth=2, embed=embed)
    post.fit(n_train=32, epochs=1, batch=16, verbose=False,
             retokenize=prob.make_retokenizer())
    path = tmp_path / f"{embed}.pt"
    torch.save(post.state_dict(), path)

    loaded = load_posterior(prob, str(path), device="cpu")
    assert type(loaded.encoder.embed) is getattr(encoder, cls_name)
    assert type(loaded.encoder.embed) is type(post.encoder.embed)
    assert loaded.embed_mode == post.embed_mode
    assert loaded.encoder.dim == 32
    assert len(loaded.encoder.blocks) == len(post.encoder.blocks)
    assert len(loaded.velocity.blocks) == 2

    gen = torch.Generator().manual_seed(0)
    _, raw = prob.simulate(1)
    tidx, cidx = prob.sample_design(gen, prob.k_min + 4)
    tok = prob.tokens_for(raw[0], tidx, cidx, gen)
    post.eval()
    a = post.sample_batch([tok], n=8, seed=0, n_steps=6)
    b = loaded.sample_batch([tok], n=8, seed=0, n_steps=6)
    assert b.shape == (1, 8, prob.prior.dim) and torch.isfinite(b).all()
    assert torch.allclose(a, b, atol=1e-5), "reloaded model draws differ"


@pytest.mark.parametrize("problem_name, cls_name", [
    ("linear_gaussian", "Linear"),   # plain linear embed: keys embed.weight/bias
    ("ou", "WarpDiffEmbed"),         # PathObserver default (wbasis)
])
def test_load_posterior_roundtrip_gallery(tmp_path, problem_name, cls_name):
    """The audit case and the SDE default: neither embedding has a ``proj``
    submodule at the place the previous loader assumed (linear has none;
    the warp classes do, the linear one does not), and both must load."""
    import torch.nn as nn
    from amortix import encoder
    from amortix.evaluation import load_posterior

    mod = importlib.import_module(f"amortix.problems.{problem_name}")
    prob = mod.make()
    post = FlowPosterior(prob, dim_model=32, depth=2)
    post.fit(n_train=64, epochs=1, batch=32, verbose=False)
    path = tmp_path / f"{problem_name}.pt"
    torch.save(post.state_dict(), path)

    loaded = load_posterior(prob, str(path), device="cpu")
    expected = getattr(nn, cls_name, None) or getattr(encoder, cls_name)
    assert type(loaded.encoder.embed) is expected
    assert type(loaded.encoder.embed) is type(post.encoder.embed)
    tk, _ = prob.observe(prob.prior.sample(1, generator=torch.Generator().manual_seed(0)))
    post.eval()
    a = post.sample_batch(tk, n=8, seed=0, n_steps=6)
    b = loaded.sample_batch(tk, n=8, seed=0, n_steps=6)
    assert b.shape == (1, 8, prob.prior.dim) and torch.isfinite(b).all()
    assert torch.allclose(a, b, atol=1e-5), "reloaded model draws differ"
