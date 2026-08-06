"""Is the velocity field the one CFM theory says it should be?

For a Gaussian target the optimal field is available in closed form, so we can
compare the LEARNED field to the TRUE one pointwise instead of judging the flow
only by its samples. That separates two very different failures:

    field wrong          -> a learning problem (capacity, optimization, objective)
    field right, samples wrong -> an integration / sampling problem

Setup (deliberately bypasses the Problem/encoder machinery so only the velocity
and the CFM training loop are under test): context c, target z1|c ~ N(W c, S)
with a strongly correlated S, base z0 ~ N(0, I), independent coupling, linear
path z_t = (1-t) z0 + t z1.

With z0 ~ N(0,I) and z1 ~ N(m,S) independent, (z_t, u = z1 - z0) is jointly
Gaussian, so

    E[u | z_t] = m + C_uz C_zz^{-1} (z_t - t m),
    C_zz = (1-t)^2 I + t^2 S,      C_uz = t S - (1-t) I.

    uv run python examples/verify_velocity.py
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from amortix.flow import CrossCondVelocity

D, CDIM, DIM, T_TOK = 2, 4, 64, 4
RHO = 0.85


def analytic_velocity(z, t, m, S):
    """E[u | z_t = z] for the Gaussian case, in closed form. z [N,D], m [N,D]."""
    I = torch.eye(D, dtype=torch.float64)
    out = torch.empty_like(z, dtype=torch.float64)
    for i in range(z.shape[0]):
        ti = float(t[i])
        Czz = (1 - ti) ** 2 * I + ti ** 2 * S
        Cuz = ti * S - (1 - ti) * I
        out[i] = m[i].double() + Cuz @ torch.linalg.solve(Czz, (z[i].double() - ti * m[i].double()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    torch.manual_seed(0)

    # correlated Gaussian target, mean linear in the context
    L = torch.tensor([[1.0, 0.0], [RHO, (1 - RHO ** 2) ** 0.5]], dtype=torch.float64)
    S = (L @ L.T)
    Lf = L.float()
    W = torch.randn(D, CDIM) * 0.5
    proj = torch.randn(CDIM, DIM) * 0.5          # fixed, untrained context -> memory

    def batch(n):
        c = torch.randn(n, CDIM)
        m = c @ W.T
        z1 = m + torch.randn(n, D) @ Lf.T
        memory = (c @ proj)[:, None, :].expand(n, T_TOK, DIM).contiguous()
        return c, m, z1, memory

    vel = CrossCondVelocity(D, DIM, n_head=4, n_layer=3)
    opt = torch.optim.Adam(vel.parameters(), lr=args.lr)
    print(f"training the velocity on a known Gaussian target (rho={RHO}) ...")
    for step in range(args.steps):
        c, m, z1, memory = batch(args.batch)
        z0 = torch.randn(args.batch, D)
        t = torch.rand(args.batch)
        zt = (1 - t)[:, None] * z0 + t[:, None] * z1
        pred = vel(zt, t, vel.encode_memory(memory))
        loss = ((pred - (z1 - z0)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % (args.steps // 5) == 0 or step == args.steps - 1:
            print(f"  step {step:5d}  loss {loss.item():.4f}")

    # ---- 1. learned field vs the analytic optimum ------------------------
    with torch.no_grad():
        c, m, z1, memory = batch(400)
        cache = vel.encode_memory(memory)
        print(f"\n{'t':>5} | {'rel. err of the learned field':>30}")
        for tv in (0.05, 0.25, 0.5, 0.75, 0.95):
            z0 = torch.randn(400, D)
            t = torch.full((400,), tv)
            zt = (1 - tv) * z0 + tv * z1
            learned = vel(zt, t, cache).double()
            exact = analytic_velocity(zt, t, m, S)
            rel = float((learned - exact).norm() / exact.norm())
            print(f"{tv:5.2f} | {rel:29.1%}")

        # ---- 2. does integrating the learned field give the right law? ---
        n = 20000
        c1 = torch.randn(1, CDIM)
        m1 = c1 @ W.T
        mem1 = (c1 @ proj)[:, None, :].expand(1, T_TOK, DIM).contiguous()
        cache1 = vel.encode_memory(mem1)
        z = torch.randn(n, D)
        cache_n = [(k.expand(n, -1, -1, -1), v.expand(n, -1, -1, -1)) for k, v in cache1]
        n_steps = 100
        dt = 1.0 / n_steps
        for i in range(n_steps):                       # midpoint, as in sample_batch
            t = torch.full((n,), i * dt)
            k1 = vel(z, t, cache_n)
            z = z + dt * vel(z + 0.5 * dt * k1, t + 0.5 * dt, cache_n)
        got_m, got_S = z.mean(0).double(), torch.cov(z.T.double())
        print(f"\nintegrating the learned field (exact answer: m={m1[0].tolist()}, "
              f"corr={RHO}):")
        print(f"  mean      : got {got_m.tolist()}  |  err {float((got_m-m1[0].double()).norm()):.4f}")
        print(f"  std       : got {[round(float(x),3) for x in got_S.diag().sqrt()]}  "
              f"|  exact {[round(float(x),3) for x in S.diag().sqrt()]}")
        r_got = float(got_S[0, 1] / (got_S[0, 0] * got_S[1, 1]).sqrt())
        print(f"  correlation: got {r_got:.3f}  |  exact {RHO:.3f}")
        print("\nfield accurate but law wrong -> integration; field wrong -> learning")


if __name__ == "__main__":
    main()
