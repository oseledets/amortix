"""Sanity anchor: amortix (CFM) vs `sbi` NPE and FMPE on the linear-Gaussian
testbed -- the one task built for THEIR interface.

    m ~ U([-3, 3]^4),   y = A m + eps,   eps ~ N(0, 0.5^2 I_6)

The observation is a fixed-length 6-vector with homogeneous scales: exactly
the input NPE/FMPE consume natively, no summary network, no representation
question. The posterior is exact (a correlated Gaussian restricted to the
prior box, sampled by rejection -- amortix.problems.linear_gaussian, the same
instrument the report's exact references use). So this script is the control
for the external-tool comparison: if the sbi arms match amortix here at
package defaults, their failures on raw SDE time series elsewhere read as an
input-representation effect, not a broken harness.

Both learners get the same simulation budget from the same simulator and
prior, run on the same device at package-default hyperparameters, and are
scored on the same test datasets against the same exact references. Dumps go
to $AMX_DUMP as lingauss_<arm>.npz / ref_lingauss.npz for
scripts/score_baselines.py.

Needs `sbi` (not a package dependency):

    uv run --with "sbi==0.27.0" python examples/baseline_lingauss.py --device cuda
    uv run --with "sbi==0.27.0" python examples/baseline_lingauss.py --quick
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from amortix import FlowPosterior
from amortix.problems.linear_gaussian import make, exact_posterior


def score(samples, exact, names):
    """samples [B, n, d] vs exact list of [n_e, d] -> per-param bias/width."""
    B = len(exact)
    out = {}
    for j, nm in enumerate(names):
        bias = np.empty(B)
        width = np.empty(B)
        for i in range(B):
            em, es = exact[i][:, j].mean(), exact[i][:, j].std()
            bias[i] = (samples[i, :, j].mean() - em) / max(es, 1e-12)
            width[i] = samples[i, :, j].std() / max(es, 1e-12)
        out[nm] = dict(bias=float(bias.mean()),
                       bias_se=float(bias.std() / np.sqrt(B)),
                       width=float(width.mean()),
                       width_se=float(width.std() / np.sqrt(B)))
    return out


def fmt(tag, res, names):
    cells = "  ".join(
        f"{nm}: bias {res[nm]['bias']:+.3f}±{res[nm]['bias_se']:.3f} "
        f"width {res[nm]['width']:.3f}" for nm in names)
    print(f"  {tag:<10s} {cells}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--n_draw", type=int, default=2000)
    ap.add_argument("--n_ref", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--arms", type=str, default="amortix,npe,fmpe",
                    help="comma list from {amortix,npe,fmpe}")
    ap.add_argument("--device", type=str, default="cpu",
                    choices=["cpu", "cuda", "auto"],
                    help="device for amortix and the sbi arms; timings are "
                         "only comparable within one device")
    ap.add_argument("--save", type=str, default="results/BASELINE_LINGAUSS.json")
    args = ap.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.quick:
        args.n_train, args.steps, args.n_test, args.n_draw, args.n_ref = \
            2000, 600, 24, 500, 1000

    prob = make()
    names = prob.prior.names
    print(f"[setup] linear-Gaussian fixed design, d={prob.prior.dim} params, "
          f"{prob.observer.n_tokens} observation components, "
          f"budget {args.n_train} sims", flush=True)

    # ---- shared test set + exact references --------------------------------
    gen = torch.Generator().manual_seed(999)
    m_true = prob.prior.sample(args.n_test, gen)
    tokens_test, y_test = prob.observe(m_true, gen)     # tokens [B,6,2], y [B,6]
    exact = []
    for i in range(args.n_test):
        ex = exact_posterior(y_test[i], prob, n=args.n_ref, seed=5000 + i).numpy()
        if ex.shape[0] < args.n_ref:
            raise RuntimeError(
                f"exact reference starved on set {i}: {ex.shape[0]} draws")
        exact.append(ex)
    print("[setup] exact references ready", flush=True)
    dump = os.environ.get("AMX_DUMP", "")
    if dump:
        os.makedirs(dump, exist_ok=True)
        np.savez(f"{dump}/ref_lingauss.npz", samples=np.stack(exact))

    report = dict(n_train=args.n_train, steps=args.steps, n_test=args.n_test,
                  n_draw=args.n_draw, device=args.device,
                  torch_threads=torch.get_num_threads())
    arms = set(args.arms.split(","))

    # ---- amortix CFM --------------------------------------------------------
    if "amortix" in arms:
        torch.manual_seed(args.seed)
        post = FlowPosterior(prob)                       # package defaults
        n_par = sum(p.numel() for p in post.parameters())
        t0 = time.time()
        post.fit(n_train=args.n_train, steps=args.steps, seed=args.seed,
                 verbose=False, device=args.device)
        t_train_amx = time.time() - t0
        t0 = time.time()
        amx = post.sample_batch(tokens_test, n=args.n_draw, seed=0).numpy()
        t_inf_amx = time.time() - t0
        if dump:
            np.savez(f"{dump}/lingauss_amortix.npz", samples=amx)
        res_amx = score(amx, exact, names)
        print(f"[amortix] {n_par:,} params, train {t_train_amx:.0f}s, "
              f"batch inference {1e3 * t_inf_amx / args.n_test:.0f} ms/dataset",
              flush=True)
        fmt("amortix", res_amx, names)
        report["amortix"] = dict(params=n_par, train_s=t_train_amx,
                                 inf_ms_per_ds=1e3 * t_inf_amx / args.n_test,
                                 **res_amx)

    # ---- sbi arms: NPE (MAF, MLE) and FMPE (flow matching) ------------------
    # Input is the raw 6-vector y -- the natural format, nothing engineered.
    def run_sbi_arm(tag, method="NPE"):
        import sbi
        import sbi.inference as sbi_inf
        from sbi.utils import BoxUniform as SbiBox

        sbi_prior = SbiBox(low=prob.prior.low.float(),
                           high=prob.prior.high.float(), device=args.device)
        gen2 = torch.Generator().manual_seed(args.seed + 1)
        theta = prob.prior.sample(args.n_train, gen2)
        x_tr = prob._forward(theta, gen2)                # raw y, [n_train, 6]
        t0 = time.time()
        inf = getattr(sbi_inf, method)(prior=sbi_prior,  # package defaults
                                       device=args.device)
        de = inf.append_simulations(theta.float(), x_tr.float()) \
                .train(show_train_summary=False)
        arm_post = inf.build_posterior(de)
        t_train = time.time() - t0
        n_par = sum(p.numel() for p in de.parameters())
        xt = y_test.float().to(args.device)
        t0 = time.time()
        smp = np.stack([
            arm_post.sample((args.n_draw,), x=xt[i],
                            show_progress_bars=False).cpu().numpy()
            for i in range(args.n_test)])
        t_inf = time.time() - t0
        if dump:
            np.savez(f"{dump}/lingauss_{tag}.npz", samples=smp)
        res = score(smp, exact, names)
        print(f"[sbi {sbi.__version__} {tag}] {n_par:,} params, "
              f"train {t_train:.0f}s, "
              f"inference {1e3 * t_inf / args.n_test:.0f} ms/dataset",
              flush=True)
        fmt(tag, res, names)
        report[tag] = dict(sbi_version=sbi.__version__, params=n_par,
                           train_s=t_train,
                           inf_ms_per_ds=1e3 * t_inf / args.n_test, **res)

    if "npe" in arms:
        run_sbi_arm("npe")
    if "fmpe" in arms:
        run_sbi_arm("fmpe", method="FMPE")

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(report, f, indent=1)
        print(f"[saved] {args.save}", flush=True)
    print("DONE_BASELINE", flush=True)


if __name__ == "__main__":
    main()
