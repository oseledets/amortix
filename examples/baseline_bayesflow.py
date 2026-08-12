"""BayesFlow-2 arm of the GBM head-to-head (see baseline_npe.py for the setup).

Same protocol as the sbi arms: same simulation budget, same test datasets,
same exact conjugate references, package defaults. BayesFlow 2 (keras 3) is
run on the torch backend. Two arms: their default CouplingFlow and their
FlowMatching network, both conditioned on the same raw observed prices the
other methods get (plus a log-return variant for the hand-engineering
ladder).

    KERAS_BACKEND=torch uv run --with bayesflow python examples/baseline_bayesflow.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_npe import exact_posterior, score, fmt  # noqa: E402

from amortix.problems import gbm                       # noqa: E402
from amortix.mcmc import observed_indices              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--n_draw", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--arms", type=str, default="coupling,flow_matching")
    ap.add_argument("--input", type=str, default="raw",
                    choices=["raw", "returns"])
    ap.add_argument("--save", type=str, default="results/BASELINE_BAYESFLOW.json")
    args = ap.parse_args()
    if args.quick:
        args.n_train, args.n_test, args.n_draw, args.epochs = 2000, 24, 500, 8

    import bayesflow as bf
    import keras

    prob = gbm.make()
    names = prob.prior.names
    idx = observed_indices(prob)
    x_idx = idx[idx > 0]

    def transform(x):
        if args.input == "returns":
            xl = x.clamp_min(1e-8).log()
            return torch.diff(xl, dim=1,
                              prepend=torch.zeros(xl.shape[0], 1)).contiguous()
        return x.contiguous()

    # shared test set + exact references (same seeds as baseline_npe.py)
    gen = torch.Generator().manual_seed(999)
    m_true = prob.prior.sample(args.n_test, gen)
    traj = prob.simulate_paths(m_true, gen)
    x_test = transform(traj[:, x_idx, 0]).numpy().astype("float32")
    exact = [exact_posterior(prob, traj[i, :, 0].numpy(), idx, seed=5000 + i)
             for i in range(args.n_test)]
    print(f"[setup] {len(x_idx)}-dim conditions ({args.input}), "
          f"{args.n_train} sims, bayesflow {bf.__version__} "
          f"on keras/{keras.backend.backend()}", flush=True)

    gen2 = torch.Generator().manual_seed(args.seed + 1)
    theta_t = prob.prior.sample(args.n_train, gen2)
    x_train = transform(prob.simulate_paths(theta_t, gen2)[:, x_idx, 0]) \
        .numpy().astype("float32")
    theta = theta_t.numpy().astype("float32")
    data = dict(theta=theta, x=x_train)

    report = dict(n_train=args.n_train, n_test=args.n_test,
                  n_draw=args.n_draw, epochs=args.epochs, input=args.input,
                  bayesflow=bf.__version__)

    nets = dict(coupling=bf.networks.CouplingFlow,
                flow_matching=bf.networks.FlowMatching)
    for tag in args.arms.split(","):
        keras.utils.set_random_seed(args.seed)
        adapter = (bf.Adapter()
                   .convert_dtype("float64", "float32")
                   .rename("theta", "inference_variables")
                   .rename("x", "inference_conditions"))
        approx = bf.ContinuousApproximator(
            inference_network=nets[tag](), adapter=adapter,
            standardize="all")     # their built-in z-scoring, same as sbi's
        approx.compile(optimizer=keras.optimizers.Adam(1e-3))
        dataset = bf.OfflineDataset(data=dict(data), batch_size=64,
                                    adapter=approx.adapter)
        t0 = time.time()
        approx.fit(dataset=dataset, epochs=args.epochs, verbose=0)
        t_train = time.time() - t0
        n_par = sum(int(np.prod(w.shape)) for w in approx.weights)
        t0 = time.time()
        smp = approx.sample(conditions=dict(x=x_test),
                            num_samples=args.n_draw)["theta"]
        t_inf = time.time() - t0
        smp = np.asarray(smp)
        res = score(smp, exact, names)
        print(f"[bayesflow {tag}] {n_par:,} params, train {t_train:.0f}s, "
              f"inference {1e3 * t_inf / args.n_test:.1f} ms/dataset",
              flush=True)
        fmt(tag, res, names)
        report[tag] = dict(params=n_par, train_s=t_train,
                           inf_ms_per_ds=1e3 * t_inf / args.n_test, **res)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(report, f, indent=1)
        print(f"[saved] {args.save}", flush=True)
    print("DONE_BAYESFLOW", flush=True)


if __name__ == "__main__":
    main()
