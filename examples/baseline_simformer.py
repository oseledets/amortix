"""Simformer arm of the GBM head-to-head (see baseline_npe.py for the setup).

Faithful port of the authors' minimal example
(mackelab/simformer, example/1_minimal_code_example.ipynb: score-based
diffusion over the joint (theta, x) with a transformer, random condition
masks, their architecture and VESDE settings) to the amortix GBM benchmark
data. Two adaptations: global per-node z-scoring of inputs (the same role
sbi/BayesFlow default z-scoring plays; their toy data was already O(1)) and
a fixed simulation set instead of an on-the-fly simulator.

Simformer is research code with a pinned, incompatible-with-torch stack, so
this arm runs in its own environment (python 3.10):

    uv venv --python 3.10 sfenv
    uv pip install --python sfenv/bin/python "jax==0.4.23" "jaxlib==0.4.23" \
        "dm-haiku==0.0.12" "optax==0.1.9" "numpy<2" "scipy<1.13" \
        "jaxtyping<0.3" ipython matplotlib pandas networkx
    git clone https://github.com/mackelab/simformer
    uv pip install --python sfenv/bin/python --no-deps -e simformer/src/probjax

Data interchange is a .npz produced from the amortix environment (theta,
x_train, x_test, exact reference samples; see BASELINES.md). Point SF_HERE
at the directory holding simformer_data.npz:

    SF_HERE=/path/to/data sfenv/bin/python baseline_simformer.py \
        --steps 75000 --batch 1024 --n_draw 2000 --time_steps 500
"""
import argparse
import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
import haiku as hk
import optax
from functools import partial
from typing import Optional

from probjax.nn.transformers import Transformer
from probjax.nn.helpers import GaussianFourierEmbedding
from probjax.nn.loss_fn import denoising_score_matching_loss
from probjax.distributions.sde import VESDE
from probjax.distributions import Empirical, Independent
from probjax.utils.sdeint import sdeint

HERE = os.environ.get("SF_HERE", ".")

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--n_draw", type=int, default=500)
ap.add_argument("--time_steps", type=int, default=200)
ap.add_argument("--input", type=str, default="raw", choices=["raw", "returns"])
ap.add_argument("--save", type=str, default=f"{HERE}/simformer_result.json")
args = ap.parse_args()

d = np.load(f"{HERE}/simformer_data.npz")
theta, x_train, x_test, exact = (d["theta"], d["x_train"], d["x_test"],
                                 d["exact"])
if args.input == "returns":
    def to_ret(x):
        xl = np.log(np.clip(x, 1e-8, None))
        return np.diff(np.concatenate([np.zeros((len(x), 1)), xl], 1), axis=1)
    x_train, x_test = to_ret(x_train), to_ret(x_test)

joint = np.concatenate([theta, x_train], axis=1).astype(np.float32)
mu_n, sd_n = joint.mean(0), joint.std(0) + 1e-8
joint_z = (joint - mu_n) / sd_n
x_test_z = (x_test - mu_n[2:]) / sd_n[2:]

n, nodes_max = joint_z.shape
data = jnp.asarray(joint_z).reshape(n, nodes_max, 1)
node_ids = jnp.arange(nodes_max)
print(f"[setup] {n} sims, {nodes_max} nodes, input={args.input}", flush=True)

# --- SDE (notebook defaults) -------------------------------------------------
T_min = 1e-2
sigma_min, sigma_max = 1e-3, 15.0
p0 = Independent(Empirical(data), 1)
sde = VESDE(p0, sigma_min=sigma_min, sigma_max=sigma_max)


def output_scale_fn(t, x):
    scale = jnp.clip(sde.marginal_stddev(t, jnp.ones_like(x)), 1e-2, None)
    return (1 / scale * x).reshape(x.shape)


dim_value, dim_id, dim_condition = 20, 20, 10


def model(t, x, node_ids, condition_mask, edge_mask: Optional[jnp.ndarray] = None):
    batch_size, seq_len, _ = x.shape
    condition_mask = condition_mask.astype(jnp.bool_).reshape(-1, seq_len, 1)
    node_ids = node_ids.reshape(-1, seq_len)
    t = t.reshape(-1, 1, 1)
    time_embeddings = GaussianFourierEmbedding(64)(t)

    embedding_net_value = lambda x: jnp.repeat(x, dim_value, axis=-1)
    embedding_net_id = hk.Embed(nodes_max, dim_id,
                                w_init=hk.initializers.RandomNormal(stddev=3.))
    condition_embedding = hk.get_parameter(
        "condition_embedding", shape=(1, 1, dim_condition),
        init=hk.initializers.RandomNormal(stddev=0.5))
    condition_embedding = condition_embedding * condition_mask
    condition_embedding = jnp.broadcast_to(
        condition_embedding, (batch_size, seq_len, dim_condition))

    value_embeddings = embedding_net_value(x)
    id_embeddings = embedding_net_id(node_ids)
    value_embeddings, id_embeddings = jnp.broadcast_arrays(
        value_embeddings, id_embeddings)
    x_encoded = jnp.concatenate(
        [value_embeddings, id_embeddings, condition_embedding], axis=-1)

    net = Transformer(num_heads=2, num_layers=2, attn_size=10,
                      widening_factor=3)
    h = net(x_encoded, context=time_embeddings, mask=edge_mask)
    out = hk.Linear(1)(h)
    return output_scale_fn(t, out)


key = jrandom.PRNGKey(0)
init, model_fn = hk.without_apply_rng(hk.transform(model))
params = init(key, jnp.ones(args.batch), data[:args.batch], node_ids,
              jnp.zeros_like(node_ids))
n_params = jax.tree_util.tree_reduce(
    lambda a, b: a + b, jax.tree_map(lambda x: x.size, params))
print(f"[model] {n_params:,} params", flush=True)


def weight_fn(t):
    return jnp.clip(sde.diffusion(t, jnp.ones((1, 1, 1))) ** 2, 1e-4)


def loss_fn(params, key):
    rng_time, rng_data, rng_condition, rng_sample = jrandom.split(key, 4)
    times = jrandom.uniform(rng_time, (args.batch, 1, 1),
                            minval=T_min, maxval=1.0)
    idx = jrandom.randint(rng_data, (args.batch,), 0, n)
    batch_xs = data[idx]
    condition_mask = jrandom.bernoulli(
        rng_condition, 0.333, shape=(args.batch, nodes_max))
    all_one = jnp.all(condition_mask, axis=-1, keepdims=True)
    condition_mask = (condition_mask & ~all_one)[..., None]

    loss = denoising_score_matching_loss(
        params, rng_sample, times, batch_xs, condition_mask,
        model_fn=lambda p, t, x: model_fn(p, t, x, node_ids, condition_mask),
        mean_fn=sde.marginal_mean, std_fn=sde.marginal_stddev,
        weight_fn=weight_fn, rebalance_loss=True)
    return loss


optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)


@jax.jit
def update(params, rng, opt_state):
    loss, grads = jax.value_and_grad(loss_fn)(params, rng)
    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state


t0 = time.time()
key = jrandom.PRNGKey(42)
run = 0.0
for i in range(args.steps):
    key, sub = jrandom.split(key)
    loss, params, opt_state = update(params, sub, opt_state)
    run += float(loss)
    if (i + 1) % 2000 == 0:
        print(f"  step {i+1}: loss {run/2000:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        run = 0.0
t_train = time.time() - t0
print(f"[trained in {t_train:.0f}s]", flush=True)

# --- posterior sampling: condition on the 94 x-nodes -------------------------
condition_mask = jnp.concatenate(
    [jnp.zeros(2), jnp.ones(nodes_max - 2)]).astype(jnp.float32)
end_std = jnp.squeeze(sde.marginal_stddev(jnp.ones(1)))
end_mean = jnp.squeeze(sde.marginal_mean(jnp.ones(1)))


def drift_backward(t, x, condition_value):
    score = model_fn(params, t.reshape(-1, 1, 1),
                     x.reshape(-1, nodes_max, 1), node_ids,
                     condition_mask.reshape(1, -1, 1))
    score = score.reshape(x.shape)
    f = sde.drift(t, x) - sde.diffusion(t, x) ** 2 * score
    return f * (1 - condition_mask)


def diffusion_backward(t, x):
    return sde.diffusion(t, x) * (1 - condition_mask)


@jax.jit
def sample_one(key, condition_value):
    key1, key2 = jrandom.split(key)
    x_T = jrandom.normal(key1, (args.n_draw, nodes_max)) * end_std + end_mean
    x_T = x_T * (1 - condition_mask) + condition_value * condition_mask
    keys = jrandom.split(key2, args.n_draw)
    ys = jax.vmap(
        lambda k, x0: sdeint(
            k, lambda t, x: drift_backward(t, x, condition_value),
            diffusion_backward, x0,
            jnp.linspace(1.0, T_min, args.time_steps),
            noise_type="diagonal"))(keys, x_T)
    return ys[:, -1, :2]


t0 = time.time()
B = x_test_z.shape[0]
samples = np.empty((B, args.n_draw, 2), dtype=np.float64)
for i in range(B):
    cv = jnp.concatenate([jnp.zeros(2), jnp.asarray(x_test_z[i])])
    s = np.asarray(sample_one(jrandom.PRNGKey(100 + i), cv))
    samples[i] = s * sd_n[:2] + mu_n[:2]
    if (i + 1) % 50 == 0:
        print(f"  sampled {i+1}/{B} ({time.time()-t0:.0f}s)", flush=True)
t_inf = time.time() - t0

res = {}
for j, nm in enumerate(["mu", "sigma"]):
    bias = np.empty(B); width = np.empty(B)
    for i in range(B):
        em, es = exact[i, :, j].mean(), exact[i, :, j].std()
        bias[i] = (samples[i, :, j].mean() - em) / max(es, 1e-12)
        width[i] = samples[i, :, j].std() / max(es, 1e-12)
    res[nm] = dict(bias=float(bias.mean()),
                   bias_se=float(bias.std() / np.sqrt(B)),
                   width=float(width.mean()))
    print(f"  {nm}: bias {res[nm]['bias']:+.3f}±{res[nm]['bias_se']:.3f} "
          f"width {res[nm]['width']:.3f}", flush=True)

out = dict(params=int(n_params), train_s=t_train,
           inf_ms_per_ds=1e3 * t_inf / B, steps=args.steps,
           batch=args.batch, n_draw=args.n_draw,
           time_steps=args.time_steps, input=args.input, **res)
with open(args.save, "w") as f:
    json.dump(out, f, indent=1)
print(f"[saved] {args.save}")
print("DONE_SIMFORMER", flush=True)
