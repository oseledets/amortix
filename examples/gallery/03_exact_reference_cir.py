"""The full evaluation instrument on Cox-Ingersoll-Ross.

CIR's transition density is a noncentral chi-square, so the package can build
a frozen evaluation set with EXACT-likelihood MCMC references, validated by
two independent chains, and score a trained model against it -- with the
set's own resolution floor reported alongside. ~10 minutes end to end on CPU.

Run:  python examples/gallery/03_exact_reference_cir.py
"""
from amortix.evaluation import build_eval_set, evaluate, model_of_size
from amortix.problems.design_basic import CIRDesign

prob = CIRDesign()
post = model_of_size(prob, "tiny")
post.fit(n_train=8000, steps=3000, batch=256,
         retokenize=prob.make_retokenizer(), verbose=True)

es = build_eval_set(prob, "cir", K=20, n_sets=4, n_chain=20000,
                    seed=11, workers=4)
print(f"\nevaluation set: {es!r}")
r = evaluate(post, es, n_draw=2000)
print(f"median FID {r['fid_median']:.4f} against a floor of "
      f"{r['null_median']:.4f}")
