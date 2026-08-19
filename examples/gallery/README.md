# Gallery

Small, self-contained scripts; each runs from the repo root and prints what it
demonstrates. Budgets are deliberately tiny -- these show the API, not the
paper's numbers (the technical report's tables use ~40x these budgets).

| script | shows | runtime |
|---|---|---|
| `01_quickstart_gbm.py` | train + sample + check against an exact posterior | ~2 min CPU |
| `02_any_design_pk.py` | one network, any number of observation points | minutes (GPU), ~15 min CPU |
| `03_exact_reference_cir.py` | frozen evaluation sets with validated MCMC references and resolution floors | ~10 min CPU |
| `04_custom_problem.py` | add your own system in ~25 lines | ~2 min CPU |

The other scripts in `examples/` are the experiment drivers behind the
technical report, kept for reproducibility.
