# amortix — use-case catalog

A curated, cited catalog of real-world problems that fit the `amortix` abstraction:

> **prior over parameters `m`  +  a (possibly expensive / stochastic) forward simulator `ℱ`  →  fast amortized posterior `p(m | data)`.**

The engine is conditional flow matching with a transformer set-encoder (Sherki, Oseledets,
Muravleva, *Bayesian Inverse Problems Meet Flow Matching*, [arXiv:2503.01375](https://arxiv.org/abs/2503.01375)),
which is one instance of the broader **amortized simulation-based inference (SBI)** / **neural
posterior estimation (NPE)** family. The cases below are drawn from that literature to seed the
example gallery and show the method's reach. Each entry lists what is recovered, the simulator,
why amortization wins, the canonical reference, and how it maps onto `amortix`'s `Problem`
contract (simulator type + observation structure). A priority table is at the end.

Legend for *fit*: **ODE / PDE / SDE / ABM** (agent-based / Markov-jump) simulator family;
observation structure = time series / fields / point sets / summary vector.

---

## 1. Gravitational-wave astronomy

### 1.1 Real-time GW parameter estimation — DINGO
- **Domain & task:** Infer the source parameters of a compact-binary merger from LIGO/Virgo strain data.
- **Recovered `m`:** ~15 binary parameters — masses, spins (6D), sky location, distance, inclination, polarization, coalescence time/phase. **Simulator `ℱ`:** waveform model (e.g. IMRPhenom / SEOBNR / surrogate) + detector noise PSD added to the template; the "likelihood" is Gaussian-in-noise but the posterior is highly multimodal and degenerate.
- **Why amortized SBI:** classical Bayesian PE (MCMC / nested sampling) takes hours to days per event; DINGO does it in **~minutes (seconds with importance sampling)** per event, essential for the growing O4/O5 event rate and multi-messenger alerts.
- **Key refs:** Dax, Green, Gair, Macke, Buonanno, Schölkopf, *Real-Time Gravitational-Wave Science with Neural Posterior Estimation*, PRL 127, 241103 (2021), [arXiv:2106.12594](https://arxiv.org/abs/2106.12594); *Neural Importance Sampling for Rapid and Reliable GW Inference*, PRL 130, 171403 (2023). Repo: [github.com/dingo-gw/dingo](https://github.com/dingo-gw). Recent transformer variant: *Flexible Gravitational-Wave Parameter Estimation with Transformers*, [arXiv:2512.02968](https://arxiv.org/abs/2512.02968).
- **Fit to amortix:** other (forward "ODE" = waveform generator); observation = **multi-detector time series / frequency-domain strain**; conditioning on noise PSD per event = exactly the "set of tokens + context" pattern. Flagship demonstration that amortized NPE handles high-dim, repeated, real-time inference.

---

## 2. Computational neuroscience

### 2.1 Mechanistic single-neuron / circuit models — the `sbi` flagship
- **Domain & task:** Find the biophysical parameters of mechanistic neuron / channel / network models consistent with recorded activity.
- **Recovered `m`:** Hodgkin–Huxley conductances and kinetics; ion-channel parameters from voltage-clamp; pyloric-network (STG) conductances; receptive-field params. **Simulator `ℱ`:** the HH ODE system (or stochastic conductance model) integrated to produce a voltage trace, then reduced to summary features (spike count, ISI stats, etc.).
- **Why amortized SBI:** the map params→features is nonlinear and the likelihood over raw traces is intractable; degeneracy means you want the *full posterior*, not a point fit. Amortization lets neuroscientists re-infer across many cells/protocols cheaply.
- **Key refs:** Gonçalves et al., *Training Deep Neural Density Estimators to Identify Mechanistic Models of Neural Dynamics*, eLife 9:e56261 (2020), [elifesciences.org/articles/56261](https://elifesciences.org/articles/56261). Toolkit: `sbi` — Tejero-Cantero et al., JOSS (2020), [arXiv:2007.09114](https://arxiv.org/abs/2007.09114); "sbi reloaded", [arXiv:2411.17337](https://arxiv.org/abs/2411.17337). Repo: [github.com/sbi-dev/sbi](https://github.com/sbi-dev/sbi).
- **Fit to amortix:** **ODE** (optionally SDE with channel noise); observation = **voltage time series → summary-statistic vector** (and/or raw trace via the set-encoder). Canonical "small ODE, many repeated inferences" case — a natural gallery port.

### 2.2 Whole-brain network models of epilepsy (VEP)
- **Domain & task:** Estimate the *epileptogenicity* map of brain regions from intracranial EEG to plan surgery.
- **Recovered `m`:** per-region excitability parameters of a neural-mass (Epileptor) model on a structural connectome. **Simulator `ℱ`:** coupled neural-mass ODE/SDE network producing seizure-like dynamics.
- **Why amortized SBI:** patient-specific, high-dim, must be reusable across patients; MCMC on a whole-brain model is prohibitive.
- **Key ref:** Hashemi et al., *Simulation-Based Inference for Whole-Brain Network Modeling of Epilepsy* (medRxiv 2022; Machine Learning: Science and Technology), [medrxiv 2022.06.02.22275860](https://www.medrxiv.org/content/10.1101/2022.06.02.22275860).
- **Fit to amortix:** **SDE on a graph**; observation = multichannel time-series features. High-dim, patient-amortized.

---

## 3. Cosmology & large-scale structure

### 3.1 Galaxy-clustering cosmology — SimBIG
- **Domain & task:** Constrain cosmological parameters from the 3D galaxy distribution beyond the standard power-spectrum analysis.
- **Recovered `m`:** Ωm, σ8, ns, H0, etc. (+ nuisance/HOD params). **Simulator `ℱ`:** N-body / quijote-style simulations → halo occupation → mock galaxy catalogs → summary statistics (bispectrum, wavelet scattering, CNN compression, skew spectra).
- **Why amortized SBI:** the likelihood of nonlinear, small-scale statistics is unknown; SBI extracts information inaccessible to analytic Gaussian likelihoods, with calibrated coverage.
- **Key refs:** Cranmer, Brehmer, Louppe, *The Frontier of Simulation-Based Inference*, PNAS 117(48):30055 (2020), [arXiv:1911.01429](https://arxiv.org/abs/1911.01429) (the field's review). Hahn et al., *SimBIG: A Forward Modeling Approach…*, PNAS (2023), [arXiv:2211.00723](https://arxiv.org/abs/2211.00723); first non-Gaussian constraints, [arXiv:2310.15246](https://arxiv.org/abs/2310.15246).
- **Fit to amortix:** PDE/N-body (expensive) → **summary-statistic vector / point-cloud of galaxies**; observation = compressed-statistics vector. Showcase for "expensive simulator, amortize once" + coverage diagnostics.

### 3.2 Strong-lensing & dark-matter substructure — swyft / TMNRE
- **Domain & task:** Infer dark-matter subhalo / warm-dark-matter mass from strong-lensing images.
- **Recovered `m`:** WDM particle mass, subhalo population params. **Simulator `ℱ`:** ray-traced lensing image generator.
- **Why amortized SBI:** image likelihood intractable; marginal posteriors on a few params needed cheaply across many lenses.
- **Key ref:** Miller, Cole, Weniger et al., *swyft: Truncated Marginal Neural Ratio Estimation*, JOSS (2022), [joss 10.21105/joss.04205](https://joss.theoj.org/papers/10.21105/joss.04205); repo [github.com/undark-lab/swyft](https://github.com/undark-lab/swyft).
- **Fit to amortix:** PDE/ray-tracing → **image field**; targeted marginal inference.

---

## 4. Epidemiology

### 4.1 Outbreak dynamics — OutbreakFlow (BayesFlow)
- **Domain & task:** Infer epidemic parameters from reported case/death time series during an outbreak.
- **Recovered `m`:** transmission rate, generation time, fraction undetected, reporting delay, intervention effects (SIR/SEIR-type, ~6+ params). **Simulator `ℱ`:** compartmental ODE (or stochastic compartmental model) → noisy observation model for daily counts.
- **Why amortized SBI:** must update *daily* as new data arrive and re-run across regions; amortization gives Bayesian updating without refitting; handles short and long time series.
- **Key refs:** Radev et al., *OutbreakFlow*, PLOS Comput. Biol. 17(10):e1009472 (2021), [journal link](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009472). Toolkit: **BayesFlow** — Radev et al., IEEE TNNLS (2020), [arXiv:2003.06281](https://arxiv.org/abs/2003.06281); JOSS (2023), [arXiv:2306.16015](https://arxiv.org/abs/2306.16015); BayesFlow 2, [arXiv:2602.07098](https://arxiv.org/abs/2602.07098). The disease-dynamics example also appears as a case in [arXiv:2503.01375](https://arxiv.org/abs/2503.01375). Stochastic-compartmental SBI assessment: [arXiv:2512.02528](https://arxiv.org/abs/2512.02528).
- **Fit to amortix:** **ODE (SEIR) or SDE/Markov-jump (stochastic compartmental)**; observation = **multivariate count time series**. Already on the amortix roadmap (`SEIR`); direct port of the 2503.01375 case.

### 4.2 Phylodynamics — epidemiological + phylogenetic NPE
- **Domain & task:** Infer epidemic parameters jointly from case counts and pathogen phylogenies.
- **Recovered `m`:** birth/death/sampling (R0, become-uninfectious rate). **Simulator `ℱ`:** birth–death branching process generating trees.
- **Key ref:** *Simulation-based inference of epidemiological and phylodynamic models via NPE* (bioRxiv 2025), [biorxiv 2025.11.25.690436](https://www.biorxiv.org/content/10.1101/2025.11.25.690436).
- **Fit to amortix:** ABM/branching process → **tree / summary vector**.

---

## 5. Systems biology, PK/PD & bioprocess

### 5.1 Bioprocess kinetic calibration — itaconic-acid fermentation (sister paper)
- **Domain & task:** Calibrate a bioprocess kinetic model from fermentation measurements.
- **Recovered `m`:** Monod-type growth, substrate-uptake and product-formation kinetic constants. **Simulator `ℱ`:** ODE system for biomass / substrate / product (itaconic acid) over the batch.
- **Why amortized SBI:** stochastic, sparse measurements; need re-calibration across batches/strains; the *same CFM recipe as `amortix`* was applied here.
- **Key ref:** companion paper to 2503.01375 — bioprocess (itaconic acid) CFM calibration, [arXiv:2604.22496](https://arxiv.org/abs/2604.22496) *(user-supplied companion; cite per the repo README)*.
- **Fit to amortix:** **ODE**; observation = **multi-species concentration time series**. Already a planned gallery case; the most direct transfer of the engine.

### 5.2 Stochastic nonlinear mixed-effects models (SNLMEM) in systems biology
- **Domain & task:** Population (mixed-effects) inference where each individual is a stochastic dynamical system — e.g. mRNA transfection, gene expression.
- **Recovered `m`:** per-individual + population (hyper)parameters of an SDE mixed-effects model. **Simulator `ℱ`:** SDE per individual with random effects.
- **Why amortized SBI:** intractable per-individual likelihood × many individuals = a prime amortization target; a global posterior surrogate refined per individual.
- **Key ref:** Häggström, Persson, Cvijovic, Picchini, *Simulation-Based Inference for Stochastic Nonlinear Mixed-Effects Models with Applications in Systems Biology*, Statistics and Computing (2026), [arXiv:2504.11279](https://arxiv.org/abs/2504.11279).
- **Fit to amortix:** **SDE (hierarchical)**; observation = **per-individual time series**. Strong fit for the SDE-recovery focus; hierarchical extension is a differentiator.

---

## 6. Population genetics & phylogenetics

### 6.1 Demographic-history inference from allele frequencies — `donni` / NPE
- **Domain & task:** Infer a population's demographic history from the site-frequency spectrum (SFS).
- **Recovered `m`:** ancestral/derived population sizes, split times, migration rates. **Simulator `ℱ`:** coalescent / diffusion model of the expected SFS (dadi / moments / msprime).
- **Why amortized SBI:** ABC is simulation-hungry; trained networks give *instantaneous* parameter inference from any new SFS — pure amortization payoff.
- **Key refs:** `donni` — *Computationally Efficient Demographic History Inference from Allele Frequencies with Supervised ML*, Mol. Biol. Evol. 41(5):msae077 (2024), [doi 10.1093/molbev/msae077](https://academic.oup.com/mbe/article/41/5/msae077/7651223). Review: *Deep Learning in Population Genetics*, GBE 15(2):evad008 (2023). Recent NPE: *Neural Posterior Estimation for Population Genetics* (bioRxiv 2025).
- **Fit to amortix:** ABM/coalescent → **SFS summary vector** (and sequence-based variants → point sets). Clean "summary-vector in, params out" amortization case.

---

## 7. SDE parameter & dynamics recovery  *(amortix's flagship focus)*

For `dX = a(X,m)dt + b(X,m)dW` the transition likelihood is generally **intractable**, so amortized
SBI is the natural tool. Drift and diffusion live on different timescales (drift from the long horizon,
diffusion from quadratic variation of high-frequency increments) — `amortix`'s multi-channel
`PathObserver` is built for exactly this.

### 7.1 Ornstein–Uhlenbeck (validation seed) — done in amortix
- **Recovered `m`:** mean-reversion θ, mean μ, volatility σ. **Simulator `ℱ`:** OU SDE via Euler–Maruyama.
- **Why:** OU has a *closed-form MLE*, so it validates the amortized posterior against a near-optimal classical estimator (see README table); the same code then transfers to SDEs with no tractable likelihood.
- **Fit to amortix:** **SDE**; observation = **multi-resolution path tokens**. ✅ implemented.

### 7.2 Finance: geometric Brownian / CIR / Heston stochastic volatility
- **Domain & task:** Calibrate a stochastic-volatility model from observed prices / option surfaces.
- **Recovered `m`:** drift, mean-reversion κ, long variance θ, vol-of-vol, correlation ρ (+ rate). **Simulator `ℱ`:** Heston / rough-vol SDE integrated to produce price paths or option prices.
- **Why amortized SBI:** likelihood intractable; calibration must run fast and repeatedly (per underlying, intraday) — amortize offline, calibrate online instantly.
- **Key refs:** Barmaz, *SDE Model Calibration through Neural Posterior Estimation* (Heston via NPE/MAF, diffrax simulator), [ybarmaz.github.io blog (2024)](https://ybarmaz.github.io/blog/posts/2024-07-07-SDE-model-calibration-through-NPE.html). Deep calibration of (rough) volatility: Horvath, Muguruza, Tomas, *Deep Learning Volatility*, Quant. Finance (2021), [arXiv:1901.09647](https://arxiv.org/abs/1901.09647).
- **Fit to amortix:** **SDE**; observation = **price time series and/or option-price surface (point set over strike × maturity)**. High-priority gallery target (already on roadmap as GBM / CIR–Heston).

### 7.3 Stochastic predator–prey / double-well / multimodal SDEs
- **Domain & task:** Recover rate/landscape params of stochastic ecological or bistable systems.
- **Recovered `m`:** reaction rates (stochastic Lotka–Volterra) or potential-barrier params (double-well). **Simulator `ℱ`:** Markov-jump SSA (LV) or Langevin SDE (double-well).
- **Why amortized SBI:** LV is the **canonical likelihood-free benchmark** — narrow, hard-to-recover posteriors; double-well stresses multimodality.
- **Key refs:** stochastic Lotka–Volterra as SBI benchmark — see Cranmer et al. review [arXiv:1911.01429](https://arxiv.org/abs/1911.01429) and the `sbi` benchmark suite (Lueckmann et al., *Benchmarking SBI*, AISTATS 2021, [arXiv:2101.04653](https://arxiv.org/abs/2101.04653)).
- **Fit to amortix:** **SDE / Markov-jump**; observation = **population time series** (2D). On roadmap; tests multimodal/2D posteriors.

### 7.4 Nonparametric drift/diffusion discovery — SINDy-for-SDE bridge
- **Domain & task:** Recover the *functional form* (basis coefficients) of drift `a(X)` and diffusion `b(X)` from a trajectory, not just scalar params.
- **Recovered `m`:** coefficients of a sparse basis expansion of drift & diffusion. **Simulator `ℱ`:** the data-generating SDE (and its Kramers–Moyal moments).
- **Why amortized SBI:** turns equation discovery into amortized posterior inference over coefficients — gives uncertainty on the discovered SDE, which sparse-regression (point-estimate) methods lack.
- **Key refs:** Boninsegna, Nüske, Clementi, *Sparse Learning of Stochastic Dynamical Equations*, J. Chem. Phys. 148:241723 (2018), [arXiv:1712.02432](https://arxiv.org/abs/1712.02432); *Data-Driven Discovery of SDEs* (Comm. App. Math. Comp. Sci., 2022); higher-order stochastic-SINDy estimates, [arXiv:2306.17814](https://arxiv.org/abs/2306.17814).
- **Fit to amortix:** **SDE (nonparametric `m`)**; observation = **single/multiple trajectories**. Differentiator vs all existing SBI packages; on amortix roadmap (basis-coefficient recovery).

---

## 8. Physical-science & engineering calibration (broader scan)

### 8.1 Particle physics — LHC effective field theory (MadMiner)
- **Domain & task:** Constrain EFT Wilson coefficients / couplings from LHC events.
- **Recovered `m`:** dimension-six EFT coefficients. **Simulator `ℱ`:** MadGraph + Pythia + detector simulation; the "gold-mining" trick exploits joint likelihood ratio/score from the simulator.
- **Why amortized SBI:** intractable detector-level likelihood; high-dim observables; far stronger bounds than histogram methods.
- **Key refs:** Brehmer, Kling, Espejo, Cranmer, *MadMiner*, Comput. Softw. Big Sci. (2020), [arXiv:1907.10621](https://arxiv.org/abs/1907.10621); *Constraining EFTs with Machine Learning*, PRL/PRD (2018). Repo: [github.com/madminer-tool/madminer](https://github.com/johannbrehmer/madminer).
- **Fit to amortix:** other (event generator) → **per-event observable vectors / point sets**.

### 8.2 Cosmology/astro X-ray spectral fitting & beyond (NPE)
- **Domain & task:** Fit astrophysical spectral models to X-ray data with NPE.
- **Key ref:** *Simulation-Based Inference with NPE applied to X-ray Spectral Fitting*, [arXiv:2401.06061](https://arxiv.org/abs/2401.06061).
- **Fit to amortix:** other (spectral model) → **spectrum vector**.

### 8.3 Climate / weather model calibration
- **Domain & task:** Calibrate subgrid parameterizations (clouds, microphysics) of climate/atmosphere models.
- **Recovered `m`:** parameterization constants (entrainment, autoconversion, etc.). **Simulator `ℱ`:** single-column / GCM run (expensive) or its emulator.
- **Why amortized SBI:** each model run is hours; emulator + amortized posterior gives tractable UQ over parameters.
- **Key refs:** UQ/Bayesian calibration of SCAM6 cloud parameterization, *Frontiers in Climate* (2021), [doi 10.3389/fclim.2021.670740](https://www.frontiersin.org/journals/climate/articles/10.3389/fclim.2021.670740/full); surrogate-based Bayesian calibration comparison, [arXiv:2508.13071](https://arxiv.org/abs/2508.13071). (Closely related to the Calibrate–Emulate–Sample line, Cleary et al., J. Comput. Phys. 2021, [arXiv:2001.03689](https://arxiv.org/abs/2001.03689).)
- **Fit to amortix:** **PDE (emulated)** → **field / summary-statistic vector**.

### 8.4 Battery / electrochemical parameter identification
- **Domain & task:** Identify electrochemical parameters of Li-ion battery models from voltage/impedance.
- **Recovered `m`:** diffusivities, reaction-rate constants, transport params of SPM/SPMe/DFN models. **Simulator `ℱ`:** PyBaMM DFN/SPMe (DAE/PDE) → voltage-vs-time or EIS spectrum.
- **Why amortized SBI:** identifiability is hard, parameters are degenerate, and packs need *per-cell* re-identification at scale.
- **Key refs:** Bayesian DFN parameter ID, [arXiv:2001.09890](https://arxiv.org/abs/2001.09890); PINN-surrogate for P2D parameter inference, [arXiv:2312.17336](https://arxiv.org/abs/2312.17336). Simulator: PyBaMM, [github.com/pybamm-team/PyBaMM](https://github.com/pybamm-team/PyBaMM).
- **Fit to amortix:** **PDE/DAE**; observation = **voltage time series / impedance spectrum**. Good comp-math/engineering gallery fit + classical-comparison harness.

### 8.5 Robotics — sim-to-real system identification
- **Domain & task:** Identify dynamics parameters (mass, friction, contact) to bridge the sim-to-real gap.
- **Recovered `m`:** physical parameters of a rigid-body / contact simulator. **Simulator `ℱ`:** physics engine (MuJoCo / contact dynamics).
- **Why amortized SBI:** real-time, repeated identification on-robot; posterior over plausible dynamics supports robust/adaptive control.
- **Key ref:** *Bridging the Sim-to-Real Gap with Bayesian Inference*, [arXiv:2403.16644](https://arxiv.org/abs/2403.16644); domain-randomization SysID line (Peng et al. 2018).
- **Fit to amortix:** ABM/physics-engine → **trajectory / sensor time series**.

---

## Existing packages & positioning

| package | what it does | flow matching? | transformer / set summary nets? |
|---------|--------------|----------------|--------------------------------|
| **sbi** (mackelab, [github](https://github.com/sbi-dev/sbi), [arXiv:2411.17337](https://arxiv.org/abs/2411.17337)) | PyTorch reference toolkit: NPE/NLE/NRE + sequential variants; large benchmark suite | yes (FMPE added in "sbi reloaded") | embedding nets (CNN/RNN); permutation-invariant nets available |
| **BayesFlow** (Radev et al., [bayesflow.org](https://bayesflow.org), [arXiv:2306.16015](https://arxiv.org/abs/2306.16015)) | amortized Bayesian *workflows*; summary+inference nets; strong diagnostics (SBC) | yes (flow-matching + diffusion backbones in BayesFlow 2) | yes — DeepSet / SetTransformer / time-series summary nets first-class |
| **lampe** (probabilists, [github](https://github.com/probabilists/lampe)) | lightweight PyTorch NPE/NRE; composable normalizing flows (via `zuko`) | partial (flow library focus) | bring-your-own embedding |
| **swyft** (undark-lab, [github](https://github.com/undark-lab/swyft), [JOSS](https://joss.theoj.org/papers/10.21105/joss.04205)) | TMNRE — simulation-efficient *marginal* ratio estimation; coverage tests; used in cosmology | no (ratio-based) | custom embedding nets |
| **sbijax** (Dirmeier, [github](https://github.com/dirmeier/sbijax), [arXiv:2409.19435](https://arxiv.org/abs/2409.19435)) | JAX SBI: NLE/NPE, surjective flows, **consistency-model** posterior estimation | consistency models (flow-matching-adjacent) | JAX-verse nets |

**Where `amortix` sits.** The engine (CFM + transformer set-encoder) overlaps most with BayesFlow's
flow-matching backbone and sbi's FMPE. `amortix` does **not** try to out-engine those general
libraries. Its niche is the **catalog + comparison** angle of the numerical-methods archaeology
program:

1. **A curated comp-math / engineering problem gallery** wired to *real numerical solvers*
   (Euler–Maruyama, `torchsde`, PDE/DAE simulators), each a self-contained `Problem`
   = prior + simulator + observation spec.
2. **A benchmark harness against classical methods** (exact MLE, MCMC, sparse regression) so each
   case ships an honest amortized-vs-classical comparison table (as already done for OU).
3. **Dedicated SDE-recovery tooling** — multi-resolution `PathObserver` that separates drift from
   diffusion, scalar *and* nonparametric (SINDy-for-SDE) drift/diffusion recovery with uncertainty.
   This SDE-first focus is the clearest differentiator: most SBI packages treat the simulator as a
   black box, whereas amortix is built around intractable-likelihood stochastic dynamics.

---

## Summary table — case → simulator type → observation → priority

| # | Use case | Simulator | Observation structure | amortix priority |
|---|----------|-----------|------------------------|:----------------:|
| 7.1 | Ornstein–Uhlenbeck (seed) | SDE | multi-resolution path tokens | **done** |
| 7.2 | GBM / CIR / Heston finance | SDE | price series + option surface (point set) | **high** |
| 5.1 | Itaconic-acid bioprocess | ODE | multi-species conc. time series | **high** |
| 4.1 | SEIR / outbreak dynamics | ODE / SDE-jump | count time series | **high** |
| 7.3 | Stochastic Lotka–Volterra / double-well | SDE / Markov-jump | 2D population series (multimodal) | **high** |
| 7.4 | Nonparametric drift/diffusion (SINDy-SDE) | SDE (functional `m`) | trajectory | **high** |
| 5.2 | Stochastic mixed-effects (systems bio) | SDE (hierarchical) | per-individual series | **high** |
| 2.1 | Mechanistic neuron / HH model | ODE / SDE | voltage series → summary vector | med |
| 8.4 | Li-ion battery (DFN/SPMe) | PDE / DAE | voltage series / impedance | med |
| 8.3 | Climate parameterization | PDE (emulated) | field / summary vector | med |
| 6.1 | Demographic inference (pop. gen.) | ABM / coalescent | SFS summary vector | med |
| 3.1 | Galaxy-clustering cosmology (SimBIG) | N-body / PDE | compressed statistics vector | med |
| 8.5 | Robotics sim-to-real SysID | physics engine / ABM | sensor trajectory | med |
| 1.1 | Gravitational-wave PE (DINGO) | waveform + noise | multi-detector strain series | med (showcase) |
| 8.1 | LHC EFT (MadMiner) | event generator | per-event observables / point set | med |
| 3.2 | Strong-lensing DM (swyft) | ray-tracing PDE | image field | low/showcase |
| 2.2 | Whole-brain epilepsy (VEP) | SDE on graph | multichannel EEG features | low |
| 4.2 | Phylodynamics | branching ABM | phylogenetic tree | low |
| 8.2 | X-ray spectral fitting | spectral model | spectrum vector | low |

*Priority = strategic fit to amortix's SDE-first, comp-math gallery + classical-comparison mission,
not scientific importance. High = direct simulator + tractable to add + reinforces SDE/ODE focus;
med = strong showcase but heavier simulator or summary-stat dependent; low = great citation, harder
or off-core simulator.*
