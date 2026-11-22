# ticktack

[![integration](https://github.com/SharmaLlama/ticktack/actions/workflows/tests.yml/badge.svg)](https://github.com/SharmaLlama/ticktack/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/ticktack.svg)](https://badge.fury.io/py/ticktack)
[![DOI](https://zenodo.org/badge/386222197.svg)](https://zenodo.org/badge/latestdoi/386222197)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.05084/status.svg)](https://doi.org/10.21105/joss.05084)

Open-source carbon-box forward modelling and Bayesian inference for tree-ring radiocarbon ($\Delta^{14}\mathrm{C}$) time series in JAX.

## Contributors

[Utkarsh Sharma](https://github.com/SharmaLlama), [Qingyuan Zhang](https://github.com/qingyuanzhang3), [Jordan Dennis](https://github.com/Jordan-Dennis), [Benjamin Pope](https://github.com/benjaminpope)

## Overview (`ticktack 2.0`)

Radiocarbon ($\Delta^{14}\mathrm{C}$) measurements from annually resolved tree rings record past cosmic-ray production events (Miyake events), solar modulation cycles, and carbon-cycle dynamics. Inferring cosmic-ray production rates $Q(t)$ from $\Delta^{14}\mathrm{C}$ data requires propagating $^{14}\mathrm{C}$ through coupled atmospheric, biospheric, and oceanic carbon reservoirs while accounting for seasonal growth windows and inter-laboratory offsets.

`ticktack 2.0` is a ground-up JAX (`float64`) redesign built on [`zodiax`](https://github.com/LouisDesdoigts/zodiax) / [`equinox`](https://github.com/patrick-kidger/equinox):

- **Immutable `zodiax` Carbon Box Models (`ticktack.cbm`, `ticktack.presets`)**: Literature presets (`Guttler15`, `Brehm21`, `Buntgen18`, and unit/closure-corrected `Miyake17`) plus gauge-minimal parametrizations (`FixedCBM`, Perron–Frobenius cycle-space `BalancedCBM`, `OpenCBM`, `ReversibleCBM`, and ordered gauge-free `EffectiveCBM(K)`).
- **Exact LTI & ODE Solvers (`ticktack.solvers`)**: `ExactLinearSolver` using Van Loan block-matrix exponentials, complex solar resolvents, and exact seasonal sub-window integration (`0` time-stepping truncation error), alongside adaptive `DiffraxSolver` (supporting non-autonomous Suess $^{12}\mathrm{C}$ dilution) and optional CUDA-backed `ModaxSolver`.
- **Analytic Linear Marginalization & Spectral GPs (`ticktack.linear`, `ticktack.likelihood`)**: Exact QR-whitened Gaussian marginalization (`LinearMarginal`) over hierarchical global/lab $\Delta^{14}\mathrm{C}$ offsets, `HarmonicSolar` quadrature amplitudes, secular `LinearTrend` slopes, and causal Solin–Särkkä Hilbert-space `GPProduction` modes (with Smith-doubled stationary pre-history covariance and `canonical_pod_modes`), plus strictly positive `LogGPProduction`.
- **Bayesian Inference & Fisher Geometry (`ticktack.fitting`, `ticktack.sampling`, `ticktack.fisher`)**: Multi-start `fit_map` (`optimistix` BFGS / Levenberg–Marquardt, `optax` Adam) with Laplace covariance, `NumPyro` `sample_nuts` (returning `arviz.InferenceData` with `sample_stats` and joint linear reconstructions) and `fit_svi`, `BlackJAX` tempered `sample_smc` with calibrated Lee & Whiteley (2018) $\ln Z$ uncertainties, and Schur-marginalized `cbm_fisher_spectrum`.

## Installation

Requires Python $\ge 3.11$. Install from PyPI:

```bash
pip install ticktack
```

For development (editable install with `pytest`, `ruff`, and notebook tools):

```bash
git clone https://github.com/SharmaLlama/ticktack.git
cd ticktack
pip install -e ".[dev]"
```

## Quickstart

```python
import ticktack as tt

dataset = tt.load_bundled_event("775CE")
cbm = tt.load_preset("Brehm21", kind="fixed")
production = (
    tt.SteadyProduction()
    + tt.MiyakeSpike(t_start=774.5, duration=0.25, fluence=6.5, profile="supergaussian")
    + tt.HarmonicSolar.from_polar(periods=11.0, amplitudes=0.25, phases=0.0, t_ref=775.0)
)
model = tt.RadiocarbonModel(cbm=cbm, production=production)

space = tt.ParameterSpace(
    model,
    [
        tt.ParamSpec("production.spike.t_start", "sigmoid", low=773.5, high=775.8, prior_mean=0.0, prior_sd=1.5),
        tt.ParamSpec("production.spike.duration", "sigmoid", low=0.04, high=1.5, prior_mean=-1.2, prior_sd=1.2),
        tt.ParamSpec("production.spike.fluence", "exp", prior_mean=1.8, prior_sd=0.6),
    ],
)
map_res = tt.fit_map(model, dataset, space)
idata = tt.sample_nuts(map_res.fitted_model, dataset, space, init_map=False)
```

## Tutorials (`notebooks/v2/`)

Worked end-to-end tutorials for `ticktack 2.0` are provided in [`notebooks/v2/`](notebooks/v2/):

1. [`01_miyake_774ce_inference.ipynb`](notebooks/v2/01_miyake_774ce_inference.ipynb) – Multi-laboratory 774 CE Miyake spike inference with exact linear marginalization, MAP + Laplace warm-start, and `NumPyro` NUTS.
2. [`02_effective_cbm_fisher_smc.ipynb`](notebooks/v2/02_effective_cbm_fisher_smc.ipynb) – Carbon-box Fisher eigenspectra (`Brehm21` vs `EffectiveCBM(K)`) and Bayesian model selection via `BlackJAX` tempered SMC.
3. [`03_nonparametric_gp_pod.ipynb`](notebooks/v2/03_nonparametric_gp_pod.ipynb) – Nonparametric cosmic-ray production recovery using Solin–Särkkä GP marginalization, canonical POD modes, and strictly positive `LogGPProduction` SVI.

> **Migrating from `ticktack 1.x`:** Legacy v1 classes (`CarbonBoxModel` mutable builder, `SingleFitter`, `MultiFitter`, `load_presets`, and `emcee` wrappers) have been replaced in `2.0` by `load_preset`, `RadiocarbonModel`, `ParameterSpace`, `fit_map`, `sample_nuts`, `fit_svi`, and `sample_smc`. Pin `pip install "ticktack<2.0"` if running unmodified v1 scripts.

## Citation & Licence

If you use `ticktack` in academic work, please cite [Sharma et al. (2023), JOSS 8(84), 5084](https://doi.org/10.21105/joss.05084). Released under the [MIT Licence](LICENSE).

## Name

The 'little boxes' in [Malvina Reynolds' famous song](https://www.youtube.com/watch?v=2_2lGkEU4Xs) are all made of ticky-tacky, and they all look just the same. Here we provide an open-source toolkit for reproducing and extending carbon box models for radiocarbon analysis, and we expect they will be as interchangeable as Malvina Reynolds' boxes!
