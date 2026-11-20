# Tutorials

`ticktack 2.0` ships with four self-contained, pre-executed minimal-working-example (MWE) Jupyter notebooks under [`notebooks/v2/`](https://github.com/SharmaLlama/ticktack/tree/main/notebooks/v2). Each notebook is structured around five focused code cells demonstrating one core pillar of the `ticktack 2.0` inference stack end to end on audited tree-ring $\Delta^{14}\text{C}$ data.

---

## Executed `v2` Walkthroughs

### 1. Semiparametric Inference & Fluence Reporting for the 774 CE Miyake Event
**Notebook:** [`notebooks/v2/01_miyake_774ce_inference.ipynb`](https://github.com/SharmaLlama/ticktack/blob/main/notebooks/v2/01_miyake_774ce_inference.ipynb)

Demonstrates the standard parametric event-inference workflow on the two-hemisphere 774–775 CE compilation (`28` tree-ring series, `313` measurements) using the 22-box `Brehm21` carbon-box model:

- Composes `SteadyProduction + HarmonicSolar + GPProduction + MiyakeSpike` inside `RadiocarbonModel`.
- Analytically marginalizes the global Holocene offset $\Delta_{\text{global}}$, all 28 per-laboratory offsets $\delta_{\text{lab},s}$, the 11-year solar quadrature modes, and 32 Solin–Särkkä GP modes via `LinearMarginalConfig` — eliminating the `ticktack 1.x` four-ring `mean(y[:4])` boundary bias.
- Finds the MAP solution with `fit_map` (`BFGS`), samples the 3D nonlinear spike posterior with `sample_nuts` (`NumPyro`), reconstructs continuous NH and SH $\Delta^{14}\text{C}(t)$ trajectories (`plot_d14c_fit`), displays a full `ChainConsumer` parametric cornerplot (`plot_corner`), and reports cosmic-ray yield in $\text{atoms cm}^{-2}$ as well as both pre-industrial ($q_0 = Q_{\text{std}}$) and epoch-local ($q_0(\Delta_0)$) production-year equivalents.

---

### 2. Carbon-Cycle Fisher Information Spectrum, `SubspaceCBM`, & Tempered SMC Model Selection
**Notebook:** [`notebooks/v2/02_effective_cbm_fisher_smc.ipynb`](https://github.com/SharmaLlama/ticktack/blob/main/notebooks/v2/02_effective_cbm_fisher_smc.ipynb)

Connects the diffusive Laplace-transform ill-conditioning of multi-reservoir carbon box models to gauge-free subspace inference and rigorous Bayesian model selection:

- Uses `cbm_fisher_spectrum` (with Schur-complement marginalization over spike and laboratory-offset nuisances and exact projection of the total-mass gauge zero mode) to contrast the 56 directed flow-balanced log rates of `Brehm21` against minimal canonical `EffectiveCBM(K=2, 3, 4)` models (`plot_fisher_spectrum`), resolving `12` genuine eigenmodes above $\mu_k > 10^{-9}$ prior to the observation-span nullspace cliff at $k > 12$.
- Visualizes both the data-dominated atmospheric/inter-hemispheric directions ($\mu_k > 1$) and the small-eigenvalue surface-ocean, litter, and biospheric modes ($10^{-9} < \mu_k < 1$, modes $4\text{--}12$) via `plot_cbm_tikz_modes` on the [Zhang et al. (2022, arXiv:2210.13775)](https://arxiv.org/abs/2210.13775) two-hemisphere TikZ reservoir geometry with node `alpha` opacity weighting and a symmetric diverging `RdBu_r` colourbar, paired with `plot_cbm_mode_impulse_responses` showing each mode's `0–30 yr` NH/SH impulse-response perturbation $\delta h_k(\tau)$ against the `0–6.5 yr` observation window.
- Fits `SubspaceCBM.from_fisher(cbm_brehm, spec_brehm, n_modes=m)` jointly with `MiyakeSpike` via `fit_map` and `sample_nuts`, rendering the full `ChainConsumer` posterior cornerplot (`plot_corner`) to show how modes beyond $m = 3$ couple to spike duration and fluence.
- Runs `BlackJAX` adaptive tempered Sequential Monte Carlo (`sample_smc`) across reservoir counts $K = 1, 2, 3, 4$ to demonstrate Occam saturation of the thermodynamic log-evidence $\ln \widehat{Z}_K \pm \widehat{\sigma}_{\ln Z}$ at $K = 2$–$3$.

---

### 3. Non-Parametric Cosmic-Ray Production Reconstruction: Causal POD GP vs Log-Gaussian Process
**Notebook:** [`notebooks/v2/03_nonparametric_gp_pod.ipynb`](https://github.com/SharmaLlama/ticktack/blob/main/notebooks/v2/03_nonparametric_gp_pod.ipynb)

Contrasts exact linear Wiener-filter Gaussian-process inversion against positivity-preserving log-Gaussian variational inference on the Solin–Särkkä Hilbert-space spectral basis:

- Propagates a 48-mode Matérn-3/2 `GPProduction` term analytically through the carbon-box operator (`project_causal_gp` + stationary Smith-doubling pre-history covariance) to obtain closed-form posterior curves via `linear_posterior` in milliseconds, illustrating why stationary linear GPs produce negative lower credible bands ($Q < 0$) away from an impulsive Miyake burst.
- Fits a 32-mode `LogGPProduction` ($Q(t) = Q_{\text{std}}\exp[f(t) - \tfrac{1}{2}\sigma_f^2(t)]$) with `fit_svi` (`AutoMultivariateNormal` full-rank guide) while analytically marginalizing hierarchical laboratory offsets inside `joint_log_likelihood`, enforcing strict physical positivity $Q(t) > 0$ across every posterior draw (`plot_production_posterior`).

---

### 4. Information Field Theory & Automatic GP Kernel Discovery for Radiocarbon
**Notebook:** [`notebooks/v2/04_ift_kernel_selection.ipynb`](https://github.com/SharmaLlama/ticktack/blob/main/notebooks/v2/04_ift_kernel_selection.ipynb)

Adapts [`virgil`](https://github.com/benjaminpope/virgil)'s Information Field Theory and MacKay empirical-Bayes framework (`ticktack.fields`) to 1D cosmic-ray production reconstruction and solar-cycle analysis:

- Demonstrates reflecting-boundary (orthonormal DCT-II) non-centred `GaussianField` synthesis across rough (`MaternSpectrum(order=1)`), smooth (`MaternSpectrum(order=2)`), and composite solar-modulated (`MaternSpectrum + QuasiPeriodicSpectrum`) priors driven by identical standardized excitations $z \sim \mathcal{N}(0, I_N)$.
- Evaluates 2D MacKay marginal log-evidence landscapes (`evidence_grid` + `plot_evidence_grid`) on a multi-decade `760–800 CE` 774 CE compilation ($\sim 3.5$ Schwabe cycles, combining `775CE` with the longer repository records) alongside a spike profiled from a rough initial guess (`select_gp_kernel(..., optimize_spike=True)`), and ranks all eight canonical kernel families in `DEFAULT_CANDIDATE_KERNELS` – demonstrating why $\ge 2$ Schwabe cycles are required to select `"sho_solar"` (`QuasiPeriodicSpectrum`) over short-baseline aperiodic `"matern12"` solutions.
- Applies MacKay's closed-form `error_scale` diagnostic (both global and `by_series=True`) to detect understated AMS measurement uncertainties and compares linear Wiener-filter `GaussianField` vs strictly positive Itô-corrected `LogGaussianField(..., ito_corrected=True)` reconstructions (`plot_field_draws`).

---

## Companion Docs-Example Walkthroughs (`notebooks/v2/docs_examples/`)

Six focused companion examples under [`notebooks/v2/docs_examples/`](https://github.com/SharmaLlama/ticktack/tree/main/notebooks/v2/docs_examples) cover single-series and two-hemisphere 774 CE inference, calendar and Southern Hemisphere growth-season auditing, `LogGPProduction` SVI reconstruction, direct `EffectiveCBM(K=3)` joint sampling, and a **3-way background-marginalization SMC Bayes-factor comparison** (`05_flat_baseline_smc_evidence.ipynb`) contrasting `Flat Baseline + Spike`, `HarmonicSolar (11-yr) + Spike` (`marginalize_solar=True`), and `Marginalized GP + Spike` (`GaussianField` / `GPProduction` marginalized via `LinearMarginalConfig`) with a 3-chain `plot_corner` overlay.

---

## Running the Notebooks Locally

Install the development extras and launch Jupyter Lab from the repository root:

```bash
uv pip install -e ".[dev]"
jupyter lab notebooks/v2/
```
