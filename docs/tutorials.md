# Tutorials

`ticktack 2.0` ships with two complementary suites of self-contained, pre-executed minimal-working-example (MWE) Jupyter notebooks under [`notebooks/v2/`](https://github.com/SharmaLlama/ticktack/tree/main/notebooks/v2):

1. **Core `v2` Pillar Walkthroughs (`notebooks/v2/01..05_*.ipynb`)** – end-to-end demonstrations of semiparametric event inference, Fisher information & tempered SMC, causal POD vs log-GP reconstruction, Information Field Theory kernel discovery, and multi-event Miyake reanalysis.
2. **`v1` → `v2` Reproduction & Benchmark Suite (`notebooks/v2/docs_examples/01..06_*.ipynb`)** – direct head-to-head reproductions of every `ticktack 1.x` documentation tutorial (`docs/notebooks/01..05_*.ipynb`) with cold-JIT and warm wall-clock benchmarks, plus direct Bayesian inference of carbon-box transfer coefficients (`06_direct_cbm_inference.ipynb`).

---

## Executed `v2` Walkthroughs

### 1. Semiparametric Inference & Fluence Reporting for the 774 CE Miyake Event
**Notebook:** [`notebooks/v2/01_miyake_774ce_inference.ipynb`](https://github.com/SharmaLlama/ticktack/blob/main/notebooks/v2/01_miyake_774ce_inference.ipynb)

Demonstrates the standard parametric event-inference workflow on the two-hemisphere 774–775 CE compilation (`28` tree-ring series, `313` measurements) using the 22-box `Brehm21` carbon-box model:

- Composes `SteadyProduction + HarmonicSolar + GPProduction + MiyakeSpike` inside `RadiocarbonModel`.
- Analytically marginalizes the global Holocene offset $\Delta_{\text{global}}$, all 28 per-laboratory offsets $\delta_{\text{lab},s}$, the 11-year solar quadrature modes, and 32 Solin–Särkkä GP modes via `LinearMarginalConfig` – eliminating the `ticktack 1.x` four-ring `mean(y[:4])` boundary bias.
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

### 5. Holocene Miyake Event Re-Analysis & Epoch-Local Fluence Scaling
**Notebook:** [`notebooks/v2/05_holocene_miyake_reanalysis.ipynb`](https://github.com/SharmaLlama/ticktack/blob/main/notebooks/v2/05_holocene_miyake_reanalysis.ipynb)

Homogeneously re-analyses all six bundled Holocene Miyake cosmic-ray excursions (`7176 BCE` to `993 CE`) across both hemispheres on 22-box `Brehm21`:

- Audits proleptic astronomical year conversion (`historical_bce` $-Y \mapsto -(Y - 1)$), Austral summer Southern Hemisphere ring windows (`october-march`), and `Brehm21`'s `4.39‰` inter-hemispheric steady-state gradient.
- Marginalizes hierarchical Holocene offsets, 11-year solar quadratures, and Solin–Särkkä GP modes analytically inside `fit_map`, contrasting 1950 CE standard yields (`relative_to="standard"`) against epoch-local ambient yields (`relative_to="ambient"`, $(1 + \Delta_0/1000)^{-1}$) when mid-Holocene atmospheric $\Delta^{14}\text{C}$ exceeded $+75\text{‰}$.
- Runs `fit_ift_field` + `error_scale(..., by_series=True)` to isolate over-dispersed laboratory series across multi-site compilations without MCMC.

---

## `v1` → `v2` Reproduction & Benchmark Suite (`docs_examples`)

The six executed five-cell notebooks in [`notebooks/v2/docs_examples/`](https://github.com/SharmaLlama/ticktack/tree/main/notebooks/v2/docs_examples) reproduce all five `ticktack 1.x` tutorials (`docs/notebooks/01..05_*.ipynb`) on `ticktack 2.0` – adding calibrated diagnostic figures (`plot_d14c_fit` with $(d - m)/\sigma$ pulls, `plot_residual_map`, `plot_data_model_correlation`, and `ChainConsumer` `plot_corner` overlays) – followed by a sixth tutorial on direct carbon-box coefficient inference:

1. **Single-Series 774 CE Fit** ([`01_single_series_774ce.ipynb`](https://github.com/SharmaLlama/ticktack/blob/main/notebooks/v2/docs_examples/01_single_series_774ce.ipynb), replaces `01_Fitting.ipynb`): Fits the 28-ring `Miyake12_Cedar` record (`notebooks/miyake12.csv`, loaded directly via whitespace-delimited `TreeRingSeries.from_csv`) on `Guttler15` with exact frequency-domain solar burn-in, analytical offset + solar quadrature marginalization (`LinearMarginalConfig`), `fit_map`, `NumPyro` `sample_nuts`, and a full `plot_corner` posterior cornerplot.
2. **Multi-Series Two-Hemisphere 774 CE Fit** ([`02_multiseries_774ce_nh_sh.ipynb`](https://github.com/SharmaLlama/ticktack/blob/main/notebooks/v2/docs_examples/02_multiseries_774ce_nh_sh.ipynb), replaces `02_MultiFitter.ipynb`): Jointly fits all 14 `notebooks/775CE/*.csv` NH and SH series on 22-box `Brehm21` (`4.39‰` inter-hemispheric gradient) with exact Cholesky marginalization of all 14 per-laboratory offsets, a 2D `series × calendar year` pull heatmap (`plot_residual_map` with symmetric diverging colourbar), and a full `plot_corner` posterior cornerplot.
3. **Non-Parametric Cosmic-Ray Inverse Reconstruction** ([`03_nonparametric_inverse_reconstruction.ipynb`](https://github.com/SharmaLlama/ticktack/blob/main/notebooks/v2/docs_examples/03_nonparametric_inverse_reconstruction.ipynb), replaces `03_InverseSolver.ipynb`): Eliminates `v1` `MC_reconstruct`'s unphysical negative production rates (`-1.73 atoms cm^-2 s^-1`) via closed-form Solin–Särkkä `GPProduction` (`linear_posterior`) and strictly positive IFT `LogGaussianField` + `select_gp_kernel`.
4. **Synthetic Solar-Cycle Injection–Recovery** ([`04_gp_injection_recovery.ipynb`](https://github.com/SharmaLlama/ticktack/blob/main/notebooks/v2/docs_examples/04_gp_injection_recovery.ipynb), replaces `04_Injection_Recovery_ControlPoints.ipynb`): Recovers a high-amplitude 11-year solar sinusoid injected into 30 June–August rings automatically via `select_gp_kernel` (`Matern32`, `Matern52`, `SquaredExponential`, `QuasiPeriodic11yr`) + `laplace_samples`, replacing `117.0 s` finite-difference spline `L-BFGS-B` and `20 min 36 s` `emcee`.
5. **3-Way Background-Marginalization SMC Evidence & Cornerplot** ([`05_flat_baseline_smc_evidence.ipynb`](https://github.com/SharmaLlama/ticktack/blob/main/notebooks/v2/docs_examples/05_flat_baseline_smc_evidence.ipynb), replaces `05_Fitting_Flat.ipynb`): Replaces `v1`'s `adaptive=False, oversample=1008` constant-step ODE hack with clamped Van Loan sub-interval quadrature (`ExactLinearSolver`) and computes thermodynamic Bayes factors $\Delta\ln\widehat{Z}$ via `BlackJAX` `sample_smc` across `Flat Baseline + Spike`, `HarmonicSolar (11-yr) + Spike` (`marginalize_solar=True`), and `Marginalized GP + Spike` (`GaussianField` / `GPProduction` marginalized via `LinearMarginalConfig`), overlaid in a 3-chain `plot_corner` comparison.
6. **Direct Carbon-Box Coefficient Inference & Reservoir Selection** ([`06_direct_cbm_inference.ipynb`](https://github.com/SharmaLlama/ticktack/blob/main/notebooks/v2/docs_examples/06_direct_cbm_inference.ipynb)): Quantifies carbon-cycle parameter identifiability via `cbm_fisher_spectrum`, jointly infers `EffectiveCBM(K=3)` and Kirchhoff flow-balanced `ReversibleCBM` (`Miyake17`) transfer coefficients alongside 774 CE spike parameters (`fit_map` + `sample_nuts` + `plot_corner`), and selects reservoir count $K \in \{1, 2, 3\}$ via `sample_smc`.

---

## Timing & XLA Optimization (`v1` vs `v2`)

Three architectural changes drive the one-to-three order-of-magnitude speedup over `ticktack 1.x`:

- **Exact modal LTI spectral propagation**: Replaces step-by-step `diffrax` ODE integration and per-interval Padé `expm` loops with exact complex modal eigendecomposition (using symmetric detailed-balance `jax.scipy.linalg.eigh` whenever available) and analytical frequency-domain solar steady state, eliminating 2000-year numerical burn-ins and `oversample=1008` step-clipping workarounds.
- **Equinox `@eqx.filter_jit` XLA cache stability**: Partitioning callable `CarbonBoxModel`, `ProductionTerm`, and `ExactLinearSolver` PyTrees cleanly into dynamic JAX array leaves and static topology metadata ensures zero spurious XLA re-tracing across `fit_map`, `sample_nuts`, and `sample_smc` iterations.
- **Closed-form linear nuisance marginalization**: Marginalizing global and hierarchical laboratory offsets, 11-year solar quadrature amplitudes, and Solin–Särkkä GP modes analytically inside `marginal_log_likelihood` collapses 15–45-dimensional random-walk `emcee` chains to 2–6 well-conditioned nonlinear parameters with exact `jax.grad` geometry and vectorized 2-matmul posterior trajectory reconstruction.

| Tutorial | `ticktack 1.x` Workflow & Recorded Wall Time | `ticktack 2.0` Cold (JIT + Run) | `ticktack 2.0` Warm Execution |
| :--- | :--- | :--- | :--- |
| **`01` Single-Series 774 CE** | `SingleFitter` + `emcee` (`119.0 s`) | `9.9 s` (`fit_map`) / `33.0 s` (`sample_nuts`) | `47 ms` (`fit_map`) / `11.3 s` (`sample_nuts`, `10.6×`) |
| **`02` Multi-Series NH + SH** | `MultiFitter` 14× ODE burn-in + `emcee` (`234.0 s`) | `28.3 s` (`fit_map`) / `64.1 s` (`sample_nuts`) | `0.91 s` (`fit_map`) / `24.0 s` (`sample_nuts`, `9.7×`) |
| **`03` Non-Parametric Inverse** | `InverseSolver.MC_reconstruct` (`11.6 s`, $Q_{\min} = -1.73$) | `1.17 s` (`GPProduction`) / `13.9 s` (`LogGaussianField`) | `3.8 ms` (`GPProduction`, `3,000×`) / `13.6 s` ($Q_{\min} = +0.20$) |
| **`04` Solar Injection–Recovery** | `fit_ControlPoints` `L-BFGS-B` (`117.0 s`) + `emcee` (`1236 s`) | `48.1 s` (4-kernel `select_gp_kernel` + draws) | `6.5 ms` (`GPProduction`) / `1.95 s` (`select_gp_kernel`, `60×`–`635×`) |
| **`05` Flat-Baseline + Evidence** | `oversample=1008` `emcee` (`26.9 s`, no $\ln Z$) | `7.6 s` (`sample_smc`, 400 particles) | `0.20 s` (`sample_smc`, `137×`, $\Delta\ln\widehat{Z} = +3.81 \pm 0.11$) |
| **`06` Direct CBM Inference** | Static frozen ODE matrix (not supported in `v1`) | `17.3 s` (Fisher) / `13.5 s` (joint NUTS) / `5.6 s` (`K=3` SMC) | `41 ms` (Fisher) / `3.4 s` (joint NUTS) / `0.05 s` (`K=3` SMC) |

---

## GP Basis Representations & Automatic Kernel Selection

`ticktack 2.0` offers two spectral Gaussian-process representations tailored to complementary physical regimes (contrasted side by side in `docs_examples/03` and `docs_examples/04`):

- **Causal Solin–Särkkä Hilbert-space POD (`GPProduction` / `LogGPProduction`)**: Expands stationary `tinygp` kernels on Dirichlet spectral eigenfunctions propagated causally from exact Smith-doubling stationary pre-history covariance (`project_causal_gp`). Best suited to rapid closed-form Wiener filtering (`linear_posterior` in milliseconds) or semiparametric background nuisance marginalization alongside a parametric `MiyakeSpike`.
- **Reflecting-boundary orthonormal DCT-II IFT (`GaussianField` / `LogGaussianField` + `select_gp_kernel`)**: Represents $f(t)$ on a padded finite interval $[t_{\min} - T_{\mathrm{pad}}, t_{\max} + T_{\mathrm{pad}}]$ with reflecting Neumann boundaries (`ortho` DCT-II), coupling any power-spectral density (`MaternSpectrum`, `SquaredExpSpectrum`, `QuasiPeriodicSpectrum`) to a precomputed exact first-order-hold `DCTResponseOperator`. Because the carbon-box response matrix is built once outside the optimizer, `select_gp_kernel` ranks candidate kernel families via Laplace marginal hyper-evidence $\ln\mathcal{Z}_{\mathrm{hyper}}$ in under a second while auditing stated AMS uncertainties via MacKay's closed-form `error_scale` diagnostic.

---

## Direct Carbon-Box Coefficient Inference & Fisher Identifiability

Decadal tree-ring $\Delta^{14}\text{C}$ observations act as a seasonal-window-averaged Laplace transform of carbon-reservoir residence times. As demonstrated in `v2/02_effective_cbm_fisher_smc.ipynb` and `docs_examples/06_direct_cbm_inference.ipynb`:

- **Exponential singular-value decay (`cbm_fisher_spectrum`)**: Even in 22-box `Brehm21` (56 directed flow-balanced log rates, or 52 under N/S symmetry tying, after projecting out the zero-Fisher global carbon-mass gauge mode), Schur-complement marginalization over cosmic-ray production and laboratory offsets leaves only $d_{\text{eff}} \approx 2\text{–}3$ collective parameter combinations with prior-whitened Fisher eigenvalues $\mu_k > 1$.
- **Exact Kirchhoff mass balance & symmetric `eigh` autodiff**: `EffectiveCBM(K)` parametrizes the identifiable stratosphere–troposphere exchange and $K - 2$ ordered effective sinks directly without redundant null-space directions, while `ReversibleCBM` (`load_preset("Miyake17", kind="reversible", miyake17_closure="closed_shallow")`) parametrizes undirected symmetric exchange fluxes so steady-state Kirchhoff mass conservation holds identically at every `fit_map`, `sample_nuts`, and `sample_smc` evaluation.

---

## Running the Notebooks Locally

Install the development extras and launch Jupyter Lab from the repository root:

```bash
uv pip install -e ".[dev]"
jupyter lab notebooks/v2/
```
