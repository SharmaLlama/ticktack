# Gaussian Processes & Information Field Theory

Cosmic-ray $^{14}\text{C}$ production fluctuates continuously across 11-year Schwabe cycles, centennial Gleissberg / Suess grand solar minima, and geomagnetic dipole excursions, superposed on impulsive solar energetic particle bursts. `ticktack 2.0` provides two complementary, closely unified frameworks for non-parametric and empirical-Bayes cosmic-ray inference — mirroring the Gaussian-process and Information Field Theory architecture of [`virgil`](https://github.com/benjaminpope/virgil):

1. **`ticktack.linear` + `GPProduction` / `LogGPProduction`**: Continuous-time Solin–Särkkä Hilbert-space proper orthogonal decomposition (POD) with stationary pre-history reservoir covariance (`stationary_cbm_gp_cov` via Smith doubling) and state-space Kalman marginalization (`kalman_gp_marginal`).
2. **`ticktack.fields`**: 1D reflecting-boundary (orthonormal DCT-II) Information Field Theory ([Enßlin, Frommert & Kitaura 2009](https://doi.org/10.1103/PhysRevD.80.105005); [Edenhofer et al. 2024](https://doi.org/10.21105/joss.06593)) with `GaussianField` and `LogGaussianField`, composable spectral kernels, MacKay marginal evidence (`log_evidence`), laboratory uncertainty auditing (`error_scale`), and automatic GP kernel selection (`select_gp_kernel`).

---

## 1. The GP $\leftrightarrow$ IFT Dictionary

Information Field Theory and Gaussian-process regression solve the same infinite-dimensional Bayesian inverse problem in dual bases:

| Physical / Statistical Concept | Gaussian-Process View (`ticktack.linear`) | Information Field Theory View (`ticktack.fields`) |
| :--- | :--- | :--- |
| Latent cosmic-ray fluctuation $\eta(t)$ | Zero-mean GP $\eta \sim \mathcal{GP}(0, k_{\boldsymbol{\phi}})$ | Continuous **signal field** |
| Diagonalized spectral prior | Solin–Särkkä Dirichlet Laplacian weights $\lambda_j = S(\sqrt{\nu_j})$ | Reflecting Neumann DCT-II **power spectrum** $S(\omega_k; \boldsymbol{\phi})$ |
| Non-centred latent coordinates $\mathbf{z}$ | Standard normal weights $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I_M)$ | **Standardized excitations** $\eta = C^\top\!\big(\sqrt{\mathbf{s}(\boldsymbol{\phi})} \odot \mathbf{z}\big)$ |
| Forward carbon-box transport | Design matrix $H = \mathcal{B}_{\text{season}}\mathcal{L}_A[\Phi]$ | Linear **response operator** $R$: $\mathbf{d} = R\boldsymbol{\eta} + \mathbf{n}$ |
| Posterior mean $\widehat{\boldsymbol{\eta}}$ | Exact GP posterior mean (`linear_posterior`) | **Wiener filter** $\widehat{\mathbf{z}} = (I + J^\top J)^{-1}J^\top\widetilde{\mathbf{d}}$ |
| Posterior precision $D^{-1}$ | Woodbury / Cholesky posterior precision | **Information propagator** $D = (I + J^\top J)^{-1}$ |
| Marginal log-likelihood $\ln p(\mathbf{d}\mid\boldsymbol{\phi})$ | `marginal_log_likelihood` / `kalman_gp_marginal` | Log **partition function** `log_evidence` (MacKay Type-II ML) |

---

## 2. Causal Solin–Särkkä POD & Pre-History Covariance (`ticktack.linear`)

### Hilbert-Space Spectral Basis (`GPProduction` & `LogGPProduction`)

Following [Solin & Särkkä (2020)](https://doi.org/10.1007/s11222-019-09886-w), any stationary [`tinygp`](https://tinygp.readthedocs.io/) kernel $k(\tau)$ with spectral density $S(\omega) = \int_{-\infty}^\infty k(\tau)e^{-i\omega\tau}\,\mathrm{d}\tau$ on a padded Dirichlet interval $[t_{\text{lo}} - L_{\text{pad}},\, t_{\text{hi}} + L_{\text{pad}}]$ of half-width $L$ admits the uniformly convergent Karhunen–Loève approximation

$$
f(t) = \sum_{j=1}^M \sqrt{S(\omega_j)}\,\phi_j(t)\,z_j, \qquad z_j \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, 1),
$$

where $\phi_j(t) = L^{-1/2}\sin\!\big(\omega_j(t - t_{\text{centre}} + L)\big)$ and $\omega_j = j\pi / (2L)$.

- **`GPProduction(kernel, n_modes=32)`**: Models additive fluctuations $\delta Q(t) = f(t)$ linearly, so every basis function $\phi_j(t)$ is propagated analytically through the complex LTI resolvent (`project_causal_gp`) and marginalized in closed form inside `marginal_log_likelihood` and `linear_posterior` with **zero MCMC dimensions**.
- **`LogGPProduction(kernel, coefficients, exact_basis_variance=True)`**: Enforces strict physical positivity $Q(t) > 0$ across large solar or Miyake excursions via
  $$
  Q(t) = Q_{\text{std}}\,\exp\!\left(f(t) - \frac{1}{2}\sigma_f^2(t)\right),
  $$
  where the point-wise basis variance $\sigma_f^2(t) = \sum_{j=1}^M S(\omega_j)\,\phi_j(t)^2$ provides the exact Itô mean correction $\mathbb{E}[Q(t)] = Q_{\text{std}}$ everywhere on the domain. Fit nonlinearly with `fit_svi` (`AutoMultivariateNormal`) or `sample_nuts` while analytically marginalizing laboratory offsets!
- **Canonical Observable POD (`canonical_pod_modes`)**: Because the carbon-box operator acts as a strong low-pass filter, many high-frequency GP modes produce negligible $\Delta^{14}\text{C}$ variance. Singular-value decomposition of the noise-whitened projected basis $\Sigma_n^{-1/2} H_{\text{GP}}$ compresses $M$ prior modes down to the $r \ll M$ orthogonal observable trajectories that actually reach tree-ring precision.

### Stationary Pre-History Reservoir Covariance via 24-Step Smith Doubling

A subtle physical boundary condition arises at the start $t_0$ of any finite reconstruction window: cosmic-ray production has been fluctuating for millennia *before* $t_0$, so the ocean–biosphere–atmosphere state perturbation $\delta\mathbf{y}(t_0)$ is not zero — it is drawn from a correlated stationary Gaussian distribution $\mathcal{N}(\mathbf{0}, \Sigma_{y,0})$! Neglecting $\Sigma_{y,0}$ ("cold-starting" the GP at $t_0$) pinches credible bands artificially at the left edge of the dataset.

For any state-space GP (`tinygp.kernels.quasisep`) with Markov generator $(F_{\text{gp}}, L_{\text{gp}}, Q_c, H_{\text{gp}})$, coupling the GP state $\mathbf{x}(t)$ into the CBM yields the joint linear stochastic differential equation

$$
\mathrm{d}\begin{bmatrix} \delta\mathbf{y} \\ \mathbf{x} \end{bmatrix}
=
\underbrace{\begin{bmatrix} A & \alpha_{\text{unit}}\mathbf{b} H_{\text{gp}} \\ 0 & F_{\text{gp}} \end{bmatrix}}_{A_{\text{aug}}}
\begin{bmatrix} \delta\mathbf{y} \\ \mathbf{x} \end{bmatrix}\mathrm{d}t
+
\begin{bmatrix} 0 \\ L_{\text{gp}} \end{bmatrix}\mathrm{d}\mathbf{W}_t,
$$

whose stationary covariance $P_\infty$ satisfies the continuous-time Lyapunov equation $A_{\text{aug}} P_\infty + P_\infty A_{\text{aug}}^\top + Q_{\text{aug}} = 0$. `stationary_cbm_gp_cov` discretizes $(A_{\text{aug}}, Q_{\text{aug}})$ at $\Delta t_0 = 0.25\text{ yr}$ via Van Loan's $(2d_{\text{aug}}) \times (2d_{\text{aug}})$ matrix exponential and executes **24 squaring iterations of Smith's doubling recurrence**

$$
P_{k+1} = P_k + \Phi_k P_k \Phi_k^\top, \qquad \Phi_{k+1} = \Phi_k^2,
$$

accumulating exact stationary pre-history covariance over $0.25 \times 2^{24} \approx 4.19 \times 10^6\text{ years}$ ($\gg 8267\text{ yr}$ $^{14}\text{C}$ mean life) in 24 tiny matrix multiplications (`include_gp_initial_cov=True` by default on `LinearMarginalConfig`).

For long annual records where $\mathcal{O}(N)$ state-space complexity is preferred over modal bases, `kalman_gp_marginal` integrates the augmented CBM+GP system directly with a stabilized Joseph-form Kalman filter.

---

## 3. Reflecting-Boundary IFT Signal Fields (`ticktack.fields`)

### Why Orthonormal DCT-II Beats Periodic FFTs on Radiocarbon Windows

Standard 1D FFT spectral priors enforce periodic boundary conditions $\eta(t_{\text{start}}) = \eta(t_{\text{end}})$. Whenever a radiocarbon window spans part of a Hallstatt/Suess ramp or ends during a post-Miyake ocean relaxation tail, periodicity creates a step jump across the boundary that rings across the entire reconstructed $Q(t)$ trajectory (Gibbs phenomenon).

`ticktack.fields` diagonalizes the 1D discrete Neumann Laplacian on an $N$-knot uniform grid (`dct_laplacian_eigenvalues`: $\lambda_k = \big[\frac{2}{\Delta t}\sin(\frac{\pi k}{2N})\big]^2$, $k = 0, \dots, N-1$) via the **orthonormal Type-II Discrete Cosine Transform (`dct2` / `idct2`)**, imposing natural **reflecting boundary conditions ($\partial_t\eta = 0$)** with zero wrap-around artefacts and $\mathcal{O}(N \log N)$ complexity.

Given standardized excitations $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I_N)$ and mode variances $\mathbf{s}(\boldsymbol{\phi})$ (`field_spectrum`), the non-centred synthesis map is

$$
\boldsymbol{\eta}(\mathbf{z}; \boldsymbol{\phi}) = \operatorname{IDCT-II}\!\left(\sqrt{\mathbf{s}(\boldsymbol{\phi})} \odot \mathbf{z}\right).
$$

Because stationary covariance on a reflecting grid has slightly inhomogeneous knot-wise marginal variance near the boundaries, `dct_knot_variance` evaluates exact diagonal variances $\sigma_\eta^2(t_i) = \sum_{k=0}^{N-1} C_{ki}^2\,s_k$ in closed form:

- **`GaussianField`**: Linear cosmic-ray production $Q(t) = q_{\text{mean}} + \eta(t)$. Under Gaussian AMS errors, MAP optimization (`fit_ift_field`), posterior sampling (`laplace_samples`), and marginal log-evidence (`log_evidence`) are **closed-form exact**.
- **`LogGaussianField(..., ito_corrected=True)`**: Strictly positive cosmic-ray production
  $$
  Q(t_i) = \max(m_i, m_{\text{floor}})\,\exp\!\left(\eta_i - \frac{1}{2}\sigma_\eta^2(t_i)\right),
  $$
  using `dct_knot_variance` when `ito_corrected=True` so that $\mathbb{E}[Q(t_i)] = q_{\text{mean}}$ identically across every interior and boundary knot!

### Composable Spectral Kernels (`SpectralKernel`)

All spectral kernels inherit from `SpectralKernel` (`zodiax.Base`) and compose additively with `+` (`SumSpectrum`):

| Kernel Class | Spectral Density $S(\omega)$ | Physical Role |
| :--- | :--- | :--- |
| `MaternSpectrum(sigma, length_yr, order, *, learn_order=False)` | $\propto \big(1 + \ell^2 \lambda_j\big)^{-\text{order}}$ (`order` $= 1, 2, 3$ for Matérn-$1/2$, $3/2$, $5/2$, or continuous `order > 0.5`) | Rough (`order=1` OU) to differentiable (`order=2, 3`) aperiodic cosmic-ray continuum |
| `SquaredExpSpectrum(sigma, length_yr)` | $\propto \exp(-\tfrac{1}{2}\ell^2 \lambda_j)$ | Smooth multi-century geomagnetic / Hallstatt baseline modulation |
| `QuasiPeriodicSpectrum(sigma, period_yr, quality)` | $\propto \big[(\lambda_j - \omega_0^2)^2 + \omega_0^2 \lambda_j / Q^2\big]^{-1}$ ($\omega_0 = 2\pi / P_{\text{yr}}$, quality factor $Q$) | 11-year Schwabe (`period_yr=11.0`) or 22-year Hale damped-oscillator solar modulation |
| `CorrelatedFieldSpectrum(sigma, slope, log_deviations, *, n_modes=4, flexibility=0.5)` | Power-law slope plus integrated cosine log-amplitude basis on $\ln\omega$ | Non-parametric power-spectrum slope and deviation learning ([Arras et al. 2022](https://doi.org/10.1038/s41550-021-01548-0)) |
| `SumSpectrum((k1, k2, ...))` | $\sum_m S_m(\omega)$ (`k1 + k2`) | Multi-scale continuum + solar cycle (`MaternSpectrum + QuasiPeriodicSpectrum`) |

---

## 4. MacKay Empirical Bayes, `error_scale`, & Kernel Discovery

### Marginal Log-Evidence (`log_evidence`)

Integrating out both the $N$ standardized DCT excitations $\mathbf{z}$ and the linear baseline offsets over a padded forward response operator `DCTResponseOperator` (`build_dct_response_operator`), the exact Gaussian marginal log-likelihood (or Laplace log-partition function at $\widehat{\mathbf{z}}$ for `LogGaussianField`) is

$$
\ln Z(\boldsymbol{\phi}) = -\mathcal{H}(\widehat{\mathbf{z}}; \boldsymbol{\phi}) - \frac{1}{2}\ln\det\!\big(I_N + J(\widehat{\mathbf{z}})^\top J(\widehat{\mathbf{z}})\big) - \frac{1}{2}\ln\det\!\big(I + U_{\text{off}}^\top U_{\text{off}}\big) - \frac{N_{\text{obs}}}{2}\ln(2\pi) - \sum_{i=1}^{N_{\text{obs}}} \ln\sigma_i,
$$

where $J$ is the offset-whitened Jacobian and the log-determinant penalizes overly flexible kernels automatically (Occam's razor; [MacKay 1992](https://doi.org/10.1162/neco.1992.4.3.415)).

### Auditing Laboratory Error Bars (`error_scale`)

Published AMS uncertainties occasionally omit sample-pretreatment or inter-ring scatter, inflating $\chi^2$ and biasing GP correlation lengths toward spurious short-timescale roughness. At any fitted `IFTFieldFitResult`, projecting orthogonal to per-series baseline offsets ($N_{\text{eff}}$ residual degrees of freedom) and solving MacKay's empirical-Bayes stationary condition

$$
\widehat{s}^2 = \frac{\chi^2(\widehat{s})}{N_{\text{eff}} - \gamma(\widehat{s})}, \qquad \gamma(s) = \operatorname{tr}\!\big(J(s)^\top J(s)\,[I_N + J(s)^\top J(s)]^{-1}\big)
$$

via safeguarded Newton–bisection yields the noise scale `error_scale(fit, dataset)`:

- $\widehat{s} \approx 1.0$ confirms self-consistent laboratory error bars; $\widehat{s} > 1.2$ diagnoses under-reported scatter; $\widehat{s} < 1$ indicates conservative quoted errors.
- Calling `error_scale(fit, dataset, by_series=True)` solves the coupled block fixed point per `TreeRingSeries` (`dict[str, float]`), so outlier laboratories can be identified and rescaled cleanly via `dataset.with_error_scale(scales)` or `apply_error_scale`.

### Grid Scans, Hyperparameter Optimization, & `select_gp_kernel`

- **`evidence_grid(cbm, dataset, sigmas, lengths_yr, ...)`**: Evaluates $\ln Z(\sigma, \ell)$ across a 2D hyperparameter mesh for visualizing multimodal or flat evidence ridges (`plot_evidence_grid`).
- **`optimize_gp_kernel(cbm, dataset, kernel, ...)`**: Maximizes $\ln Z(\mathbf{d} \mid \boldsymbol{\phi}) + \ln p(\boldsymbol{\phi})$ via multi-start BFGS over a precomputed `DCTResponseOperator`, returning `OptimizedKernelResult` with the Laplace hyper-evidence $\ln \mathcal{Z}_{\text{hyper}} = \ln Z(\mathbf{d} \mid \widehat{\boldsymbol{\phi}}) + \ln p(\widehat{\boldsymbol{\phi}}) + \tfrac{d_\phi}{2}\ln(2\pi) - \tfrac{1}{2}\ln\det H_\phi$.
- **`select_gp_kernel(cbm, dataset, ...)`**: Optimizes all seven canonical IFT / GP spectral families (`"matern12"`, `"matern32"`, `"matern52"`, `"squared_exp"`, `"continuous_matern"`, `"matern32_plus_solar"`, `"correlated_field"`) sharing one precomputed operator and ranks them by Laplace hyper-evidence $\ln \mathcal{Z}_{\text{hyper}}$ (`KernelSelectionResult`, formatted via `sel.format_table()`).
