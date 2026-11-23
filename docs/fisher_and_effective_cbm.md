# Fisher Information & EffectiveCBMs

Published multi-reservoir carbon box models contain up to 22 boxes and more than 50 exchange rates (`Brehm21` has 22 boxes and 56 directed flow-balanced rates, or 39 when the 17 equal-rate N/S pairs are tied via `tie_symmetric_edges=True`; `Guttler15` has 11 boxes and 24 directed rates). Yet tree-ring $\Delta^{14}\text{C}$ observations measure only **one** (or in two-hemisphere compilations, two closely coupled) tropospheric reservoir(s) over a finite window of decades. Attempting to infer all 56 transfer rates jointly with cosmic-ray production $Q(t)$ produces severe posterior ridges and ill-conditioning.

`ticktack.fisher`, `EffectiveCBM`, `SubspaceCBM`, and `sample_smc` turn this ill-conditioning into a quantitative model-selection and subspace-inference framework.

---

## 1. Why Multi-Box Carbon Models Are Ill-Conditioned

### Single-Port Identifiability ($2n - 2$) vs Multi-Port Ill-Conditioning

In single-hemisphere (single-port) radiocarbon inversion, cosmic-ray $^{14}\text{C}$ enters through a fixed normalized allocation vector $\mathbf{b}$ (stratosphere + troposphere) and tree rings sample a single linear readout $\mathbf{c}^\top \mathbf{y}(t)$ (tropospheric specific activity). The forward LTI transfer function in the Laplace domain $s \in \mathbb{C}$ is

$$
H(s) = \mathbf{c}^\top (s I_n - A)^{-1}\mathbf{b} = \frac{p_{n-1}(s)}{q_n(s)},
$$

a strictly proper rational function of degree at most $n$. Even with **noiseless continuous measurements over $t \in [0, \infty)$**, an arbitrary $n$-box single-port CBM has a continuous similarity gauge group $\{P \in \operatorname{GL}_n(\mathbb{R}) : P\mathbf{b} = \mathbf{b},\;\mathbf{c}^\top P^{-1} = \mathbf{c}^\top\}$ of dimension $(n - 1)^2$ that leaves $H(s)$ – and hence every tropospheric $\Delta^{14}\text{C}(t)$ curve for *every* cosmic-ray input $Q(t)$ – invariant ([Bellman & Åström 1970](https://doi.org/10.1016/0025-5564(70)90069-X)). Taking into account the steady-state normalization against $R_{\text{ref}}$ (which leaves global total mass $M_{\text{tot}}$ as an exact zero-Fisher gauge mode when $Q_{\text{std}}$ is scaled), at most **$2n - 2$** independent combinations of box parameters are identifiable in principle for a single-port model.

The two main literature presets illustrate the sharp distinction between exact single-port gauge directions and multi-port ill-conditioning:

- **Single-port `Guttler15` (`11` boxes, `24` directed rates)**: Confirmed transfer-function rank **`20`** ($2n - 2 = 20$) with **`4` exact gauge directions** (in addition to global total mass $\ln M_{\text{tot}}$).
- **Two-port untied `Brehm21` (`22` boxes, `56` directed rates)**: Measuring both Northern and Southern Hemisphere tropospheres yields a two-port transfer function whose Jacobian has **full rank `56`**; its singular values fall smoothly to about $10^{-21}$ with no spectral gap, so its `float64` numerical rank is `38`–`50` depending on tolerance. Untied `Brehm21` is therefore **severely ill-conditioned** rather than rank-deficient with exact zero modes (beyond global total mass). Moreover, a full north/south quotient of `Brehm21` is only **approximate**: the published rate matrix is about `6%` asymmetric between hemispheres, with some paired rates differing by `30–60%`.
- **Spectrum realness**: For non-reversible `BalancedCBM` presets (`Brehm21`, `Guttler15`), all eigenvalues of $A$ are real at the nominal published rates, but realness is a property of the nominal rates only — about `23%` of `Brehm21` draws and `0.3%` of `Guttler15` draws from the assumed fiducial `prior_cbm_sd = 0.35` log-rate prior have complex-conjugate eigenvalue pairs. Only detailed-balance models (`ReversibleCBM`, `EffectiveCBM`) guarantee strictly real negative eigenvalues everywhere.

### Exponential Singular-Value Decay of the Laplace Transform

Even where parameters are formally identifiable, reconstructing the distribution of reservoir relaxation rates $\gamma > 0$ from noisy time-domain $\Delta^{14}\text{C}(t)$ data is a **real Laplace-transform inversion**:

$$
g_{\text{trop}}(\tau) = \int_0^\infty e^{-\gamma \tau}\,\rho(\gamma)\,\mathrm{d}\gamma, \qquad \tau > 0.
$$

Under the logarithmic variable substitution $\tau = e^u$, $\gamma = e^{-v}$, the Laplace operator becomes a translation-invariant convolution on $\mathbb{R}$ whose Mellin-transform singular-value spectrum was derived analytically by [McWhirter & Pike (1978)](https://doi.org/10.1088/0305-4470/11/9/007) (and recently analysed for multi-exponential physical inference by [Langker et al. 2026](https://arxiv.org/abs/2610.41092v1)):

$$
\sigma(\omega_{\ln}) = \sqrt{\frac{\pi}{\cosh(\pi \omega_{\ln})}} \;\sim\; \sqrt{2\pi}\,\exp\!\left(-\frac{\pi}{2}|\omega_{\ln}|\right) \quad (|\omega_{\ln}| \to \infty).
$$

Because $\sigma(\omega_{\ln})$ decays **exponentially** with log-timescale frequency $\omega_{\ln}$, resolving $m$ distinct carbon-cycle relaxation modes separated across a finite observational baseline $[t_{\min}, t_{\max}]$ of span $T \sim 30$–$100\text{ yr}$ at AMS signal-to-noise ratio $\text{SNR}$ obeys a logarithmic capacity scaling:

$$
d_{\text{eff}} \;\lesssim\; \frac{1}{\pi}\ln\!\left(\frac{T}{\Delta t}\right)\,\ln(\text{SNR}),
$$

which gives $d_{\text{eff}} \approx 2$–$3$ on decadal tree-ring compilations (with exact numerical values depending on the dataset and assumed log-rate prior width).

Physically, decadal tree-ring data around a Miyake spike constrain:

1. **Fast stratosphere–troposphere–biosphere dilution** ($\tau_1 \sim 1$–$3\text{ yr}$), governing the peak amplitude and initial sharp drop of the spike;
2. **Intermediate mixed-layer / thermocline ocean uptake** ($\tau_2 \sim 10$–$30\text{ yr}$), governing the multi-decadal tail; and
3. At high SNR or multi-century baselines, **collective deep-ocean exchange** ($\tau_3 \gtrsim 100\text{ yr}$), whereas fine subdivisions among 22 internal biospheric and abyssal boxes lie deep inside the prior-dominated tail ($\mu_k \ll 1$).

---

## 2. Schur-Marginalized Fisher Information (`ticktack.fisher`)

`ticktack.fisher` computes numerically stable local Fisher information spectra on any `zodiax` PyTree model without forming ill-conditioned normal-equation products $J^\top \Sigma_n^{-1} J$ directly.

### QR-Stabilized Expected Fisher & Schur Nuisance Marginalization

Given noise-whitened forward Jacobians $\widetilde{J}_\theta = \Sigma_n^{-1/2} \partial_\theta \mathbf{m}$ for carbon-cycle log-rate parameters $\boldsymbol{\theta} \in \mathbb{R}^{d_\theta}$ and $\widetilde{J}_\psi = \Sigma_n^{-1/2} \partial_\psi \mathbf{m}$ for nuisance parameters $\boldsymbol{\psi} \in \mathbb{R}^{d_\psi}$ (spike onset/duration/fluence, solar harmonics, and per-series offsets), `cbm_fisher_spectrum`:

1. **Projects out nuisance degeneracies via thin QR**: With optional Gaussian nuisance prior precision $\Pi_\psi$, forms the augmented nuisance matrix $A_\psi = \begin{bmatrix} \widetilde{J}_\psi \\ \Pi_\psi^{1/2} \end{bmatrix} = Q_\psi R_\psi$ (`jnp.linalg.qr`) and projects the CBM sensitivity columns orthogonal to the nuisance range, so the profile/marginal Fisher information
   $$
   F_{\theta|\psi} = F_{\theta\theta} - F_{\theta\psi}\big(F_{\psi\psi} + \Pi_\psi\big)^{-1} F_{\psi\theta}
   $$
   is obtained directly at $\mathcal{O}(\kappa(\widetilde{J}))$ condition-number accuracy rather than $\mathcal{O}(\kappa(\widetilde{J})^2)$.
2. **Projects gauge zero modes**: Any parameter with infinite prior standard deviation (`prior_cbm_sd = jnp.inf`, such as global total carbon mass $\ln M_{\text{tot}}$ in `BalancedCBM`) is treated as an exact gauge nuisance and projected out before eigendecomposition.
3. **Prior-whitened Generalized Spectrum**: Whitening against the log-rate prior covariance $\Sigma_0 = \operatorname{diag}(\sigma_{0,1}^2, \dots, \sigma_{0,d}^2)$ via $\widetilde{F} = \Sigma_0^{1/2} F_{\theta|\psi} \Sigma_0^{1/2} = U \operatorname{diag}(\mu_1, \dots, \mu_d) U^\top$ ($\mu_1 \ge \mu_2 \ge \dots \ge 0$) yields two canonical summary metrics on `FisherSpectrumResult`:
   - **Effective constrained parameter count** ([MacKay 1992](https://doi.org/10.1162/neco.1992.4.3.415); [Spiegelhalter et al. 2002](https://doi.org/10.1111/1467-9868.00353)):
     $$
     d_{\text{eff}} = \sum_{k=1}^d \frac{\mu_k}{1 + \mu_k} \in [0, d],
     $$
     where modes with $\mu_k \gg 1$ are data-dominated ($\mu_k/(1+\mu_k) \to 1$) and modes with $\mu_k \ll 1$ are prior-dominated ($\mu_k/(1+\mu_k) \to 0$).
   - **Shannon information gain** (in bits, relative to the prior):
     $$
     I_{\text{bits}} = \frac{1}{2\ln 2}\sum_{k=1}^d \ln(1 + \mu_k).
     $$

For relaxation-mode analysis in observable pole–residue coordinates $\{(r_k, \gamma_k)\}$ of the impulse response $g_{\text{trop}}(\tau) = \sum_k r_k e^{-\gamma_k \tau}$, `relaxation_fisher_modes` maps the data window directly onto the Laplace eigenmodes of $A$. To inspect which physical reservoirs participate in each Fisher direction, `plot_cbm_tikz_modes` renders the normalized signed reservoir loadings on the published [Zhang et al. (2022, arXiv:2210.13775)](https://arxiv.org/abs/2210.13775) TikZ box geometries using node `alpha` opacity weighting and a symmetric diverging `RdBu_r` colourbar.

To connect reservoir loadings directly to observable time-domain signals, `cbm_mode_impulse_responses` computes the causal tropospheric $\Delta^{14}\text{C}$ impulse response $h_{\text{hem}}(\tau; \theta)$ (`permil_per_q0_yr`, `permil_per_fluence`, or `raw`) and each eigenmode's directional perturbation $\delta h_{\text{hem}, k}(\tau)$ for a $+1\sigma_{\text{prior}}$ step along mode $k$:

- **Fisher-mode pushforward (`spectrum is not None`)**: Evaluates exact forward-mode `jax.jvp` directional derivatives along `spectrum.parameter_loadings[:, k]` across lag $\tau \ge 0$.
- **Direct impulse-response Jacobian SVD (`spectrum is None`)**: Computes the $\sqrt{\Delta\tau}$ quadrature-weighted prior-whitened impulse-response Jacobian $J_h L_\Pi = U_h \operatorname{diag}(s_k) V_h^\top$ directly via singular-value decomposition (excluding `log_total_mass`), returning `singular_values`, `eigenvalues = s_k**2`, `parameter_loadings = L_Pi @ V_h`, and temporal mode contributions.

Rendering these trajectories with `plot_cbm_mode_impulse_responses` across `0–30 yr` with the `0–6.5 yr` post-spike observation window shaded makes the rank structure visually transparent: Modes $1\text{--}3$ peak sharply inside `0–6.5 yr`, Modes $4\text{--}6$ straddle the `3–15 yr` boundary, and slow deep-water directions such as Mode $9$ live primarily in the `> 10 yr` tail.

### Small-Eigenvalue Fisher Modes ($10^{-9} < \mu_k < 1$) and Finite-Dataset Conditioning

Every quoted $d_{\text{eff}}$ and information-gain figure in bits depends directly on both the **tree-ring compilation** (series and ring count) and the **assumed log-rate prior width** (`prior_cbm_sd = 0.35`, an assumed fiducial width with no cited literature source; see [Bundled Datasets](data.md) for the three 774 CE compilations of `313`, `432`, and `531` rings). Evaluating `cbm_fisher_spectrum` for untied `Brehm21` (`22` boxes, `56` directed free rates) on the bundled `775CE` compilation (`load_bundled_event("775CE")`: `28` series, `313` rings, `760.25–781.25 CE`) with fiducial `prior_cbm_sd = 0.35` separates three regimes:

- **Data-dominated modes ($\mu_k > 1$, modes $1\text{--}3$)**: At the MAP-like boxcar spike of Tutorial 2 (`t_start = 774.11 CE`, `duration = 1.175 yr`, `fluence = 6.16`), three modes exceed $\mu_k = 1$ ($\mu_1 = 39.96$, $\mu_2 = 13.9$, $\mu_3 = 6.53$), yielding $d_{\text{eff}} = 3.16$ and $I = 6.40\text{ bits}$ (at the snippet's `t_start = 774.5 CE`, `duration = 0.35 yr`, `fluence = 6.5`, $\mu_1 = 46.71$, $d_{\text{eff}} = 3.18$, and $I = 6.61\text{ bits}$).
- **Small-eigenvalue physical modes ($10^{-9} < \mu_k < 1$, modes $4\text{--}12$ in Tutorial 2)**: Nine further modes lie above $10^{-9}$ (down to $\mu_{12} = 2.52 \times 10^{-8}$) before dropping to numerical precision ($\mu_{13} \approx 3.2 \times 10^{-29}$). That drop is a finite-sample consequence of the discrete post-spike NH/SH seasonal bins and nuisance projection on this 28-series, 313-ring compilation — not a set of exact structural zero modes of untied `Brehm21`, whose continuous two-port transfer function has full rank `56` and is severely ill-conditioned. Although prior-dominated at `prior_cbm_sd = 0.35`, modes $4\text{--}12$ capture surface-ocean, litter, biospheric, and inter-hemispheric directions that shift inferred spike duration and fluence under broader rate priors.

```python
import jax.numpy as jnp
import ticktack as tt

# Bundled 775CE compilation: 28 series, 313 rings (760.25-781.25 CE)
dataset = tt.load_bundled_event("775CE")
spike0 = tt.MiyakeSpike(t_start=774.5, duration=0.35, fluence=6.5, profile="boxcar")
psi0 = {"spike": spike0, "offsets": jnp.zeros(dataset.n_series)}
fwd = lambda cbm_v, psi: tt.RadiocarbonModel(
    cbm_v, tt.SteadyProduction() + psi["spike"]
).predict_d14c(dataset, offsets=psi["offsets"])

cbm_brehm = tt.load_preset("Brehm21", kind="balanced")
# Assumed fiducial log-rate prior width: prior_cbm_sd = 0.35 (no cited source)
sd_brehm = jnp.append(jnp.full(cbm_brehm.log_rates_free.size, 0.35), jnp.inf)
spec = tt.cbm_fisher_spectrum(fwd, cbm_brehm, dataset.sigma_flat, psi_nuisance=psi0, prior_cbm_sd=sd_brehm)
print(
    f"Brehm21 ({cbm_brehm.log_rates_free.size} free rates) on 775CE "
    f"({dataset.n_series} series, {dataset.n_obs} rings; fiducial prior_cbm_sd=0.35): "
    f"d_eff = {float(spec.d_eff):.2f}, I = {float(spec.info_bits):.2f} bits"
)
```

---

## 3. Canonical Minimal Models (`EffectiveCBM(K)`)

In single-port settings where at most $2K - 2$ carbon-cycle parameters are identifiable from tropospheric $\Delta^{14}\text{C}(t)$, `EffectiveCBM(K)` (`K >= 2`) provides a **gauge-free canonical mammillary (star) topology**: `Stratosphere <-> Troposphere <-> {Sink_1, ..., Sink_{K-2}}` (`box 0` Stratosphere, `box 1` Troposphere, and $K - 2$ reversible biospheric/marine exchange sinks).

An arbitrary multi-box carbon web reduces to a star (mammillary) model in exact arithmetic **only when the web is reversible** (detailed balance). For general non-reversible directed webs such as `Brehm21` or `Guttler15`, a mammillary `EffectiveCBM(K)` is an **approximation** (achieving `0.044‰` RMS in tests against full non-reversible models).

### Parameter Counting & Smooth Deep-Sink Calibration

1. **Fixed stratospheric production split ($\eta_{\text{strat}} = 0.70$)**: Fixes $\mathbf{b} = (0.70, 0.30, 0, \dots, 0)^\top$, eliminating the continuous $(k_{0 \to 1}, k_{1 \to 0}, \eta_{\text{strat}})$ three-into-two similarity gauge mode.
2. **`2` stratosphere–troposphere log exchange rates**: `log_k_strat_to_trop` ($\ln k_{0 \to 1}$) and `log_k_trop_to_strat` ($\ln k_{1 \to 0}$).
3. **`2(K - 2)` ordered sink parameters (`K >= 3`)**:
   - $K - 2$ unconstrained log uptake rates $\ln k_{1 \to j}$ (`log_k_trop_to_sinks`).
   - $K - 2$ unconstrained residence-time coordinates (`sink_tau_params`) mapped via cumulative positive `softplus` increments $\ln\tau_j = \ln\tau_{j-1} + \operatorname{softplus}(\delta_j)$ (`cbm.sink_timescales()`), enforcing strict ordering $\tau_1 < \tau_2 < \dots < \tau_{K-2}$ and eliminating all $(K - 2)!$ discrete sink-permutation modes.
4. **Dynamic deep-sink calibration (`calibrate_deep_sink=True`, default)**: Given fixed tropospheric mass $M_{\text{trop}} = 501.5\text{ GtC}$ (`GUTTLER15_TROP_MASS_GTC`) and target reference activity $R_{\text{ref}}$ (`GUTTLER15_R_REF_KG_PER_GTC`), an open sequestration loss rate $k_{\text{deep}} \ge 0$ on the slowest reservoir (`Sink_{K-2}` for $K \ge 3$, or `Troposphere` for $K = 2$) is solved analytically via $C^\infty$ `softplus` barriers so steady-state tropospheric $^{14}\text{C}$ matches pre-industrial $R_{\text{ref}}$ across all PyTree updates and autodiff passes.

The total dynamic parameter count is therefore **$1 + 1 + (K - 2) + (K - 2) = 2K - 2$** (`2` parameters for $K = 2$, `4` for $K = 3$, `6` for $K = 4$; for a single-box $K = 1$ baseline, use a 1-box `OpenCBM` as in [Tutorial 2](tutorials.md)). Inspect the active open loss vector `(0, ..., 0, k_deep)` via `cbm.sink_rates()`, the scalar calibrated loss via `cbm.deep_sink_rate`, and the ordered sink residence times via `cbm.sink_timescales()`.

---

## 4. Fisher Subspace Parametrization (`SubspaceCBM`)

Rather than discarding carbon-cycle uncertainty outside a minimal star graph, `SubspaceCBM` (aliased as `FisherSubspaceCBM`) restricts a flow-balanced Perron–Frobenius model to its top $m$ Schur-marginalized Fisher log-rate eigenmodes while maintaining Kirchhoff mass balance to machine precision:

$$
\ln \mathbf{k}_{\text{free}}(\boldsymbol{\alpha}) = \ln \mathbf{k}_{\text{free},0} + B\,\boldsymbol{\alpha}, \qquad \boldsymbol{\alpha} \in \mathbb{R}^m.
$$

Constructed via `SubspaceCBM.from_fisher(base_cbm, spectrum, n_modes)`, the static basis matrix $B$ equals `spectrum.parameter_loadings[:p_free, :n_modes]` (where `p_free` is the number of free log rates in `base_cbm`). Only `alpha` is dynamic (`alpha = 0` reproduces `base_cbm` to machine precision, and `to_balanced_cbm()` converts any sample back to a `BalancedCBM`), avoiding poorly conditioned directions in `ParameterSpace` and `cbm_fisher_spectrum`. Because $B = L_{\Pi} V_{:,:m}$, placing an independent standard normal prior `ParamSpec("cbm.alpha", "identity", prior_mean=0.0, prior_sd=1.0)` on `alpha` matches restricting the $\mathcal{N}(\ln \mathbf{k}_0, \Sigma_0)$ prior of `BalancedCBM` to its leading $m$ Schur-marginalized Fisher eigenmodes – letting joint `fit_map` or `sample_nuts` fits include small-eigenvalue modes ($10^{-9} < \mu_k < 1$, e.g. $m = 6$ or $m = 12$) without exploring poorly constrained directions. Full parametric posteriors over `alpha` and spike coordinates from either `sample_nuts` or weighted `sample_smc` ensembles can then be inspected with `plot_corner` (`ChainConsumer >= 1.0`).

**Validation scope**: Synthetic recovery tests in which the synthetic truth is drawn from the same `SubspaceCBM` subspace being fitted verify only in-subspace recovery and numerical conditioning. Validating robustness against model misspecification requires synthetic tests in which the true rate perturbation lies outside the fitted subspace.

---

## 5. Tempered Sequential Monte Carlo & Bayesian Evidence (`sample_smc`)

To select the reservoir count $K \in \{1, 2, 3, 4\}$ (or compare competing spike profiles and CBMs) via Monte Carlo Bayes factors $B_{K, K'} = Z_K / Z_{K'}$ without harmonic-mean or Laplace approximations, `ticktack.sampling.sample_smc` wraps [`BlackJAX`](https://blackjax-devs.github.io/blackjax/)'s **adaptive tempered Sequential Monte Carlo** sampler:

- Constructs a thermodynamic bridge $\pi_\beta(\mathbf{u}) \propto p(\mathbf{u})\,L(\mathbf{u})^{\beta}$ from the normalized unconstrained prior ($\beta_0 = 0$, where $\int p(\mathbf{u})\,\mathrm{d}\mathbf{u} = 1$) to the full posterior ($\beta_T = 1$), choosing each inverse-temperature increment $\Delta\beta_t$ adaptively so the effective sample size (ESS) equals `target_ess * num_particles` (default `0.75`).
- Rejuvenates particles at every tempering stage using a non-interacting random-walk Metropolis Markov kernel preconditioned by tempered Laplace curvature $( \Lambda_0 + \beta H_{\text{lik}} )^{-1}$ (`proposal_mode="laplace_tempered"`).
- Uses **multinomial resampling** (`blackjax.smc.resampling.multinomial`) alongside genealogy-tracked ancestor indices so the telescoping product of incremental normalizing constants provides an asymptotically unbiased log-evidence $\ln \widehat{Z}$ together with the non-negative **[Lee & Whiteley (2018)](https://doi.org/10.1093/biomet/asy028)** single-run standard-error estimator $\widehat{\sigma}_{\ln Z}$ (`SMCResult.log_evidence_err`):

$$
\ln \widehat{Z} = \sum_{t=1}^T \ln\!\left(\frac{1}{N}\sum_{i=1}^N w_t^{(i)}\right).
$$

On the bundled `775CE` compilation (`28` series, `313` rings) under the assumed fiducial log-rate prior width `prior_cbm_sd = 0.35` (see [Tutorial 2](tutorials.md)), $\ln \widehat{Z}_K$ rises sharply to $K = 3$ and then plateaus at $K = 4$ due to the Occam penalty on unconstrained slow-exchange parameters – consistent with `EffectiveCBM` $d_{\text{eff}} = 1.59$ (`I = 3.17 bits`) at $K = 3$ and $d_{\text{eff}} = 1.50$ (`I = 2.90 bits`) at $K = 4$ (`Brehm21`: $d_{\text{eff}} = 3.16$, `I = 6.40 bits`).
