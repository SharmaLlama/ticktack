# Fisher Information & EffectiveCBMs

Published multi-reservoir carbon box models contain up to 22 boxes and more than 50 exchange rates (`Brehm21` has 22 boxes and 52 flow-balanced rates; `Guttler15` has 11 boxes). Yet tree-ring $\Delta^{14}\text{C}$ observations measure only **one** (or in two-hemisphere compilations, two closely coupled) tropospheric reservoir(s) over a finite window of decades. Attempting to infer all 52 transfer rates jointly with cosmic-ray production $Q(t)$ produces severe posterior ridges and non-identifiability.

`ticktack.fisher`, `EffectiveCBM`, and `sample_smc` turn this structural ill-conditioning into a quantitative, gauge-free model-selection framework.

---

## 1. Why Multi-Box Carbon Models Are Ill-Conditioned

### Control-Theory Identifiability Bound ($2n - 2$)

In single-hemisphere radiocarbon inversion, cosmic-ray $^{14}\text{C}$ enters through a fixed normalized allocation vector $\mathbf{b}$ (stratosphere + troposphere) and tree rings sample a single linear readout $\mathbf{c}^\top \mathbf{y}(t)$ (tropospheric specific activity). The forward LTI transfer function in the Laplace domain $s \in \mathbb{C}$ is

$$
H(s) = \mathbf{c}^\top (s I_n - A)^{-1}\mathbf{b} = \frac{p_{n-1}(s)}{q_n(s)},
$$

a strictly proper rational function of degree at most $n$. Even with **noiseless continuous measurements over $t \in [0, \infty)$**, an arbitrary $n$-box CBM has a continuous similarity gauge group $\{P \in \operatorname{GL}_n(\mathbb{R}) : P\mathbf{b} = \mathbf{b},\;\mathbf{c}^\top P^{-1} = \mathbf{c}^\top\}$ of dimension $(n - 1)^2$ that leaves $H(s)$ — and hence every tropospheric $\Delta^{14}\text{C}(t)$ curve for *every* cosmic-ray input $Q(t)$ — **identically invariant** ([Bellman & Åström 1970](https://doi.org/10.1016/0025-5564(70)90069-X)). Taking into account the steady-state normalization against $R_{\text{ref}}$ (which fixes global total mass $M_{\text{tot}}$ as an exact zero-Fisher gauge mode when $Q_{\text{std}}$ is scaled), at most **$2n - 2$** independent combinations of box parameters (for distinct real relaxation modes: $(n - 1)$ decay rates plus $(n - 1)$ relative amplitudes) are observable in principle.

### Exponential Singular-Value Decay of the Laplace Transform

Even within that $(2n - 2)$-dimensional identifiable subspace, reconstructing the distribution of reservoir relaxation rates $\gamma > 0$ from noisy time-domain $\Delta^{14}\text{C}(t)$ data is a **real Laplace-transform inversion**:

$$
g_{\text{trop}}(\tau) = \int_0^\infty e^{-\gamma \tau}\,\rho(\gamma)\,\mathrm{d}\gamma, \qquad \tau > 0.
$$

Under the logarithmic variable substitution $\tau = e^u$, $\gamma = e^{-v}$, the Laplace operator becomes a translation-invariant convolution on $\mathbb{R}$ whose Mellin-transform singular-value spectrum was derived analytically by [McWhirter & Pike (1978)](https://doi.org/10.1088/0305-4470/11/9/007) (and recently analysed for multi-exponential physical inference by Langker et al. 2026):

$$
\sigma(\omega_{\ln}) = \sqrt{\frac{\pi}{\cosh(\pi \omega_{\ln})}} \;\sim\; \sqrt{2\pi}\,\exp\!\left(-\frac{\pi}{2}|\omega_{\ln}|\right) \quad (|\omega_{\ln}| \to \infty).
$$

Because $\sigma(\omega_{\ln})$ decays **exponentially** with log-timescale frequency $\omega_{\ln}$, resolving $m$ distinct carbon-cycle relaxation modes separated across a finite observational baseline $[t_{\min}, t_{\max}]$ of span $T \sim 30$–$100\text{ yr}$ at AMS signal-to-noise ratio $\text{SNR}$ obeys a logarithmic capacity bound:

$$
d_{\text{eff}} \;\lesssim\; \frac{1}{\pi}\ln\!\left(\frac{T}{\Delta t}\right)\,\ln(\text{SNR}) \;\approx\; 2\text{--}3\text{ modes}.
$$

Physically, decadal tree-ring data around a Miyake spike sharply constrain:

1. **Fast stratosphere–troposphere–biosphere dilution** ($\tau_1 \sim 1$–$3\text{ yr}$), governing the peak amplitude and initial sharp drop of the spike;
2. **Intermediate mixed-layer / thermocline ocean uptake** ($\tau_2 \sim 10$–$30\text{ yr}$), governing the multi-decadal tail; and
3. At high SNR or multi-century baselines, **collective deep-ocean exchange** ($\tau_3 \gtrsim 100\text{ yr}$), whereas fine subdivisions among 22 internal biospheric and abyssal boxes lie deep inside the prior-dominated null space ($\mu_k \ll 1$).

---

## 2. Schur-Marginalized Fisher Information (`ticktack.fisher`)

`ticktack.fisher` computes exact, numerically stable local Fisher information spectra on any `zodiax` PyTree model without ever forming ill-conditioned normal-equation products $J^\top \Sigma_n^{-1} J$ directly.

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

For relaxation-mode analysis in observable pole–residue coordinates $\{(r_k, \gamma_k)\}$ of the impulse response $g_{\text{trop}}(\tau) = \sum_k r_k e^{-\gamma_k \tau}$, `relaxation_fisher_modes` maps the data window directly onto the Laplace eigenmodes of $A$.

```python
import jax.numpy as jnp
import ticktack as tt

dataset = tt.load_bundled_event("775CE")
spike0 = tt.MiyakeSpike(t_start=774.5, duration=0.35, fluence=6.5, profile="boxcar")
psi0 = {"spike": spike0, "offsets": jnp.zeros(dataset.n_series)}
fwd = lambda cbm_v, psi: tt.RadiocarbonModel(
    cbm_v, tt.SteadyProduction() + psi["spike"]
).predict_d14c(dataset, offsets=psi["offsets"])

cbm_brehm = tt.load_preset("Brehm21", kind="balanced")
sd_brehm = jnp.append(jnp.full(cbm_brehm.log_rates_free.size, 0.35), jnp.inf)
spec = tt.cbm_fisher_spectrum(fwd, cbm_brehm, dataset.sigma_flat, psi_nuisance=psi0, prior_cbm_sd=sd_brehm)
print(f"Brehm21 (52 free rates): d_eff = {float(spec.d_eff):.2f}, I = {float(spec.info_bits):.2f} bits")
```

---

## 3. Canonical Minimal Models (`EffectiveCBM(K)`)

Because at most $2K - 2$ carbon-cycle parameters are identifiable from tropospheric $\Delta^{14}\text{C}(t)$, `EffectiveCBM(K)` (`K >= 2`) provides a **gauge-free canonical mammillary topology**: `Stratosphere <-> Troposphere <-> {Sink_1, ..., Sink_{K-2}}` (`box 0` Stratosphere, `box 1` Troposphere, and $K - 2$ reversible biospheric/marine exchange sinks).

### Exact Parameter Counting & Smooth Deep-Sink Calibration

1. **Fixed stratospheric production split ($\eta_{\text{strat}} = 0.70$)**: Fixes $\mathbf{b} = (0.70, 0.30, 0, \dots, 0)^\top$, eliminating the continuous $(k_{0 \to 1}, k_{1 \to 0}, \eta_{\text{strat}})$ three-into-two similarity gauge mode.
2. **`2` stratosphere–troposphere log exchange rates**: `log_k_strat_to_trop` ($\ln k_{0 \to 1}$) and `log_k_trop_to_strat` ($\ln k_{1 \to 0}$).
3. **`2(K - 2)` ordered sink parameters (`K >= 3`)**:
   - $K - 2$ unconstrained log uptake rates $\ln k_{1 \to j}$ (`log_k_trop_to_sinks`).
   - $K - 2$ unconstrained residence-time coordinates (`sink_tau_params`) mapped via cumulative positive `softplus` increments $\ln\tau_j = \ln\tau_{j-1} + \operatorname{softplus}(\delta_j)$ (`cbm.sink_timescales()`), enforcing strict ordering $\tau_1 < \tau_2 < \dots < \tau_{K-2}$ and eliminating all $(K - 2)!$ discrete sink-permutation modes!
4. **Dynamic deep-sink calibration (`calibrate_deep_sink=True`, default)**: Given fixed tropospheric mass $M_{\text{trop}} = 501.5\text{ GtC}$ (`GUTTLER15_TROP_MASS_GTC`) and target reference activity $R_{\text{ref}}$ (`GUTTLER15_R_REF_KG_PER_GTC`), an open sequestration loss rate $k_{\text{deep}} \ge 0$ on the slowest reservoir (`Sink_{K-2}` for $K \ge 3$, or `Troposphere` for $K = 2$) is solved analytically via $C^\infty$ `softplus` barriers so steady-state tropospheric $^{14}\text{C}$ matches pre-industrial $R_{\text{ref}}$ across all PyTree updates and autodiff passes.

The total dynamic parameter count is therefore **$1 + 1 + (K - 2) + (K - 2) = 2K - 2$** (`2` parameters for $K = 2$, `4` for $K = 3$, `6` for $K = 4$; for a single-box $K = 1$ baseline, use a 1-box `OpenCBM` as in [Tutorial 2](tutorials.md)). Inspect the active open loss vector `(0, ..., 0, k_deep)` via `cbm.sink_rates()`, the scalar calibrated loss via `cbm.deep_sink_rate`, and the ordered sink residence times via `cbm.sink_timescales()`.

---

## 4. Tempered Sequential Monte Carlo & Bayesian Evidence (`sample_smc`)

To select the reservoir count $K \in \{1, 2, 3, 4\}$ (or compare competing spike profiles and CBMs) via exact Bayes factors $B_{K, K'} = Z_K / Z_{K'}$ without harmonic-mean or Laplace approximations, `ticktack.sampling.sample_smc` wraps [`BlackJAX`](https://blackjax-devs.github.io/blackjax/)'s **adaptive tempered Sequential Monte Carlo** sampler:

- Constructs a thermodynamic bridge $\pi_\beta(\mathbf{u}) \propto p(\mathbf{u})\,L(\mathbf{u})^{\beta}$ from the normalized unconstrained prior ($\beta_0 = 0$, where $\int p(\mathbf{u})\,\mathrm{d}\mathbf{u} = 1$ exactly) to the full posterior ($\beta_T = 1$), choosing each inverse-temperature increment $\Delta\beta_t$ adaptively so the effective sample size (ESS) equals `target_ess * num_particles` (default `0.75`).
- Rejuvenates particles at every tempering stage using a non-interacting random-walk Metropolis Markov kernel preconditioned by tempered Laplace curvature $( \Lambda_0 + \beta H_{\text{lik}} )^{-1}$ (`proposal_mode="laplace_tempered"`).
- Uses **multinomial resampling** (`blackjax.smc.resampling.multinomial`) alongside genealogy-tracked ancestor indices so the telescoping product of incremental normalizing constants provides an asymptotically unbiased log-evidence $\ln \widehat{Z}$ together with the non-negative **Lee & Whiteley (2018)** single-run standard-error estimator $\widehat{\sigma}_{\ln Z}$ (`SMCResult.log_evidence_err`):

$$
\ln \widehat{Z} = \sum_{t=1}^T \ln\!\left(\frac{1}{N}\sum_{i=1}^N w_t^{(i)}\right).
$$

On the 774 CE compilation (see [Tutorial 2](tutorials.md)), $\ln \widehat{Z}_K$ rises sharply from $K = 1$ to $K = 2$ and $K = 3$, then plateaus or declines for $K \ge 4$ due to the automatic Occam penalty on unconstrained slow-exchange parameters — matching the $d_{\text{eff}} \approx 2$–$3$ prediction of `cbm_fisher_spectrum`.
