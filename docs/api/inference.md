# Likelihood, Fitting, Sampling & Fisher

Unconstrained bijector parameter spaces and analytical marginal likelihoods (`ticktack.likelihood`), gradient-based MAP optimization (`ticktack.fitting`), `NumPyro` NUTS / SVI and `BlackJAX` adaptive tempered SMC samplers (`ticktack.sampling`), and Schur-marginalized Fisher information spectra (`ticktack.fisher`). See [Fisher Information & EffectiveCBMs](../fisher_and_effective_cbm.md).

---

## `ticktack.likelihood`

::: ticktack.likelihood
    options:
      show_root_heading: false

---

## `ticktack.fitting`

::: ticktack.fitting
    options:
      show_root_heading: false

---

## `ticktack.sampling`

::: ticktack.sampling
    options:
      show_root_heading: false

---

## `ticktack.fisher`

::: ticktack.fisher
    options:
      show_root_heading: false

### Mode Impulse-Response Contributions (`cbm_mode_impulse_responses` & `CBMModeImpulseResult`)

```python
def cbm_mode_impulse_responses(
    cbm: CarbonBoxModel,
    t_lag: ArrayLike,
    spectrum: FisherSpectrumResult | None = None,
    *,
    modes: Sequence[int] | None = None,
    hemispheres: Sequence[Literal["north", "south"]] = ("north", "south"),
    prior_cbm_sd: float = 0.35,
    q_std: float = DEFAULT_Q_STD,
    alpha_unit: float = ALPHA_ATOMS_CM2_S_TO_KG_YR,
    units: Literal["permil_per_q0_yr", "permil_per_fluence", "raw"] = "permil_per_q0_yr",
) -> CBMModeImpulseResult
```

Evaluates the nominal tropospheric $\Delta^{14}\text{C}$ impulse response $h_{\text{hem},0}(\tau)$ (`baseline`) and per-mode $+1\sigma_{\text{prior}}$ perturbations $\delta h_{\text{hem},k}(\tau)$ (`mode_contributions`) either via exact `jax.jvp` pushforward of `spectrum.parameter_loadings` (`spectrum is not None`) or via direct $\sqrt{\Delta\tau}$ quadrature-weighted Jacobian SVD $J_h L_\Pi = U_h \operatorname{diag}(s_k) V_h^\top$ (`spectrum is None`). Returns a `CBMModeImpulseResult` holding `t_lag`, `hemispheres`, `baseline`, `mode_contributions`, `mode_indices`, `eigenvalues`, and `units`.

### GP-Marginalized Fisher Spectrum (`cbm_gp_fisher_spectrum`)

`cbm_gp_fisher_spectrum(cbm, dataset, gp_or_field, *, spike=None, marginal_config=None, prior_sd=0.35, ...)` constructs the GP- and offset-marginalized data covariance $\Sigma_{\mathrm{eff}} = \operatorname{diag}(\sigma_d^2) + H_{\mathrm{GP}} H_{\mathrm{GP}}^\top + X_{\mathrm{off}} \Sigma_{\mathrm{off}} X_{\mathrm{off}}^\top$ (supporting `SpectralKernel`, `GaussianField`, `GPProduction`, or `None` across `"hierarchical"`, `"per_series"`, `"shared"`, and `"none"` offset modes) and projects out `MiyakeSpike` tangents via Schur complement, returning a `FisherSpectrumResult` directly compatible with `SubspaceCBM.from_fisher`.

### Two-Stage `SubspaceCBM` + GP Sampling & Exact IS Reweighting (`fit_subspace_cbm_gp`, `reweight_exact_marginal`)

- **`fit_subspace_cbm_gp(model, dataset, parameter_space, marginal_config=None, *, sampler="nuts", reweight_exact=True, ...)`**: Runs Stage-1 exact `fit_map`, constructs a MAP-centred `LinearizedDCTOperator` (`LinearMarginalConfig(dct_operator=lin_op)`) and regularized Laplace covariance, runs Laplace-preconditioned `sample_nuts`, `sample_smc`, or Gaussian Laplace draws, and (when `reweight_exact=True`) calls `reweight_exact_marginal` to attach exact log importance weights `log_weight_is` to `idata.sample_stats` alongside `kish_ess_ratio`, returning a `SubspaceGPFitResult`.
- **`reweight_exact_marginal(idata, model_template, dataset, parameter_space, marginal_config_exact, marginal_config_tan, ...)`**: Evaluates $\ln\mathcal{L}_{\mathrm{exact}}(\mathbf{u}_s) - \ln\mathcal{L}_{\mathrm{tan}}(\mathbf{u}_s)$ across posterior draws and returns `(idata_reweighted, diagnostics)` with `kish_ess`, `kish_ess_ratio`, `median_abs_dlnL`, `max_abs_dlnL`, and `log_z_correction`.

---

## `ticktack.multievent`

::: ticktack.multievent
    options:
      show_root_heading: false
