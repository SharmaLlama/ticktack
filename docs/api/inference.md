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

Evaluates the nominal tropospheric $\Delta^{14}\text{C}$ impulse response $h_{\text{hem},0}(\tau)$ (`baseline`) and per-mode $+1\sigma_{\text{prior}}$ perturbations $\delta h_{\text{hem},k}(\tau)$ (`mode_contributions`) either via forward-mode `jax.jvp` pushforward of `spectrum.parameter_loadings` (`spectrum is not None`) or via direct $\sqrt{\Delta\tau}$ quadrature-weighted Jacobian SVD $J_h L_\Pi = U_h \operatorname{diag}(s_k) V_h^\top$ (`spectrum is None`, using the assumed fiducial default `prior_cbm_sd = 0.35`). Returns a `CBMModeImpulseResult` holding `t_lag`, `hemispheres`, `baseline`, `mode_contributions`, `mode_indices`, `eigenvalues`, and `units`.
