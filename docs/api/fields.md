# IFT Fields & Kernel Selection (`ticktack.fields`)

1D reflecting-boundary (orthonormal DCT-II) Information Field Theory signal fields (`GaussianField`, `LogGaussianField`), composable spectral priors (`MaternSpectrum`, `SquaredExpSpectrum`, `QuasiPeriodicSpectrum`, `CorrelatedFieldSpectrum`, `SumSpectrum`), MAP-centred tangent-linear response operators (`LinearizedDCTOperator`, `build_linearized_dct_operator`), MacKay marginal evidence (`log_evidence`), laboratory uncertainty diagnostics (`error_scale`, `apply_error_scale`), and automatic empirical-Bayes GP kernel selection (`evidence_grid`, `optimize_gp_kernel`, `select_gp_kernel` across the eight canonical families in `DEFAULT_CANDIDATE_KERNELS`, including `"sho_solar"` and joint `optimize_cbm=True` subspace inference). See [Gaussian Processes & Information Field Theory](../gp_and_ift.md).

---

::: ticktack.fields
    options:
      show_root_heading: false

### Joint Spike & `SubspaceCBM` Profiling in Kernel Optimization (`optimize_spike`, `optimize_cbm`)

Both `optimize_gp_kernel` and `select_gp_kernel` accept optional keyword arguments for profiling a `MiyakeSpike` and/or `SubspaceCBM.alpha` jointly with the marginalized `GaussianField`:

- **`optimize_spike: bool = False`**: When `True` and `extra_production` contains a `MiyakeSpike`, pre-profiles `(t_start, duration, fluence)` on top of a marginalized reference `GaussianField` via `fit_ift_field(..., optimize_spike=True, spike_priors=spike_priors)` before optimizing candidate kernel hyperparameters (while `fluence` continues to be profiled per candidate kernel).
- **`spike_priors: Mapping[str, Any] | None = None`**: Optional mapping of Gaussian `(mean, sd)` prior constraints on spike parameters (`t_start`, `duration`, `fluence`) passed through to the variable-projection spike profiler.
- **`optimize_cbm: bool = False`** & **`cbm_prior_sd: float = 1.0`**: When `True` and `cbm` is a `SubspaceCBM` (`n_modes > 0`), jointly optimizes `cbm.alpha` under $\mathcal{N}(\mathbf{0}, \sigma_{\mathrm{cbm}}^2 I_m)$ alongside kernel hyperparameters (and spike parameters) via a shared MAP-centred `LinearizedDCTOperator`, including the `cbm.alpha` prior and Laplace Occam determinant in `log_hyper_evidence` and storing `cbm_opt` and `alpha_opt` on `OptimizedKernelResult`.

### MAP-Centred Tangent-Linear Operator (`LinearizedDCTOperator` & `build_linearized_dct_operator`)

`build_linearized_dct_operator(cbm_ref, dataset, ...)` evaluates `build_dct_response_operator` at a reference `SubspaceCBM` (`alpha_ref`) together with its $m$ forward-mode `jax.jvp` tangent matrices `dw_dalpha`, returning a `LinearizedDCTOperator` accepted by `LinearMarginalConfig(dct_operator=...)` so `marginal_log_likelihood` and `linear_posterior` evaluate $W_{\mathrm{tan}}(\boldsymbol{\alpha}) = W_{\mathrm{ref}} + (\boldsymbol{\alpha} - \boldsymbol{\alpha}_{\mathrm{ref}}) \cdot \partial W / \partial \boldsymbol{\alpha}$ in $\mathcal{O}(m N n_{\mathrm{grid}})$ arithmetic while keeping `SteadyProduction + MiyakeSpike` and steady-state offset ratios exact.
