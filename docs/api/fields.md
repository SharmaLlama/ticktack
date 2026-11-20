# IFT Fields & Kernel Selection (`ticktack.fields`)

1D reflecting-boundary (orthonormal DCT-II) Information Field Theory signal fields (`GaussianField`, `LogGaussianField`), composable spectral priors (`MaternSpectrum`, `SquaredExpSpectrum`, `QuasiPeriodicSpectrum`, `CorrelatedFieldSpectrum`, `SumSpectrum`), MacKay marginal evidence (`log_evidence`), laboratory uncertainty diagnostics (`error_scale`, `apply_error_scale`), and automatic empirical-Bayes GP kernel selection (`evidence_grid`, `optimize_gp_kernel`, `select_gp_kernel` across the eight canonical families in `DEFAULT_CANDIDATE_KERNELS`, including `"sho_solar"`). See [Gaussian Processes & Information Field Theory](../gp_and_ift.md).

---

::: ticktack.fields
    options:
      show_root_heading: false

### Joint Spike Profiling in Kernel Optimization (`optimize_spike` & `spike_priors`)

Both `optimize_gp_kernel` and `select_gp_kernel` accept optional keyword arguments matching `fit_ift_field` for profiling a `MiyakeSpike` inside `extra_production` jointly with the marginalized `GaussianField`:

- **`optimize_spike: bool = False`**: When `True` and `extra_production` contains a `MiyakeSpike`, pre-profiles `(t_start, duration, fluence)` on top of a marginalized reference `GaussianField` via `fit_ift_field(..., optimize_spike=True, spike_priors=spike_priors)` before optimizing candidate kernel hyperparameters (while `fluence` continues to be profiled per candidate kernel).
- **`spike_priors: Mapping[str, Any] | None = None`**: Optional mapping of Gaussian `(mean, sd)` prior constraints on spike parameters (`t_start`, `duration`, `fluence`) passed through to the variable-projection spike profiler.
