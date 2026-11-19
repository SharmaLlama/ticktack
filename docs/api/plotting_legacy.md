# Plotting & Legacy Compatibility

Publication-ready diagnostic figures (`ticktack.plotting`) and explicit opt-in flags/constants for regression comparisons against `ticktack 1.x`.

---

## `ticktack.plotting`

Every diagnostic helper in `ticktack.plotting` is strictly stateless (never mutating `matplotlib.rcParams` or calling `plt.gca()` / `plt.gcf()`), accepts optional caller-supplied `Axes`, and returns `(fig, axes)`:

- `plot_d14c_fit` – observed tree-ring $\Delta^{14}\text{C}$ windows and continuous posterior curves with optional standardized pull panel $(d - m) / \sigma$.
- `plot_data_model_correlation` – 1-to-1 observed vs model-predicted $\Delta^{14}\text{C}$ calibration scatter with $1\sigma$ error bars, `1:1` identity line, matched square limits, and reduced $\chi^2 / N_{\text{obs}}$ + WRMS pull summary.
- `plot_residual_map` – 2D `series × calendar year` standardized pull (or raw permil residual) heatmap on a symmetric diverging colour scale (`vmin = -vmax`) with distinct neutral-grey `NaN` cells for unobserved years and a labelled colourbar.
- `plot_production_posterior` – continuous cosmic-ray production $Q(t)$ credible bands, spaghetti draws, and optional injected truth overlay.
- `plot_fisher_spectrum` & `plot_cbm_eigenmodes` – prior-whitened carbon-cycle Fisher information scree and leading collective reservoir eigenmodes.
- `plot_field_draws` & `plot_evidence_grid` – Information Field Theory (DCT-II) realization galleries and 2D MacKay log-evidence landscapes (re-exported from `ticktack.fields`).

::: ticktack.plotting
    options:
      show_root_heading: false

---

## Legacy `ticktack 1.x` Compatibility Options

`ticktack 2.0` removes the mutable `ticktack 1.x` `SingleFitter` / `MultiFitter` / `emcee` classes rather than wrapping their buggy physical defaults silently. When reproducing published `ticktack 1.x` numbers for regression audits (see [Migrating from v1 to v2](../migration.md)), use the explicit opt-in parameters and constants below:

| Symbol / Parameter | Location | Purpose |
| :--- | :--- | :--- |
| `load_preset("Miyake17", legacy_v1_units=True)` | `ticktack.presets.load_preset` | Re-enables `ticktack 1.x`'s `14.003242 / 12` flux inflation, closed 4-box shallow steady state (`sink_rates = 0`), and $5700\text{-yr}$ Kutschera decay rate on `Miyake17`. |
| `LEGACY_V1_C14_OVER_C12_RATIO` | `ticktack.presets` | Exact `14.003242 / 12.0` factor serialized into `v1` `Miyake17.hd5`. |
| `ALPHA_LEGACY_V1` | `ticktack.constants` | `ticktack 1.x` production conversion factor ($365\text{-day}$ year and $4\pi R_\oplus^2 = 5.11 \times 10^{18}\text{ cm}^2$). |
| `DECAY_KUTSCHERA` | `ticktack.constants` | $\ln 2 / 5700\;\text{yr}^{-1}$ decay rate used as the default in `ticktack 1.x`. |
| `pip install "ticktack[legacy]"` | `pyproject.toml` optional extra | Installs `astropy` alongside `ticktack` for scripts converting legacy `astropy.units` quantities. |
