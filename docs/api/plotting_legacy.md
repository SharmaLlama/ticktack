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

### Additional Diagnostic Helpers (`plot_cbm_tikz_modes` & `plot_corner`)

- **`plot_cbm_tikz_modes(...)`**: Renders Fisher eigenmodes on the published [Zhang et al. (2022, arXiv:2210.13775)](https://arxiv.org/abs/2210.13775) TikZ reservoir geometries (`Brehm21` / `Buntgen18` 22-box two-hemisphere with Northern Hemisphere `#ffe0b2` on top and Southern Hemisphere `#cceeff` on the bottom, mirrored across the equator so `S`, `T`, `SW`, and `I&DW` face directly across the equatorial interface; `Guttler15` 11-box global; `Miyake17` 4-box; and circular fallback for custom topologies) with standard short labels (`S`, `T`, `SW`, `SB`, `I&DW`, `SLB`, `LLB`, `L`, `So`, `P`, `SS`, `B`, `MS`). Normalized signed node weights $u_b \in [-1, 1]$ (`"mass"` or `"net_rate"` / `"incident_rate"` loadings) map fill colours through a symmetric diverging `RdBu_r` (`vmin = -1, vmax = +1`) colourbar and modulate node fill opacity via `alpha = 0.12 + 0.88 * |u_b|` alongside directed edge opacity.
- **`plot_cbm_mode_impulse_responses(...)`**: Plots each Fisher or direct-SVD eigenmode's Northern and Southern Hemisphere impulse-response perturbation $\delta h_{\text{hem},k}(\tau)$ across `t_lag` (default `0–30 yr`), accepting either a precomputed `CBMModeImpulseResult` or a `(cbm, spectrum)` pair, with optional nominal $h_0(\tau)$ baseline reference and shaded observation window (`obs_window_yr=(0.0, 6.5)`):

```python
def plot_cbm_mode_impulse_responses(
    cbm_or_result: Any,
    spectrum: Any | None = None,
    *,
    t_lag: ArrayLike | None = None,
    modes: Sequence[int] = (0, 1, 2, 3, 5, 8),
    obs_window_yr: tuple[float, float] | None = (0.0, 6.5),
    ncols: int = 3,
    show_baseline: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, np.ndarray]
```

  Uses stateless `plt.rc_context()` and labels lag ($\text{yr}$) and $\delta h(\tau)$ ($\text{‰}\,(q_0\,\mathrm{yr})^{-1}$) with scientific-notation $\mu_k$ panel titles matching `plot_cbm_tikz_modes`.
- **`plot_corner(...)`**: Wraps [`ChainConsumer`](https://samreay.github.io/ChainConsumer/) ($\ge 1.0$) to render full parametric cornerplots from sample mappings, ArviZ `InferenceData` NUTS posteriors, weighted `SMCResult` ensembles, 2D arrays, or multi-model chain comparisons (`Mapping` / `Sequence` of chains):

```python
def plot_corner(
    chains_or_samples: Any,
    *,
    param_names: Sequence[str] | Mapping[str, str] | None = None,
    weights: Any | None = None,
    name: str = "Posterior",
    truth: Mapping[str, float] | Sequence[float] | None = None,
    truth_name: str = "MAP",
    colors: Sequence[str] | None = None,
    figsize: tuple[float, float] | None = None,
    summarize: bool = False,
    plot_kwargs: Mapping[str, Any] | None = None,
) -> tuple[Figure, np.ndarray]
```

  Uses Okabe–Ito colours by default, marks optional `truth` values (e.g. MAP estimates), suppresses `chainconsumer` and `matplotlib.font_manager` log warnings, and restores `matplotlib.rcParams` identically on exit.

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
