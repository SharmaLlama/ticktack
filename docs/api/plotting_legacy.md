# Plotting & Legacy Compatibility

Publication-ready diagnostic figures (`ticktack.plotting`) and explicit opt-in flags/constants for regression comparisons against `ticktack 1.x`.

---

## `ticktack.plotting`

::: ticktack.plotting
    options:
      show_root_heading: false

### Additional Diagnostic Helpers (`plot_cbm_tikz_modes` & `plot_corner`)

- **`plot_cbm_tikz_modes(...)`**: Renders Fisher eigenmodes on the published [Zhang et al. (2022, arXiv:2210.13775)](https://arxiv.org/abs/2210.13775) TikZ reservoir geometries (`Brehm21` / `Buntgen18` 22-box two-hemisphere with Northern Hemisphere `#ffe0b2` on top and Southern Hemisphere `#cceeff` on the bottom, mirrored across the equator so `S`, `T`, `SW`, and `I&DW` face directly across the equatorial interface; `Guttler15` 11-box global; `Miyake17` 4-box; and circular fallback for custom topologies) with standard short labels (`S`, `T`, `SW`, `SB`, `I&DW`, `SLB`, `LLB`, `L`, `So`, `P`, `SS`, `B`, `MS`). Normalized signed node weights $u_b \in [-1, 1]$ (`"mass"` or `"net_rate"` / `"incident_rate"` loadings) map fill colours through a symmetric diverging `RdBu_r` (`vmin = -1, vmax = +1`) colourbar and modulate node fill opacity via `alpha = 0.12 + 0.88 * |u_b|` alongside directed edge opacity.
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
