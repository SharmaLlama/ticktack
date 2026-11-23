# Migrating from `ticktack 1.x` to `2.0`

`ticktack 2.0` replaces the stateful `ticktack 1.x` object hierarchy (`Box`, `Flow`, `CarbonBoxModel.compile()`, `SingleFitter`, `MultiFitter`, `emcee`, and `dynesty`) with composable, immutable [`zodiax`](https://github.com/LouisDesdoigts/zodiax) PyTrees, exact matrix-exponential LTI propagation, analytical Gaussian marginalization, and gradient-based inference (`optimistix`, `NumPyro` NUTS/SVI, and `BlackJAX` tempered SMC).

Beyond dramatic speedups – eliminating the 2000-year numerical burn-in on every likelihood evaluation and marginalizing dozens of linear nuisance parameters in closed form – `ticktack 2.0` corrects several subtle physical and dendrochronological bugs discovered during a comprehensive audit of `ticktack 1.x`.

---

## 1. Side-by-Side API Mapping (`v1` $\to$ `v2`)

| Task / Concept | `ticktack 1.x` (Removed) | `ticktack 2.0` Replacement |
| :--- | :--- | :--- |
| Load a published CBM | `cbm = ticktack.load_presaved_model("Brehm21"); cbm.compile()` | `cbm = tt.load_preset("Brehm21")` (immutable PyTree, no `.compile()`) |
| Build a custom CBM | Imperative `Box(...)` + `Flow(...)` + `CarbonBoxModel(...)` | `Topology(...)` + `FixedCBM` / `BalancedCBM` / `ReversibleCBM` / `OpenCBM` or `EffectiveCBM(K=3)` |
| Parametric Miyake + solar forcing | Ad hoc Python function `production(t, *args)` passed to `SingleFitter` | `tt.SteadyProduction() + tt.MiyakeSpike(...) + tt.HarmonicSolar.from_polar(...)` |
| Non-parametric $Q(t)$ inversion | `sf.dc14_fine` control-point splines / `emcee` over annual knots | `tt.GPProduction` + `tt.linear_posterior` (exact POD) or `tt.GaussianField` / `tt.LogGaussianField(..., ito_corrected=True)` (`ticktack.fields`) |
| Load a single CSV series | `SingleFitter(cbm, cbm_model="Brehm21"); sf.load_data("file.csv")` | `tt.TreeRingSeries.from_csv("file.csv", hemisphere="north", calendar="astronomical_ce", growth_season="april-september")` or `tt.load_bundled_series(key)` |
| Load a multi-series event | `MultiFitter` list of `SingleFitter` objects | `tt.Dataset.from_csvs(...)` or `tt.load_bundled_event("775CE")` |
| Forward $\Delta^{14}\text{C}$ simulation | `cbm.run_burn_in(...)` + `cbm.run_production(...)` | `model = tt.RadiocarbonModel(cbm, production); model.predict_d14c(dataset)` or `model.predict_fine(t_grid)` |
| Baseline / lab offset handling | Clamped to first 4 rings `np.mean(d14c[:4])` | Analytical Gaussian marginalization via `tt.LinearMarginalConfig(offset_mode="hierarchical")` |
| Point optimization (MAP) | `scipy.optimize.minimize` finite-difference Nelder–Mead / Powell | `tt.fit_map(model, dataset, param_space, cfg, method="BFGS", map_space="constrained")` (exact `jax.grad`) |
| Posterior MCMC sampling | Gradient-free `sf.MarkovChainSampler` (`emcee`) | `tt.sample_nuts(model, dataset, param_space, cfg)` (`NumPyro` NUTS, returns `arviz.InferenceData`) |
| Bayesian evidence $\ln Z$ | Nested sampling via `dynesty` | `tt.sample_smc(...)` (`BlackJAX` adaptive tempered SMC with Lee–Whiteley SE) or `tt.log_evidence` (MacKay IFT) |

### Minimal Before / After Example

**`ticktack 1.x`:**

```python
import ticktack
from ticktack import fitting

cbm = ticktack.load_presaved_model("Brehm21", production_rate_units="atoms/cm^2/s")
sf = fitting.SingleFitter(cbm, cbm_model="Brehm21")
sf.load_data("miyake12.csv", growth_season="april-september")
sf.compile_production_model(model="simple_sinusoid")
chain = sf.MarkovChainSampler(params, sf.log_joint_likelihood, burnin=500, production=1000)
```

**`ticktack 2.0`:**

```python
import math
import jax
import ticktack as tt

cbm = tt.load_preset("Brehm21")
series = tt.TreeRingSeries.from_csv(
    "notebooks/775CE/Miyake12_Cedar.csv",
    hemisphere="north",
    calendar="astronomical_ce",
    growth_season="april-september",
)
dataset = tt.Dataset([series], name="Miyake12")

production = tt.ProductionSum(
    steady=tt.SteadyProduction(),
    solar=tt.HarmonicSolar.from_polar(amplitudes=0.0, phases=0.0),
    spike=tt.MiyakeSpike(t_start=774.4, duration=0.35, fluence=6.5, profile="boxcar"),
)
model = tt.RadiocarbonModel(cbm, production)

param_space = tt.ParameterSpace(model, [
    tt.ParamSpec("production.spike.t_start", "identity", prior_mean=774.4, prior_sd=0.4),
    tt.ParamSpec("production.spike.duration", "sigmoid", prior_mean=-1.0, prior_sd=1.2, low=0.05, high=2.0),
    tt.ParamSpec("production.spike.fluence", "exp", prior_mean=math.log(6.5), prior_sd=0.5),
])
cfg = tt.LinearMarginalConfig(offset_mode="hierarchical", marginalize_solar=True)
idata = tt.sample_nuts(model, dataset, param_space, cfg, rng_key=jax.random.PRNGKey(0))
```

---

## 2. Physical & Dendrochronological Bugs Corrected in `2.0`

When comparing `ticktack 2.0` inferences against legacy `ticktack 1.x` outputs, five physical differences are intentional bug fixes:

### 1. The `Miyake17` 4-Box Preset Bugs (`14.003242 / 12` Inflation & Missing Deep Ocean)

In `ticktack 1.x`, the bundled `Miyake17.hd5` 4-box model ([Miyake et al. 2017](https://doi.org/10.1073/pnas.1613144114): Stratosphere $112.5\text{ GtC}$, Troposphere $637.5\text{ GtC}$, Marine surface $900\text{ GtC}$, Biosphere $1600\text{ GtC}$) suffered from two compounding errors:

- **Spurious `14.003242 / 12` flux multiplication**: During v1 HDF5 serialization, all $^{12}\text{C}$ exchange fluxes $F_{i \to j}$ were inadvertently multiplied by the isotopic mass ratio $m_{14}/m_{12} = 14.003242 / 12 \approx 1.16694$, inflating every first-order residence rate $k_{i \to j} = F_{i \to j}/M_i$ by **$+16.7\%$**.
- **Closed shallow 4-box steady-state trapping**: Because `Miyake17` omits the $\sim 38\,000\text{ GtC}$ intermediate/deep ocean and sedimentary reservoirs (`total mass = 3250 GtC` vs `~42,000 GtC` in `Guttler15`), treating those 4 shallow boxes as a closed conservative graph forced all steady-state cosmic-ray $^{14}\text{C}$ production to decay solely within $3250\text{ GtC}$. This inflated the steady-state tropospheric $^{14}\text{C}$ inventory $y_{\text{trop,ss}}$ (and hence $R_{\text{ref}}$) by **$\sim 13\times$**, suppressing predicted $\Delta^{14}\text{C}$ spike amplitudes by an order of magnitude unless compensated by ad hoc production rescaling.

In `ticktack 2.0`, `tt.load_preset("Miyake17")` automatically divides out the erroneous `14.003242 / 12` factor and calibrates an open thermocline export sink $s_{\text{MS}} > 0$ on the Marine surface box (`miyake17_closure="open_ocean_sink"`) so steady-state tropospheric $R_{\text{ref}}$ matches `Guttler15`.

### 2. Inter-Hemispheric $\Delta^{14}\text{C}$ Normalization (`4.38‰` Gradient)

`ticktack 1.x` computed $\Delta^{14}\text{C}_c(t) = 1000\,(y_c(t) / y_{c,\text{ss}} - 1)$ per box, forcing both Northern and Southern Hemisphere tropospheres to $0\text{‰}$ in steady state and erasing `Brehm21`'s physical $4.38\text{‰}$ NH–SH Southern Ocean upwelling offset. `ticktack 2.0` normalizes all reservoirs against a single global mass-weighted tropospheric $R_{\text{ref}}$; see [Physical & Calendar Conventions](conventions.md).

### 3. Southern Hemisphere Ring-Dating (`schulman_start` vs `end_year`) & Historical BCE

`ticktack 1.x` applied a single Southern Hemisphere season shift to all Austral series without distinguishing Edmund Schulman's start-year convention (`Büntgen18_TAS01.csv`: 774 CE jump at nominal year `774`) from the growth-completion end-year convention (`Guttler15_Kauri.csv`: 774 CE jump at nominal year `775`), introducing a $1.0\text{-year}$ timing discrepancy between published Southern Hemisphere series. Similarly, negative BCE CSV labels (`-660`, `-5259`, `-5410`, `-7176`) lacked astronomical year-zero (`+1` yr) conversion. Both are enforced explicitly in `ticktack.data`.

### 4. Eliminating the 4-Ring `mean(y[:4])` Baseline Clamp

`ticktack 1.x`'s `SingleFitter` shifted model predictions by matching the sample mean of the first four data points in each series. Noise on those four points – or a rising 11-year solar phase at the start of the window – injected correlated $\sim 1$–$2\text{‰}$ bias into every ring residual and distorted inferred spike fluences. `LinearMarginalConfig(offset_mode="hierarchical")` marginalizes $\Delta_{\text{global}}$ and $\delta_{\text{lab},s}$ over the *entire* time series weighted by exact inverse covariance.

### 5. Decay Constant & Julian-Year Unit Conversion

`ticktack 1.x` defaulted to the Kutschera $5700\text{-yr}$ half-life with a $365.0\text{-day}$ year and rounded Earth surface area $5.11 \times 10^{18}\text{ cm}^2$ (`ALPHA_LEGACY_V1`). `ticktack 2.0` uses the international Cambridge $5730\text{-yr}$ physical half-life (`DECAY_CAMBRIDGE`), Julian year ($365.25\text{ d}$), and CODATA / IUGG constants (`ALPHA_ATOMS_CM2_S_TO_KG_YR`).

---

## 3. Reproducing Legacy `v1` Numbers for Regression Audits

When auditing published `ticktack 1.x` results directly, `ticktack 2.0` exposes explicit opt-in flags and constants so legacy unit choices can be reproduced cleanly without reintroducing `v1` code:

```python
import ticktack as tt
from ticktack.constants import ALPHA_LEGACY_V1, DECAY_KUTSCHERA
from ticktack.presets import LEGACY_V1_C14_OVER_C12_RATIO

# Reproduce legacy v1 Miyake17 (14.003242/12 flux inflation + closed 4-box shallow steady state + 5700-yr half-life)
cbm_v1_miyake = tt.load_preset("Miyake17", legacy_v1_units=True)

# Use legacy v1 conversion factor on RadiocarbonModel
model_v1_units = tt.RadiocarbonModel(
    tt.load_preset("Brehm21", decay_rate=DECAY_KUTSCHERA),
    tt.SteadyProduction(),
    alpha_unit=ALPHA_LEGACY_V1,
)
```

If you also use `astropy.units` in legacy analysis scripts, install the optional extra `pip install "ticktack[legacy]"`.
