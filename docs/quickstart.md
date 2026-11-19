# Quickstart

Every workflow in `ticktack 2.0` composes four immutable building blocks:

1. A **carbon box model** (`BalancedCBM`, `ReversibleCBM`, `OpenCBM`, or `EffectiveCBM`), loaded via `tt.load_preset(...)` or constructed directly.
2. A **cosmic-ray production model** (`SteadyProduction`, `MiyakeSpike`, `HarmonicSolar`, `GPProduction`, `LogGPProduction`, or an Information Field Theory `GaussianField` / `LogGaussianField`).
3. An audited **tree-ring dataset** (`TreeRingSeries` / `Dataset`) carrying explicit calendar and hemispheric growth-season metadata.
4. A **forward or inverse operator** (`RadiocarbonModel`, `fit_map`, `sample_nuts`, `sample_smc`, `fit_svi`, or `select_gp_kernel`).

!!! important "Import `ticktack` before `jax`"
    Importing `ticktack` automatically sets `jax_enable_x64 = True`. Always `import ticktack as tt` at the top of your script or notebook so IEEE 754 `float64` precision is active before any JAX arrays are traced.

---

## 1. Presets, Effective CBMs, & Audited Datasets

Published multi-reservoir topologies (`"Guttler15"`, `"Brehm21"`, `"Miyake17"`, `"Buntgen18"`) and canonical minimal models (`EffectiveCBM`) share the same `CarbonBoxModel` interface:

```python
import jax.numpy as jnp
import ticktack as tt

# 22-box two-hemisphere Brehm et al. (2021) model and 11-box Güttler et al. (2015) model
cbm_brehm = tt.load_preset("Brehm21")
cbm_guttler = tt.load_preset("Guttler15")

# Minimal identifiable K=3 effective CBM (4 identifiable rate parameters, zero gauge redundancy)
cbm_eff = tt.EffectiveCBM(K=3)
print("EffectiveCBM(K=3) sink rates [yr^-1]:", cbm_eff.sink_rates())

# Load the audited 774–775 CE two-hemisphere compilation (or a single series by catalogue key)
dataset = tt.load_bundled_event("775CE")
tas01 = tt.load_bundled_series("775AD-early-S/Büntgen18_TAS01.csv")
print(f"775CE compilation: {dataset.n_series} series, {dataset.n_obs} rings")
print("Sample catalogue keys:", list(tt.BUNDLED_DATASET_CATALOGUE)[:3])
```

When loading your own CSV files with `TreeRingSeries.from_csv` or `Dataset.from_csvs`, `hemisphere` (`"north"` or `"south"`) is mandatory, and cross-New-Year Southern Hemisphere growth seasons (`growth_season="october-march"`) require an explicit `year_convention` (`"schulman_start"` or `"end_year"`) alongside `calendar` (`"astronomical_ce"`, `"cal_bp"`, or `"historical_bce"`) so hemispheric ring-dating shifts can never enter silently; see [Physical & Calendar Conventions](conventions.md).

---

## 2. Forward $\Delta^{14}\text{C}$ Simulation

Combine any `CarbonBoxModel` and `ProductionTerm` inside `RadiocarbonModel` (which defaults to `ExactLinearSolver`). Continuous tropospheric $\Delta^{14}\text{C}(t)$ trajectories and growth-season-integrated ring predictions are exact and `jax.jit` / `jax.grad` differentiable:

```python
production = tt.ProductionSum(
    steady=tt.SteadyProduction(),
    solar=tt.HarmonicSolar.from_polar(amplitudes=0.18, phases=0.4, periods=(11.0,), t_ref=775.0),
    spike=tt.MiyakeSpike(t_start=774.45, duration=0.25, fluence=6.8, profile="boxcar"),
)
model = tt.RadiocarbonModel(cbm_brehm, production)

# Continuous Northern vs Southern Hemisphere trajectories (preserving Brehm21's ~4.38‰ NH-SH offset)
t_grid = jnp.linspace(768.0, 786.0, 300)
d14c_nh = model.predict_fine(t_grid, hemisphere="north")
d14c_sh = model.predict_fine(t_grid, hemisphere="south")

# Growth-season-averaged predictions on every ring in an audited Dataset
ring_pred = model.predict_d14c(dataset)
print(f"NH-SH pre-spike gradient: {float(d14c_nh[0] - d14c_sh[0]):.2f} permil")
```

---

## 3. Semiparametric Miyake Inference (MAP, NUTS, & Tempered SMC)

In `ticktack 2.0`, linear nuisance parameters – global Holocene baseline offset $\Delta_{\text{global}}$, per-series laboratory offsets $\delta_{\text{lab},s}$, 11-year solar quadrature coefficients, and causal Solin–Särkkä GP modes – are marginalized analytically via `LinearMarginalConfig`. Samplers only explore the low-dimensional nonlinear coordinates declared in `ParameterSpace`:

```python
import math
import jax
import tinygp.kernels as tk

prod_semi = tt.ProductionSum(
    steady=tt.SteadyProduction(),
    solar=tt.HarmonicSolar.from_polar(amplitudes=0.0, phases=0.0),
    gp=tt.GPProduction(tk.quasisep.Matern32(scale=6.0, sigma=0.08), n_modes=32),
    spike=tt.MiyakeSpike(t_start=774.4, duration=0.35, fluence=6.5, profile="boxcar"),
)
semi_model = tt.RadiocarbonModel(cbm_brehm, prod_semi)

param_space = tt.ParameterSpace(semi_model, [
    tt.ParamSpec("production.spike.t_start", "identity", prior_mean=774.4, prior_sd=0.4),
    tt.ParamSpec("production.spike.duration", "sigmoid", prior_mean=-1.0, prior_sd=1.2, low=0.05, high=2.0),
    tt.ParamSpec("production.spike.fluence", "exp", prior_mean=math.log(6.5), prior_sd=0.5),
])
cfg = tt.LinearMarginalConfig(
    offset_mode="hierarchical",
    solar_prior_sd=0.25,
    marginalize_solar=True,
    marginalize_gp=True,
)

# 1. Constrained-space MAP optimization (includes bijector log-Jacobian)
map_res = tt.fit_map(semi_model, dataset, param_space, cfg, method="BFGS", map_space="constrained")
lp = tt.linear_posterior(map_res.fitted_model, dataset, cfg, hemisphere="north")
fl_map = float(map_res.theta_map["production.spike.fluence"])
print(
    f"MAP t_start = {float(map_res.theta_map['production.spike.t_start']):.3f} CE | "
    f"Fluence = {float(tt.RadiocarbonModel.fluence_atoms_cm2(fl_map)):.3e} atoms cm^-2 | "
    f"Epoch q0 = {float(tt.RadiocarbonModel.fluence_q0_years(fl_map, baseline_d14c_permil=lp.baseline_d14c)):.2f} yr"
)

# 2. NumPyro NUTS posterior sampling (warm-started from the MAP model; returns ArviZ InferenceData)
idata = tt.sample_nuts(
    map_res.model, dataset, param_space, cfg,
    num_warmup=400, num_samples=400, num_chains=2,
    init_map=False, rng_key=jax.random.PRNGKey(0),
)

# 3. BlackJAX adaptive tempered SMC for Bayesian model evidence ln Z +/- SE
smc_res = tt.sample_smc(
    semi_model, dataset, param_space, cfg,
    num_particles=256, target_ess=0.75, rng_key=jax.random.PRNGKey(1),
)
print(f"ln Z = {smc_res.log_evidence:.2f} +/- {smc_res.log_evidence_err:.2f}")
```

---

## 4. Non-Parametric IFT Reconstruction & Automatic GP Kernel Selection

To reconstruct cosmic-ray production $Q(t)$ non-parametrically without assuming a parametric spike shape – or to select among all seven candidate GP spectral families (`"matern12"`, `"matern32"`, `"matern52"`, `"squared_exp"`, `"continuous_matern"`, `"matern32_plus_solar"`, `"correlated_field"`) by empirical Bayes – use `ticktack.fields`:

```python
# Automatic empirical-Bayes GP kernel selection via Laplace hyper-evidence
sel = tt.select_gp_kernel(cbm_brehm, dataset, n_grid=128)
print(f"Selected kernel: {sel.best_name} (ln Z_hyper = {float(sel.best_result.log_hyper_evidence):.2f})")

# MacKay laboratory uncertainty diagnostic (s > 1 indicates understated AMS errors; s < 1 conservative)
s_global = float(tt.error_scale(sel.best_result.fit, dataset))
dataset_rescaled = dataset.with_error_scale(s_global)
print(f"Global MacKay error_scale = {s_global:.3f}")

# Strictly positive log-Gaussian IFT reconstruction with exact Ito mean correction E[Q(t)] = Q_std
op = sel.best_result.fit.operator
log_field = tt.LogGaussianField(
    jnp.zeros(op.n_grid),
    op.t_start,
    op.t_end,
    kernel=sel.best_kernel,
    ito_corrected=True,
)
pos_fit = tt.fit_ift_field(tt.RadiocarbonModel(cbm_brehm, log_field), dataset_rescaled, operator=op)
pos_draws = tt.laplace_samples(pos_fit, dataset_rescaled, n_samples=200, rng_key=jax.random.PRNGKey(42))
print(f"Peak reconstructed Q(t): {float(jnp.max(pos_fit.q_grid)):.2f} atoms cm^-2 s^-1")
```
