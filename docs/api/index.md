# API Reference

Everyday classes and functions are re-exported directly from the top-level `ticktack` namespace:

```python
import ticktack as tt
```

Specialized constants, preset serialization helpers, and low-level linear-algebra utilities can also be imported from their submodules listed below.

---

## Submodules

### Models & Forward Operators

- **[Carbon Box Models & Presets](cbm.md)** (`ticktack.cbm`, `ticktack.presets`, `ticktack.constants`): `Topology`, `CarbonBoxModel`, `FixedCBM`, `BalancedCBM`, `ReversibleCBM`, `OpenCBM`, `EffectiveCBM`, `SubspaceCBM` (`FisherSubspaceCBM`), `load_preset`, `PresetSpec`, and radiometric/isotopic constants.
- **[Production Terms](production.md)** (`ticktack.production`): `ProductionTerm`, `SteadyProduction`, `MiyakeSpike`, `HarmonicSolar`, `LinearTrend`, `GriddedProduction`, `GPProduction`, `LogGPProduction`, and `ProductionSum`.
- **[Solvers & RadiocarbonModel](solvers_model.md)** (`ticktack.solvers`, `ticktack.model`): `ExactLinearSolver`, `DiffraxSolver`, `ModaxSolver`, `wrap_jvp_only_scalar_fn`, and `RadiocarbonModel`.

### Data & Non-Parametric Fields

- **[Data & Calendar Auditing](data.md)** (`ticktack.data`): `SeasonalWeights`, `TreeRingSeries`, `Dataset`, `BUNDLED_DATASET_CATALOGUE`, `load_bundled_series`, `load_bundled_event`, `convert_calendar_years`, and `parse_growth_season_bounds`.
- **[Linear Marginalization & POD](linear.md)** (`ticktack.linear`): `LinearMarginal`, `posterior`, `solin_sarkka_basis`, `project_causal_gp`, `canonical_pod_modes`, `stationary_cbm_gp_cov`, `kalman_gp_marginal`, and whitening/arrow-matrix solvers.
- **[IFT Fields & Kernel Selection](fields.md)** (`ticktack.fields`): `GaussianField`, `LogGaussianField`, `MaternSpectrum`, `SquaredExpSpectrum`, `QuasiPeriodicSpectrum`, `CorrelatedFieldSpectrum`, `SumSpectrum`, `fit_ift_field`, `log_evidence`, `error_scale`, `evidence_grid`, `optimize_gp_kernel`, and `select_gp_kernel`.

### Bayesian Inference, Fisher Geometry, & Visualization

- **[Likelihood, Fitting, Sampling & Fisher](inference.md)** (`ticktack.likelihood`, `ticktack.fitting`, `ticktack.sampling`, `ticktack.fisher`): `ParamSpec`, `ParameterSpace`, `LinearMarginalConfig`, `marginal_log_likelihood`, `linear_posterior`, `fit_map`, `sample_nuts`, `fit_svi`, `sample_smc`, `cbm_fisher_spectrum`, `relaxation_fisher_modes`, and `cbm_mode_impulse_responses` (`CBMModeImpulseResult`).
- **[Plotting & Legacy Compatibility](plotting_legacy.md)** (`ticktack.plotting` & legacy `v1` reproduction flags): `plot_d14c_fit`, `plot_production_posterior`, `plot_fisher_spectrum`, `plot_cbm_eigenmodes`, `plot_cbm_tikz_modes`, `plot_cbm_mode_impulse_responses`, `plot_corner`, `plot_field_draws`, `plot_evidence_grid`, and legacy `ticktack 1.x` audit options.
