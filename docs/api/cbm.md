# Carbon Box Models, Presets & Constants

Immutable [`zodiax`](https://github.com/LouisDesdoigts/zodiax) PyTree carbon-box parametrizations (`ticktack.cbm`), audited literature presets (`ticktack.presets`), and physical/radiometric constants (`ticktack.constants`). See [Physical & Calendar Conventions](../conventions.md), [Solvers & Carbon Box Models](../solvers_and_cbms.md), and [Fisher Information & EffectiveCBMs](../fisher_and_effective_cbm.md) for mathematical derivations.

---

## `ticktack.cbm`

::: ticktack.cbm
    options:
      show_root_heading: false

### Fisher Subspace Parametrization (`SubspaceCBM` / `FisherSubspaceCBM`)

`SubspaceCBM` (re-exported as `FisherSubspaceCBM`) parametrizes a flow-balanced Perron–Frobenius carbon box model in the leading $m$ Schur-marginalized Fisher log-rate eigenmodes $\ln \mathbf{k}_{\text{free}}(\boldsymbol{\alpha}) = \ln \mathbf{k}_{\text{free},0} + B\,\boldsymbol{\alpha}$:

```python
@classmethod
def from_fisher(
    cls,
    base_cbm: BalancedCBM,
    spectrum: FisherSpectrumResult,
    n_modes: int,
) -> SubspaceCBM
```

- **`alpha`**: Dynamic `(m,)` unconstrained subspace coordinates (`alpha = 0` reproduces `base_cbm` to machine precision). Under `ParamSpec("cbm.alpha", "identity", prior_mean=0.0, prior_sd=1.0)`, the prior matches the $\mathcal{N}(\ln \mathbf{k}_0, \Sigma_0)$ prior of `BalancedCBM` restricted to the top $m$ Fisher modes, with zero gauge nullspace.
- **`base_log_rates_free`, `basis_rows`, `base_log_total_mass`**: Static `equinox` fields storing $\ln \mathbf{k}_{\text{free},0}$, `spectrum.parameter_loadings[:p_free, :n_modes]`, and $\ln M_{\text{tot}}$.
- **`to_balanced_cbm()`**: Materializes the corresponding `BalancedCBM` at the current `alpha`.

---

## `ticktack.presets`

::: ticktack.presets
    options:
      show_root_heading: false

---

## `ticktack.constants`

::: ticktack.constants
    options:
      show_root_heading: false
