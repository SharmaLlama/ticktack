# Contributing to ticktack

Contributions to `ticktack` — bug fixes, new carbon-cycle presets, audited radiocarbon datasets, inference diagnostics, and documentation improvements — are welcome.

## Scope

`ticktack` provides fast, exact, differentiable linear and non-autonomous carbon box modelling and Bayesian inference for tree-ring and atmospheric $\Delta^{14}\text{C}$ time series in JAX. Contributions within this scope (published carbon-box topologies, cosmogenic production parameterizations, Gaussian-process / Information Field Theory spectral priors, calibrated radiocarbon datasets, and numerical tests) fit naturally; coupled general circulation or nonlinear biogeochemical Earth-system models lie outside the scope of the package.

## Development Setup

`ticktack 2.0` uses a PEP 621 `pyproject.toml` layout under `src/ticktack/` and requires Python $\ge 3.11$ with 64-bit JAX (`jax_enable_x64 = True` is enabled automatically on import).

We recommend [`uv`](https://docs.astral.sh/uv/) for fast, reproducible virtual environments:

```bash
git clone https://github.com/SharmaLlama/ticktack.git
cd ticktack
uv venv --python 3.12
uv pip install -e ".[dev]"
```

Standard `pip` editable installs work identically:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Architecture & Coding Conventions

- **Immutable PyTrees (`zodiax` / `equinox`)**: All models (`BalancedCBM`, `ReversibleCBM`, `OpenCBM`, `EffectiveCBM`, `ProductionTerm` subclasses, `ExactLinearSolver`, `DiffraxSolver`, `ModaxSolver`, `RadiocarbonModel`, `GaussianField`, `LogGaussianField`) inherit from `zodiax.Base` (`equinox.Module`). Never mutate attributes in place; update leaves functionally with `model.set("path.to.leaf", new_value)` or `equinox.tree_at`.
- **Float64 precision**: Matrix exponentials across four orders of magnitude in reservoir relaxation timescales ($0.2\text{ yr}$ stratosphere to $5000\text{ yr}$ deep ocean) and Schur-complement Fisher spectra require IEEE 754 `float64` throughout. Cast array inputs explicitly with `jnp.asarray(..., dtype=jnp.float64)`.
- **Pure JAX paths**: Forward operators, marginal log-likelihoods, and IFT field energies must remain `jax.jit`-, `jax.grad`-, and `jax.vmap`-compatible without Python-side mutation or data-dependent control flow on traced arrays.
- **Physical & calendar conventions**: Every dataset and forward calculation must obey the explicit conventions documented in `docs/conventions.md` (Cambridge vs Stuiver–Polach decay constants, single global tropospheric reference specific activity $R_{\text{ref}}$, and mandatory Southern Hemisphere Schulman vs growth-end ring-dating conventions).
- **Docstrings & spelling**: Public classes and functions use NumPy-style docstrings rendered by `mkdocstrings`. Documentation follows Oxford (OUP) spelling (`-ize`, `-ization`, `-our`, `-re`, `catalogue`, `modelled`, `modelling`).

## Testing, Linting, & Documentation

Run the test suite and linter before opening a pull request:

```bash
pytest
ruff check src tests
```

Build the documentation site in strict mode to verify that all `mkdocstrings` cross-references, internal links, and MathJax blocks resolve without warnings:

```bash
pip install mkdocs mkdocs-material "mkdocstrings[python]"
mkdocs build --strict
```

To preview the documentation locally with live reload:

```bash
mkdocs serve
```

## Pull Request Workflow

1. Fork the repository and create a feature branch from `main`.
2. Add unit tests under `tests/` covering any new functionality, analytical limiting cases, and `jax.grad` / finite-difference agreement where applicable.
3. Ensure `pytest`, `ruff check src tests`, and `mkdocs build --strict` pass cleanly.
4. Open a pull request against [`SharmaLlama/ticktack`](https://github.com/SharmaLlama/ticktack/pulls) describing the physical motivation, mathematical formulation, and verification checks.

## Questions, Issues, & Discussions

- Bug reports and concrete feature requests: open an [Issue](https://github.com/SharmaLlama/ticktack/issues).
- Design questions, scientific use cases, and broader proposals: start a thread in [Discussions](https://github.com/SharmaLlama/ticktack/discussions).

## Code of Conduct

All contributors and maintainers are expected to adhere to the project [Code of Conduct](https://github.com/SharmaLlama/ticktack/blob/main/CODE_OF_CONDUCT.md).
