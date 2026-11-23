# Contributing to ticktack

Contributions to `ticktack` — bug fixes, new carbon-cycle presets, audited radiocarbon datasets, inference diagnostics, and documentation improvements — are welcome.

## Scope

`ticktack` provides fast, differentiable linear and non-autonomous carbon box modelling and Bayesian inference for tree-ring and atmospheric $\Delta^{14}\text{C}$ time series in JAX. Contributions within this scope (published carbon-box topologies, cosmogenic production parametrizations, Gaussian-process / Information Field Theory spectral priors, calibrated radiocarbon datasets, and numerical tests) fit naturally; coupled general circulation or nonlinear biogeochemical Earth-system models lie outside the scope of the package.

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

## Pull Request Workflow & Review Gates

Every pull request passes through six ordered gates so that an error must get past deterministic checks, a same-family independent reviewer, and (for core science or large changes) a cross-family reviewer before merge.

1. **Scope.** One purpose per PR, at most ~400 changed `src/ticktack/` code lines (tests, notebooks, and docs excluded), linked to a written task or issue. Split larger changes into independent PRs. Bug fixes start with a test that fails before the fix and passes after it.
2. **Continuous integration.** All CI checks must pass:
   - `ruff check src tests` plus the complexity ratchet (`C901`, `PLR0912`, `PLR0913`, `PLR0915` violations must not exceed `.github/ci_ratchets.json`).
   - Source code line budget (`src/ticktack/**/*.py` non-docstring, non-comment code lines must not exceed `.github/ci_ratchets.json` unless the PR description includes a non-empty `Line-budget-reason:` justification).
   - `pytest` with coverage reporting and the coverage floor in `.github/ci_ratchets.json`.
   - Public-surface snapshot (`tests/test_public_surface.py` fails if `ticktack.__all__` changes without updating `tests/public_surface_snapshot.json`).
   - Execution of a fast subset of `notebooks/v2/docs_examples/`.
   - Trial merge against every other open PR targeting the same base branch that modifies any overlapping file, running `ruff` and `pytest` on clean merges to catch silent semantic clashes.
3. **Evidence table.** The PR description includes an Evidence Table (`| Claim | Verification (test name or command + output) |`). Every factual, numerical, or behavioural claim maps to a test name or a command and its output; claims without a row are deleted.
4. **Independent Carbon review.** An instance that did not write the code reviews the PR given only the diff and a five-line statement of intent (not the author's summary or conclusions). Its job is to write failing tests, not opinions. Reviewer output lists findings most severe first (`file:line`, what is wrong, and a concrete failing case or why it is only a suspicion), followed by an explicit list of what was not checked.
5. **Claude review.** Required for any PR touching `cbm.py`, `fields.py`, `fisher.py`, `likelihood.py`, `linear.py`, or `solvers.py`, and for any PR changing more than 200 `src/ticktack/` code lines. Apply the `needs-claude-review` label and stop; Ben starts the Claude session (Sonnet by default). Mechanical PRs outside those modules and within 200 changed source lines need only Gates 1–4. Conversely, code written by Claude receives an independent Carbon review before Ben sees it.
6. **Ben merges.** Agents never merge pull requests; Ben merges once all applicable gates are satisfied.

## Code Shape Rules (New & Changed Code)

1. **Reuse first.** Search for an existing helper before writing one, and list reused functions in the PR description.
2. **Private helpers.** Keep a private helper only if it has at least two callers or is a numerical kernel with its own unit test; otherwise inline it.
3. **No algorithm-selecting flags.** A boolean or string flag that selects between whole algorithms should become separate functions sharing setup.
4. **Function size and arity.** Keep new or edited functions to at most ~80 code lines and 8 parameters; any exception carries a one-line justification comment.
5. **Validate once.** Validate inputs at the public entry point; internal functions trust their callers. Use one conversion helper rather than repeating inline casts.
6. **One accepted form.** Convert heterogeneous inputs at a single constructor instead of accepting wide union types throughout the package.
7. **Tight typing, no aliases.** Avoid `Any` when the type is known. Use one spelling and one class name without aliases.
8. **Replace, do not duplicate.** A PR that adds a second way of doing something removes or wraps the first.
9. **Deliberate public surface.** Every new public name requires a docs example, a test, and an explicit update to `tests/public_surface_snapshot.json`. Keep internals out of `__all__`.
10. **Honest docstrings.** State behaviour, units, and array shapes; omit performance claims and provenance notes. Use "exact" only when true in exact arithmetic; otherwise state the tolerance and name the test.
11. **Independent verification.** Test every new mathematical formula against an independent reference (closed-form expression, dense matrix, finite differences, or external solver), never solely against another code path in `ticktack`.

## Questions, Issues, & Discussions

- Bug reports and concrete feature requests: open an [Issue](https://github.com/SharmaLlama/ticktack/issues).
- Design questions, scientific use cases, and broader proposals: start a thread in [Discussions](https://github.com/SharmaLlama/ticktack/discussions).

## Code of Conduct

All contributors and maintainers are expected to adhere to the project [Code of Conduct](https://github.com/SharmaLlama/ticktack/blob/main/CODE_OF_CONDUCT.md).
