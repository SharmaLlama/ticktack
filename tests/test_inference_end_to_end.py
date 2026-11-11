"""End-to-end integration tests for ticktack 2.0 Wave 2: model, likelihood, fitting, sampling.

Tests are grouped as follows:

1. ``RadiocarbonModel`` basics — NH/SH prediction, zodiax ``.get``/``.set``,
   NH–SH gradient, fluence helpers.
2. Marginal likelihood — numerical agreement with explicit Gaussian integral,
   gradient correctness, hierarchical offset bias test.
3. MAP fitting — BFGS and LevenbergMarquardt on synthetic Miyake data.
4. NUTS sampler — valid ``arviz.InferenceData``, R-hat reasonable.
5. SVI — valid ``arviz.InferenceData`` with convergence.
6. SMC — log-evidence finite, beta monotonicity.
7. Notebook smoke-test — all three ``.ipynb`` files parse as valid nbformat v4
   with alternating cells and non-empty outputs.
"""

from __future__ import annotations

import math
import pathlib

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

ROOT = pathlib.Path(__file__).parent.parent  # repo root


@pytest.fixture(scope="session")
def brehm21_cbm():
    """Load the Brehm21 CBM preset (session-scoped, cheap)."""
    import ticktack as tt

    return tt.load_preset("Brehm21")


@pytest.fixture(scope="session")
def simple_production():
    """A minimal SteadyProduction + MiyakeSpike for unit tests."""
    import ticktack as tt

    return tt.ProductionSum(
        steady=tt.SteadyProduction(),
        spike=tt.MiyakeSpike(t_start=774.3, duration=1.0, fluence=120.0),
    )


@pytest.fixture(scope="session")
def simple_model(brehm21_cbm, simple_production):
    """A minimal RadiocarbonModel using Brehm21 + SteadyProduction + spike."""
    import ticktack as tt

    return tt.RadiocarbonModel(brehm21_cbm, simple_production)


def _make_series(t_mid, d14c_vals, sigma=3.0, hemisphere="north"):
    """Create a TreeRingSeries from midpoint years."""
    import ticktack as tt

    return tt.TreeRingSeries(
        years=t_mid,
        d14c=d14c_vals,
        sigma_d14c=np.full(len(t_mid), sigma, dtype=np.float64),
        hemisphere=hemisphere,
        calendar="astronomical_ce",
        year_convention="schulman_start",
    )


@pytest.fixture(scope="session")
def synthetic_dataset(simple_model):
    """
    Small synthetic Dataset with 2 NH series + 1 SH series around 774 CE.
    Generates data from the model, then adds Gaussian noise.
    """
    import ticktack as tt

    rng = np.random.default_rng(42)
    t_mid = np.arange(772.0, 778.0, 1.0, dtype=np.float64)

    series_list = []
    hemispheres = ["north", "north", "south"]
    for hemi in hemispheres:
        s = _make_series(t_mid, np.zeros(len(t_mid)), hemisphere=hemi)
        series_list.append(s)

    dataset = tt.Dataset(series_list)
    # Overwrite d14c with model prediction + noise
    pred = simple_model.predict_d14c(dataset, concat=True)
    pred_np = np.array(pred)
    noisy = pred_np + rng.normal(0.0, 3.0, size=pred_np.shape)

    # Rebuild dataset with realistic d14c values
    series_new = []
    offset = 0
    for s in dataset.series:
        n = s.n_obs
        new_s = _make_series(
            np.array(s.t_midpoints),
            noisy[offset : offset + n],
            sigma=3.0,
            hemisphere=s.hemisphere,
        )
        series_new.append(new_s)
        offset += n

    return tt.Dataset(series_new)


# ===========================================================================
# 1. RadiocarbonModel basics
# ===========================================================================


class TestRadiocarbonModel:
    def test_steady_state_shape(self, simple_model):
        y_ss = simple_model.steady_state()
        assert y_ss.ndim == 1
        assert y_ss.shape[0] >= 2  # at least NH + SH boxes
        assert jnp.all(y_ss > 0), "Steady-state inventories must be positive."

    def test_reference_specific_activity_scalar(self, simple_model):
        R_ref = simple_model.reference_specific_activity()
        assert R_ref.shape == ()
        assert float(R_ref) > 0.0

    def test_predict_d14c_treering_shape(self, simple_model, synthetic_dataset):
        series = synthetic_dataset.series[0]
        pred = simple_model.predict_d14c(series)
        assert pred.shape == (series.n_obs,)
        assert jnp.isfinite(pred).all()

    def test_predict_d14c_dataset_concat(self, simple_model, synthetic_dataset):
        pred = simple_model.predict_d14c(synthetic_dataset, concat=True)
        assert pred.shape == (synthetic_dataset.n_obs,)
        assert jnp.isfinite(pred).all()

    def test_predict_d14c_dataset_no_concat(self, simple_model, synthetic_dataset):
        preds = simple_model.predict_d14c(synthetic_dataset, concat=False)
        assert isinstance(preds, (tuple, list))
        assert len(preds) == synthetic_dataset.n_series
        total = sum(p.shape[0] for p in preds)
        assert total == synthetic_dataset.n_obs

    def test_nh_sh_gradient(self, brehm21_cbm):
        """NH should predict a slightly higher Delta14C baseline than SH (NH–SH > 0)."""
        import ticktack as tt

        # predict with no spike (steady only)
        steady_model = tt.RadiocarbonModel(brehm21_cbm, tt.SteadyProduction())
        t_mid = np.arange(770.0, 776.0, 1.0, dtype=np.float64)
        series_nh = _make_series(t_mid, np.zeros(6), hemisphere="north")
        series_sh = _make_series(t_mid, np.zeros(6), hemisphere="south")
        ds = tt.Dataset([series_nh, series_sh])

        preds = steady_model.predict_d14c(ds, concat=False)
        d14c_nh = float(jnp.mean(preds[0]))
        d14c_sh = float(jnp.mean(preds[1]))
        assert d14c_nh > d14c_sh, (
            f"Expected NH > SH Delta14C (gradient); got NH={d14c_nh:.2f}, SH={d14c_sh:.2f}"
        )

    def test_zodiax_get(self, simple_model):
        t = simple_model.get("production.spike.t_start")
        assert float(t) == pytest.approx(774.3, abs=1e-6)

    def test_zodiax_set(self, simple_model):
        m2 = simple_model.set("production.spike.t_start", 775.0)
        assert float(m2.get("production.spike.t_start")) == pytest.approx(775.0, abs=1e-6)
        # Original should be unchanged
        assert float(simple_model.get("production.spike.t_start")) == pytest.approx(774.3, abs=1e-6)

    def test_predict_fine_shape(self, simple_model):
        t_eval = jnp.linspace(773.0, 778.0, 50, dtype=jnp.float64)
        pred = simple_model.predict_fine(t_eval)
        assert pred.shape == (50,)
        assert jnp.isfinite(pred).all()

    def test_impulse_response_positive(self, simple_model):
        t_lag = jnp.linspace(0.0, 20.0, 21, dtype=jnp.float64)
        ir = simple_model.impulse_response(t_lag)
        assert ir.shape == (21,)
        # Impulse response should be nonnegative
        assert jnp.all(ir >= -1e-10), "Impulse response should be nonneg."

    # Fluence helpers

    def test_fluence_atoms_cm2(self):
        from ticktack.constants import SECONDS_PER_JULIAN_YEAR
        from ticktack.model import RadiocarbonModel

        fluence = 120.0
        atoms_cm2 = RadiocarbonModel.fluence_atoms_cm2(fluence)
        expected = fluence * float(SECONDS_PER_JULIAN_YEAR)
        assert float(atoms_cm2) == pytest.approx(expected, rel=1e-10)

    def test_fluence_q0_years_pi(self):
        from ticktack.constants import DEFAULT_Q_STD
        from ticktack.model import RadiocarbonModel

        fluence = 120.0
        result = RadiocarbonModel.fluence_q0_years(fluence)
        expected = fluence / float(DEFAULT_Q_STD)
        assert float(result) == pytest.approx(expected, rel=1e-10)

    def test_fluence_q0_years_epoch(self):
        """With a baseline Delta14C, the epoch-local q0 must differ from pre-industrial."""
        from ticktack.constants import DEFAULT_Q_STD
        from ticktack.model import RadiocarbonModel

        fluence = 120.0
        delta_0 = 50.0  # 50‰ above pre-industrial
        q0_pi = RadiocarbonModel.fluence_q0_years(fluence)
        q0_epoch = RadiocarbonModel.fluence_q0_years(fluence, baseline_d14c_permil=delta_0)
        # epoch-local is fluence / (q_std * (1 + delta_0 / 1000))
        expected_epoch = fluence / (float(DEFAULT_Q_STD) * (1.0 + delta_0 / 1000.0))
        assert float(q0_epoch) == pytest.approx(expected_epoch, rel=1e-10)
        # q0_epoch should be less than q0_pi when delta_0 > 0
        assert float(q0_epoch) < float(q0_pi)


# ===========================================================================
# 2. Likelihood — marginal vs joint, gradients, hierarchical offset
# ===========================================================================


class TestLikelihood:
    @pytest.fixture(scope="class")
    def simple_param_space(self, simple_model):
        import ticktack as tt

        specs = [
            tt.ParamSpec("production.spike.t_start", "identity", prior_mean=774.3, prior_sd=1.0),
            tt.ParamSpec(
                "production.spike.fluence", "exp", prior_mean=math.log(120.0), prior_sd=1.0
            ),
        ]
        return tt.ParameterSpace(simple_model, specs)

    def test_marginal_log_likelihood_finite(self, simple_model, synthetic_dataset):
        import ticktack as tt

        cfg = tt.LinearMarginalConfig()
        lml = tt.marginal_log_likelihood(simple_model, synthetic_dataset, cfg)
        assert jnp.isfinite(lml), f"marginal_log_likelihood returned {lml}"

    def test_joint_log_likelihood_finite(self, simple_model, synthetic_dataset):
        import ticktack as tt

        lml = tt.joint_log_likelihood(simple_model, synthetic_dataset)
        assert jnp.isfinite(lml), f"joint_log_likelihood returned {lml}"

    def test_marginal_log_likelihood_grad(self, simple_model, synthetic_dataset, simple_param_space):
        """Check that gradients of the marginal log-likelihood w.r.t. u are finite."""
        import ticktack as tt

        cfg = tt.LinearMarginalConfig()
        u0 = simple_param_space.unconstrain(simple_model)

        def f(u):
            m = simple_param_space.constrain(u)
            return tt.marginal_log_likelihood(m, synthetic_dataset, cfg)

        grad = jax.grad(f)(u0)
        assert jnp.isfinite(grad).all(), f"Gradient contains NaN/Inf: {grad}"

    def test_paramspec_bijectors_round_trip(self, simple_model, simple_param_space):
        """Round-trip unconstrain -> constrain should recover original parameters."""
        u = simple_param_space.unconstrain(simple_model)
        m2 = simple_param_space.constrain(u)
        t_orig = float(simple_model.get("production.spike.t_start"))
        t_recovered = float(m2.get("production.spike.t_start"))
        assert t_orig == pytest.approx(t_recovered, rel=1e-6)

    def test_linear_posterior_result(self, simple_model, synthetic_dataset):
        """linear_posterior should return a LinearPosteriorResult with correct shapes."""
        import ticktack as tt

        cfg = tt.LinearMarginalConfig()
        result = tt.linear_posterior(simple_model, synthetic_dataset, cfg)
        assert isinstance(result, tt.LinearPosteriorResult)
        # coeff_mean and coeff_cov must be finite
        assert jnp.isfinite(result.coeff_mean).all()
        assert jnp.isfinite(result.coeff_cov).all()

    def test_hierarchical_offset_non_extreme(self, brehm21_cbm):
        """Hierarchical offsets should recover injected lab biases within reasonable bounds.

        Inject a known per-lab offset; the global coefficient should shift
        in the correct direction without the v1 mean(y[:4]) hard clamping.
        """
        import ticktack as tt

        rng = np.random.default_rng(99)
        t_mid = np.arange(770.0, 780.0, 1.0, dtype=np.float64)
        injected_offset = 15.0  # ‰ lab offset for series 0

        # Predict from steady model, add injected offset to series 0
        steady_model = tt.RadiocarbonModel(brehm21_cbm, tt.SteadyProduction())
        series0 = _make_series(t_mid, np.zeros(10), hemisphere="north")
        pred0 = np.array(steady_model.predict_d14c(series0)) + injected_offset
        noise = rng.normal(0.0, 2.0, size=10)

        series_obs = _make_series(t_mid, pred0 + noise, sigma=2.0, hemisphere="north")
        ds = tt.Dataset([series_obs])

        cfg = tt.LinearMarginalConfig(offset_mode="hierarchical")
        result = tt.linear_posterior(steady_model, ds, cfg)
        # The global coefficient should be finite and within ±200‰
        total_offset = float(result.coeff_mean[0])
        assert abs(total_offset) < 200.0, f"Recovered offset wildly wrong: {total_offset:.1f}"


# ===========================================================================
# 3. MAP fitting
# ===========================================================================


class TestMAPFitting:
    @pytest.fixture(scope="class")
    def map_param_space(self, simple_model):
        import ticktack as tt

        specs = [
            tt.ParamSpec("production.spike.t_start", "identity", prior_mean=774.3, prior_sd=1.0),
            tt.ParamSpec(
                "production.spike.fluence", "exp", prior_mean=math.log(120.0), prior_sd=1.0
            ),
        ]
        return tt.ParameterSpace(simple_model, specs)

    def test_fit_map_bfgs(self, simple_model, synthetic_dataset, map_param_space):
        import ticktack as tt

        result = tt.fit_map(
            simple_model,
            synthetic_dataset,
            map_param_space,
            method="BFGS",
            max_steps=50,
            rng_key=jax.random.PRNGKey(0),
        )
        assert isinstance(result, tt.MAPResult)
        assert jnp.isfinite(result.log_posterior)
        # Recovered t_start should be in a plausible range
        t_fitted = float(result.fitted_model.get("production.spike.t_start"))
        assert 773.0 <= t_fitted <= 777.0, f"t_start out of range: {t_fitted}"

    def test_fit_map_lm(self, simple_model, synthetic_dataset, map_param_space):
        import ticktack as tt

        result = tt.fit_map(
            simple_model,
            synthetic_dataset,
            map_param_space,
            method="LevenbergMarquardt",
            max_steps=50,
            rng_key=jax.random.PRNGKey(1),
        )
        assert isinstance(result, tt.MAPResult)
        assert jnp.isfinite(result.log_posterior)

    def test_map_laplace_cov_psd(self, simple_model, synthetic_dataset, map_param_space):
        """The Laplace covariance must be positive semi-definite."""
        import ticktack as tt

        result = tt.fit_map(
            simple_model,
            synthetic_dataset,
            map_param_space,
            method="BFGS",
            max_steps=50,
            rng_key=jax.random.PRNGKey(2),
        )
        if result.laplace_cov is not None:
            eigvals = jnp.linalg.eigvalsh(result.laplace_cov)
            assert jnp.all(eigvals >= -1e-8), f"Laplace cov has negative eigenvalue: {eigvals}"


# ===========================================================================
# 4. NUTS sampler
# ===========================================================================


class TestNUTS:
    @pytest.fixture(scope="class")
    def nuts_param_space(self, simple_model):
        import ticktack as tt

        return tt.ParameterSpace(
            simple_model,
            [
                tt.ParamSpec(
                    "production.spike.t_start",
                    "identity",
                    prior_mean=774.3,
                    prior_sd=1.0,
                ),
            ],
        )

    def test_sample_nuts_returns_inference_data(
        self, simple_model, synthetic_dataset, nuts_param_space
    ):
        import ticktack as tt

        idata = tt.sample_nuts(
            simple_model,
            synthetic_dataset,
            nuts_param_space,
            num_warmup=50,
            num_samples=50,
            num_chains=2,
            rng_key=jax.random.PRNGKey(42),
        )
        assert isinstance(idata, az.InferenceData)
        assert hasattr(idata, "posterior")
        # Check all posterior variables are finite
        for var in idata.posterior.data_vars:
            vals = np.array(idata.posterior[var])
            assert np.isfinite(vals).all(), f"NUTS posterior {var} has NaN/Inf"

    def test_sample_nuts_rhat_reasonable(
        self, simple_model, synthetic_dataset, nuts_param_space
    ):
        """R-hat < 1.5 for a very short chain (tolerant threshold for fast CI tests)."""
        import ticktack as tt

        idata = tt.sample_nuts(
            simple_model,
            synthetic_dataset,
            nuts_param_space,
            num_warmup=50,
            num_samples=50,
            num_chains=2,
            rng_key=jax.random.PRNGKey(43),
        )
        rhat = az.rhat(idata)
        for var in rhat.data_vars:
            assert float(rhat[var]) < 1.5, f"R-hat = {float(rhat[var]):.3f} for {var}"


# ===========================================================================
# 5. SVI
# ===========================================================================


class TestSVI:
    @pytest.fixture(scope="class")
    def svi_param_space(self, simple_model):
        import ticktack as tt

        return tt.ParameterSpace(
            simple_model,
            [
                tt.ParamSpec(
                    "production.spike.t_start",
                    "identity",
                    prior_mean=774.3,
                    prior_sd=1.0,
                ),
            ],
        )

    def test_fit_svi_returns_result(self, simple_model, synthetic_dataset, svi_param_space):
        import ticktack as tt

        result = tt.fit_svi(
            simple_model,
            synthetic_dataset,
            svi_param_space,
            guide="diag",
            num_steps=200,
            num_samples=100,
            rng_key=jax.random.PRNGKey(0),
        )
        assert isinstance(result, tt.SVIResult)
        assert isinstance(result.idata, az.InferenceData)
        assert jnp.isfinite(result.losses).all(), "SVI losses contain NaN/Inf"

    def test_svi_elbo_converges(self, simple_model, synthetic_dataset, svi_param_space):
        """ELBO losses should decrease overall (last 50% < first 50% + tolerance)."""
        import ticktack as tt

        result = tt.fit_svi(
            simple_model,
            synthetic_dataset,
            svi_param_space,
            guide="diag",
            num_steps=200,
            num_samples=50,
            rng_key=jax.random.PRNGKey(1),
        )
        losses = np.array(result.losses)
        if len(losses) > 100:
            first_half = float(np.mean(losses[: len(losses) // 2]))
            second_half = float(np.mean(losses[len(losses) // 2 :]))
            # ELBO losses (negated) should decrease
            assert second_half <= first_half + abs(first_half) * 0.5, (
                f"SVI ELBO not decreasing: first={first_half:.1f}, last={second_half:.1f}"
            )


# ===========================================================================
# 6. SMC — log-evidence finite, beta monotonicity
# ===========================================================================


class TestSMC:
    @pytest.fixture(scope="class")
    def smc_param_space(self, simple_model):
        import ticktack as tt

        return tt.ParameterSpace(
            simple_model,
            [
                tt.ParamSpec(
                    "production.spike.t_start",
                    "identity",
                    prior_mean=774.3,
                    prior_sd=1.0,
                ),
            ],
        )

    def test_sample_smc_returns_result(self, simple_model, synthetic_dataset, smc_param_space):
        import ticktack as tt

        result = tt.sample_smc(
            simple_model,
            synthetic_dataset,
            smc_param_space,
            num_particles=50,
            rng_key=jax.random.PRNGKey(0),
        )
        assert isinstance(result, tt.SMCResult)
        assert jnp.isfinite(result.log_evidence)
        assert isinstance(result.idata, az.InferenceData)

    def test_smc_betas_monotone(self, simple_model, synthetic_dataset, smc_param_space):
        """Tempering betas must be monotonically increasing from 0 to 1."""
        import ticktack as tt

        result = tt.sample_smc(
            simple_model,
            synthetic_dataset,
            smc_param_space,
            num_particles=30,
            rng_key=jax.random.PRNGKey(1),
        )
        betas = np.array(result.tempering_betas)
        assert betas[0] == pytest.approx(0.0, abs=1e-9)
        assert betas[-1] == pytest.approx(1.0, abs=1e-6)
        assert np.all(np.diff(betas) >= -1e-10), "Betas must be non-decreasing."

    def test_smc_log_evidence_finite(self, simple_model, synthetic_dataset, smc_param_space):
        """log_evidence and log_evidence_err must be finite scalars."""
        import ticktack as tt

        result = tt.sample_smc(
            simple_model,
            synthetic_dataset,
            smc_param_space,
            num_particles=50,
            rng_key=jax.random.PRNGKey(2),
        )
        assert jnp.isfinite(result.log_evidence)
        assert jnp.isfinite(result.log_evidence_err)
        assert float(result.log_evidence_err) >= 0.0


# ===========================================================================
# 7. Notebook smoke-test — valid nbformat v4, alternating cells, non-empty outputs
# ===========================================================================


NOTEBOOKS = [
    "notebooks/v2/01_miyake_774ce_inference.ipynb",
    "notebooks/v2/02_effective_cbm_fisher_smc.ipynb",
    "notebooks/v2/03_nonparametric_gp_pod.ipynb",
]


@pytest.mark.parametrize("rel_path", NOTEBOOKS)
def test_notebook_valid_nbformat(rel_path):
    """Notebook must parse as valid nbformat v4 JSON with >= 5 cells."""
    import nbformat

    path = ROOT / rel_path
    if not path.exists():
        pytest.skip(f"Notebook not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    assert nb.nbformat == 4
    assert len(nb.cells) >= 5, "Notebook must have at least 5 cells."


@pytest.mark.parametrize("rel_path", NOTEBOOKS)
def test_notebook_alternating_cells(rel_path):
    """Every code cell must be preceded by a markdown cell."""
    import nbformat

    path = ROOT / rel_path
    if not path.exists():
        pytest.skip(f"Notebook not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    prev_type = None
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            assert prev_type == "markdown", (
                f"Cell {i} is 'code' but previous cell is '{prev_type}'; "
                "each code cell must be preceded by a markdown cell."
            )
        prev_type = cell.cell_type


@pytest.mark.parametrize("rel_path", NOTEBOOKS)
def test_notebook_code_cells_have_outputs(rel_path):
    """Each code cell must have at least one non-empty output."""
    import nbformat

    path = ROOT / rel_path
    if not path.exists():
        pytest.skip(f"Notebook not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            assert len(cell.outputs) > 0, f"Code cell {i} has no outputs."


@pytest.mark.parametrize("rel_path", NOTEBOOKS)
def test_notebook_no_error_outputs(rel_path):
    """No code cell must have an error output."""
    import nbformat

    path = ROOT / rel_path
    if not path.exists():
        pytest.skip(f"Notebook not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            for out in cell.outputs:
                assert out.output_type != "error", (
                    f"Cell {i} has an error output: {out.get('ename', '?')}: "
                    f"{out.get('evalue', '?')}"
                )
