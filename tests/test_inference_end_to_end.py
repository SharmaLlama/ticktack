"""End-to-end integration tests for ticktack 2.0 Wave 2: model, likelihood, fitting, sampling, notebooks.

Covers:
1. ``RadiocarbonModel`` — NH/SH box routing, ``Brehm21`` inter-hemispheric gradient,
   ``zodiax`` ``.get``/``.set``, continuous ``predict_fine``, ``impulse_response``,
   ``transfer_function``, and pre-industrial vs epoch-local ``q_0`` fluence helpers.
2. ``ParameterSpace`` & ``likelihood`` — bijector round-trips, vector & zero-sized array
   leaves, normalized unconstrained priors, exact ``< 1e-10`` agreement of
   ``marginal_log_likelihood`` (hierarchical offsets + ``HarmonicSolar`` + ``LinearTrend``
   + causal ``GPProduction``) with dense Gaussian marginal likelihood and exact Gaussian
   integral of ``joint_log_likelihood``, 2nd-order ``jax.test_util.check_grads``,
   ``linear_posterior`` continuous ``Q(t)``/``Delta14C(t)`` reconstruction, and
   staggered-start secular-ramp + Miyake-spike unbiased fluence recovery vs v1 ``mean(y[:4])``.
3. ``fit_map`` — ``BFGS``, ``LevenbergMarquardt``, and ``adam`` optimization plus SPD
   Laplace covariance.
4. ``sample_nuts`` & ``fit_svi`` — 2-chain ``R-hat < 1.05``, conditional linear posterior
   draws (``offset_global``, ``offset_lab``, ``q_fine``, ``d14c_fine``), and strictly
   positive ``LogGPProduction`` MAP & SVI reconstructions ``Q(t) > 0``.
5. ``sample_smc`` — exact 2D multivariate Gaussian log-evidence test
   ``|ln Z_hat - ln Z_exact| < max(3.5 * sigma_lnZ, 0.12)`` with ``sigma_lnZ > 0``, plus
   ``EffectiveCBM(K=2, 3)`` CBM parameter SMC runs.
6. Notebook validation — stdlib ``json`` audit of ``notebooks/v2/01..03.ipynb`` checking
   nbformat v4, alternating prose/code structure, genuine execution counts, inline
   ``image/png`` outputs, zero error outputs, single-output-per-cell discipline, and
   zero stray ``.png`` files in ``notebooks/v2/``.
"""

from __future__ import annotations

import json
import math
import pathlib

import arviz as az
import jax
import jax.numpy as jnp
import jax.scipy.stats as jss
import numpy as np
import pytest
import tinygp.kernels as tk
from jax.test_util import check_grads

import ticktack as tt
from ticktack.likelihood import _build_marginal_design, _nonlinear_residual

ROOT = pathlib.Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Module-level fixtures (zero class-scoped fixture warnings)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def brehm21_cbm():
    return tt.load_preset("Brehm21")


@pytest.fixture(scope="session")
def simple_production():
    return tt.ProductionSum(
        steady=tt.SteadyProduction(),
        spike=tt.MiyakeSpike(t_start=774.4, duration=0.4, fluence=7.0, profile="boxcar"),
    )


@pytest.fixture(scope="session")
def simple_model(brehm21_cbm, simple_production):
    return tt.RadiocarbonModel(brehm21_cbm, simple_production)


def _make_series(
    t_years,
    d14c_vals,
    *,
    sigma: float = 1.5,
    hemisphere: str = "north",
    growth_season: str | None = None,
    name: str = "series",
):
    season = (
        growth_season
        if growth_season is not None
        else ("april-september" if hemisphere == "north" else "october-march")
    )
    return tt.TreeRingSeries(
        years=np.asarray(t_years, dtype=np.float64),
        d14c=np.asarray(d14c_vals, dtype=np.float64),
        sigma_d14c=np.full(len(t_years), sigma, dtype=np.float64),
        hemisphere=hemisphere,
        calendar="astronomical_ce",
        year_convention="schulman_start",
        seasonal_weights=season,
        name=name,
    )


@pytest.fixture(scope="session")
def synthetic_dataset(simple_model):
    """2 NH + 1 SH synthetic series (mixed 1- and 2-subwindow growth seasons) around 774 CE."""
    rng = np.random.default_rng(42)
    years = np.arange(771.0, 779.0, 1.0, dtype=np.float64)
    placeholders = [
        _make_series(years, np.zeros_like(years), hemisphere="north", name="nh_1"),
        _make_series(years, np.zeros_like(years), hemisphere="north", name="nh_2"),
        _make_series(years, np.zeros_like(years), hemisphere="south", name="sh_1"),
    ]
    ds0 = tt.Dataset(placeholders)
    true_offsets = jnp.array([12.0, 13.2, 11.4], dtype=jnp.float64)
    clean = np.asarray(simple_model.predict_d14c(ds0, offsets=true_offsets, concat=True))
    noisy = clean + rng.normal(0.0, 1.5, size=clean.shape)

    rebuilt = []
    cursor = 0
    for idx, s in enumerate(ds0.series):
        n = s.n_obs
        rebuilt.append(
            _make_series(
                years,
                noisy[cursor : cursor + n],
                sigma=1.5,
                hemisphere=s.hemisphere,
                name=f"tree_{idx}",
            )
        )
        cursor += n
    return tt.Dataset(rebuilt)


@pytest.fixture(scope="session")
def spike_param_space(simple_model):
    return tt.ParameterSpace(
        simple_model,
        [
            tt.ParamSpec("production.spike.t_start", "identity", prior_mean=774.4, prior_sd=0.5),
            tt.ParamSpec(
                "production.spike.fluence", "exp", prior_mean=math.log(7.0), prior_sd=0.5
            ),
        ],
    )


# ===========================================================================
# 1. RadiocarbonModel basics & fluence reporting conventions
# ===========================================================================


class TestRadiocarbonModel:
    def test_steady_state_and_reference_activity(self, simple_model):
        y_ss = simple_model.steady_state()
        assert y_ss.shape == (simple_model.cbm.n_boxes,)
        assert jnp.all(y_ss > 0.0)
        r_ref = simple_model.reference_specific_activity()
        assert r_ref.shape == ()
        assert float(r_ref) > 0.0

    def test_nh_sh_gradient_preserved(self, brehm21_cbm):
        steady_model = tt.RadiocarbonModel(brehm21_cbm, tt.SteadyProduction())
        years = np.arange(770.0, 776.0, 1.0)
        s_nh = _make_series(years, np.zeros(6), hemisphere="north", name="nh")
        s_sh = _make_series(years, np.zeros(6), hemisphere="south", name="sh")
        preds = steady_model.predict_d14c(tt.Dataset([s_nh, s_sh]), concat=False)
        diff = float(jnp.mean(preds[0]) - jnp.mean(preds[1]))
        assert 4.0 < diff < 4.8, f"Expected ~4.38 permil NH-SH gradient, got {diff:.3f}"

    def test_zodiax_paths_and_differentiability(self, brehm21_cbm):
        prod = tt.ProductionSum(
            steady=tt.SteadyProduction(),
            spike=tt.MiyakeSpike(t_start=774.4, duration=0.4, fluence=7.0, profile="supergaussian"),
        )
        m = tt.RadiocarbonModel(brehm21_cbm, prod)
        s = _make_series(np.arange(773.0, 777.0), np.zeros(4), hemisphere="north")
        m2 = m.set("production.spike.t_start", 774.8)
        assert float(m2.get("production.spike.t_start")) == pytest.approx(774.8)
        assert float(m.get("production.spike.t_start")) == pytest.approx(774.4)
        g = jax.grad(lambda ts: jnp.sum(m.set("production.spike.t_start", ts).predict_d14c(s)))(
            jnp.asarray(774.4, dtype=jnp.float64)
        )
        assert jnp.isfinite(g)

    def test_predict_fine_impulse_and_transfer(self, simple_model):
        t_eval = jnp.linspace(773.0, 778.0, 40)
        d14c_nh = simple_model.predict_fine(t_eval, hemisphere="north")
        d14c_sh = simple_model.predict_fine(t_eval, hemisphere="south")
        assert d14c_nh.shape == (40,) and d14c_sh.shape == (40,)
        assert not jnp.allclose(d14c_nh, d14c_sh)

        ir = simple_model.impulse_response(jnp.linspace(0.0, 20.0, 21), hemisphere="north")
        assert ir.shape == (21,)
        assert jnp.all(ir >= 0.0) and float(ir[0]) > float(ir[-1])

        tf = simple_model.transfer_function(jnp.array([2.0 * math.pi / 11.0]), hemisphere="north")
        assert tf.shape == (1,) and jnp.iscomplexobj(tf) and jnp.all(jnp.isfinite(tf))

    def test_fluence_helpers_pi_and_epoch_local(self):
        fluence = 6.6
        atoms = float(tt.RadiocarbonModel.fluence_atoms_cm2(fluence))
        assert atoms == pytest.approx(fluence * float(tt.SECONDS_PER_JULIAN_YEAR), rel=1e-12)

        q0_pi = float(tt.RadiocarbonModel.fluence_q0_years(fluence))
        q0_ep = float(
            tt.RadiocarbonModel.fluence_q0_years(fluence, baseline_d14c_permil=40.0)
        )
        assert q0_pi == pytest.approx(fluence / float(tt.DEFAULT_Q_STD), rel=1e-12)
        assert q0_ep == pytest.approx(q0_pi / 1.04, rel=1e-12)
        assert q0_ep < q0_pi


# ===========================================================================
# 2. ParameterSpace, Exact Marginal Likelihood, Gradients, & Bias Tests
# ===========================================================================


class TestLikelihoodAndLinearPosterior:
    def test_paramspec_bijectors_and_normalized_prior(self, brehm21_cbm):
        from scipy.integrate import quad

        eff = tt.EffectiveCBM(K=3)
        m = tt.RadiocarbonModel(
            eff,
            tt.ProductionSum(
                steady=tt.SteadyProduction(),
                spike=tt.MiyakeSpike(t_start=774.4, duration=0.4, fluence=6.5, skewness=1.5, profile="skewgaussian"),
            ),
        )
        specs = [
            tt.ParamSpec("production.spike.t_start", "identity", 774.4, 0.5),
            tt.ParamSpec("production.spike.fluence", "exp", math.log(6.5), 0.6),
            tt.ParamSpec("production.spike.duration", "softplus", 0.5, 0.4),
            tt.ParamSpec("production.spike.skewness", "sigmoid", 0.0, 0.8, low=-4.0, high=4.0),
            tt.ParamSpec("cbm.log_k_trop_to_sinks", "identity", eff.log_k_trop_to_sinks, 0.3),
            tt.ParamSpec("cbm.sink_tau_params", "identity", eff.sink_tau_params, 0.4),
        ]
        ps = tt.ParameterSpace(m, specs)
        u0 = ps.unconstrain(m)
        m_rt = ps.constrain(u0)
        for sp in specs:
            np.testing.assert_allclose(m_rt.get(sp.path), m.get(sp.path), rtol=1e-11, atol=1e-11)

        # Unconstrained prior density must integrate to 1 (crucial for SMC ln Z)
        ps_1d = tt.ParameterSpace(
            m, [tt.ParamSpec("production.spike.fluence", "exp", math.log(6.5), 0.7)]
        )
        mass, _ = quad(
            lambda u: math.exp(float(ps_1d.log_prior_unconstrained(jnp.array([u])))),
            math.log(6.5) - 8.0,
            math.log(6.5) + 8.0,
        )
        assert mass == pytest.approx(1.0, rel=1e-9)

    def test_marginal_log_likelihood_exact_agreement_and_second_order_grads(self, brehm21_cbm):
        """Mandatory Test 6.1: < 1e-10 agreement with dense Gaussian & joint_log_likelihood integral + 2nd-order check_grads."""
        rng = np.random.default_rng(2026)
        years_nh = np.arange(771.0, 779.0, 1.0)
        years_sh = np.arange(771.0, 778.0, 1.0)
        s_nh = _make_series(years_nh, rng.normal(-15.0, 4.0, len(years_nh)), sigma=1.6, hemisphere="north", name="nh")
        s_sh = _make_series(years_sh, rng.normal(-19.0, 4.0, len(years_sh)), sigma=1.8, hemisphere="south", name="sh")
        ds = tt.Dataset([s_nh, s_sh])

        domain = (766.0, 784.0)
        n_modes = 8
        kernel = tk.quasisep.Matern32(scale=3.5, sigma=0.45)
        prod = tt.ProductionSum(
            steady=tt.SteadyProduction(),
            spike=tt.MiyakeSpike(t_start=774.4, duration=0.45, fluence=6.8, profile="supergaussian"),
            solar=tt.HarmonicSolar.from_polar(periods=11.0, amplitudes=0.0, phases=0.0, t_ref=775.0),
            trend=tt.LinearTrend(slope=0.0, t_ref=775.0, switched_on=False),
            gp=tt.GPProduction(kernel, n_modes=n_modes, domain=domain),
        )
        model = tt.RadiocarbonModel(brehm21_cbm, prod)

        # (a) Check both include_gp_initial_cov=True and False against dense MVN logpdf (< 1e-10)
        for inc_init in (True, False):
            cfg = tt.LinearMarginalConfig(
                offset_mode="hierarchical",
                global_prior_mean=-15.0,
                global_prior_sd=50.0,
                lab_prior_sd=1.5,
                solar_prior_sd=0.4,
                trend_prior_sd=0.2,
                marginalize_solar=True,
                marginalize_trend=True,
                marginalize_gp=True,
                include_gp_initial_cov=inc_init,
            )
            lml = float(tt.marginal_log_likelihood(model, ds, cfg))
            resid = _nonlinear_residual(model, ds, cfg)
            X, mu0, sd0 = _build_marginal_design(model, ds, cfg)
            cov_dense = jnp.diag(jnp.square(ds.sigma_flat)) + (X * jnp.square(sd0)[None, :]) @ X.T
            lml_dense = float(jss.multivariate_normal.logpdf(resid, mean=X @ mu0, cov=cov_dense))
            assert abs(lml - lml_dense) < 1e-10, (
                f"inc_init={inc_init}: marginal_log_likelihood={lml:.12f} vs dense={lml_dense:.12f}"
            )

        # (b1) Exact (< 1e-11) Gaussian integral agreement with explicit joint_log_likelihood
        # over exact continuous LTI terms w = (d_global, d_lab_0, d_lab_1, c_cos, c_sin, slope)
        prod_exact = tt.ProductionSum(
            steady=tt.SteadyProduction(),
            spike=prod.spike,
            solar=prod.solar,
            trend=prod.trend,
        )
        model_exact = tt.RadiocarbonModel(brehm21_cbm, prod_exact)
        cfg_exact = tt.LinearMarginalConfig(
            offset_mode="hierarchical",
            global_prior_mean=-15.0,
            global_prior_sd=50.0,
            lab_prior_sd=1.5,
            solar_prior_sd=0.4,
            trend_prior_sd=0.2,
            marginalize_solar=True,
            marginalize_trend=True,
            marginalize_gp=False,
        )
        lml_exact = float(tt.marginal_log_likelihood(model_exact, ds, cfg_exact))

        y_ss = brehm21_cbm.steady_state_c14()
        trop_idx = jnp.asarray(brehm21_cbm.topology.trop_box_indices, dtype=jnp.int32)
        r_ss = (y_ss / jnp.sum(y_ss[trop_idx])) / brehm21_cbm.relative_reservoir_masses()
        r_series = jnp.array([r_ss[model._box_index_for_series(s)] for s in ds.series])

        prior_mu_ex = jnp.array([-15.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
        prior_sd_ex = jnp.array([50.0, 1.5, 1.5, 0.4, 0.4, 0.2], dtype=jnp.float64)
        cfg_none = tt.LinearMarginalConfig(
            offset_mode="none",
            marginalize_solar=False,
            marginalize_trend=False,
            marginalize_gp=False,
        )

        def log_joint_exact(w: jax.Array) -> jax.Array:
            d_glob = w[0]
            d_labs = w[1:3]
            c_cos, c_sin = w[3:4], w[4:5]
            slope = w[5]
            realized_prod = tt.ProductionSum(
                steady=tt.SteadyProduction(),
                spike=prod.spike,
                solar=tt.HarmonicSolar(periods=11.0, c_cos=c_cos, c_sin=c_sin, t_ref=775.0),
                trend=tt.LinearTrend(slope=slope, t_ref=775.0, switched_on=False),
            )
            realized_model = tt.RadiocarbonModel(brehm21_cbm, realized_prod)
            offsets_s = r_series * d_glob + d_labs
            pred = realized_model.predict_d14c(ds, offsets=offsets_s, concat=True)
            log_lk = jnp.sum(jss.norm.logpdf(ds.d14c_flat, loc=pred, scale=ds.sigma_flat))
            log_pr = jnp.sum(jss.norm.logpdf(w, loc=prior_mu_ex, scale=prior_sd_ex))
            return log_lk + log_pr

        # Check joint_log_likelihood consistency at w = 0 with offset_mode="none"
        m_at_mu = tt.RadiocarbonModel(
            brehm21_cbm,
            tt.ProductionSum(
                steady=tt.SteadyProduction(),
                spike=prod.spike,
                solar=tt.HarmonicSolar(periods=11.0, c_cos=0.0, c_sin=0.0, t_ref=775.0),
                trend=tt.LinearTrend(slope=0.0, t_ref=775.0, switched_on=False),
                gp=tt.GPProduction(kernel, n_modes=n_modes, domain=domain, coefficients=jnp.zeros(n_modes)),
            ),
        )
        assert float(tt.joint_log_likelihood(m_at_mu, ds, cfg=cfg_none)) == pytest.approx(
            float(jnp.sum(jss.norm.logpdf(ds.d14c_flat, loc=m_at_mu.predict_d14c(ds), scale=ds.sigma_flat))),
            abs=1e-12,
        )

        f_ex = log_joint_exact(prior_mu_ex)
        g_ex = jax.grad(log_joint_exact)(prior_mu_ex)
        H_ex = -jax.hessian(log_joint_exact)(prior_mu_ex)
        sign_ex, logdet_H_ex = jnp.linalg.slogdet(0.5 * (H_ex + H_ex.T))
        assert float(sign_ex) > 0.0
        lml_from_exact_integral = float(
            f_ex
            + 0.5 * (g_ex @ jnp.linalg.solve(0.5 * (H_ex + H_ex.T), g_ex))
            + 0.5 * prior_mu_ex.shape[0] * math.log(2.0 * math.pi)
            - 0.5 * logdet_H_ex
        )
        assert abs(lml_exact - lml_from_exact_integral) < 1e-11, (
            f"marginal_log_likelihood={lml_exact:.12f} vs exact integral={lml_from_exact_integral:.12f}"
        )

        # (c) 2nd-order autodiff verification wrt nonlinear spike & GP kernel hyperparameters
        eff_cbm = tt.EffectiveCBM(K=3)
        grad_model = tt.RadiocarbonModel(eff_cbm, prod)
        cfg_grad = tt.LinearMarginalConfig(
            offset_mode="hierarchical",
            marginalize_solar=True,
            marginalize_trend=True,
            marginalize_gp=True,
        )

        def scalar_lml(params_vec: jax.Array) -> jax.Array:
            ts, log_dur, log_fl, log_ell, log_sig, lk_up = params_vec
            new_k = tk.quasisep.Matern32(scale=jnp.exp(log_ell), sigma=jnp.exp(log_sig))
            m_v = (
                grad_model.set("production.spike.t_start", ts)
                .set("production.spike.duration", jnp.exp(log_dur))
                .set("production.spike.fluence", jnp.exp(log_fl))
                .set("production.gp.kernel", new_k)
                .set("cbm.log_k_trop_to_sinks", jnp.array([lk_up]))
            )
            return tt.marginal_log_likelihood(m_v, ds, cfg_grad)

        p0 = jnp.array(
            [
                774.4,
                math.log(0.45),
                math.log(6.8),
                math.log(3.5),
                math.log(0.45),
                float(eff_cbm.log_k_trop_to_sinks[0]),
            ],
            dtype=jnp.float64,
        )
        check_grads(scalar_lml, (p0,), order=2, atol=1e-4, rtol=1e-4)

    def test_staggered_start_ramp_and_spike_fluence_bias(self, brehm21_cbm):
        """Mandatory Test 6.2: hierarchical offset + trend marginalization recovers unbiased fluence where v1 mean(y[:4]) fails."""
        true_fluence = 8.0
        true_prod = tt.ProductionSum(
            steady=tt.SteadyProduction(),
            trend=tt.LinearTrend(slope=0.07, t_ref=770.0, switched_on=False),
            spike=tt.MiyakeSpike(t_start=774.5, duration=0.4, fluence=true_fluence, profile="boxcar"),
        )
        true_model = tt.RadiocarbonModel(brehm21_cbm, true_prod)

        # Three staggered NH tree-ring series (the third begins at 773 CE so its first 4 rings span the 774 CE spike!)
        y0_yrs = np.arange(756.0, 785.0, 1.0)
        y1_yrs = np.arange(766.0, 785.0, 1.0)
        y2_yrs = np.arange(773.0, 785.0, 1.0)
        raw_series = [
            _make_series(y0_yrs, np.zeros_like(y0_yrs), sigma=1.2, hemisphere="north", name="long_tree"),
            _make_series(y1_yrs, np.zeros_like(y1_yrs), sigma=1.2, hemisphere="north", name="mid_tree"),
            _make_series(y2_yrs, np.zeros_like(y2_yrs), sigma=1.2, hemisphere="north", name="short_tree"),
        ]
        ds_tmpl = tt.Dataset(raw_series)
        y_ss = brehm21_cbm.steady_state_c14()
        trop_idx = jnp.asarray(brehm21_cbm.topology.trop_box_indices, dtype=jnp.int32)
        r_nh = float(((y_ss / jnp.sum(y_ss[trop_idx])) / brehm21_cbm.relative_reservoir_masses())[brehm21_cbm.box_index("Troposphere", "north")])
        true_offsets = jnp.array([35.0 * r_nh - 0.9, 35.0 * r_nh + 0.3, 35.0 * r_nh + 0.8])
        clean = np.asarray(true_model.predict_d14c(ds_tmpl, offsets=true_offsets, concat=True))

        rng = np.random.default_rng(314159)
        noisy = clean + rng.normal(0.0, 0.6, size=clean.shape)

        obs_series = []
        v1_clamped_series = []
        cursor = 0
        for s in raw_series:
            n = s.n_obs
            seg = noisy[cursor : cursor + n]
            obs_series.append(_make_series(np.asarray(s.years_ce), seg, sigma=1.2, hemisphere="north", name=s.name))
            # v1 baseline hack: subtract mean(y[:4]) independently from each series
            v1_seg = seg - float(np.mean(seg[:4]))
            v1_clamped_series.append(
                _make_series(np.asarray(s.years_ce), v1_seg, sigma=1.2, hemisphere="north", name=s.name)
            )
            cursor += n

        ds_obs = tt.Dataset(obs_series)
        ds_v1 = tt.Dataset(v1_clamped_series)

        fit_prod = tt.ProductionSum(
            steady=tt.SteadyProduction(),
            trend=tt.LinearTrend(slope=0.0, t_ref=770.0, switched_on=False),
            spike=tt.MiyakeSpike(t_start=774.5, duration=0.4, fluence=6.0, profile="boxcar"),
        )
        fit_model = tt.RadiocarbonModel(brehm21_cbm, fit_prod)
        ps = tt.ParameterSpace(
            fit_model,
            [tt.ParamSpec("production.spike.fluence", "exp", prior_mean=math.log(6.0), prior_sd=1.0)],
        )

        # ticktack 2.0 hierarchical offset + trend marginalization
        cfg_hier = tt.LinearMarginalConfig(offset_mode="hierarchical", marginalize_trend=True)
        res_hier = tt.fit_map(fit_model, ds_obs, ps, cfg_hier, method="BFGS", max_steps=100)
        fl_hier = float(res_hier.theta_map["production.spike.fluence"])

        # Legacy v1 approach: pre-subtract mean(y[:4]) with zero offset/trend marginalization
        v1_model = tt.RadiocarbonModel(
            brehm21_cbm,
            tt.ProductionSum(
                steady=tt.SteadyProduction(),
                spike=tt.MiyakeSpike(t_start=774.5, duration=0.4, fluence=6.0, profile="boxcar"),
            ),
        )
        ps_v1 = tt.ParameterSpace(
            v1_model,
            [tt.ParamSpec("production.spike.fluence", "exp", prior_mean=math.log(6.0), prior_sd=1.0)],
        )
        cfg_v1 = tt.LinearMarginalConfig(offset_mode="none", marginalize_trend=False)
        res_v1 = tt.fit_map(v1_model, ds_v1, ps_v1, cfg_v1, method="BFGS", max_steps=100)
        fl_v1 = float(res_v1.theta_map["production.spike.fluence"])

        err_hier = abs(fl_hier - true_fluence)
        err_v1 = abs(fl_v1 - true_fluence)
        assert err_hier < 0.6, f"Hierarchical fluence recovery biased: {fl_hier:.3f} vs {true_fluence}"
        assert err_v1 > 1.5, f"Expected v1 mean(y[:4]) clamp to be strongly biased, got {fl_v1:.3f}"
        assert err_hier < 0.25 * err_v1

    def test_linear_posterior_continuous_trajectories_and_offset_modes(
        self, brehm21_cbm, synthetic_dataset
    ):
        kernel = tk.quasisep.Matern32(scale=4.0, sigma=0.3)
        prod = tt.ProductionSum(
            steady=tt.SteadyProduction(),
            solar=tt.HarmonicSolar.from_polar(amplitudes=0.0, phases=0.0),
            gp=tt.GPProduction(kernel, n_modes=16, domain=(765.0, 785.0)),
            spike=tt.MiyakeSpike(t_start=774.4, duration=0.4, fluence=7.0, profile="boxcar"),
        )
        m = tt.RadiocarbonModel(brehm21_cbm, prod)
        t_fine = jnp.linspace(771.0, 779.0, 40)

        for mode in ("hierarchical", "per_series", "shared", "none"):
            cfg = tt.LinearMarginalConfig(
                offset_mode=mode, marginalize_solar=True, marginalize_gp=True
            )
            lp = tt.linear_posterior(
                m,
                synthetic_dataset,
                cfg,
                rng_key=jax.random.PRNGKey(9),
                n_draws=12,
                t_fine=t_fine,
            )
            assert lp.q_fine_mean is not None and lp.q_fine_mean.shape == (40,)
            assert lp.d14c_fine_mean is not None and lp.d14c_fine_mean.shape == (40,)
            assert lp.q_fine_draws is not None and lp.q_fine_draws.shape == (12, 40)
            assert lp.d14c_fine_draws is not None and lp.d14c_fine_draws.shape == (12, 40)
            assert lp.ring_pred_mean is not None and lp.ring_pred_mean.shape == (synthetic_dataset.n_obs,)
            assert jnp.all(jnp.isfinite(lp.q_fine_mean))
            assert jnp.all(jnp.isfinite(lp.d14c_fine_mean))


# ===========================================================================
# 3. MAP Fitting (BFGS, LevenbergMarquardt, adam)
# ===========================================================================


class TestMAPFitting:
    def test_fit_map_bfgs_and_lm_and_adam(
        self, simple_model, synthetic_dataset, spike_param_space
    ):
        cfg = tt.LinearMarginalConfig(offset_mode="hierarchical")
        init_m = simple_model.set("production.spike.t_start", 774.1).set(
            "production.spike.fluence", 5.0
        )

        res_bfgs = tt.fit_map(init_m, synthetic_dataset, spike_param_space, cfg, method="BFGS", max_steps=80)
        res_lm = tt.fit_map(init_m, synthetic_dataset, spike_param_space, cfg, method="LevenbergMarquardt", max_steps=80)
        res_adam = tt.fit_map(init_m, synthetic_dataset, spike_param_space, cfg, method="adam", max_steps=120)

        assert jnp.allclose(res_bfgs.u_map, res_lm.u_map, atol=1e-3)
        assert float(res_adam.log_posterior) > float(
            tt.marginal_log_posterior(init_m, synthetic_dataset, spike_param_space, spike_param_space.unconstrain(init_m), cfg)
        )
        eigvals = jnp.linalg.eigvalsh(res_bfgs.laplace_cov)
        assert jnp.all(eigvals > 0.0)


# ===========================================================================
# 4. NUTS, SVI, & LogGPProduction Positivity
# ===========================================================================


class TestNUTSAndSVI:
    def test_sample_nuts_two_chains_rhat_and_linear_reconstruction(
        self, simple_model, synthetic_dataset, spike_param_space
    ):
        t_fine = jnp.linspace(772.0, 778.0, 20)
        idata = tt.sample_nuts(
            simple_model,
            synthetic_dataset,
            spike_param_space,
            num_warmup=80,
            num_samples=100,
            num_chains=2,
            reconstruct_linear=True,
            t_fine=t_fine,
            rng_key=jax.random.PRNGKey(42),
        )
        assert hasattr(idata, "posterior")
        assert "offset_global" in idata.posterior.data_vars
        assert "offset_lab" in idata.posterior.data_vars
        assert hasattr(idata, "posterior_predictive")
        assert idata.posterior_predictive["q_fine"].shape == (2, 100, 20)
        assert idata.posterior_predictive["d14c_fine"].shape == (2, 100, 20)

        rhat_ds = az.rhat(idata)
        for v in ("production__spike__t_start", "production__spike__fluence"):
            assert float(rhat_ds[v].max()) < 1.05

    def test_loggp_production_map_and_svi_strictly_positive(
        self, brehm21_cbm, synthetic_dataset
    ):
        """Mandatory Test 6.4: LogGPProduction MAP and SVI strictly positive Q(t) > 0."""
        domain = (768.0, 782.0)
        n_modes = 12
        kernel = tk.quasisep.Matern32(scale=1.5, sigma=0.8)
        loggp_model = tt.RadiocarbonModel(
            brehm21_cbm,
            tt.ProductionSum(
                steady=tt.SteadyProduction(),
                log_gp=tt.LogGPProduction(
                    kernel, jnp.zeros(n_modes, dtype=jnp.float64), domain=domain
                ),
            ),
            solver=tt.ExactLinearSolver(oversample=24),
        )
        ps = tt.ParameterSpace(
            loggp_model,
            [tt.ParamSpec("production.log_gp.coefficients", "identity", 0.0, 1.0)],
        )
        cfg = tt.LinearMarginalConfig(offset_mode="hierarchical", marginalize_gp=False)

        u_zero = ps.unconstrain(loggp_model)
        lp_init = float(tt.joint_log_posterior(loggp_model, synthetic_dataset, ps, u_zero, cfg))

        # (a) MAP BFGS
        map_res = tt.fit_map(loggp_model, synthetic_dataset, ps, cfg, method="BFGS", max_steps=120)
        assert float(map_res.log_posterior) > lp_init
        t_eval = jnp.linspace(768.0, 782.0, 120)
        q_map = map_res.fitted_model.production(t_eval, q_std=loggp_model.q_std)
        assert jnp.all(q_map > 0.0), f"LogGP MAP Q(t) had non-positive values: min={float(q_map.min())}"

        # (b) AutoMVN SVI
        svi_res = tt.fit_svi(
            loggp_model,
            synthetic_dataset,
            ps,
            cfg,
            guide="mvn",
            num_steps=250,
            lr=0.03,
            num_samples=80,
            rng_key=jax.random.PRNGKey(11),
        )
        assert jnp.all(jnp.isfinite(svi_res.losses))
        coeff_draws = jnp.asarray(svi_res.idata.posterior["production__log_gp__coefficients"])[0]
        assert coeff_draws.shape == (80, n_modes)
        q_draws = jax.vmap(
            lambda c: loggp_model.set("production.log_gp.coefficients", c).production(
                t_eval, q_std=loggp_model.q_std
            )
        )(coeff_draws)
        assert jnp.all(q_draws > 0.0)


# ===========================================================================
# 5. Tempered SMC Analytical Multivariate Gaussian & EffectiveCBM(K=2, 3)
# ===========================================================================


class TestSMC:
    def test_smc_analytical_2d_gaussian_log_evidence(self, brehm21_cbm):
        """Mandatory Test 6.3a: exact 2D affine-Gaussian ln Z test with calibrated sigma_lnZ > 0."""
        rng = np.random.default_rng(777)
        years = np.arange(770.0, 780.0, 1.0)
        base_prod = tt.ProductionSum(
            steady=tt.SteadyProduction(1.76),
            trend=tt.LinearTrend(slope=0.0, t_ref=775.0, switched_on=False),
        )
        model = tt.RadiocarbonModel(brehm21_cbm, base_prod)

        true_model = model.set("production.steady.rate", 1.88).set("production.trend.slope", 0.04)
        s0 = _make_series(years, np.zeros_like(years), sigma=1.8, hemisphere="north")
        d_obs = np.asarray(true_model.predict_d14c(s0)) + rng.normal(0.0, 1.8, size=len(years))
        ds = tt.Dataset([_make_series(years, d_obs, sigma=1.8, hemisphere="north")])

        ps = tt.ParameterSpace(
            model,
            [
                tt.ParamSpec("production.steady.rate", "identity", prior_mean=1.76, prior_sd=0.25),
                tt.ParamSpec("production.trend.slope", "identity", prior_mean=0.0, prior_sd=0.08),
            ],
        )
        cfg = tt.LinearMarginalConfig(offset_mode="none", marginalize_trend=False)

        # Exact analytical marginal log-evidence for affine forward model d ~ N(m(mu0) + J(u - mu0), Sigma_d)
        mu0 = ps.prior_means
        sd0 = ps.prior_sds
        m_mu0 = ps.constrain(mu0).predict_d14c(ds, concat=True)
        J = jax.jacfwd(lambda u: ps.constrain(u).predict_d14c(ds, concat=True))(mu0)
        cov_marg = jnp.diag(jnp.square(ds.sigma_flat)) + (J * jnp.square(sd0)[None, :]) @ J.T
        ln_z_exact = float(jss.multivariate_normal.logpdf(ds.d14c_flat, mean=m_mu0, cov=cov_marg))

        smc_res = tt.sample_smc(
            model,
            ds,
            ps,
            cfg,
            num_particles=400,
            target_ess=0.75,
            num_mcmc_steps=5,
            rng_key=jax.random.PRNGKey(2026),
        )
        ln_z_hat = float(smc_res.log_evidence)
        sigma_lnz = float(smc_res.log_evidence_err)

        assert sigma_lnz > 0.0, f"Expected calibrated sigma_lnZ > 0, got {sigma_lnz}"
        tol = max(3.5 * sigma_lnz, 0.12)
        assert abs(ln_z_hat - ln_z_exact) < tol, (
            f"SMC ln Z={ln_z_hat:.4f} +/- {sigma_lnz:.4f} missed exact {ln_z_exact:.4f} (tol={tol:.4f})"
        )
        betas = np.asarray(smc_res.tempering_betas)
        assert betas[0] == pytest.approx(0.0) and betas[-1] == pytest.approx(1.0)
        assert np.all(np.diff(betas) > 0.0)

    def test_smc_effective_cbm_k2_and_k3(self, synthetic_dataset):
        """Mandatory Test 6.3b: EffectiveCBM(K=2, 3) SMC with dynamic CBM rate & timescale arrays in ParameterSpace."""
        cfg = tt.LinearMarginalConfig(offset_mode="hierarchical", marginalize_solar=False, marginalize_gp=False)
        for k in (2, 3):
            cbm_k = tt.EffectiveCBM(K=k)
            prod_k = tt.ProductionSum(
                steady=tt.SteadyProduction(),
                spike=tt.MiyakeSpike(t_start=774.4, duration=0.4, fluence=7.0, profile="boxcar"),
            )
            m_k = tt.RadiocarbonModel(cbm_k, prod_k)
            specs = [
                tt.ParamSpec("cbm.log_k_strat_to_trop", "identity", float(cbm_k.log_k_strat_to_trop), 0.3),
                tt.ParamSpec("cbm.log_k_trop_to_strat", "identity", float(cbm_k.log_k_trop_to_strat), 0.3),
                tt.ParamSpec("cbm.log_k_trop_to_sinks", "identity", cbm_k.log_k_trop_to_sinks, 0.3),
                tt.ParamSpec("cbm.sink_tau_params", "identity", cbm_k.sink_tau_params, 0.4),
                tt.ParamSpec("production.spike.fluence", "exp", math.log(7.0), 0.4),
            ]
            ps_k = tt.ParameterSpace(m_k, specs)
            assert ps_k.n_params == (2 * k - 2) + 1

            res = tt.sample_smc(
                m_k,
                synthetic_dataset,
                ps_k,
                cfg,
                num_particles=60,
                num_mcmc_steps=3,
                rng_key=jax.random.PRNGKey(50 + k),
            )
            assert jnp.isfinite(res.log_evidence)
            assert float(res.log_evidence_err) > 0.0
            assert hasattr(res.idata, "posterior")


# ===========================================================================
# 6. Notebook Validation (stdlib json, executed cells, single output per cell)
# ===========================================================================


NOTEBOOKS = [
    "notebooks/v2/01_miyake_774ce_inference.ipynb",
    "notebooks/v2/02_effective_cbm_fisher_smc.ipynb",
    "notebooks/v2/03_nonparametric_gp_pod.ipynb",
]


def _load_notebook(rel_path: str) -> dict:
    path = ROOT / rel_path
    assert path.exists(), f"Missing notebook: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("rel_path", NOTEBOOKS)
def test_notebook_structure_and_execution(rel_path: str):
    nb = _load_notebook(rel_path)
    assert nb.get("nbformat") == 4
    cells = nb.get("cells", [])
    assert len(cells) >= 5
    assert cells[0].get("cell_type") == "markdown", "Notebook must open with prose markdown."
    assert cells[-1].get("cell_type") == "markdown", "Notebook must close with prose markdown."

    n_png_cells = 0
    prev_type = None
    for idx, cell in enumerate(cells):
        ctype = cell.get("cell_type")
        src = "".join(cell.get("source", []))
        assert ".png" not in src and "savefig" not in src, (
            f"{rel_path} cell {idx} saves a .png file instead of inline display."
        )
        if ctype == "code":
            assert prev_type == "markdown", (
                f"{rel_path} code cell {idx} must be preceded by a markdown cell."
            )
            assert cell.get("execution_count") is not None, (
                f"{rel_path} code cell {idx} was not executed."
            )
            outputs = cell.get("outputs", [])
            assert len(outputs) > 0, f"{rel_path} code cell {idx} has empty outputs."

            for out in outputs:
                assert out.get("output_type") != "error", (
                    f"{rel_path} cell {idx} raised {out.get('ename')}: {out.get('evalue')}"
                )

            has_image = any(
                "image/png" in out.get("data", {})
                for out in outputs
                if out.get("output_type") in ("display_data", "execute_result")
            )
            has_stream_or_text_result = any(
                out.get("output_type") == "stream"
                or (
                    out.get("output_type") == "execute_result"
                    and "image/png" not in out.get("data", {})
                )
                for out in outputs
            )
            assert has_image or has_stream_or_text_result, (
                f"{rel_path} code cell {idx} has neither image/png nor text output."
            )
            assert not (has_image and has_stream_or_text_result), (
                f"{rel_path} code cell {idx} mixes text and image/png in the same cell."
            )
            if has_image:
                n_png_cells += 1
        prev_type = ctype

    assert n_png_cells >= 1, f"{rel_path} must contain at least one executed inline figure."


def test_no_stray_png_files_in_notebooks_v2():
    pngs = list((ROOT / "notebooks" / "v2").glob("*.png"))
    assert pngs == [], f"Found stray .png files in notebooks/v2/: {pngs}"
