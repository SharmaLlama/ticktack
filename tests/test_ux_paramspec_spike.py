"""Tests for physical-unit ``ParamSpec`` constructors, ``MiyakeSpike.param_specs``,
``zodiax.Base`` PyTree upgrades on ``ParamSpec`` / ``LinearMarginalConfig``, and
Rule 7 ``MAPResult.model`` single-spelling cleanup.
"""

from __future__ import annotations

from decimal import Decimal, getcontext
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate
import zodiax as zdx

import ticktack as tt


# ===========================================================================
# 1. Bijector round-trips, Jacobians, quantiles & quadrature normalization
# ===========================================================================


class TestParamSpecPhysicalConstructorsAndQuadrature:
    def test_normal_scalar_and_vector_roundtrip_and_jacobian(self) -> None:
        spec_s = tt.ParamSpec.normal("production.spike.t_start", mean=774.58, sd=0.47)
        assert spec_s.bijector == "identity"
        np.testing.assert_allclose(
            float(spec_s.constrain(spec_s.prior_mean)), 774.58, rtol=1e-13, atol=0.0
        )
        for u_val in (-3.7, -0.42, 0.0, 1.85, 774.91):
            u_arr = jnp.asarray(u_val, dtype=jnp.float64)
            np.testing.assert_allclose(
                float(spec_s.unconstrain(spec_s.constrain(u_arr))),
                u_val,
                rtol=1e-12,
                atol=1e-12,
            )
            ladj_autodiff = jnp.log(jnp.abs(jax.grad(spec_s.constrain)(u_arr)))
            np.testing.assert_allclose(
                float(spec_s.log_abs_det_jacobian(u_arr)),
                float(ladj_autodiff),
                rtol=1e-13,
                atol=1e-14,
            )

        mean_v = np.array([-0.65, 0.35, 1.15], dtype=np.float64)
        sd_v = np.array([0.41, 0.73, 0.92], dtype=np.float64)
        spec_v = tt.ParamSpec.normal("cbm.alpha", mean=mean_v, sd=sd_v)
        np.testing.assert_allclose(
            np.asarray(spec_v.constrain(spec_v.prior_mean)), mean_v, rtol=1e-13
        )
        u_vec = jnp.array([-1.3, 0.77, -0.24], dtype=jnp.float64)
        np.testing.assert_allclose(
            np.asarray(spec_v.unconstrain(spec_v.constrain(u_vec))),
            np.asarray(u_vec),
            rtol=1e-12,
            atol=1e-12,
        )
        _, ladj_vec = jnp.linalg.slogdet(jax.jacfwd(spec_v.constrain)(u_vec))
        np.testing.assert_allclose(
            float(spec_v.log_abs_det_jacobian(u_vec)),
            float(ladj_vec),
            rtol=1e-13,
            atol=1e-14,
        )

    def test_lognormal_scalar_and_vector_quantiles_and_jacobian(self) -> None:
        med_s, sig_s = 6.42, 0.53
        spec_s = tt.ParamSpec.lognormal(
            "production.spike.fluence", median=med_s, sigma_log=sig_s
        )
        assert spec_s.bijector == "exp"
        pm_s = float(spec_s.prior_mean)
        ps_s = float(spec_s.prior_sd)
        np.testing.assert_allclose(
            float(spec_s.constrain(pm_s)), med_s, rtol=1e-13, atol=0.0
        )
        np.testing.assert_allclose(
            float(spec_s.constrain(pm_s + ps_s)),
            med_s * math.exp(sig_s),
            rtol=1e-13,
        )
        np.testing.assert_allclose(
            float(spec_s.constrain(pm_s - ps_s)),
            med_s * math.exp(-sig_s),
            rtol=1e-13,
        )
        for u_val in (-2.85, -0.63, 0.41, 1.93, 3.45):
            u_arr = jnp.asarray(u_val, dtype=jnp.float64)
            np.testing.assert_allclose(
                float(spec_s.unconstrain(spec_s.constrain(u_arr))),
                u_val,
                rtol=1e-12,
                atol=1e-12,
            )
            ladj_autodiff = jnp.log(jnp.abs(jax.grad(spec_s.constrain)(u_arr)))
            np.testing.assert_allclose(
                float(spec_s.log_abs_det_jacobian(u_arr)),
                float(ladj_autodiff),
                rtol=1e-13,
                atol=1e-14,
            )

        med_v = np.array([0.27, 4.85, 19.3], dtype=np.float64)
        sig_v = np.array([0.31, 0.64, 0.88], dtype=np.float64)
        spec_v = tt.ParamSpec.lognormal("vec_leaf", median=med_v, sigma_log=sig_v)
        np.testing.assert_allclose(
            np.asarray(spec_v.constrain(spec_v.prior_mean)), med_v, rtol=1e-13
        )
        np.testing.assert_allclose(
            np.asarray(spec_v.constrain(spec_v.prior_mean + spec_v.prior_sd)),
            med_v * np.exp(sig_v),
            rtol=1e-13,
        )
        u_vec = jnp.array([-1.45, 0.82, 2.37], dtype=jnp.float64)
        np.testing.assert_allclose(
            np.asarray(spec_v.unconstrain(spec_v.constrain(u_vec))),
            np.asarray(u_vec),
            rtol=1e-12,
            atol=1e-12,
        )
        _, ladj_vec = jnp.linalg.slogdet(jax.jacfwd(spec_v.constrain)(u_vec))
        np.testing.assert_allclose(
            float(spec_v.log_abs_det_jacobian(u_vec)),
            float(ladj_vec),
            rtol=1e-13,
            atol=1e-13,
        )

    def test_bounded_midpoint_asymmetric_and_narrow_interval_cancellation(self) -> None:
        getcontext().prec = 80
        # Default midpoint (median=None) and unimodality default logit_sd = 1.2 <= sqrt(2)
        spec_mid = tt.ParamSpec.bounded("x", -1.35, 4.85)
        assert spec_mid.bijector == "sigmoid"
        assert spec_mid.prior_mean == 0.0
        assert spec_mid.prior_sd == 1.2
        assert float(spec_mid.prior_sd) <= math.sqrt(2.0)
        # Verify induced physical log-density is strictly concave (unimodal) at u = 0
        d2_log_p_theta = jax.grad(
            jax.grad(
                lambda u: spec_mid.log_prior_unconstrained(u)
                - spec_mid.log_abs_det_jacobian(u)
            )
        )(jnp.asarray(0.0, dtype=jnp.float64))
        assert float(d2_log_p_theta) < 0.0
        np.testing.assert_allclose(
            float(spec_mid.constrain(spec_mid.prior_mean)),
            0.5 * (-1.35 + 4.85),
            rtol=1e-13,
            atol=1e-15,
        )

        # Sweep narrow (1e-4) through wide (10.0) intervals and asymmetric medians
        intervals = [
            (0.01, 0.01 + 1e-4),
            (0.05, 0.15),
            (0.01, 1.5),
            (-5.0, 5.0),
            (2.5, 12.5),
        ]
        fracs = [1e-8, 0.02, 0.5 + 1e-14, 0.5 + 1e-10, 0.96, 1.0 - 1e-8]
        for lo, hi in intervals:
            span = hi - lo
            for frac in fracs:
                med = (lo + frac * span) if frac <= 0.5 else (hi - (1.0 - frac) * span)
                spec = tt.ParamSpec.bounded("dur", lo, hi, median=med, logit_sd=1.25)
                np.testing.assert_allclose(
                    float(spec.constrain(spec.prior_mean)),
                    med,
                    rtol=1e-13,
                    atol=1e-15,
                )
                # Verify unconstrain(med) == prior_mean without midpoint cancellation or 1e-12 clamp
                np.testing.assert_allclose(
                    float(spec.unconstrain(med)),
                    float(spec.prior_mean),
                    rtol=1e-13,
                    atol=1e-15,
                )
                # Compare logit prior_mean directly against 80-digit Decimal oracle
                med_d = Decimal.from_float(float(med))
                lo_d = Decimal.from_float(float(lo))
                hi_d = Decimal.from_float(float(hi))
                u_exact = float((med_d - lo_d).ln() - (hi_d - med_d).ln())
                np.testing.assert_allclose(
                    float(spec.prior_mean),
                    u_exact,
                    rtol=1e-11,
                    atol=1e-14,
                )

            # Round-trip and Jacobian at generic non-zero u != 0
            spec_rt = tt.ParamSpec.bounded(
                "dur", lo, hi, median=lo + 0.19 * span, logit_sd=1.25
            )
            for u_val in (-4.1, -0.73, 0.38, 3.65):
                u_arr = jnp.asarray(u_val, dtype=jnp.float64)
                np.testing.assert_allclose(
                    float(spec_rt.unconstrain(spec_rt.constrain(u_arr))),
                    u_val,
                    rtol=1e-12,
                    atol=1e-12,
                )
                ladj_ad = jnp.log(jnp.abs(jax.grad(spec_rt.constrain)(u_arr)))
                np.testing.assert_allclose(
                    float(spec_rt.log_abs_det_jacobian(u_arr)),
                    float(ladj_ad),
                    rtol=1e-13,
                    atol=1e-13,
                )

        # Extreme tail (theta - low) / (high - low) = 1e-20 (< 1e-12, no premature clamp!)
        spec_unit = tt.ParamSpec.bounded("u01", 0.0, 1.0)
        np.testing.assert_allclose(
            float(spec_unit.unconstrain(1e-20)),
            math.log(1e-20),
            rtol=1e-13,
        )

        # Raw np.float32 narrow-interval sweep (1e-7..1e0) promoted cleanly without float() wrap
        for base in (0.01, 1.0, 0.0):
            for exp in (-7, -6, -5, -4, -3, -2, -1, 0):
                span32 = 10.0**exp
                lo32 = np.float32(base - 0.5 * span32 if base == 0.0 else base)
                hi32 = np.float32(
                    base + 0.5 * span32 if base == 0.0 else base + span32
                )
                if not (float(hi32) > float(lo32)):
                    continue
                for frac32 in (0.03, 0.5 + 1e-3, 0.97):
                    med32 = np.float32(
                        float(lo32) + frac32 * (float(hi32) - float(lo32))
                    )
                    if not (float(lo32) < float(med32) < float(hi32)):
                        continue
                    spec32 = tt.ParamSpec.bounded(
                        "dur32", lo32, hi32, median=med32, logit_sd=np.float32(1.2)
                    )
                    lo_d = Decimal.from_float(float(lo32))
                    hi_d = Decimal.from_float(float(hi32))
                    med_d = Decimal.from_float(float(med32))
                    u_ref32 = float((med_d - lo_d).ln() - (hi_d - med_d).ln())
                    assert abs(float(spec32.prior_mean)) < 60.0  # unclipped interior
                    np.testing.assert_allclose(
                        float(spec32.prior_mean), u_ref32, rtol=1e-13, atol=1e-15
                    )
                    np.testing.assert_allclose(
                        float(spec32.unconstrain(med32)),
                        u_ref32,
                        rtol=1e-13,
                        atol=1e-15,
                    )
                    np.testing.assert_allclose(
                        float(spec32.constrain(spec32.prior_mean)),
                        float(med32),
                        rtol=1e-13,
                        atol=1e-15,
                    )

        # 1D vector leaf bounded spec
        med_vec = np.array([0.04, 0.5, 1.85], dtype=np.float64)
        spec_vec = tt.ParamSpec.bounded(
            "vec_b", 0.01, 2.0, median=med_vec, logit_sd=np.array([1.1, 1.2, 1.35])
        )
        np.testing.assert_allclose(
            np.asarray(spec_vec.constrain(spec_vec.prior_mean)), med_vec, rtol=1e-13
        )

    def test_quadrature_normalization_and_negative_jacobian_control(self) -> None:
        cbm = tt.load_preset("Brehm21")
        spike = tt.MiyakeSpike(t_start=774.5, duration=0.06, fluence=6.4)
        model = tt.RadiocarbonModel(cbm, spike)

        cases = [
            (
                "normal",
                tt.ParamSpec.normal("production.t_start", mean=774.37, sd=0.73),
                774.37 - 12.0 * 0.73,
                774.37 + 12.0 * 0.73,
                False,
            ),
            (
                "lognormal",
                tt.ParamSpec.lognormal(
                    "production.fluence", median=6.4, sigma_log=0.53
                ),
                0.0,
                np.inf,
                True,
            ),
            (
                "bounded_sharp",
                tt.ParamSpec.bounded(
                    "production.duration", 0.01, 1.5, median=0.06, logit_sd=1.2
                ),
                0.01,
                1.5,
                True,
            ),
            (
                "bounded_narrow",
                tt.ParamSpec.bounded(
                    "production.duration",
                    0.01,
                    0.0101,
                    median=0.010023,
                    logit_sd=1.35,
                ),
                0.01,
                0.0101,
                True,
            ),
        ]

        for label, spec, supp_lo, supp_hi, is_nonlinear in cases:
            ps = tt.ParameterSpace(model, [spec])
            pm = float(spec.prior_mean)
            sd = float(spec.prior_sd)

            # 1. Unconstrained density integrates to 1
            int_u, _ = integrate.quad(
                lambda u: math.exp(float(spec.log_prior_unconstrained(u))),
                pm - 14.0 * sd,
                pm + 14.0 * sd,
                epsabs=1e-13,
                epsrel=1e-13,
                limit=200,
            )
            assert int_u == pytest.approx(1.0, abs=1e-11), f"{label} unconstrained quad={int_u}"

            # 2. Physical constrained density p_theta(theta) = p_u(u(theta)) / |J(u(theta))| integrates to 1
            def _exact_u(theta: float, _s: tt.ParamSpec = spec) -> float:
                if _s.bijector == "sigmoid":
                    return math.log(theta - _s.low) - math.log(_s.high - theta)
                if _s.bijector == "exp":
                    return math.log(theta)
                return theta

            def _pdf_theta(
                theta: float,
                _s: tt.ParamSpec = spec,
                _ps: tt.ParameterSpace = ps,
            ) -> float:
                u = _exact_u(theta, _s)
                lp_spec = float(_s.log_prior_unconstrained(u)) - float(
                    _s.log_abs_det_jacobian(u)
                )
                lp_ps = float(
                    _ps.log_prior_unconstrained(
                        jnp.array([u], dtype=jnp.float64), target_space="constrained"
                    )
                )
                assert abs(lp_spec - lp_ps) < 1e-13
                return math.exp(lp_spec)

            int_theta, _ = integrate.quad(
                _pdf_theta,
                supp_lo,
                supp_hi,
                epsabs=1e-13,
                epsrel=1e-13,
                limit=250,
            )
            assert int_theta == pytest.approx(1.0, abs=1e-11), (
                f"{label} constrained quad={int_theta}"
            )

            # 3. Negative control: dropping -log_abs_det_jacobian breaks normalization!
            if is_nonlinear:
                int_missing_jac, _ = integrate.quad(
                    lambda th, _s=spec: math.exp(
                        float(_s.log_prior_unconstrained(_exact_u(th, _s)))
                    ),
                    supp_lo,
                    supp_hi if np.isfinite(supp_hi) else 80.0,
                    epsabs=1e-8,
                    epsrel=1e-8,
                    limit=150,
                )
                assert abs(int_missing_jac - 1.0) > 0.5, (
                    f"{label} negative control failed: integral without Jacobian was {int_missing_jac}"
                )


# ===========================================================================
# 2. zodiax.Base PyTree methods, hashing, and @eqx.filter_jit caching
# ===========================================================================


class TestZodiaxPyTreeAndJitCaching:
    def test_paramspec_and_linearmarginalconfig_zdx_get_set_and_hashing(self) -> None:
        spec1 = tt.ParamSpec.bounded(
            "production.spike.duration", 0.05, 2.0, median=0.25, logit_sd=1.2
        )
        spec2 = tt.ParamSpec.bounded(
            "production.spike.duration", 0.05, 2.0, median=0.25, logit_sd=1.2
        )
        assert isinstance(spec1, zdx.Base)
        assert spec1 == spec2
        assert hash(spec1) == hash(spec2)
        assert spec1.get("path", to_array=False) == "production.spike.duration"
        np.testing.assert_allclose(float(spec1.get("low")), 0.05)

        spec_updated = spec1.set("low", 0.01).set(prior_sd=0.95)
        assert isinstance(spec_updated, tt.ParamSpec)
        assert spec_updated.low == 0.01
        assert spec_updated.prior_sd == 0.95
        assert spec1.low == 0.05  # immutable

        spec_dict = spec1.set({"low": 0.02, "high": 1.8})
        assert (spec_dict.low, spec_dict.high) == (0.02, 1.8)

        # Broadcast scalar across list of paths, tuple dict keys, and single-list array values
        spec_id = tt.ParamSpec.normal("v", mean=np.array([0.0, 0.0]), sd=1.0)
        spec_bcast = spec_id.set(["low", "high"], 0.5)
        assert (spec_bcast.low, spec_bcast.high) == (0.5, 0.5)
        spec_tup = spec_id.set({("low", "high"): 0.75})
        assert (spec_tup.low, spec_tup.high) == (0.75, 0.75)
        spec_arr = spec_id.set(["prior_mean"], np.array([1.2, -0.8]))
        np.testing.assert_allclose(np.asarray(spec_arr.prior_mean), [1.2, -0.8])

        cfg1 = tt.LinearMarginalConfig(offset_mode="hierarchical", lab_prior_sd=1.5)
        cfg2 = tt.LinearMarginalConfig(offset_mode="hierarchical", lab_prior_sd=1.5)
        assert isinstance(cfg1, zdx.Base)
        assert cfg1 == cfg2
        assert hash(cfg1) == hash(cfg2)
        assert cfg1.get("offset_mode", to_array=False) == "hierarchical"
        np.testing.assert_allclose(float(cfg1.get("lab_prior_sd")), 1.5)

        cbm = tt.load_preset("Brehm21")
        years = np.arange(772.0, 778.0, 1.0)
        ds = tt.Dataset(
            [
                tt.TreeRingSeries(
                    years=years,
                    d14c=np.zeros_like(years),
                    sigma_d14c=np.ones_like(years),
                    hemisphere="north",
                    name="nh",
                )
            ]
        )
        op = tt.build_dct_response_operator(cbm, ds, t_start=770.0, t_end=780.0, n_grid=16)
        cfg_op = cfg1.set("dct_operator", op)
        assert cfg_op.get("dct_operator", to_array=False) is op
        assert cfg_op.get("dct_operator.w_dct").shape == op.w_dct.shape

        # Mixed top-level static + nested PyTree leaf update in one .set(...) call
        mixed_cfg = cfg_op.set(
            {"offset_mode": "shared", "dct_operator.w_dct": 2.0 * op.w_dct}
        )
        assert mixed_cfg.offset_mode == "shared"
        np.testing.assert_allclose(
            np.asarray(mixed_cfg.dct_operator.w_dct), 2.0 * np.asarray(op.w_dct)
        )

        # Broadcast scalar and tuple dict key on LinearMarginalConfig
        cfg_bcast = cfg1.set(["marginalize_solar", "marginalize_gp"], False)
        assert not cfg_bcast.marginalize_solar and not cfg_bcast.marginalize_gp
        cfg_tup = cfg1.set({("marginalize_solar", "marginalize_trend"): False})
        assert not cfg_tup.marginalize_solar and not cfg_tup.marginalize_trend

        cfg_cleared = cfg_op.set("dct_operator", None).set(
            offset_mode="shared", marginalize_gp=False
        )
        assert cfg_cleared.dct_operator is None
        assert cfg_cleared.offset_mode == "shared"
        assert cfg_cleared.marginalize_gp is False

    def test_filter_jit_caching_across_likelihood_fitting_sampling_fisher(self) -> None:
        cbm = tt.load_preset("Brehm21")
        prod = tt.ProductionSum(
            steady=tt.SteadyProduction(),
            spike=tt.MiyakeSpike(t_start=774.5, duration=0.12, fluence=6.2),
        )
        model = tt.RadiocarbonModel(cbm, prod)
        years = np.arange(771.0, 780.0, 1.0)
        clean = np.asarray(
            model.predict_d14c(
                tt.TreeRingSeries(
                    years=years,
                    d14c=np.zeros_like(years),
                    sigma_d14c=np.ones_like(years),
                    hemisphere="north",
                    name="nh",
                )
            )
        )
        ds = tt.Dataset(
            [
                tt.TreeRingSeries(
                    years=years,
                    d14c=clean + 2.5,
                    sigma_d14c=np.full_like(years, 1.2),
                    hemisphere="north",
                    name="nh",
                )
            ]
        )

        ps_a = tt.ParameterSpace(
            model,
            model.production.spike.param_specs(t_start_sd=0.5, duration_bounds=(0.01, 2.0)),
        )
        ps_b = tt.ParameterSpace(
            model,
            model.production.spike.param_specs(t_start_sd=0.5, duration_bounds=(0.01, 2.0)),
        )
        cfg_a = tt.LinearMarginalConfig(offset_mode="shared")
        cfg_b = tt.LinearMarginalConfig(offset_mode="shared")

        trace_calls = 0

        @eqx.filter_jit
        def _cached_eval(
            m: tt.RadiocarbonModel,
            d: tt.Dataset,
            ps: tt.ParameterSpace,
            c: tt.LinearMarginalConfig,
            u: jax.Array,
        ) -> jax.Array:
            nonlocal trace_calls
            trace_calls += 1
            return tt.marginal_log_posterior(m, d, ps, u, c)

        u0 = ps_a.prior_means
        val1 = float(_cached_eval(model, ds, ps_a, cfg_a, u0))
        val2 = float(_cached_eval(model, ds, ps_b, cfg_b, u0 + 0.05))
        assert np.isfinite(val1) and np.isfinite(val2)
        assert trace_calls == 1, f"Expected 1 JIT compilation for equal static specs/cfg, got {trace_calls}"

        # Exercise linear_posterior, fit_svi, and cbm_gp_fisher_spectrum with zdx.Base cfg/specs
        lp = tt.linear_posterior(model, ds, cfg_a)
        assert np.isfinite(float(lp.offset_global_mean))

        svi_res = tt.fit_svi(
            model,
            ds,
            ps_a,
            cfg_a,
            guide="diag",
            num_steps=15,
            num_samples=16,
            rng_key=jax.random.PRNGKey(7),
        )
        assert jnp.all(jnp.isfinite(svi_res.losses))

        eff_cbm = tt.EffectiveCBM(K=2)
        spec_fisher = tt.cbm_gp_fisher_spectrum(
            eff_cbm,
            ds,
            None,
            spike=prod.spike,
            marginal_config=cfg_a,
            prior_sd=0.3,
        )
        assert jnp.all(jnp.isfinite(spec_fisher.eigenvalues))


# ===========================================================================
# 3. Sharp MiyakeSpike.param_specs UX & NH+SH seasonal MAP recovery
# ===========================================================================


class TestMiyakeSpikeParamSpecsAndSharpRecovery:
    def test_miyake_spike_param_specs_prefixes_and_skewgaussian(self) -> None:
        cbm = tt.load_preset("Brehm21")
        sp = tt.MiyakeSpike(t_start=774.55, duration=0.04, fluence=6.3)

        specs_default = sp.param_specs(t_start_sd=0.4, duration_bounds=(0.01, 1.5))
        assert [s.path for s in specs_default] == [
            "production.spike.t_start",
            "production.spike.duration",
            "production.spike.fluence",
        ]

        specs_dot = sp.param_specs(
            "production.spike.", t_start_sd=0.4, duration_bounds=(0.01, 1.5)
        )
        assert specs_dot == specs_default

        specs_bare = sp.param_specs("", t_start_sd=0.4, duration_bounds=(0.01, 1.5))
        assert [s.path for s in specs_bare] == ["t_start", "duration", "fluence"]

        # Direct spike model (prefix="production")
        m_direct = tt.RadiocarbonModel(cbm, sp)
        ps_direct = tt.ParameterSpace(
            m_direct,
            sp.param_specs("production", t_start_sd=0.4, duration_bounds=(0.01, 1.5)),
        )
        m_round = ps_direct.constrain(ps_direct.prior_means)
        np.testing.assert_allclose(float(m_round.production.t_start), 774.55, rtol=1e-13)
        np.testing.assert_allclose(float(m_round.production.duration), 0.04, rtol=1e-13)
        np.testing.assert_allclose(float(m_round.production.fluence), 6.3, rtol=1e-13)

        # Skew-Gaussian with and without skewness_sd
        sp_skew = tt.MiyakeSpike(
            t_start=993.3,
            duration=0.18,
            fluence=4.2,
            profile="skewgaussian",
            skewness=2.4,
        )
        assert len(sp_skew.param_specs(t_start_sd=0.5)) == 3
        specs_sk4 = sp_skew.param_specs(t_start_sd=0.5, skewness_sd=0.9)
        assert len(specs_sk4) == 4
        assert specs_sk4[3].path == "production.spike.skewness"
        np.testing.assert_allclose(
            float(specs_sk4[3].constrain(specs_sk4[3].prior_mean)), 2.4, rtol=1e-13
        )

    def test_sharp_miyake_spike_nh_sh_recovery_and_mapresult_single_spelling(self) -> None:
        assert not hasattr(tt.MAPResult, "fitted_model")

        cbm = tt.load_preset("Brehm21")
        true_spike = tt.MiyakeSpike(t_start=774.58, duration=0.06, fluence=6.4)
        true_model = tt.RadiocarbonModel(
            cbm,
            tt.ProductionSum(steady=tt.SteadyProduction(), spike=true_spike),
        )

        years = np.arange(770.0, 783.0, 1.0, dtype=np.float64)
        # Sub-seasonal NH monthly slices + Austral SH series resolve sharp sub-seasonal width
        templates = [
            tt.TreeRingSeries(
                years=years,
                d14c=np.zeros_like(years),
                sigma_d14c=np.full_like(years, 0.35),
                hemisphere="north",
                seasonal_weights="may-may",
                name="nh_may",
            ),
            tt.TreeRingSeries(
                years=years,
                d14c=np.zeros_like(years),
                sigma_d14c=np.full_like(years, 0.35),
                hemisphere="north",
                seasonal_weights="july-july",
                name="nh_july",
            ),
            tt.TreeRingSeries(
                years=years,
                d14c=np.zeros_like(years),
                sigma_d14c=np.full_like(years, 0.35),
                hemisphere="north",
                seasonal_weights="september-september",
                name="nh_sep",
            ),
            tt.TreeRingSeries(
                years=years,
                d14c=np.zeros_like(years),
                sigma_d14c=np.full_like(years, 0.45),
                hemisphere="south",
                seasonal_weights="october-march",
                name="sh_annual",
            ),
        ]
        ds0 = tt.Dataset(templates)
        true_offsets = jnp.array([4.2, 3.6, 3.1, 1.8], dtype=jnp.float64)
        clean_flat = np.asarray(
            true_model.predict_d14c(ds0, offsets=true_offsets, concat=True)
        )
        rng = np.random.default_rng(2026)
        sig_flat = np.asarray(ds0.sigma_flat)
        noisy_flat = clean_flat + rng.normal(0.0, 0.15 * sig_flat)

        series_list = []
        cursor = 0
        for s in ds0.series:
            n = s.n_obs
            series_list.append(s.set("d14c", noisy_flat[cursor : cursor + n]))
            cursor += n
        ds = tt.Dataset(series_list)

        # Start displaced from truth (sharp prior median 0.04 yr < default 0.05 yr bound!)
        init_spike = tt.MiyakeSpike(t_start=774.38, duration=0.04, fluence=4.9)
        init_model = tt.RadiocarbonModel(
            cbm,
            tt.ProductionSum(steady=tt.SteadyProduction(), spike=init_spike),
        )
        ps = tt.ParameterSpace(
            init_model,
            init_model.production.spike.param_specs(
                t_start_sd=0.5,
                duration_bounds=(0.01, 1.5),
            ),
        )
        cfg = tt.LinearMarginalConfig(offset_mode="hierarchical")
        result = tt.fit_map(init_model, ds, ps, cfg, method="BFGS", max_steps=300)

        assert not hasattr(result, "fitted_model")
        assert result.converged
        rec_spike = result.model.production.spike
        assert float(rec_spike.t_start) == pytest.approx(774.58, abs=0.05)
        assert float(rec_spike.duration) == pytest.approx(0.06, abs=0.04)
        assert float(rec_spike.fluence) == pytest.approx(6.4, rel=0.04)


# ===========================================================================
# 4. Input validation errors
# ===========================================================================


class TestValidationErrors:
    def test_paramspec_init_and_constructor_validation(self) -> None:
        with pytest.raises(ValueError, match="bijector"):
            tt.ParamSpec("x", bijector="bogus")  # type: ignore[arg-type]

        for bad_sd in (0.0, -0.5, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="prior_sd"):
                tt.ParamSpec("x", prior_mean=0.0, prior_sd=bad_sd)

        with pytest.raises(ValueError, match="prior_sd"):
            tt.ParamSpec("x", prior_mean=0.0, prior_sd=np.array([1.0, 0.0]))

        for bad_mean in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValueError, match="prior_mean"):
                tt.ParamSpec("x", prior_mean=bad_mean, prior_sd=1.0)

        with pytest.raises(ValueError, match="low < high"):
            tt.ParamSpec("x", "sigmoid", low=2.0, high=2.0)

        # ParamSpec.normal
        with pytest.raises(ValueError, match="sd"):
            tt.ParamSpec.normal("x", mean=0.0, sd=0.0)
        with pytest.raises(ValueError, match="mean"):
            tt.ParamSpec.normal("x", mean=float("nan"), sd=1.0)

        # ParamSpec.lognormal
        for bad_med in (0.0, -1.5, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="median"):
                tt.ParamSpec.lognormal("fluence", median=bad_med, sigma_log=0.5)
        with pytest.raises(ValueError, match="median"):
            tt.ParamSpec.lognormal(
                "fluence", median=np.array([1.0, -0.2]), sigma_log=0.5
            )
        for bad_sig in (0.0, -0.3, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="sigma_log"):
                tt.ParamSpec.lognormal("fluence", median=6.5, sigma_log=bad_sig)

        # ParamSpec.bounded (must name path, median, and (low, high)!)
        with pytest.raises(ValueError, match="low < high"):
            tt.ParamSpec.bounded("dur", 1.5, 0.5)
        with pytest.raises(ValueError, match="logit_sd"):
            tt.ParamSpec.bounded("dur", 0.05, 2.0, logit_sd=-1.0)
        for out_med in (0.05, 2.0, 0.01, 2.5):
            with pytest.raises(
                ValueError,
                match=r"production\.spike\.duration.*median.*\(0\.05, 2\.0\)",
            ):
                tt.ParamSpec.bounded(
                    "production.spike.duration", 0.05, 2.0, median=out_med
                )

        # .set(...) validation on ParamSpec and LinearMarginalConfig
        s = tt.ParamSpec.normal("x", 0.0, 1.0)
        with pytest.raises(ValueError, match="prior_sd"):
            s.set("prior_sd", -1.0)
        with pytest.raises(AttributeError, match="no settable attribute"):
            s.set("nonexistent", 1.0)
        with pytest.raises(AttributeError, match="nested"):
            s.set("foo.bar", 1.0)

        cfg = tt.LinearMarginalConfig()
        with pytest.raises(ValueError, match="offset_mode"):
            cfg.set("offset_mode", "bad_mode")
        with pytest.raises(TypeError, match="dct_operator"):
            cfg.set("dct_operator", "not_an_operator")
        with pytest.raises(AttributeError, match="no settable attribute"):
            cfg.set("nonexistent", 1)
        with pytest.raises(AttributeError, match="nested"):
            cfg.set("offset_mode.sub", 1)
        with pytest.raises(AttributeError, match="dct_operator is None"):
            cfg.set("dct_operator.w_dct", 1.0)

    def test_miyake_spike_param_specs_validation(self) -> None:
        # Sharp spike outside default duration_bounds=(0.05, 2.0)
        sharp = tt.MiyakeSpike(t_start=774.5, duration=0.02, fluence=6.5)
        with pytest.raises(ValueError, match="duration_bounds"):
            sharp.param_specs(t_start_sd=0.5)

        with pytest.raises(ValueError, match="0 <= low < high"):
            sharp.param_specs(t_start_sd=0.5, duration_bounds=(-0.1, 1.0))

        with pytest.raises(ValueError, match="sd"):
            tt.MiyakeSpike(duration=0.2).param_specs(t_start_sd=0.0)

        with pytest.raises(ValueError, match="skewgaussian"):
            tt.MiyakeSpike(duration=0.2, profile="supergaussian").param_specs(
                t_start_sd=0.5, skewness_sd=0.8
            )

        # Non-scalar spike leaf rejected with actionable MiyakeSpike.param_specs ValueError
        bad_spike = eqx.tree_at(
            lambda sp: sp.t_start,
            tt.MiyakeSpike(duration=0.2),
            jnp.array([774.4, 774.6], dtype=jnp.float64),
        )
        with pytest.raises(ValueError, match=r"MiyakeSpike\.param_specs.*scalar"):
            bad_spike.param_specs(t_start_sd=0.5)

        # Traced MiyakeSpike rejected cleanly
        with pytest.raises(ValueError, match="concrete"):
            jax.jit(
                lambda t0: tt.MiyakeSpike(t_start=t0, duration=0.2, fluence=6.5).param_specs(
                    t_start_sd=0.5
                )
            )(jnp.asarray(774.5))
