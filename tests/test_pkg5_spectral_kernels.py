"""Tests for Package 5: declarative SpectralKernel parameter table and methods."""

from __future__ import annotations

import math
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ticktack.data import load_bundled_series
from ticktack.fields import (
    DEFAULT_CANDIDATE_KERNELS,
    DEFAULT_LOG_LENGTH_PRIOR,
    DEFAULT_LOG_NU_PRIOR,
    DEFAULT_LOG_PERIOD_PRIOR,
    DEFAULT_LOG_QUALITY_PRIOR,
    DEFAULT_LOG_SIGMA_PRIOR,
    DEFAULT_SLOPE_PRIOR,
    CorrelatedFieldSpectrum,
    MaternSpectrum,
    QuasiPeriodicSpectrum,
    SpectralKernel,
    SquaredExpSpectrum,
    SumSpectrum,
    _KERNEL_NAME_ALIASES,
    _KernelParamSpec,
    _kernel_builders_from_spec,
    _normal_logpdf,
    optimize_gp_kernel,
)
from ticktack.presets import load_preset

jax.config.update("jax_enable_x64", True)


def _ref_normal_logpdf(x: np.ndarray | float, mu: float, sd: float) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    z = (arr - mu) / sd
    return -0.5 * z * z - math.log(sd) - 0.5 * math.log(2.0 * math.pi)


def test_no_duplicate_unconstrained_aliases() -> None:
    """SpectralKernel exposes only unconstrain_hyperparams / from_unconstrained_hyperparams."""
    assert not hasattr(SpectralKernel, "unconstrained_init")
    assert not hasattr(SpectralKernel, "from_unconstrained")


def test_leaf_kernels_declarative_roundtrip_prior_summary_and_cache_tag() -> None:
    """All 4 leaf kernels match independent closed-form transforms, priors, summaries, and tags."""
    cases = [
        (
            "matern_fixed",
            MaternSpectrum(
                sigma=0.37, length_yr=6.3, order=1.35, learn_order=False, zero_mode=False
            ),
            ("log_sigma", "log_length_yr"),
            np.array([math.log(0.37), math.log(6.3)]),
            np.array([-1.15, 2.05]),
            [DEFAULT_LOG_SIGMA_PRIOR, DEFAULT_LOG_LENGTH_PRIOR],
            {"sigma": math.exp(-1.15), "length_yr": math.exp(2.05), "order": 1.35},
            "custom_matern:ord=1.35:zm=False",
        ),
        (
            "matern_free",
            MaternSpectrum(
                sigma=0.42, length_yr=3.8, order=2.7, learn_order=True, zero_mode=True
            ),
            ("log_sigma", "log_length_yr", "log_nu"),
            np.array([math.log(0.42), math.log(3.8), math.log(2.7 - 0.5)]),
            np.array([-0.82, 1.44, 0.31]),
            [DEFAULT_LOG_SIGMA_PRIOR, DEFAULT_LOG_LENGTH_PRIOR, DEFAULT_LOG_NU_PRIOR],
            {
                "sigma": math.exp(-0.82),
                "length_yr": math.exp(1.44),
                "order": 0.5 + math.exp(0.31),
            },
            "canon:continuous_matern:zm=True",
        ),
        (
            "squared_exp",
            SquaredExpSpectrum(sigma=0.33, length_yr=7.4, zero_mode=False),
            ("log_sigma", "log_length_yr"),
            np.array([math.log(0.33), math.log(7.4)]),
            np.array([-1.41, 1.88]),
            [DEFAULT_LOG_SIGMA_PRIOR, DEFAULT_LOG_LENGTH_PRIOR],
            {"sigma": math.exp(-1.41), "length_yr": math.exp(1.88)},
            "canon:squared_exp:zm=False",
        ),
        (
            "quasi_periodic",
            QuasiPeriodicSpectrum(sigma=0.29, period_yr=10.7, quality=4.3, zero_mode=True),
            ("log_sigma", "log_period_yr", "log_q_minus_half"),
            np.array([math.log(0.29), math.log(10.7), math.log(4.3 - 0.5)]),
            np.array([-1.27, 2.35, 1.12]),
            [DEFAULT_LOG_SIGMA_PRIOR, DEFAULT_LOG_PERIOD_PRIOR, DEFAULT_LOG_QUALITY_PRIOR],
            {
                "sigma": math.exp(-1.27),
                "period_yr": math.exp(2.35),
                "quality": 0.5 + math.exp(1.12),
            },
            "canon:sho_solar:zm=True",
        ),
        (
            "correlated_field",
            CorrelatedFieldSpectrum(
                sigma=0.31,
                slope=-2.3,
                log_deviations=[0.4, -0.7, 0.15, -0.25, 0.6],
                flexibility=0.65,
                zero_mode=False,
            ),
            ("log_sigma", "slope", "xi_spec_1", "xi_spec_2", "xi_spec_3", "xi_spec_4", "xi_spec_5"),
            np.array([math.log(0.31), -2.3, 0.4, -0.7, 0.15, -0.25, 0.6]),
            np.array([-1.05, -1.75, -0.32, 0.58, -0.11, 0.44, -0.73]),
            [DEFAULT_LOG_SIGMA_PRIOR, DEFAULT_SLOPE_PRIOR] + [(0.0, 1.0)] * 5,
            {
                "sigma": math.exp(-1.05),
                "slope": -1.75,
                "rms_xi": float(np.sqrt(np.mean(np.square([-0.32, 0.58, -0.11, 0.44, -0.73])))),
            },
            "custom_cf:m=5:flex=0.65:zm=False",
        ),
    ]

    for label, kern, exp_names, exp_phi0, phi_new, priors, exp_summary, exp_tag in cases:
        assert kern.param_names() == exp_names, label
        assert kern.n_hyperparams == len(exp_names), label

        phi0 = kern.unconstrain_hyperparams()
        np.testing.assert_allclose(phi0, exp_phi0, rtol=1e-14, atol=1e-14)

        k_new = kern.from_unconstrained_hyperparams(phi_new)
        assert type(k_new) is type(kern)
        assert jax.tree_util.tree_structure(k_new) == jax.tree_util.tree_structure(kern)
        np.testing.assert_allclose(k_new.unconstrain_hyperparams(), phi_new, rtol=1e-14, atol=1e-14)

        # Hyperprior log-prob & analytical gradient against closed-form Gaussian
        exp_lp = sum(float(_ref_normal_logpdf(u, mu, sd)) for u, (mu, sd) in zip(phi_new, priors))
        lp_direct = kern.hyperprior_log_prob(phi_new)
        lp_from_state = k_new.hyperprior_log_prob()
        assert isinstance(lp_direct, jax.Array) and lp_direct.dtype == jnp.float64
        np.testing.assert_allclose(lp_direct, exp_lp, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(lp_from_state, exp_lp, rtol=1e-14, atol=1e-14)

        g_prior = jax.grad(
            lambda u: kern.from_unconstrained_hyperparams(u).hyperprior_log_prob(u)
        )(jnp.asarray(phi_new, dtype=jnp.float64))
        exp_grad = np.array([-(u - mu) / (sd * sd) for u, (mu, sd) in zip(phi_new, priors)])
        np.testing.assert_allclose(g_prior, exp_grad, rtol=1e-14, atol=1e-14)

        # Unclipped restoring gradient at extreme |u| = 25
        u_ext = jnp.linspace(-25.0, 25.0, kern.n_hyperparams, dtype=jnp.float64)
        g_ext = jax.grad(
            lambda u: kern.from_unconstrained_hyperparams(u).hyperprior_log_prob(u)
        )(u_ext)
        exp_g_ext = np.array([-(float(u) - mu) / (sd * sd) for u, (mu, sd) in zip(u_ext, priors)])
        np.testing.assert_allclose(g_ext, exp_g_ext, rtol=1e-14, atol=1e-14)

        # Summary dictionary
        got_summary = k_new.summarize()
        assert list(got_summary.keys()) == list(exp_summary.keys()), label
        for sk, sv in exp_summary.items():
            np.testing.assert_allclose(got_summary[sk], sv, rtol=1e-14, atol=1e-14)

        assert kern.cache_tag() == exp_tag, label


def test_scalar_and_float32_set_updates_preserve_float64() -> None:
    """zodiax.Base.set with Python floats or np.float32 works cleanly and stays float64."""
    m = MaternSpectrum(sigma=0.3, length_yr=5.0, order=2.0).set("sigma", 0.5)
    assert m.param_names() == ("log_sigma", "log_length_yr")
    assert m.n_hyperparams == 2
    phi_m = m.unconstrain_hyperparams()
    assert phi_m.dtype == jnp.float64
    np.testing.assert_allclose(phi_m, [math.log(0.5), math.log(5.0)], rtol=1e-14, atol=1e-14)
    m_rt = m.from_unconstrained_hyperparams(phi_m)
    assert m_rt.sigma.dtype == jnp.float64
    np.testing.assert_allclose(m_rt.sigma, 0.5, rtol=1e-14, atol=1e-14)
    assert m.summarize() == {"sigma": 0.5, "length_yr": 5.0, "order": 2.0}

    m32 = MaternSpectrum(sigma=0.3, length_yr=5.0, order=2.0).set("sigma", np.float32(0.5))
    phi_m32 = m32.unconstrain_hyperparams()
    assert phi_m32.dtype == jnp.float64
    np.testing.assert_allclose(phi_m32[0], math.log(0.5), rtol=1e-7, atol=1e-7)

    devs32 = np.array([0.25, -0.5, 0.75, -0.125], dtype=np.float32)
    cf32 = (
        CorrelatedFieldSpectrum(sigma=0.3, slope=-2.0, n_modes=4)
        .set("sigma", np.float32(0.5))
        .set("slope", np.float32(-1.75))
        .set("log_deviations", devs32)
    )
    assert cf32.n_hyperparams == 6
    phi_cf32 = cf32.unconstrain_hyperparams()
    assert phi_cf32.dtype == jnp.float64
    cf_rt = cf32.from_unconstrained_hyperparams(phi_cf32)
    assert cf_rt.sigma.dtype == jnp.float64
    assert cf_rt.slope.dtype == jnp.float64
    assert cf_rt.log_deviations.dtype == jnp.float64
    np.testing.assert_allclose(cf_rt.slope, -1.75, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(cf_rt.log_deviations, devs32.astype(np.float64), rtol=1e-14)


def test_summarize_decouples_renaming_from_vector_reduction() -> None:
    """summarize() keeps signed scalars under custom summary_key and RMS-reduces unnamed vectors."""

    class _MixedSummaryKernel(SpectralKernel):
        signed_val: jax.Array
        vec_val: jax.Array

        _PARAM_SPECS: ClassVar[tuple[_KernelParamSpec, ...]] = (
            _KernelParamSpec("signed_val", "signed_val", (0.0, 1.0), "identity", "renamed_signed"),
            _KernelParamSpec("vec", "vec_val", (0.0, 1.0), "identity", None),
        )

        def __init__(self, signed_val: float = -2.35, vec_val: tuple[float, ...] = (0.3, -0.4)):
            self.signed_val = jnp.asarray(signed_val, dtype=jnp.float64)
            self.vec_val = jnp.asarray(vec_val, dtype=jnp.float64)

        def spectrum(self, n_grid: int, dt: float) -> jax.Array:
            return jnp.ones((int(n_grid),), dtype=jnp.float64)

    kern = _MixedSummaryKernel(signed_val=-2.35, vec_val=(0.3, -0.4))
    summary = kern.summarize()
    assert list(summary.keys()) == ["renamed_signed", "vec_val"]
    assert summary["renamed_signed"] == pytest.approx(-2.35, rel=1e-14)
    assert summary["vec_val"] == pytest.approx(math.sqrt((0.09 + 0.16) / 2.0), rel=1e-14)


def test_correlated_field_zero_mode_and_matern_fixed_order_invariants() -> None:
    """CorrelatedFieldSpectrum(zero_mode=True/False) and MaternSpectrum(learn_order=False)."""
    for zm in (False, True):
        cf = CorrelatedFieldSpectrum(
            sigma=0.28,
            slope=-1.85,
            log_deviations=[0.2, -0.4, 0.6],
            flexibility=0.45,
            zero_mode=zm,
        )
        phi = jnp.array([math.log(0.41), -2.1, -0.3, 0.5, -0.2], dtype=jnp.float64)
        cf_upd = cf.from_unconstrained_hyperparams(phi)
        assert cf_upd.zero_mode is zm
        assert cf_upd.flexibility == 0.45
        spec = cf_upd.spectrum(64, 0.5)
        np.testing.assert_allclose(jnp.mean(spec), 0.41**2, rtol=1e-14, atol=1e-14)
        if zm:
            assert float(spec[0]) > 0.0
        else:
            assert float(spec[0]) == 0.0
        assert cf_upd.cache_tag() == f"custom_cf:m=3:flex=0.45:zm={zm}"

    m_fixed = MaternSpectrum(
        sigma=0.2, length_yr=4.0, order=2.35, learn_order=False, zero_mode=True
    )
    m_upd = m_fixed.from_unconstrained_hyperparams(jnp.array([-0.5, 1.9], dtype=jnp.float64))
    assert m_upd.learn_order is False
    assert m_upd.zero_mode is True
    np.testing.assert_allclose(m_upd.order, 2.35, rtol=0.0, atol=0.0)
    assert m_upd.summarize()["order"] == pytest.approx(2.35, rel=1e-14)


def test_nested_sum_spectrum_slicing_summary_cache_tag_and_jit_grad() -> None:
    """Nested SumSpectrum flattens and composes all methods cleanly under jit, vmap, and grad."""
    m = MaternSpectrum(sigma=0.23, length_yr=5.4, order=1.8, learn_order=True, zero_mode=True)
    qp = QuasiPeriodicSpectrum(sigma=0.19, period_yr=10.6, quality=3.7, zero_mode=False)
    cf = CorrelatedFieldSpectrum(
        sigma=0.14, slope=-2.2, log_deviations=[0.25, -0.35], flexibility=0.4, zero_mode=False
    )

    left_nested = (m + qp) + cf
    right_nested = m + (qp + cf)
    explicit_nested = SumSpectrum([SumSpectrum([m, qp]), cf])

    for comp in (left_nested, right_nested, explicit_nested):
        assert isinstance(comp, SumSpectrum)
        assert len(comp.parts) == 3
        assert comp.n_hyperparams == 3 + 3 + 4
        assert comp.param_names() == (
            "p1_log_sigma",
            "p1_log_length_yr",
            "p1_log_nu",
            "p2_log_sigma",
            "p2_log_period_yr",
            "p2_log_q_minus_half",
            "p3_log_sigma",
            "p3_slope",
            "p3_xi_spec_1",
            "p3_xi_spec_2",
        )
        assert comp.cache_tag() == (
            "sum:(canon:continuous_matern:zm=True+canon:sho_solar:zm=False"
            "+custom_cf:m=2:flex=0.4:zm=False)"
        )

    phi_parts = [
        jnp.array([-1.3, 1.6, 0.25], dtype=jnp.float64),
        jnp.array([-1.6, 2.38, 1.05], dtype=jnp.float64),
        jnp.array([-1.9, -1.7, 0.4, -0.5], dtype=jnp.float64),
    ]
    phi_all = jnp.concatenate(phi_parts)

    k_sum = left_nested.from_unconstrained_hyperparams(phi_all)
    k_parts = [p.from_unconstrained_hyperparams(u) for p, u in zip((m, qp, cf), phi_parts)]

    # Exact slice composition for spectrum, hyperprior, and summary
    np.testing.assert_allclose(
        k_sum.spectrum(64, 0.5),
        sum(kp.spectrum(64, 0.5) for kp in k_parts),
        rtol=1e-14,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        k_sum.hyperprior_log_prob(phi_all),
        sum(kp.hyperprior_log_prob(u) for kp, u in zip((m, qp, cf), phi_parts)),
        rtol=1e-14,
        atol=1e-14,
    )
    expected_summary = {
        f"p{i + 1}_{k}": v for i, kp in enumerate(k_parts) for k, v in kp.summarize().items()
    }
    assert k_sum.summarize() == expected_summary

    # JIT, vmap, and finite-difference gradient agreement
    weights = jnp.linspace(0.4, 2.1, 48, dtype=jnp.float64)

    @eqx.filter_jit
    def loss_fn(u: jax.Array) -> jax.Array:
        ku = left_nested.from_unconstrained_hyperparams(u)
        return jnp.sum(ku.spectrum(48, 0.5) * weights) + ku.hyperprior_log_prob(u)

    g_ad = jax.grad(loss_fn)(phi_all)
    eps = 1e-6
    g_fd = np.empty_like(np.asarray(phi_all))
    for i in range(phi_all.shape[0]):
        ei = jnp.zeros_like(phi_all).at[i].set(eps)
        g_fd[i] = float((loss_fn(phi_all + ei) - loss_fn(phi_all - ei)) / (2.0 * eps))
    np.testing.assert_allclose(g_ad, g_fd, rtol=1e-7, atol=1e-7)

    # Batch vmap evaluation
    batch_u = jnp.stack([phi_all, phi_all + 0.15, phi_all - 0.2])
    batch_lp = jax.vmap(left_nested.hyperprior_log_prob)(batch_u)
    expected_batch = jnp.stack([left_nested.hyperprior_log_prob(row) for row in batch_u])
    np.testing.assert_allclose(batch_lp, expected_batch, rtol=1e-14, atol=1e-14)


def test_kernel_builders_shorthand_aliases_custom_and_hyperprior_override() -> None:
    """_kernel_builders_from_spec resolves all shorthands, custom kernels, and overrides."""
    expected_dims = {
        "matern12": 2,
        "matern32": 2,
        "matern52": 2,
        "squared_exp": 2,
        "continuous_matern": 3,
        "sho_solar": 3,
        "matern32_plus_solar": 5,
        "correlated_field": 6,
    }
    assert set(DEFAULT_CANDIDATE_KERNELS) == set(expected_dims)

    def custom_hp(u: jax.Array) -> jax.Array:
        return -0.3 * jnp.sum(jnp.square(u - 0.2))

    for alias, canon in _KERNEL_NAME_ALIASES.items():
        for zm in (False, True):
            b1, p1, n1, phi1, tag1 = _kernel_builders_from_spec(
                f"  {alias.upper()} ", zero_mode=zm
            )
            b2, p2, n2, phi2, tag2 = _kernel_builders_from_spec(canon, zero_mode=zm)
            assert n1 == n2
            assert len(n1) == expected_dims[canon]
            assert tag1 == tag2 == f"canon:{canon}:zm={zm}"
            np.testing.assert_allclose(phi1, phi2, rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(
                b1(phi1 + 0.1).spectrum(32, 0.5),
                b2(phi2 + 0.1).spectrum(32, 0.5),
                rtol=1e-14,
                atol=1e-14,
            )
            np.testing.assert_allclose(p1(phi1 + 0.1), p2(phi2 + 0.1), rtol=1e-14, atol=1e-14)

            # With hyperprior_override
            _, p_ov, _, _, tag_ov = _kernel_builders_from_spec(
                alias, zero_mode=zm, hyperprior_override=custom_hp
            )
            assert p_ov is custom_hp
            assert tag_ov == f"canon:{canon}:zm={zm}:hp={id(custom_hp)}"
            np.testing.assert_allclose(
                p_ov(phi1 + 0.1), custom_hp(phi1 + 0.1), rtol=1e-14, atol=1e-14
            )

    # Custom kernel keeps its own zero_mode in cache_tag (no aliasing against caller zero_mode)
    k_zm0 = MaternSpectrum(sigma=0.3, length_yr=5.0, order=2.0, zero_mode=False)
    k_zm1 = MaternSpectrum(sigma=0.3, length_yr=5.0, order=2.0, zero_mode=True)
    _, _, _, _, tag_c0 = _kernel_builders_from_spec(k_zm0, zero_mode=True)
    _, _, _, _, tag_c1 = _kernel_builders_from_spec(k_zm1, zero_mode=True)
    assert tag_c0 == "custom_matern:ord=2.0:zm=False"
    assert tag_c1 == "custom_matern:ord=2.0:zm=True"
    assert tag_c0 != tag_c1


def test_subclass_cache_tag_and_no_jit_cache_collision() -> None:
    """Subclasses of built-in kernels get distinct custom cache tags and never collide in JIT."""

    class NarrowSolar(QuasiPeriodicSpectrum):
        def hyperprior_log_prob(self, phi: jax.Array | None = None) -> jax.Array:
            p = (
                self.unconstrain_hyperparams()
                if phi is None
                else jnp.asarray(phi, dtype=jnp.float64).reshape(-1)
            )
            return (
                _normal_logpdf(p[0], *DEFAULT_LOG_SIGMA_PRIOR)
                + _normal_logpdf(p[1], math.log(25.0), 0.02)
                + _normal_logpdf(p[2], *DEFAULT_LOG_QUALITY_PRIOR)
            )

    class SubMatern(MaternSpectrum):
        pass

    class SubSqExp(SquaredExpSpectrum):
        pass

    class SubCF(CorrelatedFieldSpectrum):
        pass

    class SubSum(SumSpectrum):
        pass

    qp_base = QuasiPeriodicSpectrum(sigma=0.22, period_yr=11.0, quality=5.0)
    qp_narrow = NarrowSolar(sigma=0.22, period_yr=11.0, quality=5.0)
    assert qp_base.cache_tag() == "canon:sho_solar:zm=False"
    assert qp_narrow.cache_tag().startswith("custom:")
    assert qp_narrow.cache_tag() != qp_base.cache_tag()
    assert SubMatern().cache_tag().startswith("custom:")
    assert SubSqExp().cache_tag().startswith("custom:")
    assert SubCF().cache_tag().startswith("custom:")
    assert SubSum([qp_base]).cache_tag().startswith("custom:")
    assert (qp_narrow + MaternSpectrum()).cache_tag() != (qp_base + MaternSpectrum()).cache_tag()

    # Run optimize_gp_kernel with base first to warm _SPEC_JIT_CACHE, then NarrowSolar
    cbm = load_preset("Brehm21")
    series = load_bundled_series("775AD-early-S/Büntgen18_TAS01.csv")
    res_base = optimize_gp_kernel(cbm, series, kernel=qp_base, n_grid=28, multi_start=1)
    res_narrow = optimize_gp_kernel(cbm, series, kernel=qp_narrow, n_grid=28, multi_start=1)
    assert res_narrow.hyperparameters["period_yr"] > 20.0
    assert abs(res_narrow.hyperparameters["period_yr"] - res_base.hyperparameters["period_yr"]) > 5.0
    np.testing.assert_allclose(
        res_narrow.log_hyperprior,
        qp_narrow.hyperprior_log_prob(res_narrow.phi_map),
        rtol=1e-12,
        atol=1e-12,
    )


def test_length_validation_and_custom_subclass_standalone_and_sum() -> None:
    """Wrong phi lengths raise clear errors; legacy custom subclasses work standalone and in Sum."""
    m = MaternSpectrum(sigma=0.3, length_yr=5.0, order=2.0, learn_order=False)
    s = m + QuasiPeriodicSpectrum(sigma=0.2, period_yr=11.0, quality=4.0)

    with pytest.raises(ValueError, match="Expected 2 hyperparameters"):
        m.from_unconstrained_hyperparams(jnp.zeros((3,), dtype=jnp.float64))
    with pytest.raises(ValueError, match="Expected 2 hyperparameters"):
        m.hyperprior_log_prob(jnp.zeros((1,), dtype=jnp.float64))
    with pytest.raises(ValueError, match="Expected 5 hyperparameters"):
        s.from_unconstrained_hyperparams(jnp.zeros((4,), dtype=jnp.float64))
    with pytest.raises(ValueError, match="Expected 5 hyperparameters"):
        s.hyperprior_log_prob(jnp.zeros((6,), dtype=jnp.float64))
    with pytest.raises(ValueError, match="Unknown kernel name"):
        _kernel_builders_from_spec("not_a_kernel", zero_mode=False)
    with pytest.raises(TypeError, match="str or SpectralKernel"):
        _kernel_builders_from_spec(123, zero_mode=False)  # type: ignore[arg-type]

    class _BareKernel(SpectralKernel):
        def spectrum(self, n_grid: int, dt: float) -> jax.Array:
            return jnp.ones((int(n_grid),), dtype=jnp.float64)

    bare = _BareKernel()
    with pytest.raises(NotImplementedError):
        bare.param_names()

    class _LegacyWhiteKernel(SpectralKernel):
        sigma: jax.Array

        def __init__(self, sigma: float = 0.25) -> None:
            self.sigma = jnp.asarray(sigma, dtype=jnp.float64).reshape(())

        def spectrum(self, n_grid: int, dt: float) -> jax.Array:
            n = int(n_grid)
            return jnp.concatenate(
                [
                    jnp.zeros((1,), dtype=jnp.float64),
                    jnp.full((n - 1,), jnp.square(self.sigma) * n / max(n - 1, 1)),
                ]
            )

        def param_names(self) -> tuple[str, ...]:
            return ("log_sigma",)

        def unconstrain_hyperparams(self) -> jax.Array:
            return jnp.array([jnp.log(jnp.maximum(self.sigma, 1e-30))], dtype=jnp.float64)

        def from_unconstrained_hyperparams(self, phi: jax.Array) -> _LegacyWhiteKernel:
            u = jnp.asarray(phi, dtype=jnp.float64).reshape(-1)[0]
            return _LegacyWhiteKernel(jnp.exp(jnp.clip(u, -80.0, 80.0)))

        def hyperprior_log_prob(self, phi: jax.Array | None = None) -> jax.Array:
            u = (
                self.unconstrain_hyperparams()
                if phi is None
                else jnp.asarray(phi, dtype=jnp.float64).reshape(-1)
            )
            return _normal_logpdf(u[0], *DEFAULT_LOG_SIGMA_PRIOR)

    leg = _LegacyWhiteKernel(0.25)
    assert leg.n_hyperparams == 1
    np.testing.assert_allclose(leg.unconstrain_hyperparams(), [math.log(0.25)], rtol=1e-14)
    assert leg.summarize() == {"log_sigma": pytest.approx(math.log(0.25), rel=1e-14)}
    assert leg.cache_tag().startswith("custom:")

    # Standalone optimize_gp_kernel with custom subclass (include_field_initial_cov=False since
    # custom subclass has no continuous-time prehistory density)
    cbm = load_preset("Brehm21")
    series = load_bundled_series("775AD-early-S/Büntgen18_TAS01.csv")
    res_leg = optimize_gp_kernel(
        cbm,
        series,
        kernel=leg,
        n_grid=24,
        multi_start=1,
        include_field_initial_cov=False,
    )
    assert res_leg.n_hyperparams == 1
    assert res_leg.hyperparameters == res_leg.kernel.summarize()

    # Inside SumSpectrum
    comp = leg + MaternSpectrum(sigma=0.2, length_yr=5.0, order=2.0)
    assert comp.n_hyperparams == 3
    assert comp.param_names() == ("p1_log_sigma", "p2_log_sigma", "p2_log_length_yr")
    phi_comp = comp.unconstrain_hyperparams()
    comp_rt = comp.from_unconstrained_hyperparams(phi_comp + 0.1)
    np.testing.assert_allclose(comp_rt.unconstrain_hyperparams(), phi_comp + 0.1, rtol=1e-14)
    assert set(comp_rt.summarize().keys()) == {"p1_log_sigma", "p2_sigma", "p2_length_yr", "p2_order"}
    assert comp.cache_tag().startswith("sum:(custom:")

    res_comp = optimize_gp_kernel(
        cbm,
        series,
        kernel=comp,
        n_grid=24,
        multi_start=1,
        include_field_initial_cov=False,
    )
    assert res_comp.n_hyperparams == 3
    assert res_comp.hyperparameters == res_comp.kernel.summarize()


def test_optimize_gp_kernel_uses_kernel_summarize_and_n_hyperparams() -> None:
    """End-to-end optimize_gp_kernel returns hyperparameters matching kernel.summarize()."""
    cbm = load_preset("Brehm21")
    series = load_bundled_series("775AD-early-S/Büntgen18_TAS01.csv")
    kern_in = MaternSpectrum(sigma=0.25, length_yr=5.0, order=2.0, learn_order=True)
    res = optimize_gp_kernel(cbm, series, kernel=kern_in, n_grid=32, multi_start=1)
    assert res.n_hyperparams == res.kernel.n_hyperparams == 3
    assert res.hyperparameters == res.kernel.summarize()
    assert set(res.hyperparameters.keys()) == {"sigma", "length_yr", "order"}
