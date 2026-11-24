"""Tests for Packages 1 & 9 and Decisions 5.2 & 5.6 (whitening, offsets, -ize, explicit Fisher priors)."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tinygp.kernels as tgk

import ticktack as tt
import ticktack.fitting as ttf
from ticktack.fisher import cbm_fisher_spectrum


def _make_asymmetric_fixture() -> tuple[tt.CarbonBoxModel, tt.Dataset, tt.GaussianField]:
    cbm = tt.load_preset("Brehm21", kind="balanced")
    lr0 = jnp.asarray(cbm.log_rates_free, dtype=jnp.float64)
    perturb = 0.11 * jnp.cos(jnp.arange(1, lr0.size + 1, dtype=jnp.float64) * 0.63)
    cbm = eqx.tree_at(lambda m: m.log_rates_free, cbm, lr0 + perturb)

    s1 = tt.TreeRingSeries(
        years=np.array([772.0, 773.0, 774.0, 775.0, 776.0, 777.0]),
        d14c=np.array([-14.2, -13.6, -5.8, 2.1, 0.8, -0.9]),
        sigma_d14c=np.array([1.15, 1.37, 0.94, 1.48, 1.21, 1.09]),
        hemisphere="north",
        seasonal_weights="Apr-Sep",
        name="nh_a",
    )
    s2 = tt.TreeRingSeries(
        years=np.array([773.0, 774.0, 775.0, 776.0, 777.0]),
        d14c=np.array([-11.7, -4.1, 3.6, 1.9, 0.5]),
        sigma_d14c=np.array([1.61, 1.05, 1.44, 1.19, 1.52]),
        hemisphere="north",
        seasonal_weights="May-Aug",
        name="nh_b",
    )
    s3 = tt.TreeRingSeries(
        years=np.array([772.5, 773.5, 774.5, 775.5, 776.5]),
        d14c=np.array([-18.4, -17.8, -11.0, -3.2, -4.3]),
        sigma_d14c=np.array([1.41, 1.12, 1.57, 1.26, 1.38]),
        hemisphere="south",
        seasonal_weights="Oct-Mar",
        name="sh_c",
    )
    ds = tt.Dataset((s1, s2, s3), name="asym_nh_sh")
    gf = tt.GaussianField(
        jnp.linspace(-0.25, 0.22, 12, dtype=jnp.float64),
        t_start=769.0,
        t_end=779.0,
        kernel=tt.MaternSpectrum(sigma=0.31, length_yr=4.2, order=1.5),
    )
    return cbm, ds, gf


# ---------------------------------------------------------------------------
# 1. Decision 5.2: LinearMarginalConfig accepts -ize only (rejects -ise)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_kwarg",
    [
        "marginalise_solar",
        "marginalise_trend",
        "marginalise_gp",
        "marginalise_field",
    ],
)
def test_linear_marginal_config_rejects_ise_kwargs_and_attrs(bad_kwarg: str) -> None:
    """Decision 5.2: ``marginalise_*`` kwargs and attributes are removed."""
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        tt.LinearMarginalConfig(**{bad_kwarg: True})  # type: ignore[arg-type]

    cfg = tt.LinearMarginalConfig()
    assert not hasattr(cfg, bad_kwarg)
    with pytest.raises(AttributeError):
        getattr(cfg, bad_kwarg)


def test_linear_marginal_config_ize_defaults_and_field_inheritance() -> None:
    """``marginalize_*`` attributes preserve exact default and ``marginalize_gp`` inheritance."""
    cfg_def = tt.LinearMarginalConfig()
    assert cfg_def.marginalize_solar is True
    assert cfg_def.marginalize_trend is True
    assert cfg_def.marginalize_gp is True
    assert cfg_def.marginalize_field is True

    cfg_no_gp = tt.LinearMarginalConfig(marginalize_gp=False)
    assert cfg_no_gp.marginalize_gp is False
    assert cfg_no_gp.marginalize_field is False

    cfg_split = tt.LinearMarginalConfig(marginalize_gp=False, marginalize_field=True)
    assert cfg_split.marginalize_gp is False
    assert cfg_split.marginalize_field is True


# ---------------------------------------------------------------------------
# 2. Decision 5.6: Explicit Fisher prior widths & removed parameterisation
# ---------------------------------------------------------------------------


def test_fisher_requires_explicit_prior_widths_and_drops_parameterisation() -> None:
    """Decision 5.6: omitting ``prior_cbm_sd`` (SVD mode) or ``prior_sd`` raises ``ValueError``."""
    cbm, ds, gf = _make_asymmetric_fixture()
    t_lag = jnp.linspace(0.0, 10.0, 11)

    with pytest.raises(ValueError, match="prior_cbm_sd"):
        tt.cbm_mode_impulse_responses(cbm, t_lag, None)

    with pytest.raises(ValueError, match="prior_sd"):
        tt.cbm_gp_fisher_spectrum(cbm, ds, gf)  # type: ignore[call-arg]

    with pytest.raises(ValueError, match="prior_sd"):
        tt.cbm_gp_fisher_spectrum(cbm, ds, gf, prior_sd=None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="parameterisation"):
        tt.cbm_gp_fisher_spectrum(  # type: ignore[call-arg]
            cbm, ds, gf, prior_sd=0.35, parameterisation="log_rates"
        )

    # When spectrum is supplied, cbm_mode_impulse_responses needs no prior_cbm_sd:
    spec = tt.cbm_gp_fisher_spectrum(cbm, ds, gf, prior_sd=0.41)
    imp = tt.cbm_mode_impulse_responses(cbm, t_lag, spec, modes=(0, 1))
    assert imp.source == "fisher"
    assert imp.mode_contributions["north"].shape == (2, 11)


# ---------------------------------------------------------------------------
# 3. Package 1: ticktack._utils shared helpers & inlined _uses_joint_mode
# ---------------------------------------------------------------------------


def test_pkg1_utils_concrete_sigma_jitter_and_safe_psd_root() -> None:
    """Verify ``_concrete``, ``_sigma_with_jitter``, ``_safe_psd_root``, and removed ``_use_joint_mode``."""
    from ticktack._utils import _concrete, _safe_psd_root, _sigma_with_jitter

    assert not hasattr(ttf, "_use_joint_mode")

    x = jnp.array([1.7, -2.3], dtype=jnp.float64)
    c_eager = _concrete(x)
    assert isinstance(c_eager, np.ndarray)
    np.testing.assert_allclose(c_eager, [1.7, -2.3], rtol=1e-15)

    traced_flag: list[bool] = []

    @jax.jit
    def _jit_probe(v: jax.Array) -> jax.Array:
        traced_flag.append(_concrete(v) is None)
        return v * 2.0

    _ = _jit_probe(x)
    assert traced_flag == [True]

    # _sigma_with_jitter at asymmetric non-unit inputs + exact gradient vs finite differences
    sig = jnp.array([1.13, 2.47, 0.83], dtype=jnp.float64)
    jit_s = jnp.asarray(0.61, dtype=jnp.float64)
    np.testing.assert_allclose(_sigma_with_jitter(sig, None), sig, rtol=1e-15)
    got_sj = _sigma_with_jitter(sig, jit_s)
    ref_sj = jnp.sqrt(sig**2 + jit_s**2)
    np.testing.assert_allclose(got_sj, ref_sj, rtol=1e-15)

    w = jnp.array([0.7, -1.3, 2.1], dtype=jnp.float64)
    g_sig, g_jit = jax.grad(lambda s, j: jnp.dot(w, _sigma_with_jitter(s, j)), argnums=(0, 1))(
        sig, jit_s
    )
    np.testing.assert_allclose(g_sig, w * sig / ref_sj, rtol=1e-15)
    np.testing.assert_allclose(float(g_jit), float(jnp.dot(w, jit_s / ref_sj)), rtol=1e-15)
    eps = 1e-5
    fd_jit = (
        float(jnp.dot(w, _sigma_with_jitter(sig, jit_s + eps)))
        - float(jnp.dot(w, _sigma_with_jitter(sig, jit_s - eps)))
    ) / (2.0 * eps)
    np.testing.assert_allclose(float(g_jit), fd_jit, rtol=1e-10)

    # _safe_psd_root: smooth finite gradient at repeated eigenvalues (where raw eigh is NaN)
    q_orth, _ = np.linalg.qr(np.array([[1.2, -0.7, 0.4], [0.5, 1.4, -0.9], [-0.8, 0.3, 1.1]]))
    q_jax = jnp.asarray(q_orth, dtype=jnp.float64)
    evals_rep = jnp.array([2.7, 2.7, 0.85], dtype=jnp.float64)
    sym_rep = q_jax @ jnp.diag(evals_rep) @ q_jax.T
    root_rep = _safe_psd_root(sym_rep)
    np.testing.assert_allclose(root_rep @ root_rep.T, sym_rep, rtol=1e-12, atol=1e-12)

    probe_u = jnp.array([0.4, -0.9, 1.3], dtype=jnp.float64)
    probe_v = jnp.array([-0.6, 1.1, 0.5], dtype=jnp.float64)
    grad_rep = jax.grad(lambda m: jnp.dot(probe_u, _safe_psd_root(m) @ probe_v))(sym_rep)
    assert bool(jnp.all(jnp.isfinite(grad_rep)))

    # _safe_psd_root(jitter=0.0) on partial-flat precision diag([0.0, 1e8]) rotated by Q:
    # must NOT leak trace-scaled 1e-13 * 1e8 = 1e-5 diagonal precision onto the flat mode!
    q2 = jnp.array([[0.8, -0.6], [0.6, 0.8]], dtype=jnp.float64)
    null_dir = q2[:, 0]
    pi_partial = q2 @ jnp.diag(jnp.array([0.0, 1.0e8], dtype=jnp.float64)) @ q2.T
    root_z = _safe_psd_root(pi_partial, jitter=0.0)
    np.testing.assert_allclose(root_z @ root_z.T, pi_partial, rtol=1e-13, atol=1e-20)
    # Rounding floor on [-6000, 8000] @ [0.8, 0.6] is ~9e-13 (vs 2.24e-3 if jitter=1e-13 leaked!)
    assert float(jnp.linalg.norm(root_z.T @ null_dir)) < 1e-11

    # Slight negative numerical overshoot (-1e-14) falls back cleanly under jit
    sym_indef = q_jax @ jnp.diag(jnp.array([3.1, 0.9, -1e-14], dtype=jnp.float64)) @ q_jax.T
    root_indef = jax.jit(_safe_psd_root)(sym_indef)
    assert bool(jnp.all(jnp.isfinite(root_indef)))
    expected_psd = q_jax @ jnp.diag(jnp.array([3.1, 0.9, 0.0], dtype=jnp.float64)) @ q_jax.T
    np.testing.assert_allclose(root_indef @ root_indef.T, expected_psd, rtol=1e-12, atol=1e-13)


# ---------------------------------------------------------------------------
# 4. Package 9: Shared whitening & offset design/unpacking + active_parameter_mask
# ---------------------------------------------------------------------------


def test_pkg9_shared_whitening_and_offset_design_against_mvn_oracle() -> None:
    """Verify shared whitening & offset helpers against brute-force MVN across all offset_modes."""
    from ticktack.likelihood import (
        _build_marginal_design,
        _nonlinear_residual,
        _unpack_offset_coeffs,
        _whiten_model_data,
    )

    cbm, ds, gf = _make_asymmetric_fixture()
    spike = tt.MiyakeSpike(t_start=774.33, duration=0.58, fluence=6.27, profile="boxcar")
    solar = tt.HarmonicSolar(periods=10.9, c_cos=0.23, c_sin=-0.16, t_ref=773.1)
    gp = tt.GPProduction(tgk.quasisep.Matern32(scale=4.1, sigma=0.27), n_modes=10, pad_years=5.0)
    model = tt.RadiocarbonModel(
        cbm,
        tt.ProductionSum(
            steady=tt.SteadyProduction(1.81), spike=spike, solar=solar, gp=gp, field=gf
        ),
        q_std=1.81,
    )
    jitter_val = 0.43
    sigma_eff = jnp.sqrt(jnp.square(ds.sigma_flat) + jitter_val**2)
    n_obs = ds.n_obs

    for mode in ("hierarchical", "per_series", "shared", "none"):
        cfg = tt.LinearMarginalConfig(
            offset_mode=mode,
            global_prior_mean=-16.4,
            global_prior_sd=88.0,
            lab_prior_sd=1.75,
            solar_prior_sd=0.68,
            marginalize_solar=True,
            marginalize_gp=True,
            marginalize_field=True,
            whiten_method="qr",
        )
        u_data, hld = _whiten_model_data(model, ds, cfg, sigma_eff, joint_mode=False)
        ll_got = float(tt.marginal_log_likelihood(model, ds, cfg, sigma_jitter=jitter_val))
        ll_from_whiten = float(-0.5 * jnp.dot(u_data, u_data) - hld - 0.5 * n_obs * math.log(2.0 * math.pi))
        np.testing.assert_allclose(ll_got, ll_from_whiten, rtol=1e-14, atol=1e-14)

        # Brute-force dense multivariate normal oracle:
        r_nl = np.asarray(_nonlinear_residual(model, ds, cfg), dtype=np.float64)
        x_mat, pm, ps = (np.asarray(a, dtype=np.float64) for a in _build_marginal_design(model, ds, cfg))
        cov_dense = np.diag(np.asarray(sigma_eff, dtype=np.float64) ** 2) + (x_mat * (ps**2)[None, :]) @ x_mat.T
        sign_ld, logdet_dense = np.linalg.slogdet(cov_dense)
        assert sign_ld > 0
        r_shifted = r_nl - x_mat @ pm
        ll_mvn = -0.5 * (
            float(r_shifted @ np.linalg.solve(cov_dense, r_shifted))
            + float(logdet_dense)
            + n_obs * math.log(2.0 * math.pi)
        )
        np.testing.assert_allclose(ll_got, ll_mvn, rtol=5e-12, atol=5e-12)
        np.testing.assert_allclose(float(hld), 0.5 * float(logdet_dense), rtol=1e-12, atol=1e-12)

        # Posterior offset unpacking parity (1D and 2D batched)
        lp = tt.linear_posterior(
            model, ds, cfg, rng_key=jax.random.PRNGKey(19), n_draws=5, sigma_jitter=jitter_val
        )
        g_1d, l_1d = _unpack_offset_coeffs(lp.coeff_mean, cfg, ds.n_series)
        np.testing.assert_allclose(g_1d, lp.offset_global_mean, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(l_1d, lp.offset_lab_means, rtol=1e-14, atol=1e-14)
        assert lp.coeff_draws is not None
        g_2d, l_2d = _unpack_offset_coeffs(lp.coeff_draws, cfg, ds.n_series)
        assert g_2d.shape == (5,)
        assert l_2d.shape == (5, ds.n_series)
        for d_idx in range(5):
            gd_i, ld_i = _unpack_offset_coeffs(lp.coeff_draws[d_idx], cfg, ds.n_series)
            np.testing.assert_allclose(g_2d[d_idx], gd_i, rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(l_2d[d_idx], ld_i, rtol=1e-14, atol=1e-14)


def test_pkg9_fisher_active_parameter_mask_protocol() -> None:
    """``cbm_fisher_spectrum`` honours ``theta_cbm.active_parameter_mask()`` before attribute fallback."""

    class _CustomChartTheta(eqx.Module):
        rates_a: jax.Array
        gauge_scale: jax.Array
        rates_b: jax.Array

        def active_parameter_mask(self) -> jax.Array:
            return jnp.array([True, False, True])

    theta0 = _CustomChartTheta(
        rates_a=jnp.asarray(0.42, dtype=jnp.float64),
        gauge_scale=jnp.asarray(9.1, dtype=jnp.float64),
        rates_b=jnp.asarray(-0.31, dtype=jnp.float64),
    )
    t_grid = jnp.linspace(0.5, 6.5, 9, dtype=jnp.float64)
    sigma = jnp.linspace(0.8, 1.4, 9, dtype=jnp.float64)

    def _fwd(th: _CustomChartTheta) -> jax.Array:
        # Independent of th.gauge_scale (exact gauge invariance)
        return 12.3 * jnp.exp(-jnp.exp(th.rates_a) * t_grid) + 4.7 * jnp.exp(
            -jnp.exp(th.rates_b) * t_grid
        )

    spec = cbm_fisher_spectrum(_fwd, theta0, sigma, prior_cbm_sd=0.37)
    np.testing.assert_array_equal(np.asarray(spec.active_mask), np.array([True, False, True]))
    assert spec.eigenvalues.shape == (2,)
    assert spec.parameter_loadings.shape == (3, 2)
    np.testing.assert_allclose(spec.parameter_loadings[1, :], 0.0, atol=1e-15)
    assert bool(jnp.all(jnp.isfinite(spec.eigenvalues)))
    assert float(spec.eigenvalues[0]) > float(spec.eigenvalues[1]) > 0.0

    # exclude_total_mass=False disables auto-masking
    spec_all = cbm_fisher_spectrum(
        _fwd, theta0, sigma, prior_cbm_sd=0.37, exclude_total_mass=False
    )
    np.testing.assert_array_equal(np.asarray(spec_all.active_mask), np.array([True, True, True]))
    assert spec_all.eigenvalues.shape == (3,)
    assert float(spec_all.eigenvalues[-1]) == pytest.approx(0.0, abs=1e-24)
