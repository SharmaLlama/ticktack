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
import ticktack.sampling as tts
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
# 2. Decision 5.6: Explicit Fisher prior widths & active-length 1D prior_sd
# ---------------------------------------------------------------------------


def test_fisher_requires_explicit_prior_widths_and_supports_active_1d_prior() -> None:
    """Decision 5.6: explicit priors, active-length 1D ``prior_sd``, and SVD plotting contract."""
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

    # Heterogeneous 1D active-length prior_sd (56,) on 57-leaf Brehm21 (56 free rates + log_total_mass):
    n_free = int(cbm.log_rates_free.size)
    sd_active = jnp.linspace(0.18, 0.54, n_free, dtype=jnp.float64)
    sd_full_inf = jnp.concatenate([sd_active, jnp.array([jnp.inf], dtype=jnp.float64)])
    spec_act_1d = tt.cbm_gp_fisher_spectrum(cbm, ds, gf, prior_sd=sd_active)
    spec_full_1d = tt.cbm_gp_fisher_spectrum(cbm, ds, gf, prior_sd=sd_full_inf)
    assert spec_act_1d.eigenvalues.shape == (n_free,)
    np.testing.assert_array_equal(
        np.asarray(spec_act_1d.active_mask),
        np.array([True] * n_free + [False]),
    )
    np.testing.assert_allclose(
        spec_act_1d.eigenvalues, spec_full_1d.eigenvalues, rtol=1e-12, atol=1e-13
    )
    np.testing.assert_allclose(float(spec_act_1d.d_eff), float(spec_full_1d.d_eff), rtol=1e-12)
    np.testing.assert_allclose(
        float(spec_act_1d.info_nats), float(spec_full_1d.info_nats), rtol=1e-12
    )
    np.testing.assert_allclose(
        spec_act_1d.posterior_cov, spec_full_1d.posterior_cov, rtol=1e-11, atol=1e-12
    )

    # plot_cbm_mode_impulse_responses rejects raw CBM with spectrum=None and accepts CBMModeImpulseResult:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cbm_m17 = tt.load_preset("Miyake17", kind="open")
    with pytest.raises(ValueError, match="cbm_mode_impulse_responses"):
        tt.plot_cbm_mode_impulse_responses(cbm_m17, None, t_lag=t_lag)

    with pytest.raises(TypeError, match="prior_cbm_sd"):
        tt.plot_cbm_mode_impulse_responses(  # type: ignore[call-arg]
            cbm_m17, spec, prior_cbm_sd=0.35
        )

    svd_imp = tt.cbm_mode_impulse_responses(cbm_m17, t_lag, None, prior_cbm_sd=0.35)
    fig_svd, axes_svd = tt.plot_cbm_mode_impulse_responses(svd_imp, modes=(0, 1), ncols=2)
    assert axes_svd.size >= 2
    plt.close(fig_svd)


# ---------------------------------------------------------------------------
# 3. Package 1: ticktack._utils shared helpers & inlined _uses_joint_mode
# ---------------------------------------------------------------------------


def test_pkg1_utils_concrete_sigma_jitter_and_safe_psd_root() -> None:
    """Verify ``_concrete``, ``_sigma_with_jitter``, ``_safe_psd_root``, and removed ``_use_joint_mode``."""
    from ticktack._utils import _concrete, _safe_psd_root, _sigma_with_jitter

    assert not hasattr(ttf, "_use_joint_mode")
    assert not hasattr(tts, "_use_joint_mode")

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
    np.testing.assert_allclose(root_rep @ root_rep.T, sym_rep, rtol=1e-11, atol=1e-11)

    probe_u = jnp.array([0.4, -0.9, 1.3], dtype=jnp.float64)
    probe_v = jnp.array([-0.6, 1.1, 0.5], dtype=jnp.float64)
    grad_rep = jax.grad(lambda m: jnp.dot(probe_u, _safe_psd_root(m) @ probe_v))(sym_rep)
    assert bool(jnp.all(jnp.isfinite(grad_rep)))

    # _safe_psd_root(jitter=0.0) on positive-rounding rank-1 outer product v @ v.T
    # (where unguarded cholesky produces a spurious ~1.29e-4 pivot):
    v_round = 1.0e4 * jnp.array(
        [1.0 + 3.0 * 2.0**-28, 1.0 + 5.0 * 2.0**-28], dtype=jnp.float64
    )
    pi_round = v_round[:, None] @ v_round[None, :]
    null_round = jnp.array([-v_round[1], v_round[0]], dtype=jnp.float64) / jnp.linalg.norm(
        v_round
    )
    jit_zero_root = jax.jit(lambda m: _safe_psd_root(m, jitter=0.0))
    for r_v in (_safe_psd_root(pi_round, jitter=0.0), jit_zero_root(pi_round)):
        np.testing.assert_allclose(r_v @ r_v.T, pi_round, rtol=1e-13, atol=1e-20)
        assert float(jnp.linalg.norm(r_v.T @ null_round)) < 1e-11

    # Sweep non-Pythagorean rotation angles on partial-flat precision diag([0.0, 1e8]):
    for ang in (0.17, 0.73, 1.19, 2.04, 2.81):
        c_a, s_a = math.cos(ang), math.sin(ang)
        q_rot = jnp.array([[c_a, -s_a], [s_a, c_a]], dtype=jnp.float64)
        null_dir = q_rot[:, 0]
        pi_partial = q_rot @ jnp.diag(jnp.array([0.0, 1.0e8], dtype=jnp.float64)) @ q_rot.T
        root_z = jit_zero_root(pi_partial)
        np.testing.assert_allclose(root_z @ root_z.T, pi_partial, rtol=1e-13, atol=1e-20)
        assert float(jnp.linalg.norm(root_z.T @ null_dir)) < 1e-11

    # Scale-equivariant jitter=0.0 handling of tiny valid precision [0.0, 1e-16] (not zeroed by 1.0 floor):
    c_s, s_s = math.cos(0.61), math.sin(0.61)
    q_s = jnp.array([[c_s, -s_s], [s_s, c_s]], dtype=jnp.float64)
    for pi_tiny in (
        jnp.diag(jnp.array([0.0, 1.0e-16], dtype=jnp.float64)),
        q_s @ jnp.diag(jnp.array([0.0, 1.0e-16], dtype=jnp.float64)) @ q_s.T,
    ):
        root_tiny = jit_zero_root(pi_tiny)
        np.testing.assert_allclose(root_tiny @ root_tiny.T, pi_tiny, rtol=1e-13, atol=1e-30)
        assert float(jnp.linalg.norm(root_tiny)) == pytest.approx(1.0e-8, rel=1e-12)

    # Indefinite eigenvalue < -1e-8 (exceeding jitter * trace / n) falls back cleanly to eigh under jit:
    sym_indef = q_jax @ jnp.diag(jnp.array([3.1, 0.9, -3.5e-6], dtype=jnp.float64)) @ q_jax.T
    root_indef = jax.jit(_safe_psd_root)(sym_indef)
    assert bool(jnp.all(jnp.isfinite(root_indef)))
    expected_psd = q_jax @ jnp.diag(jnp.array([3.1, 0.9, 0.0], dtype=jnp.float64)) @ q_jax.T
    np.testing.assert_allclose(root_indef @ root_indef.T, expected_psd, rtol=1e-12, atol=1e-13)
    grad_indef = jax.grad(jax.jit(lambda m: jnp.dot(probe_u, _safe_psd_root(m) @ probe_v)))(
        sym_indef
    )
    assert bool(jnp.all(jnp.isfinite(grad_indef)))


# ---------------------------------------------------------------------------
# 4. Package 9: Shared whitening & offset design/unpacking + active_parameter_mask
# ---------------------------------------------------------------------------


def test_pkg9_shared_whitening_and_offset_design_against_mvn_oracle() -> None:
    """Verify marginal & joint whitening and offset unpacking against independent dense-MVN oracles."""
    from ticktack.likelihood import (
        _build_marginal_design,
        _nonlinear_residual,
        _offset_only_cfg,
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
    sigma_np = np.asarray(sigma_eff, dtype=np.float64)
    n_obs = ds.n_obs
    n_s = ds.n_series

    cases = (
        ("hierarchical", 1.75),
        ("hierarchical", 0.0),
        ("per_series", 1.75),
        ("shared", 1.75),
        ("none", 1.75),
    )
    for mode, lab_sd in cases:
        cfg = tt.LinearMarginalConfig(
            offset_mode=mode,
            global_prior_mean=-16.4,
            global_prior_sd=88.0,
            lab_prior_sd=lab_sd,
            solar_prior_sd=0.68,
            marginalize_solar=True,
            marginalize_gp=True,
            marginalize_field=True,
            whiten_method="qr",
        )
        u_data, hld = _whiten_model_data(model, ds, cfg, sigma_eff, joint_mode=False)
        ll_got = float(tt.marginal_log_likelihood(model, ds, cfg, sigma_jitter=jitter_val))
        ll_from_whiten = float(
            -0.5 * jnp.dot(u_data, u_data) - hld - 0.5 * n_obs * math.log(2.0 * math.pi)
        )
        np.testing.assert_allclose(ll_got, ll_from_whiten, rtol=1e-14, atol=1e-14)

        # Brute-force dense multivariate normal oracle (marginal mode):
        r_nl = np.asarray(_nonlinear_residual(model, ds, cfg), dtype=np.float64)
        x_mat, pm, ps = (
            np.asarray(a, dtype=np.float64) for a in _build_marginal_design(model, ds, cfg)
        )
        cov_dense = np.diag(sigma_np**2) + (x_mat * (ps**2)[None, :]) @ x_mat.T
        sign_ld, logdet_dense = np.linalg.slogdet(cov_dense)
        assert sign_ld > 0
        r_shifted = r_nl - x_mat @ pm
        cov_inv_r = np.linalg.solve(cov_dense, r_shifted)
        ll_mvn = -0.5 * (
            float(r_shifted @ cov_inv_r) + float(logdet_dense) + n_obs * math.log(2.0 * math.pi)
        )
        np.testing.assert_allclose(ll_got, ll_mvn, rtol=5e-12, atol=5e-12)
        np.testing.assert_allclose(float(hld), 0.5 * float(logdet_dense), rtol=1e-12, atol=1e-12)

        # Independent dense-Gaussian posterior mean oracle w_mvn for offset unpacking:
        w_mvn = pm + (ps**2) * (x_mat.T @ cov_inv_r)
        lp = tt.linear_posterior(
            model, ds, cfg, rng_key=jax.random.PRNGKey(19), n_draws=5, sigma_jitter=jitter_val
        )
        assert lp.coeff_draws is not None
        draws_np = np.asarray(lp.coeff_draws, dtype=np.float64)
        if mode == "hierarchical" and lab_sd > 0.0:
            g_oracle = float(w_mvn[0])
            l_oracle = w_mvn[1 : 1 + n_s]
            g_draws_oracle = draws_np[:, 0]
            l_draws_oracle = draws_np[:, 1 : 1 + n_s]
            assert abs(g_oracle) > 1.0 and float(np.linalg.norm(l_oracle)) > 0.1
        elif mode == "hierarchical":
            g_oracle = float(w_mvn[0])
            l_oracle = np.zeros(n_s, dtype=np.float64)
            g_draws_oracle = draws_np[:, 0]
            l_draws_oracle = np.zeros((5, n_s), dtype=np.float64)
            assert abs(g_oracle) > 1.0
        elif mode == "per_series":
            l_oracle = w_mvn[:n_s]
            g_oracle = float(np.mean(l_oracle))
            l_draws_oracle = draws_np[:, :n_s]
            g_draws_oracle = np.mean(l_draws_oracle, axis=1)
            assert abs(g_oracle) > 1.0 and float(np.linalg.norm(l_oracle)) > 1.0
        elif mode == "shared":
            g_oracle = float(w_mvn[0])
            l_oracle = np.zeros(n_s, dtype=np.float64)
            g_draws_oracle = draws_np[:, 0]
            l_draws_oracle = np.zeros((5, n_s), dtype=np.float64)
            assert abs(g_oracle) > 1.0
        else:
            g_oracle = 0.0
            l_oracle = np.zeros(n_s, dtype=np.float64)
            g_draws_oracle = np.zeros(5, dtype=np.float64)
            l_draws_oracle = np.zeros((5, n_s), dtype=np.float64)

        g_1d, l_1d = _unpack_offset_coeffs(lp.coeff_mean, cfg, n_s)
        np.testing.assert_allclose(float(g_1d), g_oracle, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(l_1d, l_oracle, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(float(lp.offset_global_mean), g_oracle, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(lp.offset_lab_means, l_oracle, rtol=1e-11, atol=1e-11)

        g_2d, l_2d = _unpack_offset_coeffs(lp.coeff_draws, cfg, n_s)
        assert g_2d.shape == (5,) and l_2d.shape == (5, n_s)
        np.testing.assert_allclose(g_2d, g_draws_oracle, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(l_2d, l_draws_oracle, rtol=1e-14, atol=1e-14)

        # Dense-MVN parity for _whiten_model_data(..., joint_mode=True):
        u_joint, hld_joint = _whiten_model_data(model, ds, cfg, sigma_eff, joint_mode=True)
        ll_joint_whiten = float(
            -0.5 * jnp.dot(u_joint, u_joint)
            - hld_joint
            - 0.5 * n_obs * math.log(2.0 * math.pi)
        )
        ll_joint_api = float(tt.joint_log_likelihood(model, ds, cfg=cfg, sigma_jitter=jitter_val))
        np.testing.assert_allclose(ll_joint_whiten, ll_joint_api, rtol=1e-14, atol=1e-14)

        pred_full = np.asarray(model.predict_d14c(ds, offsets=None, concat=True), dtype=np.float64)
        r_joint_raw = np.asarray(ds.d14c_flat, dtype=np.float64) - pred_full
        if mode == "none":
            np.testing.assert_allclose(u_joint, r_joint_raw / sigma_np, rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(
                float(hld_joint), float(np.sum(np.log(sigma_np))), rtol=1e-14, atol=1e-14
            )
        else:
            x_off, pm_off, ps_off = (
                np.asarray(a, dtype=np.float64)
                for a in _build_marginal_design(model, ds, _offset_only_cfg(cfg))
            )
            exp_cols = (1 + n_s) if (mode == "hierarchical" and lab_sd > 0.0) else (n_s if mode == "per_series" else 1)
            assert x_off.shape == (n_obs, exp_cols)
            cov_joint = np.diag(sigma_np**2) + (x_off * (ps_off**2)[None, :]) @ x_off.T
            sign_j, logdet_joint = np.linalg.slogdet(cov_joint)
            assert sign_j > 0
            r_j_shift = r_joint_raw - x_off @ pm_off
            ll_joint_mvn = -0.5 * (
                float(r_j_shift @ np.linalg.solve(cov_joint, r_j_shift))
                + float(logdet_joint)
                + n_obs * math.log(2.0 * math.pi)
            )
            np.testing.assert_allclose(ll_joint_whiten, ll_joint_mvn, rtol=5e-12, atol=5e-12)
            np.testing.assert_allclose(
                float(hld_joint), 0.5 * float(logdet_joint), rtol=1e-12, atol=1e-12
            )


def test_pkg9_fisher_active_parameter_mask_protocol() -> None:
    """``cbm_fisher_spectrum`` honours ``theta_cbm.active_parameter_mask()`` across scalar, 1D, and 2D priors."""

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

    # Active-length 1D prior_cbm_sd (2,) and 2D prior_cbm_cov (2, 2) match explicit 3-element +inf oracle:
    sd_act2 = jnp.array([0.29, 0.53], dtype=jnp.float64)
    sd_full3 = jnp.array([0.29, jnp.inf, 0.53], dtype=jnp.float64)
    spec_1d = cbm_fisher_spectrum(_fwd, theta0, sigma, prior_cbm_sd=sd_act2)
    spec_1d_ref = cbm_fisher_spectrum(_fwd, theta0, sigma, prior_cbm_sd=sd_full3)
    np.testing.assert_allclose(spec_1d.eigenvalues, spec_1d_ref.eigenvalues, rtol=1e-13)
    np.testing.assert_allclose(spec_1d.posterior_cov, spec_1d_ref.posterior_cov, rtol=1e-13)

    cov_act2 = jnp.array([[0.11, 0.03], [0.03, 0.27]], dtype=jnp.float64)
    spec_2d = jax.jit(lambda th: cbm_fisher_spectrum(_fwd, th, sigma, prior_cbm_cov=cov_act2))(
        theta0
    )
    assert spec_2d.eigenvalues.shape == (2,)
    assert bool(jnp.all(jnp.isfinite(spec_2d.eigenvalues)))

    # exclude_total_mass=False disables auto-masking
    spec_all = cbm_fisher_spectrum(
        _fwd, theta0, sigma, prior_cbm_sd=0.37, exclude_total_mass=False
    )
    np.testing.assert_array_equal(np.asarray(spec_all.active_mask), np.array([True, True, True]))
    assert spec_all.eigenvalues.shape == (3,)
    assert float(spec_all.eigenvalues[-1]) == pytest.approx(0.0, abs=1e-24)
