"""Comprehensive test suite for ``ticktack.linear`` and ``ticktack.fisher``.

Verifies:
1. Analytic linear marginalization (`LinearMarginal`, `whiten_cholesky`,
   `whiten_qr`, `whiten_rank_one`, `whiten_blocks`, `arrow_problem`/`arrow_at`,
   `posterior`, `hierarchical_offset_design`, `harmonic_solar_design`),
   including the exact ``-0.5 * ln det(I + U^T U)`` Occam factor and closed-form
   posterior mean/covariance.
2. Causal Solin & Särkkä (2020) spectral GP projection (`solin_sarkka_basis`,
   `spectral_density`, `stationary_cbm_gp_cov`, `project_causal_gp`,
   `canonical_pod_modes`, `kalman_gp_marginal`):
   - Cross-checked against BOTH brute-force dense fine-grid integration from
     ``t_0 - 500 yr`` and exact continuous-time `kalman_gp_marginal` (point and
     seasonal-binned observations, multi-box stiff CBMs with slow ``λ_14``).
   - Exponential / rapid convergence with rank ``r`` (``< 1e-8`` at ``r = 80..120``).
   - Zero ``2×`` initial-state variance double-counting at ``t = a``.
   - Second-order autodiff check (`jax.test_util.check_grads`) wrt both ``A``
     and GP log-hyperparameters ``(ln σ_Q, ln ℓ)`` on both SDE Smith-doubling
     and spectral-quadrature paths.
3. Carbon-cycle Fisher information (`expected_fisher`, `cbm_fisher_spectrum`,
   `relaxation_fisher_modes`), including exact coordinate-reparameterization
   invariance, QR Schur-complement accuracy vs joint inverse covariance, zero
   ``inf * 0 = NaN`` hazard on gauge parameters, and McWhirter & Pike's
   analytical ``π / cosh(π ω_ln)`` Mellin singular-value decay.
"""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Guard against legacy v1 ticktack/__init__.py importing pkg_resources (and matplotlib/emcee)
# before Wave 2 replaces the package root __init__.py:
if (SRC_DIR / "ticktack" / "ticktack.py").exists():
    pkg = types.ModuleType("ticktack")
    pkg.__path__ = [str(SRC_DIR / "ticktack")]
    sys.modules.setdefault("ticktack", pkg)

import jax.numpy as jnp
import jax.scipy.linalg as jsl
import jax.scipy.stats as jss
import numpy as np
import pytest
import tinygp.kernels as tgk
import zodiax as zdx
from jax.test_util import check_grads
from tinygp.kernels import quasisep as tgq

from ticktack.fisher import (
    FisherSpectrumResult,
    RelaxationModesResult,
    cbm_fisher_spectrum,
    expected_fisher,
    fisher_projection,
    laplace_covariance,
    regularized_inverse,
    relaxation_fisher_modes,
)
from ticktack.linear import (
    LinearMarginal,
    arrow_at,
    arrow_problem,
    canonical_pod_modes,
    harmonic_solar_design,
    hierarchical_offset_design,
    kalman_gp_marginal,
    posterior,
    project_causal_gp,
    solin_sarkka_basis,
    spectral_density,
    stationary_cbm_gp_cov,
    whiten_blocks,
    whiten_cholesky,
    whiten_qr,
    whiten_rank_one,
)

LAMBDA_14 = math.log(2.0) / 5730.0


def _make_stiff_three_box_cbm(
    theta_rates: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Construct a realistic stiff 3-box CBM (troposphere, biosphere, deep ocean).

    Eigenvalues span fast tropospheric exchange (~2.5 yr⁻¹) down to radiocarbon
    decay ``λ_14 ≈ 1.21e-4 yr⁻¹`` (condition number ``κ(A) > 2 × 10⁴``).
    """
    if theta_rates is None:
        theta_rates = jnp.log(
            jnp.array([1.4, 0.8, 0.9, 0.025, 0.002], dtype=jnp.float64)
        )
    rates = jnp.exp(theta_rates)
    k01, k10, k02, k20, k_sink = rates[0], rates[1], rates[2], rates[3], rates[4]
    a_mat = jnp.array(
        [
            [-(k01 + k02 + LAMBDA_14), k10, k20],
            [k01, -(k10 + LAMBDA_14), 0.0],
            [k02, 0.0, -(k20 + k_sink + LAMBDA_14)],
        ],
        dtype=jnp.float64,
    )
    b_vec = jnp.array([0.75, 0.25, 0.0], dtype=jnp.float64)
    c_obs = jnp.array([1.2, 0.15, 0.0], dtype=jnp.float64)
    return a_mat, b_vec, c_obs


def _psi1(mat: jax.Array) -> jax.Array:
    n = mat.shape[0]
    z = jnp.zeros((n, n), dtype=jnp.float64)
    eye = jnp.eye(n, dtype=jnp.float64)
    return jsl.expm(jnp.block([[mat, eye], [z, z]]))[:n, n:]


# =============================================================================
# 1. Analytic Linear Marginalization & Design Helpers
# =============================================================================


def test_whitening_methods_and_linear_marginal_exactness():
    rng = np.random.default_rng(42)
    n_obs, k_modes = 36, 9
    x_std = jnp.asarray(rng.standard_normal((n_obs,)))
    u_mat = jnp.asarray(rng.standard_normal((n_obs, k_modes)) * 1.7)

    u_chol, ld_chol = whiten_cholesky(x_std, u_mat)
    u_qr, ld_qr = whiten_qr(x_std, u_mat)
    u_r1, ld_r1 = whiten_rank_one(x_std, u_mat)

    cov_std = jnp.eye(n_obs) + u_mat @ u_mat.T
    quad_ref = x_std @ jnp.linalg.solve(cov_std, x_std)
    _, logdet_full = jnp.linalg.slogdet(cov_std)
    half_ld_ref = 0.5 * logdet_full

    for u_vec, half_ld in ((u_chol, ld_chol), (u_qr, ld_qr), (u_r1, ld_r1)):
        np.testing.assert_allclose(u_vec @ u_vec, quad_ref, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(half_ld, half_ld_ref, rtol=1e-13, atol=1e-13)

    om_mean, om_cov = posterior(x_std, u_mat)
    s_std = jnp.eye(k_modes) + u_mat.T @ u_mat
    np.testing.assert_allclose(
        om_mean, jnp.linalg.solve(s_std, u_mat.T @ x_std), rtol=1e-13, atol=1e-13
    )
    np.testing.assert_allclose(om_cov, jnp.linalg.inv(s_std), rtol=1e-13, atol=1e-13)

    # Full LinearMarginal with hierarchical offsets + harmonic quadratures + GP columns:
    t_starts = jnp.linspace(770.0, 787.5, n_obs)
    t_ends = t_starts + 0.5
    a_mat, b_vec, _ = _make_stiff_three_box_cbm()
    series_idx = jnp.asarray(np.arange(n_obs) % 3, dtype=jnp.int32)
    box_ratios = jnp.where(series_idx == 2, 0.99562, 1.0)
    x_off, mu_off, sd_off = hierarchical_offset_design(
        series_idx,
        n_series=3,
        box_activity_ratios=box_ratios,
        global_prior_mean=-2.5,
        global_prior_sd=200.0,
        lab_prior_sd=1.2,
    )
    x_harm = harmonic_solar_design(
        a_mat,
        b_vec,
        omegas=jnp.array([2.0 * math.pi / 11.0, 2.0 * math.pi / 22.0]),
        t_starts=t_starts,
        t_ends=t_ends,
        alpha_unit=1.7,
        box_indices=0,
        obs_scale=1.2,
    )
    x_gp = jnp.asarray(rng.standard_normal((n_obs, 5)))
    design = jnp.concatenate([x_gp, x_harm, x_off], axis=1)
    prior_mean = jnp.concatenate([jnp.zeros(9), mu_off])
    prior_sd = jnp.concatenate([jnp.full(5, 1.0), jnp.full(4, 2.5), sd_off])

    m0 = jnp.sin(0.3 * t_starts)
    true_w = prior_mean + jnp.asarray(rng.standard_normal(design.shape[1])) * 0.4
    sigma_1d = jnp.linspace(0.9, 1.8, n_obs)
    data = m0 + design @ true_w + jnp.asarray(rng.standard_normal(n_obs)) * sigma_1d
    resid = data - m0

    dense_prior_cov = jnp.diag(prior_sd**2)
    dense_data_cov_1d = jnp.diag(sigma_1d**2) + design @ dense_prior_cov @ design.T
    ll_ref_1d = jss.multivariate_normal.logpdf(
        data, m0 + design @ prior_mean, dense_data_cov_1d
    )

    ll_by_method = {}
    for method in ("cholesky", "qr", "rank_one"):
        lm = LinearMarginal(
            design, prior_mean=prior_mean, prior_sd=prior_sd, method=method
        )
        assert isinstance(lm, zdx.Base)
        ll_by_method[method] = lm.loglike(resid, sigma_1d)
        # Dense D + X Λ X^T has κ ~ 2e6 from σ_global = 200‰; QR/Cholesky/rank_one agree to 1e-13!
        np.testing.assert_allclose(
            ll_by_method[method], ll_ref_1d, rtol=1e-10, atol=1e-10
        )
    np.testing.assert_allclose(
        ll_by_method["cholesky"], ll_by_method["qr"], rtol=1e-13, atol=1e-13
    )
    np.testing.assert_allclose(
        ll_by_method["rank_one"], ll_by_method["qr"], rtol=1e-13, atol=1e-13
    )

    # Explicitly check the -0.5 * ln det(I + U^T U) Occam penalty difference between two priors:
    lm_wide = LinearMarginal(
        design, prior_mean=prior_mean, prior_sd=2.0 * prior_sd, method="qr"
    )
    cov_wide = jnp.diag(sigma_1d**2) + design @ (4.0 * dense_prior_cov) @ design.T
    ll_ref_wide = jss.multivariate_normal.logpdf(
        data, m0 + design @ prior_mean, cov_wide
    )
    np.testing.assert_allclose(
        lm_wide.loglike(resid, sigma_1d) - ll_by_method["qr"],
        ll_ref_wide - ll_ref_1d,
        rtol=1e-10,
        atol=1e-10,
    )

    # Check posterior mean & covariance against exact dense Bayes formula (with 2D correlated noise D):
    b_corr = jnp.asarray(rng.standard_normal((n_obs, 4))) * 0.35
    d_cov_2d = jnp.diag(sigma_1d**2) + b_corr @ b_corr.T
    lm_cov = LinearMarginal(
        design, prior_mean=prior_mean, prior_cov=dense_prior_cov, method="qr"
    )
    np.testing.assert_allclose(
        lm_cov.loglike(resid, d_cov_2d),
        jss.multivariate_normal.logpdf(
            data,
            m0 + design @ prior_mean,
            d_cov_2d + design @ dense_prior_cov @ design.T,
        ),
        rtol=1e-10,
        atol=1e-10,
    )

    post_mean, post_cov = lm_cov.posterior(resid, d_cov_2d)
    d_inv = jnp.linalg.inv(d_cov_2d)
    post_prec_ref = jnp.linalg.inv(dense_prior_cov) + design.T @ d_inv @ design
    post_cov_ref = jnp.linalg.inv(post_prec_ref)
    post_mean_ref = prior_mean + post_cov_ref @ (
        design.T @ d_inv @ (resid - design @ prior_mean)
    )
    np.testing.assert_allclose(post_mean, post_mean_ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(post_cov, post_cov_ref, rtol=1e-10, atol=1e-10)

    pred_mean, pred_cov = lm_cov.predict_posterior(resid, d_cov_2d)
    np.testing.assert_allclose(pred_mean, design @ post_mean_ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        pred_cov, design @ post_cov_ref @ design.T, rtol=1e-10, atol=1e-10
    )


def test_arrow_problem_and_harmonic_solar_eternal_response():
    rng = np.random.default_rng(7)
    n_obs = 24
    x = jnp.asarray(rng.standard_normal(n_obs))
    rows = jnp.arange(n_obs).reshape(4, 6)
    local = jnp.asarray(rng.standard_normal((4, 6, 2)))
    spanning = jnp.asarray(rng.standard_normal((n_obs, 3)))

    prob = arrow_problem(x, rows, local, spanning)
    scale2 = 1.65
    chi2, norm, r2, tr = arrow_at(prob, scale2, trace=True)

    u_local = np.zeros((n_obs, 4 * 2))
    for b_idx in range(4):
        u_local[np.asarray(rows[b_idx]), 2 * b_idx : 2 * (b_idx + 1)] = np.asarray(
            local[b_idx]
        )
    u_dense = jnp.concatenate([jnp.asarray(u_local), spanning], axis=1)
    cov_s = scale2 * jnp.eye(n_obs) + u_dense @ u_dense.T
    chi2_ref = x @ jnp.linalg.solve(cov_s, x)
    _, logdet_cov = jnp.linalg.slogdet(cov_s)
    norm_ref = 0.5 * (logdet_cov - n_obs * jnp.log(scale2))
    tr_ref = jnp.trace(
        jnp.linalg.inv(scale2 * jnp.eye(u_dense.shape[1]) + u_dense.T @ u_dense)
    )
    np.testing.assert_allclose(chi2, chi2_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(norm, norm_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(tr, tr_ref, rtol=1e-12, atol=1e-12)
    assert float(r2) >= 0.0

    u_wb, extra_wb = whiten_blocks(x, rows, local, spanning)
    chi2_1, norm_1, _ = arrow_at(prob, 1.0)
    np.testing.assert_allclose(u_wb @ u_wb, chi2_1, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(jnp.sum(extra_wb), norm_1, rtol=1e-12, atol=1e-12)

    # Verify harmonic_solar_design against direct ODE convolution on an asymptotically fast CBM
    a_fast = jnp.array([[-1.2, 0.4], [0.5, -0.7]], dtype=jnp.float64)
    b_fast = jnp.array([0.8, 0.2], dtype=jnp.float64)
    w_schwabe = 2.0 * math.pi / 11.0
    t_starts = jnp.array([[100.2, 100.55], [103.2, 103.55]], dtype=jnp.float64)
    t_ends = jnp.array([[100.45, 100.75], [103.45, 103.75]], dtype=jnp.float64)
    sub_w = jnp.array([[0.7, 0.3], [0.7, 0.3]], dtype=jnp.float64)
    h_mat = harmonic_solar_design(
        a_fast,
        b_fast,
        omegas=jnp.array([w_schwabe]),
        t_starts=t_starts,
        t_ends=t_ends,
        sub_weights=sub_w,
        alpha_unit=1.5,
        t_ref=95.0,
        box_indices=0,
        obs_scale=1.3,
    )
    c_harm = jnp.array([0.9, -0.6], dtype=jnp.float64)
    pred_analytic = h_mat @ c_harm

    gl_x, gl_w = np.polynomial.legendre.leggauss(32)
    tau_nodes, tau_w = np.polynomial.legendre.leggauss(160)
    tau_grid = 0.5 * (tau_nodes + 1.0) * 60.0
    tau_weights = 0.5 * tau_w * 60.0
    impulse_vals = jax.vmap(lambda s: (jsl.expm(a_fast * s) @ (1.5 * b_fast))[0] * 1.3)(
        jnp.asarray(tau_grid)
    )

    pred_num = []
    for i in range(2):
        acc = 0.0
        for m in range(2):
            ts, te = float(t_starts[i, m]), float(t_ends[i, m])
            t_pts = 0.5 * (te - ts) * gl_x + 0.5 * (te + ts)
            w_pts = 0.5 * gl_w
            vals = []
            for t_val in t_pts:
                phase = w_schwabe * ((t_val - tau_grid) - 95.0)
                q_hist = 0.9 * np.cos(phase) - 0.6 * np.sin(phase)
                vals.append(np.sum(tau_weights * np.asarray(impulse_vals) * q_hist))
            acc += float(sub_w[i, m]) * float(np.sum(w_pts * np.array(vals)))
        pred_num.append(acc)
    np.testing.assert_allclose(pred_analytic, jnp.asarray(pred_num), rtol=1e-10, atol=1e-10)


# =============================================================================
# 2. Spectral Densities, 24-Step Smith Doubling, POD GP & Kalman/RTS Cross-Checks
# =============================================================================


def test_spectral_density_wiener_khinchin_and_smith_doubling():
    # Piecewise Gauss-Legendre on [0, 80] + [80, 1200] + [1200, 30000] resolves both
    # oscillatory cos(ω τ) and algebraic ω⁻² tails (Exp) accurately:
    gl_x, gl_w = np.polynomial.legendre.leggauss(320)
    segs = [
        (0.0, 4.0),
        (4.0, 60.0),
        (60.0, 400.0),
        (400.0, 900.0),
        (900.0, 1400.0),
        (1400.0, 1900.0),
        (1900.0, 2400.0),
    ]
    om_parts, wt_parts = [], []
    for lo, hi in segs:
        om_parts.append(0.5 * (hi - lo) * gl_x + 0.5 * (hi + lo))
        wt_parts.append(0.5 * (hi - lo) * gl_w)
    omegas = jnp.asarray(np.concatenate(om_parts))
    dw_weights = jnp.asarray(np.concatenate(wt_parts))

    smooth_kernels = [
        1.5**2 * tgk.Matern32(2.2),
        0.9**2 * tgk.Matern52(1.7),
        1.1**2 * tgk.ExpSquared(1.4),
        tgk.RationalQuadratic(alpha=2.5, scale=1.6),
        tgq.SHO(omega=2.0 * math.pi / 11.0, quality=4.5, sigma=0.8),
        0.7**2 * (tgk.Matern32(18.0) * tgk.Cosine(11.0)),
        0.6**2 * (tgk.ExpSquared(20.0) * tgk.Cosine(11.0)),
        tgk.ExpSquared(15.0) * tgk.ExpSineSquared(scale=11.0, gamma=1.4),
        1.2**2 * tgk.Matern52(2.5)
        + tgq.SHO(omega=2.0 * math.pi / 11.0, quality=3.0, sigma=0.5),
    ]
    lags = jnp.array([0.0, 0.6, 2.3], dtype=jnp.float64)
    for kern in smooth_kernels:
        s_w = spectral_density(kern, omegas)
        assert jnp.all(s_w >= 0.0)
        k_from_spec = (
            jnp.sum(
                (dw_weights * s_w)[:, None] * jnp.cos(omegas[:, None] * lags[None, :]),
                axis=0,
            )
            / math.pi
        )
        k_exact = jax.vmap(lambda tau, k=kern: k.evaluate(jnp.asarray(0.0), tau))(lags)
        np.testing.assert_allclose(k_from_spec, k_exact, rtol=1e-7, atol=1e-7)

    # Non-differentiable Ornstein-Uhlenbeck Exp(1.8) has ω⁻² tail (~ 2*1.69/(π*1.8*2400) ≈ 2.5e-4 at τ=0):
    kern_exp = 1.3**2 * tgk.Exp(1.8)
    s_exp = spectral_density(kern_exp, omegas)
    k_exp_spec = (
        jnp.sum((dw_weights * s_exp)[:, None] * jnp.cos(omegas[:, None] * lags[None, :]), axis=0)
        / math.pi
    )
    k_exp_exact = jax.vmap(lambda tau: kern_exp.evaluate(jnp.asarray(0.0), tau))(lags)
    np.testing.assert_allclose(k_exp_spec, k_exp_exact, rtol=3e-4, atol=3e-4)

    # Verify 24-step Smith doubling on stiff CBM (with slow λ_14 = ln 2 / 5730 yr⁻¹):
    a_mat, b_vec, _ = _make_stiff_three_box_cbm()
    eye_c = jnp.eye(3, dtype=jnp.complex128)
    xi_q = 0.5 * (jnp.asarray(np.polynomial.legendre.leggauss(240)[0]) + 1.0)
    w_q = 0.5 * jnp.asarray(np.polynomial.legendre.leggauss(240)[1])
    al_lo = math.asinh(0.05 / 1e-6)
    om_q1 = 1e-6 * jnp.sinh(al_lo * xi_q)
    dw_q1 = w_q * (1e-6 * al_lo * jnp.cosh(al_lo * xi_q))
    om_q2 = 0.05 + (4.0 - 0.05) * xi_q
    dw_q2 = w_q * (4.0 - 0.05)
    al_hi = math.asinh((4000.0 - 4.0) / 0.2)
    om_q3 = 4.0 + 0.2 * jnp.sinh(al_hi * xi_q)
    dw_q3 = w_q * (0.2 * al_hi * jnp.cosh(al_hi * xi_q))
    om_q = jnp.concatenate([om_q1, om_q2, om_q3])
    dw_q = jnp.concatenate([dw_q1, dw_q2, dw_q3])
    z_q = jax.vmap(
        lambda w: jnp.linalg.solve(
            1j * w * eye_c - a_mat.astype(jnp.complex128),
            (1.3 * b_vec).astype(jnp.complex128),
        )
    )(om_q)

    for sde_kern in (
        1.4**2 * tgk.Matern32(2.5),
        1.1**2 * tgk.Matern52(3.2),
        tgq.SHO(omega=2.0 * math.pi / 11.0, quality=5.0, sigma=0.9),
        0.8**2 * (tgk.Matern32(22.0) * tgk.Cosine(11.0)),
        1.2**2 * tgk.Matern52(3.0) + 0.5**2 * tgk.Exp(0.8),
    ):
        sig_smith, root_smith, sig_joint = stationary_cbm_gp_cov(
            a_mat, b_vec, sde_kern, alpha_unit=1.3, return_full_state=True
        )
        assert sig_joint.shape[0] > 3
        assert jnp.all(jnp.linalg.eigvalsh(sig_joint) >= -1e-14)
        np.testing.assert_allclose(
            root_smith @ root_smith.T, sig_smith, rtol=1e-10, atol=1e-9
        )

        wt_q = (dw_q * spectral_density(sde_kern, om_q)) / math.pi
        sig_quad = jnp.real(jnp.einsum("wn,w,wm->nm", z_q, wt_q, jnp.conj(z_q)))
        np.testing.assert_allclose(sig_smith, sig_quad, rtol=1e-9, atol=1e-9)


def _bruteforce_fine_grid_gp_quantities(
    a_mat: jax.Array,
    b_vec: jax.Array,
    c_obs_vec: jax.Array,
    kernel: tgk.Kernel,
    t_starts: jax.Array,
    t_ends: jax.Array,
    resid: jax.Array,
    sigma: jax.Array,
    t_eval: jax.Array,
    *,
    alpha_unit: float = 1.0,
    lookback_yr: float = 500.0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fast vectorized brute-force dense fine-grid ``D + W_full K_prior W_fullᵀ``
    integrated from ``t_0 - 500 yr``.

    Uses a two-tier piecewise-constant step grid:
    - ``[t_0 - 500, t_0 - 20]`` at ``Δτ = 0.5 yr``
    - ``[t_0 - 20, t_max]`` at ``Δτ = 1/16 yr`` (aligned with seasonal windows!)
    propagated via exact recursive discrete steps (zero per-pair ``expm`` overhead!).
    """
    t0 = float(jnp.min(t_starts))
    t_max = float(jnp.max(t_ends))
    edges_far = np.arange(t0 - lookback_yr, t0 - 18.0, 0.5, dtype=np.float64)
    edges_near = np.arange(t0 - 18.0, t_max + 1e-12, 1.0 / 32.0, dtype=np.float64)
    edges = np.concatenate([edges_far, edges_near])
    tau_a = jnp.asarray(edges[:-1])
    tau_b = jnp.asarray(edges[1:])
    dtau = tau_b - tau_a
    n_steps = dtau.shape[0]
    n_obs = t_starts.shape[0]
    n_near = edges_near.shape[0] - 1
    n_far = edges_far.shape[0]
    b_eff = alpha_unit * b_vec

    # Propagate unit impulse on each step m = 0..n_steps-1 to every fine sub-step
    # k = 0..n_near-1 on [t0 - 18, t_max] using only 2 matrix exponentials total!
    dt_far = 0.5
    dt_near = 1.0 / 32.0
    phi_far = jsl.expm(a_mat * dt_far)
    step_far = dt_far * (_psi1(a_mat * dt_far) @ b_eff)

    # Exact average of δy(t) over fine step [t_k, t_{k+1}]:
    # 1) Contribution of state δy(t_k) at start of fine step: ψ_1(A * dt_near) @ δy(t_k)
    # 2) Contribution of constant forcing q_k = 1 inside [t_k, t_{k+1}]:
    #    (1/dt_near) ∫_0^{dt_near} s ψ_1(A s) b_eff ds via 3n x 3n Van Loan expm!
    n_box = a_mat.shape[0]
    z_n = jnp.zeros((n_box, n_box), dtype=jnp.float64)
    eye_n = jnp.eye(n_box, dtype=jnp.float64)
    e_3n = jsl.expm(
        jnp.block([[a_mat, eye_n, z_n], [z_n, z_n, eye_n], [z_n, z_n, z_n]]) * dt_near
    )
    phi_near = e_3n[:n_box, :n_box]
    psi1_near = e_3n[:n_box, n_box : 2 * n_box] / dt_near
    step_near = e_3n[:n_box, n_box : 2 * n_box] @ b_eff  # ∫_0^{dt_near} exp(A s) b ds
    self_avg_near = (e_3n[:n_box, 2 * n_box :] / dt_near) @ b_eff  # avg over step of self forcing

    # Powers of phi_far (0..n_far-1) and phi_near (0..n_near-1) via fast lax.scan:
    def _pow_scan(mat: jax.Array, length: int) -> jax.Array:
        def _step(cur, _):
            return cur @ mat, cur

        _, out = jax.lax.scan(_step, eye_n, None, length=length)
        return out  # out[j] = mat^j

    pows_far = _pow_scan(phi_far, n_far)  # (n_far, n, n)
    pows_near = _pow_scan(phi_near, n_near)  # (n_near, n, n)

    # Far step m in [0, n_far-1] arrives at t_split = t0 - 20 with state phi_far^{n_far - 1 - m} @ step_far:
    far_at_split = jnp.einsum(
        "mij,j->mi", pows_far[::-1], step_far
    )  # (n_far, n)
    # Scalar obs c_obs @ avg_{t in near_k} δy(t) for far step m:
    c_psi_pows = jnp.einsum("i,ij,kjl->kl", c_obs_vec, psi1_near, pows_near)  # (n_near, n)
    w_near_from_far = c_psi_pows @ far_at_split.T  # (n_near, n_far)

    # Near step m in [0, n_near-1] observed on near step k in [0, n_near-1]:
    idx_k = jnp.arange(n_near)[:, None]
    idx_m = jnp.arange(n_near)[None, :]
    diff_km = idx_k - idx_m
    prior_vals = (c_psi_pows @ step_near)[jnp.clip(diff_km - 1, 0, n_near - 1)]
    self_val = c_obs_vec @ self_avg_near
    w_near_from_near = jnp.where(
        diff_km > 0, prior_vals, jnp.where(diff_km == 0, self_val, 0.0)
    )  # (n_near, n_near)

    w_all_near = jnp.concatenate([w_near_from_far, w_near_from_near], axis=1)  # (n_near, n_steps)

    # Average fine 1/16-yr sub-steps inside each observation window [t_starts[i], t_ends[i]]:
    near_a = jnp.asarray(edges_near[:-1])
    near_b = jnp.asarray(edges_near[1:])
    in_win = (near_a[None, :] >= t_starts[:, None] - 1e-11) & (
        near_b[None, :] <= t_ends[:, None] + 1e-11
    )  # (n_obs, n_near)
    win_weights = in_win.astype(jnp.float64) / jnp.sum(
        in_win.astype(jnp.float64), axis=1, keepdims=True
    )
    w_full = win_weights @ w_all_near  # (n_obs, n_steps)

    # 2-point Gauss-Legendre cell-average GP covariance across all steps (vectorized!):
    tau_mid = 0.5 * (tau_a + tau_b)
    gl2 = jnp.array([-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)])
    flat_pts = (tau_mid[:, None] + 0.5 * dtau[:, None] * gl2[None, :]).reshape(-1)
    k_flat = kernel(flat_pts, flat_pts)
    k_prior = jnp.mean(k_flat.reshape(n_steps, 2, n_steps, 2), axis=(1, 3))

    cov_data = jnp.diag(sigma**2) + w_full @ k_prior @ w_full.T
    cov_data = 0.5 * (cov_data + cov_data.T)
    ll_bf = jss.multivariate_normal.logpdf(resid, jnp.zeros(n_obs), cov_data)

    k_eval_steps = jnp.mean(
        kernel(t_eval, flat_pts).reshape(t_eval.shape[0], n_steps, 2), axis=2
    )
    k_eval_obs = k_eval_steps @ w_full.T
    alpha_vec = jnp.linalg.solve(cov_data, resid)
    q_mean_bf = k_eval_obs @ alpha_vec
    k_eval_diag = jnp.diag(kernel(t_eval, t_eval))
    q_var_bf = k_eval_diag - jnp.sum(
        k_eval_obs * jnp.linalg.solve(cov_data, k_eval_obs.T).T, axis=1
    )

    sig_data_cov = w_full @ k_prior @ w_full.T
    data_mean_bf = sig_data_cov @ alpha_vec
    data_var_bf = jnp.diag(sig_data_cov) - jnp.sum(
        sig_data_cov * jnp.linalg.solve(cov_data, sig_data_cov).T, axis=1
    )
    return ll_bf, q_mean_bf, q_var_bf, data_mean_bf, data_var_bf


def test_pod_gp_vs_kalman_and_bruteforce_fine_grid():
    """Cross-check POD LinearMarginal against BOTH brute-force fine-grid (t_0 - 500 yr)
    and exact continuous-time kalman_gp_marginal (point + seasonal + nuisance).
    """
    rng = np.random.default_rng(123)
    a_mod = jnp.array(
        [
            [-1.5, 0.7, 0.15],
            [0.9, -0.8, 0.0],
            [0.5, 0.0, -0.25],
        ],
        dtype=jnp.float64,
    )
    b_mod = jnp.array([0.8, 0.2, 0.0], dtype=jnp.float64)
    c_mod = jnp.array([1.25, 0.1, 0.0], dtype=jnp.float64)

    n_years = 14
    years = jnp.arange(n_years, dtype=jnp.float64)
    t_starts = years + 0.25
    t_ends = years + 0.625
    sigma = jnp.full(n_years, 0.85, dtype=jnp.float64)
    resid = jnp.asarray(rng.standard_normal(n_years) * 1.4, dtype=jnp.float64)
    t_eval = years + 0.4375
    obs_mat = jnp.broadcast_to(c_mod[None, :], (n_years, 3))

    # (A) Markov SDE kernel (Matern52): POD vs Brute-Force (t0 - 500 yr) AND Kalman/RTS!
    kern_m52 = (1.35**2) * tgk.Matern52(2.8)
    proj_m52 = project_causal_gp(
        a_mod,
        b_mod,
        t_starts,
        t_ends,
        pad=24.0,
        n_modes=180,
        kernel=kern_m52,
        alpha_unit=1.2,
        obs_matrix=obs_mat,
    )
    lm_m52 = proj_m52.to_linear_marginal()
    ll_pod = lm_m52.loglike(resid, sigma)
    q_mean_pod, q_var_pod = proj_m52.posterior_production(t_eval, resid, sigma)
    data_mean_pod, data_cov_pod = lm_m52.predict_posterior(resid, sigma)

    kal_m52 = kalman_gp_marginal(
        a_mod,
        b_mod,
        kern_m52,
        resid,
        sigma,
        t_starts,
        t_ends,
        dt=1.0 / 16.0,
        alpha_unit=1.2,
        obs_matrix=obs_mat,
    )
    eval_idx = jnp.round((t_eval - kal_m52.t_grid[0]) * 16.0).astype(jnp.int32)

    np.testing.assert_allclose(ll_pod, kal_m52.loglike, rtol=1e-8, atol=5e-8)
    np.testing.assert_allclose(
        q_mean_pod, kal_m52.production_mean[eval_idx], rtol=1e-5, atol=3e-6
    )
    np.testing.assert_allclose(
        q_var_pod, kal_m52.production_var[eval_idx], rtol=3e-5, atol=6e-6
    )

    ll_bf, q_mean_bf, q_var_bf, d_mean_bf, d_var_bf = _bruteforce_fine_grid_gp_quantities(
        a_mod,
        b_mod,
        c_mod,
        kern_m52,
        t_starts,
        t_ends,
        resid,
        sigma,
        t_eval,
        alpha_unit=1.2,
        lookback_yr=500.0,
    )
    np.testing.assert_allclose(ll_pod, ll_bf, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(q_mean_pod, q_mean_bf, rtol=3e-4, atol=3e-4)
    np.testing.assert_allclose(q_var_pod, q_var_bf, rtol=3e-4, atol=3e-4)
    np.testing.assert_allclose(data_mean_pod, d_mean_bf, rtol=3e-4, atol=3e-4)
    np.testing.assert_allclose(jnp.diag(data_cov_pod), d_var_bf, rtol=3e-4, atol=3e-4)

    # (B) Non-quasiseparable ExpSquared kernel: POD (spectral Σ_{y,0}) vs Brute-Force (t0 - 500 yr)!
    kern_se = (1.25**2) * tgk.ExpSquared(2.4)
    proj_se = project_causal_gp(
        a_mod,
        b_mod,
        t_starts,
        t_ends,
        pad=20.0,
        n_modes=110,
        kernel=kern_se,
        alpha_unit=1.2,
        obs_matrix=obs_mat,
    )
    lm_se = proj_se.to_linear_marginal()
    ll_se_bf, qm_se_bf, qv_se_bf, dm_se_bf, dv_se_bf = _bruteforce_fine_grid_gp_quantities(
        a_mod,
        b_mod,
        c_mod,
        kern_se,
        t_starts,
        t_ends,
        resid,
        sigma,
        t_eval,
        alpha_unit=1.2,
        lookback_yr=500.0,
    )
    qm_se_pod, qv_se_pod = proj_se.posterior_production(t_eval, resid, sigma)
    dm_se_pod, dcov_se_pod = lm_se.predict_posterior(resid, sigma)
    np.testing.assert_allclose(
        lm_se.loglike(resid, sigma), ll_se_bf, rtol=2e-4, atol=2e-4
    )
    np.testing.assert_allclose(qm_se_pod, qm_se_bf, rtol=3e-4, atol=3e-4)
    np.testing.assert_allclose(qv_se_pod, qv_se_bf, rtol=3e-4, atol=3e-4)
    np.testing.assert_allclose(dm_se_pod, dm_se_bf, rtol=3e-4, atol=3e-4)
    np.testing.assert_allclose(jnp.diag(dcov_se_pod), dv_se_bf, rtol=3e-4, atol=3e-4)

    # (C) Stiff radiocarbon CBM (slow λ_14 = ln 2 / 5730 yr⁻¹) + hierarchical offsets + solar harmonics
    # + overlapping NH/SH seasonal windows: POD vs exact kalman_gp_marginal!
    a_stiff, b_stiff, c_stiff = _make_stiff_three_box_cbm()
    t_nh_s = jnp.arange(10, dtype=jnp.float64) + 0.25
    t_nh_e = jnp.arange(10, dtype=jnp.float64) + 0.625
    t_sh_s = jnp.arange(10, dtype=jnp.float64) + 0.50
    t_sh_e = jnp.arange(10, dtype=jnp.float64) + 1.00
    t_comb_s = jnp.concatenate([t_nh_s, t_sh_s])
    t_comb_e = jnp.concatenate([t_nh_e, t_sh_e])
    series_comb = jnp.concatenate(
        [jnp.zeros(10, dtype=jnp.int32), jnp.ones(10, dtype=jnp.int32)]
    )
    obs_comb = jnp.broadcast_to(c_stiff[None, :], (20, 3))
    resid_comb = jnp.asarray(rng.standard_normal(20) * 2.2, dtype=jnp.float64)
    sig_comb = jnp.linspace(0.9, 1.5, 20)

    x_off, mu_off, sd_off = hierarchical_offset_design(
        series_comb,
        n_series=2,
        global_prior_mean=1.5,
        global_prior_sd=200.0,
        lab_prior_sd=1.0,
    )
    x_sol = harmonic_solar_design(
        a_stiff,
        b_stiff,
        omegas=jnp.array([2.0 * math.pi / 11.0]),
        t_starts=t_comb_s,
        t_ends=t_comb_e,
        obs_matrix=obs_comb,
    )
    nuis_x = jnp.concatenate([x_off, x_sol], axis=1)
    nuis_mu = jnp.concatenate([mu_off, jnp.zeros(2)])
    nuis_sd = jnp.concatenate([sd_off, jnp.array([1.5, 1.5])])

    kern_stiff = (1.6**2) * tgk.Matern52(3.2)
    proj_stiff = project_causal_gp(
        a_stiff,
        b_stiff,
        t_comb_s,
        t_comb_e,
        pad=22.0,
        n_modes=150,
        kernel=kern_stiff,
        obs_matrix=obs_comb,
    )
    lm_stiff = proj_stiff.to_linear_marginal(
        nuisance_design=nuis_x,
        nuisance_prior_mean=nuis_mu,
        nuisance_prior_sd=nuis_sd,
        method="qr",
    )
    kal_stiff = kalman_gp_marginal(
        a_stiff,
        b_stiff,
        kern_stiff,
        resid_comb,
        sig_comb,
        t_comb_s,
        t_comb_e,
        dt=1.0 / 16.0,
        obs_matrix=obs_comb,
        nuisance_design=nuis_x,
        nuisance_prior_mean=nuis_mu,
        nuisance_prior_sd=nuis_sd,
    )
    np.testing.assert_allclose(
        lm_stiff.loglike(resid_comb, sig_comb),
        kal_stiff.loglike,
        rtol=1e-8,
        atol=5e-8,
    )
    w_mean_pod, w_cov_pod = lm_stiff.posterior(resid_comb, sig_comb)
    n_gp_cols = proj_stiff.whitened_design.shape[1]
    np.testing.assert_allclose(
        w_mean_pod[n_gp_cols:], kal_stiff.nuisance_mean, rtol=1e-6, atol=5e-7
    )
    np.testing.assert_allclose(
        w_cov_pod[n_gp_cols:, n_gp_cols:], kal_stiff.nuisance_cov, rtol=1e-6, atol=5e-7
    )


def test_pod_exponential_rank_convergence_and_zero_double_counting():
    """Prove exponential convergence with rank ``r`` (``< 1e-8`` at ``r = 80..120``)
    and confirm ZERO ``2×`` initial-state variance double-counting at ``t = a``.
    """
    rng = np.random.default_rng(999)
    a_mat, b_vec, c_obs = _make_stiff_three_box_cbm()
    t_starts = jnp.arange(10, dtype=jnp.float64) + 0.25
    t_ends = t_starts + 0.5
    n_obs = t_starts.shape[0]
    obs_mat = jnp.broadcast_to(c_obs[None, :], (n_obs, 3))
    sigma = jnp.full(n_obs, 0.9, dtype=jnp.float64)
    resid = jnp.asarray(rng.standard_normal(n_obs) * 1.8, dtype=jnp.float64)

    # 1) Smooth ExpSquared GP: true exponential spectral convergence (< 1e-11 by r=80)!
    kern_se = (1.5**2) * tgk.ExpSquared(2.8)
    ll_se_exact = (
        project_causal_gp(
            a_mat,
            b_vec,
            t_starts,
            t_ends,
            pad=18.0,
            n_modes=200,
            kernel=kern_se,
            obs_matrix=obs_mat,
        )
        .to_linear_marginal()
        .loglike(resid, sigma)
    )
    errs_se = []
    for r in (8, 16, 32, 64, 96):
        ll_r = (
            project_causal_gp(
                a_mat,
                b_vec,
                t_starts,
                t_ends,
                pad=18.0,
                n_modes=r,
                kernel=kern_se,
                obs_matrix=obs_mat,
            )
            .to_linear_marginal()
            .loglike(resid, sigma)
        )
        errs_se.append(abs(float(ll_r - ll_se_exact)))
    assert errs_se[0] > errs_se[1] > errs_se[2] > errs_se[3]
    assert errs_se[3] < 1e-8, f"Expected < 1e-8 at r=64 for ExpSquared; got {errs_se[3]}"
    assert errs_se[4] < 1e-11, f"Expected < 1e-11 at r=96 for ExpSquared; got {errs_se[4]}"

    # 2) Seasonal Matern52 GP vs exact continuous-time kalman_gp_marginal (< 1e-8 at r=120):
    t_s_short = jnp.array([0.25, 1.25], dtype=jnp.float64)
    t_e_short = t_s_short + 0.75
    obs_short = jnp.broadcast_to(c_obs[None, :], (2, 3))
    sig_short = jnp.full(2, 1.6, dtype=jnp.float64)
    res_short = resid[:2]
    kern_m52 = (1.5**2) * tgk.Matern52(3.5)
    kal_ref = kalman_gp_marginal(
        a_mat,
        b_vec,
        kern_m52,
        res_short,
        sig_short,
        t_s_short,
        t_e_short,
        dt=0.25,
        obs_matrix=obs_short,
    )
    errs_m52 = []
    for r in (8, 24, 56, 110):
        ll_r = (
            project_causal_gp(
                a_mat,
                b_vec,
                t_s_short,
                t_e_short,
                pad=40.0,
                b_bound=22.0,
                n_modes=r,
                kernel=kern_m52,
                obs_matrix=obs_short,
            )
            .to_linear_marginal()
            .loglike(res_short, sig_short)
        )
        errs_m52.append(abs(float(ll_r - kal_ref.loglike)))
    assert errs_m52[0] > errs_m52[1] > errs_m52[2] > errs_m52[3]
    assert errs_m52[2] < 5e-6, f"Expected < 5e-6 at r=56 vs Kalman; got {errs_m52[2]}"
    assert errs_m52[3] < 1e-8, f"Expected < 1e-8 at r=110 vs Kalman; got {errs_m52[3]}"

    # 3) Explicitly confirm ZERO 2x initial-state variance double-counting at t = a:
    t_pts = jnp.arange(0.0, 10.0, 0.5, dtype=jnp.float64)
    obs_pts = jnp.broadcast_to(c_obs[None, :], (t_pts.shape[0], 3))
    proj_at_a = project_causal_gp(
        a_mat,
        b_vec,
        t_pts,
        pad=0.0,
        b_bound=24.0,
        n_modes=100,
        kernel=kern_m52,
        obs_matrix=obs_pts,
    )
    np.testing.assert_allclose(
        proj_at_a.causal_basis_design[0],
        jnp.zeros(100),
        rtol=0.0,
        atol=1e-14,
    )
    prior_obs_var_row0 = jnp.sum(proj_at_a.whitened_design[0] ** 2)
    exact_var_at_a = c_obs @ proj_at_a.sigma_y0 @ c_obs
    np.testing.assert_allclose(
        prior_obs_var_row0, exact_var_at_a, rtol=1e-10, atol=1e-9
    )

    # Without -exp(A(t-a)) v_j subtraction, slow reservoir (γ << 1/ℓ) double-counts ~2x at t=a:
    a_slow = jnp.array([[-0.15]], dtype=jnp.float64)
    b_slow = jnp.array([1.0], dtype=jnp.float64)
    kern_fast = tgk.Matern52(0.7)
    proj_slow = project_causal_gp(
        a_slow, b_slow, t_pts, pad=0.0, b_bound=40.0, n_modes=160, kernel=kern_fast
    )
    exact_slow_a = proj_slow.sigma_y0[0, 0]
    np.testing.assert_allclose(
        jnp.sum(proj_slow.whitened_design[0] ** 2), exact_slow_a, rtol=1e-12, atol=1e-12
    )
    v_naive_slow = jnp.imag(1.0 / (1j * proj_slow.omegas + 0.15))
    naive_row0_slow = (v_naive_slow / jnp.sqrt(proj_slow.half_length)) * jnp.sqrt(
        proj_slow.spectral_weights
    )
    naive_total_slow = jnp.sum(naive_row0_slow**2) + exact_slow_a
    assert float(naive_total_slow / exact_slow_a) > 1.85


def test_canonical_pod_modes_wiener_equivalence():
    rng = np.random.default_rng(2026)
    a_mat, b_vec, c_obs = _make_stiff_three_box_cbm()
    t_starts = jnp.arange(16, dtype=jnp.float64) + 0.25
    t_ends = t_starts + 0.5
    obs_mat = jnp.broadcast_to(c_obs[None, :], (16, 3))
    sigma = jnp.linspace(0.8, 1.3, 16)
    resid = jnp.asarray(rng.standard_normal(16) * 1.5, dtype=jnp.float64)
    t_eval = jnp.linspace(0.25, 15.75, 40)

    basis = solin_sarkka_basis(t_eval, -20.0, 36.0, 48, kernel=tgk.Matern52(2.5))
    assert basis.psi.shape == (40, 48)
    assert basis.spectral_weights is not None and basis.spectral_weights.shape == (48,)

    proj = project_causal_gp(
        a_mat,
        b_vec,
        t_starts,
        t_ends,
        pad=20.0,
        n_modes=64,
        kernel=(1.4**2) * tgk.Matern52(2.5),
        obs_matrix=obs_mat,
    )
    nuis_col = jnp.ones((16, 1), dtype=jnp.float64) * 200.0
    q_mean_ref, q_var_ref = proj.posterior_production(
        t_eval,
        resid,
        sigma,
        nuisance_design=jnp.ones((16, 1)),
        nuisance_prior_sd=200.0,
    )
    pod = canonical_pod_modes(
        proj, t_eval, resid, sigma, nuisance_whitened_design=nuis_col
    )
    np.testing.assert_allclose(pod.posterior_mean, q_mean_ref, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        pod.posterior_sd, jnp.sqrt(q_var_ref), rtol=1e-11, atol=1e-11
    )
    np.testing.assert_allclose(
        pod.prior_rotation.T @ pod.prior_rotation, jnp.eye(64), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        pod.data_modes.T @ pod.data_modes,
        jnp.eye(pod.data_modes.shape[1]),
        rtol=1e-12,
        atol=1e-12,
    )


# =============================================================================
# 3. Second-Order Autodiff Checks (jax.test_util.check_grads)
# =============================================================================


def test_pod_marginal_loglike_check_grads():
    """Run ``jax.test_util.check_grads`` wrt BOTH ``A`` and ``(ln σ_Q, ln ℓ)``."""
    rng = np.random.default_rng(314)
    t_starts = jnp.arange(6, dtype=jnp.float64) + 0.25
    t_ends = t_starts + 0.5
    resid = jnp.asarray(rng.standard_normal(6) * 1.3, dtype=jnp.float64)
    sigma = jnp.linspace(0.85, 1.25, 6)

    # Compact 2-box Hurwitz CBM matrix A (4 matrix entries) + 2 GP log-hyperparameters
    # so 2nd-order check_grads tests every entry of raw matrix A and (ln σ_Q, ln ℓ) fast!
    a2_0 = jnp.array(
        [[-1.3 - LAMBDA_14, 0.45], [1.3, -0.45 - 0.01 - LAMBDA_14]],
        dtype=jnp.float64,
    )
    b2 = jnp.array([0.85, 0.15], dtype=jnp.float64)
    obs2 = jnp.broadcast_to(jnp.array([[1.2, 0.0]], dtype=jnp.float64), (6, 2))
    log_hyp0 = jnp.array([math.log(1.3), math.log(2.4)], dtype=jnp.float64)

    # 1) SDE + 24-step Smith doubling path (Matern52), order=2 wrt A and (ln σ_Q, ln ℓ):
    @jax.jit
    def pod_ll_sde(a_matrix: jax.Array, log_hyp: jax.Array) -> jax.Array:
        kern = jnp.exp(2.0 * log_hyp[0]) * tgk.Matern52(jnp.exp(log_hyp[1]))
        proj = project_causal_gp(
            a_matrix,
            b2,
            t_starts,
            t_ends,
            pad=14.0,
            n_modes=28,
            kernel=kern,
            obs_matrix=obs2,
        )
        return proj.to_linear_marginal().loglike(resid, sigma)

    check_grads(pod_ll_sde, (a2_0, log_hyp0), order=2, eps=1e-6, atol=1e-4, rtol=1e-4)

    # 2) Non-quasiseparable spectral-quadrature path (ExpSquared), order=2 wrt A and (ln σ_Q, ln ℓ):
    @jax.jit
    def pod_ll_se(a_matrix: jax.Array, log_hyp: jax.Array) -> jax.Array:
        kern = jnp.exp(2.0 * log_hyp[0]) * tgk.ExpSquared(jnp.exp(log_hyp[1]))
        proj = project_causal_gp(
            a_matrix,
            b2,
            t_starts,
            t_ends,
            pad=14.0,
            n_modes=28,
            kernel=kern,
            obs_matrix=obs2,
        )
        return proj.to_linear_marginal().loglike(resid, sigma)

    check_grads(pod_ll_se, (a2_0, log_hyp0), order=2, eps=1e-6, atol=1e-4, rtol=1e-4)

    # 3) Exact gradient cross-check: high-rank POD vs kalman_gp_marginal on 3-box stiff CBM!
    theta0 = jnp.log(jnp.array([1.4, 0.8, 0.9, 0.025, 0.002], dtype=jnp.float64))
    _, _, c3 = _make_stiff_three_box_cbm(theta0)
    obs3 = jnp.broadcast_to(c3[None, :], (6, 3))

    def pod_high_rank(theta: jax.Array, log_hyp: jax.Array) -> jax.Array:
        am, bv, _ = _make_stiff_three_box_cbm(theta)
        kern = jnp.exp(2.0 * log_hyp[0]) * tgk.Matern52(jnp.exp(log_hyp[1]))
        return (
            project_causal_gp(
                am,
                bv,
                t_starts,
                t_ends,
                pad=22.0,
                n_modes=140,
                kernel=kern,
                obs_matrix=obs3,
            )
            .to_linear_marginal()
            .loglike(resid, sigma)
        )

    def kal_exact(theta: jax.Array, log_hyp: jax.Array) -> jax.Array:
        am, bv, _ = _make_stiff_three_box_cbm(theta)
        kern = jnp.exp(2.0 * log_hyp[0]) * tgk.Matern52(jnp.exp(log_hyp[1]))
        return kalman_gp_marginal(
            am,
            bv,
            kern,
            resid,
            sigma,
            t_starts,
            t_ends,
            dt=1.0 / 16.0,
            obs_matrix=obs3,
        ).loglike

    g_pod = jax.grad(pod_high_rank, argnums=(0, 1))(theta0, log_hyp0)
    g_kal = jax.grad(kal_exact, argnums=(0, 1))(theta0, log_hyp0)
    np.testing.assert_allclose(g_pod[0], g_kal[0], rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(g_pod[1], g_kal[1], rtol=1e-7, atol=1e-7)


# =============================================================================
# 4. Carbon-Cycle Fisher Information & Continuous Relaxation Modes
# =============================================================================


class _ToyBalancedCBM(zdx.Base):
    """Toy balanced 3-box CBM PyTree with an exact ``log_total_mass`` gauge leaf."""

    log_total_mass: jax.Array
    frac_logits: jax.Array
    log_flux_01: jax.Array
    log_flux_02: jax.Array

    def masses(self) -> jax.Array:
        m_tot = jnp.exp(self.log_total_mass)
        logits = jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), self.frac_logits])
        return m_tot * jax.nn.softmax(logits)

    def fluxes(self) -> jax.Array:
        m_tot = jnp.exp(self.log_total_mass)
        f01 = m_tot * jnp.exp(self.log_flux_01)
        f02 = m_tot * jnp.exp(self.log_flux_02)
        return jnp.stack([f01, f01, f02, f02])

    def matrix_and_b(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        m = self.masses()
        f01, _, f02, _ = self.fluxes()
        k01 = f01 / m[0]
        k10 = f01 / m[1]
        k02 = f02 / m[0]
        k20 = f02 / m[2]
        a_mat = jnp.array(
            [
                [-(k01 + k02 + LAMBDA_14), k10, k20],
                [k01, -(k10 + LAMBDA_14), 0.0],
                [k02, 0.0, -(k20 + LAMBDA_14)],
            ],
            dtype=jnp.float64,
        )
        b_vec = jnp.array([0.8, 0.2, 0.0], dtype=jnp.float64)
        y_ss = -jnp.linalg.solve(a_mat, b_vec)
        c_obs = jnp.array([1000.0 / y_ss[0], 0.0, 0.0], dtype=jnp.float64)
        return a_mat, b_vec, c_obs


def _miyake_forward(
    cbm: _ToyBalancedCBM, psi: jax.Array, t_obs: jax.Array
) -> jax.Array:
    a_mat, b_vec, c_obs = cbm.matrix_and_b()
    amp = psi[0]
    offset = psi[1]
    w0 = 2.0 * math.pi / 11.0

    def _at_t(t: jax.Array) -> jax.Array:
        return amp * (c_obs @ (jsl.expm(a_mat * t) @ b_vec))

    spike = jax.vmap(_at_t)(t_obs)
    harm = psi[2] * jnp.cos(w0 * t_obs) + psi[3] * jnp.sin(w0 * t_obs)
    return offset + spike + harm


def test_expected_fisher_and_regularized_laplace():
    cbm = _ToyBalancedCBM(
        log_total_mass=jnp.asarray(math.log(40000.0)),
        frac_logits=jnp.array([1.2, 3.5]),
        log_flux_01=jnp.asarray(math.log(0.03)),
        log_flux_02=jnp.asarray(math.log(0.02)),
    )
    psi = jnp.array([12.0, -1.0, 0.8, -0.4], dtype=jnp.float64)
    t_obs = jnp.linspace(0.25, 15.0, 30)
    sigma = jnp.linspace(0.9, 1.4, 30)

    f_pytree, unravel = expected_fisher(
        lambda m: _miyake_forward(m, psi, t_obs), cbm, sigma, return_unravel=True
    )
    assert f_pytree.shape == (5, 5)
    assert isinstance(unravel(jnp.zeros(5)), _ToyBalancedCBM)

    np.testing.assert_allclose(f_pytree[0, :], 0.0, atol=1e-12)
    np.testing.assert_allclose(f_pytree[:, 0], 0.0, atol=1e-12)

    cov_2d = jnp.diag(sigma**2) + 0.15 * jnp.ones((30, 30))
    prec_2d = jnp.linalg.inv(cov_2d)
    f_prec = expected_fisher(lambda m: _miyake_forward(m, psi, t_obs), cbm, prec_2d)
    f_cov = expected_fisher(
        lambda m: _miyake_forward(m, psi, t_obs), cbm, cov_2d, kind="cov"
    )
    np.testing.assert_allclose(f_prec, f_cov, rtol=1e-12, atol=1e-12)

    # Numerically singular f_pytree triggers RuntimeWarning in regularized_inverse and stays finite:
    with pytest.warns(RuntimeWarning):
        inv_reg = regularized_inverse(f_pytree, ridge=1e-8)
    assert jnp.all(jnp.isfinite(inv_reg))

    proj_p = fisher_projection(f_pytree[1:, 1:])
    np.testing.assert_allclose(
        proj_p @ proj_p.T, jnp.linalg.inv(f_pytree[1:, 1:]), rtol=1e-11, atol=1e-11
    )

    y_synth = _miyake_forward(cbm, psi, t_obs)
    th_act0 = jnp.array([1.2, 3.5, math.log(0.03), math.log(0.02)], dtype=jnp.float64)

    def neg_log_post(th_act: jax.Array) -> jax.Array:
        m = _ToyBalancedCBM(
            log_total_mass=jnp.asarray(math.log(40000.0)),
            frac_logits=th_act[:2],
            log_flux_01=th_act[2],
            log_flux_02=th_act[3],
        )
        r = (y_synth - _miyake_forward(m, psi, t_obs)) / sigma
        return 0.5 * (r @ r) + 0.5 * jnp.sum((th_act - th_act0) ** 2)

    lap_cov = laplace_covariance(neg_log_post, th_act0)
    np.testing.assert_allclose(
        lap_cov, jnp.linalg.inv(f_pytree[1:, 1:] + jnp.eye(4)), rtol=1e-11, atol=1e-11
    )


def test_cbm_fisher_spectrum_reparam_invariance_schur_and_gauge_safety():
    t_obs = jnp.linspace(0.2, 18.0, 45)
    sigma = jnp.linspace(0.8, 1.3, 45)
    psi0 = jnp.array([14.0, -2.0, 0.9, -0.5], dtype=jnp.float64)
    sd_psi = jnp.array([50.0, 200.0, 2.0, 2.0], dtype=jnp.float64)

    theta0 = jnp.array([1.1, 3.3, math.log(0.035), math.log(0.018)], dtype=jnp.float64)
    m_tot_fixed = jnp.asarray(math.log(38000.0))

    def _to_cbm(th: jax.Array) -> _ToyBalancedCBM:
        return _ToyBalancedCBM(
            log_total_mass=m_tot_fixed,
            frac_logits=th[:2],
            log_flux_01=th[2],
            log_flux_02=th[3],
        )

    rng = np.random.default_rng(77)
    raw_l = rng.standard_normal((4, 4)) * 0.4 + np.eye(4) * 0.8
    prior_cov_theta = jnp.asarray(raw_l @ raw_l.T, dtype=jnp.float64)

    res_orig = cbm_fisher_spectrum(
        lambda th, ps: _miyake_forward(_to_cbm(th), ps, t_obs),
        theta0,
        sigma,
        psi_nuisance=psi0,
        prior_cbm_cov=prior_cov_theta,
        prior_psi_sd=sd_psi,
        mass_fn=lambda th: _to_cbm(th).masses(),
        flux_fn=lambda th: _to_cbm(th).fluxes(),
    )
    assert isinstance(res_orig, FisherSpectrumResult)

    # 1) Exact Schur-complement accuracy vs joint (8 x 8) inverse covariance:
    pi_theta = jnp.linalg.inv(prior_cov_theta)
    pi_psi = jnp.diag(1.0 / (sd_psi**2))
    pi_joint = jsl.block_diag(pi_theta, pi_psi)
    joint_post_cov = jnp.linalg.inv(res_orig.joint_fisher + pi_joint)
    np.testing.assert_allclose(
        res_orig.posterior_cov, joint_post_cov[:4, :4], rtol=1e-11, atol=1e-11
    )
    f_tt = res_orig.joint_fisher[:4, :4]
    f_tp = res_orig.joint_fisher[:4, 4:]
    f_pp = res_orig.joint_fisher[4:, 4:]
    schur_dense = f_tt - f_tp @ jnp.linalg.solve(f_pp + pi_psi, f_tp.T)
    np.testing.assert_allclose(
        res_orig.marginal_fisher, schur_dense, rtol=1e-11, atol=1e-11
    )

    # 2) Coordinate-reparameterization invariance under θ' = M θ + c:
    m_xform = jnp.array(
        [
            [1.3, -0.4, 0.2, 0.1],
            [0.2, 0.9, -0.3, 0.5],
            [-0.1, 0.3, 1.5, -0.2],
            [0.4, -0.2, 0.1, 0.8],
        ],
        dtype=jnp.float64,
    )
    c_shift = jnp.array([-0.7, 1.2, 0.4, -0.9], dtype=jnp.float64)
    m_inv = jnp.linalg.inv(m_xform)
    phi0 = m_xform @ theta0 + c_shift
    prior_cov_phi = m_xform @ prior_cov_theta @ m_xform.T

    def _phi_to_cbm(ph: jax.Array) -> _ToyBalancedCBM:
        return _to_cbm(m_inv @ (ph - c_shift))

    res_reparam = cbm_fisher_spectrum(
        lambda ph, ps: _miyake_forward(_phi_to_cbm(ph), ps, t_obs),
        phi0,
        sigma,
        psi_nuisance=psi0,
        prior_cbm_cov=prior_cov_phi,
        prior_psi_sd=sd_psi,
        mass_fn=lambda ph: _phi_to_cbm(ph).masses(),
        flux_fn=lambda ph: _phi_to_cbm(ph).fluxes(),
    )

    np.testing.assert_allclose(
        res_reparam.eigenvalues, res_orig.eigenvalues, rtol=1e-11, atol=1e-11
    )
    np.testing.assert_allclose(res_reparam.d_eff, res_orig.d_eff, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        res_reparam.info_nats, res_orig.info_nats, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        res_reparam.info_bits, res_orig.info_bits, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        jnp.abs(res_reparam.mass_loadings),
        jnp.abs(res_orig.mass_loadings),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        jnp.abs(res_reparam.flux_loadings),
        jnp.abs(res_orig.flux_loadings),
        rtol=1e-10,
        atol=1e-10,
    )

    # 3) Zero inf * 0 = NaN gauge safety on full 5-parameter _ToyBalancedCBM (with log_total_mass):
    cbm_full = _to_cbm(theta0)
    cov_5 = jsl.block_diag(jnp.array([[jnp.inf]]), prior_cov_theta)
    prec_5 = jsl.block_diag(jnp.zeros((1, 1)), pi_theta)
    sd_5 = jnp.array([jnp.inf, 1.0, 1.0, 1.0, 1.0], dtype=jnp.float64)

    res_auto = cbm_fisher_spectrum(
        lambda m, ps: _miyake_forward(m, ps, t_obs),
        cbm_full,
        sigma,
        psi_nuisance=psi0,
        prior_psi_sd=sd_psi,
    )
    res_inf_sd = cbm_fisher_spectrum(
        lambda m, ps: _miyake_forward(m, ps, t_obs),
        cbm_full,
        sigma,
        psi_nuisance=psi0,
        prior_cbm_sd=sd_5,
        prior_psi_sd=sd_psi,
    )
    res_inf_cov = cbm_fisher_spectrum(
        lambda m, ps: _miyake_forward(m, ps, t_obs),
        cbm_full,
        sigma,
        psi_nuisance=psi0,
        prior_cbm_cov=cov_5,
        prior_psi_sd=sd_psi,
    )
    res_zero_prec = cbm_fisher_spectrum(
        lambda m, ps: _miyake_forward(m, ps, t_obs),
        cbm_full,
        sigma,
        psi_nuisance=psi0,
        prior_cbm_prec=prec_5,
        prior_psi_sd=sd_psi,
    )

    np.testing.assert_allclose(res_auto.d_eff, res_inf_sd.d_eff, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        res_inf_cov.eigenvalues, res_orig.eigenvalues, rtol=1e-11, atol=1e-11
    )
    np.testing.assert_allclose(
        res_zero_prec.eigenvalues, res_orig.eigenvalues, rtol=1e-11, atol=1e-11
    )
    for r_chk in (res_auto, res_inf_sd, res_inf_cov, res_zero_prec):
        assert jnp.all(jnp.isfinite(r_chk.eigenvalues))
        assert jnp.all(jnp.isfinite(r_chk.parameter_loadings))
        assert jnp.all(jnp.isfinite(r_chk.posterior_cov))

    # Also verify JIT + reverse-mode grad of info_nats with traced +inf prior_cbm_sd has ZERO NaNs!
    @jax.jit
    def _jit_info(th: jax.Array, traced_sd: jax.Array) -> jax.Array:
        return cbm_fisher_spectrum(
            lambda m, ps: _miyake_forward(m, ps, t_obs),
            _to_cbm(th),
            sigma,
            psi_nuisance=psi0,
            prior_cbm_sd=traced_sd,
            prior_psi_sd=sd_psi,
        ).info_nats

    val_jit, grad_jit = jax.value_and_grad(_jit_info)(theta0, sd_5)
    assert jnp.isfinite(val_jit)
    assert jnp.all(jnp.isfinite(grad_jit))
    np.testing.assert_allclose(val_jit, res_inf_sd.info_nats, rtol=1e-11, atol=1e-11)


def test_relaxation_fisher_modes_mcwhirter_pike_spectrum():
    """Verify ``relaxation_fisher_modes`` reproduces both the exact Mellin-Toeplitz
    operator ``1 / (2 cosh((u - u')/2))`` and McWhirter & Pike's (1978) analytical
    ``π / cosh(π ω_ln)`` singular-value decay under ``measure="l2_ds"``.
    """
    tau_min, tau_max = 1e-3, 1e3
    n_lags = 220
    w_ln = math.log(tau_max / tau_min)
    d_ln_tau = w_ln / n_lags
    ln_tau = math.log(tau_min) + (jnp.arange(n_lags, dtype=jnp.float64) + 0.5) * d_ln_tau
    taus = jnp.exp(ln_tau)
    sigma_quad = 1.0 / jnp.sqrt(taus * d_ln_tau)

    res_l2 = relaxation_fisher_modes(
        taus,
        sigma_d=sigma_quad,
        s_min=1e-6,
        s_max=1e6,
        n_rates=480,
        measure="l2_ds",
    )
    assert isinstance(res_l2, RelaxationModesResult)
    assert res_l2.measure == "l2_ds"
    assert float(res_l2.eigenvalues[0]) < math.pi

    # 1) Exact Mellin-Toeplitz operator M(u, u') = d_ln_tau / (2 cosh((u - u') / 2)):
    diff_u = ln_tau[:, None] - ln_tau[None, :]
    m_toeplitz = d_ln_tau / (2.0 * jnp.cosh(0.5 * diff_u))
    evals_toeplitz = jnp.linalg.eigvalsh(m_toeplitz)[::-1]
    np.testing.assert_allclose(
        res_l2.eigenvalues[:12], evals_toeplitz[:12], rtol=1e-3, atol=1e-4
    )

    # 2) McWhirter & Pike / Widom closed-form symbol μ_k ≈ π / cosh(π ω_ln,k):
    k_idx = jnp.arange(12, dtype=jnp.float64)
    omega_mp = math.pi * (k_idx + 0.5) / (w_ln + 2.0 * math.log(2.0))
    mu_mp = math.pi / jnp.cosh(math.pi * omega_mp)
    np.testing.assert_allclose(
        jnp.log(res_l2.eigenvalues[:12]), jnp.log(mu_mp), atol=0.13
    )
    np.testing.assert_allclose(
        jnp.log(res_l2.eigenvalues[5:12]), jnp.log(mu_mp[5:12]), atol=0.06
    )

    slope_emp = float(
        np.polyfit(np.arange(4, 12), np.log(np.asarray(res_l2.eigenvalues[4:12])), 1)[0]
    )
    slope_mp = -(math.pi**2) / (w_ln + 2.0 * math.log(2.0))
    np.testing.assert_allclose(slope_emp, slope_mp, rtol=0.02)

    # 3) Seasonal windows + measure="d_ln_s" + marginalize_dc_and_scale=True:
    t_s = jnp.arange(0.0, 20.0, 1.0) + 0.25
    t_e = t_s + 0.4
    res_dlns = relaxation_fisher_modes(
        t_s,
        t_e,
        sigma_d=1.2,
        s_min=1e-4,
        s_max=20.0,
        n_rates=128,
        measure="d_ln_s",
        marginalize_dc_and_scale=True,
    )
    assert res_dlns.measure == "d_ln_s"
    assert res_dlns.rate_modes.shape == (128, 20)
    np.testing.assert_allclose(
        jnp.sum(res_dlns.design_matrix, axis=0), 0.0, atol=1e-12
    )
