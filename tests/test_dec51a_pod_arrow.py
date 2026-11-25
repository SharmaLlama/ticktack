"""Direct tests for ``ticktack.kalman`` (Arrow & Canonical POD) and ``_whiten_rank_one``.

Decision 5.1 Part 1 verification suite:
1. Direct imports and ``__all__`` contract on ``ticktack.kalman``, removal of
   moved/privatized names from ``ticktack.linear.__all__``, and temporary
   module-level compatibility attributes on ``ticktack.linear`` and ``ticktack``.
2. ``arrow_problem`` and ``arrow_at`` verified against an independent dense
   block-matrix Schur complement and dense covariance formula (padded unequal
   blocks with both ``n_obs`` and ``-1`` sentinels, duplicate active-row
   rejection, outside rows, zero spanning columns, ``float32`` ``scale2``
   upcasting, off-unit variance scales) plus ``jax.grad`` cross-checked against
   both dense AD and central finite differences.
3. ``_wiener_pod_svd`` and ``canonical_pod_modes`` verified against full
   ``N_modes`` dense GP marginal covariance, posterior covariance, effective
   degrees of freedom, Shannon information gain, dense SVD, high-SNR variance
   cancellation resistance, ill-conditioned QR nuisance whitening, input
   validation, and ``jax.grad``/``jax.jvp`` when ``N_obs < r_modes``.
4. ``ticktack.linear._whiten_rank_one`` verified against dense ``I + U Uᵀ``
   quadratic form, log-determinant, autodiff gradients, zero-column edge case,
   and ``LinearMarginal(..., method="rank_one")``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import jax.numpy as jnp
import jax.scipy.stats as jss
import numpy as np
import pytest
import tinygp.kernels as tgk
from jax.test_util import check_grads

import ticktack
import ticktack.kalman as tt_kalman
import ticktack.linear as tt_linear
from ticktack.kalman import (
    ArrowProblem,
    CanonicalPODResult,
    _wiener_pod_svd,
    arrow_at,
    arrow_problem,
    canonical_pod_modes,
)
from ticktack.linear import (
    LinearMarginal,
    _whiten_rank_one,
    project_causal_gp,
    solin_sarkka_basis,
)

LAMBDA_14 = math.log(2.0) / 5730.0


def _assemble_dense_arrow_design(
    n_obs: int,
    rows: np.ndarray,
    local: np.ndarray,
    spanning: np.ndarray | None,
) -> jnp.ndarray:
    """Independently assemble dense ``U = [U_local, S]`` ignoring inactive rows."""
    n_block, _, n_mode = local.shape
    u_local = np.zeros((n_obs, n_block * n_mode), dtype=np.float64)
    for b_idx in range(n_block):
        for r_slot, row_idx in enumerate(rows[b_idx]):
            if 0 <= int(row_idx) < n_obs:
                u_local[int(row_idx), n_mode * b_idx : n_mode * (b_idx + 1)] = local[
                    b_idx, r_slot
                ]
    if spanning is None or spanning.shape[1] == 0:
        return jnp.asarray(u_local, dtype=jnp.float64)
    return jnp.concatenate(
        [jnp.asarray(u_local, dtype=jnp.float64), jnp.asarray(spanning, dtype=jnp.float64)],
        axis=1,
    )


def _dense_block_schur_oracle(
    x: jax.Array,
    u_local: jax.Array,
    spanning: jax.Array,
    scale2: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Independent dense 2x2 block-matrix Schur complement oracle for ``s² I + U Uᵀ``."""
    p_loc = u_local.shape[1]
    p_span = spanning.shape[1]
    a_bb = scale2 * jnp.eye(p_loc, dtype=jnp.float64) + u_local.T @ u_local
    b_bs = u_local.T @ spanning
    c_ss = scale2 * jnp.eye(p_span, dtype=jnp.float64) + spanning.T @ spanning

    a_inv = jnp.linalg.inv(a_bb)
    schur = c_ss - b_bs.T @ a_inv @ b_bs
    schur_inv = jnp.linalg.inv(schur)

    rhs_b = u_local.T @ x
    rhs_s = spanning.T @ x
    y_s = schur_inv @ (rhs_s - b_bs.T @ (a_inv @ rhs_b))
    y_b = a_inv @ (rhs_b - b_bs @ y_s)

    resid = x - u_local @ y_b - spanning @ y_s
    r2 = jnp.sum(resid**2)
    chi2 = r2 / scale2 + jnp.sum(y_b**2) + jnp.sum(y_s**2)

    _, ld_a = jnp.linalg.slogdet(a_bb)
    _, ld_s = jnp.linalg.slogdet(schur)
    norm = 0.5 * (ld_a + ld_s - (p_loc + p_span) * jnp.log(scale2))

    # Block inverse trace: tr(A^{-1} + A^{-1} B S^{-1} B^T A^{-1}) + tr(S^{-1})
    a_inv_b = a_inv @ b_bs
    tr = jnp.trace(a_inv) + jnp.trace(schur_inv @ (a_inv_b.T @ a_inv_b + jnp.eye(p_span)))
    return chi2, norm, r2, tr


def _make_asymmetric_cbm() -> tuple[jax.Array, jax.Array, jax.Array]:
    """Asymmetric stiff 3-box CBM with unequal rates, injection, and observation weights."""
    a_mat = jnp.array(
        [
            [-(1.55 + 0.72 + LAMBDA_14), 0.63, 0.031],
            [1.55, -(0.63 + 0.008 + LAMBDA_14), 0.0],
            [0.72, 0.0, -(0.031 + 0.0035 + LAMBDA_14)],
        ],
        dtype=jnp.float64,
    )
    b_vec = jnp.array([0.81, 0.19, 0.0], dtype=jnp.float64)
    c_obs = jnp.array([1.35, 0.22, -0.04], dtype=jnp.float64)
    return a_mat, b_vec, c_obs


def test_kalman_exports_and_compatibility_aliases():
    assert tt_kalman.__all__ == [
        "ArrowProblem",
        "CanonicalPODResult",
        "arrow_at",
        "arrow_problem",
        "canonical_pod_modes",
    ]
    for name in tt_kalman.__all__:
        obj = getattr(tt_kalman, name)
        assert name not in tt_linear.__all__
        assert getattr(tt_linear, name) is obj
        assert getattr(ticktack, name) is obj

    assert "whiten_rank_one" not in tt_linear.__all__
    assert tt_linear.whiten_rank_one is _whiten_rank_one
    assert ticktack.whiten_rank_one is _whiten_rank_one


def test_arrow_problem_and_arrow_at_against_dense_schur_and_grad():
    rng = np.random.default_rng(5101)
    n_obs = 29
    # 4 unequal blocks of sizes (5, 7, 4, 6) padded with sentinel index n_obs=29;
    # rows 22..28 remain outside all blocks so gram_out is genuinely non-zero.
    block_rows = [
        [0, 3, 7, 11, 18, n_obs, n_obs],
        [1, 4, 8, 12, 15, 19, 21],
        [2, 5, 9, 14, n_obs, n_obs, n_obs],
        [6, 10, 13, 16, 17, 20, n_obs],
    ]
    rows_np = np.asarray(block_rows, dtype=np.int32)
    n_block, max_r, n_mode = 4, 7, 3
    n_span = 4

    x_np = 1.4 + rng.standard_normal(n_obs) * 2.3
    mode_scales = np.array([0.35, 2.7, 11.4], dtype=np.float64)
    local_np = rng.standard_normal((n_block, max_r, n_mode)) * mode_scales[None, None, :]
    # Deliberately populate padded slots with large garbage to prove rows >= n_obs are masked.
    local_np[rows_np >= n_obs] = 999.0
    span_scales = np.array([0.6, 3.8, 19.0, 0.15], dtype=np.float64)
    spanning_np = rng.standard_normal((n_obs, n_span)) * span_scales[None, :]

    x = jnp.asarray(x_np, dtype=jnp.float64)
    rows = jnp.asarray(rows_np, dtype=jnp.int32)
    local = jnp.asarray(local_np, dtype=jnp.float64)
    spanning = jnp.asarray(spanning_np, dtype=jnp.float64)

    u_local = _assemble_dense_arrow_design(n_obs, rows_np, local_np, None)
    u_dense = _assemble_dense_arrow_design(n_obs, rows_np, local_np, spanning_np)

    prob = arrow_problem(x, rows, local, spanning)
    assert isinstance(prob, ArrowProblem)

    for scale2_val in (0.17, 2.85, 13.4):
        scale2 = jnp.asarray(scale2_val, dtype=jnp.float64)
        chi2, norm, r2, tr = arrow_at(prob, scale2, trace=True)
        chi2_notr, norm_notr, r2_notr = arrow_at(prob, scale2, trace=False)
        np.testing.assert_allclose(chi2_notr, chi2, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(norm_notr, norm, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(r2_notr, r2, rtol=1e-14, atol=1e-14)

        # 1. Cross-check against independent dense 2x2 block-matrix Schur complement
        chi2_schur, norm_schur, r2_schur, tr_schur = _dense_block_schur_oracle(
            x, u_local, spanning, scale2
        )
        np.testing.assert_allclose(chi2, chi2_schur, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(norm, norm_schur, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(r2, r2_schur, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(tr, tr_schur, rtol=1e-12, atol=1e-12)

        # 2. Cross-check against direct dense observation covariance s² I + U Uᵀ
        cov_s = scale2 * jnp.eye(n_obs, dtype=jnp.float64) + u_dense @ u_dense.T
        chi2_cov = x @ jnp.linalg.solve(cov_s, x)
        _, ld_cov = jnp.linalg.slogdet(cov_s)
        norm_cov = 0.5 * (ld_cov - n_obs * jnp.log(scale2))
        np.testing.assert_allclose(chi2, chi2_cov, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(norm, norm_cov, rtol=1e-12, atol=1e-12)

    # 3. Zero-spanning (spanning=None) path
    prob_nospan = arrow_problem(x, rows, local, spanning=None)
    scale2_ns = jnp.asarray(1.93, dtype=jnp.float64)
    chi2_ns, norm_ns, r2_ns, tr_ns = arrow_at(prob_nospan, scale2_ns, trace=True)
    cov_ns = scale2_ns * jnp.eye(n_obs, dtype=jnp.float64) + u_local @ u_local.T
    chi2_ns_ref = x @ jnp.linalg.solve(cov_ns, x)
    _, ld_ns = jnp.linalg.slogdet(cov_ns)
    norm_ns_ref = 0.5 * (ld_ns - n_obs * jnp.log(scale2_ns))
    m_loc = scale2_ns * jnp.eye(u_local.shape[1], dtype=jnp.float64) + u_local.T @ u_local
    y_loc_ref = jnp.linalg.solve(m_loc, u_local.T @ x)
    r2_ns_ref = jnp.sum((x - u_local @ y_loc_ref) ** 2)
    tr_ns_ref = jnp.trace(jnp.linalg.inv(m_loc))
    np.testing.assert_allclose(chi2_ns, chi2_ns_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(norm_ns, norm_ns_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(r2_ns, r2_ns_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(tr_ns, tr_ns_ref, rtol=1e-12, atol=1e-12)

    # 4. jax.grad wrt (scale2, x, local, spanning) vs dense Schur AD and central FD
    weights = jnp.array([1.3, -0.7, 0.4, 2.1], dtype=jnp.float64)

    def arrow_scalar(
        s2: jax.Array, x_in: jax.Array, loc_in: jax.Array, span_in: jax.Array
    ) -> jax.Array:
        out = arrow_at(arrow_problem(x_in, rows, loc_in, span_in), s2, trace=True)
        return sum(w * v for w, v in zip(weights, out))

    def dense_scalar(
        s2: jax.Array, x_in: jax.Array, loc_in: jax.Array, span_in: jax.Array
    ) -> jax.Array:
        loc_masked = jnp.where((rows < n_obs)[..., None], loc_in, 0.0)
        u_loc_dyn = jnp.zeros((n_obs, n_block * n_mode), dtype=jnp.float64)
        for b_idx in range(n_block):
            u_loc_dyn = u_loc_dyn.at[
                rows[b_idx], n_mode * b_idx : n_mode * (b_idx + 1)
            ].add(loc_masked[b_idx], mode="drop")
        out = _dense_block_schur_oracle(x_in, u_loc_dyn, span_in, s2)
        return sum(w * v for w, v in zip(weights, out))

    s2_test = jnp.asarray(2.85, dtype=jnp.float64)
    g_arrow = jax.grad(arrow_scalar, argnums=(0, 1, 2, 3))(s2_test, x, local, spanning)
    g_dense = jax.grad(dense_scalar, argnums=(0, 1, 2, 3))(s2_test, x, local, spanning)
    for ga, gd in zip(g_arrow, g_dense):
        np.testing.assert_allclose(ga, gd, rtol=1e-10, atol=1e-10)
    # Masked padded slots must carry exact zero gradient.
    np.testing.assert_array_equal(g_arrow[2][rows_np >= n_obs], 0.0)

    # Independent central finite-difference check on scale2 and 2nd-order check_grads
    eps = 1e-6
    fd_s2 = (
        arrow_scalar(s2_test + eps, x, local, spanning)
        - arrow_scalar(s2_test - eps, x, local, spanning)
    ) / (2.0 * eps)
    np.testing.assert_allclose(g_arrow[0], fd_s2, rtol=1e-7, atol=1e-7)
    check_grads(
        lambda s2: arrow_scalar(s2, x, local, spanning),
        (s2_test,),
        order=2,
        modes=["rev"],
        rtol=1e-5,
        atol=1e-5,
    )


def test_arrow_negative_sentinel_duplicate_rows_and_float32_scale2():
    rng = np.random.default_rng(5104)
    n_obs = 17
    # Mix -1, -5, and n_obs sentinel padding while leaving row n_obs - 1 = 16 outside blocks
    # with large spanning weight so any negative-index wraparound would visibly corrupt schur.
    rows_neg_np = np.asarray(
        [
            [0, 2, 5, -1, -1],
            [1, 3, 6, 9, 12],
            [4, 7, 10, -5, n_obs],
        ],
        dtype=np.int32,
    )
    local_np = rng.standard_normal((3, 5, 2)) * np.array([0.9, 4.2])[None, None, :]
    local_np[(rows_neg_np < 0) | (rows_neg_np >= n_obs)] = 777.0
    spanning_np = rng.standard_normal((n_obs, 3)) * np.array([1.2, 5.5, 0.4])[None, :]
    spanning_np[-1] = np.array([12.0, -8.5, 6.0], dtype=np.float64)
    x_np = rng.standard_normal(n_obs) * 2.1

    x = jnp.asarray(x_np, dtype=jnp.float64)
    local = jnp.asarray(local_np, dtype=jnp.float64)
    spanning = jnp.asarray(spanning_np, dtype=jnp.float64)
    prob = arrow_problem(x, jnp.asarray(rows_neg_np, dtype=jnp.int32), local, spanning)

    # Inactive entries must be normalized to sentinel n_obs (never negative).
    expected_safe = np.where((rows_neg_np >= 0) & (rows_neg_np < n_obs), rows_neg_np, n_obs)
    np.testing.assert_array_equal(np.asarray(prob.rows), expected_safe)

    u_local = _assemble_dense_arrow_design(n_obs, rows_neg_np, local_np, None)
    s2_f32 = jnp.float32(1.73)
    s2_f64 = jnp.asarray(s2_f32, dtype=jnp.float64)

    out_f32 = arrow_at(prob, s2_f32, trace=True)
    out_ref = _dense_block_schur_oracle(x, u_local, spanning, s2_f64)
    for val, ref in zip(out_f32, out_ref):
        assert val.dtype == jnp.float64
        np.testing.assert_allclose(val, ref, rtol=1e-12, atol=1e-12)

    # Reject duplicate active row indices (both cross-block and intra-block)
    rows_dup_cross = np.asarray([[0, 2, 5, -1], [2, 3, 6, -1]], dtype=np.int32)
    with pytest.raises(ValueError, match="duplicate active"):
        arrow_problem(
            x,
            jnp.asarray(rows_dup_cross, dtype=jnp.int32),
            jnp.ones((2, 4, 2), dtype=jnp.float64),
            spanning,
        )

    rows_dup_intra = np.asarray([[0, 0, 5, -1], [1, 3, 6, -1]], dtype=np.int32)
    with pytest.raises(ValueError, match="duplicate active"):
        arrow_problem(
            x,
            jnp.asarray(rows_dup_intra, dtype=jnp.int32),
            jnp.ones((2, 4, 2), dtype=jnp.float64),
            spanning,
        )


def test_wiener_pod_svd_kernel_high_snr_and_underdetermined_grad():
    rng = np.random.default_rng(5105)
    n_obs, r_modes, n_eval = 6, 14, 19
    g_base = jnp.asarray(rng.standard_normal((n_obs, r_modes)), dtype=jnp.float64)
    r_tilde = jnp.asarray(rng.standard_normal(n_obs) * 1.9, dtype=jnp.float64)
    phi_eval = jnp.asarray(rng.standard_normal((n_eval, r_modes)) * 0.7, dtype=jnp.float64)

    # High-SNR regime (s_k ~ 1e7) where naive prior_var - reduction cancels catastrophically
    g_high = g_base * 1e7
    res_high = _wiener_pod_svd(g_high, r_tilde, phi_eval)
    assert res_high.singular_values.shape == (n_obs,)
    assert res_high.wiener_weights.shape == (r_modes,)
    assert res_high.data_modes.shape == (n_obs, n_obs)
    assert res_high.prior_rotation.shape == (r_modes, r_modes)
    assert jnp.all(jnp.isfinite(res_high.posterior_sd))
    assert jnp.all(res_high.posterior_sd > 0.0)

    # Thin SVD reconstruction G̃ = U_data @ diag(s) @ V_thinᵀ
    v_thin = res_high.prior_rotation[:, :n_obs]
    g_rec = (res_high.data_modes * res_high.singular_values[None, :]) @ v_thin.T
    np.testing.assert_allclose(g_rec, g_high, rtol=1e-12, atol=1e-5)

    # High-SNR nullspace variance matches orthogonal-complement projection
    _, _, vt_ref = np.linalg.svd(np.asarray(g_high), full_matrices=False)
    phi_np = np.asarray(phi_eval)
    phi_perp_ref = phi_np - (phi_np @ vt_ref.T) @ vt_ref
    phi_perp_ref = phi_perp_ref - (phi_perp_ref @ vt_ref.T) @ vt_ref
    s_np = np.linalg.svd(np.asarray(g_high), compute_uv=False)
    var_ref = np.sum(((phi_np @ vt_ref.T) ** 2) / (1.0 + s_np[None, :] ** 2), axis=1) + np.sum(
        phi_perp_ref**2, axis=1
    )
    np.testing.assert_allclose(res_high.posterior_sd, np.sqrt(var_ref), rtol=1e-12, atol=1e-12)

    # Direct jax.grad and jax.jvp check on _wiener_pod_svd when r_modes - n_obs = 8 >= 2
    def kernel_obj(g_in: jax.Array, r_in: jax.Array) -> jax.Array:
        pod = _wiener_pod_svd(g_in, r_in, phi_eval)
        return (
            jnp.sum(pod.posterior_mean * jnp.linspace(-0.5, 1.3, n_eval))
            + jnp.sum(pod.posterior_sd * jnp.linspace(0.4, 1.1, n_eval))
            + 0.7 * pod.d_eff
            - 0.3 * pod.info_nats
        )

    def dense_obj(g_in: jax.Array, r_in: jax.Array) -> jax.Array:
        prec = jnp.eye(r_modes, dtype=jnp.float64) + g_in.T @ g_in
        cov_c = jnp.linalg.inv(prec)
        mean_c = cov_c @ (g_in.T @ r_in)
        post_mean = phi_eval @ mean_c
        post_var = jnp.sum((phi_eval @ cov_c) * phi_eval, axis=1)
        post_sd = jnp.sqrt(jnp.maximum(post_var, 0.0))
        d_eff = jnp.trace(jnp.eye(r_modes, dtype=jnp.float64) - cov_c)
        _, ld_p = jnp.linalg.slogdet(prec)
        info_nats = 0.5 * ld_p
        return (
            jnp.sum(post_mean * jnp.linspace(-0.5, 1.3, n_eval))
            + jnp.sum(post_sd * jnp.linspace(0.4, 1.1, n_eval))
            + 0.7 * d_eff
            - 0.3 * info_nats
        )

    g_pod_g, g_pod_r = jax.grad(kernel_obj, argnums=(0, 1))(g_base, r_tilde)
    g_ref_g, g_ref_r = jax.grad(dense_obj, argnums=(0, 1))(g_base, r_tilde)
    assert jnp.all(jnp.isfinite(g_pod_g)) and jnp.all(jnp.isfinite(g_pod_r))
    np.testing.assert_allclose(g_pod_g, g_ref_g, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(g_pod_r, g_ref_r, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    ("n_obs", "r_modes", "include_initial_cov", "n_nuis"),
    [
        (14, 32, True, 3),  # underdetermined N_obs < r with initial cov + multi-col nuisance
        (24, 12, False, 0),  # overdetermined N_obs > r with zero nuisance
    ],
)
def test_canonical_pod_modes_against_full_gp_covariance_and_svd(
    n_obs: int,
    r_modes: int,
    include_initial_cov: bool,
    n_nuis: int,
):
    rng = np.random.default_rng(5102 + n_obs + r_modes)
    a_mat, b_vec, c_obs = _make_asymmetric_cbm()
    t_starts = jnp.linspace(770.2, 789.4, n_obs, dtype=jnp.float64)
    durations = jnp.linspace(0.32, 0.78, n_obs, dtype=jnp.float64)
    t_ends = t_starts + durations
    obs_mat = jnp.broadcast_to(c_obs[None, :], (n_obs, 3))
    sigma = jnp.linspace(0.65, 2.4, n_obs, dtype=jnp.float64)
    resid = jnp.asarray(-0.9 + rng.standard_normal(n_obs) * 2.1, dtype=jnp.float64)
    t_eval = jnp.linspace(769.5, 790.5, 37, dtype=jnp.float64)

    kernel = (1.65**2) * tgk.Matern52(2.8)
    proj = project_causal_gp(
        a_mat,
        b_vec,
        t_starts,
        t_ends,
        pad=18.0,
        n_modes=r_modes,
        kernel=kernel,
        alpha_unit=1.75,
        obs_matrix=obs_mat,
        include_initial_cov=include_initial_cov,
    )

    if n_nuis > 0:
        nuis_scales = np.array([150.0, 2.3, 0.8][:n_nuis], dtype=np.float64)
        nuis_w = jnp.asarray(
            rng.standard_normal((n_obs, n_nuis)) * nuis_scales[None, :],
            dtype=jnp.float64,
        )
    else:
        nuis_w = None

    pod = canonical_pod_modes(
        proj,
        t_eval,
        resid,
        sigma,
        nuisance_whitened_design=nuis_w,
        include_initial_cov=include_initial_cov,
    )
    assert isinstance(pod, CanonicalPODResult)

    # Independent dense full-N_modes GP marginal covariance and SVD oracle
    h_gp = proj.causal_basis_design * jnp.sqrt(proj.spectral_weights)[None, :]  # (N_obs, r)
    sigma_tilde = jnp.diag(sigma**2)
    if include_initial_cov and proj.anchor_cov_root is not None:
        h_init = proj.hom_design_anchor @ proj.anchor_cov_root
        sigma_tilde = sigma_tilde + h_init @ h_init.T
    if nuis_w is not None:
        sigma_tilde = sigma_tilde + nuis_w @ nuis_w.T
    sigma_tot = sigma_tilde + h_gp @ h_gp.T

    chol_tilde = jnp.linalg.cholesky(0.5 * (sigma_tilde + sigma_tilde.T))
    g_dense = jax.scipy.linalg.solve_triangular(chol_tilde, h_gp, lower=True)
    s_ref = jnp.linalg.svd(g_dense, compute_uv=False)

    # Full-N_modes GP posterior via dense normal equations
    sig_tilde_inv_h = jax.scipy.linalg.cho_solve((chol_tilde, True), h_gp)
    sig_tilde_inv_r = jax.scipy.linalg.cho_solve((chol_tilde, True), resid)
    prec_c = jnp.eye(r_modes, dtype=jnp.float64) + h_gp.T @ sig_tilde_inv_h
    cov_c = jnp.linalg.inv(prec_c)
    mean_c = cov_c @ (h_gp.T @ sig_tilde_inv_r)

    basis_eval = solin_sarkka_basis(t_eval, proj.a, proj.b_bound, r_modes)
    phi_eval = basis_eval.psi * jnp.sqrt(proj.spectral_weights)[None, :]
    q_mean_ref = phi_eval @ mean_c
    q_cov_ref = phi_eval @ cov_c @ phi_eval.T
    q_sd_ref = jnp.sqrt(jnp.maximum(jnp.diag(q_cov_ref), 0.0))

    # 1. Singular values, Wiener shrinkage weights, d_eff, and information gain (nats)
    np.testing.assert_allclose(pod.singular_values, s_ref, rtol=1e-10, atol=1e-10)
    s_padded = jnp.zeros((r_modes,), dtype=jnp.float64).at[: len(s_ref)].set(s_ref)
    np.testing.assert_allclose(
        pod.wiener_weights, s_padded**2 / (1.0 + s_padded**2), rtol=1e-12, atol=1e-12
    )
    d_eff_ref = jnp.trace(jnp.eye(r_modes, dtype=jnp.float64) - cov_c)
    _, ld_tot = jnp.linalg.slogdet(sigma_tot)
    _, ld_tilde = jnp.linalg.slogdet(sigma_tilde)
    info_nats_ref = 0.5 * (ld_tot - ld_tilde)
    np.testing.assert_allclose(pod.d_eff, d_eff_ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(pod.info_nats, info_nats_ref, rtol=1e-10, atol=1e-10)

    # 2. Posterior mean, SD, and full off-diagonal posterior covariance reconstruction
    np.testing.assert_allclose(pod.posterior_mean, q_mean_ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(pod.posterior_sd, q_sd_ref, rtol=1e-10, atol=1e-10)
    q_cov_pod = (pod.production_modes * pod.coeff_var[None, :]) @ pod.production_modes.T
    np.testing.assert_allclose(q_cov_pod, q_cov_ref, rtol=1e-10, atol=1e-10)

    # 3. Prior & data orthonormality and prior covariance invariance
    np.testing.assert_allclose(
        pod.prior_rotation.T @ pod.prior_rotation,
        jnp.eye(r_modes, dtype=jnp.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        pod.data_modes.T @ pod.data_modes,
        jnp.eye(min(n_obs, r_modes), dtype=jnp.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        pod.production_modes @ pod.production_modes.T,
        phi_eval @ phi_eval.T,
        rtol=1e-11,
        atol=1e-11,
    )


def test_canonical_pod_modes_grad_and_jvp_underdetermined():
    """Verify NaN-free jax.grad and jax.jvp when r_modes - n_obs >= 2."""
    rng = np.random.default_rng(5106)
    n_obs, r_modes = 8, 20
    a_mat0, b_vec, c_obs = _make_asymmetric_cbm()
    t_starts = jnp.linspace(770.2, 777.2, n_obs, dtype=jnp.float64)
    t_ends = t_starts + jnp.linspace(0.35, 0.72, n_obs, dtype=jnp.float64)
    obs_mat = jnp.broadcast_to(c_obs[None, :], (n_obs, 3))
    resid0 = jnp.asarray(-0.6 + rng.standard_normal(n_obs) * 1.7, dtype=jnp.float64)
    sigma0 = jnp.linspace(0.75, 1.95, n_obs, dtype=jnp.float64)
    nuis_w = jnp.asarray(
        rng.standard_normal((n_obs, 2)) * np.array([120.0, 1.8])[None, :],
        dtype=jnp.float64,
    )
    t_eval = jnp.linspace(770.0, 778.0, 15, dtype=jnp.float64)
    log_hyp0 = jnp.array([math.log(1.45), math.log(2.6)], dtype=jnp.float64)
    w_mean = jnp.linspace(-0.8, 1.2, t_eval.shape[0], dtype=jnp.float64)
    w_sd = jnp.linspace(0.5, 1.4, t_eval.shape[0], dtype=jnp.float64)

    def pod_scalar(
        resid_in: jax.Array,
        sigma_in: jax.Array,
        a_in: jax.Array,
        log_hyp: jax.Array,
    ) -> jax.Array:
        kern = jnp.exp(2.0 * log_hyp[0]) * tgk.Matern52(jnp.exp(log_hyp[1]))
        proj = project_causal_gp(
            a_in,
            b_vec,
            t_starts,
            t_ends,
            pad=14.0,
            n_modes=r_modes,
            kernel=kern,
            alpha_unit=1.5,
            obs_matrix=obs_mat,
            include_initial_cov=True,
        )
        pod = canonical_pod_modes(
            proj,
            t_eval,
            resid_in,
            sigma_in,
            nuisance_whitened_design=nuis_w,
            include_initial_cov=True,
        )
        return (
            jnp.dot(w_mean, pod.posterior_mean)
            + jnp.dot(w_sd, pod.posterior_sd)
            + 0.6 * pod.d_eff
            - 0.4 * pod.info_nats
        )

    def dense_scalar(
        resid_in: jax.Array,
        sigma_in: jax.Array,
        a_in: jax.Array,
        log_hyp: jax.Array,
    ) -> jax.Array:
        kern = jnp.exp(2.0 * log_hyp[0]) * tgk.Matern52(jnp.exp(log_hyp[1]))
        proj = project_causal_gp(
            a_in,
            b_vec,
            t_starts,
            t_ends,
            pad=14.0,
            n_modes=r_modes,
            kernel=kern,
            alpha_unit=1.5,
            obs_matrix=obs_mat,
            include_initial_cov=True,
        )
        h_gp = proj.causal_basis_design * jnp.sqrt(proj.spectral_weights)[None, :]
        h_init = proj.hom_design_anchor @ proj.anchor_cov_root
        sigma_tilde = jnp.diag(sigma_in**2) + h_init @ h_init.T + nuis_w @ nuis_w.T
        chol_t = jnp.linalg.cholesky(0.5 * (sigma_tilde + sigma_tilde.T))
        st_inv_h = jax.scipy.linalg.cho_solve((chol_t, True), h_gp)
        st_inv_r = jax.scipy.linalg.cho_solve((chol_t, True), resid_in)
        prec_c = jnp.eye(r_modes, dtype=jnp.float64) + h_gp.T @ st_inv_h
        cov_c = jnp.linalg.inv(prec_c)
        mean_c = cov_c @ (h_gp.T @ st_inv_r)
        basis_eval = solin_sarkka_basis(t_eval, proj.a, proj.b_bound, r_modes)
        phi_eval = basis_eval.psi * jnp.sqrt(proj.spectral_weights)[None, :]
        post_mean = phi_eval @ mean_c
        post_sd = jnp.sqrt(jnp.maximum(jnp.sum((phi_eval @ cov_c) * phi_eval, axis=1), 0.0))
        d_eff = jnp.trace(jnp.eye(r_modes, dtype=jnp.float64) - cov_c)
        _, ld_p = jnp.linalg.slogdet(prec_c)
        return (
            jnp.dot(w_mean, post_mean)
            + jnp.dot(w_sd, post_sd)
            + 0.6 * d_eff
            - 0.4 * (0.5 * ld_p)
        )

    grads_pod = jax.grad(pod_scalar, argnums=(0, 1, 2, 3))(resid0, sigma0, a_mat0, log_hyp0)
    grads_ref = jax.grad(dense_scalar, argnums=(0, 1, 2, 3))(resid0, sigma0, a_mat0, log_hyp0)
    for gp, gr in zip(grads_pod, grads_ref):
        assert jnp.all(jnp.isfinite(gp))
        np.testing.assert_allclose(gp, gr, rtol=1e-9, atol=1e-9)

    # Forward-mode JVP check along asymmetric tangent directions
    tangents = (
        jnp.linspace(-0.2, 0.3, n_obs, dtype=jnp.float64),
        jnp.linspace(0.05, -0.07, n_obs, dtype=jnp.float64),
        jnp.zeros_like(a_mat0),
        jnp.array([0.11, -0.09], dtype=jnp.float64),
    )
    val_pod, dval_pod = jax.jvp(pod_scalar, (resid0, sigma0, a_mat0, log_hyp0), tangents)
    val_ref, dval_ref = jax.jvp(dense_scalar, (resid0, sigma0, a_mat0, log_hyp0), tangents)
    assert jnp.isfinite(dval_pod)
    np.testing.assert_allclose(val_pod, val_ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(dval_pod, dval_ref, rtol=1e-9, atol=1e-9)


def test_canonical_pod_modes_validation_and_ill_conditioned_nuisance():
    rng = np.random.default_rng(5107)
    n_obs, r_modes = 14, 24
    a_mat, b_vec, c_obs = _make_asymmetric_cbm()
    t_starts = jnp.linspace(770.0, 783.0, n_obs, dtype=jnp.float64)
    t_ends = t_starts + 0.5
    obs_mat = jnp.broadcast_to(c_obs[None, :], (n_obs, 3))
    resid = jnp.asarray(rng.standard_normal(n_obs) * 1.4, dtype=jnp.float64)
    sigma = jnp.linspace(0.8, 1.6, n_obs, dtype=jnp.float64)
    t_eval = jnp.linspace(770.0, 783.5, 21, dtype=jnp.float64)

    proj_no_kern = project_causal_gp(
        a_mat, b_vec, t_starts, t_ends, n_modes=8, kernel=None, obs_matrix=obs_mat
    )
    with pytest.raises(ValueError, match="kernel"):
        canonical_pod_modes(proj_no_kern, t_eval, resid, sigma)

    proj = project_causal_gp(
        a_mat,
        b_vec,
        t_starts,
        t_ends,
        pad=15.0,
        n_modes=r_modes,
        kernel=(1.3**2) * tgk.Matern52(2.4),
        obs_matrix=obs_mat,
        include_initial_cov=True,
    )

    # 1D nuisance_whitened_design must be rejected (never silently broadcast to (n_obs, n_obs))
    with pytest.raises(ValueError, match="nuisance_whitened_design"):
        canonical_pod_modes(
            proj, t_eval, resid, sigma, nuisance_whitened_design=jnp.ones((n_obs,))
        )
    with pytest.raises(ValueError, match="nuisance_whitened_design"):
        canonical_pod_modes(
            proj, t_eval, resid, sigma, nuisance_whitened_design=jnp.ones((n_obs - 1, 2))
        )

    # Non-positive / non-finite / 2D sigma must be rejected
    with pytest.raises(ValueError, match="sigma"):
        canonical_pod_modes(proj, t_eval, resid, sigma.at[2].set(0.0))
    with pytest.raises(ValueError, match="sigma"):
        canonical_pod_modes(proj, t_eval, resid, -1.0)
    with pytest.raises(ValueError, match="sigma"):
        canonical_pod_modes(proj, t_eval, resid, jnp.eye(n_obs))

    # Extreme diffuse / near-collinear nuisance where I + U_nuis^T U_nuis loses ~15 digits
    # in Cholesky: column 0 has scale 1e9 (||u||^2 ~ 1e19 > 2^53), column 1 has kappa ~ 1e7.
    col0 = jnp.ones((n_obs,), dtype=jnp.float64) * 1e9
    col1 = col0 + jnp.linspace(-1.0, 1.0, n_obs, dtype=jnp.float64) * 1e2
    nuis_ill = jnp.stack([col0, col1], axis=1)

    pod_ill = canonical_pod_modes(
        proj, t_eval, resid, sigma, nuisance_whitened_design=nuis_ill
    )
    assert jnp.all(jnp.isfinite(pod_ill.posterior_mean))
    assert jnp.all(jnp.isfinite(pod_ill.posterior_sd))
    assert jnp.all(jnp.isfinite(pod_ill.singular_values))

    # Compare against an independent augmented-QR/SVD oracle on [U_all; I_p]
    # (which never forms U^T U and therefore avoids squaring kappa(U_nuis)).
    u_gp = (proj.causal_basis_design * jnp.sqrt(proj.spectral_weights)[None, :]) / sigma[:, None]
    u_init = (proj.hom_design_anchor @ proj.anchor_cov_root) / sigma[:, None]
    u_nuis = nuis_ill / sigma[:, None]
    u_all = np.asarray(jnp.concatenate([u_gp, u_init, u_nuis], axis=1))
    p_tot = u_all.shape[1]
    aug_a = np.vstack([u_all, np.eye(p_tot, dtype=np.float64)])
    aug_b = np.concatenate([np.asarray(resid / sigma), np.zeros(p_tot, dtype=np.float64)])
    w_ls, _, _, _ = np.linalg.lstsq(aug_a, aug_b, rcond=None)
    _, r_aug = np.linalg.qr(aug_a, mode="reduced")
    r_inv = np.linalg.solve(r_aug, np.eye(p_tot, dtype=np.float64))
    cov_w = r_inv @ r_inv.T

    basis_eval = solin_sarkka_basis(t_eval, proj.a, proj.b_bound, r_modes)
    phi_eval = np.asarray(basis_eval.psi * jnp.sqrt(proj.spectral_weights)[None, :])
    q_mean_ref = phi_eval @ w_ls[:r_modes]
    q_sd_ref = np.sqrt(np.sum((phi_eval @ cov_w[:r_modes, :r_modes]) * phi_eval, axis=1))

    np.testing.assert_allclose(pod_ill.posterior_mean, q_mean_ref, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(pod_ill.posterior_sd, q_sd_ref, rtol=1e-8, atol=1e-8)


def test_private_whiten_rank_one_exactness_and_linear_marginal():
    rng = np.random.default_rng(5103)
    n_obs, k_cols = 19, 6
    x = jnp.asarray(-0.7 + rng.standard_normal(n_obs) * 1.8, dtype=jnp.float64)
    col_scales = np.array([0.2, 1.5, 8.0, 0.05, 3.7, 25.0], dtype=np.float64)
    u_mat = jnp.asarray(
        rng.standard_normal((n_obs, k_cols)) * col_scales[None, :], dtype=jnp.float64
    )

    u_out, half_ld = _whiten_rank_one(x, u_mat)
    cov = jnp.eye(n_obs, dtype=jnp.float64) + u_mat @ u_mat.T
    quad_ref = x @ jnp.linalg.solve(cov, x)
    _, ld_ref = jnp.linalg.slogdet(cov)
    np.testing.assert_allclose(u_out @ u_out, quad_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(half_ld, 0.5 * ld_ref, rtol=1e-12, atol=1e-12)

    # Zero-column edge case
    u_empty, ld_empty = _whiten_rank_one(x, jnp.zeros((n_obs, 0), dtype=jnp.float64))
    np.testing.assert_array_equal(u_empty, x)
    np.testing.assert_array_equal(ld_empty, 0.0)

    # Autodiff wrt (x, U) against dense reference
    def obj_r1(x_in: jax.Array, u_in: jax.Array) -> jax.Array:
        u_w, hld = _whiten_rank_one(x_in, u_in)
        return 0.5 * (u_w @ u_w) + hld

    def obj_dense(x_in: jax.Array, u_in: jax.Array) -> jax.Array:
        c_mat = jnp.eye(n_obs, dtype=jnp.float64) + u_in @ u_in.T
        _, ld = jnp.linalg.slogdet(c_mat)
        return 0.5 * (x_in @ jnp.linalg.solve(c_mat, x_in)) + 0.5 * ld

    g_r1_x, g_r1_u = jax.grad(obj_r1, argnums=(0, 1))(x, u_mat)
    g_ref_x, g_ref_u = jax.grad(obj_dense, argnums=(0, 1))(x, u_mat)
    np.testing.assert_allclose(g_r1_x, g_ref_x, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(g_r1_u, g_ref_u, rtol=1e-11, atol=1e-11)

    # Verify LinearMarginal(method="rank_one") uses _whiten_rank_one cleanly
    sigma = jnp.linspace(0.7, 1.9, n_obs, dtype=jnp.float64)
    prior_mean = jnp.linspace(-0.4, 0.6, k_cols, dtype=jnp.float64)
    prior_sd = jnp.array([0.4, 1.2, 5.0, 0.3, 2.1, 12.0], dtype=jnp.float64)
    design = u_mat
    resid = x * sigma + design @ prior_mean
    lm_r1 = LinearMarginal(design, prior_mean=prior_mean, prior_sd=prior_sd, method="rank_one")
    lm_ch = LinearMarginal(design, prior_mean=prior_mean, prior_sd=prior_sd, method="cholesky")
    dense_cov = jnp.diag(sigma**2) + design @ jnp.diag(prior_sd**2) @ design.T
    ll_dense = jss.multivariate_normal.logpdf(resid, design @ prior_mean, dense_cov)
    np.testing.assert_allclose(lm_r1.loglike(resid, sigma), ll_dense, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        lm_r1.loglike(resid, sigma), lm_ch.loglike(resid, sigma), rtol=1e-12, atol=1e-12
    )
