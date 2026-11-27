"""State-space Kalman/RTS GP marginalization, block-arrow solvers, and canonical POD.

Houses:

1. **Exact continuous-time state-space GP marginalization and smoothing**
   ([`KalmanGPResult`][ticktack.kalman.KalmanGPResult],
   [`kalman_gp_marginal`][ticktack.kalman.kalman_gp_marginal]) for Markov
   ``tinygp.kernels.quasisep`` Gaussian processes coupled to autonomous carbon
   box models across point, sub-annual seasonal, and IntCal20-style multi-year
   block-averaged observations with analytical Woodbury nuisance marginalization.
2. **Block-arrow linear-Gaussian solvers**
   ([`ArrowProblem`][ticktack.kalman.ArrowProblem],
   [`arrow_problem`][ticktack.kalman.arrow_problem],
   [`arrow_at`][ticktack.kalman.arrow_at]).
3. **Canonical data- and prior-orthogonal Wiener SVD modes**
   ([`CanonicalPODResult`][ticktack.kalman.CanonicalPODResult],
   [`canonical_pod_modes`][ticktack.kalman.canonical_pod_modes]).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
import tinygp.kernels as tgk
from jax.typing import ArrayLike

from ._utils import _concrete

if TYPE_CHECKING:
    from .linear import CausalGPProjection

__all__ = [
    "ArrowProblem",
    "CanonicalPODResult",
    "KalmanGPResult",
    "arrow_at",
    "arrow_problem",
    "canonical_pod_modes",
    "kalman_gp_marginal",
]


# =============================================================================
# 1. Block-Arrow Linear-Gaussian Marginalization
# =============================================================================


class ArrowProblem(NamedTuple):
    """Pre-reduced Gram sums of a block-arrow design ``Ũ = [U_local, S]``.

    Models marginal covariance ``s² I_N + Ũ Ũᵀ`` where ``U_local`` acts on ``B``
    disjoint observation blocks of up to ``R`` rows with ``M`` local modes per
    block and ``S`` spans all ``N`` observations with ``K`` shared modes.

    Attributes
    ----------
    x : jax.Array
        Noise-standardized data vector of shape ``(N,)`` (dimensionless).
    rows : jax.Array
        Integer row-index array of shape ``(B, R)`` after normalizing every
        inactive padding index (``< 0`` or ``>= N``) to the sentinel ``N``.
    local : jax.Array
        Masked block-local design tensor of shape ``(B, R, M)`` (dimensionless),
        with zeros on every inactive padded slot.
    spanning : jax.Array
        Shared spanning design matrix ``S`` of shape ``(N, K)`` (dimensionless).
    gram_bb : jax.Array
        Per-block local Gram matrices ``U_bᵀ U_b`` of shape ``(B, M, M)``.
    gram_bs : jax.Array
        Per-block local–spanning cross-Gram matrices ``U_bᵀ S_b`` of shape
        ``(B, M, K)``.
    gram_out : jax.Array
        Spanning Gram matrix ``S_outᵀ S_out`` of shape ``(K, K)`` summed over
        all rows not belonging to any block.
    proj_b : jax.Array
        Per-block data projections ``U_bᵀ x_b`` of shape ``(B, M)``.
    """

    x: jax.Array
    rows: jax.Array
    local: jax.Array
    spanning: jax.Array
    gram_bb: jax.Array
    gram_bs: jax.Array
    gram_out: jax.Array
    proj_b: jax.Array


def arrow_problem(
    x: ArrayLike,
    rows: ArrayLike,
    local: ArrayLike,
    spanning: ArrayLike | None = None,
) -> ArrowProblem:
    """Pre-reduce a block-arrow design ``Ũ = [U_local, S]`` for [`arrow_at`][ticktack.kalman.arrow_at].

    Parameters
    ----------
    x : array-like
        Noise-standardized residual vector of shape ``(N,)`` (dimensionless).
    rows : array-like of int
        Block membership indices of shape ``(B, R)``. Active entries must satisfy
        ``0 <= rows < N`` and be pairwise disjoint across all blocks. Entries
        outside ``[0, N)`` (such as ``-1`` or ``N``) are treated as inactive
        padding and normalized to sentinel ``N``.
    local : array-like
        Per-block local design factors of shape ``(B, R, M)`` (dimensionless).
        Slots corresponding to inactive indices in ``rows`` are zeroed.
    spanning : array-like, optional
        Shared design matrix ``S`` of shape ``(N, K)`` (dimensionless). When
        omitted, an empty ``(N, 0)`` matrix is used.

    Returns
    -------
    ArrowProblem
        Pre-reduced Gram sums and normalized indices ready for repeated
        evaluations of [`arrow_at`][ticktack.kalman.arrow_at].

    Raises
    ------
    ValueError
        If array ranks, shapes, or ``rows`` integer dtype are invalid, or if
        ``rows`` is concrete (not JAX-traced) and contains duplicate active
        indices in ``[0, N)``.
    """
    x_arr = jnp.asarray(x, dtype=jnp.float64)
    if x_arr.ndim != 1:
        raise ValueError(f"x must be a 1D array of shape (N,); got shape {x_arr.shape}.")
    rows_arr = jnp.asarray(rows)
    if rows_arr.ndim != 2 or not jnp.issubdtype(rows_arr.dtype, jnp.integer):
        raise ValueError(
            f"rows must be a 2D integer array of shape (B, R); "
            f"got shape {rows_arr.shape} and dtype {rows_arr.dtype}."
        )
    local_arr = jnp.asarray(local, dtype=jnp.float64)
    if local_arr.ndim != 3 or local_arr.shape[:2] != rows_arr.shape:
        raise ValueError(
            f"local must have shape ({rows_arr.shape[0]}, {rows_arr.shape[1]}, M); "
            f"got {local_arr.shape}."
        )
    n_obs = x_arr.shape[0]
    if spanning is None:
        span_arr = jnp.zeros((n_obs, 0), dtype=x_arr.dtype)
    else:
        span_arr = jnp.asarray(spanning, dtype=jnp.float64)
        if span_arr.ndim != 2 or span_arr.shape[0] != n_obs:
            raise ValueError(f"spanning must have shape ({n_obs}, K); got {span_arr.shape}.")
    valid = (rows_arr >= 0) & (rows_arr < n_obs)
    safe_rows = jnp.where(valid, rows_arr, n_obs)
    c_rows = _concrete(rows_arr)
    if c_rows is not None:
        active = c_rows[(c_rows >= 0) & (c_rows < n_obs)]
        if active.size != np.unique(active).size:
            raise ValueError(
                "arrow_problem requires disjoint active row indices across blocks; "
                "duplicate active indices found in rows."
            )
    local_masked = jnp.where(valid[..., None], local_arr, 0.0)
    hi = jax.lax.Precision.HIGHEST
    x_b = x_arr.at[safe_rows].get(mode="fill", fill_value=0.0)
    s_b = span_arr.at[safe_rows].get(mode="fill", fill_value=0.0)
    outside = jnp.ones(n_obs, dtype=x_arr.dtype).at[safe_rows].set(0.0, mode="drop")
    return ArrowProblem(
        x_arr,
        safe_rows,
        local_masked,
        span_arr,
        jnp.einsum("brm,brn->bmn", local_masked, local_masked, precision=hi),
        jnp.einsum("brm,brk->bmk", local_masked, s_b, precision=hi),
        jnp.matmul(span_arr.T * outside, span_arr, precision=hi),
        jnp.einsum("brm,br->bm", local_masked, x_b, precision=hi),
    )


def _solve_spd(chol: jax.Array, rhs: jax.Array) -> jax.Array:
    return jsl.cho_solve((chol, True), rhs)


def arrow_at(
    problem: ArrowProblem,
    scale2: ArrayLike,
    trace: bool = False,
) -> tuple[jax.Array, ...]:
    """Evaluate block-arrow Gaussian marginal quantities at variance scale ``s²``.

    Solves the ``P × P`` (``P = B M + K``) regularized normal equations
    ``(s² I_P + Ũᵀ Ũ) ŷ = Ũᵀ x`` via the block-arrow Schur complement plus one
    residual-space iterative refinement step.

    Parameters
    ----------
    problem : ArrowProblem
        Pre-reduced block-arrow problem returned by
        [`arrow_problem`][ticktack.kalman.arrow_problem].
    scale2 : float or jax.Array
        Strictly positive scalar variance scale ``s² > 0`` (dimensionless);
        cast to ``float64`` at entry.
    trace : bool, default False
        If ``True``, also return the parameter-space inverse trace ``tr``.

    Returns
    -------
    chi2 : jax.Array
        Scalar marginal quadratic form ``xᵀ (s² I_N + Ũ Ũᵀ)⁻¹ x``.
    norm : jax.Array
        Scalar half log-determinant ratio
        ``½ ln det(I_N + s⁻² Ũ Ũᵀ) = ½ (ln det(s² I_N + Ũ Ũᵀ) - N ln s²)``,
        with the ``½ N ln s²`` noise-scale term already subtracted.
    r2 : jax.Array
        Scalar squared residual norm ``‖x - Ũ ŷ‖²`` at the ridge solution ``ŷ``.
    tr : jax.Array, optional
        Returned only when ``trace=True``: scalar parameter-space trace
        ``tr((s² I_P + Ũᵀ Ũ)⁻¹)`` of size ``P × P`` (not the ``N × N``
        data-space covariance trace).

    Raises
    ------
    ValueError
        If ``scale2`` is not a scalar, or is concrete and non-finite or
        non-positive.
    """
    p = problem
    s2 = jnp.asarray(scale2, dtype=jnp.float64)
    if s2.ndim != 0:
        raise ValueError(f"scale2 must be a scalar; got shape {s2.shape}.")
    c_s2 = _concrete(s2)
    if c_s2 is not None and (not np.isfinite(c_s2) or float(c_s2) <= 0.0):
        raise ValueError(f"scale2 must be finite and strictly positive; got {c_s2}.")
    hi = jax.lax.Precision.HIGHEST
    n_block, n_mode = p.gram_bb.shape[:2]
    n_span = p.gram_out.shape[0]
    eye = jnp.eye(n_mode, dtype=p.gram_bb.dtype)
    chol_b = jnp.linalg.cholesky(p.gram_bb + s2 * eye)
    rhs = jnp.concatenate([p.proj_b[..., None], p.gram_bs], axis=-1)
    sol = jax.vmap(_solve_spd)(chol_b, rhs)
    m_g, m_gs = sol[..., 0], sol[..., 1:]
    eye_s = jnp.eye(n_span, dtype=p.gram_out.dtype)
    s_b = p.spanning.at[p.rows].get(mode="fill", fill_value=0.0)
    z = s_b - jnp.einsum("brm,bmk->brk", p.local, m_gs, precision=hi)
    schur = (
        s2 * eye_s
        + p.gram_out
        + jnp.einsum("brk,brl->kl", z, z, precision=hi)
        + s2 * jnp.einsum("bmk,bml->kl", m_gs, m_gs, precision=hi)
    )
    chol_s = jnp.linalg.cholesky(schur)
    fit_b = (
        jnp.zeros_like(p.x)
        .at[p.rows]
        .add(jnp.einsum("brm,bm->br", p.local, m_g, precision=hi), mode="drop")
    )
    rhs_s = jnp.matmul(p.spanning.T, p.x - fit_b, precision=hi)
    y_s = _solve_spd(chol_s, rhs_s)
    y_b = m_g - jnp.einsum("bmk,k->bm", m_gs, y_s, precision=hi)

    def residual(yb: jax.Array, ys: jax.Array) -> jax.Array:
        fit = (
            jnp.zeros_like(p.x)
            .at[p.rows]
            .add(jnp.einsum("brm,bm->br", p.local, yb, precision=hi), mode="drop")
        )
        return p.x - fit - jnp.matmul(p.spanning, ys, precision=hi)

    resid = residual(y_b, y_s)
    r_b = resid.at[p.rows].get(mode="fill", fill_value=0.0)
    g_b = jnp.einsum("brm,br->bm", p.local, r_b, precision=hi) - s2 * y_b
    g_s = jnp.matmul(p.spanning.T, resid, precision=hi) - s2 * y_s
    w_b = jax.vmap(_solve_spd)(chol_b, g_b[..., None])[..., 0]
    d_s = _solve_spd(chol_s, g_s - jnp.einsum("bmk,bm->k", p.gram_bs, w_b, precision=hi))
    y_s = y_s + d_s
    y_b = y_b + w_b - jnp.einsum("bmk,k->bm", m_gs, d_s, precision=hi)
    r2 = jnp.sum(residual(y_b, y_s) ** 2)
    chi2 = r2 / s2 + jnp.sum(y_b**2) + jnp.sum(y_s**2)
    log_diag = jnp.sum(jnp.log(jnp.diagonal(chol_b, axis1=1, axis2=2)))
    log_diag = log_diag + jnp.sum(jnp.log(jnp.diagonal(chol_s)))
    norm = log_diag - 0.5 * (n_block * n_mode + n_span) * jnp.log(s2)
    if not trace:
        return chi2, norm, r2
    inv_b = jax.vmap(lambda c: _solve_spd(c, eye))(chol_b)
    z_tr = jnp.einsum("bmk,bml->kl", m_gs, m_gs, precision=hi)
    tr = jnp.sum(jnp.diagonal(inv_b, axis1=1, axis2=2))
    tr = tr + jnp.trace(_solve_spd(chol_s, z_tr + eye_s))
    return chi2, norm, r2, tr


# =============================================================================
# 2. Canonical Data- and Prior-Orthogonal Wiener SVD Modes
# =============================================================================


class CanonicalPODResult(NamedTuple):
    """Simultaneously data-orthogonal and prior-orthonormal Wiener GP modes.

    Let ``N_d`` be the number of observations, ``r`` the number of spectral GP
    modes, ``m_sing = min(N_d, r)`` the number of observable singular modes,
    and ``N_eval`` the number of evaluation epochs ``t_eval``.

    Attributes
    ----------
    singular_values : jax.Array
        Descending observable singular values ``s_k`` of shape ``(m_sing,)``
        (dimensionless) satisfying
        ``G̃ = data_modes @ diag(singular_values) @ prior_rotation[:, :m_sing].T``.
    wiener_weights : jax.Array
        Wiener shrinkage factors of shape ``(r,)`` (dimensionless):
        ``s_k² / (1 + s_k²)`` for observable modes ``k < m_sing``, and ``0.0``
        for unobservable prior nullspace modes ``k >= m_sing`` when ``N_d < r``.
    data_modes : jax.Array
        Orthonormal left singular vectors ``U_data`` of shape ``(N_d, m_sing)``
        in whitened data space (``U_dataᵀ U_data = I_{m_sing}``).
    prior_rotation : jax.Array
        Orthogonal rotation matrix ``V_prior = [V_thin, V_null]`` of shape
        ``(r, r)`` in standardized prior-mode space (``V_priorᵀ V_prior = I_r``).
        When ``N_d < r``, the trailing ``r - m_sing`` nullspace columns are
        seeded via ``jax.lax.stop_gradient`` complete QR and projected
        orthogonal to live ``V_thin`` so ``V_priorᵀ V_prior = I_r`` holds
        through second-order autodiff without differentiating degenerate zero
        singular values.
    production_modes : jax.Array
        Canonical production basis ``Φ_eval V_prior`` of shape ``(N_eval, r)``
        in production-anomaly units (matching ``project_causal_gp``).
    coeff_mean : jax.Array
        Posterior mean of canonical prior-unit coefficients of shape ``(r,)``
        (dimensionless): ``(s_k / (1 + s_k²)) u_kᵀ r̃`` for ``k < m_sing``,
        and ``0.0`` for ``k >= m_sing``.
    coeff_var : jax.Array
        Posterior variance of canonical prior-unit coefficients of shape ``(r,)``
        (dimensionless): ``1 / (1 + s_k²)`` for ``k < m_sing``, and ``1.0``
        (unshrunk prior variance) for ``k >= m_sing``.
    posterior_mean : jax.Array
        Posterior mean production anomaly ``E[δq(t_eval) | d]`` of shape
        ``(N_eval,)`` in production-anomaly units.
    posterior_sd : jax.Array
        Pointwise posterior standard deviation ``sqrt(Var[δq(t_eval) | d])`` of
        shape ``(N_eval,)`` in production-anomaly units, computed from
        ``(singular_values, V_thin)`` and the two-pass orthogonal complement
        ``Φ_⊥`` as a non-negative sum of squares.
    d_eff : jax.Array
        Scalar effective degrees of freedom ``Σ_{k=1}^{m_sing} s_k² / (1 + s_k²)``
        (dimensionless).
    info_nats : jax.Array
        Scalar Shannon information gain ``½ Σ_{k=1}^{m_sing} ln(1 + s_k²)`` in
        nats.
    """

    singular_values: jax.Array
    wiener_weights: jax.Array
    data_modes: jax.Array
    prior_rotation: jax.Array
    production_modes: jax.Array
    coeff_mean: jax.Array
    coeff_var: jax.Array
    posterior_mean: jax.Array
    posterior_sd: jax.Array
    d_eff: jax.Array
    info_nats: jax.Array


def _wiener_pod_svd(
    g_tilde: jax.Array,
    r_tilde: jax.Array,
    phi_eval: jax.Array,
) -> CanonicalPODResult:
    """Compute thin-SVD Wiener POD shrinkage, DGKS variance, and nullspace basis."""
    r_modes = g_tilde.shape[1]
    u_data, s_vals, vt_thin = jnp.linalg.svd(g_tilde, full_matrices=False)
    v_thin = vt_thin.T
    m_sing = s_vals.shape[0]
    proj_thin = u_data.T @ r_tilde

    s_sq = s_vals**2
    inv_one_plus_s2 = 1.0 / (1.0 + s_sq)
    wiener_thin = s_sq * inv_one_plus_s2
    coeff_mean_thin = s_vals * inv_one_plus_s2 * proj_thin
    coeff_var_thin = inv_one_plus_s2

    phi_rot_thin = phi_eval @ v_thin
    phi_perp = phi_eval - phi_rot_thin @ vt_thin
    phi_perp = phi_perp - (phi_perp @ v_thin) @ vt_thin

    post_mean = phi_rot_thin @ coeff_mean_thin
    post_var = jnp.sum((phi_rot_thin**2) * coeff_var_thin[None, :], axis=1) + jnp.sum(
        phi_perp**2, axis=1
    )
    pos_var = post_var > 0.0
    post_sd = jnp.where(pos_var, jnp.sqrt(jnp.where(pos_var, post_var, 1.0)), 0.0)
    d_eff = jnp.sum(wiener_thin)
    info_nats = 0.5 * jnp.sum(jnp.log1p(s_sq))

    if m_sing < r_modes:
        q_full, _ = jnp.linalg.qr(jax.lax.stop_gradient(v_thin), mode="complete")
        v_null0 = q_full[:, m_sing:]
        proj_null = v_thin.T @ v_null0
        v_null = v_null0 - v_thin @ proj_null
        v_null = v_null + 0.5 * (v_null @ proj_null.T) @ proj_null
        v_prior = jnp.concatenate([v_thin, v_null], axis=1)
        n_null = r_modes - m_sing
        zeros_null = jnp.zeros((n_null,), dtype=jnp.float64)
        wiener = jnp.concatenate([wiener_thin, zeros_null])
        coeff_mean = jnp.concatenate([coeff_mean_thin, zeros_null])
        coeff_var = jnp.concatenate([coeff_var_thin, jnp.ones((n_null,), dtype=jnp.float64)])
    else:
        v_prior = v_thin
        wiener = wiener_thin
        coeff_mean = coeff_mean_thin
        coeff_var = coeff_var_thin

    prod_modes = phi_eval @ v_prior
    return CanonicalPODResult(
        singular_values=s_vals,
        wiener_weights=wiener,
        data_modes=u_data,
        prior_rotation=v_prior,
        production_modes=prod_modes,
        coeff_mean=coeff_mean,
        coeff_var=coeff_var,
        posterior_mean=post_mean,
        posterior_sd=post_sd,
        d_eff=d_eff,
        info_nats=info_nats,
    )


def _validate_resid_and_sigma(
    resid: ArrayLike,
    sigma: ArrayLike,
    n_obs: int,
) -> tuple[jax.Array, jax.Array]:
    """Validate and cast ``(resid, sigma)`` to ``(n_obs,)`` ``float64`` arrays."""
    resid_arr = jnp.asarray(resid, dtype=jnp.float64).reshape(-1)
    if resid_arr.shape[0] != n_obs:
        raise ValueError(
            f"resid length {resid_arr.shape[0]} does not match observation count {n_obs}."
        )
    sd_raw = jnp.asarray(sigma, dtype=jnp.float64)
    if sd_raw.ndim > 1:
        raise ValueError(
            f"sigma must be a scalar or 1D array of length {n_obs}; got shape {sd_raw.shape}."
        )
    sd_arr = jnp.broadcast_to(sd_raw, (n_obs,))
    c_sd = _concrete(sd_arr)
    if c_sd is not None and not np.all(np.isfinite(c_sd) & (c_sd > 0.0)):
        raise ValueError("sigma must be finite and strictly positive.")
    return resid_arr, sd_arr


def canonical_pod_modes(
    projection: CausalGPProjection,
    t_eval: ArrayLike,
    resid: ArrayLike,
    sigma: ArrayLike,
    *,
    nuisance_whitened_design: ArrayLike | None = None,
    include_initial_cov: bool = True,
) -> CanonicalPODResult:
    """Extract canonical Wiener-shrunk production modes via thin SVD.

    Applies a single augmented-QR whitening pass against initial-state and
    optional linear nuisance covariance, then takes the unpadded thin SVD
    ``G̃ = U_data diag(s) V_thinᵀ`` of the whitened GP operator
    ``G̃ = D̃⁻¹/² W_causal Ψ_r Λ¹/²`` so observable canonical coefficients
    ``c̃_k`` have decoupled 1D posteriors
    ``N(s_k / (1 + s_k²) u_kᵀ r̃, 1 / (1 + s_k²))``, matching
    [`CausalGPProjection.posterior_production`][ticktack.linear.CausalGPProjection.posterior_production]
    to ``rtol=1e-10`` (verified in ``test_dec51a_pod_arrow.py``).

    Parameters
    ----------
    projection : CausalGPProjection
        Causal GP projection built with ``kernel=...`` (so ``spectral_weights``
        is populated) for ``N_d`` observations and ``r`` spectral modes.
    t_eval : array-like
        1D evaluation timestamps of shape ``(N_eval,)`` in years within
        ``[projection.a, projection.b_bound]``.
    resid : array-like
        Observation residuals ``d - m_0`` of shape ``(N_d,)`` in data units
        (typically ``‰`` for Δ¹⁴C).
    sigma : float or array-like
        Strictly positive finite measurement standard deviations (scalar or 1D
        array of shape ``(N_d,)``, same units as ``resid``).
    nuisance_whitened_design : array-like, optional
        Prior-scaled linear nuisance design ``X_nuis L_nuis`` of shape
        ``(N_d, K_nuis)`` in data units (before division by ``sigma``).
    include_initial_cov : bool, default True
        If ``True`` and ``projection.anchor_cov_root`` is present, include the
        propagated pre-history stationary state covariance in the nuisance
        whitening pass.

    Returns
    -------
    CanonicalPODResult
        Canonical SVD modes, Wiener weights, posterior production summary on
        ``t_eval``, effective degrees of freedom, and information gain.

    Raises
    ------
    ValueError
        If ``projection.spectral_weights`` is ``None``, ``sigma`` is not a
        strictly positive finite scalar/1D array, or
        ``nuisance_whitened_design`` is not 2D with ``N_d`` rows.
    """
    if projection.spectral_weights is None:
        raise ValueError("projection must be constructed with kernel=... for canonical_pod_modes.")

    n_obs, r_modes = projection.causal_basis_design.shape
    resid_arr, sd = _validate_resid_and_sigma(resid, sigma, n_obs)

    spec_root = jnp.sqrt(projection.spectral_weights)
    g_raw = (projection.causal_basis_design * spec_root[None, :]) / sd[:, None]
    r_raw = resid_arr / sd

    nuis_parts = []
    if include_initial_cov and projection.anchor_cov_root is not None:
        nuis_parts.append((projection.hom_design_anchor @ projection.anchor_cov_root) / sd[:, None])
    if nuisance_whitened_design is not None:
        nuis_arr = jnp.asarray(nuisance_whitened_design, dtype=jnp.float64)
        if nuis_arr.ndim != 2 or nuis_arr.shape[0] != n_obs:
            raise ValueError(
                f"nuisance_whitened_design must have shape ({n_obs}, k); got {nuis_arr.shape}."
            )
        nuis_parts.append(nuis_arr / sd[:, None])

    if nuis_parts:
        u_nuis = jnp.concatenate(nuis_parts, axis=1)
        combined = jnp.concatenate([r_raw[:, None], g_raw], axis=1)
        whitened, _ = whiten_qr(combined, u_nuis)
        r_tilde, g_tilde = whitened[:, 0], whitened[:, 1:]
    else:
        r_tilde, g_tilde = r_raw, g_raw

    basis_eval = solin_sarkka_basis(t_eval, projection.a, projection.b_bound, r_modes)
    phi_eval = basis_eval.psi * spec_root[None, :]
    return _wiener_pod_svd(g_tilde, r_tilde, phi_eval)


# =============================================================================
# 3. Exact Continuous-Time State-Space Kalman Filter & Adjoint Smoother
# =============================================================================


class KalmanGPResult(NamedTuple):
    """Marginal log-likelihood and smoothed state-space GP + CBM posterior.

    Attributes
    ----------
    loglike : jax.Array
        Scalar Gaussian marginal log-likelihood ``ln p(d | m_0, φ)`` in nats,
        integrating over the stationary CBM + Markov GP state trajectory and
        any optional linear nuisance parameters.
    t_grid : jax.Array
        Uniform state-space evaluation grid of shape ``(N_grid,)`` in years
        (``yr``).
    production_mean : jax.Array
        Smoothed posterior mean production anomaly ``E[δq(t_k) | d]`` of shape
        ``(N_grid,)`` in kernel amplitude units (e.g. ``atoms cm^-2 s^-1`` or
        steady-state ``q_0`` units).
    production_var : jax.Array
        Smoothed pointwise posterior production variance ``Var[δq(t_k) | d]``
        of shape ``(N_grid,)`` in squared production units (clamped ``>= 0``).
    cbm_state_mean : jax.Array
        Smoothed posterior mean reservoir ``¹⁴C`` inventory anomaly
        ``E[δy(t_k) | d]`` of shape ``(N_grid, n_boxes)`` in CBM inventory
        units (``alpha_unit * b * q``).
    cbm_state_var : jax.Array
        Smoothed pointwise posterior reservoir variance ``Var[δy_c(t_k) | d]``
        of shape ``(N_grid, n_boxes)`` in squared inventory units.
    nuisance_mean : jax.Array or None
        Posterior mean ``E[β | d]`` of linear nuisance parameters of shape
        ``(p_nuis,)`` when ``nuisance_design`` is supplied, else ``None``.
    nuisance_cov : jax.Array or None
        Posterior covariance ``Cov[β | d]`` of shape ``(p_nuis, p_nuis)`` when
        ``nuisance_design`` is supplied, else ``None``.
    """

    loglike: jax.Array
    t_grid: jax.Array
    production_mean: jax.Array
    production_var: jax.Array
    cbm_state_mean: jax.Array
    cbm_state_var: jax.Array
    nuisance_mean: jax.Array | None
    nuisance_cov: jax.Array | None


class _TrackAssignment(NamedTuple):
    """Grid-index window bounds and greedy interval-colouring track indices.

    Attributes
    ----------
    obs_s_min : np.ndarray
        Earliest sub-window start grid index per observation, shape ``(N_d,)``.
    obs_e_max : np.ndarray
        Latest sub-window end grid index per observation, shape ``(N_d,)``.
    obs_track : np.ndarray
        Assigned accumulator track index in ``[0, n_tracks)`` for positive-width
        windows, or ``-1`` for instantaneous point observations, shape ``(N_d,)``.
    sub_bounds : list[list[tuple[int, int, float]]]
        Per-observation list of ``(k_start, k_end, weight)`` sub-window tuples.
    n_tracks : int
        Minimal number of concurrent accumulator tracks required across ``t_grid``.
    """

    obs_s_min: np.ndarray
    obs_e_max: np.ndarray
    obs_track: np.ndarray
    sub_bounds: list[list[tuple[int, int, float]]]
    n_tracks: int


class _KalmanSchedule(NamedTuple):
    """Static step-accumulator schedule and per-node observation design tensors.

    Attributes
    ----------
    n_tracks : int
        Number of concurrent integral-accumulator state slots (``>= 0``).
    max_simul : int
        Maximum simultaneous scalar measurement updates at any grid node (``>= 1``).
    keep_mask : jax.Array
        Accumulator retention flags of shape ``(N_steps, n_tracks)`` (``0.0``
        at window start ``k == s_lo``, ``1.0`` on continuation steps).
    w_step : jax.Array
        Step integration weights of shape ``(N_steps, n_tracks, n_boxes)`` in
        ``‰ yr^-1`` per unit inventory.
    node_obs_idx : jax.Array
        Observation row indices at each grid node, shape ``(N_grid, max_simul)``.
    node_obs_active : jax.Array
        Boolean active-slot mask at each grid node, shape ``(N_grid, max_simul)``.
    node_h_full : jax.Array
        State-space measurement vectors of shape ``(N_grid, max_simul, d_state)``.
    """

    n_tracks: int
    max_simul: int
    keep_mask: jax.Array
    w_step: jax.Array
    node_obs_idx: jax.Array
    node_obs_active: jax.Array
    node_h_full: jax.Array


class _KalmanStepDynamics(NamedTuple):
    """Continuous-time Van Loan step transition and step-integral covariances.

    Attributes
    ----------
    phi_xi : jax.Array
        Joint ``(CBM + GP)`` state transition ``exp(F_joint Δt)`` of shape
        ``(d_xi, d_xi)``.
    sig_xi : jax.Array
        Step state process covariance ``Σ_ξ(Δt)`` of shape ``(d_xi, d_xi)``.
    phi_s_xi : jax.Array
        Step box-integral coupling ``∫_0^{Δt} [I_n, 0] exp(F_joint τ) dτ`` of
        shape ``(n_boxes, d_xi)`` in ``yr``.
    sig_ss : jax.Array
        Step box-integral covariance ``Σ_{ss}(Δt)`` of shape ``(n_boxes, n_boxes)``.
    sig_s_xi : jax.Array
        Step integral–state cross-covariance ``Σ_{sξ}(Δt)`` of shape
        ``(n_boxes, d_xi)``.
    """

    phi_xi: jax.Array
    sig_xi: jax.Array
    phi_s_xi: jax.Array
    sig_ss: jax.Array
    sig_s_xi: jax.Array


class _KalmanNuisanceSpec(NamedTuple):
    """Validated linear nuisance prior and mean-shifted observation residuals.

    Attributes
    ----------
    p_nuis : int
        Number of linear nuisance parameters (``0`` when ``nuisance_design`` is
        omitted).
    resid_shifted : jax.Array
        Prior-mean-shifted residual vector ``resid - X_nuis μ_nuis`` of shape
        ``(N_d,)`` in data units.
    nuis_u_raw : jax.Array
        Prior-scaled nuisance design ``X_nuis L_nuis`` of shape
        ``(N_d, p_nuis)`` in data units.
    prior_mean : jax.Array
        Nuisance prior mean ``μ_nuis`` of shape ``(p_nuis,)``.
    prior_root : jax.Array
        Nuisance prior square-root factor ``L_nuis`` of shape
        ``(p_nuis, p_nuis)`` satisfying ``Σ_β = L_nuis L_nuisᵀ``.
    """

    p_nuis: int
    resid_shifted: jax.Array
    nuis_u_raw: jax.Array
    prior_mean: jax.Array
    prior_root: jax.Array


class _JosephUpdateResult(NamedTuple):
    """Result of one stabilized scalar Joseph-form Kalman measurement update.

    Attributes
    ----------
    mean : jax.Array
        Updated state mean of shape ``(d_state,)``.
    cov : jax.Array
        Updated state covariance of shape ``(d_state, d_state)``.
    delta_log_norm : jax.Array
        Scalar Gaussian log-normalizer increment ``-½ ln(2π s)`` in nats
        (``0.0`` when inactive).
    std_innov : jax.Array
        Standardized scalar innovation ``v / sqrt(s)`` (``0.0`` when inactive).
    std_extra_row : jax.Array
        Standardized nuisance innovation row of shape ``(p_nuis,)``.
    m_extra : jax.Array
        Updated state nuisance sensitivity matrix of shape ``(d_state, p_nuis)``.
    k_gain : jax.Array
        Kalman gain vector ``P h / s`` of shape ``(d_state,)`` (``0.0`` when
        inactive).
    inv_sqrt_s : jax.Array
        Inverse innovation standard deviation ``1 / sqrt(s)`` (``0.0`` when
        inactive).
    """

    mean: jax.Array
    cov: jax.Array
    delta_log_norm: jax.Array
    std_innov: jax.Array
    std_extra_row: jax.Array
    m_extra: jax.Array
    k_gain: jax.Array
    inv_sqrt_s: jax.Array


class _KalmanForwardPass(NamedTuple):
    """Filtered trajectories and scalar innovation factors from the forward scan.

    Attributes
    ----------
    log_norm : jax.Array
        Accumulated Gaussian log-normalizer ``Σ_j -½ ln(2π s_j)`` in nats.
    m_filt_all : jax.Array
        Filtered state means ``m_{k|k}`` of shape ``(N_grid, d_state)``.
    p_filt_all : jax.Array
        Filtered state covariances ``P_{k|k}`` of shape
        ``(N_grid, d_state, d_state)``.
    mx_filt_all : jax.Array
        Filtered nuisance sensitivity matrices ``M_{x, k|k}`` of shape
        ``(N_grid, d_state, p_nuis)``.
    phi_seq : jax.Array
        Step transition matrices ``Φ_k`` of shape ``(N_steps, d_state, d_state)``.
    sin_all : jax.Array
        Standardized scalar innovations of shape ``(N_grid, max_simul)``.
    sex_all : jax.Array
        Standardized nuisance innovation rows of shape
        ``(N_grid, max_simul, p_nuis)``.
    kg_all : jax.Array
        Per-slot Kalman gains of shape ``(N_grid, max_simul, d_state)``.
    inv_sqs_all : jax.Array
        Per-slot inverse innovation SDs of shape ``(N_grid, max_simul)``.
    """

    log_norm: jax.Array
    m_filt_all: jax.Array
    p_filt_all: jax.Array
    mx_filt_all: jax.Array
    phi_seq: jax.Array
    sin_all: jax.Array
    sex_all: jax.Array
    kg_all: jax.Array
    inv_sqs_all: jax.Array


class _KalmanMarginalizedNuisance(NamedTuple):
    """Woodbury-marginalized log-likelihood, nuisance posterior, and adjusted means.

    Attributes
    ----------
    loglike : jax.Array
        Marginal log-likelihood ``ln p(d | m_0, φ)`` in nats.
    nuisance_mean : jax.Array or None
        Posterior nuisance mean ``E[β | d]`` of shape ``(p_nuis,)``, or ``None``.
    nuisance_cov : jax.Array or None
        Posterior nuisance covariance ``Cov[β | d]`` of shape
        ``(p_nuis, p_nuis)``, or ``None``.
    cov_omega_nuis : jax.Array
        Standardized nuisance posterior covariance ``(I + Ũᵀ Ũ)⁻¹`` of shape
        ``(p_nuis, p_nuis)``.
    m_filt_adj : jax.Array
        Nuisance-adjusted filtered means ``m_{k|k} - M_{x, k|k} ω̂`` of shape
        ``(N_grid, d_state)``.
    sin_adj_all : jax.Array
        Nuisance-adjusted standardized innovations of shape
        ``(N_grid, max_simul)``.
    """

    loglike: jax.Array
    nuisance_mean: jax.Array | None
    nuisance_cov: jax.Array | None
    cov_omega_nuis: jax.Array
    m_filt_adj: jax.Array
    sin_adj_all: jax.Array


def _validate_cbm_system(
    A: ArrayLike,
    b: ArrayLike,
    alpha_unit: ArrayLike,
) -> tuple[jax.Array, jax.Array, int]:
    """Validate CBM rate matrix ``A`` and branching vector ``b`` at entry."""
    a_mat = jnp.asarray(A, dtype=jnp.float64)
    if a_mat.ndim != 2 or a_mat.shape[0] != a_mat.shape[1] or a_mat.shape[0] < 1:
        raise ValueError(
            f"A must be a non-empty 2D square matrix (n_boxes, n_boxes); got shape {a_mat.shape}."
        )
    n_boxes = a_mat.shape[0]
    b_arr = jnp.asarray(b, dtype=jnp.float64)
    if b_arr.ndim != 1 or b_arr.shape[0] != n_boxes:
        raise ValueError(f"b must be a 1D array of length {n_boxes}; got shape {b_arr.shape}.")
    return a_mat, jnp.asarray(alpha_unit, dtype=jnp.float64) * b_arr, n_boxes


def _normalize_concrete_windows(
    t_starts: ArrayLike,
    t_ends: ArrayLike | None,
    sub_weights: ArrayLike | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalize concrete observation windows to ``(N_d, M_sub)`` NumPy arrays."""
    c_s = _concrete(t_starts)
    c_e = c_s if t_ends is None else _concrete(t_ends)
    c_w = np.asarray(1.0) if sub_weights is None else _concrete(sub_weights)
    if c_s is None or c_e is None or c_w is None:
        raise ValueError(
            "Observation window timestamps (t_starts, t_ends) and sub_weights "
            "must be concrete arrays to build the static Kalman schedule."
        )
    starts = np.asarray(c_s, dtype=np.float64)
    ends = np.asarray(c_e, dtype=np.float64)
    if starts.ndim == 1 and ends.ndim == 1 and starts.shape == ends.shape:
        starts = starts[:, None]
        ends = ends[:, None]
    elif starts.ndim != 2 or ends.shape != starts.shape:
        raise ValueError(
            "t_starts and t_ends must be matching 1D (N_d,) or 2D (N_d, M_sub) arrays; "
            f"got {starts.shape} and {ends.shape}."
        )
    n_obs, m_sub = starts.shape
    if n_obs < 1:
        raise ValueError("At least one observation window is required.")
    if m_sub < 1:
        raise ValueError("Observation windows must have at least 1 sub-window column (m_sub >= 1).")
    if not (np.all(np.isfinite(starts)) and np.all(np.isfinite(ends))):
        raise ValueError("Observation window timestamps (t_starts, t_ends) must be finite.")
    if np.any(ends < starts):
        raise ValueError("Seasonal windows require t_ends >= t_starts.")

    if sub_weights is None:
        weights = np.full((n_obs, m_sub), 1.0 / m_sub, dtype=np.float64)
    else:
        w_arr = np.asarray(c_w, dtype=np.float64)
        if not np.all(np.isfinite(w_arr)):
            raise ValueError("sub_weights must contain finite values.")
        if np.any(w_arr < 0.0):
            raise ValueError("sub_weights must be non-negative.")
        try:
            w_raw = np.broadcast_to(w_arr, (n_obs, m_sub))
        except ValueError as exc:
            raise ValueError(
                f"sub_weights of shape {w_arr.shape} cannot broadcast to ({n_obs}, {m_sub})."
            ) from exc
        w_sum = np.sum(w_raw, axis=-1, keepdims=True)
        if np.any(w_sum <= 0.0):
            raise ValueError(
                "Each observation row in sub_weights must have strictly positive total weight."
            )
        weights = w_raw / w_sum
    return starts, ends, weights


def _resolve_kalman_time_grid(
    t_grid: ArrayLike | None,
    c_starts: np.ndarray,
    c_ends: np.ndarray,
    dt: float | None,
    *,
    c_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Build or validate a concrete uniform step grid ``(t_grid_np, step_dt)``."""
    if c_starts.size == 0:
        raise ValueError("At least one observation window is required.")
    if not (np.all(np.isfinite(c_starts)) and np.all(np.isfinite(c_ends))):
        raise ValueError("Observation window timestamps (t_starts, t_ends) must be finite.")
    if np.any(c_ends < c_starts):
        raise ValueError("Seasonal windows require t_ends >= t_starts.")

    if t_grid is None:
        if dt is None:
            step_dt = 1.0 / 24.0
        else:
            c_dt = _concrete(dt)
            if c_dt is None or c_dt.ndim != 0:
                raise ValueError("dt must be a concrete scalar float.")
            step_dt = float(c_dt)
        if not math.isfinite(step_dt) or step_dt <= 0.0:
            raise ValueError(f"dt must be finite and strictly positive; got {dt}.")
        if c_weights is not None:
            active = c_weights > 0.0
            t_min = float(np.min(c_starts[active]))
            t_max = float(np.max(c_ends[active]))
        else:
            t_min = float(np.min(c_starts))
            t_max = float(np.max(c_ends))
        n_steps = max(round((t_max - t_min) / step_dt), 1)
        t_grid_np = t_min + step_dt * np.arange(n_steps + 1, dtype=np.float64)
        return t_grid_np, step_dt

    c_grid = _concrete(t_grid)
    if c_grid is None:
        raise ValueError("t_grid must be a concrete array to build the Kalman schedule.")
    t_grid_np = np.asarray(c_grid, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(t_grid_np)):
        raise ValueError("t_grid must contain finite values.")
    if t_grid_np.size < 2:
        raise ValueError("t_grid must contain at least 2 points.")
    step_dt = float((t_grid_np[-1] - t_grid_np[0]) / (t_grid_np.size - 1))
    if step_dt <= 0.0:
        raise ValueError(f"t_grid must be strictly increasing (step_dt > 0); got {step_dt}.")
    expected_grid = t_grid_np[0] + step_dt * np.arange(t_grid_np.size, dtype=np.float64)
    if not np.allclose(t_grid_np, expected_grid, rtol=0.0, atol=1e-9):
        raise ValueError("t_grid must be uniformly spaced.")
    return t_grid_np, step_dt


def _assign_kalman_tracks(
    t_grid_np: np.ndarray,
    starts_np: np.ndarray,
    ends_np: np.ndarray,
    weights_np: np.ndarray,
    step_dt: float | None = None,
) -> _TrackAssignment:
    """Map active sub-windows to ``t_grid_np`` indices and greedily colour interval tracks."""
    n_grid = t_grid_np.shape[0]
    dt = float((t_grid_np[-1] - t_grid_np[0]) / (n_grid - 1)) if step_dt is None else float(step_dt)
    t0 = float(t_grid_np[0])
    n_obs, m_sub = starts_np.shape

    def _to_idx(val: float) -> int:
        k_float = (val - t0) / dt
        k_int = round(k_float)
        if abs(k_float - k_int) > 1e-7 or k_int < 0 or k_int >= n_grid:
            raise ValueError(
                f"Observation boundary {val} is not aligned to uniform t_grid "
                f"(t0={t0}, dt={dt}, t_end={float(t_grid_np[-1])})."
            )
        return k_int

    obs_s_min = np.zeros(n_obs, dtype=int)
    obs_e_max = np.zeros(n_obs, dtype=int)
    sub_bounds: list[list[tuple[int, int, float]]] = []
    for i in range(n_obs):
        pairs: list[tuple[int, int, float]] = []
        s_lo = n_grid
        e_hi = 0
        for m in range(m_sub):
            w_m = float(weights_np[i, m])
            if w_m == 0.0:
                continue
            ks = _to_idx(float(starts_np[i, m]))
            ke = _to_idx(float(ends_np[i, m]))
            s_lo = min(s_lo, ks)
            e_hi = max(e_hi, ke)
            pairs.append((ks, ke, w_m))
        obs_s_min[i] = s_lo
        obs_e_max[i] = e_hi
        sub_bounds.append(pairs)

    order = np.argsort(obs_s_min, kind="stable")
    track_free_step: list[int] = []
    obs_track = np.full(n_obs, -1, dtype=int)
    for idx in order:
        i = int(idx)
        if obs_s_min[i] == obs_e_max[i]:
            continue
        for tr, free_at in enumerate(track_free_step):
            if free_at <= obs_s_min[i]:
                obs_track[i] = tr
                track_free_step[tr] = int(obs_e_max[i])
                break
        else:
            obs_track[i] = len(track_free_step)
            track_free_step.append(int(obs_e_max[i]))

    return _TrackAssignment(
        obs_s_min=obs_s_min,
        obs_e_max=obs_e_max,
        obs_track=obs_track,
        sub_bounds=sub_bounds,
        n_tracks=len(track_free_step),
    )


def _build_seasonal_kalman_schedule(
    t_grid_np: np.ndarray,
    starts_np: np.ndarray,
    ends_np: np.ndarray,
    weights_np: np.ndarray,
    c_obs: ArrayLike,
    step_dt: float | None = None,
    *,
    d_gp: int = 0,
) -> _KalmanSchedule:
    """Assemble accumulator weights and per-node observation design tensors."""
    c_obs_jax = jnp.asarray(c_obs, dtype=jnp.float64)
    n_grid = t_grid_np.shape[0]
    n_steps = n_grid - 1
    dt = float((t_grid_np[-1] - t_grid_np[0]) / n_steps) if step_dt is None else float(step_dt)
    tracks = _assign_kalman_tracks(t_grid_np, starts_np, ends_np, weights_np, step_dt=dt)
    n_obs, n_boxes = c_obs_jax.shape
    n_tracks = tracks.n_tracks

    keep_mask = np.zeros((n_steps, n_tracks), dtype=np.float64)
    step_rate = np.zeros((n_steps, n_tracks), dtype=np.float64)
    step_obs_idx = np.zeros((n_steps, n_tracks), dtype=np.int32)
    for i in range(n_obs):
        tr = int(tracks.obs_track[i])
        if tr < 0:
            continue
        s_lo = int(tracks.obs_s_min[i])
        e_hi = int(tracks.obs_e_max[i])
        step_obs_idx[s_lo:e_hi, tr] = i
        if e_hi > s_lo + 1:
            keep_mask[s_lo + 1 : e_hi, tr] = 1.0
        for ks, ke, w_m in tracks.sub_bounds[i]:
            if ke <= ks:
                raise ValueError(
                    "Mixed zero-width and positive-width sub-windows in one ring are not supported."
                )
            step_rate[ks:ke, tr] += w_m / ((ke - ks) * dt)

    w_step = (
        jnp.asarray(step_rate, dtype=jnp.float64)[:, :, None]
        * c_obs_jax[jnp.asarray(step_obs_idx, dtype=jnp.int32)]
    )

    events_at_node: list[list[int]] = [[] for _ in range(n_grid)]
    for i in range(n_obs):
        events_at_node[int(tracks.obs_e_max[i])].append(i)
    max_simul = max(max((len(ev) for ev in events_at_node), default=1), 1)

    node_obs_idx = np.zeros((n_grid, max_simul), dtype=np.int32)
    node_obs_active = np.zeros((n_grid, max_simul), dtype=bool)
    node_h_tracks = np.zeros((n_grid, max_simul, n_tracks), dtype=np.float64)
    node_is_point = np.zeros((n_grid, max_simul), dtype=np.float64)
    for k_node, ev in enumerate(events_at_node):
        for slot, i in enumerate(ev):
            node_obs_idx[k_node, slot] = i
            node_obs_active[k_node, slot] = True
            tr = int(tracks.obs_track[i])
            if tr >= 0:
                node_h_tracks[k_node, slot, tr] = 1.0
            else:
                node_is_point[k_node, slot] = 1.0

    idx_jax = jnp.asarray(node_obs_idx, dtype=jnp.int32)
    h_boxes = jnp.asarray(node_is_point, dtype=jnp.float64)[:, :, None] * c_obs_jax[idx_jax]
    node_h_full = jnp.concatenate(
        [
            jnp.asarray(node_h_tracks, dtype=jnp.float64),
            h_boxes,
            jnp.zeros((n_grid, max_simul, d_gp), dtype=jnp.float64),
        ],
        axis=-1,
    )
    return _KalmanSchedule(
        n_tracks=n_tracks,
        max_simul=max_simul,
        keep_mask=jnp.asarray(keep_mask, dtype=jnp.float64),
        w_step=w_step,
        node_obs_idx=idx_jax,
        node_obs_active=jnp.asarray(node_obs_active, dtype=bool),
        node_h_full=node_h_full,
    )


def _prepare_kalman_nuisance(
    resid_arr: jax.Array,
    nuisance_design: ArrayLike | None,
    nuisance_prior_mean: ArrayLike,
    nuisance_prior_sd: ArrayLike | None,
    nuisance_prior_root: ArrayLike | None,
) -> _KalmanNuisanceSpec:
    """Validate optional linear nuisance design/prior and shift ``resid_arr``."""
    n_obs = resid_arr.shape[0]
    if nuisance_design is None:
        if nuisance_prior_sd is not None or nuisance_prior_root is not None:
            raise ValueError("nuisance_prior_sd and nuisance_prior_root require nuisance_design.")
        return _KalmanNuisanceSpec(
            p_nuis=0,
            resid_shifted=resid_arr,
            nuis_u_raw=jnp.zeros((n_obs, 0), dtype=jnp.float64),
            prior_mean=jnp.zeros((0,), dtype=jnp.float64),
            prior_root=jnp.zeros((0, 0), dtype=jnp.float64),
        )

    nuis_x = jnp.asarray(nuisance_design, dtype=jnp.float64)
    if nuis_x.ndim != 2 or nuis_x.shape[0] != n_obs:
        raise ValueError(
            f"nuisance_design must be a 2D array of shape ({n_obs}, p_nuis); "
            f"got shape {nuis_x.shape}."
        )
    p_nuis = nuis_x.shape[1]
    if (nuisance_prior_sd is None) == (nuisance_prior_root is None):
        raise ValueError("Specify exactly one of nuisance_prior_sd or nuisance_prior_root.")

    if nuisance_prior_sd is not None:
        sd_raw = jnp.asarray(nuisance_prior_sd, dtype=jnp.float64)
        if sd_raw.ndim > 1:
            raise ValueError(
                f"nuisance_prior_sd must be a scalar or 1D array of length {p_nuis}; "
                f"got shape {sd_raw.shape}."
            )
        sd_vec = jnp.broadcast_to(sd_raw, (p_nuis,))
        c_sd = _concrete(sd_vec)
        if c_sd is not None and not np.all(np.isfinite(c_sd) & (c_sd > 0.0)):
            raise ValueError("nuisance_prior_sd must be finite and strictly positive.")
        nuis_root = jnp.diag(sd_vec)
    else:
        nuis_root = jnp.asarray(nuisance_prior_root, dtype=jnp.float64)
        if nuis_root.ndim != 2 or nuis_root.shape != (p_nuis, p_nuis):
            raise ValueError(
                f"nuisance_prior_root must have shape ({p_nuis}, {p_nuis}); got {nuis_root.shape}."
            )
        c_root = _concrete(nuis_root)
        if c_root is not None and not np.all(np.isfinite(c_root)):
            raise ValueError("nuisance_prior_root must contain finite entries.")

    pm_raw = jnp.asarray(nuisance_prior_mean, dtype=jnp.float64)
    if pm_raw.ndim > 1:
        raise ValueError(
            f"nuisance_prior_mean must be a scalar or 1D array of length {p_nuis}; "
            f"got shape {pm_raw.shape}."
        )
    nuis_mean = jnp.broadcast_to(pm_raw, (p_nuis,))
    c_pm = _concrete(nuis_mean)
    if c_pm is not None and not np.all(np.isfinite(c_pm)):
        raise ValueError("nuisance_prior_mean must be finite.")

    return _KalmanNuisanceSpec(
        p_nuis=p_nuis,
        resid_shifted=resid_arr - nuis_x @ nuis_mean,
        nuis_u_raw=nuis_x @ nuis_root,
        prior_mean=nuis_mean,
        prior_root=nuis_root,
    )


def _van_loan_kalman_step_dynamics(
    a_mat: jax.Array,
    b_eff: jax.Array,
    f_gp: jax.Array,
    q_sde: jax.Array,
    h_gp: jax.Array,
    step_dt: float,
    *,
    has_tracks: bool,
) -> _KalmanStepDynamics:
    """Compute exact continuous-time Van Loan step transition and step-integral covariances."""
    n_boxes = a_mat.shape[0]
    d_gp = f_gp.shape[0]
    d_xi = n_boxes + d_gp
    f_joint = jnp.block(
        [
            [a_mat, jnp.outer(b_eff, h_gp)],
            [jnp.zeros((d_gp, n_boxes), dtype=jnp.float64), f_gp],
        ]
    )
    q_joint = jsl.block_diag(jnp.zeros((n_boxes, n_boxes), dtype=jnp.float64), q_sde)

    if not has_tracks:
        m_step_vl = (
            jnp.block(
                [
                    [f_joint, q_joint],
                    [jnp.zeros((d_xi, d_xi), dtype=jnp.float64), -f_joint.T],
                ]
            )
            * step_dt
        )
        e_step = jsl.expm(m_step_vl)
        phi_xi = e_step[:d_xi, :d_xi]
        sig_raw = e_step[:d_xi, d_xi:] @ phi_xi.T
        sig_xi = 0.5 * (sig_raw + sig_raw.T)
        return _KalmanStepDynamics(
            phi_xi=phi_xi,
            sig_xi=sig_xi,
            phi_s_xi=jnp.zeros((n_boxes, d_xi), dtype=jnp.float64),
            sig_ss=jnp.zeros((n_boxes, n_boxes), dtype=jnp.float64),
            sig_s_xi=jnp.zeros((n_boxes, d_xi), dtype=jnp.float64),
        )

    d_aug = n_boxes + d_xi
    c_int = jnp.concatenate(
        [
            jnp.eye(n_boxes, dtype=jnp.float64),
            jnp.zeros((n_boxes, d_gp), dtype=jnp.float64),
        ],
        axis=1,
    )
    f_aug = jnp.block(
        [
            [jnp.zeros((n_boxes, n_boxes), dtype=jnp.float64), c_int],
            [jnp.zeros((d_xi, n_boxes), dtype=jnp.float64), f_joint],
        ]
    )
    q_aug = jsl.block_diag(jnp.zeros((n_boxes, n_boxes), dtype=jnp.float64), q_joint)
    m_aug_vl = (
        jnp.block(
            [
                [f_aug, q_aug],
                [jnp.zeros((d_aug, d_aug), dtype=jnp.float64), -f_aug.T],
            ]
        )
        * step_dt
    )
    e_aug = jsl.expm(m_aug_vl)
    phi_aug = e_aug[:d_aug, :d_aug]
    sig_aug = e_aug[:d_aug, d_aug:] @ phi_aug.T
    sig_aug = 0.5 * (sig_aug + sig_aug.T)
    return _KalmanStepDynamics(
        phi_xi=phi_aug[n_boxes:, n_boxes:],
        sig_xi=sig_aug[n_boxes:, n_boxes:],
        phi_s_xi=phi_aug[:n_boxes, n_boxes:],
        sig_ss=sig_aug[:n_boxes, :n_boxes],
        sig_s_xi=sig_aug[:n_boxes, n_boxes:],
    )


def _joseph_scalar_update(
    mean: jax.Array,
    cov: jax.Array,
    h_vec: jax.Array,
    y_val: jax.Array,
    r_var: jax.Array,
    active: jax.Array,
    extra_cols: jax.Array,
) -> _JosephUpdateResult:
    """Perform one stabilized Joseph-form scalar Kalman update and nuisance projection."""
    d_state = mean.shape[0]
    eye = jnp.eye(d_state, dtype=jnp.float64)
    ph = cov @ h_vec
    s_innov = jnp.dot(h_vec, ph) + r_var
    safe_s = jnp.where(active, s_innov, 1.0)
    sqrt_s = jnp.sqrt(safe_s)
    inv_sqrt_s = jnp.where(active, 1.0 / sqrt_s, 0.0)

    innov = y_val - jnp.dot(h_vec, mean)
    k_gain = ph / safe_s

    new_mean = mean + k_gain * innov
    i_kh = eye - jnp.outer(k_gain, h_vec)
    new_cov = i_kh @ cov @ i_kh.T + r_var * jnp.outer(k_gain, k_gain)
    new_cov = 0.5 * (new_cov + new_cov.T)

    delta_log_norm = -0.5 * (math.log(2.0 * math.pi) + jnp.log(safe_s))
    std_innov = jnp.where(active, innov * inv_sqrt_s, 0.0)

    h_row_nuis = extra_cols[0]
    m_extra = extra_cols[1:]
    innov_extra = h_row_nuis - h_vec @ m_extra
    new_m_extra = m_extra + k_gain[:, None] * innov_extra[None, :]
    std_extra_row = jnp.where(active, innov_extra * inv_sqrt_s, 0.0)

    return _JosephUpdateResult(
        mean=jnp.where(active, new_mean, mean),
        cov=jnp.where(active, new_cov, cov),
        delta_log_norm=jnp.where(active, delta_log_norm, 0.0),
        std_innov=std_innov,
        std_extra_row=std_extra_row,
        m_extra=jnp.where(active, new_m_extra, m_extra),
        k_gain=jnp.where(active, k_gain, 0.0),
        inv_sqrt_s=inv_sqrt_s,
    )


def _build_kalman_step_matrices(
    k_idx: jax.Array,
    schedule: _KalmanSchedule,
    dynamics: _KalmanStepDynamics,
) -> tuple[jax.Array, jax.Array]:
    """Assemble augmented step transition ``Φ_k`` and covariance ``Σ_k`` at step ``k_idx``."""
    n_tracks = schedule.n_tracks
    if n_tracks == 0:
        return dynamics.phi_xi, dynamics.sig_xi
    d_xi = dynamics.phi_xi.shape[0]
    kp = schedule.keep_mask[k_idx]
    wk = schedule.w_step[k_idx]
    phi_k = jnp.block(
        [
            [jnp.diag(kp), wk @ dynamics.phi_s_xi],
            [jnp.zeros((d_xi, n_tracks), dtype=jnp.float64), dynamics.phi_xi],
        ]
    )
    sig_aa = wk @ dynamics.sig_ss @ wk.T
    sig_a_xi = wk @ dynamics.sig_s_xi
    sig_k = jnp.block([[sig_aa, sig_a_xi], [sig_a_xi.T, dynamics.sig_xi]])
    return phi_k, 0.5 * (sig_k + sig_k.T)


def _node_measurement_updates(
    mean: jax.Array,
    cov: jax.Array,
    m_extra: jax.Array,
    k_node: jax.Array,
    schedule: _KalmanSchedule,
    resid_shifted: jax.Array,
    r_vars: jax.Array,
    nuis_u_raw: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Apply all active scalar observation updates sequentially at grid node ``k_node``."""
    ll_node = jnp.asarray(0.0, dtype=jnp.float64)
    std_innovs = []
    std_extras = []
    k_gains = []
    inv_sqrt_ss = []
    for slot in range(schedule.max_simul):
        idx_i = schedule.node_obs_idx[k_node, slot]
        act_i = schedule.node_obs_active[k_node, slot]
        h_i = schedule.node_h_full[k_node, slot]
        y_i = resid_shifted[idx_i]
        r_i = r_vars[idx_i]
        extra_pack = jnp.concatenate([nuis_u_raw[idx_i][None, :], m_extra], axis=0)
        upd = _joseph_scalar_update(mean, cov, h_i, y_i, r_i, act_i, extra_pack)
        mean, cov, m_extra = upd.mean, upd.cov, upd.m_extra
        ll_node = ll_node + upd.delta_log_norm
        std_innovs.append(upd.std_innov)
        std_extras.append(upd.std_extra_row)
        k_gains.append(upd.k_gain)
        inv_sqrt_ss.append(upd.inv_sqrt_s)
    return (
        mean,
        cov,
        m_extra,
        ll_node,
        jnp.stack(std_innovs),
        jnp.stack(std_extras),
        jnp.stack(k_gains),
        jnp.stack(inv_sqrt_ss),
    )


def _run_kalman_forward_pass(
    schedule: _KalmanSchedule,
    dynamics: _KalmanStepDynamics,
    sigma_xi_inf: jax.Array,
    resid_shifted: jax.Array,
    r_vars: jax.Array,
    nuis_u_raw: jax.Array,
) -> _KalmanForwardPass:
    """Run the forward Joseph Kalman filter scan across all uniform grid steps."""
    n_tracks = schedule.n_tracks
    n_grid = schedule.node_obs_idx.shape[0]
    n_steps = n_grid - 1
    d_state = n_tracks + sigma_xi_inf.shape[0]
    p_nuis = nuis_u_raw.shape[1]

    init_mean = jnp.zeros((d_state,), dtype=jnp.float64)
    init_cov = jsl.block_diag(jnp.zeros((n_tracks, n_tracks), dtype=jnp.float64), sigma_xi_inf)
    init_mx = jnp.zeros((d_state, p_nuis), dtype=jnp.float64)

    m0, p0, mx0, ll0, sin0, sex0, kg0, isqs0 = _node_measurement_updates(
        init_mean,
        init_cov,
        init_mx,
        jnp.int32(0),
        schedule,
        resid_shifted,
        r_vars,
        nuis_u_raw,
    )

    def _forward_step(
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        k_step: jax.Array,
    ):
        m_prev, p_prev, mx_prev, ll_acc = carry
        phi_k, sig_k = _build_kalman_step_matrices(k_step, schedule, dynamics)
        m_pred = phi_k @ m_prev
        p_pred = phi_k @ p_prev @ phi_k.T + sig_k
        p_pred = 0.5 * (p_pred + p_pred.T)
        mx_pred = phi_k @ mx_prev
        m_f, p_f, mx_f, dll, s_in, s_ex, kg, isqs = _node_measurement_updates(
            m_pred, p_pred, mx_pred, k_step + 1, schedule, resid_shifted, r_vars, nuis_u_raw
        )
        return (
            (m_f, p_f, mx_f, ll_acc + dll),
            (m_f, p_f, mx_f, phi_k, s_in, s_ex, kg, isqs),
        )

    (_, _, _, log_norm), scan_out = jax.lax.scan(
        _forward_step, (m0, p0, mx0, ll0), jnp.arange(n_steps, dtype=jnp.int32)
    )
    m_seq, p_seq, mx_seq, phi_seq, sin_seq, sex_seq, kg_seq, isqs_seq = scan_out
    return _KalmanForwardPass(
        log_norm=log_norm,
        m_filt_all=jnp.concatenate([m0[None, :], m_seq], axis=0),
        p_filt_all=jnp.concatenate([p0[None, :, :], p_seq], axis=0),
        mx_filt_all=jnp.concatenate([mx0[None, :, :], mx_seq], axis=0),
        phi_seq=phi_seq,
        sin_all=jnp.concatenate([sin0[None, :], sin_seq], axis=0),
        sex_all=jnp.concatenate([sex0[None, :, :], sex_seq], axis=0),
        kg_all=jnp.concatenate([kg0[None, :, :], kg_seq], axis=0),
        inv_sqs_all=jnp.concatenate([isqs0[None, :], isqs_seq], axis=0),
    )


def _qr_whiten_and_posterior(
    x: jax.Array,
    u_mat: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Factor ``[U; I_p] = Q R`` once to return ``(quad, ½ ln det, ω̂, Σ_{ω|d})``."""
    p_cols = u_mat.shape[1]
    eye = jnp.eye(p_cols, dtype=jnp.float64)
    aug = jnp.concatenate([u_mat, eye], axis=0)
    q_aug, r_upper = jnp.linalg.qr(aug, mode="reduced")
    n_rows = u_mat.shape[0]
    q_top = q_aug[:n_rows]
    q_bot = q_aug[n_rows:]
    half_logdet = jnp.sum(jnp.log(jnp.abs(jnp.diag(r_upper))))

    omega0 = jsl.solve_triangular(r_upper, q_top.T @ x, lower=False)
    r_top = x - u_mat @ omega0
    d_omega = jsl.solve_triangular(r_upper, q_top.T @ r_top - q_bot.T @ omega0, lower=False)
    omega = omega0 + d_omega
    r_final = x - u_mat @ omega
    quad = jnp.dot(r_final, r_final) + jnp.dot(omega, omega)

    r_inv = jsl.solve_triangular(r_upper, eye, lower=False)
    cov_raw = r_inv @ r_inv.T
    return quad, half_logdet, omega, 0.5 * (cov_raw + cov_raw.T)


def _marginalize_kalman_nuisance(
    fwd: _KalmanForwardPass,
    nuis: _KalmanNuisanceSpec,
) -> _KalmanMarginalizedNuisance:
    """Marginalize linear nuisance columns analytically via a single thin-QR pass."""
    all_sin = fwd.sin_all.reshape(-1)
    if nuis.p_nuis == 0:
        return _KalmanMarginalizedNuisance(
            loglike=fwd.log_norm - 0.5 * jnp.dot(all_sin, all_sin),
            nuisance_mean=None,
            nuisance_cov=None,
            cov_omega_nuis=jnp.zeros((0, 0), dtype=jnp.float64),
            m_filt_adj=fwd.m_filt_all,
            sin_adj_all=fwd.sin_all,
        )

    all_sex = fwd.sex_all.reshape(-1, nuis.p_nuis)
    quad_nuis, half_logdet_nuis, omega_nuis, cov_omega_nuis = _qr_whiten_and_posterior(
        all_sin, all_sex
    )
    total_ll = fwd.log_norm - 0.5 * quad_nuis - half_logdet_nuis
    nuis_mean = nuis.prior_mean + nuis.prior_root @ omega_nuis
    raw_cov = nuis.prior_root @ cov_omega_nuis @ nuis.prior_root.T
    nuis_cov = 0.5 * (raw_cov + raw_cov.T)
    m_filt_adj = fwd.m_filt_all - jnp.einsum("tip,p->ti", fwd.mx_filt_all, omega_nuis)
    sin_adj_all = fwd.sin_all - jnp.einsum("tsp,p->ts", fwd.sex_all, omega_nuis)
    return _KalmanMarginalizedNuisance(
        loglike=total_ll,
        nuisance_mean=nuis_mean,
        nuisance_cov=nuis_cov,
        cov_omega_nuis=cov_omega_nuis,
        m_filt_adj=m_filt_adj,
        sin_adj_all=sin_adj_all,
    )


def _run_kalman_rts_smoother(
    fwd: _KalmanForwardPass,
    schedule: _KalmanSchedule,
    marg: _KalmanMarginalizedNuisance,
) -> tuple[jax.Array, jax.Array]:
    """Run the backward inversion-free Rauch–Tung–Striebel / Bryson–Frazier smoother.

    Propagates backward adjoints ``(λ_fut, Λ_fut, Λ_{x,fut})`` across the scalar
    Joseph innovation factors ``I - h_j k_jᵀ`` without inverting ``P_{k+1|k}``,
    preserving machine precision on rank-deficient accumulator tracks and
    uncontrollable multi-box CBM modes, then restores marginalized nuisance
    uncertainty ``M_{x, k|N} Σ_{ω|d} M_{x, k|N}ᵀ`` by the law of total covariance.
    """
    d_state = fwd.m_filt_all.shape[1]
    p_nuis = fwd.mx_filt_all.shape[2]
    eye_d = jnp.eye(d_state, dtype=jnp.float64)
    phi_pad = jnp.concatenate(
        [jnp.zeros((1, d_state, d_state), dtype=jnp.float64), fwd.phi_seq], axis=0
    )

    def _rts_adjoint_step(carry, inputs):
        lam_fut, cap_fut, lam_x_fut = carry
        m_f, p_f, mx_f, phi_prev, h_s, act_s, kg_s, isqs_s, sin_s, sex_s = inputs

        m_s = m_f + p_f @ lam_fut
        p_s = 0.5 * ((p_f - p_f @ cap_fut @ p_f) + (p_f - p_f @ cap_fut @ p_f).T)
        mx_s = mx_f + p_f @ lam_x_fut

        lam, cap, lam_x = lam_fut, cap_fut, lam_x_fut
        for slot in range(schedule.max_simul - 1, -1, -1):
            act = act_s[slot]
            h_i = h_s[slot]
            l_tr = eye_d - jnp.outer(h_i, kg_s[slot])
            h_std = h_i * isqs_s[slot]
            new_lam = h_std * sin_s[slot] + l_tr @ lam
            new_cap = jnp.outer(h_std, h_std) + l_tr @ cap @ l_tr.T
            new_cap = 0.5 * (new_cap + new_cap.T)
            new_lam_x = jnp.outer(h_std, sex_s[slot]) + l_tr @ lam_x
            lam = jnp.where(act, new_lam, lam)
            cap = jnp.where(act, new_cap, cap)
            lam_x = jnp.where(act, new_lam_x, lam_x)

        cap_prev = phi_prev.T @ cap @ phi_prev
        return (
            (phi_prev.T @ lam, 0.5 * (cap_prev + cap_prev.T), phi_prev.T @ lam_x),
            (m_s, p_s, mx_s),
        )

    init_adj = (
        jnp.zeros((d_state,), dtype=jnp.float64),
        jnp.zeros((d_state, d_state), dtype=jnp.float64),
        jnp.zeros((d_state, p_nuis), dtype=jnp.float64),
    )
    _, (m_smooth, p_smooth, mx_smooth) = jax.lax.scan(
        _rts_adjoint_step,
        init_adj,
        (
            marg.m_filt_adj,
            fwd.p_filt_all,
            fwd.mx_filt_all,
            phi_pad,
            schedule.node_h_full,
            schedule.node_obs_active,
            fwd.kg_all,
            fwd.inv_sqs_all,
            marg.sin_adj_all,
            fwd.sex_all,
        ),
        reverse=True,
    )
    if p_nuis > 0:
        p_smooth = p_smooth + jnp.einsum(
            "tik,kl,tjl->tij", mx_smooth, marg.cov_omega_nuis, mx_smooth
        )
    return m_smooth, p_smooth


# shape-exempt(carbon): existing public API moved from linear.py preserving 20-parameter signature until Decision 5.3 surface cleanup
def kalman_gp_marginal(
    A: ArrayLike,
    b: ArrayLike,
    kernel: tgk.Kernel,
    resid: ArrayLike,
    sigma: ArrayLike,
    t_starts: ArrayLike,
    t_ends: ArrayLike | None = None,
    sub_weights: ArrayLike | None = None,
    *,
    t_grid: ArrayLike | None = None,
    dt: float | None = None,
    alpha_unit: float = 1.0,
    box_indices: ArrayLike = 0,
    obs_scale: ArrayLike = 1.0,
    obs_matrix: ArrayLike | None = None,
    nuisance_design: ArrayLike | None = None,
    nuisance_prior_mean: ArrayLike = 0.0,
    nuisance_prior_sd: ArrayLike | None = None,
    nuisance_prior_root: ArrayLike | None = None,
    dtau0: float = 1.0 / 16.0,
    n_squarings: int = 24,
) -> KalmanGPResult:
    """Compute exact state-space GP marginal likelihood and smoothed posterior.

    Propagates the continuous-time joint ``(CBM + quasisep GP)`` stochastic
    differential equation initialized at its stationary covariance ``Σ_∞`` (from
    ``n_squarings`` discrete Smith squarings), supporting instantaneous point
    observations, sub-annual seasonal growth windows, and IntCal20-style
    multi-year block-integral observations ``[t_starts, t_ends]`` aligned to
    ``t_grid`` via Van Loan step-integral state augmentation. Matches the dense
    continuous-time double-integral Gaussian posterior to ``rtol=1e-10`` across
    ``Matern12``, ``Matern32``, and ``Matern52`` kernels (verified in
    ``test_dec51b_kalman_gp.py``).

    Parameters
    ----------
    A : array-like
        Hurwitz carbon-box rate matrix of shape ``(n_boxes, n_boxes)`` in
        ``yr^-1`` (including radioactive ``¹⁴C`` decay ``-λ_14``).
    b : array-like
        Dimensionless cosmic-ray production branching vector of shape
        ``(n_boxes,)`` summing to ``1`` across atmospheric boxes.
    kernel : tinygp.kernels.Kernel
        Stationary Markov / ``tinygp.kernels.quasisep`` GP kernel over
        production rate anomalies ``δq(t)`` (time argument in ``yr``; amplitude
        in ``atoms cm^-2 s^-1`` or steady-state ``q_0`` units).
    resid : array-like
        1D observation residual vector ``d - m_0`` of shape ``(N_d,)`` in data
        units (typically ``‰`` for ``Δ¹⁴C``).
    sigma : float or array-like
        Strictly positive finite 1σ measurement uncertainties (scalar or 1D
        array of shape ``(N_d,)``, same units as ``resid``).
    t_starts : array-like
        Concrete observation window start timestamps of shape ``(N_d,)`` or
        ``(N_d, M_sub)`` in ``yr``, aligned to ``t_grid``.
    t_ends : array-like, optional
        Concrete observation window end timestamps matching ``t_starts`` in
        ``yr`` (``t_ends >= t_starts``). When ``None``, observations are treated
        as instantaneous point evaluations at ``t_starts``.
    sub_weights : array-like, optional
        Non-negative seasonal sub-window weights broadcastable to
        ``(N_d, M_sub)`` (normalized per row). Defaults to uniform ``1 / M_sub``.
    t_grid : array-like, optional
        Explicit concrete uniformly spaced state-space evaluation grid of shape
        ``(N_grid,)`` in ``yr`` covering all ``[t_starts, t_ends]`` boundaries.
    dt : float, optional
        Uniform grid spacing in ``yr`` used when ``t_grid`` is ``None``.
        Defaults to ``1 / 24`` yr (half-monthly, commensurate with standard
        monthly NH and SH tree-ring growth windows).
    alpha_unit : float, default 1.0
        Unit conversion factor mapping ``b * δq(t)`` into ``d(δy)/dt`` inventory
        units (e.g. ``ALPHA_ATOMS_CM2_S_TO_KG_YR``).
    box_indices : int or array-like of int, default 0
        Observed box index (or per-observation ``(N_d,)`` indices) when
        ``obs_matrix`` is ``None``.
    obs_scale : float or array-like, default 1.0
        Linear conversion factor from box ``¹⁴C`` inventory ``δy_c`` to data
        units (``‰``), scalar or 1D array of shape ``(N_d,)``.
    obs_matrix : array-like, optional
        Explicit observation matrix ``C_obs`` of shape ``(N_d, n_boxes)``
        overriding ``box_indices`` and ``obs_scale``.
    nuisance_design : array-like, optional
        Linear nuisance design matrix ``X_nuis`` of shape ``(N_d, p_nuis)``
        (e.g. hierarchical ``Δ_global + δ_{lab,s}`` offsets and solar harmonics).
    nuisance_prior_mean : float or array-like, default 0.0
        Gaussian prior mean ``μ_nuis`` for ``β`` (scalar or ``(p_nuis,)``).
    nuisance_prior_sd : float or array-like, optional
        Strictly positive diagonal prior standard deviations for ``β`` (scalar
        or ``(p_nuis,)``). Mutually exclusive with ``nuisance_prior_root``.
    nuisance_prior_root : array-like, optional
        Prior covariance square-root factor ``L_nuis`` of shape
        ``(p_nuis, p_nuis)`` with ``Σ_β = L_nuis L_nuisᵀ``. Mutually exclusive
        with ``nuisance_prior_sd``.
    dtau0 : float, default 0.0625
        Base step size in ``yr`` for 24-step Smith doubling of ``Σ_∞``.
    n_squarings : int, default 24
        Number of discrete Smith squarings for stationary covariance ``Σ_∞``.

    Returns
    -------
    KalmanGPResult
        Exact marginal log-likelihood, ``t_grid``, smoothed production and CBM
        state posterior means/variances, and optional nuisance posterior.

    Raises
    ------
    ValueError
        If ``kernel`` is not Markov/quasiseparable, window timestamps or
        ``t_grid`` are traced or misaligned, ``sigma`` is non-positive, or
        nuisance shapes/priors are invalid.
    """
    sde = _try_extract_sde(kernel)
    if sde is None:
        raise ValueError(
            f"kalman_gp_marginal requires a Markov / quasisep kernel; got "
            f"{type(kernel).__name__!r}. Use project_causal_gp for non-Markov kernels."
        )
    f_gp, q_sde, _, h_gp = sde

    a_mat, b_eff, n_boxes = _validate_cbm_system(A, b, alpha_unit)
    c_starts, c_ends, c_weights = _normalize_concrete_windows(t_starts, t_ends, sub_weights)
    n_obs = c_starts.shape[0]
    resid_arr, sd_arr = _validate_resid_and_sigma(resid, sigma, n_obs)
    c_obs = _build_obs_matrix(
        n_obs, n_boxes, box_indices=box_indices, obs_scale=obs_scale, obs_matrix=obs_matrix
    )
    t_grid_np, step_dt = _resolve_kalman_time_grid(
        t_grid, c_starts, c_ends, dt, c_weights=c_weights
    )
    schedule = _build_seasonal_kalman_schedule(
        t_grid_np, c_starts, c_ends, c_weights, c_obs, step_dt, d_gp=f_gp.shape[0]
    )
    nuis = _prepare_kalman_nuisance(
        resid_arr,
        nuisance_design,
        nuisance_prior_mean,
        nuisance_prior_sd,
        nuisance_prior_root,
    )
    _, _, sigma_xi_inf = stationary_cbm_gp_cov(
        a_mat,
        b,
        kernel,
        alpha_unit=alpha_unit,
        dtau0=dtau0,
        n_squarings=n_squarings,
        return_full_state=True,
    )
    dynamics = _van_loan_kalman_step_dynamics(
        a_mat, b_eff, f_gp, q_sde, h_gp, step_dt, has_tracks=schedule.n_tracks > 0
    )
    fwd = _run_kalman_forward_pass(
        schedule, dynamics, sigma_xi_inf, nuis.resid_shifted, jnp.square(sd_arr), nuis.nuis_u_raw
    )
    marg = _marginalize_kalman_nuisance(fwd, nuis)
    m_smooth, p_smooth = _run_kalman_rts_smoother(fwd, schedule, marg)

    cbm_slice = slice(schedule.n_tracks, schedule.n_tracks + n_boxes)
    gp_slice = slice(schedule.n_tracks + n_boxes, m_smooth.shape[1])
    cbm_var = jnp.diagonal(p_smooth[:, cbm_slice, cbm_slice], axis1=1, axis2=2)
    q_var = jnp.einsum("i,tij,j->t", h_gp, p_smooth[:, gp_slice, gp_slice], h_gp)
    return KalmanGPResult(
        loglike=marg.loglike,
        t_grid=jnp.asarray(t_grid_np, dtype=jnp.float64),
        production_mean=m_smooth[:, gp_slice] @ h_gp,
        production_var=jnp.maximum(q_var, 0.0),
        cbm_state_mean=m_smooth[:, cbm_slice],
        cbm_state_var=jnp.maximum(cbm_var, 0.0),
        nuisance_mean=marg.nuisance_mean,
        nuisance_cov=marg.nuisance_cov,
    )


from .linear import (
    _build_obs_matrix,
    _try_extract_sde,
    solin_sarkka_basis,
    stationary_cbm_gp_cov,
    whiten_qr,
)
