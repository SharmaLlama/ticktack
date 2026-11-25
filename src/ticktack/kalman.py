"""Block-arrow marginalization and canonical Wiener POD modes.

Houses the block-arrow linear-Gaussian solver
([`ArrowProblem`][ticktack.kalman.ArrowProblem],
[`arrow_problem`][ticktack.kalman.arrow_problem],
[`arrow_at`][ticktack.kalman.arrow_at]) and the canonical data- and
prior-orthogonal Wiener SVD decomposition
([`CanonicalPODResult`][ticktack.kalman.CanonicalPODResult],
[`canonical_pod_modes`][ticktack.kalman.canonical_pod_modes]), preserved as a
clean submodule for state-space and low-rank linear network inference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as onp

if TYPE_CHECKING:
    from .linear import CausalGPProjection

__all__ = [
    "ArrowProblem",
    "CanonicalPODResult",
    "arrow_at",
    "arrow_problem",
    "canonical_pod_modes",
]


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
    x: Any,
    rows: Any,
    local: Any,
    spanning: Any | None = None,
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
    from .linear import _concrete

    x = jnp.asarray(x, dtype=jnp.float64)
    if x.ndim != 1:
        raise ValueError(f"x must be a 1D array of shape (N,); got shape {x.shape}.")
    rows = jnp.asarray(rows)
    if rows.ndim != 2 or not jnp.issubdtype(rows.dtype, jnp.integer):
        raise ValueError(
            f"rows must be a 2D integer array of shape (B, R); "
            f"got shape {rows.shape} and dtype {rows.dtype}."
        )
    local = jnp.asarray(local, dtype=jnp.float64)
    if local.ndim != 3 or local.shape[:2] != rows.shape:
        raise ValueError(
            f"local must have shape ({rows.shape[0]}, {rows.shape[1]}, M); got {local.shape}."
        )
    n_obs = x.shape[0]
    if spanning is None:
        spanning = jnp.zeros((n_obs, 0), dtype=x.dtype)
    else:
        spanning = jnp.asarray(spanning, dtype=jnp.float64)
        if spanning.ndim != 2 or spanning.shape[0] != n_obs:
            raise ValueError(
                f"spanning must have shape ({n_obs}, K); got {spanning.shape}."
            )
    valid = (rows >= 0) & (rows < n_obs)
    safe_rows = jnp.where(valid, rows, n_obs)
    c_rows = _concrete(rows)
    if c_rows is not None:
        active = c_rows[(c_rows >= 0) & (c_rows < n_obs)]
        if active.size != onp.unique(active).size:
            raise ValueError(
                "arrow_problem requires disjoint active row indices across blocks; "
                "duplicate active indices found in rows."
            )
    local = jnp.where(valid[..., None], local, 0.0)
    hi = jax.lax.Precision.HIGHEST
    x_b = x.at[safe_rows].get(mode="fill", fill_value=0.0)
    s_b = spanning.at[safe_rows].get(mode="fill", fill_value=0.0)
    outside = jnp.ones(n_obs, dtype=x.dtype).at[safe_rows].set(0.0, mode="drop")
    return ArrowProblem(
        x,
        safe_rows,
        local,
        spanning,
        jnp.einsum("brm,brn->bmn", local, local, precision=hi),
        jnp.einsum("brm,brk->bmk", local, s_b, precision=hi),
        jnp.matmul(spanning.T * outside, spanning, precision=hi),
        jnp.einsum("brm,br->bm", local, x_b, precision=hi),
    )


def _solve_spd(chol: jax.Array, rhs: jax.Array) -> jax.Array:
    return jsl.cho_solve((chol, True), rhs)


def arrow_at(
    problem: ArrowProblem,
    scale2: Any,
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
    from .linear import _concrete

    p = problem
    scale2 = jnp.asarray(scale2, dtype=jnp.float64)
    if scale2.ndim != 0:
        raise ValueError(f"scale2 must be a scalar; got shape {scale2.shape}.")
    c_s2 = _concrete(scale2)
    if c_s2 is not None and (not onp.isfinite(c_s2) or float(c_s2) <= 0.0):
        raise ValueError(f"scale2 must be finite and strictly positive; got {c_s2}.")
    hi = jax.lax.Precision.HIGHEST
    n_block, n_mode = p.gram_bb.shape[:2]
    n_span = p.gram_out.shape[0]
    eye = jnp.eye(n_mode, dtype=p.gram_bb.dtype)
    chol_b = jnp.linalg.cholesky(p.gram_bb + scale2 * eye)
    rhs = jnp.concatenate([p.proj_b[..., None], p.gram_bs], axis=-1)
    sol = jax.vmap(_solve_spd)(chol_b, rhs)
    m_g, m_gs = sol[..., 0], sol[..., 1:]
    eye_s = jnp.eye(n_span, dtype=p.gram_out.dtype)
    s_b = p.spanning.at[p.rows].get(mode="fill", fill_value=0.0)
    z = s_b - jnp.einsum("brm,bmk->brk", p.local, m_gs, precision=hi)
    schur = (
        scale2 * eye_s
        + p.gram_out
        + jnp.einsum("brk,brl->kl", z, z, precision=hi)
        + scale2 * jnp.einsum("bmk,bml->kl", m_gs, m_gs, precision=hi)
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
    g_b = jnp.einsum("brm,br->bm", p.local, r_b, precision=hi) - scale2 * y_b
    g_s = jnp.matmul(p.spanning.T, resid, precision=hi) - scale2 * y_s
    w_b = jax.vmap(_solve_spd)(chol_b, g_b[..., None])[..., 0]
    d_s = _solve_spd(
        chol_s, g_s - jnp.einsum("bmk,bm->k", p.gram_bs, w_b, precision=hi)
    )
    y_s = y_s + d_s
    y_b = y_b + w_b - jnp.einsum("bmk,k->bm", m_gs, d_s, precision=hi)
    r2 = jnp.sum(residual(y_b, y_s) ** 2)
    chi2 = r2 / scale2 + jnp.sum(y_b**2) + jnp.sum(y_s**2)
    log_diag = jnp.sum(jnp.log(jnp.diagonal(chol_b, axis1=1, axis2=2)))
    log_diag = log_diag + jnp.sum(jnp.log(jnp.diagonal(chol_s)))
    norm = log_diag - 0.5 * (n_block * n_mode + n_span) * jnp.log(scale2)
    if not trace:
        return chi2, norm, r2
    inv_b = jax.vmap(lambda c: _solve_spd(c, eye))(chol_b)
    z_tr = jnp.einsum("bmk,bml->kl", m_gs, m_gs, precision=hi)
    tr = jnp.sum(jnp.diagonal(inv_b, axis1=1, axis2=2))
    tr = tr + jnp.trace(_solve_spd(chol_s, z_tr + eye_s))
    return chi2, norm, r2, tr


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
        coeff_var = jnp.concatenate(
            [coeff_var_thin, jnp.ones((n_null,), dtype=jnp.float64)]
        )
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


def canonical_pod_modes(
    projection: CausalGPProjection,
    t_eval: Any,
    resid: Any,
    sigma: Any,
    *,
    nuisance_whitened_design: Any | None = None,
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
    from .linear import _concrete, solin_sarkka_basis, whiten_qr

    if projection.spectral_weights is None:
        raise ValueError(
            "projection must be constructed with kernel=... for canonical_pod_modes."
        )

    n_obs, r_modes = projection.causal_basis_design.shape
    resid_arr = jnp.asarray(resid, dtype=jnp.float64).reshape(n_obs)
    sd_raw = jnp.asarray(sigma, dtype=jnp.float64)
    if sd_raw.ndim > 1:
        raise ValueError(
            f"sigma must be a scalar or 1D array of length {n_obs}; got shape {sd_raw.shape}."
        )
    sd = jnp.broadcast_to(sd_raw, (n_obs,))
    c_sd = _concrete(sd)
    if c_sd is not None and not onp.all(onp.isfinite(c_sd) & (c_sd > 0.0)):
        raise ValueError("sigma must be finite and strictly positive.")

    spec_root = jnp.sqrt(projection.spectral_weights)
    g_raw = (projection.causal_basis_design * spec_root[None, :]) / sd[:, None]
    r_raw = resid_arr / sd

    nuis_parts = []
    if include_initial_cov and projection.anchor_cov_root is not None:
        nuis_parts.append(
            (projection.hom_design_anchor @ projection.anchor_cov_root) / sd[:, None]
        )
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
