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

if TYPE_CHECKING:
    from .linear import CausalGPProjection

__all__ = ["ArrowProblem", "CanonicalPODResult", "arrow_at", "arrow_problem", "canonical_pod_modes"]


class ArrowProblem(NamedTuple):
    """Block-arrow ``Ũ`` reduced to Gram sums for fast noise-scale evaluation."""

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
    """Build Gram sums of ``s² I + Ũ Ũᵀ`` for [`arrow_at`][ticktack.kalman.arrow_at]."""
    x = jnp.asarray(x, dtype=jnp.float64)
    local = jnp.asarray(local, dtype=jnp.float64)
    rows = jnp.asarray(rows)
    if spanning is None:
        spanning = jnp.zeros((x.shape[0], 0), dtype=x.dtype)
    else:
        spanning = jnp.asarray(spanning, dtype=jnp.float64)
    local = jnp.where((rows < x.shape[0])[..., None], local, 0.0)
    hi = jax.lax.Precision.HIGHEST
    x_b = x.at[rows].get(mode="fill", fill_value=0.0)
    s_b = spanning.at[rows].get(mode="fill", fill_value=0.0)
    outside = jnp.ones(x.shape[0], dtype=x.dtype).at[rows].set(0.0, mode="drop")
    return ArrowProblem(
        x,
        rows,
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
    """Evaluate marginal ``(chi2, norm, r2[, tr])`` at variance scale ``s² = scale2``."""
    p = problem
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
    """Simultaneously data-orthogonal and prior-orthonormal Wiener GP modes."""

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

    Takes the SVD of the (nuisance- and initial-state-whitened) GP operator
    ``D̃⁻¹/² W_causal Ψ_r Λ¹/² = U_data diag(s) V_priorᵀ`` so each canonical
    coefficient ``c̃_k`` has decoupled 1D posterior
    ``N(s_k / (1 + s_k²) u_kᵀ r̃, 1 / (1 + s_k²))``, matching
    [`CausalGPProjection.posterior_production`][ticktack.linear.CausalGPProjection.posterior_production]
    to machine precision.
    """
    from .linear import solin_sarkka_basis, whiten_cholesky

    if projection.spectral_weights is None:
        raise ValueError("projection must be constructed with kernel=... for canonical_pod_modes.")

    n_obs, r_modes = projection.causal_basis_design.shape
    resid_arr = jnp.asarray(resid, dtype=jnp.float64).reshape(n_obs)
    sd = jnp.broadcast_to(jnp.asarray(sigma, dtype=jnp.float64), (n_obs,))

    gp_prior_design = projection.causal_basis_design * jnp.sqrt(
        projection.spectral_weights
    )[None, :]
    g_raw = gp_prior_design / sd[:, None]  # (N_d, r)
    r_raw = resid_arr / sd  # (N_d,)

    nuis_parts = []
    if include_initial_cov and projection.anchor_cov_root is not None:
        nuis_parts.append(
            (projection.hom_design_anchor @ projection.anchor_cov_root) / sd[:, None]
        )
    if nuisance_whitened_design is not None:
        nuis_parts.append(
            jnp.asarray(nuisance_whitened_design, dtype=jnp.float64) / sd[:, None]
        )

    if nuis_parts:
        u_nuis = jnp.concatenate(nuis_parts, axis=1)
        r_tilde, _ = whiten_cholesky(r_raw, u_nuis)
        g_tilde = jax.vmap(lambda col: whiten_cholesky(col, u_nuis)[0], in_axes=1, out_axes=1)(
            g_raw
        )
    else:
        r_tilde = r_raw
        g_tilde = g_raw

    m_sing = min(n_obs, r_modes)
    if n_obs < r_modes:
        g_svd_in = jnp.concatenate(
            [g_tilde, jnp.zeros((r_modes - n_obs, r_modes), dtype=jnp.float64)],
            axis=0,
        )
    else:
        g_svd_in = g_tilde
    u_svd, s_full, vt_prior = jnp.linalg.svd(g_svd_in, full_matrices=False)
    v_prior = vt_prior.T  # (r, r)
    s_vals = s_full[:m_sing]
    u_data = u_svd[:n_obs, :m_sing]
    proj_full = u_svd[:n_obs, :].T @ r_tilde  # (r,)

    wiener = s_full**2 / (1.0 + s_full**2)
    coeff_mean = (s_full / (1.0 + s_full**2)) * proj_full
    coeff_var = 1.0 / (1.0 + s_full**2)

    basis_eval = solin_sarkka_basis(t_eval, projection.a, projection.b_bound, r_modes)
    prod_modes = (
        basis_eval.psi * jnp.sqrt(projection.spectral_weights)[None, :]
    ) @ v_prior  # (N_eval, r)
    post_mean = prod_modes @ coeff_mean
    post_var = jnp.sum((prod_modes**2) * coeff_var[None, :], axis=1)
    post_sd = jnp.sqrt(jnp.maximum(post_var, 0.0))

    d_eff = jnp.sum(wiener)
    info_nats = 0.5 * jnp.sum(jnp.log1p(s_full**2))

    return CanonicalPODResult(
        singular_values=s_vals,
        wiener_weights=wiener,
        data_modes=u_data[:, :m_sing],
        prior_rotation=v_prior,
        production_modes=prod_modes,
        coeff_mean=coeff_mean,
        coeff_var=coeff_var,
        posterior_mean=post_mean,
        posterior_sd=post_sd,
        d_eff=d_eff,
        info_nats=info_nats,
    )
