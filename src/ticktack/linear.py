"""Exact linear-Gaussian marginalization, causal Solin--Särkkä spectral GP
projection, 24-step Smith-doubling stationary initial covariance, canonical
Wiener SVD modes, and Kalman/RTS state-space inference.

Adapts the linear marginalization primitives of ``virgil._linear`` to
``ticktack 2.0``'s ``zodiax.Base`` architecture and pairs them with:

1. **Causal Solin & Särkkä (2020) Hilbert-space GP projection** without ODE
   scans or eternal-resolvent boundary leakage:
   ``(W_causal Ψ_r)_{:,j} = L⁻¹/² B_season[v_j cos ω_j(t-a) + u_j sin ω_j(t-a)
   - exp(A(t-a)) v_j]``, solving ``(i ω_j I_n - A) z_j = α_unit b`` directly
   in complex arithmetic without squaring ``A``.
2. **Exact pre-history stationary covariance** ``Σ_{y,0}(φ)`` via 24 discrete
   Smith squarings of the Van Loan step covariance at ``Δτ_0 = 1/16 yr``
   (spanning ``1.05 × 10⁶ yr``) for ``tinygp.kernels.quasisep`` / Markov SDE
   kernels, and smooth ``sinh``-log Gauss--Legendre spectral quadrature for
   non-quasiseparable kernels such as ``ExpSquared``.
3. **Canonical data- and prior-orthogonal Wiener SVD modes**
   ([`canonical_pod_modes`][ticktack.linear.canonical_pod_modes]) and an
   **exact continuous-time Kalman filter + RTS smoother**
   ([`kalman_gp_marginal`][ticktack.linear.kalman_gp_marginal]) for uniform
   step grids with point or seasonal-window integral observations.
"""

from __future__ import annotations

import math
from typing import Any, Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as onp
import tinygp.kernels as tgk
import zodiax as zdx
from tinygp.kernels import quasisep as tgq
from tinygp.solvers.quasisep.block import Block

__all__ = [
    "ArrowProblem",
    "CanonicalPODResult",
    "CausalGPProjection",
    "KalmanGPResult",
    "LinearMarginal",
    "SolinSarkkaBasis",
    "arrow_at",
    "arrow_problem",
    "canonical_pod_modes",
    "harmonic_solar_design",
    "hierarchical_offset_design",
    "kalman_gp_marginal",
    "posterior",
    "project_causal_gp",
    "solin_sarkka_basis",
    "spectral_density",
    "stationary_cbm_gp_cov",
    "whiten_blocks",
    "whiten_cholesky",
    "whiten_qr",
    "whiten_rank_one",
]


def _concrete(value: Any) -> onp.ndarray | None:
    """Return ``value`` as a NumPy array, or ``None`` when traced under JAX."""
    try:
        return onp.asarray(value)
    except (
        jax.errors.TracerArrayConversionError,
        jax.errors.ConcretizationTypeError,
    ):
        return None


def _safe_psd_root(sym_mat: jax.Array, jitter: float = 1e-13) -> jax.Array:
    """Return ``L`` with ``L @ L.T ≈ sym_mat`` and smooth autodiff across spectra.

    Attempts trace-scaled jittered Cholesky first (real-analytic ``C^∞`` even at
    repeated eigenvalues where ``eigh`` reverse-mode gradients diverge) and
    falls back via ``jax.lax.cond`` to double-``where`` guarded symmetric
    eigendecomposition only if a slight numerical overshoot leaves ``sym_mat``
    indefinite.
    """
    n = sym_mat.shape[0]
    if n == 0:
        return jnp.zeros((0, 0), dtype=jnp.float64)
    sym = 0.5 * (sym_mat + sym_mat.T)
    tr_scale = jnp.trace(sym) / n
    reg_diag = jitter * jnp.maximum(tr_scale, 0.0) + 1e-30
    reg_mat = sym + reg_diag * jnp.eye(n, dtype=jnp.float64)
    chol = jnp.linalg.cholesky(reg_mat)
    chol_ok = jnp.all(jnp.isfinite(chol))

    def _eigh_fallback(m: jax.Array) -> jax.Array:
        evals, evecs = jnp.linalg.eigh(0.5 * (m + m.T))
        pos = evals > 1e-30
        safe_evals = jnp.where(pos, evals, 1.0)
        roots = jnp.where(pos, jnp.sqrt(safe_evals), 0.0)
        return evecs * roots[None, :]

    c_ok = _concrete(chol_ok)
    if c_ok is not None:
        return chol if bool(c_ok) else _eigh_fallback(sym)
    return jax.lax.cond(chol_ok, lambda _: chol, _eigh_fallback, sym)


# =============================================================================
# 1. Analytic Linear Marginalization (adapted from virgil._linear)
# =============================================================================


def _rank_one_step(stack: jax.Array, j: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Whiten ``stack`` ``(..., n_row, n_col)`` for column ``j`` as a rank-1 mode."""
    w = stack[..., j]
    s = jnp.sum(w**2, axis=-1)
    q = jnp.sqrt(1.0 + s)
    coef = jnp.einsum("...r,...rk->...k", w, stack) / (q * (q + 1.0))[..., None]
    return stack - w[..., None] * coef[..., None, :], jnp.log1p(s)


def whiten_blocks(
    x: Any,
    rows: Any,
    local: Any,
    spanning: Any | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Whiten ``x`` for ``I + Σ w_j w_jᵀ`` by rank-one steps, block by block."""
    x = jnp.asarray(x, dtype=jnp.float64)
    rows = jnp.asarray(rows)
    local = jnp.asarray(local, dtype=jnp.float64)
    n = x.shape[0]
    if spanning is None:
        spanning = jnp.zeros((n, 0), dtype=x.dtype)
    else:
        spanning = jnp.asarray(spanning, dtype=jnp.float64)
    n_span = spanning.shape[1]
    stack = jnp.concatenate(
        [
            x.at[rows].get(mode="fill", fill_value=0.0)[..., None],
            spanning.at[rows].get(mode="fill", fill_value=0.0),
            local,
        ],
        axis=-1,
    )
    stack, logdets = jax.lax.scan(
        _rank_one_step, stack, 1 + n_span + jnp.arange(local.shape[-1])
    )
    done = (
        jnp.concatenate([x[:, None], spanning], axis=1)
        .at[rows]
        .set(stack[..., : 1 + n_span], mode="drop")
    )
    n_rows = jnp.sum(rows < n, axis=1)
    per_row = 0.5 * jnp.sum(logdets, axis=0) / jnp.maximum(n_rows, 1)
    extra = (
        jnp.zeros_like(x)
        .at[rows]
        .set(jnp.broadcast_to(per_row[:, None], rows.shape), mode="drop")
    )
    if n_span:
        done, span_logdets = jax.lax.scan(
            _rank_one_step, done[None], 1 + jnp.arange(n_span)
        )
        done = done[0]
        extra = extra + 0.5 * jnp.sum(span_logdets) / n
    return done[:, 0], extra


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
    """Build Gram sums of ``s² I + Ũ Ũᵀ`` for [`arrow_at`][ticktack.linear.arrow_at]."""
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


def whiten_rank_one(x: Any, U: Any) -> tuple[jax.Array, jax.Array]:
    """Whiten ``x`` for ``I + U Uᵀ`` by successive rank-one steps.

    Returns ``(u, ½ log det(I + Uᵀ U))`` with ``uᵀ u = xᵀ (I + U Uᵀ)⁻¹ x``.
    """
    x_arr = jnp.asarray(x, dtype=jnp.float64)
    u_mat = jnp.asarray(U, dtype=jnp.float64)
    if u_mat.shape[1] == 0:
        return x_arr, jnp.asarray(0.0, dtype=jnp.float64)
    rows = jnp.arange(x_arr.shape[0])[None]
    u_out, extra = whiten_blocks(x_arr, rows, u_mat[None])
    return u_out, jnp.sum(extra)


def whiten_cholesky(x: Any, U: Any) -> tuple[jax.Array, jax.Array]:
    """Whiten ``x`` for ``I + U Uᵀ`` via a dense ``k × k`` Cholesky factorization.

    Uses the exact non-commutative square-root identity
    ``(I - W (I + L⁻ᵀ)⁻¹ Wᵀ)ᵀ (I - W (I + L⁻ᵀ)⁻¹ Wᵀ) = I - U (I + Uᵀ U)⁻¹ Uᵀ``
    from ``virgil._linear``, returning ``(u, ½ log det(I + Uᵀ U))``.
    """
    x_arr = jnp.asarray(x, dtype=jnp.float64)
    u_mat = jnp.asarray(U, dtype=jnp.float64)
    k = u_mat.shape[1]
    if k == 0:
        return x_arr, jnp.asarray(0.0, dtype=jnp.float64)
    eye = jnp.eye(k, dtype=u_mat.dtype)
    chol = jnp.linalg.cholesky(eye + u_mat.T @ u_mat)
    l_inv_t = jsl.solve_triangular(chol, eye, lower=True).T
    w_mat = u_mat @ l_inv_t
    y = jsl.solve_triangular(eye + l_inv_t, w_mat.T @ x_arr, lower=False)
    return x_arr - w_mat @ y, jnp.sum(jnp.log(jnp.diag(chol)))


def whiten_qr(x: Any, U: Any) -> tuple[jax.Array, jax.Array]:
    """Whiten ``x`` for ``I + U Uᵀ`` via augmented thin QR without squaring ``κ(U)``.

    Factors ``[U; I_k] = Q R`` so ``Rᵀ R = I_k + Uᵀ U`` (preserving precision
    even when ``κ(U) ~ 10¹⁰``) and returns ``(u, ½ log det(I + Uᵀ U))`` where
    ``uᵀ u = ‖x - U ω̂‖² + ‖ω̂‖² = xᵀ (I + U Uᵀ)⁻¹ x``.
    """
    x_arr = jnp.asarray(x, dtype=jnp.float64)
    u_mat = jnp.asarray(U, dtype=jnp.float64)
    k = u_mat.shape[1]
    if k == 0:
        return x_arr, jnp.asarray(0.0, dtype=jnp.float64)
    eye = jnp.eye(k, dtype=u_mat.dtype)
    aug = jnp.concatenate([u_mat, eye], axis=0)
    _, r_upper = jnp.linalg.qr(aug, mode="reduced")
    # Ensure positive diagonal on L = R.T for exact square-root whitening formula
    signs = jnp.where(jnp.diag(r_upper) >= 0.0, 1.0, -1.0)
    chol = (signs[:, None] * r_upper).T  # lower triangular with chol @ chol.T = I + UᵀU
    l_inv_t = jsl.solve_triangular(chol, eye, lower=True).T
    w_mat = u_mat @ l_inv_t
    y = jsl.solve_triangular(eye + l_inv_t, w_mat.T @ x_arr, lower=False)
    return x_arr - w_mat @ y, jnp.sum(jnp.log(jnp.abs(jnp.diag(r_upper))))


def posterior(x: Any, U: Any) -> tuple[jax.Array, jax.Array]:
    """Conditional posterior ``N(S⁻¹ Uᵀ x, S⁻¹)`` of standardized ``ω`` (``w = μ + Λ¹/² ω``).

    Includes one residual-space iterative refinement step so broad baseline
    offsets (``σ_global = 200‰``) retain full ``float64`` precision.
    """
    x_arr = jnp.asarray(x, dtype=jnp.float64)
    u_mat = jnp.asarray(U, dtype=jnp.float64)
    k = u_mat.shape[1]
    if k == 0:
        return jnp.zeros((0,), dtype=jnp.float64), jnp.zeros((0, 0), dtype=jnp.float64)
    eye = jnp.eye(k, dtype=u_mat.dtype)
    s_mat = eye + u_mat.T @ u_mat
    factor = jsl.cho_factor(s_mat, lower=True)
    omega = jsl.cho_solve(factor, u_mat.T @ x_arr)
    resid = x_arr - u_mat @ omega
    omega = omega + jsl.cho_solve(factor, u_mat.T @ resid - omega)
    cov = jsl.cho_solve(factor, eye)
    cov = 0.5 * (cov + cov.T)
    return omega, cov


class LinearMarginal(zdx.Base):
    """Analytic Gaussian marginalization over linear parameters ``w ~ N(μ_w, Λ_w)``.

    Models observations ``d = m_0 + X w + ε`` with Gaussian measurement noise
    ``ε ~ N(0, D)`` and stated finite Gaussian prior ``w ~ N(μ_w, Λ_w)``
    (``Λ_w = L_w L_wᵀ``), yielding exact marginal log-likelihood

    .. math::

        \\ln p(\\mathbf{d} \\mid \\mathbf{m}_0) =
        -\\frac{1}{2}\\mathbf{r}^\\top (D + X\\Lambda_w X^\\top)^{-1}\\mathbf{r}
        - \\frac{1}{2}\\ln\\det D
        - \\frac{1}{2}\\ln\\det(I + U^\\top U)
        - \\frac{N_d}{2}\\ln(2\\pi),

    where ``r = d - m_0 - X μ_w`` and ``U = D⁻¹/² X L_w``, plus closed-form
    posterior mean and covariance for ``w`` (GP spectral modes, harmonic solar
    quadratures, and hierarchical ``Δ_global + δ_{lab,s}`` baseline offsets).
    """

    design: jax.Array
    prior_mean: jax.Array
    prior_root: jax.Array
    method: str = eqx.field(static=True)

    def __init__(
        self,
        design: Any,
        prior_mean: Any = 0.0,
        prior_sd: Any | None = None,
        prior_cov: Any | None = None,
        prior_root: Any | None = None,
        method: Literal["cholesky", "qr", "rank_one"] = "cholesky",
    ):
        design_arr = jnp.asarray(design, dtype=jnp.float64)
        if design_arr.ndim != 2:
            raise ValueError(f"design must be 2D (n_obs, k); got shape {design_arr.shape}.")
        k = design_arr.shape[1]

        n_given = sum(x is not None for x in (prior_sd, prior_cov, prior_root))
        if k == 0 and n_given == 0:
            root = jnp.zeros((0, 0), dtype=jnp.float64)
        elif n_given != 1:
            raise ValueError(
                "State the finite Gaussian prior via exactly one of "
                "prior_sd, prior_cov, or prior_root."
            )
        elif prior_sd is not None:
            sd = jnp.broadcast_to(jnp.asarray(prior_sd, dtype=jnp.float64), (k,))
            check = _concrete(sd)
            if check is not None and not onp.all(onp.isfinite(check) & (check > 0.0)):
                raise ValueError(
                    "prior_sd must be positive and finite; flat priors are not supported."
                )
            root = jnp.diag(sd)
        elif prior_cov is not None:
            cov = jnp.asarray(prior_cov, dtype=jnp.float64)
            if cov.shape != (k, k):
                raise ValueError(f"prior_cov must have shape ({k}, {k}); got {cov.shape}.")
            root = jnp.linalg.cholesky(0.5 * (cov + cov.T))
            check = _concrete(root)
            if check is not None and not onp.all(onp.isfinite(check)):
                raise ValueError("prior_cov must be finite and strictly positive definite.")
        else:
            root = jnp.asarray(prior_root, dtype=jnp.float64)
            if root.ndim != 2 or root.shape[0] != k:
                raise ValueError(
                    f"prior_root must have shape ({k}, m); got {root.shape}."
                )
            check = _concrete(root)
            if check is not None and not onp.all(onp.isfinite(check)):
                raise ValueError("prior_root must contain finite entries.")

        mean = jnp.broadcast_to(jnp.asarray(prior_mean, dtype=jnp.float64), (k,))
        check_mean = _concrete(mean)
        if check_mean is not None and not onp.all(onp.isfinite(check_mean)):
            raise ValueError("prior_mean must be finite.")
        if method not in ("cholesky", "qr", "rank_one"):
            raise ValueError(
                f"method must be 'cholesky', 'qr', or 'rank_one'; got {method!r}."
            )

        self.design = design_arr
        self.prior_mean = mean
        self.prior_root = root
        self.method = method

    def _standardize_with_logdet(
        self, resid: Any, sigma: Any
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return ``(x, U, ½ ln det D)`` for 1D ``sigma`` or 2D noise covariance ``D``."""
        resid_arr = jnp.asarray(resid, dtype=jnp.float64).reshape(-1)
        n_obs = self.design.shape[0]
        if resid_arr.shape[0] != n_obs:
            raise ValueError(
                f"resid length {resid_arr.shape[0]} does not match design rows {n_obs}."
            )
        shifted = resid_arr - self.design @ self.prior_mean
        xw = self.design @ self.prior_root

        sig_arr = jnp.asarray(sigma, dtype=jnp.float64)
        if sig_arr.ndim <= 1:
            sd = jnp.broadcast_to(sig_arr, (n_obs,))
            check = _concrete(sd)
            if check is not None and not onp.all(onp.isfinite(check) & (check > 0.0)):
                raise ValueError("sigma must be finite and strictly positive.")
            x = shifted / sd
            u_mat = xw / sd[:, None]
            half_logdet_d = jnp.sum(jnp.log(sd))
            return x, u_mat, half_logdet_d

        if sig_arr.shape != (n_obs, n_obs):
            raise ValueError(
                f"2D noise covariance must have shape ({n_obs}, {n_obs}); got {sig_arr.shape}."
            )
        chol_d = jnp.linalg.cholesky(0.5 * (sig_arr + sig_arr.T))
        x = jsl.solve_triangular(chol_d, shifted, lower=True)
        u_mat = jsl.solve_triangular(chol_d, xw, lower=True)
        half_logdet_d = jnp.sum(jnp.log(jnp.diag(chol_d)))
        return x, u_mat, half_logdet_d

    def standardize(self, resid: Any, sigma: Any) -> tuple[jax.Array, jax.Array]:
        """Return ``(x, U)`` for residual ``d - m_0`` and noise ``sigma``."""
        x, u_mat, _ = self._standardize_with_logdet(resid, sigma)
        return x, u_mat

    def whiten(self, resid: Any, sigma: Any) -> tuple[jax.Array, jax.Array]:
        """Return whitened residual ``u`` and ``½ ln det(D + X Λ_w Xᵀ)``."""
        x, u_mat, half_logdet_d = self._standardize_with_logdet(resid, sigma)
        if self.method == "cholesky":
            u_out, half_logdet_u = whiten_cholesky(x, u_mat)
        elif self.method == "qr":
            u_out, half_logdet_u = whiten_qr(x, u_mat)
        else:
            u_out, half_logdet_u = whiten_rank_one(x, u_mat)
        return u_out, half_logdet_d + half_logdet_u

    def loglike(self, resid: Any, sigma: Any) -> jax.Array:
        """Exact normalized Gaussian marginal log-likelihood ``ln p(d | m_0)``."""
        u_out, log_norm = self.whiten(resid, sigma)
        return -0.5 * (u_out @ u_out) - log_norm - 0.5 * u_out.size * math.log(2.0 * math.pi)

    def posterior(self, resid: Any, sigma: Any) -> tuple[jax.Array, jax.Array]:
        """Closed-form conditional posterior ``(mean, cov)`` of ``w`` given ``d``."""
        x, u_mat = self.standardize(resid, sigma)
        mean_std, cov_std = posterior(x, u_mat)
        root = self.prior_root
        mean_w = self.prior_mean + root @ mean_std
        cov_w = root @ cov_std @ root.T
        return mean_w, 0.5 * (cov_w + cov_w.T)

    def predict_posterior(
        self,
        resid: Any,
        sigma: Any,
        eval_design: Any | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Posterior ``(mean, cov)`` of a linear functional ``E @ w``."""
        e_mat = self.design if eval_design is None else jnp.asarray(eval_design, dtype=jnp.float64)
        mean_w, cov_w = self.posterior(resid, sigma)
        pred_mean = e_mat @ mean_w
        pred_cov = e_mat @ cov_w @ e_mat.T
        return pred_mean, 0.5 * (pred_cov + pred_cov.T)


# =============================================================================
# 2. Linear Design Helpers (Hierarchical Offsets & Harmonic Solar Quadratures)
# =============================================================================


def hierarchical_offset_design(
    series_indices: Any,
    n_series: int | None = None,
    box_activity_ratios: Any | None = None,
    *,
    global_prior_mean: float = 0.0,
    global_prior_sd: float = 200.0,
    lab_prior_sd: float = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build ``(design, prior_mean, prior_sd)`` for ``Δ_global * r_c + δ_{lab,s}``.

    Matches ``ticktack 2.0`` §1.1 & §3.1: one shared global ambient baseline
    ``Δ_global ~ N(global_prior_mean, global_prior_sd²)`` scaled by each box's
    steady-state specific-activity ratio ``r_c = (y_{ss,c}/M_{0,c}) / R_ref``
    (preserving ``Brehm21``'s ``4.38‰`` inter-hemispheric gradient) plus
    independent per-series inter-laboratory offsets ``δ_{lab,s} ~ N(0, lab_prior_sd²)``.
    """
    idx = jnp.asarray(series_indices, dtype=jnp.int32).reshape(-1)
    n_obs = idx.shape[0]
    if n_series is None:
        c_idx = _concrete(idx)
        n_series = int(onp.max(c_idx)) + 1 if (c_idx is not None and c_idx.size > 0) else 1

    if box_activity_ratios is None:
        global_col = jnp.ones((n_obs, 1), dtype=jnp.float64)
    else:
        global_col = jnp.broadcast_to(
            jnp.asarray(box_activity_ratios, dtype=jnp.float64), (n_obs,)
        )[:, None]

    if lab_prior_sd > 0.0 and n_series > 0:
        lab_cols = jax.nn.one_hot(idx, n_series, dtype=jnp.float64)
        design = jnp.concatenate([global_col, lab_cols], axis=1)
        mean = jnp.concatenate(
            [
                jnp.asarray([global_prior_mean], dtype=jnp.float64),
                jnp.zeros((n_series,), dtype=jnp.float64),
            ]
        )
        sd = jnp.concatenate(
            [
                jnp.asarray([global_prior_sd], dtype=jnp.float64),
                jnp.full((n_series,), lab_prior_sd, dtype=jnp.float64),
            ]
        )
    else:
        design = global_col
        mean = jnp.asarray([global_prior_mean], dtype=jnp.float64)
        sd = jnp.asarray([global_prior_sd], dtype=jnp.float64)
    return design, mean, sd


def _normalize_windows(
    t_starts: Any,
    t_ends: Any | None,
    sub_weights: Any | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Normalize seasonal sub-windows to ``(N_d, M_sub)`` ``float64`` arrays."""
    starts = jnp.asarray(t_starts, dtype=jnp.float64)
    ends = starts if t_ends is None else jnp.asarray(t_ends, dtype=jnp.float64)
    if starts.ndim == 1:
        starts = starts[:, None]
        ends = ends[:, None]
    elif starts.ndim != 2 or ends.shape != starts.shape:
        raise ValueError(
            "t_starts and t_ends must be matching 1D (N_d,) or 2D (N_d, M_sub) arrays; "
            f"got {starts.shape} and {ends.shape}."
        )
    n_obs, m_sub = starts.shape
    if sub_weights is None:
        weights = jnp.full((n_obs, m_sub), 1.0 / m_sub, dtype=jnp.float64)
    else:
        w_raw = jnp.broadcast_to(
            jnp.asarray(sub_weights, dtype=jnp.float64), (n_obs, m_sub)
        )
        w_sum = jnp.sum(w_raw, axis=-1, keepdims=True)
        weights = w_raw / jnp.where(w_sum > 0.0, w_sum, 1.0)
    c_s, c_e = _concrete(starts), _concrete(ends)
    if c_s is not None and c_e is not None and onp.any(c_e < c_s):
        raise ValueError("Seasonal windows require t_ends >= t_starts.")
    return starts, ends, weights


def _build_obs_matrix(
    n_obs: int,
    n_boxes: int,
    *,
    box_indices: Any = 0,
    obs_scale: Any = 1.0,
    obs_matrix: Any | None = None,
) -> jax.Array:
    """Build ``(N_d, n)`` observation matrix ``C_obs`` mapping ``δy(t) -> Δ¹⁴C(t)``."""
    if obs_matrix is not None:
        c_mat = jnp.asarray(obs_matrix, dtype=jnp.float64)
        if c_mat.shape != (n_obs, n_boxes):
            raise ValueError(
                f"obs_matrix must have shape ({n_obs}, {n_boxes}); got {c_mat.shape}."
            )
        return c_mat
    idx = jnp.broadcast_to(jnp.asarray(box_indices, dtype=jnp.int32), (n_obs,))
    scale_arr = jnp.asarray(obs_scale, dtype=jnp.float64)
    if scale_arr.ndim == 1 and scale_arr.shape[0] == n_boxes and n_boxes != n_obs:
        row_scale = scale_arr[idx]
    else:
        row_scale = jnp.broadcast_to(scale_arr, (n_obs,))
    return jax.nn.one_hot(idx, n_boxes, dtype=jnp.float64) * row_scale[:, None]


def _solve_complex_resolvents(
    A: jax.Array,
    b_eff: jax.Array,
    omegas: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Solve ``(i ω_j I_n - A) z_j = b_eff`` directly in ``complex128``.

    Never forms ``A² + ω² I_n`` (which would square ``κ(A) ~ 1.3 × 10⁵`` to
    ``~ 1.7 × 10¹⁰``). Returns ``(U, V)`` each of shape ``(n, r)`` with
    ``z_j = u_j + i v_j``.
    """
    n = A.shape[0]
    eye_c = jnp.eye(n, dtype=jnp.complex128)
    a_c = A.astype(jnp.complex128)
    b_c = b_eff.astype(jnp.complex128)

    def _one_freq(w: jax.Array) -> jax.Array:
        return jnp.linalg.solve(1j * w * eye_c - a_c, b_c)

    z_all = jax.vmap(_one_freq)(omegas.astype(jnp.float64))  # (r, n)
    return jnp.real(z_all).T, jnp.imag(z_all).T


def harmonic_solar_design(
    A: Any,
    b: Any,
    omegas: Any,
    t_starts: Any,
    t_ends: Any | None = None,
    sub_weights: Any | None = None,
    *,
    alpha_unit: float = 1.0,
    t_ref: float = 0.0,
    box_indices: Any = 0,
    obs_scale: Any = 1.0,
    obs_matrix: Any | None = None,
) -> jax.Array:
    """Exact seasonal-binned design matrix for eternal harmonic solar modulation.

    For ``δq_harm(t) = Σ_m [c_{1,m} cos ω_m(t - t_ref) + c_{2,m} sin ω_m(t - t_ref)]``,
    returns ``H`` of shape ``(N_d, 2 * M_harm)`` ordered
    ``[cos_1, sin_1, ..., cos_M, sin_M]`` using the unsquared complex resolvent
    and exact ``sinc`` sub-window integration (zero burn-in).
    """
    a_mat = jnp.asarray(A, dtype=jnp.float64)
    b_eff = alpha_unit * jnp.asarray(b, dtype=jnp.float64).reshape(-1)
    w_vec = jnp.asarray(omegas, dtype=jnp.float64).reshape(-1)
    starts, ends, weights = _normalize_windows(t_starts, t_ends, sub_weights)
    n_obs = starts.shape[0]
    n_boxes = a_mat.shape[0]
    c_obs = _build_obs_matrix(
        n_obs, n_boxes, box_indices=box_indices, obs_scale=obs_scale, obs_matrix=obs_matrix
    )
    if w_vec.size == 0:
        return jnp.zeros((n_obs, 0), dtype=jnp.float64)

    u_mat, v_mat = _solve_complex_resolvents(a_mat, b_eff, w_vec)  # (n, M)
    u_obs = c_obs @ u_mat  # (N_d, M)
    v_obs = c_obs @ v_mat  # (N_d, M)

    durations = ends - starts  # (N_d, M_sub)
    mids = 0.5 * (starts + ends) - t_ref  # (N_d, M_sub)
    # jnp.sinc(x) = sin(π x) / (π x), so sinc_rad(y) = jnp.sinc(y / π)
    sinc_factor = jnp.sinc(0.5 * durations[:, :, None] * w_vec[None, None, :] / jnp.pi)
    cos_avg = jnp.sum(
        weights[:, :, None] * sinc_factor * jnp.cos(mids[:, :, None] * w_vec[None, None, :]),
        axis=1,
    )
    sin_avg = jnp.sum(
        weights[:, :, None] * sinc_factor * jnp.sin(mids[:, :, None] * w_vec[None, None, :]),
        axis=1,
    )
    # Response to cos θ_m: u_m cos θ_m - v_m sin θ_m
    # Response to sin θ_m: v_m cos θ_m + u_m sin θ_m
    col_cos = u_obs * cos_avg - v_obs * sin_avg  # (N_d, M)
    col_sin = v_obs * cos_avg + u_obs * sin_avg  # (N_d, M)
    return jnp.stack([col_cos, col_sin], axis=-1).reshape(n_obs, 2 * w_vec.size)


# =============================================================================
# 3. tinygp >= 0.3 Spectral Densities & Continuous SDE Extraction
# =============================================================================


def _quasisep_mat_to_dense(mat: Any) -> jax.Array:
    """Recursively convert ``tinygp`` ``Block`` (including nested ``Sum`` blocks) to dense."""
    if isinstance(mat, Block):
        sub = [_quasisep_mat_to_dense(b) for b in mat.blocks]
        return jsl.block_diag(*sub)
    return jnp.asarray(mat, dtype=jnp.float64)


def _try_extract_sde(
    kernel: tgk.Kernel,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array] | None:
    """Extract continuous-time LTI SDE ``(F_gp, Q_sde, P_inf, h_gp)`` if Markovian.

    Supports both ``tinygp.kernels.quasisep.*`` (``Exp``, ``Matern32``,
    ``Matern52``, ``SHO``, ``Cosine``, ``Celerite``, ``CARMA``, ``Scale``,
    ``Wrapper``, ``Sum``, ``Product``) and ``tinygp.kernels.*`` stationary
    counterparts (``Exp``, ``Matern32``, ``Matern52``, ``Cosine``, ``Constant``-scaled
    ``Product``, ``Sum``, and Markov ``Product``). Returns ``None`` for
    non-quasiseparable kernels such as ``ExpSquared`` or ``RationalQuadratic``.
    """
    if isinstance(kernel, tgq.Quasisep):
        f_gp = _quasisep_mat_to_dense(kernel.design_matrix())
        p_inf = _quasisep_mat_to_dense(kernel.stationary_covariance())
        p_inf = 0.5 * (p_inf + p_inf.T)
        h_gp = jnp.asarray(
            kernel.observation_model(jnp.asarray(0.0, dtype=jnp.float64)),
            dtype=jnp.float64,
        ).reshape(-1)
        q_sde = -(f_gp @ p_inf + p_inf @ f_gp.T)
        q_sde = 0.5 * (q_sde + q_sde.T)
        return f_gp, q_sde, p_inf, h_gp

    if isinstance(kernel, tgk.Exp):
        q_eq = tgq.Exp(scale=jnp.asarray(kernel.scale, dtype=jnp.float64))
        return _try_extract_sde(q_eq)

    if isinstance(kernel, tgk.Matern32):
        q_eq = tgq.Matern32(scale=jnp.asarray(kernel.scale, dtype=jnp.float64))
        return _try_extract_sde(q_eq)

    if isinstance(kernel, tgk.Matern52):
        q_eq = tgq.Matern52(scale=jnp.asarray(kernel.scale, dtype=jnp.float64))
        return _try_extract_sde(q_eq)

    if isinstance(kernel, tgk.Cosine):
        q_eq = tgq.Cosine(scale=jnp.asarray(kernel.scale, dtype=jnp.float64))
        return _try_extract_sde(q_eq)

    if isinstance(kernel, tgk.Sum):
        sde1 = _try_extract_sde(kernel.kernel1)
        sde2 = _try_extract_sde(kernel.kernel2)
        if sde1 is None or sde2 is None:
            return None
        f1, q1, p1, h1 = sde1
        f2, q2, p2, h2 = sde2
        return (
            jsl.block_diag(f1, f2),
            jsl.block_diag(q1, q2),
            jsl.block_diag(p1, p2),
            jnp.concatenate([h1, h2]),
        )

    if isinstance(kernel, tgk.Product):
        if isinstance(kernel.kernel1, tgk.Constant):
            sub = _try_extract_sde(kernel.kernel2)
            if sub is None:
                return None
            f2, q2, p2, h2 = sub
            c = jnp.asarray(kernel.kernel1.value, dtype=jnp.float64)
            return f2, c * q2, c * p2, h2
        if isinstance(kernel.kernel2, tgk.Constant):
            sub = _try_extract_sde(kernel.kernel1)
            if sub is None:
                return None
            f1, q1, p1, h1 = sub
            c = jnp.asarray(kernel.kernel2.value, dtype=jnp.float64)
            return f1, c * q1, c * p1, h1
        sde1 = _try_extract_sde(kernel.kernel1)
        sde2 = _try_extract_sde(kernel.kernel2)
        if sde1 is not None and sde2 is not None:
            f1, _, p1, h1 = sde1
            f2, _, p2, h2 = sde2
            f_prod = jnp.kron(f1, jnp.eye(f2.shape[0], dtype=jnp.float64)) + jnp.kron(
                jnp.eye(f1.shape[0], dtype=jnp.float64), f2
            )
            p_prod = jnp.kron(p1, p2)
            p_prod = 0.5 * (p_prod + p_prod.T)
            q_prod = -(f_prod @ p_prod + p_prod @ f_prod.T)
            q_prod = 0.5 * (q_prod + q_prod.T)
            h_prod = jnp.kron(h1, h2)
            return f_prod, q_prod, p_prod, h_prod

    return None


def _try_scale_jax(kernel: tgk.Kernel) -> jax.Array | None:
    """Extract a smooth JAX correlation timescale (yr) without concrete branching."""
    if isinstance(kernel, tgk.Constant):
        return None
    if isinstance(kernel, tgq.Scale):
        return _try_scale_jax(kernel.kernel)
    if isinstance(kernel, tgq.Wrapper):
        return _try_scale_jax(kernel.kernel)
    if isinstance(kernel, tgq.SHO):
        w0 = jnp.maximum(jnp.asarray(kernel.omega, dtype=jnp.float64), 1e-6)
        qual = jnp.maximum(jnp.asarray(kernel.quality, dtype=jnp.float64), 1e-6)
        return jnp.hypot(2.0 * qual / w0, 2.0 * math.pi / w0)
    if isinstance(kernel, (tgk.Sum, tgq.Sum)):
        s1 = _try_scale_jax(kernel.kernel1)
        s2 = _try_scale_jax(kernel.kernel2)
        if s1 is not None and s2 is not None:
            return jnp.hypot(s1, s2)
        return s1 if s1 is not None else s2
    if isinstance(kernel, (tgk.Product, tgq.Product)):
        s1 = _try_scale_jax(kernel.kernel1)
        s2 = _try_scale_jax(kernel.kernel2)
        if s1 is not None and s2 is not None:
            return (s1 * s2) / jnp.hypot(s1, s2)
        return s1 if s1 is not None else s2
    if hasattr(kernel, "scale"):
        return jnp.abs(jnp.asarray(kernel.scale, dtype=jnp.float64))
    return None


def _characteristic_scale(kernel: tgk.Kernel) -> jax.Array:
    """Return a smooth positive JAX characteristic timescale (yr) for quadrature."""
    sc = _try_scale_jax(kernel)
    if sc is None:
        return jnp.asarray(2.0, dtype=jnp.float64)
    return jnp.hypot(sc, 1e-4)


# Precomputed 64-node Gauss--Legendre quadrature on [-1, 1] for log-transformed RationalQuadratic
_GLEG64_X, _GLEG64_W = onp.polynomial.legendre.leggauss(64)
# Precomputed 96-node Gauss--Legendre quadrature on [-1, 1] for general Product fallback
_GLEG96_X, _GLEG96_W = onp.polynomial.legendre.leggauss(96)
# Precomputed 128-node Gauss--Legendre quadrature on [-1, 1] for spectral Σ_{y,0}
_GLEG128_X, _GLEG128_W = onp.polynomial.legendre.leggauss(128)


def _scaled_bessel_ie_series(order_max: int, z: jax.Array) -> jax.Array:
    """Compute ``exp(-z) I_m(z)`` for ``m = 0..order_max`` smoothly for ``z >= 0``."""
    # Power series I_m(z) = Σ_{n=0}^28 (z/2)^{2n + m} / (n! (n + m)!)
    m = jnp.arange(order_max + 1, dtype=jnp.float64)[:, None]
    n = jnp.arange(28, dtype=jnp.float64)[None, :]
    safe_z = jnp.maximum(z, 1e-30)
    log_half_z = jnp.log(0.5 * safe_z)
    log_terms = (
        (2.0 * n + m) * log_half_z
        - jax.scipy.special.gammaln(n + 1.0)
        - jax.scipy.special.gammaln(n + m + 1.0)
        - z
    )
    vals = jnp.sum(jnp.exp(log_terms), axis=1)
    # At z == 0 exactly: I_0(0) = 1, I_{m>0}(0) = 0
    zero_vec = jnp.zeros(order_max + 1, dtype=jnp.float64).at[0].set(1.0)
    return jnp.where(z > 1e-30, vals, zero_vec)


def spectral_density(kernel: tgk.Kernel, omega: Any) -> jax.Array:
    """Evaluate Wiener--Khinchin spectral density ``S_k(ω) = ∫_{-∞}^∞ k(τ) e^{-iωτ} dτ``.

    Supports stationary ``tinygp >= 0.3`` kernels: ``Exp``, ``Matern32``,
    ``Matern52``, ``ExpSquared``, ``RationalQuadratic``, ``SHO``, ``Celerite``,
    ``CARMA``, damped harmonic ``Cosine``/``ExpSineSquared`` products, and
    arbitrary ``Sum``/``Product``/``Scale``/``Constant`` compositions.
    """
    w = jnp.asarray(omega, dtype=jnp.float64)
    w2 = jnp.square(w)

    # 1. Elementary stationary & quasisep closed forms (fast & zero matrix solves)
    if isinstance(kernel, (tgk.Exp, tgq.Exp)):
        ell = jnp.asarray(kernel.scale, dtype=jnp.float64)
        sig2 = (
            jnp.square(jnp.asarray(kernel.sigma, dtype=jnp.float64))
            if isinstance(kernel, tgq.Exp)
            else 1.0
        )
        return sig2 * (2.0 * ell) / (1.0 + jnp.square(ell) * w2)

    if isinstance(kernel, (tgk.Matern32, tgq.Matern32)):
        ell = jnp.asarray(kernel.scale, dtype=jnp.float64)
        sig2 = (
            jnp.square(jnp.asarray(kernel.sigma, dtype=jnp.float64))
            if isinstance(kernel, tgq.Matern32)
            else 1.0
        )
        f = math.sqrt(3.0) / ell
        f2 = jnp.square(f)
        return sig2 * (4.0 * f2 * f) / jnp.square(f2 + w2)

    if isinstance(kernel, (tgk.Matern52, tgq.Matern52)):
        ell = jnp.asarray(kernel.scale, dtype=jnp.float64)
        sig2 = (
            jnp.square(jnp.asarray(kernel.sigma, dtype=jnp.float64))
            if isinstance(kernel, tgq.Matern52)
            else 1.0
        )
        f = math.sqrt(5.0) / ell
        f2 = jnp.square(f)
        return sig2 * ((16.0 / 3.0) * f2 * f2 * f) / ((f2 + w2) ** 3)

    if isinstance(kernel, tgk.ExpSquared):
        ell = jnp.asarray(kernel.scale, dtype=jnp.float64)
        return math.sqrt(2.0 * math.pi) * ell * jnp.exp(-0.5 * jnp.square(ell) * w2)

    if isinstance(kernel, tgk.RationalQuadratic):
        ell = jnp.asarray(kernel.scale, dtype=jnp.float64)
        alpha = jnp.asarray(kernel.alpha, dtype=jnp.float64)
        # Log-transformed gamma scale mixture u = exp(t) on t ∈ [-32, 4.5] (double-exponential
        # upper decay -e^t and smooth C^∞ integrand across all α > 0, ω >= 0):
        t_lo, t_hi = -32.0, 4.5
        t_nodes = 0.5 * (t_hi - t_lo) * jnp.asarray(_GLEG64_X, dtype=jnp.float64) + 0.5 * (
            t_hi + t_lo
        )
        dt_weights = 0.5 * (t_hi - t_lo) * jnp.asarray(_GLEG64_W, dtype=jnp.float64)
        log_pref = (
            0.5 * math.log(2.0 * math.pi)
            + jnp.log(ell)
            + 0.5 * jnp.log(alpha)
            - jax.scipy.special.gammaln(alpha)
        )
        log_integrand = (
            log_pref
            + ((alpha - 0.5) * t_nodes - jnp.exp(t_nodes))[:, None]
            - 0.5 * alpha * jnp.square(ell) * jnp.exp(-t_nodes)[:, None] * w2.reshape(1, -1)
        )
        return jnp.sum(dt_weights[:, None] * jnp.exp(log_integrand), axis=0).reshape(
            w.shape
        )

    if isinstance(kernel, tgq.SHO):
        w0 = jnp.asarray(kernel.omega, dtype=jnp.float64)
        qual = jnp.asarray(kernel.quality, dtype=jnp.float64)
        sig2 = jnp.square(jnp.asarray(kernel.sigma, dtype=jnp.float64))
        w0_2 = jnp.square(w0)
        num = 2.0 * sig2 * w0_2 * w0 / qual
        den = jnp.square(w0_2 - w2) + w2 * w0_2 / jnp.square(qual)
        return num / den

    if isinstance(kernel, tgq.Scale):
        return jnp.asarray(kernel.scale, dtype=jnp.float64) * spectral_density(
            kernel.kernel, w
        )

    if isinstance(kernel, tgq.Wrapper):
        return spectral_density(kernel.kernel, w)

    if isinstance(kernel, (tgk.Sum, tgq.Sum)):
        return spectral_density(kernel.kernel1, w) + spectral_density(kernel.kernel2, w)

    if isinstance(kernel, (tgk.Product, tgq.Product)):
        k1, k2 = kernel.kernel1, kernel.kernel2
        if isinstance(k1, tgk.Constant):
            return jnp.asarray(k1.value, dtype=jnp.float64) * spectral_density(k2, w)
        if isinstance(k2, tgk.Constant):
            return jnp.asarray(k2.value, dtype=jnp.float64) * spectral_density(k1, w)
        if isinstance(k1, (tgk.Cosine, tgq.Cosine)):
            period = jnp.asarray(k1.scale, dtype=jnp.float64)
            sig2 = (
                jnp.square(jnp.asarray(k1.sigma, dtype=jnp.float64))
                if isinstance(k1, tgq.Cosine)
                else 1.0
            )
            w0 = 2.0 * math.pi / period
            return 0.5 * sig2 * (
                spectral_density(k2, w - w0) + spectral_density(k2, w + w0)
            )
        if isinstance(k2, (tgk.Cosine, tgq.Cosine)):
            period = jnp.asarray(k2.scale, dtype=jnp.float64)
            sig2 = (
                jnp.square(jnp.asarray(k2.sigma, dtype=jnp.float64))
                if isinstance(k2, tgq.Cosine)
                else 1.0
            )
            w0 = 2.0 * math.pi / period
            return 0.5 * sig2 * (
                spectral_density(k1, w - w0) + spectral_density(k1, w + w0)
            )
        if isinstance(k1, tgk.ExpSineSquared) or isinstance(k2, tgk.ExpSineSquared):
            ess, base = (k1, k2) if isinstance(k1, tgk.ExpSineSquared) else (k2, k1)
            period = jnp.asarray(ess.scale, dtype=jnp.float64)
            gamma = jnp.asarray(ess.gamma, dtype=jnp.float64)
            w0 = 2.0 * math.pi / period
            ie = _scaled_bessel_ie_series(12, 0.5 * gamma)
            out = ie[0] * spectral_density(base, w)
            for m in range(1, 13):
                out = out + ie[m] * (
                    spectral_density(base, w - m * w0)
                    + spectral_density(base, w + m * w0)
                )
            return out
        if isinstance(k1, tgk.ExpSquared) and isinstance(k2, tgk.ExpSquared):
            l1 = jnp.asarray(k1.scale, dtype=jnp.float64)
            l2 = jnp.asarray(k2.scale, dtype=jnp.float64)
            l_eff = 1.0 / jnp.sqrt(1.0 / jnp.square(l1) + 1.0 / jnp.square(l2))
            return math.sqrt(2.0 * math.pi) * l_eff * jnp.exp(-0.5 * jnp.square(l_eff) * w2)

        # General Markov SDE product (e.g. Matern32 * SHO, Matern52 * Matern32):
        sde = _try_extract_sde(kernel)
        if sde is not None:
            f_gp, q_sde, p_inf, h_gp = sde
            return _sde_spectral_density(f_gp, q_sde, p_inf, h_gp, w)

        # General stationary product fallback (e.g. ExpSquared * Matern32):
        return _quadrature_spectral_density(kernel, w)

    # General Quasisep fallback (Celerite, CARMA)
    sde = _try_extract_sde(kernel)
    if sde is not None and not isinstance(kernel, (tgk.Cosine, tgq.Cosine)):
        f_gp, q_sde, p_inf, h_gp = sde
        return _sde_spectral_density(f_gp, q_sde, p_inf, h_gp, w)

    if isinstance(kernel, (tgk.Cosine, tgq.Cosine)):
        raise TypeError(
            "Undamped Cosine has a pure Dirac line spectrum; multiply by a "
            "damping kernel (e.g. Matern32 * Cosine or ExpSquared * Cosine), "
            "use tinygp.kernels.quasisep.SHO, or use harmonic_solar_design."
        )

    raise TypeError(
        f"Unsupported kernel type {type(kernel).__name__!r} for spectral_density."
    )


def _sde_spectral_density(
    f_gp: jax.Array,
    q_sde: jax.Array,
    p_inf: jax.Array,
    h_gp: jax.Array,
    omega: jax.Array,
) -> jax.Array:
    """Evaluate ``S(ω) = hᵀ (iωI - F)⁻¹ Q_sde (-iωI - Fᵀ)⁻¹ h`` for any Hurwitz SDE."""
    del p_inf
    d = f_gp.shape[0]
    eye_c = jnp.eye(d, dtype=jnp.complex128)
    f_c = f_gp.astype(jnp.complex128)
    h_c = h_gp.astype(jnp.complex128)
    q_c = q_sde.astype(jnp.complex128)
    w_flat = omega.reshape(-1)

    def _at_w(w_val: jax.Array) -> jax.Array:
        # Solve (i ω I - Fᵀ) r = h => r* = hᵀ (-i ω I - Fᵀ)⁻¹?
        # Wait: ((i ω I - F)⁻¹)ᵀ h: solve (-i ω I - Fᵀ) g = h => g* = hᵀ (i ω I - F)⁻¹
        g = jnp.linalg.solve(-1j * w_val * eye_c - f_c.T, h_c)
        return jnp.maximum(jnp.real(jnp.vdot(g, q_c @ g)), 0.0)

    return jax.vmap(_at_w)(w_flat).reshape(omega.shape)


def _quadrature_spectral_density(kernel: tgk.Kernel, omega: jax.Array) -> jax.Array:
    """Smooth Fourier-cosine quadrature ``2 ∫_0^∞ k(τ) cos(ω τ) dτ`` for general products."""
    ell_char = _characteristic_scale(kernel)
    tau_max = 18.0 * ell_char
    xi = 0.5 * (jnp.asarray(_GLEG96_X, dtype=jnp.float64) + 1.0)
    wt = 0.5 * jnp.asarray(_GLEG96_W, dtype=jnp.float64) * tau_max
    taus = xi * tau_max
    k_vals = jax.vmap(lambda t: kernel.evaluate(jnp.asarray(0.0, dtype=jnp.float64), t))(
        taus
    )
    w_flat = omega.reshape(-1)
    spec = 2.0 * jnp.sum(
        (wt * k_vals)[:, None] * jnp.cos(taus[:, None] * w_flat[None, :]),
        axis=0,
    )
    return jnp.maximum(spec, 0.0).reshape(omega.shape)


# =============================================================================
# 4. Exact Pre-a Stationary Covariance Σ_{y,0}(φ)
# =============================================================================


def stationary_cbm_gp_cov(
    A: Any,
    b: Any,
    kernel: tgk.Kernel,
    *,
    alpha_unit: float = 1.0,
    dtau0: float = 1.0 / 16.0,
    n_squarings: int = 24,
    n_quad: int = 128,
    jitter: float = 1e-13,
    return_full_state: bool = False,
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
    """Compute exact stationary CBM covariance ``Σ_{y,0}(φ)`` under eternal GP forcing.

    - **Markov / ``quasisep`` kernels** (``Matern32``, ``Matern52``, ``SHO``,
      ``Exp``, ``Cosine``, and sums/products): extracts continuous SDE
      ``(F_GP, Q_SDE, P_∞, h_GP)``, computes the exact 1-step joint covariance
      at ``Δτ_0 = 1/16 yr`` via a ``2(n + d_GP)`` Van Loan matrix exponential,
      and runs ``n_squarings = 24`` discrete Smith doublings
      (``Σ_{k+1} = Σ_k + Φ_k Σ_k Φ_kᵀ, Φ_{k+1} = Φ_k²``, spanning
      ``2²⁴ / 16 = 1.048576 × 10⁶ yr`` without continuous Hamiltonian overflow).
    - **Non-quasiseparable kernels** (``ExpSquared``, ``RationalQuadratic``,
      etc.): integrates ``Σ_{y,0} = π⁻¹ ∫_0^∞ Re[z(ω) z(ω)*] S_k(ω) dω`` via
      128-point Gauss--Legendre quadrature under an analytic ``sinh``-log map
      that resolves both the narrow ``λ_14 ≈ 1.21 × 10⁻⁴ yr⁻¹`` radiocarbon
      decay peak and the high-frequency GP passband to machine precision.

    Returns ``(Sigma_y0, Sigma_y0_root)`` (plus ``Sigma_joint_inf`` when
    ``return_full_state=True`` and ``kernel`` is SDE-representable).
    """
    a_mat = jnp.asarray(A, dtype=jnp.float64)
    b_eff = alpha_unit * jnp.asarray(b, dtype=jnp.float64).reshape(-1)
    n = a_mat.shape[0]

    sde = _try_extract_sde(kernel)
    if sde is not None:
        f_gp, q_sde, p_inf, h_gp = sde
        d_gp = f_gp.shape[0]
        d_tot = n + d_gp
        f_joint = jnp.block(
            [
                [a_mat, jnp.outer(b_eff, h_gp)],
                [jnp.zeros((d_gp, n), dtype=jnp.float64), f_gp],
            ]
        )
        q_joint = jnp.block(
            [
                [
                    jnp.zeros((n, n), dtype=jnp.float64),
                    jnp.zeros((n, d_gp), dtype=jnp.float64),
                ],
                [jnp.zeros((d_gp, n), dtype=jnp.float64), q_sde],
            ]
        )
        m_vl = (
            jnp.block(
                [
                    [f_joint, q_joint],
                    [jnp.zeros((d_tot, d_tot), dtype=jnp.float64), -f_joint.T],
                ]
            )
            * dtau0
        )
        e_vl = jsl.expm(m_vl)
        phi_0 = e_vl[:d_tot, :d_tot]
        sigma_0 = e_vl[:d_tot, d_tot:] @ phi_0.T
        sigma_0 = 0.5 * (sigma_0 + sigma_0.T)

        def _smith_step(
            carry: tuple[jax.Array, jax.Array], _: None
        ) -> tuple[tuple[jax.Array, jax.Array], None]:
            sig, phi = carry
            sig_next = sig + phi @ sig @ phi.T
            sig_next = 0.5 * (sig_next + sig_next.T)
            phi_next = phi @ phi
            return (sig_next, phi_next), None

        (sigma_inf, phi_inf), _ = jax.lax.scan(
            _smith_step, (sigma_0, phi_0), None, length=n_squarings
        )
        # Include asymptotic initial GP covariance term Φ_24 diag(0, P_∞) Φ_24ᵀ:
        # identically ~10⁻¹¹⁰ for strictly damped SDEs and exact for undamped Cosine!
        init_gp = jsl.block_diag(jnp.zeros((n, n), dtype=jnp.float64), p_inf)
        sigma_inf = sigma_inf + phi_inf @ init_gp @ phi_inf.T
        sigma_inf = 0.5 * (sigma_inf + sigma_inf.T)
        sigma_y0 = sigma_inf[:n, :n]
        sigma_y0 = 0.5 * (sigma_y0 + sigma_y0.T)
        root_y0 = _safe_psd_root(sigma_y0, jitter=jitter)
        if return_full_state:
            return sigma_y0, root_y0, sigma_inf
        return sigma_y0, root_y0

    if return_full_state:
        raise ValueError(
            f"Kernel {type(kernel).__name__!r} does not have a finite-dimensional "
            "Markov SDE state space; full joint SDE state is unavailable."
        )

    # Three-zone composite Gauss-Legendre spectral quadrature for non-quasiseparable kernels:
    # - Zone 1: [0, 0.05] via sinh map (c0 = 1e-6) resolving slow radiocarbon λ_14 ≈ 1.21e-4 yr⁻¹
    # - Zone 2: [0.05, 4.0] linear GL resolving narrow solar-cycle harmonics (11 yr, 5.5 yr, ...)
    # - Zone 3: [4.0, w_max] via shifted sinh map resolving high-frequency tails ω >> 1/ℓ
    if n_quad == 128:
        gl_x, gl_w = _GLEG128_X, _GLEG128_W
    else:
        gl_x, gl_w = onp.polynomial.legendre.leggauss(n_quad)
    xi = 0.5 * (jnp.asarray(gl_x, dtype=jnp.float64) + 1.0)
    w_xi = 0.5 * jnp.asarray(gl_w, dtype=jnp.float64)

    ell_char = _characteristic_scale(kernel)
    w_max = jnp.hypot(250.0, 28.0 / jnp.hypot(ell_char, 1e-3))

    # Zone 1: [0, 0.05]
    c0_lo = 1e-6
    w_z1_hi = 0.05
    alpha_lo = math.asinh(w_z1_hi / c0_lo)
    om_z1 = c0_lo * jnp.sinh(alpha_lo * xi)
    dw_z1 = w_xi * (c0_lo * alpha_lo * jnp.cosh(alpha_lo * xi))

    # Zone 2: [0.05, 4.0]
    w_z2_hi = 4.0
    om_z2 = w_z1_hi + (w_z2_hi - w_z1_hi) * xi
    dw_z2 = w_xi * (w_z2_hi - w_z1_hi)

    # Zone 3: [4.0, w_max]
    c0_hi = 0.2
    alpha_hi = jnp.arcsinh((w_max - w_z2_hi) / c0_hi)
    om_z3 = w_z2_hi + c0_hi * jnp.sinh(alpha_hi * xi)
    dw_z3 = w_xi * (c0_hi * alpha_hi * jnp.cosh(alpha_hi * xi))

    omegas = jnp.concatenate([om_z1, om_z2, om_z3])
    dw_total = jnp.concatenate([dw_z1, dw_z2, dw_z3])

    u_mat, v_mat = _solve_complex_resolvents(a_mat, b_eff, omegas)  # (n, 3 * n_quad)
    s_vals = spectral_density(kernel, omegas)  # (3 * n_quad,)
    quad_weights = (dw_total * s_vals) / math.pi  # (3 * n_quad,)
    sigma_y0 = (u_mat * quad_weights[None, :]) @ u_mat.T + (
        v_mat * quad_weights[None, :]
    ) @ v_mat.T
    sigma_y0 = 0.5 * (sigma_y0 + sigma_y0.T)
    root_y0 = _safe_psd_root(sigma_y0, jitter=jitter)
    return sigma_y0, root_y0


# =============================================================================
# 5. Causal Solin & Särkkä (2020) Spectral GP Basis & Zero-Scan Projection
# =============================================================================


class SolinSarkkaBasis(NamedTuple):
    """Dirichlet Laplacian spectral GP basis on ``[a, b]`` (Solin & Särkkä 2020)."""

    omegas: jax.Array
    psi: jax.Array
    spectral_weights: jax.Array | None
    a: jax.Array
    b: jax.Array
    half_length: jax.Array


def solin_sarkka_basis(
    t: Any,
    a: Any,
    b: Any,
    n_modes: int,
    kernel: tgk.Kernel | None = None,
) -> SolinSarkkaBasis:
    """Evaluate Solin & Särkkä (2020) Laplace eigenfunctions ``ψ_j(t) = L⁻¹/² sin(ω_j(t - a))``.

    Parameters
    ----------
    t : array-like
        1D evaluation timestamps in ``[a, b]``.
    a, b : float
        Interval endpoints satisfying ``a < b``; ``L = ½(b - a)`` and
        ``ω_j = j π / (2 L)`` for ``j = 1, ..., n_modes``.
    n_modes : int
        Number of spectral modes ``r >= 1``.
    kernel : tinygp.kernels.Kernel, optional
        If provided, also evaluates ``spectral_weights = S_k(ω_j; φ)``.
    """
    if n_modes < 1:
        raise ValueError(f"n_modes must be >= 1; got {n_modes}.")
    t_arr = jnp.asarray(t, dtype=jnp.float64).reshape(-1)
    a_val = jnp.asarray(a, dtype=jnp.float64)
    b_val = jnp.asarray(b, dtype=jnp.float64)
    c_ab = _concrete(b_val - a_val)
    if c_ab is not None and float(c_ab) <= 0.0:
        raise ValueError(f"Interval requires b > a; got a={a}, b={b}.")
    half_len = 0.5 * (b_val - a_val)
    j_idx = jnp.arange(1, n_modes + 1, dtype=jnp.float64)
    omegas = j_idx * (math.pi / (2.0 * half_len))
    psi = jnp.sin((t_arr[:, None] - a_val) * omegas[None, :]) / jnp.sqrt(half_len)
    spec_w = spectral_density(kernel, omegas) if kernel is not None else None
    return SolinSarkkaBasis(
        omegas=omegas,
        psi=psi,
        spectral_weights=spec_w,
        a=a_val,
        b=b_val,
        half_length=half_len,
    )


def _psi1_matrix(x_mat: jax.Array) -> jax.Array:
    """Compute ``ψ_1(X) = ∫_0^1 exp(s X) ds`` via a ``2n × 2n`` Van Loan ``expm``.

    Exact for all ``X`` (including singular ``X = 0`` at instantaneous ``Δt = 0``)
    with zero ``(exp(X) - I) X⁻¹`` cancellation on the slow ``λ_14`` mode.
    """
    n = x_mat.shape[0]
    zeros = jnp.zeros((n, n), dtype=x_mat.dtype)
    eye = jnp.eye(n, dtype=x_mat.dtype)
    block = jnp.block([[x_mat, eye], [zeros, zeros]])
    return jsl.expm(block)[:n, n:]


def _seasonal_homogeneous_operator(
    a_mat: jax.Array,
    c_obs: jax.Array,
    starts: jax.Array,
    ends: jax.Array,
    weights: jax.Array,
    t_origin: jax.Array,
) -> jax.Array:
    """Compute ``W_hom(t_origin) = C_obs B_season[exp(A(t - t_origin))]`` ``(N_d, n)``."""
    durations = ends - starts  # (N_d, M_sub)
    offsets = starts - t_origin  # (N_d, M_sub)

    def _sub_prop(dur: jax.Array, off: jax.Array) -> jax.Array:
        return _psi1_matrix(a_mat * dur) @ jsl.expm(a_mat * off)

    # Vectorize across all (N_d * M_sub) sub-windows in a single batched vmap:
    n_obs, m_sub = starts.shape
    n_boxes = a_mat.shape[0]
    props = jax.vmap(_sub_prop)(durations.reshape(-1), offsets.reshape(-1)).reshape(
        n_obs, m_sub, n_boxes, n_boxes
    )
    avg_prop = jnp.sum(weights[:, :, None, None] * props, axis=1)  # (N_d, n, n)
    return jnp.einsum("dn,dnk->dk", c_obs, avg_prop)


class CausalGPProjection(NamedTuple):
    """Result of causal Solin--Särkkä GP projection through an autonomous CBM.

    Attributes
    ----------
    causal_basis_design : jax.Array
        ``W_causal Ψ_r`` ``(N_d, r)``, independent of GP kernel hyperparameters.
    hom_design_a : jax.Array
        ``W_hom(a) = B_season[exp(A(t - a))]`` ``(N_d, n)``.
    hom_design_anchor : jax.Array
        ``W_hom(t_anchor) = B_season[exp(A(t - t_anchor))]`` ``(N_d, n)`` used
        for propagating residual pre-history covariance.
    omegas : jax.Array
        Solin--Särkkä frequencies ``ω_j = j π / (2 L)`` ``(r,)``.
    a : jax.Array
        Left domain boundary ``a <= t_0``.
    b_bound : jax.Array
        Right domain boundary ``b_bound >= T``.
    t_anchor : jax.Array
        Anchor epoch (``t_0`` or ``a``) at which stationary covariance is matched.
    half_length : jax.Array
        Half-domain width ``L = ½(b_bound - a)``.
    spectral_weights : jax.Array | None
        ``S_k(ω_j; φ)`` ``(r,)`` when ``kernel`` was provided, else ``None``.
    sigma_y0 : jax.Array | None
        Exact eternal stationary CBM state covariance ``Σ_{y,0}(φ)`` ``(n, n)``.
    anchor_cov_root : jax.Array | None
        Square-root factor ``ΔΣ_y(t_anchor)¹/²`` ``(n, n)`` of the residual
        pre-history covariance at ``t_anchor``.
    whitened_design : jax.Array | None
        Combined unit-normal prior design
        ``[W_causal Ψ_r Λ¹/², W_hom(t_anchor) ΔΣ_y(t_anchor)¹/²]`` ``(N_d, r + n)``
        (or ``(N_d, r)`` if ``include_initial_cov=False``) when ``kernel`` is given.
    """

    causal_basis_design: jax.Array
    hom_design_a: jax.Array
    hom_design_anchor: jax.Array
    omegas: jax.Array
    a: jax.Array
    b_bound: jax.Array
    t_anchor: jax.Array
    half_length: jax.Array
    spectral_weights: jax.Array | None
    sigma_y0: jax.Array | None
    anchor_cov_root: jax.Array | None
    whitened_design: jax.Array | None

    def to_linear_marginal(
        self,
        *,
        nuisance_design: Any | None = None,
        nuisance_prior_mean: Any = 0.0,
        nuisance_prior_sd: Any | None = None,
        nuisance_prior_root: Any | None = None,
        method: Literal["cholesky", "qr", "rank_one"] = "cholesky",
    ) -> LinearMarginal:
        """Build a [`LinearMarginal`][ticktack.linear.LinearMarginal] over GP + initial state (+ optional nuisance terms)."""
        if self.whitened_design is None:
            raise ValueError(
                "Pass kernel=... to project_causal_gp before calling to_linear_marginal()."
            )
        n_gp_cols = self.whitened_design.shape[1]
        if nuisance_design is None:
            return LinearMarginal(
                self.whitened_design,
                prior_mean=0.0,
                prior_sd=1.0,
                method=method,
            )

        nuis_x = jnp.asarray(nuisance_design, dtype=jnp.float64)
        k_nuis = nuis_x.shape[1]
        if (nuisance_prior_sd is None) == (nuisance_prior_root is None):
            raise ValueError(
                "Provide exactly one of nuisance_prior_sd or nuisance_prior_root "
                "when nuisance_design is passed."
            )
        if nuisance_prior_sd is not None:
            sd_n = jnp.broadcast_to(
                jnp.asarray(nuisance_prior_sd, dtype=jnp.float64), (k_nuis,)
            )
            nuis_root = jnp.diag(sd_n)
        else:
            nuis_root = jnp.asarray(nuisance_prior_root, dtype=jnp.float64)
        nuis_mean = jnp.broadcast_to(
            jnp.asarray(nuisance_prior_mean, dtype=jnp.float64), (k_nuis,)
        )

        full_design = jnp.concatenate([self.whitened_design, nuis_x], axis=1)
        full_mean = jnp.concatenate(
            [jnp.zeros((n_gp_cols,), dtype=jnp.float64), nuis_mean]
        )
        full_root = jsl.block_diag(
            jnp.eye(n_gp_cols, dtype=jnp.float64), nuis_root
        )
        return LinearMarginal(
            full_design,
            prior_mean=full_mean,
            prior_root=full_root,
            method=method,
        )

    def posterior_production(
        self,
        t_eval: Any,
        resid: Any,
        sigma: Any,
        *,
        nuisance_design: Any | None = None,
        nuisance_prior_mean: Any = 0.0,
        nuisance_prior_sd: Any | None = None,
        nuisance_prior_root: Any | None = None,
        method: Literal["cholesky", "qr", "rank_one"] = "cholesky",
        return_cov: bool = False,
    ) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
        """Compute closed-form posterior production ``(mean, var[, cov])`` on ``t_eval``."""
        if self.spectral_weights is None:
            raise ValueError(
                "Pass kernel=... to project_causal_gp before calling posterior_production()."
            )
        lm = self.to_linear_marginal(
            nuisance_design=nuisance_design,
            nuisance_prior_mean=nuisance_prior_mean,
            nuisance_prior_sd=nuisance_prior_sd,
            nuisance_prior_root=nuisance_prior_root,
            method=method,
        )
        r_modes = self.omegas.shape[0]
        basis = solin_sarkka_basis(t_eval, self.a, self.b_bound, r_modes)
        phi_eval = basis.psi * jnp.sqrt(self.spectral_weights)[None, :]  # (N_eval, r)
        mean_all, cov_all = lm.posterior(resid, sigma)
        c_mean = mean_all[:r_modes]
        c_cov = cov_all[:r_modes, :r_modes]
        q_mean = phi_eval @ c_mean
        if return_cov:
            q_cov = phi_eval @ c_cov @ phi_eval.T
            q_cov = 0.5 * (q_cov + q_cov.T)
            return q_mean, jnp.diag(q_cov), q_cov
        q_var = jnp.sum((phi_eval @ c_cov) * phi_eval, axis=1)
        return q_mean, jnp.maximum(q_var, 0.0)


def project_causal_gp(
    A: Any,
    b: Any,
    t_starts: Any,
    t_ends: Any | None = None,
    sub_weights: Any | None = None,
    *,
    a: float | jax.Array | None = None,
    b_bound: float | jax.Array | None = None,
    pad: float | None = None,
    n_modes: int = 96,
    kernel: tgk.Kernel | None = None,
    alpha_unit: float = 1.0,
    box_indices: Any = 0,
    obs_scale: Any = 1.0,
    obs_matrix: Any | None = None,
    include_initial_cov: bool = True,
    initial_anchor: Literal["t0", "a"] = "t0",
    dtau0: float = 1.0 / 16.0,
    n_squarings: int = 24,
) -> CausalGPProjection:
    """Build the exact zero-scan causal Solin--Särkkä GP projection through CBM ``(A, b)``.

    Evaluates

    .. math::

        (W_{\\text{causal}}\\Psi_r)_{:, j} = \\frac{1}{\\sqrt{L}}\\,
        \\mathcal{B}_{\\text{season}}\\!\\Big[
            \\mathbf{v}_j\\cos\\omega_j(t - a)
            + \\mathbf{u}_j\\sin\\omega_j(t - a)
            - e^{A(t - a)}\\mathbf{v}_j
        \\Big],

    subtracting ``exp(A(t - a)) v_j`` so ``δy_GP(a) = 0`` identically (zero
    mirror-history leakage and zero ``2×`` initial-variance double-counting),
    and pairs it orthogonally with the exact pre-history stationary covariance
    ``Σ_{y,0}(φ)`` from [`stationary_cbm_gp_cov`][ticktack.linear.stationary_cbm_gp_cov].
    """
    if n_modes < 1:
        raise ValueError(f"n_modes must be >= 1; got {n_modes}.")
    a_mat = jnp.asarray(A, dtype=jnp.float64)
    b_eff = alpha_unit * jnp.asarray(b, dtype=jnp.float64).reshape(-1)
    n_boxes = a_mat.shape[0]

    starts, ends, weights = _normalize_windows(t_starts, t_ends, sub_weights)
    n_obs = starts.shape[0]
    c_obs = _build_obs_matrix(
        n_obs, n_boxes, box_indices=box_indices, obs_scale=obs_scale, obs_matrix=obs_matrix
    )

    t0_obs = jnp.min(starts)
    t_max_obs = jnp.max(ends)
    pad_val = 30.0 if pad is None else float(pad)

    a_val = (t0_obs - pad_val) if a is None else jnp.asarray(a, dtype=jnp.float64)
    b_val = (
        (t_max_obs + pad_val)
        if b_bound is None
        else jnp.asarray(b_bound, dtype=jnp.float64)
    )
    c_a, c_t0, c_b, c_tmax = (
        _concrete(a_val),
        _concrete(t0_obs),
        _concrete(b_val),
        _concrete(t_max_obs),
    )
    if c_a is not None and c_t0 is not None and float(c_a) > float(c_t0) + 1e-12:
        raise ValueError(
            f"Causal left boundary a={float(c_a)} must be <= earliest observation "
            f"t_starts={float(c_t0)}."
        )
    if c_b is not None and c_tmax is not None and float(c_b) <= float(c_tmax):
        raise ValueError(
            f"Right boundary b_bound={float(c_b)} must exceed latest observation "
            f"t_ends={float(c_tmax)}."
        )

    half_len = 0.5 * (b_val - a_val)
    j_idx = jnp.arange(1, n_modes + 1, dtype=jnp.float64)
    omegas = j_idx * (math.pi / (2.0 * half_len))

    # 1. Direct complex resolvent (i ω_j I_n - A)⁻¹ (α_unit b) = u_j + i v_j
    u_mat, v_mat = _solve_complex_resolvents(a_mat, b_eff, omegas)  # (n, r)
    u_obs = c_obs @ u_mat  # (N_d, r)
    v_obs = c_obs @ v_mat  # (N_d, r)

    # 2. Exact analytic sinc seasonal averages of cos ω_j(t - a) and sin ω_j(t - a)
    durations = ends - starts  # (N_d, M_sub)
    mids_from_a = 0.5 * (starts + ends) - a_val  # (N_d, M_sub)
    sinc_factor = jnp.sinc(
        0.5 * durations[:, :, None] * omegas[None, None, :] / jnp.pi
    )
    cos_season = jnp.sum(
        weights[:, :, None]
        * sinc_factor
        * jnp.cos(mids_from_a[:, :, None] * omegas[None, None, :]),
        axis=1,
    )
    sin_season = jnp.sum(
        weights[:, :, None]
        * sinc_factor
        * jnp.sin(mids_from_a[:, :, None] * omegas[None, None, :]),
        axis=1,
    )

    # 3. Exact Van Loan homogeneous propagator W_hom(a) = C_obs B_season[exp(A(t - a))]
    w_hom_a = _seasonal_homogeneous_operator(
        a_mat, c_obs, starts, ends, weights, a_val
    )  # (N_d, n)

    causal_design = (
        v_obs * cos_season + u_obs * sin_season - w_hom_a @ v_mat
    ) / jnp.sqrt(half_len)

    if initial_anchor == "t0":
        t_anchor = t0_obs
        w_hom_anchor = _seasonal_homogeneous_operator(
            a_mat, c_obs, starts, ends, weights, t_anchor
        )
    elif initial_anchor == "a":
        t_anchor = a_val
        w_hom_anchor = w_hom_a
    else:
        raise ValueError(
            f"initial_anchor must be 't0' or 'a'; got {initial_anchor!r}."
        )

    if kernel is None:
        return CausalGPProjection(
            causal_basis_design=causal_design,
            hom_design_a=w_hom_a,
            hom_design_anchor=w_hom_anchor,
            omegas=omegas,
            a=a_val,
            b_bound=b_val,
            t_anchor=t_anchor,
            half_length=half_len,
            spectral_weights=None,
            sigma_y0=None,
            anchor_cov_root=None,
            whitened_design=None,
        )

    spec_w = spectral_density(kernel, omegas)  # (r,)
    gp_cols = causal_design * jnp.sqrt(spec_w)[None, :]  # (N_d, r)

    if not include_initial_cov:
        return CausalGPProjection(
            causal_basis_design=causal_design,
            hom_design_a=w_hom_a,
            hom_design_anchor=w_hom_anchor,
            omegas=omegas,
            a=a_val,
            b_bound=b_val,
            t_anchor=t_anchor,
            half_length=half_len,
            spectral_weights=spec_w,
            sigma_y0=None,
            anchor_cov_root=None,
            whitened_design=gp_cols,
        )

    sigma_y0, root_y0 = stationary_cbm_gp_cov(
        a_mat,
        b,
        kernel,
        alpha_unit=alpha_unit,
        dtau0=dtau0,
        n_squarings=n_squarings,
    )

    if initial_anchor == "t0":
        # Residual stationary covariance at t_0 not yet generated on [a, t_0]:
        # ΔΣ_y(t_0) = Σ_{y,0} - Y_SS(t_0) diag(S_k(ω)) Y_SS(t_0)ᵀ.
        # When t_0 == a, Y_SS(a) == 0 identically, reducing to Σ_{y,0}.
        dt_anchor = t_anchor - a_val
        expm_dt = jsl.expm(a_mat * dt_anchor)
        c_t0 = jnp.cos(omegas * dt_anchor)[None, :]
        s_t0 = jnp.sin(omegas * dt_anchor)[None, :]
        y_ss_t0 = (v_mat * c_t0 + u_mat * s_t0 - expm_dt @ v_mat) / jnp.sqrt(half_len)
        cov_ss_t0 = (y_ss_t0 * spec_w[None, :]) @ y_ss_t0.T
        delta_cov = 0.5 * ((sigma_y0 - cov_ss_t0) + (sigma_y0 - cov_ss_t0).T)
        anchor_root = _safe_psd_root(delta_cov)
    else:
        anchor_root = root_y0

    hom_cols = w_hom_anchor @ anchor_root  # (N_d, n)
    whitened_design = jnp.concatenate([gp_cols, hom_cols], axis=1)

    return CausalGPProjection(
        causal_basis_design=causal_design,
        hom_design_a=w_hom_a,
        hom_design_anchor=w_hom_anchor,
        omegas=omegas,
        a=a_val,
        b_bound=b_val,
        t_anchor=t_anchor,
        half_length=half_len,
        spectral_weights=spec_w,
        sigma_y0=sigma_y0,
        anchor_cov_root=anchor_root,
        whitened_design=whitened_design,
    )


# =============================================================================
# 6. Canonical Data- & Prior-Orthogonal Wiener SVD Modes
# =============================================================================


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


# =============================================================================
# 7. Exact Continuous-Time Kalman Filter + RTS Smoother on Uniform Step Grids
# =============================================================================


class KalmanGPResult(NamedTuple):
    """Exact marginal likelihood and RTS smoothed posterior for ``quasisep`` GP + CBM."""

    loglike: jax.Array
    t_grid: jax.Array
    production_mean: jax.Array
    production_var: jax.Array
    cbm_state_mean: jax.Array
    cbm_state_var: jax.Array
    nuisance_mean: jax.Array | None
    nuisance_cov: jax.Array | None


def _joseph_scalar_update(
    mean: jax.Array,
    cov: jax.Array,
    h_vec: jax.Array,
    y_val: jax.Array,
    r_var: jax.Array,
    active: jax.Array,
    extra_cols: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Single scalar Joseph-form Kalman measurement update (plus linear column whitening).

    Simultaneously updates ``mean`` (for ``resid``) and a matrix ``extra_mean``
    carried in ``extra_cols`` (for analytical Woodbury marginalization of global
    offsets / solar harmonics!). Returns ``(mean, cov, delta_ll, std_innov, std_extra_row)``.
    """
    d = mean.shape[0]
    eye = jnp.eye(d, dtype=jnp.float64)
    ph = cov @ h_vec
    s_innov = jnp.dot(h_vec, ph) + r_var
    safe_s = jnp.where(active, s_innov, 1.0)
    sqrt_s = jnp.sqrt(safe_s)

    innov = y_val - jnp.dot(h_vec, mean)
    k_gain = ph / safe_s

    new_mean = mean + k_gain * innov
    i_kh = eye - jnp.outer(k_gain, h_vec)
    new_cov = i_kh @ cov @ i_kh.T + r_var * jnp.outer(k_gain, k_gain)
    new_cov = 0.5 * (new_cov + new_cov.T)

    delta_ll = -0.5 * (math.log(2.0 * math.pi) + jnp.log(safe_s) + (innov**2) / safe_s)
    std_innov = jnp.where(active, innov / sqrt_s, 0.0)

    # Parallel innovation transform for linear nuisance columns H_row - hᵀ M_extra:
    # extra_cols has shape (1 + d, p_nuis): row 0 is current H_row, rows 1: are M_extra!
    h_row_nuis = extra_cols[0]
    m_extra = extra_cols[1:]
    innov_extra = h_row_nuis - h_vec @ m_extra
    new_m_extra = m_extra + k_gain[:, None] * innov_extra[None, :]
    std_extra_row = jnp.where(active, innov_extra / sqrt_s, 0.0)

    out_mean = jnp.where(active, new_mean, mean)
    out_cov = jnp.where(active, new_cov, cov)
    out_ll = jnp.where(active, delta_ll, 0.0)
    out_m_extra = jnp.where(active, new_m_extra, m_extra)
    return out_mean, out_cov, out_ll, std_innov, std_extra_row, out_m_extra


def _build_seasonal_kalman_schedule(
    t_grid_np: onp.ndarray,
    starts_np: onp.ndarray,
    ends_np: onp.ndarray,
    weights_np: onp.ndarray,
    c_obs_np: onp.ndarray,
) -> tuple[
    int,
    int,
    onp.ndarray,
    onp.ndarray,
    onp.ndarray,
    onp.ndarray,
    onp.ndarray,
]:
    """Assign observations to minimal concurrent accumulator tracks on ``t_grid_np``."""
    n_grid = t_grid_np.shape[0]
    n_steps = n_grid - 1
    dt = float(t_grid_np[1] - t_grid_np[0])
    t0 = float(t_grid_np[0])
    n_obs, m_sub = starts_np.shape
    n_boxes = c_obs_np.shape[1]

    def _to_idx(val: float) -> int:
        k_float = (val - t0) / dt
        k_int = round(k_float)
        if abs(k_float - k_int) > 1e-7 or k_int < 0 or k_int >= n_grid:
            raise ValueError(
                f"Observation boundary {val} is not aligned to uniform t_grid "
                f"(t0={t0}, dt={dt}, t_end={float(t_grid_np[-1])})."
            )
        return k_int

    obs_s_min = onp.zeros(n_obs, dtype=int)
    obs_e_max = onp.zeros(n_obs, dtype=int)
    sub_bounds: list[list[tuple[int, int, float]]] = []
    for i in range(n_obs):
        pairs = []
        s_lo = n_grid
        e_hi = 0
        for m in range(m_sub):
            ks = _to_idx(float(starts_np[i, m]))
            ke = _to_idx(float(ends_np[i, m]))
            s_lo = min(s_lo, ks)
            e_hi = max(e_hi, ke)
            pairs.append((ks, ke, float(weights_np[i, m])))
        obs_s_min[i] = s_lo
        obs_e_max[i] = e_hi
        sub_bounds.append(pairs)

    # Greedy interval-graph colouring for positive-duration windows (ks < ke)
    order = onp.argsort(obs_s_min, kind="stable")
    track_free_step: list[int] = []
    obs_track = onp.full(n_obs, -1, dtype=int)
    for idx in order:
        i = int(idx)
        if obs_s_min[i] == obs_e_max[i]:
            continue  # Instantaneous point observation needs no accumulator track!
        placed = False
        for tr, free_at in enumerate(track_free_step):
            if free_at <= obs_s_min[i]:
                obs_track[i] = tr
                track_free_step[tr] = obs_e_max[i]
                placed = True
                break
        if not placed:
            obs_track[i] = len(track_free_step)
            track_free_step.append(obs_e_max[i])

    n_tracks = len(track_free_step)
    keep_mask = onp.zeros((n_steps, n_tracks), dtype=onp.float64)
    w_step = onp.zeros((n_steps, n_tracks, n_boxes), dtype=onp.float64)

    for i in range(n_obs):
        tr = int(obs_track[i])
        if tr < 0:
            continue
        s_lo = int(obs_s_min[i])
        e_hi = int(obs_e_max[i])
        for k in range(s_lo, e_hi):
            keep_mask[k, tr] = 0.0 if k == s_lo else 1.0
        for ks, ke, w_m in sub_bounds[i]:
            if ke > ks:
                dur = (ke - ks) * dt
                for k in range(ks, ke):
                    w_step[k, tr, :] += (w_m / dur) * c_obs_np[i, :]
            else:
                # Zero-width sub-window inside a composite observation is unsupported in step accum
                raise ValueError(
                    "Mixed zero-width and positive-width sub-windows in one ring are not supported."
                )

    # Group measurement updates at each grid node k = 0..n_grid-1
    events_at_node: list[list[int]] = [[] for _ in range(n_grid)]
    for i in range(n_obs):
        events_at_node[int(obs_e_max[i])].append(i)
    max_simul = max((len(ev) for ev in events_at_node), default=1)
    max_simul = max(max_simul, 1)

    node_obs_idx = onp.zeros((n_grid, max_simul), dtype=onp.int32)
    node_obs_active = onp.zeros((n_grid, max_simul), dtype=bool)
    # Observation vector h_node has shape (n_grid, max_simul, n_tracks + n_boxes)
    node_h_prefix = onp.zeros(
        (n_grid, max_simul, n_tracks + n_boxes), dtype=onp.float64
    )
    for k_node, ev in enumerate(events_at_node):
        for slot, i in enumerate(ev):
            node_obs_idx[k_node, slot] = i
            node_obs_active[k_node, slot] = True
            tr = int(obs_track[i])
            if tr >= 0:
                node_h_prefix[k_node, slot, tr] = 1.0
            else:
                # Instantaneous point observation directly reads δy(t_k)!
                node_h_prefix[k_node, slot, n_tracks : n_tracks + n_boxes] = c_obs_np[i, :]

    return (
        n_tracks,
        max_simul,
        keep_mask,
        w_step,
        node_obs_idx,
        node_obs_active,
        node_h_prefix,
    )


def kalman_gp_marginal(
    A: Any,
    b: Any,
    kernel: tgk.Kernel,
    resid: Any,
    sigma: Any,
    t_starts: Any,
    t_ends: Any | None = None,
    sub_weights: Any | None = None,
    *,
    t_grid: Any | None = None,
    dt: float | None = None,
    alpha_unit: float = 1.0,
    box_indices: Any = 0,
    obs_scale: Any = 1.0,
    obs_matrix: Any | None = None,
    nuisance_design: Any | None = None,
    nuisance_prior_mean: Any = 0.0,
    nuisance_prior_sd: Any | None = None,
    nuisance_prior_root: Any | None = None,
    dtau0: float = 1.0 / 16.0,
    n_squarings: int = 24,
) -> KalmanGPResult:
    """Exact ``O(N_steps (n + d_GP)³)`` Kalman filter + RTS smoother for SDE GPs.

    Propagates the exact continuous-time joint ``(CBM + quasisep GP)`` SDE
    initialized at its stationary distribution ``Σ_∞`` from 24-step Smith
    doubling, supporting both instantaneous point observations and **exact
    continuous-time seasonal-window integral observations** ``[t_starts, t_ends]``
    aligned to ``t_grid`` (via Van Loan step-integral state augmentation with
    zero discretization or midpoint error), plus analytical marginalization of
    optional linear nuisance columns (``Δ_global + δ_{lab,s}``, solar harmonics).
    """
    sde = _try_extract_sde(kernel)
    if sde is None:
        raise ValueError(
            f"kalman_gp_marginal requires a Markov / quasisep kernel; got "
            f"{type(kernel).__name__!r}. Use project_causal_gp for non-Markov kernels."
        )
    f_gp, q_sde, _, h_gp = sde
    d_gp = f_gp.shape[0]

    a_mat = jnp.asarray(A, dtype=jnp.float64)
    b_eff = alpha_unit * jnp.asarray(b, dtype=jnp.float64).reshape(-1)
    n_boxes = a_mat.shape[0]
    d_xi = n_boxes + d_gp

    starts, ends, weights = _normalize_windows(t_starts, t_ends, sub_weights)
    n_obs = starts.shape[0]
    resid_arr = jnp.asarray(resid, dtype=jnp.float64).reshape(n_obs)
    sd_arr = jnp.broadcast_to(jnp.asarray(sigma, dtype=jnp.float64), (n_obs,))
    r_vars = jnp.square(sd_arr)
    c_obs = _build_obs_matrix(
        n_obs, n_boxes, box_indices=box_indices, obs_scale=obs_scale, obs_matrix=obs_matrix
    )

    # Build or validate concrete uniform step grid
    c_starts, c_ends = _concrete(starts), _concrete(ends)
    c_weights, c_cobs = _concrete(weights), _concrete(c_obs)
    if c_starts is None or c_ends is None or c_weights is None or c_cobs is None:
        raise ValueError(
            "Observation window timestamps (t_starts, t_ends) must be concrete "
            "arrays to build the static Kalman step schedule (A, b, kernel, "
            "resid, and sigma may all be JAX-traced for jit/grad)."
        )

    if t_grid is None:
        step_dt = 1.0 / 16.0 if dt is None else float(dt)
        t_min = float(onp.min(c_starts))
        t_max = float(onp.max(c_ends))
        n_steps = max(round((t_max - t_min) / step_dt), 1)
        t_grid_np = t_min + step_dt * onp.arange(n_steps + 1, dtype=onp.float64)
    else:
        t_grid_np = onp.asarray(_concrete(t_grid), dtype=onp.float64).reshape(-1)
        if t_grid_np.size < 2:
            raise ValueError("t_grid must contain at least 2 points.")
        diffs = onp.diff(t_grid_np)
        step_dt = float(diffs[0])
        if not onp.allclose(diffs, step_dt, rtol=0.0, atol=1e-9):
            raise ValueError("t_grid must be uniformly spaced.")

    (
        n_tracks,
        max_simul,
        keep_mask_np,
        w_step_np,
        node_obs_idx_np,
        node_obs_active_np,
        node_h_prefix_np,
    ) = _build_seasonal_kalman_schedule(
        t_grid_np, c_starts, c_ends, c_weights, c_cobs
    )

    t_grid_jax = jnp.asarray(t_grid_np, dtype=jnp.float64)
    n_grid = t_grid_np.shape[0]
    n_steps = n_grid - 1
    d_state = n_tracks + d_xi

    # Optional linear nuisance setup (carried through Kalman innovation whitening!)
    if nuisance_design is None:
        p_nuis = 0
        nuis_u_raw = jnp.zeros((n_obs, 0), dtype=jnp.float64)
        nuis_mean_prior = jnp.zeros((0,), dtype=jnp.float64)
        nuis_root = jnp.zeros((0, 0), dtype=jnp.float64)
    else:
        nuis_x = jnp.asarray(nuisance_design, dtype=jnp.float64)
        p_nuis = nuis_x.shape[1]
        if (nuisance_prior_sd is None) == (nuisance_prior_root is None):
            raise ValueError(
                "Specify exactly one of nuisance_prior_sd or nuisance_prior_root."
            )
        if nuisance_prior_sd is not None:
            nuis_root = jnp.diag(
                jnp.broadcast_to(
                    jnp.asarray(nuisance_prior_sd, dtype=jnp.float64), (p_nuis,)
                )
            )
        else:
            nuis_root = jnp.asarray(nuisance_prior_root, dtype=jnp.float64)
        nuis_mean_prior = jnp.broadcast_to(
            jnp.asarray(nuisance_prior_mean, dtype=jnp.float64), (p_nuis,)
        )
        resid_arr = resid_arr - nuis_x @ nuis_mean_prior
        nuis_u_raw = nuis_x @ nuis_root  # (N_d, p_nuis)

    # 1. Stationary initial covariance Σ_∞ at t_grid[0] via 24-step Smith doubling
    _, _, sigma_xi_inf = stationary_cbm_gp_cov(
        a_mat,
        b,
        kernel,
        alpha_unit=alpha_unit,
        dtau0=dtau0,
        n_squarings=n_squarings,
        return_full_state=True,
    )

    # 2. Single Van Loan expm for exact step transition + step integral covariance!
    f_joint = jnp.block(
        [
            [a_mat, jnp.outer(b_eff, h_gp)],
            [jnp.zeros((d_gp, n_boxes), dtype=jnp.float64), f_gp],
        ]
    )
    q_joint = jnp.block(
        [
            [
                jnp.zeros((n_boxes, n_boxes), dtype=jnp.float64),
                jnp.zeros((n_boxes, d_gp), dtype=jnp.float64),
            ],
            [jnp.zeros((d_gp, n_boxes), dtype=jnp.float64), q_sde],
        ]
    )

    if n_tracks == 0:
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
        sig_xi = 0.5 * (
            e_step[:d_xi, d_xi:] @ phi_xi.T + (e_step[:d_xi, d_xi:] @ phi_xi.T).T
        )
        phi_s_xi = jnp.zeros((n_boxes, d_xi), dtype=jnp.float64)
        sig_ss = jnp.zeros((n_boxes, n_boxes), dtype=jnp.float64)
        sig_s_xi = jnp.zeros((n_boxes, d_xi), dtype=jnp.float64)
    else:
        # Augmented continuous system [s(t); ξ(t)] with ds/dt = [I_n, 0] ξ(t)
        d_aug = n_boxes + d_xi
        c_int = jnp.concatenate(
            [jnp.eye(n_boxes, dtype=jnp.float64), jnp.zeros((n_boxes, d_gp), dtype=jnp.float64)],
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

        phi_s_xi = phi_aug[:n_boxes, n_boxes:]
        phi_xi = phi_aug[n_boxes:, n_boxes:]
        sig_ss = sig_aug[:n_boxes, :n_boxes]
        sig_s_xi = sig_aug[:n_boxes, n_boxes:]
        sig_xi = sig_aug[n_boxes:, n_boxes:]

    keep_jax = jnp.asarray(keep_mask_np, dtype=jnp.float64)
    w_step_jax = jnp.asarray(w_step_np, dtype=jnp.float64)
    node_obs_idx = jnp.asarray(node_obs_idx_np, dtype=jnp.int32)
    node_obs_active = jnp.asarray(node_obs_active_np, dtype=bool)
    node_h_full = jnp.concatenate(
        [
            jnp.asarray(node_h_prefix_np, dtype=jnp.float64),
            jnp.zeros((n_grid, max_simul, d_gp), dtype=jnp.float64),
        ],
        axis=-1,
    )

    def _step_mats(k_idx: jax.Array) -> tuple[jax.Array, jax.Array]:
        if n_tracks == 0:
            return phi_xi, sig_xi
        kp = keep_jax[k_idx]  # (n_tracks,)
        wk = w_step_jax[k_idx]  # (n_tracks, n_boxes)
        phi_k = jnp.block(
            [
                [jnp.diag(kp), wk @ phi_s_xi],
                [jnp.zeros((d_xi, n_tracks), dtype=jnp.float64), phi_xi],
            ]
        )
        # Tiny 1e-30 variance on idle accumulator slots keeps P_{k+1|k} strictly SPD for RTS
        idle_reg = 1e-30 * jnp.eye(n_tracks, dtype=jnp.float64)
        sig_aa = wk @ sig_ss @ wk.T + idle_reg
        sig_a_xi = wk @ sig_s_xi
        sig_k = jnp.block([[sig_aa, sig_a_xi], [sig_a_xi.T, sig_xi]])
        sig_k = 0.5 * (sig_k + sig_k.T)
        return phi_k, sig_k

    def _node_measurement_updates(
        mean: jax.Array,
        cov: jax.Array,
        m_extra: jax.Array,
        k_node: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        ll_node = jnp.asarray(0.0, dtype=jnp.float64)
        std_innovs = []
        std_extras = []
        for slot in range(max_simul):
            idx_i = node_obs_idx[k_node, slot]
            act_i = node_obs_active[k_node, slot]
            h_i = node_h_full[k_node, slot]
            y_i = resid_arr[idx_i]
            r_i = r_vars[idx_i]
            h_nuis_i = nuis_u_raw[idx_i]
            extra_pack = jnp.concatenate([h_nuis_i[None, :], m_extra], axis=0)
            mean, cov, dll, s_in, s_ex, m_extra = _joseph_scalar_update(
                mean, cov, h_i, y_i, r_i, act_i, extra_pack
            )
            ll_node = ll_node + dll
            std_innovs.append(s_in)
            std_extras.append(s_ex)
        return (
            mean,
            cov,
            m_extra,
            ll_node,
            jnp.stack(std_innovs),
            jnp.stack(std_extras),
        )

    # Initial state at node 0:
    init_mean = jnp.zeros((d_state,), dtype=jnp.float64)
    init_cov = jsl.block_diag(
        1e-30 * jnp.eye(n_tracks, dtype=jnp.float64), sigma_xi_inf
    )
    init_m_extra = jnp.zeros((d_state, p_nuis), dtype=jnp.float64)

    mean0, cov0, m_extra0, ll0, sin0, sex0 = _node_measurement_updates(
        init_mean, init_cov, init_m_extra, jnp.asarray(0, dtype=jnp.int32)
    )

    def _forward_step(
        carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        k_step: jax.Array,
    ):
        m_prev, p_prev, mx_prev, ll_acc = carry
        phi_k, sig_k = _step_mats(k_step)
        m_pred = phi_k @ m_prev
        p_pred = phi_k @ p_prev @ phi_k.T + sig_k
        p_pred = 0.5 * (p_pred + p_pred.T)
        mx_pred = phi_k @ mx_prev

        m_filt, p_filt, mx_filt, dll, s_in, s_ex = _node_measurement_updates(
            m_pred, p_pred, mx_pred, k_step + 1
        )
        return (
            (m_filt, p_filt, mx_filt, ll_acc + dll),
            (m_filt, p_filt, m_pred, p_pred, phi_k, mx_filt, mx_pred, s_in, s_ex),
        )

    (_, _, _, total_ll), scan_out = jax.lax.scan(
        _forward_step,
        (mean0, cov0, m_extra0, ll0),
        jnp.arange(n_steps, dtype=jnp.int32),
    )
    (
        m_filt_seq,
        p_filt_seq,
        m_pred_seq,
        p_pred_seq,
        phi_seq,
        mx_filt_seq,
        mx_pred_seq,
        sin_seq,
        sex_seq,
    ) = scan_out

    m_filt_all = jnp.concatenate([mean0[None, :], m_filt_seq], axis=0)
    p_filt_all = jnp.concatenate([cov0[None, :, :], p_filt_seq], axis=0)
    mx_filt_all = jnp.concatenate([m_extra0[None, :, :], mx_filt_seq], axis=0)

    # Analytical Woodbury marginalization + posterior over optional linear nuisance terms!
    if p_nuis > 0:
        all_sin = jnp.concatenate([sin0[None, :], sin_seq], axis=0).reshape(-1)
        all_sex = jnp.concatenate([sex0[None, :, :], sex_seq], axis=0).reshape(
            -1, p_nuis
        )
        # Note: inactive slots have 0.0 in both all_sin and all_sex, adding 0 to UᵀU and Uᵀx!
        _, half_logdet_nuis = whiten_cholesky(all_sin, all_sex)
        omega_nuis, cov_omega_nuis = posterior(all_sin, all_sex)
        # Replace -0.5 * ||all_sin||² inside total_ll with whitened quadratic form:
        u_whitened, _ = whiten_cholesky(all_sin, all_sex)
        total_ll = (
            total_ll
            + 0.5 * jnp.dot(all_sin, all_sin)
            - 0.5 * jnp.dot(u_whitened, u_whitened)
            - half_logdet_nuis
        )
        nuis_mean = nuis_mean_prior + nuis_root @ omega_nuis
        nuis_cov = 0.5 * (
            nuis_root @ cov_omega_nuis @ nuis_root.T
            + (nuis_root @ cov_omega_nuis @ nuis_root.T).T
        )
        # Adjust filtered state means & covariances for marginalized nuisance posterior:
        m_filt_adj = m_filt_all - jnp.einsum("tij,j->ti", mx_filt_all, omega_nuis)
        m_pred_adj = m_pred_seq - jnp.einsum("tij,j->ti", mx_pred_seq, omega_nuis)
    else:
        nuis_mean = None
        nuis_cov = None
        m_filt_adj = m_filt_all
        m_pred_adj = m_pred_seq
        cov_omega_nuis = jnp.zeros((0, 0), dtype=jnp.float64)

    # 3. Backward Rauch--Tung--Striebel (RTS) smoother scan
    def _rts_step(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        inputs: tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ],
    ):
        m_next_s, p_next_s, mx_next_s = carry
        m_k_f, p_k_f, m_kp1_p, p_kp1_p, phi_k, mx_k_f, mx_kp1_p = inputs
        # G_k = P_{k|k} Φ_kᵀ P_{k+1|k}⁻¹ => G_kᵀ = P_{k+1|k}⁻¹ (Φ_k P_{k|k})
        chol_p = jnp.linalg.cholesky(
            p_kp1_p + 1e-30 * jnp.eye(d_state, dtype=jnp.float64)
        )
        g_t = jsl.cho_solve((chol_p, True), phi_k @ p_k_f)
        g_k = g_t.T
        m_k_s = m_k_f + g_k @ (m_next_s - m_kp1_p)
        p_k_s = p_k_f + g_k @ (p_next_s - p_kp1_p) @ g_t
        p_k_s = 0.5 * (p_k_s + p_k_s.T)
        mx_k_s = mx_k_f + g_k @ (mx_next_s - mx_kp1_p)
        return (m_k_s, p_k_s, mx_k_s), (m_k_s, p_k_s, mx_k_s)

    _, (m_smooth_rev, p_smooth_rev, mx_smooth_rev) = jax.lax.scan(
        _rts_step,
        (m_filt_adj[-1], p_filt_all[-1], mx_filt_all[-1]),
        (
            m_filt_adj[:-1],
            p_filt_all[:-1],
            m_pred_adj,
            p_pred_seq,
            phi_seq,
            mx_filt_all[:-1],
            mx_pred_seq,
        ),
        reverse=True,
    )

    m_smooth = jnp.concatenate([m_smooth_rev, m_filt_adj[-1:]], axis=0)
    p_smooth = jnp.concatenate([p_smooth_rev, p_filt_all[-1:]], axis=0)
    if p_nuis > 0:
        mx_smooth = jnp.concatenate([mx_smooth_rev, mx_filt_all[-1:]], axis=0)
        p_smooth = p_smooth + jnp.einsum(
            "tik,kl,tjl->tij", mx_smooth, cov_omega_nuis, mx_smooth
        )

    # Extract CBM state δy(t_k) and GP production q(t_k) = h_GPᵀ z_GP(t_k):
    cbm_slice = slice(n_tracks, n_tracks + n_boxes)
    gp_slice = slice(n_tracks + n_boxes, d_state)

    cbm_mean = m_smooth[:, cbm_slice]
    cbm_var = jnp.diagonal(p_smooth[:, cbm_slice, cbm_slice], axis1=1, axis2=2)

    z_gp_mean = m_smooth[:, gp_slice]
    z_gp_cov = p_smooth[:, gp_slice, gp_slice]
    q_mean = z_gp_mean @ h_gp
    q_var = jnp.einsum("i,tij,j->t", h_gp, z_gp_cov, h_gp)
    q_var = jnp.maximum(q_var, 0.0)

    return KalmanGPResult(
        loglike=total_ll,
        t_grid=t_grid_jax,
        production_mean=q_mean,
        production_var=q_var,
        cbm_state_mean=cbm_mean,
        cbm_state_var=jnp.maximum(cbm_var, 0.0),
        nuisance_mean=nuis_mean,
        nuisance_cov=nuis_cov,
    )
