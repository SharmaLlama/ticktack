"""Fisher information analysis, Schur-marginalized carbon-cycle eigenmodes,
and continuous Laplace/Mellin relaxation spectra.

Implements the carbon-cycle information theory of ``ticktack 2.0`` (the
time-domain Laplace/Mellin counterpart of Langker, Greer, Pope & Tuthill
2026, arXiv:2610.41092, and McWhirter & Pike 1978):

1. [`expected_fisher`][ticktack.fisher.expected_fisher]: Gauss--Newton expected
   Fisher information matrix ``Jᵀ Σ_d⁻¹ J`` via forward-mode autodiff
   (``jax.jacfwd``), accepting arbitrary parameter PyTrees (``zodiax.Base`` /
   ``equinox.Module``, dicts, tuples) or flat unconstrained parameter vectors,
   and either 1D Gaussian uncertainties ``sigma`` or a 2D noise precision
   matrix ``prec``.
2. [`cbm_fisher_spectrum`][ticktack.fisher.cbm_fisher_spectrum]: Joint
   CBM + nuisance Fisher calculation, QR-stabilized Schur-complement
   marginalization over production and baseline-offset nuisance parameters,
   and coordinate-invariant prior-whitened spectral decomposition
   ``F̃_CBM = L_Πᵀ F_CBM^marg L_Π = V diag(μ) Vᵀ``, returning effective
   constrained dimensionality ``d_eff``, Shannon/KL information gain in nats
   and bits, and physical log-mass / log-flux loading projections with zero
   ``inf * 0 = NaN`` hazard on flat-prior gauge modes.
3. [`relaxation_fisher_modes`][ticktack.fisher.relaxation_fisher_modes]:
   Model-agnostic SVD of the seasonal-binned Laplace relaxation operator on
   log-spaced decay rates ``s_j``, reproducing the analytical McWhirter--Pike
   ``π / cosh(π ω_ln)`` Mellin power spectrum under ``measure="l2_ds"`` and
   scale-invariant residue modes under ``measure="d_ln_s"``.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import Any, Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as onp
from jax.flatten_util import ravel_pytree

__all__ = [
    "FisherSpectrumResult",
    "RelaxationModesResult",
    "cbm_fisher_spectrum",
    "expected_fisher",
    "fisher_projection",
    "laplace_covariance",
    "regularized_inverse",
    "relaxation_fisher_modes",
]


def _concrete(value: Any) -> onp.ndarray | None:
    """Return ``value`` as a NumPy array, or ``None`` inside JAX transformations."""
    try:
        return onp.asarray(value)
    except (
        jax.errors.TracerArrayConversionError,
        jax.errors.ConcretizationTypeError,
    ):
        return None


def _partition_inexact(params: Any) -> tuple[jax.Array, Callable[[jax.Array], Any]]:
    """Flatten floating-point leaves of ``params`` to a 1D ``float64`` vector."""
    if isinstance(params, (jax.Array, onp.ndarray)) and params.ndim == 1:
        flat = jnp.asarray(params, dtype=jnp.float64)
        return flat, lambda x: x
    diff_tree, static_tree = eqx.partition(params, eqx.is_inexact_array)
    flat_raw, unravel_diff = ravel_pytree(diff_tree)
    flat = jnp.asarray(flat_raw, dtype=jnp.float64)

    def unravel(vec: jax.Array) -> Any:
        return eqx.combine(unravel_diff(vec), static_tree)

    return flat, unravel


def _whiten_jacobian(
    jacobian: jax.Array,
    sigma_or_prec: Any,
    kind: Literal["auto", "sigma", "precision", "cov"] = "auto",
) -> jax.Array:
    """Return ``L_d⁻¹ J`` such that ``(L_d⁻¹ J)ᵀ (L_d⁻¹ J) = Jᵀ Σ_d⁻¹ J``."""
    n_obs = jacobian.shape[0]
    arr = jnp.asarray(sigma_or_prec, dtype=jnp.float64)

    resolved_kind = kind
    if resolved_kind == "auto":
        resolved_kind = "sigma" if arr.ndim <= 1 else "precision"

    if resolved_kind == "sigma":
        sigma = jnp.broadcast_to(arr, (n_obs,)).reshape(n_obs)
        c_sigma = _concrete(sigma)
        if c_sigma is not None and not onp.all(onp.isfinite(c_sigma) & (c_sigma > 0.0)):
            raise ValueError("Gaussian standard deviations sigma must be finite and > 0.")
        return jacobian / sigma[:, None]

    if resolved_kind == "precision":
        if arr.shape != (n_obs, n_obs):
            raise ValueError(
                f"Noise precision matrix must have shape ({n_obs}, {n_obs}); "
                f"got {arr.shape}."
            )
        sym_prec = 0.5 * (arr + arr.T)
        c_prec = _concrete(sym_prec)
        if c_prec is not None:
            if not onp.all(onp.isfinite(c_prec)):
                raise ValueError("Noise precision matrix must be finite.")
            if onp.min(onp.linalg.eigvalsh(c_prec)) <= 0.0:
                raise ValueError("Noise precision matrix must be positive definite.")
        chol_prec = jnp.linalg.cholesky(sym_prec)
        # sym_prec = chol_prec @ chol_prec.T => J.T @ sym_prec @ J = (chol_prec.T @ J).T @ (chol_prec.T @ J)
        return chol_prec.T @ jacobian

    if resolved_kind == "cov":
        if arr.shape != (n_obs, n_obs):
            raise ValueError(
                f"Noise covariance matrix must have shape ({n_obs}, {n_obs}); "
                f"got {arr.shape}."
            )
        sym_cov = 0.5 * (arr + arr.T)
        c_cov = _concrete(sym_cov)
        if c_cov is not None:
            if not onp.all(onp.isfinite(c_cov)):
                raise ValueError("Noise covariance matrix must be finite.")
            if onp.min(onp.linalg.eigvalsh(c_cov)) <= 0.0:
                raise ValueError("Noise covariance matrix must be positive definite.")
        chol_cov = jnp.linalg.cholesky(sym_cov)
        return jsl.solve_triangular(chol_cov, jacobian, lower=True)

    raise ValueError(
        f"kind must be 'auto', 'sigma', 'precision', or 'cov'; got {kind!r}."
    )


def expected_fisher(
    forward_fn: Callable[[Any], Any],
    params: Any,
    sigma_or_prec: Any,
    *,
    ridge: float = 0.0,
    kind: Literal["auto", "sigma", "precision", "cov"] = "auto",
    return_unravel: bool = False,
) -> jax.Array | tuple[jax.Array, Callable[[jax.Array], Any]]:
    """Compute the Gauss--Newton expected Fisher matrix ``Jᵀ Σ_d⁻¹ J``.

    Parameters
    ----------
    forward_fn : callable
        Deterministic forward prediction mapping ``params`` to a 1D array of
        predicted observables of length ``N_d``.
    params : pytree or 1D array
        Point in parameter space at which sensitivities are evaluated. May be
        a flat 1D ``jax.Array`` or any ``zodiax`` / ``equinox`` PyTree.
    sigma_or_prec : array-like
        Either a 1D array (or scalar) of Gaussian standard deviations ``sigma``
        of length ``N_d``, or a 2D ``(N_d, N_d)`` noise precision matrix
        ``Σ_d⁻¹`` (or covariance if ``kind="cov"``).
    ridge : float, optional
        Non-negative isotropic diagonal regularization ``max(ridge, 0) * I``.
    kind : {"auto", "sigma", "precision", "cov"}, optional
        Interpretation of ``sigma_or_prec``. ``"auto"`` (default) treats 0D/1D
        inputs as ``"sigma"`` and 2D inputs as ``"precision"``.
    return_unravel : bool, optional
        If ``True``, return ``(fisher_matrix, unravel_fn)`` matching
        ``virgil.inference.expected_fisher``; if ``False`` (default), return
        ``fisher_matrix`` directly.

    Returns
    -------
    jax.Array or tuple[jax.Array, callable]
        Symmetric ``(p, p)`` expected Fisher information matrix (and
        ``unravel_fn`` if ``return_unravel=True``).
    """
    flat_params, unravel = _partition_inexact(params)

    def flat_forward(vec: jax.Array) -> jax.Array:
        return jnp.asarray(forward_fn(unravel(vec)), dtype=jnp.float64).reshape(-1)

    jacobian = jax.jacfwd(flat_forward)(flat_params)
    whitened = _whiten_jacobian(jacobian, sigma_or_prec, kind=kind)
    fmat = whitened.T @ whitened
    fmat = 0.5 * (fmat + fmat.T)
    if ridge != 0.0:
        p = fmat.shape[-1]
        fmat = fmat + jnp.maximum(ridge, 0.0) * jnp.eye(p, dtype=fmat.dtype)
    if return_unravel:
        return fmat, unravel
    return fmat


def regularized_inverse(matrix: Any, ridge: float = 1e-10) -> jax.Array:
    """Invert a symmetric positive-definite curvature matrix with relative ridge fallback.

    Follows ``virgil.inference.regularized_inverse``: if ``matrix`` is strictly
    positive definite, its exact inverse is returned; otherwise a scale-aware
    diagonal ridge ``ridge * |matrix_ii|`` is added first and a
    ``RuntimeWarning`` is emitted outside ``jax.jit``.
    """
    mat = jnp.asarray(matrix, dtype=jnp.float64)
    mat = 0.5 * (mat + mat.T)
    if mat.shape[0] == 0:
        return jnp.zeros((0, 0), dtype=jnp.float64)
    scale = jnp.abs(jnp.diagonal(mat))
    top = jnp.max(scale)
    scale = jnp.where(scale > 0.0, scale, jnp.where(top > 0.0, top, 1.0))
    regularized = mat + ridge * jnp.diag(scale)
    evals = jnp.linalg.eigvalsh(mat)
    max_ev = jnp.maximum(jnp.max(evals), 0.0)
    plain = (
        jnp.all(jnp.isfinite(mat))
        & (jnp.min(evals) > 0.0)
        & (jnp.min(evals) > 1e-13 * max_ev)
        & jnp.all(jnp.isfinite(jnp.linalg.cholesky(mat)))
    )
    c_raw = _concrete(mat)
    if c_raw is not None and onp.all(onp.isfinite(c_raw)):
        c_evals = onp.linalg.eigvalsh(c_raw)
        c_max = max(float(onp.max(c_evals)), 0.0)
        if float(onp.min(c_evals)) <= max(0.0, 1e-13 * c_max):
            warnings.warn(
                "Information matrix is not strictly positive definite; a scale-relative "
                "diagonal ridge was added before inversion.",
                RuntimeWarning,
                stacklevel=2,
            )
    target = jnp.where(plain, mat, regularized)
    chol = jnp.linalg.cholesky(target)
    return jsl.cho_solve((chol, True), jnp.eye(mat.shape[0], dtype=mat.dtype))


def laplace_covariance(
    neg_log_posterior_fn: Callable[[Any], jax.Array],
    params: Any,
    *,
    ridge: float = 1e-10,
) -> jax.Array:
    """Return the Laplace posterior covariance ``(∇²(-ln p(θ | d)))⁻¹``."""
    flat_params, unravel = _partition_inexact(params)
    hess = jax.hessian(lambda v: jnp.asarray(neg_log_posterior_fn(unravel(v)), dtype=jnp.float64))(
        flat_params
    )
    return regularized_inverse(hess, ridge=ridge)


def fisher_projection(fmat: Any, eps: float = 1e-12) -> jax.Array:
    """Return ``P = V diag(λ⁻¹/²)`` mapping standard-normal draws to ``F⁻¹`` steps."""
    evals, evecs = jnp.linalg.eigh(0.5 * (jnp.asarray(fmat, dtype=jnp.float64) + jnp.asarray(fmat, dtype=jnp.float64).T))
    largest = jnp.max(jnp.abs(evals))
    floor = jnp.where(largest > 0.0, eps * largest, eps)
    safe = jnp.maximum(evals, floor)
    return evecs * (1.0 / jnp.sqrt(safe))[None, :]


class FisherSpectrumResult(NamedTuple):
    """Coordinate-invariant carbon-cycle Fisher eigenspectrum and loadings.

    Attributes
    ----------
    eigenvalues : jax.Array
        Descending prior-whitened eigenvalues ``μ_1 >= ... >= μ_p >= 0`` of
        ``F̃_CBM = L_Πᵀ F_CBM^marg L_Π``. ``sqrt(μ_k)`` is the posterior-to-prior
        precision gain along mode ``k``.
    d_eff : jax.Array
        Spiegelhalter / MacKay effective number of constrained CBM combinations,
        ``Σ_k μ_k / (μ_k + 1)``.
    info_nats : jax.Array
        Shannon / Kullback--Leibler information gain ``½ Σ_k ln(1 + μ_k)`` in nats.
    info_bits : jax.Array
        Shannon / Kullback--Leibler information gain in bits (``info_nats / ln 2``).
    prior_whitened_modes : jax.Array
        Orthonormal eigenvectors ``V`` ``(p_active, p_active)`` in prior-whitened
        coordinates.
    parameter_loadings : jax.Array
        Physical parameter perturbations ``L_Π V`` ``(p_full, p_active)`` per
        1-sigma prior step along each eigenmode (zero on inactive/gauge coords).
    marginal_fisher : jax.Array
        Schur-complement marginalized CBM Fisher matrix ``F_CBM^marg`` ``(p_full, p_full)``.
    whitened_fisher : jax.Array
        Prior-whitened symmetric CBM Fisher matrix ``F̃_CBM`` ``(p_active, p_active)``.
    joint_fisher : jax.Array
        Full joint Gauss--Newton Fisher matrix over ``(θ_CBM, ψ_nuisance)``.
    posterior_cov : jax.Array
        Marginalized Gaussian posterior covariance ``(F_CBM,act^marg + Π_θ,act)⁻¹``
        ``(p_active, p_active)``.
    mass_loadings : jax.Array | None
        Reservoir log-mass sensitivities ``∂(ln M)/∂θ @ (L_Π V)`` ``(n_boxes, p_active)``
        when ``mass_fn`` is supplied, else ``None``.
    flux_loadings : jax.Array | None
        Gross edge log-flux sensitivities ``∂(ln F)/∂θ @ (L_Π V)`` ``(n_edges, p_active)``
        when ``flux_fn`` is supplied, else ``None``.
    active_mask : jax.Array
        Boolean mask ``(p_full,)`` indicating active finite-prior CBM coordinates.
    """

    eigenvalues: jax.Array
    d_eff: jax.Array
    info_nats: jax.Array
    info_bits: jax.Array
    prior_whitened_modes: jax.Array
    parameter_loadings: jax.Array
    marginal_fisher: jax.Array
    whitened_fisher: jax.Array
    joint_fisher: jax.Array
    posterior_cov: jax.Array
    mass_loadings: jax.Array | None
    flux_loadings: jax.Array | None
    active_mask: jax.Array


def _psd_square_root_factor(psd_mat: jax.Array, tol: float = 1e-15) -> jax.Array:
    """Return ``B`` with ``B @ B.T = psd_mat`` safely for symmetric PSD matrices.

    Uses Cholesky directly when strictly positive definite (preserving exact
    autodiff across repeated eigenvalues) and falls back to double-``where``
    guarded symmetric eigendecomposition when singular or zero.
    """
    n = psd_mat.shape[0]
    if n == 0:
        return jnp.zeros((0, 0), dtype=jnp.float64)
    sym = 0.5 * (psd_mat + psd_mat.T)
    chol = jnp.linalg.cholesky(sym)
    chol_ok = jnp.all(jnp.isfinite(chol))

    # Guarded eigh branch for rank-deficient / zero nuisance precision matrices:
    c_sym = _concrete(sym)
    def _eigh_branch(m: jax.Array) -> jax.Array:
        evals, evecs = jnp.linalg.eigh(m)
        scale = jnp.maximum(jnp.max(evals), 1.0)
        pos = evals > tol * scale
        safe_evals = jnp.where(pos, evals, 1.0)
        root_evals = jnp.where(pos, jnp.sqrt(safe_evals), 0.0)
        return evecs * root_evals[None, :]

    if c_sym is not None:
        if onp.all(onp.isfinite(onp.asarray(chol))):
            return chol
        return _eigh_branch(sym)

    return jax.lax.cond(chol_ok, lambda _: chol, _eigh_branch, sym)


def _resolve_cbm_prior_root(
    p_full: int,
    *,
    active_cbm_mask: Any,
    prior_cbm_root: Any,
    prior_cbm_cov: Any,
    prior_cbm_prec: Any,
    prior_cbm_sd: Any,
    whitened_j_theta: jax.Array,
) -> tuple[jax.Array, onp.ndarray]:
    """Resolve active CBM mask and prior covariance root ``L_Π`` without ``inf * 0 = NaN``.

    Returns ``(L_Pi, active_mask)`` where ``L_Pi`` has shape ``(p_active, p_active)``
    satisfying ``L_Pi @ L_Pi.T = Σ_{Π,active}``, and ``active_mask`` is a concrete
    1D boolean NumPy array of length ``p_full``.
    """
    n_given = sum(
        x is not None
        for x in (prior_cbm_root, prior_cbm_cov, prior_cbm_prec, prior_cbm_sd)
    )
    if n_given > 1:
        raise ValueError(
            "Specify at most one of prior_cbm_root, prior_cbm_cov, "
            "prior_cbm_prec, or prior_cbm_sd."
        )

    if active_cbm_mask is not None:
        mask = onp.asarray(active_cbm_mask, dtype=bool).reshape(p_full)
    else:
        mask = onp.ones(p_full, dtype=bool)

    c_j_col_norm = _concrete(jnp.linalg.norm(whitened_j_theta, axis=0))

    if n_given == 0:
        p_act = int(onp.sum(mask))
        return jnp.eye(p_act, dtype=jnp.float64), mask

    if prior_cbm_sd is not None:
        sd = jnp.broadcast_to(
            jnp.asarray(prior_cbm_sd, dtype=jnp.float64), (p_full,)
        )
        c_sd = _concrete(sd)
        if c_sd is not None:
            if onp.any(c_sd <= 0.0) or onp.any(onp.isnan(c_sd)):
                raise ValueError("prior_cbm_sd must be strictly positive (or +inf on gauge coords).")
            inf_cols = ~onp.isfinite(c_sd)
            if onp.any(inf_cols & mask):
                if c_j_col_norm is not None and onp.any(c_j_col_norm[inf_cols & mask] > 1e-12):
                    raise ValueError(
                        "Infinite prior SD is only permitted on zero-Fisher gauge "
                        "coordinates (such as log_total_mass); data-sensitive "
                        "parameters require a finite prior."
                    )
                mask = mask & (~inf_cols)
        idx = jnp.asarray(onp.flatnonzero(mask), dtype=jnp.int32)
        sd_act = sd[idx]
        safe_sd = jnp.where(jnp.isfinite(sd_act), sd_act, 0.0)
        return jnp.diag(safe_sd), mask

    if prior_cbm_root is not None:
        root = jnp.asarray(prior_cbm_root, dtype=jnp.float64)
        p_act = int(onp.sum(mask))
        if root.shape == (p_act, p_act):
            return root, mask
        if root.shape == (p_full, p_full):
            idx = jnp.asarray(onp.flatnonzero(mask), dtype=jnp.int32)
            return root[jnp.ix_(idx, idx)], mask
        raise ValueError(
            f"prior_cbm_root must have shape ({p_act}, {p_act}) or ({p_full}, {p_full}); "
            f"got {root.shape}."
        )

    if prior_cbm_cov is not None:
        cov = jnp.asarray(prior_cbm_cov, dtype=jnp.float64)
        p_act_init = int(onp.sum(mask))
        if cov.shape == (p_full, p_full):
            c_diag = _concrete(jnp.diag(cov))
            if c_diag is not None:
                inf_cols = ~onp.isfinite(c_diag)
                if onp.any(inf_cols & mask):
                    if c_j_col_norm is not None and onp.any(c_j_col_norm[inf_cols & mask] > 1e-12):
                        raise ValueError(
                            "Infinite prior variance is only permitted on zero-Fisher "
                            "gauge coordinates."
                        )
                    mask = mask & (~inf_cols)
            idx = jnp.asarray(onp.flatnonzero(mask), dtype=jnp.int32)
            cov_act = cov[jnp.ix_(idx, idx)]
        elif cov.shape == (p_act_init, p_act_init):
            cov_act = cov
        else:
            raise ValueError(
                f"prior_cbm_cov must have shape ({p_act_init}, {p_act_init}) or "
                f"({p_full}, {p_full}); got {cov.shape}."
            )
        cov_act = 0.5 * (cov_act + cov_act.T)
        # Guard traced infinities on diagonal before Cholesky so inf * 0 never occurs:
        finite = jnp.isfinite(jnp.diag(cov_act))
        p_act = cov_act.shape[0]
        safe_cov = jnp.where(
            finite[:, None] & finite[None, :],
            cov_act,
            jnp.eye(p_act, dtype=jnp.float64),
        )
        c_safe = _concrete(safe_cov)
        if c_safe is not None and p_act > 0 and onp.min(onp.linalg.eigvalsh(c_safe)) <= 0.0:
            raise ValueError("Active prior_cbm_cov block must be positive definite.")
        root = jnp.linalg.cholesky(safe_cov) if p_act > 0 else jnp.zeros((0, 0), dtype=jnp.float64)
        root = jnp.where(finite[:, None] & finite[None, :], root, 0.0)
        return root, mask

    # prior_cbm_prec is not None
    prec = jnp.asarray(prior_cbm_prec, dtype=jnp.float64)
    p_act_init = int(onp.sum(mask))
    if prec.shape == (p_full, p_full):
        c_prec = _concrete(prec)
        if c_prec is not None:
            zero_rc = onp.linalg.norm(c_prec, axis=0) == 0.0
            if onp.any(zero_rc & mask):
                if c_j_col_norm is not None and onp.any(c_j_col_norm[zero_rc & mask] > 1e-12):
                    raise ValueError(
                        "Zero prior precision (flat prior) is only permitted on "
                        "zero-Fisher gauge coordinates."
                    )
                mask = mask & (~zero_rc)
        idx = jnp.asarray(onp.flatnonzero(mask), dtype=jnp.int32)
        prec_act = prec[jnp.ix_(idx, idx)]
    elif prec.shape == (p_act_init, p_act_init):
        prec_act = prec
    else:
        raise ValueError(
            f"prior_cbm_prec must have shape ({p_act_init}, {p_act_init}) or "
            f"({p_full}, {p_full}); got {prec.shape}."
        )
    prec_act = 0.5 * (prec_act + prec_act.T)
    p_act = prec_act.shape[0]
    if p_act == 0:
        return jnp.zeros((0, 0), dtype=jnp.float64), mask
    c_pa = _concrete(prec_act)
    if c_pa is not None and onp.min(onp.linalg.eigvalsh(c_pa)) <= 0.0:
        raise ValueError("Active prior_cbm_prec block must be strictly positive definite.")
    # Π_θ = L_prec @ L_prec.T => Σ_Π = Π_θ⁻¹ = L_prec⁻ᵀ @ L_prec⁻¹ => L_Π = L_prec⁻ᵀ
    nonzero = jnp.linalg.norm(prec_act, axis=0) > 0.0
    safe_prec = jnp.where(
        nonzero[:, None] & nonzero[None, :],
        prec_act,
        jnp.eye(p_act, dtype=jnp.float64),
    )
    chol_prec = jnp.linalg.cholesky(safe_prec)
    l_pi = jsl.solve_triangular(
        chol_prec, jnp.eye(p_act, dtype=jnp.float64), lower=True
    ).T
    l_pi = jnp.where(nonzero[:, None] & nonzero[None, :], l_pi, 0.0)
    return l_pi, mask


def _resolve_psi_prior_prec(
    p_psi: int,
    *,
    prior_psi_prec: Any,
    prior_psi_cov: Any,
    prior_psi_sd: Any,
) -> jax.Array:
    """Resolve ``(p_psi, p_psi)`` prior precision matrix ``Π_ψ`` for nuisance params."""
    n_given = sum(x is not None for x in (prior_psi_prec, prior_psi_cov, prior_psi_sd))
    if n_given > 1:
        raise ValueError(
            "Specify at most one of prior_psi_prec, prior_psi_cov, or prior_psi_sd."
        )
    if n_given == 0:
        return jnp.zeros((p_psi, p_psi), dtype=jnp.float64)
    if prior_psi_sd is not None:
        sd = jnp.broadcast_to(jnp.asarray(prior_psi_sd, dtype=jnp.float64), (p_psi,))
        finite = jnp.isfinite(sd) & (sd > 0.0)
        safe_sd = jnp.where(finite, sd, 1.0)
        inv_var = jnp.where(finite, 1.0 / (safe_sd**2), 0.0)
        return jnp.diag(inv_var)
    if prior_psi_cov is not None:
        cov = jnp.asarray(prior_psi_cov, dtype=jnp.float64)
        if cov.shape != (p_psi, p_psi):
            raise ValueError(f"prior_psi_cov must have shape ({p_psi}, {p_psi}).")
        sym_cov = 0.5 * (cov + cov.T)
        chol = jnp.linalg.cholesky(sym_cov)
        return jsl.cho_solve((chol, True), jnp.eye(p_psi, dtype=jnp.float64))
    prec = jnp.asarray(prior_psi_prec, dtype=jnp.float64)
    if prec.shape != (p_psi, p_psi):
        raise ValueError(f"prior_psi_prec must have shape ({p_psi}, {p_psi}).")
    return 0.5 * (prec + prec.T)


def cbm_fisher_spectrum(
    forward_fn: Callable[..., Any],
    theta_cbm: Any,
    sigma_or_prec: Any,
    *,
    psi_nuisance: Any = None,
    prior_cbm_cov: Any = None,
    prior_cbm_prec: Any = None,
    prior_cbm_root: Any = None,
    prior_cbm_sd: Any = None,
    prior_psi_prec: Any = None,
    prior_psi_cov: Any = None,
    prior_psi_sd: Any = None,
    active_cbm_mask: Any = None,
    exclude_total_mass: bool = True,
    mass_fn: Callable[[Any], Any] | None = None,
    flux_fn: Callable[[Any], Any] | None = None,
    log_space_observables: bool = True,
    noise_kind: Literal["auto", "sigma", "precision", "cov"] = "auto",
) -> FisherSpectrumResult:
    """Compute the Schur-marginalized, prior-whitened CBM Fisher eigenspectrum.

    Parameters
    ----------
    forward_fn : callable
        Forward prediction ``forward_fn(theta_cbm) -> (N_d,)`` when
        ``psi_nuisance is None``, or ``forward_fn(theta_cbm, psi_nuisance) -> (N_d,)``
        when ``psi_nuisance`` is provided.
    theta_cbm : pytree or 1D array
        Carbon box model parameters ``θ`` (e.g. unconstrained ``BalancedCBM``
        or ``EffectiveCBM`` PyTree, or a flat 1D vector).
    sigma_or_prec : array-like
        1D Gaussian standard deviations ``sigma`` or 2D precision matrix ``Σ_d⁻¹``.
    psi_nuisance : pytree or 1D array, optional
        Nuisance production and baseline-offset parameters ``ψ`` to marginalize
        via stabilized QR Schur complement.
    prior_cbm_cov, prior_cbm_prec, prior_cbm_root, prior_cbm_sd : array-like, optional
        Gaussian prior specification on ``θ_CBM`` (specify at most one; defaults
        to standard normal ``I`` on active coordinates). Infinite variances or
        zero precisions on zero-Fisher gauge coordinates (such as ``θ_tot``) are
        projected out analytically without ``inf * 0 = NaN``.
    prior_psi_prec, prior_psi_cov, prior_psi_sd : array-like, optional
        Gaussian prior specification on nuisance parameters ``ψ`` (defaults to
        zero precision / uninformative prior on identifiable nuisance columns).
    active_cbm_mask : array-like of bool, optional
        Explicit mask selecting active CBM parameters among flattened ``theta_cbm``.
    exclude_total_mass : bool, optional
        When ``True`` (default) and ``theta_cbm`` is an ``equinox.Module`` with
        a ``log_total_mass`` or ``theta_tot`` attribute and no explicit
        ``active_cbm_mask`` or finite prior on it, automatically masks that
        exact zero-Fisher gauge coordinate.
    mass_fn, flux_fn : callable, optional
        Functions mapping ``theta_cbm`` to reservoir masses ``M`` ``(n_boxes,)``
        and directed edge fluxes ``F`` ``(n_edges,)``. When ``log_space_observables``
        is ``True`` (default), sensitivities are returned for ``ln M`` and ``ln F``.
    log_space_observables : bool, optional
        If ``True`` (default), differentiate ``jnp.log(mass_fn(theta))`` and
        ``jnp.log(flux_fn(theta))``; if ``False``, differentiate ``mass_fn`` and
        ``flux_fn`` directly.
    noise_kind : {"auto", "sigma", "precision", "cov"}, optional
        Interpretation of ``sigma_or_prec``.

    Returns
    -------
    FisherSpectrumResult
        Coordinate-invariant eigenvalues, ``d_eff``, KL information gain in nats
        and bits, whitened eigenvectors, parameter loadings, marginalized Fisher,
        posterior covariance, and physical mass/flux loadings.
    """
    flat_theta, unravel_theta = _partition_inexact(theta_cbm)
    p_full = flat_theta.shape[0]

    # Auto-detect log_total_mass / theta_tot gauge leaf on PyTree CBM objects
    # if no explicit active_cbm_mask is given and no prior was supplied for it.
    if (
        active_cbm_mask is None
        and exclude_total_mass
        and isinstance(theta_cbm, eqx.Module)
        and all(
            x is None
            for x in (prior_cbm_cov, prior_cbm_prec, prior_cbm_root, prior_cbm_sd)
        )
    ):
        for gauge_attr in ("log_total_mass", "theta_tot"):
            if hasattr(theta_cbm, gauge_attr) and eqx.is_inexact_array(
                getattr(theta_cbm, gauge_attr)
            ):
                diff_tree, _ = eqx.partition(theta_cbm, eqx.is_inexact_array)
                ones_tree = jax.tree_util.tree_map(
                    lambda leaf: onp.ones(leaf.shape, dtype=onp.float64), diff_tree
                )
                marked_tree = eqx.tree_at(
                    lambda m, attr=gauge_attr: getattr(m, attr),
                    ones_tree,
                    onp.zeros(getattr(diff_tree, gauge_attr).shape, dtype=onp.float64),
                )
                flat_mask_np = onp.concatenate(
                    [onp.ravel(leaf) for leaf in jax.tree_util.tree_leaves(marked_tree)]
                )
                active_cbm_mask = flat_mask_np > 0.5
                break

    if psi_nuisance is None:
        j_theta_raw = jax.jacfwd(
            lambda th: jnp.asarray(forward_fn(unravel_theta(th)), dtype=jnp.float64).reshape(-1)
        )(flat_theta)
        w_theta = _whiten_jacobian(j_theta_raw, sigma_or_prec, kind=noise_kind)
        w_psi = jnp.zeros((w_theta.shape[0], 0), dtype=jnp.float64)
        joint_fisher = 0.5 * (w_theta.T @ w_theta + (w_theta.T @ w_theta).T)
    else:
        flat_psi, unravel_psi = _partition_inexact(psi_nuisance)

        def joint_forward(th: jax.Array, ps: jax.Array) -> jax.Array:
            return jnp.asarray(
                forward_fn(unravel_theta(th), unravel_psi(ps)), dtype=jnp.float64
            ).reshape(-1)

        j_theta_raw, j_psi_raw = jax.jacfwd(joint_forward, argnums=(0, 1))(
            flat_theta, flat_psi
        )
        w_joint = _whiten_jacobian(
            jnp.concatenate([j_theta_raw, j_psi_raw], axis=1),
            sigma_or_prec,
            kind=noise_kind,
        )
        w_theta = w_joint[:, :p_full]
        w_psi = w_joint[:, p_full:]
        joint_fisher = w_joint.T @ w_joint
        joint_fisher = 0.5 * (joint_fisher + joint_fisher.T)

    l_pi, active_mask = _resolve_cbm_prior_root(
        p_full,
        active_cbm_mask=active_cbm_mask,
        prior_cbm_root=prior_cbm_root,
        prior_cbm_cov=prior_cbm_cov,
        prior_cbm_prec=prior_cbm_prec,
        prior_cbm_sd=prior_cbm_sd,
        whitened_j_theta=w_theta,
    )
    c_mask = active_mask
    active_idx = jnp.asarray(onp.flatnonzero(c_mask), dtype=jnp.int32)
    p_act = active_idx.shape[0]

    # Stabilized QR Schur-complement marginalization over nuisance ψ:
    # Project A_θ = [W_θ; 0] orthogonally off col(A_ψ), A_ψ = [W_ψ; L_{Π_ψ}ᵀ],
    # so J_{θ,⊥}ᵀ J_{θ,⊥} = F_θθ - F_θψ (F_ψψ + Π_ψ)⁻¹ F_ψθ without normal-equation
    # condition-number squaring or cancellation.
    p_psi = w_psi.shape[1]
    if p_psi > 0:
        pi_psi = _resolve_psi_prior_prec(
            p_psi,
            prior_psi_prec=prior_psi_prec,
            prior_psi_cov=prior_psi_cov,
            prior_psi_sd=prior_psi_sd,
        )
        # pi_psi = L_psi @ L_psi.T => L_psi.T has Gram matrix L_psi @ L_psi.T = pi_psi
        l_psi = _psd_square_root_factor(pi_psi)
        a_psi = jnp.concatenate([w_psi, l_psi.T], axis=0)
        a_theta_full = jnp.concatenate(
            [w_theta, jnp.zeros((p_psi, p_full), dtype=jnp.float64)], axis=0
        )
        q_psi, r_psi = jnp.linalg.qr(a_psi, mode="reduced")
        r_diag = jnp.abs(jnp.diag(r_psi))
        r_scale = jnp.maximum(jnp.max(r_diag), 1.0)
        col_ok = (r_diag > 1e-13 * r_scale).astype(jnp.float64)
        q_psi = q_psi * col_ok[None, :]
        # Two-pass DGKS orthogonal projection guarantees ||Q_ψᵀ J_⊥|| < eps_mach:
        j_full_perp = a_theta_full - q_psi @ (q_psi.T @ a_theta_full)
        j_full_perp = j_full_perp - q_psi @ (q_psi.T @ j_full_perp)
    else:
        j_full_perp = w_theta

    marginal_fisher = j_full_perp.T @ j_full_perp
    marginal_fisher = 0.5 * (marginal_fisher + marginal_fisher.T)
    j_act_perp = j_full_perp[:, active_idx]

    # Prior-whitened operator W_whitened = J_{θ,act,⊥} @ L_Π:
    # SVD directly yields descending μ_k = s_k² >= 0 with full dynamic range down
    # to 1e-30 (never squaring condition numbers via F̃ = Wᵀ W).
    if p_act > 0:
        w_whitened = j_act_perp @ l_pi
        whitened_fisher = w_whitened.T @ w_whitened
        whitened_fisher = 0.5 * (whitened_fisher + whitened_fisher.T)
        if w_whitened.shape[0] < p_act:
            w_svd_in = jnp.concatenate(
                [
                    w_whitened,
                    jnp.zeros(
                        (p_act - w_whitened.shape[0], p_act), dtype=jnp.float64
                    ),
                ],
                axis=0,
            )
        else:
            w_svd_in = w_whitened
        _, s_vals, vt = jnp.linalg.svd(w_svd_in, full_matrices=False)
        v_modes = vt.T
        eigenvalues = s_vals[:p_act] ** 2
        loadings_act = l_pi @ v_modes  # (p_act, p_act)
        parameter_loadings = (
            jnp.zeros((p_full, p_act), dtype=jnp.float64)
            .at[active_idx, :]
            .set(loadings_act)
        )
        posterior_cov = (loadings_act * (1.0 / (1.0 + eigenvalues))[None, :]) @ loadings_act.T
        posterior_cov = 0.5 * (posterior_cov + posterior_cov.T)
    else:
        eigenvalues = jnp.zeros((0,), dtype=jnp.float64)
        v_modes = jnp.zeros((0, 0), dtype=jnp.float64)
        whitened_fisher = jnp.zeros((0, 0), dtype=jnp.float64)
        parameter_loadings = jnp.zeros((p_full, 0), dtype=jnp.float64)
        posterior_cov = jnp.zeros((0, 0), dtype=jnp.float64)

    d_eff = jnp.sum(eigenvalues / (eigenvalues + 1.0))
    info_nats = 0.5 * jnp.sum(jnp.log1p(eigenvalues))
    info_bits = info_nats / math.log(2.0)

    mass_loadings: jax.Array | None = None
    if mass_fn is not None:
        def _eval_mass(th: jax.Array) -> jax.Array:
            val = jnp.asarray(mass_fn(unravel_theta(th)), dtype=jnp.float64).reshape(-1)
            return jnp.log(val) if log_space_observables else val

        j_mass = jax.jacfwd(_eval_mass)(flat_theta)
        mass_loadings = j_mass @ parameter_loadings

    flux_loadings: jax.Array | None = None
    if flux_fn is not None:
        def _eval_flux(th: jax.Array) -> jax.Array:
            val = jnp.asarray(flux_fn(unravel_theta(th)), dtype=jnp.float64).reshape(-1)
            return jnp.log(val) if log_space_observables else val

        j_flux = jax.jacfwd(_eval_flux)(flat_theta)
        flux_loadings = j_flux @ parameter_loadings

    return FisherSpectrumResult(
        eigenvalues=eigenvalues,
        d_eff=d_eff,
        info_nats=info_nats,
        info_bits=info_bits,
        prior_whitened_modes=v_modes,
        parameter_loadings=parameter_loadings,
        marginal_fisher=marginal_fisher,
        whitened_fisher=whitened_fisher,
        joint_fisher=joint_fisher,
        posterior_cov=posterior_cov,
        mass_loadings=mass_loadings,
        flux_loadings=flux_loadings,
        active_mask=jnp.asarray(c_mask),
    )


class RelaxationModesResult(NamedTuple):
    """Singular-value spectrum and eigenmodes of continuous Laplace relaxation.

    Attributes
    ----------
    rates : jax.Array
        Log-spaced relaxation rates ``s_j`` in ``yr⁻¹``, shape ``(n_rates,)``.
    residence_times : jax.Array
        Corresponding residence times ``τ_res,j = 1 / s_j`` in ``yr``.
    singular_values : jax.Array
        Descending singular values ``σ_1 >= σ_2 >= ... >= 0`` of the noise- and
        measure-weighted seasonal Laplace operator.
    eigenvalues : jax.Array
        Squared singular values ``μ_k = σ_k²`` (comparing directly against
        ``π / cosh(π ω_ln)`` under ``measure="l2_ds"``).
    rate_modes : jax.Array
        Right singular vectors ``V`` ``(n_rates, n_modes)`` in discrete
        measure-orthonormal coordinates (columns are modes).
    data_modes : jax.Array
        Left singular vectors ``U`` ``(N_d, n_modes)`` in noise-whitened data space.
    design_matrix : jax.Array
        Weighted ``(N_d, n_rates)`` operator subjected to SVD.
    d_eff : jax.Array
        Effective mode count ``Σ_k σ_k² / (σ_k² + 1)`` at unit prior scale.
    info_nats : jax.Array
        Information gain ``½ Σ_k ln(1 + σ_k²)`` in nats.
    info_bits : jax.Array
        Information gain in bits.
    measure : str
        Measure convention used (``"l2_ds"`` or ``"d_ln_s"``).
    """

    rates: jax.Array
    residence_times: jax.Array
    singular_values: jax.Array
    eigenvalues: jax.Array
    rate_modes: jax.Array
    data_modes: jax.Array
    design_matrix: jax.Array
    d_eff: jax.Array
    info_nats: jax.Array
    info_bits: jax.Array
    measure: str


def _normalize_seasonal_windows(
    t_starts: Any,
    t_ends: Any | None,
    sub_weights: Any | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return ``(starts, ends, weights)`` all of shape ``(N_d, M_sub)`` in ``float64``."""
    starts = jnp.asarray(t_starts, dtype=jnp.float64)
    if t_ends is None:
        ends = starts
    else:
        ends = jnp.asarray(t_ends, dtype=jnp.float64)

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
        safe_sum = jnp.where(w_sum > 0.0, w_sum, 1.0)
        weights = w_raw / safe_sum

    c_starts, c_ends = _concrete(starts), _concrete(ends)
    if c_starts is not None and c_ends is not None and onp.any(c_ends < c_starts):
        raise ValueError("All seasonal windows must satisfy t_ends >= t_starts.")
    return starts, ends, weights


def relaxation_fisher_modes(
    t_starts: Any,
    t_ends: Any | None = None,
    sub_weights: Any | None = None,
    sigma_d: Any = 1.0,
    s_min: float = 1e-4,
    s_max: float = 20.0,
    n_rates: int = 128,
    measure: Literal["l2_ds", "d_ln_s"] = "l2_ds",
    marginalize_dc_and_scale: bool = False,
    reference_profile: Any | None = None,
) -> RelaxationModesResult:
    """Compute singular values and eigenmodes of the seasonal-binned Laplace kernel.

    Models post-impulse relaxation ``g(τ) = ∫₀^∞ ρ(s) exp(-s τ) ds`` integrated
    analytically across each tree ring's seasonal growth sub-windows
    ``[a_{i,m}, b_{i,m}]`` (``a_{i,m} >= 0`` relative to impulse time):

    .. math::

        K_{\\text{raw}, i, j} = \\sum_{m=1}^{M_{\\text{sub}}} w_{i,m}\\,
        \\psi_1\\!\\big(-s_j (b_{i,m} - a_{i,m})\\big)\\, e^{-s_j a_{i,m}},
        \\qquad \\psi_1(-x) = \\frac{1 - e^{-x}}{x}.

    Under ``measure="l2_ds"``, column ``j`` carries weight
    ``sqrt(s_j * Δ(ln s))``, making the discrete inner product isometric to
    ``L²(ℝ_+, ds)`` and reproducing McWhirter & Pike's (1978) analytical Mellin
    power spectrum ``σ_k² ≈ π / cosh(π ω_ln,k)`` (with ``ω_ln,k = π k / W_ln``
    over observable log-time passband ``W_ln = ln(τ_max / τ_min)``). Under
    ``measure="d_ln_s"``, column ``j`` carries weight ``sqrt(Δ(ln s))`` for
    scale-invariant dimensionless residue densities ``w(s) = s ρ(s)``.
    """
    if s_min <= 0.0 or s_max <= s_min:
        raise ValueError(f"Require 0 < s_min < s_max; got s_min={s_min}, s_max={s_max}.")
    if n_rates < 2:
        raise ValueError(f"n_rates must be >= 2; got {n_rates}.")
    if measure not in ("l2_ds", "d_ln_s"):
        raise ValueError(f"measure must be 'l2_ds' or 'd_ln_s'; got {measure!r}.")

    starts, ends, weights = _normalize_seasonal_windows(t_starts, t_ends, sub_weights)
    c_starts = _concrete(starts)
    if c_starts is not None and onp.any(c_starts < 0.0):
        raise ValueError(
            "t_starts for relaxation_fisher_modes must be >= 0 (post-impulse lag τ)."
        )

    n_obs = starts.shape[0]
    sigma = jnp.broadcast_to(jnp.asarray(sigma_d, dtype=jnp.float64), (n_obs,))
    c_sigma = _concrete(sigma)
    if c_sigma is not None and not onp.all(onp.isfinite(c_sigma) & (c_sigma > 0.0)):
        raise ValueError("sigma_d must be finite and strictly positive.")

    ln_s = jnp.linspace(math.log(s_min), math.log(s_max), n_rates, dtype=jnp.float64)
    d_ln_s = (math.log(s_max) - math.log(s_min)) / (n_rates - 1)
    rates = jnp.exp(ln_s)
    residence_times = 1.0 / rates

    # Exact seasonal window average of exp(-s_j * t) over [a_{i,m}, b_{i,m}]:
    # (N_d, M_sub, n_rates)
    durations = ends - starts
    x = durations[:, :, None] * rates[None, None, :]
    # Double-where guard prevents 0/0 VJP NaNs when duration == 0 (point sampling)
    big = x > 1e-8
    safe_x = jnp.where(big, x, 1.0)
    psi1 = jnp.where(
        big,
        -jnp.expm1(-safe_x) / safe_x,
        1.0 - 0.5 * x + (x**2) / 6.0 - (x**3) / 24.0,
    )
    sub_integrals = psi1 * jnp.exp(-starts[:, :, None] * rates[None, None, :])
    k_raw = jnp.sum(weights[:, :, None] * sub_integrals, axis=1)  # (N_d, n_rates)

    if measure == "l2_ds":
        col_weights = jnp.sqrt(rates * d_ln_s)
    else:
        col_weights = jnp.full_like(rates, math.sqrt(d_ln_s))

    design = (k_raw / sigma[:, None]) * col_weights[None, :]

    if marginalize_dc_and_scale:
        dc_col = 1.0 / sigma
        if reference_profile is None:
            ref_raw = jnp.mean(k_raw, axis=1)
        else:
            ref_raw = jnp.asarray(reference_profile, dtype=jnp.float64).reshape(n_obs)
        scale_col = ref_raw / sigma
        nuisance = jnp.stack([dc_col, scale_col], axis=1)
        q_nuis, r_nuis = jnp.linalg.qr(nuisance, mode="reduced")
        r_diag = jnp.abs(jnp.diag(r_nuis))
        r_top = jnp.maximum(jnp.max(r_diag), 1.0)
        col_ok = (r_diag > 1e-13 * r_top).astype(jnp.float64)
        q_nuis = q_nuis * col_ok[None, :]
        design = design - q_nuis @ (q_nuis.T @ design)
        design = design - q_nuis @ (q_nuis.T @ design)

    u_modes, s_vals, vt = jnp.linalg.svd(design, full_matrices=False)
    v_modes = vt.T
    evals = s_vals**2
    d_eff = jnp.sum(evals / (evals + 1.0))
    info_nats = 0.5 * jnp.sum(jnp.log1p(evals))
    info_bits = info_nats / math.log(2.0)

    return RelaxationModesResult(
        rates=rates,
        residence_times=residence_times,
        singular_values=s_vals,
        eigenvalues=evals,
        rate_modes=v_modes,
        data_modes=u_modes,
        design_matrix=design,
        d_eff=d_eff,
        info_nats=info_nats,
        info_bits=info_bits,
        measure=measure,
    )
