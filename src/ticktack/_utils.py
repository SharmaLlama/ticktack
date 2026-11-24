"""Shared internal array, Tracer-concretization, jitter, and PSD square-root helpers."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike


def _concrete(value: Any) -> np.ndarray | None:
    """Return ``value`` as a NumPy array, or ``None`` when traced under JAX."""
    try:
        return np.asarray(value)
    except (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError):
        return None


def _sigma_with_jitter(sigma: ArrayLike, sigma_jitter: ArrayLike | None = None) -> jax.Array:
    """Return ``float64`` uncertainties ``sqrt(sigma^2 + sigma_jitter^2)``."""
    sig = jnp.asarray(sigma, dtype=jnp.float64)
    if sigma_jitter is None:
        return sig
    return jnp.sqrt(jnp.square(sig) + jnp.square(jnp.asarray(sigma_jitter, dtype=jnp.float64)))


def _safe_psd_root(sym_mat: ArrayLike, jitter: float = 1e-13) -> jax.Array:
    """Return ``L`` with ``L @ L.T ≈ sym_mat`` and smooth autodiff across spectra."""
    arr = jnp.asarray(sym_mat, dtype=jnp.float64)
    n = arr.shape[0]
    if n == 0:
        return jnp.zeros((0, 0), dtype=jnp.float64)
    sym = 0.5 * (arr + arr.T)
    p0 = jax.lax.stop_gradient(jnp.linalg.cholesky(sym))
    ok0 = jnp.all(jnp.isfinite(p0)) & jnp.all(jnp.diag(p0) > 0.0)

    def _eigh_branch(m: jax.Array) -> jax.Array:
        evals, evecs = jnp.linalg.eigh(0.5 * (m + m.T))
        tol = 1e-30 if jitter > 0.0 else 1e-15 * jnp.maximum(jnp.max(evals), 1.0)
        pos = evals > tol
        return evecs * jnp.where(pos, jnp.sqrt(jnp.where(pos, evals, 1.0)), 0.0)[None, :]

    def _fallback(m: jax.Array) -> jax.Array:
        if jitter <= 0.0:
            return _eigh_branch(m)
        reg = m + (jitter * jnp.maximum(jnp.trace(m) / n, 0.0) + 1e-30) * jnp.eye(n, dtype=jnp.float64)
        pr = jax.lax.stop_gradient(jnp.linalg.cholesky(reg))
        okr = jnp.all(jnp.isfinite(pr)) & jnp.all(jnp.diag(pr) > 0.0)
        if (cr := _concrete(okr)) is not None:
            return jnp.linalg.cholesky(reg) if bool(cr) else _eigh_branch(m)
        return jax.lax.cond(okr, lambda p: jnp.linalg.cholesky(p[0]), lambda p: _eigh_branch(p[1]), (reg, m))

    if (c0 := _concrete(ok0)) is not None:
        return jnp.linalg.cholesky(sym) if bool(c0) else _fallback(sym)
    return jax.lax.cond(ok0, jnp.linalg.cholesky, _fallback, sym)
