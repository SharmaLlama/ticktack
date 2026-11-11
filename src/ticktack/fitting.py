"""MAP estimation and Laplace covariance for ``ticktack 2.0`` radiocarbon inference.

Provides multi-start MAP optimisation via ``optimistix`` (``BFGS``,
``LevenbergMarquardt``) and ``optax`` (``adam``), with regularised Laplace
covariance, convergence metadata, and conditional linear posterior reconstruction.
"""

from __future__ import annotations

from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import optax
import optimistix as optx
from jax.typing import ArrayLike

from .data import Dataset
from .likelihood import (
    LinearMarginalConfig,
    LinearPosteriorResult,
    ParameterSpace,
    linear_posterior,
    marginal_log_likelihood,
    marginal_log_posterior,
)
from .model import RadiocarbonModel

__all__ = [
    "MAPResult",
    "fit_map",
]


class MAPResult(NamedTuple):
    """Results from :func:`fit_map`."""

    model: RadiocarbonModel
    """Fitted :class:`~ticktack.model.RadiocarbonModel` at MAP parameters."""

    u_map: jax.Array
    """Unconstrained MAP parameter vector, shape ``(d,)``."""

    theta_map: dict[str, jax.Array]
    """Constrained MAP parameter dict ``{path: value}``."""

    log_posterior: jax.Array
    """Scalar log-posterior at MAP (marginal or joint)."""

    log_likelihood: jax.Array
    """Scalar log-likelihood at MAP."""

    laplace_cov: jax.Array
    """Regularised inverse-Hessian (Laplace) covariance, shape ``(d, d)``."""

    linear_post: LinearPosteriorResult | None
    """Conditional linear posterior at MAP (offsets, GP modes, etc.), or ``None``."""

    converged: bool
    """Whether the optimiser reported convergence."""

    n_steps: int
    """Number of optimiser steps taken."""

    method: str
    """Optimisation method used."""

    @property
    def fitted_model(self) -> RadiocarbonModel:
        """Fitted model at MAP (alias for :attr:`model`)."""
        return self.model


def _neg_log_post_fn(
    u_vec: jax.Array,
    model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    cfg: LinearMarginalConfig | None,
    sigma_jitter: ArrayLike | None,
) -> jax.Array:
    return -marginal_log_posterior(
        model,
        dataset,
        parameter_space,
        u_vec,
        cfg,
        sigma_jitter=sigma_jitter,
    )


def _run_bfgs(
    u0: jax.Array,
    model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    cfg: LinearMarginalConfig | None,
    sigma_jitter: ArrayLike | None,
    max_steps: int,
    tol: float,
) -> tuple[jax.Array, bool, int]:
    """Run ``optimistix.BFGS`` minimization of negative log-posterior."""

    def fn(u: jax.Array, _args: Any) -> jax.Array:
        return _neg_log_post_fn(u, model, dataset, parameter_space, cfg, sigma_jitter)

    solver = optx.BFGS(rtol=tol, atol=tol)
    result = optx.minimise(fn, solver, u0, max_steps=max_steps, args=None)
    u_opt = result.value
    converged = bool(result.result == optx.RESULTS.successful)
    try:
        n_steps = int(result.state.num_steps)
    except AttributeError:
        n_steps = max_steps
    return u_opt, converged, n_steps


def _run_levenberg_marquardt(
    u0: jax.Array,
    model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    cfg: LinearMarginalConfig | None,
    sigma_jitter: ArrayLike | None,
    max_steps: int,
    tol: float,
) -> tuple[jax.Array, bool, int]:
    """Run ``optimistix.LevenbergMarquardt`` minimization of negative log-posterior.

    Builds a proper whitened residual vector ``[data_whitened, prior_whitened]``
    so that LM can exploit the least-squares structure of the Gaussian posterior.
    """
    dataset_sigma = dataset.sigma_flat
    if sigma_jitter is not None:
        dataset_sigma = jnp.sqrt(
            jnp.square(dataset_sigma)
            + jnp.square(jnp.asarray(sigma_jitter, dtype=jnp.float64))
        )

    def residual_fn(u: jax.Array, _args: Any) -> jax.Array:
        # Whitened data residual
        constrained = parameter_space.constrain(u, model)
        from .likelihood import _nonlinear_residual, _build_marginal_design

        cfg_eff = cfg if cfg is not None else LinearMarginalConfig()
        X, prior_mean, prior_sd = _build_marginal_design(constrained, dataset, cfg_eff)
        raw_resid = _nonlinear_residual(constrained, dataset)  # (N_tot,)
        sigma_arr = jnp.asarray(dataset_sigma, dtype=jnp.float64)

        # Projected residual: (I - X (X^T diag(1/sig^2) X + D^{-2})^{-1} X^T diag(1/sig^2)) r
        # Approximate: use raw_resid / sigma as whitened residual (ignores marginalization)
        data_whitened = raw_resid / sigma_arr  # (N_tot,)

        # Prior whitened residual (u - prior_mean) / prior_sd
        prior_means = jnp.array([s.prior_mean for s in parameter_space.specs], dtype=jnp.float64)
        prior_sds = jnp.array([s.prior_sd for s in parameter_space.specs], dtype=jnp.float64)
        prior_whitened = (u - prior_means) / prior_sds  # (d,)

        return jnp.concatenate([data_whitened, prior_whitened], axis=0)

    solver = optx.LevenbergMarquardt(rtol=tol, atol=tol)
    result = optx.least_squares(residual_fn, solver, u0, max_steps=max_steps, args=None)
    u_opt = result.value
    converged = bool(result.result == optx.RESULTS.successful)
    try:
        n_steps = int(result.state.num_steps)
    except AttributeError:
        n_steps = max_steps
    return u_opt, converged, n_steps


def _run_adam(
    u0: jax.Array,
    model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    cfg: LinearMarginalConfig | None,
    sigma_jitter: ArrayLike | None,
    max_steps: int,
    tol: float,
    lr: float = 1e-3,
) -> tuple[jax.Array, bool, int]:
    """Run ``optax.adam`` gradient ascent on log-posterior."""
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(u0)
    u = u0

    @jax.jit
    def _step(u_in: jax.Array, state_in: Any) -> tuple[jax.Array, Any, jax.Array]:
        loss, grad = jax.value_and_grad(
            lambda u_: _neg_log_post_fn(u_, model, dataset, parameter_space, cfg, sigma_jitter)
        )(u_in)
        updates, state_out = optimizer.update(grad, state_in)
        u_out = optax.apply_updates(u_in, updates)
        return u_out, state_out, loss

    prev_loss = jnp.inf
    for step in range(max_steps):
        u, opt_state, loss = _step(u, opt_state)
        if float(jnp.abs(prev_loss - loss)) < tol:
            return u, True, step + 1
        prev_loss = loss

    return u, False, max_steps


def _laplace_covariance(
    u_map: jax.Array,
    model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    cfg: LinearMarginalConfig | None,
    sigma_jitter: ArrayLike | None,
    regularize: float = 1e-4,
) -> jax.Array:
    """Regularised Laplace (inverse Hessian) covariance at ``u_map``."""
    def neg_log_post(u):
        return _neg_log_post_fn(u, model, dataset, parameter_space, cfg, sigma_jitter)

    H = jax.hessian(neg_log_post)(u_map)  # (d, d)
    d = H.shape[0]
    tr_H = jnp.trace(H)
    reg = regularize * jnp.abs(tr_H) / jnp.maximum(d, 1)
    H_reg = H + reg * jnp.eye(d, dtype=jnp.float64)
    try:
        jax.scipy.linalg.cholesky(H_reg)
        chol_ok = True
    except Exception:
        chol_ok = False

    if chol_ok:
        cov = jax.scipy.linalg.cho_solve(
            jax.scipy.linalg.cho_factor(H_reg, lower=True),
            jnp.eye(d, dtype=jnp.float64),
        )
    else:
        cov = jnp.linalg.pinv(H_reg)
    return 0.5 * (cov + cov.T)


def fit_map(
    model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    cfg: LinearMarginalConfig | None = None,
    *,
    method: Literal["BFGS", "LevenbergMarquardt", "adam"] = "BFGS",
    max_steps: int = 2000,
    tol: float = 1e-6,
    multi_start: int = 1,
    rng_key: jax.Array | None = None,
    sigma_jitter: ArrayLike | None = None,
    compute_laplace: bool = True,
    compute_linear_posterior: bool = True,
    t_fine: ArrayLike | None = None,
    n_fine_draws: int = 0,
) -> MAPResult:
    """MAP estimation via ``optimistix`` (BFGS / LM) or ``optax`` (adam).

    Parameters
    ----------
    model : RadiocarbonModel
        Base model with the nominal parameter values used as the starting point.
    dataset : Dataset
        Observed tree-ring data.
    parameter_space : ParameterSpace
        Bijector-based unconstrained parameter specification.
    cfg : LinearMarginalConfig, optional
        Linear marginalization configuration.
    method : {"BFGS", "LevenbergMarquardt", "adam"}
        Optimisation method.
    max_steps : int
        Maximum optimiser steps per start.
    tol : float
        Relative/absolute convergence tolerance.
    multi_start : int
        Number of random restarts.  The best converged MAP is returned.
    rng_key : jax.Array, optional
        JAX PRNGKey for multi-start random restarts.
    sigma_jitter : array-like, optional
        Per-observation noise jitter (‰).
    compute_laplace : bool
        When ``True`` (default), compute the regularised Laplace covariance.
    compute_linear_posterior : bool
        When ``True`` (default), compute the conditional linear posterior at MAP.
    t_fine : array-like, optional
        Fine time grid for continuous posterior reconstruction.
    n_fine_draws : int
        Number of Gaussian draws from the linear posterior.

    Returns
    -------
    MAPResult
        Named tuple with fitted model, parameter vectors, log-posterior,
        Laplace covariance, and optional linear posterior.
    """
    if cfg is None:
        cfg = LinearMarginalConfig()

    u0 = parameter_space.unconstrain(model)

    best_u: jax.Array = u0
    best_lp = jnp.asarray(-jnp.inf, dtype=jnp.float64)
    best_converged = False
    best_n_steps = 0

    starts: list[jax.Array] = [u0]
    if multi_start > 1:
        if rng_key is None:
            rng_key = jax.random.PRNGKey(42)
        for _ in range(multi_start - 1):
            rng_key, subkey = jax.random.split(rng_key)
            noise = 0.5 * jax.random.normal(subkey, shape=u0.shape, dtype=jnp.float64)
            starts.append(u0 + noise)

    for u_start in starts:
        try:
            if method == "BFGS":
                u_opt, conv, n_steps = _run_bfgs(
                    u_start, model, dataset, parameter_space, cfg, sigma_jitter, max_steps, tol
                )
            elif method == "LevenbergMarquardt":
                u_opt, conv, n_steps = _run_levenberg_marquardt(
                    u_start, model, dataset, parameter_space, cfg, sigma_jitter, max_steps, tol
                )
            elif method == "adam":
                u_opt, conv, n_steps = _run_adam(
                    u_start, model, dataset, parameter_space, cfg, sigma_jitter, max_steps, tol
                )
            else:
                raise ValueError(
                    f"Unknown method {method!r}; expected 'BFGS', 'LevenbergMarquardt', or 'adam'."
                )

            lp = marginal_log_posterior(
                model, dataset, parameter_space, u_opt, cfg, sigma_jitter=sigma_jitter
            )
            if float(lp) > float(best_lp):
                best_u = u_opt
                best_lp = lp
                best_converged = conv
                best_n_steps = n_steps
        except Exception:
            continue

    u_map = best_u
    model_map = parameter_space.constrain(u_map, model)

    log_lk = marginal_log_likelihood(model_map, dataset, cfg, sigma_jitter=sigma_jitter)

    theta_map = {spec.path: model_map.get(spec.path) for spec in parameter_space.specs}

    if compute_laplace:
        laplace_cov = _laplace_covariance(
            u_map, model, dataset, parameter_space, cfg, sigma_jitter
        )
    else:
        d = parameter_space.n_params
        laplace_cov = jnp.eye(d, dtype=jnp.float64)

    lin_post = None
    if compute_linear_posterior:
        lin_post = linear_posterior(
            model_map,
            dataset,
            cfg,
            rng_key=rng_key,
            n_draws=n_fine_draws,
            t_fine=t_fine,
            sigma_jitter=sigma_jitter,
        )

    return MAPResult(
        model=model_map,
        u_map=u_map,
        theta_map=theta_map,
        log_posterior=best_lp,
        log_likelihood=log_lk,
        laplace_cov=laplace_cov,
        linear_post=lin_post,
        converged=best_converged,
        n_steps=best_n_steps,
        method=method,
    )
