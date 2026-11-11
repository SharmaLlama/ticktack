"""MAP estimation and Laplace covariance for ``ticktack 2.0`` radiocarbon inference.

Provides multi-start MAP optimization via ``optimistix`` (``BFGS``,
``LevenbergMarquardt``) and ``optax`` (``adam``), with eigenspectrum-regularized
Laplace covariance, convergence metadata, and conditional linear posterior
reconstruction.
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
    _build_marginal_design,
    _has_loggp,
    _make_fast_unconstrained_loglike,
    _nonlinear_residual,
    _offset_only_cfg,
    joint_log_likelihood,
    joint_log_posterior,
    linear_posterior,
    marginal_log_likelihood,
    marginal_log_posterior,
)
from .linear import LinearMarginal
from .model import RadiocarbonModel
from .solvers import ModaxSolver

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
    """Constrained MAP parameter dictionary ``{path: value}``."""

    log_posterior: jax.Array
    """Scalar log-posterior at MAP (marginal or joint)."""

    log_likelihood: jax.Array
    """Scalar log-likelihood at MAP."""

    laplace_cov: jax.Array
    """Regularized inverse-Hessian (Laplace) covariance in ``u`` space, shape ``(d, d)``."""

    linear_post: LinearPosteriorResult | None
    """Conditional linear posterior at MAP (offsets, GP modes, etc.), or ``None``."""

    converged: bool
    """Whether the optimizer reported convergence."""

    n_steps: int
    """Number of optimizer steps taken."""

    method: str
    """Optimization method used."""

    @property
    def fitted_model(self) -> RadiocarbonModel:
        """Fitted model at MAP (alias for :attr:`model`)."""
        return self.model


def _use_joint_mode(model: RadiocarbonModel) -> bool:
    return _has_loggp(model.production) or isinstance(model.solver, ModaxSolver)


def _eval_log_posterior(
    u_vec: jax.Array,
    model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    cfg: LinearMarginalConfig,
    sigma_jitter: ArrayLike | None,
) -> jax.Array:
    if _use_joint_mode(model):
        return joint_log_posterior(
            model,
            dataset,
            parameter_space,
            u_vec,
            cfg=cfg,
            sigma_jitter=sigma_jitter,
        )
    return marginal_log_posterior(
        model,
        dataset,
        parameter_space,
        u_vec,
        cfg,
        sigma_jitter=sigma_jitter,
    )


def _eval_log_likelihood(
    model_eval: RadiocarbonModel,
    dataset: Dataset,
    cfg: LinearMarginalConfig,
    sigma_jitter: ArrayLike | None,
) -> jax.Array:
    if _use_joint_mode(model_eval):
        return joint_log_likelihood(
            model_eval,
            dataset,
            cfg=cfg,
            sigma_jitter=sigma_jitter,
        )
    return marginal_log_likelihood(
        model_eval,
        dataset,
        cfg,
        sigma_jitter=sigma_jitter,
    )


def _neg_log_post_fn(
    u_vec: jax.Array,
    model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    cfg: LinearMarginalConfig,
    sigma_jitter: ArrayLike | None,
) -> jax.Array:
    return -_eval_log_posterior(
        u_vec, model, dataset, parameter_space, cfg, sigma_jitter
    )


def _run_bfgs(
    u0: jax.Array,
    model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    cfg: LinearMarginalConfig,
    sigma_jitter: ArrayLike | None,
    max_steps: int,
    tol: float,
) -> tuple[jax.Array, bool, int]:
    """Run JIT-compiled ``optimistix.BFGS`` minimization of negative log-posterior."""
    fast_lk_fn = _make_fast_unconstrained_loglike(
        model, dataset, parameter_space, cfg, sigma_jitter
    )

    def fn(u: jax.Array, _args: Any) -> jax.Array:
        return -(fast_lk_fn(u) + parameter_space.log_prior_unconstrained(u))

    solver = optx.BFGS(rtol=tol, atol=tol)

    @jax.jit
    def _solve(u_init: jax.Array) -> tuple[jax.Array, Any, Any]:
        res = optx.minimise(
            fn, solver, u_init, max_steps=max_steps, args=None, throw=False
        )
        steps = (
            res.stats.get("num_steps", max_steps)
            if hasattr(res, "stats")
            else max_steps
        )
        return res.value, res.result == optx.RESULTS.successful, steps

    u_opt, conv_arr, steps_arr = _solve(u0)
    return u_opt, bool(conv_arr), int(steps_arr)


def _run_levenberg_marquardt(
    u0: jax.Array,
    model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    cfg: LinearMarginalConfig,
    sigma_jitter: ArrayLike | None,
    max_steps: int,
    tol: float,
) -> tuple[jax.Array, bool, int]:
    """Run JIT-compiled ``optimistix.LevenbergMarquardt`` on the whitened ``[data, prior]`` residual vector."""
    sigma_arr = jnp.asarray(dataset.sigma_flat, dtype=jnp.float64)
    if sigma_jitter is not None:
        sigma_arr = jnp.sqrt(
            jnp.square(sigma_arr)
            + jnp.square(jnp.asarray(sigma_jitter, dtype=jnp.float64))
        )

    prior_means = parameter_space.prior_means
    prior_sds = parameter_space.prior_sds
    joint_mode = _use_joint_mode(model)
    eff_cfg = _offset_only_cfg(cfg) if joint_mode else cfg

    @jax.jit
    def _whitened_pair(u: jax.Array) -> tuple[jax.Array, jax.Array]:
        constrained = parameter_space.constrain(u, model)
        if joint_mode and eff_cfg.offset_mode == "none":
            pred = constrained.predict_d14c(dataset, offsets=None, concat=True)
            u_data = (dataset.d14c_flat - pred) / sigma_arr
            hld = jnp.sum(jnp.log(sigma_arr))
        else:
            X, pm, ps = _build_marginal_design(constrained, dataset, eff_cfg)
            raw_resid = _nonlinear_residual(constrained, dataset, eff_cfg)
            lm = LinearMarginal(
                X, prior_mean=pm, prior_sd=ps, method=eff_cfg.whiten_method
            )
            u_data, hld = lm.whiten(raw_resid, sigma_arr)
        u_prior = (u - prior_means) / prior_sds
        return jnp.concatenate([u_data, u_prior], axis=0), hld

    _, hld0 = _whitened_pair(u0)

    def residual_fn(u: jax.Array, _args: Any) -> jax.Array:
        r_vec, _ = _whitened_pair(u)
        return r_vec

    solver = optx.LevenbergMarquardt(rtol=tol, atol=tol)

    @jax.jit
    def _solve_lm(u_init: jax.Array) -> tuple[jax.Array, Any, Any]:
        res = optx.least_squares(
            residual_fn, solver, u_init, max_steps=max_steps, args=None, throw=False
        )
        steps = (
            res.stats.get("num_steps", max_steps)
            if hasattr(res, "stats")
            else max_steps
        )
        return res.value, res.result == optx.RESULTS.successful, steps

    u_opt, conv_arr, steps_arr = _solve_lm(u0)
    converged = bool(conv_arr)
    n_steps = int(steps_arr)

    # If log-determinant varied with u (e.g. CBM or GP kernel hyperparameters),
    # polish with BFGS on the exact log-det-included objective.
    _, hld_opt = _whitened_pair(u_opt)
    if float(jnp.abs(hld_opt - hld0)) > 1e-9:
        u_opt, conv_bfgs, steps_bfgs = _run_bfgs(
            u_opt,
            model,
            dataset,
            parameter_space,
            cfg,
            sigma_jitter,
            max(max_steps // 2, 50),
            tol,
        )
        converged = converged or conv_bfgs
        n_steps = n_steps + steps_bfgs

    return u_opt, converged, n_steps


def _run_adam(
    u0: jax.Array,
    model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    cfg: LinearMarginalConfig,
    sigma_jitter: ArrayLike | None,
    max_steps: int,
    tol: float,
    lr: float = 1e-2,
) -> tuple[jax.Array, bool, int]:
    """Run ``optax.adam`` gradient descent on negative log-posterior."""
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(u0)
    u = u0
    fast_lk_fn = _make_fast_unconstrained_loglike(
        model, dataset, parameter_space, cfg, sigma_jitter
    )

    @jax.jit
    def _step(u_in: jax.Array, state_in: Any) -> tuple[jax.Array, Any, jax.Array]:
        loss, grad = jax.value_and_grad(
            lambda u_: -(
                fast_lk_fn(u_) + parameter_space.log_prior_unconstrained(u_)
            )
        )(u_in)
        updates, state_out = optimizer.update(grad, state_in)
        u_out = optax.apply_updates(u_in, updates)
        return u_out, state_out, loss

    prev_loss = jnp.inf
    best_u = u
    best_loss = jnp.inf
    for step in range(max_steps):
        u, opt_state, loss = _step(u, opt_state)
        if float(loss) < float(best_loss):
            best_loss = loss
            best_u = u
        if step > 5 and float(jnp.abs(prev_loss - loss)) < tol:
            return best_u, True, step + 1
        prev_loss = loss

    return best_u, False, max_steps


def _laplace_covariance(
    u_map: jax.Array,
    model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    cfg: LinearMarginalConfig,
    sigma_jitter: ArrayLike | None,
    regularize: float = 1e-6,
) -> jax.Array:
    """Strictly SPD eigenspectrum-regularized Laplace covariance at ``u_map``."""
    d = parameter_space.n_params
    if d == 0:
        return jnp.zeros((0, 0), dtype=jnp.float64)

    if isinstance(model.solver, ModaxSolver):
        # ModaxSolver supports 1st-order custom_vjp only; fall back to diagonal prior cov
        return jnp.diag(jnp.square(parameter_space.prior_sds))

    fast_lk_fn = _make_fast_unconstrained_loglike(
        model, dataset, parameter_space, cfg, sigma_jitter
    )

    def neg_log_post(u: jax.Array) -> jax.Array:
        return -(fast_lk_fn(u) + parameter_space.log_prior_unconstrained(u))

    hess_fn = (
        jax.jit(jax.jacfwd(jax.jacfwd(neg_log_post)))
        if d <= 16
        else jax.jit(jax.hessian(neg_log_post))
    )
    H = hess_fn(u_map)
    H_sym = 0.5 * (H + H.T)
    prior_prec_floor = 1.0 / jnp.maximum(jnp.max(jnp.square(parameter_space.prior_sds)), 1.0)
    eigvals, eigvecs = jnp.linalg.eigh(H_sym)
    scale_floor = jnp.maximum(
        regularize * jnp.maximum(jnp.max(jnp.abs(eigvals)), 1.0),
        1e-8 * prior_prec_floor,
    )
    safe_eigvals = jnp.maximum(eigvals, scale_floor)
    cov = (eigvecs * (1.0 / safe_eigvals)[None, :]) @ eigvecs.T
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
    """MAP estimation via ``optimistix`` (``BFGS``, ``LevenbergMarquardt``) or ``optax`` (``adam``)."""
    if cfg is None:
        cfg = LinearMarginalConfig()

    u0 = parameter_space.unconstrain(model)
    if parameter_space.n_params == 0:
        lp0 = _eval_log_posterior(u0, model, dataset, parameter_space, cfg, sigma_jitter)
        lk0 = _eval_log_likelihood(model, dataset, cfg, sigma_jitter)
        lp_lin = (
            linear_posterior(
                model,
                dataset,
                _offset_only_cfg(cfg) if _use_joint_mode(model) else cfg,
                rng_key=rng_key,
                n_draws=n_fine_draws,
                t_fine=t_fine,
                sigma_jitter=sigma_jitter,
            )
            if compute_linear_posterior
            else None
        )
        return MAPResult(
            model=model,
            u_map=u0,
            theta_map={},
            log_posterior=lp0,
            log_likelihood=lk0,
            laplace_cov=jnp.zeros((0, 0), dtype=jnp.float64),
            linear_post=lp_lin,
            converged=True,
            n_steps=0,
            method=method,
        )

    best_u: jax.Array = u0
    best_lp = _eval_log_posterior(u0, model, dataset, parameter_space, cfg, sigma_jitter)
    best_converged = False
    best_n_steps = 0

    starts: list[jax.Array] = [u0]
    if multi_start > 1:
        if rng_key is None:
            rng_key = jax.random.PRNGKey(42)
        prior_sds = parameter_space.prior_sds
        for _ in range(multi_start - 1):
            rng_key, subkey = jax.random.split(rng_key)
            noise = 0.35 * prior_sds * jax.random.normal(
                subkey, shape=u0.shape, dtype=jnp.float64
            )
            starts.append(u0 + noise)

    if method not in ("BFGS", "LevenbergMarquardt", "adam"):
        raise ValueError(
            f"Unknown method {method!r}; expected 'BFGS', 'LevenbergMarquardt', or 'adam'."
        )

    for u_start in starts:
        if method == "BFGS":
            u_opt, conv, n_steps = _run_bfgs(
                u_start,
                model,
                dataset,
                parameter_space,
                cfg,
                sigma_jitter,
                max_steps,
                tol,
            )
        elif method == "LevenbergMarquardt":
            u_opt, conv, n_steps = _run_levenberg_marquardt(
                u_start,
                model,
                dataset,
                parameter_space,
                cfg,
                sigma_jitter,
                max_steps,
                tol,
            )
        else:
            u_opt, conv, n_steps = _run_adam(
                u_start,
                model,
                dataset,
                parameter_space,
                cfg,
                sigma_jitter,
                max_steps,
                tol,
            )

        lp = _eval_log_posterior(
            u_opt, model, dataset, parameter_space, cfg, sigma_jitter
        )
        if bool(jnp.isfinite(lp)) and float(lp) >= float(best_lp):
            best_u = u_opt
            best_lp = lp
            best_converged = conv
            best_n_steps = n_steps

    u_map = best_u
    model_map = parameter_space.constrain(u_map, model)
    log_lk = _eval_log_likelihood(model_map, dataset, cfg, sigma_jitter)
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
        lin_cfg = _offset_only_cfg(cfg) if _use_joint_mode(model_map) else cfg
        lin_post = linear_posterior(
            model_map,
            dataset,
            lin_cfg,
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
