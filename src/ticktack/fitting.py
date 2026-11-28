"""MAP estimation and Laplace covariance for ``ticktack 2.0`` radiocarbon inference.

Provides multi-start MAP optimization via ``optimistix`` (``BFGS``,
``LevenbergMarquardt``) and ``optax`` (``adam``), with eigenspectrum-regularized
Laplace covariance, convergence metadata, and conditional linear posterior
reconstruction.
"""

from __future__ import annotations

import math
from typing import Any, Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import optimistix as optx
from jax.typing import ArrayLike

from ._utils import _sigma_with_jitter
from .data import Dataset
from .likelihood import (
    FastLogLikeSpec,
    LinearMarginalConfig,
    LinearPosteriorResult,
    ParameterSpace,
    _extract_single_lgf,
    _make_fast_unconstrained_loglike,
    _offset_only_cfg,
    _uses_joint_mode,
    _whiten_model_data,
    joint_log_likelihood,
    joint_log_posterior,
    linear_posterior,
    marginal_log_likelihood,
    marginal_log_posterior,
)
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


def _eval_log_posterior(
    u_vec: jax.Array,
    model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    cfg: LinearMarginalConfig,
    sigma_jitter: ArrayLike | None,
    map_space: Literal["unconstrained", "constrained"] = "unconstrained",
) -> jax.Array:
    if _uses_joint_mode(model, parameter_space, cfg):
        return joint_log_posterior(
            model,
            dataset,
            parameter_space,
            u_vec,
            cfg=cfg,
            sigma_jitter=sigma_jitter,
            target_space=map_space,
        )
    return marginal_log_posterior(
        model,
        dataset,
        parameter_space,
        u_vec,
        cfg,
        sigma_jitter=sigma_jitter,
        target_space=map_space,
    )


def _eval_log_likelihood(
    model_eval: RadiocarbonModel,
    dataset: Dataset,
    cfg: LinearMarginalConfig,
    sigma_jitter: ArrayLike | None,
    parameter_space: ParameterSpace | None = None,
) -> jax.Array:
    if _uses_joint_mode(model_eval, parameter_space, cfg):
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


def _neg_log_post_from_spec(
    u: jax.Array,
    fast_lk_spec: FastLogLikeSpec,
    map_space: Literal["unconstrained", "constrained"],
) -> jax.Array:
    return -(
        fast_lk_spec(u)
        + fast_lk_spec.parameter_space.log_prior_unconstrained(
            u, target_space=map_space
        )
    )


@eqx.filter_jit
def _eval_fast_log_post_jit(
    fast_lk_spec: FastLogLikeSpec,
    u_vec: jax.Array,
    map_space: Literal["unconstrained", "constrained"],
) -> jax.Array:
    return fast_lk_spec(u_vec) + fast_lk_spec.parameter_space.log_prior_unconstrained(
        u_vec, target_space=map_space
    )


@eqx.filter_jit
def _eval_fast_log_lk_jit(
    fast_lk_spec: FastLogLikeSpec,
    u_vec: jax.Array,
) -> jax.Array:
    return fast_lk_spec(u_vec)


def _bfgs_obj(u: jax.Array, args: Any) -> jax.Array:
    fast_lk_spec, map_space = args
    return _neg_log_post_from_spec(u, fast_lk_spec, map_space)


@eqx.filter_jit
def _solve_bfgs_jit(
    u_init: jax.Array,
    fast_lk_spec: FastLogLikeSpec,
    max_steps: int,
    tol: float,
    map_space: Literal["unconstrained", "constrained"],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    solver = optx.BFGS(rtol=tol, atol=tol)
    res = optx.minimise(
        _bfgs_obj,
        solver,
        u_init,
        max_steps=max_steps,
        args=(fast_lk_spec, map_space),
        throw=False,
    )
    steps = (
        res.stats.get("num_steps", max_steps)
        if hasattr(res, "stats")
        else max_steps
    )
    return res.value, res.result == optx.RESULTS.successful, jnp.asarray(steps)


def _run_bfgs(
    u0: jax.Array,
    model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    cfg: LinearMarginalConfig,
    sigma_jitter: ArrayLike | None,
    max_steps: int,
    tol: float,
    map_space: Literal["unconstrained", "constrained"] = "unconstrained",
    *,
    fast_lk_spec: FastLogLikeSpec | None = None,
) -> tuple[jax.Array, bool, int]:
    """Run module-cached ``@eqx.filter_jit`` ``optimistix.BFGS`` minimization."""
    spec = (
        _make_fast_unconstrained_loglike(
            model, dataset, parameter_space, cfg, sigma_jitter
        )
        if fast_lk_spec is None
        else fast_lk_spec
    )
    u_opt, conv_arr, steps_arr = _solve_bfgs_jit(
        u0, spec, int(max_steps), float(tol), map_space
    )
    return u_opt, bool(conv_arr), int(steps_arr)


def _lm_whitened_pair_core(
    u: jax.Array,
    model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    eff_cfg: LinearMarginalConfig,
    sigma_arr: jax.Array,
    joint_mode: bool,
) -> tuple[jax.Array, jax.Array]:
    constrained = parameter_space.constrain(u, model)
    u_data, hld = _whiten_model_data(
        constrained, dataset, eff_cfg, sigma_arr, joint_mode=joint_mode
    )
    u_prior = (u - parameter_space.prior_means) / parameter_space.prior_sds
    return jnp.concatenate([u_data, u_prior], axis=0), hld


_lm_whitened_pair_jit = eqx.filter_jit(_lm_whitened_pair_core)


@eqx.filter_jit
def _lm_whitened_hld_grad_jit(
    u: jax.Array,
    model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    eff_cfg: LinearMarginalConfig,
    sigma_arr: jax.Array,
    joint_mode: bool,
) -> jax.Array:
    return jax.grad(
        lambda x: _lm_whitened_pair_core(
            x, model, dataset, parameter_space, eff_cfg, sigma_arr, joint_mode
        )[1]
    )(u)


def _lm_residual_obj(u: jax.Array, args: Any) -> jax.Array:
    model, dataset, parameter_space, eff_cfg, sigma_arr, joint_mode = args
    r_vec, _ = _lm_whitened_pair_core(
        u, model, dataset, parameter_space, eff_cfg, sigma_arr, joint_mode
    )
    return r_vec


@eqx.filter_jit
def _solve_lm_jit(
    u_init: jax.Array,
    model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    eff_cfg: LinearMarginalConfig,
    sigma_arr: jax.Array,
    joint_mode: bool,
    max_steps: int,
    tol: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    solver = optx.LevenbergMarquardt(rtol=tol, atol=tol)
    res = optx.least_squares(
        _lm_residual_obj,
        solver,
        u_init,
        max_steps=max_steps,
        args=(model, dataset, parameter_space, eff_cfg, sigma_arr, joint_mode),
        throw=False,
    )
    steps = (
        res.stats.get("num_steps", max_steps)
        if hasattr(res, "stats")
        else max_steps
    )
    return res.value, res.result == optx.RESULTS.successful, jnp.asarray(steps)


def _run_levenberg_marquardt(
    u0: jax.Array,
    model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    cfg: LinearMarginalConfig,
    sigma_jitter: ArrayLike | None,
    max_steps: int,
    tol: float,
    map_space: Literal["unconstrained", "constrained"] = "unconstrained",
    *,
    fast_lk_spec: FastLogLikeSpec | None = None,
) -> tuple[jax.Array, bool, int]:
    """Run module-cached ``@eqx.filter_jit`` ``optimistix.LevenbergMarquardt``."""
    sigma_arr = _sigma_with_jitter(dataset.sigma_flat, sigma_jitter)

    joint_mode = bool(_uses_joint_mode(model, parameter_space, cfg))
    if not joint_mode and _extract_single_lgf(model.production)[0] is not None:
        return _run_bfgs(
            u0,
            model,
            dataset,
            parameter_space,
            cfg,
            sigma_jitter,
            max_steps,
            tol,
            map_space=map_space,
            fast_lk_spec=fast_lk_spec,
        )
    eff_cfg = _offset_only_cfg(cfg) if joint_mode else cfg
    max_steps_int = int(max_steps)
    tol_flt = float(tol)

    _, hld0 = _lm_whitened_pair_jit(
        u0, model, dataset, parameter_space, eff_cfg, sigma_arr, joint_mode
    )
    u_opt, conv_arr, steps_arr = _solve_lm_jit(
        u0,
        model,
        dataset,
        parameter_space,
        eff_cfg,
        sigma_arr,
        joint_mode,
        max_steps_int,
        tol_flt,
    )
    converged = bool(conv_arr)
    n_steps = int(steps_arr)

    _, hld_opt = _lm_whitened_pair_jit(
        u_opt, model, dataset, parameter_space, eff_cfg, sigma_arr, joint_mode
    )
    hld_grad = _lm_whitened_hld_grad_jit(
        u_opt, model, dataset, parameter_space, eff_cfg, sigma_arr, joint_mode
    )
    hld_grad_norm = float(jnp.linalg.norm(hld_grad))
    needs_jacobian_polish = map_space == "constrained" and any(
        s.bijector != "identity" for s in parameter_space.specs
    )
    if (
        float(jnp.abs(hld_opt - hld0)) > 1e-9
        or hld_grad_norm > tol_flt
        or needs_jacobian_polish
    ):
        spec = (
            _make_fast_unconstrained_loglike(
                model, dataset, parameter_space, cfg, sigma_jitter
            )
            if fast_lk_spec is None
            else fast_lk_spec
        )
        lp0_chk = _eval_fast_log_post_jit(spec, u0, map_space)
        lp_lm_chk = _eval_fast_log_post_jit(spec, u_opt, map_space)
        u_polish_init = (
            u_opt
            if (bool(jnp.isfinite(lp_lm_chk)) and float(lp_lm_chk) >= float(lp0_chk))
            else u0
        )
        u_opt, conv_bfgs, steps_bfgs = _run_bfgs(
            u_polish_init,
            model,
            dataset,
            parameter_space,
            cfg,
            sigma_jitter,
            max_steps_int,
            tol_flt,
            map_space=map_space,
            fast_lk_spec=spec,
        )
        converged = bool(conv_bfgs)
        n_steps = n_steps + steps_bfgs

    return u_opt, converged, n_steps


@eqx.filter_jit
def _adam_step_jit(
    u_in: jax.Array,
    state_in: Any,
    fast_lk_spec: FastLogLikeSpec,
    lr: float,
    map_space: Literal["unconstrained", "constrained"],
) -> tuple[jax.Array, Any, jax.Array, jax.Array]:
    optimizer = optax.adam(lr)
    loss, grad = jax.value_and_grad(
        lambda u_: _neg_log_post_from_spec(u_, fast_lk_spec, map_space)
    )(u_in)
    updates, state_out = optimizer.update(grad, state_in)
    u_out = optax.apply_updates(u_in, updates)
    return u_out, state_out, loss, jnp.linalg.norm(grad)


@eqx.filter_jit
def _adam_val_grad_jit(
    u_in: jax.Array,
    fast_lk_spec: FastLogLikeSpec,
    map_space: Literal["unconstrained", "constrained"],
) -> tuple[jax.Array, jax.Array]:
    return jax.value_and_grad(
        lambda u_: _neg_log_post_from_spec(u_, fast_lk_spec, map_space)
    )(u_in)


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
    map_space: Literal["unconstrained", "constrained"] = "unconstrained",
    *,
    fast_lk_spec: FastLogLikeSpec | None = None,
) -> tuple[jax.Array, bool, int]:
    """Run module-cached ``@eqx.filter_jit`` ``optax.adam`` gradient descent."""
    lr_flt = float(lr)
    tol_flt = float(tol)
    max_steps_int = int(max_steps)
    optimizer = optax.adam(lr_flt)
    opt_state = optimizer.init(u0)
    u = u0
    spec = (
        _make_fast_unconstrained_loglike(
            model, dataset, parameter_space, cfg, sigma_jitter
        )
        if fast_lk_spec is None
        else fast_lk_spec
    )
    grad_tol = max(math.sqrt(max(tol_flt, 0.0)), tol_flt)

    prev_loss = jnp.inf
    best_u = u
    best_loss = jnp.inf
    for step in range(max_steps_int):
        u_next, opt_state, loss, grad_norm = _adam_step_jit(
            u, opt_state, spec, lr_flt, map_space
        )
        if float(loss) < float(best_loss):
            best_loss = loss
            best_u = u
        if (
            step > 5
            and float(jnp.abs(prev_loss - loss)) < tol_flt
            and float(grad_norm) <= grad_tol
        ):
            return best_u, True, step + 1
        prev_loss = loss
        u = u_next

    final_loss, final_grad = _adam_val_grad_jit(u, spec, map_space)
    if float(final_loss) < float(best_loss):
        best_u = u
        if float(jnp.linalg.norm(final_grad)) <= grad_tol:
            return best_u, True, max_steps_int

    return best_u, False, max_steps_int


@eqx.filter_jit
def _laplace_cov_fwd_jit(
    u_map: jax.Array,
    fast_lk_spec: FastLogLikeSpec,
    regularize: float,
    map_space: Literal["unconstrained", "constrained"],
) -> jax.Array:
    H = jax.jacfwd(
        jax.jacfwd(lambda u_: _neg_log_post_from_spec(u_, fast_lk_spec, map_space))
    )(u_map)
    H_sym = 0.5 * (H + H.T)
    prior_sds = fast_lk_spec.parameter_space.prior_sds
    prior_prec_floor = 1.0 / jnp.maximum(jnp.max(jnp.square(prior_sds)), 1.0)
    eigvals, eigvecs = jnp.linalg.eigh(H_sym)
    scale_floor = jnp.maximum(
        regularize * jnp.maximum(jnp.max(jnp.abs(eigvals)), 1.0),
        1e-8 * prior_prec_floor,
    )
    safe_eigvals = jnp.maximum(eigvals, scale_floor)
    cov = (eigvecs * (1.0 / safe_eigvals)[None, :]) @ eigvecs.T
    return 0.5 * (cov + cov.T)


@eqx.filter_jit
def _laplace_cov_rev_jit(
    u_map: jax.Array,
    fast_lk_spec: FastLogLikeSpec,
    regularize: float,
    map_space: Literal["unconstrained", "constrained"],
) -> jax.Array:
    H = jax.hessian(
        lambda u_: _neg_log_post_from_spec(u_, fast_lk_spec, map_space)
    )(u_map)
    H_sym = 0.5 * (H + H.T)
    prior_sds = fast_lk_spec.parameter_space.prior_sds
    prior_prec_floor = 1.0 / jnp.maximum(jnp.max(jnp.square(prior_sds)), 1.0)
    eigvals, eigvecs = jnp.linalg.eigh(H_sym)
    scale_floor = jnp.maximum(
        regularize * jnp.maximum(jnp.max(jnp.abs(eigvals)), 1.0),
        1e-8 * prior_prec_floor,
    )
    safe_eigvals = jnp.maximum(eigvals, scale_floor)
    cov = (eigvecs * (1.0 / safe_eigvals)[None, :]) @ eigvecs.T
    return 0.5 * (cov + cov.T)


@eqx.filter_jit
def _neg_log_post_grad_jit(
    u_vec: jax.Array,
    fast_lk_spec: FastLogLikeSpec,
    map_space: Literal["unconstrained", "constrained"],
) -> jax.Array:
    return jax.grad(lambda u_: _neg_log_post_from_spec(u_, fast_lk_spec, map_space))(
        u_vec
    )


def _regularize_hessian_to_cov(
    H: jax.Array,
    prior_sds: jax.Array,
    regularize: float,
) -> jax.Array:
    H_sym = 0.5 * (H + H.T)
    prior_prec_floor = 1.0 / jnp.maximum(jnp.max(jnp.square(prior_sds)), 1.0)
    eigvals, eigvecs = jnp.linalg.eigh(H_sym)
    scale_floor = jnp.maximum(
        regularize * jnp.maximum(jnp.max(jnp.abs(eigvals)), 1.0),
        1e-8 * prior_prec_floor,
    )
    safe_eigvals = jnp.maximum(eigvals, scale_floor)
    cov = (eigvecs * (1.0 / safe_eigvals)[None, :]) @ eigvecs.T
    return 0.5 * (cov + cov.T)


def _laplace_covariance(
    u_map: jax.Array,
    model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    cfg: LinearMarginalConfig,
    sigma_jitter: ArrayLike | None,
    regularize: float = 1e-6,
    map_space: Literal["unconstrained", "constrained"] = "unconstrained",
    *,
    fast_lk_spec: FastLogLikeSpec | None = None,
) -> jax.Array:
    """Strictly SPD eigenspectrum-regularized Laplace covariance at ``u_map``."""
    d = parameter_space.n_params
    if d == 0:
        return jnp.zeros((0, 0), dtype=jnp.float64)

    if isinstance(model.solver, ModaxSolver):
        # ModaxSolver supports 1st-order custom_vjp only; fall back to diagonal prior cov
        return jnp.diag(jnp.square(parameter_space.prior_sds))

    spec = (
        _make_fast_unconstrained_loglike(
            model, dataset, parameter_space, cfg, sigma_jitter
        )
        if fast_lk_spec is None
        else fast_lk_spec
    )
    reg_flt = float(regularize)
    if not spec.is_static and model.cbm.n_boxes > 8:
        # Central finite-difference of exact 1st-order grad avoids massive
        # 2nd-order Van Loan expm XLA graphs on large multi-box CBMs while
        # achieving O(eps^2) ~ 1e-10 Hessian accuracy in float64.
        eps = 1e-5
        eye = jnp.eye(d, dtype=jnp.float64)
        cols = [
            (
                _neg_log_post_grad_jit(u_map + eps * eye[i], spec, map_space)
                - _neg_log_post_grad_jit(u_map - eps * eye[i], spec, map_space)
            )
            / (2.0 * eps)
            for i in range(d)
        ]
        H = jnp.column_stack(cols)
        return _regularize_hessian_to_cov(
            H, spec.parameter_space.prior_sds, reg_flt
        )
    if d <= 16:
        return _laplace_cov_fwd_jit(u_map, spec, reg_flt, map_space)
    return _laplace_cov_rev_jit(u_map, spec, reg_flt, map_space)


def fit_map(
    model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    cfg: LinearMarginalConfig | None = None,
    *,
    method: Literal["BFGS", "LevenbergMarquardt", "adam"] = "BFGS",
    map_space: Literal["unconstrained", "constrained"] = "unconstrained",
    max_steps: int = 2000,
    tol: float = 1e-6,
    lr: float = 1e-2,
    multi_start: int = 1,
    rng_key: jax.Array | None = None,
    sigma_jitter: ArrayLike | None = None,
    compute_laplace: bool = True,
    compute_linear_posterior: bool = True,
    t_fine: ArrayLike | None = None,
    n_fine_draws: int = 0,
) -> MAPResult:
    """MAP estimation via ``optimistix`` (``BFGS``, ``LevenbergMarquardt``) or ``optax`` (``adam``).

    Parameters
    ----------
    map_space : {"unconstrained", "constrained"}, optional
        Coordinate density maximized by ``fit_map``:

        * ``"unconstrained"`` (default): maximizes ``log p(u | d)`` in unconstrained
          ``u``-space (where ``u ~ N(prior_mean, diag(prior_sd^2))``, matching the
          internal parameterization used by NUTS/SVI/SMC) and maps ``u_map`` through
          ``g(u)`` to obtain ``theta_map``. Note that nonlinear bijectors
          (``"exp"``, ``"softplus"``, ``"sigmoid"``) shift the mode relative to
          constrained ``theta``-space by ``-log |det dg/du|``.
        * ``"constrained"``: maximizes the physical-space posterior density
          ``log p_theta(g(u) | d) = log p(u | d) - log |det dg/du|``.
    """
    if map_space not in ("unconstrained", "constrained"):
        raise ValueError(
            f"map_space must be 'unconstrained' or 'constrained'; got {map_space!r}."
        )
    if cfg is None:
        cfg = LinearMarginalConfig()

    u0 = parameter_space.unconstrain(model)
    if parameter_space.n_params == 0:
        lp0 = _eval_log_posterior(
            u0, model, dataset, parameter_space, cfg, sigma_jitter, map_space
        )
        lk0 = _eval_log_likelihood(model, dataset, cfg, sigma_jitter, parameter_space)
        lp_lin = (
            linear_posterior(
                model,
                dataset,
                _offset_only_cfg(cfg) if _uses_joint_mode(model, parameter_space, cfg) else cfg,
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

    if method not in ("BFGS", "LevenbergMarquardt", "adam"):
        raise ValueError(
            f"Unknown method {method!r}; expected 'BFGS', 'LevenbergMarquardt', or 'adam'."
        )

    fast_lk_spec = _make_fast_unconstrained_loglike(
        model, dataset, parameter_space, cfg, sigma_jitter
    )
    best_u: jax.Array = u0
    best_lp = _eval_fast_log_post_jit(fast_lk_spec, u0, map_space)
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
                map_space=map_space,
                fast_lk_spec=fast_lk_spec,
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
                map_space=map_space,
                fast_lk_spec=fast_lk_spec,
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
                lr=lr,
                map_space=map_space,
                fast_lk_spec=fast_lk_spec,
            )

        lp = _eval_fast_log_post_jit(fast_lk_spec, u_opt, map_space)
        if bool(jnp.isfinite(lp)) and float(lp) >= float(best_lp):
            best_u = u_opt
            best_lp = lp
            best_converged = conv
            best_n_steps = n_steps

    u_map = best_u
    model_map = parameter_space.constrain(u_map, model)
    log_lk = _eval_fast_log_lk_jit(fast_lk_spec, u_map)
    theta_map = {spec.path: model_map.get(spec.path) for spec in parameter_space.specs}

    if compute_laplace:
        laplace_cov = _laplace_covariance(
            u_map,
            model,
            dataset,
            parameter_space,
            cfg,
            sigma_jitter,
            map_space=map_space,
            fast_lk_spec=fast_lk_spec,
        )
    else:
        d = parameter_space.n_params
        laplace_cov = jnp.eye(d, dtype=jnp.float64)

    lin_post = None
    if compute_linear_posterior:
        is_lgf_marg = (
            not _uses_joint_mode(model_map, parameter_space, cfg)
            and _extract_single_lgf(model_map.production)[0] is not None
        )
        lin_cfg = (
            _offset_only_cfg(cfg)
            if _uses_joint_mode(model_map, parameter_space, cfg) or is_lgf_marg
            else cfg
        )
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
