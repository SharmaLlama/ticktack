"""``NumPyro`` NUTS/SVI and ``BlackJAX`` tempered SMC samplers for ``ticktack 2.0``.

Implements three complementary posterior samplers:

1. :func:`sample_nuts`: ``NumPyro`` ``NUTS`` + ``MCMC``, warm-started at the MAP
   estimate in non-centred prior coordinates, returning ``arviz.InferenceData``
   with constrained physical parameters and optional exact conditional Gaussian
   linear posterior draws per sample.
2. :func:`fit_svi`: ``NumPyro`` ``SVI`` with ``AutoMultivariateNormal`` /
   ``AutoNormal`` variational guide via JIT-compiled ``svi.run``, ideal for fast
   ``LogGPProduction`` inference.
3. :func:`sample_smc`: ``BlackJAX`` adaptive tempered SMC with **multinomial**
   resampling (essential for unbiased Lee & Whiteley 2018 ``ln Z`` variance
   estimation), trace-regularized covariance mutation kernel, and calibrated
   ``(log_evidence, log_evidence_err)`` for ``EffectiveCBM(K)`` model selection.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Literal, NamedTuple

import arviz as az
import blackjax.smc.resampling as bj_resample
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
from jax.typing import ArrayLike
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, autoguide

from .data import Dataset
from .fitting import _laplace_covariance, fit_map
from .likelihood import (
    FastLogLikeSpec,
    LinearMarginalConfig,
    ParameterSpace,
    _build_marginal_design,
    _has_loggp,
    _make_fast_unconstrained_loglike,
    _marginal_design_is_static,
    _offset_column_count,
    _offset_only_cfg,
    joint_log_likelihood,
    linear_posterior,
    marginal_log_likelihood,
)
from .linear import LinearMarginal
from .model import RadiocarbonModel
from .solvers import ModaxSolver

__all__ = [
    "SVIResult",
    "SMCResult",
    "sample_nuts",
    "fit_svi",
    "sample_smc",
]


def _use_joint_mode(model: RadiocarbonModel) -> bool:
    return _has_loggp(model.production) or isinstance(model.solver, ModaxSolver)


def _eval_single_loglike(
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


def _unpack_constrained_samples(
    parameter_space: ParameterSpace,
    u_samples: jax.Array,
) -> dict[str, np.ndarray]:
    """Convert ``(..., d)`` unconstrained draws into ``{name: np.ndarray}`` constrained draws."""
    batch_shape = u_samples.shape[:-1]
    out: dict[str, np.ndarray] = {}
    for spec, shape, (start, stop) in zip(
        parameter_space.specs,
        parameter_space.leaf_shapes,
        parameter_space.spec_slices,
    ):
        if stop == start:
            continue
        param_name = spec.path.replace(".", "__")
        u_leaf = u_samples[..., start:stop].reshape(*batch_shape, *shape)
        theta_leaf = spec.constrain(u_leaf)
        out[param_name] = np.asarray(theta_leaf)
    return out


@eqx.filter_jit
def _vmap_fast_lk_jit(
    fast_lk_spec: FastLogLikeSpec,
    u_batch: jax.Array,
) -> jax.Array:
    return jax.vmap(fast_lk_spec)(u_batch)


@eqx.filter_jit
def _vmap_fast_resid_jit(
    fast_lk_spec: FastLogLikeSpec,
    u_batch: jax.Array,
) -> jax.Array:
    return jax.vmap(fast_lk_spec.nonlinear_residual_from_u)(u_batch)


@eqx.filter_jit
def _jit_static_linear_draw_factors(
    model: RadiocarbonModel,
    dataset: Dataset,
    lin_cfg: LinearMarginalConfig,
    sigma: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    X, prior_mean, prior_sd = _build_marginal_design(model, dataset, lin_cfg)
    k_tot = X.shape[1]
    if k_tot == 0:
        return (
            jnp.zeros((0,), dtype=jnp.float64),
            jnp.zeros((dataset.n_obs, 0), dtype=jnp.float64),
            jnp.zeros((0, 0), dtype=jnp.float64),
        )
    lm = LinearMarginal(
        X,
        prior_mean=prior_mean,
        prior_sd=prior_sd,
        method=lin_cfg.whiten_method,
    )
    mu_0, coeff_cov = lm.posterior(jnp.zeros_like(dataset.d14c_flat), sigma)
    coeff_cov = 0.5 * (coeff_cov + coeff_cov.T)
    L_cov = jnp.linalg.cholesky(
        coeff_cov + 1e-12 * jnp.eye(k_tot, dtype=jnp.float64)
    )
    gain_T = (X / jnp.square(sigma)[:, None]) @ coeff_cov
    return mu_0, gain_T, L_cov


@eqx.filter_jit
def _vmap_log_prior_jit(
    parameter_space: ParameterSpace,
    u_batch: jax.Array,
) -> jax.Array:
    return jax.vmap(parameter_space.log_prior_unconstrained)(u_batch)


@eqx.filter_jit
def _nuts_cond_draw_batch_jit(
    u_flat: jax.Array,
    draw_keys: jax.Array,
    model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    lin_cfg: LinearMarginalConfig,
    t_fine_arr: jax.Array | None,
    sigma_jitter_arr: jax.Array | None,
) -> tuple[jax.Array, jax.Array, jax.Array | None, jax.Array | None]:
    n_series = dataset.n_series
    n_off = _offset_column_count(lin_cfg, n_series)

    def _cond_draw_one(
        u_one: jax.Array, key_one: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array | None, jax.Array | None]:
        m_one = parameter_space.constrain(u_one, model)
        lp = linear_posterior(
            m_one,
            dataset,
            lin_cfg,
            rng_key=key_one,
            n_draws=1,
            t_fine=t_fine_arr,
            sigma_jitter=sigma_jitter_arr,
        )
        k_tot = lp.coeff_mean.shape[0]
        if lp.coeff_draws is not None and k_tot > 0:
            w_draw = lp.coeff_draws[0]
        else:
            w_draw = lp.coeff_mean

        if n_off > 0 and k_tot >= n_off:
            if lin_cfg.offset_mode == "hierarchical":
                g_draw = w_draw[0]
                l_draw = (
                    w_draw[1 : 1 + n_series]
                    if n_off == 1 + n_series
                    else jnp.zeros((n_series,), dtype=jnp.float64)
                )
            elif lin_cfg.offset_mode == "shared":
                g_draw = w_draw[0]
                l_draw = jnp.zeros((n_series,), dtype=jnp.float64)
            else:
                l_draw = w_draw[:n_series]
                g_draw = jnp.mean(l_draw)
        else:
            g_draw = jnp.asarray(0.0, dtype=jnp.float64)
            l_draw = jnp.zeros((n_series,), dtype=jnp.float64)

        qf_d = lp.q_fine_draws[0] if lp.q_fine_draws is not None else None
        df_d = lp.d14c_fine_draws[0] if lp.d14c_fine_draws is not None else None
        return g_draw, l_draw, qf_d, df_d

    return jax.lax.map(
        lambda pair: _cond_draw_one(pair[0], pair[1]),
        (u_flat, draw_keys),
        batch_size=16,
    )


# ---------------------------------------------------------------------------
# 1. NumPyro NUTS sampler
# ---------------------------------------------------------------------------


def _make_numpyro_model(
    base_model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    cfg: LinearMarginalConfig,
    sigma_jitter: ArrayLike | None,
) -> Any:
    """Return a non-centred NumPyro model function over ``z ~ N(0, I_d)``."""
    d = parameter_space.n_params
    prior_means = parameter_space.prior_means
    prior_sds = parameter_space.prior_sds
    fast_lk_fn = _make_fast_unconstrained_loglike(
        base_model, dataset, parameter_space, cfg, sigma_jitter
    )

    def numpyro_model() -> None:
        z = numpyro.sample(
            "z",
            dist.Normal(
                jnp.zeros((d,), dtype=jnp.float64),
                jnp.ones((d,), dtype=jnp.float64),
            ),
        )
        u_vec = prior_means + prior_sds * z
        log_lk = fast_lk_fn(u_vec)
        numpyro.factor("log_likelihood", log_lk)

    return numpyro_model


def sample_nuts(
    model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    cfg: LinearMarginalConfig | None = None,
    *,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 2,
    target_accept_prob: float = 0.85,
    max_tree_depth: int = 10,
    init_map: bool = True,
    dense_mass: bool = True,
    reconstruct_linear: bool = True,
    t_fine: ArrayLike | None = None,
    rng_key: jax.Array | None = None,
    sigma_jitter: ArrayLike | None = None,
    progress_bar: bool = False,
) -> az.InferenceData:
    """``NumPyro`` NUTS + MCMC sampler warm-started near the MAP estimate."""
    if cfg is None:
        cfg = LinearMarginalConfig()
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    d = parameter_space.n_params
    prior_means = parameter_space.prior_means
    prior_sds = parameter_space.prior_sds

    # --- Warm-start in non-centred z-space + Laplace mass-matrix preconditioning ---
    init_params = None
    inv_mass_init = None
    if d > 0:
        z_center: jax.Array | None = None
        cov_u: jax.Array | None = None
        if init_map:
            rng_key, map_key = jax.random.split(rng_key)
            try:
                map_result = fit_map(
                    model,
                    dataset,
                    parameter_space,
                    cfg,
                    method="BFGS",
                    max_steps=500,
                    tol=1e-5,
                    rng_key=map_key,
                    sigma_jitter=sigma_jitter,
                    compute_laplace=(d <= 16),
                    compute_linear_posterior=False,
                )
                z_center = (map_result.u_map - prior_means) / prior_sds
                if d <= 16:
                    cov_u = map_result.laplace_cov
            except Exception as exc:
                warnings.warn(
                    f"sample_nuts MAP warm-start fallback to prior initialization: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                z_center = None
        else:
            try:
                u_curr = parameter_space.unconstrain(model)
                z_cand = (u_curr - prior_means) / prior_sds
                if bool(jnp.all(jnp.isfinite(z_cand))):
                    z_center = z_cand
                    if d <= 16 and not isinstance(model.solver, ModaxSolver):
                        cov_u = _laplace_covariance(
                            u_curr, model, dataset, parameter_space, cfg, sigma_jitter
                        )
            except Exception as exc:
                warnings.warn(
                    f"sample_nuts initial-point warm-start fallback: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                z_center = None

        L_z: jax.Array | None = None
        if cov_u is not None:
            try:
                inv_sds = 1.0 / prior_sds
                cov_z = 0.5 * (cov_u + cov_u.T) * (inv_sds[:, None] * inv_sds[None, :])
                cov_z = jnp.clip(cov_z, -1e4, 1e4)
                eigvals, eigvecs = jnp.linalg.eigh(cov_z)
                safe_ev = jnp.clip(eigvals, 1e-6, 10.0)
                cov_z_reg = (eigvecs * safe_ev[None, :]) @ eigvecs.T
                cov_z_reg = 0.5 * (cov_z_reg + cov_z_reg.T)
                if bool(jnp.all(jnp.isfinite(cov_z_reg))):
                    L_z = jnp.linalg.cholesky(cov_z_reg)
                    inv_mass_init = {
                        ("z",): cov_z_reg if dense_mass else jnp.diag(cov_z_reg)
                    }
            except Exception as exc:
                warnings.warn(
                    f"sample_nuts Laplace mass-matrix fallback to identity: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                inv_mass_init = None
                L_z = None

        if z_center is not None:
            rng_key, perturb_key = jax.random.split(rng_key)
            raw_eps = jax.random.normal(
                perturb_key,
                shape=(num_chains, d),
                dtype=jnp.float64,
            )
            perturb = (
                0.15 * (raw_eps @ L_z.T) if L_z is not None else 0.05 * raw_eps
            )
            z_starts = z_center[None, :] + perturb
            init_params = {"z": z_starts[0] if num_chains == 1 else z_starts}

    numpyro_model_fn = _make_numpyro_model(
        model, dataset, parameter_space, cfg, sigma_jitter
    )
    nuts_kernel = NUTS(
        numpyro_model_fn,
        target_accept_prob=target_accept_prob,
        dense_mass=dense_mass,
        max_tree_depth=max_tree_depth,
        inverse_mass_matrix=inv_mass_init,
    )
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="vectorized" if num_chains > 1 else "sequential",
        progress_bar=progress_bar,
    )

    extra_fields = (
        "potential_energy",
        "energy",
        "diverging",
        "num_steps",
        "accept_prob",
        "mean_accept_prob",
        "adapt_state.step_size",
    )
    rng_key, run_key = jax.random.split(rng_key)
    mcmc.run(run_key, init_params=init_params, extra_fields=extra_fields)

    extra = mcmc.get_extra_fields(group_by_chain=True)
    pe_np = np.asarray(extra["potential_energy"])
    en_np = np.asarray(extra["energy"])
    div_np = np.asarray(extra["diverging"], dtype=bool)
    n_steps_np = np.asarray(extra["num_steps"], dtype=np.int64)
    acc_np = np.asarray(extra["accept_prob"])
    step_sz_np = np.asarray(extra["adapt_state.step_size"])
    tree_depth_np = (
        np.floor(np.log2(np.maximum(n_steps_np, 1))).astype(np.int64) + 1
    )
    max_steps_cap = max((1 << int(max_tree_depth)) - 1, 1)
    max_td_np = (n_steps_np >= max_steps_cap) & (~div_np)

    n_div = int(np.sum(div_np))
    if n_div > 0:
        warnings.warn(
            f"sample_nuts detected {n_div} divergent transition(s) out of "
            f"{div_np.size} post-warmup draws; consider increasing target_accept_prob "
            "or reparameterizing.",
            RuntimeWarning,
            stacklevel=2,
        )
    n_max_td = int(np.sum(max_td_np))
    if n_max_td > 0:
        warnings.warn(
            f"sample_nuts saturated max_tree_depth={max_tree_depth} on {n_max_td} "
            f"out of {max_td_np.size} post-warmup draws; consider increasing max_tree_depth.",
            RuntimeWarning,
            stacklevel=2,
        )

    z_samples = mcmc.get_samples(group_by_chain=True)["z"]  # (C, S, d)
    u_samples = prior_means[None, None, :] + prior_sds[None, None, :] * z_samples

    posterior_dict = _unpack_constrained_samples(parameter_space, u_samples)

    fast_lk_batch_fn = _make_fast_unconstrained_loglike(
        model, dataset, parameter_space, cfg, sigma_jitter
    )
    u_flat = u_samples.reshape(-1, d)
    log_lk_arr = _vmap_fast_lk_jit(fast_lk_batch_fn, u_flat).reshape(
        num_chains, num_samples
    )

    idata_groups: dict[str, Any] = {
        "posterior": posterior_dict,
        "sample_stats": {
            "lp": -pe_np,
            "energy": en_np,
            "diverging": div_np,
            "n_steps": n_steps_np,
            "tree_depth": tree_depth_np,
            "reached_max_treedepth": max_td_np,
            "acceptance_rate": acc_np,
            "step_size": step_sz_np,
        },
        "log_likelihood": {"obs": np.asarray(log_lk_arr[:, :, None])},
    }

    # --- Conditional Gaussian reconstruction of marginalized linear parameters ---
    if reconstruct_linear:
        joint_mode = _use_joint_mode(model)
        lin_cfg = _offset_only_cfg(cfg) if joint_mode else cfg
        n_off = _offset_column_count(lin_cfg, dataset.n_series)
        n_series = dataset.n_series

        if n_off > 0 or t_fine is not None:
            rng_key, lin_key = jax.random.split(rng_key)
            draw_keys = jax.random.split(lin_key, u_flat.shape[0])
            t_fine_arr = (
                jnp.asarray(t_fine, dtype=jnp.float64).reshape(-1)
                if t_fine is not None
                else None
            )
            jit_arr = (
                jnp.asarray(sigma_jitter, dtype=jnp.float64)
                if sigma_jitter is not None
                else None
            )
            if isinstance(model.solver, ModaxSolver):
                def _cond_draw_mdx(u_one: jax.Array, key_one: jax.Array) -> Any:
                    m_one = parameter_space.constrain(u_one, model)
                    lp = linear_posterior(
                        m_one,
                        dataset,
                        lin_cfg,
                        rng_key=key_one,
                        n_draws=1,
                        t_fine=t_fine_arr,
                        sigma_jitter=jit_arr,
                    )
                    w_draw = (
                        lp.coeff_draws[0]
                        if (lp.coeff_draws is not None and lp.coeff_mean.shape[0] > 0)
                        else lp.coeff_mean
                    )
                    g_d = w_draw[0] if n_off > 0 and w_draw.shape[0] >= 1 else jnp.asarray(0.0, dtype=jnp.float64)
                    l_d = (
                        w_draw[1 : 1 + n_series]
                        if (lin_cfg.offset_mode == "hierarchical" and n_off == 1 + n_series)
                        else (w_draw[:n_series] if lin_cfg.offset_mode == "per_series" and n_off == n_series else jnp.zeros((n_series,), dtype=jnp.float64))
                    )
                    if lin_cfg.offset_mode == "per_series" and n_off > 0:
                        g_d = jnp.mean(l_d)
                    return (
                        g_d,
                        l_d,
                        lp.q_fine_draws[0] if lp.q_fine_draws is not None else None,
                        lp.d14c_fine_draws[0] if lp.d14c_fine_draws is not None else None,
                    )
                g_all, l_all, qf_all, df_all = jax.vmap(_cond_draw_mdx)(u_flat, draw_keys)
            elif (
                t_fine_arr is None
                and dataset.n_obs > 0
                and _marginal_design_is_static(model, parameter_space, lin_cfg)
            ):
                sigma_eff = jnp.asarray(dataset.sigma_flat, dtype=jnp.float64)
                if jit_arr is not None:
                    sigma_eff = jnp.sqrt(
                        jnp.square(sigma_eff) + jnp.square(jit_arr)
                    )
                mu_0, gain_T, L_cov = _jit_static_linear_draw_factors(
                    model, dataset, lin_cfg, sigma_eff
                )
                resids_all = _vmap_fast_resid_jit(fast_lk_batch_fn, u_flat)
                n_draws_tot = u_flat.shape[0]
                k_tot = mu_0.shape[0]
                if k_tot > 0:
                    coeff_means = mu_0[None, :] + resids_all @ gain_T
                    z_lin = jax.random.normal(
                        lin_key, (n_draws_tot, k_tot), dtype=jnp.float64
                    )
                    w_draws = coeff_means + z_lin @ L_cov.T
                else:
                    w_draws = jnp.zeros((n_draws_tot, 0), dtype=jnp.float64)

                if n_off > 0 and k_tot >= n_off:
                    if lin_cfg.offset_mode == "hierarchical":
                        g_all = w_draws[:, 0]
                        l_all = (
                            w_draws[:, 1 : 1 + n_series]
                            if n_off == 1 + n_series
                            else jnp.zeros((n_draws_tot, n_series), dtype=jnp.float64)
                        )
                    elif lin_cfg.offset_mode == "shared":
                        g_all = w_draws[:, 0]
                        l_all = jnp.zeros((n_draws_tot, n_series), dtype=jnp.float64)
                    else:
                        l_all = w_draws[:, :n_series]
                        g_all = jnp.mean(l_all, axis=1)
                else:
                    g_all = jnp.zeros((n_draws_tot,), dtype=jnp.float64)
                    l_all = jnp.zeros((n_draws_tot, n_series), dtype=jnp.float64)
                qf_all, df_all = None, None
            else:
                g_all, l_all, qf_all, df_all = _nuts_cond_draw_batch_jit(
                    u_flat,
                    draw_keys,
                    model,
                    dataset,
                    parameter_space,
                    lin_cfg,
                    t_fine_arr,
                    jit_arr,
                )
            if n_off > 0:
                posterior_dict["offset_global"] = np.asarray(
                    g_all.reshape(num_chains, num_samples)
                )
                posterior_dict["offset_lab"] = np.asarray(
                    l_all.reshape(num_chains, num_samples, n_series)
                )
            if t_fine is not None and qf_all is not None and df_all is not None:
                idata_groups["posterior_predictive"] = {
                    "q_fine": np.asarray(
                        qf_all.reshape(num_chains, num_samples, -1)
                    ),
                    "d14c_fine": np.asarray(
                        df_all.reshape(num_chains, num_samples, -1)
                    ),
                }

    return az.from_dict(idata_groups)


# ---------------------------------------------------------------------------
# 2. NumPyro SVI
# ---------------------------------------------------------------------------


class SVIResult(NamedTuple):
    """Results from :func:`fit_svi`."""

    model: RadiocarbonModel
    """Posterior-mean constrained :class:`~ticktack.model.RadiocarbonModel`."""

    guide: Any
    """Trained NumPyro variational guide."""

    losses: jax.Array
    """ELBO loss history, shape ``(num_steps,)``."""

    idata: az.InferenceData
    """``arviz.InferenceData`` with constrained posterior samples from the guide."""

    parameter_space: ParameterSpace
    """Parameter space used for the variational fit."""

    params: dict[str, Any] = {}
    """Trained variational guide parameter dictionary."""


def fit_svi(
    model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    cfg: LinearMarginalConfig | None = None,
    *,
    guide: Literal["mvn", "diag"] = "mvn",
    num_steps: int = 2000,
    lr: float = 1e-2,
    num_samples: int = 1000,
    rng_key: jax.Array | None = None,
    sigma_jitter: ArrayLike | None = None,
    progress_bar: bool = False,
) -> SVIResult:
    """``NumPyro`` SVI variational inference via JIT-compiled ``svi.run``."""
    if cfg is None:
        cfg = LinearMarginalConfig()
    if rng_key is None:
        rng_key = jax.random.PRNGKey(1)

    numpyro_model_fn = _make_numpyro_model(
        model, dataset, parameter_space, cfg, sigma_jitter
    )
    if guide == "mvn":
        guide_fn = autoguide.AutoMultivariateNormal(numpyro_model_fn)
    elif guide == "diag":
        guide_fn = autoguide.AutoNormal(numpyro_model_fn)
    else:
        raise ValueError(f"guide must be 'mvn' or 'diag'; got {guide!r}.")

    optimizer = numpyro.optim.optax_to_numpyro(optax.adam(lr))
    svi = SVI(numpyro_model_fn, guide_fn, optimizer, loss=Trace_ELBO())

    rng_key, run_key = jax.random.split(rng_key)
    svi_result = svi.run(
        run_key,
        num_steps,
        stable_update=True,
        progress_bar=progress_bar,
    )
    losses_arr = jnp.asarray(svi_result.losses, dtype=jnp.float64)

    rng_key, sample_key = jax.random.split(rng_key)
    guide_samples = guide_fn.sample_posterior(
        sample_key, svi_result.params, sample_shape=(num_samples,)
    )

    d = parameter_space.n_params
    prior_means = parameter_space.prior_means
    prior_sds = parameter_space.prior_sds
    z_samples = guide_samples.get(
        "z", jnp.zeros((num_samples, d), dtype=jnp.float64)
    )
    u_samples = prior_means[None, :] + prior_sds[None, :] * z_samples

    posterior_dict = _unpack_constrained_samples(
        parameter_space, u_samples[None, :, :]
    )
    idata = az.from_dict({"posterior": posterior_dict})

    u_mean = jnp.mean(u_samples, axis=0) if d > 0 else jnp.zeros((0,), dtype=jnp.float64)
    fitted_model = parameter_space.constrain(u_mean, model)

    return SVIResult(
        model=fitted_model,
        guide=guide_fn,
        losses=losses_arr,
        idata=idata,
        parameter_space=parameter_space,
        params=dict(svi_result.params),
    )


# ---------------------------------------------------------------------------
# 3. BlackJAX adaptive tempered SMC with multinomial resampling
# ---------------------------------------------------------------------------


class SMCResult(NamedTuple):
    """Results from :func:`sample_smc`."""

    particles: jax.Array
    """Final unconstrained particle positions, shape ``(num_particles, d)``."""

    weights: jax.Array
    """Final normalized particle weights, shape ``(num_particles,)``."""

    idata: az.InferenceData
    """``arviz.InferenceData`` with constrained posterior samples."""

    log_evidence: jax.Array
    """Marginal log-evidence estimate ``ln Z_hat`` (scalar)."""

    log_evidence_err: jax.Array
    """Calibrated standard error ``sigma_{ln Z}`` via Lee & Whiteley (2018)."""

    tempering_betas: jax.Array
    """Adaptive tempering schedule ``beta_0 = 0 < ... < beta_T = 1``."""

    ess_history: jax.Array
    """Effective sample size at each tempering bridge."""

    root_ess: jax.Array = jnp.asarray(0.0, dtype=jnp.float64)
    """Terminal root-ancestor effective sample size ``(sum_r B_r^2)^{-1}``."""

    mean_accept_prob: jax.Array = jnp.asarray(0.0, dtype=jnp.float64)
    """Mean Metropolis-Hastings acceptance rate across intermediate SMC bridges."""


def _trace_regularized_covariance(
    particles: jax.Array,
    weights: jax.Array,
    eps: float = 1e-2,
) -> jax.Array:
    """Trace-regularized weighted covariance ``(1 - eps) * cov + eps * (Tr(cov)/d) * I_d``."""
    d = particles.shape[1]
    if d == 0:
        return jnp.zeros((0, 0), dtype=jnp.float64)
    w = jnp.asarray(weights, dtype=jnp.float64)
    w = w / jnp.sum(w)
    mu = jnp.sum(w[:, None] * particles, axis=0)
    diffs = particles - mu[None, :]
    cov = jnp.einsum("ni,nj,n->ij", diffs, diffs, w)
    tr = jnp.maximum(jnp.trace(cov) / max(d, 1), 1e-12)
    return (1.0 - eps) * cov + eps * tr * jnp.eye(d, dtype=jnp.float64)


@eqx.filter_jit
def _smc_mutate_particles_jit(
    key_in: jax.Array,
    parts_in: jax.Array,
    lp_in: jax.Array,
    lk_in: jax.Array,
    chol_cov: jax.Array,
    beta_val: jax.Array,
    fast_lk_spec: FastLogLikeSpec,
    parameter_space: ParameterSpace,
    num_mcmc_steps_eff: int,
    rwm_scale: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    n_part = parts_in.shape[0]

    def _one_mh_step(carry: Any, step_key: jax.Array) -> tuple[Any, jax.Array]:
        p_curr, lp_c, lk_c = carry
        k_prop, k_acc = jax.random.split(step_key)
        noise = jax.random.normal(k_prop, shape=p_curr.shape, dtype=jnp.float64)
        props = p_curr + rwm_scale * (noise @ chol_cov.T)
        lp_prop = jax.vmap(parameter_space.log_prior_unconstrained)(props)
        lk_prop = jax.vmap(fast_lk_spec)(props)
        log_alpha = (lp_prop + beta_val * lk_prop) - (lp_c + beta_val * lk_c)
        u_acc = jax.random.uniform(k_acc, shape=(n_part,), dtype=jnp.float64)
        acc = jnp.log(u_acc) < log_alpha
        p_next = jnp.where(acc[:, None], props, p_curr)
        lp_next = jnp.where(acc, lp_prop, lp_c)
        lk_next = jnp.where(acc, lk_prop, lk_c)
        return (p_next, lp_next, lk_next), jnp.mean(acc.astype(jnp.float64))

    step_keys = jax.random.split(key_in, num_mcmc_steps_eff)
    (p_out, lp_out, lk_out), step_accs = jax.lax.scan(
        _one_mh_step, (parts_in, lp_in, lk_in), step_keys
    )
    return p_out, lp_out, lk_out, jnp.mean(step_accs)


def sample_smc(
    model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    cfg: LinearMarginalConfig | None = None,
    *,
    num_particles: int = 500,
    target_ess: float = 0.75,
    num_mcmc_steps: int = 5,
    proposal_mode: Literal["laplace_tempered", "prior", "ensemble"] = "laplace_tempered",
    rng_key: jax.Array | None = None,
    sigma_jitter: ArrayLike | None = None,
    progress_bar: bool = False,
) -> SMCResult:
    """``BlackJAX`` adaptive tempered SMC with multinomial resampling and Lee-Whiteley ``sigma_{ln Z}``.

    By default (``proposal_mode="laplace_tempered"``), the RWM mutation covariance at
    inverse temperature ``beta`` is ``(Lambda_0 + beta * H_lik)^{-1}`` where ``H_lik``
    is pre-estimated once before particle initialization; this keeps the Markov
    rejuvenation kernel strictly non-interacting across particle lineages as required
    by Lee & Whiteley (2018) while tracking posterior contraction across ``beta``.
    """
    if proposal_mode not in ("laplace_tempered", "prior", "ensemble"):
        raise ValueError(
            "proposal_mode must be 'laplace_tempered', 'prior', or 'ensemble'; "
            f"got {proposal_mode!r}."
        )
    if cfg is None:
        cfg = LinearMarginalConfig()
    if rng_key is None:
        rng_key = jax.random.PRNGKey(2)

    d = parameter_space.n_params
    n_part = int(num_particles)
    if n_part < 2:
        raise ValueError(f"num_particles must be >= 2; got {n_part}.")

    if d == 0:
        lk0 = _eval_single_loglike(model, dataset, cfg, sigma_jitter)
        empty_p = jnp.zeros((n_part, 0), dtype=jnp.float64)
        unif_w = jnp.ones((n_part,), dtype=jnp.float64) / n_part
        return SMCResult(
            particles=empty_p,
            weights=unif_w,
            idata=az.from_dict({"posterior": {"_dummy": np.zeros((1, n_part))}}),
            log_evidence=lk0,
            log_evidence_err=jnp.asarray(0.0, dtype=jnp.float64),
            tempering_betas=jnp.array([0.0, 1.0], dtype=jnp.float64),
            ess_history=jnp.array([float(n_part)], dtype=jnp.float64),
            root_ess=jnp.asarray(float(n_part), dtype=jnp.float64),
            mean_accept_prob=jnp.asarray(1.0, dtype=jnp.float64),
        )

    prior_means = parameter_space.prior_means
    prior_sds = parameter_space.prior_sds
    prec_prior_diag = 1.0 / jnp.square(prior_sds)

    # Pre-estimate likelihood curvature independently of the SMC particle ensemble
    # so the mutation kernel satisfies Lee & Whiteley's non-interacting Markov assumption.
    H_lik_fixed: jax.Array | None = None
    if (
        proposal_mode == "laplace_tempered"
        and int(num_mcmc_steps) > 0
        and d <= 16
        and not isinstance(model.solver, ModaxSolver)
    ):
        rng_key, pre_key = jax.random.split(rng_key)
        try:
            pre_map = fit_map(
                model,
                dataset,
                parameter_space,
                cfg,
                method="BFGS",
                max_steps=300,
                tol=1e-5,
                rng_key=pre_key,
                sigma_jitter=sigma_jitter,
                compute_laplace=True,
                compute_linear_posterior=False,
            )
            cov_map = 0.5 * (pre_map.laplace_cov + pre_map.laplace_cov.T)
            H_post = jnp.linalg.inv(cov_map)
            H_diff = H_post - jnp.diag(prec_prior_diag)
            H_diff = 0.5 * (H_diff + H_diff.T)
            ev_d, evec_d = jnp.linalg.eigh(H_diff)
            cand_h = (evec_d * jnp.maximum(ev_d, 0.0)[None, :]) @ evec_d.T
            if bool(jnp.all(jnp.isfinite(cand_h))):
                H_lik_fixed = 0.5 * (cand_h + cand_h.T)
        except Exception as exc:
            warnings.warn(
                f"sample_smc Laplace pre-calibration fallback to prior proposal scale: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    fast_lk_smc_fn = _make_fast_unconstrained_loglike(
        model, dataset, parameter_space, cfg, sigma_jitter
    )
    rwm_scale = float(2.38 / math.sqrt(max(d, 1)))
    num_mcmc_steps_eff = max(int(num_mcmc_steps), 1)

    # --- Initialize particles directly from the exact prior ---
    rng_key, init_key, smc_key = jax.random.split(rng_key, 3)
    z_init = jax.random.normal(init_key, shape=(n_part, d), dtype=jnp.float64)
    particles = prior_means[None, :] + prior_sds[None, :] * z_init
    curr_lp = _vmap_log_prior_jit(parameter_space, particles)
    current_log_lks = _vmap_fast_lk_jit(fast_lk_smc_fn, particles)

    beta = 0.0
    log_z = jnp.asarray(0.0, dtype=jnp.float64)
    betas: list[float] = [0.0]
    ess_list: list[float] = []
    step_rel_vars: list[float] = []
    acc_rates: list[float] = []
    eve = jnp.arange(n_part, dtype=jnp.int32)
    lw_var_final: float | None = None
    root_ess_val: float = float(n_part)
    n_steps_total = 0
    unif_weights = jnp.ones((n_part,), dtype=jnp.float64) / n_part
    final_weights = unif_weights

    try:
        from tqdm.auto import tqdm

        pbar = tqdm(desc="SMC tempering", disable=not progress_bar)
    except ImportError:
        pbar = None

    target_ess_count = float(target_ess) * n_part

    while beta < 1.0:
        n_steps_total += 1
        lk_np = np.asarray(current_log_lks, dtype=np.float64)

        def _ess_at_beta(b_cand: float) -> float:
            lw = (b_cand - beta) * lk_np
            lw = lw - float(np.max(lw))
            w = np.exp(lw)
            w = w / np.sum(w)
            return float(1.0 / np.sum(w * w))

        if _ess_at_beta(1.0) >= target_ess_count:
            beta_new = 1.0
        else:
            lo, hi = beta, 1.0
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if _ess_at_beta(mid) >= target_ess_count:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < 1e-7:
                    break
            beta_new = min(max(lo, beta + 1e-6), 1.0)

        log_inc = (beta_new - beta) * current_log_lks
        log_w_lse = jax.scipy.special.logsumexp(log_inc)
        w_norm = jnp.exp(log_inc - log_w_lse)
        ess_now = float(1.0 / jnp.sum(jnp.square(w_norm)))
        ess_list.append(ess_now)

        # Incremental evidence update: (1/N) sum_i exp((beta_new - beta) * log_lk_i)
        log_z = log_z + (log_w_lse - math.log(n_part))

        # Per-bridge relative variance increment + Lee & Whiteley (2018) root-lineage estimator
        sum_w2 = float(jnp.sum(jnp.square(w_norm)))
        step_rel_vars.append(max(n_part * sum_w2 - 1.0, 0.0) / max(n_part - 1, 1))

        if beta_new >= 1.0:
            lineage_masses = (
                jnp.zeros((n_part,), dtype=jnp.float64).at[eve].add(w_norm)
            )
            sum_lin2 = float(jnp.sum(jnp.square(lineage_masses)))
            root_ess_val = 1.0 / max(sum_lin2, 1e-30)
            lw_ratio = ((n_part / max(n_part - 1, 1)) ** n_steps_total) * (
                1.0 - sum_lin2
            )
            indep_var = float(sum(step_rel_vars))
            if lw_ratio >= 1.0:
                # Upward finite-sample fluctuation when lineages remain fully diverse
                lw_var_final = max(indep_var, 1e-12)
            else:
                lw_var_final = max(
                    -math.log(max(lw_ratio, 1.0 / n_part)),
                    0.5 * indep_var,
                )
                if lw_ratio <= 0.05:
                    warnings.warn(
                        f"sample_smc detected severe root-ancestor depletion "
                        f"(lw_ratio={lw_ratio:.4g}, root_ess={root_ess_val:.2f}/{n_part}); "
                        "increase num_particles or num_mcmc_steps.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            final_weights = w_norm
            beta = 1.0
            betas.append(1.0)
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(beta=f"{beta:.3f}", ESS=f"{ess_now:.0f}")
            break

        # Multinomial resampling on intermediate bridges (mandatory for unbiased Lee-Whiteley estimator)
        smc_key, resample_key, mut_key = jax.random.split(smc_key, 3)
        indices = bj_resample.multinomial(resample_key, w_norm, n_part)
        particles = particles[indices]
        curr_lp = curr_lp[indices]
        current_log_lks = current_log_lks[indices]
        eve = eve[indices]

        if int(num_mcmc_steps) > 0:
            if proposal_mode == "ensemble":
                cov = _trace_regularized_covariance(particles, unif_weights)
            elif H_lik_fixed is not None:
                prec_b = jnp.diag(prec_prior_diag) + float(beta_new) * H_lik_fixed
                ev_b, evec_b = jnp.linalg.eigh(0.5 * (prec_b + prec_b.T))
                cov = (evec_b * (1.0 / jnp.maximum(ev_b, 1e-12))[None, :]) @ evec_b.T
                cov = 0.5 * (cov + cov.T)
            else:
                cov = jnp.diag(jnp.square(prior_sds))
            L_cov = jnp.linalg.cholesky(cov)
            particles, curr_lp, current_log_lks, stage_acc = _smc_mutate_particles_jit(
                mut_key,
                particles,
                curr_lp,
                curr_lp * 0.0 + current_log_lks,
                L_cov,
                jnp.asarray(beta_new, dtype=jnp.float64),
                fast_lk_smc_fn,
                parameter_space,
                num_mcmc_steps_eff,
                rwm_scale,
            )
            acc_rates.append(float(stage_acc))

        beta = beta_new
        betas.append(float(beta))
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(beta=f"{beta:.3f}", ESS=f"{ess_now:.0f}")

    if pbar is not None:
        pbar.close()

    var_lnz = lw_var_final if lw_var_final is not None else float(sum(step_rel_vars))
    log_evidence_err = jnp.asarray(math.sqrt(max(var_lnz, 0.0)), dtype=jnp.float64)

    # Low-variance systematic resampling only for equally weighted idata draws;
    # SMCResult.particles and SMCResult.weights preserve full terminal diversity.
    if float(jnp.max(final_weights) - jnp.min(final_weights)) > 1e-12:
        smc_key, sys_key = jax.random.split(smc_key)
        idata_idx = bj_resample.systematic(sys_key, final_weights, n_part)
        idata_particles = particles[idata_idx]
    else:
        idata_particles = particles

    posterior_dict = _unpack_constrained_samples(
        parameter_space, idata_particles[None, :, :]
    )
    idata = az.from_dict({"posterior": posterior_dict})

    if acc_rates:
        mean_acc_val = float(np.mean(acc_rates))
    else:
        mean_acc_val = 1.0 if int(num_mcmc_steps) > 0 else 0.0

    return SMCResult(
        particles=particles,
        weights=final_weights,
        idata=idata,
        log_evidence=log_z,
        log_evidence_err=log_evidence_err,
        tempering_betas=jnp.asarray(betas, dtype=jnp.float64),
        ess_history=jnp.asarray(ess_list, dtype=jnp.float64),
        root_ess=jnp.asarray(root_ess_val, dtype=jnp.float64),
        mean_accept_prob=jnp.asarray(mean_acc_val, dtype=jnp.float64),
    )
