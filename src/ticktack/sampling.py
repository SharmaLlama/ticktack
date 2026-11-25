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
import numpyro.distributions.distribution as _np_dist_mod
import optax
import xarray as xr

if not hasattr(dist, "distribution"):
    dist.distribution = _np_dist_mod
from jax.typing import ArrayLike
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, autoguide

from ._utils import _safe_psd_root, _sigma_with_jitter
from .cbm import SubspaceCBM
from .data import Dataset
from .fields import LinearizedDCTOperator, build_linearized_dct_operator
from .fitting import MAPResult, _laplace_covariance, fit_map
from .likelihood import (
    FastLogLikeSpec,
    LinearMarginalConfig,
    ParameterSpace,
    _build_marginal_design,
    _extract_single_lgf,
    _make_fast_unconstrained_loglike,
    _marginal_design_is_static,
    _offset_column_count,
    _offset_only_cfg,
    _partition_terms_for_cfg,
    _unpack_offset_coeffs,
    _uses_joint_mode,
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
    "SubspaceGPFitResult",
    "sample_nuts",
    "fit_svi",
    "sample_smc",
    "reweight_exact_marginal",
    "fit_subspace_cbm_gp",
]


def _eval_single_loglike(
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
    if not fast_lk_spec.is_static:
        return jax.lax.map(fast_lk_spec, u_batch, batch_size=4)
    return jax.vmap(fast_lk_spec)(u_batch)


@eqx.filter_jit
def _vmap_fast_resid_jit(
    fast_lk_spec: FastLogLikeSpec,
    u_batch: jax.Array,
) -> jax.Array:
    if not fast_lk_spec.is_static:
        return jax.lax.map(fast_lk_spec.nonlinear_residual_from_u, u_batch, batch_size=4)
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
    L_cov = _safe_psd_root(coeff_cov, jitter=1e-12)
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
    n_solar: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None, jax.Array | None]:
    n_series = dataset.n_series
    n_off = _offset_column_count(lin_cfg, n_series)

    def _cond_draw_one(
        u_one: jax.Array, key_one: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None, jax.Array | None]:
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

        g_draw, l_draw = _unpack_offset_coeffs(w_draw, lin_cfg, n_series)

        if n_solar > 0:
            s_draw = (
                w_draw[n_off : n_off + n_solar]
                if k_tot >= n_off + n_solar
                else jnp.zeros((n_solar,), dtype=jnp.float64)
            )
        else:
            s_draw = None

        qf_d = lp.q_fine_draws[0] if lp.q_fine_draws is not None else None
        df_d = lp.d14c_fine_draws[0] if lp.d14c_fine_draws is not None else None
        return g_draw, l_draw, s_draw, qf_d, df_d

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


def sample_nuts(  # shape-exempt(carbon): public NUTS sampler adding Laplace mass-matrix warm-start keyword options
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
    adapt_mass_matrix: bool = True,
    init_u: ArrayLike | None = None,
    init_cov_u: ArrayLike | None = None,
    chain_method: Literal["auto", "vectorized", "sequential", "parallel"] = "auto",
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
        if init_u is not None:
            u_in = jnp.asarray(init_u, dtype=jnp.float64).reshape(d)
            z_cand = (u_in - prior_means) / prior_sds
            if bool(jnp.all(jnp.isfinite(z_cand))):
                z_center = z_cand
            if init_cov_u is not None:
                cov_u = jnp.asarray(init_cov_u, dtype=jnp.float64).reshape(d, d)
            elif d <= 16 and not isinstance(model.solver, ModaxSolver):
                cov_u = _laplace_covariance(
                    u_in, model, dataset, parameter_space, cfg, sigma_jitter
                )
        elif init_map:
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
            step_scale = 0.05 if init_u is not None else 0.15
            perturb = (
                step_scale * (raw_eps @ L_z.T) if L_z is not None else 0.05 * raw_eps
            )
            z_starts = z_center[None, :] + perturb
            init_params = {"z": z_starts[0] if num_chains == 1 else z_starts}

    numpyro_model_fn = _make_numpyro_model(
        model, dataset, parameter_space, cfg, sigma_jitter
    )
    nuts_kwargs: dict[str, Any] = {
        "target_accept_prob": target_accept_prob,
        "dense_mass": dense_mass,
        "adapt_mass_matrix": bool(adapt_mass_matrix),
        "max_tree_depth": max_tree_depth,
        "inverse_mass_matrix": inv_mass_init,
    }
    if init_cov_u is not None and inv_mass_init is not None:
        nuts_kwargs["step_size"] = 0.5
        nuts_kwargs["find_heuristic_step_size"] = False
    nuts_kernel = NUTS(numpyro_model_fn, **nuts_kwargs)
    eff_chain_method = (
        ("vectorized" if num_chains > 1 else "sequential")
        if chain_method == "auto"
        else chain_method
    )
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=eff_chain_method,
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
        joint_mode = _uses_joint_mode(model, parameter_space, cfg)
        is_lgf_marg = (
            not joint_mode
            and _extract_single_lgf(model.production)[0] is not None
        )
        lin_cfg = _offset_only_cfg(cfg) if (joint_mode or is_lgf_marg) else cfg
        n_off = _offset_column_count(lin_cfg, dataset.n_series)
        n_series = dataset.n_series
        _, _solar_terms, _, _, _ = _partition_terms_for_cfg(model.production, lin_cfg)
        n_solar = sum(
            2 * int(jnp.asarray(s.omegas, dtype=jnp.float64).reshape(-1).size)
            for s in _solar_terms
        )

        if n_off > 0 or n_solar > 0 or t_fine is not None:
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
                    g_d, l_d = _unpack_offset_coeffs(w_draw, lin_cfg, n_series)
                    s_d = (
                        w_draw[n_off : n_off + n_solar]
                        if n_solar > 0 and w_draw.shape[0] >= n_off + n_solar
                        else jnp.zeros((n_solar,), dtype=jnp.float64)
                    )
                    return (
                        g_d,
                        l_d,
                        s_d,
                        lp.q_fine_draws[0] if lp.q_fine_draws is not None else None,
                        lp.d14c_fine_draws[0] if lp.d14c_fine_draws is not None else None,
                    )
                g_all, l_all, s_all, qf_all, df_all = jax.vmap(_cond_draw_mdx)(u_flat, draw_keys)
            elif (
                t_fine_arr is None
                and dataset.n_obs > 0
                and _marginal_design_is_static(model, parameter_space, lin_cfg)
            ):
                sigma_eff = _sigma_with_jitter(dataset.sigma_flat, jit_arr)
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

                g_all, l_all = _unpack_offset_coeffs(w_draws, lin_cfg, n_series)
                if n_solar > 0 and k_tot >= n_off + n_solar:
                    s_all = w_draws[:, n_off : n_off + n_solar]
                else:
                    s_all = jnp.zeros((n_draws_tot, n_solar), dtype=jnp.float64)
                qf_all, df_all = None, None
            else:
                g_all, l_all, s_all, qf_all, df_all = _nuts_cond_draw_batch_jit(
                    u_flat,
                    draw_keys,
                    model,
                    dataset,
                    parameter_space,
                    lin_cfg,
                    t_fine_arr,
                    jit_arr,
                    n_solar=n_solar,
                )
            if n_off > 0:
                posterior_dict["offset_global"] = np.asarray(
                    g_all.reshape(num_chains, num_samples)
                )
                posterior_dict["offset_lab"] = np.asarray(
                    l_all.reshape(num_chains, num_samples, n_series)
                )
            if n_solar > 0 and s_all is not None:
                posterior_dict["solar_coeffs"] = np.asarray(
                    s_all.reshape(num_chains, num_samples, n_solar)
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
            L_cov = _safe_psd_root(cov)
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


# ---------------------------------------------------------------------------
# 4. Two-stage SubspaceCBM + marginalized GP sampling & exact IS reweighting
# ---------------------------------------------------------------------------


class SubspaceGPFitResult(NamedTuple):
    """Results from :func:`fit_subspace_cbm_gp`.

    Supports both attribute access (``res.idata``, ``res.map_result``, ``res.lin_op``,
    ``res.diagnostics``) and 4-tuple unpacking
    ``idata, map_result, lin_op, diagnostics = fit_subspace_cbm_gp(...)``.
    """

    idata: az.InferenceData
    """Posterior ``arviz.InferenceData`` (with ``sample_stats['log_weight_is']`` when reweighted)."""

    map_result: MAPResult
    """Stage-1 exact MAP estimate and regularized Laplace covariance."""

    lin_op: LinearizedDCTOperator
    """MAP-centred :class:`~ticktack.fields.LinearizedDCTOperator` used in Stage 2."""

    diagnostics: dict[str, Any]
    """Dictionary of sampler and exact importance-sampling diagnostics."""

    @property
    def smc_result(self) -> SMCResult | None:
        """Underlying :class:`SMCResult` when ``sampler='smc'``, else ``None``."""
        return self.diagnostics.get("smc_result")

    @property
    def kish_ess_ratio(self) -> float | None:
        """Kish effective sample size fraction ``ESS_IS / S`` when reweighted."""
        return self.diagnostics.get("kish_ess_ratio")

    @property
    def pareto_k(self) -> float | None:
        """PSIS Pareto-``k`` tail shape diagnostic when reweighted."""
        return self.diagnostics.get("pareto_k")


def _pack_unconstrained_from_idata(
    idata: az.InferenceData,
    parameter_space: ParameterSpace,
) -> tuple[np.ndarray, int, int]:
    """Recover ``(n_chains, n_draws, d)`` unconstrained ``u`` vectors from ``idata.posterior``."""
    post = idata.posterior
    n_chains = int(post.sizes["chain"])
    n_draws = int(post.sizes["draw"])
    d = parameter_space.n_params
    if d == 0:
        return np.zeros((n_chains, n_draws, 0), dtype=np.float64), n_chains, n_draws

    parts: list[np.ndarray] = []
    for spec, shape, (start, stop) in zip(
        parameter_space.specs,
        parameter_space.leaf_shapes,
        parameter_space.spec_slices,
    ):
        if stop == start:
            continue
        param_name = spec.path.replace(".", "__")
        if param_name not in post:
            raise KeyError(
                f"Parameter {param_name!r} (from spec {spec.path!r}) not found in idata.posterior."
            )
        theta_vals = jnp.asarray(post[param_name].values, dtype=jnp.float64)
        u_leaf = np.asarray(spec.unconstrain(theta_vals), dtype=np.float64)
        parts.append(u_leaf.reshape(n_chains, n_draws, stop - start))
    return np.concatenate(parts, axis=-1), n_chains, n_draws


@eqx.filter_jit
def _eval_marginal_ll_batch_jit(
    u_batch: jax.Array,
    model_template: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    cfg: LinearMarginalConfig,
    sigma_jitter_arr: jax.Array | None,
) -> jax.Array:
    """Evaluate marginal log-likelihood over ``(S, d)`` draws via memory-bounded ``lax.map``."""
    def _one(u_vec: jax.Array) -> jax.Array:
        m = parameter_space.constrain(u_vec, model_template)
        return marginal_log_likelihood(m, dataset, cfg, sigma_jitter=sigma_jitter_arr)

    return jax.lax.map(_one, u_batch, batch_size=4)


def _attach_log_weight_is(
    idata: az.InferenceData,
    log_w_2d: np.ndarray,
) -> az.InferenceData:
    """Attach ``(chain, draw)`` ``log_weight_is`` to ``idata.sample_stats`` (ArviZ 0.x & 1.x)."""
    idata_out = idata.copy()
    coords = {
        "chain": idata_out.posterior.coords["chain"],
        "draw": idata_out.posterior.coords["draw"],
    }
    da_lw = xr.DataArray(log_w_2d, dims=("chain", "draw"), coords=coords)
    has_ss = hasattr(idata_out, "sample_stats") and idata_out.sample_stats is not None
    if has_ss:
        ss_obj = idata_out.sample_stats
        ds_ss = ss_obj.to_dataset() if hasattr(ss_obj, "to_dataset") else ss_obj
        ds_ss = ds_ss.assign(log_weight_is=da_lw)
        if hasattr(xr, "DataTree") and isinstance(idata_out, xr.DataTree):
            idata_out["sample_stats"] = xr.DataTree(ds_ss)
        else:
            idata_out.sample_stats = ds_ss
    else:
        ds_new = xr.Dataset({"log_weight_is": da_lw})
        if hasattr(xr, "DataTree") and isinstance(idata_out, xr.DataTree):
            idata_out["sample_stats"] = xr.DataTree(ds_new)
        else:
            idata_out.add_groups({"sample_stats": ds_new})
    return idata_out


def _thin_idata_for_is(
    idata: az.InferenceData,
    n_chains: int,
    n_draws: int,
    is_max_draws: int | None,
) -> tuple[az.InferenceData, np.ndarray, int, int]:
    """Return ``(idata_sel, flat_eval_indices, n_chains_out, n_draws_out)`` with aligned draws."""
    if is_max_draws is not None and int(is_max_draws) < 1:
        raise ValueError(f"is_max_draws must be >= 1 or None; got {is_max_draws}.")
    s_total = n_chains * n_draws
    if is_max_draws is None or s_total <= int(is_max_draws):
        return idata, np.arange(s_total, dtype=np.int64), n_chains, n_draws

    cap = int(is_max_draws)
    if cap >= n_chains:
        per_chain = max(cap // n_chains, 1)
        chain_idx = np.arange(n_chains, dtype=np.int64)
        draw_idx = np.linspace(0, n_draws - 1, per_chain, dtype=np.int64)
    else:
        chain_idx = np.linspace(0, n_chains - 1, cap, dtype=np.int64)
        draw_idx = np.array([0], dtype=np.int64)

    flat_idx = (chain_idx[:, None] * n_draws + draw_idx[None, :]).reshape(-1)
    if hasattr(xr, "DataTree") and isinstance(idata, xr.DataTree):
        new_groups: dict[str, xr.Dataset] = {}
        for gname in idata.children:
            ds_g = idata[gname].to_dataset()
            sel: dict[str, Any] = {}
            if "chain" in ds_g.dims:
                sel["chain"] = chain_idx
            if "draw" in ds_g.dims:
                sel["draw"] = draw_idx
            new_groups[gname] = ds_g.isel(**sel) if sel else ds_g
        idata_sel = xr.DataTree.from_dict(new_groups)
    else:
        idata_sel = idata.isel(chain=chain_idx, draw=draw_idx)
    return idata_sel, flat_idx, int(chain_idx.shape[0]), int(draw_idx.shape[0])


def _compute_psis_pareto_k(log_w_2d: np.ndarray, *, r_eff: float = 1.0) -> float:
    """Estimate PSIS Pareto-``k`` tail index of importance weights ``exp(log_w_2d)``.

    ``arviz_stats.psislw`` follows the LOO convention and negates its input
    internally (``-ary``), so passing ``-log_w_2d`` fits the generalized Pareto
    tail to the right tail of ``exp(+log_w_2d)`` across ``("chain", "draw")``.
    When the upper tail collapses onto a single repeated atom above the bulk
    minimum (e.g. systematic resampling of one dominant particle), returns
    ``inf`` rather than ``nan`` so heavy-tail warnings still fire.
    """
    arr = np.asarray(log_w_2d, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if np.any(np.isnan(arr)) or not np.any(np.isfinite(arr)):
        return float("nan")
    if np.any(np.isposinf(arr)):
        return float("inf")
    finite = arr[np.isfinite(arr)]
    tiny = np.finfo(np.float64).tiny
    if finite.size == arr.size and float(np.ptp(finite)) <= tiny:
        return 0.0
    if arr.size < 25:
        return float("nan")
    r_eff_pos = max(float(r_eff), 1e-12)
    n_tail = min(arr.size // 5, int(3.0 * math.sqrt(arr.size / r_eff_pos)))
    if n_tail < 5:
        return float("nan")
    s_vals = np.sort(arr.reshape(-1))
    min_tail, max_tail = float(s_vals[-n_tail]), float(s_vals[-1])
    if np.isfinite(min_tail) and abs(max_tail - min_tail) <= tiny and max_tail > float(s_vals[0]) + tiny:
        return float("inf")
    arr_shifted = arr - float(np.max(finite))
    try:
        _, khat_da = xr.DataArray(-arr_shifted, dims=("chain", "draw")).azstats.psislw(
            r_eff=r_eff_pos
        )
        return float(np.max(np.asarray(khat_da, dtype=np.float64)))
    except ValueError:
        return float("nan")


def _warn_if_high_pareto_k(
    pareto_k: float,
    caller: str,
    *,
    n_eval: int | None = None,
) -> None:
    """Emit ``UserWarning`` when PSIS ``pareto_k > 0.7`` or ``pareto_k`` is ``NaN``."""
    if math.isnan(pareto_k):
        if n_eval is not None and n_eval < 25:
            reason = f"n_eval={n_eval} < 25 draws, too few draws for PSIS tail estimation"
        else:
            reason = "indeterminate or tied upper-tail order statistics"
        warnings.warn(
            f"{caller} importance-sampling Pareto-k diagnostic is NaN ({reason}); "
            "importance-weight tail heaviness could not be verified.",
            UserWarning,
            stacklevel=3,
        )
    elif pareto_k > 0.7:
        warnings.warn(
            f"{caller} importance-sampling Pareto-k diagnostic is high "
            f"(pareto_k={pareto_k:.3f} > 0.7); the surrogate posterior may be narrower "
            "than the exact target, yielding heavy-tailed weights with infinite variance.",
            UserWarning,
            stacklevel=3,
        )


def _summarize_is_weights(
    log_w_eval: np.ndarray,
    *,
    center_for_spread: bool = False,
) -> tuple[np.ndarray, float, float, float, float, float]:
    """Compute ``(w_is, kish_ess, kish_ess_ratio, log_mean_w, median_abs, max_abs)``."""
    finite = np.isfinite(log_w_eval)
    if not np.any(finite):
        return (
            np.full_like(log_w_eval, np.nan),
            0.0,
            0.0,
            float("nan"),
            float("nan"),
            float("nan"),
        )
    lw_max = float(np.max(log_w_eval[finite]))
    w_unnorm = np.where(
        finite, np.exp(np.where(finite, log_w_eval - lw_max, -np.inf)), 0.0
    )
    w_sum = float(np.sum(w_unnorm))
    w_is = w_unnorm / max(w_sum, 1e-300)
    kish_ess = float((np.sum(w_is) ** 2) / max(float(np.sum(w_is**2)), 1e-300))
    kish_ess_ratio = float(kish_ess / max(len(w_is), 1))
    log_mean_w = float(lw_max + math.log(max(w_sum, 1e-300)) - math.log(len(w_is)))
    spread_vals = (
        np.abs(log_w_eval[finite] - log_mean_w)
        if center_for_spread
        else np.abs(log_w_eval[finite])
    )
    return (
        w_is,
        kish_ess,
        kish_ess_ratio,
        log_mean_w,
        float(np.median(spread_vals)),
        float(np.max(spread_vals)),
    )


def reweight_exact_marginal(
    idata: az.InferenceData,
    model_template: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    marginal_config_exact: LinearMarginalConfig,
    marginal_config_tan: LinearMarginalConfig,
    *,
    is_max_draws: int | None = None,
    sigma_jitter: ArrayLike | None = None,
) -> tuple[az.InferenceData, dict[str, Any]]:
    r"""Compute exact importance-sampling weights ``ln L_exact(u_s) - ln L_tan(u_s)``.

    Evaluates the discrepancy between the exact marginalized log-likelihood and the
    MAP-centred tangent-linear surrogate via memory-bounded ``jax.lax.map`` batches
    across all posterior draws by default (``is_max_draws=None``). When an explicit
    cap ``is_max_draws < S_total`` is requested, evenly thins ``idata`` along the
    draw/chain axes so ``idata_out.sample_stats["log_weight_is"]`` contains zero
    ``NaN``s and ``diagnostics["weights_is"]`` stays shape-aligned with ``idata_out``.
    ``diagnostics["pareto_k"]`` reports the PSIS Pareto-``k`` tail index over
    ``("chain", "draw")`` and emits a :class:`UserWarning` when ``pareto_k > 0.7`` or
    ``pareto_k`` is ``NaN`` (e.g. ``n_eval < 25`` or tied upper-tail order statistics).
    """
    u_chains, n_chains, n_draws = _pack_unconstrained_from_idata(idata, parameter_space)
    s_total = n_chains * n_draws
    d = parameter_space.n_params
    u_flat = u_chains.reshape(s_total, d)

    idata_sel, eval_idx, n_c_out, n_d_out = _thin_idata_for_is(
        idata, n_chains, n_draws, is_max_draws
    )
    u_eval = jnp.asarray(u_flat[eval_idx], dtype=jnp.float64)

    jit_arr = (
        jnp.asarray(sigma_jitter, dtype=jnp.float64)
        if sigma_jitter is not None
        else None
    )

    ll_ex = _eval_marginal_ll_batch_jit(
        u_eval,
        model_template,
        dataset,
        parameter_space,
        marginal_config_exact,
        jit_arr,
    )
    ll_tn = _eval_marginal_ll_batch_jit(
        u_eval,
        model_template,
        dataset,
        parameter_space,
        marginal_config_tan,
        jit_arr,
    )
    log_w_eval = np.asarray(ll_ex - ll_tn, dtype=np.float64)

    w_is, kish_ess, kish_ess_ratio, log_z_corr, med_abs, max_abs = _summarize_is_weights(
        log_w_eval, center_for_spread=False
    )
    log_w_2d = log_w_eval.reshape(n_c_out, n_d_out)
    pareto_k = _compute_psis_pareto_k(log_w_2d)
    _warn_if_high_pareto_k(pareto_k, "reweight_exact_marginal", n_eval=int(eval_idx.shape[0]))
    idata_out = _attach_log_weight_is(idata_sel, log_w_2d)

    diagnostics: dict[str, Any] = {
        "log_weight_is": log_w_eval,
        "weights_is": w_is,
        "eval_indices": eval_idx,
        "n_eval": int(eval_idx.shape[0]),
        "n_total": int(s_total),
        "kish_ess": kish_ess,
        "kish_ess_ratio": kish_ess_ratio,
        "pareto_k": pareto_k,
        "median_abs_dlnL": med_abs,
        "max_abs_dlnL": max_abs,
        "log_z_correction": log_z_corr,
    }
    return idata_out, diagnostics


def _run_subspace_stage2_sampler(
    sampler: str,
    model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    cfg_tan: LinearMarginalConfig,
    map_result: MAPResult,
    samp_key: jax.Array,
    sigma_jitter: ArrayLike | None,
    skw: dict[str, Any],
) -> tuple[az.InferenceData, dict[str, Any], jax.Array | None, jax.Array | None]:  # shape-exempt(carbon): internal Stage-2 dispatch forwarding model/dataset/config/MAP state
    """Run Stage-2 NUTS, SMC, or Laplace sampling under the MAP-centred tangent operator."""
    extra_diag: dict[str, Any] = {}
    if sampler == "nuts":
        num_warmup = int(skw.pop("num_warmup", 200))
        adapt_mm = skw.pop("adapt_mass_matrix", num_warmup >= 150)
        for k_ignore in ("init_map", "init_u", "init_cov_u"):
            skw.pop(k_ignore, None)
        idata = sample_nuts(
            model,
            dataset,
            parameter_space,
            cfg_tan,
            num_warmup=num_warmup,
            init_map=False,
            init_u=map_result.u_map,
            init_cov_u=map_result.laplace_cov,
            adapt_mass_matrix=bool(adapt_mm),
            rng_key=samp_key,
            sigma_jitter=sigma_jitter,
            **skw,
        )
        return idata, extra_diag, None, None
    if sampler == "smc":
        smc_res = sample_smc(
            model,
            dataset,
            parameter_space,
            cfg_tan,
            rng_key=samp_key,
            sigma_jitter=sigma_jitter,
            **skw,
        )
        extra_diag["smc_result"] = smc_res
        extra_diag["log_evidence_tan"] = float(smc_res.log_evidence)
        extra_diag["log_evidence_err"] = float(smc_res.log_evidence_err)
        return smc_res.idata, extra_diag, None, None

    num_samples = int(skw.pop("num_samples", 200))
    d = parameter_space.n_params
    cov_u = 0.5 * (map_result.laplace_cov + map_result.laplace_cov.T)
    ev, evec = jnp.linalg.eigh(cov_u)
    cov_spd = (evec * jnp.maximum(ev, 1e-12)[None, :]) @ evec.T
    l_u = jnp.linalg.cholesky(0.5 * (cov_spd + cov_spd.T))
    z_draw = jax.random.normal(samp_key, (1, num_samples, d), dtype=jnp.float64)
    laplace_u_draws = map_result.u_map[None, None, :] + z_draw @ l_u.T
    laplace_log_q = (
        -0.5 * jnp.sum(jnp.square(z_draw), axis=-1)
        - jnp.sum(jnp.log(jnp.diag(l_u)))
        - 0.5 * float(d) * math.log(2.0 * math.pi)
    )
    post_dict = _unpack_constrained_samples(parameter_space, laplace_u_draws)
    return az.from_dict({"posterior": post_dict}), extra_diag, laplace_u_draws, laplace_log_q


def _reweight_laplace_draws(
    idata: az.InferenceData,
    laplace_u_draws: jax.Array,
    laplace_log_q: jax.Array,
    model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    cfg_exact: LinearMarginalConfig,
    *,
    is_max_draws: int | None = None,
    sigma_jitter: ArrayLike | None = None,
) -> tuple[az.InferenceData, dict[str, Any]]:  # shape-exempt(carbon): internal Stage-3 Laplace IS evaluator matching reweight_exact_marginal
    """Compute exact IS weights and PSIS Pareto-``k`` for Gaussian Laplace draws."""
    n_c_lap, n_d_lap, d_lap = laplace_u_draws.shape
    idata_sel, eval_idx, n_c_out, n_d_out = _thin_idata_for_is(
        idata, n_c_lap, n_d_lap, is_max_draws
    )
    u_eval = laplace_u_draws.reshape(-1, d_lap)[eval_idx]
    log_q_eval = laplace_log_q.reshape(-1)[eval_idx]
    jit_arr = (
        jnp.asarray(sigma_jitter, dtype=jnp.float64)
        if sigma_jitter is not None
        else None
    )
    ll_ex = _eval_marginal_ll_batch_jit(
        u_eval, model, dataset, parameter_space, cfg_exact, jit_arr
    )
    lp_ex = _vmap_log_prior_jit(parameter_space, u_eval)
    log_w_eval = np.asarray(ll_ex + lp_ex - log_q_eval, dtype=np.float64)
    w_is, kish_ess, kish_ess_ratio, log_z_exact, med_abs, max_abs = (
        _summarize_is_weights(log_w_eval, center_for_spread=True)
    )
    log_w_2d = log_w_eval.reshape(n_c_out, n_d_out)
    pareto_k = _compute_psis_pareto_k(log_w_2d, r_eff=1.0)
    _warn_if_high_pareto_k(pareto_k, "fit_subspace_cbm_gp", n_eval=int(eval_idx.shape[0]))
    idata_out = _attach_log_weight_is(idata_sel, log_w_2d)
    return idata_out, {
        "log_weight_is": log_w_eval,
        "weights_is": w_is,
        "eval_indices": eval_idx,
        "n_eval": int(eval_idx.shape[0]),
        "n_total": int(n_c_lap * n_d_lap),
        "kish_ess": kish_ess,
        "kish_ess_ratio": kish_ess_ratio,
        "pareto_k": pareto_k,
        "median_abs_dlnL": med_abs,
        "max_abs_dlnL": max_abs,
        "log_evidence_exact": log_z_exact,
    }


def _reweight_smc_particles(
    smc_res: SMCResult,
    model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    cfg_exact: LinearMarginalConfig,
    cfg_tan: LinearMarginalConfig,
    *,
    is_max_draws: int | None = None,
    sigma_jitter: ArrayLike | None = None,
) -> tuple[az.InferenceData, dict[str, Any]]:
    """Reweight unresampled SMC particles ``(particles, weights)`` to the exact marginal target."""
    parts = jnp.asarray(smc_res.particles, dtype=jnp.float64)
    w_smc = np.asarray(smc_res.weights, dtype=np.float64).reshape(-1)
    n_part, d_part = int(parts.shape[0]), int(parts.shape[1])
    idata_raw = az.from_dict(
        {"posterior": _unpack_constrained_samples(parameter_space, parts[None, :, :])}
    )
    idata_sel, eval_idx, n_c_out, n_d_out = _thin_idata_for_is(
        idata_raw, 1, n_part, is_max_draws
    )
    u_eval = parts.reshape(-1, d_part)[eval_idx]
    w_sel = np.maximum(w_smc[eval_idx], 0.0)
    w_sel_sum = float(np.sum(w_sel))
    w_norm = (
        w_sel / w_sel_sum
        if w_sel_sum > 0.0
        else np.full_like(w_sel, 1.0 / max(len(w_sel), 1))
    )
    jit_arr = None if sigma_jitter is None else jnp.asarray(sigma_jitter, dtype=jnp.float64)
    ll_ex = _eval_marginal_ll_batch_jit(u_eval, model, dataset, parameter_space, cfg_exact, jit_arr)
    ll_tn = _eval_marginal_ll_batch_jit(u_eval, model, dataset, parameter_space, cfg_tan, jit_arr)
    dlnl_eval = np.asarray(ll_ex - ll_tn, dtype=np.float64)
    n_ev = max(len(eval_idx), 1)
    log_w_smc = np.log(np.maximum(n_ev * w_norm, 1e-300))
    log_w_eval = log_w_smc + dlnl_eval
    w_is, kish_ess, kish_ess_ratio, log_z_corr, _, _ = _summarize_is_weights(
        log_w_eval, center_for_spread=False
    )
    ess_smc = float(1.0 / max(float(np.sum(w_norm**2)), 1e-300))
    fin_d = np.isfinite(dlnl_eval)
    abs_d = np.abs(dlnl_eval[fin_d]) if np.any(fin_d) else np.array([np.nan])
    log_w_2d = log_w_eval.reshape(n_c_out, n_d_out)
    pareto_k = _compute_psis_pareto_k(log_w_2d, r_eff=1.0)
    _warn_if_high_pareto_k(pareto_k, "fit_subspace_cbm_gp", n_eval=int(eval_idx.shape[0]))
    idata_out = _attach_log_weight_is(idata_sel, log_w_2d)
    return idata_out, {
        "log_weight_is": log_w_eval,
        "log_likelihood_diff": dlnl_eval,
        "weights_is": w_is,
        "eval_indices": eval_idx,
        "n_eval": int(eval_idx.shape[0]),
        "n_total": n_part,
        "kish_ess": kish_ess,
        "kish_ess_ratio": kish_ess_ratio,
        "is_ess_ratio": float(kish_ess / ess_smc),
        "pareto_k": pareto_k,
        "median_abs_dlnL": float(np.median(abs_d)),
        "max_abs_dlnL": float(np.max(abs_d)),
        "log_z_correction": log_z_corr,
        "log_evidence_exact": float(smc_res.log_evidence) + log_z_corr,
    }


def fit_subspace_cbm_gp(  # shape-exempt(carbon): public two-stage sampler accepting **sampler_kwargs passthrough
    model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    marginal_config: LinearMarginalConfig | None = None,
    *,
    sampler: Literal["nuts", "smc", "laplace"] = "nuts",
    reweight_exact: bool = True,
    is_max_draws: int | None = None,
    rng_key: jax.Array | None = None,
    sigma_jitter: ArrayLike | None = None,
    **sampler_kwargs: Any,
) -> SubspaceGPFitResult:
    """Two-stage MAP-centred tangent-linear sampler + exact IS reweighting for ``SubspaceCBM``.

    Stage 1 runs exact :func:`~ticktack.fitting.fit_map` under ``marginal_log_posterior``,
    builds a MAP-centred :class:`~ticktack.fields.LinearizedDCTOperator` around
    ``map_result.model.cbm``, and computes the regularized Laplace covariance.
    Stage 2 samples with :func:`sample_nuts` (preconditioned by the Stage-1 Laplace dense
    mass matrix), :func:`sample_smc`, or Gaussian Laplace draws using
    ``marginal_config_tan = eqx.tree_at(lambda c: c.dct_operator, cfg_exact, lin_op)``
    while keeping ``SteadyProduction + MiyakeSpike``, hierarchical offset ratios, and
    any sampled GP kernel hyperparameters (``production.field.kernel.*``) exact at every step.
    Stage 3 (when ``reweight_exact=True``) applies exact importance reweighting:
    ``ln L_exact(u_s) - ln L_tan(u_s)`` for ``sampler`` in ``{"nuts", "smc"}`` and
    ``ln L_exact(u_s) + ln p(u_s) - ln N(u_s | u_map, Sigma_spd)`` for ``sampler="laplace"``.
    """
    if not isinstance(model.cbm, SubspaceCBM):
        raise TypeError(
            "fit_subspace_cbm_gp requires model.cbm to be a SubspaceCBM; "
            f"got {type(model.cbm).__name__}."
        )
    if sampler not in ("nuts", "smc", "laplace"):
        raise ValueError(
            f"sampler must be 'nuts', 'smc', or 'laplace'; got {sampler!r}."
        )

    cfg_in = LinearMarginalConfig() if marginal_config is None else marginal_config
    cfg_exact = (
        eqx.tree_at(lambda c: c.dct_operator, cfg_in, None)
        if cfg_in.dct_operator is not None
        else cfg_in
    )

    _, _, _, _, field_terms = _partition_terms_for_cfg(model.production, cfg_exact)
    if len(field_terms) != 1:
        raise ValueError(
            "fit_subspace_cbm_gp requires exactly one marginalized GaussianField in "
            f"model.production; found {len(field_terms)}."
        )
    gf = field_terms[0]

    key = jax.random.PRNGKey(0) if rng_key is None else rng_key
    _, map_key, samp_key = jax.random.split(key, 3)
    skw = dict(sampler_kwargs)
    map_method = skw.pop("map_method", "BFGS")
    map_max_steps = int(skw.pop("map_max_steps", 1000))
    map_tol = float(skw.pop("map_tol", 1e-6))

    map_result = fit_map(
        model,
        dataset,
        parameter_space,
        cfg_exact,
        method=map_method,
        max_steps=map_max_steps,
        tol=map_tol,
        rng_key=map_key,
        sigma_jitter=sigma_jitter,
        compute_laplace=False,
        compute_linear_posterior=False,
    )
    if not map_result.converged:
        warnings.warn(
            f"fit_subspace_cbm_gp Stage-1 MAP optimization did not converge "
            f"(n_steps={map_result.n_steps}, method={map_method!r}); centring "
            "LinearizedDCTOperator at the best iterate anyway — inspect IS diagnostics.",
            RuntimeWarning,
            stacklevel=2,
        )
    assert isinstance(map_result.model.cbm, SubspaceCBM)
    lin_op = build_linearized_dct_operator(
        map_result.model.cbm,
        dataset,
        t_start=gf.t_start,
        t_end=gf.t_end,
        n_grid=gf.n_grid,
        q_std=model.q_std,
        alpha_unit=model.alpha_unit,
        template=gf.template,
    )
    cfg_tan = eqx.tree_at(
        lambda c: c.dct_operator, cfg_exact, lin_op, is_leaf=lambda x: x is None
    )
    laplace_cov = _laplace_covariance(
        map_result.u_map, model, dataset, parameter_space, cfg_tan, sigma_jitter
    )
    map_result = map_result._replace(laplace_cov=laplace_cov)

    diagnostics: dict[str, Any] = {
        "sampler": sampler,
        "reweight_exact": bool(reweight_exact),
        "map_converged": bool(map_result.converged),
        "map_log_posterior": float(map_result.log_posterior),
        "map_log_likelihood": float(map_result.log_likelihood),
    }

    idata, stage2_diag, lap_u, lap_lq = _run_subspace_stage2_sampler(
        sampler, model, dataset, parameter_space, cfg_tan, map_result, samp_key, sigma_jitter, skw
    )
    diagnostics.update(stage2_diag)

    if reweight_exact:
        if sampler == "laplace":
            assert lap_u is not None and lap_lq is not None
            idata, is_diag = _reweight_laplace_draws(
                idata,
                lap_u,
                lap_lq,
                model,
                dataset,
                parameter_space,
                cfg_exact,
                is_max_draws=is_max_draws,
                sigma_jitter=sigma_jitter,
            )
        elif sampler == "smc":
            smc_obj = diagnostics["smc_result"]
            idata, is_diag = _reweight_smc_particles(
                smc_obj,
                model,
                dataset,
                parameter_space,
                cfg_exact,
                cfg_tan,
                is_max_draws=is_max_draws,
                sigma_jitter=sigma_jitter,
            )
        else:
            idata, is_diag = reweight_exact_marginal(
                idata,
                model,
                dataset,
                parameter_space,
                cfg_exact,
                cfg_tan,
                is_max_draws=is_max_draws,
                sigma_jitter=sigma_jitter,
            )
        diagnostics.update(is_diag)
        eff_ratio = float(is_diag.get("is_ess_ratio", is_diag["kish_ess_ratio"]))
        if math.isfinite(eff_ratio) and eff_ratio < 0.85:
            warnings.warn(
                f"fit_subspace_cbm_gp exact IS efficiency is low "
                f"(is_ess_ratio={eff_ratio:.3f} < 0.85, "
                f"max|ΔlnL|={is_diag['max_abs_dlnL']:.3f} nats); consider checking "
                "posterior multimodality or sampling with exact marginal_log_likelihood.",
                RuntimeWarning,
                stacklevel=2,
            )

    return SubspaceGPFitResult(
        idata=idata,
        map_result=map_result,
        lin_op=lin_op,
        diagnostics=diagnostics,
    )
