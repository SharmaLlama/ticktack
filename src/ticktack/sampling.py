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
from typing import Any, Literal, NamedTuple

import arviz as az
import blackjax.smc.resampling as bj_resample
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
    LinearMarginalConfig,
    ParameterSpace,
    _has_loggp,
    _make_fast_unconstrained_loglike,
    _offset_column_count,
    _offset_only_cfg,
    joint_log_likelihood,
    linear_posterior,
    marginal_log_likelihood,
)
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
    max_tree_depth: int = 7,
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
            except Exception:
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
            except Exception:
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
            except Exception:
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

    rng_key, run_key = jax.random.split(rng_key)
    mcmc.run(run_key, init_params=init_params, extra_fields=("potential_energy",))

    z_samples = mcmc.get_samples(group_by_chain=True)["z"]  # (C, S, d)
    u_samples = prior_means[None, None, :] + prior_sds[None, None, :] * z_samples

    posterior_dict = _unpack_constrained_samples(parameter_space, u_samples)

    fast_lk_batch_fn = _make_fast_unconstrained_loglike(
        model, dataset, parameter_space, cfg, sigma_jitter
    )

    @jax.jit
    def _log_lk_batch(u_flat_in: jax.Array) -> jax.Array:
        return jax.vmap(fast_lk_batch_fn)(u_flat_in)

    u_flat = u_samples.reshape(-1, d)
    log_lk_arr = _log_lk_batch(u_flat).reshape(num_chains, num_samples)

    idata_groups: dict[str, Any] = {
        "posterior": posterior_dict,
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

            @jax.jit
            def _cond_draw_one(
                u_one: jax.Array, key_one: jax.Array
            ) -> tuple[jax.Array, jax.Array, jax.Array | None, jax.Array | None]:
                k1, k2 = jax.random.split(key_one)
                m_one = parameter_space.constrain(u_one, model)
                lp = linear_posterior(
                    m_one,
                    dataset,
                    lin_cfg,
                    rng_key=k1,
                    n_draws=1 if t_fine is not None else 0,
                    t_fine=t_fine,
                    sigma_jitter=sigma_jitter,
                )
                k_tot = lp.coeff_mean.shape[0]
                if k_tot > 0:
                    jit_eye = 1e-12 * jnp.eye(k_tot, dtype=jnp.float64)
                    L_c = jnp.linalg.cholesky(0.5 * (lp.coeff_cov + lp.coeff_cov.T) + jit_eye)
                    w_draw = lp.coeff_mean + L_c @ jax.random.normal(
                        k2, shape=(k_tot,), dtype=jnp.float64
                    )
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

            g_all, l_all, qf_all, df_all = jax.vmap(_cond_draw_one)(u_flat, draw_keys)
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


def sample_smc(
    model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    cfg: LinearMarginalConfig | None = None,
    *,
    num_particles: int = 500,
    target_ess: float = 0.75,
    num_mcmc_steps: int = 5,
    rng_key: jax.Array | None = None,
    sigma_jitter: ArrayLike | None = None,
    progress_bar: bool = False,
) -> SMCResult:
    """``BlackJAX`` adaptive tempered SMC with multinomial resampling and Lee-Whiteley ``sigma_{ln Z}``."""
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
        )

    prior_means = parameter_space.prior_means
    prior_sds = parameter_space.prior_sds

    fast_lk_smc_fn = _make_fast_unconstrained_loglike(
        model, dataset, parameter_space, cfg, sigma_jitter
    )

    @jax.jit
    def vmap_log_prior(u_batch: jax.Array) -> jax.Array:
        return jax.vmap(parameter_space.log_prior_unconstrained)(u_batch)

    @jax.jit
    def vmap_log_likelihood(u_batch: jax.Array) -> jax.Array:
        return jax.vmap(fast_lk_smc_fn)(u_batch)

    rwm_scale = 2.38 / math.sqrt(max(d, 1))

    @jax.jit
    def _mutate_particles(
        key_in: jax.Array,
        parts_in: jax.Array,
        lp_in: jax.Array,
        lk_in: jax.Array,
        chol_cov: jax.Array,
        beta_val: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        def _one_mh_step(carry: Any, step_key: jax.Array) -> tuple[Any, None]:
            p_curr, lp_c, lk_c = carry
            k_prop, k_acc = jax.random.split(step_key)
            noise = jax.random.normal(k_prop, shape=p_curr.shape, dtype=jnp.float64)
            props = p_curr + rwm_scale * (noise @ chol_cov.T)
            lp_prop = vmap_log_prior(props)
            lk_prop = vmap_log_likelihood(props)
            log_alpha = (lp_prop + beta_val * lk_prop) - (lp_c + beta_val * lk_c)
            u_acc = jax.random.uniform(k_acc, shape=(n_part,), dtype=jnp.float64)
            acc = jnp.log(u_acc) < log_alpha
            p_next = jnp.where(acc[:, None], props, p_curr)
            lp_next = jnp.where(acc, lp_prop, lp_c)
            lk_next = jnp.where(acc, lk_prop, lk_c)
            return (p_next, lp_next, lk_next), None

        step_keys = jax.random.split(key_in, max(int(num_mcmc_steps), 1))
        (p_out, lp_out, lk_out), _ = jax.lax.scan(
            _one_mh_step, (parts_in, lp_in, lk_in), step_keys
        )
        return p_out, lp_out, lk_out

    # --- Initialize particles directly from the exact prior ---
    rng_key, init_key, smc_key = jax.random.split(rng_key, 3)
    z_init = jax.random.normal(init_key, shape=(n_part, d), dtype=jnp.float64)
    particles = prior_means[None, :] + prior_sds[None, :] * z_init
    curr_lp = vmap_log_prior(particles)
    current_log_lks = vmap_log_likelihood(particles)

    beta = 0.0
    log_z = jnp.asarray(0.0, dtype=jnp.float64)
    betas: list[float] = [0.0]
    ess_list: list[float] = []
    step_rel_vars: list[float] = []
    eve = jnp.arange(n_part, dtype=jnp.int32)
    lw_var_final: float | None = None
    n_steps_total = 0
    unif_weights = jnp.ones((n_part,), dtype=jnp.float64) / n_part

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
            lw_ratio = ((n_part / max(n_part - 1, 1)) ** n_steps_total) * (
                1.0 - sum_lin2
            )
            indep_var = float(sum(step_rel_vars))
            if 0.05 < lw_ratio < 1.0:
                lw_var_final = max(-math.log(lw_ratio), 0.5 * indep_var)
            else:
                lw_var_final = max(indep_var, 1e-12)

        # Multinomial resampling (mandatory for unbiased Lee-Whiteley estimator)
        smc_key, resample_key, mut_key = jax.random.split(smc_key, 3)
        indices = bj_resample.multinomial(resample_key, w_norm, n_part)
        particles = particles[indices]
        curr_lp = curr_lp[indices]
        current_log_lks = current_log_lks[indices]
        eve = eve[indices]

        if int(num_mcmc_steps) > 0:
            cov = _trace_regularized_covariance(particles, unif_weights)
            L_cov = jnp.linalg.cholesky(cov)
            particles, curr_lp, current_log_lks = _mutate_particles(
                mut_key,
                particles,
                curr_lp,
                current_log_lks,
                L_cov,
                jnp.asarray(beta_new, dtype=jnp.float64),
            )

        beta = beta_new
        betas.append(float(beta))
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(beta=f"{beta:.3f}", ESS=f"{ess_now:.0f}")

    if pbar is not None:
        pbar.close()

    var_lnz = lw_var_final if lw_var_final is not None else float(sum(step_rel_vars))
    log_evidence_err = jnp.asarray(math.sqrt(max(var_lnz, 0.0)), dtype=jnp.float64)

    posterior_dict = _unpack_constrained_samples(
        parameter_space, particles[None, :, :]
    )
    idata = az.from_dict({"posterior": posterior_dict})

    return SMCResult(
        particles=particles,
        weights=unif_weights,
        idata=idata,
        log_evidence=log_z,
        log_evidence_err=log_evidence_err,
        tempering_betas=jnp.asarray(betas, dtype=jnp.float64),
        ess_history=jnp.asarray(ess_list, dtype=jnp.float64),
    )
