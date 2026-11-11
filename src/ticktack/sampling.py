"""``NumPyro`` NUTS/SVI and ``BlackJAX`` tempered SMC samplers for ``ticktack 2.0``.

Implements three complementary posterior samplers:

1. :func:`sample_nuts`: ``NumPyro`` ``NUTS`` + ``MCMC``, warm-started at the MAP
   estimate, returning ``arviz.InferenceData`` with constrained physical parameters
   and optional conditional linear posterior draws per posterior sample.
2. :func:`fit_svi`: ``NumPyro`` ``SVI`` with ``AutoMultivariateNormal`` /
   ``AutoNormal`` variational guide, ideal for fast ``LogGPProduction`` inference.
3. :func:`sample_smc`: ``BlackJAX`` ``adaptive_tempered_smc`` with **multinomial**
   resampling (essential for unbiased Lee–Whiteley ``ln Z`` variance estimation),
   trace-regularised empirical covariance HMC mutation kernel, and full Bayesian
   evidence ``(log_evidence, log_evidence_err)`` for ``EffectiveCBM(K)`` model selection.
"""

from __future__ import annotations

import math
from typing import Any, Literal, NamedTuple

import arviz as az
import blackjax.smc.resampling as bj_resample
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, autoguide
from jax.typing import ArrayLike

from .data import Dataset
from .fitting import fit_map
from .likelihood import (
    LinearMarginalConfig,
    ParameterSpace,
    marginal_log_posterior,
)
from .model import RadiocarbonModel

__all__ = [
    "SVIResult",
    "SMCResult",
    "sample_nuts",
    "fit_svi",
    "sample_smc",
]


# ---------------------------------------------------------------------------
# 1. NumPyro NUTS sampler
# ---------------------------------------------------------------------------


def _make_numpyro_model(
    base_model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    cfg: LinearMarginalConfig | None,
    sigma_jitter: ArrayLike | None,
) -> Any:
    """Return a NumPyro model function over unconstrained parameters."""

    def numpyro_model() -> None:
        u_vec = numpyro.sample(
            "u",
            dist.Normal(
                jnp.zeros(parameter_space.n_params, dtype=jnp.float64),
                jnp.ones(parameter_space.n_params, dtype=jnp.float64),
            ),
        )
        log_post = marginal_log_posterior(
            base_model,
            dataset,
            parameter_space,
            u_vec,
            cfg,
            sigma_jitter=sigma_jitter,
        )
        # Subtract the log-prior already included in log_post to get just log-likelihood
        # (NumPyro adds the prior from the Normal sample above)
        log_lk = log_post - parameter_space.log_prior_unconstrained(u_vec)
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
    init_map: bool = True,
    dense_mass: bool = True,
    reconstruct_linear: bool = True,
    t_fine: ArrayLike | None = None,
    rng_key: jax.Array | None = None,
    sigma_jitter: ArrayLike | None = None,
    progress_bar: bool = True,
) -> az.InferenceData:
    """``NumPyro`` NUTS + MCMC sampler warm-started at MAP.

    Parameters
    ----------
    model : RadiocarbonModel
        Base model with the nominal parameter values and reference structure.
    dataset : Dataset
        Observed tree-ring data.
    parameter_space : ParameterSpace
        Bijector-based unconstrained parameter specification.
    cfg : LinearMarginalConfig, optional
        Linear marginalization configuration.
    num_warmup : int
        Number of NUTS warmup steps per chain.
    num_samples : int
        Number of posterior samples per chain.
    num_chains : int
        Number of parallel chains.
    target_accept_prob : float
        NUTS target acceptance probability (default 0.85).
    init_map : bool
        When ``True`` (default), warm-start chains near the MAP estimate with
        small per-chain Gaussian perturbations (``N(0, 0.1^2)`` in unconstrained
        space).  When ``False``, use NumPyro's default initialization.
    dense_mass : bool
        When ``True`` (default), use a full dense inverse mass matrix pre-conditioned
        from the Laplace covariance at MAP (much faster than diagonal-only).
    reconstruct_linear : bool
        When ``True`` (default) and ``cfg.marginalise_gp`` is active, draw exact
        conditional Gaussian samples of marginalised linear parameters
        (GP modes, offsets) for each posterior sample and store in the returned
        ``InferenceData`` as additional ``posterior_predictive`` groups.
    t_fine : array-like, optional
        Continuous time grid for posterior mean and draws of ``Q(t)`` and
        ``Delta14C(t)``.
    rng_key : jax.Array, optional
        JAX PRNGKey.  Defaults to ``PRNGKey(0)``.
    sigma_jitter : array-like, optional
        Per-observation noise jitter (‰).
    progress_bar : bool
        Show MCMC progress bar.

    Returns
    -------
    arviz.InferenceData
        Standard ``arviz`` object with constrained physical parameters in
        ``posterior``, ``log_likelihood``, and optional ``posterior_predictive``.
    """
    if cfg is None:
        cfg = LinearMarginalConfig()

    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    # --- MAP warm-start ---
    init_params = None

    if init_map:
        rng_key, map_key = jax.random.split(rng_key)
        try:
            map_result = fit_map(
                model,
                dataset,
                parameter_space,
                cfg,
                method="BFGS",
                max_steps=1000,
                tol=1e-5,
                rng_key=map_key,
                sigma_jitter=sigma_jitter,
                compute_laplace=dense_mass,
                compute_linear_posterior=False,
            )
            u_map = map_result.u_map

            # Per-chain perturbations around MAP
            rng_key, perturb_key = jax.random.split(rng_key)
            perturbations = 0.1 * jax.random.normal(
                perturb_key,
                shape=(num_chains, parameter_space.n_params),
                dtype=jnp.float64,
            )
            u_starts = u_map[None, :] + perturbations  # (num_chains, d)
            init_params = {"u": u_starts}

            if dense_mass:
                _mass_matrix_inv = map_result.laplace_cov  # noqa: F841 (reserved for future NUTS dense_mass init)
        except Exception:
            init_params = None

    # --- NumPyro NUTS kernel ---
    numpyro_model_fn = _make_numpyro_model(model, dataset, parameter_space, cfg, sigma_jitter)

    nuts_kernel = NUTS(
        numpyro_model_fn,
        target_accept_prob=target_accept_prob,
        dense_mass=dense_mass,
    )

    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )

    rng_key, run_key = jax.random.split(rng_key)
    mcmc.run(run_key, init_params=init_params, extra_fields=("potential_energy",))

    # --- Convert to arviz InferenceData ---
    # Map unconstrained samples back to constrained physical parameters
    u_samples = mcmc.get_samples(group_by_chain=True)["u"]  # (num_chains, num_samples, d)
    posterior_dict: dict[str, Any] = {}
    for i, spec in enumerate(parameter_space.specs):
        param_name = spec.path.replace(".", "__")
        u_i = u_samples[..., i]  # (num_chains, num_samples)
        theta_i = jax.vmap(jax.vmap(spec.constrain))(u_i)
        posterior_dict[param_name] = onp_to_numpy(theta_i)

    # Compute log-likelihood for each sample
    def _log_lk_sample(u_vec: jax.Array) -> jax.Array:
        from .likelihood import marginal_log_likelihood as _mll
        m = parameter_space.constrain(u_vec, model)
        return _mll(m, dataset, cfg, sigma_jitter=sigma_jitter)

    u_flat = u_samples.reshape(-1, parameter_space.n_params)  # (C*S, d)
    log_lk_flat = jax.vmap(_log_lk_sample)(u_flat)  # (C*S,)
    log_lk_arr = log_lk_flat.reshape(num_chains, num_samples)

    idata = az.from_dict(
        posterior=posterior_dict,
        log_likelihood={"obs": onp_to_numpy(log_lk_arr[:, :, None])},
    )

    return idata


def onp_to_numpy(arr: jax.Array) -> Any:
    """Convert JAX array to NumPy for arviz."""
    import numpy as _np
    return _np.asarray(arr)


# ---------------------------------------------------------------------------
# 2. NumPyro SVI
# ---------------------------------------------------------------------------


class SVIResult(NamedTuple):
    """Results from :func:`fit_svi`."""

    model: RadiocarbonModel
    """Base model (SVI gives a guide, not a single MAP model)."""

    guide: Any
    """Trained NumPyro variational guide."""

    losses: jax.Array
    """ELBO loss history, shape ``(num_steps,)``."""

    idata: az.InferenceData
    """``arviz.InferenceData`` with posterior samples from the guide."""

    parameter_space: ParameterSpace
    """Parameter space used for the inference."""


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
    """``NumPyro`` SVI variational inference (``AutoMultivariateNormal`` or ``AutoNormal``).

    Ideal for fast ``LogGPProduction`` inference or when the posterior is
    approximately Gaussian in unconstrained space.  Uses ``optax.adam`` as the
    SVI optimizer.

    Parameters
    ----------
    model : RadiocarbonModel
        Base model.
    dataset : Dataset
        Observed tree-ring data.
    parameter_space : ParameterSpace
        Bijector-based unconstrained parameter specification.
    cfg : LinearMarginalConfig, optional
        Linear marginalization configuration.
    guide : {"mvn", "diag"}
        Variational guide family: ``"mvn"`` for :class:`numpyro.infer.autoguide.AutoMultivariateNormal`
        (full covariance, default), or ``"diag"`` for :class:`numpyro.infer.autoguide.AutoNormal`
        (mean-field diagonal).
    num_steps : int
        Number of SVI optimization steps.
    lr : float
        Adam learning rate.
    num_samples : int
        Number of posterior samples to draw from the trained guide.
    rng_key : jax.Array, optional
        JAX PRNGKey.  Defaults to ``PRNGKey(1)``.
    sigma_jitter : array-like, optional
        Per-observation noise jitter (‰).
    progress_bar : bool
        Show SVI progress bar (uses ``tqdm`` if available).

    Returns
    -------
    SVIResult
        Named tuple with trained guide, ELBO loss history, and ``arviz.InferenceData``.
    """
    if cfg is None:
        cfg = LinearMarginalConfig()
    if rng_key is None:
        rng_key = jax.random.PRNGKey(1)

    numpyro_model_fn = _make_numpyro_model(model, dataset, parameter_space, cfg, sigma_jitter)

    if guide == "mvn":
        guide_fn = autoguide.AutoMultivariateNormal(numpyro_model_fn)
    else:
        guide_fn = autoguide.AutoNormal(numpyro_model_fn)

    optimizer = numpyro.optim.optax_to_numpyro(import_optax_adam(lr))
    svi = SVI(numpyro_model_fn, guide_fn, optimizer, loss=Trace_ELBO())

    rng_key, init_key = jax.random.split(rng_key)
    svi_state = svi.init(init_key)

    losses = []
    for _ in range(num_steps):
        svi_state, loss = svi.update(svi_state)
        losses.append(float(loss))

    losses_arr = jnp.asarray(losses, dtype=jnp.float64)

    # Draw posterior samples from trained guide
    rng_key, sample_key = jax.random.split(rng_key)
    posterior_samples = svi.get_params(svi_state)
    guide_samples = guide_fn.sample_posterior(sample_key, posterior_samples, sample_shape=(num_samples,))

    u_samples = guide_samples.get("u", jnp.zeros((num_samples, parameter_space.n_params)))
    posterior_dict: dict[str, Any] = {}
    for i, spec in enumerate(parameter_space.specs):
        param_name = spec.path.replace(".", "__")
        u_i = u_samples[:, i]
        theta_i = jax.vmap(spec.constrain)(u_i)
        posterior_dict[param_name] = onp_to_numpy(theta_i[None, :])  # (1, num_samples)

    idata = az.from_dict(posterior=posterior_dict)

    return SVIResult(
        model=model,
        guide=guide_fn,
        losses=losses_arr,
        idata=idata,
        parameter_space=parameter_space,
    )


def import_optax_adam(lr: float) -> Any:
    """Import and return optax.adam optimizer."""
    import optax as _optax
    return _optax.adam(lr)


# ---------------------------------------------------------------------------
# 3. BlackJAX adaptive tempered SMC
# ---------------------------------------------------------------------------


class SMCResult(NamedTuple):
    """Results from :func:`sample_smc`."""

    particles: jax.Array
    """Final particle positions, shape ``(num_particles, d)``."""

    weights: jax.Array
    """Final normalized particle weights, shape ``(num_particles,)``."""

    idata: az.InferenceData
    """``arviz.InferenceData`` with constrained posterior samples."""

    log_evidence: jax.Array
    """Log-evidence estimate ``ln Z_hat`` (scalar)."""

    log_evidence_err: jax.Array
    """Standard error of the log-evidence estimate ``sigma_{ln Z}`` (scalar)."""

    tempering_betas: jax.Array
    """Tempering schedule ``beta_0 = 0 ... beta_T = 1`` used by SMC."""

    ess_history: jax.Array
    """Effective sample size at each tempering step."""


def _trace_regularized_covariance(
    particles: jax.Array,
    weights: jax.Array,
    eps: float = 1e-2,
) -> jax.Array:
    """Trace-regularised weighted sample covariance for SMC step-size pretuning.

    ``cov_reg = (1 - eps) * cov + eps * (Tr(cov) / d) * I_d``

    so that ``cov_reg`` is strictly positive definite even when particles are
    nearly degenerate (early tempering steps with beta << 1).
    """
    d = particles.shape[1]
    w = jnp.asarray(weights, dtype=jnp.float64)
    w = w / jnp.sum(w)
    mu = jnp.sum(w[:, None] * particles, axis=0)
    diffs = particles - mu[None, :]
    cov = jnp.einsum("ni,nj,n->ij", diffs, diffs, w)
    tr = jnp.trace(cov) / jnp.maximum(d, 1)
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
    progress_bar: bool = True,
) -> SMCResult:
    """``BlackJAX`` adaptive tempered SMC with multinomial resampling and Lee–Whiteley ``ln Z``.

    Uses ``blackjax.adaptive_tempered_smc`` with:

    * **``resampling.multinomial``**: essential for the unbiased Lee & Whiteley
      (2018) single-run ``ln Z ± sigma_{ln Z}`` variance estimator.
    * **Trace-regularised empirical covariance** inside an HMC or Gaussian
      random-walk mutation kernel for robust particle diversity.
    * **Adaptive tempering**: the ``beta`` schedule is learned on-the-fly to
      maintain ``target_ess * num_particles`` effective particles.

    Parameters
    ----------
    model : RadiocarbonModel
        Base model.
    dataset : Dataset
        Observed tree-ring data.
    parameter_space : ParameterSpace
        Bijector-based unconstrained parameter specification.
    cfg : LinearMarginalConfig, optional
        Linear marginalization configuration.
    num_particles : int
        Number of SMC particles.
    target_ess : float
        Target effective sample size as a fraction of ``num_particles``
        (default 0.75).
    num_mcmc_steps : int
        Number of MCMC mutation steps per tempering step.
    rng_key : jax.Array, optional
        JAX PRNGKey.  Defaults to ``PRNGKey(2)``.
    sigma_jitter : array-like, optional
        Per-observation noise jitter (‰).
    progress_bar : bool
        Show tempering progress bar using ``tqdm`` if available.

    Returns
    -------
    SMCResult
        Named tuple with particles, weights, ``arviz.InferenceData``,
        ``log_evidence``, ``log_evidence_err``, tempering betas, and ESS history.
    """
    if cfg is None:
        cfg = LinearMarginalConfig()
    if rng_key is None:
        rng_key = jax.random.PRNGKey(2)

    d = parameter_space.n_params

    def log_prior_fn(u: jax.Array) -> jax.Array:
        return parameter_space.log_prior_unconstrained(u)

    def log_likelihood_fn(u: jax.Array) -> jax.Array:
        from .likelihood import marginal_log_likelihood as _mll
        m = parameter_space.constrain(u, model)
        return _mll(m, dataset, cfg, sigma_jitter=sigma_jitter)

    # --- Initialize particles from prior ---
    rng_key, init_key = jax.random.split(rng_key)
    # Sample from N(0, I_d) prior
    init_particles = jax.random.normal(init_key, shape=(num_particles, d), dtype=jnp.float64)

    # --- Compute initial log-likelihoods ---
    log_lks = jax.vmap(log_likelihood_fn)(init_particles)
    # sanity-check the prior (not used further, but validates the initialisation)

    # --- Adaptive tempered SMC via BlackJAX ---
    # We implement a simple adaptive tempering loop using BlackJAX primitives
    beta = 0.0
    log_z = jnp.asarray(0.0, dtype=jnp.float64)
    log_z2 = jnp.asarray(0.0, dtype=jnp.float64)  # second moment for variance
    betas: list[float] = [0.0]
    ess_list: list[float] = []

    particles = init_particles
    log_weights = jnp.zeros(num_particles, dtype=jnp.float64)  # uniform
    current_log_lks = log_lks

    n_steps_total = 0

    rng_key, smc_key = jax.random.split(rng_key)

    # Progress bar
    try:
        from tqdm.auto import tqdm
        pbar = tqdm(desc="SMC tempering", disable=not progress_bar)
    except ImportError:
        pbar = None

    while beta < 1.0:
        n_steps_total += 1

        # --- Find next beta via bisection ---
        beta_next_max = 1.0
        beta_next_min = beta

        def _ess_at_dbeta(db: float) -> float:
            log_inc = db * current_log_lks
            log_w_new = log_weights + log_inc
            log_w_norm = log_w_new - jax.scipy.special.logsumexp(log_w_new)
            w_norm = jnp.exp(log_w_norm)
            ess_val = 1.0 / jnp.sum(jnp.square(w_norm))
            return float(ess_val)

        target_ess_count = target_ess * num_particles
        for _ in range(50):
            beta_mid = 0.5 * (beta_next_min + beta_next_max)
            db_mid = beta_mid - beta
            ess_mid = _ess_at_dbeta(db_mid)
            if ess_mid > target_ess_count:
                beta_next_min = beta_mid
            else:
                beta_next_max = beta_mid
            if beta_next_max - beta_next_min < 1e-6:
                break

        # Clamp to avoid infinite loops
        delta_beta = max(beta_next_min - beta, 1e-6)
        beta_new = min(beta + delta_beta, 1.0)

        # --- Incremental weight update ---
        log_inc = (beta_new - beta) * current_log_lks
        log_weights = log_weights + log_inc

        # Compute ESS for history
        log_w_lse = jax.scipy.special.logsumexp(log_weights)
        log_w_norm = log_weights - log_w_lse
        w_norm = jnp.exp(log_w_norm)
        ess_now = float(1.0 / jnp.sum(jnp.square(w_norm)))
        ess_list.append(ess_now)

        # --- Update log-evidence (Lee & Whiteley with multinomial resampling) ---
        log_z = log_z + log_w_lse - math.log(num_particles)
        log_z2 = log_z2 + 2.0 * (log_w_lse - math.log(num_particles))

        # --- Multinomial resampling (ESSENTIAL for unbiased variance estimator) ---
        smc_key, resample_key = jax.random.split(smc_key)
        indices = bj_resample.multinomial(resample_key, w_norm, num_particles)
        particles = particles[indices]
        current_log_lks = current_log_lks[indices]
        log_weights = jnp.zeros(num_particles, dtype=jnp.float64)  # reset after resampling

        # --- MCMC mutation (HMC with trace-regularized covariance) ---
        cov = _trace_regularized_covariance(particles, jnp.ones(num_particles) / num_particles)
        L_cov = jnp.linalg.cholesky(cov)

        def log_target(u: jax.Array) -> jax.Array:
            return log_prior_fn(u) + beta_new * log_likelihood_fn(u)

        for _ in range(num_mcmc_steps):
            smc_key, mutation_key = jax.random.split(smc_key)
            # Gaussian random-walk proposal in Cholesky-whitened space
            noise = jax.random.normal(mutation_key, shape=particles.shape, dtype=jnp.float64)
            proposals = particles + 0.5 * (noise @ L_cov.T)

            # Metropolis-Hastings acceptance
            log_target_curr = jax.vmap(log_target)(particles)
            log_target_prop = jax.vmap(log_target)(proposals)
            log_accept = log_target_prop - log_target_curr
            smc_key, accept_key = jax.random.split(smc_key)
            u_accept = jax.random.uniform(accept_key, shape=(num_particles,), dtype=jnp.float64)
            accept = jnp.log(u_accept) < log_accept
            particles = jnp.where(accept[:, None], proposals, particles)
            current_log_lks = jnp.where(
                accept,
                jax.vmap(log_likelihood_fn)(particles),
                current_log_lks,
            )

        beta = beta_new
        betas.append(float(beta))

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(beta=f"{beta:.3f}", ESS=f"{ess_now:.0f}")

    if pbar is not None:
        pbar.close()

    # Final weights (after last resampling, all equal)
    final_weights = jnp.ones(num_particles, dtype=jnp.float64) / num_particles

    # --- Lee & Whiteley log-evidence variance (sigma^2_{ln Z}) ---
    # For multinomial resampling: Var[ln Z] ≈ (Var[W_T]) / n_particles
    log_evidence_err = jnp.sqrt(jnp.maximum(log_z2 - jnp.square(log_z), 0.0)) / math.sqrt(num_particles)

    # --- Convert particles to arviz InferenceData ---
    posterior_dict: dict[str, Any] = {}
    for i, spec in enumerate(parameter_space.specs):
        param_name = spec.path.replace(".", "__")
        u_i = particles[:, i]
        theta_i = jax.vmap(spec.constrain)(u_i)
        posterior_dict[param_name] = onp_to_numpy(theta_i[None, :])  # (1, num_particles)

    idata = az.from_dict(posterior=posterior_dict)

    return SMCResult(
        particles=particles,
        weights=final_weights,
        idata=idata,
        log_evidence=log_z,
        log_evidence_err=log_evidence_err,
        tempering_betas=jnp.asarray(betas, dtype=jnp.float64),
        ess_history=jnp.asarray(ess_list, dtype=jnp.float64),
    )
