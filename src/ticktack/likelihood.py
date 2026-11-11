"""Marginalized and joint log-likelihoods for ``ticktack 2.0`` radiocarbon inference.

Implements the full Bayesian parameter infrastructure:

1. :class:`ParamSpec` and :class:`ParameterSpace`: smooth bijector-based
   mapping between unconstrained ``R^d`` vectors and constrained
   :class:`~ticktack.model.RadiocarbonModel` leaves (both scalar and 1D/ND
   array leaves), with exact log-prior and optional log-abs-det-Jacobian.

2. :class:`LinearMarginalConfig` + :func:`marginal_log_likelihood` /
   :func:`marginal_log_posterior`: analytically marginalize baseline
   ``Delta14C`` offsets, harmonic solar quadrature coefficients, secular linear
   trends, and causal Solin--Särkkä POD ``GPProduction`` modes in closed form
   via :mod:`ticktack.linear`, eliminating v1's 4-point ``mean(y[:4])``
   boundary bias while preserving inter-hemispheric gradients.

3. :func:`joint_log_likelihood` / :func:`joint_log_posterior`: non-Gaussian
   ``LogGPProduction`` log-likelihood (unit-Gaussian prior on Hilbert-space
   mode weights) compatible with :class:`~ticktack.solvers.ExactLinearSolver`,
   :class:`~ticktack.solvers.DiffraxSolver`, and the
   :class:`~ticktack.solvers.ModaxSolver` ``@jax.custom_vjp`` wrapper.

4. :func:`linear_posterior`: closed-form posterior mean and covariance of all
   marginalized linear coefficients, reconstructed global and per-series
   offsets, and exact continuous posterior mean, marginal standard deviation,
   and correlated Gaussian draws of ``Q(t)`` and ``Delta14C(t)`` on a fine grid.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
import zodiax as zdx
from jax.typing import ArrayLike

from .data import Dataset
from .linear import (
    CausalGPProjection,
    LinearMarginal,
    _solve_complex_resolvents,
    harmonic_solar_design,
    hierarchical_offset_design,
    project_causal_gp,
    solin_sarkka_basis,
)
from .model import RadiocarbonModel
from .production import (
    GPProduction,
    HarmonicSolar,
    LinearTrend,
    LogGPProduction,
    ProductionSum,
    ProductionTerm,
    SteadyProduction,
)
from .solvers import ModaxSolver, wrap_jvp_only_scalar_fn

__all__ = [
    "ParamSpec",
    "ParameterSpace",
    "LinearMarginalConfig",
    "LinearPosteriorResult",
    "marginal_log_likelihood",
    "marginal_log_posterior",
    "joint_log_likelihood",
    "joint_log_posterior",
    "linear_posterior",
]

# ---------------------------------------------------------------------------
# 1. ParamSpec: bijector specification for a scalar or ND array leaf
# ---------------------------------------------------------------------------

_BIJECTOR_TYPES = Literal["identity", "exp", "softplus", "sigmoid"]


class ParamSpec(eqx.Module):
    """Specification for a parameter leaf (scalar or array) with a smooth bijector.

    Maps elementwise between unconstrained real values ``u`` and constrained
    physical values ``theta`` via a differentiable bijection:

    * ``"identity"``: ``theta = u`` (unbounded reals).
    * ``"exp"``: ``theta = exp(u)`` (strictly positive quantities).
    * ``"softplus"``: ``theta = log(1 + exp(u)) + 1e-6`` (soft positive).
    * ``"sigmoid"``: ``theta = low + (high - low) * sigmoid(u)`` (bounded).

    The Gaussian prior ``N(prior_mean, prior_sd^2)`` is specified directly in
    unconstrained ``u`` space so that ``int exp(log_prior_unconstrained(u)) du = 1``
    identically (essential for calibrated SMC evidence estimation).
    """

    path: str = eqx.field(static=True)
    bijector: str = eqx.field(static=True)
    _prior_mean_tuple: tuple[float, ...] = eqx.field(static=True)
    _prior_sd_tuple: tuple[float, ...] = eqx.field(static=True)
    _prior_shape: tuple[int, ...] = eqx.field(static=True)
    low: float = eqx.field(static=True)
    high: float = eqx.field(static=True)

    def __init__(
        self,
        path: str,
        bijector: _BIJECTOR_TYPES = "identity",
        prior_mean: ArrayLike = 0.0,
        prior_sd: ArrayLike = 10.0,
        *,
        low: float = 0.0,
        high: float = 1.0,
    ) -> None:
        if bijector not in ("identity", "exp", "softplus", "sigmoid"):
            raise ValueError(
                f"bijector must be 'identity', 'exp', 'softplus', or 'sigmoid'; got {bijector!r}."
            )
        if bijector == "sigmoid" and not (float(high) > float(low)):
            raise ValueError(
                f"sigmoid bijector requires high > low; got low={low}, high={high}."
            )
        pm_np = np.asarray(prior_mean, dtype=np.float64)
        ps_np = np.maximum(np.asarray(prior_sd, dtype=np.float64), 1e-30)
        b_shape = np.broadcast_shapes(pm_np.shape, ps_np.shape)
        pm_b = np.broadcast_to(pm_np, b_shape)
        ps_b = np.broadcast_to(ps_np, b_shape)

        self.path = str(path)
        self.bijector = str(bijector)
        self._prior_shape = tuple(int(s) for s in b_shape)
        self._prior_mean_tuple = tuple(float(x) for x in pm_b.reshape(-1))
        self._prior_sd_tuple = tuple(float(x) for x in ps_b.reshape(-1))
        self.low = float(low)
        self.high = float(high)

    @property
    def prior_mean(self) -> float | jax.Array:
        """Prior mean in unconstrained space (float for scalar specs, Array otherwise)."""
        if self._prior_shape == ():
            return self._prior_mean_tuple[0]
        return jnp.asarray(self._prior_mean_tuple, dtype=jnp.float64).reshape(
            self._prior_shape
        )

    @property
    def prior_sd(self) -> float | jax.Array:
        """Prior standard deviation in unconstrained space."""
        if self._prior_shape == ():
            return self._prior_sd_tuple[0]
        return jnp.asarray(self._prior_sd_tuple, dtype=jnp.float64).reshape(
            self._prior_shape
        )

    def constrain(self, u: ArrayLike) -> jax.Array:
        """Apply bijector: unconstrained ``u`` -> constrained ``theta``."""
        u_arr = jnp.asarray(u, dtype=jnp.float64)
        if self.bijector == "identity":
            return u_arr
        if self.bijector == "exp":
            return jnp.exp(u_arr)
        if self.bijector == "softplus":
            return jax.nn.softplus(u_arr) + 1e-6
        if self.bijector == "sigmoid":
            lo = jnp.asarray(self.low, dtype=jnp.float64)
            hi = jnp.asarray(self.high, dtype=jnp.float64)
            return lo + (hi - lo) * jax.nn.sigmoid(u_arr)
        return u_arr

    def unconstrain(self, theta: ArrayLike) -> jax.Array:
        """Inverse bijector: constrained ``theta`` -> unconstrained ``u``."""
        t_arr = jnp.asarray(theta, dtype=jnp.float64)
        if self.bijector == "identity":
            return t_arr
        if self.bijector == "exp":
            return jnp.log(jnp.maximum(t_arr, 1e-30))
        if self.bijector == "softplus":
            safe_t = jnp.maximum(t_arr - 1e-6, 1e-30)
            return jnp.where(
                safe_t > 20.0,
                safe_t + jnp.log1p(-jnp.exp(-safe_t)),
                jnp.log(jnp.expm1(safe_t)),
            )
        if self.bijector == "sigmoid":
            lo = jnp.asarray(self.low, dtype=jnp.float64)
            hi = jnp.asarray(self.high, dtype=jnp.float64)
            p = jnp.clip((t_arr - lo) / (hi - lo), 1e-12, 1.0 - 1e-12)
            return jnp.log(p) - jnp.log1p(-p)
        return t_arr

    def log_abs_det_jacobian(self, u: ArrayLike) -> jax.Array:
        """Scalar sum of ``log |d theta / d u|`` over all elements of ``u``."""
        u_arr = jnp.asarray(u, dtype=jnp.float64)
        if u_arr.size == 0 or self.bijector == "identity":
            return jnp.asarray(0.0, dtype=jnp.float64)
        if self.bijector == "exp":
            return jnp.sum(u_arr)
        if self.bijector == "softplus":
            return jnp.sum(-jax.nn.softplus(-u_arr))
        if self.bijector == "sigmoid":
            span = math.log(self.high - self.low)
            log_sig = -jax.nn.softplus(-u_arr)
            log_one_minus = -jax.nn.softplus(u_arr)
            return jnp.sum(span + log_sig + log_one_minus)
        return jnp.asarray(0.0, dtype=jnp.float64)

    def log_prior_unconstrained(self, u: ArrayLike) -> jax.Array:
        """Normalized Gaussian log-prior density in unconstrained space."""
        u_arr = jnp.asarray(u, dtype=jnp.float64)
        if u_arr.size == 0:
            return jnp.asarray(0.0, dtype=jnp.float64)
        pm = jnp.broadcast_to(jnp.asarray(self.prior_mean, dtype=jnp.float64), u_arr.shape)
        ps = jnp.broadcast_to(jnp.asarray(self.prior_sd, dtype=jnp.float64), u_arr.shape)
        z = (u_arr - pm) / ps
        return jnp.sum(-0.5 * jnp.square(z) - jnp.log(ps) - 0.5 * math.log(2.0 * math.pi))


# ---------------------------------------------------------------------------
# 2. ParameterSpace: multi-parameter unconstrained <-> constrained mapping
# ---------------------------------------------------------------------------


class ParameterSpace(zdx.Base):
    """Immutable bijector-based map between unconstrained ``R^d`` and ``RadiocarbonModel``.

    Supports both scalar leaves (e.g. ``"production.spike.t_start"``) and 1D/ND
    array leaves (e.g. ``"cbm.log_k_trop_to_sinks"``, ``"cbm.log_rates_free"``,
    ``"production.log_gp.coefficients"``), including zero-sized array leaves
    (such as ``EffectiveCBM(K=2).log_k_trop_to_sinks``).
    """

    reference_model: RadiocarbonModel
    specs: tuple[ParamSpec, ...] = eqx.field(static=True)
    leaf_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    spec_slices: tuple[tuple[int, int], ...] = eqx.field(static=True)
    _total_dim: int = eqx.field(static=True)
    _prior_means_tuple: tuple[float, ...] = eqx.field(static=True)
    _prior_sds_tuple: tuple[float, ...] = eqx.field(static=True)

    def __init__(
        self,
        reference_model: RadiocarbonModel,
        specs: list[ParamSpec] | tuple[ParamSpec, ...],
    ) -> None:
        self.reference_model = reference_model
        self.specs = tuple(specs)

        shapes: list[tuple[int, ...]] = []
        slices: list[tuple[int, int]] = []
        means_flat: list[float] = []
        sds_flat: list[float] = []
        cursor = 0

        for spec in self.specs:
            leaf = jnp.asarray(reference_model.get(spec.path), dtype=jnp.float64)
            shape = tuple(int(s) for s in leaf.shape)
            size = int(math.prod(shape))
            shapes.append(shape)
            slices.append((cursor, cursor + size))
            cursor += size

            if size > 0:
                pm_spec = np.asarray(spec._prior_mean_tuple, dtype=np.float64).reshape(
                    spec._prior_shape
                )
                ps_spec = np.asarray(spec._prior_sd_tuple, dtype=np.float64).reshape(
                    spec._prior_shape
                )
                pm_full = np.broadcast_to(pm_spec, shape).reshape(-1)
                ps_full = np.broadcast_to(ps_spec, shape).reshape(-1)
                means_flat.extend(float(x) for x in pm_full)
                sds_flat.extend(float(x) for x in ps_full)

        self.leaf_shapes = tuple(shapes)
        self.spec_slices = tuple(slices)
        self._total_dim = cursor
        self._prior_means_tuple = tuple(means_flat)
        self._prior_sds_tuple = tuple(sds_flat)

    @property
    def n_params(self) -> int:
        """Total flattened dimensionality ``d`` of the unconstrained parameter space."""
        return self._total_dim

    @property
    def prior_means(self) -> jax.Array:
        """Flattened unconstrained prior mean vector ``(d,)``."""
        return jnp.asarray(self._prior_means_tuple, dtype=jnp.float64)

    @property
    def prior_sds(self) -> jax.Array:
        """Flattened unconstrained prior standard deviation vector ``(d,)``."""
        return jnp.asarray(self._prior_sds_tuple, dtype=jnp.float64)

    def constrain(
        self,
        u_vec: ArrayLike,
        base_model: RadiocarbonModel | None = None,
    ) -> RadiocarbonModel:
        """Map unconstrained vector ``u_vec`` of shape ``(d,)`` to a ``RadiocarbonModel``."""
        u = jnp.asarray(u_vec, dtype=jnp.float64).reshape(-1)
        if u.shape[0] != self._total_dim:
            raise ValueError(
                f"u_vec length {u.shape[0]} does not match n_params={self._total_dim}."
            )
        model = self.reference_model if base_model is None else base_model
        for spec, shape, (start, stop) in zip(
            self.specs, self.leaf_shapes, self.spec_slices
        ):
            if stop == start:
                continue
            u_leaf = u[start:stop].reshape(shape)
            theta_leaf = spec.constrain(u_leaf)
            model = model.set(spec.path, theta_leaf)
        return model

    def unconstrain(self, model: RadiocarbonModel) -> jax.Array:
        """Map a ``RadiocarbonModel`` to its flattened unconstrained vector ``(d,)``."""
        parts: list[jax.Array] = []
        for spec, (start, stop) in zip(self.specs, self.spec_slices):
            if stop == start:
                continue
            val = jnp.asarray(model.get(spec.path), dtype=jnp.float64)
            parts.append(spec.unconstrain(val).reshape(-1))
        if not parts:
            return jnp.zeros((0,), dtype=jnp.float64)
        return jnp.concatenate(parts, axis=0)

    def apply_unconstrained(
        self,
        u_vec: ArrayLike,
        fn: Any,
        base_model: RadiocarbonModel | None = None,
    ) -> Any:
        """Evaluate ``fn(constrained_model)`` from unconstrained vector ``u_vec``."""
        return fn(self.constrain(u_vec, base_model))

    def log_prior_unconstrained(
        self,
        u_vec: ArrayLike,
        *,
        include_jacobian: bool = False,
    ) -> jax.Array:
        """Normalized Gaussian log-prior density on ``u_vec`` (without Jacobian by default)."""
        u = jnp.asarray(u_vec, dtype=jnp.float64).reshape(-1)
        if u.shape[0] != self._total_dim:
            raise ValueError(
                f"u_vec length {u.shape[0]} does not match n_params={self._total_dim}."
            )
        if self._total_dim == 0:
            return jnp.asarray(0.0, dtype=jnp.float64)

        pm = self.prior_means
        ps = self.prior_sds
        z = (u - pm) / ps
        log_p = jnp.sum(
            -0.5 * jnp.square(z) - jnp.log(ps) - 0.5 * math.log(2.0 * math.pi)
        )
        if include_jacobian:
            for spec, shape, (start, stop) in zip(
                self.specs, self.leaf_shapes, self.spec_slices
            ):
                if stop > start and spec.bijector != "identity":
                    log_p = log_p + spec.log_abs_det_jacobian(
                        u[start:stop].reshape(shape)
                    )
        return log_p


# ---------------------------------------------------------------------------
# 3. LinearMarginalConfig: configuration for analytical linear marginalization
# ---------------------------------------------------------------------------


class LinearMarginalConfig(eqx.Module):
    """Configuration for analytical Gaussian marginalization of linear parameters."""

    offset_mode: str = eqx.field(static=True)
    global_prior_mean: float = eqx.field(static=True)
    global_prior_sd: float = eqx.field(static=True)
    lab_prior_sd: float = eqx.field(static=True)
    solar_prior_sd: float = eqx.field(static=True)
    trend_prior_sd: float = eqx.field(static=True)
    marginalise_solar: bool = eqx.field(static=True)
    marginalise_trend: bool = eqx.field(static=True)
    marginalise_gp: bool = eqx.field(static=True)
    n_gp_modes: int | None = eqx.field(static=True)
    include_gp_initial_cov: bool = eqx.field(static=True)
    whiten_method: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        offset_mode: Literal[
            "hierarchical", "per_series", "shared", "none"
        ] = "hierarchical",
        global_prior_mean: float = 0.0,
        global_prior_sd: float = 200.0,
        lab_prior_sd: float = 1.5,
        solar_prior_sd: float = 1.0,
        trend_prior_sd: float = 5.0,
        marginalise_solar: bool | None = None,
        marginalise_trend: bool | None = None,
        marginalise_gp: bool | None = None,
        marginalize_solar: bool | None = None,
        marginalize_trend: bool | None = None,
        marginalize_gp: bool | None = None,
        n_gp_modes: int | None = None,
        include_gp_initial_cov: bool = True,
        whiten_method: Literal["cholesky", "qr", "rank_one"] = "cholesky",
    ) -> None:
        valid_offset_modes = ("hierarchical", "per_series", "shared", "none")
        if offset_mode not in valid_offset_modes:
            raise ValueError(
                f"offset_mode must be one of {valid_offset_modes}; got {offset_mode!r}."
            )
        eff_solar = (
            marginalize_solar
            if marginalize_solar is not None
            else (True if marginalise_solar is None else marginalise_solar)
        )
        eff_trend = (
            marginalize_trend
            if marginalize_trend is not None
            else (True if marginalise_trend is None else marginalise_trend)
        )
        eff_gp = (
            marginalize_gp
            if marginalize_gp is not None
            else (True if marginalise_gp is None else marginalise_gp)
        )
        self.offset_mode = str(offset_mode)
        self.global_prior_mean = float(global_prior_mean)
        self.global_prior_sd = float(global_prior_sd)
        self.lab_prior_sd = float(lab_prior_sd)
        self.solar_prior_sd = float(solar_prior_sd)
        self.trend_prior_sd = float(trend_prior_sd)
        self.marginalise_solar = bool(eff_solar)
        self.marginalise_trend = bool(eff_trend)
        self.marginalise_gp = bool(eff_gp)
        self.n_gp_modes = None if n_gp_modes is None else int(n_gp_modes)
        self.include_gp_initial_cov = bool(include_gp_initial_cov)
        self.whiten_method = str(whiten_method)

    @property
    def marginalize_solar(self) -> bool:
        return self.marginalise_solar

    @property
    def marginalize_trend(self) -> bool:
        return self.marginalise_trend

    @property
    def marginalize_gp(self) -> bool:
        return self.marginalise_gp


# ---------------------------------------------------------------------------
# 4. Internal helpers: conditional term partitioning, padded windows, designs
# ---------------------------------------------------------------------------


def _has_loggp(production: ProductionTerm) -> bool:
    """Return True if ``production`` is or contains a :class:`LogGPProduction`."""
    if isinstance(production, LogGPProduction):
        return True
    if isinstance(production, ProductionSum):
        return any(_has_loggp(p) for p in production.parts)
    return False


def _partition_terms_for_cfg(
    production: ProductionTerm,
    cfg: LinearMarginalConfig,
) -> tuple[
    list[ProductionTerm],
    list[HarmonicSolar],
    list[LinearTrend],
    list[GPProduction],
]:
    """Partition production components into unmarginalized vs marginalized lists.

    Crucially, terms whose marginalization flag in ``cfg`` is ``False`` remain
    in ``unmarginalized`` so their deterministic trajectories are never dropped!
    """
    parts = (
        list(production.parts)
        if isinstance(production, ProductionSum)
        else [production]
    )
    unmarginalized: list[ProductionTerm] = []
    solar_terms: list[HarmonicSolar] = []
    trend_terms: list[LinearTrend] = []
    gp_terms: list[GPProduction] = []

    for term in parts:
        if isinstance(term, HarmonicSolar) and cfg.marginalise_solar:
            solar_terms.append(term)
        elif isinstance(term, LinearTrend) and cfg.marginalise_trend:
            trend_terms.append(term)
        elif isinstance(term, GPProduction) and cfg.marginalise_gp:
            gp_terms.append(term)
        else:
            unmarginalized.append(term)

    return unmarginalized, solar_terms, trend_terms, gp_terms


def _build_base_production(
    unmarginalized: list[ProductionTerm],
    q_std: jax.Array,
) -> ProductionTerm:
    """Combine unmarginalized terms into a single ``ProductionTerm`` (defaulting to ``SteadyProduction(q_std)``)."""
    if not unmarginalized:
        return SteadyProduction(q_std)
    if len(unmarginalized) == 1:
        return unmarginalized[0]
    return ProductionSum({f"t{i}": term for i, term in enumerate(unmarginalized)})


def _concat_padded_windows(
    dataset: Dataset,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Concatenate ``(sub_t_starts, sub_t_ends, sub_weights)`` across all series with zero-weight padding."""
    m_max = max(int(s.sub_t_starts.shape[1]) for s in dataset.series)
    ts_parts: list[jax.Array] = []
    te_parts: list[jax.Array] = []
    w_parts: list[jax.Array] = []

    for s in dataset.series:
        n_s, m_s = s.sub_t_starts.shape
        w_s = jnp.broadcast_to(
            jnp.asarray(s.seasonal_weights.weights, dtype=jnp.float64),
            (n_s, m_s),
        )
        if m_s < m_max:
            pad_k = m_max - m_s
            ts_pad = jnp.repeat(s.sub_t_starts[:, -1:], pad_k, axis=1)
            te_pad = jnp.repeat(s.sub_t_ends[:, -1:], pad_k, axis=1)
            w_pad = jnp.zeros((n_s, pad_k), dtype=jnp.float64)
            ts_s = jnp.concatenate([s.sub_t_starts, ts_pad], axis=1)
            te_s = jnp.concatenate([s.sub_t_ends, te_pad], axis=1)
            w_s = jnp.concatenate([w_s, w_pad], axis=1)
        else:
            ts_s = s.sub_t_starts
            te_s = s.sub_t_ends
        ts_parts.append(ts_s)
        te_parts.append(te_s)
        w_parts.append(w_s)

    return (
        jnp.concatenate(ts_parts, axis=0),
        jnp.concatenate(te_parts, axis=0),
        jnp.concatenate(w_parts, axis=0),
    )


def _project_single_gp_dataset(
    model: RadiocarbonModel,
    dataset: Dataset,
    gp: GPProduction,
    cfg: LinearMarginalConfig,
    A: jax.Array,
    b: jax.Array,
    box_indices_flat: jax.Array,
    all_obs_scale: jax.Array,
    all_t_starts: jax.Array,
    all_t_ends: jax.Array,
    all_weights: jax.Array,
) -> CausalGPProjection:
    """Project one ``GPProduction`` across the entire ``Dataset`` via ``project_causal_gp``."""
    r_modes = gp.n_modes if cfg.n_gp_modes is None else cfg.n_gp_modes
    a_arg = gp.domain[0] if gp.domain is not None else None
    b_arg = gp.domain[1] if gp.domain is not None else None
    return project_causal_gp(
        A=A,
        b=b,
        t_starts=all_t_starts,
        t_ends=all_t_ends,
        sub_weights=all_weights,
        a=a_arg,
        b_bound=b_arg,
        pad=gp.pad_years,
        n_modes=r_modes,
        kernel=gp.kernel,
        alpha_unit=model.alpha_unit,
        box_indices=box_indices_flat,
        obs_scale=all_obs_scale,
        include_initial_cov=cfg.include_gp_initial_cov,
    )


def _build_marginal_design_with_projections(
    model: RadiocarbonModel,
    dataset: Dataset,
    cfg: LinearMarginalConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, list[CausalGPProjection]]:
    """Build ``(X, prior_mean, prior_sd, gp_projections)`` for all marginalized linear terms."""
    cbm = model.cbm
    A = cbm.matrix_A()
    b = cbm.production_vector()

    n_tot = dataset.n_obs
    n_series = dataset.n_series
    series_idx = dataset.series_indices

    y_ss = cbm.steady_state_c14(q_std=model.q_std, alpha_unit=model.alpha_unit)
    trop_idx = jnp.asarray(cbm.topology.trop_box_indices, dtype=jnp.int32)
    y_trop = jnp.sum(y_ss[trop_idx])
    m_rel = cbm.relative_reservoir_masses()
    r_ss_ratio = (y_ss / y_trop) / m_rel

    box_indices_flat = jnp.array(
        [
            model._box_index_for_series(s)
            for s in dataset.series
            for _ in range(s.n_obs)
        ],
        dtype=jnp.int32,
    )
    obs_activity_ratios = r_ss_ratio[box_indices_flat]

    design_cols: list[jax.Array] = []
    pmean_parts: list[jax.Array] = []
    psd_parts: list[jax.Array] = []
    gp_projs: list[CausalGPProjection] = []

    # 1. Baseline offset columns
    if cfg.offset_mode == "hierarchical":
        H_off, mu_off, sd_off = hierarchical_offset_design(
            series_idx,
            n_series=n_series,
            box_activity_ratios=obs_activity_ratios,
            global_prior_mean=cfg.global_prior_mean,
            global_prior_sd=cfg.global_prior_sd,
            lab_prior_sd=cfg.lab_prior_sd,
        )
        design_cols.append(H_off)
        pmean_parts.append(mu_off)
        psd_parts.append(sd_off)
    elif cfg.offset_mode == "per_series":
        H_off = dataset.lab_offset_design_matrix()
        mu_off = jnp.full((n_series,), cfg.global_prior_mean, dtype=jnp.float64)
        sd_off = jnp.full((n_series,), cfg.global_prior_sd, dtype=jnp.float64)
        design_cols.append(H_off)
        pmean_parts.append(mu_off)
        psd_parts.append(sd_off)
    elif cfg.offset_mode == "shared":
        H_off = obs_activity_ratios[:, None]
        mu_off = jnp.array([cfg.global_prior_mean], dtype=jnp.float64)
        sd_off = jnp.array([cfg.global_prior_sd], dtype=jnp.float64)
        design_cols.append(H_off)
        pmean_parts.append(mu_off)
        psd_parts.append(sd_off)

    _, solar_terms, trend_terms, gp_terms = _partition_terms_for_cfg(
        model.production, cfg
    )

    if (solar_terms or trend_terms or gp_terms) and n_tot > 0:
        all_ts, all_te, all_w = _concat_padded_windows(dataset)
        all_obs_scale = 1000.0 / (y_trop * m_rel[box_indices_flat])
    else:
        all_ts = all_te = all_w = all_obs_scale = jnp.zeros((0,), dtype=jnp.float64)

    # 2. Marginalized HarmonicSolar quadrature columns (single dataset-wide call)
    for solar in solar_terms:
        H_solar = harmonic_solar_design(
            A,
            b,
            solar.omegas,
            all_ts,
            all_te,
            all_w,
            alpha_unit=model.alpha_unit,
            t_ref=solar.t_ref,
            box_indices=box_indices_flat,
            obs_scale=all_obs_scale,
        )
        n_harm_cols = H_solar.shape[1]
        design_cols.append(H_solar)
        pmean_parts.append(jnp.zeros((n_harm_cols,), dtype=jnp.float64))
        psd_parts.append(
            jnp.full((n_harm_cols,), cfg.solar_prior_sd, dtype=jnp.float64)
        )

    # 3. Marginalized LinearTrend exact LTI ramp columns
    for trend in trend_terms:
        unit_trend = LinearTrend(
            slope=1.0,
            t_ref=trend.t_ref,
            switched_on=trend.switched_on,
        )
        if not isinstance(model.solver, ModaxSolver) and n_tot > 0:
            dy_all = model.solver.solve_delta_binned(
                cbm,
                unit_trend,
                all_ts,
                all_te,
                sub_weights=all_w,
                q_std=model.q_std,
                alpha_unit=model.alpha_unit,
            )
            H_trend = (
                dy_all[jnp.arange(n_tot), box_indices_flat] * all_obs_scale
            )[:, None]
        else:
            trend_col_parts: list[jax.Array] = []
            for series in dataset.series:
                box_idx = model._box_index_for_series(series)
                dy_s = model.solver.solve_delta_binned(
                    cbm,
                    unit_trend,
                    series.sub_t_starts,
                    series.sub_t_ends,
                    sub_weights=series.seasonal_weights.weights,
                    q_std=model.q_std,
                    alpha_unit=model.alpha_unit,
                )
                obs_scale_s = 1000.0 / (y_trop * m_rel[box_idx])
                trend_col_parts.append((dy_s[:, box_idx] * obs_scale_s)[:, None])
            H_trend = jnp.concatenate(trend_col_parts, axis=0)
        design_cols.append(H_trend)
        pmean_parts.append(jnp.zeros((1,), dtype=jnp.float64))
        psd_parts.append(jnp.array([cfg.trend_prior_sd], dtype=jnp.float64))

    # 4. Marginalized GPProduction causal Solin-Särkkä columns (dataset-wide!)
    if gp_terms:
        for gp in gp_terms:
            proj = _project_single_gp_dataset(
                model,
                dataset,
                gp,
                cfg,
                A,
                b,
                box_indices_flat,
                all_obs_scale,
                all_ts,
                all_te,
                all_w,
            )
            gp_projs.append(proj)
            assert proj.whitened_design is not None
            H_gp = proj.whitened_design
            r_total = H_gp.shape[1]
            design_cols.append(H_gp)
            pmean_parts.append(jnp.zeros((r_total,), dtype=jnp.float64))
            psd_parts.append(jnp.ones((r_total,), dtype=jnp.float64))

    if design_cols:
        X = jnp.concatenate(design_cols, axis=1)
        pm = jnp.concatenate(pmean_parts)
        ps = jnp.concatenate(psd_parts)
    else:
        X = jnp.zeros((n_tot, 0), dtype=jnp.float64)
        pm = jnp.zeros((0,), dtype=jnp.float64)
        ps = jnp.zeros((0,), dtype=jnp.float64)

    return X, pm, ps, gp_projs


def _build_marginal_design(
    model: RadiocarbonModel,
    dataset: Dataset,
    cfg: LinearMarginalConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build ``(X, prior_mean, prior_sd)`` for all marginalized linear terms."""
    X, pm, ps, _ = _build_marginal_design_with_projections(model, dataset, cfg)
    return X, pm, ps


def _nonlinear_residual(
    model: RadiocarbonModel,
    dataset: Dataset,
    cfg: LinearMarginalConfig | None = None,
) -> jax.Array:
    """Compute ``d - m_0`` subtracting steady-state NH-SH gradient and all unmarginalized terms."""
    if dataset.n_obs == 0:
        return jnp.zeros((0,), dtype=jnp.float64)
    cfg_eff = LinearMarginalConfig() if cfg is None else cfg
    unmarginalized, _, _, _ = _partition_terms_for_cfg(model.production, cfg_eff)
    base_prod = _build_base_production(unmarginalized, model.q_std)

    if not isinstance(model.solver, ModaxSolver):
        all_ts, all_te, all_w = _concat_padded_windows(dataset)
        box_indices_flat = jnp.array(
            [
                model._box_index_for_series(s)
                for s in dataset.series
                for _ in range(s.n_obs)
            ],
            dtype=jnp.int32,
        )
        pred_flat = model.solver.predict_d14c(
            model.cbm,
            base_prod,
            all_ts,
            all_te,
            sub_weights=all_w,
            box_indices=box_indices_flat,
            q_std=model.q_std,
            alpha_unit=model.alpha_unit,
        )
        return dataset.d14c_flat - pred_flat

    per_series_preds: list[jax.Array] = []
    for series in dataset.series:
        box_idx = model._box_index_for_series(series)
        pred_s = model.solver.predict_d14c(
            model.cbm,
            base_prod,
            series.sub_t_starts,
            series.sub_t_ends,
            sub_weights=series.seasonal_weights.weights,
            box_indices=box_idx,
            q_std=model.q_std,
            alpha_unit=model.alpha_unit,
        )
        per_series_preds.append(pred_s)

    pred_flat = jnp.concatenate(per_series_preds, axis=0)
    return dataset.d14c_flat - pred_flat


def _marginal_design_is_static(
    base_model: RadiocarbonModel,
    parameter_space: ParameterSpace,
    cfg: LinearMarginalConfig,
) -> bool:
    """Return True if ``(X, prior_mean, prior_sd)`` is invariant across ``parameter_space``."""
    if isinstance(base_model.solver, ModaxSolver):
        return False

    joint_mode = _has_loggp(base_model.production)
    for spec in parameter_space.specs:
        p = spec.path
        if p in ("cbm", "q_std", "alpha_unit") or p.startswith("cbm."):
            return False

    if joint_mode:
        # In joint mode only baseline offsets are marginalized, which depend solely on CBM/q_std/alpha_unit.
        return True

    prod = base_model.production
    if isinstance(prod, ProductionSum):
        marg_names = {
            name
            for name, part in zip(prod.names, prod.parts)
            if (isinstance(part, HarmonicSolar) and cfg.marginalise_solar)
            or (isinstance(part, LinearTrend) and cfg.marginalise_trend)
            or (isinstance(part, GPProduction) and cfg.marginalise_gp)
        }
        for spec in parameter_space.specs:
            p = spec.path
            if p == "production":
                return False
            for mn in marg_names:
                if p == f"production.{mn}" or p.startswith(f"production.{mn}."):
                    return False
        return True

    is_marg_single = (
        (isinstance(prod, HarmonicSolar) and cfg.marginalise_solar)
        or (isinstance(prod, LinearTrend) and cfg.marginalise_trend)
        or (isinstance(prod, GPProduction) and cfg.marginalise_gp)
    )
    if is_marg_single:
        for spec in parameter_space.specs:
            if spec.path == "production" or spec.path.startswith("production."):
                return False
    return True


def _prefactor_static_whiten(
    lm: LinearMarginal,
    sigma_1d: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Precompute ``(shift_const, inv_sigma, w_mat, p_right, const_norm)`` once."""
    n_obs = lm.design.shape[0]
    shift_const = lm.design @ lm.prior_mean
    inv_sigma = 1.0 / sigma_1d
    u_mat = (lm.design @ lm.prior_root) * inv_sigma[:, None]
    k = u_mat.shape[1]
    half_logdet_d = jnp.sum(jnp.log(sigma_1d))

    if k == 0:
        w_mat = jnp.zeros((n_obs, 0), dtype=jnp.float64)
        p_right = jnp.zeros((0, n_obs), dtype=jnp.float64)
        const_norm = -half_logdet_d - 0.5 * n_obs * math.log(2.0 * math.pi)
        return shift_const, inv_sigma, w_mat, p_right, const_norm

    eye = jnp.eye(k, dtype=jnp.float64)
    if lm.method == "qr":
        aug = jnp.concatenate([u_mat, eye], axis=0)
        _, r_upper = jnp.linalg.qr(aug, mode="reduced")
        signs = jnp.where(jnp.diag(r_upper) >= 0.0, 1.0, -1.0)
        chol = (signs[:, None] * r_upper).T
        half_logdet_u = jnp.sum(jnp.log(jnp.abs(jnp.diag(r_upper))))
    else:
        gram = eye + u_mat.T @ u_mat
        chol = jnp.linalg.cholesky(0.5 * (gram + gram.T))
        half_logdet_u = jnp.sum(jnp.log(jnp.diag(chol)))

    l_inv_t = jsl.solve_triangular(chol, eye, lower=True).T
    w_mat = u_mat @ l_inv_t
    p_right = jsl.solve_triangular(eye + l_inv_t, w_mat.T, lower=False)
    const_norm = (
        -half_logdet_d - half_logdet_u - 0.5 * n_obs * math.log(2.0 * math.pi)
    )
    return shift_const, inv_sigma, w_mat, p_right, const_norm


def _make_fast_unconstrained_loglike(
    base_model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    cfg: LinearMarginalConfig | None = None,
    sigma_jitter: ArrayLike | None = None,
) -> Callable[[jax.Array], jax.Array]:
    """Build a fast ``u_vec -> log_likelihood`` closure, pre-factoring ``LinearMarginal`` when static."""
    cfg_eff = LinearMarginalConfig() if cfg is None else cfg
    joint_mode = _has_loggp(base_model.production) or isinstance(
        base_model.solver, ModaxSolver
    )

    if (
        dataset.n_obs > 0
        and not isinstance(sigma_jitter, jax.core.Tracer)
        and _marginal_design_is_static(base_model, parameter_space, cfg_eff)
    ):
        sigma = jnp.asarray(dataset.sigma_flat, dtype=jnp.float64)
        if sigma_jitter is not None:
            sigma = jnp.sqrt(
                jnp.square(sigma)
                + jnp.square(jnp.asarray(sigma_jitter, dtype=jnp.float64))
            )

        if joint_mode:
            if cfg_eff.offset_mode == "none":
                inv_sig = 1.0 / sigma
                const_n = -jnp.sum(jnp.log(sigma)) - 0.5 * dataset.n_obs * math.log(
                    2.0 * math.pi
                )

                def _fast_joint_no_off(u_vec: jax.Array) -> jax.Array:
                    m_eval = parameter_space.constrain(u_vec, base_model)
                    pred = m_eval.predict_d14c(dataset, offsets=None, concat=True)
                    x_arr = (dataset.d14c_flat - pred) * inv_sig
                    return -0.5 * jnp.dot(x_arr, x_arr) + const_n

                return _fast_joint_no_off

            off_cfg = _offset_only_cfg(cfg_eff)
            X_off, pm_off, ps_off = _build_marginal_design(
                base_model, dataset, off_cfg
            )
            lm_off = LinearMarginal(
                X_off,
                prior_mean=pm_off,
                prior_sd=ps_off,
                method=off_cfg.whiten_method,
            )
            shift_c, inv_sig, w_mat, p_right, const_n = _prefactor_static_whiten(
                lm_off, sigma
            )

            def _fast_joint_off(u_vec: jax.Array) -> jax.Array:
                m_eval = parameter_space.constrain(u_vec, base_model)
                pred = m_eval.predict_d14c(dataset, offsets=None, concat=True)
                x_arr = (dataset.d14c_flat - pred - shift_c) * inv_sig
                u_out = x_arr - w_mat @ (p_right @ x_arr)
                return -0.5 * jnp.dot(u_out, u_out) + const_n

            return _fast_joint_off

        X_m, pm_m, ps_m = _build_marginal_design(base_model, dataset, cfg_eff)
        lm_m = LinearMarginal(
            X_m,
            prior_mean=pm_m,
            prior_sd=ps_m,
            method=cfg_eff.whiten_method,
        )
        shift_c, inv_sig, w_mat, p_right, const_n = _prefactor_static_whiten(
            lm_m, sigma
        )

        def _fast_marginal(u_vec: jax.Array) -> jax.Array:
            m_eval = parameter_space.constrain(u_vec, base_model)
            resid = _nonlinear_residual(m_eval, dataset, cfg_eff)
            x_arr = (resid - shift_c) * inv_sig
            u_out = x_arr - w_mat @ (p_right @ x_arr)
            return -0.5 * jnp.dot(u_out, u_out) + const_n

        return _fast_marginal

    def _dynamic_fallback(u_vec: jax.Array) -> jax.Array:
        m_eval = parameter_space.constrain(u_vec, base_model)
        if joint_mode:
            return joint_log_likelihood(
                m_eval, dataset, cfg=cfg_eff, sigma_jitter=sigma_jitter
            )
        return marginal_log_likelihood(
            m_eval, dataset, cfg_eff, sigma_jitter=sigma_jitter
        )

    return _dynamic_fallback


# ---------------------------------------------------------------------------
# 5. Marginal log-likelihood and log-posterior
# ---------------------------------------------------------------------------


def marginal_log_likelihood(
    model: RadiocarbonModel,
    dataset: Dataset,
    cfg: LinearMarginalConfig | None = None,
    *,
    sigma_jitter: ArrayLike | None = None,
) -> jax.Array:
    """Exact marginal Gaussian log-likelihood (analytically marginalized linear parameters)."""
    if cfg is None:
        cfg = LinearMarginalConfig()

    sigma = dataset.sigma_flat
    if sigma_jitter is not None:
        sigma = jnp.sqrt(
            jnp.square(sigma)
            + jnp.square(jnp.asarray(sigma_jitter, dtype=jnp.float64))
        )

    resid = _nonlinear_residual(model, dataset, cfg)
    X, prior_mean, prior_sd = _build_marginal_design(model, dataset, cfg)

    lm = LinearMarginal(
        X,
        prior_mean=prior_mean,
        prior_sd=prior_sd,
        method=cfg.whiten_method,
    )
    return lm.loglike(resid, sigma)


def marginal_log_posterior(
    model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    u_vec: ArrayLike,
    cfg: LinearMarginalConfig | None = None,
    *,
    sigma_jitter: ArrayLike | None = None,
) -> jax.Array:
    """Marginal log-posterior (marginalized log-likelihood + unconstrained log-prior)."""
    constrained_model = parameter_space.constrain(u_vec, model)
    log_lk = marginal_log_likelihood(
        constrained_model, dataset, cfg, sigma_jitter=sigma_jitter
    )
    log_pr = parameter_space.log_prior_unconstrained(u_vec)
    return log_lk + log_pr


# ---------------------------------------------------------------------------
# 6. Joint log-likelihood (for LogGPProduction and ModaxSolver)
# ---------------------------------------------------------------------------


def _offset_only_cfg(cfg: LinearMarginalConfig) -> LinearMarginalConfig:
    return LinearMarginalConfig(
        offset_mode=cfg.offset_mode,
        global_prior_mean=cfg.global_prior_mean,
        global_prior_sd=cfg.global_prior_sd,
        lab_prior_sd=cfg.lab_prior_sd,
        solar_prior_sd=cfg.solar_prior_sd,
        trend_prior_sd=cfg.trend_prior_sd,
        marginalise_solar=False,
        marginalise_trend=False,
        marginalise_gp=False,
        whiten_method=cfg.whiten_method,
    )


def _eval_joint_loglike_core(
    model: RadiocarbonModel,
    dataset: Dataset,
    cfg: LinearMarginalConfig | None,
    sigma_jitter: ArrayLike | None,
) -> jax.Array:
    if cfg is not None and cfg.offset_mode != "none":
        return marginal_log_likelihood(
            model,
            dataset,
            _offset_only_cfg(cfg),
            sigma_jitter=sigma_jitter,
        )

    sigma = jnp.asarray(dataset.sigma_flat, dtype=jnp.float64)
    if sigma_jitter is not None:
        sigma = jnp.sqrt(
            jnp.square(sigma)
            + jnp.square(jnp.asarray(sigma_jitter, dtype=jnp.float64))
        )
    pred_flat = model.predict_d14c(dataset, offsets=None, concat=True)
    resid = dataset.d14c_flat - pred_flat
    return (
        -0.5 * jnp.sum(jnp.square(resid / sigma))
        - jnp.sum(jnp.log(sigma))
        - 0.5 * dataset.n_obs * math.log(2.0 * math.pi)
    )


def joint_log_likelihood(
    model: RadiocarbonModel,
    dataset: Dataset,
    cfg: LinearMarginalConfig | None = None,
    *,
    sigma_jitter: ArrayLike | None = None,
) -> jax.Array:
    """Joint Gaussian log-likelihood (for ``LogGPProduction``, explicit realizations, and ``ModaxSolver``).

    Evaluates all production terms explicitly (no GP/solar/trend marginalization),
    while analytically marginalizing baseline offsets whenever ``cfg`` is supplied
    with ``cfg.offset_mode != "none"``. Automatically wraps ``ModaxSolver`` in
    :func:`~ticktack.solvers.wrap_jvp_only_scalar_fn` for reverse-mode autodiff.
    """
    if isinstance(model.solver, ModaxSolver):
        wrapped = wrap_jvp_only_scalar_fn(
            lambda m: _eval_joint_loglike_core(m, dataset, cfg, sigma_jitter)
        )
        return wrapped(model)
    return _eval_joint_loglike_core(model, dataset, cfg, sigma_jitter)


def joint_log_posterior(
    model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    u_vec: ArrayLike,
    cfg: LinearMarginalConfig | None = None,
    *,
    sigma_jitter: ArrayLike | None = None,
) -> jax.Array:
    """Joint log-posterior (joint log-likelihood + unconstrained log-prior)."""
    constrained_model = parameter_space.constrain(u_vec, model)
    log_lk = joint_log_likelihood(
        constrained_model, dataset, cfg=cfg, sigma_jitter=sigma_jitter
    )
    log_pr = parameter_space.log_prior_unconstrained(u_vec)
    return log_lk + log_pr


# ---------------------------------------------------------------------------
# 7. linear_posterior: closed-form posterior of marginalized linear params
# ---------------------------------------------------------------------------


class LinearPosteriorResult(NamedTuple):
    """Results from :func:`linear_posterior`."""

    coeff_mean: jax.Array
    """Posterior mean of all marginalized linear coefficients, shape ``(k_total,)``."""

    coeff_cov: jax.Array
    """Posterior covariance of coefficients, shape ``(k_total, k_total)``."""

    offset_global_mean: jax.Array
    """Reconstructed global Holocene ambient offset ``Delta_global`` posterior mean (‰)."""

    offset_lab_means: jax.Array
    """Reconstructed per-series inter-laboratory offset posterior means (‰), shape ``(S,)``."""

    baseline_d14c: jax.Array
    """Epoch-local baseline ``Delta14C_0`` estimate (‰) = ``offset_global_mean``."""

    q_fine_mean: jax.Array | None
    """Posterior mean of continuous ``Q(t)`` on ``t_fine`` (``atoms cm^-2 s^-1``), or ``None``."""

    q_fine_draws: jax.Array | None
    """Correlated Gaussian posterior draws of ``Q(t)`` shape ``(n_draws, N_fine)``, or ``None``."""

    d14c_fine_mean: jax.Array | None
    """Posterior mean of continuous ``Delta14C(t)`` on ``t_fine`` (‰), or ``None``."""

    d14c_fine_draws: jax.Array | None
    """Correlated Gaussian posterior draws of ``Delta14C(t)`` shape ``(n_draws, N_fine)``, or ``None``."""

    q_fine_std: jax.Array | None = None
    """Exact marginal posterior standard deviation of ``Q(t)`` on ``t_fine``, or ``None``."""

    d14c_fine_std: jax.Array | None = None
    """Exact marginal posterior standard deviation of ``Delta14C(t)`` on ``t_fine``, or ``None``."""

    ring_pred_mean: jax.Array | None = None
    """Posterior mean binned ``Delta14C`` prediction at all dataset rings ``(N_tot,)``."""


def _offset_column_count(cfg: LinearMarginalConfig, n_series: int) -> int:
    if cfg.offset_mode == "hierarchical":
        return 1 + (n_series if cfg.lab_prior_sd > 0.0 and n_series > 0 else 0)
    if cfg.offset_mode == "shared":
        return 1
    if cfg.offset_mode == "per_series":
        return n_series
    return 0


def linear_posterior(
    model: RadiocarbonModel,
    dataset: Dataset,
    cfg: LinearMarginalConfig | None = None,
    *,
    rng_key: jax.Array | None = None,
    n_draws: int = 0,
    t_fine: ArrayLike | None = None,
    hemisphere: Literal["north", "south"] = "north",
    box: int | None = None,
    sigma_jitter: ArrayLike | None = None,
) -> LinearPosteriorResult:
    """Closed-form Gaussian posterior over all marginalized linear parameters and trajectories."""
    if cfg is None:
        cfg = LinearMarginalConfig()

    sigma = dataset.sigma_flat
    if sigma_jitter is not None:
        sigma = jnp.sqrt(
            jnp.square(sigma)
            + jnp.square(jnp.asarray(sigma_jitter, dtype=jnp.float64))
        )

    resid = _nonlinear_residual(model, dataset, cfg)
    X, prior_mean, prior_sd, gp_projs = _build_marginal_design_with_projections(
        model, dataset, cfg
    )

    lm = LinearMarginal(
        X,
        prior_mean=prior_mean,
        prior_sd=prior_sd,
        method=cfg.whiten_method,
    )
    coeff_mean, coeff_cov = lm.posterior(resid, sigma)
    ring_pred_mean = dataset.d14c_flat - resid + X @ coeff_mean

    # --- Reconstruct offsets ---
    n_series = dataset.n_series
    n_off = _offset_column_count(cfg, n_series)
    if n_off > 0 and coeff_mean.shape[0] >= n_off:
        if cfg.offset_mode == "hierarchical":
            offset_global_mean = coeff_mean[0]
            if n_off == 1 + n_series:
                offset_lab_means = coeff_mean[1 : 1 + n_series]
            else:
                offset_lab_means = jnp.zeros((n_series,), dtype=jnp.float64)
        elif cfg.offset_mode == "shared":
            offset_global_mean = coeff_mean[0]
            offset_lab_means = jnp.zeros((n_series,), dtype=jnp.float64)
        else:  # per_series
            offset_lab_means = coeff_mean[:n_series]
            offset_global_mean = jnp.mean(offset_lab_means)
    else:
        offset_global_mean = jnp.asarray(0.0, dtype=jnp.float64)
        offset_lab_means = jnp.zeros((n_series,), dtype=jnp.float64)

    baseline_d14c = offset_global_mean

    # --- Optional continuous Q(t) and Delta14C(t) reconstruction on t_fine ---
    q_fine_mean: jax.Array | None = None
    q_fine_draws: jax.Array | None = None
    d14c_fine_mean: jax.Array | None = None
    d14c_fine_draws: jax.Array | None = None
    q_fine_std: jax.Array | None = None
    d14c_fine_std: jax.Array | None = None

    if t_fine is not None:
        t_fine_arr = jnp.asarray(t_fine, dtype=jnp.float64).reshape(-1)
        n_fine = t_fine_arr.shape[0]

        cbm = model.cbm
        A = cbm.matrix_A()
        b = cbm.production_vector()
        y_ss = cbm.steady_state_c14(q_std=model.q_std, alpha_unit=model.alpha_unit)
        trop_idx = jnp.asarray(cbm.topology.trop_box_indices, dtype=jnp.int32)
        y_trop = jnp.sum(y_ss[trop_idx])
        m_rel = cbm.relative_reservoir_masses()
        r_ss_ratio = (y_ss / y_trop) / m_rel

        target_box = model._resolve_box(box, hemisphere)
        r_box = r_ss_ratio[target_box]
        scale_box = 1000.0 / (y_trop * m_rel[target_box])

        unmarg, solar_terms, trend_terms, _ = _partition_terms_for_cfg(
            model.production, cfg
        )
        base_prod = _build_base_production(unmarg, model.q_std)
        base_model_fine = RadiocarbonModel(
            cbm,
            base_prod,
            solver=model.solver,
            q_std=model.q_std,
            alpha_unit=model.alpha_unit,
        )
        q_base = base_prod(t_fine_arr, q_std=model.q_std)
        d14c_base = base_model_fine.predict_fine(
            t_fine_arr, box=target_box, hemisphere=hemisphere
        )

        phi_q_cols: list[jax.Array] = []
        phi_d14c_cols: list[jax.Array] = []

        # 1. Offset columns
        if n_off > 0:
            phi_q_cols.append(jnp.zeros((n_fine, n_off), dtype=jnp.float64))
            if cfg.offset_mode == "hierarchical":
                g_col = jnp.full((n_fine, 1), r_box, dtype=jnp.float64)
                if n_off > 1:
                    l_cols = jnp.zeros((n_fine, n_off - 1), dtype=jnp.float64)
                    phi_d14c_cols.append(jnp.concatenate([g_col, l_cols], axis=1))
                else:
                    phi_d14c_cols.append(g_col)
            elif cfg.offset_mode == "shared":
                phi_d14c_cols.append(jnp.full((n_fine, 1), r_box, dtype=jnp.float64))
            else:  # per_series: average across series for atmospheric baseline
                phi_d14c_cols.append(
                    jnp.full((n_fine, n_off), 1.0 / max(n_off, 1), dtype=jnp.float64)
                )

        # 2. Marginalized HarmonicSolar columns ([cos_1, sin_1, ..., cos_M, sin_M])
        for solar in solar_terms:
            w_vec = jnp.asarray(solar.omegas, dtype=jnp.float64).reshape(-1)
            theta_f = (t_fine_arr[:, None] - solar.t_ref) * w_vec[None, :]
            phi_q_s = jnp.stack(
                [jnp.cos(theta_f), jnp.sin(theta_f)], axis=-1
            ).reshape(n_fine, 2 * w_vec.size)
            phi_d14c_s = harmonic_solar_design(
                A,
                b,
                w_vec,
                t_fine_arr,
                None,
                None,
                alpha_unit=model.alpha_unit,
                t_ref=solar.t_ref,
                box_indices=target_box,
                obs_scale=scale_box,
            )
            phi_q_cols.append(phi_q_s)
            phi_d14c_cols.append(phi_d14c_s)

        # 3. Marginalized LinearTrend columns
        for trend in trend_terms:
            unit_trend = LinearTrend(
                slope=1.0,
                t_ref=trend.t_ref,
                switched_on=trend.switched_on,
            )
            phi_q_t = unit_trend.delta_q(t_fine_arr, q_std=model.q_std)[:, None]
            dy_t = model.solver.solve_delta_trajectory(
                cbm,
                unit_trend,
                t_fine_arr,
                q_std=model.q_std,
                alpha_unit=model.alpha_unit,
            )
            phi_d14c_t = (dy_t[:, target_box] * scale_box)[:, None]
            phi_q_cols.append(phi_q_t)
            phi_d14c_cols.append(phi_d14c_t)

        # 4. Marginalized GPProduction columns
        b_eff = model.alpha_unit * jnp.asarray(b, dtype=jnp.float64).reshape(-1)
        for proj in gp_projs:
            assert proj.spectral_weights is not None
            assert proj.whitened_design is not None
            r_modes = proj.omegas.shape[0]
            n_gp_cols = proj.whitened_design.shape[1]
            n_anc_cols = n_gp_cols - r_modes

            basis_f = solin_sarkka_basis(
                t_fine_arr, proj.a, proj.b_bound, r_modes
            )
            sqrt_psd = jnp.sqrt(jnp.maximum(proj.spectral_weights, 0.0))
            in_domain = (t_fine_arr >= proj.a) & (t_fine_arr <= proj.b_bound)
            psi_masked = jnp.where(in_domain[:, None], basis_f.psi, 0.0)
            phi_q_modes = psi_masked * sqrt_psd[None, :]
            if n_anc_cols > 0:
                phi_q_gp = jnp.concatenate(
                    [phi_q_modes, jnp.zeros((n_fine, n_anc_cols), dtype=jnp.float64)],
                    axis=1,
                )
            else:
                phi_q_gp = phi_q_modes
            phi_q_cols.append(phi_q_gp)

            u_mat, v_mat = _solve_complex_resolvents(A, b_eff, proj.omegas)
            u_box = scale_box * u_mat[target_box]
            v_box = scale_box * v_mat[target_box]
            dt_a = jnp.maximum(t_fine_arr - proj.a, 0.0)
            expm_a_row = jax.vmap(
                lambda dt: jax.scipy.linalg.expm(A * dt)[target_box]
            )(dt_a)
            causal_f = (
                jnp.cos(dt_a[:, None] * proj.omegas[None, :]) * v_box[None, :]
                + jnp.sin(dt_a[:, None] * proj.omegas[None, :]) * u_box[None, :]
                - scale_box * (expm_a_row @ v_mat)
            ) / jnp.sqrt(proj.half_length)
            causal_f = jnp.where((t_fine_arr >= proj.a)[:, None], causal_f, 0.0)
            phi_d14c_modes = causal_f * sqrt_psd[None, :]

            if n_anc_cols > 0 and proj.anchor_cov_root is not None:
                dt_anc = jnp.maximum(t_fine_arr - proj.t_anchor, 0.0)
                expm_anc_row = jax.vmap(
                    lambda dt: jax.scipy.linalg.expm(A * dt)[target_box]
                )(dt_anc)
                phi_d14c_anc = scale_box * (expm_anc_row @ proj.anchor_cov_root)
                phi_d14c_gp = jnp.concatenate(
                    [phi_d14c_modes, phi_d14c_anc], axis=1
                )
            else:
                phi_d14c_gp = phi_d14c_modes
            phi_d14c_cols.append(phi_d14c_gp)

        if phi_q_cols:
            phi_q = jnp.concatenate(phi_q_cols, axis=1)
            phi_d14c = jnp.concatenate(phi_d14c_cols, axis=1)
        else:
            phi_q = jnp.zeros((n_fine, 0), dtype=jnp.float64)
            phi_d14c = jnp.zeros((n_fine, 0), dtype=jnp.float64)

        q_fine_mean = q_base + phi_q @ coeff_mean
        d14c_fine_mean = d14c_base + phi_d14c @ coeff_mean

        q_var = jnp.sum((phi_q @ coeff_cov) * phi_q, axis=1)
        d14c_var = jnp.sum((phi_d14c @ coeff_cov) * phi_d14c, axis=1)
        q_fine_std = jnp.sqrt(jnp.maximum(q_var, 0.0))
        d14c_fine_std = jnp.sqrt(jnp.maximum(d14c_var, 0.0))

        if n_draws > 0:
            if rng_key is None:
                raise ValueError("rng_key is required when n_draws > 0.")
            k_tot = coeff_mean.shape[0]
            if k_tot > 0:
                jitter_eye = 1e-12 * jnp.eye(k_tot, dtype=jnp.float64)
                L_cov = jnp.linalg.cholesky(0.5 * (coeff_cov + coeff_cov.T) + jitter_eye)
                z = jax.random.normal(
                    rng_key, shape=(n_draws, k_tot), dtype=jnp.float64
                )
                w_draws = coeff_mean[None, :] + z @ L_cov.T
                q_fine_draws = q_base[None, :] + w_draws @ phi_q.T
                d14c_fine_draws = d14c_base[None, :] + w_draws @ phi_d14c.T
            else:
                q_fine_draws = jnp.broadcast_to(q_base[None, :], (n_draws, n_fine))
                d14c_fine_draws = jnp.broadcast_to(
                    d14c_base[None, :], (n_draws, n_fine)
                )

    return LinearPosteriorResult(
        coeff_mean=coeff_mean,
        coeff_cov=coeff_cov,
        offset_global_mean=offset_global_mean,
        offset_lab_means=offset_lab_means,
        baseline_d14c=baseline_d14c,
        q_fine_mean=q_fine_mean,
        q_fine_draws=q_fine_draws,
        d14c_fine_mean=d14c_fine_mean,
        d14c_fine_draws=d14c_fine_draws,
        q_fine_std=q_fine_std,
        d14c_fine_std=d14c_fine_std,
        ring_pred_mean=ring_pred_mean,
    )
