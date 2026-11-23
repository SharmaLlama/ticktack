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
from typing import Any, Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
import optimistix as optx
import zodiax as zdx
from jax.typing import ArrayLike

from .cbm import CarbonBoxModel, SubspaceCBM
from .data import Dataset
from .fields import (
    DCTResponseOperator,
    GaussianField,
    LinearizedDCTOperator,
    LogGaussianField,
    _compute_cbm_digest,
    _compute_dataset_digest,
    _compute_subspace_basis_digest,
    _compute_template_digest,
    _field_prehistory_root_y0,
    _safe_sqrt,
    build_dct_response_operator,
    idct2,
)
from .linear import (
    CausalGPProjection,
    LinearMarginal,
    _concrete,
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
from .solvers import ExactLinearSolver, ModaxSolver, _modal_decomposition, wrap_jvp_only_scalar_fn

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
        target_space: Literal["unconstrained", "constrained"] = "unconstrained",
        include_jacobian: bool | None = None,
    ) -> jax.Array:
        """Normalized Gaussian log-prior density in ``u``-space or induced ``theta``-space.

        Since the Gaussian prior ``u ~ N(prior_mean, diag(prior_sd^2))`` is placed on
        unconstrained coordinates ``u`` with physical parameters ``theta = g(u)``,
        change of variables gives:

        * ``target_space="unconstrained"`` (default): ``log p_u(u)``.
        * ``target_space="constrained"``: ``log p_theta(g(u)) = log p_u(u) - log |det dg/du|``.
        """
        if target_space not in ("unconstrained", "constrained"):
            raise ValueError(
                f"target_space must be 'unconstrained' or 'constrained'; got {target_space!r}."
            )
        if include_jacobian is not None:
            implied = "constrained" if include_jacobian else "unconstrained"
            if target_space != "unconstrained" and target_space != implied:
                raise ValueError(
                    "Conflicting target_space and include_jacobian arguments."
                )
            eff_space = implied
        else:
            eff_space = target_space

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
        if eff_space == "constrained":
            for spec, shape, (start, stop) in zip(
                self.specs, self.leaf_shapes, self.spec_slices
            ):
                if stop > start and spec.bijector != "identity":
                    log_p = log_p - spec.log_abs_det_jacobian(
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
    marginalise_field: bool = eqx.field(static=True)
    n_gp_modes: int | None = eqx.field(static=True)
    include_gp_initial_cov: bool = eqx.field(static=True)
    include_field_initial_cov: bool = eqx.field(static=True)
    whiten_method: str = eqx.field(static=True)
    dct_operator: DCTResponseOperator | LinearizedDCTOperator | None

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
        marginalise_field: bool | None = None,
        marginalize_solar: bool | None = None,
        marginalize_trend: bool | None = None,
        marginalize_gp: bool | None = None,
        marginalize_field: bool | None = None,
        n_gp_modes: int | None = None,
        include_gp_initial_cov: bool = True,
        include_field_initial_cov: bool = True,
        whiten_method: Literal["cholesky", "qr", "rank_one"] = "cholesky",
        dct_operator: DCTResponseOperator | LinearizedDCTOperator | None = None,
    ) -> None:
        valid_offset_modes = ("hierarchical", "per_series", "shared", "none")
        if offset_mode not in valid_offset_modes:
            raise ValueError(
                f"offset_mode must be one of {valid_offset_modes}; got {offset_mode!r}."
            )
        if dct_operator is not None and not isinstance(
            dct_operator, (DCTResponseOperator, LinearizedDCTOperator)
        ):
            raise TypeError(
                "dct_operator must be a DCTResponseOperator, LinearizedDCTOperator, or None; "
                f"got {type(dct_operator).__name__}."
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
        eff_field = (
            marginalize_field
            if marginalize_field is not None
            else (eff_gp if marginalise_field is None else marginalise_field)
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
        self.marginalise_field = bool(eff_field)
        self.n_gp_modes = None if n_gp_modes is None else int(n_gp_modes)
        self.include_gp_initial_cov = bool(include_gp_initial_cov)
        self.include_field_initial_cov = bool(include_field_initial_cov)
        self.whiten_method = str(whiten_method)
        self.dct_operator = dct_operator

    @property
    def marginalize_solar(self) -> bool:
        return self.marginalise_solar

    @property
    def marginalize_trend(self) -> bool:
        return self.marginalise_trend

    @property
    def marginalize_gp(self) -> bool:
        return self.marginalise_gp

    @property
    def marginalize_field(self) -> bool:
        return self.marginalise_field


# ---------------------------------------------------------------------------
# 4. Internal helpers: conditional term partitioning, padded windows, designs
# ---------------------------------------------------------------------------


def _has_loggp(production: ProductionTerm) -> bool:
    """Return True if ``production`` is or contains a :class:`LogGPProduction` or :class:`LogGaussianField`."""
    if isinstance(production, (LogGPProduction, LogGaussianField)):
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
    list[GaussianField],
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
    field_terms: list[GaussianField] = []

    for term in parts:
        if isinstance(term, HarmonicSolar) and cfg.marginalise_solar:
            solar_terms.append(term)
        elif isinstance(term, LinearTrend) and cfg.marginalise_trend:
            trend_terms.append(term)
        elif isinstance(term, GPProduction) and cfg.marginalise_gp:
            gp_terms.append(term)
        elif isinstance(term, GaussianField) and cfg.marginalise_field:
            field_terms.append(term)
        else:
            unmarginalized.append(term)

    return unmarginalized, solar_terms, trend_terms, gp_terms, field_terms


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


def _gp_anchor_state_map(
    A: jax.Array,
    b_eff: jax.Array,
    proj: CausalGPProjection,
    cbm: CarbonBoxModel,
) -> jax.Array:
    """Return ``(n_boxes, r_modes + n_anc)`` map from whitened GP coords to ``delta_y(t_anchor)``."""
    assert proj.spectral_weights is not None and proj.anchor_cov_root is not None
    u_mat, v_mat = _solve_complex_resolvents(A, b_eff, proj.omegas)
    dt_anc = proj.t_anchor - proj.a
    modal = _modal_decomposition(A, cbm=cbm)
    if modal is not None:
        lam_m, v_eig_m, v_inv_m = modal
        expm_dt_v = jnp.real(
            v_eig_m
            @ (
                jnp.exp(lam_m * dt_anc)[:, None]
                * (v_inv_m @ v_mat.astype(v_inv_m.dtype))
            )
        )
    else:
        expm_dt_v = jsl.expm(A * dt_anc) @ v_mat
    c_t0 = jnp.cos(proj.omegas * dt_anc)[None, :]
    s_t0 = jnp.sin(proj.omegas * dt_anc)[None, :]
    y_ss_t0 = (v_mat * c_t0 + u_mat * s_t0 - expm_dt_v) / jnp.sqrt(proj.half_length)
    mode_map = y_ss_t0 * jnp.sqrt(jnp.maximum(proj.spectral_weights, 0.0))[None, :]
    return jnp.concatenate([mode_map, proj.anchor_cov_root], axis=1)


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
        cbm=model.cbm,
    )


def _build_marginal_design_with_projections(
    model: RadiocarbonModel,
    dataset: Dataset,
    cfg: LinearMarginalConfig,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    list[CausalGPProjection],
    list[jax.Array | None],
    list[tuple[slice, jax.Array, jax.Array]],
]:
    """Build ``(X, prior_mean, prior_sd, gp_projections, field_init_roots, init_state_blocks)``."""
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

    _, solar_terms, trend_terms, gp_terms, field_terms = _partition_terms_for_cfg(
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
    init_state_blocks: list[tuple[slice, jax.Array, jax.Array]] = []
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
            r_modes = proj.omegas.shape[0]
            col_start = sum(c.shape[1] for c in design_cols)
            design_cols.append(H_gp)
            pmean_parts.append(jnp.zeros((r_total,), dtype=jnp.float64))
            psd_parts.append(jnp.ones((r_total,), dtype=jnp.float64))
            if r_total > r_modes and proj.anchor_cov_root is not None:
                b_eff_gp = model.alpha_unit * jnp.asarray(b, dtype=jnp.float64).reshape(-1)
                gp_state_map = _gp_anchor_state_map(A, b_eff_gp, proj, cbm)
                init_state_blocks.append(
                    (
                        slice(col_start, col_start + r_total),
                        gp_state_map,
                        proj.t_anchor,
                    )
                )

    # 5. Marginalized GaussianField reflecting-boundary DCT-II columns
    field_init_roots: list[jax.Array | None] = []
    if field_terms and n_tot > 0:
        for gf in field_terms:
            op_cfg = cfg.dct_operator
            if isinstance(op_cfg, LinearizedDCTOperator):
                if not isinstance(cbm, SubspaceCBM):
                    raise TypeError(
                        "LinearMarginalConfig.dct_operator is a LinearizedDCTOperator, "
                        f"which requires model.cbm to be a SubspaceCBM; got {type(cbm).__name__}."
                    )
                if op_cfg.w_ref.shape != (n_tot, gf.n_grid) or (
                    abs(op_cfg.t_start - gf.t_start) > 1e-9
                    or abs(op_cfg.t_end - gf.t_end) > 1e-9
                ):
                    raise ValueError(
                        "LinearizedDCTOperator geometry "
                        f"(shape={op_cfg.w_ref.shape}, [{op_cfg.t_start}, {op_cfg.t_end}]) "
                        f"does not match dataset/GaussianField "
                        f"(shape=({n_tot}, {gf.n_grid}), [{gf.t_start}, {gf.t_end}])."
                    )
                if op_cfg.dataset_digest:
                    ds_dig = _compute_dataset_digest(cbm, dataset)
                    if ds_dig and ds_dig != op_cfg.dataset_digest:
                        raise ValueError(
                            "LinearizedDCTOperator dataset_digest does not match dataset."
                        )
                if op_cfg.template_digest:
                    tmpl_dig = _compute_template_digest(gf.template)
                    if tmpl_dig and tmpl_dig != op_cfg.template_digest:
                        raise ValueError(
                            "LinearizedDCTOperator template_digest does not match GaussianField.template."
                        )
                if op_cfg.subspace_basis_digest:
                    basis_dig = _compute_subspace_basis_digest(cbm)
                    if basis_dig != op_cfg.subspace_basis_digest:
                        raise ValueError(
                            "LinearizedDCTOperator subspace_basis_digest does not match model.cbm."
                        )
                w_dct_gf = op_cfg.evaluate_w_dct(cbm.alpha)
            elif isinstance(op_cfg, DCTResponseOperator):
                if op_cfg.w_dct.shape != (n_tot, gf.n_grid) or (
                    abs(op_cfg.t_start - gf.t_start) > 1e-9
                    or abs(op_cfg.t_end - gf.t_end) > 1e-9
                ):
                    raise ValueError(
                        "DCTResponseOperator geometry "
                        f"(shape={op_cfg.w_dct.shape}, [{op_cfg.t_start}, {op_cfg.t_end}]) "
                        f"does not match dataset/GaussianField "
                        f"(shape=({n_tot}, {gf.n_grid}), [{gf.t_start}, {gf.t_end}])."
                    )
                if op_cfg.dataset_digest:
                    ds_dig = _compute_dataset_digest(cbm, dataset)
                    if ds_dig and ds_dig != op_cfg.dataset_digest:
                        raise ValueError(
                            "DCTResponseOperator dataset_digest does not match dataset."
                        )
                if op_cfg.template_digest:
                    tmpl_dig = _compute_template_digest(gf.template)
                    if tmpl_dig and tmpl_dig != op_cfg.template_digest:
                        raise ValueError(
                            "DCTResponseOperator template_digest does not match GaussianField.template."
                        )
                cur_cbm_dig = _compute_cbm_digest(cbm)
                if cur_cbm_dig == "" or (
                    op_cfg.cbm_digest != "" and cur_cbm_dig != op_cfg.cbm_digest
                ):
                    op_gf = build_dct_response_operator(
                        cbm,
                        dataset,
                        t_start=gf.t_start,
                        t_end=gf.t_end,
                        n_grid=gf.n_grid,
                        q_std=model.q_std,
                        alpha_unit=model.alpha_unit,
                        template=gf.template,
                    )
                    w_dct_gf = op_gf.w_dct
                else:
                    w_dct_gf = op_cfg.w_dct
            else:
                op_gf = build_dct_response_operator(
                    cbm,
                    dataset,
                    t_start=gf.t_start,
                    t_end=gf.t_end,
                    n_grid=gf.n_grid,
                    q_std=model.q_std,
                    alpha_unit=model.alpha_unit,
                    template=gf.template,
                )
                w_dct_gf = op_gf.w_dct
            H_gf = w_dct_gf * _safe_sqrt(gf.power_spectrum())[None, :]
            design_cols.append(H_gf)
            pmean_parts.append(jnp.zeros((gf.n_grid,), dtype=jnp.float64))
            psd_parts.append(jnp.ones((gf.n_grid,), dtype=jnp.float64))
            if cfg.include_field_initial_cov:
                root_y0_gf = _field_prehistory_root_y0(cbm, model.alpha_unit, gf)
                H_init = op_gf.init_state_basis @ root_y0_gf
                m_rank = H_init.shape[1]
                init_start = sum(c.shape[1] for c in design_cols)
                design_cols.append(H_init)
                pmean_parts.append(jnp.zeros((m_rank,), dtype=jnp.float64))
                psd_parts.append(jnp.ones((m_rank,), dtype=jnp.float64))
                init_state_blocks.append(
                    (
                        slice(init_start, init_start + m_rank),
                        root_y0_gf,
                        jnp.asarray(gf.t_start, dtype=jnp.float64),
                    )
                )
                field_init_roots.append(root_y0_gf)
            else:
                field_init_roots.append(None)

    if design_cols:
        X = jnp.concatenate(design_cols, axis=1)
        pm = jnp.concatenate(pmean_parts)
        ps = jnp.concatenate(psd_parts)
    else:
        X = jnp.zeros((n_tot, 0), dtype=jnp.float64)
        pm = jnp.zeros((0,), dtype=jnp.float64)
        ps = jnp.zeros((0,), dtype=jnp.float64)

    return X, pm, ps, gp_projs, field_init_roots, init_state_blocks


def _build_marginal_design(
    model: RadiocarbonModel,
    dataset: Dataset,
    cfg: LinearMarginalConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build ``(X, prior_mean, prior_sd)`` for all marginalized linear terms."""
    X, pm, ps, _, _, _ = _build_marginal_design_with_projections(model, dataset, cfg)
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
    unmarginalized, _, _, _, _ = _partition_terms_for_cfg(model.production, cfg_eff)
    base_prod = _build_base_production(unmarginalized, model.q_std)

    if not isinstance(model.solver, ModaxSolver):
        base_model = RadiocarbonModel(
            model.cbm,
            base_prod,
            solver=model.solver,
            q_std=model.q_std,
            alpha_unit=model.alpha_unit,
        )
        pred_flat = base_model.predict_d14c(dataset, concat=True)
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


def _uses_joint_mode(
    model: RadiocarbonModel,
    parameter_space: ParameterSpace | None,
    cfg: LinearMarginalConfig,
) -> bool:
    """Return True when inference runs in joint mode rather than marginalizing ``production``."""
    if isinstance(model.solver, ModaxSolver):
        return True
    lgf, _ = _extract_single_lgf(model.production)
    if lgf is not None:
        if not cfg.marginalise_field:
            return True
        if parameter_space is None:
            return False
        return any(
            spec.path == "production.latent" or spec.path.endswith(".latent")
            for spec in parameter_space.specs
        )
    return _has_loggp(model.production)


def _marginal_design_is_static(
    base_model: RadiocarbonModel,
    parameter_space: ParameterSpace,
    cfg: LinearMarginalConfig,
) -> bool:
    """Return True if ``(X, prior_mean, prior_sd)`` is invariant across ``parameter_space``."""
    if isinstance(base_model.solver, ModaxSolver):
        return False

    joint_mode = _uses_joint_mode(base_model, parameter_space, cfg)
    for spec in parameter_space.specs:
        p = spec.path
        if p in ("cbm", "q_std", "alpha_unit") or p.startswith("cbm."):
            return False

    if joint_mode:
        # In joint mode only baseline offsets are marginalized, which depend solely on CBM/q_std/alpha_unit.
        return True

    if _extract_single_lgf(base_model.production)[0] is not None:
        return False

    prod = base_model.production
    if isinstance(prod, ProductionSum):
        marg_names = {
            name
            for name, part in zip(prod.names, prod.parts)
            if (isinstance(part, HarmonicSolar) and cfg.marginalise_solar)
            or (isinstance(part, LinearTrend) and cfg.marginalise_trend)
            or (isinstance(part, GPProduction) and cfg.marginalise_gp)
            or (isinstance(part, GaussianField) and cfg.marginalise_field)
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
        or (isinstance(prod, GaussianField) and cfg.marginalise_field)
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


@eqx.filter_jit
def _jit_prefactor_joint_off(
    base_model: RadiocarbonModel,
    dataset: Dataset,
    off_cfg: LinearMarginalConfig,
    sigma: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    X_off, pm_off, ps_off = _build_marginal_design(base_model, dataset, off_cfg)
    lm_off = LinearMarginal(
        X_off,
        prior_mean=pm_off,
        prior_sd=ps_off,
        method=off_cfg.whiten_method,
    )
    return _prefactor_static_whiten(lm_off, sigma)


@eqx.filter_jit
def _jit_prefactor_marginal(
    base_model: RadiocarbonModel,
    dataset: Dataset,
    cfg_eff: LinearMarginalConfig,
    sigma: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    X_m, pm_m, ps_m = _build_marginal_design(base_model, dataset, cfg_eff)
    lm_m = LinearMarginal(
        X_m,
        prior_mean=pm_m,
        prior_sd=ps_m,
        method=cfg_eff.whiten_method,
    )
    return _prefactor_static_whiten(lm_m, sigma)


class FastLogLikeSpec(eqx.Module):
    """PyTree-callable unconstrained log-likelihood for module-level ``@eqx.filter_jit`` caching."""

    base_model: RadiocarbonModel
    dataset: Dataset
    parameter_space: ParameterSpace
    cfg_eff: LinearMarginalConfig
    mode: str = eqx.field(static=True)
    joint_mode: bool = eqx.field(static=True)
    sigma_jitter_arr: jax.Array | None
    shift_c: jax.Array | None
    inv_sig: jax.Array | None
    w_mat: jax.Array | None
    p_right: jax.Array | None
    const_n: jax.Array | None
    modal_const: tuple[jax.Array, jax.Array, jax.Array] | None = None
    box_indices_flat: jax.Array | None = None
    base_ss_flat: jax.Array | None = None
    obs_scale_flat: jax.Array | None = None

    def _predict_exact_static(
        self, m_eval: RadiocarbonModel, prod: ProductionTerm
    ) -> jax.Array:
        assert (
            isinstance(m_eval.solver, ExactLinearSolver)
            and self.box_indices_flat is not None
            and self.base_ss_flat is not None
            and self.obs_scale_flat is not None
        )
        all_ts, all_te, all_w = _concat_padded_windows(self.dataset)
        dy = m_eval.solver.solve_delta_binned(
            m_eval.cbm,
            prod,
            all_ts,
            all_te,
            sub_weights=all_w,
            q_std=m_eval.q_std,
            alpha_unit=m_eval.alpha_unit,
            modal=self.modal_const,
        )
        n_tot = self.box_indices_flat.shape[0]
        return (
            self.base_ss_flat
            + dy[jnp.arange(n_tot), self.box_indices_flat] * self.obs_scale_flat
        )

    def nonlinear_residual_from_u(self, u_vec: jax.Array) -> jax.Array:
        """Compute ``d - m_0(u_vec)`` using precomputed modal/observation factors when static."""
        m_eval = self.parameter_space.constrain(u_vec, self.base_model)
        can_fast_exact = (
            isinstance(m_eval.solver, ExactLinearSolver)
            and self.box_indices_flat is not None
            and self.base_ss_flat is not None
            and self.obs_scale_flat is not None
        )
        if self.joint_mode:
            pred = (
                self._predict_exact_static(m_eval, m_eval.production)
                if can_fast_exact
                else m_eval.predict_d14c(self.dataset, offsets=None, concat=True)
            )
            return self.dataset.d14c_flat - pred
        if can_fast_exact:
            unmarginalized, _, _, _, _ = _partition_terms_for_cfg(
                m_eval.production, self.cfg_eff
            )
            base_prod = _build_base_production(unmarginalized, m_eval.q_std)
            pred = self._predict_exact_static(m_eval, base_prod)
            return self.dataset.d14c_flat - pred
        return _nonlinear_residual(m_eval, self.dataset, self.cfg_eff)

    def __call__(self, u_vec: jax.Array) -> jax.Array:
        m_eval = self.parameter_space.constrain(u_vec, self.base_model)
        can_fast_exact = (
            isinstance(m_eval.solver, ExactLinearSolver)
            and self.box_indices_flat is not None
            and self.base_ss_flat is not None
            and self.obs_scale_flat is not None
        )
        if self.mode == "joint_no_off":
            assert self.inv_sig is not None and self.const_n is not None
            pred = (
                self._predict_exact_static(m_eval, m_eval.production)
                if can_fast_exact
                else m_eval.predict_d14c(self.dataset, offsets=None, concat=True)
            )
            x_arr = (self.dataset.d14c_flat - pred) * self.inv_sig
            return -0.5 * jnp.dot(x_arr, x_arr) + self.const_n
        if self.mode == "joint_off":
            assert (
                self.shift_c is not None
                and self.inv_sig is not None
                and self.w_mat is not None
                and self.p_right is not None
                and self.const_n is not None
            )
            pred = (
                self._predict_exact_static(m_eval, m_eval.production)
                if can_fast_exact
                else m_eval.predict_d14c(self.dataset, offsets=None, concat=True)
            )
            x_arr = (self.dataset.d14c_flat - pred - self.shift_c) * self.inv_sig
            u_out = x_arr - self.w_mat @ (self.p_right @ x_arr)
            return -0.5 * jnp.dot(u_out, u_out) + self.const_n
        if self.mode == "marginal":
            assert (
                self.shift_c is not None
                and self.inv_sig is not None
                and self.w_mat is not None
                and self.p_right is not None
                and self.const_n is not None
            )
            if can_fast_exact:
                unmarginalized, _, _, _, _ = _partition_terms_for_cfg(
                    m_eval.production, self.cfg_eff
                )
                base_prod = _build_base_production(unmarginalized, m_eval.q_std)
                pred = self._predict_exact_static(m_eval, base_prod)
                resid = self.dataset.d14c_flat - pred
            else:
                resid = _nonlinear_residual(m_eval, self.dataset, self.cfg_eff)
            x_arr = (resid - self.shift_c) * self.inv_sig
            u_out = x_arr - self.w_mat @ (self.p_right @ x_arr)
            return -0.5 * jnp.dot(u_out, u_out) + self.const_n
        if self.joint_mode:
            return joint_log_likelihood(
                m_eval,
                self.dataset,
                cfg=self.cfg_eff,
                sigma_jitter=self.sigma_jitter_arr,
            )
        return marginal_log_likelihood(
            m_eval,
            self.dataset,
            self.cfg_eff,
            sigma_jitter=self.sigma_jitter_arr,
        )


@eqx.filter_jit
def _jit_static_cbm_arrays(
    base_model: RadiocarbonModel,
    box_idx_flat: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    cbm = base_model.cbm
    A_c = cbm.matrix_A()
    y_ss = cbm.steady_state_c14(
        q_std=base_model.q_std, alpha_unit=base_model.alpha_unit
    )
    trop_idx = jnp.asarray(cbm.topology.trop_box_indices, dtype=jnp.int32)
    y_trop = jnp.sum(y_ss[trop_idx])
    m_rel = cbm.relative_reservoir_masses()
    r_ss_ratio = (y_ss / y_trop) / m_rel
    base_ss_flat = 1000.0 * (r_ss_ratio[box_idx_flat] - 1.0)
    obs_scale_flat = 1000.0 / (y_trop * m_rel[box_idx_flat])
    return A_c, base_ss_flat, obs_scale_flat


def _precompute_static_cbm_obs(
    base_model: RadiocarbonModel, dataset: Dataset
) -> tuple[
    tuple[jax.Array, jax.Array, jax.Array] | None,
    jax.Array | None,
    jax.Array | None,
    jax.Array | None,
]:
    if not isinstance(base_model.solver, ExactLinearSolver):
        return None, None, None, None
    box_idx_flat = jnp.array(
        [
            base_model._box_index_for_series(s)
            for s in dataset.series
            for _ in range(s.n_obs)
        ],
        dtype=jnp.int32,
    )
    A_c, base_ss_flat, obs_scale_flat = _jit_static_cbm_arrays(
        base_model, box_idx_flat
    )
    modal_c = _modal_decomposition(A_c, cbm=base_model.cbm)
    return modal_c, box_idx_flat, base_ss_flat, obs_scale_flat


def _make_fast_unconstrained_loglike(
    base_model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    cfg: LinearMarginalConfig | None = None,
    sigma_jitter: ArrayLike | None = None,
) -> FastLogLikeSpec:
    """Build a PyTree-callable ``u_vec -> log_likelihood``, pre-factoring ``LinearMarginal`` when static."""
    cfg_eff = LinearMarginalConfig() if cfg is None else cfg
    joint_mode = _uses_joint_mode(base_model, parameter_space, cfg_eff)
    jit_arr = (
        jnp.asarray(sigma_jitter, dtype=jnp.float64)
        if sigma_jitter is not None
        else None
    )

    if (
        dataset.n_obs > 0
        and not isinstance(sigma_jitter, jax.core.Tracer)
        and not isinstance(base_model.solver, ModaxSolver)
        and _marginal_design_is_static(base_model, parameter_space, cfg_eff)
    ):
        sigma = jnp.asarray(dataset.sigma_flat, dtype=jnp.float64)
        if jit_arr is not None:
            sigma = jnp.sqrt(jnp.square(sigma) + jnp.square(jit_arr))
        modal_c, box_idx_f, base_ss_f, obs_sc_f = _precompute_static_cbm_obs(
            base_model, dataset
        )

        if joint_mode:
            if cfg_eff.offset_mode == "none":
                inv_sig = 1.0 / sigma
                const_n = -jnp.sum(jnp.log(sigma)) - 0.5 * dataset.n_obs * math.log(
                    2.0 * math.pi
                )
                return FastLogLikeSpec(
                    base_model=base_model,
                    dataset=dataset,
                    parameter_space=parameter_space,
                    cfg_eff=cfg_eff,
                    mode="joint_no_off",
                    joint_mode=True,
                    sigma_jitter_arr=jit_arr,
                    shift_c=None,
                    inv_sig=inv_sig,
                    w_mat=None,
                    p_right=None,
                    const_n=const_n,
                    modal_const=modal_c,
                    box_indices_flat=box_idx_f,
                    base_ss_flat=base_ss_f,
                    obs_scale_flat=obs_sc_f,
                )

            off_cfg = _offset_only_cfg(cfg_eff)
            shift_c, inv_sig, w_mat, p_right, const_n = _jit_prefactor_joint_off(
                base_model, dataset, off_cfg, sigma
            )
            return FastLogLikeSpec(
                base_model=base_model,
                dataset=dataset,
                parameter_space=parameter_space,
                cfg_eff=cfg_eff,
                mode="joint_off",
                joint_mode=True,
                sigma_jitter_arr=jit_arr,
                shift_c=shift_c,
                inv_sig=inv_sig,
                w_mat=w_mat,
                p_right=p_right,
                const_n=const_n,
                modal_const=modal_c,
                box_indices_flat=box_idx_f,
                base_ss_flat=base_ss_f,
                obs_scale_flat=obs_sc_f,
            )

        shift_c, inv_sig, w_mat, p_right, const_n = _jit_prefactor_marginal(
            base_model, dataset, cfg_eff, sigma
        )
        return FastLogLikeSpec(
            base_model=base_model,
            dataset=dataset,
            parameter_space=parameter_space,
            cfg_eff=cfg_eff,
            mode="marginal",
            joint_mode=False,
            sigma_jitter_arr=jit_arr,
            shift_c=shift_c,
            inv_sig=inv_sig,
            w_mat=w_mat,
            p_right=p_right,
            const_n=const_n,
            modal_const=modal_c,
            box_indices_flat=box_idx_f,
            base_ss_flat=base_ss_f,
            obs_scale_flat=obs_sc_f,
        )

    return FastLogLikeSpec(
        base_model=base_model,
        dataset=dataset,
        parameter_space=parameter_space,
        cfg_eff=cfg_eff,
        mode="dynamic",
        joint_mode=bool(joint_mode),
        sigma_jitter_arr=jit_arr,
        shift_c=None,
        inv_sig=None,
        w_mat=None,
        p_right=None,
        const_n=None,
    )


# ---------------------------------------------------------------------------
# 5. Marginal log-likelihood and log-posterior
# ---------------------------------------------------------------------------


def _extract_single_lgf(
    production: ProductionTerm,
) -> tuple[LogGaussianField | None, list[ProductionTerm]]:
    parts = (
        list(production.parts)
        if isinstance(production, ProductionSum)
        else [production]
    )
    lgfs = [p for p in parts if isinstance(p, LogGaussianField)]
    if len(lgfs) != 1:
        return None, parts
    others = [p for p in parts if not isinstance(p, LogGaussianField)]
    return lgfs[0], others


def _laplace_marginal_log_likelihood_lgf(
    model: RadiocarbonModel,
    dataset: Dataset,
    lgf: LogGaussianField,
    other_parts: list[ProductionTerm],
    cfg: LinearMarginalConfig,
    sigma: jax.Array,
) -> jax.Array:
    """Laplace-marginalized log-likelihood over ``LogGaussianField`` latent ``z ~ N(0, I)``."""
    if dataset.n_obs == 0:
        return jnp.asarray(0.0, dtype=jnp.float64)
    cfg_no_field = LinearMarginalConfig(
        offset_mode=cfg.offset_mode,
        global_prior_mean=cfg.global_prior_mean,
        global_prior_sd=cfg.global_prior_sd,
        lab_prior_sd=cfg.lab_prior_sd,
        solar_prior_sd=cfg.solar_prior_sd,
        trend_prior_sd=cfg.trend_prior_sd,
        marginalise_solar=cfg.marginalise_solar,
        marginalise_trend=cfg.marginalise_trend,
        marginalise_gp=cfg.marginalise_gp,
        marginalise_field=False,
        n_gp_modes=cfg.n_gp_modes,
        include_gp_initial_cov=cfg.include_gp_initial_cov,
        include_field_initial_cov=False,
        whiten_method=cfg.whiten_method,
    )
    model_no_lgf = RadiocarbonModel(
        model.cbm,
        _build_base_production(other_parts, model.q_std),
        solver=model.solver,
        q_std=model.q_std,
        alpha_unit=model.alpha_unit,
    )
    resid_base = _nonlinear_residual(model_no_lgf, dataset, cfg_no_field)
    X_o, pm_o, ps_o = _build_marginal_design(model_no_lgf, dataset, cfg_no_field)
    op_lgf = build_dct_response_operator(
        model.cbm,
        dataset,
        t_start=lgf.t_start,
        t_end=lgf.t_end,
        n_grid=lgf.n_grid,
        q_std=model.q_std,
        alpha_unit=model.alpha_unit,
    )
    if cfg.include_field_initial_cov:
        root_y0_lgf = _field_prehistory_root_y0(model.cbm, model.alpha_unit, lgf)
        H_init_lgf = op_lgf.init_state_basis @ root_y0_lgf
        m_rank = H_init_lgf.shape[1]
        X_o = jnp.concatenate([X_o, H_init_lgf], axis=1)
        pm_o = jnp.concatenate([pm_o, jnp.zeros((m_rank,), dtype=jnp.float64)], axis=0)
        ps_o = jnp.concatenate([ps_o, jnp.ones((m_rank,), dtype=jnp.float64)], axis=0)
    lm_o = LinearMarginal(X_o, prior_mean=pm_o, prior_sd=ps_o, method=cfg.whiten_method)
    shift_c, inv_sig, w_mat, p_right, const_n = _prefactor_static_whiten(lm_o, sigma)

    x_arr = (resid_base - shift_c) * inv_sig
    x_target = x_arr - w_mat @ (p_right @ x_arr)
    wg_scaled = inv_sig[:, None] * op_lgf.w_grid
    w_grid_eff = wg_scaled - w_mat @ (p_right @ wg_scaled)

    def _r_eff_of_z(z_in: jax.Array) -> jax.Array:
        dq = lgf.with_latent(z_in).evaluate_grid(q_std=model.q_std)
        return w_grid_eff @ dq - x_target

    def _ham_z(z_in: jax.Array) -> jax.Array:
        r_e = _r_eff_of_z(z_in)
        return 0.5 * (r_e @ r_e) + 0.5 * (z_in @ z_in)

    sol_z = optx.least_squares(
        lambda z_in, _: jnp.concatenate([_r_eff_of_z(z_in), z_in], axis=0),
        optx.LevenbergMarquardt(rtol=1e-9, atol=1e-9),
        jax.lax.stop_gradient(lgf.latent),
        max_steps=50,
        throw=False,
    )
    z_anchor = jax.lax.stop_gradient(sol_z.value)
    z_c = z_anchor
    for _ in range(2):
        g_z = jax.grad(_ham_z)(z_c)
        k_zz = jax.hessian(_ham_z)(z_c)
        k_zz = 0.5 * (k_zz + k_zz.T)
        z_c = z_c - jnp.linalg.solve(k_zz, g_z)

    r_map = _r_eff_of_z(z_c)
    j_map = jax.jacfwd(_r_eff_of_z)(z_c)
    chol = jnp.linalg.cholesky(jnp.eye(lgf.n_grid, dtype=jnp.float64) + j_map.T @ j_map)
    half_ld_field = jnp.sum(jnp.log(jnp.diag(chol)))
    return const_n - 0.5 * (r_map @ r_map) - 0.5 * (z_c @ z_c) - half_ld_field


def marginal_log_likelihood(
    model: RadiocarbonModel,
    dataset: Dataset,
    cfg: LinearMarginalConfig | None = None,
    *,
    sigma_jitter: ArrayLike | None = None,
) -> jax.Array:
    """Exact marginal Gaussian log-likelihood (or Laplace-marginalized for ``LogGaussianField``)."""
    if cfg is None:
        cfg = LinearMarginalConfig()

    sigma = dataset.sigma_flat
    if sigma_jitter is not None:
        sigma = jnp.sqrt(
            jnp.square(sigma)
            + jnp.square(jnp.asarray(sigma_jitter, dtype=jnp.float64))
        )

    if cfg.marginalise_field and not isinstance(model.solver, ModaxSolver):
        lgf, other_parts = _extract_single_lgf(model.production)
        if lgf is not None:
            return _laplace_marginal_log_likelihood_lgf(
                model, dataset, lgf, other_parts, cfg, sigma
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
    target_space: Literal["unconstrained", "constrained"] = "unconstrained",
) -> jax.Array:
    """Marginal log-posterior in ``u``-space (default) or induced ``theta``-space."""
    constrained_model = parameter_space.constrain(u_vec, model)
    log_lk = marginal_log_likelihood(
        constrained_model, dataset, cfg, sigma_jitter=sigma_jitter
    )
    log_pr = parameter_space.log_prior_unconstrained(
        u_vec, target_space=target_space
    )
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
        marginalise_field=False,
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
    target_space: Literal["unconstrained", "constrained"] = "unconstrained",
) -> jax.Array:
    """Joint log-posterior in ``u``-space (default) or induced ``theta``-space."""
    constrained_model = parameter_space.constrain(u_vec, model)
    log_lk = joint_log_likelihood(
        constrained_model, dataset, cfg=cfg, sigma_jitter=sigma_jitter
    )
    log_pr = parameter_space.log_prior_unconstrained(
        u_vec, target_space=target_space
    )
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

    coeff_draws: jax.Array | None = None
    """Joint conditional Gaussian coefficient draws ``(n_draws, k_total)``, or ``None``."""

    initial_state_mean: jax.Array | None = None
    """Posterior mean of initial reservoir perturbation ``delta_y(t_0)`` (``14C kg``), shape ``(n_boxes,)``, or ``None`` when no single shared initial-state anchor exists."""

    initial_state_sd: jax.Array | None = None
    """Marginal posterior SD of ``delta_y(t_0)`` (``14C kg``), shape ``(n_boxes,)``, or ``None`` when no single shared initial-state anchor exists."""

    initial_state_d14c_mean: jax.Array | None = None
    """Posterior mean of initial reservoir ``Delta14C(t_0)`` perturbation (‰), shape ``(n_boxes,)``, or ``None`` when no single shared initial-state anchor exists."""

    initial_state_d14c_sd: jax.Array | None = None
    """Marginal posterior SD of initial reservoir ``Delta14C(t_0)`` (‰), shape ``(n_boxes,)``, or ``None`` when no single shared initial-state anchor exists."""


def _offset_column_count(cfg: LinearMarginalConfig, n_series: int) -> int:
    if cfg.offset_mode == "hierarchical":
        return 1 + (n_series if cfg.lab_prior_sd > 0.0 and n_series > 0 else 0)
    if cfg.offset_mode == "shared":
        return 1
    if cfg.offset_mode == "per_series":
        return n_series
    return 0


def _reconstruct_initial_state_posterior(
    model: RadiocarbonModel,
    init_state_blocks: list[tuple[slice, jax.Array, jax.Array]],
    coeff_mean: jax.Array,
    coeff_cov: jax.Array,
) -> tuple[
    jax.Array | None,
    jax.Array | None,
    jax.Array | None,
    jax.Array | None,
]:
    """Reconstruct initial-state posterior moments when all blocks share one anchor epoch."""
    if not init_state_blocks:
        return None, None, None, None
    if len(init_state_blocks) == 1:
        anc_slice, anc_root, _ = init_state_blocks[0]
        initial_state_mean = anc_root @ coeff_mean[anc_slice]
        cov_y0 = anc_root @ coeff_cov[anc_slice, anc_slice] @ anc_root.T
    else:
        concrete_anchors = [
            float(c)
            for _, _, t_anc in init_state_blocks
            if (c := _concrete(t_anc)) is not None
        ]
        if len(concrete_anchors) > 1 and any(
            abs(t - concrete_anchors[0]) > 1e-9 for t in concrete_anchors[1:]
        ):
            return None, None, None, None
        idx_anc = jnp.concatenate(
            [
                jnp.arange(sl.start, sl.stop, dtype=jnp.int32)
                for sl, _, _ in init_state_blocks
            ]
        )
        anc_root_all = jnp.concatenate([r for _, r, _ in init_state_blocks], axis=1)
        initial_state_mean = anc_root_all @ coeff_mean[idx_anc]
        cov_y0 = anc_root_all @ coeff_cov[jnp.ix_(idx_anc, idx_anc)] @ anc_root_all.T

    cov_y0 = 0.5 * (cov_y0 + cov_y0.T)
    initial_state_sd = jnp.sqrt(jnp.maximum(jnp.diag(cov_y0), 0.0))

    y_ss_init = model.cbm.steady_state_c14(
        q_std=model.q_std, alpha_unit=model.alpha_unit
    )
    trop_idx_init = jnp.asarray(
        model.cbm.topology.trop_box_indices, dtype=jnp.int32
    )
    y_trop_init = jnp.sum(y_ss_init[trop_idx_init])
    m_rel_init = model.cbm.relative_reservoir_masses()
    scale_all_boxes = 1000.0 / (y_trop_init * m_rel_init)
    return (
        initial_state_mean,
        initial_state_sd,
        scale_all_boxes * initial_state_mean,
        scale_all_boxes * initial_state_sd,
    )


def _linear_posterior_impl(
    model: RadiocarbonModel,
    dataset: Dataset,
    cfg: LinearMarginalConfig,
    rng_key: jax.Array | None,
    n_draws: int,
    t_fine_arr: jax.Array | None,
    hemisphere: Literal["north", "south"],
    box: int | None,
    sigma_jitter_arr: jax.Array | None,
) -> LinearPosteriorResult:
    sigma = dataset.sigma_flat
    if sigma_jitter_arr is not None:
        sigma = jnp.sqrt(jnp.square(sigma) + jnp.square(sigma_jitter_arr))

    resid = _nonlinear_residual(model, dataset, cfg)
    X, prior_mean, prior_sd, gp_projs, field_init_roots, init_state_blocks = (
        _build_marginal_design_with_projections(model, dataset, cfg)
    )

    lm = LinearMarginal(
        X,
        prior_mean=prior_mean,
        prior_sd=prior_sd,
        method=cfg.whiten_method,
    )
    coeff_mean, coeff_cov = lm.posterior(resid, sigma)
    ring_pred_mean = dataset.d14c_flat - resid + X @ coeff_mean

    # --- Reconstruct initial-state posterior vector (delta_y(t_0) across all boxes) ---
    (
        initial_state_mean,
        initial_state_sd,
        initial_state_d14c_mean,
        initial_state_d14c_sd,
    ) = _reconstruct_initial_state_posterior(
        model, init_state_blocks, coeff_mean, coeff_cov
    )

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

    # --- Optional joint conditional Gaussian coefficient draws ---
    coeff_draws: jax.Array | None = None
    if n_draws > 0:
        assert rng_key is not None
        k_tot = coeff_mean.shape[0]
        if k_tot > 0:
            jitter_eye = 1e-12 * jnp.eye(k_tot, dtype=jnp.float64)
            L_cov = jnp.linalg.cholesky(0.5 * (coeff_cov + coeff_cov.T) + jitter_eye)
            z = jax.random.normal(
                rng_key, shape=(n_draws, k_tot), dtype=jnp.float64
            )
            coeff_draws = coeff_mean[None, :] + z @ L_cov.T
        else:
            coeff_draws = jnp.zeros((n_draws, 0), dtype=jnp.float64)

    # --- Optional continuous Q(t) and Delta14C(t) reconstruction on t_fine ---
    q_fine_mean: jax.Array | None = None
    q_fine_draws: jax.Array | None = None
    d14c_fine_mean: jax.Array | None = None
    d14c_fine_draws: jax.Array | None = None
    q_fine_std: jax.Array | None = None
    d14c_fine_std: jax.Array | None = None

    if t_fine_arr is not None:
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

        unmarg, solar_terms, trend_terms, _, field_terms = _partition_terms_for_cfg(
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

            modal_fine = _modal_decomposition(A, cbm=cbm)
            u_mat, v_mat = _solve_complex_resolvents(A, b_eff, proj.omegas)
            u_box = scale_box * u_mat[target_box]
            v_box = scale_box * v_mat[target_box]
            dt_a = jnp.maximum(t_fine_arr - proj.a, 0.0)
            if modal_fine is not None:
                lam_f, v_eig_f, v_inv_f = modal_fine
                expm_a_row = jnp.real(
                    (
                        v_eig_f[target_box, :][None, :]
                        * jnp.exp(dt_a[:, None] * lam_f[None, :])
                    )
                    @ v_inv_f
                )
            else:
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
                if modal_fine is not None:
                    lam_f, v_eig_f, v_inv_f = modal_fine
                    expm_anc_row = jnp.real(
                        (
                            v_eig_f[target_box, :][None, :]
                            * jnp.exp(dt_anc[:, None] * lam_f[None, :])
                        )
                        @ v_inv_f
                    )
                else:
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

        # 5. Marginalized GaussianField columns
        modal_fine_gf = _modal_decomposition(A, cbm=cbm) if field_terms else None
        for gf, root_y0_gf in zip(field_terms, field_init_roots):
            n_g = gf.n_grid
            amp_gf = _safe_sqrt(gf.power_spectrum())
            b_knots = idct2(jnp.eye(n_g, dtype=jnp.float64), axis=0) * amp_gf[None, :]
            if gf.template is not None:
                b_knots = jnp.asarray(gf.template, dtype=jnp.float64)[:, None] * b_knots
            in_win = (t_fine_arr >= gf.t_start) & (t_fine_arr <= gf.t_end)
            pos = jnp.clip((t_fine_arr - gf.t_start) / gf.dt, 0.0, float(n_g - 1))
            idx0 = jnp.clip(jnp.floor(pos).astype(jnp.int32), 0, n_g - 2)
            alpha_w = pos - idx0.astype(jnp.float64)
            phi_q_gf = jnp.where(
                in_win[:, None],
                (1.0 - alpha_w)[:, None] * b_knots[idx0]
                + alpha_w[:, None] * b_knots[idx0 + 1],
                0.0,
            )
            phi_q_cols.append(phi_q_gf)

            def _gf_d14c_of_z(z_in: jax.Array, _gf: GaussianField = gf) -> jax.Array:
                dy_gf = model.solver.solve_delta_trajectory(
                    cbm,
                    _gf.with_latent(z_in),
                    t_fine_arr,
                    q_std=model.q_std,
                    alpha_unit=model.alpha_unit,
                )
                return dy_gf[:, target_box] * scale_box

            phi_d14c_gf = jax.jacfwd(_gf_d14c_of_z)(jnp.zeros((n_g,), dtype=jnp.float64))
            phi_d14c_cols.append(phi_d14c_gf)

            if root_y0_gf is not None:
                m_rank = root_y0_gf.shape[1]
                phi_q_cols.append(jnp.zeros((n_fine, m_rank), dtype=jnp.float64))
                dt_start = jnp.maximum(t_fine_arr - gf.t_start, 0.0)
                if modal_fine_gf is not None:
                    lam_f, v_eig_f, v_inv_f = modal_fine_gf
                    expm_tstart_row = jnp.real(
                        (
                            v_eig_f[target_box, :][None, :]
                            * jnp.exp(dt_start[:, None] * lam_f[None, :])
                        )
                        @ v_inv_f
                    )
                else:
                    expm_tstart_row = jax.vmap(
                        lambda dt: jax.scipy.linalg.expm(A * dt)[target_box]
                    )(dt_start)
                phi_d14c_init = jnp.where(
                    (t_fine_arr >= gf.t_start)[:, None],
                    scale_box * (expm_tstart_row @ root_y0_gf),
                    0.0,
                )
                phi_d14c_cols.append(phi_d14c_init)

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

        if coeff_draws is not None:
            if coeff_draws.shape[1] > 0:
                q_fine_draws = q_base[None, :] + coeff_draws @ phi_q.T
                d14c_fine_draws = d14c_base[None, :] + coeff_draws @ phi_d14c.T
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
        coeff_draws=coeff_draws,
        initial_state_mean=initial_state_mean,
        initial_state_sd=initial_state_sd,
        initial_state_d14c_mean=initial_state_d14c_mean,
        initial_state_d14c_sd=initial_state_d14c_sd,
    )


_linear_posterior_jit = eqx.filter_jit(_linear_posterior_impl)


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
    cfg_eff = LinearMarginalConfig() if cfg is None else cfg
    n_draws_int = int(n_draws)
    if n_draws_int > 0 and rng_key is None:
        raise ValueError("rng_key is required when n_draws > 0.")
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
    box_int = int(box) if box is not None else None
    _, _, _, gp_terms, field_terms = _partition_terms_for_cfg(model.production, cfg_eff)
    field_anchors = (
        [float(gf.t_start) for gf in field_terms]
        if cfg_eff.include_field_initial_cov and dataset.n_obs > 0
        else []
    )
    anchors_match = True
    if len(field_anchors) > 1 and any(
        abs(t - field_anchors[0]) > 1e-9 for t in field_anchors[1:]
    ):
        anchors_match = False
    elif gp_terms and cfg_eff.include_gp_initial_cov and field_anchors:
        # _project_single_gp_dataset uses project_causal_gp's default initial_anchor="t0",
        # so every GP term's anchor epoch equals min(sub_t_starts) across the dataset.
        t0_c = _concrete(
            jnp.min(
                jnp.stack(
                    [jnp.min(s.sub_t_starts) for s in dataset.series if s.n_obs > 0]
                )
            )
        )
        anchors_match = (
            t0_c is not None and abs(float(t0_c) - field_anchors[0]) <= 1e-9
        )

    impl = (
        _linear_posterior_impl
        if isinstance(model.solver, ModaxSolver)
        else _linear_posterior_jit
    )
    res = impl(
        model,
        dataset,
        cfg_eff,
        rng_key,
        n_draws_int,
        t_fine_arr,
        hemisphere,
        box_int,
        jit_arr,
    )
    if not anchors_match:
        return res._replace(
            initial_state_mean=None,
            initial_state_sd=None,
            initial_state_d14c_mean=None,
            initial_state_d14c_sd=None,
        )
    return res
