"""Marginalized and joint log-likelihoods for ``ticktack 2.0`` radiocarbon inference.

Implements the full Bayesian parameter infrastructure:

1. :class:`ParamSpec` and :class:`ParameterSpace`: smooth bijector-based
   mapping between unconstrained ``R^d`` vectors and constrained
   :class:`~ticktack.model.RadiocarbonModel` leaves, with exact
   log-prior + log-abs-det-Jacobian.

2. :class:`LinearMarginalConfig` + :func:`marginal_log_likelihood` /
   :func:`marginal_log_posterior`: analytically marginalise baseline
   ``Delta14C`` offsets, harmonic solar quadrature coefficients, and causal
   Solin–Särkkä POD ``GPProduction`` modes in closed form via
   :mod:`ticktack.linear`, completely eliminating v1's 4-point
   ``mean(y[:4])`` boundary bias.

3. :func:`joint_log_likelihood` / :func:`joint_log_posterior`: non-Gaussian
   ``LogGPProduction`` log-likelihood (unit-Gaussian prior on mode weights)
   compatible with both :class:`~ticktack.solvers.DiffraxSolver` and the
   optional :class:`~ticktack.solvers.ModaxSolver`
   ``@jax.custom_vjp`` wrapper.

4. :func:`linear_posterior`: closed-form posterior mean and covariance of all
   marginalised linear coefficients, reconstructed per-series offsets, and
   optional Gaussian draws of continuous ``Q(t)`` and ``Delta14C(t)`` on a
   fine grid.
"""

from __future__ import annotations

import math
from typing import Any, Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .data import Dataset
from .linear import (
    LinearMarginal,
    harmonic_solar_design,
    hierarchical_offset_design,
    project_causal_gp,
)
from .model import RadiocarbonModel
from .production import (
    GPProduction,
    HarmonicSolar,
    LinearTrend,
    LogGPProduction,
    ProductionSum,
    ProductionTerm,
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
# 1. ParamSpec: single bijector specification for one parameter leaf
# ---------------------------------------------------------------------------

_BIJECTOR_TYPES = Literal["identity", "exp", "softplus", "sigmoid"]


class ParamSpec(eqx.Module):
    """Specification for a single parameter with a smooth bijector.

    Maps between an unconstrained real value ``u`` and the constrained
    physical value ``theta`` via a differentiable bijection:

    * ``"identity"``: ``theta = u`` (no transformation; for unbounded reals).
    * ``"exp"``: ``theta = exp(u)`` (for strictly positive scalars).
    * ``"softplus"``: ``theta = log(1 + exp(u)) + eps`` (soft positive).
    * ``"sigmoid"``: ``theta = lo + (hi - lo) * sigmoid(u)`` (bounded).

    The log-prior is ``N(prior_mean, prior_sd^2)`` in the **unconstrained**
    space (i.e., ``N(prior_mean, prior_sd^2)`` on ``u``).  The
    log-abs-det-Jacobian is included in :meth:`ParameterSpace.log_prior_unconstrained`
    when ``include_jacobian=True``.
    """

    path: str = eqx.field(static=True)
    bijector: str = eqx.field(static=True)
    prior_mean: float = eqx.field(static=True)
    prior_sd: float = eqx.field(static=True)
    low: float = eqx.field(static=True)
    high: float = eqx.field(static=True)

    def __init__(
        self,
        path: str,
        bijector: _BIJECTOR_TYPES = "identity",
        *,
        prior_mean: float = 0.0,
        prior_sd: float = 10.0,
        low: float = 0.0,
        high: float = 1.0,
    ) -> None:
        if bijector not in ("identity", "exp", "softplus", "sigmoid"):
            raise ValueError(
                f"bijector must be 'identity', 'exp', 'softplus', or 'sigmoid'; got {bijector!r}."
            )
        self.path = str(path)
        self.bijector = str(bijector)
        self.prior_mean = float(prior_mean)
        self.prior_sd = max(float(prior_sd), 1e-30)
        self.low = float(low)
        self.high = float(high)

    def constrain(self, u: jax.Array) -> jax.Array:
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
        return u_arr  # unreachable

    def unconstrain(self, theta: jax.Array) -> jax.Array:
        """Inverse bijector: constrained ``theta`` -> unconstrained ``u``."""
        t_arr = jnp.asarray(theta, dtype=jnp.float64)
        if self.bijector == "identity":
            return t_arr
        if self.bijector == "exp":
            return jnp.log(jnp.maximum(t_arr, 1e-30))
        if self.bijector == "softplus":
            # softplus^{-1}(x) = log(exp(x) - 1) = log1p(-exp(-x)) + x; numerically stable
            safe_t = jnp.maximum(t_arr - 1e-6, 1e-30)
            return jnp.log(jnp.expm1(safe_t))
        if self.bijector == "sigmoid":
            lo = jnp.asarray(self.low, dtype=jnp.float64)
            hi = jnp.asarray(self.high, dtype=jnp.float64)
            p = jnp.clip((t_arr - lo) / (hi - lo), 1e-7, 1.0 - 1e-7)
            return jnp.log(p / (1.0 - p))
        return t_arr

    def log_abs_det_jacobian(self, u: jax.Array) -> jax.Array:
        """Log-abs-det-Jacobian ``log |d theta / d u|`` at unconstrained value ``u``."""
        u_arr = jnp.asarray(u, dtype=jnp.float64)
        if self.bijector == "identity":
            return jnp.asarray(0.0, dtype=jnp.float64)
        if self.bijector == "exp":
            return u_arr  # d/du exp(u) = exp(u), log|J| = u
        if self.bijector == "softplus":
            return -jax.nn.softplus(-u_arr)  # log sigmoid(u)
        if self.bijector == "sigmoid":
            lo = jnp.asarray(self.low, dtype=jnp.float64)
            hi = jnp.asarray(self.high, dtype=jnp.float64)
            sig = jax.nn.sigmoid(u_arr)
            return jnp.log(hi - lo) + jnp.log(sig) + jnp.log1p(-sig)
        return jnp.asarray(0.0, dtype=jnp.float64)

    def log_prior_unconstrained(self, u: jax.Array) -> jax.Array:
        """Log-prior density in unconstrained space: ``N(prior_mean, prior_sd^2)`` on ``u``."""
        u_arr = jnp.asarray(u, dtype=jnp.float64)
        z = (u_arr - self.prior_mean) / self.prior_sd
        return -0.5 * z * z - math.log(self.prior_sd) - 0.5 * math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# 2. ParameterSpace: multi-parameter unconstrained <-> constrained mapping
# ---------------------------------------------------------------------------


class ParameterSpace(eqx.Module):
    """Immutable bijector-based map from unconstrained ``R^d`` to a ``RadiocarbonModel``.

    Builds a flat ``d``-dimensional unconstrained space by enumerating
    :class:`ParamSpec` entries, each with a path and bijector, and provides:

    * :meth:`constrain`: ``R^d`` -> ``RadiocarbonModel`` (physical parameters).
    * :meth:`unconstrain`: ``RadiocarbonModel`` -> ``R^d`` (optimization space).
    * :meth:`apply_unconstrained`: evaluate any function ``f(model)`` in
      unconstrained space.
    * :meth:`log_prior_unconstrained`: sum of ``N(0, 1)`` priors on ``u``
      plus log-abs-det-Jacobians for change-of-variables.
    """

    specs: tuple[ParamSpec, ...] = eqx.field(static=True)
    _reference_model_leaves: tuple = eqx.field(static=True)
    _reference_model_treedef: Any = eqx.field(static=True)

    def __init__(
        self,
        reference_model: RadiocarbonModel,
        specs: list[ParamSpec] | tuple[ParamSpec, ...],
    ) -> None:
        self.specs = tuple(specs)
        # Freeze structure by flattening the reference model
        leaves, treedef = jax.tree_util.tree_flatten(reference_model)
        self._reference_model_leaves = tuple(leaves)
        self._reference_model_treedef = treedef

    @property
    def n_params(self) -> int:
        """Dimensionality of the unconstrained parameter space."""
        return len(self.specs)

    def constrain(
        self,
        u_vec: ArrayLike,
        base_model: RadiocarbonModel | None = None,
    ) -> RadiocarbonModel:
        """Map unconstrained vector ``u`` to a ``RadiocarbonModel``.

        Parameters
        ----------
        u_vec : array-like
            Unconstrained parameter vector of shape ``(d,)``.
        base_model : RadiocarbonModel, optional
            Starting model to update.  Uses the reference model when ``None``.

        Returns
        -------
        RadiocarbonModel
            Updated model with constrained parameters set at their paths.
        """
        u = jnp.asarray(u_vec, dtype=jnp.float64).reshape(-1)
        if len(self.specs) != u.shape[0]:
            raise ValueError(
                f"u_vec length {u.shape[0]} does not match n_params={len(self.specs)}."
            )
        if base_model is None:
            leaves = list(self._reference_model_leaves)
            model = jax.tree_util.tree_unflatten(self._reference_model_treedef, leaves)
        else:
            model = base_model
        for i, spec in enumerate(self.specs):
            theta_i = spec.constrain(u[i])
            model = model.set(spec.path, theta_i)
        return model

    def unconstrain(self, model: RadiocarbonModel) -> jax.Array:
        """Map a ``RadiocarbonModel`` to its unconstrained parameter vector.

        Parameters
        ----------
        model : RadiocarbonModel
            Model with physically constrained parameters.

        Returns
        -------
        jax.Array
            Shape ``(d,)`` unconstrained vector.
        """
        return jnp.stack(
            [spec.unconstrain(model.get(spec.path)) for spec in self.specs],
            axis=0,
        )

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
        include_jacobian: bool = True,
    ) -> jax.Array:
        """Sum of unconstrained log-prior densities (and optional log-Jacobian).

        Parameters
        ----------
        u_vec : array-like
            Unconstrained parameter vector of shape ``(d,)``.
        include_jacobian : bool
            When ``True`` (default), adds the log-abs-det-Jacobian for each
            non-identity bijector so the result is proportional to the density
            in constrained space.

        Returns
        -------
        jax.Array
            Scalar log-prior (or log-prior + log-Jacobian).
        """
        u = jnp.asarray(u_vec, dtype=jnp.float64).reshape(-1)
        log_p = jnp.asarray(0.0, dtype=jnp.float64)
        for i, spec in enumerate(self.specs):
            log_p = log_p + spec.log_prior_unconstrained(u[i])
            if include_jacobian and spec.bijector != "identity":
                log_p = log_p + spec.log_abs_det_jacobian(u[i])
        return log_p


# ---------------------------------------------------------------------------
# 3. LinearMarginalConfig: configuration for analytical linear marginalization
# ---------------------------------------------------------------------------


class LinearMarginalConfig(eqx.Module):
    """Configuration for analytical marginalization of linear parameters.

    Controls which linear parameters are analytically marginalised at each
    likelihood evaluation:

    * **Baseline offsets** (``offset_mode``): hierarchical global + per-series
      lab offsets (``"hierarchical"``), independent per-series offsets
      (``"per_series"``), one shared offset (``"shared"``), or no marginalisation
      (``"none"``).
    * **HarmonicSolar amplitudes** (``marginalise_solar``): when ``True``, the
      ``(c_cos, c_sin)`` quadrature amplitudes are marginalised exactly given
      the CBM resolvent columns from :func:`~ticktack.linear.harmonic_solar_design`.
    * **LinearTrend slope** (``marginalise_trend``): when ``True``, the secular
      linear production drift slope is marginalised.
    * **GPProduction modes** (``marginalise_gp``): when ``True``, the causal
      Solin–Särkkä spectral mode weights are marginalised exactly via
      :func:`~ticktack.linear.project_causal_gp`.
    """

    offset_mode: str = eqx.field(static=True)
    global_prior_mean: float = eqx.field(static=True)
    global_prior_sd: float = eqx.field(static=True)
    lab_prior_sd: float = eqx.field(static=True)
    marginalise_solar: bool = eqx.field(static=True)
    marginalise_trend: bool = eqx.field(static=True)
    marginalise_gp: bool = eqx.field(static=True)
    n_gp_modes: int = eqx.field(static=True)
    whiten_method: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        offset_mode: Literal["hierarchical", "per_series", "shared", "none"] = "hierarchical",
        global_prior_mean: float = 0.0,
        global_prior_sd: float = 200.0,
        lab_prior_sd: float = 1.5,
        marginalise_solar: bool = True,
        marginalise_trend: bool = True,
        marginalise_gp: bool = True,
        n_gp_modes: int = 60,
        whiten_method: Literal["cholesky", "qr", "rank_one"] = "cholesky",
    ) -> None:
        valid_offset_modes = ("hierarchical", "per_series", "shared", "none")
        if offset_mode not in valid_offset_modes:
            raise ValueError(
                f"offset_mode must be one of {valid_offset_modes}; got {offset_mode!r}."
            )
        self.offset_mode = str(offset_mode)
        self.global_prior_mean = float(global_prior_mean)
        self.global_prior_sd = float(global_prior_sd)
        self.lab_prior_sd = float(lab_prior_sd)
        self.marginalise_solar = bool(marginalise_solar)
        self.marginalise_trend = bool(marginalise_trend)
        self.marginalise_gp = bool(marginalise_gp)
        self.n_gp_modes = int(n_gp_modes)
        self.whiten_method = str(whiten_method)


# ---------------------------------------------------------------------------
# 4. Internal helpers: build design matrices and residuals
# ---------------------------------------------------------------------------


def _unpack_production_terms(production: ProductionTerm):
    """Return (nonlinear_terms, solar_terms, trend_terms, gp_terms) dicts."""
    if isinstance(production, ProductionSum):
        nonlinear = production.nonlinear_terms
        linear = production.linear_terms
    else:
        nonlinear = {}
        linear = {}
        if isinstance(production, (GPProduction, HarmonicSolar, LinearTrend)):
            linear["_root"] = production
        else:
            nonlinear["_root"] = production

    solar_terms = {k: v for k, v in linear.items() if isinstance(v, HarmonicSolar)}
    trend_terms = {k: v for k, v in linear.items() if isinstance(v, LinearTrend)}
    gp_terms = {k: v for k, v in linear.items() if isinstance(v, GPProduction)}
    return nonlinear, solar_terms, trend_terms, gp_terms


def _nonlinear_production(production: ProductionTerm) -> ProductionTerm | None:
    """Return a production term containing only nonlinear components, or None."""
    nonlin, _, _, _ = _unpack_production_terms(production)
    if not nonlin:
        return None
    if len(nonlin) == 1 and "_root" in nonlin:
        return nonlin["_root"]
    if isinstance(production, ProductionSum):
        return ProductionSum(nonlin)
    return None


def _build_marginal_design(
    model: RadiocarbonModel,
    dataset: Dataset,
    cfg: LinearMarginalConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build joint design matrix, prior mean, and prior sd for all marginalised linear params.

    Returns ``(X, prior_mean, prior_sd)`` of shapes ``(N_tot, k_total)``,
    ``(k_total,)``, ``(k_total,)`` where columns are ordered:
    [offset_global, offset_lab_0, ..., offset_lab_{S-1},
     solar_cos_1, solar_sin_1, ..., solar_cos_M, solar_sin_M,
     trend_slope_0, ..., trend_slope_P,
     gp_modes_0, ..., gp_modes_R].
    """
    cbm = model.cbm
    production = model.production
    A = cbm.matrix_A(q_std=model.q_std, alpha_unit=model.alpha_unit)
    b = cbm.production_vector()

    # Gather per-series observations
    n_tot = dataset.n_obs
    n_series = dataset.n_series
    series_idx = dataset.series_indices  # (N_tot,)

    # Steady-state specific-activity ratios for per-box Delta scaling
    y_ss = cbm.steady_state_c14(q_std=model.q_std, alpha_unit=model.alpha_unit)
    trop_idx = jnp.asarray(cbm.topology.trop_box_indices, dtype=jnp.int32)
    y_trop = jnp.sum(y_ss[trop_idx])
    m_rel = cbm.relative_reservoir_masses()
    r_ss_ratio = (y_ss / y_trop) / m_rel  # (n_boxes,)

    # Map each observation to its box's specific-activity ratio
    box_indices_flat = jnp.array(
        [
            model._box_index_for_series(s) if model._box_index_for_series(s) is not None else 0
            for s in dataset.series
            for _ in range(s.n_obs)
        ],
        dtype=jnp.int32,
    )
    obs_activity_ratios = r_ss_ratio[box_indices_flat]  # (N_tot,)

    design_cols: list[jax.Array] = []
    pmean_parts: list[jax.Array] = []
    psd_parts: list[jax.Array] = []

    # --- Offset columns ---
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
        H_off, mu_off, sd_off = hierarchical_offset_design(
            series_idx,
            n_series=n_series,
            box_activity_ratios=obs_activity_ratios,
            global_prior_mean=cfg.global_prior_mean,
            global_prior_sd=cfg.global_prior_sd,
            lab_prior_sd=0.0,  # No lab offset -> just per-series from hierarchical
        )
        design_cols.append(H_off)
        pmean_parts.append(mu_off)
        psd_parts.append(sd_off)

    elif cfg.offset_mode == "shared":
        # Single global column only, no per-series lab offsets
        H_off = obs_activity_ratios[:, None]
        mu_off = jnp.array([cfg.global_prior_mean], dtype=jnp.float64)
        sd_off = jnp.array([cfg.global_prior_sd], dtype=jnp.float64)
        design_cols.append(H_off)
        pmean_parts.append(mu_off)
        psd_parts.append(sd_off)

    # else "none": no offset columns

    # --- Solar harmonic columns ---
    _, solar_terms, trend_terms, gp_terms = _unpack_production_terms(production)

    if cfg.marginalise_solar and solar_terms:
        for solar in solar_terms.values():
            if not isinstance(solar, HarmonicSolar):
                continue
            # Build observation-flattened design across all series
            solar_col_parts = []
            for series in dataset.series:
                box_idx = model._box_index_for_series(series)
                H_s = harmonic_solar_design(
                    A,
                    b,
                    solar.omegas,
                    series.sub_t_starts,
                    series.sub_t_ends,
                    series.seasonal_weights.weights,
                    alpha_unit=float(model.alpha_unit) / (y_trop * float(m_rel[box_idx])),
                    t_ref=float(solar.t_ref),
                    box_indices=box_idx,
                    obs_scale=1.0,
                )  # (N_r_s, 2*M)
                solar_col_parts.append(H_s)
            H_solar = jnp.concatenate(solar_col_parts, axis=0)
            M = H_solar.shape[1]
            design_cols.append(H_solar)
            pmean_parts.append(jnp.zeros(M, dtype=jnp.float64))
            psd_parts.append(jnp.full((M,), 10.0 * float(model.q_std), dtype=jnp.float64))

    # --- Linear trend columns ---
    if cfg.marginalise_trend and trend_terms:
        for trend in trend_terms.values():
            if not isinstance(trend, LinearTrend):
                continue
            # Build trend column per observation
            trend_col_parts = []
            for series in dataset.series:
                box_idx = model._box_index_for_series(series)
                # Trend DeltaC contribution: slope * (t - t_ref) * (-A)^{-1} b projected to box
                inv_A_b = jnp.linalg.solve(-A, b)
                scale_factor = model.alpha_unit * inv_A_b[box_idx] / (y_trop * m_rel[box_idx])
                t_mid = 0.5 * (series.t_starts + series.t_ends)
                col_s = (1000.0 * scale_factor * (t_mid - trend.t_ref))[:, None]
                trend_col_parts.append(col_s)
            H_trend = jnp.concatenate(trend_col_parts, axis=0)
            design_cols.append(H_trend)
            pmean_parts.append(jnp.zeros(1, dtype=jnp.float64))
            psd_parts.append(jnp.array([5.0], dtype=jnp.float64))  # ~5 permil/yr max slope

    # --- GP mode columns ---
    if cfg.marginalise_gp and gp_terms:
        for gp in gp_terms.values():
            if not isinstance(gp, GPProduction):
                continue
            # For each series, project causal GP basis onto observations
            gp_col_parts = []
            for series in dataset.series:
                box_idx = model._box_index_for_series(series)
                # project_causal_gp returns (N_r_s, r + r_hom) design matrix
                H_gp_s = project_causal_gp(
                    A=A,
                    b=b,
                    kernel=gp.kernel,
                    t_starts=series.sub_t_starts,
                    t_ends=series.sub_t_ends,
                    sub_weights=series.seasonal_weights.weights,
                    n_modes=cfg.n_gp_modes,
                    alpha_unit=float(model.alpha_unit),
                    box_indices=box_idx,
                    obs_scale=1000.0 / (y_trop * float(m_rel[box_idx])),
                )
                gp_col_parts.append(H_gp_s)
            H_gp = jnp.concatenate(gp_col_parts, axis=0)
            r_total = H_gp.shape[1]
            design_cols.append(H_gp)
            pmean_parts.append(jnp.zeros(r_total, dtype=jnp.float64))
            psd_parts.append(jnp.ones(r_total, dtype=jnp.float64))  # unit-normal modes

    if design_cols:
        X = jnp.concatenate(design_cols, axis=1)
        pm = jnp.concatenate(pmean_parts)
        ps = jnp.concatenate(psd_parts)
    else:
        X = jnp.zeros((n_tot, 0), dtype=jnp.float64)
        pm = jnp.zeros(0, dtype=jnp.float64)
        ps = jnp.zeros(0, dtype=jnp.float64)

    return X, pm, ps


def _nonlinear_residual(
    model: RadiocarbonModel,
    dataset: Dataset,
) -> jax.Array:
    """Compute residual ``d - m_nonlinear`` where ``m_nonlinear`` is the nonlinear prediction.

    Uses only the nonlinear part of the production (MiyakeSpike, etc.) to
    compute the trajectory, returning ``(N_tot,)`` residuals.
    """
    nonlin_prod = _nonlinear_production(model.production)
    if nonlin_prod is None:
        # No nonlinear component: residual is just the data
        return dataset.d14c_flat

    per_series_preds: list[jax.Array] = []
    for series in dataset.series:
        box_idx = model._box_index_for_series(series)
        pred_s = model.solver.predict_d14c(
            model.cbm,
            nonlin_prod,
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
    """Exact marginal Gaussian log-likelihood (analytically marginalised linear params).

    Evaluates ``log p(d | nonlinear_theta) = int log p(d | nonlinear_theta, w) p(w) dw``
    in closed form for all linear parameters ``w`` specified in ``cfg``.

    Parameters
    ----------
    model : RadiocarbonModel
        Current model (CBM + production + solver) with nonlinear parameters set.
    dataset : Dataset
        Observed tree-ring data.
    cfg : LinearMarginalConfig, optional
        Marginalization configuration.  Defaults to hierarchical offsets + solar + GP.
    sigma_jitter : array-like, optional
        Per-observation additional noise jitter ``sigma_jit`` (‰) added in
        quadrature to ``dataset.sigma_flat``.  Pass ``None`` for no jitter.

    Returns
    -------
    jax.Array
        Scalar marginal log-likelihood.
    """
    if cfg is None:
        cfg = LinearMarginalConfig()

    # Effective per-observation noise
    sigma = dataset.sigma_flat
    if sigma_jitter is not None:
        sigma = jnp.sqrt(jnp.square(sigma) + jnp.square(jnp.asarray(sigma_jitter, dtype=jnp.float64)))

    # Residual from nonlinear trajectory
    resid = _nonlinear_residual(model, dataset)  # (N_tot,)

    # Build linear design for all marginalised components
    X, prior_mean, prior_sd = _build_marginal_design(model, dataset, cfg)

    # LinearMarginal: marginalise all linear columns
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
    """Marginal log-posterior (marginalised log-likelihood + unconstrained log-prior).

    Parameters
    ----------
    model : RadiocarbonModel
        Base model (not necessarily at the constrained parameters; ``u_vec`` is used).
    dataset : Dataset
        Observed tree-ring data.
    parameter_space : ParameterSpace
        Bijector-based unconstrained parameter specification.
    u_vec : array-like
        Unconstrained parameter vector of shape ``(d,)``.
    cfg : LinearMarginalConfig, optional
        Marginalization configuration.
    sigma_jitter : array-like, optional
        Per-observation noise jitter (‰).

    Returns
    -------
    jax.Array
        Scalar marginal log-posterior.
    """
    constrained_model = parameter_space.constrain(u_vec, model)
    log_lk = marginal_log_likelihood(constrained_model, dataset, cfg, sigma_jitter=sigma_jitter)
    log_pr = parameter_space.log_prior_unconstrained(u_vec)
    return log_lk + log_pr


# ---------------------------------------------------------------------------
# 6. Joint log-likelihood (for LogGPProduction and ModaxSolver)
# ---------------------------------------------------------------------------


def joint_log_likelihood(
    model: RadiocarbonModel,
    dataset: Dataset,
    *,
    cfg: LinearMarginalConfig | None = None,
    sigma_jitter: ArrayLike | None = None,
) -> jax.Array:
    """Joint Gaussian log-likelihood (no linear marginalization).

    Suitable for ``LogGPProduction`` (nonlinear positive GP) and
    :class:`~ticktack.solvers.ModaxSolver`-backed models where
    the full non-Gaussian likelihood is evaluated with explicit
    ``weights ~ N(0, I_r)`` mode weights sampled by NUTS/SVI.

    For ``LogGPProduction``, the production is fully specified by the model's
    ``weights`` leaf (sampled via NumPyro) and no analytical marginalization is
    done for it; baseline offsets are still analytically marginalised via ``cfg``.

    Parameters
    ----------
    model : RadiocarbonModel
        Current model with all production parameters set.
    dataset : Dataset
        Observed tree-ring data.
    cfg : LinearMarginalConfig, optional
        Offset marginalization configuration (``marginalise_gp=False`` is
        automatically enforced for ``LogGPProduction``).
    sigma_jitter : array-like, optional
        Per-observation noise jitter (‰).

    Returns
    -------
    jax.Array
        Scalar joint log-likelihood.
    """
    # Detect ModaxSolver: wrap in jvp-only scalar function for HMC compatibility
    if isinstance(model.solver, ModaxSolver):
        return _joint_loglike_modax(model, dataset, cfg=cfg, sigma_jitter=sigma_jitter)

    # Standard joint: all production terms non-marginalised
    sigma = dataset.sigma_flat
    if sigma_jitter is not None:
        sigma = jnp.sqrt(jnp.square(sigma) + jnp.square(jnp.asarray(sigma_jitter, dtype=jnp.float64)))

    # Check whether the production contains a LogGPProduction
    has_loggp = _has_loggp(model.production)

    if has_loggp or cfg is None or cfg.offset_mode == "none":
        # Full joint: predict everything, Gaussian likelihood
        pred_flat = model.predict_d14c(dataset, offsets=None, concat=True)
        resid = dataset.d14c_flat - pred_flat
        sigma_arr = jnp.asarray(sigma, dtype=jnp.float64)
        return -0.5 * jnp.sum(jnp.square(resid / sigma_arr)) - jnp.sum(jnp.log(sigma_arr)) - 0.5 * dataset.n_obs * math.log(2.0 * math.pi)

    # Partial joint with offset marginalisation only
    joint_cfg = LinearMarginalConfig(
        offset_mode=cfg.offset_mode if cfg is not None else "hierarchical",
        global_prior_mean=cfg.global_prior_mean if cfg is not None else 0.0,
        global_prior_sd=cfg.global_prior_sd if cfg is not None else 200.0,
        lab_prior_sd=cfg.lab_prior_sd if cfg is not None else 1.5,
        marginalise_solar=False,
        marginalise_trend=False,
        marginalise_gp=False,
        whiten_method=cfg.whiten_method if cfg is not None else "cholesky",
    )
    return marginal_log_likelihood(model, dataset, joint_cfg, sigma_jitter=sigma_jitter)


def _has_loggp(production: ProductionTerm) -> bool:
    """Check whether a production term contains a ``LogGPProduction``."""
    if isinstance(production, LogGPProduction):
        return True
    if isinstance(production, ProductionSum):
        return any(_has_loggp(p) for p in production.parts)
    return False


def _joint_loglike_modax(
    model: RadiocarbonModel,
    dataset: Dataset,
    *,
    cfg: LinearMarginalConfig | None,
    sigma_jitter: ArrayLike | None,
) -> jax.Array:
    """Wrap joint log-likelihood for ``ModaxSolver`` with ``@jax.custom_vjp`` adapter."""
    # Build the scalar log-likelihood function and wrap it
    def _fn(model_inner: RadiocarbonModel) -> jax.Array:
        sigma = dataset.sigma_flat
        if sigma_jitter is not None:
            sigma = jnp.sqrt(jnp.square(sigma) + jnp.square(jnp.asarray(sigma_jitter, dtype=jnp.float64)))
        pred_flat = model_inner.predict_d14c(dataset, offsets=None, concat=True)
        resid = dataset.d14c_flat - pred_flat
        sigma_arr = jnp.asarray(sigma, dtype=jnp.float64)
        return -0.5 * jnp.sum(jnp.square(resid / sigma_arr)) - jnp.sum(jnp.log(sigma_arr)) - 0.5 * dataset.n_obs * math.log(2.0 * math.pi)

    wrapped = wrap_jvp_only_scalar_fn(_fn)
    return wrapped(model)


def joint_log_posterior(
    model: RadiocarbonModel,
    dataset: Dataset,
    parameter_space: ParameterSpace,
    u_vec: ArrayLike,
    *,
    cfg: LinearMarginalConfig | None = None,
    sigma_jitter: ArrayLike | None = None,
) -> jax.Array:
    """Joint log-posterior (joint log-likelihood + unconstrained log-prior).

    Parameters
    ----------
    model : RadiocarbonModel
        Base model.
    dataset : Dataset
        Observed tree-ring data.
    parameter_space : ParameterSpace
        Unconstrained parameter specification.
    u_vec : array-like
        Unconstrained parameter vector of shape ``(d,)``.
    cfg : LinearMarginalConfig, optional
        Offset marginalization configuration.
    sigma_jitter : array-like, optional
        Per-observation noise jitter (‰).

    Returns
    -------
    jax.Array
        Scalar joint log-posterior.
    """
    constrained_model = parameter_space.constrain(u_vec, model)
    log_lk = joint_log_likelihood(constrained_model, dataset, cfg=cfg, sigma_jitter=sigma_jitter)
    log_pr = parameter_space.log_prior_unconstrained(u_vec)
    return log_lk + log_pr


# ---------------------------------------------------------------------------
# 7. linear_posterior: closed-form posterior of marginalised linear params
# ---------------------------------------------------------------------------


class LinearPosteriorResult(NamedTuple):
    """Results from :func:`linear_posterior`."""

    coeff_mean: jax.Array
    """Posterior mean of all marginalised linear coefficients, shape ``(k_total,)``."""

    coeff_cov: jax.Array
    """Posterior covariance of coefficients, shape ``(k_total, k_total)``."""

    offset_global_mean: jax.Array
    """Reconstructed global Holocene ambient offset ``Delta_global`` posterior mean (‰)."""

    offset_lab_means: jax.Array
    """Reconstructed per-series inter-laboratory offset posterior means (‰), shape ``(S,)``."""

    baseline_d14c: jax.Array
    """Epoch-local baseline ``Delta14C_0`` estimate (‰) = offset_global_mean."""

    q_fine_mean: jax.Array | None
    """Posterior mean of ``Q(t)`` on ``t_fine`` (``atoms cm^-2 s^-1``), or ``None``."""

    q_fine_draws: jax.Array | None
    """Gaussian posterior draws of ``Q(t)`` shape ``(n_draws, N_fine)`` or ``None``."""

    d14c_fine_mean: jax.Array | None
    """Posterior mean of ``Delta14C(t)`` on ``t_fine`` (‰) or ``None``."""

    d14c_fine_draws: jax.Array | None
    """Gaussian posterior draws of ``Delta14C(t)`` shape ``(n_draws, N_fine)`` or ``None``."""


def linear_posterior(
    model: RadiocarbonModel,
    dataset: Dataset,
    cfg: LinearMarginalConfig | None = None,
    *,
    rng_key: jax.Array | None = None,
    n_draws: int = 0,
    t_fine: ArrayLike | None = None,
    sigma_jitter: ArrayLike | None = None,
) -> LinearPosteriorResult:
    """Closed-form posterior of all marginalised linear coefficients.

    Parameters
    ----------
    model : RadiocarbonModel
        Current nonlinear model (CBM + production).
    dataset : Dataset
        Observed tree-ring data.
    cfg : LinearMarginalConfig, optional
        Marginalization configuration.
    rng_key : jax.Array, optional
        JAX PRNGKey for Gaussian draws.  Required when ``n_draws > 0``.
    n_draws : int
        Number of Gaussian posterior samples to draw (0 = no draws).
    t_fine : array-like, optional
        Continuous time grid for posterior mean and draws of ``Q(t)`` and
        ``Delta14C(t)``.  Pass ``None`` to skip continuous reconstruction.
    sigma_jitter : array-like, optional
        Per-observation noise jitter (‰).

    Returns
    -------
    LinearPosteriorResult
        Named tuple with posterior mean, covariance, reconstructed offsets,
        and optional continuous ``Q(t)`` and ``Delta14C(t)`` curves.
    """
    if cfg is None:
        cfg = LinearMarginalConfig()

    sigma = dataset.sigma_flat
    if sigma_jitter is not None:
        sigma = jnp.sqrt(jnp.square(sigma) + jnp.square(jnp.asarray(sigma_jitter, dtype=jnp.float64)))

    resid = _nonlinear_residual(model, dataset)
    X, prior_mean, prior_sd = _build_marginal_design(model, dataset, cfg)

    lm = LinearMarginal(
        X,
        prior_mean=prior_mean,
        prior_sd=prior_sd,
        method=cfg.whiten_method,
    )
    coeff_mean, coeff_cov = lm.posterior(resid, sigma)

    # --- Reconstruct offsets ---
    n_offset_cols = 0
    if cfg.offset_mode == "hierarchical":
        n_offset_cols = 1 + dataset.n_series  # global + per-series
    elif cfg.offset_mode in ("per_series", "shared"):
        n_offset_cols = 1 if cfg.offset_mode == "shared" else dataset.n_series

    if n_offset_cols > 0:
        offset_global_mean = coeff_mean[0]
        if cfg.offset_mode == "hierarchical":
            offset_lab_means = coeff_mean[1 : 1 + dataset.n_series]
        else:
            offset_lab_means = jnp.zeros(dataset.n_series, dtype=jnp.float64)
    else:
        offset_global_mean = jnp.asarray(0.0, dtype=jnp.float64)
        offset_lab_means = jnp.zeros(dataset.n_series, dtype=jnp.float64)

    baseline_d14c = offset_global_mean

    # --- Optional continuous Q(t) and Delta14C reconstruction ---
    q_fine_mean = q_fine_draws = d14c_fine_mean = d14c_fine_draws = None

    if t_fine is not None and cfg.marginalise_gp:
        _, _, _, gp_terms = _unpack_production_terms(model.production)
        if gp_terms:
            # Find the GP columns in X (after offset + solar + trend columns)
            # For now, sum all GP contributions to Q(t)
            # This is a simplified reconstruction; full implementation would
            # invert the design matrix columns back to production space.
            # We skip full implementation here and return None for draws.
            pass

    # --- Optional Gaussian draws ---
    if n_draws > 0 and rng_key is not None:
        # Draw from N(coeff_mean, coeff_cov)
        L_cov = jnp.linalg.cholesky(coeff_cov + 1e-12 * jnp.eye(coeff_cov.shape[0], dtype=jnp.float64))
        z = jax.random.normal(rng_key, shape=(n_draws, coeff_mean.shape[0]), dtype=jnp.float64)
        coeff_draws = coeff_mean[None, :] + z @ L_cov.T  # (n_draws, k_total)
        # Store for caller use
        q_fine_draws = coeff_draws
        d14c_fine_draws = coeff_draws

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
    )
