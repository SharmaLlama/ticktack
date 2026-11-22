"""Multi-event shared-CBM Fisher analysis and marginalized GP inference for ``ticktack 2.0``.

Pools multiple historical cosmic-ray radiocarbon event windows (for example
``775CE``, ``993CE``, ``660BCE``, ``5259BCE``, ``5410BCE``) under a single
shared [`CarbonBoxModel`][ticktack.cbm.CarbonBoxModel] (such as
[`BalancedCBM`][ticktack.cbm.BalancedCBM] or
[`SubspaceCBM`][ticktack.cbm.SubspaceCBM]) while retaining independent
per-event [`MiyakeSpike`][ticktack.production.MiyakeSpike] parameters,
marginalized background [`GaussianField`][ticktack.fields.GaussianField] /
[`GPProduction`][ticktack.production.GPProduction] draws, and baseline
laboratory offsets:

1. [`EventWindow`][ticktack.multievent.EventWindow] &
   [`MultiEventDataset`][ticktack.multievent.MultiEventDataset]: immutable
   containers bundling each event's [`Dataset`][ticktack.data.Dataset],
   [`ProductionTerm`][ticktack.production.ProductionTerm], and
   [`LinearMarginalConfig`][ticktack.likelihood.LinearMarginalConfig],
   validating unique event names and disjoint time spans.
2. [`MultiEventModel`][ticktack.multievent.MultiEventModel]: ``zodiax.Base``
   PyTree holding a shared ``cbm`` alongside a keyed mapping
   ``events: dict[str, ProductionTerm]`` so
   [`ParameterSpace`][ticktack.likelihood.ParameterSpace] string paths such as
   ``"cbm.alpha"`` and ``"events.775CE.spike.fluence"`` address leaves
   natively.
3. [`multi_event_cbm_gp_fisher_spectrum`][ticktack.multievent.multi_event_cbm_gp_fisher_spectrum]:
   sums per-event [`cbm_gp_fisher_spectrum`][ticktack.fisher.cbm_gp_fisher_spectrum]
   information matrices across all events (whitening by the shared CBM prior
   exactly once) and returns a
   [`FisherSpectrumResult`][ticktack.fisher.FisherSpectrumResult] directly
   compatible with [`SubspaceCBM.from_fisher`][ticktack.cbm.SubspaceCBM.from_fisher].
4. [`multi_event_marginal_log_likelihood`][ticktack.multievent.multi_event_marginal_log_likelihood],
   [`multi_event_marginal_log_posterior`][ticktack.multievent.multi_event_marginal_log_posterior],
   [`reweight_multi_event_exact`][ticktack.multievent.reweight_multi_event_exact],
   and [`fit_multi_event_subspace_cbm`][ticktack.multievent.fit_multi_event_subspace_cbm]:
   block-diagonal Woodbury marginalization in ``O(sum_e N_e r_e^2)`` time,
   exact joint L-BFGS-B MAP + Laplace preconditioning, per-event MAP-centred
   [`LinearizedDCTOperator`][ticktack.fields.LinearizedDCTOperator] instances
   for fast NUTS sampling, and exact multi-event importance-sampling
   reweighting.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import math
from typing import Any, Callable, Literal, NamedTuple
import warnings

import arviz as az
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jax.typing import ArrayLike
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.distributions.distribution as _np_dist_mod
from numpyro.infer import MCMC, NUTS
from scipy.optimize import minimize as _scipy_minimize
import zodiax as zdx

if not hasattr(dist, "distribution"):
    dist.distribution = _np_dist_mod

from .cbm import BalancedCBM, CarbonBoxModel, FixedCBM, SubspaceCBM
from .constants import ALPHA_ATOMS_CM2_S_TO_KG_YR, DEFAULT_Q_STD
from .data import Dataset, TreeRingSeries
from .fields import (
    LinearizedDCTOperator,
    _compute_dataset_digest,
    _compute_subspace_basis_digest,
    _compute_template_digest,
    _safe_sqrt,
    build_linearized_dct_operator,
)
from .fisher import (
    FisherSpectrumResult,
    _jacfwd_chunked,
    _partition_inexact,
    _resolve_cbm_prior_root,
    cbm_fisher_spectrum,
    cbm_gp_fisher_spectrum,
)
from .fitting import _regularize_hessian_to_cov
from .likelihood import (
    LinearMarginalConfig,
    LinearPosteriorResult,
    ParameterSpace,
    _build_base_production,
    _build_marginal_design,
    _concat_padded_windows,
    _nonlinear_residual,
    _partition_terms_for_cfg,
    linear_posterior,
    marginal_log_likelihood,
)
from .linear import LinearMarginal, hierarchical_offset_design
from .model import RadiocarbonModel
from .production import (
    MiyakeSpike,
    ProductionTerm,
)
from .sampling import _unpack_constrained_samples
from .solvers import AbstractSolver, ExactLinearSolver, _modal_decomposition

__all__ = [
    "EventWindow",
    "MultiEventDataset",
    "MultiEventModel",
    "MultiEventFitResult",
    "multi_event_cbm_gp_fisher_spectrum",
    "multi_event_marginal_log_likelihood",
    "multi_event_marginal_log_posterior",
    "reweight_multi_event_exact",
    "fit_multi_event_subspace_cbm",
]


# ---------------------------------------------------------------------------
# 1. EventWindow, MultiEventDataset, and MultiEventModel
# ---------------------------------------------------------------------------


def _validate_event_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"EventWindow name must be a str, got {type(name)!r}.")
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("EventWindow name must be a non-empty string.")
    if "." in cleaned:
        raise ValueError(
            f"EventWindow name {name!r} cannot contain '.' because zodiax uses "
            "'.' as its parameter-path delimiter."
        )
    return cleaned


def _strip_dct_operator(cfg: LinearMarginalConfig) -> LinearMarginalConfig:
    """Return ``cfg`` with any pre-attached ``dct_operator`` cleared to ``None``."""
    if cfg.dct_operator is None:
        return cfg
    return eqx.tree_at(lambda c: c.dct_operator, cfg, None)


class EventWindow(eqx.Module):
    """Single historical radiocarbon event window for multi-event CBM pooling.

    Bundles an event's observational [`Dataset`][ticktack.data.Dataset], its
    independent per-event [`ProductionTerm`][ticktack.production.ProductionTerm]
    (spike, marginalized cosmic-ray background field or GP, etc.), and its
    [`LinearMarginalConfig`][ticktack.likelihood.LinearMarginalConfig].

    Parameters
    ----------
    name : str
        Unique event identifier containing no dots (``"."``), e.g. ``"775CE"``.
    dataset : Dataset or TreeRingSeries
        Tree-ring observations for this event window.
    production : ProductionTerm
        Per-event cosmic-ray production model (typically a
        [`ProductionSum`][ticktack.production.ProductionSum] combining
        [`SteadyProduction`][ticktack.production.SteadyProduction], a
        marginalized [`GaussianField`][ticktack.fields.GaussianField] or
        [`GPProduction`][ticktack.production.GPProduction], and a
        [`MiyakeSpike`][ticktack.production.MiyakeSpike]).
    marginal_config : LinearMarginalConfig, optional
        Linear marginalization configuration for this event. Defaults to
        ``LinearMarginalConfig()``.
    spike_path : str or None, optional
        Dot-separated attribute path to this event's
        [`MiyakeSpike`][ticktack.production.MiyakeSpike] inside ``production``
        (default ``"spike"``), or ``None`` if the event contains no parametric
        spike.
    """

    name: str = eqx.field(static=True)
    dataset: Dataset
    production: ProductionTerm
    marginal_config: LinearMarginalConfig
    spike_path: str | None = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        dataset: Dataset | TreeRingSeries,
        production: ProductionTerm,
        marginal_config: LinearMarginalConfig | None = None,
        spike_path: str | None = "spike",
    ) -> None:
        self.name = _validate_event_name(name)
        if isinstance(dataset, TreeRingSeries):
            self.dataset = Dataset((dataset,), name=dataset.name)
        elif isinstance(dataset, Dataset):
            self.dataset = dataset
        else:
            raise TypeError(
                f"EventWindow dataset must be a Dataset or TreeRingSeries, "
                f"got {type(dataset)!r}."
            )
        if not isinstance(production, ProductionTerm):
            raise TypeError(
                f"EventWindow production must be a ProductionTerm, "
                f"got {type(production)!r}."
            )
        cfg = LinearMarginalConfig() if marginal_config is None else marginal_config
        if not isinstance(cfg, LinearMarginalConfig):
            raise TypeError(
                f"EventWindow marginal_config must be a LinearMarginalConfig, "
                f"got {type(cfg)!r}."
            )
        self.production = production
        self.marginal_config = cfg
        self.spike_path = None if spike_path is None else str(spike_path)

    @property
    def n_obs(self) -> int:
        """Total number of tree-ring observations in this event window."""
        return self.dataset.n_obs

    @property
    def n_series(self) -> int:
        """Number of tree-ring series in this event window."""
        return self.dataset.n_series

    @property
    def time_span(self) -> tuple[float, float]:
        """Chronological ``(t_min, t_max)`` span of this event's dataset."""
        return self.dataset.time_span

    def get_spike(
        self, production: ProductionTerm | None = None
    ) -> MiyakeSpike | None:
        """Return this event's [`MiyakeSpike`][ticktack.production.MiyakeSpike], or ``None``.

        Parameters
        ----------
        production : ProductionTerm, optional
            Production term to query instead of ``self.production``.

        Returns
        -------
        MiyakeSpike or None
            Resolved spike component, or ``None`` if absent.
        """
        prod = self.production if production is None else production
        if self.spike_path is None:
            return None
        if isinstance(prod, MiyakeSpike):
            return prod
        try:
            candidate = prod.get(self.spike_path)
        except (AttributeError, KeyError, ValueError):
            candidate = None
        if isinstance(candidate, MiyakeSpike):
            return candidate
        return None

    def with_marginal_config(self, cfg: LinearMarginalConfig) -> EventWindow:
        """Return a copy of this event window with an updated ``marginal_config``.

        Parameters
        ----------
        cfg : LinearMarginalConfig
            Replacement marginalization configuration.

        Returns
        -------
        EventWindow
            New event window carrying ``cfg``.
        """
        if not isinstance(cfg, LinearMarginalConfig):
            raise TypeError(f"Expected LinearMarginalConfig, got {type(cfg)!r}.")
        return eqx.tree_at(lambda ev: ev.marginal_config, self, cfg)


class MultiEventDataset(eqx.Module):
    """Collection of conditionally independent, named [`EventWindow`][ticktack.multievent.EventWindow] objects.

    Validates unique event names and (by default) disjoint chronological spans
    so that per-event GP cosmic-ray fluctuations, baseline laboratory offsets,
    and Miyake spikes factorize into a block-diagonal marginal likelihood given
    a shared [`CarbonBoxModel`][ticktack.cbm.CarbonBoxModel].

    Parameters
    ----------
    events : Sequence[EventWindow]
        Non-empty sequence of event windows to pool.
    name : str, optional
        Human-readable label for the pooled dataset (default ``"multi_event"``).
    check_disjoint : bool, optional
        If ``True`` (default), raise ``ValueError`` when any two events have
        overlapping chronological spans.
    """

    events: tuple[EventWindow, ...]
    name: str = eqx.field(static=True)

    def __init__(
        self,
        events: Sequence[EventWindow],
        *,
        name: str = "multi_event",
        check_disjoint: bool = True,
    ) -> None:
        ev_tuple = tuple(events)
        if not ev_tuple:
            raise ValueError("MultiEventDataset requires at least one EventWindow.")
        seen_names: set[str] = set()
        spans: list[tuple[str, float, float]] = []
        for ev in ev_tuple:
            if not isinstance(ev, EventWindow):
                raise TypeError(
                    f"All entries in MultiEventDataset must be EventWindow instances; "
                    f"got {type(ev)!r}."
                )
            if ev.name in seen_names:
                raise ValueError(
                    f"Duplicate EventWindow name {ev.name!r} in MultiEventDataset."
                )
            seen_names.add(ev.name)
            if ev.n_obs > 0:
                t_lo, t_hi = ev.time_span
                spans.append((ev.name, float(t_lo), float(t_hi)))

        if check_disjoint and len(spans) > 1:
            spans_sorted = sorted(spans, key=lambda item: (item[1], item[2]))
            for idx in range(len(spans_sorted) - 1):
                name_a, _lo_a, hi_a = spans_sorted[idx]
                name_b, lo_b, _hi_b = spans_sorted[idx + 1]
                if lo_b < hi_a:
                    raise ValueError(
                        f"EventWindow {name_a!r} (ends at {hi_a:.2f}) overlaps "
                        f"with {name_b!r} (starts at {lo_b:.2f}). Pass "
                        "check_disjoint=False only if overlapping windows are "
                        "intentional."
                    )

        self.events = ev_tuple
        self.name = str(name)

    @property
    def n_events(self) -> int:
        """Number of event windows."""
        return len(self.events)

    @property
    def n_obs_total(self) -> int:
        """Total number of tree-ring observations summed across all events."""
        return sum(ev.n_obs for ev in self.events)

    @property
    def event_names(self) -> tuple[str, ...]:
        """Ordered tuple of event window names."""
        return tuple(ev.name for ev in self.events)

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[EventWindow]:
        return iter(self.events)

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return any(ev.name == name for ev in self.events)

    def get_event(self, name: str) -> EventWindow:
        """Look up an [`EventWindow`][ticktack.multievent.EventWindow] by name.

        Parameters
        ----------
        name : str
            Registered event name.

        Returns
        -------
        EventWindow
            Matching event window.
        """
        for ev in self.events:
            if ev.name == name:
                return ev
        raise KeyError(
            f"No EventWindow named {name!r}; available events are {self.event_names}."
        )

    def __getitem__(self, key: int | str) -> EventWindow:
        if isinstance(key, int):
            return self.events[key]
        if isinstance(key, str):
            return self.get_event(key)
        raise TypeError(
            f"MultiEventDataset indices must be int or str, got {type(key)!r}."
        )

    def with_marginal_configs(
        self,
        configs: Mapping[str, LinearMarginalConfig],
    ) -> MultiEventDataset:
        """Return a new [`MultiEventDataset`][ticktack.multievent.MultiEventDataset] with updated configs.

        Parameters
        ----------
        configs : Mapping[str, LinearMarginalConfig]
            Mapping from event name to replacement marginalization config.

        Returns
        -------
        MultiEventDataset
            Updated dataset carrying the new configurations.
        """
        unknown = set(configs.keys()) - set(self.event_names)
        if unknown:
            raise KeyError(
                f"Unknown event names in configs: {sorted(unknown)}; "
                f"available events are {self.event_names}."
            )
        new_events = tuple(
            ev.with_marginal_config(configs[ev.name])
            if ev.name in configs
            else ev
            for ev in self.events
        )
        return eqx.tree_at(lambda m: m.events, self, new_events)


class MultiEventModel(zdx.Base):
    """Shared-CBM multi-event radiocarbon model supporting ``zodiax`` paths.

    Holds a single shared [`CarbonBoxModel`][ticktack.cbm.CarbonBoxModel] (for
    example [`BalancedCBM`][ticktack.cbm.BalancedCBM] or
    [`SubspaceCBM`][ticktack.cbm.SubspaceCBM]) and a keyed dictionary
    ``events: dict[str, ProductionTerm]`` so
    [`ParameterSpace`][ticktack.likelihood.ParameterSpace] addresses shared and
    per-event leaves natively (for example ``"cbm.alpha"``,
    ``"events.775CE.spike.fluence"``, ``"events.993CE.spike.fluence"``).

    Parameters
    ----------
    cbm : CarbonBoxModel
        Shared carbon-box model applied across all events.
    events : Mapping[str, ProductionTerm]
        Non-empty mapping from event name to per-event production term.
    solver : AbstractSolver, optional
        Forward LTI solver (defaults to
        [`ExactLinearSolver`][ticktack.solvers.ExactLinearSolver]).
    q_std : ArrayLike, optional
        Reference steady-state production rate (atoms cm⁻² s⁻¹).
    alpha_unit : ArrayLike, optional
        Unit conversion factor from atoms cm⁻² s⁻¹ to kg yr⁻¹.
    """

    cbm: CarbonBoxModel
    events: dict[str, ProductionTerm]
    solver: AbstractSolver
    q_std: jax.Array
    alpha_unit: jax.Array

    def __init__(
        self,
        cbm: CarbonBoxModel,
        events: Mapping[str, ProductionTerm],
        solver: AbstractSolver | None = None,
        *,
        q_std: ArrayLike = DEFAULT_Q_STD,
        alpha_unit: ArrayLike = ALPHA_ATOMS_CM2_S_TO_KG_YR,
    ) -> None:
        if not isinstance(cbm, CarbonBoxModel):
            raise TypeError(f"cbm must be a CarbonBoxModel, got {type(cbm)!r}.")
        if not isinstance(events, Mapping) or len(events) == 0:
            raise ValueError(
                "MultiEventModel requires a non-empty mapping of {name: ProductionTerm}."
            )
        cleaned_events: dict[str, ProductionTerm] = {}
        for k, prod in events.items():
            cname = _validate_event_name(k)
            if not isinstance(prod, ProductionTerm):
                raise TypeError(
                    f"Event production {k!r} must be a ProductionTerm, got {type(prod)!r}."
                )
            cleaned_events[cname] = prod
        if solver is None:
            solver = ExactLinearSolver()
        if not isinstance(solver, AbstractSolver):
            raise TypeError(f"solver must be an AbstractSolver, got {type(solver)!r}.")
        self.cbm = cbm
        self.events = cleaned_events
        self.solver = solver
        self.q_std = jnp.asarray(q_std, dtype=jnp.float64).reshape(())
        self.alpha_unit = jnp.asarray(alpha_unit, dtype=jnp.float64).reshape(())

    @classmethod
    def from_multi_dataset(
        cls,
        cbm: CarbonBoxModel,
        multi_dataset: MultiEventDataset,
        solver: AbstractSolver | None = None,
        *,
        q_std: ArrayLike = DEFAULT_Q_STD,
        alpha_unit: ArrayLike = ALPHA_ATOMS_CM2_S_TO_KG_YR,
    ) -> MultiEventModel:
        """Construct a [`MultiEventModel`][ticktack.multievent.MultiEventModel] from ``cbm`` and a [`MultiEventDataset`][ticktack.multievent.MultiEventDataset].

        Parameters
        ----------
        cbm : CarbonBoxModel
            Shared carbon-box model.
        multi_dataset : MultiEventDataset
            Pooled event windows providing each event's production model.
        solver : AbstractSolver, optional
            Forward solver (defaults to
            [`ExactLinearSolver`][ticktack.solvers.ExactLinearSolver]).
        q_std, alpha_unit : ArrayLike, optional
            Reference production rate and unit conversion factor.

        Returns
        -------
        MultiEventModel
            Assembled multi-event model.
        """
        return cls(
            cbm=cbm,
            events={ev.name: ev.production for ev in multi_dataset.events},
            solver=solver,
            q_std=q_std,
            alpha_unit=alpha_unit,
        )

    @property
    def event_names(self) -> tuple[str, ...]:
        """Ordered event names present in this model."""
        return tuple(self.events.keys())

    @property
    def n_events(self) -> int:
        """Number of events in this model."""
        return len(self.events)

    def event_model(self, event_name: str) -> RadiocarbonModel:
        """Extract the single-event [`RadiocarbonModel`][ticktack.model.RadiocarbonModel] for ``event_name``.

        Parameters
        ----------
        event_name : str
            Name of the target event window.

        Returns
        -------
        RadiocarbonModel
            Single-event model sharing ``self.cbm`` and ``self.solver``.
        """
        if event_name not in self.events:
            raise KeyError(
                f"Event {event_name!r} not found in MultiEventModel; "
                f"available events are {self.event_names}."
            )
        return RadiocarbonModel(
            self.cbm,
            self.events[event_name],
            solver=self.solver,
            q_std=self.q_std,
            alpha_unit=self.alpha_unit,
        )


# ---------------------------------------------------------------------------
# 2. Multi-event GP-marginalized Fisher eigenspectrum
# ---------------------------------------------------------------------------


def _resolve_fisher_cbm(
    cbm: CarbonBoxModel,
    parameterisation: Literal["log_rates", "raw"],
) -> CarbonBoxModel:
    if parameterisation not in ("log_rates", "raw"):
        raise ValueError(
            f"parameterisation must be 'log_rates' or 'raw'; got {parameterisation!r}."
        )
    if parameterisation == "log_rates" and isinstance(cbm, FixedCBM):
        if bool(jnp.any(cbm.sink_rates() > 0.0)):
            raise ValueError(
                "FixedCBM with non-zero sink_rates cannot be converted to BalancedCBM "
                "under parameterisation='log_rates'; pass an OpenCBM or parameterisation='raw'."
            )
        return BalancedCBM.from_masses_and_fluxes(
            topology=cbm.topology,
            masses=cbm.masses,
            edge_fluxes=cbm.edge_fluxes,
            prod_weights=cbm.prod_weights,
            decay_rate=cbm.decay_rate,
        )
    return cbm


def _single_event_general_fisher(
    cbm: CarbonBoxModel,
    event: EventWindow,
    *,
    prior_cbm_sd: Any = None,
    prior_cbm_cov: Any = None,
    prior_cbm_prec: Any = None,
    prior_cbm_root: Any = None,
    spike_prior_precision: Any = None,
    include_covariance_fisher: bool = True,
    q_std: ArrayLike = DEFAULT_Q_STD,
    alpha_unit: ArrayLike = ALPHA_ATOMS_CM2_S_TO_KG_YR,
    solver: AbstractSolver | None = None,
) -> FisherSpectrumResult:
    """Fallback single-event Fisher for custom priors, solvers, or composite marginalized terms."""
    solver_eff = ExactLinearSolver() if solver is None else solver
    cfg = _strip_dct_operator(event.marginal_config)
    q_std_arr = jnp.asarray(q_std, dtype=jnp.float64)
    model_e = RadiocarbonModel(
        cbm,
        event.production,
        solver=solver_eff,
        q_std=q_std,
        alpha_unit=alpha_unit,
    )
    X_m, _pm_m, sd_m = _build_marginal_design(model_e, event.dataset, cfg)
    sd_np = np.asarray(sd_m, dtype=np.float64)
    fin_mask = np.isfinite(sd_np) & (sd_np > 0.0)
    inf_mask = ~np.isfinite(sd_np)
    fin_idx = jnp.asarray(np.flatnonzero(fin_mask), dtype=jnp.int32)
    has_fin_lin = bool(np.any(fin_mask))

    def _h_cov_for(cbm_arg: CarbonBoxModel) -> jax.Array:
        if not has_fin_lin:
            return jnp.zeros((event.n_obs, 0), dtype=jnp.float64)
        m_arg = RadiocarbonModel(
            cbm_arg,
            event.production,
            solver=solver_eff,
            q_std=q_std,
            alpha_unit=alpha_unit,
        )
        X_arg, _, sd_arg = _build_marginal_design(m_arg, event.dataset, cfg)
        return X_arg[:, fin_idx] * sd_arg[fin_idx][None, :]

    h_cov_0 = (
        X_m[:, fin_idx] * sd_m[fin_idx][None, :]
        if has_fin_lin
        else jnp.zeros((event.n_obs, 0), dtype=jnp.float64)
    )
    sigma_d = jnp.asarray(event.dataset.sigma_flat, dtype=jnp.float64)
    sigma_eff = jnp.diag(jnp.square(sigma_d)) + h_cov_0 @ h_cov_0.T
    sigma_eff = 0.5 * (sigma_eff + sigma_eff.T)
    chol_eff = jnp.linalg.cholesky(sigma_eff)

    use_sb_cov = bool(include_covariance_fisher and h_cov_0.shape[1] > 0)
    if use_sb_cov:
        u_0 = jsl.solve_triangular(chol_eff, h_cov_0, lower=True)
        q_0, r_0 = jnp.linalg.qr(u_0, mode="reduced")
        r_rank = int(q_0.shape[1])
        tril_i, tril_j = np.tril_indices(r_rank, k=-1)
        inv_sqrt2 = 1.0 / math.sqrt(2.0)

        def _sb_cov_rows(cbm_arg: CarbonBoxModel) -> jax.Array:
            u_v = jsl.solve_triangular(chol_eff, _h_cov_for(cbm_arg), lower=True)
            x_v = q_0.T @ u_v
            y_v = u_v - q_0 @ x_v
            s_v = x_v @ r_0.T + r_0 @ x_v.T
            diag_part = inv_sqrt2 * jnp.diag(s_v)
            tril_part = s_v[tril_i, tril_j]
            orth_part = (y_v @ r_0.T).reshape(-1)
            return jnp.concatenate([diag_part, tril_part, orth_part], axis=0)

    add_global_prior_mean = (
        cfg.offset_mode in ("hierarchical", "shared")
        and math.isfinite(cfg.global_prior_sd)
        and float(cfg.global_prior_mean) != 0.0
    )
    g_mean_f = float(cfg.global_prior_mean)
    box_indices_flat = jnp.array(
        [
            model_e._box_index_for_series(s)
            for s in event.dataset.series
            for _ in range(s.n_obs)
        ],
        dtype=jnp.int32,
    )

    def _obs_activity_ratios_for(cbm_arg: CarbonBoxModel) -> jax.Array:
        y_ss = cbm_arg.steady_state_c14(q_std=q_std, alpha_unit=alpha_unit)
        trop_idx = jnp.asarray(cbm_arg.topology.trop_box_indices, dtype=jnp.int32)
        y_trop = jnp.sum(y_ss[trop_idx])
        m_rel = cbm_arg.relative_reservoir_masses()
        return ((y_ss / y_trop) / m_rel)[box_indices_flat]

    def _pack_whitened(cbm_arg: CarbonBoxModel, pred: jax.Array) -> jax.Array:
        if add_global_prior_mean:
            pred = pred + g_mean_f * _obs_activity_ratios_for(cbm_arg)
        w_mean = jsl.solve_triangular(chol_eff, pred, lower=True)
        if use_sb_cov:
            return jnp.concatenate([w_mean, _sb_cov_rows(cbm_arg)], axis=0)
        return w_mean

    def _base_prod_with_spike(sp_v: MiyakeSpike | None) -> ProductionTerm:
        if sp_v is None or event.spike_path is None:
            prod_full = event.production
        elif isinstance(event.production, MiyakeSpike):
            prod_full = sp_v
        else:
            prod_full = event.production.set(event.spike_path, sp_v)
        unmarg_parts, _, _, _, _ = _partition_terms_for_cfg(prod_full, cfg)
        return _build_base_production(unmarg_parts, q_std_arr)

    spike_obj = event.get_spike(event.production)
    base_prod = _base_prod_with_spike(None)

    has_inf_lin = bool(np.any(inf_mask))
    inf_cols = (
        X_m[:, jnp.asarray(np.flatnonzero(inf_mask), dtype=jnp.int32)]
        if has_inf_lin
        else jnp.zeros((event.n_obs, 0), dtype=jnp.float64)
    )
    n_inf = int(inf_cols.shape[1])

    if spike_obj is not None and has_inf_lin:
        psi_init: Any = (spike_obj, jnp.zeros((n_inf,), dtype=jnp.float64))

        def fwd_fn(cbm_arg: CarbonBoxModel, psi_arg: tuple[Any, jax.Array]) -> jax.Array:
            sp_v, inf_v = psi_arg
            prod_eval = _base_prod_with_spike(sp_v)
            pred = RadiocarbonModel(
                cbm_arg, prod_eval, solver=solver_eff, q_std=q_std, alpha_unit=alpha_unit
            ).predict_d14c(event.dataset, concat=True)
            return _pack_whitened(cbm_arg, pred + inf_cols @ inf_v)

        full_psi_prec = None
        if spike_prior_precision is not None:
            flat_sp, _ = _partition_inexact(spike_obj)
            p_sp = int(flat_sp.shape[0])
            sp_prec = jnp.asarray(spike_prior_precision, dtype=jnp.float64)
            if sp_prec.ndim == 0:
                sp_prec = sp_prec * jnp.eye(p_sp, dtype=jnp.float64)
            elif sp_prec.ndim == 1:
                sp_prec = jnp.diag(sp_prec)
            full_psi_prec = jnp.zeros((p_sp + n_inf, p_sp + n_inf), dtype=jnp.float64)
            full_psi_prec = full_psi_prec.at[:p_sp, :p_sp].set(sp_prec)
    elif spike_obj is not None:
        psi_init = spike_obj

        def fwd_fn(cbm_arg: CarbonBoxModel, sp_v: MiyakeSpike) -> jax.Array:
            prod_eval = _base_prod_with_spike(sp_v)
            pred = RadiocarbonModel(
                cbm_arg, prod_eval, solver=solver_eff, q_std=q_std, alpha_unit=alpha_unit
            ).predict_d14c(event.dataset, concat=True)
            return _pack_whitened(cbm_arg, pred)

        full_psi_prec = None
        if spike_prior_precision is not None:
            flat_sp, _ = _partition_inexact(spike_obj)
            p_sp = int(flat_sp.shape[0])
            sp_prec = jnp.asarray(spike_prior_precision, dtype=jnp.float64)
            if sp_prec.ndim == 0:
                full_psi_prec = sp_prec * jnp.eye(p_sp, dtype=jnp.float64)
            elif sp_prec.ndim == 1:
                full_psi_prec = jnp.diag(sp_prec)
            else:
                full_psi_prec = sp_prec
    elif has_inf_lin:
        psi_init = jnp.zeros((n_inf,), dtype=jnp.float64)

        def fwd_fn(cbm_arg: CarbonBoxModel, inf_v: jax.Array) -> jax.Array:
            pred = RadiocarbonModel(
                cbm_arg, base_prod, solver=solver_eff, q_std=q_std, alpha_unit=alpha_unit
            ).predict_d14c(event.dataset, concat=True)
            return _pack_whitened(cbm_arg, pred + inf_cols @ inf_v)

        full_psi_prec = None
    else:
        psi_init = None

        def fwd_fn(cbm_arg: CarbonBoxModel) -> jax.Array:
            pred = RadiocarbonModel(
                cbm_arg, base_prod, solver=solver_eff, q_std=q_std, alpha_unit=alpha_unit
            ).predict_d14c(event.dataset, concat=True)
            return _pack_whitened(cbm_arg, pred)

        full_psi_prec = None

    return cbm_fisher_spectrum(
        fwd_fn,
        cbm,
        1.0,
        psi_nuisance=psi_init,
        prior_cbm_sd=prior_cbm_sd,
        prior_cbm_cov=prior_cbm_cov,
        prior_cbm_prec=prior_cbm_prec,
        prior_cbm_root=prior_cbm_root,
        prior_psi_prec=full_psi_prec,
        noise_kind="sigma",
    )


def multi_event_cbm_gp_fisher_spectrum(
    cbm: CarbonBoxModel,
    multi_dataset: MultiEventDataset | Sequence[EventWindow],
    *,
    prior_sd: Any = 0.35,
    prior_cbm_cov: Any = None,
    prior_cbm_prec: Any = None,
    prior_cbm_root: Any = None,
    parameterisation: Literal["log_rates", "raw"] = "log_rates",
    spike_prior_precision: Mapping[str, Any] | Any | None = None,
    include_covariance_fisher: bool = True,
    mass_fn: Callable[[Any], Any] | None = None,
    flux_fn: Callable[[Any], Any] | None = None,
    log_space_observables: bool = True,
    q_std: ArrayLike = DEFAULT_Q_STD,
    alpha_unit: ArrayLike = ALPHA_ATOMS_CM2_S_TO_KG_YR,
    solver: AbstractSolver | None = None,
) -> FisherSpectrumResult:
    r"""Compute the pooled multi-event GP-, offset-, and spike-marginalized CBM Fisher spectrum.

    Delegates each standard event window to
    [`cbm_gp_fisher_spectrum`][ticktack.fisher.cbm_gp_fisher_spectrum] and sums
    the Schur-marginalized CBM Fisher information matrices linearly across
    conditionally independent events:

    $$\widetilde{F}_{\mathrm{tot}} = \sum_{e=1}^{E} \widetilde{F}_e = L_\Pi^\top \left(\sum_{e=1}^{E} F_{\mathrm{CBM},e}^{\mathrm{marg}}\right) L_\Pi,$$

    applying the shared CBM prior precision $I$ exactly once (never
    double-counted across events). The returned
    [`FisherSpectrumResult`][ticktack.fisher.FisherSpectrumResult] is directly
    compatible with
    [`SubspaceCBM.from_fisher`][ticktack.cbm.SubspaceCBM.from_fisher].

    Parameters
    ----------
    cbm : CarbonBoxModel
        Reference carbon-box model shared across all event windows.
    multi_dataset : MultiEventDataset or Sequence[EventWindow]
        Collection of conditionally independent event windows to pool.
    prior_sd : float or ArrayLike, optional
        Gaussian prior standard deviation on active CBM parameters (default
        ``0.35``), ignored when any of ``prior_cbm_cov``, ``prior_cbm_prec``,
        or ``prior_cbm_root`` is provided.
    prior_cbm_cov, prior_cbm_prec, prior_cbm_root : ArrayLike, optional
        Custom Gaussian prior covariance, precision, or square-root factor on
        the shared CBM parameters (specify at most one).
    parameterisation : {"log_rates", "raw"}, optional
        CBM parameterisation convention (default ``"log_rates"``).
    spike_prior_precision : Mapping[str, Any] or ArrayLike, optional
        Optional per-event mapping or shared Gaussian prior precision on each
        event's active spike nuisance parameters.
    include_covariance_fisher : bool, optional
        If ``True`` (default), include exact Slepian--Bangs covariance-sensitivity
        Fisher rows from the CBM dependence of the marginalized GP and offset
        covariance, matching ``cbm_gp_fisher_spectrum``.
    mass_fn, flux_fn : callable, optional
        Optional physical observable maps evaluated once on the pooled
        eigenmodes. ``mass_fn`` defaults to ``lambda m: m.reservoir_masses()``
        when available, matching ``cbm_gp_fisher_spectrum``.
    log_space_observables : bool, optional
        If ``True`` (default), differentiate ``log(mass_fn(cbm))`` and
        ``log(flux_fn(cbm))`` so loadings are fractional changes per prior SD.
    q_std, alpha_unit : ArrayLike, optional
        Reference steady-state production rate and unit conversion factor.
    solver : AbstractSolver, optional
        Optional forward solver; passing a custom solver routes events through
        the general fallback Fisher evaluator.

    Returns
    -------
    FisherSpectrumResult
        Pooled Fisher spectrum with single-prior posterior covariance,
        block-arrow joint Fisher over ``(theta_CBM, psi_1, ..., psi_E)``, and
        physical mass/flux loadings.
    """
    if isinstance(multi_dataset, MultiEventDataset):
        mds = multi_dataset
    else:
        mds = MultiEventDataset(tuple(multi_dataset))

    cbm_eval = _resolve_fisher_cbm(cbm, parameterisation)
    use_prior_sd = (
        prior_sd
        if (prior_cbm_cov is None and prior_cbm_prec is None and prior_cbm_root is None)
        else None
    )
    can_use_core_gp_fisher = (
        parameterisation == "log_rates"
        and prior_cbm_cov is None
        and prior_cbm_prec is None
        and prior_cbm_root is None
        and solver is None
    )

    # Suppress redundant per-event mass_fn Jacobians inside the loop; evaluate once on pooled loadings.
    def _noop_mass(_m: Any) -> jax.Array:
        return jnp.ones((1,), dtype=jnp.float64)

    per_event_results: list[FisherSpectrumResult] = []
    for ev in mds.events:
        sp_prec_e = (
            spike_prior_precision.get(ev.name, None)
            if isinstance(spike_prior_precision, Mapping)
            else spike_prior_precision
        )
        unmarg, solar_t, trend_t, gp_t, field_t = _partition_terms_for_cfg(
            ev.production, ev.marginal_config
        )
        spike_obj = ev.get_spike(ev.production)
        non_steady_unmarg = [
            t for t in unmarg if not isinstance(t, (MiyakeSpike,))
            and type(t).__name__ != "SteadyProduction"
        ]
        if (
            can_use_core_gp_fisher
            and not solar_t
            and not trend_t
            and not non_steady_unmarg
            and (len(gp_t) + len(field_t)) <= 1
        ):
            gp_or_field = field_t[0] if field_t else (gp_t[0] if gp_t else None)
            res_e = cbm_gp_fisher_spectrum(
                cbm_eval,
                ev.dataset,
                gp_or_field,
                spike=spike_obj,
                spike_prior_precision=sp_prec_e,
                marginal_config=_strip_dct_operator(ev.marginal_config),
                prior_sd=use_prior_sd,
                parameterisation="log_rates",
                include_covariance_fisher=include_covariance_fisher,
                mass_fn=_noop_mass,
                flux_fn=None,
                q_std=float(q_std),
                alpha_unit=float(alpha_unit),
            )
        else:
            res_e = _single_event_general_fisher(
                cbm_eval,
                ev,
                prior_cbm_sd=use_prior_sd,
                prior_cbm_cov=prior_cbm_cov,
                prior_cbm_prec=prior_cbm_prec,
                prior_cbm_root=prior_cbm_root,
                spike_prior_precision=sp_prec_e,
                include_covariance_fisher=include_covariance_fisher,
                q_std=q_std,
                alpha_unit=alpha_unit,
                solver=solver,
            )
        per_event_results.append(res_e)

    active_mask = per_event_results[0].active_mask
    c_mask = np.asarray(active_mask, dtype=bool)
    for ev, res_e in zip(mds.events[1:], per_event_results[1:]):
        if not np.array_equal(np.asarray(res_e.active_mask, dtype=bool), c_mask):
            raise ValueError(
                f"Inconsistent active CBM parameter masks between event "
                f"{mds.events[0].name!r} and event {ev.name!r}."
            )

    active_idx = jnp.asarray(np.flatnonzero(c_mask), dtype=jnp.int32)
    p_full = int(c_mask.shape[0])
    p_act = int(active_idx.shape[0])

    # Exact linear additivity of Schur-marginalized CBM Fisher information across events
    marginal_fisher = jnp.zeros((p_full, p_full), dtype=jnp.float64)
    whitened_fisher = jnp.zeros((p_act, p_act), dtype=jnp.float64)
    p_psi_list: list[int] = []
    for res_e in per_event_results:
        marginal_fisher = marginal_fisher + res_e.marginal_fisher
        whitened_fisher = whitened_fisher + res_e.whitened_fisher
        p_psi_list.append(max(int(res_e.joint_fisher.shape[0]) - p_full, 0))

    marginal_fisher = 0.5 * (marginal_fisher + marginal_fisher.T)
    whitened_fisher = 0.5 * (whitened_fisher + whitened_fisher.T)

    # Assemble block-arrow joint Fisher over (theta_CBM, psi_1, ..., psi_E)
    p_psi_tot = sum(p_psi_list)
    joint_fisher = jnp.zeros(
        (p_full + p_psi_tot, p_full + p_psi_tot), dtype=jnp.float64
    )
    psi_offset = p_full
    for res_e, p_psi_e in zip(per_event_results, p_psi_list):
        jf_e = res_e.joint_fisher
        joint_fisher = joint_fisher.at[:p_full, :p_full].add(jf_e[:p_full, :p_full])
        if p_psi_e > 0:
            sl = slice(psi_offset, psi_offset + p_psi_e)
            joint_fisher = joint_fisher.at[:p_full, sl].set(jf_e[:p_full, p_full:])
            joint_fisher = joint_fisher.at[sl, :p_full].set(jf_e[p_full:, :p_full])
            joint_fisher = joint_fisher.at[sl, sl].set(jf_e[p_full:, p_full:])
            psi_offset += p_psi_e
    joint_fisher = 0.5 * (joint_fisher + joint_fisher.T)

    # Shared prior covariance root L_Pi applied exactly once, preserving flat-prior guards
    j_col_norms = jnp.sqrt(jnp.maximum(jnp.diag(marginal_fisher), 0.0))[None, :]
    l_pi, _ = _resolve_cbm_prior_root(
        p_full,
        active_cbm_mask=c_mask,
        prior_cbm_root=prior_cbm_root,
        prior_cbm_cov=prior_cbm_cov,
        prior_cbm_prec=prior_cbm_prec,
        prior_cbm_sd=use_prior_sd,
        whitened_j_theta=j_col_norms,
    )

    if p_act > 0:
        evals_asc, evecs_asc = jnp.linalg.eigh(whitened_fisher)
        eigenvalues = jnp.maximum(evals_asc[::-1], 0.0)
        v_modes = evecs_asc[:, ::-1]
        loadings_act = l_pi @ v_modes
        parameter_loadings = (
            jnp.zeros((p_full, p_act), dtype=jnp.float64)
            .at[active_idx, :]
            .set(loadings_act)
        )
        # Shared prior precision I enters exactly once: Sigma_post = L_Pi (F_tot + I)^-1 L_Pi^T
        posterior_cov = (
            loadings_act * (1.0 / (1.0 + eigenvalues))[None, :]
        ) @ loadings_act.T
        posterior_cov = 0.5 * (posterior_cov + posterior_cov.T)
    else:
        eigenvalues = jnp.zeros((0,), dtype=jnp.float64)
        v_modes = jnp.zeros((0, 0), dtype=jnp.float64)
        parameter_loadings = jnp.zeros((p_full, 0), dtype=jnp.float64)
        posterior_cov = jnp.zeros((0, 0), dtype=jnp.float64)

    d_eff = jnp.sum(eigenvalues / (eigenvalues + 1.0))
    info_nats = 0.5 * jnp.sum(jnp.log1p(eigenvalues))
    info_bits = info_nats / math.log(2.0)

    eff_mass_fn = (
        mass_fn
        if mass_fn is not None
        else (
            (lambda m: m.reservoir_masses())
            if hasattr(cbm_eval, "reservoir_masses")
            else None
        )
    )

    flat_theta, unravel_theta = _partition_inexact(cbm_eval)
    mass_loadings: jax.Array | None = None
    if eff_mass_fn is not None:
        def _eval_mass(th: jax.Array) -> jax.Array:
            val = jnp.asarray(eff_mass_fn(unravel_theta(th)), dtype=jnp.float64).reshape(-1)
            return jnp.log(val) if log_space_observables else val

        j_mass = _jacfwd_chunked(_eval_mass, flat_theta)
        mass_loadings = j_mass @ parameter_loadings

    flux_loadings: jax.Array | None = None
    if flux_fn is not None:
        def _eval_flux(th: jax.Array) -> jax.Array:
            val = jnp.asarray(flux_fn(unravel_theta(th)), dtype=jnp.float64).reshape(-1)
            return jnp.log(val) if log_space_observables else val

        j_flux = _jacfwd_chunked(_eval_flux, flat_theta)
        flux_loadings = j_flux @ parameter_loadings

    return FisherSpectrumResult(
        eigenvalues=eigenvalues,
        d_eff=d_eff,
        info_nats=info_nats,
        info_bits=info_bits,
        prior_whitened_modes=v_modes,
        parameter_loadings=parameter_loadings,
        marginal_fisher=marginal_fisher,
        whitened_fisher=whitened_fisher,
        joint_fisher=joint_fisher,
        posterior_cov=posterior_cov,
        mass_loadings=mass_loadings,
        flux_loadings=flux_loadings,
        active_mask=jnp.asarray(c_mask),
    )


# ---------------------------------------------------------------------------
# 3. Block-diagonal multi-event marginal log-likelihood & log-posterior
# ---------------------------------------------------------------------------


def _resolve_event_sigma_jitter(
    sigma_jitter: Mapping[str, ArrayLike] | ArrayLike | None,
    event_name: str,
) -> ArrayLike | None:
    if sigma_jitter is None:
        return None
    if isinstance(sigma_jitter, Mapping):
        return sigma_jitter.get(event_name, None)
    return sigma_jitter


def multi_event_marginal_log_likelihood(
    model: MultiEventModel,
    multi_dataset: MultiEventDataset,
    *,
    sigma_jitter: Mapping[str, ArrayLike] | ArrayLike | None = None,
) -> jax.Array:
    """Block-diagonal multi-event marginal Gaussian log-likelihood.

    Sums per-event [`marginal_log_likelihood`][ticktack.likelihood.marginal_log_likelihood]
    values across all [`EventWindow`][ticktack.multievent.EventWindow] entries in
    ``O(sum_e N_e r_e^2)`` time without forming any cross-event dense covariance
    matrix. Each event window may independently carry a precomputed
    [`LinearizedDCTOperator`][ticktack.fields.LinearizedDCTOperator] in
    ``ev.marginal_config.dct_operator``.

    Parameters
    ----------
    model : MultiEventModel
        Shared-CBM multi-event radiocarbon model.
    multi_dataset : MultiEventDataset
        Pooled event windows containing observations and marginal configs.
    sigma_jitter : Mapping[str, ArrayLike] or ArrayLike, optional
        Optional per-event mapping or scalar additive observation jitter (‰).

    Returns
    -------
    jax.Array
        Scalar total marginal log-likelihood summed over all events.
    """
    total = jnp.asarray(0.0, dtype=jnp.float64)
    for ev in multi_dataset.events:
        m_e = model.event_model(ev.name)
        jit_e = _resolve_event_sigma_jitter(sigma_jitter, ev.name)
        total = total + marginal_log_likelihood(
            m_e,
            ev.dataset,
            ev.marginal_config,
            sigma_jitter=jit_e,
        )
    return total


def multi_event_marginal_log_posterior(
    model: MultiEventModel,
    multi_dataset: MultiEventDataset,
    parameter_space: ParameterSpace,
    u_vec: ArrayLike,
    *,
    sigma_jitter: Mapping[str, ArrayLike] | ArrayLike | None = None,
    target_space: Literal["unconstrained", "constrained"] = "unconstrained",
) -> jax.Array:
    """Multi-event marginal log-posterior evaluating shared-CBM priors exactly once.

    Parameters
    ----------
    model : MultiEventModel
        Template multi-event model whose free leaves are replaced via
        ``parameter_space.constrain(u_vec, model)``.
    multi_dataset : MultiEventDataset
        Pooled event windows.
    parameter_space : ParameterSpace
        Joint parameter space over shared CBM parameters and per-event leaves.
    u_vec : ArrayLike
        Unconstrained parameter vector of shape ``(d,)``.
    sigma_jitter : Mapping[str, ArrayLike] or ArrayLike, optional
        Optional per-event mapping or scalar additive observation jitter (‰).
    target_space : {"unconstrained", "constrained"}, optional
        Coordinate space in which ``parameter_space.log_prior_unconstrained``
        is evaluated (default ``"unconstrained"``, including change-of-variables
        log-Jacobian).

    Returns
    -------
    jax.Array
        Scalar joint marginal log-posterior.
    """
    constrained = parameter_space.constrain(u_vec, model)
    log_lk = multi_event_marginal_log_likelihood(
        constrained,
        multi_dataset,
        sigma_jitter=sigma_jitter,
    )
    log_pr = parameter_space.log_prior_unconstrained(
        u_vec, target_space=target_space
    )
    return log_lk + log_pr


# ---------------------------------------------------------------------------
# 4. Shared-modal tangent evaluation & exact multi-event IS reweighting
# ---------------------------------------------------------------------------


def _shared_cbm_context(
    model: MultiEventModel,
) -> tuple[
    tuple[jax.Array, jax.Array, jax.Array] | None,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Compute shared CBM modal decomposition and steady-state ratios once across all events."""
    cbm = model.cbm
    A = cbm.matrix_A()
    modal = (
        _modal_decomposition(A, cbm=cbm)
        if isinstance(model.solver, ExactLinearSolver)
        else None
    )
    y_ss = cbm.steady_state_c14(q_std=model.q_std, alpha_unit=model.alpha_unit)
    trop_idx = jnp.asarray(cbm.topology.trop_box_indices, dtype=jnp.int32)
    y_trop = jnp.sum(y_ss[trop_idx])
    m_rel = cbm.relative_reservoir_masses()
    r_ss_ratio = (y_ss / y_trop) / m_rel
    ss_d14c = 1000.0 * (r_ss_ratio - 1.0)
    obs_scale_boxes = 1000.0 / (y_trop * m_rel)
    return modal, r_ss_ratio, ss_d14c, obs_scale_boxes


def _fast_exact_nonlinear_residual(
    model_e: RadiocarbonModel,
    dataset: Dataset,
    cfg: LinearMarginalConfig,
    box_indices_flat: jax.Array,
    shared_ctx: tuple[
        tuple[jax.Array, jax.Array, jax.Array] | None,
        jax.Array,
        jax.Array,
        jax.Array,
    ],
) -> jax.Array:
    """Compute exact ``d - m_0`` reusing the shared CBM modal decomposition and steady state."""
    modal, _r_ss_ratio, ss_d14c, obs_scale_boxes = shared_ctx
    if not isinstance(model_e.solver, ExactLinearSolver):
        return _nonlinear_residual(model_e, dataset, cfg)

    unmarg, _, _, _, _ = _partition_terms_for_cfg(model_e.production, cfg)
    base_prod = _build_base_production(unmarg, model_e.q_std)
    all_ts, all_te, all_w = _concat_padded_windows(dataset)
    n_tot = int(box_indices_flat.shape[0])

    uniq_cache = getattr(dataset, "_uniq_subwindows", None)
    if uniq_cache is not None:
        uniq_a, uniq_b, inv_idx, m_c = uniq_cache
        dy_uniq = model_e.solver.solve_delta_binned(
            model_e.cbm,
            base_prod,
            jnp.asarray(uniq_a, dtype=jnp.float64),
            jnp.asarray(uniq_b, dtype=jnp.float64),
            q_std=model_e.q_std,
            alpha_unit=model_e.alpha_unit,
            modal=modal,
        )
        w_norm = all_w / jnp.sum(all_w, axis=-1, keepdims=True)
        dy_sub = dy_uniq[jnp.asarray(inv_idx, dtype=jnp.int32)].reshape(n_tot, m_c, -1)
        dy_binned = jnp.sum(w_norm[:, :, None] * dy_sub, axis=1)
    else:
        dy_binned = model_e.solver.solve_delta_binned(
            model_e.cbm,
            base_prod,
            all_ts,
            all_te,
            sub_weights=all_w,
            q_std=model_e.q_std,
            alpha_unit=model_e.alpha_unit,
            modal=modal,
        )

    pred_flat = (
        ss_d14c[box_indices_flat]
        + dy_binned[jnp.arange(n_tot), box_indices_flat]
        * obs_scale_boxes[box_indices_flat]
    )
    return dataset.d14c_flat - pred_flat


def _event_tan_log_likelihood_shared(
    model_e: RadiocarbonModel,
    dataset: Dataset,
    cfg: LinearMarginalConfig,
    lin_op: LinearizedDCTOperator | None,
    *,
    sigma_jitter: ArrayLike | None = None,
    shared_ctx: tuple[
        tuple[jax.Array, jax.Array, jax.Array] | None,
        jax.Array,
        jax.Array,
        jax.Array,
    ]
    | None = None,
) -> jax.Array:
    """Evaluate single-event tangent log-likelihood reusing the shared CBM modal context.

    ``SteadyProduction + MiyakeSpike`` trajectories and ``y_ss`` steady-state
    hierarchical offset ratios are evaluated **exactly** at ``model_e.cbm``;
    only ``W_DCT(alpha)`` uses ``lin_op.evaluate_w_dct(model_e.cbm.alpha)``.
    """
    if lin_op is None or not isinstance(model_e.cbm, SubspaceCBM):
        return marginal_log_likelihood(
            model_e, dataset, cfg, sigma_jitter=sigma_jitter
        )

    sigma = jnp.asarray(dataset.sigma_flat, dtype=jnp.float64)
    if sigma_jitter is not None:
        sigma = jnp.sqrt(
            jnp.square(sigma)
            + jnp.square(jnp.asarray(sigma_jitter, dtype=jnp.float64))
        )

    _, solar_terms, trend_terms, gp_terms, field_terms = _partition_terms_for_cfg(
        model_e.production, cfg
    )
    if (
        len(field_terms) != 1
        or solar_terms
        or trend_terms
        or gp_terms
        or dataset.n_obs == 0
    ):
        cfg_tan = eqx.tree_at(
            lambda c: c.dct_operator, cfg, lin_op, is_leaf=lambda x: x is None
        )
        return marginal_log_likelihood(
            model_e, dataset, cfg_tan, sigma_jitter=sigma_jitter
        )

    gf = field_terms[0]
    cbm = model_e.cbm
    if lin_op.w_ref.shape != (dataset.n_obs, gf.n_grid) or (
        abs(lin_op.t_start - gf.t_start) > 1e-9
        or abs(lin_op.t_end - gf.t_end) > 1e-9
    ):
        raise ValueError(
            "LinearizedDCTOperator geometry "
            f"(shape={lin_op.w_ref.shape}, [{lin_op.t_start}, {lin_op.t_end}]) "
            f"does not match dataset/GaussianField "
            f"(shape=({dataset.n_obs}, {gf.n_grid}), [{gf.t_start}, {gf.t_end}])."
        )
    if lin_op.dataset_digest:
        ds_dig = _compute_dataset_digest(cbm, dataset)
        if ds_dig and ds_dig != lin_op.dataset_digest:
            raise ValueError(
                "LinearizedDCTOperator dataset_digest does not match dataset."
            )
    if lin_op.template_digest:
        tmpl_dig = _compute_template_digest(gf.template)
        if tmpl_dig and tmpl_dig != lin_op.template_digest:
            raise ValueError(
                "LinearizedDCTOperator template_digest does not match GaussianField.template."
            )
    if lin_op.subspace_basis_digest:
        basis_dig = _compute_subspace_basis_digest(cbm)
        if basis_dig != lin_op.subspace_basis_digest:
            raise ValueError(
                "LinearizedDCTOperator subspace_basis_digest does not match model.cbm."
            )

    box_indices_flat = jnp.array(
        [
            model_e._box_index_for_series(s)
            for s in dataset.series
            for _ in range(s.n_obs)
        ],
        dtype=jnp.int32,
    )

    if shared_ctx is not None:
        _, r_ss_ratio, _, _ = shared_ctx
        resid = _fast_exact_nonlinear_residual(
            model_e, dataset, cfg, box_indices_flat, shared_ctx
        )
    else:
        resid = _nonlinear_residual(model_e, dataset, cfg)
        y_ss = cbm.steady_state_c14(q_std=model_e.q_std, alpha_unit=model_e.alpha_unit)
        trop_idx = jnp.asarray(cbm.topology.trop_box_indices, dtype=jnp.int32)
        y_trop = jnp.sum(y_ss[trop_idx])
        m_rel = cbm.relative_reservoir_masses()
        r_ss_ratio = (y_ss / y_trop) / m_rel

    obs_activity_ratios = r_ss_ratio[box_indices_flat]

    design_cols: list[jax.Array] = []
    pmean_parts: list[jax.Array] = []
    psd_parts: list[jax.Array] = []

    if cfg.offset_mode == "hierarchical":
        h_off, mu_off, sd_off = hierarchical_offset_design(
            dataset.series_indices,
            n_series=dataset.n_series,
            box_activity_ratios=obs_activity_ratios,
            global_prior_mean=cfg.global_prior_mean,
            global_prior_sd=cfg.global_prior_sd,
            lab_prior_sd=cfg.lab_prior_sd,
        )
        design_cols.append(h_off)
        pmean_parts.append(mu_off)
        psd_parts.append(sd_off)
    elif cfg.offset_mode == "per_series":
        h_off = dataset.lab_offset_design_matrix()
        design_cols.append(h_off)
        pmean_parts.append(
            jnp.full((dataset.n_series,), cfg.global_prior_mean, dtype=jnp.float64)
        )
        psd_parts.append(
            jnp.full((dataset.n_series,), cfg.global_prior_sd, dtype=jnp.float64)
        )
    elif cfg.offset_mode == "shared":
        design_cols.append(obs_activity_ratios[:, None])
        pmean_parts.append(jnp.array([cfg.global_prior_mean], dtype=jnp.float64))
        psd_parts.append(jnp.array([cfg.global_prior_sd], dtype=jnp.float64))

    w_tan = lin_op.evaluate_w_dct(cbm.alpha)
    h_gf = w_tan * _safe_sqrt(gf.power_spectrum())[None, :]
    design_cols.append(h_gf)
    pmean_parts.append(jnp.zeros((gf.n_grid,), dtype=jnp.float64))
    psd_parts.append(jnp.ones((gf.n_grid,), dtype=jnp.float64))

    X = jnp.concatenate(design_cols, axis=1)
    pm = jnp.concatenate(pmean_parts)
    ps = jnp.concatenate(psd_parts)
    lm = LinearMarginal(X, prior_mean=pm, prior_sd=ps, method=cfg.whiten_method)
    return lm.loglike(resid, sigma)


def _multi_event_tan_log_likelihood(
    model: MultiEventModel,
    multi_dataset: MultiEventDataset,
    tan_ops: Mapping[str, LinearizedDCTOperator],
    *,
    sigma_jitter: Mapping[str, ArrayLike] | ArrayLike | None = None,
) -> jax.Array:
    shared_ctx = _shared_cbm_context(model) if tan_ops else None
    total = jnp.asarray(0.0, dtype=jnp.float64)
    for ev in multi_dataset.events:
        m_e = model.event_model(ev.name)
        jit_e = _resolve_event_sigma_jitter(sigma_jitter, ev.name)
        op_e = tan_ops.get(ev.name, None)
        total = total + _event_tan_log_likelihood_shared(
            m_e,
            ev.dataset,
            ev.marginal_config,
            op_e,
            sigma_jitter=jit_e,
            shared_ctx=shared_ctx,
        )
    return total


@eqx.filter_jit
def _single_event_dlnl_one_jit(
    u_one: jax.Array,
    model_template: MultiEventModel,
    ev: EventWindow,
    parameter_space: ParameterSpace,
    cfg_tan_e: LinearMarginalConfig,
    sigma_jitter_e: Any,
) -> jax.Array:
    m_one = parameter_space.constrain(u_one, model_template)
    m_e = m_one.event_model(ev.name)
    cfg_exact_e = _strip_dct_operator(ev.marginal_config)
    ll_ex = marginal_log_likelihood(
        m_e, ev.dataset, cfg_exact_e, sigma_jitter=sigma_jitter_e
    )
    ll_tan = marginal_log_likelihood(
        m_e, ev.dataset, cfg_tan_e, sigma_jitter=sigma_jitter_e
    )
    return ll_ex - ll_tan


@eqx.filter_jit
def _single_event_ll_vg_jit(
    u: jax.Array,
    model: MultiEventModel,
    ev: EventWindow,
    parameter_space: ParameterSpace,
    sigma_jitter_e: Any,
) -> tuple[jax.Array, jax.Array]:
    cfg_exact_e = _strip_dct_operator(ev.marginal_config)

    def _ll_e(x: jax.Array) -> jax.Array:
        m_x = parameter_space.constrain(x, model)
        return marginal_log_likelihood(
            m_x.event_model(ev.name),
            ev.dataset,
            cfg_exact_e,
            sigma_jitter=sigma_jitter_e,
        )

    return jax.value_and_grad(_ll_e)(u)


@eqx.filter_jit
def _prior_vg_jit(
    u: jax.Array,
    parameter_space: ParameterSpace,
) -> tuple[jax.Array, jax.Array]:
    return jax.value_and_grad(parameter_space.log_prior_unconstrained)(u)


def _multi_neg_log_post_vg(
    u: jax.Array,
    model: MultiEventModel,
    multi_dataset: MultiEventDataset,
    parameter_space: ParameterSpace,
    sigma_jitter: Mapping[str, ArrayLike] | ArrayLike | None,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate exact ``(-ln p(u | d), grad)`` accumulating one event at a time.

    Accumulating per-event ``value_and_grad`` keeps XLA compilation memory
    ``O(1)`` in the number of events ``E`` rather than inlining ``E`` full
    DCT response operators into a single reverse-mode graph.
    """
    lp_val, lp_grad = _prior_vg_jit(u, parameter_space)
    total_val = lp_val
    total_grad = lp_grad
    for ev in multi_dataset.events:
        jit_e = _resolve_event_sigma_jitter(sigma_jitter, ev.name)
        ll_e, g_e = _single_event_ll_vg_jit(u, model, ev, parameter_space, jit_e)
        total_val = total_val + ll_e
        total_grad = total_grad + g_e
    return -total_val, -total_grad


def reweight_multi_event_exact(
    u_samples: ArrayLike,
    model_template: MultiEventModel,
    multi_dataset: MultiEventDataset,
    parameter_space: ParameterSpace,
    tan_ops: Mapping[str, LinearizedDCTOperator],
    *,
    max_draws: int | None = None,
    sigma_jitter: Mapping[str, ArrayLike] | ArrayLike | None = None,
) -> dict[str, Any]:
    r"""Compute exact multi-event importance-sampling weights.

    Evaluates the summed log-likelihood discrepancy

    $$\log w_s = \sum_{e=1}^{E} \left[\ln \mathcal{L}_{\mathrm{exact}, e}(\boldsymbol{u}_s) - \ln \mathcal{L}_{\mathrm{tan}, e}(\boldsymbol{u}_s)\right]$$

    across unconstrained posterior draws ``u_samples`` of shape ``(..., d)``
    and returns normalized importance weights, Kish effective sample size
    diagnostics, and log-evidence correction ``log_z_correction``.

    Parameters
    ----------
    u_samples : ArrayLike
        Unconstrained posterior draws of shape ``(..., d)``.
    model_template : MultiEventModel
        Template multi-event model used by ``parameter_space.constrain``.
    multi_dataset : MultiEventDataset
        Pooled event windows.
    parameter_space : ParameterSpace
        Joint parameter space matching the trailing axis of ``u_samples``.
    tan_ops : Mapping[str, LinearizedDCTOperator]
        Per-event MAP-centred tangent operators used during Stage-2 sampling.
    max_draws : int or None, optional
        Maximum number of evenly spaced draws to evaluate. ``None`` (default)
        evaluates all draws so ``weights`` aligns 1-to-1 with every draw.
    sigma_jitter : Mapping[str, ArrayLike] or ArrayLike, optional
        Optional per-event mapping or scalar additive observation jitter (‰).

    Returns
    -------
    dict[str, Any]
        Dictionary containing ``indices`` / ``eval_indices``, ``n_eval``,
        ``n_total``, ``log_weights``, ``weights``, ``kish_ess``,
        ``kish_ess_ratio``, ``log_z_correction``, ``median_abs_dlnL``,
        ``max_abs_dlnL``, and ``all_draws_evaluated``.
    """
    if max_draws is not None and int(max_draws) < 1:
        raise ValueError(f"max_draws must be >= 1 or None; got {max_draws!r}.")

    u_arr = jnp.asarray(u_samples, dtype=jnp.float64)
    d = parameter_space.n_params
    u_flat = u_arr.reshape(-1, d)
    n_tot = int(u_flat.shape[0])
    if n_tot == 0:
        empty_idx = np.zeros((0,), dtype=np.int64)
        empty_arr = jnp.zeros((0,), dtype=jnp.float64)
        return {
            "indices": empty_idx,
            "eval_indices": empty_idx,
            "n_eval": 0,
            "n_total": 0,
            "log_weights": empty_arr,
            "weights": empty_arr,
            "kish_ess": 0.0,
            "kish_ess_ratio": 1.0,
            "log_z_correction": 0.0,
            "median_abs_dlnL": 0.0,
            "max_abs_dlnL": 0.0,
            "all_draws_evaluated": True,
        }

    if max_draws is not None and n_tot > int(max_draws):
        idx = np.linspace(0, n_tot - 1, int(max_draws), dtype=np.int64)
        u_eval = u_flat[jnp.asarray(idx)]
        all_eval = False
    else:
        idx = np.arange(n_tot, dtype=np.int64)
        u_eval = u_flat
        all_eval = True

    s_eval = int(u_eval.shape[0])
    log_w_np = np.zeros((s_eval,), dtype=np.float64)
    for ev in multi_dataset.events:
        op_e = tan_ops.get(ev.name, None)
        if op_e is None:
            continue
        cfg_tan_e = eqx.tree_at(
            lambda c: c.dct_operator,
            ev.marginal_config,
            op_e,
            is_leaf=lambda x: x is None,
        )
        jit_e = _resolve_event_sigma_jitter(sigma_jitter, ev.name)
        for s_idx in range(s_eval):
            log_w_np[s_idx] += float(
                _single_event_dlnl_one_jit(
                    u_eval[s_idx],
                    model_template,
                    ev,
                    parameter_space,
                    cfg_tan_e,
                    jit_e,
                )
            )

    finite = np.isfinite(log_w_np)
    if np.any(finite):
        lw_max = float(np.max(log_w_np[finite]))
        w_unnorm = np.where(
            finite,
            np.exp(np.where(finite, log_w_np - lw_max, -np.inf)),
            0.0,
        )
        w_sum = float(np.sum(w_unnorm))
        w_is = w_unnorm / max(w_sum, 1e-300)
        kish_ess = float((np.sum(w_is) ** 2) / max(float(np.sum(w_is**2)), 1e-300))
        kish_ratio = float(kish_ess / max(s_eval, 1))
        log_z_corr = float(lw_max + math.log(max(w_sum, 1e-300)) - math.log(s_eval))
        abs_lw = np.abs(log_w_np[finite])
        med_abs = float(np.median(abs_lw))
        max_abs = float(np.max(abs_lw))
    else:
        w_is = np.full_like(log_w_np, np.nan)
        kish_ess = 0.0
        kish_ratio = 0.0
        log_z_corr = float("nan")
        med_abs = float("nan")
        max_abs = float("nan")

    return {
        "indices": idx,
        "eval_indices": idx,
        "n_eval": s_eval,
        "n_total": n_tot,
        "log_weights": jnp.asarray(log_w_np, dtype=jnp.float64),
        "weights": jnp.asarray(w_is, dtype=jnp.float64),
        "kish_ess": kish_ess,
        "kish_ess_ratio": kish_ratio,
        "log_z_correction": log_z_corr,
        "median_abs_dlnL": med_abs,
        "max_abs_dlnL": max_abs,
        "all_draws_evaluated": all_eval,
    }


# ---------------------------------------------------------------------------
# 5. End-to-end joint multi-event MAP + tangent-linear NUTS + IS fit
# ---------------------------------------------------------------------------


class MultiEventFitResult(NamedTuple):
    """Results from [`fit_multi_event_subspace_cbm`][ticktack.multievent.fit_multi_event_subspace_cbm].

    Attributes
    ----------
    model : MultiEventModel
        Fitted multi-event model at the joint exact MAP estimate.
    u_map : jax.Array
        Unconstrained joint MAP parameter vector, shape ``(d,)``.
    theta_map : dict[str, jax.Array]
        Constrained joint MAP parameter dictionary ``{path: value}``.
    log_posterior_map : jax.Array
        Exact multi-event marginal log-posterior at MAP.
    log_likelihood_map : jax.Array
        Exact multi-event marginal log-likelihood at MAP.
    laplace_cov : jax.Array
        Regularized inverse-Hessian Laplace covariance in ``u``-space, shape ``(d, d)``.
    map_converged : bool
        Whether the exact MAP L-BFGS-B optimization converged to a finite state.
    map_steps : int
        Number of MAP optimizer steps taken.
    idata : arviz.InferenceData or None
        ArviZ ``InferenceData`` from NUTS sampling (with
        ``sample_stats["log_weight_is"]`` populated whenever
        ``reweight_exact=True``), or ``None`` when ``sampler="laplace"``.
    parameter_space : ParameterSpace
        Joint parameter space over shared CBM and per-event parameters.
    dct_operators : dict[str, LinearizedDCTOperator]
        Per-event MAP-centred
        [`LinearizedDCTOperator`][ticktack.fields.LinearizedDCTOperator]
        instances keyed by event name.
    log_weights_is : jax.Array or None
        Exact importance-sampling log-weights
        ``sum_e (ln L_exact,e - ln L_tan,e)`` on the evaluated draws.
    weights_is : jax.Array or None
        Normalized importance-sampling weights summing to 1 on the evaluated draws.
    kish_ess_ratio : float or None
        Kish effective sample size fraction ``ESS_IS / S`` under exact IS reweighting.
    median_abs_dlnL : float or None
        Median ``|ln L_exact - ln L_tan|`` across evaluated posterior draws.
    max_abs_dlnL : float or None
        Maximum ``|ln L_exact - ln L_tan|`` across evaluated posterior draws.
    event_linear_posteriors : dict[str, LinearPosteriorResult]
        Per-event conditional
        [`LinearPosteriorResult`][ticktack.likelihood.LinearPosteriorResult] at MAP.
    diagnostics : dict[str, Any]
        Summary diagnostics including ``map_converged``, ``n_divergences``,
        ``n_max_treedepth``, ``max_rhat``, ``min_ess_bulk``,
        ``laplace_mass_fallback``, ``eval_indices``, ``log_z_correction``,
        ``kish_ess_ratio``, and ``per_event_chi2_per_n``.
    """

    model: MultiEventModel
    u_map: jax.Array
    theta_map: dict[str, jax.Array]
    log_posterior_map: jax.Array
    log_likelihood_map: jax.Array
    laplace_cov: jax.Array
    map_converged: bool
    map_steps: int
    idata: az.InferenceData | None
    parameter_space: ParameterSpace
    dct_operators: dict[str, LinearizedDCTOperator]
    log_weights_is: jax.Array | None
    weights_is: jax.Array | None
    kish_ess_ratio: float | None
    median_abs_dlnL: float | None
    max_abs_dlnL: float | None
    event_linear_posteriors: dict[str, LinearPosteriorResult]
    diagnostics: dict[str, Any]


def _compute_mcmc_convergence_diagnostics(
    idata: az.InferenceData,
    num_chains: int,
) -> tuple[float, float]:
    """Return ``(max_rhat, min_ess_bulk)`` across all posterior variables in ``idata``."""
    max_rhat = float("nan")
    if num_chains >= 2:
        try:
            rhat_ds = az.rhat(idata)
            rhat_vals = [
                np.ravel(np.asarray(da.values, dtype=np.float64))
                for da in rhat_ds.data_vars.values()
            ]
            if rhat_vals:
                flat_rh = np.concatenate(rhat_vals)
                fin_rh = flat_rh[np.isfinite(flat_rh)]
                if fin_rh.size > 0:
                    max_rhat = float(np.max(fin_rh))
        except Exception:
            max_rhat = float("nan")

    min_ess_bulk = float("nan")
    try:
        ess_ds = az.ess(idata, method="bulk")
        ess_vals = [
            np.ravel(np.asarray(da.values, dtype=np.float64))
            for da in ess_ds.data_vars.values()
        ]
        if ess_vals:
            flat_ess = np.concatenate(ess_vals)
            fin_ess = flat_ess[np.isfinite(flat_ess)]
            if fin_ess.size > 0:
                min_ess_bulk = float(np.min(fin_ess))
    except Exception:
        min_ess_bulk = float("nan")

    return max_rhat, min_ess_bulk


def fit_multi_event_subspace_cbm(
    model: MultiEventModel,
    multi_dataset: MultiEventDataset,
    parameter_space: ParameterSpace,
    *,
    sampler: Literal["nuts", "laplace"] = "nuts",
    reweight_exact: bool = True,
    use_linearized_dct: bool = True,
    is_max_draws: int | None = None,
    map_max_steps: int = 500,
    map_tol: float = 1e-6,
    num_warmup: int = 200,
    num_samples: int = 300,
    num_chains: int = 2,
    chain_method: Literal["sequential", "vectorized", "parallel"] = "sequential",
    target_accept_prob: float = 0.85,
    max_tree_depth: int = 10,
    dense_mass: bool = True,
    adapt_mass_matrix: bool | None = None,
    rng_key: jax.Array | None = None,
    sigma_jitter: Mapping[str, ArrayLike] | ArrayLike | None = None,
    compute_linear_posterior: bool = True,
    progress_bar: bool = False,
) -> MultiEventFitResult:
    """Jointly infer a shared ``SubspaceCBM`` and per-event spikes across multiple events.

    Runs a three-stage exact / asymptotically exact workflow:

    1. **Exact joint MAP + Laplace covariance**: maximizes
       [`multi_event_marginal_log_posterior`][ticktack.multievent.multi_event_marginal_log_posterior]
       using L-BFGS-B with per-event gradient accumulation (always stripping any
       pre-attached tangent operators) and computes the regularized
       inverse-Hessian covariance at ``u_map``.
    2. **MAP-centred per-event**
       [`LinearizedDCTOperator`][ticktack.fields.LinearizedDCTOperator] **+
       Laplace-preconditioned NUTS** (when ``sampler="nuts"``): builds each
       event's tangent-linear DCT operator via
       [`build_linearized_dct_operator`][ticktack.fields.build_linearized_dct_operator]
       around ``alpha_MAP`` while keeping ``SteadyProduction + MiyakeSpike`` and
       steady-state ``y_ss`` offset ratios **exact**, and samples the joint
       posterior in non-centred coordinates preconditioned by the Laplace
       covariance.
    3. **Exact multi-event IS reweighting** (when ``reweight_exact=True``):
       evaluates ``log_w = sum_e (ln L_exact,e - ln L_tan,e)`` across posterior
       draws via
       [`reweight_multi_event_exact`][ticktack.multievent.reweight_multi_event_exact],
       always attaching a ``(num_chains, num_samples)`` ``log_weight_is`` array
       to ``idata.sample_stats`` (with ``NaN`` on unevaluated draws if
       ``is_max_draws`` subsamples).

    Parameters
    ----------
    model : MultiEventModel
        Template multi-event model with shared ``SubspaceCBM`` (or other
        ``CarbonBoxModel``) and per-event production terms.
    multi_dataset : MultiEventDataset
        Pooled event windows.
    parameter_space : ParameterSpace
        Joint parameter space over shared CBM parameters and per-event leaves.
    sampler : {"nuts", "laplace"}, optional
        Posterior inference mode (default ``"nuts"``).
    reweight_exact : bool, optional
        If ``True`` (default), compute exact multi-event importance-sampling
        weights correcting tangent-linear NUTS draws back to the exact marginal
        posterior.
    use_linearized_dct : bool, optional
        If ``True`` (default) and ``model.cbm`` is a ``SubspaceCBM``, build
        MAP-centred per-event ``LinearizedDCTOperator``s for Stage-2 NUTS.
    is_max_draws : int or None, optional
        Maximum evenly spaced draws to evaluate in Stage-3 IS reweighting.
        ``None`` (default) evaluates all ``num_chains * num_samples`` draws so
        ``weights_is`` aligns 1-to-1 with every posterior draw.
    map_max_steps : int, optional
        Maximum L-BFGS-B iterations in Stage 1 (default ``500``).
    map_tol : float, optional
        Projected-gradient tolerance ``gtol`` for Stage-1 L-BFGS-B (default ``1e-6``).
    num_warmup, num_samples, num_chains : int, optional
        NUTS warmup steps, post-warmup samples per chain, and chain count.
    chain_method : {"sequential", "vectorized", "parallel"}, optional
        NumPyro chain execution strategy (default ``"sequential"``).
    target_accept_prob : float, optional
        Target NUTS acceptance probability (default ``0.85``).
    max_tree_depth : int, optional
        Maximum NUTS binary tree depth (default ``10``).
    dense_mass : bool, optional
        Whether to use a dense inverse mass matrix initialized from the Laplace
        covariance (default ``True``).
    adapt_mass_matrix : bool or None, optional
        Whether NumPyro adapts the mass matrix during warmup. ``None``
        (default) enables adaptation whenever Laplace preconditioning fell back
        or ``num_warmup >= 150``.
    rng_key : jax.Array, optional
        PRNG key for NUTS initialization and sampling (defaults to ``PRNGKey(0)``).
    sigma_jitter : Mapping[str, ArrayLike] or ArrayLike, optional
        Optional per-event mapping or scalar additive observation jitter (‰).
    compute_linear_posterior : bool, optional
        If ``True`` (default), compute conditional per-event linear posteriors
        and raw unrescaled ``chi2 / N`` at MAP.
    progress_bar : bool, optional
        Whether to show NumPyro's progress bar (default ``False``).

    Returns
    -------
    MultiEventFitResult
        Joint MAP, Laplace covariance, NUTS ``InferenceData``, exact IS weights,
        conditional linear posteriors, and convergence diagnostics.
    """
    if sampler not in ("nuts", "laplace"):
        raise ValueError(f"sampler must be 'nuts' or 'laplace'; got {sampler!r}.")
    if is_max_draws is not None and int(is_max_draws) < 1:
        raise ValueError(f"is_max_draws must be >= 1 or None; got {is_max_draws!r}.")
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    d = parameter_space.n_params
    if sampler == "nuts" and d == 0:
        raise ValueError(
            "sampler='nuts' requires at least one free parameter in parameter_space."
        )
    u_init = parameter_space.unconstrain(model)

    # ------------------------------------------------------------------
    # Stage 1: Exact joint MAP & central-diff Laplace covariance
    # ------------------------------------------------------------------
    if d > 0:
        def _scipy_obj(x_np: np.ndarray) -> tuple[float, np.ndarray]:
            val, grad = _multi_neg_log_post_vg(
                jnp.asarray(x_np, dtype=jnp.float64),
                model,
                multi_dataset,
                parameter_space,
                sigma_jitter,
            )
            val_f = float(val)
            grad_np = np.asarray(grad, dtype=np.float64)
            if not (math.isfinite(val_f) and np.all(np.isfinite(grad_np))):
                return float("inf"), np.full_like(x_np, np.nan, dtype=np.float64)
            return val_f, grad_np

        opt_res = _scipy_minimize(
            _scipy_obj,
            np.asarray(u_init, dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": int(map_max_steps),
                "gtol": float(map_tol),
                "ftol": max(float(map_tol) * 1e-3, 1e-12),
            },
        )
        u_map = jnp.asarray(opt_res.x, dtype=jnp.float64)
        map_converged = bool(opt_res.success)
        map_steps = int(getattr(opt_res, "nit", 0))
    else:
        u_map = jnp.zeros((0,), dtype=jnp.float64)
        map_converged = True
        map_steps = 0

    model_map = parameter_space.constrain(u_map, model)
    theta_map: dict[str, jax.Array] = {}
    for spec, shape, (start, stop) in zip(
        parameter_space.specs,
        parameter_space.leaf_shapes,
        parameter_space.spec_slices,
    ):
        if stop == start:
            theta_map[spec.path] = jnp.zeros(shape, dtype=jnp.float64)
        else:
            theta_map[spec.path] = spec.constrain(u_map[start:stop].reshape(shape))

    if d > 0:
        neg_lp_map, _ = _multi_neg_log_post_vg(
            u_map, model, multi_dataset, parameter_space, sigma_jitter
        )
        log_post_map = -neg_lp_map
        log_lk_map = log_post_map - parameter_space.log_prior_unconstrained(u_map)
        map_converged = bool(
            map_converged
            and math.isfinite(float(log_post_map))
            and bool(jnp.all(jnp.isfinite(u_map)))
        )

        u_map_np = np.asarray(u_map, dtype=np.float64)
        h_step = 1e-5
        hess_cols = np.empty((d, d), dtype=np.float64)
        for j in range(d):
            e_j = np.zeros(d, dtype=np.float64)
            e_j[j] = h_step
            _, g_p = _multi_neg_log_post_vg(
                jnp.asarray(u_map_np + e_j, dtype=jnp.float64),
                model,
                multi_dataset,
                parameter_space,
                sigma_jitter,
            )
            _, g_m = _multi_neg_log_post_vg(
                jnp.asarray(u_map_np - e_j, dtype=jnp.float64),
                model,
                multi_dataset,
                parameter_space,
                sigma_jitter,
            )
            hess_cols[:, j] = (np.asarray(g_p) - np.asarray(g_m)) / (2.0 * h_step)
        hess = jnp.asarray(0.5 * (hess_cols + hess_cols.T), dtype=jnp.float64)
        laplace_cov = _regularize_hessian_to_cov(
            hess, parameter_space.prior_sds, regularize=1e-6
        )
    else:
        mds_exact = multi_dataset.with_marginal_configs(
            {ev.name: _strip_dct_operator(ev.marginal_config) for ev in multi_dataset.events}
        )
        log_lk_map = multi_event_marginal_log_likelihood(
            model_map, mds_exact, sigma_jitter=sigma_jitter
        )
        log_post_map = log_lk_map
        map_converged = bool(math.isfinite(float(log_post_map)))
        laplace_cov = jnp.zeros((0, 0), dtype=jnp.float64)

    if not map_converged:
        warnings.warn(
            f"fit_multi_event_subspace_cbm Stage-1 MAP optimization did not converge "
            f"(steps={map_steps}, log_posterior={float(log_post_map):.4g}).",
            RuntimeWarning,
            stacklevel=2,
        )

    # Conditional per-event linear posteriors & raw chi2 / N at MAP (exact operator)
    event_linear_post: dict[str, LinearPosteriorResult] = {}
    per_event_chi2_per_n: dict[str, float] = {}
    if compute_linear_posterior:
        for ev in multi_dataset.events:
            jit_e = _resolve_event_sigma_jitter(sigma_jitter, ev.name)
            lp_e = linear_posterior(
                model_map.event_model(ev.name),
                ev.dataset,
                _strip_dct_operator(ev.marginal_config),
                sigma_jitter=jit_e,
            )
            event_linear_post[ev.name] = lp_e
            if ev.n_obs > 0 and lp_e.ring_pred_mean is not None:
                z_raw = (ev.dataset.d14c_flat - lp_e.ring_pred_mean) / ev.dataset.sigma_flat
                per_event_chi2_per_n[ev.name] = float(
                    jnp.sum(jnp.square(z_raw)) / ev.n_obs
                )

    if sampler == "laplace":
        return MultiEventFitResult(
            model=model_map,
            u_map=u_map,
            theta_map=theta_map,
            log_posterior_map=log_post_map,
            log_likelihood_map=log_lk_map,
            laplace_cov=laplace_cov,
            map_converged=map_converged,
            map_steps=map_steps,
            idata=None,
            parameter_space=parameter_space,
            dct_operators={},
            log_weights_is=None,
            weights_is=None,
            kish_ess_ratio=None,
            median_abs_dlnL=None,
            max_abs_dlnL=None,
            event_linear_posteriors=event_linear_post,
            diagnostics={
                "sampler": "laplace",
                "map_converged": map_converged,
                "map_steps": map_steps,
                "per_event_chi2_per_n": per_event_chi2_per_n,
            },
        )

    # ------------------------------------------------------------------
    # Stage 2: Per-event MAP-centred LinearizedDCTOperator + NUTS
    # ------------------------------------------------------------------
    tan_ops: dict[str, LinearizedDCTOperator] = {}
    if use_linearized_dct and isinstance(model_map.cbm, SubspaceCBM):
        for ev in multi_dataset.events:
            _, solar_t, trend_t, gp_t, field_t = _partition_terms_for_cfg(
                model_map.events[ev.name], ev.marginal_config
            )
            if (
                len(field_t) == 1
                and not solar_t
                and not trend_t
                and not gp_t
                and ev.n_obs > 0
            ):
                gf = field_t[0]
                tan_ops[ev.name] = build_linearized_dct_operator(
                    model_map.cbm,
                    ev.dataset,
                    t_start=gf.t_start,
                    t_end=gf.t_end,
                    n_grid=gf.n_grid,
                    q_std=model_map.q_std,
                    alpha_unit=model_map.alpha_unit,
                    template=gf.template,
                )

    prior_means = parameter_space.prior_means
    prior_sds = parameter_space.prior_sds

    init_params = None
    inv_mass_init = None
    laplace_mass_fallback = False
    if d > 0:
        z_center = (u_map - prior_means) / prior_sds
        if not bool(jnp.all(jnp.isfinite(z_center))):
            warnings.warn(
                "fit_multi_event_subspace_cbm MAP z_center contains non-finite values; "
                "initializing chains around the prior mean.",
                RuntimeWarning,
                stacklevel=2,
            )
            z_center = jnp.zeros((d,), dtype=jnp.float64)

        L_z: jax.Array | None = None
        fallback_reason: str | None = None
        try:
            inv_sds = 1.0 / prior_sds
            cov_z = 0.5 * (laplace_cov + laplace_cov.T) * (
                inv_sds[:, None] * inv_sds[None, :]
            )
            cov_z = jnp.clip(cov_z, -1e4, 1e4)
            eigvals, eigvecs = jnp.linalg.eigh(cov_z)
            safe_ev = jnp.clip(eigvals, 1e-6, 10.0)
            cov_z_reg = (eigvecs * safe_ev[None, :]) @ eigvecs.T
            cov_z_reg = 0.5 * (cov_z_reg + cov_z_reg.T)
            if not bool(jnp.all(jnp.isfinite(cov_z_reg))):
                fallback_reason = "regularized Laplace covariance is non-finite"
            else:
                L_z_cand = jnp.linalg.cholesky(cov_z_reg)
                if not bool(jnp.all(jnp.isfinite(L_z_cand))):
                    fallback_reason = "Cholesky factor of regularized Laplace covariance is non-finite"
                else:
                    L_z = L_z_cand
                    inv_mass_init = {
                        ("z",): cov_z_reg if dense_mass else jnp.diag(cov_z_reg)
                    }
        except Exception as exc:
            fallback_reason = str(exc)

        if fallback_reason is not None:
            laplace_mass_fallback = True
            inv_mass_init = None
            L_z = None
            warnings.warn(
                "fit_multi_event_subspace_cbm Laplace mass-matrix fallback to identity: "
                f"{fallback_reason}.",
                RuntimeWarning,
                stacklevel=2,
            )

        rng_key, perturb_key = jax.random.split(rng_key)
        raw_eps = jax.random.normal(
            perturb_key, shape=(num_chains, d), dtype=jnp.float64
        )
        perturb = 0.15 * (raw_eps @ L_z.T) if L_z is not None else 0.05 * raw_eps
        z_starts = z_center[None, :] + perturb
        init_params = {"z": z_starts[0] if num_chains == 1 else z_starts}

    def _numpyro_multi_model() -> None:
        z = numpyro.sample(
            "z",
            dist.Normal(
                jnp.zeros((d,), dtype=jnp.float64),
                jnp.ones((d,), dtype=jnp.float64),
            ),
        )
        u_vec = prior_means + prior_sds * z
        m_u = parameter_space.constrain(u_vec, model)
        if tan_ops:
            log_lk = _multi_event_tan_log_likelihood(
                m_u, multi_dataset, tan_ops, sigma_jitter=sigma_jitter
            )
        else:
            log_lk = multi_event_marginal_log_likelihood(
                m_u, multi_dataset, sigma_jitter=sigma_jitter
            )
        numpyro.factor("log_likelihood", log_lk)

    eff_adapt_mass = (
        bool(adapt_mass_matrix)
        if adapt_mass_matrix is not None
        else (inv_mass_init is None or num_warmup >= 150)
    )
    nuts_kwargs: dict[str, Any] = {
        "target_accept_prob": target_accept_prob,
        "dense_mass": dense_mass,
        "adapt_mass_matrix": eff_adapt_mass,
        "max_tree_depth": max_tree_depth,
        "inverse_mass_matrix": inv_mass_init,
    }
    if inv_mass_init is not None:
        nuts_kwargs["step_size"] = 0.35
        nuts_kwargs["find_heuristic_step_size"] = False
    nuts_kernel = NUTS(_numpyro_multi_model, **nuts_kwargs)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method if num_chains > 1 else "sequential",
        progress_bar=progress_bar,
    )

    extra_fields = (
        "potential_energy",
        "energy",
        "diverging",
        "num_steps",
        "accept_prob",
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
            f"fit_multi_event_subspace_cbm NUTS detected {n_div} divergent transition(s) "
            f"out of {div_np.size} post-warmup draws; consider increasing "
            "target_accept_prob or num_warmup.",
            RuntimeWarning,
            stacklevel=2,
        )
    n_max_td = int(np.sum(max_td_np))
    if n_max_td > 0:
        warnings.warn(
            f"fit_multi_event_subspace_cbm NUTS saturated max_tree_depth={max_tree_depth} "
            f"on {n_max_td} out of {max_td_np.size} post-warmup draws; consider increasing "
            "max_tree_depth.",
            RuntimeWarning,
            stacklevel=2,
        )

    z_samples = mcmc.get_samples(group_by_chain=True)["z"]  # (C, S, d)
    u_samples = prior_means[None, None, :] + prior_sds[None, None, :] * z_samples
    posterior_dict = _unpack_constrained_samples(parameter_space, u_samples)

    # ------------------------------------------------------------------
    # Stage 3: Exact multi-event importance-sampling reweighting
    # ------------------------------------------------------------------
    log_weights_is: jax.Array | None = None
    weights_is: jax.Array | None = None
    kish_ess: float | None = None
    kish_ess_ratio: float | None = None
    log_z_correction: float | None = None
    median_abs_dlnL: float | None = None
    max_abs_dlnL: float | None = None
    eval_indices: np.ndarray | None = None
    n_eval: int | None = None
    n_tot_draws = num_chains * num_samples
    all_draws_evaluated: bool | None = None

    sample_stats_dict: dict[str, Any] = {
        "lp": -pe_np,
        "energy": en_np,
        "diverging": div_np,
        "n_steps": n_steps_np,
        "tree_depth": tree_depth_np,
        "reached_max_treedepth": max_td_np,
        "acceptance_rate": acc_np,
        "step_size": step_sz_np,
    }

    if reweight_exact:
        if tan_ops:
            is_info = reweight_multi_event_exact(
                u_samples,
                model,
                multi_dataset,
                parameter_space,
                tan_ops,
                max_draws=is_max_draws,
                sigma_jitter=sigma_jitter,
            )
            log_weights_is = is_info["log_weights"]
            weights_is = is_info["weights"]
            eval_indices = np.asarray(is_info["eval_indices"], dtype=np.int64)
            n_eval = int(is_info["n_eval"])
            all_draws_evaluated = bool(is_info["all_draws_evaluated"])
            kish_ess = float(is_info["kish_ess"])
            kish_ess_ratio = float(is_info["kish_ess_ratio"])
            log_z_correction = float(is_info["log_z_correction"])
            median_abs_dlnL = float(is_info["median_abs_dlnL"])
            max_abs_dlnL = float(is_info["max_abs_dlnL"])

            log_w_full = np.full((n_tot_draws,), np.nan, dtype=np.float64)
            log_w_full[eval_indices] = np.asarray(log_weights_is, dtype=np.float64)
            sample_stats_dict["log_weight_is"] = log_w_full.reshape(
                num_chains, num_samples
            )
        else:
            eval_indices = np.arange(n_tot_draws, dtype=np.int64)
            n_eval = n_tot_draws
            all_draws_evaluated = True
            log_weights_is = jnp.zeros((n_tot_draws,), dtype=jnp.float64)
            weights_is = jnp.full(
                (n_tot_draws,), 1.0 / max(n_tot_draws, 1), dtype=jnp.float64
            )
            kish_ess = float(n_tot_draws)
            kish_ess_ratio = 1.0
            log_z_correction = 0.0
            median_abs_dlnL = 0.0
            max_abs_dlnL = 0.0
            sample_stats_dict["log_weight_is"] = np.zeros(
                (num_chains, num_samples), dtype=np.float64
            )

        if (
            kish_ess_ratio is None
            or not math.isfinite(kish_ess_ratio)
            or kish_ess_ratio < 0.85
        ):
            warnings.warn(
                f"fit_multi_event_subspace_cbm exact IS efficiency is low "
                f"(kish_ess_ratio={kish_ess_ratio}, max|ΔlnL|={max_abs_dlnL} nats); "
                "consider checking posterior multimodality or sampling with "
                "use_linearized_dct=False.",
                RuntimeWarning,
                stacklevel=2,
            )

    idata = az.from_dict(
        {
            "posterior": posterior_dict,
            "sample_stats": sample_stats_dict,
        }
    )
    max_rhat, min_ess_bulk = _compute_mcmc_convergence_diagnostics(idata, num_chains)

    diagnostics = {
        "sampler": "nuts",
        "map_converged": map_converged,
        "map_steps": map_steps,
        "laplace_mass_fallback": laplace_mass_fallback,
        "n_divergences": n_div,
        "n_max_treedepth": n_max_td,
        "mean_accept_prob": float(np.mean(acc_np)),
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess_bulk,
        "used_linearized_dct": bool(tan_ops),
        "eval_indices": eval_indices,
        "n_eval": n_eval,
        "n_total": n_tot_draws,
        "all_draws_evaluated": all_draws_evaluated,
        "kish_ess": kish_ess,
        "kish_ess_ratio": kish_ess_ratio,
        "log_z_correction": log_z_correction,
        "median_abs_dlnL": median_abs_dlnL,
        "max_abs_dlnL": max_abs_dlnL,
        "per_event_chi2_per_n": per_event_chi2_per_n,
    }

    return MultiEventFitResult(
        model=model_map,
        u_map=u_map,
        theta_map=theta_map,
        log_posterior_map=log_post_map,
        log_likelihood_map=log_lk_map,
        laplace_cov=laplace_cov,
        map_converged=map_converged,
        map_steps=map_steps,
        idata=idata,
        parameter_space=parameter_space,
        dct_operators=tan_ops,
        log_weights_is=log_weights_is,
        weights_is=weights_is,
        kish_ess_ratio=kish_ess_ratio,
        median_abs_dlnL=median_abs_dlnL,
        max_abs_dlnL=max_abs_dlnL,
        event_linear_posteriors=event_linear_post,
        diagnostics=diagnostics,
    )
