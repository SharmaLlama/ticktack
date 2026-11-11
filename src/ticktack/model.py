"""Top-level ``RadiocarbonModel`` integrating CBM, production, and solver.

:class:`RadiocarbonModel` is the central user-facing ``zodiax.Base`` PyTree,
wrapping a :class:`~ticktack.cbm.CarbonBoxModel`, a
:class:`~ticktack.production.ProductionTerm`, and an
:class:`~ticktack.solvers.AbstractSolver`, exposing:

* ``steady_state()`` / ``reference_specific_activity()``: closed-form
  pre-industrial steady-state ``14C`` inventory and single global
  tropospheric reference specific activity (preserving the ``~4.38‰``
  NH–SH gradient).
* ``predict_d14c(dataset_or_series, ...)``: seasonal-window-binned
  ``Delta14C`` (‰) for a :class:`~ticktack.data.TreeRingSeries` or a
  :class:`~ticktack.data.Dataset`, routing each series to its hemisphere's
  tropospheric box and optionally concatenating the per-series arrays.
* ``predict_fine(t_eval, ...)``: ``Delta14C`` on a continuous time grid.
* ``impulse_response(t_lag, ...)`` / ``transfer_function(omegas, ...)``:
  system identification helpers.
* Explicit fluence reporting helpers: ``fluence_atoms_cm2(fluence)`` and
  ``fluence_q0_years(fluence, ...)``.

All methods are differentiable via JAX and are designed to be ``jit``-compiled
through ``zodiax`` / ``equinox`` ``filter_jit``.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import zodiax as zdx

from .cbm import CarbonBoxModel
from .constants import (
    ALPHA_ATOMS_CM2_S_TO_KG_YR,
    DEFAULT_Q_STD,
    SECONDS_PER_JULIAN_YEAR,
)
from .data import Dataset, TreeRingSeries
from .production import ProductionTerm
from .solvers import AbstractSolver, ExactLinearSolver

__all__ = [
    "RadiocarbonModel",
]


class RadiocarbonModel(zdx.Base):
    """Immutable ``zodiax`` top-level radiocarbon model.

    Wraps a :class:`~ticktack.cbm.CarbonBoxModel` (CBM), a
    :class:`~ticktack.production.ProductionTerm`, and an
    :class:`~ticktack.solvers.AbstractSolver` into a unified differentiable
    PyTree supporting ``zodiax`` string-path parameter addressing:

    .. code-block:: python

        model = RadiocarbonModel(cbm, production, solver)
        t_start = model.get("production.spike.t_start")     # read leaf
        model2  = model.set("production.spike.t_start", 774.6)  # update

    Parameters
    ----------
    cbm : CarbonBoxModel
        Carbon box model (e.g. :class:`~ticktack.cbm.FixedCBM`,
        :class:`~ticktack.cbm.BalancedCBM`, or :class:`~ticktack.cbm.EffectiveCBM`).
    production : ProductionTerm
        Cosmic-ray production model (e.g.
        :class:`~ticktack.production.ProductionSum`,
        :class:`~ticktack.production.MiyakeSpike`).
    solver : AbstractSolver
        ODE or exact matrix-exponential solver.  Defaults to
        :class:`~ticktack.solvers.ExactLinearSolver`.
    q_std : float
        Pre-industrial reference production rate (``atoms cm^-2 s^-1``;
        default ``1.76``).
    alpha_unit : float
        Conversion factor from ``atoms cm^-2 s^-1`` to ``kg yr^-1``
        (default CODATA value from :mod:`~ticktack.constants`).
    """

    cbm: CarbonBoxModel
    production: ProductionTerm
    solver: AbstractSolver
    q_std: jax.Array
    alpha_unit: jax.Array

    def __init__(
        self,
        cbm: CarbonBoxModel,
        production: ProductionTerm,
        solver: AbstractSolver | None = None,
        *,
        q_std: ArrayLike = DEFAULT_Q_STD,
        alpha_unit: ArrayLike = ALPHA_ATOMS_CM2_S_TO_KG_YR,
    ) -> None:
        if not isinstance(cbm, CarbonBoxModel):
            raise TypeError(f"cbm must be a CarbonBoxModel, got {type(cbm)!r}.")
        if not isinstance(production, ProductionTerm):
            raise TypeError(f"production must be a ProductionTerm, got {type(production)!r}.")
        if solver is None:
            solver = ExactLinearSolver()
        if not isinstance(solver, AbstractSolver):
            raise TypeError(f"solver must be an AbstractSolver, got {type(solver)!r}.")
        self.cbm = cbm
        self.production = production
        self.solver = solver
        self.q_std = jnp.asarray(q_std, dtype=jnp.float64).reshape(())
        self.alpha_unit = jnp.asarray(alpha_unit, dtype=jnp.float64).reshape(())

    # ------------------------------------------------------------------
    # Steady-state helpers
    # ------------------------------------------------------------------

    def steady_state(self) -> jax.Array:
        """Pre-industrial steady-state ``14C`` inventory vector ``y_ss`` (``kg``).

        Returns
        -------
        jax.Array
            Shape ``(n_boxes,)`` ``float64`` steady-state ``14C`` masses in ``kg``.
        """
        return self.cbm.steady_state_c14(q_std=self.q_std, alpha_unit=self.alpha_unit)

    def reference_specific_activity(self) -> jax.Array:
        """Single global tropospheric reference specific activity ``R_ref`` (``kg GtC^-1``).

        Computed as the mass-weighted mean of steady-state ``14C / carbon`` over
        all tropospheric boxes, so a **single** ``R_ref`` is used for both NH and
        SH, preserving the ``~4.38‰`` inter-hemispheric gradient in
        ``Delta14C`` rather than normalising each hemisphere to ``0‰``.

        Returns
        -------
        jax.Array
            Scalar ``float64`` reference specific activity in ``kg GtC^-1``.
        """
        return self.cbm.reference_specific_activity(q_std=self.q_std, alpha_unit=self.alpha_unit)

    # ------------------------------------------------------------------
    # Prediction on Dataset / TreeRingSeries
    # ------------------------------------------------------------------

    def _box_index_for_series(self, series: TreeRingSeries) -> int | None:
        """Return the box index to project onto for a given series.

        Uses ``series.box_name`` to look up in the CBM topology; falls back
        to the first tropospheric box for the matching hemisphere.
        """
        topo = self.cbm.topology
        # Try exact box name first
        if series.box_name in topo.box_names:
            return topo.box_names.index(series.box_name)
        # Fall back to hemisphere tropospheric box
        hemi = series.hemisphere
        for i, (name, h) in enumerate(zip(topo.box_names, topo.hemispheres)):
            if h == hemi and "trop" in name.lower():
                return i
        # Return first tropospheric box if hemisphere match fails
        trop_idx = topo.trop_box_indices
        if len(trop_idx) > 0:
            return int(trop_idx[0])
        return None

    def predict_d14c(
        self,
        dataset_or_series: Dataset | TreeRingSeries,
        *,
        offsets: jax.Array | None = None,
        concat: bool = True,
        delta_global: ArrayLike = 0.0,
    ) -> jax.Array | tuple[jax.Array, ...]:
        """Predict seasonal-window-binned ``Delta14C`` (‰).

        Routes each :class:`~ticktack.data.TreeRingSeries` to its hemisphere's
        tropospheric box. When ``offsets`` is provided, it is added per series.

        Parameters
        ----------
        dataset_or_series : Dataset or TreeRingSeries
            Observational data with precomputed seasonal windows.
        offsets : jax.Array, optional
            Per-series scalar baseline offsets of shape ``(S,)`` (‰) added to
            the predicted ``Delta14C``.  Pass ``None`` (default) for zero offsets.
        concat : bool
            When ``True`` (default) and input is a :class:`~ticktack.data.Dataset`,
            return a flat ``(N_tot,)`` array.  When ``False``, return a tuple of
            per-series ``(N_r_s,)`` arrays.
        delta_global : float or jax.Array
            Global Holocene ambient offset ``Delta_global`` (‰).  Applied
            uniformly across all series before ``offsets``.

        Returns
        -------
        jax.Array or tuple of jax.Array
            Predicted ``Delta14C`` (‰).  Shape ``(N_r,)`` for a single
            :class:`~ticktack.data.TreeRingSeries`; ``(N_tot,)`` or tuple for
            a :class:`~ticktack.data.Dataset`.
        """
        if isinstance(dataset_or_series, TreeRingSeries):
            series = dataset_or_series
            box_idx = self._box_index_for_series(series)
            delta_lab = offsets[0] if offsets is not None else 0.0
            return self.solver.predict_d14c(
                self.cbm,
                self.production,
                series.sub_t_starts,
                series.sub_t_ends,
                sub_weights=series.seasonal_weights.weights,
                box_indices=box_idx,
                q_std=self.q_std,
                alpha_unit=self.alpha_unit,
                delta_global=delta_global,
                delta_lab=delta_lab,
            )

        # Dataset: predict per series
        dataset = dataset_or_series
        per_series: list[jax.Array] = []
        for s_idx, series in enumerate(dataset.series):
            box_idx = self._box_index_for_series(series)
            delta_lab = offsets[s_idx] if offsets is not None else 0.0
            pred_s = self.solver.predict_d14c(
                self.cbm,
                self.production,
                series.sub_t_starts,
                series.sub_t_ends,
                sub_weights=series.seasonal_weights.weights,
                box_indices=box_idx,
                q_std=self.q_std,
                alpha_unit=self.alpha_unit,
                delta_global=delta_global,
                delta_lab=delta_lab,
            )
            per_series.append(pred_s)

        if concat:
            return jnp.concatenate(per_series, axis=0)
        return tuple(per_series)

    def predict_fine(
        self,
        t_eval: ArrayLike,
        *,
        box: int | None = None,
        hemisphere: Literal["north", "south"] = "north",
    ) -> jax.Array:
        """Predict ``Delta14C`` (‰) on a continuous time grid ``t_eval``.

        Parameters
        ----------
        t_eval : array-like
            Continuous time grid in astronomical CE years.
        box : int, optional
            Box index to project onto.  When ``None``, uses the first
            tropospheric box for the given ``hemisphere``.
        hemisphere : {"north", "south"}
            Hemisphere for box selection when ``box`` is ``None``.

        Returns
        -------
        jax.Array
            Shape ``(len(t_eval),)`` predicted ``Delta14C`` (‰).
        """
        t_arr = jnp.asarray(t_eval, dtype=jnp.float64)
        box_idx = self._resolve_box(box, hemisphere)
        y_traj = self.solver.solve_trajectory(
            self.cbm,
            self.production,
            t_arr,
            q_std=self.q_std,
            alpha_unit=self.alpha_unit,
        )  # (N_t, n_boxes)
        y_ss = self.steady_state()  # (n_boxes,)
        trop_idx = jnp.asarray(self.cbm.topology.trop_box_indices, dtype=jnp.int32)
        y_trop_ss = jnp.sum(y_ss[trop_idx])
        m_rel = self.cbm.relative_reservoir_masses()
        # Delta14C for selected box
        y_box = y_traj[:, box_idx]  # (N_t,)
        m_box = m_rel[box_idx]
        return 1000.0 * (y_box / (y_trop_ss * m_box) - 1.0)

    def _resolve_box(
        self,
        box: int | None,
        hemisphere: Literal["north", "south"],
    ) -> int:
        """Resolve a box index by hemisphere when ``box`` is ``None``."""
        if box is not None:
            return int(box)
        topo = self.cbm.topology
        hemi_map = {"north": "north", "south": "south"}
        h = hemi_map.get(hemisphere, "north")
        for i, (name, bh) in enumerate(zip(topo.box_names, topo.hemispheres)):
            if bh == h and "trop" in name.lower():
                return i
        # Fallback: first tropospheric box
        trop_idx = topo.trop_box_indices
        return int(trop_idx[0]) if len(trop_idx) > 0 else 0

    # ------------------------------------------------------------------
    # System identification
    # ------------------------------------------------------------------

    def impulse_response(
        self,
        t_lag: ArrayLike,
        *,
        box: int | None = None,
        hemisphere: Literal["north", "south"] = "north",
    ) -> jax.Array:
        """Tropospheric impulse response ``g_c(tau)`` = ``e_c^T exp(A tau) b`` (``yr^-1``).

        Parameters
        ----------
        t_lag : array-like
            Time lags ``tau >= 0`` (years).
        box : int, optional
            Box index.  Defaults to first tropospheric box for ``hemisphere``.
        hemisphere : {"north", "south"}
            Hemisphere for box selection.

        Returns
        -------
        jax.Array
            Shape ``(len(t_lag),)`` impulse response values.
        """
        A = self.cbm.matrix_A(q_std=self.q_std, alpha_unit=self.alpha_unit)
        b = self.cbm.production_vector()
        box_idx = self._resolve_box(box, hemisphere)
        tau_arr = jnp.asarray(t_lag, dtype=jnp.float64)

        def _g(tau: jax.Array) -> jax.Array:
            safe_tau = jnp.maximum(tau, 0.0)
            expm_At = jax.scipy.linalg.expm(A * safe_tau)
            return self.alpha_unit * expm_At[box_idx] @ b

        return jax.vmap(_g)(tau_arr.reshape(-1)).reshape(tau_arr.shape)

    def transfer_function(
        self,
        omegas: ArrayLike,
        *,
        box: int | None = None,
        hemisphere: Literal["north", "south"] = "north",
    ) -> jax.Array:
        """Complex tropospheric transfer function ``H_c(omega)`` = ``e_c^T (i omega I - A)^{-1} b``.

        Parameters
        ----------
        omegas : array-like
            Angular frequencies ``omega`` (``yr^-1 * 2 pi``).
        box : int, optional
            Box index.  Defaults to first tropospheric box for ``hemisphere``.
        hemisphere : {"north", "south"}
            Hemisphere for box selection.

        Returns
        -------
        jax.Array
            Complex ``float64`` array of shape ``(len(omegas),)`` with the
            Fourier transform of the causal impulse response.
        """
        A = self.cbm.matrix_A(q_std=self.q_std, alpha_unit=self.alpha_unit)
        b = self.cbm.production_vector()
        box_idx = self._resolve_box(box, hemisphere)
        w_arr = jnp.asarray(omegas, dtype=jnp.float64).reshape(-1)
        n = A.shape[0]
        eye_c = jnp.eye(n, dtype=jnp.complex128)
        A_c = A.astype(jnp.complex128)
        b_c = (self.alpha_unit * b).astype(jnp.complex128)

        def _h(w: jax.Array) -> jax.Array:
            z = jnp.linalg.solve(1j * w * eye_c - A_c, b_c)
            return z[box_idx]

        return jax.vmap(_h)(w_arr)

    # ------------------------------------------------------------------
    # Fluence reporting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def fluence_atoms_cm2(fluence: ArrayLike) -> jax.Array:
        """Convert ``MiyakeSpike.fluence`` (``atoms cm^-2 s^-1 * yr``) to ``atoms cm^-2``.

        ``MiyakeSpike.fluence`` stores the time-integrated excess production:
        ``integral delta_q(t) dt`` in units of ``atoms cm^-2 s^-1 * yr``.
        Multiplying by ``SECONDS_PER_JULIAN_YEAR`` converts to the total
        column fluence in ``atoms cm^-2``.

        Parameters
        ----------
        fluence : float or jax.Array
            :class:`~ticktack.production.MiyakeSpike` ``.fluence`` attribute
            value (``atoms cm^-2 s^-1 * yr``).

        Returns
        -------
        jax.Array
            Total fluence in ``atoms cm^-2``.
        """
        return jnp.asarray(fluence, dtype=jnp.float64) * SECONDS_PER_JULIAN_YEAR

    @staticmethod
    def fluence_q0_years(
        fluence: ArrayLike,
        *,
        q_std: ArrayLike = DEFAULT_Q_STD,
        baseline_d14c_permil: ArrayLike | None = None,
    ) -> jax.Array:
        """Express spike fluence as equivalent pre-industrial ``q_0`` years.

        Returns the spike fluence in units of equivalent pre-industrial
        production-rate-years:

        * **Pre-industrial** (``baseline_d14c_permil=None``):
          ``q0_pi_yr = fluence / q_std``
          (independent of Holocene ambient ``Delta14C``).
        * **Epoch-local** (``baseline_d14c_permil`` supplied):
          ``q0_local_yr = fluence / (q_std * (1 + Delta_0 / 1000))``
          (corrects for the ambient ``14C`` level at the event epoch, so
          the yield fraction is correctly normalized to the local ``R_ref``
          rather than the 1950 CE reference).

        Parameters
        ----------
        fluence : float or jax.Array
            :class:`~ticktack.production.MiyakeSpike` ``.fluence`` attribute
            (``atoms cm^-2 s^-1 * yr``).
        q_std : float
            Pre-industrial production rate (``atoms cm^-2 s^-1``; default ``1.76``).
        baseline_d14c_permil : float or jax.Array, optional
            Epoch-local ambient ``Delta14C_0`` in ‰ at the event time.
            When ``None``, the pre-industrial convention ``q0_pi = fluence / q_std``
            is returned.  When supplied, the epoch-local convention
            ``q0_local = fluence / (q_std * (1 + Delta_0 / 1000))`` is used.

        Returns
        -------
        jax.Array
            Scalar fluence in equivalent ``q_0`` years (pre-industrial or epoch-local).
        """
        fl = jnp.asarray(fluence, dtype=jnp.float64)
        qs = jnp.asarray(q_std, dtype=jnp.float64)
        if baseline_d14c_permil is None:
            return fl / qs
        d0 = jnp.asarray(baseline_d14c_permil, dtype=jnp.float64)
        return fl / (qs * (1.0 + d0 / 1000.0))

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RadiocarbonModel(\n"
            f"  cbm={self.cbm!r},\n"
            f"  production={self.production!r},\n"
            f"  solver={self.solver!r},\n"
            f"  q_std={float(self.q_std):.4f},\n"
            f")"
        )
