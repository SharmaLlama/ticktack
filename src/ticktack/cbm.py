"""Immutable ``zodiax`` / ``equinox`` Carbon Box Models (CBMs) for ``ticktack 2.0``.

Every CBM component is an immutable PyTree subclassing :class:`zodiax.Base` with
``float64`` JAX array leaves for continuous parameters and hashable static tuples
for graph topology and structural metadata:

* :class:`Topology`: directed carbon-cycle graph metadata with Tarjan strongly
  connected components (SCC) verification and hemisphere-aware box lookup.
* :class:`CarbonBoxModel`: abstract base interface defining ``rate_matrix_K()``,
  ``sink_rates()``, ``matrix_A()``, ``production_vector()``, ``reservoir_masses()``,
  ``flux_matrix()``, ``steady_state_c14()``, single global tropospheric
  ``reference_specific_activity()`` (preserving inter-hemispheric gradients),
  and closed-form Güttler ``production_for_target_c14()``.
* :class:`FixedCBM`: fixed reservoir inventories, inter-box fluxes, and optional
  sink rates.
* :class:`BalancedCBM`: unconstrained Perron–Frobenius equilibrium flow-balance
  chart ``theta = M_tie @ theta_free`` plus ``theta_tot = ln M_tot``, with optional
  edge-symmetry tying and exact inverse ``from_masses_and_fluxes``.
* :class:`ReversibleCBM`: detailed-balance symmetric-flux chart (unconstrained
  ``ln M`` and undirected ``ln F_{i,j}``).
* :class:`OpenCBM`: Perron–Frobenius inter-box exchange augmented with explicit
  unconstrained ``log_sink_rates`` on open reservoirs (e.g. truncated shallow
  models losing ``14C`` to the deep ocean).
* :class:`EffectiveCBM`: minimal gauge-free ``K``-box mammillary model
  (``Stratosphere <-> Troposphere <-> {Sink_1, ..., Sink_{K-2}}``, ``K >= 2``)
  with fixed ``eta_strat = 0.70``, calibrated deep-ocean loss rate on the slowest
  box, and strict ``softplus`` sink timescale ordering ``ln tau_1 < ... < ln tau_{K-2}``,
  giving exact ``2K - 2`` identifiable parameters.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
import zodiax as zdx

from .constants import (
    ALPHA_ATOMS_CM2_S_TO_KG_YR,
    DECAY_CAMBRIDGE,
    DEFAULT_Q_STD,
    GUTTLER15_R_REF_KG_PER_GTC,
    GUTTLER15_TROP_MASS_GTC,
)

__all__ = [
    "tarjan_scc",
    "Topology",
    "CarbonBoxModel",
    "FixedCBM",
    "BalancedCBM",
    "ReversibleCBM",
    "OpenCBM",
    "EffectiveCBM",
]


def _concrete_array(x: Any) -> np.ndarray | None:
    """Return ``x`` as a NumPy array when concrete, or ``None`` when JAX-traced."""
    try:
        return np.asarray(x, dtype=np.float64)
    except (
        jax.errors.TracerArrayConversionError,
        jax.errors.ConcretizationTypeError,
    ):
        return None


def _normalize_hemisphere(h: str | None) -> str:
    if h is None:
        return "none"
    val = str(h).strip().lower()
    if val in ("none", "", "global"):
        return "none"
    if val in ("north", "n", "nh"):
        return "north"
    if val in ("south", "s", "sh"):
        return "south"
    raise ValueError(
        f"Invalid hemisphere {h!r}; expected 'none', 'north', or 'south'."
    )


def tarjan_scc(
    n_nodes: int,
    edges_src: Sequence[int],
    edges_dst: Sequence[int],
) -> list[tuple[int, ...]]:
    """Compute strongly connected components of a directed graph via Tarjan's algorithm.

    Parameters
    ----------
    n_nodes : int
        Number of vertices ``0, ..., n_nodes - 1``.
    edges_src, edges_dst : Sequence[int]
        Directed edge endpoints ``src -> dst``.

    Returns
    -------
    list[tuple[int, ...]]
        Strongly connected components, each sorted in ascending vertex order.
    """
    if n_nodes < 1:
        raise ValueError(f"n_nodes must be >= 1, got {n_nodes}.")
    if len(edges_src) != len(edges_dst):
        raise ValueError(
            f"edges_src ({len(edges_src)}) and edges_dst ({len(edges_dst)}) must have equal length."
        )

    adj: list[list[int]] = [[] for _ in range(n_nodes)]
    for s, d in zip(edges_src, edges_dst):
        adj[int(s)].append(int(d))

    index_counter = 0
    indices = [-1] * n_nodes
    lowlink = [0] * n_nodes
    on_stack = [False] * n_nodes
    stack: list[int] = []
    sccs: list[tuple[int, ...]] = []

    def strongconnect(v: int) -> None:
        nonlocal index_counter
        indices[v] = index_counter
        lowlink[v] = index_counter
        index_counter += 1
        stack.append(v)
        on_stack[v] = True

        for w in adj[v]:
            if indices[w] == -1:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack[w]:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            comp: list[int] = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                comp.append(w)
                if w == v:
                    break
            sccs.append(tuple(sorted(comp)))

    for node in range(n_nodes):
        if indices[node] == -1:
            strongconnect(node)

    return sccs


def _perron_frobenius_simplex(
    K: jax.Array,
    edges_src: tuple[int, ...],
    k_edges: jax.Array,
) -> jax.Array:
    """Compute the unique positive null vector ``pi`` of irreducible Metzler ``K``.

    Solves ``K @ pi = 0`` with ``sum(pi) = 1`` by right-preconditioning ``K`` with
    the inverse exit rates ``d_j = -K_{jj} > 0`` so ``K_hat = K @ diag(d)^{-1}``
    has unit-scale columns across arbitrarily stiff reservoir residence times
    (e.g. ``1.2 GtC`` surface biota alongside ``378,000 GtC`` sediments), then
    solving ``K_tilde @ q = e_n`` where the last row of ``K_hat`` is replaced by ones.
    """
    n = K.shape[0]
    if n == 1:
        return jnp.ones((1,), dtype=jnp.float64)

    exit_rates = jnp.zeros((n,), dtype=jnp.float64).at[np.asarray(edges_src)].add(k_edges)
    safe_exit = jnp.where(exit_rates > 0.0, exit_rates, 1.0)
    K_hat = K / safe_exit[None, :]
    inv_n = 1.0 / float(n)
    K_tilde = K_hat + inv_n
    rhs = jnp.full((n,), inv_n, dtype=jnp.float64)
    q = jnp.linalg.solve(K_tilde, rhs)
    # One iterative refinement step on the equilibrated nullspace system
    res = K_tilde @ q - rhs
    q = q - jnp.linalg.solve(K_tilde, res)
    pi_unnorm = q / safe_exit
    return pi_unnorm / jnp.sum(pi_unnorm)


def _assemble_rate_matrix(
    n_boxes: int,
    edges_src: tuple[int, ...],
    edges_dst: tuple[int, ...],
    k_edges: jax.Array,
) -> jax.Array:
    """Assemble ``K`` with ``K[dst, src] = k_{src->dst}`` and zero column sums."""
    K = jnp.zeros((n_boxes, n_boxes), dtype=jnp.float64)
    if len(edges_src) == 0:
        return K
    src = np.asarray(edges_src, dtype=np.int64)
    dst = np.asarray(edges_dst, dtype=np.int64)
    K = K.at[dst, src].add(k_edges)
    K = K.at[src, src].add(-k_edges)
    return K


def _build_tie_matrix(
    n_edges: int,
    *,
    tie_matrix: ArrayLike | None = None,
    symmetric_edge_groups: Sequence[Sequence[int]] | None = None,
) -> tuple[tuple[float, ...], ...] | None:
    """Build a static hashable ``(n_edges, p_free)`` tie matrix if tying is specified."""
    if tie_matrix is not None and symmetric_edge_groups is not None:
        raise ValueError("Specify at most one of tie_matrix or symmetric_edge_groups.")

    if tie_matrix is not None:
        arr = np.asarray(tie_matrix, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != n_edges:
            raise ValueError(
                f"tie_matrix must have shape ({n_edges}, p_free), got {arr.shape}."
            )
        return tuple(tuple(float(v) for v in row) for row in arr)

    if symmetric_edge_groups is None:
        return None

    grouped: dict[int, int] = {}
    cols: list[list[int]] = []
    for grp in symmetric_edge_groups:
        g_list = [int(idx) for idx in grp]
        if not g_list:
            continue
        for idx in g_list:
            if idx < 0 or idx >= n_edges:
                raise ValueError(f"Edge index {idx} out of range [0, {n_edges}).")
            if idx in grouped:
                raise ValueError(f"Edge index {idx} appears in multiple symmetric_edge_groups.")
            grouped[idx] = len(cols)
        cols.append(g_list)

    # Add singleton columns for any remaining untied edges in canonical order
    final_cols: list[list[int]] = []
    seen_groups: set[int] = set()
    for e in range(n_edges):
        if e in grouped:
            gid = grouped[e]
            if gid not in seen_groups:
                seen_groups.add(gid)
                final_cols.append(cols[gid])
        else:
            final_cols.append([e])

    mat = np.zeros((n_edges, len(final_cols)), dtype=np.float64)
    for col_idx, edge_list in enumerate(final_cols):
        for e in edge_list:
            mat[e, col_idx] = 1.0
    return tuple(tuple(float(v) for v in row) for row in mat)


class Topology(eqx.Module):
    """Directed carbon-cycle reservoir topology with static structural metadata.

    Parameters
    ----------
    box_names : Sequence[str]
        Ordered reservoir names of length ``n``.
    edges_src, edges_dst : Sequence[int]
        Directed positive-flux edge indices ``(src -> dst)`` with ``0 <= idx < n``.
        Zero-flux placeholder edges must be omitted prior to construction.
    hemispheres : Sequence[str], optional
        Per-box hemisphere labels (``"none"``, ``"north"``, or ``"south"``).
        Defaults to ``("none", ...)``.
    prod_box_indices : Sequence[int], optional
        Indices of boxes receiving cosmic-ray ``14C`` production. Defaults to
        boxes matching ``"strat"`` or ``"trop"``.
    trop_box_indices : Sequence[int], optional
        Indices of tropospheric boxes ``T`` defining the global reference
        specific activity ``R_ref``. Defaults to boxes matching ``"trop"``.
    check_scc : bool, optional
        If ``True`` (default), verify strong connectivity via Tarjan's SCC
        algorithm and raise :class:`ValueError` if reducible.
    """

    box_names: tuple[str, ...] = eqx.field(static=True)
    hemispheres: tuple[str, ...] = eqx.field(static=True)
    edges_src: tuple[int, ...] = eqx.field(static=True)
    edges_dst: tuple[int, ...] = eqx.field(static=True)
    prod_box_indices: tuple[int, ...] = eqx.field(static=True)
    trop_box_indices: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        box_names: Sequence[str],
        edges_src: Sequence[int],
        edges_dst: Sequence[int],
        *,
        hemispheres: Sequence[str | None] | None = None,
        prod_box_indices: Sequence[int] | None = None,
        trop_box_indices: Sequence[int] | None = None,
        check_scc: bool = True,
    ):
        names = tuple(str(b) for b in box_names)
        n = len(names)
        if n < 1:
            raise ValueError("Topology requires at least one box.")

        if hemispheres is None:
            hemis = ("none",) * n
        else:
            if len(hemispheres) != n:
                raise ValueError(
                    f"hemispheres length ({len(hemispheres)}) must match box_names ({n})."
                )
            hemis = tuple(_normalize_hemisphere(h) for h in hemispheres)

        if any(h == "none" for h in hemis) and not all(h == "none" for h in hemis):
            raise ValueError(
                "Hemisphere labels cannot mix 'none' with 'north'/'south'; "
                "either all boxes are 'none' or all are 'north'/'south'."
            )

        src_t = tuple(int(s) for s in edges_src)
        dst_t = tuple(int(d) for d in edges_dst)
        if len(src_t) != len(dst_t):
            raise ValueError(
                f"edges_src ({len(src_t)}) and edges_dst ({len(dst_t)}) must have the same length."
            )

        seen_edges: set[tuple[int, int]] = set()
        for s, d in zip(src_t, dst_t):
            if not (0 <= s < n and 0 <= d < n):
                raise ValueError(f"Edge ({s} -> {d}) out of bounds for {n} boxes.")
            if s == d:
                raise ValueError(f"Self-loop edge ({s} -> {d}) is not allowed.")
            if (s, d) in seen_edges:
                raise ValueError(f"Duplicate directed edge ({s} -> {d}).")
            seen_edges.add((s, d))

        if trop_box_indices is None:
            trop_t = tuple(
                i for i, name in enumerate(names) if "trop" in name.lower()
            )
        else:
            trop_t = tuple(int(i) for i in trop_box_indices)
        if not trop_t or any(i < 0 or i >= n for i in trop_t):
            raise ValueError(
                f"trop_box_indices must be non-empty and within [0, {n}), got {trop_t}."
            )

        if prod_box_indices is None:
            prod_t = tuple(
                i
                for i, name in enumerate(names)
                if ("strat" in name.lower() or "trop" in name.lower())
            )
        else:
            prod_t = tuple(int(i) for i in prod_box_indices)
        if not prod_t or any(i < 0 or i >= n for i in prod_t):
            raise ValueError(
                f"prod_box_indices must be non-empty and within [0, {n}), got {prod_t}."
            )

        self.box_names = names
        self.hemispheres = hemis
        self.edges_src = src_t
        self.edges_dst = dst_t
        self.prod_box_indices = prod_t
        self.trop_box_indices = trop_t

        if check_scc:
            self.check_strongly_connected()

    @property
    def n_boxes(self) -> int:
        return len(self.box_names)

    @property
    def n_edges(self) -> int:
        return len(self.edges_src)

    @property
    def is_hemispheric(self) -> bool:
        return self.hemispheres[0] != "none"

    def strongly_connected_components(self) -> list[tuple[int, ...]]:
        return tarjan_scc(self.n_boxes, self.edges_src, self.edges_dst)

    @property
    def is_strongly_connected(self) -> bool:
        return len(self.strongly_connected_components()) == 1

    def check_strongly_connected(self) -> None:
        sccs = self.strongly_connected_components()
        if len(sccs) != 1:
            comp_desc = [
                [f"{i}:{self.box_names[i]}" for i in comp] for comp in sccs
            ]
            raise ValueError(
                f"Carbon-box topology is not strongly connected ({len(sccs)} SCCs: {comp_desc})."
            )

    def box_index(self, name: str = "Troposphere", hemisphere: str | None = None) -> int:
        """Resolve a reservoir index by box name and optional hemisphere.

        If ``hemisphere`` is ``"north"`` or ``"south"`` on a non-hemispheric
        1-globe topology (all boxes ``"none"``), the global box matching ``name``
        is returned cleanly so hemispheric tree-ring series can be modelled with
        global CBMs such as ``Guttler15`` or ``EffectiveCBM``.
        """
        target_name = name.strip().lower()
        target_hemi = _normalize_hemisphere(hemisphere) if hemisphere is not None else None

        matches: list[int] = []
        for idx, (bname, bhemi) in enumerate(zip(self.box_names, self.hemispheres)):
            if bname.strip().lower() != target_name:
                continue
            if target_hemi is None or not self.is_hemispheric or target_hemi == "none":
                matches.append(idx)
            elif bhemi == target_hemi:
                matches.append(idx)

        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise KeyError(
                f"No box matching name={name!r}, hemisphere={hemisphere!r} in {list(zip(self.box_names, self.hemispheres))}."
            )
        raise ValueError(
            f"Ambiguous box lookup name={name!r}, hemisphere={hemisphere!r}; specify hemisphere ('north' or 'south')."
        )

    @classmethod
    def from_spec(cls, spec: Any, *, check_scc: bool = True) -> Topology:
        """Construct a :class:`Topology` from a ``PresetSpec``-compatible object."""
        pos_edges = [(int(s), int(d)) for s, d, f in spec.edges if float(f) > 0.0]
        edges_src = [s for s, _ in pos_edges]
        edges_dst = [d for _, d in pos_edges]
        prod_coeffs = np.asarray(spec.prod_coefficients, dtype=np.float64)
        prod_idx = tuple(int(i) for i in np.flatnonzero(prod_coeffs > 0.0))
        return cls(
            box_names=spec.box_names,
            edges_src=edges_src,
            edges_dst=edges_dst,
            hemispheres=spec.hemispheres,
            prod_box_indices=prod_idx if prod_idx else None,
            check_scc=check_scc,
        )


class CarbonBoxModel(zdx.Base):
    """Abstract immutable base class for all ``ticktack 2.0`` Carbon Box Models."""

    topology: Topology = eqx.field(static=True)
    decay_rate: float = eqx.field(static=True)

    @property
    def n_boxes(self) -> int:
        return self.topology.n_boxes

    def box_index(self, name: str = "Troposphere", hemisphere: str | None = None) -> int:
        return self.topology.box_index(name=name, hemisphere=hemisphere)

    def rate_matrix_K(self) -> jax.Array:
        """Column-conserving inter-box transfer matrix ``K`` (``yr^-1``)."""
        raise NotImplementedError

    def sink_rates(self) -> jax.Array:
        """Non-negative open reservoir loss rates ``k_sink`` (``yr^-1``)."""
        return jnp.zeros((self.n_boxes,), dtype=jnp.float64)

    def matrix_A(self) -> jax.Array:
        """Hurwitz LTI state matrix ``A = K - diag(k_sink) - lambda_14 * I``."""
        n = self.n_boxes
        return (
            self.rate_matrix_K()
            - jnp.diag(self.sink_rates())
            - float(self.decay_rate) * jnp.eye(n, dtype=jnp.float64)
        )

    def production_vector(self) -> jax.Array:
        """Normalized cosmic-ray ``14C`` production allocation ``b`` summing to 1."""
        raise NotImplementedError

    def reservoir_masses(self) -> jax.Array:
        """Steady-state ``12C`` reservoir inventories ``M_0`` in ``GtC``."""
        raise NotImplementedError

    def relative_reservoir_masses(self) -> jax.Array:
        """Reservoir ``12C`` inventories normalized by global tropospheric mass ``sum_{j in T} M_{0,j}``."""
        trop_idx = jnp.asarray(self.topology.trop_box_indices, dtype=jnp.int32)
        m_0 = self.reservoir_masses()
        return m_0 / jnp.sum(m_0[trop_idx])

    def flux_matrix(self) -> jax.Array:
        """Steady-state inter-box ``12C`` flux matrix ``F[src, dst]`` in ``GtC yr^-1``."""
        K = self.rate_matrix_K()
        M = self.reservoir_masses()
        off_diag = jnp.transpose(K) * M[:, None]
        return jnp.where(
            jnp.eye(self.n_boxes, dtype=bool),
            0.0,
            off_diag,
        )

    def reservoir_masses_at(self, t: ArrayLike) -> jax.Array:
        """Reservoir ``12C`` masses ``M(t)`` at epoch ``t`` (autonomous default)."""
        del t
        return self.reservoir_masses()

    def matrix_A_at(self, t: ArrayLike) -> jax.Array:
        """LTI/LTV state matrix ``A(t)`` at epoch ``t`` (autonomous default)."""
        del t
        return self.matrix_A()

    def kirchhoff_residual(self) -> jax.Array:
        """Net inter-box ``12C`` outflow minus inflow ``sum_j F_ij - sum_j F_ji``."""
        F = self.flux_matrix()
        return jnp.sum(F, axis=1) - jnp.sum(F, axis=0)

    def steady_state_c14(
        self,
        q_std: ArrayLike = DEFAULT_Q_STD,
        alpha_unit: ArrayLike = ALPHA_ATOMS_CM2_S_TO_KG_YR,
    ) -> jax.Array:
        """Equilibrium ``14C`` inventory ``y_ss = alpha_unit * q_std * (-A)^{-1} b`` (``kg``)."""
        scale = jnp.asarray(alpha_unit, dtype=jnp.float64) * jnp.asarray(
            q_std, dtype=jnp.float64
        )
        rhs = scale * self.production_vector()
        return jnp.linalg.solve(-self.matrix_A(), rhs)

    def reference_specific_activity(
        self,
        q_std: ArrayLike = DEFAULT_Q_STD,
        alpha_unit: ArrayLike = ALPHA_ATOMS_CM2_S_TO_KG_YR,
    ) -> jax.Array:
        """Single global tropospheric reference specific activity ``R_ref``.

        Defined as ``sum_{j in T} y_{ss,j} / sum_{j in T} M_{0,j}`` across all
        tropospheric boxes ``T = topology.trop_box_indices``. Using a single
        global ``R_ref`` preserves physical steady-state inter-hemispheric
        gradients (such as ``Brehm21``'s ``4.38 permil`` NH-SH offset driven by
        Southern Ocean deep-water upwelling).
        """
        trop_idx = jnp.asarray(self.topology.trop_box_indices, dtype=jnp.int32)
        y_ss = self.steady_state_c14(q_std=q_std, alpha_unit=alpha_unit)
        m_0 = self.reservoir_masses()
        return jnp.sum(y_ss[trop_idx]) / jnp.sum(m_0[trop_idx])

    def steady_state_d14c(
        self,
        q_std: ArrayLike = DEFAULT_Q_STD,
        alpha_unit: ArrayLike = ALPHA_ATOMS_CM2_S_TO_KG_YR,
    ) -> jax.Array:
        """Steady-state ``Delta14C`` (in ``permil``) of every reservoir relative to ``R_ref``."""
        trop_idx = jnp.asarray(self.topology.trop_box_indices, dtype=jnp.int32)
        y_ss = self.steady_state_c14(q_std=q_std, alpha_unit=alpha_unit)
        m_rel = self.relative_reservoir_masses()
        y_trop = jnp.sum(y_ss[trop_idx])
        return 1000.0 * ((y_ss / y_trop) / m_rel - 1.0)

    def production_for_target_c14(
        self,
        target_c14: ArrayLike,
        *,
        box: int | str | None = None,
        hemisphere: str | None = None,
        alpha_unit: ArrayLike = ALPHA_ATOMS_CM2_S_TO_KG_YR,
    ) -> jax.Array:
        """Closed-form Güttler production rate ``Q_std`` yielding ``target_c14`` (``kg``).

        Replaces ``ticktack 1.x``'s numerical BFGS loop with exact scalar division
        ``Q_std = target_c14 / (alpha_unit * unit_y)``.
        """
        unit_vec = jnp.linalg.solve(-self.matrix_A(), self.production_vector())
        if isinstance(box, int):
            unit_y = unit_vec[box]
        elif isinstance(box, str):
            idx = self.box_index(box, hemisphere=hemisphere)
            unit_y = unit_vec[idx]
        elif hemisphere is not None:
            idx = self.box_index("Troposphere", hemisphere=hemisphere)
            unit_y = unit_vec[idx]
        else:
            trop_idx = jnp.asarray(self.topology.trop_box_indices, dtype=jnp.int32)
            unit_y = jnp.sum(unit_vec[trop_idx])

        return jnp.asarray(target_c14, dtype=jnp.float64) / (
            jnp.asarray(alpha_unit, dtype=jnp.float64) * unit_y
        )


def _resolve_prod_weights(
    topology: Topology,
    prod_weights: Sequence[float] | ArrayLike | None,
    prod_fractions: Sequence[float] | ArrayLike | None = None,
) -> tuple[float, ...]:
    arg = prod_fractions if prod_fractions is not None else prod_weights
    if arg is None:
        m_p = len(topology.prod_box_indices)
        return (1.0 / m_p,) * m_p
    pw_arr = np.asarray(arg, dtype=np.float64)
    if pw_arr.shape == (topology.n_boxes,):
        pw_arr = pw_arr[np.asarray(topology.prod_box_indices, dtype=np.int64)]
    if pw_arr.shape != (len(topology.prod_box_indices),):
        raise ValueError(
            f"prod_weights must have length {len(topology.prod_box_indices)} or {topology.n_boxes}."
        )
    total = float(np.sum(pw_arr))
    if total <= 0.0:
        raise ValueError("prod_weights must have positive sum.")
    return tuple(float(v / total) for v in pw_arr)


def _make_prod_vector(
    n_boxes: int,
    prod_box_indices: tuple[int, ...],
    prod_weights: Sequence[float] | jax.Array,
) -> jax.Array:
    w = jnp.asarray(prod_weights, dtype=jnp.float64)
    w = w / jnp.sum(w)
    b = jnp.zeros((n_boxes,), dtype=jnp.float64)
    return b.at[np.asarray(prod_box_indices, dtype=np.int64)].set(w)


class FixedCBM(CarbonBoxModel):
    """Carbon Box Model with directly specified inventories, fluxes, and sink rates."""

    masses: jax.Array
    edge_fluxes: jax.Array
    sink_rates_arr: jax.Array
    prod_weights: tuple[float, ...] = eqx.field(static=True)

    def __init__(
        self,
        topology: Topology,
        masses: ArrayLike,
        edge_fluxes: ArrayLike,
        *,
        prod_weights: Sequence[float] | ArrayLike | None = None,
        prod_fractions: Sequence[float] | ArrayLike | None = None,
        sink_rates: ArrayLike | None = None,
        decay_rate: float = DECAY_CAMBRIDGE,
    ):
        self.topology = topology
        self.decay_rate = float(decay_rate)
        self.masses = jnp.asarray(masses, dtype=jnp.float64)
        self.edge_fluxes = jnp.asarray(edge_fluxes, dtype=jnp.float64)
        if sink_rates is None:
            self.sink_rates_arr = jnp.zeros((topology.n_boxes,), dtype=jnp.float64)
        else:
            self.sink_rates_arr = jnp.asarray(sink_rates, dtype=jnp.float64)

        if self.masses.shape != (topology.n_boxes,):
            raise ValueError(
                f"masses shape {self.masses.shape} does not match ({topology.n_boxes},)."
            )
        if self.edge_fluxes.shape != (topology.n_edges,):
            raise ValueError(
                f"edge_fluxes shape {self.edge_fluxes.shape} does not match ({topology.n_edges},)."
            )
        if self.sink_rates_arr.shape != (topology.n_boxes,):
            raise ValueError(
                f"sink_rates shape {self.sink_rates_arr.shape} does not match ({topology.n_boxes},)."
            )

        self.prod_weights = _resolve_prod_weights(topology, prod_weights, prod_fractions)

    def reservoir_masses(self) -> jax.Array:
        return self.masses

    def rate_matrix_K(self) -> jax.Array:
        src = np.asarray(self.topology.edges_src, dtype=np.int64)
        k_edges = self.edge_fluxes / self.masses[src]
        return _assemble_rate_matrix(
            self.n_boxes, self.topology.edges_src, self.topology.edges_dst, k_edges
        )

    def sink_rates(self) -> jax.Array:
        return self.sink_rates_arr

    def production_vector(self) -> jax.Array:
        return _make_prod_vector(
            self.n_boxes, self.topology.prod_box_indices, self.prod_weights
        )

    @classmethod
    def from_flux_matrix(
        cls,
        box_names: Sequence[str],
        masses: ArrayLike,
        flux_matrix: ArrayLike,
        *,
        hemispheres: Sequence[str | None] | None = None,
        prod_coefficients: ArrayLike | None = None,
        sink_rates: ArrayLike | None = None,
        decay_rate: float = DECAY_CAMBRIDGE,
        check_scc: bool = True,
    ) -> FixedCBM:
        F = np.asarray(flux_matrix, dtype=np.float64)
        src_idx, dst_idx = np.nonzero(F > 0.0)
        edges_flux = F[src_idx, dst_idx]
        prod_idx = None
        if prod_coefficients is not None:
            pc = np.asarray(prod_coefficients, dtype=np.float64)
            prod_idx = tuple(int(i) for i in np.flatnonzero(pc > 0.0))
        topo = Topology(
            box_names=box_names,
            edges_src=src_idx.tolist(),
            edges_dst=dst_idx.tolist(),
            hemispheres=hemispheres,
            prod_box_indices=prod_idx,
            check_scc=check_scc,
        )
        return cls(
            topology=topo,
            masses=masses,
            edge_fluxes=edges_flux,
            prod_weights=prod_coefficients if prod_coefficients is not None else None,
            sink_rates=sink_rates,
            decay_rate=decay_rate,
        )

    @classmethod
    def from_spec(cls, spec: Any, *, decay_rate: float = DECAY_CAMBRIDGE) -> FixedCBM:
        topo = Topology.from_spec(spec)
        edge_fluxes = [float(f) for _, _, f in spec.edges if float(f) > 0.0]
        return cls(
            topology=topo,
            masses=spec.masses,
            edge_fluxes=edge_fluxes,
            prod_weights=spec.prod_coefficients,
            sink_rates=spec.sink_rates,
            decay_rate=decay_rate,
        )


class BalancedCBM(CarbonBoxModel):
    """Unconstrained equilibrium flow-balanced CBM via the Perron–Frobenius chart.

    Parameterizes all positive Kirchhoff-balanced ``(M_0, F)`` configurations on a
    strongly connected topology ``G = (V, E)`` without boundary inequalities or
    redundant cycle weights:

    * ``log_rates_free`` (``R^{p_rate}``): unconstrained log transfer rates mapped
      to edge log-rates via ``theta = M_tie @ log_rates_free``.
    * ``log_total_mass`` (scalar ``R``): global carbon log-mass ``ln M_tot`` (an
      exact zero-Fisher gauge coordinate for ``Delta14C``).
    """

    log_rates_free: jax.Array
    log_total_mass: jax.Array
    prod_weights: tuple[float, ...] = eqx.field(static=True)
    tie_matrix_rows: tuple[tuple[float, ...], ...] | None = eqx.field(static=True)

    def __init__(
        self,
        topology: Topology,
        log_rates_free: ArrayLike,
        log_total_mass: ArrayLike,
        *,
        prod_weights: Sequence[float] | ArrayLike | None = None,
        prod_fractions: Sequence[float] | ArrayLike | None = None,
        tie_matrix: ArrayLike | None = None,
        symmetric_edge_groups: Sequence[Sequence[int]] | None = None,
        decay_rate: float = DECAY_CAMBRIDGE,
    ):
        topology.check_strongly_connected()
        self.topology = topology
        self.decay_rate = float(decay_rate)

        self.tie_matrix_rows = _build_tie_matrix(
            topology.n_edges,
            tie_matrix=tie_matrix,
            symmetric_edge_groups=symmetric_edge_groups,
        )
        expected_p = (
            topology.n_edges
            if self.tie_matrix_rows is None
            else len(self.tie_matrix_rows[0])
        )

        self.log_rates_free = jnp.asarray(log_rates_free, dtype=jnp.float64)
        self.log_total_mass = jnp.asarray(log_total_mass, dtype=jnp.float64).reshape(())
        if self.log_rates_free.shape != (expected_p,):
            raise ValueError(
                f"log_rates_free must have shape ({expected_p},), got {self.log_rates_free.shape}."
            )

        self.prod_weights = _resolve_prod_weights(topology, prod_weights, prod_fractions)

    def edge_log_rates(self) -> jax.Array:
        if self.tie_matrix_rows is None:
            return self.log_rates_free
        M_tie = jnp.asarray(self.tie_matrix_rows, dtype=jnp.float64)
        return M_tie @ self.log_rates_free

    def edge_rates(self) -> jax.Array:
        return jnp.exp(self.edge_log_rates())

    def rate_matrix_K(self) -> jax.Array:
        return _assemble_rate_matrix(
            self.n_boxes,
            self.topology.edges_src,
            self.topology.edges_dst,
            self.edge_rates(),
        )

    def _perron_simplex(self) -> jax.Array:
        k_edges = self.edge_rates()
        K = _assemble_rate_matrix(
            self.n_boxes, self.topology.edges_src, self.topology.edges_dst, k_edges
        )
        return _perron_frobenius_simplex(K, self.topology.edges_src, k_edges)

    def reservoir_masses(self) -> jax.Array:
        return jnp.exp(self.log_total_mass) * self._perron_simplex()

    def relative_reservoir_masses(self) -> jax.Array:
        trop_idx = jnp.asarray(self.topology.trop_box_indices, dtype=jnp.int32)
        pi = self._perron_simplex()
        return pi / jnp.sum(pi[trop_idx])

    def production_vector(self) -> jax.Array:
        return _make_prod_vector(
            self.n_boxes, self.topology.prod_box_indices, self.prod_weights
        )

    @classmethod
    def from_masses_and_fluxes(
        cls,
        topology: Topology,
        masses: ArrayLike,
        edge_fluxes: ArrayLike,
        *,
        prod_weights: Sequence[float] | ArrayLike | None = None,
        prod_fractions: Sequence[float] | ArrayLike | None = None,
        tie_matrix: ArrayLike | None = None,
        symmetric_edge_groups: Sequence[Sequence[int]] | None = None,
        decay_rate: float = DECAY_CAMBRIDGE,
    ) -> BalancedCBM:
        """Exact inverse chart mapping balanced ``(masses, edge_fluxes)`` to ``BalancedCBM``."""
        m_arr = jnp.asarray(masses, dtype=jnp.float64)
        f_arr = jnp.asarray(edge_fluxes, dtype=jnp.float64)
        if f_arr.ndim == 2:
            src = np.asarray(topology.edges_src, dtype=np.int64)
            dst = np.asarray(topology.edges_dst, dtype=np.int64)
            f_arr = f_arr[src, dst]

        src = np.asarray(topology.edges_src, dtype=np.int64)
        log_k = jnp.log(f_arr / m_arr[src])
        log_m_tot = jnp.log(jnp.sum(m_arr))

        tie_rows = _build_tie_matrix(
            topology.n_edges,
            tie_matrix=tie_matrix,
            symmetric_edge_groups=symmetric_edge_groups,
        )
        if tie_rows is None:
            theta_free = log_k
        else:
            M_tie = jnp.asarray(tie_rows, dtype=jnp.float64)
            theta_free, _, _, _ = jnp.linalg.lstsq(M_tie, log_k, rcond=None)

        return cls(
            topology=topology,
            log_rates_free=theta_free,
            log_total_mass=log_m_tot,
            prod_weights=prod_weights,
            prod_fractions=prod_fractions,
            tie_matrix=tie_matrix,
            symmetric_edge_groups=symmetric_edge_groups,
            decay_rate=decay_rate,
        )

    @classmethod
    def from_spec(
        cls,
        spec: Any,
        *,
        tie_symmetric: bool = False,
        decay_rate: float = DECAY_CAMBRIDGE,
    ) -> BalancedCBM:
        topo = Topology.from_spec(spec)
        edge_fluxes = [float(f) for _, _, f in spec.edges if float(f) > 0.0]
        groups = spec.symmetric_edge_groups if tie_symmetric else None
        return cls.from_masses_and_fluxes(
            topology=topo,
            masses=spec.masses,
            edge_fluxes=edge_fluxes,
            prod_weights=spec.prod_coefficients,
            symmetric_edge_groups=groups,
            decay_rate=decay_rate,
        )


class ReversibleCBM(CarbonBoxModel):
    """Detailed-balance CBM with symmetric undirected fluxes ``F_{ij} = F_{ji}``.

    Every undirected reservoir pair ``{i, j}`` carries equal forward and reverse
    ``12C`` fluxes ``F_{ij} = F_{ji} = exp(log_sym_fluxes[u])``, guaranteeing
    exact Kirchhoff balance for arbitrary ``masses = exp(log_masses)``.
    """

    log_masses: jax.Array
    log_sym_fluxes: jax.Array
    pair_forward_edge: tuple[int, ...] = eqx.field(static=True)
    pair_reverse_edge: tuple[int, ...] = eqx.field(static=True)
    prod_weights: tuple[float, ...] = eqx.field(static=True)

    def __init__(
        self,
        topology: Topology,
        log_masses: ArrayLike,
        log_sym_fluxes: ArrayLike,
        *,
        prod_weights: Sequence[float] | ArrayLike | None = None,
        prod_fractions: Sequence[float] | ArrayLike | None = None,
        decay_rate: float = DECAY_CAMBRIDGE,
    ):
        topology.check_strongly_connected()
        self.topology = topology
        self.decay_rate = float(decay_rate)

        edge_lookup = {
            (s, d): idx
            for idx, (s, d) in enumerate(zip(topology.edges_src, topology.edges_dst))
        }
        fwd_list: list[int] = []
        rev_list: list[int] = []
        visited: set[tuple[int, int]] = set()
        for idx, (s, d) in enumerate(zip(topology.edges_src, topology.edges_dst)):
            if (s, d) in visited:
                continue
            if (d, s) not in edge_lookup:
                raise ValueError(
                    f"ReversibleCBM requires symmetric edge pairs; missing reverse edge ({d} -> {s})."
                )
            ridx = edge_lookup[(d, s)]
            fwd_list.append(idx)
            rev_list.append(ridx)
            visited.add((s, d))
            visited.add((d, s))

        self.pair_forward_edge = tuple(fwd_list)
        self.pair_reverse_edge = tuple(rev_list)

        self.log_masses = jnp.asarray(log_masses, dtype=jnp.float64)
        self.log_sym_fluxes = jnp.asarray(log_sym_fluxes, dtype=jnp.float64)
        if self.log_masses.shape != (topology.n_boxes,):
            raise ValueError(
                f"log_masses must have shape ({topology.n_boxes},), got {self.log_masses.shape}."
            )
        if self.log_sym_fluxes.shape != (len(fwd_list),):
            raise ValueError(
                f"log_sym_fluxes must have shape ({len(fwd_list)},), got {self.log_sym_fluxes.shape}."
            )

        self.prod_weights = _resolve_prod_weights(topology, prod_weights, prod_fractions)

    def reservoir_masses(self) -> jax.Array:
        return jnp.exp(self.log_masses)

    def edge_fluxes(self) -> jax.Array:
        sym_f = jnp.exp(self.log_sym_fluxes)
        f = jnp.zeros((self.topology.n_edges,), dtype=jnp.float64)
        f = f.at[np.asarray(self.pair_forward_edge, dtype=np.int64)].set(sym_f)
        f = f.at[np.asarray(self.pair_reverse_edge, dtype=np.int64)].set(sym_f)
        return f

    def rate_matrix_K(self) -> jax.Array:
        src = np.asarray(self.topology.edges_src, dtype=np.int64)
        k_edges = self.edge_fluxes() / self.reservoir_masses()[src]
        return _assemble_rate_matrix(
            self.n_boxes, self.topology.edges_src, self.topology.edges_dst, k_edges
        )

    def production_vector(self) -> jax.Array:
        return _make_prod_vector(
            self.n_boxes, self.topology.prod_box_indices, self.prod_weights
        )

    @classmethod
    def from_masses_and_fluxes(
        cls,
        topology: Topology,
        masses: ArrayLike,
        edge_fluxes: ArrayLike,
        *,
        prod_weights: Sequence[float] | ArrayLike | None = None,
        prod_fractions: Sequence[float] | ArrayLike | None = None,
        decay_rate: float = DECAY_CAMBRIDGE,
    ) -> ReversibleCBM:
        m_arr = jnp.asarray(masses, dtype=jnp.float64)
        f_arr = jnp.asarray(edge_fluxes, dtype=jnp.float64)
        if f_arr.ndim == 2:
            src = np.asarray(topology.edges_src, dtype=np.int64)
            dst = np.asarray(topology.edges_dst, dtype=np.int64)
            f_arr = f_arr[src, dst]

        edge_lookup = {
            (s, d): idx
            for idx, (s, d) in enumerate(zip(topology.edges_src, topology.edges_dst))
        }
        fwd_indices: list[int] = []
        visited: set[tuple[int, int]] = set()
        for idx, (s, d) in enumerate(zip(topology.edges_src, topology.edges_dst)):
            if (s, d) in visited:
                continue
            ridx = edge_lookup[(d, s)]
            fwd_indices.append(idx)
            visited.add((s, d))
            visited.add((d, s))
            _ = ridx

        sym_f = f_arr[np.asarray(fwd_indices, dtype=np.int64)]
        return cls(
            topology=topology,
            log_masses=jnp.log(m_arr),
            log_sym_fluxes=jnp.log(sym_f),
            prod_weights=prod_weights,
            prod_fractions=prod_fractions,
            decay_rate=decay_rate,
        )


class OpenCBM(CarbonBoxModel):
    """Perron–Frobenius CBM with explicit open deep-reservoir loss ``log_sink_rates``.

    Used for shallow truncated models (such as 4-box ``Miyake17``) where omitting
    the deep ocean would otherwise trap all global ``14C`` in shallow reservoirs.
    """

    log_rates_free: jax.Array
    log_total_mass: jax.Array
    log_sink_rates: jax.Array
    open_box_indices: tuple[int, ...] = eqx.field(static=True)
    prod_weights: tuple[float, ...] = eqx.field(static=True)
    tie_matrix_rows: tuple[tuple[float, ...], ...] | None = eqx.field(static=True)

    def __init__(
        self,
        topology: Topology,
        log_rates_free: ArrayLike,
        log_total_mass: ArrayLike,
        log_sink_rates: ArrayLike,
        *,
        open_box_indices: Sequence[int],
        prod_weights: Sequence[float] | ArrayLike | None = None,
        prod_fractions: Sequence[float] | ArrayLike | None = None,
        tie_matrix: ArrayLike | None = None,
        symmetric_edge_groups: Sequence[Sequence[int]] | None = None,
        decay_rate: float = DECAY_CAMBRIDGE,
    ):
        topology.check_strongly_connected()
        self.topology = topology
        self.decay_rate = float(decay_rate)

        open_idx = tuple(int(i) for i in open_box_indices)
        if not open_idx or any(i < 0 or i >= topology.n_boxes for i in open_idx):
            raise ValueError(
                f"open_box_indices must be non-empty and within [0, {topology.n_boxes}), got {open_idx}."
            )
        self.open_box_indices = open_idx

        self.tie_matrix_rows = _build_tie_matrix(
            topology.n_edges,
            tie_matrix=tie_matrix,
            symmetric_edge_groups=symmetric_edge_groups,
        )
        expected_p = (
            topology.n_edges
            if self.tie_matrix_rows is None
            else len(self.tie_matrix_rows[0])
        )

        self.log_rates_free = jnp.asarray(log_rates_free, dtype=jnp.float64)
        self.log_total_mass = jnp.asarray(log_total_mass, dtype=jnp.float64).reshape(())
        self.log_sink_rates = jnp.asarray(log_sink_rates, dtype=jnp.float64)

        if self.log_rates_free.shape != (expected_p,):
            raise ValueError(
                f"log_rates_free must have shape ({expected_p},), got {self.log_rates_free.shape}."
            )
        if self.log_sink_rates.shape != (len(open_idx),):
            raise ValueError(
                f"log_sink_rates must have shape ({len(open_idx)},), got {self.log_sink_rates.shape}."
            )

        self.prod_weights = _resolve_prod_weights(topology, prod_weights, prod_fractions)

    def edge_rates(self) -> jax.Array:
        if self.tie_matrix_rows is None:
            log_k = self.log_rates_free
        else:
            M_tie = jnp.asarray(self.tie_matrix_rows, dtype=jnp.float64)
            log_k = M_tie @ self.log_rates_free
        return jnp.exp(log_k)

    def rate_matrix_K(self) -> jax.Array:
        return _assemble_rate_matrix(
            self.n_boxes,
            self.topology.edges_src,
            self.topology.edges_dst,
            self.edge_rates(),
        )

    def _perron_simplex(self) -> jax.Array:
        k_edges = self.edge_rates()
        K = _assemble_rate_matrix(
            self.n_boxes, self.topology.edges_src, self.topology.edges_dst, k_edges
        )
        return _perron_frobenius_simplex(K, self.topology.edges_src, k_edges)

    def reservoir_masses(self) -> jax.Array:
        return jnp.exp(self.log_total_mass) * self._perron_simplex()

    def relative_reservoir_masses(self) -> jax.Array:
        trop_idx = jnp.asarray(self.topology.trop_box_indices, dtype=jnp.int32)
        pi = self._perron_simplex()
        return pi / jnp.sum(pi[trop_idx])

    def sink_rates(self) -> jax.Array:
        out = jnp.zeros((self.n_boxes,), dtype=jnp.float64)
        return out.at[np.asarray(self.open_box_indices, dtype=np.int64)].set(
            jnp.exp(self.log_sink_rates)
        )

    def production_vector(self) -> jax.Array:
        return _make_prod_vector(
            self.n_boxes, self.topology.prod_box_indices, self.prod_weights
        )

    @classmethod
    def from_masses_and_fluxes(
        cls,
        topology: Topology,
        masses: ArrayLike,
        edge_fluxes: ArrayLike,
        sink_rates: ArrayLike,
        *,
        open_box_indices: Sequence[int] | None = None,
        prod_weights: Sequence[float] | ArrayLike | None = None,
        prod_fractions: Sequence[float] | ArrayLike | None = None,
        tie_matrix: ArrayLike | None = None,
        symmetric_edge_groups: Sequence[Sequence[int]] | None = None,
        decay_rate: float = DECAY_CAMBRIDGE,
    ) -> OpenCBM:
        m_arr = jnp.asarray(masses, dtype=jnp.float64)
        f_arr = jnp.asarray(edge_fluxes, dtype=jnp.float64)
        s_arr = np.asarray(sink_rates, dtype=np.float64)
        if open_box_indices is None:
            open_idx = tuple(int(i) for i in np.flatnonzero(s_arr > 0.0))
        else:
            open_idx = tuple(int(i) for i in open_box_indices)
        if not open_idx:
            raise ValueError(
                "OpenCBM requires at least one box with positive sink_rate or explicit open_box_indices."
            )

        src = np.asarray(topology.edges_src, dtype=np.int64)
        log_k = jnp.log(f_arr / m_arr[src])
        log_m_tot = jnp.log(jnp.sum(m_arr))

        tie_rows = _build_tie_matrix(
            topology.n_edges,
            tie_matrix=tie_matrix,
            symmetric_edge_groups=symmetric_edge_groups,
        )
        if tie_rows is None:
            theta_free = log_k
        else:
            M_tie = jnp.asarray(tie_rows, dtype=jnp.float64)
            theta_free, _, _, _ = jnp.linalg.lstsq(M_tie, log_k, rcond=None)

        log_s = jnp.log(jnp.asarray(s_arr[list(open_idx)], dtype=jnp.float64))
        return cls(
            topology=topology,
            log_rates_free=theta_free,
            log_total_mass=log_m_tot,
            log_sink_rates=log_s,
            open_box_indices=open_idx,
            prod_weights=prod_weights,
            prod_fractions=prod_fractions,
            tie_matrix=tie_matrix,
            symmetric_edge_groups=symmetric_edge_groups,
            decay_rate=decay_rate,
        )

    @classmethod
    def from_spec(
        cls,
        spec: Any,
        *,
        tie_symmetric: bool = False,
        decay_rate: float = DECAY_CAMBRIDGE,
    ) -> OpenCBM:
        topo = Topology.from_spec(spec)
        edge_fluxes = [float(f) for _, _, f in spec.edges if float(f) > 0.0]
        groups = spec.symmetric_edge_groups if tie_symmetric else None
        return cls.from_masses_and_fluxes(
            topology=topo,
            masses=spec.masses,
            edge_fluxes=edge_fluxes,
            sink_rates=spec.sink_rates,
            prod_weights=spec.prod_coefficients,
            symmetric_edge_groups=groups,
            decay_rate=decay_rate,
        )


def _calibrate_effective_deep_sink(
    n_boxes: int,
    k_strat_to_trop: float,
    k_trop_to_strat: float,
    k_trop_to_sinks: np.ndarray,
    sink_taus: np.ndarray,
    *,
    eta_strat: float,
    trop_mass: float,
    target_r_ref: float,
    decay_rate: float,
    q_std: float = DEFAULT_Q_STD,
    alpha_unit: float = ALPHA_ATOMS_CM2_S_TO_KG_YR,
) -> float:
    """Solve analytically for ``k_deep >= 0`` on the slowest box to match ``target_r_ref``."""
    lam = float(decay_rate)
    rhs_top = (1.0 - eta_strat) + eta_strat * k_strat_to_trop / (lam + k_strat_to_trop)
    target_g1 = (trop_mass * target_r_ref) / (alpha_unit * q_std)
    target_denom = rhs_top / target_g1

    strat_term = lam * k_trop_to_strat / (lam + k_strat_to_trop)
    if n_boxes == 2:
        k_deep = target_denom - lam - strat_term
        return max(float(k_deep), 0.0)

    m = n_boxes - 2
    other_sinks = 0.0
    for j in range(m - 1):
        inv_tau = 1.0 / float(sink_taus[j])
        other_sinks += float(k_trop_to_sinks[j]) * lam / (lam + inv_tau)

    needed_last = target_denom - lam - strat_term - other_sinks
    k_up = float(k_trop_to_sinks[m - 1])
    inv_tau_last = 1.0 / float(sink_taus[m - 1])

    # Solve k_up * (lam + k_deep) / (lam + inv_tau_last + k_deep) = needed_last
    if needed_last <= k_up * lam / (lam + inv_tau_last):
        return 0.0
    if needed_last >= k_up * 0.999:
        # Cap if single sink uptake is near saturation
        return float(10.0 * inv_tau_last)

    num = needed_last * (lam + inv_tau_last) - k_up * lam
    den = k_up - needed_last
    return max(float(num / den), 0.0)


def _inv_softplus(y: float) -> float:
    if y > 20.0:
        return y
    return math.log(math.expm1(y))


class EffectiveCBM(CarbonBoxModel):
    """Minimal gauge-free ``K``-box mammillary CBM (``K >= 2``) with ``2K - 2`` parameters.

    Topology is ``Stratosphere <-> Troposphere <-> {Sink_1, ..., Sink_{K-2}}`` with:

    1. Fixed cosmic-ray stratospheric production fraction ``eta_strat = 0.70``
       (eliminating the continuous ``(k_01, k_10, eta)`` 3-into-2 gauge mode).
    2. Fixed reference tropospheric mass ``trop_mass = 501.5 GtC`` (fixing the
       zero-Fisher global mass-scale gauge) and calibrated open deep-ocean loss rate
       ``deep_sink_rate >= 0`` on the slowest box so pre-industrial ``R_ref`` matches
       ``Guttler15`` (``~1.3195 kg/GtC``).
    3. Strict sink residence-time ordering ``ln tau_1 < ln tau_2 < ... < ln tau_{K-2}``
       via ``softplus`` increments ``ln tau_j = ln tau_{j-1} + softplus(delta_j)``,
       eliminating all ``(K - 2)!`` discrete sink-permutation modes.

    The dynamic JAX array leaves have total dimension ``1 + 1 + (K - 2) + (K - 2) = 2K - 2``.
    """

    log_k_strat_to_trop: jax.Array
    log_k_trop_to_strat: jax.Array
    log_k_trop_to_sinks: jax.Array
    sink_tau_params: jax.Array
    K: int = eqx.field(static=True)
    eta_strat: float = eqx.field(static=True)
    trop_mass: float = eqx.field(static=True)
    deep_sink_rate: float = eqx.field(static=True)

    def __init__(
        self,
        K: int = 3,
        *,
        log_k_strat_to_trop: ArrayLike | None = None,
        log_k_trop_to_strat: ArrayLike | None = None,
        log_k_trop_to_sinks: ArrayLike | None = None,
        sink_tau_params: ArrayLike | None = None,
        eta_strat: float = 0.70,
        trop_mass: float = GUTTLER15_TROP_MASS_GTC,
        deep_sink_rate: float | None = None,
        target_r_ref: float = GUTTLER15_R_REF_KG_PER_GTC,
        decay_rate: float = DECAY_CAMBRIDGE,
    ):
        k_int = int(K)
        if k_int < 2:
            raise ValueError(f"EffectiveCBM requires K >= 2 boxes, got K={k_int}.")
        if not (0.0 < float(eta_strat) < 1.0):
            raise ValueError(f"eta_strat must lie strictly in (0, 1), got {eta_strat}.")

        n_sinks = k_int - 2
        box_names = ("Stratosphere", "Troposphere") + tuple(
            f"Sink_{j + 1}" for j in range(n_sinks)
        )
        edges_src_list = [0, 1]
        edges_dst_list = [1, 0]
        for j in range(n_sinks):
            sink_idx = 2 + j
            edges_src_list.extend([1, sink_idx])
            edges_dst_list.extend([sink_idx, 1])

        self.topology = Topology(
            box_names=box_names,
            edges_src=edges_src_list,
            edges_dst=edges_dst_list,
            prod_box_indices=(0, 1),
            trop_box_indices=(1,),
            check_scc=True,
        )
        self.K = k_int
        self.eta_strat = float(eta_strat)
        self.trop_mass = float(trop_mass)
        self.decay_rate = float(decay_rate)

        # Physically motivated defaults matching Guttler15 stratosphere & decadal/centennial uptake
        default_k01 = 45.0 / 88.5
        default_k10 = 45.0 / float(trop_mass)

        if log_k_strat_to_trop is None:
            self.log_k_strat_to_trop = jnp.asarray(
                math.log(default_k01), dtype=jnp.float64
            )
        else:
            self.log_k_strat_to_trop = jnp.asarray(
                log_k_strat_to_trop, dtype=jnp.float64
            ).reshape(())

        if log_k_trop_to_strat is None:
            self.log_k_trop_to_strat = jnp.asarray(
                math.log(default_k10), dtype=jnp.float64
            )
        else:
            self.log_k_trop_to_strat = jnp.asarray(
                log_k_trop_to_strat, dtype=jnp.float64
            ).reshape(())

        if n_sinks == 0:
            self.log_k_trop_to_sinks = jnp.zeros((0,), dtype=jnp.float64)
            self.sink_tau_params = jnp.zeros((0,), dtype=jnp.float64)
        else:
            if log_k_trop_to_sinks is None:
                # Partition ~180 GtC/yr total surface-ocean + biosphere uptake across sinks
                weights = np.geomspace(1.0, 0.35, n_sinks)
                weights = weights / np.sum(weights)
                k_up_default = (180.0 * weights) / float(trop_mass)
                self.log_k_trop_to_sinks = jnp.asarray(
                    np.log(k_up_default), dtype=jnp.float64
                )
            else:
                self.log_k_trop_to_sinks = jnp.asarray(
                    log_k_trop_to_sinks, dtype=jnp.float64
                )

            if sink_tau_params is None:
                # Log-spaced residence times starting around ~8 yr with ratio ~5x per sink
                log_taus_default = math.log(8.0) + np.arange(n_sinks) * math.log(5.5)
                raw = np.empty((n_sinks,), dtype=np.float64)
                raw[0] = log_taus_default[0]
                if n_sinks > 1:
                    diffs = np.diff(log_taus_default)
                    raw[1:] = np.array([_inv_softplus(float(d)) for d in diffs])
                self.sink_tau_params = jnp.asarray(raw, dtype=jnp.float64)
            else:
                self.sink_tau_params = jnp.asarray(sink_tau_params, dtype=jnp.float64)

        if self.log_k_trop_to_sinks.shape != (n_sinks,):
            raise ValueError(
                f"log_k_trop_to_sinks must have shape ({n_sinks},), got {self.log_k_trop_to_sinks.shape}."
            )
        if self.sink_tau_params.shape != (n_sinks,):
            raise ValueError(
                f"sink_tau_params must have shape ({n_sinks},), got {self.sink_tau_params.shape}."
            )

        if deep_sink_rate is None:
            c_k01 = _concrete_array(jnp.exp(self.log_k_strat_to_trop))
            c_k10 = _concrete_array(jnp.exp(self.log_k_trop_to_strat))
            c_kup = _concrete_array(jnp.exp(self.log_k_trop_to_sinks))
            c_taus = _concrete_array(self.sink_timescales())
            if (
                c_k01 is not None
                and c_k10 is not None
                and c_kup is not None
                and c_taus is not None
            ):
                self.deep_sink_rate = _calibrate_effective_deep_sink(
                    k_int,
                    float(c_k01),
                    float(c_k10),
                    c_kup,
                    c_taus,
                    eta_strat=self.eta_strat,
                    trop_mass=self.trop_mass,
                    target_r_ref=float(target_r_ref),
                    decay_rate=self.decay_rate,
                )
            else:
                self.deep_sink_rate = 0.005
        else:
            self.deep_sink_rate = float(deep_sink_rate)

    @property
    def n_parameters(self) -> int:
        return 2 * self.K - 2

    def log_sink_timescales(self) -> jax.Array:
        """Strictly ordered log residence times ``ln tau_1 < ... < ln tau_{K-2}``."""
        n_sinks = self.K - 2
        if n_sinks == 0:
            return jnp.zeros((0,), dtype=jnp.float64)
        if n_sinks == 1:
            return self.sink_tau_params
        increments = jax.nn.softplus(self.sink_tau_params[1:])
        return jnp.concatenate(
            [
                self.sink_tau_params[:1],
                self.sink_tau_params[0] + jnp.cumsum(increments),
            ]
        )

    def sink_timescales(self) -> jax.Array:
        """Strictly ordered sink residence times ``tau_1 < ... < tau_{K-2}`` (``yr``)."""
        return jnp.exp(self.log_sink_timescales())

    def edge_rates(self) -> jax.Array:
        k01 = jnp.exp(self.log_k_strat_to_trop)
        k10 = jnp.exp(self.log_k_trop_to_strat)
        if self.K == 2:
            return jnp.stack([k01, k10])
        k_up = jnp.exp(self.log_k_trop_to_sinks)
        k_ret = jnp.exp(-self.log_sink_timescales())
        sink_pairs = jnp.stack([k_up, k_ret], axis=1).reshape(-1)
        return jnp.concatenate([jnp.stack([k01, k10]), sink_pairs])

    def rate_matrix_K(self) -> jax.Array:
        return _assemble_rate_matrix(
            self.n_boxes,
            self.topology.edges_src,
            self.topology.edges_dst,
            self.edge_rates(),
        )

    def reservoir_masses(self) -> jax.Array:
        k01 = jnp.exp(self.log_k_strat_to_trop)
        k10 = jnp.exp(self.log_k_trop_to_strat)
        m_trop = jnp.asarray(self.trop_mass, dtype=jnp.float64)
        m_strat = m_trop * (k10 / k01)
        if self.K == 2:
            return jnp.stack([m_strat, m_trop])
        m_sinks = m_trop * jnp.exp(self.log_k_trop_to_sinks + self.log_sink_timescales())
        return jnp.concatenate([jnp.stack([m_strat, m_trop]), m_sinks])

    def sink_rates(self) -> jax.Array:
        out = jnp.zeros((self.n_boxes,), dtype=jnp.float64)
        return out.at[-1].set(float(self.deep_sink_rate))

    def production_vector(self) -> jax.Array:
        out = jnp.zeros((self.n_boxes,), dtype=jnp.float64)
        out = out.at[0].set(self.eta_strat)
        out = out.at[1].set(1.0 - self.eta_strat)
        return out
