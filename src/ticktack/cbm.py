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

from collections.abc import Mapping, Sequence
import math
from typing import Any, ClassVar, Literal

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
    "CBMChart",
    "FixedChart",
    "BalancedChart",
    "ReversibleChart",
    "OpenChart",
    "CarbonBoxModel",
    "FixedCBM",
    "BalancedCBM",
    "SubspaceCBM",
    "FisherSubspaceCBM",
    "ReversibleCBM",
    "OpenCBM",
    "EffectiveCBM",
]


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


class CBMChart(zdx.Base):
    """Abstract coordinate chart mapping unconstrained JAX leaves to LTI box rates."""

    is_detailed_balance: ClassVar[bool] = False
    is_subspace: ClassVar[bool] = False

    def edge_rates(self, topology: Topology) -> jax.Array:
        """Directed inter-box transfer rates ``k_{src->dst}`` (``yr^-1``)."""
        raise NotImplementedError

    def rate_matrix_K(self, topology: Topology) -> jax.Array:
        """Column-conserving inter-box transfer matrix ``K`` (``yr^-1``)."""
        return _assemble_rate_matrix(
            topology.n_boxes, topology.edges_src, topology.edges_dst, self.edge_rates(topology)
        )

    def reservoir_masses(self, topology: Topology) -> jax.Array:
        """Steady-state ``12C`` reservoir inventories ``M_0`` in ``GtC``."""
        raise NotImplementedError

    def sink_rates(self, topology: Topology, decay_rate: float = DECAY_CAMBRIDGE) -> jax.Array:
        """Non-negative open reservoir loss rates ``k_sink`` (``yr^-1``)."""
        return jnp.zeros((topology.n_boxes,), dtype=jnp.float64)

    def production_vector(self, topology: Topology) -> jax.Array:
        """Normalized cosmic-ray ``14C`` production allocation ``b`` summing to 1."""
        pw = getattr(self, "prod_weights", None)
        if pw is None:
            raise NotImplementedError
        return _make_prod_vector(topology.n_boxes, topology.prod_box_indices, pw)

    def relative_reservoir_masses(self, topology: Topology) -> jax.Array:
        """Reservoir ``12C`` inventories normalized by global tropospheric mass."""
        m_0 = self.reservoir_masses(topology)
        return m_0 / jnp.sum(m_0[jnp.asarray(topology.trop_box_indices, dtype=jnp.int32)])

    def active_parameter_mask(self) -> np.ndarray:
        """Boolean mask over flattened inexact JAX leaves excluding gauge coordinates."""
        leaves = jax.tree_util.tree_leaves(eqx.filter(self, eqx.is_inexact_array))
        return np.ones((sum(int(x.size) for x in leaves),), dtype=bool)

    def param_specs(self, prefix: str = "cbm", *, prior_sd: float | None = None) -> tuple[Any, ...]:
        """Default :class:`~ticktack.likelihood.ParamSpec` tuple for inference."""
        return ()


def _require_prior_sd(prior_sd: float | None) -> float:
    if prior_sd is None or not math.isfinite(float(prior_sd)) or float(prior_sd) <= 0.0:
        raise ValueError("prior_sd must be specified explicitly as a finite positive float.")
    return float(prior_sd)


def _box_norm_key(k: Any) -> Any:
    return (
        (str(k[0]).strip().lower(), _normalize_hemisphere(k[1]))
        if isinstance(k, tuple) and len(k) == 2
        else str(k).strip().lower()
    )


def _lookup_box(k: Any, lookup: dict[Any, int]) -> int:
    idx = lookup.get(k, lookup.get(_box_norm_key(k)))
    if idx is None:
        raise KeyError(f"Unknown reservoir {k!r}.")
    return idx


def _resolve_named_vector(
    spec: Mapping[Any, float] | Sequence[float] | None, n: int, lookup: dict[Any, int]
) -> np.ndarray | None:
    if spec is None or not isinstance(spec, Mapping):
        arr = None if spec is None else np.asarray(spec, dtype=np.float64)
        if arr is not None and arr.shape != (n,):
            raise ValueError(f"Sequence length {arr.shape} must match reservoir count ({n},).")
        return arr
    arr = np.zeros(n, dtype=np.float64)
    for k, v in spec.items():
        arr[_lookup_box(k, lookup)] = float(v)
    return arr


class CarbonBoxModel(zdx.Base):
    """Concrete immutable ``zodiax`` Carbon Box Model combining :class:`Topology` and :class:`CBMChart`."""

    topology: Topology = eqx.field(static=True)
    chart: CBMChart | None = None
    decay_rate: float = eqx.field(static=True, default=DECAY_CAMBRIDGE)

    def __init__(self, topology: Topology, chart: CBMChart, *, decay_rate: float = DECAY_CAMBRIDGE):
        if not isinstance(chart, CBMChart):
            raise TypeError(f"chart must be a CBMChart, got {type(chart).__name__}.")
        self.topology, self.chart, self.decay_rate = topology, chart, float(decay_rate)

    def __getattr__(self, name: str) -> Any:
        if not (name.startswith("__") and name.endswith("__")):
            chart = self.__dict__.get("chart")
            if chart is not None and hasattr(chart, name):
                return getattr(chart, name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def _init_common(
        self,
        topology: Topology,
        decay_rate: float,
        prod_weights: Sequence[float] | ArrayLike | None = None,
        *,
        require_scc: bool = False,
    ) -> None:
        if require_scc:
            topology.check_strongly_connected()
        self.topology = topology
        self.decay_rate = float(decay_rate)
        self.prod_weights = _resolve_prod_weights(topology, prod_weights)

    @property
    def n_boxes(self) -> int:
        return self.topology.n_boxes

    @property
    def is_detailed_balance(self) -> bool:
        return (
            bool(self.chart.is_detailed_balance)
            if self.chart is not None
            else isinstance(self, EffectiveCBM)
        )

    @property
    def is_subspace(self) -> bool:
        return (
            bool(self.chart.is_subspace)
            if self.chart is not None
            else isinstance(self, SubspaceCBM)
        )

    def active_parameter_mask(self) -> np.ndarray:
        """Boolean mask over flattened inexact JAX leaves excluding gauge coordinates."""
        if self.chart is not None:
            return self.chart.active_parameter_mask()
        leaves = jax.tree_util.tree_leaves(eqx.filter(self, eqx.is_inexact_array))
        return np.ones((sum(int(x.size) for x in leaves),), dtype=bool)

    def box_index(self, name: str = "Troposphere", hemisphere: str | None = None) -> int:
        return self.topology.box_index(name=name, hemisphere=hemisphere)

    def edge_rates(self) -> jax.Array:
        """Directed inter-box transfer rates ``k_{src->dst}`` (``yr^-1``)."""
        if self.chart is not None:
            return self.chart.edge_rates(self.topology)
        raise NotImplementedError

    def rate_matrix_K(self) -> jax.Array:
        """Column-conserving inter-box transfer matrix ``K`` (``yr^-1``)."""
        if self.chart is not None:
            return self.chart.rate_matrix_K(self.topology)
        return _assemble_rate_matrix(
            self.n_boxes,
            self.topology.edges_src,
            self.topology.edges_dst,
            self.edge_rates(),
        )

    def sink_rates(self) -> jax.Array:
        """Non-negative open reservoir loss rates ``k_sink`` (``yr^-1``)."""
        if self.chart is not None:
            return self.chart.sink_rates(self.topology, self.decay_rate)
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
        if self.chart is not None:
            return self.chart.production_vector(self.topology)
        prod_weights = getattr(self, "prod_weights", None)
        if prod_weights is None:
            raise NotImplementedError
        return _make_prod_vector(
            self.n_boxes, self.topology.prod_box_indices, prod_weights
        )

    def reservoir_masses(self) -> jax.Array:
        """Steady-state ``12C`` reservoir inventories ``M_0`` in ``GtC``."""
        if self.chart is not None:
            return self.chart.reservoir_masses(self.topology)
        raise NotImplementedError

    def relative_reservoir_masses(self) -> jax.Array:
        """Reservoir ``12C`` inventories normalized by global tropospheric mass ``sum_{j in T} M_{0,j}``."""
        if self.chart is not None:
            return self.chart.relative_reservoir_masses(self.topology)
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

    def residence_times(self) -> jax.Array:
        """Reservoir carbon turnover times ``tau_i = 1 / (-K_ii + k_sink_i)`` in ``yr``."""
        exit_rate = -jnp.diag(self.rate_matrix_K()) + self.sink_rates()
        return jnp.where(exit_rate > 0.0, 1.0 / jnp.where(exit_rate > 0.0, exit_rate, 1.0), jnp.inf)

    def with_chart(self, chart: CBMChart) -> CarbonBoxModel:
        """Return a new :class:`CarbonBoxModel` pairing ``self.topology`` with ``chart``."""
        return CarbonBoxModel(self.topology, chart, decay_rate=self.decay_rate)

    def to_fixed(self) -> CarbonBoxModel:
        """Materialize the current box state into a :class:`FixedChart` :class:`CarbonBoxModel`."""
        masses, src = self.reservoir_masses(), np.asarray(self.topology.edges_src, dtype=np.int64)
        pw = self.production_vector() if (pw := getattr(self, "prod_weights", None)) is None else pw
        return CarbonBoxModel(
            self.topology,
            _chart_from_state(
                self.topology,
                (masses, self.edge_rates() * masses[src], pw, self.sink_rates()),
                "fixed",
            ),
            decay_rate=self.decay_rate,
        )

    def to_balanced(
        self,
        *,
        tie_matrix: ArrayLike | None = None,
        symmetric_edge_groups: Sequence[Sequence[int]] | None = None,
    ) -> CarbonBoxModel:
        """Project a closed CBM onto an equivalent :class:`BalancedChart` :class:`CarbonBoxModel`."""
        if isinstance(self, SubspaceCBM) and tie_matrix is None and symmetric_edge_groups is None:
            return CarbonBoxModel(
                self.topology,
                BalancedChart(
                    self.topology,
                    self.log_rates_free(),
                    self.base_log_total_mass,
                    prod_weights=self.prod_weights,
                    tie_matrix=self.tie_matrix_rows,
                ),
                decay_rate=self.decay_rate,
            )
        masses, src = self.reservoir_masses(), np.asarray(self.topology.edges_src, dtype=np.int64)
        pw = self.production_vector() if (pw := getattr(self, "prod_weights", None)) is None else pw
        if tie_matrix is None and symmetric_edge_groups is None:
            return CarbonBoxModel(
                self.topology,
                _chart_from_state(
                    self.topology,
                    (masses, self.edge_rates() * masses[src], pw, self.sink_rates()),
                    "balanced",
                ),
                decay_rate=self.decay_rate,
            )
        if np.any(np.asarray(self.sink_rates(), dtype=np.float64) > 0.0):
            raise ValueError("balanced chart does not allow positive sinks; use chart='open'.")
        return CarbonBoxModel(
            self.topology,
            BalancedChart.from_masses_and_fluxes(
                self.topology,
                masses,
                self.edge_rates() * masses[src],
                prod_weights=pw,
                tie_matrix=tie_matrix,
                symmetric_edge_groups=symmetric_edge_groups,
            ),
            decay_rate=self.decay_rate,
        )

    def param_specs(self, prefix: str = "cbm", **kwargs: Any) -> tuple[Any, ...]:
        """Return default :class:`~ticktack.likelihood.ParamSpec` tuple for ``ParameterSpace``."""
        if self.chart is not None:
            return self.chart.param_specs(prefix=prefix, **kwargs)
        if isinstance(self, SubspaceCBM) and self.n_modes > 0:
            from .likelihood import ParamSpec

            return (
                ParamSpec(
                    f"{prefix}.alpha",
                    "identity",
                    self.alpha,
                    _require_prior_sd(kwargs.get("prior_sd", 1.0)),
                ),
            )
        return ()

    def summary(self) -> str:
        """Format a plain-text reservoir summary table (mass, residence time, sink, steady-state d14C)."""
        cname = type(self.chart).__name__ if self.chart is not None else type(self).__name__
        masses = np.asarray(self.reservoir_masses())
        taus = np.asarray(self.residence_times())
        sinks = np.asarray(self.sink_rates())
        prod = np.asarray(self.production_vector())
        d14c = np.asarray(self.steady_state_d14c())
        r_ref = float(self.reference_specific_activity())
        max_imbalance = float(jnp.max(jnp.abs(self.kirchhoff_residual())))
        lines = [
            f"CarbonBoxModel (chart={cname}, n_boxes={self.n_boxes}, "
            f"n_edges={self.topology.n_edges}, R_ref={r_ref:.4f} kg/GtC, "
            f"max|dF|={max_imbalance:.2e} GtC/yr)",
            f"{'Idx':>3}  {'Reservoir':<28} {'Mass [GtC]':>12} {'Tau [yr]':>10} {'Sink [yr^-1]':>12} {'Prod':>6} {'d14C_ss [permil]':>16}",
            "-" * 95,
        ]
        for i, (bname, bhemi) in enumerate(zip(self.topology.box_names, self.topology.hemispheres)):
            lbl = f"{bname} ({bhemi[0].upper()})" if self.topology.is_hemispheric else bname
            lines.append(
                f"{i:>3d}  {lbl:<28.28} {masses[i]:>12.2f} {taus[i]:>10.2f} {sinks[i]:>12.4e} {prod[i]:>6.2f} {d14c[i]:>+16.2f}"
            )
        return "\n".join(lines)

    @classmethod
    def from_boxes(
        cls,
        masses: Mapping[Any, float] | Sequence[tuple[Any, float]],
        fluxes: Sequence[tuple[Any, Any, float]],
        *,
        production: Mapping[Any, float] | Sequence[float] | None = None,
        sinks: Mapping[Any, float] | Sequence[float] | None = None,
        chart: Literal["fixed", "balanced", "reversible", "open"] = "fixed",
        decay_rate: float = DECAY_CAMBRIDGE,
    ) -> CarbonBoxModel:
        """Construct a :class:`CarbonBoxModel` by reservoir name rather than 0-based indices."""
        items = list(masses.items()) if isinstance(masses, Mapping) else list(masses)
        if not items or any(float(m) <= 0.0 for _, m in items):
            raise ValueError("masses must be non-empty with strictly positive values.")
        has_hemi, names, hemis, lookup = isinstance(items[0][0], tuple), [], [], {}
        if not all(
            (isinstance(rk, tuple) and len(rk) == 2 and isinstance(rk[0], str))
            if has_hemi
            else isinstance(rk, str)
            for rk, _ in items
        ):
            raise ValueError(
                "Reservoir keys must be either all str names or all (name: str, hemisphere) 2-tuples."
            )
        for idx, (rk, _) in enumerate(items):
            ck = _box_norm_key(rk)
            if ck in lookup:
                raise ValueError(f"Duplicate reservoir key {rk!r}.")
            lookup[ck] = lookup[rk] = idx
            names.append(str(rk[0]) if has_hemi else str(rk))
            hemis.append(_normalize_hemisphere(rk[1]) if has_hemi else "none")
        m_arr = np.asarray([float(m) for _, m in items], dtype=np.float64)
        if any(float(f) <= 0.0 for _, _, f in fluxes):
            raise ValueError("All edge fluxes must be > 0.")
        src_l = [_lookup_box(s_k, lookup) for s_k, _, _ in fluxes]
        dst_l = [_lookup_box(d_k, lookup) for _, d_k, _ in fluxes]
        flx_l = [float(f) for _, _, f in fluxes]
        p_arr = _resolve_named_vector(production, len(names), lookup)
        if p_arr is not None and (np.any(p_arr < 0.0) or float(np.sum(p_arr)) <= 0.0):
            raise ValueError("production weights must be non-negative with positive sum.")
        prod_idx = None if p_arr is None else tuple(int(i) for i in np.flatnonzero(p_arr > 0.0))
        prod_w = None if p_arr is None or prod_idx is None else p_arr[list(prod_idx)]
        s_vec = _resolve_named_vector(sinks, len(names), lookup)
        s_arr = np.zeros(len(names), dtype=np.float64) if s_vec is None else s_vec
        if np.any(s_arr < 0.0):
            raise ValueError("sinks must be non-negative.")
        topo = Topology(
            names,
            src_l,
            dst_l,
            hemispheres=(tuple(hemis) if has_hemi else None),
            prod_box_indices=prod_idx,
            check_scc=True,
        )
        return cls(
            topo,
            _chart_from_state(topo, (m_arr, flx_l, prod_w, s_arr), chart),
            decay_rate=decay_rate,
        )


def _chart_from_state(topo: Topology, data: tuple[Any, Any, Any, Any], chart: str) -> CBMChart:
    m_arr, flx_l, prod_w, s_arr = data
    has_sinks = bool(np.any(np.asarray(s_arr, dtype=np.float64) > 0.0))
    if chart == "fixed":
        return FixedChart(topo, m_arr, flx_l, prod_weights=prod_w, sink_rates=s_arr)
    if chart == "balanced" and not has_sinks:
        return BalancedChart.from_masses_and_fluxes(topo, m_arr, flx_l, prod_weights=prod_w)
    if chart == "reversible" and not has_sinks:
        return ReversibleChart.from_masses_and_fluxes(topo, m_arr, flx_l, prod_weights=prod_w)
    if chart == "open":
        return OpenChart.from_masses_and_fluxes(topo, m_arr, flx_l, s_arr, prod_weights=prod_w)
    if has_sinks and chart in ("balanced", "reversible"):
        raise ValueError(f"{chart!r} chart does not allow positive sinks; use chart='open'.")
    raise ValueError(
        f"Unknown chart={chart!r}; expected 'fixed', 'balanced', 'reversible', or 'open'."
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


class FixedChart(CBMChart):
    """Coordinate chart with directly specified reservoir masses, fluxes, and sink rates."""

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
        sink_rates: ArrayLike | None = None,
    ):
        self.prod_weights = _resolve_prod_weights(topology, prod_weights)
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

    def reservoir_masses(self, topology: Topology) -> jax.Array:
        del topology
        return self.masses

    def edge_rates(self, topology: Topology) -> jax.Array:
        if topology.n_edges == 0:
            return jnp.zeros((0,), dtype=jnp.float64)
        src = np.asarray(topology.edges_src, dtype=np.int64)
        return self.edge_fluxes / self.masses[src]

    def sink_rates(self, topology: Topology, decay_rate: float = DECAY_CAMBRIDGE) -> jax.Array:
        del topology, decay_rate
        return self.sink_rates_arr


class FixedCBM(CarbonBoxModel):
    """Carbon Box Model wrapping :class:`FixedChart`."""

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
        pw = prod_fractions if prod_fractions is not None else prod_weights
        super().__init__(
            topology,
            FixedChart(topology, masses, edge_fluxes, prod_weights=pw, sink_rates=sink_rates),
            decay_rate=decay_rate,
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


def _invert_perron_chart(
    topology: Topology,
    masses: ArrayLike,
    edge_fluxes: ArrayLike,
    tie_matrix: ArrayLike | None = None,
    symmetric_edge_groups: Sequence[Sequence[int]] | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Map balanced ``(masses, edge_fluxes)`` to ``(log_rates_free, log_total_mass)``."""
    m_arr = jnp.asarray(masses, dtype=jnp.float64)
    f_arr = jnp.asarray(edge_fluxes, dtype=jnp.float64)
    src = np.asarray(topology.edges_src, dtype=np.int64)
    if f_arr.ndim == 2:
        dst = np.asarray(topology.edges_dst, dtype=np.int64)
        f_arr = f_arr[src, dst]
    log_k = jnp.log(f_arr / m_arr[src])
    log_m_tot = jnp.log(jnp.sum(m_arr))
    tie_rows = _build_tie_matrix(
        topology.n_edges,
        tie_matrix=tie_matrix,
        symmetric_edge_groups=symmetric_edge_groups,
    )
    if tie_rows is None:
        return log_k, log_m_tot
    M_tie = jnp.asarray(tie_rows, dtype=jnp.float64)
    theta_free, _, _, _ = jnp.linalg.lstsq(M_tie, log_k, rcond=None)
    return theta_free, log_m_tot


class _PerronCBM(CarbonBoxModel):
    """Internal base class for Perron–Frobenius flow-balanced CBMs."""

    prod_weights: tuple[float, ...] = eqx.field(static=True)
    tie_matrix_rows: tuple[tuple[float, ...], ...] | None = eqx.field(static=True)

    def _init_tie_matrix(
        self,
        tie_matrix: ArrayLike | None = None,
        symmetric_edge_groups: Sequence[Sequence[int]] | None = None,
    ) -> int:
        self.tie_matrix_rows = _build_tie_matrix(
            self.topology.n_edges,
            tie_matrix=tie_matrix,
            symmetric_edge_groups=symmetric_edge_groups,
        )
        return (
            self.topology.n_edges
            if self.tie_matrix_rows is None
            else len(self.tie_matrix_rows[0])
        )

    def _free_log_rates(self) -> jax.Array:
        return self.log_rates_free  # type: ignore[attr-defined]

    def _log_total_mass_value(self) -> jax.Array:
        return self.log_total_mass  # type: ignore[attr-defined]

    def edge_log_rates(self) -> jax.Array:
        lr = self._free_log_rates()
        if self.tie_matrix_rows is None:
            return lr
        M_tie = jnp.asarray(self.tie_matrix_rows, dtype=jnp.float64)
        return M_tie @ lr

    def edge_rates(self) -> jax.Array:
        return jnp.exp(self.edge_log_rates())

    def _perron_simplex(self) -> jax.Array:
        k_edges = self.edge_rates()
        K = _assemble_rate_matrix(
            self.n_boxes, self.topology.edges_src, self.topology.edges_dst, k_edges
        )
        return _perron_frobenius_simplex(K, self.topology.edges_src, k_edges)

    def reservoir_masses(self) -> jax.Array:
        return jnp.exp(self._log_total_mass_value()) * self._perron_simplex()

    def relative_reservoir_masses(self) -> jax.Array:
        trop_idx = jnp.asarray(self.topology.trop_box_indices, dtype=jnp.int32)
        pi = self._perron_simplex()
        return pi / jnp.sum(pi[trop_idx])


class _PerronChart(CBMChart):
    """Internal base coordinate chart for Perron–Frobenius flow-balanced CBMs."""

    prod_weights: tuple[float, ...] = eqx.field(static=True)
    tie_matrix_rows: tuple[tuple[float, ...], ...] | None = eqx.field(static=True)

    def _init_perron(
        self,
        topology: Topology,
        prod_weights: Sequence[float] | ArrayLike | None = None,
        tie_matrix: ArrayLike | None = None,
        symmetric_edge_groups: Sequence[Sequence[int]] | None = None,
    ) -> int:
        topology.check_strongly_connected()
        self.prod_weights = _resolve_prod_weights(topology, prod_weights)
        self.tie_matrix_rows = _build_tie_matrix(
            topology.n_edges, tie_matrix=tie_matrix, symmetric_edge_groups=symmetric_edge_groups
        )
        return topology.n_edges if self.tie_matrix_rows is None else len(self.tie_matrix_rows[0])

    def edge_log_rates(self) -> jax.Array:
        lr = self.log_rates_free  # type: ignore[attr-defined]
        return (
            lr
            if self.tie_matrix_rows is None
            else jnp.asarray(self.tie_matrix_rows, dtype=jnp.float64) @ lr
        )

    def edge_rates(self, topology: Topology) -> jax.Array:
        return jnp.exp(self.edge_log_rates())

    def _perron_simplex(self, topology: Topology) -> jax.Array:
        k_edges = self.edge_rates(topology)
        return _perron_frobenius_simplex(
            _assemble_rate_matrix(
                topology.n_boxes, topology.edges_src, topology.edges_dst, k_edges
            ),
            topology.edges_src,
            k_edges,
        )

    def reservoir_masses(self, topology: Topology) -> jax.Array:
        return jnp.exp(self.log_total_mass) * self._perron_simplex(topology)  # type: ignore[attr-defined]

    def relative_reservoir_masses(self, topology: Topology) -> jax.Array:
        pi = self._perron_simplex(topology)
        return pi / jnp.sum(pi[jnp.asarray(topology.trop_box_indices, dtype=jnp.int32)])

    def active_parameter_mask(self) -> np.ndarray:
        diff_tree, _ = eqx.partition(self, eqx.is_inexact_array)
        marked = eqx.tree_at(
            lambda m: m.log_total_mass,
            jax.tree_util.tree_map(lambda x: np.ones(x.shape, dtype=bool), diff_tree),
            np.asarray(False),
        )
        return np.concatenate([np.ravel(x) for x in jax.tree_util.tree_leaves(marked)])


class BalancedChart(_PerronChart):
    """Unconstrained equilibrium flow-balanced Perron–Frobenius coordinate chart."""

    log_rates_free: jax.Array
    log_total_mass: jax.Array

    def __init__(
        self,
        topology: Topology,
        log_rates_free: ArrayLike,
        log_total_mass: ArrayLike,
        *,
        prod_weights: Sequence[float] | ArrayLike | None = None,
        tie_matrix: ArrayLike | None = None,
        symmetric_edge_groups: Sequence[Sequence[int]] | None = None,
    ):
        expected_p = self._init_perron(topology, prod_weights, tie_matrix, symmetric_edge_groups)
        self.log_rates_free = jnp.asarray(log_rates_free, dtype=jnp.float64)
        self.log_total_mass = jnp.asarray(log_total_mass, dtype=jnp.float64).reshape(())
        if self.log_rates_free.shape != (expected_p,):
            raise ValueError(
                f"log_rates_free must have shape ({expected_p},), got {self.log_rates_free.shape}."
            )

    def param_specs(self, prefix: str = "cbm", *, prior_sd: float | None = None) -> tuple[Any, ...]:
        if self.log_rates_free.size == 0:
            return ()
        from .likelihood import ParamSpec

        return (
            ParamSpec(
                f"{prefix}.log_rates_free",
                "identity",
                self.log_rates_free,
                _require_prior_sd(prior_sd),
            ),
        )

    @classmethod
    def from_masses_and_fluxes(
        cls,
        topology: Topology,
        masses: ArrayLike,
        edge_fluxes: ArrayLike,
        *,
        prod_weights: Sequence[float] | ArrayLike | None = None,
        tie_matrix: ArrayLike | None = None,
        symmetric_edge_groups: Sequence[Sequence[int]] | None = None,
    ) -> BalancedChart:
        theta_free, log_m_tot = _invert_perron_chart(
            topology, masses, edge_fluxes, tie_matrix, symmetric_edge_groups
        )
        return cls(
            topology,
            theta_free,
            log_m_tot,
            prod_weights=prod_weights,
            tie_matrix=tie_matrix,
            symmetric_edge_groups=symmetric_edge_groups,
        )


class BalancedCBM(CarbonBoxModel):
    """Unconstrained equilibrium flow-balanced CBM wrapping :class:`BalancedChart`."""

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
        pw = prod_fractions if prod_fractions is not None else prod_weights
        super().__init__(
            topology,
            BalancedChart(
                topology,
                log_rates_free,
                log_total_mass,
                prod_weights=pw,
                tie_matrix=tie_matrix,
                symmetric_edge_groups=symmetric_edge_groups,
            ),
            decay_rate=decay_rate,
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
        theta_free, log_m_tot = _invert_perron_chart(
            topology, masses, edge_fluxes, tie_matrix, symmetric_edge_groups
        )
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


class SubspaceCBM(_PerronCBM):
    """Flow-balanced Perron–Frobenius CBM restricted to a linear log-rate subspace.

    Parameterizes equilibrium-balanced carbon-cycle perturbations along an
    ``m``-dimensional subspace of free log transfer rates:

    .. math::

        \\ln \\mathbf{k}_{\\mathrm{free}}(\\boldsymbol{\\alpha})
        = \\ln \\mathbf{k}_{\\mathrm{free}, 0} + B\\,\\boldsymbol{\\alpha},
        \\qquad \\boldsymbol{\\alpha} \\in \\mathbb{R}^m.

    All reference log-rates, basis columns ``B``, and ``base_log_total_mass`` are
    stored as static PyTree metadata so ``alpha`` is the sole dynamic JAX leaf
    (with zero gauge nullspace). When constructed via :meth:`from_fisher` from a
    prior-whitened :class:`~ticktack.fisher.FisherSpectrumResult`, placing a
    standard normal prior ``alpha ~ N(0, I_m)`` restricts the Gaussian prior of
    ``base_cbm`` to its top ``m`` Schur-marginalized Fisher eigenmodes while
    preserving exact Perron–Frobenius Kirchhoff mass balance everywhere.
    """

    alpha: jax.Array
    base_log_rates_free: tuple[float, ...] = eqx.field(static=True)
    basis_rows: tuple[tuple[float, ...], ...] = eqx.field(static=True)
    base_log_total_mass: float = eqx.field(static=True)

    def __init__(
        self,
        topology: Topology,
        base_log_rates_free: Sequence[float] | ArrayLike,
        basis: ArrayLike,
        base_log_total_mass: float,
        *,
        alpha: ArrayLike | None = None,
        prod_weights: Sequence[float] | ArrayLike | None = None,
        prod_fractions: Sequence[float] | ArrayLike | None = None,
        tie_matrix: ArrayLike | None = None,
        symmetric_edge_groups: Sequence[Sequence[int]] | None = None,
        decay_rate: float = DECAY_CAMBRIDGE,
    ):
        self._init_common(
            topology,
            decay_rate,
            prod_fractions if prod_fractions is not None else prod_weights,
            require_scc=True,
        )
        expected_p = self._init_tie_matrix(tie_matrix, symmetric_edge_groups)

        base_arr = np.asarray(base_log_rates_free, dtype=np.float64).reshape(-1)
        if base_arr.shape != (expected_p,):
            raise ValueError(
                f"base_log_rates_free must have shape ({expected_p},), got {base_arr.shape}."
            )

        basis_arr = np.asarray(basis, dtype=np.float64)
        if basis_arr.ndim != 2 or basis_arr.shape[0] != expected_p:
            raise ValueError(
                f"basis must have shape ({expected_p}, n_modes), got {basis_arr.shape}."
            )
        m_modes = int(basis_arr.shape[1])

        if alpha is None:
            self.alpha = jnp.zeros((m_modes,), dtype=jnp.float64)
        else:
            alpha_arr = jnp.asarray(alpha, dtype=jnp.float64)
            if alpha_arr.shape != (m_modes,):
                raise ValueError(
                    f"alpha must have shape ({m_modes},), got {alpha_arr.shape}."
                )
            self.alpha = alpha_arr

        self.base_log_rates_free = tuple(float(v) for v in base_arr)
        self.basis_rows = tuple(tuple(float(v) for v in row) for row in basis_arr)
        self.base_log_total_mass = float(base_log_total_mass)

    @property
    def n_modes(self) -> int:
        return int(self.alpha.shape[0])

    def log_rates_free(self) -> jax.Array:
        base = jnp.asarray(self.base_log_rates_free, dtype=jnp.float64)
        if self.alpha.shape[0] == 0:
            return base
        basis_mat = jnp.asarray(self.basis_rows, dtype=jnp.float64)
        return base + basis_mat @ self.alpha

    def _free_log_rates(self) -> jax.Array:
        return self.log_rates_free()

    def _log_total_mass_value(self) -> jax.Array:
        return jnp.asarray(self.base_log_total_mass, dtype=jnp.float64)

    def to_balanced_cbm(self) -> BalancedCBM:
        """Convert the current subspace point into an equivalent :class:`BalancedCBM`."""
        return BalancedCBM(
            topology=self.topology,
            log_rates_free=self.log_rates_free(),
            log_total_mass=self.base_log_total_mass,
            prod_weights=self.prod_weights,
            tie_matrix=self.tie_matrix_rows,
            decay_rate=self.decay_rate,
        )

    @classmethod
    def from_fisher(
        cls,
        base_cbm: BalancedCBM,
        spectrum: Any,
        n_modes: int,
    ) -> SubspaceCBM:
        """Construct a ``SubspaceCBM`` spanned by the leading ``n_modes`` Fisher loadings.

        Parameters
        ----------
        base_cbm : BalancedCBM
            Reference flow-balanced carbon-box model at ``alpha = 0``.
        spectrum : FisherSpectrumResult
            Prior-whitened Fisher spectrum computed at ``base_cbm``.
        n_modes : int
            Number of leading eigenmodes ``m`` to retain
            (``1 <= n_modes <= spectrum.parameter_loadings.shape[1]``).
        """
        if not (
            isinstance(base_cbm, BalancedCBM)
            or (isinstance(base_cbm, CarbonBoxModel) and isinstance(base_cbm.chart, BalancedChart))
        ):
            raise TypeError(
                f"SubspaceCBM.from_fisher expects a BalancedCBM or BalancedChart CarbonBoxModel, got {type(base_cbm).__name__}."
            )
        p_free = int(base_cbm.log_rates_free.shape[0])
        loadings = np.asarray(spectrum.parameter_loadings, dtype=np.float64)
        if loadings.ndim != 2 or loadings.shape[0] < p_free:
            raise ValueError(
                f"spectrum.parameter_loadings shape {loadings.shape} is incompatible with "
                f"base_cbm.log_rates_free length {p_free}."
            )
        max_modes = int(loadings.shape[1])
        m = int(n_modes)
        if not (1 <= m <= max_modes):
            raise ValueError(
                f"n_modes must satisfy 1 <= n_modes <= {max_modes}, got {m}."
            )
        return cls(
            topology=base_cbm.topology,
            base_log_rates_free=np.asarray(base_cbm.log_rates_free, dtype=np.float64),
            basis=loadings[:p_free, :m],
            base_log_total_mass=float(base_cbm.log_total_mass),
            alpha=jnp.zeros((m,), dtype=jnp.float64),
            prod_weights=base_cbm.prod_weights,
            tie_matrix=base_cbm.tie_matrix_rows,
            decay_rate=base_cbm.decay_rate,
        )


FisherSubspaceCBM = SubspaceCBM


class ReversibleChart(CBMChart):
    """Detailed-balance coordinate chart with symmetric undirected fluxes ``F_{ij} = F_{ji}``."""

    is_detailed_balance: ClassVar[bool] = True
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
    ):
        topology.check_strongly_connected()
        self.prod_weights = _resolve_prod_weights(topology, prod_weights)

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

    def reservoir_masses(self, topology: Topology) -> jax.Array:
        return jnp.exp(self.log_masses)

    def edge_fluxes(self) -> jax.Array:
        sym_f = jnp.exp(self.log_sym_fluxes)
        f = jnp.zeros((2 * len(self.pair_forward_edge),), dtype=jnp.float64)
        if len(self.pair_forward_edge) == 0:
            return f
        f = f.at[np.asarray(self.pair_forward_edge, dtype=np.int64)].set(sym_f)
        return f.at[np.asarray(self.pair_reverse_edge, dtype=np.int64)].set(sym_f)

    def edge_rates(self, topology: Topology) -> jax.Array:
        if topology.n_edges == 0:
            return jnp.zeros((0,), dtype=jnp.float64)
        src = np.asarray(topology.edges_src, dtype=np.int64)
        return self.edge_fluxes() / self.reservoir_masses(topology)[src]

    def param_specs(
        self, prefix: str = "cbm", *, prior_sd: float | None = None, include_masses: bool = True
    ) -> tuple[Any, ...]:
        from .likelihood import ParamSpec

        specs: list[Any] = []
        if include_masses and self.log_masses.size > 0:
            specs.append(
                ParamSpec(
                    f"{prefix}.log_masses", "identity", self.log_masses, _require_prior_sd(prior_sd)
                )
            )
        if self.log_sym_fluxes.size > 0:
            specs.append(
                ParamSpec(
                    f"{prefix}.log_sym_fluxes",
                    "identity",
                    self.log_sym_fluxes,
                    _require_prior_sd(prior_sd),
                )
            )
        return tuple(specs)

    @classmethod
    def from_masses_and_fluxes(
        cls,
        topology: Topology,
        masses: ArrayLike,
        edge_fluxes: ArrayLike,
        *,
        prod_weights: Sequence[float] | ArrayLike | None = None,
    ) -> ReversibleChart:
        return ReversibleCBM.from_masses_and_fluxes(
            topology, masses, edge_fluxes, prod_weights=prod_weights
        ).chart  # type: ignore[return-value]


class ReversibleCBM(CarbonBoxModel):
    """Detailed-balance CBM wrapping :class:`ReversibleChart`."""

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
        pw = prod_fractions if prod_fractions is not None else prod_weights
        super().__init__(
            topology,
            ReversibleChart(topology, log_masses, log_sym_fluxes, prod_weights=pw),
            decay_rate=decay_rate,
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
        rev_indices: list[int] = []
        visited: set[tuple[int, int]] = set()
        for idx, (s, d) in enumerate(zip(topology.edges_src, topology.edges_dst)):
            if (s, d) in visited:
                continue
            if (d, s) not in edge_lookup:
                raise ValueError(
                    f"ReversibleCBM requires symmetric edge pairs; missing reverse edge ({d} -> {s})."
                )
            ridx = edge_lookup[(d, s)]
            fwd_indices.append(idx)
            rev_indices.append(ridx)
            visited.add((s, d))
            visited.add((d, s))

        if not isinstance(f_arr, jax.core.Tracer) and fwd_indices:
            f_np = np.asarray(f_arr, dtype=np.float64)
            if not np.allclose(f_np[fwd_indices], f_np[rev_indices], rtol=1e-6, atol=1e-9):
                raise ValueError(
                    "ReversibleChart requires equal forward and reverse fluxes F_{ij} == F_{ji}."
                )

        sym_f = f_arr[np.asarray(fwd_indices, dtype=np.int64)]
        return cls(
            topology=topology,
            log_masses=jnp.log(m_arr),
            log_sym_fluxes=jnp.log(sym_f),
            prod_weights=prod_weights,
            prod_fractions=prod_fractions,
            decay_rate=decay_rate,
        )


class OpenChart(_PerronChart):
    """Perron–Frobenius coordinate chart augmented with open reservoir ``log_sink_rates``."""

    log_rates_free: jax.Array
    log_total_mass: jax.Array
    log_sink_rates: jax.Array
    open_box_indices: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        topology: Topology,
        log_rates_free: ArrayLike,
        log_total_mass: ArrayLike,
        log_sink_rates: ArrayLike,
        *,
        open_box_indices: Sequence[int],
        prod_weights: Sequence[float] | ArrayLike | None = None,
        tie_matrix: ArrayLike | None = None,
        symmetric_edge_groups: Sequence[Sequence[int]] | None = None,
    ):
        open_idx = tuple(int(i) for i in open_box_indices)
        if not open_idx or any(i < 0 or i >= topology.n_boxes for i in open_idx):
            raise ValueError(
                f"open_box_indices must be non-empty and within [0, {topology.n_boxes}), got {open_idx}."
            )
        self.open_box_indices = open_idx

        expected_p = self._init_perron(topology, prod_weights, tie_matrix, symmetric_edge_groups)
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

    def sink_rates(self, topology: Topology, decay_rate: float = DECAY_CAMBRIDGE) -> jax.Array:
        return (
            jnp.zeros((topology.n_boxes,), dtype=jnp.float64)
            .at[np.asarray(self.open_box_indices, dtype=np.int64)]
            .set(jnp.exp(self.log_sink_rates))
        )

    def param_specs(
        self, prefix: str = "cbm", *, prior_sd: float | None = None, sink_sd: float | None = None
    ) -> tuple[Any, ...]:
        from .likelihood import ParamSpec

        specs: list[Any] = []
        if self.log_rates_free.size > 0:
            specs.append(
                ParamSpec(
                    f"{prefix}.log_rates_free",
                    "identity",
                    self.log_rates_free,
                    _require_prior_sd(prior_sd),
                )
            )
        if self.log_sink_rates.size > 0:
            specs.append(
                ParamSpec(
                    f"{prefix}.log_sink_rates",
                    "identity",
                    self.log_sink_rates,
                    _require_prior_sd(prior_sd if sink_sd is None else sink_sd),
                )
            )
        return tuple(specs)

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
        tie_matrix: ArrayLike | None = None,
        symmetric_edge_groups: Sequence[Sequence[int]] | None = None,
    ) -> OpenChart:
        return OpenCBM.from_masses_and_fluxes(
            topology,
            masses,
            edge_fluxes,
            sink_rates,
            open_box_indices=open_box_indices,
            prod_weights=prod_weights,
            tie_matrix=tie_matrix,
            symmetric_edge_groups=symmetric_edge_groups,
        ).chart  # type: ignore[return-value]


class OpenCBM(CarbonBoxModel):
    """Perron–Frobenius CBM with explicit open deep-reservoir loss wrapping :class:`OpenChart`."""

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
        pw = prod_fractions if prod_fractions is not None else prod_weights
        super().__init__(
            topology,
            OpenChart(
                topology,
                log_rates_free,
                log_total_mass,
                log_sink_rates,
                open_box_indices=open_box_indices,
                prod_weights=pw,
                tie_matrix=tie_matrix,
                symmetric_edge_groups=symmetric_edge_groups,
            ),
            decay_rate=decay_rate,
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
        s_arr = np.asarray(sink_rates, dtype=np.float64)
        if open_box_indices is None:
            open_idx = tuple(int(i) for i in np.flatnonzero(s_arr > 0.0))
        else:
            open_idx = tuple(int(i) for i in open_box_indices)
        if not open_idx:
            raise ValueError(
                "OpenCBM requires at least one box with positive sink_rate or explicit open_box_indices."
            )

        theta_free, log_m_tot = _invert_perron_chart(
            topology, masses, edge_fluxes, tie_matrix, symmetric_edge_groups
        )
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


def _calibrate_effective_deep_sink_jax(
    n_boxes: int,
    log_k_strat_to_trop: jax.Array,
    log_k_trop_to_strat: jax.Array,
    log_k_trop_to_sinks: jax.Array,
    log_sink_taus: jax.Array,
    *,
    eta_strat: float,
    trop_mass: float,
    target_r_ref: float,
    decay_rate: float,
    q_std: float = DEFAULT_Q_STD,
    alpha_unit: float = ALPHA_ATOMS_CM2_S_TO_KG_YR,
) -> jax.Array:
    """Smooth ``C^infty`` JAX calibration of ``k_deep >= 0`` to match ``target_r_ref``."""
    lam = jnp.asarray(decay_rate, dtype=jnp.float64)
    k01 = jnp.exp(log_k_strat_to_trop)
    k10 = jnp.exp(log_k_trop_to_strat)

    rhs_top = (1.0 - eta_strat) + eta_strat * k01 / (lam + k01)
    target_g1 = (trop_mass * target_r_ref) / (alpha_unit * q_std)
    target_denom = rhs_top / target_g1
    strat_term = lam * k10 / (lam + k01)
    eps_lo = 1e-6 * lam

    if n_boxes == 2:
        raw_k = target_denom - lam - strat_term
        return eps_lo * jax.nn.softplus(raw_k / eps_lo)

    k_up_all = jnp.exp(log_k_trop_to_sinks)
    inv_taus = jnp.exp(-log_sink_taus)
    other_sinks = (
        jnp.sum(k_up_all[:-1] * lam / (lam + inv_taus[:-1]))
        if n_boxes > 3
        else jnp.asarray(0.0, dtype=jnp.float64)
    )
    needed_last = target_denom - lam - strat_term - other_sinks
    k_up = k_up_all[-1]
    inv_tau_last = inv_taus[-1]

    # Smooth C^infty upper barrier keeping eff_needed < k_up without any jump:
    # den = margin * k_up + eps_hi * softplus((max_needed - needed_last) / eps_hi)
    # equals k_up - needed_last to machine precision in the interior (needed_last <= 0.95 * k_up)
    # and smoothly approaches margin * k_up > 0 as needed_last -> +inf.
    margin = 1e-3
    max_needed = (1.0 - margin) * k_up
    eps_hi = margin * k_up
    den = margin * k_up + eps_hi * jax.nn.softplus((max_needed - needed_last) / eps_hi)
    eff_needed = k_up - den

    raw_k = (eff_needed * (lam + inv_tau_last) - k_up * lam) / den
    return eps_lo * jax.nn.softplus(raw_k / eps_lo)


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
       zero-Fisher global mass-scale gauge) and dynamically calibrated open loss
       rate ``deep_sink_rate >= 0`` on the slowest reservoir (``Sink_{K-2}`` for
       ``K >= 3``, or effective atmosphere-to-ocean sequestration on ``Troposphere``
       for ``K == 2``) so pre-industrial ``R_ref`` matches ``Guttler15``
       (``~1.3195 kg/GtC``) smoothly across all PyTree updates and autodiff passes.
    3. Strict sink residence-time ordering ``ln tau_1 < ln tau_2 < ... < ln tau_{K-2}``
       via ``softplus`` increments ``ln tau_j = ln tau_{j-1} + softplus(delta_j)``,
       eliminating all ``(K - 2)!`` discrete sink-permutation modes.

    The dynamic JAX array leaves have total dimension ``1 + 1 + (K - 2) + (K - 2) = 2K - 2``
    when ``deep_sink_rate`` is calibrated dynamically (default).
    """

    log_k_strat_to_trop: jax.Array
    log_k_trop_to_strat: jax.Array
    log_k_trop_to_sinks: jax.Array
    sink_tau_params: jax.Array
    explicit_deep_sink_rate: jax.Array | None
    K: int = eqx.field(static=True)
    eta_strat: float = eqx.field(static=True)
    trop_mass: float = eqx.field(static=True)
    calibrate_deep_sink: bool = eqx.field(static=True)
    target_r_ref: float = eqx.field(static=True)
    q_std_cal: float = eqx.field(static=True)
    alpha_unit_cal: float = eqx.field(static=True)

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
        deep_sink_rate: ArrayLike | None = None,
        target_r_ref: float = GUTTLER15_R_REF_KG_PER_GTC,
        decay_rate: float = DECAY_CAMBRIDGE,
        q_std: float = DEFAULT_Q_STD,
        alpha_unit: float = ALPHA_ATOMS_CM2_S_TO_KG_YR,
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
        self.target_r_ref = float(target_r_ref)
        self.q_std_cal = float(q_std)
        self.alpha_unit_cal = float(alpha_unit)

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
            self.calibrate_deep_sink = True
            self.explicit_deep_sink_rate = None
        else:
            self.calibrate_deep_sink = False
            self.explicit_deep_sink_rate = jnp.asarray(
                deep_sink_rate, dtype=jnp.float64
            ).reshape(())

    @property
    def n_parameters(self) -> int:
        base = 2 * self.K - 2
        return base if self.explicit_deep_sink_rate is None else base + 1

    @property
    def deep_sink_rate(self) -> jax.Array:
        """Active open loss rate ``k_deep`` (``yr^-1``) on the deepest box."""
        return self.sink_rates()[-1]

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
        if not self.calibrate_deep_sink:
            assert self.explicit_deep_sink_rate is not None
            return out.at[-1].set(
                jnp.maximum(jnp.asarray(self.explicit_deep_sink_rate, dtype=jnp.float64), 0.0)
            )
        k_deep = _calibrate_effective_deep_sink_jax(
            self.K,
            self.log_k_strat_to_trop,
            self.log_k_trop_to_strat,
            self.log_k_trop_to_sinks,
            self.log_sink_timescales(),
            eta_strat=self.eta_strat,
            trop_mass=self.trop_mass,
            target_r_ref=self.target_r_ref,
            decay_rate=self.decay_rate,
            q_std=self.q_std_cal,
            alpha_unit=self.alpha_unit_cal,
        )
        return out.at[-1].set(k_deep)

    def production_vector(self) -> jax.Array:
        out = jnp.zeros((self.n_boxes,), dtype=jnp.float64)
        out = out.at[0].set(self.eta_strat)
        out = out.at[1].set(1.0 - self.eta_strat)
        return out
