"""Literature carbon-box model (CBM) presets and HDF5 serialization helpers.

Provides pure-Python/JAX :class:`PresetSpec` definitions and constructors for
the four standard radiocarbon cycle topologies:

* ``"Guttler15"`` — Güttler et al. (2015) 11-box global carbon cycle.
* ``"Brehm21"`` — Brehm et al. (2021) 22-box two-hemisphere carbon cycle.
* ``"Buntgen18"`` — Büntgen et al. (2018) 22-box two-hemisphere carbon cycle
  (with the zero-flux placeholder edge ``SwN -> SwS = 0`` filtered out).
* ``"Miyake17"`` — Miyake et al. (2017) 4-box carbon cycle, with both v1.x
  defects corrected by default (``legacy_v1_units=False``):
  1. Removes the erroneous ``14.003242 / 12`` transfer-rate inflation.
  2. Calibrates an open deep-ocean loss rate ``k_sink`` on ``Marine surface``
     so the steady-state tropospheric specific activity ``R_ref`` matches
     pre-industrial ``Guttler15`` (~1.3195 kg/GtC), preventing the 16x
     shallow-box steady-state inventory suppression. Passing
     ``legacy_v1_units=True`` reproduces exact ``ticktack 1.x`` numbers for
     regression comparisons.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import h5py
import jax
import jax.numpy as jnp
import numpy as np

# Local fallback physical constants so `presets.py` works standalone even
# before `constants.py` is merged on parallel branches.
try:
    from . import constants as _const_mod

    _DEFAULT_ALPHA_UNIT = _const_mod.ALPHA_ATOMS_CM2_S_TO_KG_YR
    _ALPHA_LEGACY_V1 = _const_mod.ALPHA_LEGACY_V1
    _DECAY_CAMBRIDGE = _const_mod.DECAY_CAMBRIDGE
    _DECAY_KUTSCHERA = _const_mod.DECAY_KUTSCHERA
    _DEFAULT_Q_STD = _const_mod.DEFAULT_Q_STD
except ImportError:
    _DECAY_CAMBRIDGE = math.log(2.0) / 5730.0
    _DECAY_KUTSCHERA = math.log(2.0) / 5700.0
    _ALPHA_LEGACY_V1 = 14.003242 / 6.022 * 5.11 * 31536.0 / 1.0e5
    # CODATA 2022: m(14C) = 14.003241988 u, N_A = 6.02214076e23 mol^-1,
    # Earth mean radius R_E = 6.3710088e8 cm => 4*pi*R_E^2 = 5.1006562e18 cm^2,
    # Julian year = 31557600 s. (Cancels identically in R_ref-normalized d14c.)
    _EARTH_AREA_CM2 = 4.0 * math.pi * (6.3710088e8) ** 2
    _DEFAULT_ALPHA_UNIT = (
        14.003241988e-3 / 6.02214076e23 * _EARTH_AREA_CM2 * 31557600.0
    )
    _DEFAULT_Q_STD = 1.76

LEGACY_V1_C14_OVER_C12_RATIO: float = 14.003242 / 12.0

PresetName = Literal["Guttler15", "Brehm21", "Buntgen18", "Miyake17"]
Miyake17Closure = Literal["open_ocean_sink", "closed_shallow"]
PRESET_NAMES: tuple[PresetName, ...] = (
    "Guttler15",
    "Brehm21",
    "Buntgen18",
    "Miyake17",
)


@dataclasses.dataclass(frozen=True)
class PresetSpec:
    """Immutable specification of a carbon-box model preset.

    Attributes
    ----------
    name : str
        Canonical preset identifier.
    box_names : tuple[str, ...]
        Reservoir names in canonical index order ``0 .. n - 1``.
    hemispheres : tuple[str, ...]
        Per-box hemisphere tag (``"north"``, ``"south"``, or ``"None"`` for
        single-hemisphere global models).
    masses : tuple[float, ...]
        Pre-industrial steady-state 12C inventories ``M_0`` in GtC.
    edges : tuple[tuple[int, int, float], ...]
        Directed positive 12C fluxes ``(src_idx, dst_idx, flux_gtc_yr)`` with
        ``flux_gtc_yr > 0``. Zero-flux placeholders are excluded.
    prod_coefficients : tuple[float, ...]
        Normalized cosmic-ray 14C allocation vector ``b`` summing to ``1.0``.
    sink_rates : tuple[float, ...]
        Per-box open deep-reservoir loss rates ``k_sink`` in ``yr^-1`` (all
        zeros for closed global models).
    symmetric_edge_groups : tuple[tuple[int, ...], ...] | None
        Optional groups of edge indices (into ``edges``) sharing equal
        first-order transfer rates ``k_e = F_{ij} / M_i`` (e.g. N/S homologous
        stratospheric, atmospheric, and biological rates in ``Brehm21`` and
        ``Buntgen18``).
    """

    name: str
    box_names: tuple[str, ...]
    hemispheres: tuple[str, ...]
    masses: tuple[float, ...]
    edges: tuple[tuple[int, int, float], ...]
    prod_coefficients: tuple[float, ...]
    sink_rates: tuple[float, ...]
    symmetric_edge_groups: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        n = len(self.box_names)
        if len(self.hemispheres) != n:
            raise ValueError("hemispheres length must match box_names.")
        if len(self.masses) != n:
            raise ValueError("masses length must match box_names.")
        if len(self.prod_coefficients) != n:
            raise ValueError("prod_coefficients length must match box_names.")
        if len(self.sink_rates) != n:
            raise ValueError("sink_rates length must match box_names.")
        if any(m <= 0.0 for m in self.masses):
            raise ValueError("All reservoir masses must be strictly positive.")
        if any(s < 0.0 for s in self.sink_rates):
            raise ValueError("All sink_rates must be non-negative.")
        for idx, (src, dst, flux) in enumerate(self.edges):
            if not (0 <= src < n and 0 <= dst < n) or src == dst:
                raise ValueError(f"Invalid edge endpoints at index {idx}: {(src, dst)}.")
            if flux <= 0.0:
                raise ValueError(
                    f"Non-positive edge flux at index {idx}: {flux}. "
                    "Zero-flux placeholder edges must be filtered out."
                )

    @property
    def n_boxes(self) -> int:
        """Number of carbon reservoirs ``n``."""
        return len(self.box_names)

    @property
    def n_edges(self) -> int:
        """Number of strictly positive directed edges ``|E|``."""
        return len(self.edges)

    @property
    def trop_box_indices(self) -> tuple[int, ...]:
        """Indices of tropospheric boxes used to define global ``R_ref``."""
        indices = tuple(
            i for i, bname in enumerate(self.box_names) if bname.lower() == "troposphere"
        )
        if not indices:
            raise ValueError(f"Preset {self.name!r} has no Troposphere box.")
        return indices

    @property
    def prod_box_indices(self) -> tuple[int, ...]:
        """Indices of boxes receiving non-zero direct cosmic-ray production."""
        return tuple(i for i, c in enumerate(self.prod_coefficients) if c > 0.0)

    def box_index(self, box_name: str = "Troposphere", hemisphere: str | None = None) -> int:
        """Look up the integer index of a reservoir by name and optional hemisphere."""
        target = box_name.lower()
        for i, (bname, hem) in enumerate(zip(self.box_names, self.hemispheres)):
            if bname.lower() == target and (
                hemisphere is None or hem == "None" or hem.lower() == hemisphere.lower()
            ):
                return i
        raise KeyError(
            f"No box matching name={box_name!r}, hemisphere={hemisphere!r} in {self.name!r}."
        )

    def flux_matrix(self) -> jax.Array:
        """Return dense ``(n, n)`` 12C flux matrix ``F`` in GtC/yr (``F[i, j]`` is ``i -> j``)."""
        n = self.n_boxes
        mat = np.zeros((n, n), dtype=np.float64)
        for src, dst, flux in self.edges:
            mat[src, dst] += flux
        return jnp.asarray(mat, dtype=jnp.float64)

    def kirchhoff_imbalance(self) -> jax.Array:
        """Return per-box net outgoing minus incoming 12C flux ``sum_j F_ij - sum_j F_ji``."""
        f_mat = self.flux_matrix()
        return jnp.sum(f_mat, axis=1) - jnp.sum(f_mat, axis=0)

    def rate_matrix_K(self) -> jax.Array:
        """Return ``(n, n)`` Metzler transfer-rate matrix ``K`` in ``yr^-1``.

        Follows column-stochastic generator convention ``dy/dt = A y + ...`` with
        ``K[j, i] = F[i, j] / M_0[i]`` for ``i != j`` and ``K[i, i] = -sum_{j != i} K[j, i]``.
        """
        masses = jnp.asarray(self.masses, dtype=jnp.float64)
        f_mat = self.flux_matrix()
        rates_out = f_mat / masses[:, None]
        return rates_out.T - jnp.diag(jnp.sum(rates_out, axis=1))

    def matrix_A(self, decay_rate: float = _DECAY_CAMBRIDGE) -> jax.Array:
        """Return full LTI system matrix ``A = K - diag(k_sink) - decay_rate * I``."""
        n = self.n_boxes
        k_mat = self.rate_matrix_K()
        sinks = jnp.asarray(self.sink_rates, dtype=jnp.float64)
        return k_mat - jnp.diag(sinks) - float(decay_rate) * jnp.eye(n, dtype=jnp.float64)

    def steady_state_c14(
        self,
        q_std: float = _DEFAULT_Q_STD,
        alpha_unit: float = _DEFAULT_ALPHA_UNIT,
        decay_rate: float = _DECAY_CAMBRIDGE,
    ) -> jax.Array:
        """Return steady-state 14C reservoir masses ``y_ss = alpha_unit * q_std * (-A)^-1 b``."""
        a_mat = self.matrix_A(decay_rate=decay_rate)
        b_vec = jnp.asarray(self.prod_coefficients, dtype=jnp.float64)
        return jnp.linalg.solve(-a_mat, float(alpha_unit) * float(q_std) * b_vec)

    def reference_specific_activity(
        self,
        q_std: float = _DEFAULT_Q_STD,
        alpha_unit: float = _DEFAULT_ALPHA_UNIT,
        decay_rate: float = _DECAY_CAMBRIDGE,
    ) -> jax.Array:
        """Return global tropospheric reference specific activity ``R_ref`` in kg/GtC."""
        y_ss = self.steady_state_c14(
            q_std=q_std, alpha_unit=alpha_unit, decay_rate=decay_rate
        )
        masses = jnp.asarray(self.masses, dtype=jnp.float64)
        t_idx = jnp.asarray(self.trop_box_indices, dtype=jnp.int32)
        return jnp.sum(y_ss[t_idx]) / jnp.sum(masses[t_idx])

    def edge_tie_matrix(self, tie_symmetric_edges: bool = True) -> jax.Array:
        """Build ``(|E|, p_free)`` binary symmetry-tying matrix ``M_tie``."""
        m_edges = self.n_edges
        if not tie_symmetric_edges or not self.symmetric_edge_groups:
            return jnp.eye(m_edges, dtype=jnp.float64)

        grouped: dict[int, int] = {}
        cols: list[tuple[int, ...]] = []
        used: set[int] = set()
        for grp in self.symmetric_edge_groups:
            for e_idx in grp:
                if e_idx in used:
                    raise ValueError(f"Duplicate edge index {e_idx} in symmetric_edge_groups.")
                if not (0 <= e_idx < m_edges):
                    raise ValueError(f"Out-of-bounds edge index {e_idx} in symmetric_edge_groups.")
                used.add(e_idx)
        # Preserve canonical edge order for columns by first appearance.
        group_by_member = {e_idx: grp for grp in self.symmetric_edge_groups for e_idx in grp}
        seen_groups: set[tuple[int, ...]] = set()
        for e_idx in range(m_edges):
            if e_idx in group_by_member:
                grp = group_by_member[e_idx]
                if grp not in seen_groups:
                    col_idx = len(cols)
                    cols.append(grp)
                    seen_groups.add(grp)
                    for member in grp:
                        grouped[member] = col_idx
            else:
                col_idx = len(cols)
                cols.append((e_idx,))
                grouped[e_idx] = col_idx

        mat = np.zeros((m_edges, len(cols)), dtype=np.float64)
        for e_idx, col_idx in grouped.items():
            mat[e_idx, col_idx] = 1.0
        return jnp.asarray(mat, dtype=jnp.float64)


def _find_symmetric_rate_groups(
    masses: tuple[float, ...],
    hemispheres: tuple[str, ...],
    box_names: tuple[str, ...],
    edges: tuple[tuple[int, int, float], ...],
    *,
    include_interhemispheric_pairs: bool = True,
    rtol: float = 1e-12,
) -> tuple[tuple[int, ...], ...]:
    """Identify N/S homologous and symmetric inter-hemispheric edge pairs with equal ``k_e``."""
    edge_lookup: dict[tuple[int, int], int] = {
        (src, dst): idx for idx, (src, dst, _) in enumerate(edges)
    }
    box_by_hem_name: dict[tuple[str, str], int] = {
        (hem, bname.lower().replace("&", "and")): idx
        for idx, (bname, hem) in enumerate(zip(box_names, hemispheres))
    }

    groups: list[tuple[int, int]] = []
    visited: set[int] = set()

    for idx, (src, dst, flux) in enumerate(edges):
        if idx in visited:
            continue
        src_hem, dst_hem = hemispheres[src], hemispheres[dst]
        rate = flux / masses[src]

        # 1. Intra-hemispheric South <-> North homologous transitions (stratospheric,
        # biological, soil, peat, and marine export/sedimentation rates).
        if src_hem == "south" and dst_hem == "south":
            src_key = ("north", box_names[src].lower().replace("&", "and"))
            dst_key = ("north", box_names[dst].lower().replace("&", "and"))
            if src_key in box_by_hem_name and dst_key in box_by_hem_name:
                n_src = box_by_hem_name[src_key]
                n_dst = box_by_hem_name[dst_key]
                pair_idx = edge_lookup.get((n_src, n_dst))
                if pair_idx is not None and pair_idx not in visited:
                    n_flux = edges[pair_idx][2]
                    n_rate = n_flux / masses[n_src]
                    if math.isclose(rate, n_rate, rel_tol=rtol, abs_tol=0.0):
                        groups.append((idx, pair_idx))
                        visited.add(idx)
                        visited.add(pair_idx)
                        continue

        # 2. Symmetric inter-hemispheric atmospheric exchange (SpS <-> SpN, TpS <-> TpN).
        if include_interhemispheric_pairs and src_hem != dst_hem:
            rev_idx = edge_lookup.get((dst, src))
            if rev_idx is not None and rev_idx not in visited:
                rev_flux = edges[rev_idx][2]
                rev_rate = rev_flux / masses[dst]
                if math.isclose(rate, rev_rate, rel_tol=rtol, abs_tol=0.0):
                    groups.append((idx, rev_idx))
                    visited.add(idx)
                    visited.add(rev_idx)

    return tuple(groups)


def _build_guttler15_spec() -> PresetSpec:
    box_names = (
        "Stratosphere",
        "Troposphere",
        "Surface Water",
        "Surface Biota",
        "Intermediate & Deep water",
        "Short-lived biota",
        "Long-lived Biota",
        "Litter",
        "Soil",
        "Peat",
        "Sedimentary Sink",
    )
    hemispheres = ("None",) * len(box_names)
    masses = (
        88.5,
        501.5,
        900.0,
        3.0,
        37800.0,
        110.0,
        450.0,
        300.0,
        1350.0,
        500.0,
        378000.0,
    )
    prod_coefficients = (0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    edges = (
        (10, 1, 0.7),    # f1: SS -> Tp
        (0, 1, 45.0),    # f2: Sp -> Tp
        (1, 0, 45.0),    # f3: Tp -> Sp
        (1, 2, 60.5),    # f4: Tp -> Sw
        (2, 1, 61.0),    # f5: Sw -> Tp
        (2, 3, 40.0),    # f6: Sw -> Sb
        (3, 2, 36.0),    # f7: Sb -> Sw
        (3, 4, 4.0),     # f8: Sb -> Idw
        (4, 2, 42.0),    # f9: Idw -> Sw
        (2, 4, 38.2),    # f10: Sw -> Idw
        (2, 10, 0.3),    # f11: Sw -> SS
        (4, 10, 0.2),    # f12: Idw -> SS
        (1, 5, 115.0),   # f13: Tp -> Slb
        (5, 1, 60.0),    # f14: Slb -> Tp
        (5, 6, 15.0),    # f15: Slb -> Llb
        (6, 7, 15.0),    # f16: Llb -> L
        (5, 7, 40.0),    # f17: Slb -> L
        (7, 8, 3.0),     # f18: L -> S
        (7, 9, 1.0),     # f19: L -> P
        (7, 2, 1.0),     # f20: L -> Sw
        (7, 1, 50.0),    # f21: L -> Tp
        (8, 1, 3.0),     # f22: S -> Tp
        (9, 1, 0.8),     # f23: P -> Tp
        (9, 10, 0.2),    # f24: P -> SS
    )
    return PresetSpec(
        name="Guttler15",
        box_names=box_names,
        hemispheres=hemispheres,
        masses=masses,
        edges=edges,
        prod_coefficients=prod_coefficients,
        sink_rates=(0.0,) * len(box_names),
        symmetric_edge_groups=None,
    )


_TWO_HEM_BOX_NAMES: tuple[str, ...] = (
    "Stratosphere",
    "Troposphere",
    "Surface Water",
    "Surface Biota",
    "Intermediate and Deep water",
    "Short-lived biota",
    "Long-lived Biota",
    "Litter",
    "Soil",
    "Peat",
    "Sedimentary Sink",
    "Stratosphere",
    "Troposphere",
    "Surface Water",
    "Surface Biota",
    "Intermediate & Deep water",
    "Short-lived biota",
    "Long-lived Biota",
    "Litter",
    "Soil",
    "Peat",
    "Sedimentary Sink",
)

_TWO_HEM_TAGS: tuple[str, ...] = ("south",) * 11 + ("north",) * 11

_TWO_HEM_MASSES: tuple[float, ...] = (
    44.5,
    250.5,
    540.0,
    1.8,
    22680.0,
    33.0,
    135.0,
    90.0,
    405.0,
    150.0,
    189000.0,
    44.5,
    250.5,
    360.0,
    1.2,
    15120.0,
    77.0,
    315.0,
    210.0,
    945.0,
    350.0,
    189000.0,
)

_TWO_HEM_PROD: tuple[float, ...] = (
    0.35,
    0.15,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.35,
    0.15,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)


def _build_brehm21_spec() -> PresetSpec:
    edges = (
        (0, 1, 22.5),     # f1: SpS -> TpS
        (1, 0, 22.5),     # f2: TpS -> SpS
        (1, 5, 34.5),     # f3: TpS -> SlbS
        (5, 1, 18.0),     # f4: SlbS -> TpS
        (1, 2, 36.6),     # f5: TpS -> SwS
        (2, 1, 36.6),     # f6: SwS -> TpS
        (5, 6, 4.5),      # f7: SlbS -> LlbS
        (6, 7, 4.5),      # f8: LlbS -> LS
        (5, 7, 12.0),     # f9: SlbS -> LS
        (7, 1, 15.0),     # f10: LS -> TpS
        (7, 2, 0.3),      # f11: LS -> SwS
        (7, 9, 0.3),      # f12: LS -> PS
        (7, 8, 0.9),      # f13: LS -> SoS
        (9, 1, 0.24),     # f14: PS -> TpS
        (9, 10, 0.06),    # f15: PS -> SSS
        (8, 1, 0.9),      # f16: SoS -> TpS
        (2, 10, 0.18),    # f17: SwS -> SSS
        (4, 10, 0.12),    # f18: IdwS -> SSS
        (2, 3, 22.6),     # f19: SwS -> SbS
        (2, 4, 37.92),    # f20: SwS -> IdwS
        (2, 13, 13.0),    # f21: SwS -> SwN
        (3, 2, 20.2),     # f22: SbS -> SwS
        (3, 4, 2.4),      # f23: SbS -> IdwS
        (4, 2, 43.2),     # f24: IdwS -> SwS
        (4, 15, 15.12),   # f25: IdwS -> IdwN
        (10, 1, 0.36),    # f26: SSS -> TpS
        (0, 11, 44.1),    # f27: SpS -> SpN
        (1, 12, 71.0),    # f28: TpS -> TpN
        (11, 0, 44.1),    # f29: SpN -> SpS
        (12, 1, 71.0),    # f30: TpN -> TpS
        (11, 12, 22.5),   # f31: SpN -> TpN
        (12, 11, 22.5),   # f32: TpN -> SpN
        (12, 13, 24.4),   # f33: TpN -> SwN
        (12, 16, 80.5),   # f34: TpN -> SlbN
        (13, 12, 24.89),  # f35: SwN -> TpN
        (13, 2, 10.0),    # f36: SwN -> SwS
        (13, 14, 16.0),   # f37: SwN -> SbN
        (13, 15, 28.29),  # f38: SwN -> IdwN
        (16, 12, 42.0),   # f39: SlbN -> TpN
        (16, 17, 10.5),   # f40: SlbN -> LlbN
        (16, 18, 28.0),   # f41: SlbN -> LN
        (17, 18, 10.5),   # f42: LlbN -> LN
        (14, 13, 14.4),   # f43: SbN -> SwN
        (14, 15, 1.6),    # f44: SbN -> IdwN
        (13, 21, 0.12),   # f45: SwN -> SSN
        (15, 13, 26.8),   # f46: IdwN -> SwN
        (15, 21, 0.09),   # f47: IdwN -> SSN
        (15, 4, 18.12),   # f48: IdwN -> IdwS
        (18, 13, 0.7),    # f49: LN -> SwN
        (18, 19, 2.1),    # f50: LN -> SoN
        (18, 12, 35.0),   # f51: LN -> TpN
        (18, 20, 0.7),    # f52: LN -> PN
        (19, 12, 2.1),    # f53: SoN -> TpN
        (20, 12, 0.56),   # f54: PN -> TpN
        (20, 21, 0.14),   # f55: PN -> SSN
        (21, 12, 0.35),   # f56: SSN -> TpN
    )
    sym_groups = _find_symmetric_rate_groups(
        _TWO_HEM_MASSES, _TWO_HEM_TAGS, _TWO_HEM_BOX_NAMES, edges
    )
    return PresetSpec(
        name="Brehm21",
        box_names=_TWO_HEM_BOX_NAMES,
        hemispheres=_TWO_HEM_TAGS,
        masses=_TWO_HEM_MASSES,
        edges=edges,
        prod_coefficients=_TWO_HEM_PROD,
        sink_rates=(0.0,) * len(_TWO_HEM_BOX_NAMES),
        symmetric_edge_groups=sym_groups,
    )


def _build_buntgen18_spec() -> PresetSpec:
    raw_edges = (
        (0, 1, 23.8),     # f1: SpS -> TpS
        (1, 0, 23.8),     # f2: TpS -> SpS
        (1, 5, 35.36),    # f3: TpS -> SlbS
        (5, 1, 18.86),    # f4: SlbS -> TpS
        (1, 2, 38.25),    # f5: TpS -> SwS
        (2, 1, 38.25),    # f6: SwS -> TpS
        (5, 6, 4.5),      # f7: SlbS -> LlbS
        (6, 7, 4.5),      # f8: LlbS -> LS
        (5, 7, 12.0),     # f9: SlbS -> LS
        (7, 1, 15.0),     # f10: LS -> TpS
        (7, 2, 0.3),      # f11: LS -> SwS
        (7, 9, 0.3),      # f12: LS -> PS
        (7, 8, 0.9),      # f13: LS -> SoS
        (9, 1, 0.24),     # f14: PS -> TpS
        (9, 10, 0.06),    # f15: PS -> SSS
        (8, 1, 0.9),      # f16: SoS -> TpS
        (2, 10, 0.18),    # f17: SwS -> SSS
        (4, 10, 0.12),    # f18: IdwS -> SSS
        (2, 3, 22.6),     # f19: SwS -> SbS
        (2, 4, 8.52),     # f20: SwS -> IdwS
        (2, 13, 28.8),    # f21: SwS -> SwN
        (3, 2, 20.2),     # f22: SbS -> SwS
        (3, 4, 2.4),      # f23: SbS -> IdwS
        (4, 2, 39.6),     # f24: IdwS -> SwS
        (4, 15, 0.72),    # f25: IdwS -> IdwN
        (10, 1, 0.36),    # f26: SSS -> TpS
        (0, 11, 22.25),   # f27: SpS -> SpN
        (1, 12, 71.6),    # f28: TpS -> TpN
        (11, 0, 22.25),   # f29: SpN -> SpS
        (12, 1, 71.6),    # f30: TpN -> TpS
        (11, 12, 23.8),   # f31: SpN -> TpN
        (12, 11, 23.8),   # f32: TpN -> SpN
        (12, 13, 29.16),  # f33: TpN -> SwN
        (12, 16, 80.51),  # f34: TpN -> SlbN
        (13, 12, 29.65),  # f35: SwN -> TpN
        (13, 2, 0.0),     # f36: SwN -> SwS (zero-flux placeholder: filtered!)
        (13, 14, 16.0),   # f37: SwN -> SbN
        (13, 15, 49.69),  # f38: SwN -> IdwN
        (16, 12, 42.01),  # f39: SlbN -> TpN
        (16, 17, 10.5),   # f40: SlbN -> LlbN
        (16, 18, 28.0),   # f41: SlbN -> LN
        (17, 18, 10.5),   # f42: LlbN -> LN
        (14, 13, 14.4),   # f43: SbN -> SwN
        (14, 15, 1.6),    # f44: SbN -> IdwN
        (13, 21, 0.12),   # f45: SwN -> SSN
        (15, 13, 22.4),   # f46: IdwN -> SwN
        (15, 21, 0.09),   # f47: IdwN -> SSN
        (15, 4, 29.52),   # f48: IdwN -> IdwS
        (18, 13, 0.7),    # f49: LN -> SwN
        (18, 19, 2.1),    # f50: LN -> SoN
        (18, 12, 35.0),   # f51: LN -> TpN
        (18, 20, 0.7),    # f52: LN -> PN
        (19, 12, 2.1),    # f53: SoN -> TpN
        (20, 12, 0.56),   # f54: PN -> TpN
        (20, 21, 0.14),   # f55: PN -> SSN
        (21, 12, 0.35),   # f56: SSN -> TpN
    )
    edges = tuple((src, dst, flux) for src, dst, flux in raw_edges if flux > 0.0)
    sym_groups = _find_symmetric_rate_groups(
        _TWO_HEM_MASSES, _TWO_HEM_TAGS, _TWO_HEM_BOX_NAMES, edges
    )
    return PresetSpec(
        name="Buntgen18",
        box_names=_TWO_HEM_BOX_NAMES,
        hemispheres=_TWO_HEM_TAGS,
        masses=_TWO_HEM_MASSES,
        edges=edges,
        prod_coefficients=_TWO_HEM_PROD,
        sink_rates=(0.0,) * len(_TWO_HEM_BOX_NAMES),
        symmetric_edge_groups=sym_groups,
    )


def calibrate_miyake17_ms_sink_rate(
    *,
    decay_rate: float = _DECAY_CAMBRIDGE,
    target_r_ref_per_q_tilde: float | None = None,
    rate_scale: float = 1.0,
) -> float:
    """Closed-form ``Marine surface`` deep-ocean loss rate ``k_sink`` for ``Miyake17``.

    For the 4-box mammillary topology ``Strat(0) <-> Trop(1) <-> {MS(2), Bio(3)}``
    with cosmic-ray allocation ``b = (0.7, 0.3, 0, 0)``, steady-state balance
    ``(-A) y_ss = Q_tilde * b`` yields the exact algebraic relation (independent
    of ``alpha_unit`` and ``Q_std``!):

    .. math::
        N_{\\mathrm{MS}} = b_1 + \\frac{k_{01}}{k_{01} + \\lambda} b_0
            - \\lambda \\tilde{y}_{1,\\star}\\left(
                1 + \\frac{k_{10}}{k_{01} + \\lambda} + \\frac{k_{13}}{k_{31} + \\lambda}
            \\right), \\qquad
        k_{\\mathrm{sink}} = \\frac{k_{21} N_{\\mathrm{MS}}}{k_{12} \\tilde{y}_{1,\\star} - N_{\\mathrm{MS}}} - \\lambda,

    where ``\\tilde{y}_{1,\\star} = M_1 * (R_ref / Q_tilde)`` is the unit-production
    tropospheric 14C inventory calibrated to match ``Guttler15``.
    """
    lam = float(decay_rate)
    m0, m1, m2, m3 = 112.5, 637.5, 900.0, 1600.0
    b0, b1 = 0.7, 0.3

    f01 = m0 * 0.5 * rate_scale
    f12 = (m1 / 11.0) * rate_scale
    f13 = (m1 / 23.0) * rate_scale

    k01, k10 = f01 / m0, f01 / m1
    k12, k21 = f12 / m1, f12 / m2
    k13, k31 = f13 / m1, f13 / m3

    if target_r_ref_per_q_tilde is None:
        guttler = _build_guttler15_spec()
        # Unit production (alpha_unit = 1, q_std = 1) since Q_tilde cancels identically.
        target_r_ref_per_q_tilde = float(
            guttler.reference_specific_activity(
                q_std=1.0, alpha_unit=1.0, decay_rate=lam
            )
        )

    y1_tilde = m1 * float(target_r_ref_per_q_tilde)
    net_ms = (
        b1
        + (k01 / (k01 + lam)) * b0
        - lam * y1_tilde * (1.0 + k10 / (k01 + lam) + k13 / (k31 + lam))
    )
    denom = k12 * y1_tilde - net_ms
    if denom <= 0.0:
        raise ValueError(
            f"Unphysical target R_ref prevents positive MS inventory (denom={denom})."
        )
    k_sink = (k21 * net_ms) / denom - lam
    if k_sink < 0.0:
        raise ValueError(
            f"Target R_ref exceeds closed shallow 4-box steady state (k_sink={k_sink})."
        )
    return float(k_sink)


def _build_miyake17_spec(
    *,
    legacy_v1_units: bool = False,
    miyake17_closure: Miyake17Closure = "open_ocean_sink",
    decay_rate: float | None = None,
    target_r_ref: float | None = None,
    q_std: float = _DEFAULT_Q_STD,
    alpha_unit: float | None = None,
) -> PresetSpec:
    if miyake17_closure not in ("open_ocean_sink", "closed_shallow"):
        raise ValueError(
            f"miyake17_closure must be 'open_ocean_sink' or 'closed_shallow', "
            f"got {miyake17_closure!r}."
        )

    box_names = ("Stratosphere", "Troposphere", "Marine surface", "Biosphere")
    hemispheres = ("None",) * 4
    masses = (112.5, 637.5, 900.0, 1600.0)
    prod_coefficients = (0.7, 0.3, 0.0, 0.0)

    scale = LEGACY_V1_C14_OVER_C12_RATIO if legacy_v1_units else 1.0
    f01 = 112.5 * 0.5 * scale
    f12 = (637.5 / 11.0) * scale
    f13 = (637.5 / 23.0) * scale

    edges = (
        (0, 1, f01),  # Stratosphere -> Troposphere
        (1, 0, f01),  # Troposphere -> Stratosphere
        (1, 2, f12),  # Troposphere -> Marine surface
        (2, 1, f12),  # Marine surface -> Troposphere
        (1, 3, f13),  # Troposphere -> Biosphere
        (3, 1, f13),  # Biosphere -> Troposphere
    )

    if legacy_v1_units or miyake17_closure == "closed_shallow":
        sink_rates = (0.0, 0.0, 0.0, 0.0)
    else:
        eff_decay = _DECAY_CAMBRIDGE if decay_rate is None else float(decay_rate)
        eff_alpha = _DEFAULT_ALPHA_UNIT if alpha_unit is None else float(alpha_unit)
        target_per_q = (
            None
            if target_r_ref is None
            else float(target_r_ref) / (eff_alpha * float(q_std))
        )
        k_sink_ms = calibrate_miyake17_ms_sink_rate(
            decay_rate=eff_decay,
            target_r_ref_per_q_tilde=target_per_q,
            rate_scale=scale,
        )
        sink_rates = (0.0, 0.0, k_sink_ms, 0.0)

    return PresetSpec(
        name="Miyake17",
        box_names=box_names,
        hemispheres=hemispheres,
        masses=masses,
        edges=edges,
        prod_coefficients=prod_coefficients,
        sink_rates=sink_rates,
        symmetric_edge_groups=None,
    )


def get_preset_spec(
    name: PresetName | str,
    *,
    legacy_v1_units: bool = False,
    miyake17_closure: Miyake17Closure = "open_ocean_sink",
    decay_rate: float | None = None,
    target_r_ref: float | None = None,
    q_std: float = _DEFAULT_Q_STD,
    alpha_unit: float | None = None,
) -> PresetSpec:
    """Return the immutable :class:`PresetSpec` for a named literature CBM.

    Parameters
    ----------
    name : {"Guttler15", "Brehm21", "Buntgen18", "Miyake17"}
        Literature preset name.
    legacy_v1_units : bool, default False
        If ``True`` on ``"Miyake17"``, applies ``ticktack 1.x``'s buggy
        ``14.003242 / 12`` transfer-rate inflation and closed 4-box inventory
        for exact regression checks against v1.x outputs.
    miyake17_closure : {"open_ocean_sink", "closed_shallow"}, default "open_ocean_sink"
        Steady-state closure for ``"Miyake17"`` when ``legacy_v1_units=False``.
    decay_rate : float, optional
        14C decay constant ``lambda_14`` in ``yr^-1`` used when calibrating
        ``Miyake17``'s ``k_sink`` against ``Guttler15`` (defaults to Cambridge
        ``ln(2) / 5730``).
    target_r_ref : float, optional
        Explicit target tropospheric specific activity ``R_ref`` (kg/GtC) for
        ``Miyake17`` when ``miyake17_closure="open_ocean_sink"``. Defaults to
        ``Guttler15``'s ``R_ref`` (~1.3195 kg/GtC at ``Q_std = 1.76``).
    """
    if name == "Guttler15":
        return _build_guttler15_spec()
    if name == "Brehm21":
        return _build_brehm21_spec()
    if name == "Buntgen18":
        return _build_buntgen18_spec()
    if name == "Miyake17":
        return _build_miyake17_spec(
            legacy_v1_units=legacy_v1_units,
            miyake17_closure=miyake17_closure,
            decay_rate=decay_rate,
            target_r_ref=target_r_ref,
            q_std=q_std,
            alpha_unit=alpha_unit,
        )
    raise ValueError(
        f"Unknown preset {name!r}; expected one of {PRESET_NAMES}."
    )


def _decode_hdf5_strings(arr: np.ndarray) -> tuple[str, ...]:
    out: list[str] = []
    for item in arr.reshape(-1):
        if isinstance(item, (bytes, np.bytes_)):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return tuple(out)


def load_preset_hdf5(
    name_or_path: str | Path,
    *,
    legacy_v1_units: bool = False,
    miyake17_closure: Miyake17Closure = "open_ocean_sink",
    decay_rate: float | None = None,
) -> PresetSpec:
    """Load a :class:`PresetSpec` from a bundled preset name or ``.hd5`` file path.

    Uses :mod:`importlib.resources` (never ``pkg_resources``) to access bundled
    ``.hd5`` files inside ``ticktack.data``. Zero-flux placeholder entries are
    filtered out automatically, and ``Miyake17.hd5``'s serialized ``14.003242/12``
    flux inflation is undone when ``legacy_v1_units=False``.
    """
    is_bundled_miyake = False
    if isinstance(name_or_path, str) and name_or_path in PRESET_NAMES:
        preset_name = name_or_path
        is_bundled_miyake = preset_name == "Miyake17"
        res = resources.files("ticktack").joinpath("data", f"{preset_name}.hd5")
        with resources.as_file(res) as resolved_path:
            return _read_hdf5_file(
                resolved_path,
                default_name=preset_name,
                undo_v1_miyake_inflation=is_bundled_miyake and not legacy_v1_units,
                calibrate_miyake_sink=(
                    is_bundled_miyake
                    and not legacy_v1_units
                    and miyake17_closure == "open_ocean_sink"
                ),
                decay_rate=decay_rate,
            )

    path = Path(name_or_path)
    inferred_name = path.stem
    is_miyake_file = inferred_name == "Miyake17"
    return _read_hdf5_file(
        path,
        default_name=inferred_name,
        undo_v1_miyake_inflation=is_miyake_file and not legacy_v1_units,
        calibrate_miyake_sink=(
            is_miyake_file
            and not legacy_v1_units
            and miyake17_closure == "open_ocean_sink"
        ),
        decay_rate=decay_rate,
    )


def _read_hdf5_file(
    path: Path,
    *,
    default_name: str,
    undo_v1_miyake_inflation: bool,
    calibrate_miyake_sink: bool,
    decay_rate: float | None,
) -> PresetSpec:
    with h5py.File(path, "r") as f:
        name = str(f.attrs.get("name", default_name))
        has_v2_format = bool(f.attrs.get("ticktack_format_version", 1) >= 2)
        box_names = _decode_hdf5_strings(np.asarray(f["nodes"]))
        hemispheres = _decode_hdf5_strings(np.asarray(f["hemispheres"]))
        masses_arr = np.asarray(f["reservoir content"], dtype=np.float64).reshape(-1)
        prod_arr = np.asarray(f["production coefficients"], dtype=np.float64).reshape(-1)
        prod_sum = float(np.sum(prod_arr))
        if prod_sum <= 0.0:
            raise ValueError(f"Non-positive total production coefficient in {path}.")
        prod_arr = prod_arr / prod_sum

        flux_mat = np.asarray(f["fluxes"], dtype=np.float64)
        if undo_v1_miyake_inflation and not has_v2_format:
            flux_mat = flux_mat / LEGACY_V1_C14_OVER_C12_RATIO

        edges: list[tuple[int, int, float]] = []
        n = len(box_names)
        for src in range(n):
            for dst in range(n):
                val = float(flux_mat[src, dst])
                if val > 0.0:
                    edges.append((src, dst, val))

        if "sink_rates" in f:
            sinks_arr = np.asarray(f["sink_rates"], dtype=np.float64).reshape(-1)
            sink_rates = tuple(float(x) for x in sinks_arr)
        elif calibrate_miyake_sink:
            eff_decay = _DECAY_CAMBRIDGE if decay_rate is None else float(decay_rate)
            k_sink_ms = calibrate_miyake17_ms_sink_rate(decay_rate=eff_decay)
            sink_rates = (0.0, 0.0, k_sink_ms, 0.0)
        else:
            sink_rates = (0.0,) * n

    masses = tuple(float(x) for x in masses_arr)
    edges_tuple = tuple(edges)
    sym_groups = (
        _find_symmetric_rate_groups(masses, hemispheres, box_names, edges_tuple)
        if any(h in ("north", "south") for h in hemispheres)
        else None
    )
    return PresetSpec(
        name=name,
        box_names=box_names,
        hemispheres=hemispheres,
        masses=masses,
        edges=edges_tuple,
        prod_coefficients=tuple(float(x) for x in prod_arr),
        sink_rates=sink_rates,
        symmetric_edge_groups=sym_groups or None,
    )


def save_preset_hdf5(spec: PresetSpec, path: str | Path) -> None:
    """Serialize a :class:`PresetSpec` to HDF5 in v1/v2-compatible dataset layout."""
    out_path = Path(path)
    str_dt = h5py.string_dtype(encoding="utf-8")
    flux_mat = np.asarray(spec.flux_matrix(), dtype=np.float64)
    masses_2d = np.asarray(spec.masses, dtype=np.float64)[None, :]
    prod_1d = np.asarray(spec.prod_coefficients, dtype=np.float64)
    sinks_1d = np.asarray(spec.sink_rates, dtype=np.float64)

    with h5py.File(out_path, "w") as f:
        f.attrs["name"] = spec.name
        f.attrs["ticktack_format_version"] = 2
        f.create_dataset("fluxes", data=flux_mat)
        f.create_dataset("reservoir content", data=masses_2d)
        f.create_dataset("production coefficients", data=prod_1d)
        f.create_dataset("sink_rates", data=sinks_1d)
        f.create_dataset("nodes", data=np.asarray(spec.box_names, dtype=object), dtype=str_dt)
        f.create_dataset(
            "hemispheres", data=np.asarray(spec.hemispheres, dtype=object), dtype=str_dt
        )


def _construct_topology(cbm_mod: Any, spec: PresetSpec) -> Any:
    """Build ``cbm.Topology`` from ``spec`` adapting cleanly to static tuple fields."""
    Topology = cbm_mod.Topology
    edges_src = tuple(src for src, _, _ in spec.edges)
    edges_dst = tuple(dst for _, dst, _ in spec.edges)
    sig = inspect.signature(Topology)
    kwargs: dict[str, Any] = {}
    candidate_map = {
        "box_names": spec.box_names,
        "hemispheres": spec.hemispheres,
        "edges_src": edges_src,
        "edges_dst": edges_dst,
        "prod_box_indices": spec.prod_box_indices,
        "trop_box_indices": spec.trop_box_indices,
    }
    for param_name in sig.parameters:
        if param_name in candidate_map:
            kwargs[param_name] = candidate_map[param_name]
    return Topology(**kwargs)


def load_preset(
    name: PresetName | str,
    kind: Literal["fixed", "balanced", "open", "reversible"] = "fixed",
    *,
    legacy_v1_units: bool = False,
    miyake17_closure: Miyake17Closure = "open_ocean_sink",
    tie_symmetric_edges: bool = False,
    decay_rate: float | None = None,
) -> Any:
    """Instantiate a ``ticktack.cbm.CarbonBoxModel`` PyTree for a named preset.

    Imports ``.cbm`` lazily so :func:`get_preset_spec` works standalone before
    all Wave 1 branches are merged.
    """
    from . import cbm as cbm_mod  # Lazy import by design.

    eff_decay = (
        (_DECAY_KUTSCHERA if legacy_v1_units else _DECAY_CAMBRIDGE)
        if decay_rate is None
        else float(decay_rate)
    )
    spec = get_preset_spec(
        name,
        legacy_v1_units=legacy_v1_units,
        miyake17_closure=miyake17_closure,
        decay_rate=eff_decay,
    )
    topology = _construct_topology(cbm_mod, spec)
    masses = jnp.asarray(spec.masses, dtype=jnp.float64)
    edge_fluxes = jnp.asarray([f for _, _, f in spec.edges], dtype=jnp.float64)
    prod_vec = jnp.asarray(spec.prod_coefficients, dtype=jnp.float64)
    sink_rates = jnp.asarray(spec.sink_rates, dtype=jnp.float64)
    has_open_sink = any(s > 0.0 for s in spec.sink_rates)

    if kind == "open" or (kind == "fixed" and has_open_sink):
        # Prefer FixedCBM if it natively accepts sink_rates for fixed parameters,
        # otherwise use OpenCBM.
        if kind == "fixed" and hasattr(cbm_mod, "FixedCBM"):
            fixed_sig = inspect.signature(cbm_mod.FixedCBM)
            if "sink_rates" in fixed_sig.parameters or "_sink_rates" in fixed_sig.parameters:
                return _instantiate_fixed_cbm(
                    cbm_mod.FixedCBM, topology, masses, spec, edge_fluxes, prod_vec, sink_rates, eff_decay
                )
        return _instantiate_open_cbm(
            cbm_mod.OpenCBM, topology, masses, spec, edge_fluxes, prod_vec, sink_rates, eff_decay
        )

    if kind == "fixed":
        return _instantiate_fixed_cbm(
            cbm_mod.FixedCBM, topology, masses, spec, edge_fluxes, prod_vec, sink_rates, eff_decay
        )

    if kind == "balanced":
        if has_open_sink:
            raise ValueError(
                f"Preset {name!r} has open sink_rates; pass kind='open' or "
                "miyake17_closure='closed_shallow' for closed BalancedCBM."
            )
        tie_mat = spec.edge_tie_matrix(tie_symmetric_edges=tie_symmetric_edges)
        BalancedCBM = cbm_mod.BalancedCBM
        if hasattr(BalancedCBM, "from_masses_and_fluxes"):
            fn_sig = inspect.signature(BalancedCBM.from_masses_and_fluxes)
            b_kwargs: dict[str, Any] = {}
            for k, v in (
                ("topology", topology),
                ("masses", masses),
                ("edge_fluxes", edge_fluxes),
                ("fluxes", spec.flux_matrix()),
                ("prod_fractions", prod_vec),
                ("production_coefficients", prod_vec),
                ("prod_vector", prod_vec),
                ("tie_matrix", tie_mat if tie_symmetric_edges else None),
                ("decay_rate", eff_decay),
            ):
                if k in fn_sig.parameters and v is not None:
                    b_kwargs[k] = v
            return BalancedCBM.from_masses_and_fluxes(**b_kwargs)
        src_idx = jnp.asarray([s for s, _, _ in spec.edges], dtype=jnp.int32)
        log_rates = jnp.log(edge_fluxes / masses[src_idx])
        if tie_symmetric_edges:
            theta_free = jnp.linalg.lstsq(tie_mat, log_rates, rcond=None)[0]
            return BalancedCBM(
                topology=topology,
                theta_free=theta_free,
                log_total_mass=jnp.log(jnp.sum(masses)),
                prod_fractions=prod_vec,
                tie_matrix=tie_mat,
                decay_rate=eff_decay,
            )
        return BalancedCBM(
            topology=topology,
            theta_free=log_rates,
            log_total_mass=jnp.log(jnp.sum(masses)),
            prod_fractions=prod_vec,
            decay_rate=eff_decay,
        )

    if kind == "reversible":
        ReversibleCBM = cbm_mod.ReversibleCBM
        if hasattr(ReversibleCBM, "from_masses_and_fluxes"):
            return ReversibleCBM.from_masses_and_fluxes(
                topology=topology,
                masses=masses,
                edge_fluxes=edge_fluxes,
                prod_fractions=prod_vec,
                decay_rate=eff_decay,
            )
        raise NotImplementedError("ReversibleCBM.from_masses_and_fluxes is required.")

    raise ValueError(
        f"Unknown kind={kind!r}; expected 'fixed', 'balanced', 'open', or 'reversible'."
    )


def _instantiate_fixed_cbm(
    cls: Any,
    topology: Any,
    masses: jax.Array,
    spec: PresetSpec,
    edge_fluxes: jax.Array,
    prod_vec: jax.Array,
    sink_rates: jax.Array,
    decay_rate: float,
) -> Any:
    sig = inspect.signature(cls)
    kwargs: dict[str, Any] = {}
    candidates = {
        "topology": topology,
        "masses": masses,
        "reservoir_masses": masses,
        "edge_fluxes": edge_fluxes,
        "fluxes": spec.flux_matrix(),
        "flux_matrix": spec.flux_matrix(),
        "prod_fractions": prod_vec,
        "prod_vector": prod_vec,
        "production_coefficients": prod_vec,
        "sink_rates": sink_rates,
        "decay_rate": decay_rate,
    }
    for pname in sig.parameters:
        if pname in candidates:
            kwargs[pname] = candidates[pname]
    return cls(**kwargs)


def _instantiate_open_cbm(
    cls: Any,
    topology: Any,
    masses: jax.Array,
    spec: PresetSpec,
    edge_fluxes: jax.Array,
    prod_vec: jax.Array,
    sink_rates: jax.Array,
    decay_rate: float,
) -> Any:
    if hasattr(cls, "from_masses_and_fluxes"):
        fn_sig = inspect.signature(cls.from_masses_and_fluxes)
        kwargs: dict[str, Any] = {}
        candidates = {
            "topology": topology,
            "masses": masses,
            "edge_fluxes": edge_fluxes,
            "fluxes": spec.flux_matrix(),
            "prod_fractions": prod_vec,
            "prod_vector": prod_vec,
            "production_coefficients": prod_vec,
            "sink_rates": sink_rates,
            "decay_rate": decay_rate,
        }
        for pname in fn_sig.parameters:
            if pname in candidates:
                kwargs[pname] = candidates[pname]
        return cls.from_masses_and_fluxes(**kwargs)

    sig = inspect.signature(cls)
    src_idx = jnp.asarray([s for s, _, _ in spec.edges], dtype=jnp.int32)
    log_rates = jnp.log(edge_fluxes / masses[src_idx])
    active_sink_indices = tuple(i for i, s in enumerate(spec.sink_rates) if s > 0.0)
    log_sinks_active = jnp.log(
        jnp.asarray([spec.sink_rates[i] for i in active_sink_indices], dtype=jnp.float64)
    )
    log_sinks_all = jnp.where(sink_rates > 0.0, jnp.log(jnp.maximum(sink_rates, 1e-300)), -jnp.inf)
    candidates = {
        "topology": topology,
        "theta_free": log_rates,
        "log_rates": log_rates,
        "log_total_mass": jnp.log(jnp.sum(masses)),
        "masses": masses,
        "edge_fluxes": edge_fluxes,
        "log_sink_rates": log_sinks_active if "sink_box_indices" in sig.parameters else log_sinks_all,
        "sink_box_indices": active_sink_indices,
        "sink_rates": sink_rates,
        "prod_fractions": prod_vec,
        "prod_vector": prod_vec,
        "production_coefficients": prod_vec,
        "decay_rate": decay_rate,
    }
    kwargs = {p: candidates[p] for p in sig.parameters if p in candidates}
    return cls(**kwargs)
