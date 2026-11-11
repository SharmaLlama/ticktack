"""Unit and autodiff tests for ``ticktack 2.0`` core CBMs, production, and solvers."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import jax

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.test_util import check_grads  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import tinygp.kernels as tk  # noqa: E402
import zodiax as zdx  # noqa: E402

from ticktack.cbm import (  # noqa: E402
    BalancedCBM,
    CarbonBoxModel,
    EffectiveCBM,
    FixedCBM,
    OpenCBM,
    ReversibleCBM,
    Topology,
    tarjan_scc,
)
from ticktack.constants import (  # noqa: E402
    ALPHA_ATOMS_CM2_S_TO_KG_YR,
    ALPHA_LEGACY_V1,
    DECAY_CAMBRIDGE,
    DECAY_KUTSCHERA,
    DECAY_STUIVER_POLACH,
    DEFAULT_DECAY_RATE,
    DEFAULT_Q_STD,
    GUTTLER15_R_REF_KG_PER_GTC,
    GUTTLER15_TROP_C14_KG,
    GUTTLER15_TROP_MASS_GTC,
)
from ticktack.production import (  # noqa: E402
    GPProduction,
    GriddedProduction,
    HarmonicSolar,
    LinearTrend,
    LogGPProduction,
    MiyakeSpike,
    ProductionSum,
    SteadyProduction,
    kernel_spectral_density,
    solin_sarkka_eigenpairs,
)
from ticktack.solvers import (  # noqa: E402
    DiffraxSolver,
    ExactLinearSolver,
    ModaxSolver,
    harmonic_resolvent,
    propagate_foh_uniform,
)


# ---------------------------------------------------------------------------
# Self-contained reference CBM builders (zero cross-branch dependency)
# ---------------------------------------------------------------------------


def make_guttler15_fixed() -> FixedCBM:
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
    masses = np.array(
        [88.5, 501.5, 900.0, 3.0, 37800.0, 110.0, 450.0, 300.0, 1350.0, 500.0, 378000.0],
        dtype=np.float64,
    )
    edges = [
        (10, 1, 0.7),
        (0, 1, 45.0),
        (1, 0, 45.0),
        (1, 2, 60.5),
        (2, 1, 61.0),
        (2, 3, 40.0),
        (3, 2, 36.0),
        (3, 4, 4.0),
        (4, 2, 42.0),
        (2, 4, 38.2),
        (2, 10, 0.3),
        (4, 10, 0.2),
        (1, 5, 115.0),
        (5, 1, 60.0),
        (5, 6, 15.0),
        (6, 7, 15.0),
        (5, 7, 40.0),
        (7, 8, 3.0),
        (7, 9, 1.0),
        (7, 2, 1.0),
        (7, 1, 50.0),
        (8, 1, 3.0),
        (9, 1, 0.8),
        (9, 10, 0.2),
    ]
    topo = Topology(
        box_names=box_names,
        edges_src=[s for s, _, _ in edges],
        edges_dst=[d for _, d, _ in edges],
        prod_box_indices=(0, 1),
        trop_box_indices=(1,),
    )
    return FixedCBM(
        topology=topo,
        masses=masses,
        edge_fluxes=[f for _, _, f in edges],
        prod_weights=(0.7, 0.3),
    )


def _hemispheric_22_boxes() -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray, np.ndarray]:
    names_11 = (
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
    box_names = names_11 + names_11
    hemispheres = ("south",) * 11 + ("north",) * 11
    masses = np.array(
        [
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
        ],
        dtype=np.float64,
    )
    prod = np.zeros(22, dtype=np.float64)
    prod[0] = 0.35
    prod[1] = 0.15
    prod[11] = 0.35
    prod[12] = 0.15
    return box_names, hemispheres, masses, prod


def make_brehm21_fixed() -> FixedCBM:
    box_names, hemispheres, masses, prod = _hemispheric_22_boxes()
    # Indices: S = 0..10 (SpS=0, TpS=1, SwS=2, SbS=3, IdwS=4, SlbS=5, LlbS=6, LS=7, SoS=8, PS=9, SSS=10)
    #          N = 11..21 (SpN=11, TpN=12, SwN=13, SbN=14, IdwN=15, SlbN=16, LlbN=17, LN=18, SoN=19, PN=20, SSN=21)
    edges = [
        (0, 1, 22.5),
        (1, 0, 22.5),
        (1, 5, 34.5),
        (5, 1, 18.0),
        (1, 2, 36.6),
        (2, 1, 36.6),
        (5, 6, 4.5),
        (6, 7, 4.5),
        (5, 7, 12.0),
        (7, 1, 15.0),
        (7, 2, 0.3),
        (7, 9, 0.3),
        (7, 8, 0.9),
        (9, 1, 0.24),
        (9, 10, 0.06),
        (8, 1, 0.9),
        (2, 10, 0.18),
        (4, 10, 0.12),
        (2, 3, 22.6),
        (2, 4, 37.92),
        (2, 13, 13.0),
        (3, 2, 20.2),
        (3, 4, 2.4),
        (4, 2, 43.2),
        (4, 15, 15.12),
        (10, 1, 0.36),
        (0, 11, 44.1),
        (1, 12, 71.0),
        (11, 0, 44.1),
        (12, 1, 71.0),
        (11, 12, 22.5),
        (12, 11, 22.5),
        (12, 13, 24.4),
        (12, 16, 80.5),
        (13, 12, 24.89),
        (13, 2, 10.0),
        (13, 14, 16.0),
        (13, 15, 28.29),
        (16, 12, 42.0),
        (16, 17, 10.5),
        (16, 18, 28.0),
        (17, 18, 10.5),
        (14, 13, 14.4),
        (14, 15, 1.6),
        (13, 21, 0.12),
        (15, 13, 26.8),
        (15, 21, 0.09),
        (15, 4, 18.12),
        (18, 13, 0.7),
        (18, 19, 2.1),
        (18, 12, 35.0),
        (18, 20, 0.7),
        (19, 12, 2.1),
        (20, 12, 0.56),
        (20, 21, 0.14),
        (21, 12, 0.35),
    ]
    topo = Topology(
        box_names=box_names,
        edges_src=[s for s, _, _ in edges],
        edges_dst=[d for _, d, _ in edges],
        hemispheres=hemispheres,
        prod_box_indices=(0, 1, 11, 12),
        trop_box_indices=(1, 12),
    )
    return FixedCBM(
        topology=topo,
        masses=masses,
        edge_fluxes=[f for _, _, f in edges],
        prod_weights=prod,
    )


def make_buntgen18_fixed() -> FixedCBM:
    box_names, hemispheres, masses, prod = _hemispheric_22_boxes()
    raw_edges = [
        (0, 1, 23.8),
        (1, 0, 23.8),
        (1, 5, 35.36),
        (5, 1, 18.86),
        (1, 2, 38.25),
        (2, 1, 38.25),
        (5, 6, 4.5),
        (6, 7, 4.5),
        (5, 7, 12.0),
        (7, 1, 15.0),
        (7, 2, 0.3),
        (7, 9, 0.3),
        (7, 8, 0.9),
        (9, 1, 0.24),
        (9, 10, 0.06),
        (8, 1, 0.9),
        (2, 10, 0.18),
        (4, 10, 0.12),
        (2, 3, 22.6),
        (2, 4, 8.52),
        (2, 13, 28.8),
        (3, 2, 20.2),
        (3, 4, 2.4),
        (4, 2, 39.6),
        (4, 15, 0.72),
        (10, 1, 0.36),
        (0, 11, 22.25),
        (1, 12, 71.6),
        (11, 0, 22.25),
        (12, 1, 71.6),
        (11, 12, 23.8),
        (12, 11, 23.8),
        (12, 13, 29.16),
        (12, 16, 80.51),
        (13, 12, 29.65),
        (13, 2, 0.0),  # zero-flux placeholder edge in Buntgen18 -> filtered out
        (13, 14, 16.0),
        (13, 15, 49.69),
        (16, 12, 42.01),
        (16, 17, 10.5),
        (16, 18, 28.0),
        (17, 18, 10.5),
        (14, 13, 14.4),
        (14, 15, 1.6),
        (13, 21, 0.12),
        (15, 13, 22.4),
        (15, 21, 0.09),
        (15, 4, 29.52),
        (18, 13, 0.7),
        (18, 19, 2.1),
        (18, 12, 35.0),
        (18, 20, 0.7),
        (19, 12, 2.1),
        (20, 12, 0.56),
        (20, 21, 0.14),
        (21, 12, 0.35),
    ]
    pos_edges = [(s, d, f) for s, d, f in raw_edges if f > 0.0]
    assert len(pos_edges) == 55
    topo = Topology(
        box_names=box_names,
        edges_src=[s for s, _, _ in pos_edges],
        edges_dst=[d for _, d, _ in pos_edges],
        hemispheres=hemispheres,
        prod_box_indices=(0, 1, 11, 12),
        trop_box_indices=(1, 12),
    )
    return FixedCBM(
        topology=topo,
        masses=masses,
        edge_fluxes=[f for _, _, f in pos_edges],
        prod_weights=prod,
    )


def make_miyake17_open(with_deep_sink: bool = True) -> OpenCBM | FixedCBM:
    box_names = ("Stratosphere", "Troposphere", "Marine surface", "Biosphere")
    masses = np.array([112.5, 637.5, 900.0, 1600.0], dtype=np.float64)
    f_st = 112.5 * (1.0 / 2.0)
    f_tm = 637.5 * (1.0 / 11.0)
    f_tb = 637.5 * (1.0 / 23.0)
    edges = [
        (0, 1, f_st),
        (1, 0, f_st),
        (1, 2, f_tm),
        (2, 1, f_tm),
        (1, 3, f_tb),
        (3, 1, f_tb),
    ]
    topo = Topology(
        box_names=box_names,
        edges_src=[s for s, _, _ in edges],
        edges_dst=[d for _, d, _ in edges],
        prod_box_indices=(0, 1),
        trop_box_indices=(1,),
    )
    if not with_deep_sink:
        return FixedCBM(
            topology=topo,
            masses=masses,
            edge_fluxes=[f for _, _, f in edges],
            prod_weights=(0.7, 0.3),
        )
    # Calibrated deep-ocean loss on Marine surface (~0.029 yr^-1)
    sink_rates = np.array([0.0, 0.0, 0.029, 0.0], dtype=np.float64)
    return OpenCBM.from_masses_and_fluxes(
        topology=topo,
        masses=masses,
        edge_fluxes=[f for _, _, f in edges],
        sink_rates=sink_rates,
        prod_weights=(0.7, 0.3),
    )


# ---------------------------------------------------------------------------
# 1. Constants, Topology, and Radiometric Normalization
# ---------------------------------------------------------------------------


def test_constants_conventions() -> None:
    assert DECAY_CAMBRIDGE == pytest.approx(math.log(2.0) / 5730.0, rel=0.0, abs=1e-16)
    assert DECAY_STUIVER_POLACH == pytest.approx(1.0 / 8267.0, rel=0.0, abs=1e-16)
    assert DECAY_KUTSCHERA == pytest.approx(math.log(2.0) / 5700.0, rel=0.0, abs=1e-16)
    assert DEFAULT_DECAY_RATE == DECAY_CAMBRIDGE
    assert DEFAULT_Q_STD == 1.76
    # CODATA and legacy alpha differ by ~0.18% due to 365.25 vs 365 days & Earth area
    assert abs(ALPHA_ATOMS_CM2_S_TO_KG_YR - ALPHA_LEGACY_V1) / ALPHA_ATOMS_CM2_S_TO_KG_YR < 3e-3
    assert GUTTLER15_TROP_MASS_GTC == 501.5


def test_topology_tarjan_scc_and_box_lookup() -> None:
    # Two-component reducible graph: (0 <-> 1) -> (2 <-> 3)
    sccs = tarjan_scc(4, [0, 1, 1, 2, 3], [1, 0, 2, 3, 2])
    assert len(sccs) == 2
    assert set(sccs) == {(0, 1), (2, 3)}

    with pytest.raises(ValueError, match="not strongly connected"):
        Topology(
            box_names=("Stratosphere", "Troposphere", "Ocean", "Deep"),
            edges_src=[0, 1, 1, 2, 3],
            edges_dst=[1, 0, 2, 3, 2],
        )

    # Reject mixed 'none' and hemispheric labels
    with pytest.raises(ValueError, match="cannot mix"):
        Topology(
            box_names=("Stratosphere", "Troposphere"),
            edges_src=[0, 1],
            edges_dst=[1, 0],
            hemispheres=("none", "north"),
        )

    g15 = make_guttler15_fixed()
    # 1-globe topology accepts north/south lookup cleanly for hemispheric ring series
    assert g15.box_index("Troposphere") == 1
    assert g15.box_index("Troposphere", "north") == 1
    assert g15.box_index("Troposphere", "south") == 1

    b21 = make_brehm21_fixed()
    assert b21.box_index("Troposphere", "south") == 1
    assert b21.box_index("Troposphere", "north") == 12
    with pytest.raises(ValueError, match="Ambiguous box lookup"):
        b21.box_index("Troposphere")


def test_brehm21_global_rref_preserves_interhemispheric_gradient() -> None:
    b21 = make_brehm21_fixed()
    ss_d14c = b21.steady_state_d14c()
    tp_s = b21.box_index("Troposphere", "south")
    tp_n = b21.box_index("Troposphere", "north")

    gradient = float(ss_d14c[tp_n] - ss_d14c[tp_s])
    assert gradient == pytest.approx(4.378, abs=0.01)

    # Mass-weighted tropospheric mean Delta14C is identically 0 permil
    masses = b21.reservoir_masses()
    weighted_mean = (
        masses[tp_s] * ss_d14c[tp_s] + masses[tp_n] * ss_d14c[tp_n]
    ) / (masses[tp_s] + masses[tp_n])
    assert abs(float(weighted_mean)) < 5e-13

    # Steady production at Q_std gives constant NH/SH offset in ExactLinearSolver
    solver = ExactLinearSolver()
    prod = SteadyProduction(DEFAULT_Q_STD)
    t_s = jnp.array([770.0, 771.0])
    t_e = jnp.array([771.0, 772.0])
    nh = solver.predict_d14c(b21, prod, t_s, t_e, box_indices=tp_n)
    sh = solver.predict_d14c(b21, prod, t_s, t_e, box_indices=tp_s)
    np.testing.assert_allclose(nh - sh, gradient, rtol=0.0, atol=1e-13)


def test_alpha_unit_invariance_and_closed_form_guttler_production() -> None:
    g15 = make_guttler15_fixed()
    # Closed-form production_for_target_c14 round-trip
    y_ss = g15.steady_state_c14(q_std=DEFAULT_Q_STD, alpha_unit=ALPHA_LEGACY_V1)
    q_rec = g15.production_for_target_c14(
        y_ss[1], box="Troposphere", alpha_unit=ALPHA_LEGACY_V1
    )
    assert float(q_rec) == pytest.approx(DEFAULT_Q_STD, rel=1e-14, abs=1e-15)

    # Default Guttler15 reference target with CODATA vs legacy alpha
    y_ss_codata = g15.steady_state_c14()
    q_rec_codata = g15.production_for_target_c14(y_ss_codata[1])
    assert float(q_rec_codata) == pytest.approx(DEFAULT_Q_STD, rel=1e-14, abs=1e-15)
    assert float(y_ss_codata[1]) == pytest.approx(GUTTLER15_TROP_C14_KG, rel=1e-13)
    assert float(g15.reference_specific_activity()) == pytest.approx(
        GUTTLER15_R_REF_KG_PER_GTC, rel=1e-14
    )

    # Exact invariance of Delta14C to alpha_unit
    solver = ExactLinearSolver()
    prod = SteadyProduction(DEFAULT_Q_STD) + MiyakeSpike(
        t_start=774.55, duration=0.2, fluence=6.8, profile="supergaussian"
    )
    t_s = jnp.arange(773.0, 780.0) + 4.0 / 12.0
    t_e = jnp.arange(773.0, 780.0) + 8.0 / 12.0
    d14c_codata = solver.predict_d14c(
        g15, prod, t_s, t_e, box_indices=1, alpha_unit=ALPHA_ATOMS_CM2_S_TO_KG_YR
    )
    d14c_legacy = solver.predict_d14c(
        g15, prod, t_s, t_e, box_indices=1, alpha_unit=ALPHA_LEGACY_V1
    )
    d14c_arb = solver.predict_d14c(
        g15, prod, t_s, t_e, box_indices=1, alpha_unit=42.0
    )
    np.testing.assert_allclose(d14c_codata, d14c_legacy, rtol=0.0, atol=5e-13)
    np.testing.assert_allclose(d14c_codata, d14c_arb, rtol=0.0, atol=5e-13)


# ---------------------------------------------------------------------------
# 2. CarbonBoxModel Subclasses: Bijections, Kirchhoff Balance, Identifiability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder",
    [make_guttler15_fixed, make_brehm21_fixed, make_buntgen18_fixed],
)
def test_balanced_cbm_round_trip_and_kirchhoff_balance(builder) -> None:
    fixed = builder()
    bal = BalancedCBM.from_masses_and_fluxes(
        fixed.topology,
        fixed.reservoir_masses(),
        fixed.edge_fluxes,
        prod_weights=fixed.prod_weights,
    )

    assert bal.log_rates_free.dtype == jnp.float64
    assert bal.log_total_mass.dtype == jnp.float64
    np.testing.assert_allclose(
        bal.reservoir_masses(), fixed.reservoir_masses(), rtol=1e-13, atol=1e-12
    )
    np.testing.assert_allclose(
        bal.flux_matrix(), fixed.flux_matrix(), rtol=1e-13, atol=1e-12
    )
    np.testing.assert_allclose(
        bal.matrix_A(), fixed.matrix_A(), rtol=1e-13, atol=1e-14
    )

    # Random unconstrained draw in R^{|E|} must satisfy Kirchhoff balance to < 1e-14 rel
    rng = np.random.default_rng(1234)
    rand_theta = bal.log_rates_free + 0.35 * rng.standard_normal(bal.topology.n_edges)
    bal_rand = BalancedCBM(
        fixed.topology,
        jnp.asarray(rand_theta, dtype=jnp.float32),  # verify float64 coercion
        jnp.float32(12.5),
        prod_weights=fixed.prod_weights,
    )
    assert bal_rand.log_rates_free.dtype == jnp.float64
    assert bal_rand.log_total_mass.dtype == jnp.float64
    F = bal_rand.flux_matrix()
    total_flux = jnp.sum(F)
    rel_kirchhoff_global = jnp.max(jnp.abs(bal_rand.kirchhoff_residual())) / total_flux
    assert float(rel_kirchhoff_global) < 1e-15
    throughput = jnp.maximum(jnp.sum(F, axis=1), 1e-30)
    rel_kirchhoff_box = jnp.max(jnp.abs(bal_rand.kirchhoff_residual()) / throughput)
    assert float(rel_kirchhoff_box) < 5e-14
    # Column sums of K are identically zero to machine precision
    col_sums = jnp.sum(bal_rand.rate_matrix_K(), axis=0)
    assert float(jnp.max(jnp.abs(col_sums))) < 1e-14


def test_balanced_cbm_edge_tying_and_total_mass_gauge_invariance() -> None:
    b21 = make_brehm21_fixed()
    # Tie SpS->TpS (edge 0), TpS->SpS (edge 1), SpN->TpN (edge 30), TpN->SpN (edge 31)
    # Note SpS=SpN=44.5 so 0 and 30 share exact rate 22.5/44.5, and 1 and 31 share 22.5/250.5
    groups = [[0, 30], [1, 31], [26, 28], [27, 29]]
    bal_tied = BalancedCBM.from_masses_and_fluxes(
        b21.topology,
        b21.reservoir_masses(),
        b21.edge_fluxes,
        prod_weights=b21.prod_weights,
        symmetric_edge_groups=groups,
    )
    assert bal_tied.log_rates_free.shape == (56 - 4,)
    np.testing.assert_allclose(
        bal_tied.reservoir_masses(), b21.reservoir_masses(), rtol=1e-13, atol=1e-12
    )

    # Zero Fisher sensitivity of Delta14C to log_total_mass
    solver = ExactLinearSolver()
    prod = MiyakeSpike(t_start=774.6, duration=0.2, fluence=6.5)
    t_s = jnp.array([774.3, 775.3, 776.3])
    t_e = t_s + 0.4

    def loss_fn(log_rates: jax.Array, log_mtot: jax.Array) -> jax.Array:
        m = BalancedCBM(
            b21.topology,
            log_rates,
            log_mtot,
            prod_weights=b21.prod_weights,
            symmetric_edge_groups=groups,
        )
        pred = solver.predict_d14c(m, prod, t_s, t_e, box_indices=12)
        return jnp.sum(jnp.square(pred))

    g_rates, g_mtot = jax.grad(loss_fn, argnums=(0, 1))(
        bal_tied.log_rates_free, bal_tied.log_total_mass
    )
    assert float(jnp.abs(g_mtot)) < 1e-13
    assert float(jnp.linalg.norm(g_rates)) > 1.0
    check_grads(
        lambda r: loss_fn(r, bal_tied.log_total_mass),
        (bal_tied.log_rates_free,),
        order=1,
        modes=["fwd", "rev"],
        rtol=1e-6,
        atol=1e-6,
    )


def test_reversible_and_open_cbm() -> None:
    m17_closed = make_miyake17_open(with_deep_sink=False)
    assert isinstance(m17_closed, FixedCBM)
    rev = ReversibleCBM.from_masses_and_fluxes(
        m17_closed.topology,
        m17_closed.reservoir_masses(),
        m17_closed.edge_fluxes,
        prod_weights=m17_closed.prod_weights,
    )
    np.testing.assert_allclose(
        rev.reservoir_masses(), m17_closed.reservoir_masses(), rtol=1e-14
    )
    np.testing.assert_allclose(
        rev.flux_matrix(), m17_closed.flux_matrix(), rtol=1e-14
    )
    assert float(jnp.max(jnp.abs(rev.kirchhoff_residual()))) < 1e-14

    m17_open = make_miyake17_open(with_deep_sink=True)
    assert isinstance(m17_open, OpenCBM)
    assert float(m17_open.sink_rates()[2]) == pytest.approx(0.029, rel=1e-14)
    eigvals = jnp.linalg.eigvals(m17_open.matrix_A())
    assert bool(jnp.all(jnp.real(eigvals) < 0.0))

    solver = ExactLinearSolver()
    prod = MiyakeSpike(t_start=774.5, duration=0.25, fluence=6.5)
    t_s = jnp.array([774.0, 775.0, 776.0, 778.0])
    t_e = t_s + 0.5

    def open_loss(log_rates: jax.Array, log_sinks: jax.Array) -> jax.Array:
        cbm = OpenCBM(
            m17_open.topology,
            log_rates,
            m17_open.log_total_mass,
            log_sinks,
            open_box_indices=m17_open.open_box_indices,
            prod_weights=m17_open.prod_weights,
        )
        return jnp.sum(solver.predict_d14c(cbm, prod, t_s, t_e, box_indices=1) ** 2)

    check_grads(
        open_loss,
        (m17_open.log_rates_free, m17_open.log_sink_rates),
        order=1,
        modes=["fwd", "rev"],
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("K", [2, 3, 4, 5])
def test_effective_cbm_exact_param_count_ordering_and_full_jacobian_rank(K: int) -> None:
    eff = EffectiveCBM(K=K)
    leaves = jax.tree_util.tree_leaves(eqx.filter(eff, eqx.is_inexact_array))
    n_params = sum(int(x.size) for x in leaves)
    assert n_params == 2 * K - 2
    assert eff.n_parameters == 2 * K - 2

    if K > 2:
        taus = np.asarray(eff.sink_timescales())
        assert taus.shape == (K - 2,)
        assert np.all(np.diff(taus) > 0.0)

        # Even with strongly negative increment parameters, ordering remains strictly increasing
        neg_params = jnp.concatenate(
            [jnp.array([1.5]), jnp.full((K - 3,), -8.0)]
        )
        eff_neg = EffectiveCBM(K=K, sink_tau_params=neg_params)
        assert bool(jnp.all(jnp.diff(eff_neg.log_sink_timescales()) > 0.0))

    assert float(jnp.max(jnp.abs(eff.kirchhoff_residual()))) < 1e-13
    if K >= 3:
        assert float(eff.reference_specific_activity()) == pytest.approx(
            GUTTLER15_R_REF_KG_PER_GTC, rel=1e-12
        )

    # Full 2K - 2 Jacobian column rank on post-spike + multi-timescale ring trajectory
    solver = ExactLinearSolver()
    prod = MiyakeSpike(t_start=774.5, duration=0.15, fluence=6.5, profile="boxcar") + (
        HarmonicSolar.from_polar(periods=[11.0, 88.0], amplitudes=[0.25, 0.35])
    )
    # Span fast intra-annual, decadal, and multi-centennial lags so all K-2 sinks excite
    lags = np.concatenate(
        [
            np.linspace(-0.5, 3.0, 10),
            np.geomspace(3.5, 450.0, 22),
        ]
    )
    t_s = jnp.asarray(774.5 + lags, dtype=jnp.float64)
    t_e = t_s + 0.35

    flat0, unravel = jax.flatten_util.ravel_pytree(
        eqx.filter(eff, eqx.is_inexact_array)
    )
    static_part = eqx.filter(eff, lambda x: not eqx.is_inexact_array(x))

    def pred_from_vec(vec: jax.Array) -> jax.Array:
        cbm_v = eqx.combine(unravel(vec), static_part)
        return solver.predict_d14c(cbm_v, prod, t_s, t_e, box_indices=1)

    J = jax.jacfwd(pred_from_vec)(flat0)
    assert J.shape == (t_s.shape[0], 2 * K - 2)
    svals = jnp.linalg.svd(J, compute_uv=False)
    rank = int(jnp.linalg.matrix_rank(J, tol=1e-10))
    assert rank == 2 * K - 2
    assert float(svals[-1] / svals[0]) > 1e-6

    check_grads(
        lambda v: jnp.sum(pred_from_vec(v) ** 2),
        (flat0,),
        order=1,
        modes=["fwd", "rev"],
        rtol=1e-6,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# 3. Production Terms, Spectral Densities, LogGP Itô Mean, and zodiax Paths
# ---------------------------------------------------------------------------


class _DummyForwardModel(zdx.Base):
    cbm: CarbonBoxModel
    production: ProductionSum
    solver: ExactLinearSolver


def test_production_sum_zodiax_get_set_and_partitioning() -> None:
    prod = (
        SteadyProduction(1.76)
        + HarmonicSolar.from_polar(periods=[11.0, 208.0], amplitudes=[0.25, 0.15])
        + LinearTrend(slope=0.002, t_ref=775.0)
        + MiyakeSpike(t_start=774.55, duration=0.25, fluence=6.5)
        + MiyakeSpike(t_start=993.3, duration=0.30, fluence=4.1)
        + GPProduction(tk.quasisep.Matern32(sigma=0.3, scale=8.0), n_modes=32)
    )
    assert prod.names == ("steady", "solar", "trend", "spike", "spike_2", "gp")

    lin, nonlin = prod.partition_linear_nonlinear()
    assert set(lin.keys()) == {"solar", "trend", "gp"}
    assert set(nonlin.keys()) == {"spike", "spike_2"}
    assert not prod.is_linear_marginalizable

    lin_only = SteadyProduction(1.76) + HarmonicSolar(11.0, c_cos=0.2, c_sin=0.1)
    assert lin_only.is_linear_marginalizable

    # Nested zodiax string-path .get / .set / .add under JIT
    model = _DummyForwardModel(
        cbm=EffectiveCBM(K=3),
        production=prod,
        solver=ExactLinearSolver(),
    )
    assert float(model.get("production.spike.t_start")) == pytest.approx(774.55)
    assert float(model.get("production.spike_2.fluence")) == pytest.approx(4.1)

    updated = model.set(
        ["production.spike.t_start", "production.solar.c_cos"],
        [jnp.asarray(774.72), jnp.array([0.31, -0.08])],
    )
    assert float(updated.get("production.spike.t_start")) == pytest.approx(774.72)
    np.testing.assert_allclose(
        updated.get("production.solar.c_cos"), np.array([0.31, -0.08]), atol=1e-15
    )

    @eqx.filter_jit
    def jit_eval(m: _DummyForwardModel, new_t: jax.Array) -> jax.Array:
        m2 = m.set("production.spike.t_start", new_t)
        return m2.solver.predict_d14c(
            m2.cbm,
            m2.production,
            jnp.array([774.3, 775.3]),
            jnp.array([774.7, 775.7]),
            box_indices=1,
        )

    out1 = jit_eval(model, jnp.asarray(774.4))
    out2 = jit_eval(model, jnp.asarray(774.9))
    assert not jnp.allclose(out1, out2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"profile": "boxcar"},
        {"profile": "supergaussian", "exponent": 16.0},
        {"profile": "supergaussian", "exponent": 4.0},
        {"profile": "skewgaussian", "skewness": 2.5},
        {"profile": "skewgaussian", "skewness": -1.8},
        {"profile": "gamma", "gamma_shape": 3.0},
        {"profile": "gamma", "gamma_shape": 5.5},
    ],
)
def test_miyake_spike_fluence_and_fwhm_normalization(kwargs: dict) -> None:
    dur = 0.32
    fl = 7.25
    spike = MiyakeSpike(t_start=774.5, duration=dur, fluence=fl, **kwargs)

    # Fine-grid numerical fluence check
    t_grid = jnp.linspace(770.0, 782.0, 240_001)
    dq = spike.delta_q(t_grid)
    num_fluence = float(jnp.trapezoid(dq, t_grid))
    assert num_fluence == pytest.approx(fl, rel=2e-6, abs=1e-6)

    # Exact FWHM == duration check
    u_dense = jnp.linspace(-4.0, 8.0, 400_001)
    phi = np.asarray(spike.unit_profile(u_dense))
    half_max = 0.5 * np.max(phi)
    above = np.asarray(u_dense)[phi >= half_max]
    fwhm_u = float(above[-1] - above[0])
    assert fwhm_u == pytest.approx(1.0, abs=2e-4)


@pytest.mark.parametrize(
    "kernel",
    [
        1.44 * tk.Matern32(scale=6.5),
        0.81 * tk.Matern52(scale=4.2),
        0.64 * tk.Exp(scale=9.0),
        1.21 * tk.ExpSquared(scale=3.0),
        tk.quasisep.Matern32(sigma=1.2, scale=6.5),
        tk.quasisep.Matern52(sigma=0.9, scale=4.2),
        tk.quasisep.Exp(sigma=0.8, scale=9.0),
        tk.quasisep.SHO(omega=2.0 * math.pi / 11.0, quality=4.5, sigma=0.5),
        tk.quasisep.Matern32(sigma=0.7, scale=12.0)
        + tk.quasisep.SHO(omega=2.0 * math.pi / 11.0, quality=6.0, sigma=0.35),
    ],
)
def test_kernel_spectral_density_wiener_khinchin(kernel) -> None:
    # Verify k(tau) = (1 / pi) int_0^inf S_k(omega) cos(omega * tau) domega
    # Map omega = 10 * tan(u) on u in [0, pi/2 - 1e-4] to integrate to infinity cleanly
    u = jnp.linspace(0.0, 0.5 * math.pi - 1e-4, 60_001)
    w = 10.0 * jnp.tan(u)
    dw_du = 10.0 / jnp.square(jnp.cos(u))
    psd = kernel_spectral_density(kernel, w)
    for tau in (0.0, 0.7, 2.5, 7.0):
        integrand = psd * jnp.cos(w * tau) * dw_du
        cov_wk = float(jnp.trapezoid(integrand, u) / math.pi)
        cov_exact = float(kernel.evaluate(jnp.asarray(0.0), jnp.asarray(tau)))
        assert cov_wk == pytest.approx(cov_exact, rel=1e-5, abs=1e-5)


def test_log_gp_production_ito_mean_correction() -> None:
    kernel = tk.quasisep.Matern52(sigma=0.65, scale=5.0)
    domain = (750.0, 810.0)
    n_modes = 64
    t_eval = jnp.linspace(765.0, 795.0, 25)

    # Analytical moment-generating identity: E[exp(w^T z - 0.5 ||w||^2)] == 1 for z ~ N(0, I)
    rng = np.random.default_rng(2026)
    n_mc = 40_000
    # Antithetic + stratified normal draws for tight MC check
    z_half = rng.standard_normal((n_mc // 2, n_modes))
    z_draws = jnp.asarray(np.vstack([z_half, -z_half]), dtype=jnp.float64)

    def eval_one(z: jax.Array) -> jax.Array:
        lgp = LogGPProduction(kernel, z, domain=domain)
        return lgp(t_eval, q_std=DEFAULT_Q_STD)

    q_samples = jax.vmap(eval_one)(z_draws)
    assert bool(jnp.all(q_samples > 0.0))
    mc_mean = jnp.mean(q_samples, axis=0)
    np.testing.assert_allclose(mc_mean, DEFAULT_Q_STD, rtol=8e-3, atol=8e-3)

    # Also verify exact algebraic variance identity log_field_and_variance
    omegas, psd, half_len = solin_sarkka_eigenpairs(domain[0], domain[1], n_modes, kernel)
    psi = (1.0 / math.sqrt(half_len)) * jnp.sin((t_eval[:, None] - domain[0]) * omegas)
    exact_var = jnp.sum((psi**2) * psd, axis=-1)
    lgp0 = LogGPProduction(kernel, jnp.zeros((n_modes,)), domain=domain)
    f0, var0 = lgp0.log_field_and_variance(t_eval)
    np.testing.assert_allclose(f0, 0.0, atol=1e-15)
    np.testing.assert_allclose(var0, exact_var, rtol=1e-14, atol=1e-15)


# ---------------------------------------------------------------------------
# 4. Solvers: Complex Resolvent, < 1e-11 Diffrax Agreement, rfft, and Gradients
# ---------------------------------------------------------------------------


def test_complex_harmonic_resolvent_ode_residual_and_grads() -> None:
    b21 = make_brehm21_fixed()
    A = b21.matrix_A()
    b_scaled = ALPHA_ATOMS_CM2_S_TO_KG_YR * b21.production_vector()
    solar = HarmonicSolar(
        periods=[11.0, 88.0, 208.0],
        c_cos=[0.28, -0.14, 0.22],
        c_sin=[-0.11, 0.19, 0.07],
        t_ref=775.0,
    )
    # Direct complex linear solve residual: (1j * omega * I - A) (u + 1j * v) == b_scaled
    u_m, v_m = harmonic_resolvent(A, b_scaled, solar.omegas)
    z_m = u_m + 1j * v_m
    for m_idx, w_val in enumerate(np.asarray(solar.omegas)):
        res = (1j * w_val * jnp.eye(b21.n_boxes) - A) @ z_m[m_idx] - b_scaled
        assert float(jnp.linalg.norm(res) / jnp.linalg.norm(b_scaled)) < 1e-14

    solver = ExactLinearSolver()

    def state_at(t_scalar: jax.Array) -> jax.Array:
        return solver.solve_delta_trajectory(b21, solar, jnp.reshape(t_scalar, (1,)))[0]

    for t_val in (770.15, 775.0, 783.7, 993.4):
        t_j = jnp.asarray(t_val, dtype=jnp.float64)
        dy, ddy_dt = jax.jvp(state_at, (t_j,), (jnp.ones_like(t_j),))
        rhs = A @ dy + b_scaled * solar.delta_q(t_j)
        scale = jnp.linalg.norm(rhs)
        rel_res = jnp.linalg.norm(ddy_dt - rhs) / scale
        assert float(rel_res) < 1e-14

    t_s = jnp.arange(770.0, 776.0) + 4.0 / 12.0
    t_e = jnp.arange(770.0, 776.0) + 8.0 / 12.0

    def harm_loss(c_cos: jax.Array, c_sin: jax.Array, periods: jax.Array) -> jax.Array:
        term = HarmonicSolar(periods=periods, c_cos=c_cos, c_sin=c_sin, t_ref=775.0)
        return jnp.sum(solver.predict_d14c(b21, term, t_s, t_e, box_indices=12) ** 2)

    check_grads(
        harm_loss,
        (solar.c_cos, solar.c_sin, solar.periods),
        order=1,
        modes=["fwd", "rev"],
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "cbm_factory,box_idx",
    [
        (make_guttler15_fixed, 1),
        (make_brehm21_fixed, 12),
        (make_buntgen18_fixed, 1),
        (lambda: make_miyake17_open(with_deep_sink=True), 1),
        (lambda: EffectiveCBM(K=4), 1),
    ],
)
def test_exact_linear_vs_diffrax_tsit5_across_all_cbms(
    cbm_factory, box_idx: int
) -> None:
    cbm = cbm_factory()
    prod = (
        SteadyProduction(1.81)
        + HarmonicSolar(
            periods=[11.0, 208.0],
            c_cos=[0.24, -0.15],
            c_sin=[0.12, 0.09],
            t_ref=774.0,
        )
        + LinearTrend(slope=0.004, t_ref=774.0, switched_on=False)
        + MiyakeSpike(
            t_start=774.58, duration=0.22, fluence=6.8, profile="supergaussian"
        )
    )

    exact = ExactLinearSolver()
    dfx = DiffraxSolver("Tsit5", rtol=1e-13, atol=1e-13, max_steps=262_144)

    t_eval = jnp.linspace(773.5, 780.5, 15)
    dy_exact = exact.solve_delta_trajectory(cbm, prod, t_eval)
    dy_dfx = dfx.solve_delta_trajectory(cbm, prod, t_eval)
    np.testing.assert_allclose(dy_exact, dy_dfx, rtol=1e-11, atol=1e-11)

    years = jnp.arange(773.0, 781.0)
    t_starts = years + 4.0 / 12.0
    t_ends = years + 8.0 / 12.0
    d14c_exact = exact.predict_d14c(
        cbm,
        prod,
        t_starts,
        t_ends,
        box_indices=box_idx,
        delta_global=-1.5,
        delta_lab=0.25,
    )
    d14c_dfx = dfx.predict_d14c(
        cbm,
        prod,
        t_starts,
        t_ends,
        box_indices=box_idx,
        delta_global=-1.5,
        delta_lab=0.25,
    )
    np.testing.assert_allclose(d14c_exact, d14c_dfx, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize(
    "spike_kwargs",
    [
        {"profile": "boxcar"},
        {"profile": "supergaussian", "exponent": 16.0},
        {"profile": "skewgaussian", "skewness": 2.0},
        {"profile": "gamma", "gamma_shape": 3.5},
    ],
)
def test_exact_linear_vs_diffrax_tsit5_across_all_spike_profiles(
    spike_kwargs: dict,
) -> None:
    cbm = make_brehm21_fixed()
    prod = (
        SteadyProduction(1.81)
        + HarmonicSolar(
            periods=[11.0, 208.0],
            c_cos=[0.24, -0.15],
            c_sin=[0.12, 0.09],
            t_ref=774.0,
        )
        + LinearTrend(slope=0.004, t_ref=774.0, switched_on=False)
        + MiyakeSpike(t_start=774.58, duration=0.22, fluence=6.8, **spike_kwargs)
    )

    exact = ExactLinearSolver()
    dfx = DiffraxSolver("Tsit5", rtol=1e-13, atol=1e-13, max_steps=262_144)

    t_eval = jnp.linspace(773.5, 780.5, 15)
    dy_exact = exact.solve_delta_trajectory(cbm, prod, t_eval)
    dy_dfx = dfx.solve_delta_trajectory(cbm, prod, t_eval)
    np.testing.assert_allclose(dy_exact, dy_dfx, rtol=1e-11, atol=1e-11)

    years = jnp.arange(773.0, 781.0)
    # Test both NH (May-Aug) and SH (Oct-Feb Austral summer) windows in one combined call
    ts_both = jnp.concatenate([years + 4.0 / 12.0, years + 9.0 / 12.0])
    te_both = jnp.concatenate([years + 8.0 / 12.0, years + 14.0 / 12.0])
    b_both = jnp.concatenate(
        [
            jnp.full(years.shape, 12, dtype=jnp.int32),
            jnp.full(years.shape, 1, dtype=jnp.int32),
        ]
    )
    d14c_exact = exact.predict_d14c(cbm, prod, ts_both, te_both, box_indices=b_both)
    d14c_dfx = dfx.predict_d14c(cbm, prod, ts_both, te_both, box_indices=b_both)
    np.testing.assert_allclose(d14c_exact, d14c_dfx, rtol=1e-11, atol=1e-11)

    # Also verify pure absolute < 1e-11 permil agreement at O(1) epochs (where jump_ts ULP is ~2e-16 yr)
    g15 = make_guttler15_fixed()
    prod_rel = MiyakeSpike(t_start=1.58, duration=0.22, fluence=6.8, **spike_kwargs)
    y_rel = jnp.arange(0.0, 8.0)
    d14c_ex_rel = exact.predict_d14c(
        g15, prod_rel, y_rel + 4.0 / 12.0, y_rel + 8.0 / 12.0, box_indices=1
    )
    d14c_df_rel = dfx.predict_d14c(
        g15, prod_rel, y_rel + 4.0 / 12.0, y_rel + 8.0 / 12.0, box_indices=1
    )
    assert float(jnp.max(jnp.abs(d14c_ex_rel - d14c_df_rel))) < 1e-11


def test_switched_on_trend_gridded_foh_2d_subweights_and_kvaerno5() -> None:
    b21 = make_brehm21_fixed()
    exact = ExactLinearSolver()
    dfx_tsit5 = DiffraxSolver("Tsit5", rtol=1e-13, atol=1e-13)
    dfx_kv5 = DiffraxSolver("Kvaerno5", rtol=1e-11, atol=1e-11)

    t_knots = jnp.linspace(772.0, 779.0, 29)
    q_knots = DEFAULT_Q_STD + 0.5 * jnp.sin(1.7 * (t_knots - 772.0)) * jnp.exp(
        -0.3 * (t_knots - 772.0)
    )
    prod = LinearTrend(slope=0.03, t_ref=773.2, switched_on=True) + GriddedProduction(
        t_knots, q_knots
    )

    # 2D earlywood/latewood sub-windows with custom sub_weights
    years = jnp.arange(773.0, 778.0)
    t_starts_2d = jnp.stack([years + 3.0 / 12.0, years + 6.0 / 12.0], axis=1)
    t_ends_2d = jnp.stack([years + 6.0 / 12.0, years + 9.0 / 12.0], axis=1)
    weights = jnp.array([0.65, 0.35])

    # Non-equilibrium initial condition (t0, y0)
    y_ss = b21.steady_state_c14()
    y0 = y_ss * (1.0 + 0.002 * jnp.linspace(-1.0, 1.0, b21.n_boxes))

    pred_exact = exact.predict_d14c(
        b21,
        prod,
        t_starts_2d,
        t_ends_2d,
        sub_weights=weights,
        box_indices=12,
        t0=772.0,
        y0=y0,
    )
    pred_tsit5 = dfx_tsit5.predict_d14c(
        b21,
        prod,
        t_starts_2d,
        t_ends_2d,
        sub_weights=weights,
        box_indices=12,
        t0=772.0,
        y0=y0,
    )
    pred_kv5 = dfx_kv5.predict_d14c(
        b21,
        prod,
        t_starts_2d,
        t_ends_2d,
        sub_weights=weights,
        box_indices=12,
        t0=772.0,
        y0=y0,
    )
    assert float(jnp.max(jnp.abs(pred_exact - pred_tsit5))) < 1e-11
    assert float(jnp.max(jnp.abs(pred_exact - pred_kv5))) < 1e-8


def test_rfft_vs_scan_foh_long_horizon_exact_equivalence() -> None:
    g15 = make_guttler15_fixed()
    A = g15.matrix_A()
    b_scaled = ALPHA_ATOMS_CM2_S_TO_KG_YR * g15.production_vector()

    # N_t = 5000 > 4096 triggers causal 1D rfft branch under method="auto"
    n_t = 5000
    t_grid = jnp.arange(n_t, dtype=jnp.float64) * 0.25
    dq = 0.35 * jnp.sin(2.0 * math.pi * t_grid / 11.0) + 0.2 * jnp.cos(
        2.0 * math.pi * t_grid / 208.0
    )
    dy0 = jnp.linspace(-0.8, 1.2, g15.n_boxes)

    y_scan = propagate_foh_uniform(A, b_scaled, 0.25, dq, dy0=dy0, method="scan")
    y_rfft = propagate_foh_uniform(A, b_scaled, 0.25, dq, dy0=dy0, method="rfft")
    y_auto = propagate_foh_uniform(A, b_scaled, 0.25, dq, dy0=dy0, method="auto")

    np.testing.assert_allclose(y_rfft, y_scan, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(y_auto, y_rfft, rtol=0.0, atol=1e-14)

    # Also test ExactLinearSolver binned FOH on LogGPProduction across scan vs rfft
    kernel = tk.quasisep.Matern32(sigma=0.25, scale=6.0)
    z = jnp.sin(jnp.arange(1, 25, dtype=jnp.float64)) * 0.4
    lgp = LogGPProduction(kernel, z, domain=(760.0, 785.0))
    t_s = jnp.arange(765.0, 782.0) + 0.25
    t_e = t_s + 0.5
    s_scan = ExactLinearSolver(oversample=24, foh_method="scan")
    s_rfft = ExactLinearSolver(oversample=24, foh_method="rfft")
    p_scan = s_scan.predict_d14c(g15, lgp, t_s, t_e, box_indices=1)
    p_rfft = s_rfft.predict_d14c(g15, lgp, t_s, t_e, box_indices=1)
    np.testing.assert_allclose(p_rfft, p_scan, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "profile",
    ["boxcar", "supergaussian", "skewgaussian", "gamma"],
)
def test_clamped_spike_gradients_across_windows_and_zero_duration_dirac_limit(
    profile: str,
) -> None:
    g15 = make_guttler15_fixed()
    solver = ExactLinearSolver()
    years = jnp.arange(773.0, 779.0)
    t_starts = years + 4.0 / 12.0
    t_ends = years + 8.0 / 12.0

    # Spike straddling the growing-season boundary t_ends[1] = 774 + 8/12 = 774.6667
    def loss_spike(ts: jax.Array, dur: jax.Array, fl: jax.Array) -> jax.Array:
        sp = MiyakeSpike(t_start=ts, duration=dur, fluence=fl, profile=profile)
        pred = solver.predict_d14c(g15, sp, t_starts, t_ends, box_indices=1)
        return jnp.sum(jnp.square(pred))

    # Parameterize offset around 774.55 so FD step inside check_grads avoids 774.55 ULP loss
    check_grads(
        lambda dt_s, dur, fl: loss_spike(774.55 + dt_s, dur, fl),
        (jnp.asarray(0.0), jnp.asarray(0.24), jnp.asarray(6.4)),
        order=1,
        modes=["fwd", "rev"],
        rtol=5e-6,
        atol=5e-6,
    )

    # Zero-duration / tiny-duration Dirac impulse limit & NaN-free reverse autodiff
    t_imp = 774.52
    fl_imp = 6.5
    val_zero, grads_zero = jax.value_and_grad(loss_spike, argnums=(0, 1, 2))(
        jnp.asarray(t_imp), jnp.asarray(0.0), jnp.asarray(fl_imp)
    )
    val_tiny, grads_tiny = jax.value_and_grad(loss_spike, argnums=(0, 1, 2))(
        jnp.asarray(t_imp), jnp.asarray(1e-14), jnp.asarray(fl_imp)
    )
    assert jnp.isfinite(val_zero)
    assert jnp.isfinite(val_tiny)
    for g in (*grads_zero, *grads_tiny):
        assert bool(jnp.all(jnp.isfinite(g)))

    # Compare duration=0 against exact analytical Dirac response fluence * exp(A*(t - t_imp)) b
    sp_zero = MiyakeSpike(t_start=t_imp, duration=0.0, fluence=fl_imp, profile=profile)
    t_post = jnp.array([774.8, 775.5, 777.0])
    dy_zero = solver.solve_delta_trajectory(g15, sp_zero, t_post)
    A = g15.matrix_A()
    b_scaled = ALPHA_ATOMS_CM2_S_TO_KG_YR * g15.production_vector()
    dy_dirac = jax.vmap(
        lambda ti: fl_imp * (jax.scipy.linalg.expm(A * (ti - t_imp)) @ b_scaled)
    )(t_post)
    np.testing.assert_allclose(dy_zero, dy_dirac, rtol=1e-12, atol=1e-12)


def test_diffrax_time_varying_masses_suess_dilution() -> None:
    g15 = make_guttler15_fixed()
    m0 = g15.reservoir_masses()
    K0 = g15.rate_matrix_K()

    # Inject 14C-free fossil 12C smoothly into Troposphere (index 1) after 1850 CE
    def masses_at(t: jax.Array) -> jax.Array:
        dt = jnp.maximum(t - 1850.0, 0.0)
        added_gtc = 120.0 * (dt / 100.0) ** 2
        return m0.at[1].add(added_gtc)

    # Keep physical carbon fluxes F0 fixed so exit rates scale as M0 / M(t)
    def matrix_A_at(t: jax.Array) -> jax.Array:
        scale = m0 / masses_at(t)
        Kt = K0 * scale[None, :]
        return Kt - g15.decay_rate * jnp.eye(g15.n_boxes, dtype=jnp.float64)

    suess_solver = DiffraxSolver(
        "Tsit5",
        rtol=1e-10,
        atol=1e-10,
        masses_fn=masses_at,
        matrix_A_fn=matrix_A_at,
    )
    years = jnp.array([1845.0, 1900.0, 1950.0])
    d14c = suess_solver.predict_d14c(
        g15,
        SteadyProduction(DEFAULT_Q_STD),
        years + 0.25,
        years + 0.75,
        box_indices=1,
        t0=1840.0,
    )
    # Pre-1850 is 0 permil; 1900 and 1950 exhibit progressive negative Suess dilution
    assert abs(float(d14c[0])) < 1e-9
    assert float(d14c[1]) < -5.0
    assert float(d14c[2]) < float(d14c[1]) - 15.0


def test_modax_lazy_import_and_custom_vjp_jvp_only_wrapper() -> None:
    g15 = make_guttler15_fixed()
    modax_solver = ModaxSolver()
    with pytest.raises(ImportError, match="modax-solvers"):
        modax_solver.solve_delta_trajectory(
            g15, SteadyProduction(DEFAULT_Q_STD), jnp.array([774.0, 775.0])
        )

    # Simulate a CUDA FFI solver primitive that defines ONLY forward-mode JVP
    # (no transpose rule -> raw jax.grad raises NotImplementedError / TypeError)
    exact = ExactLinearSolver()
    t_s = jnp.array([774.2, 775.2, 776.2])
    t_e = t_s + 0.4
    y_obs = jnp.array([2.0, 12.5, 9.8])

    @jax.custom_jvp
    def jvp_only_forward(theta: jax.Array) -> jax.Array:
        sp = MiyakeSpike(t_start=theta[0], duration=theta[1], fluence=theta[2])
        return exact.predict_d14c(g15, sp, t_s, t_e, box_indices=1)

    @jvp_only_forward.defjvp
    def _jvp_rule(primals, tangents):
        (theta,) = primals
        (dtheta,) = tangents
        return jax.jvp(
            lambda p: exact.predict_d14c(
                g15,
                MiyakeSpike(t_start=p[0], duration=p[1], fluence=p[2]),
                t_s,
                t_e,
                box_indices=1,
            ),
            (theta,),
            (dtheta,),
        )

    def raw_loglike(params: dict[str, jax.Array]) -> jax.Array:
        theta = jnp.stack([params["t_start"], params["duration"], params["fluence"]])
        pred = jvp_only_forward(theta)
        return -0.5 * jnp.sum(jnp.square((pred - y_obs) / 1.5))

    wrapped_loglike = ModaxSolver.wrap_log_likelihood(raw_loglike)
    p0 = {
        "t_start": jnp.asarray(774.55),
        "duration": jnp.asarray(0.22),
        "fluence": jnp.asarray(6.4),
    }

    val, grad_dict = jax.value_and_grad(wrapped_loglike)(p0)
    # Compare against standard reverse-mode JAX gradient on ExactLinearSolver
    def ref_loglike(params: dict[str, jax.Array]) -> jax.Array:
        sp = MiyakeSpike(
            t_start=params["t_start"],
            duration=params["duration"],
            fluence=params["fluence"],
        )
        pred = exact.predict_d14c(g15, sp, t_s, t_e, box_indices=1)
        return -0.5 * jnp.sum(jnp.square((pred - y_obs) / 1.5))

    val_ref, grad_ref = jax.value_and_grad(ref_loglike)(p0)
    assert float(val) == pytest.approx(float(val_ref), rel=1e-14, abs=1e-14)
    for k in p0:
        np.testing.assert_allclose(grad_dict[k], grad_ref[k], rtol=1e-11, atol=1e-11)

    # Verify outer jax.vmap compatibility (as used in BlackJAX SMC particle mutations)
    batch_p = {
        "t_start": jnp.array([774.45, 774.60, 774.75]),
        "duration": jnp.array([0.18, 0.25, 0.35]),
        "fluence": jnp.array([6.0, 6.5, 7.0]),
    }
    vals_b, grads_b = jax.vmap(jax.value_and_grad(wrapped_loglike))(batch_p)
    vals_ref_b, grads_ref_b = jax.vmap(jax.value_and_grad(ref_loglike))(batch_p)
    np.testing.assert_allclose(vals_b, vals_ref_b, rtol=1e-14, atol=1e-14)
    for k in batch_p:
        np.testing.assert_allclose(grads_b[k], grads_ref_b[k], rtol=1e-11, atol=1e-11)
