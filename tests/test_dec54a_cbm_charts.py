"""Tests for Decision 5.4 Part 1: CarbonBoxModel + core CBMChart hierarchy."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

jax.config.update("jax_enable_x64", True)

from ticktack.cbm import (  # noqa: E402
    BalancedCBM,
    BalancedChart,
    CBMChart,
    CarbonBoxModel,
    EffectiveCBM,
    FixedCBM,
    FixedChart,
    OpenCBM,
    OpenChart,
    ReversibleCBM,
    ReversibleChart,
    SubspaceCBM,
    Topology,
)
from ticktack.fisher import cbm_fisher_spectrum  # noqa: E402
from ticktack.likelihood import ParamSpec, ParameterSpace  # noqa: E402
from ticktack.model import RadiocarbonModel  # noqa: E402
from ticktack.presets import load_preset  # noqa: E402
from ticktack.production import HarmonicSolar, MiyakeSpike, SteadyProduction  # noqa: E402
from ticktack.solvers import ExactLinearSolver  # noqa: E402


def _asym_3box_topo() -> Topology:
    return Topology(
        box_names=("Stratosphere", "Troposphere", "DeepOcean"),
        edges_src=(0, 1, 1, 2, 2),
        edges_dst=(1, 0, 2, 1, 0),
        prod_box_indices=(0, 1),
        trop_box_indices=(1,),
    )


def _sym_3box_topo() -> Topology:
    return Topology(
        box_names=("Stratosphere", "Troposphere", "Ocean"),
        edges_src=(0, 1, 1, 2),
        edges_dst=(1, 0, 2, 1),
        prod_box_indices=(0, 1),
        trop_box_indices=(1,),
    )


def test_cbm_chart_abstract_interface_and_flags() -> None:
    """Verify abstract CBMChart defaults, TypeError guard, and detailed-balance/subspace flags."""
    topo_sym = _sym_3box_topo()
    topo_asym = _asym_3box_topo()

    class _MinimalChart(CBMChart):
        pass

    chart0 = _MinimalChart()
    assert chart0.is_detailed_balance is False
    assert chart0.is_subspace is False
    assert chart0.param_specs() == ()
    np.testing.assert_array_equal(chart0.sink_rates(topo_sym), np.zeros(3))
    np.testing.assert_array_equal(chart0.active_parameter_mask(), np.zeros(0, dtype=bool))
    with pytest.raises(NotImplementedError):
        chart0.edge_rates(topo_sym)
    with pytest.raises(NotImplementedError):
        chart0.rate_matrix_K(topo_sym)
    with pytest.raises(NotImplementedError):
        chart0.reservoir_masses(topo_sym)
    with pytest.raises(NotImplementedError):
        chart0.production_vector(topo_sym)
    with pytest.raises(NotImplementedError):
        chart0.relative_reservoir_masses(topo_sym)

    with pytest.raises(TypeError, match="CBMChart"):
        CarbonBoxModel(topo_sym, "not_a_chart")  # type: ignore[arg-type]

    c_fix = CarbonBoxModel(topo_asym, FixedChart(topo_asym, [91.0, 520.0, 3400.0], [46.0, 38.0, 82.0, 74.0, 8.0], prod_weights=(0.69, 0.31)))
    c_bal = CarbonBoxModel(topo_asym, BalancedChart(topo_asym, [-0.62, -2.55, -1.84, -3.72, -5.91], 8.31, prod_weights=(0.69, 0.31)))
    c_rev = CarbonBoxModel(topo_sym, ReversibleChart(topo_sym, np.log([93.0, 515.0, 2950.0]), np.log([46.5, 92.0]), prod_weights=(0.71, 0.29)))
    c_opn = CarbonBoxModel(topo_asym, OpenChart(topo_asym, [-0.62, -2.55, -1.84, -3.72, -5.91], 8.31, [-8.15], open_box_indices=(2,), prod_weights=(0.69, 0.31)))
    c_eff = EffectiveCBM(K=3)
    c_sub = SubspaceCBM(topo_asym, [-0.62, -2.55, -1.84, -3.72, -5.91], np.ones((5, 2)) * 0.2, 8.31, prod_weights=(0.69, 0.31))

    assert ReversibleChart.is_detailed_balance is True
    assert c_rev.is_detailed_balance is True
    assert ReversibleCBM(topo_sym, np.log([93.0, 515.0, 2950.0]), np.log([46.5, 92.0])).is_detailed_balance is True
    assert c_eff.is_detailed_balance is True
    for m in (c_fix, c_bal, c_opn, c_sub):
        assert m.is_detailed_balance is False

    assert c_sub.is_subspace is True
    for m in (c_fix, c_bal, c_rev, c_opn, c_eff):
        assert m.is_subspace is False


def test_single_leaf_uniqueness_and_dual_zodiax_paths() -> None:
    """Verify zero duplicate leaves and identical 'cbm.<leaf>' vs 'cbm.chart.<leaf>' autodiff/updates."""
    topo_asym = _asym_3box_topo()
    topo_sym = _sym_3box_topo()
    prod = SteadyProduction(1.83) + MiyakeSpike(t_start=774.58, duration=0.27, fluence=6.7, profile="boxcar")
    t_s = jnp.array([773.3, 774.3, 775.3, 777.3], dtype=jnp.float64)
    t_e = t_s + 0.45
    solver = ExactLinearSolver()
    weights = jnp.array([1.2, -0.85, 0.65, 1.4], dtype=jnp.float64)

    pairs = [
        (
            CarbonBoxModel(topo_asym, FixedChart(topo_asym, [92.0, 535.0, 3150.0], [47.0, 39.0, 84.0, 76.0, 8.0], prod_weights=(0.68, 0.32), sink_rates=[0.0, 0.0, 1.5e-4])),
            FixedCBM(topo_asym, [92.0, 535.0, 3150.0], [47.0, 39.0, 84.0, 76.0, 8.0], prod_weights=(0.68, 0.32), sink_rates=[0.0, 0.0, 1.5e-4]),
            "masses",
            jnp.array([96.0, 548.0, 3220.0], dtype=jnp.float64),
            3,
        ),
        (
            CarbonBoxModel(topo_asym, BalancedChart(topo_asym, [-0.61, -2.48, -1.87, -3.68, -5.77], 8.24, prod_weights=(0.68, 0.32))),
            BalancedCBM(topo_asym, [-0.61, -2.48, -1.87, -3.68, -5.77], 8.24, prod_weights=(0.68, 0.32)),
            "log_rates_free",
            jnp.array([-0.49, -2.31, -1.95, -3.52, -5.61], dtype=jnp.float64),
            2,
        ),
        (
            CarbonBoxModel(topo_sym, ReversibleChart(topo_sym, np.log([91.0, 528.0, 2890.0]), np.log([45.5, 96.0]), prod_weights=(0.73, 0.27))),
            ReversibleCBM(topo_sym, np.log([91.0, 528.0, 2890.0]), np.log([45.5, 96.0]), prod_weights=(0.73, 0.27)),
            "log_sym_fluxes",
            jnp.log(jnp.array([51.0, 88.5], dtype=jnp.float64)),
            2,
        ),
        (
            CarbonBoxModel(topo_asym, OpenChart(topo_asym, [-0.61, -2.48, -1.87, -3.68, -5.77], 8.24, [-8.35], open_box_indices=(2,), prod_weights=(0.68, 0.32))),
            OpenCBM(topo_asym, [-0.61, -2.48, -1.87, -3.68, -5.77], 8.24, [-8.35], open_box_indices=(2,), prod_weights=(0.68, 0.32)),
            "log_sink_rates",
            jnp.array([-7.65], dtype=jnp.float64),
            3,
        ),
    ]

    for cbm_direct, cbm_wrap, leaf_name, new_val, expected_n_leaves in pairs:
        for cbm in (cbm_direct, cbm_wrap):
            # Confirm leaf appears strictly once in PyTree and not duplicated in cbm.__dict__.
            leaves = jax.tree_util.tree_leaves(eqx.filter(cbm, eqx.is_inexact_array))
            assert len(leaves) == expected_n_leaves
            assert leaf_name not in cbm.__dict__
            assert not hasattr(cbm, "__nonexistent_dunder__")
            with pytest.raises(AttributeError):
                _ = cbm.nonexistent_attr_xyz

            rm = RadiocarbonModel(cbm, prod)
            np.testing.assert_array_equal(rm.get(f"cbm.{leaf_name}"), rm.get(f"cbm.chart.{leaf_name}"))

            rm_short = rm.set(f"cbm.{leaf_name}", new_val)
            rm_long = rm.set(f"cbm.chart.{leaf_name}", new_val)
            np.testing.assert_array_equal(rm_short.get(f"cbm.{leaf_name}"), new_val)
            np.testing.assert_array_equal(rm_short.cbm.matrix_A(), rm_long.cbm.matrix_A())
            assert len(jax.tree_util.tree_leaves(eqx.filter(rm_short.cbm, eqx.is_inexact_array))) == expected_n_leaves

            # Verify eqx.filter_grad matches central finite differences and forwards via __getattr__.
            def _loss(m: RadiocarbonModel) -> jax.Array:
                pred = solver.predict_d14c(m.cbm, m.production, t_s, t_e, box_indices=1, q_std=1.83)
                return jnp.sum(weights * pred)

            g_rm = eqx.filter_grad(_loss)(rm)
            g_short = np.asarray(getattr(g_rm.cbm, leaf_name), dtype=np.float64)
            g_chart = np.asarray(getattr(g_rm.cbm.chart, leaf_name), dtype=np.float64)
            np.testing.assert_array_equal(g_short, g_chart)

            base_leaf = np.asarray(rm.get(f"cbm.{leaf_name}"), dtype=np.float64)
            eps = 1e-6 * np.maximum(np.abs(base_leaf), 1.0)
            fd = np.empty_like(base_leaf)
            for idx in range(base_leaf.size):
                p_vec = base_leaf.reshape(-1).copy()
                m_vec = base_leaf.reshape(-1).copy()
                step = float(eps.reshape(-1)[idx])
                p_vec[idx] += step
                m_vec[idx] -= step
                f_p = float(_loss(rm.set(f"cbm.{leaf_name}", jnp.asarray(p_vec.reshape(base_leaf.shape)))))
                f_m = float(_loss(rm.set(f"cbm.{leaf_name}", jnp.asarray(m_vec.reshape(base_leaf.shape)))))
                fd.reshape(-1)[idx] = (f_p - f_m) / (2.0 * step)
            np.testing.assert_allclose(g_short, fd, rtol=1e-5, atol=1e-6)
            assert np.linalg.norm(g_short) > 0.0


@pytest.mark.parametrize(
    ("preset_name", "chart_kind", "closure", "tie_sym", "expected_chart_cls"),
    [
        ("Guttler15", "fixed", "open_ocean_sink", False, FixedChart),
        ("Guttler15", "balanced", "open_ocean_sink", False, BalancedChart),
        ("Brehm21", "fixed", "open_ocean_sink", False, FixedChart),
        ("Brehm21", "balanced", "open_ocean_sink", False, BalancedChart),
        ("Brehm21", "balanced", "open_ocean_sink", True, BalancedChart),
        ("Buntgen18", "balanced", "open_ocean_sink", True, BalancedChart),
        ("Miyake17", "fixed", "open_ocean_sink", False, FixedChart),
        ("Miyake17", "open", "open_ocean_sink", False, OpenChart),
        ("Miyake17", "reversible", "closed_shallow", False, ReversibleChart),
        ("Miyake17", "balanced", "closed_shallow", False, BalancedChart),
    ],
)
def test_load_preset_chart_vs_kind_parity_and_kirchhoff(
    preset_name: str,
    chart_kind: str,
    closure: str,
    tie_sym: bool,
    expected_chart_cls: type[CBMChart],
) -> None:
    """Verify load_preset(chart=...) returns concrete CarbonBoxModel matching kind=... and ExactLinearSolver."""
    cbm_chart = load_preset(
        preset_name,
        chart=chart_kind,  # type: ignore[arg-type]
        miyake17_closure=closure,  # type: ignore[arg-type]
        tie_symmetric_edges=tie_sym,
    )
    cbm_kind = load_preset(
        preset_name,
        kind=chart_kind,  # type: ignore[arg-type]
        miyake17_closure=closure,  # type: ignore[arg-type]
        tie_symmetric_edges=tie_sym,
    )
    assert type(cbm_chart) is CarbonBoxModel
    assert isinstance(cbm_chart.chart, expected_chart_cls)
    assert isinstance(cbm_kind, CarbonBoxModel)
    assert type(cbm_kind) is not CarbonBoxModel

    np.testing.assert_array_equal(cbm_chart.reservoir_masses(), cbm_kind.reservoir_masses())
    np.testing.assert_array_equal(cbm_chart.rate_matrix_K(), cbm_kind.rate_matrix_K())
    np.testing.assert_array_equal(cbm_chart.sink_rates(), cbm_kind.sink_rates())
    np.testing.assert_array_equal(cbm_chart.matrix_A(), cbm_kind.matrix_A())
    np.testing.assert_array_equal(cbm_chart.steady_state_d14c(), cbm_kind.steady_state_d14c())

    if chart_kind in ("balanced", "reversible", "open"):
        assert float(jnp.max(jnp.abs(cbm_chart.kirchhoff_residual()))) < 1e-12

    solver = ExactLinearSolver()
    profile = "supergaussian" if preset_name == "Miyake17" else "boxcar"
    prod = (
        SteadyProduction(1.79)
        + MiyakeSpike(t_start=774.62, duration=0.31, fluence=6.4, profile=profile)  # type: ignore[arg-type]
        + HarmonicSolar(periods=(11.0,), c_cos=(0.18,), c_sin=(-0.11,))
    )
    t_s = jnp.arange(772.0, 779.0, dtype=jnp.float64) + 0.28
    t_e = t_s + 0.48
    trop_box = cbm_chart.topology.trop_box_indices[0]
    d14c_chart = solver.predict_d14c(cbm_chart, prod, t_s, t_e, box_indices=trop_box, q_std=1.79)
    d14c_kind = solver.predict_d14c(cbm_kind, prod, t_s, t_e, box_indices=trop_box, q_std=1.79)
    np.testing.assert_allclose(d14c_chart, d14c_kind, rtol=1e-13, atol=1e-13)


def test_load_preset_error_guards_and_random_kirchhoff_invariance() -> None:
    """Verify conflicting chart/kind errors and <1e-12 Kirchhoff residual at random off-origin draws."""
    with pytest.raises(ValueError, match="Conflicting"):
        load_preset("Brehm21", "fixed", chart="balanced")
    with pytest.raises(ValueError, match="open sink_rates"):
        load_preset("Miyake17", chart="balanced")
    with pytest.raises(ValueError, match="open sink_rates"):
        load_preset("Miyake17", chart="reversible")
    with pytest.raises(ValueError, match="Unknown chart/kind"):
        load_preset("Brehm21", chart="bogus")  # type: ignore[arg-type]

    rng = np.random.default_rng(5401)
    b21 = load_preset("Brehm21", chart="balanced")
    pert = b21.set("log_rates_free", b21.log_rates_free + 0.45 * rng.standard_normal(b21.log_rates_free.shape))
    assert float(jnp.max(jnp.abs(pert.kirchhoff_residual()))) < 1e-12


def test_active_parameter_mask_fisher_spectrum_and_subspace_compat() -> None:
    """Verify gauge-mask auto-detection in cbm_fisher_spectrum and SubspaceCBM.from_fisher compatibility."""
    cbm_bal_chart = load_preset("Guttler15", chart="balanced")
    cbm_bal_kind = load_preset("Guttler15", kind="balanced")
    cbm_open_chart = load_preset("Miyake17", chart="open")

    expected_bal_mask = np.array([True] * 24 + [False], dtype=bool)
    expected_open_mask = np.array([True] * 6 + [False, True], dtype=bool)
    np.testing.assert_array_equal(cbm_bal_chart.active_parameter_mask(), expected_bal_mask)
    np.testing.assert_array_equal(cbm_open_chart.active_parameter_mask(), expected_open_mask)

    solver = ExactLinearSolver()
    t_s = jnp.arange(770.0, 784.0, dtype=jnp.float64) + 0.33
    t_e = t_s + 0.42
    sigma = jnp.full(t_s.shape, 1.35, dtype=jnp.float64)
    psi0 = {"spike": MiyakeSpike(774.55, 0.32, 6.6, profile="boxcar"), "offset": jnp.asarray(-0.75)}

    def _fwd(c: CarbonBoxModel, psi: dict[str, jax.Array | MiyakeSpike]) -> jax.Array:
        p = SteadyProduction(1.76) + psi["spike"]  # type: ignore[operator]
        return solver.predict_d14c(c, p, t_s, t_e, box_indices=1) + psi["offset"]  # type: ignore[operator]

    spec_chart = cbm_fisher_spectrum(_fwd, cbm_bal_chart, sigma, psi_nuisance=psi0, prior_cbm_sd=0.37)
    spec_kind = cbm_fisher_spectrum(_fwd, cbm_bal_kind, sigma, psi_nuisance=psi0, prior_cbm_sd=0.37)
    np.testing.assert_array_equal(np.asarray(spec_chart.active_mask), expected_bal_mask)
    np.testing.assert_allclose(spec_chart.eigenvalues, spec_kind.eigenvalues, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(spec_chart.d_eff, spec_kind.d_eff, rtol=1e-13, atol=1e-14)

    spec_open = cbm_fisher_spectrum(_fwd, cbm_open_chart, sigma, psi_nuisance=psi0, prior_cbm_sd=0.37)
    np.testing.assert_array_equal(np.asarray(spec_open.active_mask), expected_open_mask)

    # Confirm test sensitivity: disabling gauge exclusion adds the exact zero-Fisher log_total_mass mode.
    spec_unmasked = cbm_fisher_spectrum(
        _fwd, cbm_bal_chart, sigma, psi_nuisance=psi0, prior_cbm_sd=jnp.full((25,), 0.37), exclude_total_mass=False
    )
    assert int(jnp.sum(spec_unmasked.active_mask)) == 25
    assert float(spec_unmasked.eigenvalues[-1]) < 1e-24

    sub = SubspaceCBM.from_fisher(cbm_bal_chart, spec_chart, n_modes=3)
    assert sub.n_modes == 3
    np.testing.assert_allclose(sub.steady_state_d14c(), cbm_bal_chart.steady_state_d14c(), rtol=1e-13, atol=1e-13)
    with pytest.raises(TypeError, match="BalancedCBM"):
        SubspaceCBM.from_fisher(load_preset("Guttler15", chart="fixed"), spec_chart, n_modes=2)  # type: ignore[arg-type]


def test_from_boxes_conversions_residence_times_summary_and_param_specs() -> None:
    """Verify CarbonBoxModel.from_boxes, to_fixed/to_balanced/with_chart, residence_times, summary, and param_specs."""
    masses_asym = {"Stratosphere": 94.0, "Troposphere": 525.0, "DeepOcean": 3600.0}
    fluxes_asym = [
        ("Stratosphere", "Troposphere", 47.0),
        ("Troposphere", "Stratosphere", 39.0),
        ("Troposphere", "DeepOcean", 84.0),
        ("DeepOcean", "Troposphere", 76.0),
        ("DeepOcean", "Stratosphere", 8.0),
    ]
    cbm_bal = CarbonBoxModel.from_boxes(
        masses=masses_asym,
        fluxes=fluxes_asym,
        production={"Stratosphere": 0.72, "Troposphere": 0.28},
        chart="balanced",
    )
    assert type(cbm_bal) is CarbonBoxModel
    assert isinstance(cbm_bal.chart, BalancedChart)
    assert float(jnp.max(jnp.abs(cbm_bal.kirchhoff_residual()))) < 1e-12
    np.testing.assert_allclose(cbm_bal.reservoir_masses(), [94.0, 525.0, 3600.0], rtol=1e-13)
    np.testing.assert_allclose(cbm_bal.residence_times(), [94.0 / 47.0, 525.0 / 123.0, 3600.0 / 84.0], rtol=1e-13)

    cbm_open = CarbonBoxModel.from_boxes(
        masses=masses_asym,
        fluxes=fluxes_asym,
        production=[0.72, 0.28, 0.0],
        sinks={"DeepOcean": 2.5e-4},
        chart="open",
    )
    assert isinstance(cbm_open.chart, OpenChart)
    expected_tau_open = [94.0 / 47.0, 525.0 / 123.0, 1.0 / (84.0 / 3600.0 + 2.5e-4)]
    np.testing.assert_allclose(cbm_open.residence_times(), expected_tau_open, rtol=1e-13)
    with pytest.raises(ValueError, match="positive sinks"):
        cbm_open.to_balanced()

    # Round-trip conversions: Balanced -> Fixed -> Balanced and with_chart
    cbm_fix = cbm_bal.to_fixed()
    assert isinstance(cbm_fix.chart, FixedChart)
    cbm_bal2 = cbm_fix.to_balanced()
    assert isinstance(cbm_bal2.chart, BalancedChart)
    np.testing.assert_allclose(cbm_bal2.matrix_A(), cbm_bal.matrix_A(), rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(cbm_fix.with_chart(cbm_bal.chart).matrix_A(), cbm_bal.matrix_A(), rtol=0.0, atol=0.0)

    # Symmetric reversible custom model + asymmetric flux rejection
    cbm_rev = CarbonBoxModel.from_boxes(
        masses={"Stratosphere": 90.0, "Troposphere": 510.0, "Ocean": 38000.0},
        fluxes=[
            ("Stratosphere", "Troposphere", 45.0),
            ("Troposphere", "Stratosphere", 45.0),
            ("Troposphere", "Ocean", 60.0),
            ("Ocean", "Troposphere", 60.0),
        ],
        production={"Stratosphere": 0.7, "Troposphere": 0.3},
        chart="reversible",
    )
    assert cbm_rev.is_detailed_balance is True
    assert float(jnp.max(jnp.abs(cbm_rev.kirchhoff_residual()))) < 1e-12
    with pytest.raises(ValueError, match="equal forward and reverse"):
        CarbonBoxModel.from_boxes(
            masses={"Stratosphere": 90.0, "Troposphere": 510.0},
            fluxes=[("Stratosphere", "Troposphere", 45.0), ("Troposphere", "Stratosphere", 41.0)],
            chart="reversible",
        )
    with pytest.raises(ValueError, match="positive sinks"):
        CarbonBoxModel.from_boxes(
            masses={"Stratosphere": 90.0, "Troposphere": 510.0},
            fluxes=[("Stratosphere", "Troposphere", 45.0), ("Troposphere", "Stratosphere", 45.0)],
            sinks={"Troposphere": 0.01},
            chart="balanced",
        )
    with pytest.raises(KeyError):
        CarbonBoxModel.from_boxes(
            masses={"Stratosphere": 90.0, "Troposphere": 510.0},
            fluxes=[("Stratosphere", "Mars", 10.0)],
        )
    # Mixed or malformed reservoir keys must raise ValueError instead of silent corruption
    for bad_masses in (
        {"Stratosphere": 45.0, ("Troposphere", "north"): 250.0},
        {("Troposphere", "north"): 250.0, "Snow": 45.0},
        {("Troposphere", "north", "extra"): 250.0, ("Stratosphere", "north", "extra"): 45.0},
    ):
        with pytest.raises(ValueError, match="Reservoir keys"):
            CarbonBoxModel.from_boxes(masses=bad_masses, fluxes=[])

    # Custom CBMChart with multi-element ndarray prod_weights must not trigger bool(ndarray) in to_fixed/to_balanced
    class _ArrayPWChart(CBMChart):
        prod_weights: jax.Array

        def __init__(self) -> None:
            self.prod_weights = jnp.array([0.72, 0.28], dtype=jnp.float64)

        def reservoir_masses(self, topology: Topology) -> jax.Array:
            return jnp.array([90.0, 510.0, 38000.0], dtype=jnp.float64)

        def edge_rates(self, topology: Topology) -> jax.Array:
            return jnp.array([45.0 / 90.0, 45.0 / 510.0, 60.0 / 510.0, 60.0 / 38000.0], dtype=jnp.float64)

    custom_cbm = CarbonBoxModel(_sym_3box_topo(), _ArrayPWChart())
    np.testing.assert_allclose(custom_cbm.to_fixed().production_vector(), [0.72, 0.28, 0.0], rtol=1e-14)
    np.testing.assert_allclose(custom_cbm.to_balanced().production_vector(), [0.72, 0.28, 0.0], rtol=1e-14)

    # Hemispheric custom model via (name, hemisphere) keys + summary()
    cbm_hemi = CarbonBoxModel.from_boxes(
        masses={
            ("Stratosphere", "south"): 45.0,
            ("Troposphere", "south"): 250.0,
            ("Stratosphere", "north"): 45.0,
            ("Troposphere", "north"): 250.0,
        },
        fluxes=[
            (("Stratosphere", "south"), ("Troposphere", "south"), 22.0),
            (("Troposphere", "south"), ("Stratosphere", "south"), 22.0),
            (("Troposphere", "south"), ("Troposphere", "north"), 70.0),
            (("Troposphere", "north"), ("Troposphere", "south"), 70.0),
            (("Stratosphere", "north"), ("Troposphere", "north"), 22.0),
            (("Troposphere", "north"), ("Stratosphere", "north"), 22.0),
        ],
        chart="balanced",
    )
    table = cbm_hemi.summary()
    assert "BalancedChart" in table and "Troposphere (N)" in table and "Troposphere (S)" in table

    # param_specs() + ParameterSpace round-trip, explicit prior_sd enforcement (Decision 5.6), and Gaussian log-prior
    spike = MiyakeSpike(774.5, 0.3, 6.2)
    sp_spec = ParamSpec("production.spike.fluence", "exp", math.log(6.2), 0.45)
    assert cbm_fix.param_specs() == ()
    for c_req in (cbm_bal, cbm_rev, cbm_open):
        with pytest.raises(ValueError, match="prior_sd"):
            c_req.param_specs()
    with pytest.raises(ValueError, match="prior_sd"):
        cbm_bal.param_specs(prior_sd=0.0)
    with pytest.raises(ValueError, match="prior_sd"):
        cbm_bal.param_specs(prior_sd=float("inf"))
    sub_default = SubspaceCBM(_asym_3box_topo(), cbm_bal.log_rates_free, np.ones((5, 2)) * 0.1, float(cbm_bal.log_total_mass))
    assert sub_default.param_specs()[0].prior_sd == pytest.approx(1.0)

    for cbm_case, kwargs, expected_cbm_dim in [
        (cbm_bal, {"prior_sd": 0.37}, 5),
        (cbm_rev, {"prior_sd": 0.41, "include_masses": True}, 5),
        (cbm_rev, {"prior_sd": 0.41, "include_masses": False}, 2),
        (cbm_open, {"prior_sd": 0.33, "sink_sd": 0.52}, 6),
    ]:
        rm0 = RadiocarbonModel(cbm_case, SteadyProduction(1.76) + spike)
        ps = ParameterSpace(rm0, [*cbm_case.param_specs(**kwargs), sp_spec])
        assert ps.n_params == expected_cbm_dim + 1
        u0 = ps.unconstrain(rm0)
        u_step = u0 + jnp.linspace(0.11, -0.23, ps.n_params, dtype=jnp.float64)
        rm_step = ps.constrain(u_step)
        np.testing.assert_allclose(ps.unconstrain(rm_step), u_step, rtol=1e-13, atol=1e-14)

        # Compare unconstrained log-prior against independent analytical normal log-density
        z = (np.asarray(u_step) - np.asarray(ps.prior_means)) / np.asarray(ps.prior_sds)
        expected_lp = float(np.sum(-0.5 * z**2 - np.log(np.asarray(ps.prior_sds)) - 0.5 * math.log(2.0 * math.pi)))
        assert float(ps.log_prior_unconstrained(u_step)) == pytest.approx(expected_lp, rel=1e-14, abs=1e-14)
