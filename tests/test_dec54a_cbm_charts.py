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

    c_fix = CarbonBoxModel(
        topo_asym,
        FixedChart(
            topo_asym,
            [91.0, 520.0, 3400.0],
            [46.0, 38.0, 82.0, 74.0, 8.0],
            prod_weights=(0.69, 0.31),
        ),
    )
    c_bal = CarbonBoxModel(
        topo_asym,
        BalancedChart(
            topo_asym, [-0.62, -2.55, -1.84, -3.72, -5.91], 8.31, prod_weights=(0.69, 0.31)
        ),
    )
    c_rev = CarbonBoxModel(
        topo_sym,
        ReversibleChart(
            topo_sym, np.log([93.0, 515.0, 2950.0]), np.log([46.5, 92.0]), prod_weights=(0.71, 0.29)
        ),
    )
    c_opn = CarbonBoxModel(
        topo_asym,
        OpenChart(
            topo_asym,
            [-0.62, -2.55, -1.84, -3.72, -5.91],
            8.31,
            [-8.15],
            open_box_indices=(2,),
            prod_weights=(0.69, 0.31),
        ),
    )
    c_eff = EffectiveCBM(K=3)
    c_sub = SubspaceCBM(
        topo_asym,
        [-0.62, -2.55, -1.84, -3.72, -5.91],
        np.ones((5, 2)) * 0.2,
        8.31,
        prod_weights=(0.69, 0.31),
    )

    assert ReversibleChart.is_detailed_balance is True
    assert c_rev.is_detailed_balance is True
    assert (
        ReversibleCBM(
            topo_sym, np.log([93.0, 515.0, 2950.0]), np.log([46.5, 92.0])
        ).is_detailed_balance
        is True
    )
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
    prod = SteadyProduction(1.83) + MiyakeSpike(
        t_start=774.58, duration=0.27, fluence=6.7, profile="boxcar"
    )
    t_s = jnp.array([773.3, 774.3, 775.3, 777.3], dtype=jnp.float64)
    t_e = t_s + 0.45
    solver = ExactLinearSolver()
    weights = jnp.array([1.2, -0.85, 0.65, 1.4], dtype=jnp.float64)

    pairs = [
        (
            CarbonBoxModel(
                topo_asym,
                FixedChart(
                    topo_asym,
                    [92.0, 535.0, 3150.0],
                    [47.0, 39.0, 84.0, 76.0, 8.0],
                    prod_weights=(0.68, 0.32),
                    sink_rates=[0.0, 0.0, 1.5e-4],
                ),
            ),
            FixedCBM(
                topo_asym,
                [92.0, 535.0, 3150.0],
                [47.0, 39.0, 84.0, 76.0, 8.0],
                prod_weights=(0.68, 0.32),
                sink_rates=[0.0, 0.0, 1.5e-4],
            ),
            "masses",
            jnp.array([96.0, 548.0, 3220.0], dtype=jnp.float64),
            3,
        ),
        (
            CarbonBoxModel(
                topo_asym,
                BalancedChart(
                    topo_asym, [-0.61, -2.48, -1.87, -3.68, -5.77], 8.24, prod_weights=(0.68, 0.32)
                ),
            ),
            BalancedCBM(
                topo_asym, [-0.61, -2.48, -1.87, -3.68, -5.77], 8.24, prod_weights=(0.68, 0.32)
            ),
            "log_rates_free",
            jnp.array([-0.49, -2.31, -1.95, -3.52, -5.61], dtype=jnp.float64),
            2,
        ),
        (
            CarbonBoxModel(
                topo_sym,
                ReversibleChart(
                    topo_sym,
                    np.log([91.0, 528.0, 2890.0]),
                    np.log([45.5, 96.0]),
                    prod_weights=(0.73, 0.27),
                ),
            ),
            ReversibleCBM(
                topo_sym,
                np.log([91.0, 528.0, 2890.0]),
                np.log([45.5, 96.0]),
                prod_weights=(0.73, 0.27),
            ),
            "log_sym_fluxes",
            jnp.log(jnp.array([51.0, 88.5], dtype=jnp.float64)),
            2,
        ),
        (
            CarbonBoxModel(
                topo_asym,
                OpenChart(
                    topo_asym,
                    [-0.61, -2.48, -1.87, -3.68, -5.77],
                    8.24,
                    [-8.35],
                    open_box_indices=(2,),
                    prod_weights=(0.68, 0.32),
                ),
            ),
            OpenCBM(
                topo_asym,
                [-0.61, -2.48, -1.87, -3.68, -5.77],
                8.24,
                [-8.35],
                open_box_indices=(2,),
                prod_weights=(0.68, 0.32),
            ),
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
            np.testing.assert_array_equal(
                rm.get(f"cbm.{leaf_name}"), rm.get(f"cbm.chart.{leaf_name}")
            )

            rm_short = rm.set(f"cbm.{leaf_name}", new_val)
            rm_long = rm.set(f"cbm.chart.{leaf_name}", new_val)
            np.testing.assert_array_equal(rm_short.get(f"cbm.{leaf_name}"), new_val)
            np.testing.assert_array_equal(rm_short.cbm.matrix_A(), rm_long.cbm.matrix_A())
            assert (
                len(jax.tree_util.tree_leaves(eqx.filter(rm_short.cbm, eqx.is_inexact_array)))
                == expected_n_leaves
            )

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
                f_p = float(
                    _loss(rm.set(f"cbm.{leaf_name}", jnp.asarray(p_vec.reshape(base_leaf.shape))))
                )
                f_m = float(
                    _loss(rm.set(f"cbm.{leaf_name}", jnp.asarray(m_vec.reshape(base_leaf.shape))))
                )
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
    pert = b21.set(
        "log_rates_free", b21.log_rates_free + 0.45 * rng.standard_normal(b21.log_rates_free.shape)
    )
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

    spec_chart = cbm_fisher_spectrum(
        _fwd, cbm_bal_chart, sigma, psi_nuisance=psi0, prior_cbm_sd=0.37
    )
    spec_kind = cbm_fisher_spectrum(_fwd, cbm_bal_kind, sigma, psi_nuisance=psi0, prior_cbm_sd=0.37)
    np.testing.assert_array_equal(np.asarray(spec_chart.active_mask), expected_bal_mask)
    np.testing.assert_allclose(
        spec_chart.eigenvalues, spec_kind.eigenvalues, rtol=1e-13, atol=1e-14
    )
    np.testing.assert_allclose(spec_chart.d_eff, spec_kind.d_eff, rtol=1e-13, atol=1e-14)

    spec_open = cbm_fisher_spectrum(
        _fwd, cbm_open_chart, sigma, psi_nuisance=psi0, prior_cbm_sd=0.37
    )
    np.testing.assert_array_equal(np.asarray(spec_open.active_mask), expected_open_mask)

    # Confirm test sensitivity: disabling gauge exclusion adds the exact zero-Fisher log_total_mass mode.
    spec_unmasked = cbm_fisher_spectrum(
        _fwd,
        cbm_bal_chart,
        sigma,
        psi_nuisance=psi0,
        prior_cbm_sd=jnp.full((25,), 0.37),
        exclude_total_mass=False,
    )
    assert int(jnp.sum(spec_unmasked.active_mask)) == 25
    assert float(spec_unmasked.eigenvalues[-1]) < 1e-24

    sub = SubspaceCBM.from_fisher(cbm_bal_chart, spec_chart, n_modes=3)
    assert sub.n_modes == 3
    np.testing.assert_allclose(
        sub.steady_state_d14c(), cbm_bal_chart.steady_state_d14c(), rtol=1e-13, atol=1e-13
    )
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
    np.testing.assert_allclose(
        cbm_bal.residence_times(), [94.0 / 47.0, 525.0 / 123.0, 3600.0 / 84.0], rtol=1e-13
    )

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
    np.testing.assert_allclose(
        cbm_fix.with_chart(cbm_bal.chart).matrix_A(), cbm_bal.matrix_A(), rtol=0.0, atol=0.0
    )

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
            return jnp.array(
                [45.0 / 90.0, 45.0 / 510.0, 60.0 / 510.0, 60.0 / 38000.0], dtype=jnp.float64
            )

    custom_cbm = CarbonBoxModel(_sym_3box_topo(), _ArrayPWChart())
    np.testing.assert_allclose(
        custom_cbm.to_fixed().production_vector(), [0.72, 0.28, 0.0], rtol=1e-14
    )
    np.testing.assert_allclose(
        custom_cbm.to_balanced().production_vector(), [0.72, 0.28, 0.0], rtol=1e-14
    )

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
    sub_nonzero = SubspaceCBM(
        _asym_3box_topo(),
        cbm_bal.log_rates_free,
        np.ones((5, 2)) * 0.1,
        float(cbm_bal.log_total_mass),
        alpha=np.array([0.8, -0.6], dtype=np.float64),
    )
    sub_spec = sub_nonzero.param_specs()[0]
    assert sub_spec.prior_sd == pytest.approx(1.0)
    assert float(np.asarray(sub_spec.prior_mean)) == pytest.approx(0.0, abs=0.0)
    rm_sub = RadiocarbonModel(sub_nonzero, SteadyProduction(1.76) + spike)
    ps_sub = ParameterSpace(rm_sub, list(sub_nonzero.param_specs(prior_sd=1.25)))
    u_sub = ps_sub.unconstrain(rm_sub)
    np.testing.assert_allclose(u_sub, [0.8, -0.6], rtol=1e-14)
    np.testing.assert_allclose(ps_sub.prior_means, [0.0, 0.0], atol=0.0)
    expected_sub_lp = float(
        np.sum(
            -0.5 * (np.array([0.8, -0.6]) / 1.25) ** 2
            - math.log(1.25)
            - 0.5 * math.log(2.0 * math.pi)
        )
    )
    assert float(ps_sub.log_prior_unconstrained(u_sub)) == pytest.approx(
        expected_sub_lp, rel=1e-14, abs=1e-14
    )

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
        expected_lp = float(
            np.sum(-0.5 * z**2 - np.log(np.asarray(ps.prior_sds)) - 0.5 * math.log(2.0 * math.pi))
        )
        assert float(ps.log_prior_unconstrained(u_step)) == pytest.approx(
            expected_lp, rel=1e-14, abs=1e-14
        )


def test_perron_kirchhoff_tying_and_open_sink_validation() -> None:
    """Threads 1 & 2: reject imbalanced fluxes, inconsistent tying, and invalid sink_rates."""
    topo_asym = _asym_3box_topo()
    masses = [94.0, 525.0, 3600.0]
    balanced_fluxes = [47.0, 39.0, 84.0, 76.0, 8.0]
    imbalanced_fluxes = [47.0, 35.0, 84.0, 76.0, 8.0]  # box 0 outflow 47 != inflow 43

    # Kirchhoff imbalance rejected across BalancedChart, OpenChart, from_boxes, and to_balanced
    with pytest.raises(ValueError, match="Kirchhoff"):
        BalancedChart.from_masses_and_fluxes(topo_asym, masses, imbalanced_fluxes)
    with pytest.raises(ValueError, match="Kirchhoff"):
        OpenChart.from_masses_and_fluxes(topo_asym, masses, imbalanced_fluxes, [0.0, 0.0, 2e-4])
    with pytest.raises(ValueError, match="Kirchhoff"):
        CarbonBoxModel.from_boxes(
            masses={"Stratosphere": 90.0, "Troposphere": 510.0},
            fluxes=[("Stratosphere", "Troposphere", 45.0), ("Troposphere", "Stratosphere", 41.0)],
            chart="balanced",
        )
    unbal_fixed = FixedCBM(topo_asym, masses, imbalanced_fluxes)
    with pytest.raises(ValueError, match="Kirchhoff"):
        unbal_fixed.to_balanced()

    # Non-positive or non-finite masses/fluxes and off-topology 2D matrix entries rejected
    for bad_m in ([0.0, 525.0, 3600.0], [-5.0, 525.0, 3600.0], [float("nan"), 525.0, 3600.0]):
        with pytest.raises(ValueError, match="masses"):
            BalancedChart.from_masses_and_fluxes(topo_asym, bad_m, balanced_fluxes)
    for bad_f in ([0.0, 39.0, 84.0, 76.0, 8.0], [float("inf"), 39.0, 84.0, 76.0, 8.0]):
        with pytest.raises(ValueError, match="edge_fluxes"):
            BalancedChart.from_masses_and_fluxes(topo_asym, masses, bad_f)
    bad_mat = np.zeros((3, 3), dtype=np.float64)
    for s, d, f in zip(topo_asym.edges_src, topo_asym.edges_dst, balanced_fluxes):
        bad_mat[s, d] = f
    bad_mat[0, 2] = 1.5  # edge (0 -> 2) is not in topo_asym
    with pytest.raises(ValueError, match="outside topology"):
        BalancedChart.from_masses_and_fluxes(topo_asym, masses, bad_mat)

    # Inconsistent edge tying rejected; consistent edge tying accepted
    with pytest.raises(ValueError, match="tying"):
        BalancedChart.from_masses_and_fluxes(
            topo_asym, masses, balanced_fluxes, symmetric_edge_groups=[[0, 1]]
        )
    topo_sym = _sym_3box_topo()
    tied_ok = BalancedChart.from_masses_and_fluxes(
        topo_sym, [120.0, 120.0, 2400.0], [48.0, 48.0, 60.0, 60.0], symmetric_edge_groups=[[0, 1]]
    )
    assert tied_ok.log_rates_free.shape == (3,)

    # OpenChart / OpenCBM sink_rates validation across ALL boxes and open_box_indices
    for bad_sinks in (
        [-1e-5, 0.0, 2.5e-4],  # negative on closed box
        [0.0, float("nan"), 2.5e-4],
        [0.0, float("inf"), 2.5e-4],
        [0.0, 2.5e-4],  # wrong shape
    ):
        with pytest.raises(ValueError, match="sink_rates"):
            OpenChart.from_masses_and_fluxes(topo_asym, masses, balanced_fluxes, bad_sinks)
    with pytest.raises(ValueError, match="strictly positive"):
        OpenChart.from_masses_and_fluxes(
            topo_asym, masses, balanced_fluxes, [0.0, 0.0, 2.5e-4], open_box_indices=(1, 2)
        )
    with pytest.raises(ValueError, match="duplicates"):
        OpenChart.from_masses_and_fluxes(
            topo_asym, masses, balanced_fluxes, [0.0, 0.0, 2.5e-4], open_box_indices=(2, 2)
        )


def test_topology_chart_compatibility_and_decay_rate_validation() -> None:
    """Thread 3: validate Topology, decay_rate, and chart._validate_topology in __init__/with_chart."""
    topo_sym = _sym_3box_topo()
    topo_asym = _asym_3box_topo()
    fix_3 = FixedChart(topo_sym, [90.0, 510.0, 3200.0], [45.0, 45.0, 60.0, 60.0])
    bal_5 = BalancedChart(topo_asym, [-0.6, -2.5, -1.9, -3.7, -5.8], 8.3)
    rev_2 = ReversibleChart(topo_sym, np.log([90.0, 510.0, 3200.0]), np.log([45.0, 60.0]))
    opn_3 = OpenChart(topo_asym, [-0.6, -2.5, -1.9, -3.7, -5.8], 8.3, [-8.1], open_box_indices=(2,))

    with pytest.raises(TypeError, match="Topology"):
        CarbonBoxModel("not_a_topology", fix_3)  # type: ignore[arg-type]
    for bad_decay in (-1e-4, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="decay_rate"):
            CarbonBoxModel(topo_sym, fix_3, decay_rate=bad_decay)
    assert CarbonBoxModel(topo_sym, fix_3, decay_rate=0.0).decay_rate == 0.0

    # Box-count, edge-count, and prod_box_indices mismatches rejected in CarbonBoxModel / with_chart
    topo_4box = Topology(
        box_names=("Stratosphere", "Troposphere", "Surface", "Deep"),
        edges_src=(0, 1, 1, 2, 2, 3),
        edges_dst=(1, 0, 2, 1, 3, 2),
    )
    cbm_4 = CarbonBoxModel(
        topo_4box,
        FixedChart(topo_4box, [90.0, 510.0, 900.0, 35000.0], [45.0, 45.0, 60.0, 60.0, 40.0, 40.0]),
    )
    for bad_chart in (fix_3, bal_5, rev_2, opn_3):
        with pytest.raises(ValueError):
            cbm_4.with_chart(bad_chart)

    # Edge-count mismatch (4-edge topo_sym vs 5-edge bal_5) and prod_weights length mismatch
    with pytest.raises(ValueError, match="log_rates_free"):
        CarbonBoxModel(topo_sym, bal_5)
    topo_1prod = Topology(
        box_names=("Stratosphere", "Troposphere", "Ocean"),
        edges_src=(0, 1, 1, 2),
        edges_dst=(1, 0, 2, 1),
        prod_box_indices=(1,),
        trop_box_indices=(1,),
    )
    with pytest.raises(ValueError, match="prod_weights"):
        CarbonBoxModel(topo_1prod, fix_3)

    # Reducible 3-box, 4-edge topology rejected by Perron/Reversible charts via with_chart
    topo_reducible = Topology(
        box_names=("Stratosphere", "Troposphere", "Sink"),
        edges_src=(0, 1, 0, 1),
        edges_dst=(1, 0, 2, 2),
        check_scc=False,
    )
    with pytest.raises(ValueError, match="strongly connected"):
        CarbonBoxModel(topo_reducible, rev_2)

    # Non-reversible strongly connected 3-box, 4-edge topology rejected by ReversibleChart
    topo_nonrev_4e = Topology(
        box_names=("Stratosphere", "Troposphere", "Ocean"),
        edges_src=(0, 1, 2, 0),
        edges_dst=(1, 2, 0, 2),
    )
    with pytest.raises(ValueError, match="symmetric edge pairs"):
        CarbonBoxModel(topo_nonrev_4e, rev_2)

    # Reordered symmetric 3-box, 4-edge topology with mismatched pair indices rejected
    topo_reordered_4e = Topology(
        box_names=("Stratosphere", "Troposphere", "Ocean"),
        edges_src=(0, 1, 1, 2),
        edges_dst=(1, 2, 0, 1),
    )
    with pytest.raises(ValueError, match="edge-pair indices"):
        CarbonBoxModel(topo_reordered_4e, rev_2)


def test_to_fixed_jit_to_balanced_tying_overflow_and_chart_from_state() -> None:
    """Threads 4 & 5: jit-safe to_fixed(), unpacked _chart_from_state, and tie/overflow-safe to_balanced()."""
    from ticktack.cbm import _chart_from_state

    topo_asym = _asym_3box_topo()
    topo_sym = _sym_3box_topo()

    # Direct verification of unpacked _chart_from_state signature
    c_f = _chart_from_state(
        topo_asym, [94.0, 525.0, 3600.0], [47.0, 39.0, 84.0, 76.0, 8.0], "fixed"
    )
    assert isinstance(c_f, FixedChart)
    c_b = _chart_from_state(
        topo_asym, [94.0, 525.0, 3600.0], [47.0, 39.0, 84.0, 76.0, 8.0], "balanced"
    )
    assert isinstance(c_b, BalancedChart)

    # Verify to_fixed() under both eqx.filter_jit and jax.jit + autodiff across all CBM types
    models: list[CarbonBoxModel] = [
        load_preset("Guttler15", chart="fixed"),
        load_preset("Guttler15", chart="balanced"),
        load_preset("Miyake17", chart="open"),
        CarbonBoxModel(
            topo_sym, ReversibleChart(topo_sym, np.log([92.0, 520.0, 3100.0]), np.log([46.0, 62.0]))
        ),
        EffectiveCBM(K=3),
        SubspaceCBM(topo_asym, [-0.6, -2.5, -1.9, -3.7, -5.8], np.ones((5, 2)) * 0.15, 8.3),
    ]
    jit_to_fixed = eqx.filter_jit(lambda m: m.to_fixed())
    for m in models:
        m_fix = jit_to_fixed(m)
        assert type(m_fix) is CarbonBoxModel and isinstance(m_fix.chart, FixedChart)
        np.testing.assert_allclose(m_fix.matrix_A(), m.matrix_A(), rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(
            m_fix.steady_state_d14c(), m.steady_state_d14c(), rtol=1e-12, atol=1e-10
        )

    # Raw jax.jit and eqx.filter_grad through to_fixed() on OpenChart & EffectiveCBM
    opn = load_preset("Miyake17", chart="open")
    raw_jit_d14c = jax.jit(lambda m: m.to_fixed().steady_state_d14c())(opn)
    np.testing.assert_allclose(raw_jit_d14c, opn.steady_state_d14c(), rtol=1e-12, atol=1e-12)
    eff = EffectiveCBM(K=3)
    g_via_fix = eqx.filter_grad(lambda m: jnp.sum(m.to_fixed().steady_state_d14c() ** 2))(eff)
    g_direct = eqx.filter_grad(lambda m: jnp.sum(m.steady_state_d14c() ** 2))(eff)
    np.testing.assert_allclose(
        g_via_fix.log_k_trop_to_sinks, g_direct.log_k_trop_to_sinks, rtol=1e-12
    )
    assert float(jnp.linalg.norm(g_via_fix.log_k_trop_to_sinks)) > 0.0

    # Thread 5: to_balanced() preserves tie_matrix_rows and avoids exp(800.0) overflow
    for b21_tied in (
        load_preset("Brehm21", chart="balanced", tie_symmetric_edges=True),
        load_preset("Brehm21", kind="balanced", tie_symmetric_edges=True),
    ):
        assert b21_tied.log_rates_free.shape == (39,)
        assert b21_tied.tie_matrix_rows is not None
        b21_hi = b21_tied.set("log_total_mass", jnp.asarray(800.0, dtype=jnp.float64))
        proj = b21_hi.to_balanced()
        assert type(proj) is CarbonBoxModel and isinstance(proj.chart, BalancedChart)
        assert proj.tie_matrix_rows == b21_tied.tie_matrix_rows
        assert proj.log_rates_free.shape == (39,)
        assert float(proj.log_total_mass) == 800.0
        assert np.all(np.isfinite(np.asarray(proj.steady_state_d14c())))
        np.testing.assert_array_equal(proj.steady_state_d14c(), b21_tied.steady_state_d14c())

    # Single collapsed fallback in to_balanced() with explicit symmetric_edge_groups
    sym_fix = FixedCBM(topo_sym, [120.0, 120.0, 2400.0], [48.0, 48.0, 60.0, 60.0])
    sym_bal_tied = sym_fix.to_balanced(symmetric_edge_groups=[[0, 1]])
    assert sym_bal_tied.log_rates_free.shape == (3,)


def test_from_boxes_subclasses_finiteness_troposphere_and_duplicate_keys() -> None:
    """Threads 6, 7 & 8: subclass from_boxes, NaN/inf guards, troposphere=..., and duplicate keys."""
    masses_ok = {"Stratosphere": 94.0, "Troposphere": 525.0, "DeepOcean": 3600.0}
    fluxes_ok = [
        ("Stratosphere", "Troposphere", 47.0),
        ("Troposphere", "Stratosphere", 39.0),
        ("Troposphere", "DeepOcean", 84.0),
        ("DeepOcean", "Troposphere", 76.0),
        ("DeepOcean", "Stratosphere", 8.0),
    ]

    # Thread 6: calling from_boxes on *CBM subclasses returns concrete CarbonBoxModel without TypeError
    for sub_cls, chart_arg in [
        (FixedCBM, "fixed"),
        (BalancedCBM, "balanced"),
        (OpenCBM, "open"),
    ]:
        kw = {"sinks": {"DeepOcean": 2e-4}} if chart_arg == "open" else {}
        built = sub_cls.from_boxes(masses_ok, fluxes_ok, chart=chart_arg, **kw)  # type: ignore[arg-type]
        assert type(built) is CarbonBoxModel
    rev_sub = ReversibleCBM.from_boxes(
        {"Stratosphere": 90.0, "Troposphere": 510.0},
        [("Stratosphere", "Troposphere", 45.0), ("Troposphere", "Stratosphere", 45.0)],
        chart="reversible",
    )
    assert type(rev_sub) is CarbonBoxModel and isinstance(rev_sub.chart, ReversibleChart)

    # Thread 6: NaN / inf rejected in masses, fluxes, production, sinks, and decay_rate
    for nonfinite in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            CarbonBoxModel.from_boxes(
                {"Troposphere": nonfinite, "Stratosphere": 90.0}, fluxes_ok[:2]
            )
        with pytest.raises(ValueError):
            CarbonBoxModel.from_boxes(
                {"Stratosphere": 90.0, "Troposphere": 510.0},
                [("Stratosphere", "Troposphere", nonfinite), ("Troposphere", "Stratosphere", 45.0)],
            )
        with pytest.raises(ValueError):
            CarbonBoxModel.from_boxes(masses_ok, fluxes_ok, production={"Troposphere": nonfinite})
        with pytest.raises(ValueError):
            CarbonBoxModel.from_boxes(masses_ok, fluxes_ok, production=[0.7, nonfinite, 0.3])
        with pytest.raises(ValueError):
            CarbonBoxModel.from_boxes(masses_ok, fluxes_ok, sinks={"DeepOcean": nonfinite})
        with pytest.raises(ValueError):
            CarbonBoxModel.from_boxes(masses_ok, fluxes_ok, decay_rate=nonfinite)

    # Thread 7: custom reservoir names without "trop" require troposphere=... and succeed when given
    custom_masses = {"UpperAir": 90.0, "Atmosphere": 510.0, "Abyss": 36000.0}
    custom_fluxes = [
        ("UpperAir", "Atmosphere", 45.0),
        ("Atmosphere", "UpperAir", 45.0),
        ("Atmosphere", "Abyss", 60.0),
        ("Abyss", "Atmosphere", 60.0),
    ]
    with pytest.raises(ValueError, match="troposphere"):
        CarbonBoxModel.from_boxes(custom_masses, custom_fluxes, chart="balanced")

    cbm_atm = CarbonBoxModel.from_boxes(
        custom_masses, custom_fluxes, troposphere="Atmosphere", chart="balanced"
    )
    assert cbm_atm.topology.trop_box_indices == (1,)
    assert float(cbm_atm.steady_state_d14c()[1]) == pytest.approx(0.0, abs=1e-12)

    # Hemispheric custom names with troposphere="Atmosphere"
    cbm_hemi_atm = CarbonBoxModel.from_boxes(
        masses={("Atmosphere", "south"): 255.0, ("Atmosphere", "north"): 255.0},
        fluxes=[
            (("Atmosphere", "south"), ("Atmosphere", "north"), 65.0),
            (("Atmosphere", "north"), ("Atmosphere", "south"), 65.0),
        ],
        troposphere="Atmosphere",
        chart="balanced",
    )
    assert cbm_hemi_atm.topology.trop_box_indices == (0, 1)

    # Thread 8: duplicate normalized reservoir keys rejected in named vectors
    with pytest.raises(ValueError, match="Duplicate"):
        CarbonBoxModel.from_boxes(
            masses_ok, fluxes_ok, production={"Troposphere": 0.3, "troposphere": 0.7}
        )
    with pytest.raises(ValueError, match="Duplicate"):
        CarbonBoxModel.from_boxes(
            masses_ok, fluxes_ok, sinks={"DeepOcean": 1e-4, " DeepOcean ": 2e-4}
        )
    with pytest.raises(ValueError, match="Duplicate"):
        CarbonBoxModel.from_boxes(
            masses={("Troposphere", "south"): 250.0, ("Troposphere", "north"): 250.0},
            fluxes=[
                (("Troposphere", "south"), ("Troposphere", "north"), 70.0),
                (("Troposphere", "north"), ("Troposphere", "south"), 70.0),
            ],
            production={("Troposphere", "north"): 0.4, ("Troposphere", "nh"): 0.6},
        )


def test_reversible_scale_gauge_docstrings_and_getattr_leafwrapper_guard() -> None:
    """Threads 7, 9 & 10: unit/gauge docstrings, ReversibleChart 1D gauge, and _LeafWrapper(None) guard."""
    for obj in (
        CarbonBoxModel.from_boxes,
        CBMChart,
        FixedChart,
        BalancedChart,
        ReversibleChart,
        OpenChart,
    ):
        doc = obj.__doc__ or ""
        assert "Parameters" in doc or "Attributes" in doc
        assert "yr^-1" in doc and "GtC" in doc
    assert "scale" in (ReversibleChart.__doc__ or "")
    assert "include_masses=False" in (ReversibleChart.param_specs.__doc__ or "")

    # Thread 9: verify exact 1D global scale gauge (1_n, 1_m) invariance on ReversibleChart
    topo_sym = _sym_3box_topo()
    rev0 = CarbonBoxModel(
        topo_sym,
        ReversibleChart(
            topo_sym, np.log([91.0, 523.0, 3150.0]), np.log([46.5, 63.0]), prod_weights=(0.71, 0.29)
        ),
    )
    shift = 1.73
    rev_scaled = rev0.set("log_masses", rev0.log_masses + shift).set(
        "log_sym_fluxes", rev0.log_sym_fluxes + shift
    )
    np.testing.assert_allclose(rev_scaled.edge_rates(), rev0.edge_rates(), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(rev_scaled.matrix_A(), rev0.matrix_A(), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(
        rev_scaled.relative_reservoir_masses(),
        rev0.relative_reservoir_masses(),
        rtol=1e-14,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        rev_scaled.steady_state_d14c(), rev0.steady_state_d14c(), rtol=1e-12, atol=1e-12
    )

    # Thread 10: _LeafWrapper(None) inside eqx.tree_at / zodiax .set() on chartless subclasses
    eff = EffectiveCBM(K=3)
    sub = SubspaceCBM(_asym_3box_topo(), [-0.6, -2.5, -1.9, -3.7, -5.8], np.ones((5, 2)) * 0.2, 8.3)
    eff2 = eff.set("log_k_strat_to_trop", jnp.asarray(-0.55, dtype=jnp.float64))
    sub2 = sub.set("alpha", jnp.array([0.15, -0.25], dtype=jnp.float64))
    assert float(eff2.log_k_strat_to_trop) == pytest.approx(-0.55)
    np.testing.assert_allclose(sub2.alpha, [0.15, -0.25])

    # Prove sensitivity: Equinox _LeafWrapper has a '.leaf' attribute; without isinstance(chart, CBMChart),
    # accessing m.leaf inside eqx.tree_at's where callable leaked _LeafWrapper(None).leaf instead of AttributeError.
    for chartless, target_fn, repl in [
        (eff, lambda m: m.log_k_strat_to_trop, jnp.asarray(-0.5)),
        (sub, lambda m: m.alpha, jnp.zeros((2,))),
    ]:
        assert eqx.tree_at(target_fn, chartless, repl) is not None
        with pytest.raises(AttributeError, match="leaf"):
            eqx.tree_at(lambda m: m.leaf, chartless, repl)


def test_effective_param_specs_jit_open_sink_guards_and_fixed_validation() -> None:
    """Verify EffectiveCBM.param_specs, SubspaceCBM kwargs, jit to_balanced sink rejection, and FixedChart guards."""
    topo_asym = _asym_3box_topo()
    masses = [94.0, 525.0, 3600.0]
    fluxes = [47.0, 39.0, 84.0, 76.0, 8.0]
    spike = MiyakeSpike(774.5, 0.3, 6.2)

    # 1. EffectiveCBM.param_specs covers all 2K - 2 (or 2K - 1) leaves and round-trips ParameterSpace
    for eff_case, expected_dim in [
        (EffectiveCBM(K=2), 2),
        (EffectiveCBM(K=3), 4),
        (EffectiveCBM(K=4), 6),
        (EffectiveCBM(K=3, deep_sink_rate=0.015), 5),
    ]:
        with pytest.raises(ValueError, match="prior_sd"):
            eff_case.param_specs()
        rm = RadiocarbonModel(eff_case, SteadyProduction(1.76) + spike)
        ps = ParameterSpace(rm, list(eff_case.param_specs(prior_sd=0.35)))
        assert ps.n_params == expected_dim
        u0 = ps.unconstrain(rm)
        u_pert = u0 + jnp.linspace(0.08, -0.14, ps.n_params, dtype=jnp.float64)
        np.testing.assert_allclose(ps.unconstrain(ps.constrain(u_pert)), u_pert, rtol=1e-13, atol=1e-14)

    with pytest.raises(ValueError, match="explicit_deep_sink_rate"):
        EffectiveCBM(K=3, deep_sink_rate=0.0).param_specs(prior_sd=0.35)

    # SubspaceCBM rejects unexpected kwargs; custom chartless subclass with leaves raises NotImplementedError
    sub = SubspaceCBM(topo_asym, [-0.6, -2.5, -1.9, -3.7, -5.8], np.ones((5, 2)) * 0.2, 8.3)
    with pytest.raises(TypeError, match="unexpected keyword"):
        sub.param_specs(bogus_kwarg=0.35)

    class _CustomDynamicCBM(CarbonBoxModel):
        theta: jax.Array

        def __init__(self, topology: Topology) -> None:
            self.topology = topology
            self.decay_rate = 1.2e-4
            self.theta = jnp.array([1.0, 2.0], dtype=jnp.float64)

    with pytest.raises(NotImplementedError, match="param_specs"):
        _CustomDynamicCBM(topo_asym).param_specs(prior_sd=0.35)

    # 2. to_balanced() rejects OpenChart, OpenCBM, and calibrated EffectiveCBM under eqx.filter_jit / jax.jit
    for open_model in (
        load_preset("Miyake17", chart="open"),
        load_preset("Miyake17", kind="open"),
        EffectiveCBM(K=3),
    ):
        with pytest.raises(ValueError, match="positive sinks"):
            eqx.filter_jit(lambda m: m.to_balanced())(open_model)
        with pytest.raises(ValueError, match="positive sinks"):
            jax.jit(lambda m: m.to_balanced().matrix_A())(open_model)

    # OpenChart / OpenCBM rejects positive sinks outside explicit open_box_indices
    with pytest.raises(ValueError, match="outside explicit open_box_indices"):
        OpenChart.from_masses_and_fluxes(
            topo_asym, masses, fluxes, [0.0, 1e-3, 2.5e-4], open_box_indices=(2,)
        )

    # 3. FixedChart / FixedCBM rejects non-positive/non-finite masses and negative/non-finite fluxes & sinks
    for bad_m in ([0.0, 525.0, 3600.0], [-10.0, 525.0, 3600.0], [float("nan"), 525.0, 3600.0]):
        with pytest.raises(ValueError, match="masses"):
            FixedChart(topo_asym, bad_m, fluxes)
    for bad_f in ([-1.0, 39.0, 84.0, 76.0, 8.0], [float("inf"), 39.0, 84.0, 76.0, 8.0]):
        with pytest.raises(ValueError, match="edge_fluxes"):
            FixedChart(topo_asym, masses, bad_f)
    for bad_s in ([0.0, -1e-4, 0.0], [0.0, float("nan"), 0.0]):
        with pytest.raises(ValueError, match="sink_rates"):
            FixedChart(topo_asym, masses, fluxes, sink_rates=bad_s)
    # Zero edge_fluxes / sink_rates remain valid on FixedChart
    assert FixedChart(topo_asym, masses, [0.0, 39.0, 84.0, 76.0, 8.0], sink_rates=[0.0, 0.0, 0.0]) is not None


def test_flag_descriptors_jit_to_balanced_guards_non_identity_prod_and_open_from_boxes() -> None:
    """Verify non-data flag descriptors, jit to_balanced/inversion guards, permuted prod_box_indices, and open from_boxes."""

    # 1. Thread PRRT_kwDOFwVIdc68J9m6: static eqx.field subclasses can set is_detailed_balance / is_subspace
    class _StaticFlagCBM(CarbonBoxModel):
        log_masses: jax.Array
        is_detailed_balance: bool = eqx.field(static=True)
        is_subspace: bool = eqx.field(static=True)

        def __init__(self, *, is_detailed_balance: bool, is_subspace: bool = False) -> None:
            self.topology = _sym_3box_topo()
            self.decay_rate = 1.2e-4
            self.log_masses = jnp.log(jnp.array([90.0, 510.0, 3200.0], dtype=jnp.float64))
            self.is_detailed_balance = bool(is_detailed_balance)
            self.is_subspace = bool(is_subspace)

        def reservoir_masses(self) -> jax.Array:
            return jnp.exp(self.log_masses)

    m_true = _StaticFlagCBM(is_detailed_balance=True, is_subspace=True)
    m_false = _StaticFlagCBM(is_detailed_balance=False, is_subspace=False)
    assert m_true.is_detailed_balance is True and m_true.is_subspace is True
    assert m_false.is_detailed_balance is False and m_false.is_subspace is False
    # Round-trip through Equinox PyTree flatten/unflatten and filter_jit
    m_rt = eqx.filter_jit(lambda m: m)(m_true)
    assert m_rt.is_detailed_balance is True and m_rt.is_subspace is True

    class _OptOutRev(ReversibleCBM):
        @property
        def is_detailed_balance(self) -> bool:
            return False

    opt_out = _OptOutRev(_sym_3box_topo(), np.log([90.0, 510.0, 3200.0]), np.log([45.0, 60.0]))
    assert opt_out.is_detailed_balance is False

    # 2. Thread PRRT_kwDOFwVIdc68J9nC: no silent sink drop, Kirchhoff bypass, or tying bypass under jit
    topo_asym = _asym_3box_topo()
    for traced_open in (
        load_preset("Miyake17", chart="fixed"),
        EffectiveCBM(K=3, deep_sink_rate=0.015),
        FixedCBM(topo_asym, [94.0, 525.0, 3600.0], [47.0, 35.0, 84.0, 76.0, 8.0]),
    ):
        with pytest.raises(ValueError, match="traced|positive sinks|concrete"):
            eqx.filter_jit(lambda m: m.to_balanced())(traced_open)
        with pytest.raises(ValueError, match="traced|positive sinks|concrete"):
            jax.jit(lambda m: m.to_balanced().matrix_A())(traced_open)

    with pytest.raises(ValueError, match="concrete"):
        jax.jit(
            lambda flx: BalancedChart.from_masses_and_fluxes(
                topo_asym, [94.0, 525.0, 3600.0], flx, symmetric_edge_groups=[[0, 1]]
            )
        )(jnp.array([47.0, 39.0, 84.0, 76.0, 8.0], dtype=jnp.float64))

    # Structurally closed ReversibleChart / BalancedChart / SubspaceCBM remain jit-safe for untied to_balanced()
    rev_base = CarbonBoxModel(
        _sym_3box_topo(),
        ReversibleChart(_sym_3box_topo(), np.log([90.0, 510.0, 3200.0]), np.log([45.0, 60.0])),
    )
    rev_hi = CarbonBoxModel(
        _sym_3box_topo(),
        ReversibleChart(
            _sym_3box_topo(), np.log([90.0, 510.0, 3200.0]) + 790.0, np.log([45.0, 60.0]) + 790.0
        ),
    )
    rev_to_bal = eqx.filter_jit(lambda m: m.to_balanced())(rev_hi)
    assert isinstance(rev_to_bal.chart, BalancedChart)
    np.testing.assert_allclose(rev_to_bal.matrix_A(), rev_base.matrix_A(), rtol=1e-13, atol=1e-15)
    assert np.all(np.isfinite(np.asarray(rev_to_bal.steady_state_d14c())))

    # 3. Thread PRRT_kwDOFwVIdc68J9nN: non-identity prod_box_indices with len == n_boxes is idempotent
    topo_perm = Topology(
        ("Troposphere", "Stratosphere"),
        edges_src=(0, 1),
        edges_dst=(1, 0),
        prod_box_indices=(1, 0),
        trop_box_indices=(0,),
    )
    expected_b = np.array([0.27, 0.73], dtype=np.float64)
    perm_models: list[CarbonBoxModel] = [
        CarbonBoxModel(
            topo_perm, FixedChart(topo_perm, [510.0, 90.0], [45.0, 45.0], prod_weights=(0.27, 0.73))
        ),
        CarbonBoxModel(
            topo_perm, BalancedChart(topo_perm, [-2.4, -0.7], 6.4, prod_weights=(0.27, 0.73))
        ),
        CarbonBoxModel(
            topo_perm,
            ReversibleChart(
                topo_perm, np.log([510.0, 90.0]), np.log([45.0]), prod_weights=(0.27, 0.73)
            ),
        ),
        SubspaceCBM(
            topo_perm, [-2.4, -0.7], np.array([[0.1], [0.2]]), 6.4, prod_weights=(0.27, 0.73)
        ),
    ]
    for pm in perm_models:
        assert pm.prod_weights == pytest.approx((0.73, 0.27))
        np.testing.assert_allclose(pm.production_vector(), expected_b, rtol=1e-14)
        np.testing.assert_allclose(pm.to_fixed().production_vector(), expected_b, rtol=1e-14)
        np.testing.assert_allclose(pm.to_balanced().production_vector(), expected_b, rtol=1e-14)
    np.testing.assert_allclose(
        perm_models[-1].to_balanced_cbm().production_vector(), expected_b, rtol=1e-14
    )  # type: ignore[union-attr]

    # 4. Thread PRRT_kwDOFwVIdc68J9nS: from_boxes(..., chart="open") without positive sinks names sinks
    masses_ok = {"Stratosphere": 94.0, "Troposphere": 525.0, "DeepOcean": 3600.0}
    fluxes_ok = [
        ("Stratosphere", "Troposphere", 47.0),
        ("Troposphere", "Stratosphere", 39.0),
        ("Troposphere", "DeepOcean", 84.0),
        ("DeepOcean", "Troposphere", 76.0),
        ("DeepOcean", "Stratosphere", 8.0),
    ]
    with pytest.raises(ValueError, match="sinks"):
        CarbonBoxModel.from_boxes(masses_ok, fluxes_ok, chart="open")
    with pytest.raises(ValueError, match="sinks"):
        CarbonBoxModel.from_boxes(masses_ok, fluxes_ok, sinks={"DeepOcean": 0.0}, chart="open")

    # 5. Round-3 threads: Python float via zodiax .set("log_total_mass", 9.5) in _validate_topology & active_parameter_mask
    cbm_bal_float = load_preset("Guttler15", chart="balanced").set("log_total_mass", 9.5)
    assert isinstance(cbm_bal_float.to_balanced().chart, BalancedChart)
    assert isinstance(cbm_bal_float.with_chart(cbm_bal_float.chart).chart, BalancedChart)
    assert cbm_bal_float.active_parameter_mask().shape == (24,)
    assert bool(np.all(cbm_bal_float.active_parameter_mask()))

    solver = ExactLinearSolver()
    t_s = jnp.arange(770.0, 776.0, dtype=jnp.float64) + 0.3
    t_e = t_s + 0.4
    spec_float = cbm_fisher_spectrum(
        lambda c: solver.predict_d14c(
            c, SteadyProduction(1.76) + MiyakeSpike(774.5, 0.3, 6.2), t_s, t_e, box_indices=1
        ),
        cbm_bal_float,
        jnp.full(t_s.shape, 1.5, dtype=jnp.float64),
        prior_cbm_sd=0.35,
    )
    assert spec_float.active_mask.shape == (24,)

    # 0-edge 1-box BalancedChart with Python float log_total_mass returns empty mask cleanly
    topo_1box = Topology(("Troposphere",), (), (), prod_box_indices=(0,), trop_box_indices=(0,))
    cbm_1box = CarbonBoxModel(topo_1box, BalancedChart(topo_1box, [], 6.2)).set(
        "log_total_mass", 6.5
    )
    assert cbm_1box.with_chart(cbm_1box.chart).active_parameter_mask().shape == (0,)
    assert SubspaceCBM.from_fisher.__annotations__["base_cbm"] == "CarbonBoxModel"


def test_claude_gate5_exact_box_matching_default_prod_kirchhoff_msg_and_kind_deprecation() -> None:
    """Verify Claude Gate 5 fixes: exact trop/strat matching, 70/30 default prod, Kirchhoff wording, and kind= deprecation."""
    import warnings

    # 1. Exact normalized whole-name matching excludes "Tropical Surface Ocean" and "Stratified Ocean"
    cbm_ocean = CarbonBoxModel.from_boxes(
        masses={
            "Stratosphere": 90.0,
            "Troposphere": 510.0,
            "Tropical Surface Ocean": 800.0,
            "Stratified Ocean": 3500.0,
        },
        fluxes=[
            ("Stratosphere", "Troposphere", 45.0),
            ("Troposphere", "Stratosphere", 45.0),
            ("Troposphere", "Tropical Surface Ocean", 60.0),
            ("Tropical Surface Ocean", "Troposphere", 60.0),
            ("Tropical Surface Ocean", "Stratified Ocean", 40.0),
            ("Stratified Ocean", "Tropical Surface Ocean", 40.0),
        ],
        chart="balanced",
    )
    assert cbm_ocean.topology.trop_box_indices == (1,)
    assert cbm_ocean.topology.prod_box_indices == (0, 1)
    np.testing.assert_allclose(cbm_ocean.production_vector(), [0.70, 0.30, 0.0, 0.0], rtol=1e-14)
    d14c_ss = np.asarray(cbm_ocean.steady_state_d14c(), dtype=np.float64)
    assert d14c_ss[1] == pytest.approx(0.0, abs=1e-12)
    assert d14c_ss[2] < -1.0  # Tropical Surface Ocean is depleted, not merged into R_ref

    # Without an exact troposphere name, "Tropical Surface Ocean" must not match
    with pytest.raises(ValueError, match="troposphere"):
        CarbonBoxModel.from_boxes(
            masses={"Tropical Surface Ocean": 800.0, "Stratified Ocean": 3500.0},
            fluxes=[
                ("Tropical Surface Ocean", "Stratified Ocean", 40.0),
                ("Stratified Ocean", "Tropical Surface Ocean", 40.0),
            ],
            chart="balanced",
        )

    # Ambiguous single-globe troposphere names or mismatched strat/trop counts must raise ValueError
    with pytest.raises(ValueError, match="Ambiguous.*troposphere"):
        CarbonBoxModel.from_boxes(
            masses={"Troposphere": 255.0, "Trop": 255.0, "Ocean": 3500.0},
            fluxes=[
                ("Troposphere", "Trop", 50.0),
                ("Trop", "Troposphere", 50.0),
                ("Troposphere", "Ocean", 40.0),
                ("Ocean", "Troposphere", 40.0),
            ],
            chart="balanced",
        )
    with pytest.raises(ValueError, match="production="):
        CarbonBoxModel.from_boxes(
            masses={"Stratosphere": 45.0, "Strat": 45.0, "Troposphere": 510.0},
            fluxes=[
                ("Stratosphere", "Troposphere", 22.0),
                ("Troposphere", "Stratosphere", 22.0),
                ("Strat", "Troposphere", 22.0),
                ("Troposphere", "Strat", 22.0),
            ],
            chart="balanced",
        )

    # Hemispheric default production assigns 70% strat / 30% trop split evenly across hemispheres
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
    np.testing.assert_allclose(cbm_hemi.production_vector(), [0.35, 0.15, 0.35, 0.15], rtol=1e-14)

    # 2. Kirchhoff imbalance ValueError recommends chart='fixed' and quotes rtol=1e-6, atol=1e-9
    with pytest.raises(ValueError, match=r"rtol=1e-6.*chart='fixed'"):
        CarbonBoxModel.from_boxes(
            masses={"Stratosphere": 90.0, "Troposphere": 510.0},
            fluxes=[("Stratosphere", "Troposphere", 45.0), ("Troposphere", "Stratosphere", 41.0)],
            chart="balanced",
        )

    # 3. load_preset(..., kind=...) emits DeprecationWarning; chart=... and default do not
    with pytest.warns(DeprecationWarning, match=r"Wave 6.*chart="):
        _ = load_preset("Guttler15", kind="balanced")
    with pytest.warns(DeprecationWarning, match=r"Wave 6.*chart="):
        _ = load_preset("Guttler15", "fixed")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        _ = load_preset("Guttler15", chart="balanced")
        _ = load_preset("Guttler15")
    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]
