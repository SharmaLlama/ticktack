"""Tests for Package 4: shared CarbonBoxModel / _PerronCBM base deduplication."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ticktack.cbm as cbm_mod
from ticktack.cbm import (
    BalancedCBM,
    CarbonBoxModel,
    EffectiveCBM,
    FisherSubspaceCBM,
    FixedCBM,
    OpenCBM,
    ReversibleCBM,
    SubspaceCBM,
    Topology,
)

jax.config.update("jax_enable_x64", True)


def _asymmetric_3box_topology() -> Topology:
    """Asymmetric 3-box strongly connected topology with non-reversible cycle."""
    return Topology(
        box_names=("Stratosphere", "Troposphere", "DeepOcean"),
        edges_src=(0, 1, 1, 2, 2),
        edges_dst=(1, 0, 2, 1, 0),
        prod_box_indices=(0, 1),
        trop_box_indices=(1,),
        check_scc=True,
    )


def _symmetric_3box_topology() -> Topology:
    """3-box symmetric topology supporting ReversibleCBM and edge tying."""
    return Topology(
        box_names=("Stratosphere", "Troposphere", "Ocean"),
        edges_src=(0, 1, 1, 2),
        edges_dst=(1, 0, 2, 1),
        prod_box_indices=(0, 1),
        trop_box_indices=(1,),
        check_scc=True,
    )


def _leaf_names(tree: CarbonBoxModel) -> list[str]:
    leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(tree)
    names: list[str] = []
    for path, _ in leaves_with_path:
        assert len(path) == 1
        names.append(path[0].name)
    return names


def test_shared_base_methods_and_dead_helper_removal() -> None:
    """All CBM subclasses inherit shared methods cleanly without leaking Perron hooks."""
    assert not hasattr(cbm_mod, "_concrete_array")
    assert not hasattr(cbm_mod, "_calibrate_effective_deep_sink")

    # Every concrete CBM inherits CarbonBoxModel.rate_matrix_K.
    for cls in (FixedCBM, BalancedCBM, SubspaceCBM, ReversibleCBM, OpenCBM, EffectiveCBM):
        assert cls.rate_matrix_K is CarbonBoxModel.rate_matrix_K

    # All prod_weights-backed CBMs inherit CarbonBoxModel.production_vector.
    for cls in (FixedCBM, BalancedCBM, SubspaceCBM, ReversibleCBM, OpenCBM):
        assert cls.production_vector is CarbonBoxModel.production_vector

    # Perron CBMs share _PerronCBM methods; non-Perron CBMs do not expose _perron_simplex.
    for cls in (BalancedCBM, SubspaceCBM, OpenCBM):
        assert issubclass(cls, cbm_mod._PerronCBM)
        assert cls.edge_log_rates is cbm_mod._PerronCBM.edge_log_rates
        assert cls.edge_rates is cbm_mod._PerronCBM.edge_rates
        assert cls._perron_simplex is cbm_mod._PerronCBM._perron_simplex
        assert cls.reservoir_masses is cbm_mod._PerronCBM.reservoir_masses
        assert cls.relative_reservoir_masses is cbm_mod._PerronCBM.relative_reservoir_masses

    for cls in (FixedCBM, ReversibleCBM, EffectiveCBM):
        assert not hasattr(cls, "_perron_simplex")
    assert not hasattr(EffectiveCBM(K=3), "prod_weights")


def test_abstract_base_raises_not_implemented() -> None:
    """A bare CarbonBoxModel subclass preserves NotImplementedError defaults."""

    class BareCBM(CarbonBoxModel):
        def __init__(self, topology: Topology):
            self.topology = topology
            self.decay_rate = 1.2e-4

    bare = BareCBM(_symmetric_3box_topology())
    with pytest.raises(NotImplementedError):
        bare.edge_rates()
    with pytest.raises(NotImplementedError):
        bare.rate_matrix_K()
    with pytest.raises(NotImplementedError):
        bare.production_vector()


def test_pytree_leaf_names_order_and_zodiax_updates() -> None:
    """Dynamic PyTree leaf names, order, and zodiax updates stay identical for all CBMs."""
    topo_asym = _asymmetric_3box_topology()
    topo_sym = _symmetric_3box_topology()

    fixed = FixedCBM(
        topo_asym,
        masses=np.array([92.0, 540.0, 3100.0]),
        edge_fluxes=np.array([48.0, 40.0, 85.0, 77.0, 8.0]),
        prod_weights=(0.68, 0.32),
        sink_rates=np.array([0.0, 0.0, 1e-4]),
    )
    bal = BalancedCBM(
        topo_asym,
        log_rates_free=np.array([-0.6, -2.5, -1.9, -3.7, -5.8]),
        log_total_mass=8.2,
        prod_weights=(0.68, 0.32),
    )
    sub = FisherSubspaceCBM(
        topo_asym,
        base_log_rates_free=np.array([-0.6, -2.5, -1.9, -3.7, -5.8]),
        basis=np.array(
            [
                [0.4, -0.1],
                [-0.2, 0.5],
                [0.3, 0.2],
                [-0.1, -0.4],
                [0.2, 0.3],
            ]
        ),
        base_log_total_mass=8.2,
        alpha=np.array([0.35, -0.25]),
        prod_weights=(0.68, 0.32),
    )
    rev = ReversibleCBM(
        topo_sym,
        log_masses=np.log([90.0, 520.0, 2800.0]),
        log_sym_fluxes=np.log([47.0, 95.0]),
        prod_weights=(0.72, 0.28),
    )
    open_cbm = OpenCBM(
        topo_asym,
        log_rates_free=np.array([-0.6, -2.5, -1.9, -3.7, -5.8]),
        log_total_mass=8.2,
        log_sink_rates=np.array([-8.5]),
        open_box_indices=(2,),
        prod_weights=(0.68, 0.32),
    )
    eff_cal = EffectiveCBM(K=3)
    eff_exp = EffectiveCBM(K=3, deep_sink_rate=0.015)

    assert _leaf_names(fixed) == ["masses", "edge_fluxes", "sink_rates_arr"]
    assert _leaf_names(bal) == ["log_rates_free", "log_total_mass"]
    assert _leaf_names(sub) == ["alpha"]
    assert _leaf_names(rev) == ["log_masses", "log_sym_fluxes"]
    assert _leaf_names(open_cbm) == ["log_rates_free", "log_total_mass", "log_sink_rates"]
    assert _leaf_names(eff_cal) == [
        "log_k_strat_to_trop",
        "log_k_trop_to_strat",
        "log_k_trop_to_sinks",
        "sink_tau_params",
    ]
    assert _leaf_names(eff_exp) == [
        "log_k_strat_to_trop",
        "log_k_trop_to_strat",
        "log_k_trop_to_sinks",
        "sink_tau_params",
        "explicit_deep_sink_rate",
    ]

    # Zodiax .get / .set round-trip on Perron subclasses.
    bal2 = bal.set("log_total_mass", jnp.asarray(9.5, dtype=jnp.float64))
    assert float(bal2.get("log_total_mass")) == pytest.approx(9.5, rel=0.0, abs=0.0)
    sub2 = sub.set("alpha", jnp.array([0.1, 0.2], dtype=jnp.float64))
    np.testing.assert_allclose(sub2.get("alpha"), [0.1, 0.2], rtol=0.0, atol=0.0)
    open2 = open_cbm.set("log_sink_rates", jnp.array([-7.0], dtype=jnp.float64))
    np.testing.assert_allclose(open2.get("log_sink_rates"), [-7.0], rtol=0.0, atol=0.0)


def test_exact_log_total_mass_gauge_invariance_perron_cbms() -> None:
    """relative_reservoir_masses and steady_state_d14c never touch exp(log_total_mass)."""
    topo = _asymmetric_3box_topology()
    tie = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    lr = np.array([-0.55, -2.4, -2.1, -5.6])

    for cbm_lo, cbm_hi in (
        (
            BalancedCBM(topo, lr, 7.5, prod_weights=(0.65, 0.35), tie_matrix=tie),
            BalancedCBM(topo, lr, 800.0, prod_weights=(0.65, 0.35), tie_matrix=tie),
        ),
        (
            OpenCBM(
                topo,
                lr,
                7.5,
                np.array([-8.2]),
                open_box_indices=(2,),
                prod_weights=(0.65, 0.35),
                tie_matrix=tie,
            ),
            OpenCBM(
                topo,
                lr,
                800.0,
                np.array([-8.2]),
                open_box_indices=(2,),
                prod_weights=(0.65, 0.35),
                tie_matrix=tie,
            ),
        ),
    ):
        # log_total_mass=800.0 overflows exp(800) to inf; gauge-invariant path must stay finite
        # and bitwise identical to log_total_mass=7.5.
        rel_lo = np.asarray(cbm_lo.relative_reservoir_masses())
        rel_hi = np.asarray(cbm_hi.relative_reservoir_masses())
        assert np.all(np.isfinite(rel_hi))
        np.testing.assert_array_equal(rel_lo, rel_hi)

        d14c_lo = np.asarray(cbm_lo.steady_state_d14c())
        d14c_hi = np.asarray(cbm_hi.steady_state_d14c())
        assert np.all(np.isfinite(d14c_hi))
        np.testing.assert_array_equal(d14c_lo, d14c_hi)

        # Autodiff gradient w.r.t. log_total_mass must be exact 0.0 (not merely 1e-16).
        grad_rel = eqx.filter_grad(lambda c: jnp.sum(c.relative_reservoir_masses() ** 2))(cbm_lo)
        grad_d14c = eqx.filter_grad(lambda c: jnp.sum(c.steady_state_d14c() ** 2))(cbm_lo)
        assert float(grad_rel.log_total_mass) == 0.0
        assert float(grad_d14c.log_total_mass) == 0.0
        # Non-trivial gradient on log_rates_free confirms the test is sensitive.
        assert float(jnp.linalg.norm(grad_d14c.log_rates_free)) > 1e-2


def test_subspace_to_balanced_and_open_2d_flux_inversion() -> None:
    """SubspaceCBM matches to_balanced_cbm() and OpenCBM.from_masses_and_fluxes accepts 2D F."""
    topo = _asymmetric_3box_topology()
    masses = np.array([92.0, 540.0, 3100.0])
    edge_fluxes = np.array([48.0, 40.0, 85.0, 77.0, 8.0])
    F = np.zeros((3, 3), dtype=np.float64)
    for s, d, f in zip(topo.edges_src, topo.edges_dst, edge_fluxes):
        F[s, d] = f

    sinks = np.array([0.0, 0.0, 2.5e-4])
    open_1d = OpenCBM.from_masses_and_fluxes(
        topo, masses, edge_fluxes, sinks, prod_fractions=(0.67, 0.33)
    )
    open_2d = OpenCBM.from_masses_and_fluxes(
        topo, masses, F, sinks, prod_fractions=(0.67, 0.33)
    )
    np.testing.assert_allclose(open_1d.edge_rates(), open_2d.edge_rates(), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        open_1d.reservoir_masses(), masses, rtol=1e-13, atol=1e-13
    )
    np.testing.assert_allclose(open_1d.flux_matrix(), F, rtol=1e-13, atol=1e-13)

    bal = BalancedCBM.from_masses_and_fluxes(
        topo, masses, F, prod_weights=(0.67, 0.33)
    )
    basis = np.array([[0.3, -0.2], [-0.4, 0.1], [0.2, 0.3], [-0.1, 0.5], [0.25, -0.35]])
    sub = SubspaceCBM(
        topo,
        base_log_rates_free=bal.log_rates_free,
        basis=basis,
        base_log_total_mass=float(bal.log_total_mass),
        alpha=np.array([0.4, -0.3]),
        prod_weights=(0.67, 0.33),
    )
    bal_eq = sub.to_balanced_cbm()
    np.testing.assert_allclose(sub.edge_log_rates(), bal_eq.edge_log_rates(), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(sub.rate_matrix_K(), bal_eq.rate_matrix_K(), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(sub.reservoir_masses(), bal_eq.reservoir_masses(), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        sub.relative_reservoir_masses(), bal_eq.relative_reservoir_masses(), rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        sub.steady_state_d14c(), bal_eq.steady_state_d14c(), rtol=0.0, atol=0.0
    )


def test_filter_grad_matches_finite_differences_all_cbms() -> None:
    """eqx.filter_grad agrees with central finite differences across all mutable CBMs."""
    topo_asym = _asymmetric_3box_topology()
    topo_sym = _symmetric_3box_topology()
    weights = jnp.array([1.3, -0.7, 0.9], dtype=jnp.float64)

    def objective(cbm: CarbonBoxModel) -> jax.Array:
        return jnp.sum(weights * cbm.steady_state_d14c()) + 0.05 * jnp.sum(
            cbm.reservoir_masses()
        )

    cases: list[CarbonBoxModel] = [
        BalancedCBM(
            topo_asym,
            np.array([-0.6, -2.5, -1.9, -3.7, -5.8]),
            8.2,
            prod_weights=(0.68, 0.32),
        ),
        SubspaceCBM(
            topo_asym,
            np.array([-0.6, -2.5, -1.9, -3.7, -5.8]),
            np.array([[0.3, -0.1], [-0.2, 0.4], [0.25, 0.2], [-0.15, -0.3], [0.1, 0.25]]),
            8.2,
            alpha=np.array([0.2, -0.35]),
            prod_weights=(0.68, 0.32),
        ),
        ReversibleCBM(
            topo_sym,
            np.log([92.0, 540.0, 2900.0]),
            np.log([47.0, 98.0]),
            prod_weights=(0.71, 0.29),
        ),
        OpenCBM(
            topo_asym,
            np.array([-0.6, -2.5, -1.9, -3.7, -5.8]),
            8.2,
            np.array([-8.3]),
            open_box_indices=(2,),
            prod_weights=(0.68, 0.32),
        ),
        EffectiveCBM(K=3),
    ]

    eps = 1e-5
    for cbm in cases:
        grad_tree = eqx.filter_grad(objective)(cbm)
        dyn, static = eqx.partition(cbm, eqx.is_array)
        leaves, treedef = jax.tree_util.tree_flatten(dyn)
        grad_leaves = jax.tree_util.tree_leaves(eqx.filter(grad_tree, eqx.is_array))
        assert len(leaves) == len(grad_leaves)
        for idx, (leaf, g_ad) in enumerate(zip(leaves, grad_leaves)):
            flat_leaf = np.asarray(leaf, dtype=np.float64).reshape(-1)
            flat_fd = np.empty_like(flat_leaf)
            for j in range(flat_leaf.size):
                plus = flat_leaf.copy()
                minus = flat_leaf.copy()
                plus[j] += eps
                minus[j] -= eps
                l_plus = list(leaves)
                l_minus = list(leaves)
                l_plus[idx] = jnp.asarray(plus.reshape(leaf.shape), dtype=jnp.float64)
                l_minus[idx] = jnp.asarray(minus.reshape(leaf.shape), dtype=jnp.float64)
                f_plus = float(objective(eqx.combine(treedef.unflatten(l_plus), static)))
                f_minus = float(objective(eqx.combine(treedef.unflatten(l_minus), static)))
                flat_fd[j] = (f_plus - f_minus) / (2.0 * eps)
            np.testing.assert_allclose(
                np.asarray(g_ad, dtype=np.float64).reshape(-1),
                flat_fd,
                rtol=1e-5,
                atol=1e-6,
            )


def test_constructor_scc_and_prod_weight_validation() -> None:
    """Constructor preambles enforce SCC on balanced/open/reversible CBMs but not FixedCBM."""
    reducible = Topology(
        box_names=("Stratosphere", "Troposphere"),
        edges_src=(0,),
        edges_dst=(1,),
        prod_box_indices=(0, 1),
        trop_box_indices=(1,),
        check_scc=False,
    )
    fixed = FixedCBM(reducible, masses=[100.0, 500.0], edge_fluxes=[50.0])
    np.testing.assert_allclose(fixed.production_vector(), [0.5, 0.5])

    with pytest.raises(ValueError, match="strongly connected"):
        BalancedCBM(reducible, log_rates_free=[0.0], log_total_mass=6.0)
    with pytest.raises(ValueError, match="strongly connected"):
        SubspaceCBM(reducible, base_log_rates_free=[0.0], basis=[[1.0]], base_log_total_mass=6.0)
    with pytest.raises(ValueError, match="strongly connected"):
        ReversibleCBM(reducible, log_masses=[4.0, 6.0], log_sym_fluxes=[3.0])
    with pytest.raises(ValueError, match="strongly connected"):
        OpenCBM(
            reducible,
            log_rates_free=[0.0],
            log_total_mass=6.0,
            log_sink_rates=[-5.0],
            open_box_indices=(1,),
        )
