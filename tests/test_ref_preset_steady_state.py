"""Independent plain-NumPy reference tests for steady-state 14C of every preset."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from ticktack.constants import (
    ALPHA_ATOMS_CM2_S_TO_KG_YR,
    ALPHA_LEGACY_V1,
    DECAY_CAMBRIDGE,
    DECAY_KUTSCHERA,
    DEFAULT_Q_STD,
)
from ticktack.presets import PresetSpec, get_preset_spec, load_preset


class PresetCase(NamedTuple):
    """Parameter bundle for one preset steady-state verification case."""

    name: str
    kind: str
    legacy_v1_units: bool
    miyake17_closure: str
    tie_symmetric_edges: bool
    q_std: float
    alpha_unit: float
    decay_rate: float


def _numpy_steady_state(
    spec: PresetSpec,
    *,
    q_std: float,
    alpha_unit: float,
    decay_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve ``-A_np y_np = b_np`` purely in NumPy from ``spec`` masses and fluxes."""
    n = spec.n_boxes
    masses = np.asarray(spec.masses, dtype=np.float64)
    flux_mat = np.zeros((n, n), dtype=np.float64)
    for src, dst, flux in spec.edges:
        flux_mat[src, dst] += flux

    k_np = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i != j:
                k_np[j, i] = flux_mat[i, j] / masses[i]
        k_np[i, i] = -np.sum(k_np[:, i])

    sink_rates = np.asarray(spec.sink_rates, dtype=np.float64)
    a_np = k_np - decay_rate * np.eye(n, dtype=np.float64) - np.diag(sink_rates)
    prod_fractions = np.asarray(spec.prod_coefficients, dtype=np.float64)
    b_np = q_std * alpha_unit * prod_fractions
    y_np = np.linalg.solve(-a_np, b_np)
    return y_np, b_np, sink_rates


PRESET_CASES: list[PresetCase] = [
    PresetCase("Guttler15", "fixed", False, "open_ocean_sink", False, DEFAULT_Q_STD, ALPHA_ATOMS_CM2_S_TO_KG_YR, DECAY_CAMBRIDGE),
    PresetCase("Guttler15", "balanced", False, "open_ocean_sink", False, DEFAULT_Q_STD, ALPHA_ATOMS_CM2_S_TO_KG_YR, DECAY_CAMBRIDGE),
    PresetCase("Guttler15", "balanced", False, "open_ocean_sink", False, 2.15, ALPHA_ATOMS_CM2_S_TO_KG_YR, DECAY_KUTSCHERA),
    PresetCase("Brehm21", "fixed", False, "open_ocean_sink", False, DEFAULT_Q_STD, ALPHA_ATOMS_CM2_S_TO_KG_YR, DECAY_CAMBRIDGE),
    PresetCase("Brehm21", "balanced", False, "open_ocean_sink", False, DEFAULT_Q_STD, ALPHA_ATOMS_CM2_S_TO_KG_YR, DECAY_CAMBRIDGE),
    PresetCase("Brehm21", "balanced", False, "open_ocean_sink", True, 1.92, ALPHA_ATOMS_CM2_S_TO_KG_YR, DECAY_CAMBRIDGE),
    PresetCase("Buntgen18", "fixed", False, "open_ocean_sink", False, DEFAULT_Q_STD, ALPHA_ATOMS_CM2_S_TO_KG_YR, DECAY_CAMBRIDGE),
    PresetCase("Buntgen18", "balanced", False, "open_ocean_sink", False, DEFAULT_Q_STD, ALPHA_ATOMS_CM2_S_TO_KG_YR, DECAY_CAMBRIDGE),
    PresetCase("Buntgen18", "balanced", False, "open_ocean_sink", True, 1.85, ALPHA_ATOMS_CM2_S_TO_KG_YR, DECAY_CAMBRIDGE),
    PresetCase("Miyake17", "fixed", False, "open_ocean_sink", False, DEFAULT_Q_STD, ALPHA_ATOMS_CM2_S_TO_KG_YR, DECAY_CAMBRIDGE),
    PresetCase("Miyake17", "open", False, "open_ocean_sink", False, 2.05, ALPHA_ATOMS_CM2_S_TO_KG_YR, DECAY_CAMBRIDGE),
    PresetCase("Miyake17", "fixed", False, "closed_shallow", False, DEFAULT_Q_STD, ALPHA_ATOMS_CM2_S_TO_KG_YR, DECAY_CAMBRIDGE),
    PresetCase("Miyake17", "balanced", False, "closed_shallow", False, DEFAULT_Q_STD, ALPHA_ATOMS_CM2_S_TO_KG_YR, DECAY_CAMBRIDGE),
    PresetCase("Miyake17", "reversible", False, "closed_shallow", False, DEFAULT_Q_STD, ALPHA_ATOMS_CM2_S_TO_KG_YR, DECAY_CAMBRIDGE),
    PresetCase("Miyake17", "fixed", True, "open_ocean_sink", False, DEFAULT_Q_STD, ALPHA_LEGACY_V1, DECAY_KUTSCHERA),
    PresetCase("Miyake17", "balanced", True, "open_ocean_sink", False, DEFAULT_Q_STD, ALPHA_LEGACY_V1, DECAY_KUTSCHERA),
    PresetCase("Miyake17", "reversible", True, "open_ocean_sink", False, DEFAULT_Q_STD, ALPHA_LEGACY_V1, DECAY_KUTSCHERA),
]


@pytest.mark.parametrize(
    "case",
    PRESET_CASES,
    ids=[
        f"{c.name}-{c.kind}-legacy={c.legacy_v1_units}-{c.miyake17_closure}-tie={c.tie_symmetric_edges}-q={c.q_std}"
        for c in PRESET_CASES
    ],
)
def test_preset_steady_state_against_plain_numpy_solve(case: PresetCase) -> None:
    """Verify ``steady_state_c14``, conservation, ``d14c``, and ``production_for_target_c14``."""
    spec = get_preset_spec(
        case.name,
        legacy_v1_units=case.legacy_v1_units,
        miyake17_closure=case.miyake17_closure,  # type: ignore[arg-type]
        decay_rate=case.decay_rate,
    )
    cbm = load_preset(
        case.name,
        kind=case.kind,  # type: ignore[arg-type]
        legacy_v1_units=case.legacy_v1_units,
        miyake17_closure=case.miyake17_closure,  # type: ignore[arg-type]
        tie_symmetric_edges=case.tie_symmetric_edges,
        decay_rate=case.decay_rate,
    )

    y_np, b_np, sink_rates = _numpy_steady_state(
        spec,
        q_std=case.q_std,
        alpha_unit=case.alpha_unit,
        decay_rate=case.decay_rate,
    )

    y_cbm = np.asarray(
        cbm.steady_state_c14(q_std=case.q_std, alpha_unit=case.alpha_unit),
        dtype=np.float64,
    )
    np.testing.assert_allclose(y_cbm, y_np, rtol=1e-12, atol=0.0)

    # Check reservoir_masses(), reference_specific_activity(), and steady_state_d14c()
    # (exercises _perron_frobenius_simplex on BalancedCBM and OpenCBM).
    masses_np = np.asarray(spec.masses, dtype=np.float64)
    np.testing.assert_allclose(
        np.asarray(cbm.reservoir_masses(), dtype=np.float64),
        masses_np,
        rtol=1e-12,
        atol=0.0,
    )
    trop_idx = list(spec.trop_box_indices)
    r_ref_np = float(np.sum(y_np[trop_idx]) / np.sum(masses_np[trop_idx]))
    np.testing.assert_allclose(
        float(cbm.reference_specific_activity(q_std=case.q_std, alpha_unit=case.alpha_unit)),
        r_ref_np,
        rtol=1e-12,
        atol=0.0,
    )
    d14c_np = 1000.0 * ((y_np / masses_np) / r_ref_np - 1.0)
    np.testing.assert_allclose(
        np.asarray(cbm.steady_state_d14c(q_std=case.q_std, alpha_unit=case.alpha_unit), dtype=np.float64),
        d14c_np,
        rtol=1e-12,
        atol=1e-11,
    )

    # Non-tautological closure check for Miyake17 open_ocean_sink:
    # R_ref(Miyake17) must match independent plain-NumPy R_ref(Guttler15).
    if (
        case.name == "Miyake17"
        and case.miyake17_closure == "open_ocean_sink"
        and not case.legacy_v1_units
    ):
        g15_spec = get_preset_spec("Guttler15", decay_rate=case.decay_rate)
        y_g15_np, _, _ = _numpy_steady_state(
            g15_spec,
            q_std=case.q_std,
            alpha_unit=case.alpha_unit,
            decay_rate=case.decay_rate,
        )
        g15_trop = list(g15_spec.trop_box_indices)
        g15_masses = np.asarray(g15_spec.masses, dtype=np.float64)
        r_ref_g15 = float(np.sum(y_g15_np[g15_trop]) / np.sum(g15_masses[g15_trop]))
        np.testing.assert_allclose(
            float(y_cbm[trop_idx[0]] / masses_np[trop_idx[0]]),
            r_ref_g15,
            rtol=1e-12,
            atol=0.0,
        )

    # Global particle conservation at steady state: decay + open sinks == total production.
    loss_total = case.decay_rate * float(np.sum(y_cbm)) + float(np.sum(sink_rates * y_cbm))
    np.testing.assert_allclose(loss_total, float(np.sum(b_np)), rtol=1e-12, atol=0.0)

    # production_for_target_c14 inverts the plain-NumPy steady-state solve.
    trop_target = float(np.sum(y_np[trop_idx]))
    q_recovered_trop = float(
        cbm.production_for_target_c14(trop_target, alpha_unit=case.alpha_unit)
    )
    np.testing.assert_allclose(q_recovered_trop, case.q_std, rtol=1e-12, atol=0.0)

    q_recovered_box0 = float(
        cbm.production_for_target_c14(float(y_np[0]), box=0, alpha_unit=case.alpha_unit)
    )
    np.testing.assert_allclose(q_recovered_box0, case.q_std, rtol=1e-12, atol=0.0)

    if any(h in ("north", "south") for h in spec.hemispheres):
        for hem in ("north", "south"):
            hem_trop_idx = spec.box_index("Troposphere", hemisphere=hem)
            q_hem_str = float(
                cbm.production_for_target_c14(
                    float(y_np[hem_trop_idx]),
                    box="Troposphere",
                    hemisphere=hem,
                    alpha_unit=case.alpha_unit,
                )
            )
            np.testing.assert_allclose(q_hem_str, case.q_std, rtol=1e-12, atol=0.0)

            q_hem_default = float(
                cbm.production_for_target_c14(
                    float(y_np[hem_trop_idx]),
                    hemisphere=hem,
                    alpha_unit=case.alpha_unit,
                )
            )
            np.testing.assert_allclose(q_hem_default, case.q_std, rtol=1e-12, atol=0.0)
