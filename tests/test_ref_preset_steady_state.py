"""Independent plain-NumPy reference tests for steady-state 14C of every preset."""

from __future__ import annotations

import sys
from pathlib import Path

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


PRESET_CASES: list[tuple[str, str, bool, str, bool, float, float, float]] = [
    # (name, kind, legacy_v1_units, miyake17_closure, tie_symmetric_edges, q_std, alpha_unit, decay_rate)
    ("Guttler15", "fixed", False, "open_ocean_sink", False, DEFAULT_Q_STD, ALPHA_ATOMS_CM2_S_TO_KG_YR, DECAY_CAMBRIDGE),
    ("Guttler15", "balanced", False, "open_ocean_sink", False, DEFAULT_Q_STD, ALPHA_ATOMS_CM2_S_TO_KG_YR, DECAY_CAMBRIDGE),
    ("Guttler15", "balanced", False, "open_ocean_sink", False, 2.15, ALPHA_ATOMS_CM2_S_TO_KG_YR, DECAY_KUTSCHERA),
    ("Brehm21", "fixed", False, "open_ocean_sink", False, DEFAULT_Q_STD, ALPHA_ATOMS_CM2_S_TO_KG_YR, DECAY_CAMBRIDGE),
    ("Brehm21", "balanced", False, "open_ocean_sink", False, DEFAULT_Q_STD, ALPHA_ATOMS_CM2_S_TO_KG_YR, DECAY_CAMBRIDGE),
    ("Brehm21", "balanced", False, "open_ocean_sink", True, 1.92, ALPHA_ATOMS_CM2_S_TO_KG_YR, DECAY_CAMBRIDGE),
    ("Buntgen18", "fixed", False, "open_ocean_sink", False, DEFAULT_Q_STD, ALPHA_ATOMS_CM2_S_TO_KG_YR, DECAY_CAMBRIDGE),
    ("Buntgen18", "balanced", False, "open_ocean_sink", False, DEFAULT_Q_STD, ALPHA_ATOMS_CM2_S_TO_KG_YR, DECAY_CAMBRIDGE),
    ("Buntgen18", "balanced", False, "open_ocean_sink", True, 1.85, ALPHA_ATOMS_CM2_S_TO_KG_YR, DECAY_CAMBRIDGE),
    ("Miyake17", "fixed", False, "open_ocean_sink", False, DEFAULT_Q_STD, ALPHA_ATOMS_CM2_S_TO_KG_YR, DECAY_CAMBRIDGE),
    ("Miyake17", "open", False, "open_ocean_sink", False, 2.05, ALPHA_ATOMS_CM2_S_TO_KG_YR, DECAY_CAMBRIDGE),
    ("Miyake17", "fixed", False, "closed_shallow", False, DEFAULT_Q_STD, ALPHA_ATOMS_CM2_S_TO_KG_YR, DECAY_CAMBRIDGE),
    ("Miyake17", "balanced", False, "closed_shallow", False, DEFAULT_Q_STD, ALPHA_ATOMS_CM2_S_TO_KG_YR, DECAY_CAMBRIDGE),
    ("Miyake17", "reversible", False, "closed_shallow", False, DEFAULT_Q_STD, ALPHA_ATOMS_CM2_S_TO_KG_YR, DECAY_CAMBRIDGE),
    ("Miyake17", "fixed", True, "open_ocean_sink", False, DEFAULT_Q_STD, ALPHA_LEGACY_V1, DECAY_KUTSCHERA),
    ("Miyake17", "balanced", True, "open_ocean_sink", False, DEFAULT_Q_STD, ALPHA_LEGACY_V1, DECAY_KUTSCHERA),
    ("Miyake17", "reversible", True, "open_ocean_sink", False, DEFAULT_Q_STD, ALPHA_LEGACY_V1, DECAY_KUTSCHERA),
]


@pytest.mark.parametrize(
    (
        "name",
        "kind",
        "legacy_v1_units",
        "miyake17_closure",
        "tie_symmetric_edges",
        "q_std",
        "alpha_unit",
        "decay_rate",
    ),
    PRESET_CASES,
)
def test_preset_steady_state_against_plain_numpy_solve(
    name: str,
    kind: str,
    legacy_v1_units: bool,
    miyake17_closure: str,
    tie_symmetric_edges: bool,
    q_std: float,
    alpha_unit: float,
    decay_rate: float,
) -> None:
    """Verify ``steady_state_c14``, conservation, and ``production_for_target_c14``."""
    spec = get_preset_spec(
        name,
        legacy_v1_units=legacy_v1_units,
        miyake17_closure=miyake17_closure,  # type: ignore[arg-type]
        decay_rate=decay_rate,
    )
    cbm = load_preset(
        name,
        kind=kind,  # type: ignore[arg-type]
        legacy_v1_units=legacy_v1_units,
        miyake17_closure=miyake17_closure,  # type: ignore[arg-type]
        tie_symmetric_edges=tie_symmetric_edges,
        decay_rate=decay_rate,
    )

    y_np, b_np, sink_rates = _numpy_steady_state(
        spec,
        q_std=q_std,
        alpha_unit=alpha_unit,
        decay_rate=decay_rate,
    )

    y_cbm = np.asarray(cbm.steady_state_c14(q_std=q_std, alpha_unit=alpha_unit), dtype=np.float64)
    np.testing.assert_allclose(y_cbm, y_np, rtol=1e-12, atol=0.0)

    # Global particle conservation at steady state: decay + open sinks == total production.
    loss_total = decay_rate * float(np.sum(y_cbm)) + float(np.sum(sink_rates * y_cbm))
    np.testing.assert_allclose(loss_total, float(np.sum(b_np)), rtol=1e-12, atol=0.0)

    # production_for_target_c14 inverts the plain-NumPy steady-state solve.
    trop_target = float(np.sum(y_np[list(spec.trop_box_indices)]))
    q_recovered_trop = float(
        cbm.production_for_target_c14(trop_target, alpha_unit=alpha_unit)
    )
    np.testing.assert_allclose(q_recovered_trop, q_std, rtol=1e-12, atol=0.0)

    q_recovered_box0 = float(
        cbm.production_for_target_c14(float(y_np[0]), box=0, alpha_unit=alpha_unit)
    )
    np.testing.assert_allclose(q_recovered_box0, q_std, rtol=1e-12, atol=0.0)
