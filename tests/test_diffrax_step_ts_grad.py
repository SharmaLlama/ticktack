"""Regression tests for ``DiffraxSolver`` spike-timing gradients.

Verifies that stopping gradients on continuous ``step_ts`` knots and on the
automatically selected integration start time ``t_start_int`` eliminates
``PIDController`` step-clipping gradient corruption and eager-vs-``jax.jit``
divergence, while keeping discontinuous ``jump_ts`` timestamps differentiable
for ``boxcar`` pulses and preserving explicit ``t0`` sensitivity when ``y0``
is provided.
"""

from __future__ import annotations

from collections.abc import Callable
import gc
from pathlib import Path
import sys

import jax

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ticktack.cbm import BalancedCBM, CarbonBoxModel, Topology  # noqa: E402
from ticktack.constants import DEFAULT_Q_STD  # noqa: E402
from ticktack.presets import load_preset  # noqa: E402
from ticktack.production import (  # noqa: E402
    GriddedProduction,
    HarmonicSolar,
    MiyakeSpike,
    ProductionTerm,
    SteadyProduction,
)
from ticktack.solvers import DiffraxSolver, ExactLinearSolver  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_jax_caches() -> None:
    """Keep XLA compilation cache bounded across parametrized ODE solves."""
    jax.clear_caches()
    gc.collect()


def _central_fd_grad(
    fn: Callable[[jax.Array], jax.Array],
    x: jax.Array,
    eps: float = 1e-6,
) -> np.ndarray:
    """Second-order central finite-difference gradient of a scalar JAX function."""
    grads = []
    for idx in range(int(x.shape[0])):
        xp = x.at[idx].add(eps)
        xm = x.at[idx].add(-eps)
        grads.append((float(fn(xp)) - float(fn(xm))) / (2.0 * eps))
    return np.asarray(grads, dtype=np.float64)


def _make_near_degenerate_3box() -> BalancedCBM:
    """Three-box chain with nearly degenerate exchange rates (reproducer #17)."""
    topo = Topology(
        box_names=("Stratosphere", "Troposphere", "Deep"),
        edges_src=(0, 1, 1, 2),
        edges_dst=(1, 0, 2, 1),
        prod_box_indices=(0,),
    )
    rates = jnp.array([1.0, 0.5, 1.0, 0.5 + 1e-13], dtype=jnp.float64)
    return BalancedCBM(
        topology=topo,
        log_rates_free=jnp.log(rates),
        log_total_mass=jnp.log(jnp.array(1000.0, dtype=jnp.float64)),
    )


def _make_spike_loss(
    cbm: CarbonBoxModel,
    solver: ExactLinearSolver | DiffraxSolver,
    spike_kwargs: dict[str, object],
    *,
    binned: bool,
    t_eval: jax.Array | None = None,
) -> Callable[[jax.Array], jax.Array]:
    """Scalar quadratic trajectory or seasonal-window loss wrt ``(t_start, duration, fluence)``."""
    t_pts = (
        jnp.array([773.8, 774.5, 775.2, 776.4], dtype=jnp.float64)
        if t_eval is None
        else t_eval
    )
    t_starts = jnp.array([773.3, 774.3, 775.3, 776.3], dtype=jnp.float64)
    t_ends = jnp.array([773.8, 774.8, 775.8, 776.8], dtype=jnp.float64)

    def _loss(params: jax.Array) -> jax.Array:
        prod: ProductionTerm = (
            SteadyProduction(1.76)
            + HarmonicSolar.from_polar(
                periods=11.0, amplitudes=0.25, phases=0.4, t_ref=774.0
            )
            + MiyakeSpike(params[0], params[1], params[2], **spike_kwargs)
        )
        if binned:
            dy = solver.solve_delta_binned(cbm, prod, t_starts, t_ends)
        else:
            dy = solver.solve_delta_trajectory(cbm, prod, t_pts)
        return jnp.sum(dy[:, 1] ** 2)

    return _loss


def _assert_grad_matches(
    loss_exact: Callable[[jax.Array], jax.Array],
    loss_dfx: Callable[[jax.Array], jax.Array],
    p0: jax.Array,
    *,
    rtol_exact: float,
    rtol_fd: float,
    rtol_eager_jit: float,
) -> None:
    """Assert eager and JIT ``DiffraxSolver`` gradients agree with exact and FD references."""
    g_exact = np.asarray(jax.grad(loss_exact)(p0), dtype=np.float64)
    g_fd = _central_fd_grad(jax.jit(loss_exact), p0)
    g_eager = np.asarray(jax.grad(loss_dfx)(p0), dtype=np.float64)
    g_jit = np.asarray(jax.jit(jax.grad(loss_dfx))(p0), dtype=np.float64)

    np.testing.assert_allclose(g_exact, g_fd, rtol=rtol_fd, atol=1e-8)
    np.testing.assert_allclose(g_eager, g_exact, rtol=rtol_exact, atol=1e-8)
    np.testing.assert_allclose(g_jit, g_exact, rtol=rtol_exact, atol=1e-8)
    np.testing.assert_allclose(g_eager, g_jit, rtol=rtol_eager_jit, atol=1e-8)


def test_3box_near_degenerate_supergaussian_grad_eager_and_jit() -> None:
    """Reproducer #17: near-degenerate 3-box CBM must not corrupt ``(t_start, duration)`` grads."""
    cbm = _make_near_degenerate_3box()
    exact = ExactLinearSolver()
    dfx = DiffraxSolver("Tsit5", rtol=1e-12, atol=1e-12)
    p0 = jnp.array([774.40, 0.25, 5.5], dtype=jnp.float64)
    kw = {"profile": "supergaussian", "exponent": 10.0}

    _assert_grad_matches(
        _make_spike_loss(cbm, exact, kw, binned=False),
        _make_spike_loss(cbm, dfx, kw, binned=False),
        p0,
        rtol_exact=1e-7,
        rtol_fd=1e-5,
        rtol_eager_jit=1e-8,
    )


@pytest.mark.parametrize(
    "spike_kwargs",
    [
        {"profile": "supergaussian", "exponent": 10.0},
        {"profile": "skewgaussian", "skewness": 2.0},
        {"profile": "gamma", "gamma_shape": 3.0},
    ],
    ids=["supergaussian", "skewgaussian", "gamma_k3"],
)
@pytest.mark.parametrize("binned", [False, True], ids=["pointwise", "windows"])
def test_guttler15_smooth_spikes_tsit5_grad_eager_and_jit(
    spike_kwargs: dict[str, object],
    binned: bool,
) -> None:
    """Smooth ``MiyakeSpike`` profiles agree with exact and FD grads in eager and JIT modes."""
    cbm = load_preset("Guttler15")
    exact = ExactLinearSolver()
    dfx = DiffraxSolver("Tsit5", rtol=1e-12, atol=1e-12)
    p0 = jnp.array([774.40, 0.25, 5.5], dtype=jnp.float64)

    _assert_grad_matches(
        _make_spike_loss(cbm, exact, spike_kwargs, binned=binned),
        _make_spike_loss(cbm, dfx, spike_kwargs, binned=binned),
        p0,
        rtol_exact=1e-7,
        rtol_fd=1e-5,
        rtol_eager_jit=1e-8,
    )


@pytest.mark.parametrize(
    "spike_kwargs,binned,rtol_exact",
    [
        ({"profile": "supergaussian", "exponent": 10.0}, False, 1e-5),
        ({"profile": "supergaussian", "exponent": 10.0}, True, 1e-5),
        ({"profile": "skewgaussian", "skewness": 2.0}, False, 1e-5),
        ({"profile": "boxcar"}, False, 1e-4),
    ],
    ids=["superg_pt", "superg_win", "skewg_pt", "boxcar_pt"],
)
def test_guttler15_kvaerno5_grad_eager_and_jit(
    spike_kwargs: dict[str, object],
    binned: bool,
    rtol_exact: float,
) -> None:
    """Implicit ``Kvaerno5`` solver matches exact and FD gradients across smooth and boxcar spikes."""
    cbm = load_preset("Guttler15")
    exact = ExactLinearSolver()
    dfx = DiffraxSolver("Kvaerno5", rtol=1e-10, atol=1e-10)
    p0 = jnp.array([774.40, 0.25, 5.5], dtype=jnp.float64)

    _assert_grad_matches(
        _make_spike_loss(cbm, exact, spike_kwargs, binned=binned),
        _make_spike_loss(cbm, dfx, spike_kwargs, binned=binned),
        p0,
        rtol_exact=rtol_exact,
        rtol_fd=1e-4,
        rtol_eager_jit=1e-7,
    )


@pytest.mark.parametrize("binned", [False, True], ids=["pointwise", "windows"])
def test_boxcar_jump_ts_grad_preserved(binned: bool) -> None:
    """Discontinuous ``boxcar`` spikes retain differentiable ``jump_ts`` boundary gradients."""
    cbm = load_preset("Guttler15")
    exact = ExactLinearSolver()
    dfx = DiffraxSolver("Tsit5", rtol=1e-12, atol=1e-12)
    p0 = jnp.array([774.40, 0.25, 5.5], dtype=jnp.float64)
    kw = {"profile": "boxcar"}

    loss_ex = _make_spike_loss(cbm, exact, kw, binned=binned)
    loss_dfx = _make_spike_loss(cbm, dfx, kw, binned=binned)
    g_eager = np.asarray(jax.grad(loss_dfx)(p0), dtype=np.float64)
    assert abs(g_eager[0]) > 10.0  # nonzero Leibniz jump derivative wrt t_start
    _assert_grad_matches(
        loss_ex,
        loss_dfx,
        p0,
        rtol_exact=1e-6,
        rtol_fd=1e-5,
        rtol_eager_jit=1e-8,
    )


def test_3box_near_degenerate_boxcar_grad_preserved() -> None:
    """Near-degenerate 3-box CBM preserves discontinuous ``boxcar`` gradients."""
    cbm = _make_near_degenerate_3box()
    exact = ExactLinearSolver()
    dfx = DiffraxSolver("Tsit5", rtol=1e-12, atol=1e-12)
    p0 = jnp.array([774.40, 0.25, 5.5], dtype=jnp.float64)
    kw = {"profile": "boxcar"}

    _assert_grad_matches(
        _make_spike_loss(cbm, exact, kw, binned=False),
        _make_spike_loss(cbm, dfx, kw, binned=False),
        p0,
        rtol_exact=1e-4,
        rtol_fd=1e-5,
        rtol_eager_jit=1e-8,
    )


def test_onset_before_earliest_obs_and_suess_time_varying_grad() -> None:
    """Auto ``t_start_int < earliest_obs`` and Suess ``predict_d14c`` match FD and eager/JIT."""
    cbm = load_preset("Guttler15")
    exact = ExactLinearSolver()
    tsit = DiffraxSolver("Tsit5", rtol=1e-12, atol=1e-12)
    p0 = jnp.array([774.40, 0.25, 5.5], dtype=jnp.float64)

    # First observation (774.5) is after spike onset (~774.32), so t_start_int = onset
    t_late = jnp.array([774.5, 775.2, 776.4], dtype=jnp.float64)
    kw_sg = {"profile": "supergaussian", "exponent": 10.0}
    _assert_grad_matches(
        _make_spike_loss(cbm, exact, kw_sg, binned=False, t_eval=t_late),
        _make_spike_loss(cbm, tsit, kw_sg, binned=False, t_eval=t_late),
        p0,
        rtol_exact=1e-7,
        rtol_fd=1e-5,
        rtol_eager_jit=1e-8,
    )

    # Non-autonomous Suess path (third PIDController site in DiffraxSolver.predict_d14c)
    base_m = cbm.reservoir_masses()
    base_a = cbm.matrix_A()
    t_starts = jnp.array([773.3, 774.3, 775.3, 776.3], dtype=jnp.float64)
    t_ends = jnp.array([773.8, 774.8, 775.8, 776.8], dtype=jnp.float64)

    def _masses_fn(t: jax.Array) -> jax.Array:
        scale = 1.0 + 0.02 * jax.nn.sigmoid((t - 774.5) / 0.5)
        return base_m.at[1].multiply(scale)

    suess_dfx = DiffraxSolver(
        "Tsit5",
        rtol=1e-12,
        atol=1e-12,
        masses_fn=_masses_fn,
        matrix_A_fn=lambda _t: base_a,
    )
    for kw in ({"profile": "skewgaussian", "skewness": 2.0}, {"profile": "boxcar"}):
        jax.clear_caches()

        def _suess_loss(params: jax.Array, _kw: dict[str, object] = kw) -> jax.Array:
            prod = SteadyProduction(1.76) + MiyakeSpike(
                params[0], params[1], params[2], **_kw
            )
            d14c = suess_dfx.predict_d14c(cbm, prod, t_starts, t_ends, box_indices=1)
            return jnp.sum(d14c**2)

        g_fd = _central_fd_grad(jax.jit(_suess_loss), p0)
        g_eager = np.asarray(jax.grad(_suess_loss)(p0), dtype=np.float64)
        g_jit = np.asarray(jax.jit(jax.grad(_suess_loss))(p0), dtype=np.float64)
        np.testing.assert_allclose(g_eager, g_fd, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(g_jit, g_fd, rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(g_eager, g_jit, rtol=1e-7, atol=1e-8)


def test_triangular_gridded_spike_and_explicit_t0_override_grad() -> None:
    """Piecewise-linear triangular ``GriddedProduction`` and explicit ``t0`` with ``y0`` grads."""
    cbm = load_preset("Guttler15")
    exact = ExactLinearSolver()
    dfx = DiffraxSolver("Tsit5", rtol=1e-12, atol=1e-12)
    t_eval = jnp.array([773.8, 774.5, 775.2, 776.4], dtype=jnp.float64)

    # Piecewise-linear triangular tent pulse on a uniform grid (t_grid goes to step_ts)
    t_grid = jnp.linspace(773.5, 777.0, 29, dtype=jnp.float64)
    tri_basis = jnp.maximum(0.0, 1.0 - jnp.abs(t_grid - 774.5) / 0.25)

    def _tri_loss(
        solver: ExactLinearSolver | DiffraxSolver,
    ) -> Callable[[jax.Array], jax.Array]:
        def _fn(q_vals: jax.Array) -> jax.Array:
            prod = GriddedProduction(t_grid, q_vals)
            return jnp.sum(solver.solve_delta_trajectory(cbm, prod, t_eval)[:, 1] ** 2)

        return _fn

    q0 = DEFAULT_Q_STD + 22.0 * tri_basis
    g_ex_tri = np.asarray(jax.grad(_tri_loss(exact))(q0), dtype=np.float64)
    g_eager = np.asarray(jax.grad(_tri_loss(dfx))(q0), dtype=np.float64)
    g_jit = np.asarray(jax.jit(jax.grad(_tri_loss(dfx)))(q0), dtype=np.float64)
    np.testing.assert_allclose(g_eager, g_ex_tri, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(g_jit, g_ex_tri, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(g_eager, g_jit, rtol=1e-8, atol=1e-8)

    # Explicit t0 with non-equilibrium y0 must remain differentiable wrt t0
    y_ss = cbm.steady_state_c14()
    y0 = y_ss * (1.0 + 0.02 * jnp.linspace(-1.0, 1.0, cbm.n_boxes))
    prod_steady = SteadyProduction(DEFAULT_Q_STD)

    def _t0_loss(
        solver: ExactLinearSolver | DiffraxSolver,
    ) -> Callable[[jax.Array], jax.Array]:
        def _fn(t0_val: jax.Array) -> jax.Array:
            dy = solver.solve_delta_trajectory(cbm, prod_steady, t_eval, t0=t0_val, y0=y0)
            return jnp.sum(dy[:, 1] ** 2)

        return _fn

    t0_init = jnp.array(773.0, dtype=jnp.float64)
    g_ex = float(jax.grad(_t0_loss(exact))(t0_init))
    g_dfx = float(jax.jit(jax.grad(_t0_loss(dfx)))(t0_init))
    assert abs(g_ex) > 1e-3
    np.testing.assert_allclose(g_dfx, g_ex, rtol=1e-7, atol=1e-8)


def test_y0_with_t0_none_and_onset_before_earliest_obs() -> None:
    """When ``y0`` is supplied with ``t0=None``, ``y0`` is anchored at ``earliest_obs``."""
    cbm = load_preset("Guttler15")
    exact = ExactLinearSolver()
    dfx = DiffraxSolver("Tsit5", rtol=1e-12, atol=1e-12)
    y_ss = cbm.steady_state_c14()
    y0 = y_ss * (1.0 + 0.05 * jnp.linspace(-1.0, 1.0, cbm.n_boxes))
    t_late = jnp.array([774.5, 775.2, 776.4], dtype=jnp.float64)
    t_starts = jnp.array([774.5, 775.5, 776.5], dtype=jnp.float64)
    t_ends = jnp.array([775.0, 776.0, 777.0], dtype=jnp.float64)
    p0 = jnp.array([774.40, 0.25, 5.5], dtype=jnp.float64)

    for binned in (False, True):
        jax.clear_caches()

        def _make_loss(
            solver: ExactLinearSolver | DiffraxSolver,
            _binned: bool = binned,
        ) -> Callable[[jax.Array], jax.Array]:
            def _fn(params: jax.Array) -> jax.Array:
                prod = MiyakeSpike(
                    params[0], params[1], params[2], profile="supergaussian", exponent=10.0
                )
                if _binned:
                    dy = solver.solve_delta_binned(cbm, prod, t_starts, t_ends, y0=y0)
                else:
                    dy = solver.solve_delta_trajectory(cbm, prod, t_late, y0=y0)
                return jnp.sum(dy[:, 1] ** 2)

            return _fn

        f_ex = _make_loss(exact)
        f_dfx = _make_loss(dfx)
        np.testing.assert_allclose(float(f_dfx(p0)), float(f_ex(p0)), rtol=1e-9, atol=1e-9)
        _assert_grad_matches(f_ex, f_dfx, p0, rtol_exact=1e-7, rtol_fd=1e-5, rtol_eager_jit=1e-8)


def test_gridded_production_shift_grad_via_stopped_jump_ts() -> None:
    """Shifted ``GriddedProduction.t_grid`` uses stopped ``jump_ts`` for accurate timing grads."""
    dfx = DiffraxSolver("Tsit5", rtol=1e-12, atol=1e-12)
    t_eval = jnp.array([773.8, 774.5, 775.2, 776.4], dtype=jnp.float64)
    t_base = jnp.array([773.5, 774.40, 774.525, 774.65, 777.0], dtype=jnp.float64)
    q_vals = DEFAULT_Q_STD + jnp.array([0.0, 0.0, 44.0, 0.0, 0.0], dtype=jnp.float64)

    for cbm in (load_preset("Guttler15"), _make_near_degenerate_3box()):
        jax.clear_caches()

        def _loss(shift: jax.Array, _cbm: CarbonBoxModel = cbm) -> jax.Array:
            prod = GriddedProduction(t_base + shift[0], q_vals)
            return jnp.sum(dfx.solve_delta_trajectory(_cbm, prod, t_eval)[:, 1] ** 2)

        s0 = jnp.array([0.0], dtype=jnp.float64)
        g_fd = _central_fd_grad(jax.jit(_loss), s0)
        g_eager = np.asarray(jax.grad(_loss)(s0), dtype=np.float64)
        g_jit = np.asarray(jax.jit(jax.grad(_loss))(s0), dtype=np.float64)
        np.testing.assert_allclose(g_eager, g_fd, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(g_jit, g_fd, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(g_eager, g_jit, rtol=1e-8, atol=1e-8)


def test_boxcar_predict_d14c_onset_before_first_obs() -> None:
    """Boxcar ``predict_d14c`` with spike onset before the first window stays accurate."""
    cbm = load_preset("Guttler15")
    exact = ExactLinearSolver()
    t_starts = jnp.array([774.5, 775.5, 776.5], dtype=jnp.float64)
    t_ends = jnp.array([775.0, 776.0, 777.0], dtype=jnp.float64)
    p0 = jnp.array([774.40, 0.25, 5.5], dtype=jnp.float64)

    def _make_loss(
        solver: ExactLinearSolver | DiffraxSolver,
    ) -> Callable[[jax.Array], jax.Array]:
        def _fn(params: jax.Array) -> jax.Array:
            prod = (
                SteadyProduction(1.76)
                + HarmonicSolar.from_polar(
                    periods=11.0, amplitudes=0.25, phases=0.4, t_ref=774.0
                )
                + MiyakeSpike(params[0], params[1], params[2], profile="boxcar")
            )
            d14c = solver.predict_d14c(cbm, prod, t_starts, t_ends, box_indices=1)
            return jnp.sum(d14c**2)

        return _fn

    f_ex = _make_loss(exact)
    for method, tol, rtol_ex in (
        ("Tsit5", 1e-10, 1e-4),
        ("Kvaerno5", 1e-10, 2e-2),
    ):
        jax.clear_caches()
        dfx = DiffraxSolver(method, rtol=tol, atol=tol)
        _assert_grad_matches(
            f_ex,
            _make_loss(dfx),
            p0,
            rtol_exact=rtol_ex,
            rtol_fd=1e-5,
            rtol_eager_jit=1e-7,
        )


def test_gamma_k2_onset_jump_grad() -> None:
    """``MiyakeSpike(profile='gamma', gamma_shape=2.0)`` resolves the ``u=0`` slope kink."""
    cbm = load_preset("Guttler15")
    exact = ExactLinearSolver()
    dfx = DiffraxSolver("Tsit5", rtol=1e-12, atol=1e-12)
    p0 = jnp.array([774.40, 0.25, 5.5], dtype=jnp.float64)
    kw = {"profile": "gamma", "gamma_shape": 2.0}

    _assert_grad_matches(
        _make_spike_loss(cbm, exact, kw, binned=False),
        _make_spike_loss(cbm, dfx, kw, binned=False),
        p0,
        rtol_exact=1e-7,
        rtol_fd=1e-5,
        rtol_eager_jit=1e-8,
    )
