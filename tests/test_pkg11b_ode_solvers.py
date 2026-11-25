"""Tests for Package 11 Part 2 ODE solver deduplication and ``is_detailed_balance`` dispatch.

Verifies:
1. ``DiffraxSolver`` eager, ``jax.jit``, and ``jax.grad`` parity across autonomous
   ``MiyakeSpike`` (``boxcar`` & ``supergaussian``), ``GriddedProduction``, explicit
   non-equilibrium ``(t0, y0)``, 1D and 2D weighted seasonal windows, and non-autonomous
   Suess ``12C`` dilution (both solver-level callbacks and CBM-overridden methods).
2. ``ModaxSolver`` CPU ``backend_fn`` and mocked ``modax`` module paths: unsorted/duplicate
   timestamp deduplication, 1D/2D seasonal GL16 quadrature, Suess ``12C`` dilution
   (``relative_to="standard"`` & ``"ambient"``), eager vs ``jax.jit`` parity, and
   ``jax.grad`` / ``wrap_log_likelihood`` accuracy against ``DiffraxSolver`` and finite
   differences.
3. ``is_detailed_balance`` duck-typed dispatch in ``_modal_decomposition`` and
   ``ExactLinearSolver.propagation_backend`` for chart-backed / custom CBMs under both
   eager execution and JAX tracing.
"""

from __future__ import annotations

from collections.abc import Callable
import gc
from pathlib import Path
import sys
import types

import diffrax
import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy import integrate as sp_integrate  # noqa: E402

from ticktack.cbm import (  # noqa: E402
    CarbonBoxModel,
    EffectiveCBM,
    FixedCBM,
    ReversibleCBM,
    Topology,
)
from ticktack.constants import (  # noqa: E402
    ALPHA_ATOMS_CM2_S_TO_KG_YR,
    DECAY_CAMBRIDGE,
    DEFAULT_Q_STD,
)
from ticktack.presets import load_preset  # noqa: E402
from ticktack.production import (  # noqa: E402
    GriddedProduction,
    HarmonicSolar,
    LinearTrend,
    MiyakeSpike,
    SteadyProduction,
)
from ticktack.solvers import (  # noqa: E402
    DiffraxSolver,
    ExactLinearSolver,
    ModaxSolver,
    _modal_decomposition,
)


@pytest.fixture(autouse=True)
def _clear_caches() -> object:
    """Release JAX/Equinox compilation caches between tests to keep RSS compact."""
    yield
    if hasattr(eqx, "clear_caches"):
        eqx.clear_caches()
    jax.clear_caches()
    gc.collect()


def _central_fd_grad(
    fn: Callable[[jax.Array], jax.Array],
    x: jax.Array,
    eps: float = 1e-6,
) -> np.ndarray:
    """Second-order central finite-difference gradient of a scalar function."""
    grads = []
    for i in range(int(x.shape[0])):
        fp = float(fn(x.at[i].add(eps)))
        fm = float(fn(x.at[i].add(-eps)))
        grads.append((fp - fm) / (2.0 * eps))
    return np.asarray(grads, dtype=np.float64)


def _install_mock_modax(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Install a lightweight CPU ``modax`` mock backed by ``diffrax.Tsit5``."""
    fake_modax = types.ModuleType("modax")
    calls = {"count": 0, "checked_strict_unique": 0}

    class _FakeSolver:
        pass

    fake_modax.Rodas5P = type("Rodas5P", (_FakeSolver,), {})  # type: ignore[attr-defined]
    fake_modax.Tsit5 = type("Tsit5", (_FakeSolver,), {})  # type: ignore[attr-defined]

    def _fake_diffeqsolve(
        rhs,
        solver,
        *,
        t0,
        t1,
        y0,
        ts,
        args,
        rtol,
        atol,
        max_steps,
    ):
        assert isinstance(solver, _FakeSolver)
        calls["count"] += 1
        ts_arr = jnp.asarray(ts, dtype=jnp.float64)
        if ts_arr.size > 1 and not isinstance(ts_arr, jax.core.Tracer):
            assert bool(jnp.all(jnp.diff(ts_arr) > 0.0))
            calls["checked_strict_unique"] += 1
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(rhs),
            diffrax.Tsit5(),
            t0=t0,
            t1=t1,
            dt0=None,
            y0=y0,
            args=args,
            saveat=diffrax.SaveAt(ts=ts_arr),
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            adjoint=diffrax.DirectAdjoint(),
            max_steps=max_steps,
        )
        return types.SimpleNamespace(ys=sol.ys)

    fake_modax.diffeqsolve = _fake_diffeqsolve  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "modax", fake_modax)
    return calls


def _make_suess_callbacks(
    cbm: CarbonBoxModel,
    t_onset: float = 1850.0,
    amp_gtc: float = 95.0,
) -> tuple[Callable[[jax.Array], jax.Array], Callable[[jax.Array], jax.Array]]:
    """Smooth fossil-fuel ``12C`` dilution callbacks ``(masses_fn, matrix_A_fn)``."""
    m0 = cbm.reservoir_masses()
    K0 = cbm.rate_matrix_K()
    eye_n = jnp.eye(cbm.n_boxes, dtype=jnp.float64)

    def masses_at(t: jax.Array) -> jax.Array:
        u = jax.nn.softplus((t - t_onset) / 10.0) * 10.0
        return m0.at[1].add(amp_gtc * (u / 100.0) ** 2)

    def matrix_A_at(t: jax.Array) -> jax.Array:
        scale = m0 / masses_at(t)
        return K0 * scale[None, :] - cbm.decay_rate * eye_n

    return masses_at, matrix_A_at


class _NonAutonomousSubCBM(CarbonBoxModel):
    """CBM subclass overriding ``reservoir_masses_at`` and ``matrix_A_at`` directly."""

    base: FixedCBM
    amp_gtc: jax.Array

    def __init__(self, base: FixedCBM, amp_gtc: float = 85.0) -> None:
        self.topology = base.topology
        self.decay_rate = float(base.decay_rate)
        self.base = base
        self.amp_gtc = jnp.asarray(amp_gtc, dtype=jnp.float64)

    def reservoir_masses(self) -> jax.Array:
        return self.base.reservoir_masses()

    def matrix_A(self) -> jax.Array:
        return self.base.matrix_A()

    def production_vector(self) -> jax.Array:
        return self.base.production_vector()

    def reservoir_masses_at(self, t: jax.Array) -> jax.Array:
        m0 = self.base.reservoir_masses()
        u = jax.nn.softplus((jnp.asarray(t, dtype=jnp.float64) - 1850.0) / 10.0) * 10.0
        return m0.at[1].add(self.amp_gtc * (u / 100.0) ** 2)

    def matrix_A_at(self, t: jax.Array) -> jax.Array:
        m0 = self.base.reservoir_masses()
        scale = m0 / self.reservoir_masses_at(t)
        return (
            self.base.rate_matrix_K() * scale[None, :]
            - self.decay_rate * jnp.eye(self.n_boxes, dtype=jnp.float64)
        )


# ---------------------------------------------------------------------------
# 1. DiffraxSolver: Autonomous, Gridded, MiyakeSpike, 2D Windows & Suess
# ---------------------------------------------------------------------------


def test_diffrax_autonomous_miyake_and_gridded_eager_jit_grad() -> None:
    """Verify eager, ``jax.jit``, and ``jax.grad`` parity on ``DiffraxSolver`` autonomous paths."""
    cbm = load_preset("Guttler15", kind="fixed")
    exact = ExactLinearSolver()
    dfx = DiffraxSolver("Tsit5", rtol=1e-12, atol=1e-12)

    # Unsorted evaluation times + 2D weighted seasonal sub-windows + non-default q_std
    t_unsorted = jnp.array([776.4, 773.7, 775.1, 774.55, 773.7], dtype=jnp.float64)
    yrs = jnp.array([773.0, 774.0, 775.0, 776.0], dtype=jnp.float64)
    starts_2d = jnp.stack([yrs + 0.20, yrs + 0.55], axis=1)
    ends_2d = jnp.stack([yrs + 0.48, yrs + 0.85], axis=1)
    weights_2d = jnp.broadcast_to(jnp.array([0.65, 0.35], dtype=jnp.float64), starts_2d.shape)
    q_std = 1.84

    for profile in ("supergaussian", "boxcar"):
        def _loss(params: jax.Array, *, binned: bool, _prof: str = profile) -> jax.Array:
            prod = (
                SteadyProduction(1.92)
                + HarmonicSolar.from_polar(periods=11.0, amplitudes=0.22, phases=0.6, t_ref=774.0)
                + LinearTrend(slope=0.004, t_ref=774.0, switched_on=False)
                + MiyakeSpike(params[0], params[1], params[2], profile=_prof)
            )
            if binned:
                out = dfx.solve_delta_binned(
                    cbm, prod, starts_2d, ends_2d, sub_weights=weights_2d, q_std=q_std
                )
            else:
                out = dfx.solve_delta_trajectory(cbm, prod, t_unsorted, q_std=q_std)
            return jnp.sum(out[:, 1] ** 2)

        def _ref_loss(params: jax.Array, *, binned: bool, _prof: str = profile) -> jax.Array:
            prod = (
                SteadyProduction(1.92)
                + HarmonicSolar.from_polar(periods=11.0, amplitudes=0.22, phases=0.6, t_ref=774.0)
                + LinearTrend(slope=0.004, t_ref=774.0, switched_on=False)
                + MiyakeSpike(params[0], params[1], params[2], profile=_prof)
            )
            if binned:
                out = exact.solve_delta_binned(
                    cbm, prod, starts_2d, ends_2d, sub_weights=weights_2d, q_std=q_std
                )
            else:
                out = exact.solve_delta_trajectory(cbm, prod, t_unsorted, q_std=q_std)
            return jnp.sum(out[:, 1] ** 2)

        p0 = jnp.array([774.38, 0.24, 5.8], dtype=jnp.float64)
        for binned in (False, True):
            fn_dfx = lambda p, _b=binned: _loss(p, binned=_b)  # noqa: E731
            fn_ref = lambda p, _b=binned: _ref_loss(p, binned=_b)  # noqa: E731

            v_eager, g_eager = jax.value_and_grad(fn_dfx)(p0)
            v_jit, g_jit = jax.jit(jax.value_and_grad(fn_dfx))(p0)
            v_ref, g_ref = jax.value_and_grad(fn_ref)(p0)

            assert float(v_eager) == pytest.approx(float(v_ref), rel=1e-9, abs=1e-9)
            assert float(v_jit) == pytest.approx(float(v_eager), rel=1e-12, abs=1e-12)
            np.testing.assert_allclose(g_eager, g_ref, rtol=1e-5, atol=1e-7)
            np.testing.assert_allclose(g_jit, g_eager, rtol=1e-8, atol=1e-8)

    # GriddedProduction pointwise & binned forward + gradient parity vs ExactLinearSolver
    t_grid = jnp.linspace(773.2, 777.2, 21, dtype=jnp.float64)
    q_init = q_std + 18.0 * jnp.exp(-0.5 * ((t_grid - 774.6) / 0.28) ** 2)

    def _grid_loss_binned(q_vals: jax.Array, solver: DiffraxSolver | ExactLinearSolver) -> jax.Array:
        prod = GriddedProduction(t_grid, q_vals)
        dy = solver.solve_delta_binned(
            cbm, prod, starts_2d, ends_2d, sub_weights=weights_2d, q_std=q_std
        )
        return jnp.sum(dy[:, 1] ** 2)

    v_g_eager, g_g_eager = jax.value_and_grad(lambda q: _grid_loss_binned(q, dfx))(q_init)
    v_g_jit, g_g_jit = jax.jit(jax.value_and_grad(lambda q: _grid_loss_binned(q, dfx)))(q_init)
    v_g_ex, g_g_ex = jax.value_and_grad(lambda q: _grid_loss_binned(q, exact))(q_init)

    assert float(v_g_eager) == pytest.approx(float(v_g_ex), rel=1e-8, abs=1e-8)
    assert float(v_g_jit) == pytest.approx(float(v_g_eager), rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(g_g_eager, g_g_ex, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(g_g_jit, g_g_eager, rtol=1e-8, atol=1e-8)

    # Explicit non-equilibrium (t0, y0) binned parity and t0 gradient
    y_ss = cbm.steady_state_c14(q_std=q_std)
    y0_neq = y_ss * (1.0 + 0.03 * jnp.linspace(-1.0, 1.0, cbm.n_boxes))

    def _ic_loss(t0_val: jax.Array, solver: DiffraxSolver | ExactLinearSolver) -> jax.Array:
        dy = solver.solve_delta_binned(
            cbm,
            SteadyProduction(q_std),
            starts_2d,
            ends_2d,
            sub_weights=weights_2d,
            q_std=q_std,
            t0=t0_val,
            y0=y0_neq,
        )
        return jnp.sum(dy[:, 1] ** 2)

    t0_val = jnp.asarray(772.3, dtype=jnp.float64)
    v_ic_e, g_ic_e = jax.value_and_grad(lambda t: _ic_loss(t, dfx))(t0_val)
    v_ic_j, g_ic_j = jax.jit(jax.value_and_grad(lambda t: _ic_loss(t, dfx)))(t0_val)
    v_ic_x, g_ic_x = jax.value_and_grad(lambda t: _ic_loss(t, exact))(t0_val)
    assert float(v_ic_e) == pytest.approx(float(v_ic_x), rel=1e-10, abs=1e-10)
    assert float(v_ic_j) == pytest.approx(float(v_ic_e), rel=1e-12, abs=1e-12)
    assert float(g_ic_e) == pytest.approx(float(g_ic_x), rel=1e-7, abs=1e-8)
    assert float(g_ic_j) == pytest.approx(float(g_ic_e), rel=1e-8, abs=1e-8)


def _scipy_suess_2d_oracle(
    cbm: FixedCBM,
    prod: SteadyProduction | object,
    starts_2d: np.ndarray,
    ends_2d: np.ndarray,
    weights_2d: np.ndarray,
    *,
    t0: float,
    t_onset: float,
    amp_gtc: float,
    t_spike: float,
    delta_global: float,
    delta_lab: float,
    relative_to: str,
    box_index: int = 1,
) -> np.ndarray:
    """Independent SciPy ``DOP853`` + ``quad`` forward oracle for 2D-window Suess ``Delta14C``."""
    m0 = np.asarray(cbm.reservoir_masses(), dtype=np.float64)
    K0 = np.asarray(cbm.rate_matrix_K(), dtype=np.float64)
    decay = float(cbm.decay_rate)
    eye_n = np.eye(cbm.n_boxes, dtype=np.float64)
    A0 = K0 - decay * eye_n
    b_sc = float(ALPHA_ATOMS_CM2_S_TO_KG_YR) * np.asarray(cbm.production_vector(), dtype=np.float64)
    y_ss = np.linalg.solve(-A0, b_sc * float(DEFAULT_Q_STD))
    r_ref = float(y_ss[box_index] / m0[box_index])

    def _m_at(t: float) -> np.ndarray:
        u = float(np.logaddexp(0.0, (t - t_onset) / 10.0) * 10.0)
        mt = m0.copy()
        mt[1] += amp_gtc * (u / 100.0) ** 2
        return mt

    def _rhs_tot(t: float, y_tot: np.ndarray) -> np.ndarray:
        At = K0 * (m0 / _m_at(t))[None, :] - decay * eye_n
        q_t = float(DEFAULT_Q_STD) + float(prod.delta_q(t, q_std=DEFAULT_Q_STD))  # type: ignore[attr-defined]
        return At @ y_tot + b_sc * q_t

    # Eternal SteadyProduction(1.83) equilibrium at t0 = 1840 (pre-spike, pre-Suess)
    y_init = y_ss + (1.83 - float(DEFAULT_Q_STD)) * np.linalg.solve(-A0, b_sc)
    t_end = float(np.max(ends_2d)) + 0.1
    segs = [
        (t0, t_spike - 1.5, np.inf),
        (t_spike - 1.5, t_spike + 2.5, 0.02),
        (t_spike + 2.5, t_end, np.inf),
    ]
    sols = []
    y_cur = y_init
    for ta, tb, hmax in segs:
        sol = sp_integrate.solve_ivp(
            _rhs_tot,
            (ta, tb),
            y_cur,
            method="DOP853",
            rtol=1e-12,
            atol=1e-13,
            max_step=hmax,
            dense_output=True,
        )
        assert sol.success
        sols.append((ta, tb, sol.sol))
        y_cur = sol.y[:, -1]

    def _y_tot_at(t: float) -> np.ndarray:
        for ta, tb, dense in sols:
            if ta <= t <= tb:
                return dense(t)
        return sols[-1][2](t)

    scale_amb = 1.0 + delta_global / 1000.0
    exc_scale = scale_amb if relative_to == "ambient" else 1.0
    y_base = y_ss[box_index] * scale_amb

    def _inst_d14c(t: float) -> float:
        dy_b = float(_y_tot_at(t)[box_index] - y_ss[box_index])
        m_b = float(_m_at(t)[box_index])
        return 1000.0 * (((y_base + exc_scale * dy_b) / m_b) / r_ref - 1.0)

    n_r, m_w = starts_2d.shape
    sub_avgs = np.empty((n_r, m_w), dtype=np.float64)
    for i in range(n_r):
        for j in range(m_w):
            a_ij, b_ij = float(starts_2d[i, j]), float(ends_2d[i, j])
            val, _ = sp_integrate.quad(_inst_d14c, a_ij, b_ij, epsabs=1e-12, epsrel=1e-12)
            sub_avgs[i, j] = val / (b_ij - a_ij)
    return np.sum(sub_avgs * weights_2d, axis=1) + delta_lab


def test_diffrax_suess_dilution_callbacks_and_cbm_override_eager_jit_grad() -> None:
    """Verify Suess ``12C`` dilution for both solver callbacks and CBM overrides (2D windows)."""
    g15: FixedCBM = load_preset("Guttler15", kind="fixed")  # type: ignore[assignment]
    masses_fn, matrix_A_fn = _make_suess_callbacks(g15, t_onset=1850.0, amp_gtc=85.0)
    dfx_cb = DiffraxSolver(
        "Tsit5", rtol=1e-11, atol=1e-11, masses_fn=masses_fn, matrix_A_fn=matrix_A_fn
    )
    dfx_plain = DiffraxSolver("Tsit5", rtol=1e-11, atol=1e-11)
    nonauton_cbm = _NonAutonomousSubCBM(g15, amp_gtc=85.0)

    yrs = jnp.array([1845.0, 1895.0, 1945.0], dtype=jnp.float64)
    starts_2d = jnp.stack([yrs + 0.15, yrs + 0.55], axis=1)
    ends_2d = jnp.stack([yrs + 0.45, yrs + 0.85], axis=1)
    weights_2d = jnp.broadcast_to(jnp.array([0.3, 0.7], dtype=jnp.float64), starts_2d.shape)

    for rel_mode in ("standard", "ambient"):
        def _suess_loss(params: jax.Array, *, use_cbm_override: bool, _m: str = rel_mode) -> jax.Array:
            prod = SteadyProduction(1.83) + MiyakeSpike(
                t_start=params[0], duration=params[1], fluence=params[2], profile="supergaussian"
            )
            solver = dfx_plain if use_cbm_override else dfx_cb
            model_cbm: CarbonBoxModel = nonauton_cbm if use_cbm_override else g15
            pred = solver.predict_d14c(
                model_cbm,
                prod,
                starts_2d,
                ends_2d,
                sub_weights=weights_2d,
                box_indices=1,
                delta_global=params[3],
                delta_lab=params[4],
                relative_to=_m,  # type: ignore[arg-type]
                t0=1840.0,
            )
            return jnp.sum(pred**2)

        p0 = jnp.array([1894.4, 0.32, 5.2, 22.5, -1.7], dtype=jnp.float64)
        prod_p0 = SteadyProduction(1.83) + MiyakeSpike(
            t_start=p0[0], duration=p0[1], fluence=p0[2], profile="supergaussian"
        )
        pred_scipy = _scipy_suess_2d_oracle(
            g15,
            prod_p0,
            np.asarray(starts_2d),
            np.asarray(ends_2d),
            np.asarray(weights_2d),
            t0=1840.0,
            t_onset=1850.0,
            amp_gtc=85.0,
            t_spike=float(p0[0]),
            delta_global=float(p0[3]),
            delta_lab=float(p0[4]),
            relative_to=rel_mode,
        )
        pred_cb = dfx_cb.predict_d14c(
            g15,
            prod_p0,
            starts_2d,
            ends_2d,
            sub_weights=weights_2d,
            box_indices=1,
            delta_global=p0[3],
            delta_lab=p0[4],
            relative_to=rel_mode,  # type: ignore[arg-type]
            t0=1840.0,
        )
        pred_ov = dfx_plain.predict_d14c(
            nonauton_cbm,
            prod_p0,
            starts_2d,
            ends_2d,
            sub_weights=weights_2d,
            box_indices=1,
            delta_global=p0[3],
            delta_lab=p0[4],
            relative_to=rel_mode,  # type: ignore[arg-type]
            t0=1840.0,
        )
        np.testing.assert_allclose(pred_cb, pred_scipy, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(pred_ov, pred_scipy, rtol=1e-9, atol=1e-9)

        fn_cb = lambda p: _suess_loss(p, use_cbm_override=False)  # noqa: E731
        fn_ov = lambda p: _suess_loss(p, use_cbm_override=True)  # noqa: E731

        v_cb_e, g_cb_e = jax.value_and_grad(fn_cb)(p0)
        v_cb_j, g_cb_j = jax.jit(jax.value_and_grad(fn_cb))(p0)
        v_ov_e, g_ov_e = jax.value_and_grad(fn_ov)(p0)
        v_ov_j, g_ov_j = jax.jit(jax.value_and_grad(fn_ov))(p0)
        g_fd = _central_fd_grad(jax.jit(fn_cb), p0)

        assert float(v_cb_j) == pytest.approx(float(v_cb_e), rel=1e-12, abs=1e-12)
        assert float(v_ov_e) == pytest.approx(float(v_cb_e), rel=1e-12, abs=1e-12)
        assert float(v_ov_j) == pytest.approx(float(v_ov_e), rel=1e-12, abs=1e-12)
        np.testing.assert_allclose(g_cb_e, g_fd, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(g_cb_j, g_cb_e, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(g_ov_e, g_cb_e, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(g_ov_j, g_ov_e, rtol=1e-7, atol=1e-7)

    # Confirm invalid relative_to fails loudly on both DiffraxSolver and ModaxSolver
    for solver in (dfx_cb, ModaxSolver()):
        with pytest.raises(ValueError, match="relative_to"):
            solver.predict_d14c(
                g15,
                SteadyProduction(1.76),
                yrs,
                yrs + 0.5,
                relative_to="invalid",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# 2. ModaxSolver: Custom CPU backend_fn & Mocked modax Module Paths
# ---------------------------------------------------------------------------


def test_modax_custom_backend_fn_binned_suess_eager_jit_grad() -> None:
    """Verify ``ModaxSolver(backend_fn=...)`` CPU path across 2D windows, Suess, JIT, and grad."""
    g15: FixedCBM = load_preset("Guttler15", kind="fixed")  # type: ignore[assignment]
    masses_fn, matrix_A_fn = _make_suess_callbacks(g15, t_onset=1850.0, amp_gtc=105.0)
    dfx_ref = DiffraxSolver(
        "Tsit5", rtol=1e-11, atol=1e-11, masses_fn=masses_fn, matrix_A_fn=matrix_A_fn
    )
    mdx_cpu = ModaxSolver(
        "Tsit5",
        rtol=1e-11,
        atol=1e-11,
        masses_fn=masses_fn,
        matrix_A_fn=matrix_A_fn,
        backend_fn=dfx_ref.solve_delta_trajectory,
    )

    yrs = jnp.array([1846.0, 1895.0, 1945.0], dtype=jnp.float64)
    starts_2d = jnp.stack([yrs + 0.2, yrs + 0.6], axis=1)
    ends_2d = jnp.stack([yrs + 0.5, yrs + 0.9], axis=1)
    weights_2d = jnp.broadcast_to(jnp.array([0.4, 0.6], dtype=jnp.float64), starts_2d.shape)

    def _binned_loss(params: jax.Array, solver: DiffraxSolver | ModaxSolver) -> jax.Array:
        prod = SteadyProduction(1.85) + MiyakeSpike(
            t_start=params[0], duration=params[1], fluence=params[2], profile="supergaussian"
        )
        dy = solver.solve_delta_binned(
            g15, prod, starts_2d, ends_2d, sub_weights=weights_2d, t0=1840.0
        )
        return jnp.sum(dy[:, 1] ** 2)

    p_bin = jnp.array([1894.3, 0.35, 6.1], dtype=jnp.float64)
    vb_m, gb_m = jax.value_and_grad(lambda p: _binned_loss(p, mdx_cpu))(p_bin)
    vb_j, gb_j = jax.jit(jax.value_and_grad(lambda p: _binned_loss(p, mdx_cpu)))(p_bin)
    vb_d, gb_d = jax.value_and_grad(lambda p: _binned_loss(p, dfx_ref))(p_bin)
    assert float(vb_m) == pytest.approx(float(vb_d), rel=1e-8, abs=1e-8)
    assert float(vb_j) == pytest.approx(float(vb_m), rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(gb_m, gb_d, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(gb_j, gb_m, rtol=1e-8, atol=1e-8)

    # Suess predict_d14c (relative_to="ambient", nonzero delta_global & delta_lab, 2D windows)
    def _suess_loss(params: jax.Array, solver: DiffraxSolver | ModaxSolver) -> jax.Array:
        prod = SteadyProduction(1.85) + MiyakeSpike(
            t_start=params[0], duration=params[1], fluence=params[2], profile="supergaussian"
        )
        pred = solver.predict_d14c(
            g15,
            prod,
            starts_2d,
            ends_2d,
            sub_weights=weights_2d,
            box_indices=1,
            delta_global=params[3],
            delta_lab=params[4],
            relative_to="ambient",
            t0=1840.0,
        )
        return jnp.sum(pred**2)

    p_suess = jnp.array([1894.3, 0.35, 6.1, 24.0, 1.4], dtype=jnp.float64)
    vs_m, gs_m = jax.value_and_grad(lambda p: _suess_loss(p, mdx_cpu))(p_suess)
    vs_j, gs_j = jax.jit(jax.value_and_grad(lambda p: _suess_loss(p, mdx_cpu)))(p_suess)
    vs_d, gs_d = jax.value_and_grad(lambda p: _suess_loss(p, dfx_ref))(p_suess)
    assert float(vs_m) == pytest.approx(float(vs_d), rel=1e-8, abs=1e-8)
    assert float(vs_j) == pytest.approx(float(vs_m), rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(gs_m, gs_d, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(gs_j, gs_m, rtol=1e-7, atol=1e-7)


def test_modax_mocked_module_dedup_jit_grad_and_suess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify mocked ``modax`` path: concrete dedup, JIT static-shape fallback, Suess, and grads."""
    calls = _install_mock_modax(monkeypatch)
    g15: FixedCBM = load_preset("Guttler15", kind="fixed")  # type: ignore[assignment]
    masses_fn, matrix_A_fn = _make_suess_callbacks(g15, t_onset=1850.0, amp_gtc=90.0)

    dfx = DiffraxSolver(
        "Tsit5", rtol=1e-11, atol=1e-11, masses_fn=masses_fn, matrix_A_fn=matrix_A_fn
    )
    mdx = ModaxSolver(
        "Rodas5P", rtol=1e-11, atol=1e-11, masses_fn=masses_fn, matrix_A_fn=matrix_A_fn
    )

    # 1. Concrete unsorted + duplicate timestamps trigger strict deduplication path
    t_dup = jnp.array([1910.0, 1848.0, 1895.0, 1848.0, 1910.0], dtype=jnp.float64)
    prod = SteadyProduction(1.81) + MiyakeSpike(
        t_start=1894.5, duration=0.30, fluence=5.5, profile="supergaussian"
    )
    dy_eager = mdx.solve_delta_trajectory(g15, prod, t_dup, t0=1840.0)
    dy_jit = jax.jit(lambda t: mdx.solve_delta_trajectory(g15, prod, t, t0=1840.0))(t_dup)
    dy_ref = dfx.solve_delta_trajectory(g15, prod, t_dup, t0=1840.0)

    assert calls["checked_strict_unique"] >= 1
    np.testing.assert_allclose(dy_eager, dy_ref, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(dy_jit, dy_eager, rtol=1e-11, atol=1e-11)

    # 2. Explicit non-equilibrium (t0, y0) parity under mocked modax
    y_ss = g15.steady_state_c14()
    y0_neq = y_ss * (1.0 + 0.025 * jnp.linspace(-1.0, 1.0, g15.n_boxes))
    dy_ic_m = mdx.solve_delta_trajectory(g15, SteadyProduction(1.76), t_dup, t0=1842.0, y0=y0_neq)
    dy_ic_d = dfx.solve_delta_trajectory(g15, SteadyProduction(1.76), t_dup, t0=1842.0, y0=y0_neq)
    np.testing.assert_allclose(dy_ic_m, dy_ic_d, rtol=1e-8, atol=1e-8)

    # 3. Suess predict_d14c gradient via ModaxSolver.wrap_log_likelihood (eager + jit)
    yrs = jnp.array([1846.0, 1896.0, 1946.0], dtype=jnp.float64)
    t_s, t_e = yrs + 0.25, yrs + 0.75

    def _raw_ll(params: jax.Array) -> jax.Array:
        pr = SteadyProduction(1.81) + MiyakeSpike(
            t_start=params[0], duration=params[1], fluence=params[2], profile="supergaussian"
        )
        pred = mdx.predict_d14c(
            g15,
            pr,
            t_s,
            t_e,
            box_indices=1,
            delta_global=params[3],
            relative_to="standard",
            t0=1840.0,
        )
        return -0.5 * jnp.sum(pred**2)

    wrapped = ModaxSolver.wrap_log_likelihood(_raw_ll)
    p0 = jnp.array([1894.5, 0.30, 5.5, 18.0], dtype=jnp.float64)
    v_e, g_e = jax.value_and_grad(wrapped)(p0)
    v_j, g_j = jax.jit(jax.value_and_grad(wrapped))(p0)
    g_fd = _central_fd_grad(jax.jit(_raw_ll), p0)

    assert float(v_j) == pytest.approx(float(v_e), rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(g_e, g_fd, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(g_j, g_e, rtol=1e-7, atol=1e-7)


# ---------------------------------------------------------------------------
# 3. is_detailed_balance Dispatch Without Concrete Subclass Checks
# ---------------------------------------------------------------------------


class _ChartDetailedBalanceCBM(CarbonBoxModel):
    """Custom detailed-balance CBM (NOT an ``EffectiveCBM`` / ``ReversibleCBM`` subclass)."""

    log_sym_fluxes: jax.Array
    log_masses: jax.Array
    is_detailed_balance: bool = eqx.field(static=True)

    def __init__(
        self,
        sym_fluxes: jax.Array,
        masses: jax.Array,
        *,
        is_detailed_balance: bool = True,
    ) -> None:
        self.topology = Topology(
            box_names=("Stratosphere", "Troposphere", "Ocean"),
            edges_src=(0, 1, 1, 2),
            edges_dst=(1, 0, 2, 1),
            prod_box_indices=(0, 1),
            trop_box_indices=(1,),
        )
        self.decay_rate = float(DECAY_CAMBRIDGE)
        self.log_sym_fluxes = jnp.log(jnp.asarray(sym_fluxes, dtype=jnp.float64))
        self.log_masses = jnp.log(jnp.asarray(masses, dtype=jnp.float64))
        self.is_detailed_balance = bool(is_detailed_balance)

    def reservoir_masses(self) -> jax.Array:
        return jnp.exp(self.log_masses)

    def matrix_A(self) -> jax.Array:
        f01, f12 = jnp.exp(self.log_sym_fluxes)
        m = self.reservoir_masses()
        K = jnp.array(
            [
                [-f01 / m[0], f01 / m[1], 0.0],
                [f01 / m[0], -(f01 + f12) / m[1], f12 / m[2]],
                [0.0, f12 / m[1], -f12 / m[2]],
            ],
            dtype=jnp.float64,
        )
        return K - self.decay_rate * jnp.eye(3, dtype=jnp.float64)

    def production_vector(self) -> jax.Array:
        return jnp.array([0.7, 0.3, 0.0], dtype=jnp.float64)


class _OptOutReversibleCBM(ReversibleCBM):
    """``ReversibleCBM`` subclass explicitly opting out of detailed-balance dispatch."""

    @property
    def is_detailed_balance(self) -> bool:
        return False


def test_is_detailed_balance_duck_typed_dispatch_eager_jit_and_grad() -> None:
    """Verify ``is_detailed_balance`` attribute controls ``modal_symmetric`` without isinstance."""
    solver = ExactLinearSolver()
    chart_db = _ChartDetailedBalanceCBM(
        sym_fluxes=jnp.array([47.0, 62.0]),
        masses=jnp.array([92.0, 515.0, 940.0]),
        is_detailed_balance=True,
    )
    chart_no_db = _ChartDetailedBalanceCBM(
        sym_fluxes=jnp.array([47.0, 62.0]),
        masses=jnp.array([92.0, 515.0, 940.0]),
        is_detailed_balance=False,
    )

    assert not isinstance(chart_db, (EffectiveCBM, ReversibleCBM))
    assert solver.propagation_backend(chart_db) == "modal_symmetric"
    assert solver.propagation_backend(chart_no_db) == "modal_general"

    # Under eqx.filter_jit tracing, detailed-balance stays modal_symmetric while non-db -> van_loan
    @eqx.filter_jit
    def _traced_backend(model: CarbonBoxModel) -> str:
        return solver.propagation_backend(model)

    assert _traced_backend(chart_db) == "modal_symmetric"
    assert _traced_backend(chart_no_db) == "van_loan"

    # Explicit is_detailed_balance=False overrides ReversibleCBM subclass default
    rev_base: ReversibleCBM = load_preset(  # type: ignore[assignment]
        "Miyake17", kind="reversible", miyake17_closure="closed_shallow"
    )
    opt_out = _OptOutReversibleCBM(
        topology=rev_base.topology,
        log_sym_fluxes=rev_base.log_sym_fluxes,
        log_masses=rev_base.log_masses,
        prod_weights=rev_base.prod_weights,
        decay_rate=rev_base.decay_rate,
    )
    assert isinstance(opt_out, ReversibleCBM)
    assert solver.propagation_backend(rev_base) == "modal_symmetric"
    assert solver.propagation_backend(opt_out) == "modal_general"
    assert _traced_backend(opt_out) == "van_loan"

    # Modal decomposition via is_detailed_balance=True reconstructs A accurately & matches Van Loan
    A = chart_db.matrix_A()
    modal = _modal_decomposition(A, cbm=chart_db)
    assert modal is not None
    lam, v_mat, v_inv = modal
    assert lam.dtype == jnp.float64
    np.testing.assert_allclose((v_mat * lam[None, :]) @ v_inv, A, rtol=1e-13, atol=1e-13)

    t_s = jnp.array([774.2, 775.2, 776.2], dtype=jnp.float64)
    t_e = t_s + 0.45
    spike = MiyakeSpike(t_start=774.5, duration=0.22, fluence=6.3, profile="supergaussian")

    def _db_loss(log_f: jax.Array, log_m: jax.Array, *, db_flag: bool) -> jax.Array:
        m = _ChartDetailedBalanceCBM(
            jnp.exp(log_f), jnp.exp(log_m), is_detailed_balance=db_flag
        )
        return jnp.sum(solver.predict_d14c(m, spike, t_s, t_e, box_indices=1) ** 2)

    v_db, g_db = jax.jit(jax.value_and_grad(lambda f, m: _db_loss(f, m, db_flag=True), (0, 1)))(
        chart_db.log_sym_fluxes, chart_db.log_masses
    )
    v_vl, g_vl = jax.jit(jax.value_and_grad(lambda f, m: _db_loss(f, m, db_flag=False), (0, 1)))(
        chart_db.log_sym_fluxes, chart_db.log_masses
    )
    assert float(v_db) == pytest.approx(float(v_vl), rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(g_db[0], g_vl[0], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(g_db[1], g_vl[1], rtol=1e-10, atol=1e-10)
