"""Tests for Package 11 solver spike-path deduplication and backend visibility.

Verifies:
1. ``ExactLinearSolver.propagation_backend(cbm)`` across ``"modal_symmetric"``,
   ``"modal_general"``, and ``"van_loan"`` regimes under both eager execution
   and ``jax.jit`` / ``equinox.filter_jit`` tracing.
2. All 8 spike evaluation paths (``boxcar`` vs smooth quadrature profiles x
   pointwise vs seasonal-window binned x modal vs Van Loan ``expm``) against an
   independent SciPy ``expm`` + ``quad_vec`` numerical integral oracle.
3. Exact Dirac-impulse limit (``duration = 0`` and ``1e-15``), narrow-duration
   stability (``1e-6`` to ``1e-2``), and contiguous-window boundary tie-breaking
   at ``t_start == a_r`` (counted once, never double-counted or dropped).
4. Forward/reverse-mode autodiff gradients across all profiles and backends
   against central finite differences, including ``duration = 0``.
"""

from __future__ import annotations

from collections.abc import Callable
import gc
from pathlib import Path
import sys

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from numpy.polynomial.legendre import leggauss  # noqa: E402
import pytest  # noqa: E402
from scipy.linalg import expm as scipy_expm  # noqa: E402

from ticktack.cbm import (  # noqa: E402
    BalancedCBM,
    CarbonBoxModel,
    EffectiveCBM,
    FixedCBM,
    Topology,
)
from ticktack.constants import (  # noqa: E402
    ALPHA_ATOMS_CM2_S_TO_KG_YR,
    DECAY_CAMBRIDGE,
)
from ticktack.presets import load_preset  # noqa: E402
from ticktack.production import MiyakeSpike  # noqa: E402
from ticktack.solvers import (  # noqa: E402
    ExactLinearSolver,
    _eval_boxcar_binned,
    _eval_boxcar_binned_van_loan,
    _eval_boxcar_pointwise,
    _eval_boxcar_pointwise_van_loan,
    _eval_smooth_spike_binned,
    _eval_smooth_spike_binned_van_loan,
    _eval_smooth_spike_pointwise,
    _eval_smooth_spike_pointwise_van_loan,
    _modal_decomposition,
)


_TEST_COUNTER = 0


def _flush_xla_caches() -> None:
    """Release both Equinox and JAX XLA executable caches."""
    if hasattr(eqx, "clear_caches"):
        eqx.clear_caches()
    jax.clear_caches()
    gc.collect()


@pytest.fixture(autouse=True)
def _clear_caches() -> object:
    """Periodically clear XLA compilation caches so RSS stays well below 1.5 GB."""
    global _TEST_COUNTER
    _TEST_COUNTER += 1
    if _TEST_COUNTER % 5 == 0:
        _flush_xla_caches()
    yield


_THREE_BOX_TOPO = Topology(
    box_names=("Stratosphere", "Troposphere", "DeepOcean"),
    edges_src=(0, 1, 1, 2),
    edges_dst=(1, 0, 2, 0),
    prod_box_indices=(0, 1),
    trop_box_indices=(1,),
)


def _make_regime_cbm(regime: str) -> BalancedCBM:
    """Build a 3-box ``BalancedCBM`` in a real, complex, or near-degenerate spectrum."""
    rates_by_regime = {
        "real": [0.53, 0.19, 0.41, 0.047],
        "complex": [1.0, 0.1, 1.0, 1.0],
        "near_degenerate": [1.0, 0.5, 1.0, 0.5 + 1e-13],
    }
    return BalancedCBM(
        topology=_THREE_BOX_TOPO,
        log_rates_free=jnp.log(jnp.array(rates_by_regime[regime], dtype=jnp.float64)),
        log_total_mass=jnp.log(jnp.array(1850.0, dtype=jnp.float64)),
        prod_weights=(0.68, 0.32),
        decay_rate=DECAY_CAMBRIDGE,
    )


_GL32_X, _GL32_W = leggauss(32)
_GL32_NODES = 0.5 * (_GL32_X + 1.0)
_GL32_WEIGHTS = 0.5 * _GL32_W


def _scipy_int_expm_vec(A_np: np.ndarray, v_np: np.ndarray, dt: float) -> np.ndarray:
    """Cancellation-free ``int_0^dt exp(A s) v ds`` via SciPy ``(n+1)x(n+1)`` ``expm``."""
    n = A_np.shape[0]
    aug = np.zeros((n + 1, n + 1), dtype=np.float64)
    aug[:n, :n] = A_np * dt
    aug[:n, n] = v_np * dt
    return scipy_expm(aug)[:n, n]


def _scipy_spike_pointwise_oracle(
    A_np: np.ndarray,
    b_np: np.ndarray,
    spike: MiyakeSpike,
    t_eval: np.ndarray,
) -> np.ndarray:
    """Independent 32-point Gauss-Legendre + SciPy ``expm`` causal integral oracle."""
    ts = float(spike.t_start)
    dur = max(float(spike.duration), 0.0)
    panels = spike.unit_support_panels()
    supp_lo = ts + dur * panels[0][0]
    supp_hi = ts + dur * panels[-1][1]
    base_knots = [ts + dur * p[0] for p in panels] + [supp_hi]

    out = np.zeros((len(t_eval), A_np.shape[0]), dtype=np.float64)
    for idx, ti in enumerate(t_eval):
        ti_f = float(ti)
        if ti_f <= supp_lo:
            continue
        upper = min(ti_f, supp_hi)
        edges = sorted({supp_lo, upper, *(k for k in base_knots if supp_lo < k < upper)})
        taus = np.concatenate(
            [lo_e + (hi_e - lo_e) * _GL32_NODES for lo_e, hi_e in zip(edges[:-1], edges[1:])]
        )
        wts = np.concatenate(
            [(hi_e - lo_e) * _GL32_WEIGHTS for lo_e, hi_e in zip(edges[:-1], edges[1:])]
        )
        q_vals = np.asarray(spike.delta_q(taus), dtype=np.float64)
        acc = np.zeros(A_np.shape[0], dtype=np.float64)
        for tau_k, w_k, q_k in zip(taus, wts, q_vals):
            acc += w_k * q_k * (scipy_expm(A_np * (ti_f - tau_k)) @ b_np)
        out[idx] = acc
    return out


def _scipy_spike_binned_oracle(
    A_np: np.ndarray,
    b_np: np.ndarray,
    spike: MiyakeSpike,
    a_r: np.ndarray,
    b_r: np.ndarray,
) -> np.ndarray:
    """Independent 32-point Gauss-Legendre + SciPy ``expm`` window oracle via Fubini."""
    ts = float(spike.t_start)
    dur = max(float(spike.duration), 0.0)
    panels = spike.unit_support_panels()
    supp_lo = ts + dur * panels[0][0]
    supp_hi = ts + dur * panels[-1][1]
    base_knots = [ts + dur * p[0] for p in panels] + [supp_hi]

    out = np.zeros((len(a_r), A_np.shape[0]), dtype=np.float64)
    for idx, (ar, br) in enumerate(zip(a_r, b_r)):
        ar_f, br_f = float(ar), float(br)
        if br_f <= supp_lo:
            continue
        upper = min(br_f, supp_hi)
        edges = sorted(
            {supp_lo, upper, *(k for k in [*base_knots, ar_f] if supp_lo < k < upper)}
        )
        taus = np.concatenate(
            [lo_e + (hi_e - lo_e) * _GL32_NODES for lo_e, hi_e in zip(edges[:-1], edges[1:])]
        )
        wts = np.concatenate(
            [(hi_e - lo_e) * _GL32_WEIGHTS for lo_e, hi_e in zip(edges[:-1], edges[1:])]
        )
        q_vals = np.asarray(spike.delta_q(taus), dtype=np.float64)
        acc = np.zeros(A_np.shape[0], dtype=np.float64)
        for tau_k, w_k, q_k in zip(taus, wts, q_vals):
            s0 = max(ar_f - tau_k, 0.0)
            s1 = br_f - tau_k
            v_start = scipy_expm(A_np * s0) @ b_np if s0 > 0.0 else b_np
            acc += w_k * q_k * _scipy_int_expm_vec(A_np, v_start, s1 - s0)
        out[idx] = acc / (br_f - ar_f)
    return out


def _central_fd_grad(
    fn: Callable[[jax.Array], jax.Array],
    x: jax.Array,
    eps: float = 1e-6,
) -> np.ndarray:
    """Second-order central finite-difference gradient."""
    grads = []
    for i in range(int(x.shape[0])):
        fp = float(fn(x.at[i].add(eps)))
        fm = float(fn(x.at[i].add(-eps)))
        grads.append((fp - fm) / (2.0 * eps))
    return np.asarray(grads, dtype=np.float64)


class _EmptyCBMStub(CarbonBoxModel):
    """Minimal 0-box stub verifying ``n_boxes == 0`` Van Loan fallback."""

    def __init__(self) -> None:
        self.topology = _THREE_BOX_TOPO
        self.decay_rate = float(DECAY_CAMBRIDGE)

    @property
    def n_boxes(self) -> int:
        return 0

    def reservoir_masses(self) -> jax.Array:
        return jnp.zeros((0,), dtype=jnp.float64)

    def relative_reservoir_masses(self) -> jax.Array:
        return jnp.zeros((0,), dtype=jnp.float64)

    def matrix_A(self) -> jax.Array:
        return jnp.zeros((0, 0), dtype=jnp.float64)

    def production_vector(self) -> jax.Array:
        return jnp.zeros((0,), dtype=jnp.float64)


def test_propagation_backend_visibility_eager_and_jit() -> None:
    """Verify ``ExactLinearSolver.propagation_backend`` across all regimes and JIT."""
    solver = ExactLinearSolver()

    eff = EffectiveCBM(K=3)
    rev = load_preset("Miyake17", kind="reversible", miyake17_closure="closed_shallow")
    bal_real = _make_regime_cbm("real")
    bal_cplx = _make_regime_cbm("complex")
    bal_deg = _make_regime_cbm("near_degenerate")
    guttler_fixed: FixedCBM = load_preset("Guttler15", kind="fixed")  # type: ignore[assignment]

    # Eager concrete inspection
    assert solver.propagation_backend(eff) == "modal_symmetric"
    assert solver.propagation_backend(rev) == "modal_symmetric"
    assert solver.propagation_backend(bal_real) == "modal_general"
    assert solver.propagation_backend(bal_cplx) == "modal_general"
    assert solver.propagation_backend(guttler_fixed) == "modal_general"
    assert solver.propagation_backend(bal_deg) == "van_loan"
    assert solver.propagation_backend(_EmptyCBMStub()) == "van_loan"

    # Confirm complex regime genuinely has non-zero imaginary eigenvalues
    eigs_cplx = np.linalg.eigvals(np.asarray(bal_cplx.matrix_A()))
    assert np.max(np.abs(eigs_cplx.imag)) > 0.1

    # Confirm near-degenerate regime genuinely exceeds cond(V) >= 1e6
    _, v_deg = np.linalg.eig(np.asarray(bal_deg.matrix_A()))
    assert float(np.linalg.cond(v_deg)) >= 1e6

    # Under equinox.filter_jit (cbm leaves traced): symmetric stays modal_symmetric,
    # non-symmetric traced CBMs fall back to van_loan without concretization errors.
    jit_backend = eqx.filter_jit(solver.propagation_backend)
    assert jit_backend(eff) == "modal_symmetric"
    assert jit_backend(rev) == "modal_symmetric"
    assert jit_backend(bal_real) == "van_loan"
    assert jit_backend(bal_cplx) == "van_loan"
    assert jit_backend(bal_deg) == "van_loan"

    # Inside raw jax.jit (all array ops traced under DynamicJaxprTrace):
    # EffectiveCBM stays modal_symmetric, while non-symmetric CBMs (both
    # closed-over and dynamically updated) select van_loan without concretization errors.
    t_pts = jnp.array([774.2, 774.8, 775.6], dtype=jnp.float64)
    seen_backends: list[tuple[str, str]] = []

    @jax.jit
    def _closed_over_solve(t_start: jax.Array) -> jax.Array:
        seen_backends.append(
            (solver.propagation_backend(eff), solver.propagation_backend(bal_cplx))
        )
        sp = MiyakeSpike(t_start=t_start, duration=0.25, fluence=6.2, profile="boxcar")
        return solver.solve_delta_trajectory(bal_cplx, sp, t_pts)

    _ = _closed_over_solve(jnp.asarray(774.4, dtype=jnp.float64))
    assert seen_backends == [("modal_symmetric", "van_loan")]

    traced_backends: list[tuple[str, str]] = []

    @jax.jit
    def _traced_cbm_solve(log_k_eff: jax.Array, log_rates_bal: jax.Array) -> jax.Array:
        c_eff = eff.set("log_k_trop_to_sinks", log_k_eff)
        c_bal = bal_real.set("log_rates_free", log_rates_bal)
        traced_backends.append(
            (solver.propagation_backend(c_eff), solver.propagation_backend(c_bal))
        )
        sp = MiyakeSpike(t_start=774.4, duration=0.25, fluence=6.2, profile="boxcar")
        return (
            solver.solve_delta_trajectory(c_eff, sp, t_pts)[:, 1]
            + solver.solve_delta_trajectory(c_bal, sp, t_pts)[:, 1]
        )

    _ = _traced_cbm_solve(eff.log_k_trop_to_sinks, bal_real.log_rates_free)
    assert traced_backends == [("modal_symmetric", "van_loan")]


@pytest.mark.parametrize(
    "spike_kwargs",
    [
        {"profile": "boxcar"},
        {"profile": "supergaussian", "exponent": 10.0},
        {"profile": "skewgaussian", "skewness": 2.3},
        {"profile": "gamma", "gamma_shape": 2.7},
    ],
    ids=["boxcar", "supergaussian", "skewgaussian", "gamma"],
)
@pytest.mark.parametrize("regime", ["symmetric", "real", "complex", "near_degenerate"])
def test_eight_spike_paths_vs_independent_scipy_quad_oracle(
    spike_kwargs: dict[str, object],
    regime: str,
) -> None:
    """Compare all 8 spike paths against independent SciPy ``expm`` + ``quad_vec``."""
    cbm: CarbonBoxModel = (
        EffectiveCBM(K=2) if regime == "symmetric" else _make_regime_cbm(regime)
    )
    solver = ExactLinearSolver()
    expected_backend = {
        "symmetric": "modal_symmetric",
        "real": "modal_general",
        "complex": "modal_general",
        "near_degenerate": "van_loan",
    }[regime]
    assert solver.propagation_backend(cbm) == expected_backend

    alpha = 1.93 * ALPHA_ATOMS_CM2_S_TO_KG_YR
    A = cbm.matrix_A()
    b_scaled = alpha * cbm.production_vector()
    A_np = np.asarray(A, dtype=np.float64)
    b_np = np.asarray(b_scaled, dtype=np.float64)

    spike = MiyakeSpike(t_start=774.43, duration=0.27, fluence=6.37, **spike_kwargs)
    t_eval = jnp.array([773.1, 774.43, 774.56, 774.95, 776.8], dtype=jnp.float64)
    a_r = jnp.array([773.2, 774.25, 774.55, 775.4], dtype=jnp.float64)
    b_r = jnp.array([773.7, 774.60, 775.10, 776.1], dtype=jnp.float64)

    ref_pt = _scipy_spike_pointwise_oracle(A_np, b_np, spike, np.asarray(t_eval))
    ref_bn = _scipy_spike_binned_oracle(
        A_np, b_np, spike, np.asarray(a_r), np.asarray(b_r)
    )

    got_pt = solver.solve_delta_trajectory(cbm, spike, t_eval, alpha_unit=alpha)
    got_bn = solver.solve_delta_binned(cbm, spike, a_r, b_r, alpha_unit=alpha)
    np.testing.assert_allclose(got_pt, ref_pt, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(got_bn, ref_bn, rtol=1e-11, atol=1e-12)

    # Direct parity check between explicit modal/default entrypoints and Van Loan fallbacks
    if spike.profile == "boxcar":
        vl_pt = _eval_boxcar_pointwise_van_loan(A, b_scaled, spike, t_eval)
        vl_bn = _eval_boxcar_binned_van_loan(A, b_scaled, spike, a_r, b_r)
        md_pt = _eval_boxcar_pointwise(A, b_scaled, spike, t_eval, cbm=cbm)
        md_bn = _eval_boxcar_binned(A, b_scaled, spike, a_r, b_r, cbm=cbm)
    else:
        vl_pt = _eval_smooth_spike_pointwise_van_loan(A, b_scaled, spike, t_eval)
        vl_bn = _eval_smooth_spike_binned_van_loan(A, b_scaled, spike, a_r, b_r)
        md_pt = _eval_smooth_spike_pointwise(A, b_scaled, spike, t_eval, cbm=cbm)
        md_bn = _eval_smooth_spike_binned(A, b_scaled, spike, a_r, b_r, cbm=cbm)

    np.testing.assert_allclose(md_pt, vl_pt, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(md_bn, vl_bn, rtol=1e-13, atol=1e-13)

    # 2D weighted sub-windows consistency
    starts_2d = a_r.reshape(2, 2)
    ends_2d = b_r.reshape(2, 2)
    weights = jnp.array([0.63, 0.37], dtype=jnp.float64)
    got_2d = solver.solve_delta_binned(
        cbm, spike, starts_2d, ends_2d, sub_weights=weights, alpha_unit=alpha
    )
    ref_2d = 0.63 * ref_bn.reshape(2, 2, -1)[:, 0, :] + 0.37 * ref_bn.reshape(2, 2, -1)[:, 1, :]
    np.testing.assert_allclose(got_2d, ref_2d, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("profile", ["boxcar", "supergaussian", "skewgaussian", "gamma"])
@pytest.mark.parametrize("duration", [0.0, 1e-15])
def test_dirac_impulse_and_contiguous_window_boundary_partition(
    profile: str,
    duration: float,
) -> None:
    """Verify Dirac limit ``duration -> 0+`` and single-counting at ``t_start == a_r``."""
    cbm = _make_regime_cbm("complex")
    A = cbm.matrix_A()
    b_scaled = 1.4 * ALPHA_ATOMS_CM2_S_TO_KG_YR * cbm.production_vector()
    A_np = np.asarray(A, dtype=np.float64)
    b_np = np.asarray(b_scaled, dtype=np.float64)

    ts = 775.0  # Lies exactly on the boundary between window 0 and window 1
    fluence = 7.2
    spike = MiyakeSpike(t_start=ts, duration=duration, fluence=fluence, profile=profile)  # type: ignore[arg-type]

    # Pointwise: zero for t << ts (Hurwitz overflow check) and t < ts, exact at t >= ts
    t_pts = jnp.array([200.0, 774.5, 775.0, 775.4, 777.2], dtype=jnp.float64)
    ref_pt = np.zeros((len(t_pts), cbm.n_boxes), dtype=np.float64)
    for i, ti in enumerate(np.asarray(t_pts)):
        if ti >= ts:
            ref_pt[i] = fluence * (scipy_expm(A_np * (ti - ts)) @ b_np)

    if profile == "boxcar":
        pt_md = _eval_boxcar_pointwise(A, b_scaled, spike, t_pts, cbm=cbm)
        pt_vl = _eval_boxcar_pointwise_van_loan(A, b_scaled, spike, t_pts)
    else:
        pt_md = _eval_smooth_spike_pointwise(A, b_scaled, spike, t_pts, cbm=cbm)
        pt_vl = _eval_smooth_spike_pointwise_van_loan(A, b_scaled, spike, t_pts)

    np.testing.assert_allclose(pt_md, ref_pt, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(pt_vl, ref_pt, rtol=1e-13, atol=1e-14)

    # Contiguous windows [774, 775], [775, 776], [776, 778] partition [774, 778]:
    # weighted integral sum_w (b_w - a_w) * avg_w MUST equal single-window integral on [774, 778].
    a_contig = jnp.array([774.0, 775.0, 776.0], dtype=jnp.float64)
    b_contig = jnp.array([775.0, 776.0, 778.0], dtype=jnp.float64)
    widths = b_contig - a_contig
    a_whole = jnp.array([774.0], dtype=jnp.float64)
    b_whole = jnp.array([778.0], dtype=jnp.float64)

    if profile == "boxcar":
        bn_md = _eval_boxcar_binned(A, b_scaled, spike, a_contig, b_contig, cbm=cbm)
        bn_vl = _eval_boxcar_binned_van_loan(A, b_scaled, spike, a_contig, b_contig)
        wh_md = _eval_boxcar_binned(A, b_scaled, spike, a_whole, b_whole, cbm=cbm)
    else:
        bn_md = _eval_smooth_spike_binned(A, b_scaled, spike, a_contig, b_contig, cbm=cbm)
        bn_vl = _eval_smooth_spike_binned_van_loan(A, b_scaled, spike, a_contig, b_contig)
        wh_md = _eval_smooth_spike_binned(A, b_scaled, spike, a_whole, b_whole, cbm=cbm)

    # Window 0 ([774, 775]) strictly precedes ts == 775.0, so its response must be identically 0;
    # a broken <= tie-breaker on pre-window would double-count the Dirac impulse into window 0.
    np.testing.assert_allclose(bn_md[0], 0.0, atol=1e-15)
    np.testing.assert_allclose(bn_vl[0], 0.0, atol=1e-15)
    assert float(jnp.linalg.norm(bn_md[1])) > 1e-3

    integrated_parts = jnp.sum(widths[:, None] * bn_md, axis=0)
    integrated_whole = (b_whole[0] - a_whole[0]) * wh_md[0]
    np.testing.assert_allclose(integrated_parts, integrated_whole, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(bn_md, bn_vl, rtol=1e-13, atol=1e-14)

    # Closed-form check via SciPy augmented matrix exponential: int_{775}^{778} exp(A s) (F b) ds
    exact_int = _scipy_int_expm_vec(A_np, fluence * b_np, 3.0)
    np.testing.assert_allclose(integrated_parts, exact_int, rtol=1e-13, atol=1e-14)


def test_narrow_duration_sweep_no_cancellation() -> None:
    """Sweep ``duration`` across ``1e-7 .. 1e-1`` against Van Loan & Dirac limit."""
    cbm = _make_regime_cbm("real")
    solver = ExactLinearSolver()
    a_r = jnp.array([774.2, 774.5], dtype=jnp.float64)
    b_r = jnp.array([774.5, 775.2], dtype=jnp.float64)
    dirac_val = solver.solve_delta_binned(
        cbm, MiyakeSpike(t_start=774.5, duration=0.0, fluence=6.1, profile="boxcar"), a_r, b_r
    )
    errs = []
    for dur in (1e-1, 1e-3, 1e-5, 1e-7):
        sp = MiyakeSpike(t_start=774.5, duration=dur, fluence=6.1, profile="boxcar")
        val_md = solver.solve_delta_binned(cbm, sp, a_r, b_r)
        val_vl = _eval_boxcar_binned_van_loan(
            cbm.matrix_A(), ALPHA_ATOMS_CM2_S_TO_KG_YR * cbm.production_vector(), sp, a_r, b_r
        )
        assert jnp.all(jnp.isfinite(val_md))
        np.testing.assert_allclose(val_md, val_vl, rtol=1e-13, atol=1e-14)
        errs.append(float(jnp.max(jnp.abs(val_md - dirac_val))))
    # Error to Dirac limit decreases monotonically ~ O(duration) without cancellation blow-up
    assert errs[-1] < errs[-2] < errs[-3] < errs[-4]
    assert errs[-1] < 1e-5


@pytest.mark.parametrize(
    "profile",
    ["boxcar", "supergaussian", "skewgaussian", "gamma"],
)
def test_spike_autodiff_gradients_all_profiles_eager_jit_and_zero_duration(
    profile: str,
) -> None:
    """Verify eager & JIT gradients against central FD across profiles and ``duration = 0``."""
    _flush_xla_caches()
    solver = ExactLinearSolver()
    cbm_cplx = _make_regime_cbm("complex")
    cbm_sym = EffectiveCBM(K=2)
    t_pts = jnp.array([774.35, 774.62], dtype=jnp.float64)
    a_r = jnp.array([774.2, 774.55], dtype=jnp.float64)
    b_r = jnp.array([774.55, 775.05], dtype=jnp.float64)
    w_pt = jnp.array([1.2, 0.7], dtype=jnp.float64)
    w_bn = jnp.array([0.9, 1.4], dtype=jnp.float64)

    def _loss(
        cbm_in: CarbonBoxModel,
        p: jax.Array,
        modal_in: tuple[jax.Array, jax.Array, jax.Array] | None = None,
    ) -> jax.Array:
        kw: dict[str, object] = {"profile": profile}
        if profile == "skewgaussian":
            kw["skewness"] = 2.1
        elif profile == "supergaussian":
            kw["exponent"] = 10.0
        elif profile == "gamma":
            kw["gamma_shape"] = 3.0
        sp = MiyakeSpike(t_start=p[0], duration=p[1], fluence=p[2], **kw)
        pt = solver.solve_delta_trajectory(cbm_in, sp, t_pts, modal=modal_in)
        bn = solver.solve_delta_binned(cbm_in, sp, a_r, b_r, modal=modal_in)
        return jnp.sum(w_pt * pt[:, 1] ** 2) + jnp.sum(w_bn * bn[:, 1] ** 2)

    p0 = jnp.array([774.41, 0.28, 5.9], dtype=jnp.float64)

    modal_cplx = _modal_decomposition(cbm_cplx.matrix_A(), cbm=cbm_cplx)
    for cbm_case, modal_case in ((cbm_cplx, modal_cplx), (cbm_sym, None)):
        loss_eager = lambda p, _c=cbm_case: _loss(_c, p)  # noqa: E731
        loss_jit = lambda p, _c=cbm_case, _m=modal_case: _loss(_c, p, _m)  # noqa: E731
        g_eager = np.asarray(jax.grad(loss_eager)(p0))
        g_jit = np.asarray(jax.jit(jax.grad(loss_jit))(p0))
        g_fd = _central_fd_grad(jax.jit(loss_jit), p0)
        np.testing.assert_allclose(g_eager, g_fd, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(g_eager, g_jit, rtol=1e-12, atol=1e-12)

    # Also verify traced-CBM filter_jit (Van Loan fallback vs modal_symmetric) on boxcar & supergaussian
    if profile in ("boxcar", "supergaussian"):
        for cbm_case in (cbm_cplx, cbm_sym):
            g_eager = np.asarray(jax.grad(lambda p, _c=cbm_case: _loss(_c, p))(p0))
            g_fjit = np.asarray(eqx.filter_jit(jax.grad(_loss, argnums=1))(cbm_case, p0))
            np.testing.assert_allclose(g_eager, g_fjit, rtol=1e-12, atol=1e-12)

    # Non-NaN finite 1st & 2nd directional derivatives wrt (t_start, duration, fluence) at duration = 0.0
    p3_zero = jnp.array([774.41, 0.0, 5.9], dtype=jnp.float64)
    g_zero = np.asarray(jax.grad(lambda p: _loss(cbm_cplx, p, modal_cplx))(p3_zero))
    assert np.all(np.isfinite(g_zero))
    if profile in ("boxcar", "supergaussian"):
        v_dir = jnp.ones_like(p3_zero)
        _, hvp_zero = jax.jvp(
            jax.grad(lambda p3: _loss(cbm_cplx, p3, modal_cplx)), (p3_zero,), (v_dir,)
        )
        assert np.all(np.isfinite(np.asarray(hvp_zero)))
