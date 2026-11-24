"""Differentiable ODE and exact matrix-exponential solvers for ``ticktack 2.0``.

Implements four solver components sharing the :class:`AbstractSolver` interface:

1. :class:`ExactLinearSolver`: default solver for autonomous linear CBMs using:
   * ``(n + 2) x (n + 2)`` Van Loan first-order-hold (FOH) ``expm`` with
     ``jax.lax.scan`` for short/medium grids (``N_t <= rfft_threshold``) and
     causal 1D ``jnp.fft.rfft`` convolution for long IntCal horizons (``N_t > rfft_threshold``).
   * Smoothly clamped sub-interval ``[clip(t_s, a_r, b_r), clip(t_s + tau_d, a_r, b_r)]``
     analytical ``boxcar`` integration (``psi_1, psi_2`` with double-``where`` guard
     at ``tau_d -> 0+``) and 16-point Gauss–Legendre smooth spike seasonal-window
     integration (Hurwitz-safe ``exp(A * nonneg)`` factorization, zero step-size jitter).
   * Direct complex harmonic resolvent ``solve(1j * omega * I - A, b)`` (never forming
     ``A^2``) with exact ``sinc`` seasonal binning, eliminating the 2000-yr burn-in.
   * Cancellation-free ``t_ref``-anchored secular ramp ``(t - t_ref)(-A)^{-1} b`` and
     switched-on Van Loan ``psi_2`` / ``psi_3`` ramp.
2. :class:`DiffraxSolver`: adaptive ``Tsit5`` (explicit) and ``Kvaerno5`` (L-stable
   stiff implicit) integration via ``diffrax``, supporting time-varying inventories
   ``M_c(t)`` and rate matrices ``A(t)`` (e.g. fossil-fuel Suess dilution).
3. :class:`ModaxSolver` & :func:`wrap_jvp_only_scalar_fn`: lazy wrapper around optional
   ``modax-solvers`` CUDA kernels with a ``@jax.custom_vjp`` scalar log-likelihood
   adapter backed by batched forward-mode ``jvp`` / ``jacfwd`` for ``NumPyro`` NUTS
   and ``BlackJAX`` HMC/SMC.
"""

from __future__ import annotations

from collections.abc import Callable
import functools
import math
from typing import Any, Literal, NamedTuple

import diffrax
import equinox as eqx
import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
from numpy.polynomial.legendre import leggauss
import zodiax as zdx

from .cbm import CarbonBoxModel, EffectiveCBM, ReversibleCBM
from .constants import ALPHA_ATOMS_CM2_S_TO_KG_YR, DEFAULT_Q_STD
from .production import (
    GPProduction,
    GriddedProduction,
    HarmonicSolar,
    LinearTrend,
    LogGPProduction,
    MiyakeSpike,
    ProductionSum,
    ProductionTerm,
    SteadyProduction,
)

__all__ = [
    "GL16_NODES_01",
    "GL16_WEIGHTS_01",
    "van_loan_foh_blocks",
    "van_loan_psi1_vec",
    "van_loan_psi3_vec",
    "harmonic_resolvent",
    "propagate_foh_uniform",
    "wrap_jvp_only_scalar_fn",
    "AbstractSolver",
    "ExactLinearSolver",
    "DiffraxSolver",
    "ModaxSolver",
]

# Precomputed 16-point Gauss-Legendre rule mapped from [-1, 1] onto [0, 1]
_GL16_X, _GL16_W = leggauss(16)
GL16_NODES_01: np.ndarray = 0.5 * (_GL16_X + 1.0)
GL16_WEIGHTS_01: np.ndarray = 0.5 * _GL16_W


def _unpack_terms(production: ProductionTerm) -> tuple[ProductionTerm, ...]:
    if isinstance(production, ProductionSum):
        res: list[ProductionTerm] = []
        for p in production.parts:
            res.extend(_unpack_terms(p))
        return tuple(res)
    return (production,)


def _expm1_safe(z: jax.Array) -> jax.Array:
    """Cancellation-free ``exp(z) - 1`` for both real ``float64`` and ``complex128``."""
    if jnp.iscomplexobj(z):
        x = jnp.real(z)
        y = jnp.imag(z)
        sh = jnp.sin(0.5 * y)
        re = jnp.expm1(x) * jnp.cos(y) - 2.0 * sh * sh
        im = jnp.exp(x) * jnp.sin(y)
        return re + 1j * im
    return jnp.expm1(z)


_PSI1_HORNER_COEFFS: tuple[float, ...] = tuple(
    1.0 / math.factorial(k + 1) for k in range(9, -1, -1)
)
_PSI2_HORNER_COEFFS: tuple[float, ...] = tuple(
    1.0 / math.factorial(k + 2) for k in range(9, -1, -1)
)


def _scalar_psi1(z: ArrayLike) -> jax.Array:
    """Holomorphic ``psi_1(z) = (exp(z) - 1) / z`` with double-``where`` AD guard at ``z = 0``."""
    z_arr = jnp.asarray(z)
    small = jnp.abs(z_arr) < 0.1
    z_safe = jnp.where(small, 1.0 + 0.0 * z_arr, z_arr)
    poly = jnp.asarray(_PSI1_HORNER_COEFFS[0], dtype=z_arr.dtype)
    for coeff in _PSI1_HORNER_COEFFS[1:]:
        poly = poly * z_arr + coeff
    far = _expm1_safe(z_safe) / z_safe
    return jnp.where(small, poly, far)


def _scalar_psi2(z: ArrayLike) -> jax.Array:
    """Holomorphic ``psi_2(z) = (exp(z) - 1 - z) / z^2`` with double-``where`` AD guard at ``z = 0``."""
    z_arr = jnp.asarray(z)
    small = jnp.abs(z_arr) < 0.1
    z_safe = jnp.where(small, 1.0 + 0.0 * z_arr, z_arr)
    poly = jnp.asarray(_PSI2_HORNER_COEFFS[0], dtype=z_arr.dtype)
    for coeff in _PSI2_HORNER_COEFFS[1:]:
        poly = poly * z_arr + coeff
    far = (_expm1_safe(z_safe) - z_safe) / (z_safe * z_safe)
    return jnp.where(small, poly, far)


@functools.lru_cache(maxsize=64)
def _cached_eig_concrete(
    shape: tuple[int, int],
    a_bytes: bytes,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """LRU-cached spectral factorization ``A = V diag(lam) V_inv`` for concrete ``A``."""
    n = shape[0]
    if n == 0 or shape[1] != n:
        return None
    mat = np.frombuffer(a_bytes, dtype=np.float64).reshape(shape).copy()
    if not np.all(np.isfinite(mat)):
        return None
    lam, v_mat = np.linalg.eig(mat)
    cond_v = float(np.linalg.cond(v_mat))
    if not np.isfinite(cond_v) or cond_v >= 1e6:
        return None
    v_inv = np.linalg.solve(v_mat, np.eye(n, dtype=np.complex128))
    recon_err = float(np.linalg.norm((v_mat * lam[None, :]) @ v_inv - mat))
    if recon_err > 1e-9 * max(float(np.linalg.norm(mat)), 1.0):
        return None
    if np.max(np.abs(lam.imag)) == 0.0 and np.max(np.abs(v_mat.imag)) == 0.0:
        return (
            np.ascontiguousarray(lam.real, dtype=np.float64),
            np.ascontiguousarray(v_mat.real, dtype=np.float64),
            np.ascontiguousarray(v_inv.real, dtype=np.float64),
        )
    return (
        np.ascontiguousarray(lam, dtype=np.complex128),
        np.ascontiguousarray(v_mat, dtype=np.complex128),
        np.ascontiguousarray(v_inv, dtype=np.complex128),
    )


PropagationBackend = Literal["modal_symmetric", "modal_general", "van_loan"]


def _is_detailed_balance(cbm: CarbonBoxModel) -> bool:
    """True for detailed-balance CBMs (chart flag, else EffectiveCBM / ReversibleCBM)."""
    return bool(getattr(cbm, "is_detailed_balance", isinstance(cbm, (EffectiveCBM, ReversibleCBM))))


def _modal_decomposition(
    A: jax.Array,
    cbm: CarbonBoxModel | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array] | None:
    """Return ``(lam, V, V_inv)`` satisfying ``A = V diag(lam) V_inv``, or ``None``."""
    n = A.shape[0]
    if n == 0:
        return None

    # Case A: Detailed-balance CBMs (EffectiveCBM, ReversibleCBM, or chart-backed models) —
    # real symmetric eigh via flux balance M_i^{-1/2} A_{ij} M_j^{1/2}, valid under eager & JIT.
    if _is_detailed_balance(cbm):
        masses = cbm.reservoir_masses()
        m_sqrt = jnp.sqrt(masses)
        s_mat = (A / m_sqrt[:, None]) * m_sqrt[None, :]
        lam, q_orth = jnp.linalg.eigh(0.5 * (s_mat + s_mat.T))
        v_mat = m_sqrt[:, None] * q_orth
        v_inv = q_orth.T / m_sqrt[None, :]
        return lam, v_mat, v_inv

    # Case B: Concrete (non-traced) A — LRU-cached complex128/float64 eigendecomposition.
    if not isinstance(A, jax.core.Tracer):
        try:
            a_np = np.asarray(A, dtype=np.float64)
        except (
            jax.errors.TracerArrayConversionError,
            jax.errors.ConcretizationTypeError,
        ):
            return None
        cached = _cached_eig_concrete(a_np.shape, a_np.tobytes())
        if cached is None:
            return None
        lam_np, v_np, v_inv_np = cached
        return jnp.asarray(lam_np), jnp.asarray(v_np), jnp.asarray(v_inv_np)

    # Case C: Traced general non-symmetric A (e.g. BalancedCBM / OpenCBM) -> Van Loan fallback.
    return None


def van_loan_psi1_vec(
    A: jax.Array,
    v: jax.Array,
    dt: ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """Compute ``(exp(A * dt) @ v, dt * psi_1(A * dt) @ v)`` via ``(n + 1) x (n + 1)`` ``expm``.

    Never inverts ``A`` and remains smooth ``C^inf`` at ``dt = 0``.
    """
    n = A.shape[0]
    dt_arr = jnp.asarray(dt, dtype=jnp.float64)
    M = jnp.zeros((n + 1, n + 1), dtype=jnp.float64)
    M = M.at[:n, :n].set(A * dt_arr)
    M = M.at[:n, n].set(v * dt_arr)
    E = jax.scipy.linalg.expm(M)
    return E[:n, :n] @ v, E[:n, n]


def van_loan_foh_blocks(
    A: jax.Array,
    b_scaled: jax.Array,
    dt: ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute ``(Phi, g0, g1)`` for step ``dt`` via a single ``(n + 2) x (n + 2)`` ``expm``.

    Returns
    -------
    Phi : jax.Array
        ``exp(A * dt)`` of shape ``(n, n)``.
    g0 : jax.Array
        ``dt * psi_1(A * dt) @ b_scaled`` of shape ``(n,)``.
    g1 : jax.Array
        ``dt * psi_2(A * dt) @ b_scaled`` of shape ``(n,)``.
    """
    n = A.shape[0]
    dt_arr = jnp.asarray(dt, dtype=jnp.float64)
    M = jnp.zeros((n + 2, n + 2), dtype=jnp.float64)
    M = M.at[:n, :n].set(A * dt_arr)
    M = M.at[:n, n].set(b_scaled * dt_arr)
    M = M.at[n, n + 1].set(1.0)
    E = jax.scipy.linalg.expm(M)
    return E[:n, :n], E[:n, n], E[:n, n + 1]


def van_loan_psi3_vec(
    A: jax.Array,
    b_scaled: jax.Array,
    dt: ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute ``(Phi, g0, g1, g2)`` where ``g2 = dt * psi_3(A * dt) @ b_scaled``."""
    n = A.shape[0]
    dt_arr = jnp.asarray(dt, dtype=jnp.float64)
    M = jnp.zeros((n + 3, n + 3), dtype=jnp.float64)
    M = M.at[:n, :n].set(A * dt_arr)
    M = M.at[:n, n].set(b_scaled * dt_arr)
    M = M.at[n, n + 1].set(1.0)
    M = M.at[n + 1, n + 2].set(1.0)
    E = jax.scipy.linalg.expm(M)
    return E[:n, :n], E[:n, n], E[:n, n + 1], E[:n, n + 2]


def harmonic_resolvent(
    A: jax.Array,
    b_scaled: jax.Array,
    omega: ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """Solve ``(1j * omega * I - A) z = b_scaled`` directly in ``complex128``.

    Avoids forming ``A^2 + omega^2 I``, preserving condition number ``<= cond(A)``
    across arbitrarily stiff carbon-box rate spectra.
    """
    n = A.shape[0]
    A_c = A.astype(jnp.complex128)
    b_c = b_scaled.astype(jnp.complex128)
    w_c = jnp.asarray(omega, dtype=jnp.complex128)
    eye_c = jnp.eye(n, dtype=jnp.complex128)
    lhs = w_c[..., None, None] * (1j * eye_c) - A_c
    rhs = jnp.broadcast_to(b_c, lhs.shape[:-1])[..., None]
    z = jnp.linalg.solve(lhs, rhs)[..., 0]
    return jnp.real(z), jnp.imag(z)


def propagate_foh_uniform(
    A: jax.Array,
    b_scaled: jax.Array,
    dt: ArrayLike,
    delta_q: jax.Array,
    dy0: jax.Array | None = None,
    *,
    method: Literal["auto", "scan", "rfft"] = "auto",
    rfft_threshold: int = 4096,
) -> jax.Array:
    """Propagate piecewise-linear ``delta_q`` on a uniform grid of spacing ``dt``.

    Parameters
    ----------
    A : jax.Array
        LTI state matrix ``(n, n)``.
    b_scaled : jax.Array
        Scaled cosmic-ray input vector ``alpha_unit * b`` of shape ``(n,)``.
    dt : float or Array
        Uniform step size ``Delta t > 0``.
    delta_q : jax.Array
        Grid perturbations ``(N_t,)``.
    dy0 : jax.Array, optional
        Initial state ``delta_y(t_0)`` of shape ``(n,)`` (zeros if ``None``).
    method : {"auto", "scan", "rfft"}, optional
        ``"scan"`` runs ``jax.lax.scan``; ``"rfft"`` runs causal 1D FFT convolution
        with exact endpoint correction ``dy0 - g1 * delta_q[0]``; ``"auto"`` uses
        ``"scan"`` for ``N_t <= rfft_threshold`` and ``"rfft"`` for ``N_t > rfft_threshold``.

    Returns
    -------
    jax.Array
        Exact FOH state trajectory ``(N_t, n)``.
    """
    dq = jnp.asarray(delta_q, dtype=jnp.float64)
    n_steps = dq.shape[0]
    n = A.shape[0]
    init_y = jnp.zeros((n,), dtype=jnp.float64) if dy0 is None else jnp.asarray(dy0, dtype=jnp.float64)

    if n_steps == 0:
        return jnp.zeros((0, n), dtype=jnp.float64)
    if n_steps == 1:
        return init_y[None, :]

    Phi, g0, g1 = van_loan_foh_blocks(A, b_scaled, dt)
    w0 = g0 - g1
    w1 = g1

    use_rfft = method == "rfft" or (method == "auto" and n_steps > rfft_threshold)
    if not use_rfft:
        inputs = w0[None, :] * dq[:-1, None] + w1[None, :] * dq[1:, None]

        def _scan_step(y_curr: jax.Array, u_curr: jax.Array) -> tuple[jax.Array, jax.Array]:
            y_next = Phi @ y_curr + u_curr
            return y_next, y_next

        _, ys_tail = jax.lax.scan(_scan_step, init_y, inputs)
        return jnp.concatenate([init_y[None, :], ys_tail], axis=0)

    # Exact causal 1D rfft FOH convolution:
    # H[0] = w1, H[l] = Phi^{l-1} (w0 + Phi @ w1) for l >= 1;
    # dy[k] = (H * dq)[k] + Phi^k (init_y - w1 * dq[0]).
    u_step = w0 + Phi @ w1
    v_corr = init_y - w1 * dq[0]
    init_pair = jnp.stack([u_step, Phi @ v_corr], axis=1)  # corresponds to lag l = 1

    def _irf_step(pair: jax.Array, _: None) -> tuple[jax.Array, jax.Array]:
        next_pair = Phi @ pair
        return next_pair, pair

    _, pairs = jax.lax.scan(_irf_step, init_pair, None, length=n_steps - 1)
    # pairs has shape (n_steps - 1, n, 2) for lags l = 1 .. n_steps - 1
    H = jnp.concatenate([w1[None, :], pairs[:, :, 0]], axis=0)
    y_hom = jnp.concatenate([v_corr[None, :], pairs[:, :, 1]], axis=0)

    n_fft = 1 << (2 * n_steps - 2).bit_length()
    H_f = jnp.fft.rfft(H, n=n_fft, axis=0)
    Q_f = jnp.fft.rfft(dq, n=n_fft)
    conv = jnp.fft.irfft(H_f * Q_f[:, None], n=n_fft, axis=0)[:n_steps]
    return conv + y_hom


def _eval_harmonic_pointwise(
    A: jax.Array,
    b_scaled: jax.Array,
    term: HarmonicSolar,
    t: jax.Array,
) -> jax.Array:
    """Exact asymptotic periodic state ``delta_y_harm(t)`` of shape ``(len(t), n)``."""
    omegas = term.omegas
    u_m, v_m = harmonic_resolvent(A, b_scaled, omegas)  # (M, n)
    cos_vec = term.c_cos[:, None] * u_m + term.c_sin[:, None] * v_m  # (M, n)
    sin_vec = term.c_sin[:, None] * u_m - term.c_cos[:, None] * v_m  # (M, n)
    theta = (t[:, None] - term.t_ref) * omegas[None, :]  # (T, M)
    return jnp.cos(theta) @ cos_vec + jnp.sin(theta) @ sin_vec


def _eval_harmonic_binned(
    A: jax.Array,
    b_scaled: jax.Array,
    term: HarmonicSolar,
    a_r: jax.Array,
    b_r: jax.Array,
) -> jax.Array:
    """Exact ``sinc`` seasonal sub-window average of ``delta_y_harm`` over ``[a_r, b_r]``."""
    omegas = term.omegas
    u_m, v_m = harmonic_resolvent(A, b_scaled, omegas)  # (M, n)
    cos_vec = term.c_cos[:, None] * u_m + term.c_sin[:, None] * v_m
    sin_vec = term.c_sin[:, None] * u_m - term.c_cos[:, None] * v_m

    mid = 0.5 * (a_r + b_r)  # (W,)
    width = b_r - a_r  # (W,)
    # Note: jnp.sinc(x) = sin(pi * x) / (pi * x)
    sinc_factor = jnp.sinc((width[:, None] * omegas[None, :]) / (2.0 * math.pi))
    theta_mid = (mid[:, None] - term.t_ref) * omegas[None, :]
    avg_cos = sinc_factor * jnp.cos(theta_mid)
    avg_sin = sinc_factor * jnp.sin(theta_mid)
    return avg_cos @ cos_vec + avg_sin @ sin_vec


def _eval_trend_pointwise(
    A: jax.Array,
    b_scaled: jax.Array,
    term: LinearTrend,
    t: jax.Array,
) -> jax.Array:
    if not term.switched_on:
        v_ss = jnp.linalg.solve(-A, b_scaled)
        return (term.slope * (t - term.t_ref))[:, None] * v_ss[None, :]

    def _single(ti: jax.Array) -> jax.Array:
        delta = jnp.maximum(ti - term.t_ref, 0.0)
        _, _, g1 = van_loan_foh_blocks(A, b_scaled, delta)
        return term.slope * delta * g1

    return jax.vmap(_single)(t)


def _eval_trend_binned(
    A: jax.Array,
    b_scaled: jax.Array,
    term: LinearTrend,
    a_r: jax.Array,
    b_r: jax.Array,
) -> jax.Array:
    if not term.switched_on:
        v_ss = jnp.linalg.solve(-A, b_scaled)
        mid = 0.5 * (a_r + b_r)
        return (term.slope * (mid - term.t_ref))[:, None] * v_ss[None, :]

    def _single_win(ar: jax.Array, br: jax.Array) -> jax.Array:
        da = jnp.maximum(ar - term.t_ref, 0.0)
        db = jnp.maximum(br - term.t_ref, 0.0)
        _, _, _, g2_a = van_loan_psi3_vec(A, b_scaled, da)
        _, _, _, g2_b = van_loan_psi3_vec(A, b_scaled, db)
        int_val = (db * db) * g2_b - (da * da) * g2_a
        return (term.slope / (br - ar)) * int_val

    return jax.vmap(_single_win)(a_r, b_r)


def _clamp_spike_panels(
    spike: MiyakeSpike, lo: jax.Array | None, hi: jax.Array, dirac_mask: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Clamp ``spike.unit_support_panels()`` onto ``(-inf, hi]`` or ``[lo, hi]``."""
    ts, tau_d = spike.t_start, jnp.maximum(spike.duration, 0.0)
    is_tiny = tau_d <= 1e-12
    safe_tau = jnp.where(is_tiny, 1.0, tau_d)
    panels = jnp.asarray(spike.unit_support_panels(), dtype=jnp.float64)
    u_los, du = panels[:, 0], panels[:, 1] - panels[:, 0]
    p_a, p_b = ts + tau_d * u_los, ts + tau_d * panels[:, 1]
    hi_e = hi[..., None]
    sub_a = jnp.minimum(p_a, hi_e) if lo is None else jnp.clip(p_a, lo[..., None], hi_e)
    sub_b = jnp.minimum(p_b, hi_e) if lo is None else jnp.clip(p_b, lo[..., None], hi_e)
    span = sub_b - sub_a
    scale = spike.fluence * jnp.where(is_tiny, jnp.where(dirac_mask[..., None], du, 0.0), span / safe_tau)
    return sub_a, sub_b, span, scale, u_los, du


def _boxcar_clamped_interval(
    spike: MiyakeSpike, lo: jax.Array | None, hi: jax.Array, dirac_mask: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Clamp single-panel boxcar support onto ``(-inf, hi]`` or ``[lo, hi]``."""
    _, sub_b, span, scale, _, _ = _clamp_spike_panels(spike, lo, hi, dirac_mask)
    return span[..., 0], hi - sub_b[..., 0], scale[..., 0]


def _smooth_clamped_panels(
    spike: MiyakeSpike, lo: jax.Array | None, hi: jax.Array, dirac_mask: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Map 16-point Gauss–Legendre nodes onto clamped smooth-spike sub-panels."""
    ts, tau_d = spike.t_start, jnp.maximum(spike.duration, 0.0)
    is_tiny = tau_d <= 1e-12
    safe_tau = jnp.where(is_tiny, 1.0, tau_d)
    sub_a, sub_b, span, scale, u_los, du = _clamp_spike_panels(spike, lo, hi, dirac_mask)
    xi = jnp.asarray(GL16_NODES_01, dtype=jnp.float64)
    wq = jnp.asarray(GL16_WEIGHTS_01, dtype=jnp.float64)
    lin_q = sub_a[..., None] + span[..., None] * xi
    u_lin = jnp.clip((lin_q - ts) / safe_tau, u_los[:, None], (u_los + du)[:, None])
    u_q = jnp.where(is_tiny, u_los[:, None] + du[:, None] * xi, u_lin)
    tau_q = jnp.where(is_tiny, ts + tau_d * u_q, lin_q)
    return sub_b, tau_q, scale, wq * spike.unit_profile(u_q)


def _smooth_prefix_van_loan_one(
    A: jax.Array, b_scaled: jax.Array, spike: MiyakeSpike, cutoff: jax.Array, dirac_mask: jax.Array
) -> jax.Array:
    """Propagate smooth-spike forcing on ``(-inf, cutoff]`` to ``cutoff`` via ``expm``."""
    sub_b, tau_q, scale, w_phi = _smooth_clamped_panels(spike, None, cutoff, dirac_mask)

    def _panel(sb: jax.Array, tq: jax.Array, sc: jax.Array, wp: jax.Array) -> jax.Array:
        lags = jnp.maximum(sb - tq, 0.0)
        vecs = jax.vmap(lambda lag: jax.scipy.linalg.expm(A * lag) @ b_scaled)(lags)
        Phi_shift = jax.scipy.linalg.expm(A * (cutoff - sb))
        return sc * (Phi_shift @ (wp @ vecs))

    return jnp.sum(jax.vmap(_panel)(sub_b, tau_q, scale, w_phi), axis=0)


def _smooth_prefix_modes(
    spike: MiyakeSpike, cutoff: jax.Array, dirac_mask: jax.Array, lam: jax.Array
) -> jax.Array:
    """Propagate smooth-spike forcing on ``(-inf, cutoff]`` to ``cutoff`` in modal coordinates."""
    sub_b, tau_q, scale, w_phi = _smooth_clamped_panels(spike, None, cutoff, dirac_mask)
    lags = jnp.maximum(sub_b[..., None] - tau_q, 0.0)
    exp_lags = jnp.exp(lags[..., None] * lam)
    panel_modes = jnp.einsum("mpq,mpqn->mpn", w_phi.astype(lam.dtype), exp_lags)
    shift = cutoff[:, None] - sub_b
    return jnp.sum(scale[:, :, None] * jnp.exp(shift[:, :, None] * lam) * panel_modes, axis=1)


@eqx.filter_jit
def _eval_boxcar_pointwise_van_loan(
    A: jax.Array,
    b_scaled: jax.Array,
    spike: MiyakeSpike,
    t: jax.Array,
) -> jax.Array:
    """Van Loan ``expm`` fallback for pointwise boxcar spike evaluation."""
    delta_act, delta_post, scale = _boxcar_clamped_interval(spike, None, t, t >= spike.t_start)

    def _one_t(da: jax.Array, dp: jax.Array, sc: jax.Array) -> jax.Array:
        _, psi1_act_b = van_loan_psi1_vec(A * da, b_scaled, 1.0)
        return sc * (jax.scipy.linalg.expm(A * dp) @ psi1_act_b)

    return jax.vmap(_one_t)(delta_act, delta_post, scale)


def _eval_boxcar_pointwise(
    A: jax.Array,
    b_scaled: jax.Array,
    spike: MiyakeSpike,
    t: jax.Array,
    *,
    cbm: CarbonBoxModel | None = None,
    modal: tuple[jax.Array, jax.Array, jax.Array] | None = None,
) -> jax.Array:
    """Exact pointwise boxcar spike response with modal spectral fast path."""
    eff_modal = _modal_decomposition(A, cbm=cbm) if modal is None else modal
    if eff_modal is None:
        return _eval_boxcar_pointwise_van_loan(A, b_scaled, spike, t)

    lam, v_mat, v_inv = eff_modal
    b_tilde = v_inv @ b_scaled.astype(v_inv.dtype)
    delta_act, delta_post, scale = _boxcar_clamped_interval(spike, None, t, t >= spike.t_start)
    y_tilde = (
        scale[:, None]
        * jnp.exp(delta_post[:, None] * lam[None, :])
        * _scalar_psi1(delta_act[:, None] * lam[None, :])
        * b_tilde[None, :]
    )
    return jnp.real(y_tilde @ v_mat.T)


@eqx.filter_jit
def _eval_boxcar_binned_van_loan(
    A: jax.Array,
    b_scaled: jax.Array,
    spike: MiyakeSpike,
    a_r: jax.Array,
    b_r: jax.Array,
) -> jax.Array:
    """Van Loan ``expm`` fallback for seasonal-window averaged boxcar pulse."""

    def _one_win(ar: jax.Array, br: jax.Array) -> jax.Array:
        width = br - ar
        delta_pre, lag_pre, s_pre = _boxcar_clamped_interval(spike, None, ar, spike.t_start < ar)
        _, psi1_pre_b = van_loan_psi1_vec(A * delta_pre, b_scaled, 1.0)
        dy_ar = s_pre * (jax.scipy.linalg.expm(A * lag_pre) @ psi1_pre_b)
        _, int_pre = van_loan_psi1_vec(A, dy_ar, width)

        in_mask = (spike.t_start >= ar) & (spike.t_start < br)
        delta_in, delta_post, s_in = _boxcar_clamped_interval(spike, ar, br, in_mask)
        _, psi1_in_b, g1_in = van_loan_foh_blocks(A * delta_in, b_scaled, 1.0)
        int_active = s_in * delta_in * g1_in
        dy_t2 = s_in * psi1_in_b
        _, int_post = van_loan_psi1_vec(A, dy_t2, delta_post)
        return (int_pre + int_active + int_post) / width

    return jax.lax.map(
        lambda ab: _one_win(ab[0], ab[1]),
        (a_r, b_r),
        batch_size=32,
    )


def _eval_boxcar_binned(
    A: jax.Array,
    b_scaled: jax.Array,
    spike: MiyakeSpike,
    a_r: jax.Array,
    b_r: jax.Array,
    *,
    cbm: CarbonBoxModel | None = None,
    modal: tuple[jax.Array, jax.Array, jax.Array] | None = None,
) -> jax.Array:
    """Exact analytical seasonal window average of a boxcar Miyake pulse."""
    eff_modal = _modal_decomposition(A, cbm=cbm) if modal is None else modal
    if eff_modal is None:
        return _eval_boxcar_binned_van_loan(A, b_scaled, spike, a_r, b_r)

    lam, v_mat, v_inv = eff_modal
    b_tilde = v_inv @ b_scaled.astype(v_inv.dtype)
    width = b_r - a_r

    delta_pre, lag_pre, s_pre = _boxcar_clamped_interval(spike, None, a_r, spike.t_start < a_r)
    int_pre_modes = (
        (s_pre * width)[:, None]
        * _scalar_psi1(width[:, None] * lam[None, :])
        * jnp.exp(lag_pre[:, None] * lam[None, :])
        * _scalar_psi1(delta_pre[:, None] * lam[None, :])
    )

    in_mask = (spike.t_start >= a_r) & (spike.t_start < b_r)
    delta_in, delta_post, s_in = _boxcar_clamped_interval(spike, a_r, b_r, in_mask)
    z_in = delta_in[:, None] * lam[None, :]
    int_active_modes = (s_in * delta_in)[:, None] * _scalar_psi2(z_in)
    int_post_modes = (
        (s_in * delta_post)[:, None]
        * _scalar_psi1(delta_post[:, None] * lam[None, :])
        * _scalar_psi1(z_in)
    )
    y_tilde = (
        (int_pre_modes + int_active_modes + int_post_modes) / width[:, None]
    ) * b_tilde[None, :]
    return jnp.real(y_tilde @ v_mat.T)


@eqx.filter_jit
def _eval_smooth_spike_pointwise_van_loan(
    A: jax.Array,
    b_scaled: jax.Array,
    spike: MiyakeSpike,
    t: jax.Array,
) -> jax.Array:
    """Van Loan ``expm`` fallback for smooth ``MiyakeSpike`` pointwise evaluation."""
    return jax.vmap(
        lambda ti: _smooth_prefix_van_loan_one(A, b_scaled, spike, ti, ti >= spike.t_start)
    )(t)


def _eval_smooth_spike_pointwise(
    A: jax.Array,
    b_scaled: jax.Array,
    spike: MiyakeSpike,
    t: jax.Array,
    *,
    cbm: CarbonBoxModel | None = None,
    modal: tuple[jax.Array, jax.Array, jax.Array] | None = None,
) -> jax.Array:
    """16-point Gauss–Legendre clamped quadrature for smooth ``MiyakeSpike`` profiles."""
    eff_modal = _modal_decomposition(A, cbm=cbm) if modal is None else modal
    if eff_modal is None:
        return _eval_smooth_spike_pointwise_van_loan(A, b_scaled, spike, t)

    lam, v_mat, v_inv = eff_modal
    b_tilde = v_inv @ b_scaled.astype(v_inv.dtype)
    y_tilde = _smooth_prefix_modes(spike, t, t >= spike.t_start, lam) * b_tilde[None, :]
    return jnp.real(y_tilde @ v_mat.T)


@eqx.filter_jit
def _eval_smooth_spike_binned_van_loan(
    A: jax.Array,
    b_scaled: jax.Array,
    spike: MiyakeSpike,
    a_r: jax.Array,
    b_r: jax.Array,
) -> jax.Array:
    """Van Loan ``expm`` fallback for seasonal-window smooth ``MiyakeSpike`` profiles."""

    def _one_win(ar: jax.Array, br: jax.Array) -> jax.Array:
        width = br - ar
        dy_ar = _smooth_prefix_van_loan_one(A, b_scaled, spike, ar, spike.t_start < ar)
        _, int_pre_total = van_loan_psi1_vec(A, dy_ar, width)
        in_mask = (spike.t_start >= ar) & (spike.t_start < br)
        _, tau_in, s_in, w_phi_in = _smooth_clamped_panels(spike, ar, br, in_mask)

        def _panel_in(tq: jax.Array, sc: jax.Array, wp: jax.Array) -> jax.Array:
            rem = jnp.maximum(br - tq, 0.0)
            int_vecs = jax.vmap(lambda r: van_loan_psi1_vec(A, b_scaled, r)[1])(rem)
            return sc * (wp @ int_vecs)

        int_in = jnp.sum(jax.vmap(_panel_in)(tau_in, s_in, w_phi_in), axis=0)
        return (int_pre_total + int_in) / width

    return jax.lax.map(
        lambda ab: _one_win(ab[0], ab[1]),
        (a_r, b_r),
        batch_size=32,
    )


def _eval_smooth_spike_binned(
    A: jax.Array,
    b_scaled: jax.Array,
    spike: MiyakeSpike,
    a_r: jax.Array,
    b_r: jax.Array,
    *,
    cbm: CarbonBoxModel | None = None,
    modal: tuple[jax.Array, jax.Array, jax.Array] | None = None,
) -> jax.Array:
    """16-point Gauss–Legendre seasonal-window integration for smooth ``MiyakeSpike`` profiles."""
    eff_modal = _modal_decomposition(A, cbm=cbm) if modal is None else modal
    if eff_modal is None:
        return _eval_smooth_spike_binned_van_loan(A, b_scaled, spike, a_r, b_r)

    lam, v_mat, v_inv = eff_modal
    b_tilde = v_inv @ b_scaled.astype(v_inv.dtype)
    width = b_r - a_r

    dy_ar_modes = _smooth_prefix_modes(spike, a_r, spike.t_start < a_r, lam)
    int_pre_modes = (
        width[:, None] * _scalar_psi1(width[:, None] * lam[None, :]) * dy_ar_modes
    )
    in_mask = (spike.t_start >= a_r) & (spike.t_start < b_r)
    _, tau_in, s_in, w_phi_in = _smooth_clamped_panels(spike, a_r, b_r, in_mask)
    rem = jnp.maximum(b_r[:, None, None] - tau_in, 0.0)
    w_phi_in = w_phi_in.astype(lam.dtype)
    rem_psi1 = rem[..., None] * _scalar_psi1(
        rem[..., None] * lam[None, None, None, :]
    )
    panel_in_modes = jnp.einsum("wpq,wpqn->wpn", w_phi_in, rem_psi1)
    int_in_modes = jnp.sum(s_in[:, :, None] * panel_in_modes, axis=1)

    y_tilde = ((int_pre_modes + int_in_modes) / width[:, None]) * b_tilde[None, :]
    return jnp.real(y_tilde @ v_mat.T)


def _foh_cumulative_integral(
    A: jax.Array,
    b_scaled: jax.Array,
    t0: jax.Array,
    dt: jax.Array,
    delta_q: jax.Array,
    dy0: jax.Array,
    query_ts: jax.Array,
    *,
    method: Literal["auto", "scan", "rfft"] = "auto",
    rfft_threshold: int = 4096,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate exact FOH state ``dy(query_ts)`` and cumulative integral ``Y(query_ts) = int_{t0}^t dy``."""
    n = A.shape[0]
    dy_grid = propagate_foh_uniform(
        A,
        b_scaled,
        dt,
        delta_q,
        dy0=dy0,
        method=method,
        rfft_threshold=rfft_threshold,
    )  # (N_t, n)

    # Augmented (2n)x(2n) Van Loan step for exact cell antiderivatives
    A_aug = jnp.zeros((2 * n, 2 * n), dtype=jnp.float64)
    A_aug = A_aug.at[:n, :n].set(A)
    A_aug = A_aug.at[n:, :n].set(jnp.eye(n, dtype=jnp.float64))
    b_aug = jnp.concatenate([b_scaled, jnp.zeros((n,), dtype=jnp.float64)])
    Phi_aug, g0_aug, g1_aug = van_loan_foh_blocks(A_aug, b_aug, dt)
    Psi_y = Phi_aug[n:, :n]  # dt * psi_1(A * dt)
    h0 = g0_aug[n:]
    h1 = g1_aug[n:]

    cell_ints = (
        dy_grid[:-1] @ Psi_y.T
        + delta_q[:-1, None] * (h0 - h1)[None, :]
        + delta_q[1:, None] * h1[None, :]
    )
    Y_grid = jnp.concatenate(
        [jnp.zeros((1, n), dtype=jnp.float64), jnp.cumsum(cell_ints, axis=0)],
        axis=0,
    )

    n_cells = delta_q.shape[0] - 1

    def _eval_at_query(tq: jax.Array) -> tuple[jax.Array, jax.Array]:
        pos = (tq - t0) / dt
        idx = jnp.clip(jnp.floor(pos).astype(jnp.int32), 0, n_cells - 1)
        s = jnp.clip(tq - (t0 + idx * dt), 0.0, dt)
        dq0 = delta_q[idx]
        dq1 = delta_q[idx + 1]
        dqs = dq0 + (s / dt) * (dq1 - dq0)

        Phi_s, g0_s, g1_s = van_loan_foh_blocks(A_aug, b_aug, s)
        dy_k = dy_grid[idx]
        dy_s = (
            Phi_s[:n, :n] @ dy_k
            + (g0_s[:n] - g1_s[:n]) * dq0
            + g1_s[:n] * dqs
        )
        Y_s = (
            Y_grid[idx]
            + Phi_s[n:, :n] @ dy_k
            + (g0_s[n:] - g1_s[n:]) * dq0
            + g1_s[n:] * dqs
        )
        return dy_s, Y_s

    return jax.vmap(_eval_at_query)(query_ts)


def _flatten_windows(
    t_starts: ArrayLike,
    t_ends: ArrayLike,
    sub_weights: ArrayLike | None,
) -> tuple[jax.Array, jax.Array, jax.Array | None, bool, tuple[int, ...]]:
    """Normalize 1D ``(N_r,)`` or 2D ``(N_r, M_w)`` seasonal window bounds."""
    ts = jnp.asarray(t_starts, dtype=jnp.float64)
    te = jnp.asarray(t_ends, dtype=jnp.float64)
    if ts.shape != te.shape:
        raise ValueError(
            f"t_starts shape {ts.shape} and t_ends shape {te.shape} must match."
        )
    if ts.ndim == 1:
        return ts, te, None, True, ts.shape
    if ts.ndim == 2:
        if sub_weights is None:
            w = jnp.full(ts.shape, 1.0 / ts.shape[1], dtype=jnp.float64)
        else:
            w_raw = jnp.broadcast_to(
                jnp.asarray(sub_weights, dtype=jnp.float64), ts.shape
            )
            w = w_raw / jnp.sum(w_raw, axis=-1, keepdims=True)
        return ts.reshape(-1), te.reshape(-1), w, False, ts.shape
    raise ValueError(f"t_starts must be 1D or 2D, got shape {ts.shape}.")


def _select_boxes(
    arr: jax.Array,
    box_indices: int | ArrayLike | None,
) -> jax.Array:
    """Select reservoir column(s) from ``(N_r, n)`` array."""
    if box_indices is None:
        return arr
    if isinstance(box_indices, int):
        return arr[:, box_indices]
    idx = jnp.asarray(box_indices, dtype=jnp.int32)
    if idx.ndim == 0:
        return arr[:, idx]
    if idx.shape == (arr.shape[0],):
        return arr[jnp.arange(arr.shape[0]), idx]
    return arr[:, idx]


def _delta_y_to_d14c(
    cbm: CarbonBoxModel,
    delta_y: jax.Array,
    *,
    box_indices: int | ArrayLike | None = None,
    q_std: ArrayLike = DEFAULT_Q_STD,
    alpha_unit: ArrayLike = ALPHA_ATOMS_CM2_S_TO_KG_YR,
    delta_global: ArrayLike = 0.0,
    delta_lab: ArrayLike = 0.0,
    relative_to: Literal["standard", "ambient"] = "standard",
) -> jax.Array:
    """Convert ``delta_y`` ``(..., n)`` to ``Delta14C`` (``permil``) via single global ``R_ref``."""
    trop_idx = jnp.asarray(cbm.topology.trop_box_indices, dtype=jnp.int32)
    y_ss = cbm.steady_state_c14(q_std=q_std, alpha_unit=alpha_unit)
    m_rel = cbm.relative_reservoir_masses()
    y_trop = jnp.sum(y_ss[trop_idx])

    r_ss_ratio = (y_ss / y_trop) / m_rel  # (n,)
    ss_d14c = 1000.0 * (r_ss_ratio - 1.0)
    excursion_scale = (
        1.0 + jnp.asarray(delta_global, dtype=jnp.float64) / 1000.0
        if relative_to == "ambient"
        else 1.0
    )
    excursion_d14c = 1000.0 * excursion_scale * ((delta_y / y_trop) / m_rel)
    d14c_all = (
        ss_d14c
        + excursion_d14c
        + jnp.asarray(delta_global, dtype=jnp.float64) * r_ss_ratio
    )
    selected = _select_boxes(d14c_all, box_indices)
    return selected + jnp.asarray(delta_lab, dtype=jnp.float64)


class AbstractSolver(zdx.Base):
    """Abstract base solver for ``ticktack 2.0`` radiocarbon propagation."""

    def solve_delta_trajectory(
        self,
        cbm: CarbonBoxModel,
        production: ProductionTerm,
        t_eval: ArrayLike,
        *,
        q_std: ArrayLike = DEFAULT_Q_STD,
        alpha_unit: ArrayLike = ALPHA_ATOMS_CM2_S_TO_KG_YR,
        t0: ArrayLike | None = None,
        y0: ArrayLike | None = None,
    ) -> jax.Array:
        """Evaluate ``delta_y(t) = y(t) - y_ss`` of shape ``(len(t_eval), n_boxes)``."""
        raise NotImplementedError

    def solve_trajectory(
        self,
        cbm: CarbonBoxModel,
        production: ProductionTerm,
        t_eval: ArrayLike,
        *,
        q_std: ArrayLike = DEFAULT_Q_STD,
        alpha_unit: ArrayLike = ALPHA_ATOMS_CM2_S_TO_KG_YR,
        t0: ArrayLike | None = None,
        y0: ArrayLike | None = None,
    ) -> jax.Array:
        """Evaluate total ``14C`` inventory ``y(t) = y_ss + delta_y(t)`` in ``kg``."""
        y_ss = cbm.steady_state_c14(q_std=q_std, alpha_unit=alpha_unit)
        dy = self.solve_delta_trajectory(
            cbm,
            production,
            t_eval,
            q_std=q_std,
            alpha_unit=alpha_unit,
            t0=t0,
            y0=y0,
        )
        return y_ss[None, :] + dy

    def solve_delta_binned(
        self,
        cbm: CarbonBoxModel,
        production: ProductionTerm,
        t_starts: ArrayLike,
        t_ends: ArrayLike,
        *,
        sub_weights: ArrayLike | None = None,
        q_std: ArrayLike = DEFAULT_Q_STD,
        alpha_unit: ArrayLike = ALPHA_ATOMS_CM2_S_TO_KG_YR,
        t0: ArrayLike | None = None,
        y0: ArrayLike | None = None,
    ) -> jax.Array:
        """Evaluate seasonal-window averaged ``delta_y`` of shape ``(N_r, n_boxes)``."""
        raise NotImplementedError

    def solve_binned(
        self,
        cbm: CarbonBoxModel,
        production: ProductionTerm,
        t_starts: ArrayLike,
        t_ends: ArrayLike,
        *,
        sub_weights: ArrayLike | None = None,
        q_std: ArrayLike = DEFAULT_Q_STD,
        alpha_unit: ArrayLike = ALPHA_ATOMS_CM2_S_TO_KG_YR,
        t0: ArrayLike | None = None,
        y0: ArrayLike | None = None,
    ) -> jax.Array:
        """Evaluate seasonal-window averaged total ``14C`` mass ``y`` (``kg``)."""
        y_ss = cbm.steady_state_c14(q_std=q_std, alpha_unit=alpha_unit)
        dy = self.solve_delta_binned(
            cbm,
            production,
            t_starts,
            t_ends,
            sub_weights=sub_weights,
            q_std=q_std,
            alpha_unit=alpha_unit,
            t0=t0,
            y0=y0,
        )
        return y_ss[None, :] + dy

    def predict_d14c(
        self,
        cbm: CarbonBoxModel,
        production: ProductionTerm,
        t_starts: ArrayLike,
        t_ends: ArrayLike,
        *,
        sub_weights: ArrayLike | None = None,
        box_indices: int | ArrayLike | None = None,
        q_std: ArrayLike = DEFAULT_Q_STD,
        alpha_unit: ArrayLike = ALPHA_ATOMS_CM2_S_TO_KG_YR,
        delta_global: ArrayLike = 0.0,
        delta_lab: ArrayLike = 0.0,
        relative_to: Literal["standard", "ambient"] = "standard",
        t0: ArrayLike | None = None,
        y0: ArrayLike | None = None,
    ) -> jax.Array:
        """Predict seasonal-window ``Delta14C`` (``permil``) normalized to global ``R_ref``."""
        dy_binned = self.solve_delta_binned(
            cbm,
            production,
            t_starts,
            t_ends,
            sub_weights=sub_weights,
            q_std=q_std,
            alpha_unit=alpha_unit,
            t0=t0,
            y0=y0,
        )
        return _delta_y_to_d14c(
            cbm,
            dy_binned,
            box_indices=box_indices,
            q_std=q_std,
            alpha_unit=alpha_unit,
            delta_global=delta_global,
            delta_lab=delta_lab,
            relative_to=relative_to,
        )


class ExactLinearSolver(AbstractSolver):
    """Exact matrix-exponential solver for autonomous linear Carbon Box Models."""

    oversample: int = eqx.field(static=True)
    rfft_threshold: int = eqx.field(static=True)
    foh_method: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        oversample: int = 48,
        rfft_threshold: int = 4096,
        foh_method: Literal["auto", "scan", "rfft"] = "auto",
    ):
        if int(oversample) < 2:
            raise ValueError(f"oversample must be >= 2, got {oversample}.")
        if foh_method not in ("auto", "scan", "rfft"):
            raise ValueError(
                f"foh_method must be 'auto', 'scan', or 'rfft', got {foh_method!r}."
            )
        self.oversample = int(oversample)
        self.rfft_threshold = int(rfft_threshold)
        self.foh_method = str(foh_method)

    def propagation_backend(self, cbm: CarbonBoxModel) -> PropagationBackend:
        """Return the homogeneous/spike propagation backend selected for ``cbm``.

        Parameters
        ----------
        cbm : CarbonBoxModel
            Autonomous carbon box model inspected in the current execution
            context (concrete or JAX-traced).

        Returns
        -------
        {"modal_symmetric", "modal_general", "van_loan"}
            * ``"modal_symmetric"``: real symmetric ``eigh`` similarity
              factorization for detailed-balance models (:class:`EffectiveCBM`,
              :class:`ReversibleCBM`), active both eagerly and under ``jax.jit``.
            * ``"modal_general"``: cached ``float64``/``complex128``
              eigendecomposition for concrete non-symmetric models whose
              eigenvector condition number satisfies ``cond(V) < 1e6`` and
              relative reconstruction error is at most ``1e-9``.
            * ``"van_loan"``: block matrix-exponential (``expm``) fallback used
              when ``cond(V) >= 1e6``, when reconstruction fails, when
              ``cbm.n_boxes == 0``, or when a non-symmetric ``cbm`` is traced
              inside ``jax.jit`` / ``equinox.filter_jit``.

        Notes
        -----
        Under ``jax.jit`` or ``equinox.filter_jit``, ``cbm.matrix_A()`` is a
        JAX tracer: detailed-balance models still select ``"modal_symmetric"``,
        while non-symmetric models select ``"van_loan"`` unless a precomputed
        concrete ``modal`` tuple is passed explicitly to
        :meth:`solve_delta_trajectory` or :meth:`solve_delta_binned`. Calling
        ``solver.propagation_backend(cbm)`` inside traced code never triggers
        concretization errors (``equinox.filter_jit`` can also return the
        Python string directly, whereas raw ``jax.jit`` requires array outputs).
        """
        if _modal_decomposition(cbm.matrix_A(), cbm=cbm) is None:
            return "van_loan"
        if _is_detailed_balance(cbm):
            return "modal_symmetric"
        return "modal_general"

    def _eval_single_term_pointwise(
        self,
        A: jax.Array,
        b_scaled: jax.Array,
        term: ProductionTerm,
        t: jax.Array,
        q_std: ArrayLike,
        *,
        cbm: CarbonBoxModel | None = None,
        modal: tuple[jax.Array, jax.Array, jax.Array] | None = None,
    ) -> jax.Array:
        n = A.shape[0]
        if isinstance(term, SteadyProduction):
            dq = term.rate - jnp.asarray(q_std, dtype=jnp.float64)
            if modal is not None:
                lam_s, v_s, vi_s = modal
                v_ss = jnp.real(v_s @ ((vi_s @ b_scaled.astype(vi_s.dtype)) / (-lam_s)))
            else:
                v_ss = jnp.linalg.solve(-A, b_scaled)
            return jnp.broadcast_to((dq * v_ss)[None, :], (t.shape[0], n))

        if isinstance(term, HarmonicSolar):
            return _eval_harmonic_pointwise(A, b_scaled, term, t)

        if isinstance(term, LinearTrend):
            return _eval_trend_pointwise(A, b_scaled, term, t)

        if isinstance(term, MiyakeSpike):
            if term.profile == "boxcar":
                return _eval_boxcar_pointwise(
                    A, b_scaled, term, t, cbm=cbm, modal=modal
                )
            return _eval_smooth_spike_pointwise(
                A, b_scaled, term, t, cbm=cbm, modal=modal
            )

        if isinstance(term, GPProduction) and term.coefficients is None:
            return jnp.zeros((t.shape[0], n), dtype=jnp.float64)

        # Gridded or continuous FOH terms (GriddedProduction, LogGPProduction, GPProduction w/ coeffs)
        return self._eval_foh_term_pointwise(A, b_scaled, term, t, q_std)

    def _build_foh_grid_for_term(
        self,
        term: ProductionTerm,
        t_min_req: jax.Array,
        t_max_req: jax.Array,
        q_std: ArrayLike,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return ``(t0_grid, dt, delta_q_grid)`` for a general/gridded FOH term."""
        raw_term = term
        if hasattr(term, "to_gridded"):
            term = term.to_gridded(q_std=q_std)

        has_static_uniform_grid = (
            isinstance(getattr(raw_term, "t_start", None), (int, float))
            and isinstance(getattr(raw_term, "t_end", None), (int, float))
            and isinstance(getattr(raw_term, "n_grid", None), int)
            and isinstance(term, GriddedProduction)
        )
        if has_static_uniform_grid:
            t0_u = float(raw_term.t_start)  # type: ignore[attr-defined]
            t1_u = float(raw_term.t_end)  # type: ignore[attr-defined]
            dt_u = (t1_u - t0_u) / float(int(raw_term.n_grid) - 1)  # type: ignore[attr-defined]
            dq_u = (
                term.values
                if term.is_delta
                else term.values - jnp.asarray(q_std, dtype=jnp.float64)
            )
            try:
                tmin_f = float(np.asarray(t_min_req))
                tmax_f = float(np.asarray(t_max_req))
            except (
                jax.errors.TracerArrayConversionError,
                jax.errors.ConcretizationTypeError,
            ):
                return (
                    jnp.asarray(t0_u, dtype=jnp.float64),
                    jnp.asarray(dt_u, dtype=jnp.float64),
                    dq_u,
                )
            if t0_u <= tmin_f + 1e-12 and t1_u >= tmax_f - 1e-12:
                return (
                    jnp.asarray(t0_u, dtype=jnp.float64),
                    jnp.asarray(dt_u, dtype=jnp.float64),
                    dq_u,
                )
            start_f = min(tmin_f, t0_u)
            end_f = max(tmax_f, t1_u)
            dt_f = 1.0 / float(self.oversample)
            n_steps = max(int(math.ceil((end_f - start_f) / dt_f)) + 2, 3)
            grid = start_f + jnp.arange(n_steps, dtype=jnp.float64) * dt_f
            dq = raw_term.delta_q(grid, q_std=q_std)
            return (
                jnp.asarray(start_f, dtype=jnp.float64),
                jnp.asarray(dt_f, dtype=jnp.float64),
                dq,
            )

        if isinstance(term, GriddedProduction):
            tg = term.t_grid
            # Check if tg is concrete and uniform and covers [t_min_req, t_max_req]
            try:
                tg_np = np.asarray(tg, dtype=np.float64)
                tmin_f = float(np.asarray(t_min_req))
                tmax_f = float(np.asarray(t_max_req))
                diffs = np.diff(tg_np)
                dt0 = float(diffs[0])
                if (
                    dt0 > 0.0
                    and np.max(np.abs(diffs - dt0)) <= 1e-11 * dt0
                    and tg_np[0] <= tmin_f + 1e-12
                    and tg_np[-1] >= tmax_f - 1e-12
                ):
                    dq_vals = (
                        term.values
                        if term.is_delta
                        else term.values - jnp.asarray(q_std, dtype=jnp.float64)
                    )
                    return tg[0], jnp.asarray(dt0, dtype=jnp.float64), dq_vals
            except (
                jax.errors.TracerArrayConversionError,
                jax.errors.ConcretizationTypeError,
            ):
                pass

        if isinstance(term, (LogGPProduction, GPProduction)) and term.domain is not None:
            start_f = float(term.domain[0])
            end_f = float(term.domain[1])
        else:
            # Must determine static grid length from concrete bounds
            try:
                tmin_c = float(np.asarray(t_min_req))
                tmax_c = float(np.asarray(t_max_req))
            except (
                jax.errors.TracerArrayConversionError,
                jax.errors.ConcretizationTypeError,
            ) as exc:
                raise ValueError(
                    "FOH grid propagation under JIT requires concrete time-grid bounds "
                    "or an explicit domain=(a, b) on GPProduction / LogGPProduction."
                ) from exc
            if isinstance(term, GriddedProduction):
                tg_c = np.asarray(term.t_grid, dtype=np.float64)
                start_f = min(tmin_c, float(tg_c[0]))
                end_f = max(tmax_c, float(tg_c[-1]))
            else:
                start_f = tmin_c
                end_f = tmax_c

        dt_f = 1.0 / float(self.oversample)
        n_steps = max(int(math.ceil((end_f - start_f) / dt_f)) + 2, 3)
        grid = start_f + jnp.arange(n_steps, dtype=jnp.float64) * dt_f
        dq = term.delta_q(grid, q_std=q_std)
        return (
            jnp.asarray(start_f, dtype=jnp.float64),
            jnp.asarray(dt_f, dtype=jnp.float64),
            dq,
        )

    def _eval_foh_term_pointwise(
        self,
        A: jax.Array,
        b_scaled: jax.Array,
        term: ProductionTerm,
        t: jax.Array,
        q_std: ArrayLike,
    ) -> jax.Array:
        t0_g, dt_g, dq_g = self._build_foh_grid_for_term(
            term, jnp.min(t), jnp.max(t), q_std
        )
        dy0 = jnp.zeros((A.shape[0],), dtype=jnp.float64)
        dy_q, _ = _foh_cumulative_integral(
            A,
            b_scaled,
            t0_g,
            dt_g,
            dq_g,
            dy0,
            t,
            method=self.foh_method,  # type: ignore[arg-type]
            rfft_threshold=self.rfft_threshold,
        )
        return dy_q

    def _eval_foh_term_binned(
        self,
        A: jax.Array,
        b_scaled: jax.Array,
        term: ProductionTerm,
        a_r: jax.Array,
        b_r: jax.Array,
        q_std: ArrayLike,
    ) -> jax.Array:
        t0_g, dt_g, dq_g = self._build_foh_grid_for_term(
            term, jnp.min(a_r), jnp.max(b_r), q_std
        )
        dy0 = jnp.zeros((A.shape[0],), dtype=jnp.float64)
        _, Y_a = _foh_cumulative_integral(
            A,
            b_scaled,
            t0_g,
            dt_g,
            dq_g,
            dy0,
            a_r,
            method=self.foh_method,  # type: ignore[arg-type]
            rfft_threshold=self.rfft_threshold,
        )
        _, Y_b = _foh_cumulative_integral(
            A,
            b_scaled,
            t0_g,
            dt_g,
            dq_g,
            dy0,
            b_r,
            method=self.foh_method,  # type: ignore[arg-type]
            rfft_threshold=self.rfft_threshold,
        )
        return (Y_b - Y_a) / (b_r - a_r)[:, None]

    def _eval_single_term_binned(
        self,
        A: jax.Array,
        b_scaled: jax.Array,
        term: ProductionTerm,
        a_r: jax.Array,
        b_r: jax.Array,
        q_std: ArrayLike,
        *,
        cbm: CarbonBoxModel | None = None,
        modal: tuple[jax.Array, jax.Array, jax.Array] | None = None,
    ) -> jax.Array:
        n = A.shape[0]
        if isinstance(term, SteadyProduction):
            dq = term.rate - jnp.asarray(q_std, dtype=jnp.float64)
            if modal is not None:
                lam_s, v_s, vi_s = modal
                v_ss = jnp.real(v_s @ ((vi_s @ b_scaled.astype(vi_s.dtype)) / (-lam_s)))
            else:
                v_ss = jnp.linalg.solve(-A, b_scaled)
            return jnp.broadcast_to((dq * v_ss)[None, :], (a_r.shape[0], n))

        if isinstance(term, HarmonicSolar):
            return _eval_harmonic_binned(A, b_scaled, term, a_r, b_r)

        if isinstance(term, LinearTrend):
            return _eval_trend_binned(A, b_scaled, term, a_r, b_r)

        if isinstance(term, MiyakeSpike):
            if term.profile == "boxcar":
                return _eval_boxcar_binned(
                    A, b_scaled, term, a_r, b_r, cbm=cbm, modal=modal
                )
            return _eval_smooth_spike_binned(
                A, b_scaled, term, a_r, b_r, cbm=cbm, modal=modal
            )

        if isinstance(term, GPProduction) and term.coefficients is None:
            return jnp.zeros((a_r.shape[0], n), dtype=jnp.float64)

        return self._eval_foh_term_binned(A, b_scaled, term, a_r, b_r, q_std)

    def solve_delta_trajectory(
        self,
        cbm: CarbonBoxModel,
        production: ProductionTerm,
        t_eval: ArrayLike,
        *,
        q_std: ArrayLike = DEFAULT_Q_STD,
        alpha_unit: ArrayLike = ALPHA_ATOMS_CM2_S_TO_KG_YR,
        t0: ArrayLike | None = None,
        y0: ArrayLike | None = None,
        modal: tuple[jax.Array, jax.Array, jax.Array] | None = None,
    ) -> jax.Array:
        t_arr = jnp.atleast_1d(jnp.asarray(t_eval, dtype=jnp.float64))
        A = cbm.matrix_A()
        b_scaled = jnp.asarray(alpha_unit, dtype=jnp.float64) * cbm.production_vector()
        terms = _unpack_terms(production)
        if modal is None:
            needs_modal = y0 is not None or any(
                isinstance(tm, (MiyakeSpike, SteadyProduction)) for tm in terms
            )
            modal = _modal_decomposition(A, cbm=cbm) if needs_modal else None

        total = jnp.zeros((t_arr.shape[0], cbm.n_boxes), dtype=jnp.float64)
        for term in terms:
            total = total + self._eval_single_term_pointwise(
                A, b_scaled, term, t_arr, q_std, cbm=cbm, modal=modal
            )

        if y0 is not None:
            t_init = t_arr[0] if t0 is None else jnp.asarray(t0, dtype=jnp.float64)
            y_ss = cbm.steady_state_c14(q_std=q_std, alpha_unit=alpha_unit)
            dy_target0 = jnp.asarray(y0, dtype=jnp.float64) - y_ss
            dy_part0 = jnp.zeros((cbm.n_boxes,), dtype=jnp.float64)
            t_init_1d = jnp.reshape(t_init, (1,))
            for term in terms:
                dy_part0 = dy_part0 + self._eval_single_term_pointwise(
                    A, b_scaled, term, t_init_1d, q_std, cbm=cbm, modal=modal
                )[0]
            diff0 = dy_target0 - dy_part0
            if modal is not None:
                lam, v_mat, v_inv = modal
                d_tilde = v_inv @ diff0.astype(v_inv.dtype)
                hom_tilde = (
                    jnp.exp((t_arr - t_init)[:, None] * lam[None, :])
                    * d_tilde[None, :]
                )
                hom = jnp.real(hom_tilde @ v_mat.T)
            else:
                hom = jax.vmap(
                    lambda ti: jax.scipy.linalg.expm(A * (ti - t_init)) @ diff0
                )(t_arr)
            total = total + hom

        return total

    def solve_delta_binned(
        self,
        cbm: CarbonBoxModel,
        production: ProductionTerm,
        t_starts: ArrayLike,
        t_ends: ArrayLike,
        *,
        sub_weights: ArrayLike | None = None,
        q_std: ArrayLike = DEFAULT_Q_STD,
        alpha_unit: ArrayLike = ALPHA_ATOMS_CM2_S_TO_KG_YR,
        t0: ArrayLike | None = None,
        y0: ArrayLike | None = None,
        modal: tuple[jax.Array, jax.Array, jax.Array] | None = None,
    ) -> jax.Array:
        a_flat, b_flat, w_2d, is_1d, orig_shape = _flatten_windows(
            t_starts, t_ends, sub_weights
        )
        A = cbm.matrix_A()
        b_scaled = jnp.asarray(alpha_unit, dtype=jnp.float64) * cbm.production_vector()
        terms = _unpack_terms(production)
        if modal is None:
            needs_modal = y0 is not None or any(
                isinstance(tm, (MiyakeSpike, SteadyProduction)) for tm in terms
            )
            modal = _modal_decomposition(A, cbm=cbm) if needs_modal else None

        total = jnp.zeros((a_flat.shape[0], cbm.n_boxes), dtype=jnp.float64)
        for term in terms:
            total = total + self._eval_single_term_binned(
                A, b_scaled, term, a_flat, b_flat, q_std, cbm=cbm, modal=modal
            )

        if y0 is not None:
            t_init = (
                jnp.min(a_flat) if t0 is None else jnp.asarray(t0, dtype=jnp.float64)
            )
            y_ss = cbm.steady_state_c14(q_std=q_std, alpha_unit=alpha_unit)
            dy_target0 = jnp.asarray(y0, dtype=jnp.float64) - y_ss
            dy_part0 = jnp.zeros((cbm.n_boxes,), dtype=jnp.float64)
            t_init_1d = jnp.reshape(t_init, (1,))
            for term in terms:
                dy_part0 = dy_part0 + self._eval_single_term_pointwise(
                    A, b_scaled, term, t_init_1d, q_std, cbm=cbm, modal=modal
                )[0]
            diff0 = dy_target0 - dy_part0

            if modal is not None:
                lam, v_mat, v_inv = modal
                d_tilde = v_inv @ diff0.astype(v_inv.dtype)
                widths = b_flat - a_flat
                hom_tilde = (
                    _scalar_psi1(widths[:, None] * lam[None, :])
                    * jnp.exp((a_flat - t_init)[:, None] * lam[None, :])
                    * d_tilde[None, :]
                )
                total = total + jnp.real(hom_tilde @ v_mat.T)
            else:
                def _hom_win(ar: jax.Array, br: jax.Array) -> jax.Array:
                    state_ar = jax.scipy.linalg.expm(A * (ar - t_init)) @ diff0
                    _, int_h = van_loan_psi1_vec(A, state_ar, br - ar)
                    return int_h / (br - ar)

                total = total + jax.vmap(_hom_win)(a_flat, b_flat)

        if is_1d:
            return total
        assert w_2d is not None
        n_r, m_w = orig_shape
        reshaped = total.reshape((n_r, m_w, cbm.n_boxes))
        return jnp.sum(reshaped * w_2d[:, :, None], axis=1)


def _cbm_has_nonautonomous_masses(cbm: CarbonBoxModel) -> bool:
    """Return ``True`` if ``cbm`` overrides ``reservoir_masses_at`` or ``matrix_A_at``."""
    return (
        type(cbm).reservoir_masses_at is not CarbonBoxModel.reservoir_masses_at
        or type(cbm).matrix_A_at is not CarbonBoxModel.matrix_A_at
    )


def _filter_event_timestamps(
    ts_list: list[jax.Array],
    t_start_int: jax.Array,
    t_end_int: jax.Array,
    *,
    stop_grad: bool = False,
) -> jax.Array | None:
    """Filter or mask solver step/jump timestamps strictly to ``(t_start_int, t_end_int)``.

    When ``stop_grad=True`` (used for continuous ``step_ts`` knots), gradients
    are stopped *after* the traced/concrete branch decision so eager and JIT
    reverse-mode runs follow the identical masking path while preventing
    ``PIDController`` from differentiating through internal mesh-clipping times.
    """
    if not ts_list:
        return None
    flat = jnp.concatenate([jnp.atleast_1d(s).reshape(-1) for s in ts_list])
    if flat.size == 0:
        return None
    if (
        not isinstance(flat, jax.core.Tracer)
        and not isinstance(t_start_int, jax.core.Tracer)
        and not isinstance(t_end_int, jax.core.Tracer)
    ):
        flat_np = np.asarray(flat, dtype=np.float64)
        lo_f = float(t_start_int)
        hi_f = float(t_end_int)
        kept = np.unique(flat_np[(flat_np > lo_f) & (flat_np < hi_f)])
        if kept.size == 0:
            return None
        return jnp.asarray(kept, dtype=jnp.float64)

    # Under JAX tracing (static shapes required): map <= t_start_int to -inf and
    # >= t_end_int to +inf, which diffrax.ClipStepSizeController ignores via
    # searchsorted(..., side="right") and minimum(_t, t1).
    masked = jnp.where(
        flat <= t_start_int,
        -jnp.inf,
        jnp.where(flat >= t_end_int, jnp.inf, flat),
    )
    out = jnp.sort(masked)
    return jax.lax.stop_gradient(out) if stop_grad else out


def _collect_initial_time_and_events(
    terms: tuple[ProductionTerm, ...],
    earliest_obs: jax.Array,
    latest_obs: jax.Array,
    t0_override: ArrayLike | None,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None]:
    """Determine integration start ``t0``, continuous ``step_ts``, and discontinuous ``jump_ts``."""
    t0_candidates: list[jax.Array] = [earliest_obs]
    step_list: list[jax.Array] = []
    jump_list: list[jax.Array] = []

    for term in terms:
        if isinstance(term, MiyakeSpike):
            panels = term.unit_support_panels()
            u_min = panels[0][0]
            dur = jnp.maximum(term.duration, 0.0)
            onset = term.t_start + dur * u_min - 0.02
            t0_candidates.append(onset)
            if term.profile == "boxcar":
                jump_list.extend([term.t_start, term.t_start + dur])
            elif term.profile == "gamma":
                if term.gamma_shape == 2.0:
                    # C^0 onset with finite slope jump at u=0; jump past t_start without
                    # differentiating the mesh boundary.
                    jump_list.append(jax.lax.stop_gradient(term.t_start))
                else:
                    step_list.append(term.t_start)
                for _, u_hi in panels[:-1]:
                    step_list.append(term.t_start + dur * u_hi)
            else:
                for u_lo, _ in panels:
                    step_list.append(term.t_start + dur * u_lo)
                step_list.append(term.t_start + dur * panels[-1][1])
        elif isinstance(term, LinearTrend) and term.switched_on:
            tr = jnp.asarray(term.t_ref, dtype=jnp.float64)
            t0_candidates.append(tr)
            step_list.append(tr)
        elif isinstance(term, GriddedProduction):
            t0_candidates.append(term.t_grid[0])
            # Piecewise-linear jnp.interp has slope kinks at t_grid knots; step past them
            # via jump_ts with stopped gradients (continuous value -> zero Leibniz jump).
            jump_list.append(jax.lax.stop_gradient(term.t_grid))
        elif isinstance(term, (LogGPProduction, GPProduction)) and term.domain is not None:
            t0_candidates.append(jnp.asarray(term.domain[0], dtype=jnp.float64))

    if t0_override is not None:
        t_start_int = jnp.asarray(t0_override, dtype=jnp.float64)
        t_start_out = t_start_int
    else:
        t_start_int = jnp.min(jnp.stack([jnp.reshape(c, ()) for c in t0_candidates]))
        # Automatic integration start is an auxiliary pre-onset epoch; stop its
        # gradient so PIDController does not amplify d(t0)/dtheta across steps.
        t_start_out = jax.lax.stop_gradient(t_start_int)

    t_end_int = jnp.maximum(jnp.reshape(latest_obs, ()), t_start_int + 1e-9)
    # Stop gradients on continuous step_ts (zero Leibniz boundary term), while
    # keeping discontinuous jump_ts differentiable for boxcar edges.
    step_ts = _filter_event_timestamps(
        step_list, t_start_int, t_end_int, stop_grad=True
    )
    jump_ts = _filter_event_timestamps(jump_list, t_start_int, t_end_int)
    return t_start_out, step_ts, jump_ts


def _equilibrium_dy0_and_drift_anchor(
    A0: jax.Array,
    b_scaled: jax.Array,
    terms: tuple[ProductionTerm, ...],
    t_start_int: jax.Array,
    q_std: ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """Compute eternal background state ``dy(t_start_int)`` and secular drift anchor."""
    dy0 = jnp.zeros((A0.shape[0],), dtype=jnp.float64)
    secular_rhs_anchor = jnp.zeros((A0.shape[0],), dtype=jnp.float64)
    v_ss = jnp.linalg.solve(-A0, b_scaled)
    t1d = jnp.reshape(t_start_int, (1,))

    for term in terms:
        if isinstance(term, SteadyProduction):
            dq = term.rate - jnp.asarray(q_std, dtype=jnp.float64)
            dy0 = dy0 + dq * v_ss
        elif isinstance(term, HarmonicSolar):
            dy0 = dy0 + _eval_harmonic_pointwise(A0, b_scaled, term, t1d)[0]
        elif isinstance(term, LinearTrend) and not term.switched_on:
            dy0 = dy0 + term.slope * (t_start_int - term.t_ref) * v_ss
            secular_rhs_anchor = secular_rhs_anchor + term.slope * v_ss

    return dy0, secular_rhs_anchor


class _ODEIVPSetup(NamedTuple):
    """Sorted save times, integration bracket, events, initial state, and RHS args."""

    ts_sorted: jax.Array
    inv_order: jax.Array
    t_start: jax.Array
    t_end: jax.Array
    step_ts: jax.Array | None
    jump_ts: jax.Array | None
    dy_init: jax.Array
    rhs_args: tuple[Any, ...]


def _prepare_ode_ivp(
    cbm: CarbonBoxModel, production: ProductionTerm, ts: jax.Array, *,
    q_std: ArrayLike, alpha_unit: ArrayLike, t0: ArrayLike | None = None,
    y0: ArrayLike | None = None, deduplicate: bool = False,
) -> _ODEIVPSetup:
    """Sort timestamps, bracket integration horizon/events, and resolve initial state."""
    if deduplicate and not isinstance(ts, jax.core.Tracer):
        ts_sorted, inv_order = jnp.unique(ts, return_inverse=True)
    else:
        order = jnp.argsort(ts)
        inv_order, ts_sorted = jnp.argsort(order), ts[order]

    A0 = cbm.matrix_A()
    b_scaled = jnp.asarray(alpha_unit, dtype=jnp.float64) * cbm.production_vector()
    y_ss = cbm.steady_state_c14(q_std=q_std, alpha_unit=alpha_unit)
    terms = _unpack_terms(production)

    eff_t0 = ts_sorted[0] if (t0 is None and y0 is not None) else t0
    t_start, step_ts, jump_ts = _collect_initial_time_and_events(
        terms, ts_sorted[0], ts_sorted[-1], eff_t0)
    t_end = jnp.maximum(ts_sorted[-1], t_start + 1e-9)

    dy_init, secular_anchor = _equilibrium_dy0_and_drift_anchor(A0, b_scaled, terms, t_start, q_std)
    if y0 is not None:
        dy_init = jnp.asarray(y0, dtype=jnp.float64) - y_ss
        secular_anchor = jnp.zeros_like(secular_anchor)

    rhs_args = (cbm, production, b_scaled, y_ss, A0, secular_anchor, q_std)
    return _ODEIVPSetup(ts_sorted, inv_order, t_start, t_end, step_ts, jump_ts, dy_init, rhs_args)


def _eval_ode_ddy(
    a_at_fn: Callable[..., jax.Array], t: jax.Array, dy: jax.Array, rhs_args: tuple[Any, ...]
) -> jax.Array:
    """Evaluate ``d(delta_y)/dt`` for autonomous or non-autonomous ``A(t)``."""
    cbm_arg, prod_arg, b_sc, y_ss_arg, A0_arg, sec_anc, q_s = rhs_args
    At = a_at_fn(cbm_arg, t)
    return At @ dy + (At - A0_arg) @ y_ss_arg + b_sc * prod_arg.delta_q(t, q_std=q_s) + sec_anc


def _combine_subwindows(values: jax.Array, subwindow_meta: tuple[Any, ...]) -> jax.Array:
    """Combine 1D or weighted 2D ``(N_r, M_w)`` seasonal sub-window values to ``(N_r, n)``."""
    w_2d, is_1d, orig_shape = subwindow_meta
    if is_1d:
        return values
    assert w_2d is not None
    return jnp.sum(values.reshape((*orig_shape, values.shape[-1])) * w_2d[:, :, None], axis=1)


def _unpack_cumulative_windows(
    cum_sorted: jax.Array, inv_order: jax.Array, a_flat: jax.Array, b_flat: jax.Array,
    subwindow_meta: tuple[Any, ...],
) -> jax.Array:
    """Convert sorted cumulative-integral boundary states to seasonal-window means."""
    cum_all, n_w = cum_sorted[inv_order], a_flat.shape[0]
    win_avg = (cum_all[n_w:] - cum_all[:n_w]) / (b_flat - a_flat)[:, None]
    return _combine_subwindows(win_avg, subwindow_meta)


def _gl16_window_nodes(a_flat: jax.Array, b_flat: jax.Array) -> jax.Array:
    """Map 16-point Gauss-Legendre nodes onto flattened seasonal windows ``[a, b]``."""
    xi = jnp.asarray(GL16_NODES_01, dtype=jnp.float64)
    return (a_flat[:, None] + (b_flat - a_flat)[:, None] * xi[None, :]).reshape(-1)


def _gl16_reduce_windows(
    quad_vals: jax.Array, n_w: int, subwindow_meta: tuple[Any, ...]
) -> jax.Array:
    """Integrate ``(N_w * 16, n)`` GL16 node values into seasonal-window means."""
    wq = jnp.asarray(GL16_WEIGHTS_01, dtype=jnp.float64)
    vals_3d = quad_vals.reshape((n_w, wq.shape[0], quad_vals.shape[-1]))
    return _combine_subwindows(jnp.einsum("wqn,q->wn", vals_3d, wq), subwindow_meta)


def _prepare_suess_params(
    cbm: CarbonBoxModel, q_std: ArrayLike, alpha_unit: ArrayLike,
    delta_global: ArrayLike, relative_to: Literal["standard", "ambient"],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute ``(y_base, excursion_scale, r_ref)`` for time-varying ``M_c(t)`` ``Delta14C``."""
    y_ss = cbm.steady_state_c14(q_std=q_std, alpha_unit=alpha_unit)
    r_ref = cbm.reference_specific_activity(q_std=q_std, alpha_unit=alpha_unit)
    scale_amb = 1.0 + jnp.asarray(delta_global, dtype=jnp.float64) / 1000.0
    excursion_scale = scale_amb if relative_to == "ambient" else jnp.asarray(1.0, dtype=jnp.float64)
    return y_ss * scale_amb, excursion_scale, r_ref


def _suess_instantaneous_d14c(
    dy: jax.Array, masses: jax.Array, suess_params: tuple[jax.Array, jax.Array, jax.Array]
) -> jax.Array:
    """Evaluate instantaneous ``Delta14C`` (``permil``) under time-varying ``M_c(t)``."""
    y_base, excursion_scale, r_ref = suess_params
    return 1000.0 * (((y_base + excursion_scale * dy) / masses) / r_ref - 1.0)


class DiffraxSolver(AbstractSolver):
    """General adaptive ODE solver backed by ``diffrax`` (``Tsit5`` or ``Kvaerno5``).

    Supports both autonomous linear CBMs (serving as an independent ``1e-13``
    reference against :class:`ExactLinearSolver`) and non-autonomous time-varying
    reservoir masses ``M_c(t)`` / transition matrices ``A(t)`` (e.g. industrial
    fossil-fuel Suess dilution).
    """

    method: str = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    atol: float = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    dt0: float | None = eqx.field(static=True)
    masses_fn: Callable[[jax.Array], jax.Array] | None = eqx.field(static=True)
    matrix_A_fn: Callable[[jax.Array], jax.Array] | None = eqx.field(static=True)

    def __init__(
        self,
        method: Literal["Tsit5", "Kvaerno5", "Dopri5", "Dopri8"] = "Tsit5",
        *,
        rtol: float = 1e-10,
        atol: float = 1e-10,
        max_steps: int = 262_144,
        dt0: float | None = None,
        masses_fn: Callable[[jax.Array], jax.Array] | None = None,
        matrix_A_fn: Callable[[jax.Array], jax.Array] | None = None,
    ):
        if method not in ("Tsit5", "Kvaerno5", "Dopri5", "Dopri8"):
            raise ValueError(
                f"Unsupported DiffraxSolver method {method!r}; choose 'Tsit5', 'Kvaerno5', 'Dopri5', or 'Dopri8'."
            )
        self.method = str(method)
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.max_steps = int(max_steps)
        self.dt0 = None if dt0 is None else float(dt0)
        self.masses_fn = masses_fn
        self.matrix_A_fn = matrix_A_fn

    def _make_diffrax_solver(self) -> diffrax.AbstractSolver:
        if self.method == "Tsit5":
            return diffrax.Tsit5()
        if self.method == "Kvaerno5":
            return diffrax.Kvaerno5()
        if self.method == "Dopri5":
            return diffrax.Dopri5()
        if self.method == "Dopri8":
            return diffrax.Dopri8()
        raise RuntimeError(f"Unhandled solver method {self.method!r}")

    def _run_diffeqsolve(
        self, rhs: Callable[..., Any], ivp: _ODEIVPSetup, y0_state: Any, args: Any
    ) -> Any:
        return diffrax.diffeqsolve(
            diffrax.ODETerm(rhs), self._make_diffrax_solver(),
            t0=ivp.t_start, t1=ivp.t_end, dt0=self.dt0, y0=y0_state, args=args,
            saveat=diffrax.SaveAt(ts=ivp.ts_sorted), max_steps=self.max_steps,
            stepsize_controller=diffrax.PIDController(
                rtol=self.rtol, atol=self.atol, step_ts=ivp.step_ts, jump_ts=ivp.jump_ts),
        ).ys

    def _A_at(self, cbm: CarbonBoxModel, t: jax.Array) -> jax.Array:
        if self.matrix_A_fn is not None:
            return self.matrix_A_fn(t)
        return cbm.matrix_A_at(t)

    def _masses_at(self, cbm: CarbonBoxModel, t: jax.Array) -> jax.Array:
        if self.masses_fn is not None:
            return self.masses_fn(t)
        return cbm.reservoir_masses_at(t)

    @property
    def has_time_varying_masses(self) -> bool:
        return self.masses_fn is not None or self.matrix_A_fn is not None

    def solve_delta_trajectory(
        self,
        cbm: CarbonBoxModel,
        production: ProductionTerm,
        t_eval: ArrayLike,
        *,
        q_std: ArrayLike = DEFAULT_Q_STD,
        alpha_unit: ArrayLike = ALPHA_ATOMS_CM2_S_TO_KG_YR,
        t0: ArrayLike | None = None,
        y0: ArrayLike | None = None,
    ) -> jax.Array:
        t_arr = jnp.atleast_1d(jnp.asarray(t_eval, dtype=jnp.float64))
        ivp = _prepare_ode_ivp(
            cbm, production, t_arr, q_std=q_std, alpha_unit=alpha_unit, t0=t0, y0=y0)

        def _rhs(t: jax.Array, dy: jax.Array, args: Any) -> jax.Array:
            return _eval_ode_ddy(self._A_at, t, dy, args)

        return self._run_diffeqsolve(_rhs, ivp, ivp.dy_init, ivp.rhs_args)[ivp.inv_order]

    def solve_delta_binned(
        self,
        cbm: CarbonBoxModel,
        production: ProductionTerm,
        t_starts: ArrayLike,
        t_ends: ArrayLike,
        *,
        sub_weights: ArrayLike | None = None,
        q_std: ArrayLike = DEFAULT_Q_STD,
        alpha_unit: ArrayLike = ALPHA_ATOMS_CM2_S_TO_KG_YR,
        t0: ArrayLike | None = None,
        y0: ArrayLike | None = None,
    ) -> jax.Array:
        a_flat, b_flat, w_2d, is_1d, orig_shape = _flatten_windows(
            t_starts, t_ends, sub_weights
        )
        all_bounds = jnp.concatenate([a_flat, b_flat])
        ivp = _prepare_ode_ivp(
            cbm, production, all_bounds, q_std=q_std, alpha_unit=alpha_unit, t0=t0, y0=y0)

        def _aug_rhs(
            t: jax.Array, state: tuple[jax.Array, jax.Array], args: Any
        ) -> tuple[jax.Array, jax.Array]:
            dy, _ = state
            return _eval_ode_ddy(self._A_at, t, dy, args), dy

        _, Y_sorted = self._run_diffeqsolve(
            _aug_rhs, ivp, (ivp.dy_init, jnp.zeros_like(ivp.dy_init)), ivp.rhs_args)
        return _unpack_cumulative_windows(
            Y_sorted, ivp.inv_order, a_flat, b_flat, (w_2d, is_1d, orig_shape))

    def predict_d14c(
        self,
        cbm: CarbonBoxModel,
        production: ProductionTerm,
        t_starts: ArrayLike,
        t_ends: ArrayLike,
        *,
        sub_weights: ArrayLike | None = None,
        box_indices: int | ArrayLike | None = None,
        q_std: ArrayLike = DEFAULT_Q_STD,
        alpha_unit: ArrayLike = ALPHA_ATOMS_CM2_S_TO_KG_YR,
        delta_global: ArrayLike = 0.0,
        delta_lab: ArrayLike = 0.0,
        relative_to: Literal["standard", "ambient"] = "standard",
        t0: ArrayLike | None = None,
        y0: ArrayLike | None = None,
    ) -> jax.Array:
        if relative_to not in ("standard", "ambient"):
            raise ValueError(
                f"relative_to must be 'standard' or 'ambient'; got {relative_to!r}."
            )
        if not (self.has_time_varying_masses or _cbm_has_nonautonomous_masses(cbm)):
            return super().predict_d14c(
                cbm,
                production,
                t_starts,
                t_ends,
                sub_weights=sub_weights,
                box_indices=box_indices,
                q_std=q_std,
                alpha_unit=alpha_unit,
                delta_global=delta_global,
                delta_lab=delta_lab,
                relative_to=relative_to,
                t0=t0,
                y0=y0,
            )

        # Time-varying M_c(t) (e.g. fossil-fuel 12C Suess dilution):
        # integrate instantaneous specific activity normalized by M_c(t) * R_ref,
        # diluting both pre-industrial ambient offset (delta_global * y_ss / 1000)
        # and dynamical excursion delta_y(t) consistently by M_c(t).
        a_flat, b_flat, w_2d, is_1d, orig_shape = _flatten_windows(
            t_starts, t_ends, sub_weights
        )
        all_bounds = jnp.concatenate([a_flat, b_flat])
        ivp = _prepare_ode_ivp(
            cbm, production, all_bounds, q_std=q_std, alpha_unit=alpha_unit, t0=t0, y0=y0)
        suess_params = _prepare_suess_params(cbm, q_std, alpha_unit, delta_global, relative_to)

        def _suess_rhs(
            t: jax.Array, state: tuple[jax.Array, jax.Array], args: Any
        ) -> tuple[jax.Array, jax.Array]:
            dy, _ = state
            rhs_args, s_params = args
            ddy = _eval_ode_ddy(self._A_at, t, dy, rhs_args)
            inst_d14c = _suess_instantaneous_d14c(dy, self._masses_at(rhs_args[0], t), s_params)
            return ddy, inst_d14c

        _, R_sorted = self._run_diffeqsolve(
            _suess_rhs, ivp, (ivp.dy_init, jnp.zeros_like(ivp.dy_init)), (ivp.rhs_args, suess_params))
        d14c_win = _unpack_cumulative_windows(
            R_sorted, ivp.inv_order, a_flat, b_flat, (w_2d, is_1d, orig_shape))

        selected = _select_boxes(d14c_win, box_indices)
        return selected + jnp.asarray(delta_lab, dtype=jnp.float64)


def wrap_jvp_only_scalar_fn(
    scalar_fn: Callable[[Any], jax.Array],
) -> Callable[[Any], jax.Array]:
    """Wrap a JVP-only scalar function (e.g. ``ModaxSolver`` log-likelihood) with ``@jax.custom_vjp``.

    ``modax-solvers`` CUDA FFI kernels expose forward sensitivity (``jvp``) only,
    which otherwise fails inside ``jax.value_and_grad`` in ``NumPyro`` NUTS and
    ``BlackJAX`` HMC-mutated SMC. This wrapper implements the reverse-mode VJP by
    pushing forward an identity tangent basis in a single ``vmap(jvp)`` pass,
    evaluating the primal once during ``value_and_grad`` and supporting outer ``vmap``.
    """

    @jax.custom_vjp
    def _wrapped(params: Any) -> jax.Array:
        return jnp.reshape(scalar_fn(params), ())

    def _wrapped_fwd(params: Any) -> tuple[jax.Array, Any]:
        flat_p, unravel = ravel_pytree(params)
        dim = flat_p.shape[0]

        def _flat_fn(v: jax.Array) -> jax.Array:
            return jnp.reshape(scalar_fn(unravel(v)), ())

        if dim == 0:
            val = _flat_fn(flat_p)
            return val, params

        basis = jnp.eye(dim, dtype=flat_p.dtype)

        def _pushfwd(tangent: jax.Array) -> tuple[jax.Array, jax.Array]:
            return jax.jvp(_flat_fn, (flat_p,), (tangent,))

        val, flat_grad = jax.vmap(_pushfwd, out_axes=(None, 0))(basis)
        return val, unravel(flat_grad)

    def _wrapped_bwd(grad_tree: Any, cotangent: jax.Array) -> tuple[Any]:
        cot = jnp.asarray(cotangent)
        scaled = jax.tree_util.tree_map(lambda g: cot * g, grad_tree)
        return (scaled,)

    _wrapped.defvjp(_wrapped_fwd, _wrapped_bwd)
    return _wrapped


def _load_modax_module() -> Any:
    """Lazily import ``modax`` (from optional ``ticktack[cuda]`` extra ``modax-solvers``)."""
    try:
        import modax  # type: ignore[import-not-found]

        return modax
    except ImportError as exc:
        raise ImportError(
            "ModaxSolver requires the optional CUDA package 'modax-solvers' "
            "(Berry et al. 2026, arXiv:2609.35661). Install via "
            "`pip install 'ticktack[cuda]'` on a CUDA-enabled machine, or use "
            "`ExactLinearSolver` / `DiffraxSolver`."
        ) from exc


class ModaxSolver(AbstractSolver):
    """Optional GPU-accelerated ODE solver backed by ``modax-solvers`` (``Tsit5`` / ``Rodas5P``).

    Lazily imports ``modax`` at solve time so ``ticktack`` runs cleanly on CPU and
    Apple Silicon without CUDA, and provides :meth:`wrap_log_likelihood` to equip
    per-particle scalar log-likelihoods with a ``@jax.custom_vjp`` rule backed by
    forward-mode ``jvp`` / ``jacfwd``.
    """

    method: str = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    atol: float = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    masses_fn: Callable[[jax.Array], jax.Array] | None = eqx.field(static=True)
    matrix_A_fn: Callable[[jax.Array], jax.Array] | None = eqx.field(static=True)
    _custom_backend_fn: Callable[..., jax.Array] | None = eqx.field(static=True)

    def __init__(
        self,
        method: Literal["Rodas5P", "Tsit5"] = "Rodas5P",
        *,
        rtol: float = 1e-8,
        atol: float = 1e-8,
        max_steps: int = 100_000,
        masses_fn: Callable[[jax.Array], jax.Array] | None = None,
        matrix_A_fn: Callable[[jax.Array], jax.Array] | None = None,
        backend_fn: Callable[..., jax.Array] | None = None,
    ):
        if method not in ("Rodas5P", "Tsit5"):
            raise ValueError(
                f"ModaxSolver method must be 'Rodas5P' or 'Tsit5', got {method!r}."
            )
        self.method = str(method)
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.max_steps = int(max_steps)
        self.masses_fn = masses_fn
        self.matrix_A_fn = matrix_A_fn
        self._custom_backend_fn = backend_fn

    def _A_at(self, cbm: CarbonBoxModel, t: jax.Array) -> jax.Array:
        if self.matrix_A_fn is not None:
            return self.matrix_A_fn(t)
        return cbm.matrix_A_at(t)

    def _masses_at(self, cbm: CarbonBoxModel, t: jax.Array) -> jax.Array:
        if self.masses_fn is not None:
            return self.masses_fn(t)
        return cbm.reservoir_masses_at(t)

    @property
    def has_time_varying_masses(self) -> bool:
        return self.masses_fn is not None or self.matrix_A_fn is not None

    @staticmethod
    def wrap_log_likelihood(
        log_prob_fn: Callable[[Any], jax.Array],
    ) -> Callable[[Any], jax.Array]:
        """Equip a scalar ``log_prob_fn(params)`` using ``ModaxSolver`` with reverse-mode VJP."""
        return wrap_jvp_only_scalar_fn(log_prob_fn)

    def solve_delta_trajectory(
        self,
        cbm: CarbonBoxModel,
        production: ProductionTerm,
        t_eval: ArrayLike,
        *,
        q_std: ArrayLike = DEFAULT_Q_STD,
        alpha_unit: ArrayLike = ALPHA_ATOMS_CM2_S_TO_KG_YR,
        t0: ArrayLike | None = None,
        y0: ArrayLike | None = None,
    ) -> jax.Array:
        t_arr = jnp.atleast_1d(jnp.asarray(t_eval, dtype=jnp.float64)).reshape(-1)
        if self._custom_backend_fn is not None:
            return self._custom_backend_fn(
                cbm,
                production,
                t_arr,
                q_std=q_std,
                alpha_unit=alpha_unit,
                t0=t0,
                y0=y0,
            )

        modax = _load_modax_module()
        ivp = _prepare_ode_ivp(
            cbm, production, t_arr, q_std=q_std, alpha_unit=alpha_unit, t0=t0, y0=y0, deduplicate=True
        )
        solver_cls = getattr(modax, self.method)

        def _rhs(t: jax.Array, dy: jax.Array, args: Any) -> jax.Array:
            return _eval_ode_ddy(self._A_at, t, dy, args)

        sol = modax.diffeqsolve(
            _rhs,
            solver_cls(),
            t0=ivp.t_start,
            t1=ivp.t_end,
            y0=ivp.dy_init,
            ts=ivp.ts_sorted,
            args=ivp.rhs_args,
            rtol=self.rtol,
            atol=self.atol,
            max_steps=self.max_steps,
        )
        ys_sorted = getattr(sol, "ys", sol)
        return ys_sorted[ivp.inv_order]

    def solve_delta_binned(
        self,
        cbm: CarbonBoxModel,
        production: ProductionTerm,
        t_starts: ArrayLike,
        t_ends: ArrayLike,
        *,
        sub_weights: ArrayLike | None = None,
        q_std: ArrayLike = DEFAULT_Q_STD,
        alpha_unit: ArrayLike = ALPHA_ATOMS_CM2_S_TO_KG_YR,
        t0: ArrayLike | None = None,
        y0: ArrayLike | None = None,
    ) -> jax.Array:
        a_flat, b_flat, w_2d, is_1d, orig_shape = _flatten_windows(
            t_starts, t_ends, sub_weights
        )
        quad_t = _gl16_window_nodes(a_flat, b_flat)
        dy_quad = self.solve_delta_trajectory(
            cbm,
            production,
            quad_t,
            q_std=q_std,
            alpha_unit=alpha_unit,
            t0=t0,
            y0=y0,
        )
        return _gl16_reduce_windows(dy_quad, a_flat.shape[0], (w_2d, is_1d, orig_shape))

    def predict_d14c(
        self,
        cbm: CarbonBoxModel,
        production: ProductionTerm,
        t_starts: ArrayLike,
        t_ends: ArrayLike,
        *,
        sub_weights: ArrayLike | None = None,
        box_indices: int | ArrayLike | None = None,
        q_std: ArrayLike = DEFAULT_Q_STD,
        alpha_unit: ArrayLike = ALPHA_ATOMS_CM2_S_TO_KG_YR,
        delta_global: ArrayLike = 0.0,
        delta_lab: ArrayLike = 0.0,
        relative_to: Literal["standard", "ambient"] = "standard",
        t0: ArrayLike | None = None,
        y0: ArrayLike | None = None,
    ) -> jax.Array:
        if relative_to not in ("standard", "ambient"):
            raise ValueError(
                f"relative_to must be 'standard' or 'ambient'; got {relative_to!r}."
            )
        if not (self.has_time_varying_masses or _cbm_has_nonautonomous_masses(cbm)):
            return super().predict_d14c(
                cbm,
                production,
                t_starts,
                t_ends,
                sub_weights=sub_weights,
                box_indices=box_indices,
                q_std=q_std,
                alpha_unit=alpha_unit,
                delta_global=delta_global,
                delta_lab=delta_lab,
                relative_to=relative_to,
                t0=t0,
                y0=y0,
            )

        a_flat, b_flat, w_2d, is_1d, orig_shape = _flatten_windows(
            t_starts, t_ends, sub_weights
        )
        quad_t = _gl16_window_nodes(a_flat, b_flat)
        dy_quad = self.solve_delta_trajectory(
            cbm,
            production,
            quad_t,
            q_std=q_std,
            alpha_unit=alpha_unit,
            t0=t0,
            y0=y0,
        )
        mt_quad = jax.vmap(lambda ti: self._masses_at(cbm, ti))(quad_t)
        suess_params = _prepare_suess_params(cbm, q_std, alpha_unit, delta_global, relative_to)
        inst_d14c = _suess_instantaneous_d14c(dy_quad, mt_quad, suess_params)
        d14c_win = _gl16_reduce_windows(inst_d14c, a_flat.shape[0], (w_2d, is_1d, orig_shape))
        selected = _select_boxes(d14c_win, box_indices)
        return selected + jnp.asarray(delta_lab, dtype=jnp.float64)
