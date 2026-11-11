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
import math
from typing import Any, Literal

import diffrax
import equinox as eqx
import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
from numpy.polynomial.legendre import leggauss
import zodiax as zdx

from .cbm import CarbonBoxModel
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


def _eval_boxcar_pointwise(
    A: jax.Array,
    b_scaled: jax.Array,
    spike: MiyakeSpike,
    t: jax.Array,
) -> jax.Array:
    """Exact pointwise boxcar spike response with double-``where`` ``tau_d -> 0+`` guard."""
    ts = spike.t_start
    tau_d = jnp.maximum(spike.duration, 0.0)
    fluence = spike.fluence
    is_tiny = tau_d <= 1e-12
    safe_tau = jnp.where(is_tiny, 1.0, tau_d)

    def _one_t(ti: jax.Array) -> jax.Array:
        delta_act = jnp.clip(ti - ts, 0.0, tau_d)
        delta_post = jnp.maximum(ti - (ts + tau_d), 0.0)
        frac = jnp.where(
            is_tiny,
            jnp.where(ti >= ts, 1.0, 0.0),
            delta_act / safe_tau,
        )
        _, g0_act, _ = van_loan_foh_blocks(A, b_scaled, delta_act)
        # Note g0_act = delta_act * psi_1(A * delta_act) @ b_scaled.
        # To keep psi_1(A * delta_act) @ b_scaled smooth at delta_act = 0 without dividing by delta_act,
        # evaluate psi_1 via unit-scaled vector column!
        _, psi1_act_b = van_loan_psi1_vec(A * delta_act, b_scaled, 1.0)
        _ = g0_act
        Phi_post = jax.scipy.linalg.expm(A * delta_post)
        return fluence * frac * (Phi_post @ psi1_act_b)

    return jax.vmap(_one_t)(t)


def _eval_boxcar_binned(
    A: jax.Array,
    b_scaled: jax.Array,
    spike: MiyakeSpike,
    a_r: jax.Array,
    b_r: jax.Array,
) -> jax.Array:
    """Exact analytical seasonal window average of a boxcar Miyake pulse.

    Partitions ``[a_r, b_r]`` at ``t_1 = clip(t_s, a_r, b_r)`` and
    ``t_2 = clip(t_s + tau_d, a_r, b_r)`` using Van Loan ``psi_1, psi_2`` blocks
    and double-``where`` ``tau_d -> 0+`` protection so both primal values and
    ``jax.grad`` wrt ``(t_start, duration, fluence)`` remain exact and ``NaN``-free
    even at ``duration == 0.0``.
    """
    ts = spike.t_start
    tau_d = jnp.maximum(spike.duration, 0.0)
    fluence = spike.fluence
    is_tiny = tau_d <= 1e-12
    safe_tau = jnp.where(is_tiny, 1.0, tau_d)

    def _one_win(ar: jax.Array, br: jax.Array) -> jax.Array:
        width = br - ar

        # 1. Pre-window overlap [u0, u1] = [min(ts, ar), min(ts + tau_d, ar)]
        u0 = jnp.minimum(ts, ar)
        u1 = jnp.minimum(ts + tau_d, ar)
        delta_pre = u1 - u0
        f_pre = jnp.where(
            is_tiny,
            jnp.where(ts < ar, 1.0, 0.0),
            delta_pre / safe_tau,
        )
        _, psi1_pre_b = van_loan_psi1_vec(A * delta_pre, b_scaled, 1.0)
        Phi_to_ar = jax.scipy.linalg.expm(A * (ar - u1))
        dy_ar = fluence * f_pre * (Phi_to_ar @ psi1_pre_b)
        _, int_pre = van_loan_psi1_vec(A, dy_ar, width)

        # 2. In-window overlap [t1, t2] = [clip(ts, ar, br), clip(ts + tau_d, ar, br)]
        t1 = jnp.clip(ts, ar, br)
        t2 = jnp.clip(ts + tau_d, ar, br)
        delta_in = t2 - t1
        delta_post = br - t2
        f_in = jnp.where(
            is_tiny,
            jnp.where((ts >= ar) & (ts < br), 1.0, 0.0),
            delta_in / safe_tau,
        )
        # Active accumulation integral over [t1, t2]: E * f_in * delta_in * psi_2(A * delta_in) b
        _, _, g1_in = van_loan_foh_blocks(A * delta_in, b_scaled, 1.0)
        int_active = fluence * f_in * delta_in * g1_in

        # Post-spike decay inside [t2, br] of state accumulated on [t1, t2]
        _, psi1_in_b = van_loan_psi1_vec(A * delta_in, b_scaled, 1.0)
        dy_t2 = fluence * f_in * psi1_in_b
        _, int_post = van_loan_psi1_vec(A, dy_t2, delta_post)

        return (int_pre + int_active + int_post) / width

    return jax.vmap(_one_win)(a_r, b_r)


def _eval_smooth_spike_pointwise(
    A: jax.Array,
    b_scaled: jax.Array,
    spike: MiyakeSpike,
    t: jax.Array,
) -> jax.Array:
    """16-point Gauss–Legendre clamped quadrature for smooth ``MiyakeSpike`` profiles."""
    ts = spike.t_start
    tau_d = jnp.maximum(spike.duration, 0.0)
    fluence = spike.fluence
    is_tiny = tau_d <= 1e-12
    safe_tau = jnp.where(is_tiny, 1.0, tau_d)
    panels_arr = jnp.asarray(spike.unit_support_panels(), dtype=jnp.float64)
    u_los = panels_arr[:, 0]
    u_his = panels_arr[:, 1]
    xi = jnp.asarray(GL16_NODES_01, dtype=jnp.float64)
    wq = jnp.asarray(GL16_WEIGHTS_01, dtype=jnp.float64)

    def _one_t(ti: jax.Array) -> jax.Array:
        def _panel_contrib(u_lo: jax.Array, u_hi: jax.Array) -> jax.Array:
            p_a = ts + tau_d * u_lo
            p_b = ts + tau_d * u_hi
            tilde_a = jnp.minimum(p_a, ti)
            tilde_b = jnp.minimum(p_b, ti)
            span = tilde_b - tilde_a
            ratio = jnp.where(
                is_tiny,
                jnp.where(ti >= ts, u_hi - u_lo, 0.0),
                span / safe_tau,
            )
            u_q = jnp.where(
                is_tiny,
                u_lo + (u_hi - u_lo) * xi,
                (tilde_a + span * xi - ts) / safe_tau,
            )
            tau_q = jnp.where(is_tiny, ts + tau_d * u_q, tilde_a + span * xi)
            phi_q = spike.unit_profile(u_q)  # (16,)

            local_lags = jnp.maximum(tilde_b - tau_q, 0.0)
            vecs = jax.vmap(lambda lag: jax.scipy.linalg.expm(A * lag) @ b_scaled)(
                local_lags
            )  # (16, n)
            panel_vec = (wq * phi_q) @ vecs
            Phi_shift = jax.scipy.linalg.expm(A * jnp.maximum(ti - tilde_b, 0.0))
            return fluence * ratio * (Phi_shift @ panel_vec)

        return jnp.sum(jax.vmap(_panel_contrib)(u_los, u_his), axis=0)

    return jax.vmap(_one_t)(t)


def _eval_smooth_spike_binned(
    A: jax.Array,
    b_scaled: jax.Array,
    spike: MiyakeSpike,
    a_r: jax.Array,
    b_r: jax.Array,
) -> jax.Array:
    """16-point Gauss–Legendre seasonal-window integration for smooth ``MiyakeSpike`` profiles.

    Applies Fubini's theorem to reduce the double ``(t, tau)`` integral on each
    window ``[a_r, b_r]`` to 1D Gauss–Legendre quadrature over smoothly clamped
    spike sub-panels with exact analytical ``psi_1`` integration in ``t``.
    """
    ts = spike.t_start
    tau_d = jnp.maximum(spike.duration, 0.0)
    fluence = spike.fluence
    is_tiny = tau_d <= 1e-12
    safe_tau = jnp.where(is_tiny, 1.0, tau_d)
    panels_arr = jnp.asarray(spike.unit_support_panels(), dtype=jnp.float64)
    u_los = panels_arr[:, 0]
    u_his = panels_arr[:, 1]
    xi = jnp.asarray(GL16_NODES_01, dtype=jnp.float64)
    wq = jnp.asarray(GL16_WEIGHTS_01, dtype=jnp.float64)

    def _one_win(ar: jax.Array, br: jax.Array) -> jax.Array:
        width = br - ar

        def _panel_win(u_lo: jax.Array, u_hi: jax.Array) -> tuple[jax.Array, jax.Array]:
            p_a = ts + tau_d * u_lo
            p_b = ts + tau_d * u_hi

            # (a) Pre-window clamped sub-panel (-inf, ar]
            pre_a = jnp.minimum(p_a, ar)
            pre_b = jnp.minimum(p_b, ar)
            pre_span = pre_b - pre_a
            pre_ratio = jnp.where(
                is_tiny,
                jnp.where(ts < ar, u_hi - u_lo, 0.0),
                pre_span / safe_tau,
            )
            u_pre = jnp.where(
                is_tiny,
                u_lo + (u_hi - u_lo) * xi,
                (pre_a + pre_span * xi - ts) / safe_tau,
            )
            tau_pre = jnp.where(is_tiny, ts + tau_d * u_pre, pre_a + pre_span * xi)
            phi_pre = spike.unit_profile(u_pre)
            lags_pre = jnp.maximum(pre_b - tau_pre, 0.0)
            vecs_pre = jax.vmap(lambda lag: jax.scipy.linalg.expm(A * lag) @ b_scaled)(
                lags_pre
            )
            panel_pre = (wq * phi_pre) @ vecs_pre
            Phi_to_ar = jax.scipy.linalg.expm(A * jnp.maximum(ar - pre_b, 0.0))
            dy_ar_p = fluence * pre_ratio * (Phi_to_ar @ panel_pre)

            # (b) In-window clamped sub-panel [ar, br]
            in_a = jnp.clip(p_a, ar, br)
            in_b = jnp.clip(p_b, ar, br)
            in_span = in_b - in_a
            in_ratio = jnp.where(
                is_tiny,
                jnp.where((ts >= ar) & (ts < br), u_hi - u_lo, 0.0),
                in_span / safe_tau,
            )
            u_in = jnp.where(
                is_tiny,
                u_lo + (u_hi - u_lo) * xi,
                (in_a + in_span * xi - ts) / safe_tau,
            )
            tau_in = jnp.where(is_tiny, ts + tau_d * u_in, in_a + in_span * xi)
            phi_in = spike.unit_profile(u_in)
            rem = jnp.maximum(br - tau_in, 0.0)
            int_vecs = jax.vmap(lambda r: van_loan_psi1_vec(A, b_scaled, r)[1])(rem)
            panel_in = (wq * phi_in) @ int_vecs
            int_in_p = fluence * in_ratio * panel_in
            return dy_ar_p, int_in_p

        dy_ar_all, int_in_all = jax.vmap(_panel_win)(u_los, u_his)
        dy_ar = jnp.sum(dy_ar_all, axis=0)
        int_in = jnp.sum(int_in_all, axis=0)
        _, int_pre_total = van_loan_psi1_vec(A, dy_ar, width)
        return (int_pre_total + int_in) / width

    return jax.vmap(_one_win)(a_r, b_r)


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

    def _eval_single_term_pointwise(
        self,
        A: jax.Array,
        b_scaled: jax.Array,
        term: ProductionTerm,
        t: jax.Array,
        q_std: ArrayLike,
    ) -> jax.Array:
        n = A.shape[0]
        if isinstance(term, SteadyProduction):
            dq = term.rate - jnp.asarray(q_std, dtype=jnp.float64)
            v_ss = jnp.linalg.solve(-A, b_scaled)
            return jnp.broadcast_to((dq * v_ss)[None, :], (t.shape[0], n))

        if isinstance(term, HarmonicSolar):
            return _eval_harmonic_pointwise(A, b_scaled, term, t)

        if isinstance(term, LinearTrend):
            return _eval_trend_pointwise(A, b_scaled, term, t)

        if isinstance(term, MiyakeSpike):
            if term.profile == "boxcar":
                return _eval_boxcar_pointwise(A, b_scaled, term, t)
            return _eval_smooth_spike_pointwise(A, b_scaled, term, t)

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
    ) -> jax.Array:
        n = A.shape[0]
        if isinstance(term, SteadyProduction):
            dq = term.rate - jnp.asarray(q_std, dtype=jnp.float64)
            v_ss = jnp.linalg.solve(-A, b_scaled)
            return jnp.broadcast_to((dq * v_ss)[None, :], (a_r.shape[0], n))

        if isinstance(term, HarmonicSolar):
            return _eval_harmonic_binned(A, b_scaled, term, a_r, b_r)

        if isinstance(term, LinearTrend):
            return _eval_trend_binned(A, b_scaled, term, a_r, b_r)

        if isinstance(term, MiyakeSpike):
            if term.profile == "boxcar":
                return _eval_boxcar_binned(A, b_scaled, term, a_r, b_r)
            return _eval_smooth_spike_binned(A, b_scaled, term, a_r, b_r)

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
    ) -> jax.Array:
        t_arr = jnp.atleast_1d(jnp.asarray(t_eval, dtype=jnp.float64))
        A = cbm.matrix_A()
        b_scaled = jnp.asarray(alpha_unit, dtype=jnp.float64) * cbm.production_vector()
        terms = _unpack_terms(production)

        total = jnp.zeros((t_arr.shape[0], cbm.n_boxes), dtype=jnp.float64)
        for term in terms:
            total = total + self._eval_single_term_pointwise(
                A, b_scaled, term, t_arr, q_std
            )

        if y0 is not None:
            t_init = t_arr[0] if t0 is None else jnp.asarray(t0, dtype=jnp.float64)
            y_ss = cbm.steady_state_c14(q_std=q_std, alpha_unit=alpha_unit)
            dy_target0 = jnp.asarray(y0, dtype=jnp.float64) - y_ss
            dy_part0 = jnp.zeros((cbm.n_boxes,), dtype=jnp.float64)
            t_init_1d = jnp.reshape(t_init, (1,))
            for term in terms:
                dy_part0 = dy_part0 + self._eval_single_term_pointwise(
                    A, b_scaled, term, t_init_1d, q_std
                )[0]
            diff0 = dy_target0 - dy_part0
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
    ) -> jax.Array:
        a_flat, b_flat, w_2d, is_1d, orig_shape = _flatten_windows(
            t_starts, t_ends, sub_weights
        )
        A = cbm.matrix_A()
        b_scaled = jnp.asarray(alpha_unit, dtype=jnp.float64) * cbm.production_vector()
        terms = _unpack_terms(production)

        total = jnp.zeros((a_flat.shape[0], cbm.n_boxes), dtype=jnp.float64)
        for term in terms:
            total = total + self._eval_single_term_binned(
                A, b_scaled, term, a_flat, b_flat, q_std
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
                    A, b_scaled, term, t_init_1d, q_std
                )[0]
            diff0 = dy_target0 - dy_part0

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

    def _collect_initial_time_and_events(
        self,
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
                    if term.gamma_shape <= 1.0:
                        jump_list.append(term.t_start)
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
                step_list.append(term.t_grid)
            elif isinstance(term, (LogGPProduction, GPProduction)) and term.domain is not None:
                t0_candidates.append(jnp.asarray(term.domain[0], dtype=jnp.float64))

        if t0_override is not None:
            t_start_int = jnp.asarray(t0_override, dtype=jnp.float64)
        else:
            t_start_int = jnp.min(jnp.stack([jnp.reshape(c, ()) for c in t0_candidates]))

        step_ts = None
        if step_list:
            flat_steps = jnp.concatenate([jnp.atleast_1d(s) for s in step_list])
            # Keep strictly inside (t_start_int, latest_obs) or clamp safely
            step_ts = jnp.sort(flat_steps)

        jump_ts = None
        if jump_list:
            flat_jumps = jnp.concatenate([jnp.atleast_1d(j) for j in jump_list])
            jump_ts = jnp.sort(flat_jumps)

        _ = latest_obs
        return t_start_int, step_ts, jump_ts

    def _equilibrium_dy0_and_drift_anchor(
        self,
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
        order = jnp.argsort(t_arr)
        inv_order = jnp.argsort(order)
        t_sorted = t_arr[order]

        A0 = cbm.matrix_A()
        b_scaled = jnp.asarray(alpha_unit, dtype=jnp.float64) * cbm.production_vector()
        y_ss = cbm.steady_state_c14(q_std=q_std, alpha_unit=alpha_unit)
        terms = _unpack_terms(production)

        t_start_int, step_ts, jump_ts = self._collect_initial_time_and_events(
            terms, t_sorted[0], t_sorted[-1], t0
        )
        t_end_int = jnp.maximum(t_sorted[-1], t_start_int + 1e-9)

        dy0_eq, secular_anchor = self._equilibrium_dy0_and_drift_anchor(
            A0, b_scaled, terms, t_start_int, q_std
        )
        if y0 is not None:
            dy_init = jnp.asarray(y0, dtype=jnp.float64) - y_ss
            secular_anchor = jnp.zeros_like(secular_anchor)
        else:
            dy_init = dy0_eq

        def _rhs(t: jax.Array, dy: jax.Array, args: Any) -> jax.Array:
            cbm_arg, prod_arg, b_sc, y_ss_arg, A0_arg, sec_anc, q_s = args
            At = self._A_at(cbm_arg, t)
            dq_t = prod_arg.delta_q(t, q_std=q_s)
            non_auton = (At - A0_arg) @ y_ss_arg
            return At @ dy + non_auton + b_sc * dq_t + sec_anc

        controller = diffrax.PIDController(
            rtol=self.rtol,
            atol=self.atol,
            step_ts=step_ts,
            jump_ts=jump_ts,
        )
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(_rhs),
            self._make_diffrax_solver(),
            t0=t_start_int,
            t1=t_end_int,
            dt0=self.dt0,
            y0=dy_init,
            args=(cbm, production, b_scaled, y_ss, A0, secular_anchor, q_std),
            saveat=diffrax.SaveAt(ts=t_sorted),
            stepsize_controller=controller,
            max_steps=self.max_steps,
        )
        return sol.ys[inv_order]

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
        order = jnp.argsort(all_bounds)
        inv_order = jnp.argsort(order)
        ts_sorted = all_bounds[order]

        A0 = cbm.matrix_A()
        b_scaled = jnp.asarray(alpha_unit, dtype=jnp.float64) * cbm.production_vector()
        y_ss = cbm.steady_state_c14(q_std=q_std, alpha_unit=alpha_unit)
        terms = _unpack_terms(production)

        t_start_int, step_ts, jump_ts = self._collect_initial_time_and_events(
            terms, ts_sorted[0], ts_sorted[-1], t0
        )
        t_end_int = jnp.maximum(ts_sorted[-1], t_start_int + 1e-9)

        dy0_eq, secular_anchor = self._equilibrium_dy0_and_drift_anchor(
            A0, b_scaled, terms, t_start_int, q_std
        )
        if y0 is not None:
            dy_init = jnp.asarray(y0, dtype=jnp.float64) - y_ss
            secular_anchor = jnp.zeros_like(secular_anchor)
        else:
            dy_init = dy0_eq

        Y_init = jnp.zeros_like(dy_init)

        def _aug_rhs(
            t: jax.Array, state: tuple[jax.Array, jax.Array], args: Any
        ) -> tuple[jax.Array, jax.Array]:
            dy, _ = state
            cbm_arg, prod_arg, b_sc, y_ss_arg, A0_arg, sec_anc, q_s = args
            At = self._A_at(cbm_arg, t)
            dq_t = prod_arg.delta_q(t, q_std=q_s)
            non_auton = (At - A0_arg) @ y_ss_arg
            ddy = At @ dy + non_auton + b_sc * dq_t + sec_anc
            return ddy, dy

        controller = diffrax.PIDController(
            rtol=self.rtol,
            atol=self.atol,
            step_ts=step_ts,
            jump_ts=jump_ts,
        )
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(_aug_rhs),
            self._make_diffrax_solver(),
            t0=t_start_int,
            t1=t_end_int,
            dt0=self.dt0,
            y0=(dy_init, Y_init),
            args=(cbm, production, b_scaled, y_ss, A0, secular_anchor, q_std),
            saveat=diffrax.SaveAt(ts=ts_sorted),
            stepsize_controller=controller,
            max_steps=self.max_steps,
        )
        _, Y_sorted = sol.ys
        Y_all = Y_sorted[inv_order]
        n_w = a_flat.shape[0]
        Y_a = Y_all[:n_w]
        Y_b = Y_all[n_w:]
        avg = (Y_b - Y_a) / (b_flat - a_flat)[:, None]

        if is_1d:
            return avg
        assert w_2d is not None
        n_r, m_w = orig_shape
        reshaped = avg.reshape((n_r, m_w, cbm.n_boxes))
        return jnp.sum(reshaped * w_2d[:, :, None], axis=1)

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
        if not self.has_time_varying_masses:
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
        # integrate instantaneous specific activity (y_c(t) / M_c(t)) / R_ref - 1
        a_flat, b_flat, w_2d, is_1d, orig_shape = _flatten_windows(
            t_starts, t_ends, sub_weights
        )
        all_bounds = jnp.concatenate([a_flat, b_flat])
        order = jnp.argsort(all_bounds)
        inv_order = jnp.argsort(order)
        ts_sorted = all_bounds[order]

        A0 = cbm.matrix_A()
        m0 = cbm.reservoir_masses()
        b_scaled = jnp.asarray(alpha_unit, dtype=jnp.float64) * cbm.production_vector()
        y_ss = cbm.steady_state_c14(q_std=q_std, alpha_unit=alpha_unit)
        r_ref = cbm.reference_specific_activity(q_std=q_std, alpha_unit=alpha_unit)
        terms = _unpack_terms(production)

        t_start_int, step_ts, jump_ts = self._collect_initial_time_and_events(
            terms, ts_sorted[0], ts_sorted[-1], t0
        )
        t_end_int = jnp.maximum(ts_sorted[-1], t_start_int + 1e-9)

        dy0_eq, secular_anchor = self._equilibrium_dy0_and_drift_anchor(
            A0, b_scaled, terms, t_start_int, q_std
        )
        dy_init = (
            jnp.asarray(y0, dtype=jnp.float64) - y_ss if y0 is not None else dy0_eq
        )
        if y0 is not None:
            secular_anchor = jnp.zeros_like(secular_anchor)
        R_int_init = jnp.zeros_like(dy_init)

        def _suess_rhs(
            t: jax.Array, state: tuple[jax.Array, jax.Array], args: Any
        ) -> tuple[jax.Array, jax.Array]:
            dy, _ = state
            cbm_arg, prod_arg, b_sc, y_ss_arg, A0_arg, sec_anc, r_ref_arg, q_s = args
            At = self._A_at(cbm_arg, t)
            mt = self._masses_at(cbm_arg, t)
            dq_t = prod_arg.delta_q(t, q_std=q_s)
            non_auton = (At - A0_arg) @ y_ss_arg
            ddy = At @ dy + non_auton + b_sc * dq_t + sec_anc
            inst_d14c = 1000.0 * (((y_ss_arg + dy) / mt) / r_ref_arg - 1.0)
            return ddy, inst_d14c

        controller = diffrax.PIDController(
            rtol=self.rtol,
            atol=self.atol,
            step_ts=step_ts,
            jump_ts=jump_ts,
        )
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(_suess_rhs),
            self._make_diffrax_solver(),
            t0=t_start_int,
            t1=t_end_int,
            dt0=self.dt0,
            y0=(dy_init, R_int_init),
            args=(cbm, production, b_scaled, y_ss, A0, secular_anchor, r_ref, q_std),
            saveat=diffrax.SaveAt(ts=ts_sorted),
            stepsize_controller=controller,
            max_steps=self.max_steps,
        )
        _, R_sorted = sol.ys
        R_all = R_sorted[inv_order]
        n_w = a_flat.shape[0]
        d14c_win = (R_all[n_w:] - R_all[:n_w]) / (b_flat - a_flat)[:, None]

        if not is_1d:
            assert w_2d is not None
            n_r, m_w = orig_shape
            d14c_win = jnp.sum(
                d14c_win.reshape((n_r, m_w, cbm.n_boxes)) * w_2d[:, :, None],
                axis=1,
            )

        r_ss_ratio = (y_ss / m0) / r_ref
        d14c_all = (
            d14c_win + jnp.asarray(delta_global, dtype=jnp.float64) * r_ss_ratio
        )
        selected = _select_boxes(d14c_all, box_indices)
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
    _custom_backend_fn: Callable[..., jax.Array] | None = eqx.field(static=True)

    def __init__(
        self,
        method: Literal["Rodas5P", "Tsit5"] = "Rodas5P",
        *,
        rtol: float = 1e-8,
        atol: float = 1e-8,
        max_steps: int = 100_000,
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
        self._custom_backend_fn = backend_fn

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
        if self._custom_backend_fn is not None:
            return self._custom_backend_fn(
                cbm,
                production,
                jnp.atleast_1d(jnp.asarray(t_eval, dtype=jnp.float64)),
                q_std=q_std,
                alpha_unit=alpha_unit,
                t0=t0,
                y0=y0,
            )

        modax = _load_modax_module()
        t_arr = jnp.atleast_1d(jnp.asarray(t_eval, dtype=jnp.float64))
        b_scaled = jnp.asarray(alpha_unit, dtype=jnp.float64) * cbm.production_vector()
        y_ss = cbm.steady_state_c14(q_std=q_std, alpha_unit=alpha_unit)
        t_init = t_arr[0] if t0 is None else jnp.asarray(t0, dtype=jnp.float64)
        dy_init = (
            jnp.zeros((cbm.n_boxes,), dtype=jnp.float64)
            if y0 is None
            else jnp.asarray(y0, dtype=jnp.float64) - y_ss
        )

        solver_cls = getattr(modax, self.method)

        def _rhs(t: jax.Array, dy: jax.Array, args: Any) -> jax.Array:
            cbm_a, prod_a, b_sc, q_s = args
            return cbm_a.matrix_A_at(t) @ dy + b_sc * prod_a.delta_q(t, q_std=q_s)

        sol = modax.diffeqsolve(
            _rhs,
            solver_cls(),
            t0=t_init,
            t1=t_arr[-1],
            y0=dy_init,
            ts=t_arr,
            args=(cbm, production, b_scaled, q_std),
            rtol=self.rtol,
            atol=self.atol,
            max_steps=self.max_steps,
        )
        return getattr(sol, "ys", sol)

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
        xi = jnp.asarray(GL16_NODES_01, dtype=jnp.float64)
        wq = jnp.asarray(GL16_WEIGHTS_01, dtype=jnp.float64)
        quad_t = (a_flat[:, None] + (b_flat - a_flat)[:, None] * xi[None, :]).reshape(-1)
        dy_quad = self.solve_delta_trajectory(
            cbm,
            production,
            quad_t,
            q_std=q_std,
            alpha_unit=alpha_unit,
            t0=t0,
            y0=y0,
        ).reshape((a_flat.shape[0], xi.shape[0], cbm.n_boxes))
        avg = jnp.einsum("wqn,q->wn", dy_quad, wq)
        if is_1d:
            return avg
        assert w_2d is not None
        n_r, m_w = orig_shape
        return jnp.sum(avg.reshape((n_r, m_w, cbm.n_boxes)) * w_2d[:, :, None], axis=1)
