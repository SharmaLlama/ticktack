"""Decision 5.1 Part 2 verification suite for ``ticktack.kalman`` (`kalman_gp_marginal`).

Covers:
1. Module exports, ``__all__`` deduplication across ``ticktack.kalman`` and
   ``ticktack.linear``, ``_utils`` helper sharing, clean fresh-interpreter
   imports in both module orders, zero ``Any`` annotations in ``kalman.py``, and
   single-call ``_qr_whiten_and_posterior`` nuisance marginalization.
2. IntCal20-style mixed-resolution raw multi-series datasets (annual 1 yr tree
   rings, bimodal NH/SH multi-subwindow seasonal weights, 5 yr and 10 yr
   contiguous and overlapping multi-year blocks, instantaneous point
   observations, multi-decade gaps, multi-box atmospheric/marine observations,
   and correlated non-diagonal ``nuisance_prior_root`` with non-zero prior mean)
   verified to ``rtol=1e-10, atol=1e-10`` against an independent dense SciPy
   continuous-time double-integral MVN oracle across ``Matern12``, ``Matern32``,
   and ``Matern52`` kernels.
3. Exact reverse-mode ``jax.grad`` verification wrt
   ``(A, b, log_scale, log_sigma_gp, sigma_obs)`` and smoothed posterior
   trajectories against 4th-order central finite differences (``rtol=1e-7``).
4. Direct unit tests for all private state-space helpers, input-validation
   ``ValueError`` guards, and negative-control assertions proving the dense-MVN
   parity checks fail if ``half_logdet_nuis`` or smoother ``cov_omega_nuis`` is
   omitted.
"""

from __future__ import annotations

import inspect
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import jax

jax.config.update("jax_enable_x64", True)

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.integrate as spi
import scipy.linalg as sla
import tinygp.kernels as tgk
from tinygp.kernels import quasisep as tgq

import ticktack
import ticktack._utils as tt_utils
import ticktack.kalman as tt_kalman
import ticktack.linear as tt_linear
from ticktack.kalman import (
    KalmanGPResult,
    _assign_kalman_tracks,
    _build_kalman_step_matrices,
    _build_seasonal_kalman_schedule,
    _joseph_scalar_update,
    _marginalize_kalman_nuisance,
    _prepare_kalman_nuisance,
    _qr_whiten_and_posterior,
    _resolve_kalman_time_grid,
    _run_kalman_forward_pass,
    _run_kalman_rts_smoother,
    _van_loan_kalman_step_dynamics,
    kalman_gp_marginal,
)

LAMBDA_14 = math.log(2.0) / 5730.0


# =============================================================================
# Independent Dense SciPy Continuous-Time Double-Integral MVN Oracle
# =============================================================================


class _DenseMVNOracleResult(NamedTuple):
    loglike: float
    production_mean: np.ndarray
    production_var: np.ndarray
    cbm_state_mean: np.ndarray
    cbm_state_var: np.ndarray
    nuisance_mean: np.ndarray | None
    nuisance_cov: np.ndarray | None


def _textbook_matern_sde(
    kind: str,
    scale: float,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form continuous-time companion SDE ``(F_gp, Q_sde, h_gp)`` in NumPy."""
    if kind == "matern12":
        lam = 1.0 / scale
        return (
            np.array([[-lam]], dtype=np.float64),
            np.array([[2.0 * lam * (sigma**2)]], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        )
    if kind == "matern32":
        lam = math.sqrt(3.0) / scale
        return (
            np.array([[0.0, 1.0], [-(lam**2), -2.0 * lam]], dtype=np.float64),
            np.array([[0.0, 0.0], [0.0, 4.0 * (lam**3) * (sigma**2)]], dtype=np.float64),
            np.array([1.0, 0.0], dtype=np.float64),
        )
    if kind == "matern52":
        lam = math.sqrt(5.0) / scale
        f_gp = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-(lam**3), -3.0 * (lam**2), -3.0 * lam],
            ],
            dtype=np.float64,
        )
        q_sde = np.zeros((3, 3), dtype=np.float64)
        q_sde[2, 2] = (16.0 / 3.0) * (lam**5) * (sigma**2)
        return f_gp, q_sde, np.array([1.0, 0.0, 0.0], dtype=np.float64)
    raise ValueError(f"Unknown Matern kind: {kind!r}")


def _scipy_expm_integrals(
    f_mat: np.ndarray,
    delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate ``(exp(F Δ), ∫_0^Δ exp(F u) du, ∫_0^Δ (Δ - u) exp(F u) du)`` via SciPy ``expm``."""
    d = f_mat.shape[0]
    eye = np.eye(d, dtype=np.float64)
    zeros = np.zeros((d, d), dtype=np.float64)
    blk = np.block([[f_mat, eye, zeros], [zeros, zeros, eye], [zeros, zeros, zeros]]) * delta
    eb = sla.expm(blk)
    return eb[:d, :d], eb[:d, d : 2 * d], eb[:d, 2 * d :]


def _scipy_window_pair_cov(
    f_mat: np.ndarray,
    sig_inf: np.ndarray,
    s1: float,
    e1: float,
    s2: float,
    e2: float,
    cache: dict[tuple[str, float], object] | None = None,
) -> np.ndarray:
    """Exact window-mean state cross-covariance ``E[I_[s1,e1](ξ) I_[s2,e2](ξ)ᵀ]``."""

    def _expm(dt_val: float) -> np.ndarray:
        key = ("expm", round(float(dt_val), 12))
        if cache is not None and key in cache:
            return cache[key]  # type: ignore[return-value]
        res = sla.expm(f_mat * dt_val)
        if cache is not None:
            cache[key] = res
        return res

    def _ints(dt_val: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = ("ints", round(float(dt_val), 12))
        if cache is not None and key in cache:
            return cache[key]  # type: ignore[return-value]
        res = _scipy_expm_integrals(f_mat, dt_val)
        if cache is not None:
            cache[key] = res
        return res

    d1 = e1 - s1
    d2 = e2 - s2
    if d1 == 0.0 and d2 == 0.0:
        if s1 >= s2:
            return _expm(s1 - s2) @ sig_inf
        return sig_inf @ _expm(s2 - s1).T

    if d1 == 0.0:
        if s2 < s1 < e2:
            w_l = (s1 - s2) / d2
            w_r = (e2 - s1) / d2
            return w_l * _scipy_window_pair_cov(f_mat, sig_inf, s1, s1, s2, s1, cache) + w_r * (
                _scipy_window_pair_cov(f_mat, sig_inf, s1, s1, s1, e2, cache)
            )
        _, i2, _ = _ints(d2)
        if s1 >= e2:
            return _expm(s1 - e2) @ (i2 / d2) @ sig_inf
        return sig_inf @ _expm(s2 - s1).T @ (i2.T / d2)

    if d2 == 0.0:
        return _scipy_window_pair_cov(f_mat, sig_inf, s2, e2, s1, e1, cache).T

    pts = sorted({s1, e1, s2, e2})
    subs1 = [
        (pts[k], pts[k + 1])
        for k in range(len(pts) - 1)
        if s1 <= pts[k] and pts[k + 1] <= e1 and pts[k + 1] > pts[k]
    ]
    subs2 = [
        (pts[k], pts[k + 1])
        for k in range(len(pts) - 1)
        if s2 <= pts[k] and pts[k + 1] <= e2 and pts[k + 1] > pts[k]
    ]
    total = np.zeros_like(sig_inf)
    for a, b in subs1:
        len_ab = b - a
        _, i_ab, j_ab = _ints(len_ab)
        for c, d in subs2:
            len_cd = d - c
            if abs(a - c) < 1e-13 and abs(b - d) < 1e-13:
                m_sq = j_ab @ sig_inf
                term = (m_sq + m_sq.T) / (len_ab * len_cd)
            elif a >= d - 1e-13:
                _, i_cd, _ = _ints(len_cd)
                term = (i_ab / len_ab) @ _expm(a - d) @ (i_cd / len_cd) @ sig_inf
            elif c >= b - 1e-13:
                _, i_cd, _ = _ints(len_cd)
                term = sig_inf @ (i_ab.T / len_ab) @ _expm(c - b).T @ (i_cd.T / len_cd)
            else:
                raise RuntimeError(f"Unexpected sub-interval overlap {(a, b)} vs {(c, d)}")
            total += (len_ab / d1) * (len_cd / d2) * term
    return total


def _dense_scipy_mvn_oracle(
    a_mat: np.ndarray,
    b_vec: np.ndarray,
    kind: str,
    scale: float,
    sigma_gp: float,
    resid: np.ndarray,
    sigma_obs: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    sub_weights: np.ndarray,
    t_grid: np.ndarray,
    *,
    alpha_unit: float,
    obs_matrix: np.ndarray,
    nuisance_design: np.ndarray | None = None,
    nuisance_prior_mean: np.ndarray | None = None,
    nuisance_prior_root: np.ndarray | None = None,
) -> _DenseMVNOracleResult:
    """Pure NumPy/SciPy continuous-time double-integral MVN posterior oracle."""
    f_gp, q_sde, h_gp = _textbook_matern_sde(kind, scale, sigma_gp)
    n_boxes = a_mat.shape[0]
    d_gp = f_gp.shape[0]
    d_xi = n_boxes + d_gp
    f_joint = np.block(
        [
            [a_mat, np.outer(alpha_unit * b_vec, h_gp)],
            [np.zeros((d_gp, n_boxes), dtype=np.float64), f_gp],
        ]
    )
    q_joint = sla.block_diag(np.zeros((n_boxes, n_boxes), dtype=np.float64), q_sde)
    sig_inf = sla.solve_continuous_lyapunov(f_joint, -q_joint)
    sig_inf = 0.5 * (sig_inf + sig_inf.T)

    w_norm = sub_weights / np.sum(sub_weights, axis=-1, keepdims=True)
    n_obs, m_sub = starts.shape
    c_xi_obs = np.hstack([obs_matrix, np.zeros((n_obs, d_gp), dtype=np.float64)])
    exp_cache: dict[tuple[str, float], object] = {}

    k_gp = np.zeros((n_obs, n_obs), dtype=np.float64)
    for i in range(n_obs):
        for j in range(i, n_obs):
            blk = np.zeros_like(sig_inf)
            for a_idx in range(m_sub):
                for b_idx in range(m_sub):
                    blk += (
                        w_norm[i, a_idx]
                        * w_norm[j, b_idx]
                        * _scipy_window_pair_cov(
                            f_joint,
                            sig_inf,
                            float(starts[i, a_idx]),
                            float(ends[i, a_idx]),
                            float(starts[j, b_idx]),
                            float(ends[j, b_idx]),
                            exp_cache,
                        )
                    )
            val_ij = float(c_xi_obs[i] @ blk @ c_xi_obs[j])
            k_gp[i, j] = k_gp[j, i] = val_ij

    cov_noise = 0.5 * (k_gp + k_gp.T) + np.diag(sigma_obs**2)
    l_noise = sla.cholesky(cov_noise, lower=True)
    logdet_noise = 2.0 * float(np.sum(np.log(np.diag(l_noise))))

    if nuisance_design is not None:
        assert nuisance_prior_mean is not None and nuisance_prior_root is not None
        r_shift = resid - nuisance_design @ nuisance_prior_mean
        x_std = sla.solve_triangular(l_noise, r_shift, lower=True)
        u_mat = sla.solve_triangular(l_noise, nuisance_design @ nuisance_prior_root, lower=True)
        p_cols = u_mat.shape[1]
        q_aug, r_up = sla.qr(np.vstack([u_mat, np.eye(p_cols, dtype=np.float64)]), mode="economic")
        q_top, q_bot = q_aug[:n_obs], q_aug[n_obs:]
        omega0 = sla.solve_triangular(r_up, q_top.T @ x_std, lower=False)
        d_omega = sla.solve_triangular(
            r_up, q_top.T @ (x_std - u_mat @ omega0) - q_bot.T @ omega0, lower=False
        )
        omega = omega0 + d_omega
        r_std = x_std - u_mat @ omega
        quad = float(r_std @ r_std + omega @ omega)
        half_ld_nuis = float(np.sum(np.log(np.abs(np.diag(r_up)))))
        loglike = (
            -0.5 * quad - 0.5 * logdet_noise - half_ld_nuis - 0.5 * n_obs * math.log(2.0 * math.pi)
        )
        alpha_vec = sla.solve_triangular(l_noise.T, r_std, lower=False)
        r_inv = sla.solve_triangular(r_up, np.eye(p_cols, dtype=np.float64), lower=False)
        cov_omega = 0.5 * (r_inv @ r_inv.T + (r_inv @ r_inv.T).T)
        nuis_mean = nuisance_prior_mean + nuisance_prior_root @ omega
        nuis_cov = nuisance_prior_root @ cov_omega @ nuisance_prior_root.T
        nuis_cov = 0.5 * (nuis_cov + nuis_cov.T)
    else:
        x_std = sla.solve_triangular(l_noise, resid, lower=True)
        alpha_vec = sla.solve_triangular(l_noise.T, x_std, lower=False)
        loglike = (
            -0.5 * float(x_std @ x_std) - 0.5 * logdet_noise - 0.5 * n_obs * math.log(2.0 * math.pi)
        )
        nuis_mean = None
        nuis_cov = None
        u_mat = None
        r_up = None

    n_grid = t_grid.shape[0]
    cross_xi_d = np.zeros((n_grid, d_xi, n_obs), dtype=np.float64)
    for k_idx, tk_val in enumerate(t_grid):
        tk = float(tk_val)
        for j in range(n_obs):
            blk_kj = np.zeros_like(sig_inf)
            for b_idx in range(m_sub):
                blk_kj += w_norm[j, b_idx] * _scipy_window_pair_cov(
                    f_joint,
                    sig_inf,
                    tk,
                    tk,
                    float(starts[j, b_idx]),
                    float(ends[j, b_idx]),
                    exp_cache,
                )
            cross_xi_d[k_idx, :, j] = blk_kj @ c_xi_obs[j]

    xi_mean = np.einsum("kdj,j->kd", cross_xi_d, alpha_vec)
    cbm_mean = xi_mean[:, :n_boxes]
    q_mean = xi_mean[:, n_boxes:] @ h_gp
    cbm_var = np.zeros((n_grid, n_boxes), dtype=np.float64)
    q_var = np.zeros((n_grid,), dtype=np.float64)
    for k_idx in range(n_grid):
        ck_std = sla.solve_triangular(l_noise, cross_xi_d[k_idx].T, lower=True)
        cov_xi = sig_inf - ck_std.T @ ck_std
        if u_mat is not None and r_up is not None:
            proj = sla.solve_triangular(r_up.T, u_mat.T @ ck_std, lower=True)
            cov_xi = cov_xi + proj.T @ proj
        cbm_var[k_idx] = np.diag(cov_xi[:n_boxes, :n_boxes])
        q_var[k_idx] = float(h_gp @ cov_xi[n_boxes:, n_boxes:] @ h_gp)

    return _DenseMVNOracleResult(
        loglike=loglike,
        production_mean=q_mean,
        production_var=q_var,
        cbm_state_mean=cbm_mean,
        cbm_state_var=cbm_var,
        nuisance_mean=nuis_mean,
        nuisance_cov=nuis_cov,
    )


def _make_asymmetric_three_box_cbm() -> tuple[np.ndarray, np.ndarray]:
    """Asymmetric 3-box CBM (NH trop, SH trop, surface ocean) with slow ¹⁴C decay."""
    a_mat = np.array(
        [
            [-(1.48 + 0.67 + LAMBDA_14), 0.61, 0.042],
            [1.48, -(0.61 + 0.59 + LAMBDA_14), 0.035],
            [0.67, 0.59, -(0.042 + 0.035 + 0.009 + LAMBDA_14)],
        ],
        dtype=np.float64,
    )
    b_vec = np.array([0.54, 0.46, 0.0], dtype=np.float64)
    return a_mat, b_vec


def _build_intcal20_style_multiseries_dataset() -> dict[str, np.ndarray]:
    """Construct a 30-yr mixed-resolution IntCal20-style BCE multi-series dataset.

    Anchored at ``t_0 = -5489.0`` CE (bundled ``IntCal20`` start epoch) on a
    non-dyadic ``dt = 1 / 24`` yr step grid (``721`` nodes) to exercise
    floating-point step recovery and monthly ring boundaries:
    - Series 0 (NH single-year + monthly/bimodal seasonal rings, ``[-5489, -5483]``):
      1 yr boxcars, standard NH ``[4/12, 8/12]`` rings, and bimodal spring/autumn
      sub-windows with summer dormancy gaps.
    - Series 1 (NH IntCal20 5 yr & 10 yr overlapping blocks, ``[-5488, -5476]``):
      4 contiguous/overlapping 5 yr and 10 yr blocks sharing epochs with Series 0.
    - Uneven 6-yr unmeasured gap on ``[-5476.0, -5470.0]``.
    - Series 2 (SH ``[9/12, 14/12]`` seasonal rings + simultaneous completions at
      ``t = -5467.0`` + point sample, ``[-5470, -5465]``).
    - Uneven 3-yr unmeasured gap on ``[-5465.0, -5462.0]``.
    - Series 3 (Surface-ocean / mixed marine 2 yr & 3 yr blocks, ``[-5462, -5459]``).
    """
    rng = np.random.default_rng(20260512)
    t0 = -5489.0
    step_dt = 1.0 / 24.0
    t_grid = t0 + step_dt * np.arange(30 * 24 + 1, dtype=np.float64)

    rows_s: list[list[float]] = []
    rows_e: list[list[float]] = []
    rows_w: list[list[float]] = []
    rows_c: list[list[float]] = []
    series_id: list[int] = []

    c_nh = [1.34, 0.0, 0.0]
    c_sh = [0.0, 1.29, 0.0]
    c_mar = [0.14, 0.08, 0.87]

    # Series 0: 1 yr boxcars + NH monthly [4/12, 8/12] + bimodal dormancy-gapped rings
    for off in (0.0, 1.0):
        yr = t0 + off
        rows_s.append([yr, yr + 0.5])
        rows_e.append([yr + 0.5, yr + 1.0])
        rows_w.append([0.5, 0.5])
        rows_c.append(c_nh)
        series_id.append(0)
    for off in (2.0, 3.0):
        yr = t0 + off
        rows_s.append([yr + 4.0 / 12.0, yr + 6.0 / 12.0])
        rows_e.append([yr + 6.0 / 12.0, yr + 8.0 / 12.0])
        rows_w.append([0.5, 0.5])
        rows_c.append(c_nh)
        series_id.append(0)
    for off in (4.0, 5.0):
        yr = t0 + off
        rows_s.append([yr + 3.0 / 12.0, yr + 9.0 / 12.0])  # dormancy gap [5/12, 9/12]
        rows_e.append([yr + 5.0 / 12.0, yr + 11.0 / 12.0])
        rows_w.append([0.68, 0.32])
        rows_c.append(c_nh)
        series_id.append(0)

    # Series 1: overlapping 5 yr and 10 yr IntCal20-style multi-year blocks
    for s_off, dur in ((1.0, 5.0), (2.0, 10.0), (6.0, 5.0), (8.0, 5.0)):
        s_blk = t0 + s_off
        mid = s_blk + 0.5 * dur
        rows_s.append([s_blk, mid])
        rows_e.append([mid, s_blk + dur])
        rows_w.append([0.5, 0.5])
        rows_c.append(c_nh)
        series_id.append(1)

    # Gap on [t0 + 13.0, t0 + 19.0] = [-5476.0, -5470.0]!

    # Series 2: SH [9/12, 14/12] rings + 3 simultaneous completions at t0 + 22.0
    for s_off in (19.0, 20.0):
        yr = t0 + s_off
        rows_s.append([yr + 9.0 / 12.0, yr + 11.0 / 12.0])
        rows_e.append([yr + 11.0 / 12.0, yr + 14.0 / 12.0])
        rows_w.append([0.42, 0.58])
        rows_c.append(c_sh)
        series_id.append(2)
    t_sim = t0 + 22.0  # -5467.0 CE
    rows_s.append([t_sim - 5.0 / 12.0, t_sim - 2.0 / 12.0])
    rows_e.append([t_sim - 2.0 / 12.0, t_sim])
    rows_w.append([0.45, 0.55])
    rows_c.append(c_sh)
    series_id.append(2)
    rows_s.append([t_sim - 2.0, t_sim - 1.0])
    rows_e.append([t_sim - 1.0, t_sim])
    rows_w.append([0.5, 0.5])
    rows_c.append(c_sh)
    series_id.append(2)
    rows_s.append([t_sim, t_sim])  # instantaneous point observation at t_sim
    rows_e.append([t_sim, t_sim])
    rows_w.append([0.5, 0.5])
    rows_c.append(c_sh)
    series_id.append(2)
    rows_s.append([t0 + 22.5, t0 + 23.25])
    rows_e.append([t0 + 23.25, t0 + 24.0])
    rows_w.append([0.5, 0.5])
    rows_c.append(c_sh)
    series_id.append(2)

    # Gap on [t0 + 24.0, t0 + 27.0] = [-5465.0, -5462.0]!

    # Series 3: Surface ocean / mixed marine multi-year blocks on [t0 + 27.0, t0 + 30.0]
    for s_off, e_off in ((27.0, 29.0), (27.5, 30.0), (28.0, 30.0)):
        s_blk = t0 + s_off
        e_blk = t0 + e_off
        mid = 0.5 * (s_blk + e_blk)
        rows_s.append([s_blk, mid])
        rows_e.append([mid, e_blk])
        rows_w.append([0.5, 0.5])
        rows_c.append(c_mar)
        series_id.append(3)

    starts = np.asarray(rows_s, dtype=np.float64)
    ends = np.asarray(rows_e, dtype=np.float64)
    weights = np.asarray(rows_w, dtype=np.float64)
    obs_matrix = np.asarray(rows_c, dtype=np.float64)
    s_idx = np.asarray(series_id, dtype=np.int32)
    n_obs = starts.shape[0]

    sigma_obs = 0.55 + 1.45 * rng.random(n_obs)
    resid = -1.35 + 2.65 * rng.standard_normal(n_obs)

    # 5-column nuisance design: global ambient baseline + 4 correlated per-series lab offsets
    global_ratio = np.where(s_idx == 3, 0.86, np.where(s_idx == 2, 0.996, 1.002))
    lab_one_hot = np.eye(4, dtype=np.float64)[s_idx]
    nuis_design = np.column_stack([global_ratio, lab_one_hot])
    nuis_prior_mean = np.array([-16.4, 0.65, -0.40, 0.25, -0.85], dtype=np.float64)
    # Non-diagonal lower-triangular nuisance prior root (correlated lab calibration standards)
    nuis_prior_root = np.array(
        [
            [85.0, 0.0, 0.0, 0.0, 0.0],
            [2.4, 1.35, 0.0, 0.0, 0.0],
            [1.8, 0.48, 1.15, 0.0, 0.0],
            [-1.5, -0.22, 0.31, 1.42, 0.0],
            [0.9, 0.15, -0.37, 0.29, 1.68],
        ],
        dtype=np.float64,
    )
    return {
        "t_grid": t_grid,
        "starts": starts,
        "ends": ends,
        "weights": weights,
        "obs_matrix": obs_matrix,
        "sigma_obs": sigma_obs,
        "resid": resid,
        "nuis_design": nuis_design,
        "nuis_prior_mean": nuis_prior_mean,
        "nuis_prior_root": nuis_prior_root,
    }


# =============================================================================
# 1. Exports, Helper Deduplication, Circular-Import Safety & Single Whitening
# =============================================================================


def test_exports_deduplication_clean_imports_and_single_whiten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert "KalmanGPResult" in tt_kalman.__all__
    assert "kalman_gp_marginal" in tt_kalman.__all__
    assert "KalmanGPResult" not in tt_linear.__all__
    assert "kalman_gp_marginal" not in tt_linear.__all__
    assert tt_linear.KalmanGPResult is KalmanGPResult
    assert tt_linear.kalman_gp_marginal is kalman_gp_marginal
    assert ticktack.kalman_gp_marginal is kalman_gp_marginal

    # Deduplicated helpers in linear.py point directly to ticktack._utils
    assert tt_linear._concrete is tt_utils._concrete
    assert tt_linear._safe_psd_root is tt_utils._safe_psd_root

    # Zero Any annotations in src/ticktack/kalman.py
    kalman_src = inspect.getsource(tt_kalman)
    assert "Any" not in kalman_src

    # Fresh Python interpreter imports succeed in BOTH orders (zero circular import)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_DIR)
    for mod in ("ticktack.kalman", "ticktack.linear"):
        proc = subprocess.run(
            [sys.executable, "-c", f"import {mod}; assert hasattr({mod}, 'kalman_gp_marginal')"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"Fresh import of {mod} failed: {proc.stderr}"

    # Verify _qr_whiten_and_posterior is invoked strictly ONCE during nuisance marginalization
    calls = 0
    orig_qr = tt_kalman._qr_whiten_and_posterior

    def _counted_qr(
        x: jax.Array, u: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        nonlocal calls
        calls += 1
        return orig_qr(x, u)

    monkeypatch.setattr(tt_kalman, "_qr_whiten_and_posterior", _counted_qr)
    a_mat, b_vec = _make_asymmetric_three_box_cbm()
    res = kalman_gp_marginal(
        a_mat,
        b_vec,
        tgq.Matern32(scale=3.2, sigma=1.4),
        jnp.array([1.2, -0.7, 0.5]),
        jnp.array([0.8, 1.1, 0.9]),
        jnp.array([0.0, 1.0, 2.0]),
        jnp.array([1.0, 2.0, 3.0]),
        dt=0.5,
        nuisance_design=jnp.ones((3, 1)),
        nuisance_prior_sd=25.0,
    )
    assert isinstance(res, KalmanGPResult)
    assert calls == 1


# =============================================================================
# 2. IntCal20-Scale Mixed-Resolution Dataset vs Dense SciPy Double-Integral Oracle
# =============================================================================


def test_scipy_window_pair_cov_matches_dblquad_quadrature() -> None:
    """Verify the SciPy matrix-exponential double integral directly against ``dblquad``."""
    a_mat, b_vec = _make_asymmetric_three_box_cbm()
    f_gp, q_sde, h_gp = _textbook_matern_sde("matern52", scale=3.1, sigma=1.55)
    f_joint = np.block([[a_mat, np.outer(1.73 * b_vec, h_gp)], [np.zeros((3, 3)), f_gp]])
    sig_inf = sla.solve_continuous_lyapunov(f_joint, -sla.block_diag(np.zeros((3, 3)), q_sde))
    sig_inf = 0.5 * (sig_inf + sig_inf.T)
    c1 = np.array([1.34, -0.21, 0.09, 0.0, 0.0, 0.0])
    c2 = np.array([0.18, 1.15, -0.33, 0.0, 0.0, 0.0])

    # Partial overlap [2.0, 7.0] vs [4.5, 11.0]
    s1, e1 = 2.0, 7.0
    s2, e2 = 4.5, 11.0
    val_exact = float(c1 @ _scipy_window_pair_cov(f_joint, sig_inf, s1, e1, s2, e2) @ c2)

    def _integrand(tp: float, t: float) -> float:
        cov_tt = (
            sla.expm(f_joint * (t - tp)) @ sig_inf
            if t >= tp
            else sig_inf @ sla.expm(f_joint.T * (tp - t))
        )
        return float(c1 @ cov_tt @ c2) / ((e1 - s1) * (e2 - s2))

    val_quad, _ = spi.dblquad(
        _integrand, s1, e1, lambda _: s2, lambda _: e2, epsabs=1e-13, epsrel=1e-13
    )
    np.testing.assert_allclose(val_exact, val_quad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("kind", "scale", "sigma_gp", "kernel"),
    [
        ("matern12", 4.6, 1.38, tgq.Exp(scale=4.6, sigma=1.38)),
        ("matern32", 3.9, 1.52, tgq.Matern32(scale=3.9, sigma=1.52)),
        ("matern52", 3.1, 1.27, tgq.Matern52(scale=3.1, sigma=1.27)),
        ("matern32", 4.2, 1.19, (1.19**2) * tgk.Matern32(scale=4.2)),
    ],
)
def test_intcal20_mixed_resolution_kalman_matches_dense_scipy_oracle(
    kind: str,
    scale: float,
    sigma_gp: float,
    kernel: tgk.Kernel,
) -> None:
    a_mat, b_vec = _make_asymmetric_three_box_cbm()
    ds = _build_intcal20_style_multiseries_dataset()
    alpha_unit = 1.73

    oracle = _dense_scipy_mvn_oracle(
        a_mat,
        b_vec,
        kind,
        scale,
        sigma_gp,
        ds["resid"],
        ds["sigma_obs"],
        ds["starts"],
        ds["ends"],
        ds["weights"],
        ds["t_grid"],
        alpha_unit=alpha_unit,
        obs_matrix=ds["obs_matrix"],
        nuisance_design=ds["nuis_design"],
        nuisance_prior_mean=ds["nuis_prior_mean"],
        nuisance_prior_root=ds["nuis_prior_root"],
    )

    kal = kalman_gp_marginal(
        jnp.asarray(a_mat),
        jnp.asarray(b_vec),
        kernel,
        jnp.asarray(ds["resid"]),
        jnp.asarray(ds["sigma_obs"]),
        jnp.asarray(ds["starts"]),
        jnp.asarray(ds["ends"]),
        jnp.asarray(ds["weights"]),
        t_grid=jnp.asarray(ds["t_grid"]),
        alpha_unit=alpha_unit,
        obs_matrix=jnp.asarray(ds["obs_matrix"]),
        nuisance_design=jnp.asarray(ds["nuis_design"]),
        nuisance_prior_mean=jnp.asarray(ds["nuis_prior_mean"]),
        nuisance_prior_root=jnp.asarray(ds["nuis_prior_root"]),
    )

    np.testing.assert_allclose(kal.loglike, oracle.loglike, rtol=1e-10, atol=1e-10)
    assert kal.nuisance_mean is not None and kal.nuisance_cov is not None
    np.testing.assert_allclose(kal.nuisance_mean, oracle.nuisance_mean, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(kal.nuisance_cov, oracle.nuisance_cov, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(kal.production_mean, oracle.production_mean, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(kal.production_var, oracle.production_var, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(kal.cbm_state_mean, oracle.cbm_state_mean, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(kal.cbm_state_var, oracle.cbm_state_var, rtol=1e-10, atol=1e-10)


def test_point_only_no_nuisance_and_box_indices_vs_scipy_oracle() -> None:
    """Exercise ``t_ends=None`` (``n_tracks == 0``), ``n_obs == n_boxes``, and ``box_indices``."""
    a_mat, b_vec = _make_asymmetric_three_box_cbm()
    t_pts = np.array([-5489.0, -5487.5, -5487.0, -5482.5, -5480.0], dtype=np.float64)
    box_idx = np.array([0, 1, 2, 0, 1], dtype=np.int32)
    box_scales = np.array([1.34, 1.29, 0.91], dtype=np.float64)
    obs_scale_per_obs = box_scales[box_idx]
    obs_mat = np.eye(3, dtype=np.float64)[box_idx] * obs_scale_per_obs[:, None]
    resid = np.array([1.9, -1.2, 0.65, -2.1, 1.3], dtype=np.float64)
    sigma_obs = np.array([0.7, 1.1, 0.85, 1.4, 0.95], dtype=np.float64)

    kal = kalman_gp_marginal(
        a_mat,
        b_vec,
        tgq.Matern52(scale=2.8, sigma=1.44),
        resid,
        sigma_obs,
        t_pts,
        t_ends=None,
        dt=0.5,
        alpha_unit=1.61,
        box_indices=box_idx,
        obs_scale=obs_scale_per_obs,
    )
    assert kal.nuisance_mean is None and kal.nuisance_cov is None
    oracle = _dense_scipy_mvn_oracle(
        a_mat,
        b_vec,
        "matern52",
        2.8,
        1.44,
        resid,
        sigma_obs,
        t_pts[:, None],
        t_pts[:, None],
        np.ones((5, 1), dtype=np.float64),
        np.asarray(kal.t_grid),
        alpha_unit=1.61,
        obs_matrix=obs_mat,
    )
    np.testing.assert_allclose(kal.loglike, oracle.loglike, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(kal.production_mean, oracle.production_mean, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(kal.production_var, oracle.production_var, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(kal.cbm_state_mean, oracle.cbm_state_mean, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(kal.cbm_state_var, oracle.cbm_state_var, rtol=1e-11, atol=1e-11)

    # Explicit n_obs == n_boxes == 3 case with permuted box_indices = [1, 2, 0]
    # and starkly unequal scales (1.34 vs 0.028 deep-reservoir style):
    idx_perm = np.array([1, 2, 0], dtype=np.int32)
    scales_3 = np.array([1.34, 1.29, 0.028], dtype=np.float64)
    scale_perm = scales_3[idx_perm]
    t3 = np.array([-5489.0, -5488.0, -5487.0], dtype=np.float64)
    kal_eq = kalman_gp_marginal(
        a_mat,
        b_vec,
        tgq.Matern32(scale=3.2, sigma=1.35),
        resid[:3],
        sigma_obs[:3],
        t3,
        t3 + 0.5,
        dt=0.5,
        alpha_unit=1.61,
        box_indices=idx_perm,
        obs_scale=scale_perm,
    )
    oracle_eq = _dense_scipy_mvn_oracle(
        a_mat,
        b_vec,
        "matern32",
        3.2,
        1.35,
        resid[:3],
        sigma_obs[:3],
        t3[:, None],
        (t3 + 0.5)[:, None],
        np.ones((3, 1), dtype=np.float64),
        np.asarray(kal_eq.t_grid),
        alpha_unit=1.61,
        obs_matrix=np.eye(3, dtype=np.float64)[idx_perm] * scale_perm[:, None],
    )
    np.testing.assert_allclose(kal_eq.loglike, oracle_eq.loglike, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        kal_eq.cbm_state_mean, oracle_eq.cbm_state_mean, rtol=1e-11, atol=1e-11
    )


def test_multimillennial_intcal20_step_dt_alignment_and_auto_grid() -> None:
    """Verify default ``dt = 1/24`` yr across bundled ``[-5489, 1749]`` and 55-kyr IntCal20."""
    # 1. Full 55-kyr IntCal20 [-53050, 1950] CE (1,320,000 steps at 1/24 yr):
    # test both auto-grid and explicit t_grid step_dt recovery with NH/SH monthly rings!
    starts_55k = np.array([[-53050.0], [-53049.0 + 4.0 / 12.0], [1948.0 + 9.0 / 12.0], [1949.0]])
    ends_55k = np.array([[-53040.0], [-53049.0 + 8.0 / 12.0], [1948.0 + 14.0 / 12.0], [1950.0]])
    weights_55k = np.ones((4, 1), dtype=np.float64)
    grid_auto, dt_auto = _resolve_kalman_time_grid(None, starts_55k, ends_55k, None)
    assert grid_auto.shape == (1_320_001,)
    assert dt_auto == pytest.approx(1.0 / 24.0, rel=1e-15)
    tracks_auto = _assign_kalman_tracks(grid_auto, starts_55k, ends_55k, weights_55k, dt_auto)
    assert tracks_auto.obs_e_max[-1] == 1_320_000

    # Explicit t_grid (where step_dt must be recovered without t_grid[1] - t_grid[0] cancellation)
    grid_exp, dt_exp = _resolve_kalman_time_grid(grid_auto, starts_55k, ends_55k, None)
    tracks_exp = _assign_kalman_tracks(grid_exp, starts_55k, ends_55k, weights_55k, dt_exp)
    np.testing.assert_array_equal(tracks_exp.obs_s_min, tracks_auto.obs_s_min)
    np.testing.assert_array_equal(tracks_exp.obs_e_max, tracks_auto.obs_e_max)

    # 2. End-to-end kalman_gp_marginal with default t_grid=None, dt=None across bundled
    # IntCal20's full [-5489.0, 1749.0] CE span (173,712 steps at 1/24 yr)
    a_1box = jnp.array([[-0.12 - LAMBDA_14]], dtype=jnp.float64)
    b_1box = jnp.array([1.0], dtype=jnp.float64)
    ts_ic = jnp.array([-5489.0, -5488.0 + 4.0 / 12.0, 1747.0 + 9.0 / 12.0, 1748.0])
    te_ic = jnp.array([-5479.0, -5488.0 + 8.0 / 12.0, 1747.0 + 14.0 / 12.0, 1749.0])
    res_ic = jnp.array([12.0, 9.5, -4.2, -5.1], dtype=jnp.float64)
    sig_ic = jnp.array([1.8, 1.2, 1.1, 1.3], dtype=jnp.float64)
    kal_ic = kalman_gp_marginal(
        a_1box,
        b_1box,
        tgq.Exp(scale=8.0, sigma=0.4),
        res_ic,
        sig_ic,
        ts_ic,
        te_ic,
    )
    assert kal_ic.t_grid.shape == (173_713,)
    oracle_ic = _dense_scipy_mvn_oracle(
        np.asarray(a_1box),
        np.asarray(b_1box),
        "matern12",
        8.0,
        0.4,
        np.asarray(res_ic),
        np.asarray(sig_ic),
        np.asarray(ts_ic)[:, None],
        np.asarray(te_ic)[:, None],
        np.ones((4, 1), dtype=np.float64),
        np.array([-5489.0, 1749.0], dtype=np.float64),
        alpha_unit=1.0,
        obs_matrix=np.ones((4, 1), dtype=np.float64),
    )
    np.testing.assert_allclose(kal_ic.loglike, oracle_ic.loglike, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        kal_ic.production_mean[jnp.array([0, -1])],
        oracle_ic.production_mean,
        rtol=1e-10,
        atol=1e-10,
    )


def test_zero_padded_sub_weights_and_high_offset_qr_cancellation() -> None:
    """Verify zero-weight sub-window padding and +400‰ uncentred QR cancellation accuracy."""
    a_mat, b_vec = _make_asymmetric_three_box_cbm()
    kern = tgq.Matern32(scale=3.5, sigma=1.25)

    # 1. Mixed M_sub series padded with w_m == 0.0 (including zero-width [s, s] dummy slots
    # and out-of-range dummy slots that must NOT widen tracks or trigger ValueError)
    starts_pad = np.array(
        [
            [-5489.0, -5488.5, -5489.0],  # slot 2 is zero-width zero-weight padding
            [-5488.0, 9999.0, 9999.0],  # slots 1..2 are zero-weight dummy timestamps
            [-5487.0, -5487.0, -5487.0],  # point obs in slot 0, zero-weight in slots 1..2
        ],
        dtype=np.float64,
    )
    ends_pad = np.array(
        [
            [-5488.5, -5488.0, -5489.0],
            [-5487.0, 10005.0, 10005.0],
            [-5487.0, -5487.0, -5487.0],
        ],
        dtype=np.float64,
    )
    weights_pad = np.array(
        [
            [0.6, 0.4, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    resid_pad = np.array([1.4, -0.9, 0.7], dtype=np.float64)
    sig_pad = np.array([0.8, 1.1, 0.9], dtype=np.float64)
    obs_pad = np.array([[1.34, 0.0, 0.0], [0.0, 1.29, 0.0], [0.14, 0.08, 0.87]])

    kal_pad = kalman_gp_marginal(
        a_mat,
        b_vec,
        kern,
        resid_pad,
        sig_pad,
        starts_pad,
        ends_pad,
        weights_pad,
        dt=0.5,
        obs_matrix=obs_pad,
    )
    assert kal_pad.t_grid.shape == (5,)  # [-5489.0, -5488.5, -5488.0, -5487.5, -5487.0]

    # Unpadded active windows are disjoint on [-5489, -5488] and [-5488, -5487] -> 1 track!
    sched_pad = _build_seasonal_kalman_schedule(
        np.asarray(kal_pad.t_grid),
        starts_pad,
        ends_pad,
        weights_pad,
        obs_pad,
        0.5,
        d_gp=2,
    )
    assert sched_pad.n_tracks == 1

    starts_clean = np.array([[-5489.0, -5488.5], [-5488.0, -5487.5], [-5487.0, -5487.0]])
    ends_clean = np.array([[-5488.5, -5488.0], [-5487.5, -5487.0], [-5487.0, -5487.0]])
    weights_clean = np.array([[0.6, 0.4], [0.5, 0.5], [0.5, 0.5]])
    oracle_pad = _dense_scipy_mvn_oracle(
        a_mat,
        b_vec,
        "matern32",
        3.5,
        1.25,
        resid_pad,
        sig_pad,
        starts_clean,
        ends_clean,
        weights_clean,
        np.asarray(kal_pad.t_grid),
        alpha_unit=1.0,
        obs_matrix=obs_pad,
    )
    np.testing.assert_allclose(kal_pad.loglike, oracle_pad.loglike, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        kal_pad.production_mean, oracle_pad.production_mean, rtol=1e-12, atol=1e-12
    )

    # 2. Large uncentred glacial offset (+400‰) with small sigma_gp = 1e-4 and diffuse prior
    rng = np.random.default_rng(7802)
    n_glac = 40
    t_s_g = -12000.0 + np.arange(n_glac, dtype=np.float64)
    t_e_g = t_s_g + 1.0
    sig_g = 0.5 + 0.3 * rng.random(n_glac)
    resid_g = 400.0 + rng.standard_normal(n_glac) * sig_g
    nuis_x_g = np.column_stack(
        [
            np.ones(n_glac),
            (np.arange(n_glac) < 20).astype(float),
            (np.arange(n_glac) >= 20).astype(float),
        ]
    )
    nuis_sd_g = np.array([200.0, 1.5, 1.5], dtype=np.float64)
    obs_g = np.broadcast_to(np.array([[1.34, 0.0, 0.0]]), (n_glac, 3))

    @jax.jit
    def _glac_ll(ls_in: jax.Array, lsig_in: jax.Array) -> jax.Array:
        return kalman_gp_marginal(
            a_mat,
            b_vec,
            tgq.Matern32(scale=jnp.exp(ls_in), sigma=jnp.exp(lsig_in)),
            resid_g,
            sig_g,
            t_s_g,
            t_e_g,
            dt=0.5,
            obs_matrix=obs_g,
            nuisance_design=nuis_x_g,
            nuisance_prior_mean=0.0,
            nuisance_prior_sd=nuis_sd_g,
        ).loglike

    ls_g0 = jnp.asarray(math.log(4.0), dtype=jnp.float64)
    lsig_g0 = jnp.asarray(math.log(1e-4), dtype=jnp.float64)
    kal_g = kalman_gp_marginal(
        a_mat,
        b_vec,
        tgq.Matern32(scale=4.0, sigma=1e-4),
        resid_g,
        sig_g,
        t_s_g,
        t_e_g,
        dt=0.5,
        obs_matrix=obs_g,
        nuisance_design=nuis_x_g,
        nuisance_prior_mean=0.0,
        nuisance_prior_sd=nuis_sd_g,
    )
    oracle_g = _dense_scipy_mvn_oracle(
        a_mat,
        b_vec,
        "matern32",
        4.0,
        1e-4,
        resid_g,
        sig_g,
        t_s_g[:, None],
        t_e_g[:, None],
        np.ones((n_glac, 1)),
        np.asarray(kal_g.t_grid),
        alpha_unit=1.0,
        obs_matrix=obs_g,
        nuisance_design=nuis_x_g,
        nuisance_prior_mean=np.zeros(3),
        nuisance_prior_root=np.diag(nuis_sd_g),
    )
    np.testing.assert_allclose(kal_g.loglike, oracle_g.loglike, rtol=1e-12, atol=1e-12)
    assert kal_g.nuisance_mean is not None
    np.testing.assert_allclose(kal_g.nuisance_mean, oracle_g.nuisance_mean, rtol=1e-12, atol=1e-12)
    g_ls_tiny, g_lsig_tiny = jax.grad(_glac_ll, argnums=(0, 1))(ls_g0, lsig_g0)
    fd_ls_tiny = _fourth_order_fd_array(lambda x: _glac_ll(x, lsig_g0), ls_g0, step=5e-2)
    fd_lsig_tiny = _fourth_order_fd_array(lambda x: _glac_ll(ls_g0, x), lsig_g0, step=5e-2)
    np.testing.assert_allclose(g_ls_tiny, fd_ls_tiny, rtol=2e-2, atol=1e-11)
    np.testing.assert_allclose(g_lsig_tiny, fd_lsig_tiny, rtol=1e-3, atol=1e-11)

    # At sigma_gp = 0.25 on the same +400‰ uncentred offset (||all_sin||^2 / quad > 10^6),
    # hyperparameter gradients are O(10^-2..10^-1) and match 4th-order FD to rtol=1e-7:
    lsig_g1 = jnp.asarray(math.log(0.25), dtype=jnp.float64)
    g_ls, g_lsig = jax.grad(_glac_ll, argnums=(0, 1))(ls_g0, lsig_g1)
    fd_ls = _fourth_order_fd_array(lambda x: _glac_ll(x, lsig_g1), ls_g0, step=1e-2)
    fd_lsig = _fourth_order_fd_array(lambda x: _glac_ll(ls_g0, x), lsig_g1, step=1e-2)
    np.testing.assert_allclose(g_ls, fd_ls, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(g_lsig, fd_lsig, rtol=1e-7, atol=1e-9)


# =============================================================================
# 3. Autodiff Verification vs 4th-Order Central Finite Differences
# =============================================================================


def _fourth_order_fd_array(fn, x0: jax.Array, step: float = 1e-4) -> np.ndarray:
    """Evaluate 4th-order central finite differences componentwise over ``x0``."""
    x_np = np.asarray(x0, dtype=np.float64)
    grad_fd = np.zeros_like(x_np)
    for idx in np.ndindex(x_np.shape):
        e = np.zeros_like(x_np)
        e[idx] = step
        ej = jnp.asarray(e, dtype=jnp.float64)
        f_p2 = float(fn(x0 + 2.0 * ej))
        f_p1 = float(fn(x0 + ej))
        f_m1 = float(fn(x0 - ej))
        f_m2 = float(fn(x0 - 2.0 * ej))
        grad_fd[idx] = (-f_p2 + 8.0 * f_p1 - 8.0 * f_m1 + f_m2) / (12.0 * step)
    return grad_fd


def test_kalman_gp_marginal_autodiff_matches_fourth_order_fd() -> None:
    """Verify ``jax.grad`` wrt ``(A, b, log_scale, log_sigma_gp, sigma_obs)`` and smoothers."""
    a0 = jnp.array(
        [
            [-(1.42 + LAMBDA_14), 0.53],
            [1.18, -(0.71 + LAMBDA_14)],
        ],
        dtype=jnp.float64,
    )
    b0 = jnp.array([0.63, 0.37], dtype=jnp.float64)
    log_scale0 = jnp.asarray(math.log(3.35), dtype=jnp.float64)
    log_sigma_gp0 = jnp.asarray(math.log(1.48), dtype=jnp.float64)
    sigma_obs0 = jnp.array([0.72, 1.18, 0.64, 1.35, 0.89, 1.07], dtype=jnp.float64)

    starts = jnp.array(
        [[0.0, 0.5], [0.5, 1.0], [1.0, 3.5], [2.0, 4.5], [7.0, 7.0], [7.5, 8.5]],
        dtype=jnp.float64,
    )
    ends = jnp.array(
        [[0.5, 1.0], [1.0, 1.5], [3.5, 6.0], [4.5, 7.0], [7.0, 7.0], [8.5, 9.5]],
        dtype=jnp.float64,
    )
    weights = jnp.array(
        [[0.65, 0.35], [0.4, 0.6], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
        dtype=jnp.float64,
    )
    obs_mat = jnp.array(
        [[1.31, 0.0], [1.31, 0.0], [1.31, 0.0], [0.25, 1.08], [0.0, 1.22], [0.92, 0.31]],
        dtype=jnp.float64,
    )
    resid = jnp.array([2.35, -1.62, 0.88, -2.74, 1.91, -0.53], dtype=jnp.float64)
    nuis_x = jnp.array(
        [[1.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.96, 0.0], [0.98, 1.0], [1.02, 0.0]],
        dtype=jnp.float64,
    )
    nuis_mean = jnp.array([-11.5, 0.45], dtype=jnp.float64)
    nuis_root = jnp.array([[35.0, 0.0], [2.1, 1.35]], dtype=jnp.float64)

    @jax.jit
    def _eval_kalman(
        a_in: jax.Array,
        b_in: jax.Array,
        ls_in: jax.Array,
        lsig_in: jax.Array,
        sobs_in: jax.Array,
    ) -> KalmanGPResult:
        kern = tgq.Matern52(scale=jnp.exp(ls_in), sigma=jnp.exp(lsig_in))
        return kalman_gp_marginal(
            a_in,
            b_in,
            kern,
            resid,
            sobs_in,
            starts,
            ends,
            weights,
            dt=0.5,
            alpha_unit=1.67,
            obs_matrix=obs_mat,
            nuisance_design=nuis_x,
            nuisance_prior_mean=nuis_mean,
            nuisance_prior_root=nuis_root,
        )

    def _ll_fn(
        a_in: jax.Array,
        b_in: jax.Array,
        ls_in: jax.Array,
        lsig_in: jax.Array,
        sobs_in: jax.Array,
    ) -> jax.Array:
        return _eval_kalman(a_in, b_in, ls_in, lsig_in, sobs_in).loglike

    grads_ad = jax.grad(_ll_fn, argnums=(0, 1, 2, 3, 4))(
        a0, b0, log_scale0, log_sigma_gp0, sigma_obs0
    )
    fd_a = _fourth_order_fd_array(
        lambda x: _ll_fn(x, b0, log_scale0, log_sigma_gp0, sigma_obs0), a0, step=1e-4
    )
    fd_b = _fourth_order_fd_array(
        lambda x: _ll_fn(a0, x, log_scale0, log_sigma_gp0, sigma_obs0), b0, step=1e-4
    )
    fd_ls = _fourth_order_fd_array(
        lambda x: _ll_fn(a0, b0, x, log_sigma_gp0, sigma_obs0), log_scale0, step=1e-4
    )
    fd_lsig = _fourth_order_fd_array(
        lambda x: _ll_fn(a0, b0, log_scale0, x, sigma_obs0), log_sigma_gp0, step=1e-4
    )
    fd_sobs = _fourth_order_fd_array(
        lambda x: _ll_fn(a0, b0, log_scale0, log_sigma_gp0, x), sigma_obs0, step=1e-4
    )

    for g_ad, g_fd in zip(grads_ad, (fd_a, fd_b, fd_ls, fd_lsig, fd_sobs)):
        assert np.linalg.norm(np.asarray(g_ad)) > 1e-3
        np.testing.assert_allclose(g_ad, g_fd, rtol=1e-7, atol=1e-9)

    # Also verify smooth autodiff through RTS smoothed posterior means & variances
    w_t = jnp.linspace(-0.6, 1.1, 20, dtype=jnp.float64)

    def _smooth_scalar(ls_in: jax.Array, lsig_in: jax.Array) -> jax.Array:
        out = _eval_kalman(a0, b0, ls_in, lsig_in, sigma_obs0)
        assert out.nuisance_mean is not None and out.nuisance_cov is not None
        return (
            jnp.dot(w_t, out.production_mean)
            + 0.4 * jnp.dot(w_t**2, out.production_var)
            + 0.3 * jnp.sum(out.cbm_state_mean * w_t[:, None])
            + 0.2 * jnp.sum(out.cbm_state_var)
            + 0.5 * jnp.sum(out.nuisance_mean)
            + 0.1 * jnp.trace(out.nuisance_cov)
        )

    gs_ls, gs_lsig = jax.grad(_smooth_scalar, argnums=(0, 1))(log_scale0, log_sigma_gp0)
    fds_ls = _fourth_order_fd_array(lambda x: _smooth_scalar(x, log_sigma_gp0), log_scale0)
    fds_lsig = _fourth_order_fd_array(lambda x: _smooth_scalar(log_scale0, x), log_sigma_gp0)
    np.testing.assert_allclose(gs_ls, fds_ls, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(gs_lsig, fds_lsig, rtol=1e-7, atol=1e-9)


# =============================================================================
# 4. Private Helper Kernels, Input Validation & Negative Controls
# =============================================================================


def test_private_kalman_helpers_and_negative_controls() -> None:
    """Test helper kernels directly and prove omitting ``half_logdet_nuis`` or ``cov_omega_nuis`` fails."""
    a_mat, b_vec = _make_asymmetric_three_box_cbm()
    ds = _build_intcal20_style_multiseries_dataset()
    alpha_unit = 1.73
    scale, sigma_gp = 3.9, 1.52
    kernel = tgq.Matern32(scale=scale, sigma=sigma_gp)

    t_grid_np, step_dt = _resolve_kalman_time_grid(ds["t_grid"], ds["starts"], ds["ends"], None)
    assert step_dt == pytest.approx(1.0 / 24.0)
    tracks = _assign_kalman_tracks(t_grid_np, ds["starts"], ds["ends"], ds["weights"], step_dt)
    assert tracks.n_tracks >= 3  # overlapping 5 yr and 10 yr blocks need >= 3 tracks
    assert np.any(tracks.obs_track == -1)  # point observation uses no track

    schedule = _build_seasonal_kalman_schedule(
        t_grid_np, ds["starts"], ds["ends"], ds["weights"], ds["obs_matrix"], step_dt, d_gp=2
    )
    assert schedule.max_simul >= 3
    nuis = _prepare_kalman_nuisance(
        jnp.asarray(ds["resid"]),
        ds["nuis_design"],
        ds["nuis_prior_mean"],
        None,
        ds["nuis_prior_root"],
    )
    sde_m32 = tt_linear._try_extract_sde(kernel)
    assert sde_m32 is not None
    f_gp, q_sde, _, h_gp = sde_m32
    dynamics = _van_loan_kalman_step_dynamics(
        jnp.asarray(a_mat),
        alpha_unit * jnp.asarray(b_vec),
        f_gp,
        q_sde,
        h_gp,
        step_dt,
        has_tracks=True,
    )
    phi_0, sig_0 = _build_kalman_step_matrices(jnp.int32(0), schedule, dynamics)
    assert phi_0.shape == sig_0.shape == (tracks.n_tracks + 5, tracks.n_tracks + 5)

    # Inactive _joseph_scalar_update leaves mean, cov, m_extra untouched and delta_log_norm == 0
    upd_idle = _joseph_scalar_update(
        jnp.ones(4),
        jnp.eye(4) * 2.0,
        jnp.ones(4),
        jnp.asarray(5.0),
        jnp.asarray(0.5),
        jnp.asarray(False),
        jnp.ones((5, 2)),
    )
    np.testing.assert_array_equal(upd_idle.mean, np.ones(4))
    assert float(upd_idle.delta_log_norm) == 0.0

    _, _, sigma_xi_inf = tt_linear.stationary_cbm_gp_cov(
        a_mat, b_vec, kernel, alpha_unit=alpha_unit, return_full_state=True
    )
    fwd = _run_kalman_forward_pass(
        schedule,
        dynamics,
        sigma_xi_inf,
        nuis.resid_shifted,
        jnp.asarray(ds["sigma_obs"] ** 2),
        nuis.nuis_u_raw,
    )
    marg = _marginalize_kalman_nuisance(fwd, nuis)
    m_smooth, p_smooth = _run_kalman_rts_smoother(fwd, schedule, marg)
    assert m_smooth.shape == (t_grid_np.shape[0], tracks.n_tracks + 5)

    oracle = _dense_scipy_mvn_oracle(
        a_mat,
        b_vec,
        "matern32",
        scale,
        sigma_gp,
        ds["resid"],
        ds["sigma_obs"],
        ds["starts"],
        ds["ends"],
        ds["weights"],
        ds["t_grid"],
        alpha_unit=alpha_unit,
        obs_matrix=ds["obs_matrix"],
        nuisance_design=ds["nuis_design"],
        nuisance_prior_mean=ds["nuis_prior_mean"],
        nuisance_prior_root=ds["nuis_prior_root"],
    )

    # Negative control 1: omitting half_logdet_nuis makes the loglike parity assertion FAIL!
    _, half_ld, _, _ = _qr_whiten_and_posterior(
        fwd.sin_all.reshape(-1), fwd.sex_all.reshape(-1, nuis.p_nuis)
    )
    ll_missing_logdet = marg.loglike + half_ld
    assert abs(float(ll_missing_logdet) - oracle.loglike) > 2.0
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(ll_missing_logdet, oracle.loglike, rtol=1e-10, atol=1e-10)

    # Negative control 2: omitting RTS cov_omega_nuis makes cbm_state_var parity FAIL!
    marg_no_cov = marg._replace(cov_omega_nuis=jnp.zeros_like(marg.cov_omega_nuis))
    _, p_smooth_bad = _run_kalman_rts_smoother(fwd, schedule, marg_no_cov)
    cbm_slice = slice(tracks.n_tracks, tracks.n_tracks + 3)
    cbm_var_bad = jnp.diagonal(p_smooth_bad[:, cbm_slice, cbm_slice], axis1=1, axis2=2)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(cbm_var_bad, oracle.cbm_state_var, rtol=1e-10, atol=1e-10)


def test_kalman_gp_marginal_input_validation_errors() -> None:
    a_mat, b_vec = _make_asymmetric_three_box_cbm()
    kern = tgq.Matern32(scale=3.0, sigma=1.2)
    resid = jnp.array([1.0, -0.5, 0.8], dtype=jnp.float64)
    sigma = jnp.array([0.9, 1.1, 0.8], dtype=jnp.float64)
    t_s = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float64)
    t_e = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)

    # 1. Non-Markov kernel
    with pytest.raises(ValueError, match="Markov / quasisep"):
        kalman_gp_marginal(a_mat, b_vec, tgk.ExpSquared(2.5), resid, sigma, t_s, t_e)

    # 2. Traced t_starts / t_grid / dt under jax.jit
    with pytest.raises(ValueError, match="concrete"):
        jax.jit(lambda ts: kalman_gp_marginal(a_mat, b_vec, kern, resid, sigma, ts, t_e).loglike)(
            t_s
        )
    with pytest.raises(ValueError, match="concrete"):
        jax.jit(
            lambda tg: (
                kalman_gp_marginal(a_mat, b_vec, kern, resid, sigma, t_s, t_e, t_grid=tg).loglike
            )
        )(jnp.array([0.0, 1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="concrete"):
        jax.jit(
            lambda step: (
                kalman_gp_marginal(a_mat, b_vec, kern, resid, sigma, t_s, t_e, dt=step).loglike
            )
        )(jnp.asarray(0.5))

    # 3. <2-point, non-uniform, decreasing, non-finite t_grid, bad dt, misaligned windows, t_ends < t_starts
    with pytest.raises(ValueError, match="at least 2 points"):
        kalman_gp_marginal(a_mat, b_vec, kern, resid, sigma, t_s, t_e, t_grid=jnp.array([0.0]))
    with pytest.raises(ValueError, match="uniformly spaced"):
        kalman_gp_marginal(
            a_mat, b_vec, kern, resid, sigma, t_s, t_e, t_grid=jnp.array([0.0, 1.0, 2.5, 3.0])
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        kalman_gp_marginal(
            a_mat, b_vec, kern, resid, sigma, t_s, t_e, t_grid=jnp.array([3.0, 2.0, 1.0, 0.0])
        )
    with pytest.raises(ValueError, match="finite"):
        _resolve_kalman_time_grid(
            np.array([0.0, np.nan, 2.0]), np.asarray(t_s)[:, None], np.asarray(t_e)[:, None], None
        )
    with pytest.raises(ValueError, match="At least one"):
        _resolve_kalman_time_grid(None, np.zeros((0, 1)), np.zeros((0, 1)), 0.5)
    with pytest.raises(ValueError, match="t_ends >= t_starts"):
        _resolve_kalman_time_grid(None, np.array([[2.0]]), np.array([[1.0]]), 0.5)
    with pytest.raises(ValueError, match="finite"):
        _resolve_kalman_time_grid(None, np.array([[np.inf]]), np.array([[2.0]]), 0.5)
    with pytest.raises(ValueError, match="dt must be finite and strictly positive"):
        kalman_gp_marginal(a_mat, b_vec, kern, resid, sigma, t_s, t_e, dt=-0.25)
    with pytest.raises(ValueError, match="not aligned"):
        kalman_gp_marginal(
            a_mat, b_vec, kern, resid, sigma, t_s + 0.13, t_e, t_grid=jnp.arange(0.0, 4.0, 0.5)
        )
    with pytest.raises(ValueError, match="Mixed zero-width and positive-width"):
        kalman_gp_marginal(
            a_mat,
            b_vec,
            kern,
            jnp.array([1.0]),
            jnp.array([0.9]),
            jnp.array([[0.0, 1.0]]),
            jnp.array([[0.0, 2.0]]),
            dt=0.5,
        )
    with pytest.raises(ValueError, match="t_ends >= t_starts"):
        kalman_gp_marginal(a_mat, b_vec, kern, resid, sigma, t_e, t_s)

    # 4. Bad resid / sigma
    with pytest.raises(ValueError, match="resid length"):
        kalman_gp_marginal(a_mat, b_vec, kern, jnp.ones(2), sigma, t_s, t_e)
    with pytest.raises(ValueError, match="sigma"):
        kalman_gp_marginal(a_mat, b_vec, kern, resid, jnp.eye(3), t_s, t_e)
    with pytest.raises(ValueError, match="sigma must be finite and strictly positive"):
        kalman_gp_marginal(a_mat, b_vec, kern, resid, jnp.array([0.9, 0.0, 0.8]), t_s, t_e)

    # 5. Nuisance design & prior validation (1D, wrong rows, wrong root shape, both/neither)
    with pytest.raises(ValueError, match="nuisance_design must be a 2D"):
        kalman_gp_marginal(
            a_mat,
            b_vec,
            kern,
            resid,
            sigma,
            t_s,
            t_e,
            nuisance_design=jnp.ones(3),
            nuisance_prior_sd=10.0,
        )
    with pytest.raises(ValueError, match="nuisance_design must be a 2D"):
        kalman_gp_marginal(
            a_mat,
            b_vec,
            kern,
            resid,
            sigma,
            t_s,
            t_e,
            nuisance_design=jnp.ones((4, 2)),
            nuisance_prior_sd=10.0,
        )
    with pytest.raises(ValueError, match="nuisance_prior_root must have shape"):
        kalman_gp_marginal(
            a_mat,
            b_vec,
            kern,
            resid,
            sigma,
            t_s,
            t_e,
            nuisance_design=jnp.ones((3, 2)),
            nuisance_prior_root=jnp.ones((2,)),
        )
    with pytest.raises(ValueError, match="nuisance_prior_root must have shape"):
        kalman_gp_marginal(
            a_mat,
            b_vec,
            kern,
            resid,
            sigma,
            t_s,
            t_e,
            nuisance_design=jnp.ones((3, 2)),
            nuisance_prior_root=jnp.ones((2, 3)),
        )
    with pytest.raises(ValueError, match="exactly one of nuisance_prior_sd or nuisance_prior_root"):
        kalman_gp_marginal(
            a_mat, b_vec, kern, resid, sigma, t_s, t_e, nuisance_design=jnp.ones((3, 2))
        )
    with pytest.raises(ValueError, match="exactly one of nuisance_prior_sd or nuisance_prior_root"):
        kalman_gp_marginal(
            a_mat,
            b_vec,
            kern,
            resid,
            sigma,
            t_s,
            t_e,
            nuisance_design=jnp.ones((3, 2)),
            nuisance_prior_sd=10.0,
            nuisance_prior_root=jnp.eye(2),
        )
    with pytest.raises(ValueError, match="require nuisance_design"):
        kalman_gp_marginal(a_mat, b_vec, kern, resid, sigma, t_s, t_e, nuisance_prior_sd=10.0)
    with pytest.raises(ValueError, match="nuisance_prior_sd must be finite and strictly positive"):
        kalman_gp_marginal(
            a_mat,
            b_vec,
            kern,
            resid,
            sigma,
            t_s,
            t_e,
            nuisance_design=jnp.ones((3, 1)),
            nuisance_prior_sd=-1.0,
        )

    # 6. Entry-point (A, b), box_indices, obs_scale, m_sub == 0, sub_weights, and drift t_grid
    with pytest.raises(ValueError, match="A must be a non-empty 2D square"):
        kalman_gp_marginal(jnp.ones(3), b_vec, kern, resid, sigma, t_s, t_e)
    with pytest.raises(ValueError, match="A must be a non-empty 2D square"):
        kalman_gp_marginal(jnp.ones((2, 3)), b_vec, kern, resid, sigma, t_s, t_e)
    with pytest.raises(ValueError, match="A must be a non-empty 2D square"):
        kalman_gp_marginal(jnp.zeros((0, 0)), jnp.zeros((0,)), kern, resid, sigma, t_s, t_e)
    with pytest.raises(ValueError, match="b must be a 1D array"):
        kalman_gp_marginal(a_mat, jnp.ones((3, 1)), kern, resid, sigma, t_s, t_e)
    with pytest.raises(ValueError, match="b must be a 1D array"):
        kalman_gp_marginal(a_mat, jnp.ones(2), kern, resid, sigma, t_s, t_e)
    with pytest.raises(ValueError, match="obs_scale must be a scalar or 1D array of length"):
        kalman_gp_marginal(
            a_mat,
            b_vec,
            kern,
            jnp.ones(4),
            jnp.ones(4),
            jnp.arange(4.0),
            jnp.arange(1.0, 5.0),
            obs_scale=jnp.ones(3),  # n_boxes=3 != n_obs=4 rejected!
        )
    with pytest.raises(ValueError, match="box_indices"):
        kalman_gp_marginal(
            a_mat, b_vec, kern, resid, sigma, t_s, t_e, box_indices=jnp.array([0, 3, 1])
        )
    with pytest.raises(ValueError, match="box_indices"):
        kalman_gp_marginal(
            a_mat, b_vec, kern, resid, sigma, t_s, t_e, box_indices=jnp.array([0.5, 1.0, 0.0])
        )
    with pytest.raises(ValueError, match="m_sub >= 1"):
        kalman_gp_marginal(a_mat, b_vec, kern, resid, sigma, np.zeros((3, 0)), np.zeros((3, 0)))
    with pytest.raises(ValueError, match="non-negative"):
        kalman_gp_marginal(
            a_mat,
            b_vec,
            kern,
            resid,
            sigma,
            t_s[:, None],
            t_e[:, None],
            sub_weights=np.array([-0.2]),
        )
    with pytest.raises(ValueError, match="strictly positive total weight"):
        kalman_gp_marginal(
            a_mat,
            b_vec,
            kern,
            resid,
            sigma,
            np.column_stack([t_s, t_s + 0.5]),
            np.column_stack([t_s + 0.5, t_e]),
            sub_weights=np.array([[0.5, 0.5], [0.0, 0.0], [1.0, 0.0]]),
            dt=0.5,
        )
    with pytest.raises(ValueError, match="finite"):
        kalman_gp_marginal(
            a_mat,
            b_vec,
            kern,
            resid,
            sigma,
            t_s[:, None],
            t_e[:, None],
            sub_weights=np.array([np.nan]),
        )
    # Cumulative drift > 1e-9 across steps despite pairwise diffs within 1e-9
    drifted_grid = np.arange(0.0, 4.0, 0.5) + 5e-10 * (np.arange(8) ** 2)
    with pytest.raises(ValueError, match="uniformly spaced"):
        kalman_gp_marginal(a_mat, b_vec, kern, resid, sigma, t_s, t_e, t_grid=drifted_grid)
