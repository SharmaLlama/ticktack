"""Independent numerical reference tests (Neon oracle suite) for ``ticktack 2.0``.

Verifies four core guarantees against genuinely independent NumPy/SciPy
implementations and literature paper transcriptions:

1. **Full-spectrum Fisher matrix vs central-difference ``J^T Sigma^-1 J``**:
   ``expected_fisher`` and ``cbm_fisher_spectrum`` (unmarginalized and
   Schur-marginalized over nuisance parameters ``psi``) against pure NumPy
   Richardson-extrapolated central-difference Jacobians and explicit Schur
   complements, checking all matrix entries, all eigenvalues, and full
   eigenvector/subspace alignment.
2. **Preset flux and mass tables vs independent paper transcriptions**:
   ``Guttler15``, ``Brehm21``, ``Buntgen18``, and ``Miyake17`` checked against
   keyed literature tables (box counts, masses, total inventory, production
   fractions, directed fluxes, hemisphere symmetry/asymmetry, and exact
   per-box Kirchhoff flux balance).
3. **Seasonal window averaging vs 20,000-point composite trapezoid quadrature**:
   ``RadiocarbonModel.predict_d14c`` / ``ExactLinearSolver`` on NH summer,
   SH year-boundary wrap-around (``schulman_start`` and ``end_year``), and
   multi-subwindow weights across sub-annual spikes, multi-harmonic solar
   modulation, and realized GP production against independent SciPy
   ``expm`` / ``DOP853`` trajectories and composite trapezoid integration,
   including explicit ``O(h^2)`` convergence-order verification.
4. **Solver eager vs ``jax.jit`` consistency (modal & Van Loan paths)**:
   ``ExactLinearSolver`` (all four spike profiles, pointwise and binned,
   nominal real spectrum, complex-eigenvalue non-normal spectrum, and
   near-degenerate Van Loan fallback, plus ``EffectiveCBM``) and
   ``DiffraxSolver`` (``Tsit5`` and ``Kvaerno5``), asserting ``rtol=1e-13``
   agreement of both forward predictions and ``jax.grad`` gradients between
   eager execution and ``jax.jit``.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import equinox as eqx
import h5py
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.integrate as spi
import scipy.linalg as spl
import scipy.optimize as spo
import tinygp.kernels as tgk

from ticktack.cbm import BalancedCBM, EffectiveCBM, FixedCBM, Topology
from ticktack.constants import (
    ALPHA_ATOMS_CM2_S_TO_KG_YR,
    DECAY_CAMBRIDGE,
    DEFAULT_Q_STD,
)
from ticktack.data import Dataset, SeasonalWeights, TreeRingSeries
from ticktack.fisher import cbm_fisher_spectrum, expected_fisher
from ticktack.model import RadiocarbonModel
from ticktack.presets import (
    LEGACY_V1_C14_OVER_C12_RATIO,
    get_preset_spec,
    load_preset,
)
from ticktack.production import (
    GPProduction,
    HarmonicSolar,
    MiyakeSpike,
    ProductionSum,
    SteadyProduction,
)
from ticktack.solvers import (
    DiffraxSolver,
    ExactLinearSolver,
    _modal_decomposition,
)


# =============================================================================
# Shared Pure-NumPy / SciPy Oracle Helpers
# =============================================================================


def _central_diff_jacobian(
    fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    h: float = 1e-3,
) -> np.ndarray:
    """Richardson-extrapolated central-difference Jacobian in pure float64 NumPy."""
    x0 = np.asarray(x0, dtype=np.float64)
    y0 = np.asarray(fn(x0), dtype=np.float64).reshape(-1)
    n_out, p = y0.size, x0.size

    def _cd(step: float) -> np.ndarray:
        jac = np.empty((n_out, p), dtype=np.float64)
        for j in range(p):
            e_j = np.zeros(p, dtype=np.float64)
            e_j[j] = step
            f_plus = np.asarray(fn(x0 + e_j), dtype=np.float64).reshape(-1)
            f_minus = np.asarray(fn(x0 - e_j), dtype=np.float64).reshape(-1)
            jac[:, j] = (f_plus - f_minus) / (2.0 * step)
        return jac

    j_h = _cd(h)
    j_half = _cd(0.5 * h)
    j_rich = (4.0 * j_half - j_h) / 3.0
    scale = max(float(np.max(np.abs(j_rich))), 1.0)
    assert np.max(np.abs(j_half - j_h)) < 1e-3 * scale, (
        "Central-difference step is not in the asymptotic O(h^2) regime."
    )
    return j_rich


def _assert_full_spectral_agreement(
    mat_pkg: np.ndarray,
    mat_ref: np.ndarray,
    evals_pkg: np.ndarray,
    evecs_pkg: np.ndarray,
    *,
    tols: tuple[float, float, float, float] = (1e-8, 1e-9, 1e-7, 1e-11),
) -> None:
    """Check all matrix entries, all eigenvalues, and all eigenvector subspaces."""
    rtol_mat, atol_mat, rtol_eig, zero_tol = tols
    p = mat_ref.shape[0]
    norm_ref = max(float(np.max(np.abs(mat_ref))), 1.0)
    np.testing.assert_allclose(mat_pkg, mat_ref, rtol=rtol_mat, atol=atol_mat * norm_ref)

    evals_ref_asc, evecs_ref_asc = np.linalg.eigh(0.5 * (mat_ref + mat_ref.T))
    evals_ref = evals_ref_asc[::-1]
    evecs_ref = evecs_ref_asc[:, ::-1]

    max_ev = max(float(evals_ref[0]), 1.0)
    for k in range(p):
        if evals_ref[k] > zero_tol * max_ev:
            np.testing.assert_allclose(
                evals_pkg[k],
                evals_ref[k],
                rtol=rtol_eig,
                atol=zero_tol * max_ev,
                err_msg=f"Eigenvalue mismatch at index {k} of {p}",
            )
        else:
            assert abs(evals_pkg[k]) <= 1e-9 * max_ev, (
                f"Expected zero eigenvalue at index {k}, got {evals_pkg[k]}"
            )

    np.testing.assert_allclose(evecs_pkg.T @ evecs_pkg, np.eye(p), atol=1e-12)
    np.testing.assert_allclose(
        evecs_pkg.T @ mat_ref @ evecs_pkg,
        np.diag(evals_pkg),
        rtol=rtol_eig,
        atol=1e-8 * max_ev,
    )

    for k in range(p):
        if evals_ref[k] <= zero_tol * max_ev:
            continue
        gap_lo = abs(evals_ref[k] - evals_ref[k + 1]) if k + 1 < p else evals_ref[k]
        gap_hi = abs(evals_ref[k - 1] - evals_ref[k]) if k > 0 else evals_ref[k]
        if min(gap_lo, gap_hi) > 1e-2 * evals_ref[k]:
            overlap = abs(float(np.dot(evecs_pkg[:, k], evecs_ref[:, k])))
            assert overlap > 1.0 - 1e-8, (
                f"Eigenvector {k} misalignment: |v_pkg . v_ref| = {overlap}"
            )


def _assert_norm_close(
    actual: np.ndarray,
    desired: np.ndarray,
    rtol: float = 1e-13,
    atol_floor: float = 0.0,
) -> None:
    """Assert elementwise agreement with norm-scaled absolute floor."""
    scale = max(1.0, float(np.max(np.abs(desired))))
    np.testing.assert_allclose(actual, desired, rtol=rtol, atol=max(rtol * scale, atol_floor))


def _make_nh_dataset(years: np.ndarray) -> Dataset:
    """Create a deterministic NH synthetic Dataset."""
    zeros = np.zeros_like(years, dtype=np.float64)
    sig_nh = np.linspace(1.1, 1.9, years.size)
    return Dataset(
        [
            TreeRingSeries(
                years,
                zeros,
                sig_nh,
                hemisphere="north",
                seasonal_weights="april-september",
                name="syn_nh",
            )
        ]
    )


# =============================================================================
# 1. Full-Spectrum Fisher Matrix vs Central-Difference J^T Sigma^-1 J
# =============================================================================


def test_expected_fisher_full_spectrum_vs_central_diff_jacobian() -> None:
    """Compare ``expected_fisher`` against pure NumPy ``J_fd^T Sigma^-1 J_fd``."""
    ds = _make_nh_dataset(np.arange(772.0, 786.0, 1.0))
    n_obs = ds.n_obs
    solver = ExactLinearSolver()

    base_cbm = EffectiveCBM(K=4)  # 6 identifiable CBM parameters
    theta_cbm0, unravel_cbm = jax.flatten_util.ravel_pytree(
        eqx.filter(base_cbm, eqx.is_inexact_array)
    )
    _, static_cbm = eqx.partition(base_cbm, eqx.is_inexact_array)
    p_cbm = int(theta_cbm0.size)

    x0_jnp = jnp.concatenate(
        [theta_cbm0, jnp.array([774.45, 0.25, 6.2, 0.22, -0.15], dtype=jnp.float64)]
    )
    x0_np = np.asarray(x0_jnp, dtype=np.float64)

    def forward_jnp(x: jax.Array) -> jax.Array:
        cbm = eqx.combine(unravel_cbm(x[:p_cbm]), static_cbm)
        prod = ProductionSum(
            steady=SteadyProduction(),
            spike=MiyakeSpike(
                t_start=x[p_cbm],
                duration=x[p_cbm + 1],
                fluence=x[p_cbm + 2],
                profile="boxcar",
            ),
            solar=HarmonicSolar(
                periods=11.0, c_cos=x[p_cbm + 3], c_sin=x[p_cbm + 4], t_ref=775.0
            ),
        )
        return RadiocarbonModel(cbm, prod, solver).predict_d14c(ds)

    fwd_jit = jax.jit(forward_jnp)
    j_fd = _central_diff_jacobian(
        lambda x: np.asarray(fwd_jit(jnp.asarray(x, dtype=jnp.float64))), x0_np
    )

    sigma = np.linspace(0.8, 2.1, n_obs)
    f_pkg_sig = np.asarray(expected_fisher(forward_jnp, x0_jnp, sigma, kind="sigma"))
    f_fd_sig = j_fd.T @ np.diag(1.0 / (sigma**2)) @ j_fd
    ev_s, evec_s = np.linalg.eigh(f_pkg_sig)
    _assert_full_spectral_agreement(f_pkg_sig, f_fd_sig, ev_s[::-1], evec_s[:, ::-1])

    rng = np.random.default_rng(2026)
    factor = rng.standard_normal((n_obs, 5)) * 0.35
    cov_mat = np.diag(sigma**2) + factor @ factor.T
    prec_mat = np.linalg.inv(cov_mat)
    ridge = 1.5e-2
    f_fd_corr = j_fd.T @ prec_mat @ j_fd + ridge * np.eye(x0_np.size)

    for arg, kind in ((prec_mat, "precision"), (cov_mat, "cov")):
        f_pkg = np.asarray(
            expected_fisher(forward_jnp, x0_jnp, arg, kind=kind, ridge=ridge)  # type: ignore[arg-type]
        )
        ev, evec = np.linalg.eigh(f_pkg)
        _assert_full_spectral_agreement(f_pkg, f_fd_corr, ev[::-1], evec[:, ::-1])


def _check_schur_spectrum_case(
    forward_joint: Callable,
    cbm0: EffectiveCBM,
    psi0: dict[str, jax.Array],
    fd_blocks: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]],
    priors: tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray],
) -> None:
    """Compare one Schur-marginalized ``cbm_fisher_spectrum`` call against NumPy FD."""
    sigma, f_joint_fd, f_tt_fd, f_tp_fd, f_pp_fd, (j_log_m, j_log_f) = fd_blocks
    prior_cbm_cov, l_pi, psi_cov_arg, pi_psi_ref = priors
    src_e = np.asarray(cbm0.topology.edges_src, dtype=np.int64)
    dst_e = np.asarray(cbm0.topology.edges_dst, dtype=np.int64)

    res = cbm_fisher_spectrum(
        forward_joint,
        cbm0,
        sigma,
        psi_nuisance=psi0,
        prior_cbm_cov=prior_cbm_cov,
        prior_psi_cov=psi_cov_arg,
        mass_fn=lambda c: c.reservoir_masses(),
        flux_fn=lambda c: c.flux_matrix()[src_e, dst_e],
    )
    f_marg_fd = f_tt_fd - f_tp_fd @ np.linalg.solve(f_pp_fd + pi_psi_ref, f_tp_fd.T)
    f_marg_fd = 0.5 * (f_marg_fd + f_marg_fd.T)
    f_white_fd = 0.5 * (l_pi.T @ f_marg_fd @ l_pi + (l_pi.T @ f_marg_fd @ l_pi).T)

    np.testing.assert_allclose(np.asarray(res.joint_fisher), f_joint_fd, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(np.asarray(res.marginal_fisher), f_marg_fd, rtol=1e-7, atol=1e-8)
    _assert_full_spectral_agreement(
        np.asarray(res.whitened_fisher),
        f_white_fd,
        np.asarray(res.eigenvalues),
        np.asarray(res.prior_whitened_modes),
        tols=(1e-7, 1e-8, 1e-6, 1e-11),
    )

    mu_fd = np.linalg.eigvalsh(f_white_fd)[::-1]
    d_eff_fd = float(np.sum(mu_fd / (mu_fd + 1.0)))
    info_nats_fd = 0.5 * float(np.sum(np.log1p(np.maximum(mu_fd, 0.0))))
    post_cov_fd = np.linalg.inv(f_marg_fd + np.linalg.inv(prior_cbm_cov))

    np.testing.assert_allclose(float(res.d_eff), d_eff_fd, rtol=1e-7)
    np.testing.assert_allclose(float(res.info_nats), info_nats_fd, rtol=1e-7)
    np.testing.assert_allclose(float(res.info_bits), info_nats_fd / math.log(2.0), rtol=1e-7)
    np.testing.assert_allclose(np.asarray(res.posterior_cov), post_cov_fd, rtol=1e-7, atol=1e-9)

    loadings = np.asarray(res.parameter_loadings)
    np.testing.assert_allclose(np.asarray(res.mass_loadings), j_log_m @ loadings, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(np.asarray(res.flux_loadings), j_log_f @ loadings, rtol=1e-8, atol=1e-9)


def test_cbm_fisher_spectrum_unmarginalized_and_schur_effective_cbm() -> None:
    """Verify full-spectrum ``cbm_fisher_spectrum`` (all modes + loadings) on ``EffectiveCBM``."""
    ds = _make_nh_dataset(np.arange(772.0, 788.0, 1.0))
    solver = ExactLinearSolver()
    cbm0 = EffectiveCBM(K=4)
    psi0 = {
        "spike": jnp.array([774.35, 0.35, 6.5], dtype=jnp.float64),
        "solar": jnp.array([0.25, -0.18], dtype=jnp.float64),
        "offset": jnp.array([-1.5], dtype=jnp.float64),
    }

    def forward_joint(cbm: EffectiveCBM, psi: dict[str, jax.Array]) -> jax.Array:
        prod = ProductionSum(
            steady=SteadyProduction(),
            spike=MiyakeSpike(
                t_start=psi["spike"][0],
                duration=psi["spike"][1],
                fluence=psi["spike"][2],
                profile="boxcar",
            ),
            solar=HarmonicSolar(
                periods=11.0, c_cos=psi["solar"][0], c_sin=psi["solar"][1], t_ref=775.0
            ),
        )
        return RadiocarbonModel(cbm, prod, solver).predict_d14c(
            ds, delta_global=psi["offset"][0]
        )

    rng = np.random.default_rng(17)
    sigma = np.linspace(0.9, 1.8, ds.n_obs)
    sig_inv = np.diag(1.0 / (sigma**2))
    p_th, p_psi = 6, 6

    a_cbm = rng.standard_normal((p_th, p_th)) * 0.12
    prior_cbm_cov = np.diag(np.full(p_th, 0.35**2)) + a_cbm @ a_cbm.T
    l_pi = np.linalg.cholesky(prior_cbm_cov)
    a_psi = rng.standard_normal((p_psi, p_psi)) * 0.08
    prior_psi_cov = np.diag([0.5**2, 0.2**2, 2.0**2, 0.5**2, 0.5**2, 5.0**2]) + a_psi @ a_psi.T

    th_diff0, static_th = eqx.partition(cbm0, eqx.is_inexact_array)
    flat_th0, unravel_th = jax.flatten_util.ravel_pytree(th_diff0)
    flat_psi0, unravel_psi = jax.flatten_util.ravel_pytree(psi0)
    z0_np = np.asarray(jnp.concatenate([flat_th0, flat_psi0]), dtype=np.float64)

    fwd_z_jit = jax.jit(
        lambda z: forward_joint(
            eqx.combine(unravel_th(z[:p_th]), static_th), unravel_psi(z[p_th:])
        )
    )
    j_joint_fd = _central_diff_jacobian(
        lambda z: np.asarray(fwd_z_jit(jnp.asarray(z, dtype=jnp.float64))), z0_np
    )
    j_th_fd, j_psi_fd = j_joint_fd[:, :p_th], j_joint_fd[:, p_th:]
    f_joint_fd = j_joint_fd.T @ sig_inv @ j_joint_fd
    f_tt_fd = j_th_fd.T @ sig_inv @ j_th_fd
    f_tp_fd = j_th_fd.T @ sig_inv @ j_psi_fd
    f_pp_fd = j_psi_fd.T @ sig_inv @ j_psi_fd

    res_unmarg = cbm_fisher_spectrum(
        lambda c: forward_joint(c, psi0), cbm0, sigma, prior_cbm_cov=prior_cbm_cov
    )
    _assert_full_spectral_agreement(
        np.asarray(res_unmarg.whitened_fisher),
        l_pi.T @ f_tt_fd @ l_pi,
        np.asarray(res_unmarg.eigenvalues),
        np.asarray(res_unmarg.prior_whitened_modes),
    )

    src_e = np.asarray(cbm0.topology.edges_src, dtype=np.int64)
    dst_e = np.asarray(cbm0.topology.edges_dst, dtype=np.int64)
    j_log_m = _central_diff_jacobian(
        lambda th: np.log(
            np.asarray(eqx.combine(unravel_th(jnp.asarray(th)), static_th).reservoir_masses())
        ),
        np.asarray(flat_th0, dtype=np.float64),
    )
    j_log_f = _central_diff_jacobian(
        lambda th: np.log(
            np.asarray(eqx.combine(unravel_th(jnp.asarray(th)), static_th).flux_matrix())[
                src_e, dst_e
            ]
        ),
        np.asarray(flat_th0, dtype=np.float64),
    )
    fd_blocks = (sigma, f_joint_fd, f_tt_fd, f_tp_fd, f_pp_fd, (j_log_m, j_log_f))

    _check_schur_spectrum_case(
        forward_joint, cbm0, psi0, fd_blocks, (prior_cbm_cov, l_pi, None, np.zeros((p_psi, p_psi)))
    )
    _check_schur_spectrum_case(
        forward_joint,
        cbm0,
        psi0,
        fd_blocks,
        (prior_cbm_cov, l_pi, prior_psi_cov, np.linalg.inv(prior_psi_cov)),
    )


def test_cbm_fisher_spectrum_guttler15_full_24_modes_and_gauge_nullspace() -> None:
    """Verify all 24 active modes and 4 exact gauge zeros of ``Guttler15`` ``BalancedCBM``."""
    ds = _make_nh_dataset(np.arange(770.0, 795.0, 1.0))
    solver = ExactLinearSolver()

    cbm0 = load_preset("Guttler15", kind="balanced")  # 24 log-rates + 1 log_total_mass
    psi0 = jnp.array([774.45, 0.25, 6.2, -1.0], dtype=jnp.float64)
    sigma = np.linspace(0.8, 1.6, ds.n_obs)
    sig_inv = np.diag(1.0 / (sigma**2))
    prior_cbm_sd = 0.35
    prior_psi_sd = np.array([0.4, 0.2, 2.0, 10.0], dtype=np.float64)

    def forward_joint(cbm: BalancedCBM, psi: jax.Array) -> jax.Array:
        prod = ProductionSum(
            steady=SteadyProduction(),
            spike=MiyakeSpike(t_start=psi[0], duration=psi[1], fluence=psi[2], profile="boxcar"),
        )
        return RadiocarbonModel(cbm, prod, solver).predict_d14c(ds, delta_global=psi[3])

    res = cbm_fisher_spectrum(
        forward_joint,
        cbm0,
        sigma,
        psi_nuisance=psi0,
        prior_cbm_sd=prior_cbm_sd,
        prior_psi_sd=prior_psi_sd,
    )

    active_mask = np.asarray(res.active_mask, dtype=bool)
    assert active_mask.shape == (25,) and int(np.sum(active_mask)) == 24

    th_diff0, static_th = eqx.partition(cbm0, eqx.is_inexact_array)
    flat_th0, unravel_th = jax.flatten_util.ravel_pytree(th_diff0)
    z0_np = np.asarray(jnp.concatenate([flat_th0, psi0]), dtype=np.float64)
    fwd_z_jit = jax.jit(
        lambda z: forward_joint(eqx.combine(unravel_th(z[:25]), static_th), z[25:])
    )
    j_joint_fd = _central_diff_jacobian(
        lambda z: np.asarray(fwd_z_jit(jnp.asarray(z, dtype=jnp.float64))), z0_np
    )
    j_th_fd, j_psi_fd = j_joint_fd[:, :25], j_joint_fd[:, 25:]
    f_tt_fd = j_th_fd.T @ sig_inv @ j_th_fd
    f_tp_fd = j_th_fd.T @ sig_inv @ j_psi_fd
    f_pp_fd = j_psi_fd.T @ sig_inv @ j_psi_fd

    pi_psi = np.diag(1.0 / (prior_psi_sd**2))
    f_marg_fd = f_tt_fd - f_tp_fd @ np.linalg.solve(f_pp_fd + pi_psi, f_tp_fd.T)
    f_marg_fd = 0.5 * (f_marg_fd + f_marg_fd.T)
    act_idx = np.flatnonzero(active_mask)
    f_white_fd = (prior_cbm_sd**2) * f_marg_fd[np.ix_(act_idx, act_idx)]

    assert np.max(np.abs(f_marg_fd[np.flatnonzero(~active_mask), :])) < 1e-12

    evals_pkg = np.asarray(res.eigenvalues)
    evecs_pkg = np.asarray(res.prior_whitened_modes)
    _assert_full_spectral_agreement(
        np.asarray(res.whitened_fisher),
        f_white_fd,
        evals_pkg,
        evecs_pkg,
        tols=(1e-7, 1e-8, 1e-6, 1e-10),
    )

    assert np.all(evals_pkg[-4:] < 1e-11 * max(float(evals_pkg[0]), 1.0))
    assert float(np.linalg.norm(f_white_fd @ evecs_pkg[:, -4:])) < 1e-9 * max(
        float(evals_pkg[0]), 1.0
    )
    jax.clear_caches()


# =============================================================================
# 2. Preset Flux and Mass Tables vs Independent Paper Transcriptions
# =============================================================================


def _canonical_box_key(name: str, hemisphere: str) -> tuple[str, str]:
    """Normalize minor punctuation differences ('&' vs 'and') and case."""
    norm_name = " ".join(name.lower().replace("&", "and").split())
    norm_hem = hemisphere.strip().lower()
    if norm_hem in ("none", "", "global"):
        norm_hem = "none"
    return (norm_name, norm_hem)


# 1. Güttler et al. (2015), EPSL 411, 290–297, Fig. 1 / Table 1 (11 boxes, 24 fluxes)
_REF_GUTTLER15_MASSES: dict[tuple[str, str], float] = {
    ("stratosphere", "none"): 88.5,
    ("troposphere", "none"): 501.5,
    ("surface water", "none"): 900.0,
    ("surface biota", "none"): 3.0,
    ("intermediate and deep water", "none"): 37800.0,
    ("short-lived biota", "none"): 110.0,
    ("long-lived biota", "none"): 450.0,
    ("litter", "none"): 300.0,
    ("soil", "none"): 1350.0,
    ("peat", "none"): 500.0,
    ("sedimentary sink", "none"): 378000.0,
}

_REF_GUTTLER15_PROD: dict[tuple[str, str], float] = {
    ("stratosphere", "none"): 0.70,
    ("troposphere", "none"): 0.30,
}

_REF_GUTTLER15_FLUXES: dict[tuple[tuple[str, str], tuple[str, str]], float] = {
    (("sedimentary sink", "none"), ("troposphere", "none")): 0.7,
    (("stratosphere", "none"), ("troposphere", "none")): 45.0,
    (("troposphere", "none"), ("stratosphere", "none")): 45.0,
    (("troposphere", "none"), ("surface water", "none")): 60.5,
    (("surface water", "none"), ("troposphere", "none")): 61.0,
    (("surface water", "none"), ("surface biota", "none")): 40.0,
    (("surface biota", "none"), ("surface water", "none")): 36.0,
    (("surface biota", "none"), ("intermediate and deep water", "none")): 4.0,
    (("intermediate and deep water", "none"), ("surface water", "none")): 42.0,
    (("surface water", "none"), ("intermediate and deep water", "none")): 38.2,
    (("surface water", "none"), ("sedimentary sink", "none")): 0.3,
    (("intermediate and deep water", "none"), ("sedimentary sink", "none")): 0.2,
    (("troposphere", "none"), ("short-lived biota", "none")): 115.0,
    (("short-lived biota", "none"), ("troposphere", "none")): 60.0,
    (("short-lived biota", "none"), ("long-lived biota", "none")): 15.0,
    (("long-lived biota", "none"), ("litter", "none")): 15.0,
    (("short-lived biota", "none"), ("litter", "none")): 40.0,
    (("litter", "none"), ("soil", "none")): 3.0,
    (("litter", "none"), ("peat", "none")): 1.0,
    (("litter", "none"), ("surface water", "none")): 1.0,
    (("litter", "none"), ("troposphere", "none")): 50.0,
    (("soil", "none"), ("troposphere", "none")): 3.0,
    (("peat", "none"), ("troposphere", "none")): 0.8,
    (("peat", "none"), ("sedimentary sink", "none")): 0.2,
}

# Shared 22-box two-hemisphere reservoir masses and production split
# (Büntgen et al. 2018, Nat. Commun. 9, 3605; Brehm et al. 2021, Nat. Geosci. 14, 10)
_REF_TWO_HEM_MASSES: dict[tuple[str, str], float] = {
    ("stratosphere", "south"): 44.5,
    ("troposphere", "south"): 250.5,
    ("surface water", "south"): 540.0,
    ("surface biota", "south"): 1.8,
    ("intermediate and deep water", "south"): 22680.0,
    ("short-lived biota", "south"): 33.0,
    ("long-lived biota", "south"): 135.0,
    ("litter", "south"): 90.0,
    ("soil", "south"): 405.0,
    ("peat", "south"): 150.0,
    ("sedimentary sink", "south"): 189000.0,
    ("stratosphere", "north"): 44.5,
    ("troposphere", "north"): 250.5,
    ("surface water", "north"): 360.0,
    ("surface biota", "north"): 1.2,
    ("intermediate and deep water", "north"): 15120.0,
    ("short-lived biota", "north"): 77.0,
    ("long-lived biota", "north"): 315.0,
    ("litter", "north"): 210.0,
    ("soil", "north"): 945.0,
    ("peat", "north"): 350.0,
    ("sedimentary sink", "north"): 189000.0,
}

_REF_TWO_HEM_PROD: dict[tuple[str, str], float] = {
    ("stratosphere", "south"): 0.35,
    ("troposphere", "south"): 0.15,
    ("stratosphere", "north"): 0.35,
    ("troposphere", "north"): 0.15,
}

# 2. Brehm et al. (2021), Nature Geoscience 14, 10–15, Supplementary Table 1 (56 edges)
_REF_BREHM21_FLUXES: dict[tuple[tuple[str, str], tuple[str, str]], float] = {
    (("stratosphere", "south"), ("troposphere", "south")): 22.5,
    (("troposphere", "south"), ("stratosphere", "south")): 22.5,
    (("troposphere", "south"), ("short-lived biota", "south")): 34.5,
    (("short-lived biota", "south"), ("troposphere", "south")): 18.0,
    (("troposphere", "south"), ("surface water", "south")): 36.6,
    (("surface water", "south"), ("troposphere", "south")): 36.6,
    (("short-lived biota", "south"), ("long-lived biota", "south")): 4.5,
    (("long-lived biota", "south"), ("litter", "south")): 4.5,
    (("short-lived biota", "south"), ("litter", "south")): 12.0,
    (("litter", "south"), ("troposphere", "south")): 15.0,
    (("litter", "south"), ("surface water", "south")): 0.3,
    (("litter", "south"), ("peat", "south")): 0.3,
    (("litter", "south"), ("soil", "south")): 0.9,
    (("peat", "south"), ("troposphere", "south")): 0.24,
    (("peat", "south"), ("sedimentary sink", "south")): 0.06,
    (("soil", "south"), ("troposphere", "south")): 0.9,
    (("surface water", "south"), ("sedimentary sink", "south")): 0.18,
    (("intermediate and deep water", "south"), ("sedimentary sink", "south")): 0.12,
    (("surface water", "south"), ("surface biota", "south")): 22.6,
    (("surface water", "south"), ("intermediate and deep water", "south")): 37.92,
    (("surface water", "south"), ("surface water", "north")): 13.0,
    (("surface biota", "south"), ("surface water", "south")): 20.2,
    (("surface biota", "south"), ("intermediate and deep water", "south")): 2.4,
    (("intermediate and deep water", "south"), ("surface water", "south")): 43.2,
    (("intermediate and deep water", "south"), ("intermediate and deep water", "north")): 15.12,
    (("sedimentary sink", "south"), ("troposphere", "south")): 0.36,
    (("stratosphere", "south"), ("stratosphere", "north")): 44.1,
    (("troposphere", "south"), ("troposphere", "north")): 71.0,
    (("stratosphere", "north"), ("stratosphere", "south")): 44.1,
    (("troposphere", "north"), ("troposphere", "south")): 71.0,
    (("stratosphere", "north"), ("troposphere", "north")): 22.5,
    (("troposphere", "north"), ("stratosphere", "north")): 22.5,
    (("troposphere", "north"), ("surface water", "north")): 24.4,
    (("troposphere", "north"), ("short-lived biota", "north")): 80.5,
    (("surface water", "north"), ("troposphere", "north")): 24.89,
    (("surface water", "north"), ("surface water", "south")): 10.0,
    (("surface water", "north"), ("surface biota", "north")): 16.0,
    (("surface water", "north"), ("intermediate and deep water", "north")): 28.29,
    (("short-lived biota", "north"), ("troposphere", "north")): 42.0,
    (("short-lived biota", "north"), ("long-lived biota", "north")): 10.5,
    (("short-lived biota", "north"), ("litter", "north")): 28.0,
    (("long-lived biota", "north"), ("litter", "north")): 10.5,
    (("surface biota", "north"), ("surface water", "north")): 14.4,
    (("surface biota", "north"), ("intermediate and deep water", "north")): 1.6,
    (("surface water", "north"), ("sedimentary sink", "north")): 0.12,
    (("intermediate and deep water", "north"), ("surface water", "north")): 26.8,
    (("intermediate and deep water", "north"), ("sedimentary sink", "north")): 0.09,
    (("intermediate and deep water", "north"), ("intermediate and deep water", "south")): 18.12,
    (("litter", "north"), ("surface water", "north")): 0.7,
    (("litter", "north"), ("soil", "north")): 2.1,
    (("litter", "north"), ("troposphere", "north")): 35.0,
    (("litter", "north"), ("peat", "north")): 0.7,
    (("soil", "north"), ("troposphere", "north")): 2.1,
    (("peat", "north"), ("troposphere", "north")): 0.56,
    (("peat", "north"), ("sedimentary sink", "north")): 0.14,
    (("sedimentary sink", "north"), ("troposphere", "north")): 0.35,
}

# 3. Büntgen et al. (2018), Nature Communications 9, 3605 (55 strictly positive edges)
_REF_BUNTGEN18_FLUXES: dict[tuple[tuple[str, str], tuple[str, str]], float] = {
    **_REF_BREHM21_FLUXES,
    (("stratosphere", "south"), ("troposphere", "south")): 23.8,
    (("troposphere", "south"), ("stratosphere", "south")): 23.8,
    (("troposphere", "south"), ("short-lived biota", "south")): 35.36,
    (("short-lived biota", "south"), ("troposphere", "south")): 18.86,
    (("troposphere", "south"), ("surface water", "south")): 38.25,
    (("surface water", "south"), ("troposphere", "south")): 38.25,
    (("surface water", "south"), ("intermediate and deep water", "south")): 8.52,
    (("surface water", "south"), ("surface water", "north")): 28.8,
    (("intermediate and deep water", "south"), ("surface water", "south")): 39.6,
    (("intermediate and deep water", "south"), ("intermediate and deep water", "north")): 0.72,
    (("stratosphere", "south"), ("stratosphere", "north")): 22.25,
    (("troposphere", "south"), ("troposphere", "north")): 71.6,
    (("stratosphere", "north"), ("stratosphere", "south")): 22.25,
    (("troposphere", "north"), ("troposphere", "south")): 71.6,
    (("stratosphere", "north"), ("troposphere", "north")): 23.8,
    (("troposphere", "north"), ("stratosphere", "north")): 23.8,
    (("troposphere", "north"), ("surface water", "north")): 29.16,
    (("troposphere", "north"), ("short-lived biota", "north")): 80.51,
    (("surface water", "north"), ("troposphere", "north")): 29.65,
    (("surface water", "north"), ("intermediate and deep water", "north")): 49.69,
    (("short-lived biota", "north"), ("troposphere", "north")): 42.01,
    (("intermediate and deep water", "north"), ("surface water", "north")): 22.4,
    (("intermediate and deep water", "north"), ("intermediate and deep water", "south")): 29.52,
}
del _REF_BUNTGEN18_FLUXES[(("surface water", "north"), ("surface water", "south"))]

# 4. Miyake et al. (2017), PNAS 114, 881–884 (4 boxes, 6 fluxes from tau = 2, 11, 23 yr)
_REF_MIYAKE17_MASSES: dict[tuple[str, str], float] = {
    ("stratosphere", "none"): 112.5,
    ("troposphere", "none"): 637.5,
    ("marine surface", "none"): 900.0,
    ("biosphere", "none"): 1600.0,
}

_REF_MIYAKE17_PROD: dict[tuple[str, str], float] = {
    ("stratosphere", "none"): 0.70,
    ("troposphere", "none"): 0.30,
}

_REF_MIYAKE17_FLUXES: dict[tuple[tuple[str, str], tuple[str, str]], float] = {
    (("stratosphere", "none"), ("troposphere", "none")): 112.5 / 2.0,
    (("troposphere", "none"), ("stratosphere", "none")): 112.5 / 2.0,
    (("troposphere", "none"), ("marine surface", "none")): 637.5 / 11.0,
    (("marine surface", "none"), ("troposphere", "none")): 637.5 / 11.0,
    (("troposphere", "none"), ("biosphere", "none")): 637.5 / 23.0,
    (("biosphere", "none"), ("troposphere", "none")): 637.5 / 23.0,
}

_PAPER_TABLES = {
    "Guttler15": (11, 420003.0, _REF_GUTTLER15_MASSES, _REF_GUTTLER15_PROD, _REF_GUTTLER15_FLUXES),
    "Brehm21": (22, 420003.0, _REF_TWO_HEM_MASSES, _REF_TWO_HEM_PROD, _REF_BREHM21_FLUXES),
    "Buntgen18": (22, 420003.0, _REF_TWO_HEM_MASSES, _REF_TWO_HEM_PROD, _REF_BUNTGEN18_FLUXES),
    "Miyake17": (4, 3250.0, _REF_MIYAKE17_MASSES, _REF_MIYAKE17_PROD, _REF_MIYAKE17_FLUXES),
}


@pytest.mark.parametrize("preset_name", ("Guttler15", "Brehm21", "Buntgen18", "Miyake17"))
def test_preset_tables_vs_independent_paper_transcriptions(preset_name: str) -> None:
    """Check masses, production fractions, fluxes, and Kirchhoff balance vs paper tables."""
    exp_n_boxes, exp_total_mass, ref_masses, ref_prod, ref_fluxes = _PAPER_TABLES[preset_name]
    spec = get_preset_spec(preset_name)
    cbm = load_preset(preset_name, kind="fixed")

    assert spec.n_boxes == exp_n_boxes and cbm.n_boxes == exp_n_boxes
    assert spec.n_edges == len(ref_fluxes)

    keys = [_canonical_box_key(n, h) for n, h in zip(spec.box_names, spec.hemispheres)]
    assert len(set(keys)) == exp_n_boxes and set(keys) == set(ref_masses.keys())

    spec_masses = np.asarray(spec.masses, dtype=np.float64)
    cbm_masses = np.asarray(cbm.reservoir_masses(), dtype=np.float64)
    np.testing.assert_allclose(float(np.sum(spec_masses)), exp_total_mass, rtol=1e-15)
    np.testing.assert_allclose(cbm_masses, spec_masses, rtol=1e-15)
    for idx, key in enumerate(keys):
        np.testing.assert_allclose(spec_masses[idx], ref_masses[key], rtol=1e-15, atol=0.0)

    spec_prod = np.asarray(spec.prod_coefficients, dtype=np.float64)
    cbm_prod = np.asarray(cbm.production_vector(), dtype=np.float64)
    np.testing.assert_allclose(float(np.sum(spec_prod)), 1.0, atol=1e-15)
    np.testing.assert_allclose(cbm_prod, spec_prod, rtol=1e-15)
    for idx, key in enumerate(keys):
        np.testing.assert_allclose(spec_prod[idx], ref_prod.get(key, 0.0), atol=1e-15)

    actual_fluxes: dict[tuple[tuple[str, str], tuple[str, str]], float] = {}
    for src, dst, flux in spec.edges:
        pair = (keys[src], keys[dst])
        assert pair not in actual_fluxes, f"Duplicate edge {pair}"
        actual_fluxes[pair] = float(flux)
    assert set(actual_fluxes.keys()) == set(ref_fluxes.keys())
    for pair, exp_val in ref_fluxes.items():
        np.testing.assert_allclose(actual_fluxes[pair], exp_val, rtol=1e-14, atol=0.0)

    f_mat = np.asarray(cbm.flux_matrix(), dtype=np.float64)
    np.testing.assert_allclose(np.sum(f_mat, axis=0) - np.sum(f_mat, axis=1), 0.0, atol=1e-13)
    k_mat = np.asarray(cbm.rate_matrix_K(), dtype=np.float64)
    np.testing.assert_allclose(np.sum(k_mat, axis=0), 0.0, atol=1e-14)
    np.testing.assert_allclose(k_mat @ cbm_masses, 0.0, atol=1e-12)

    hd5_path = REPO_ROOT / "src" / "ticktack" / "data" / f"{preset_name}.hd5"
    with h5py.File(hd5_path, "r") as hf:
        hd5_masses = np.asarray(hf["reservoir content"], dtype=np.float64).ravel()
        hd5_fluxes = np.asarray(hf["fluxes"], dtype=np.float64)
        if preset_name == "Miyake17":
            hd5_fluxes = hd5_fluxes / LEGACY_V1_C14_OVER_C12_RATIO
        np.testing.assert_allclose(hd5_masses, spec_masses, rtol=1e-14)
        np.testing.assert_allclose(hd5_fluxes, f_mat, rtol=1e-13, atol=1e-13)


def test_two_hemisphere_preset_symmetry_and_asymmetry() -> None:
    """Verify exact N/S terrestrial/atmospheric ratios and physical ocean asymmetry."""
    g_mass = {key[0]: mass for key, mass in _REF_GUTTLER15_MASSES.items()}

    for name, exp_net_north_surface in (("Brehm21", 3.0), ("Buntgen18", 28.8)):
        spec = get_preset_spec(name)
        idx_by_key = {
            _canonical_box_key(n, h): i
            for i, (n, h) in enumerate(zip(spec.box_names, spec.hemispheres))
        }
        masses = np.asarray(spec.masses, dtype=np.float64)
        f_mat = np.asarray(spec.flux_matrix(), dtype=np.float64)
        k_mat = np.asarray(spec.rate_matrix_K(), dtype=np.float64)

        atm_s = (
            masses[idx_by_key[("stratosphere", "south")]]
            + masses[idx_by_key[("troposphere", "south")]]
        )
        atm_n = (
            masses[idx_by_key[("stratosphere", "north")]]
            + masses[idx_by_key[("troposphere", "north")]]
        )
        np.testing.assert_allclose(atm_s, 295.0, rtol=1e-15)
        np.testing.assert_allclose(atm_n, 295.0, rtol=1e-15)
        np.testing.assert_allclose(
            atm_s + atm_n, g_mass["stratosphere"] + g_mass["troposphere"], rtol=1e-15
        )

        for box, frac_s in (
            ("surface water", 0.60),
            ("surface biota", 0.60),
            ("intermediate and deep water", 0.60),
            ("short-lived biota", 0.30),
            ("long-lived biota", 0.30),
            ("litter", 0.30),
            ("soil", 0.30),
            ("peat", 0.30),
            ("sedimentary sink", 0.50),
        ):
            np.testing.assert_allclose(
                masses[idx_by_key[(box, "south")]], frac_s * g_mass[box], rtol=1e-14
            )
            np.testing.assert_allclose(
                masses[idx_by_key[(box, "north")]], (1.0 - frac_s) * g_mass[box], rtol=1e-14
            )

        sw_s, sw_n = idx_by_key[("surface water", "south")], idx_by_key[("surface water", "north")]
        idw_s = idx_by_key[("intermediate and deep water", "south")]
        idw_n = idx_by_key[("intermediate and deep water", "north")]
        np.testing.assert_allclose(f_mat[sw_s, sw_n] - f_mat[sw_n, sw_s], exp_net_north_surface, rtol=1e-14)
        np.testing.assert_allclose(f_mat[idw_n, idw_s] - f_mat[idw_s, idw_n], exp_net_north_surface, rtol=1e-14)

        perm_ns = np.array(
            [
                idx_by_key[(n, "north" if h == "south" else "south")]
                for n, h in (
                    _canonical_box_key(bn, bh)
                    for bn, bh in zip(spec.box_names, spec.hemispheres)
                )
            ]
        )
        rel_asym = float(np.linalg.norm(k_mat - k_mat[np.ix_(perm_ns, perm_ns)]) / np.linalg.norm(k_mat))
        assert rel_asym > 0.04, f"Expected genuine N/S rate asymmetry in {name}, got {rel_asym}"

        sp_s, sp_n = idx_by_key[("stratosphere", "south")], idx_by_key[("stratosphere", "north")]
        tp_s, tp_n = idx_by_key[("troposphere", "south")], idx_by_key[("troposphere", "north")]
        np.testing.assert_allclose(k_mat[sp_n, sp_s], k_mat[sp_s, sp_n], rtol=1e-15)
        np.testing.assert_allclose(k_mat[tp_n, tp_s], k_mat[tp_s, tp_n], rtol=1e-15)


def test_miyake17_residence_times_and_closure_variants() -> None:
    """Verify Miyake17 published residence times, open-ocean sink, and legacy v1 units."""
    m_open = get_preset_spec("Miyake17")
    m_closed = get_preset_spec("Miyake17", miyake17_closure="closed_shallow")
    m_v1 = get_preset_spec("Miyake17", legacy_v1_units=True)

    k_open = np.asarray(m_open.rate_matrix_K(), dtype=np.float64)
    np.testing.assert_allclose(1.0 / k_open[1, 0], 2.0, rtol=1e-15)
    np.testing.assert_allclose(1.0 / k_open[2, 1], 11.0, rtol=1e-15)
    np.testing.assert_allclose(1.0 / k_open[3, 1], 23.0, rtol=1e-15)

    # Independent plain-NumPy Guttler15 R_ref from the transcribed paper table
    g_keys = list(_REF_GUTTLER15_MASSES.keys())
    g_idx = {k: i for i, k in enumerate(g_keys)}
    g_m = np.array([_REF_GUTTLER15_MASSES[k] for k in g_keys], dtype=np.float64)
    g_f = np.zeros((11, 11), dtype=np.float64)
    for (s_k, d_k), val in _REF_GUTTLER15_FLUXES.items():
        g_f[g_idx[s_k], g_idx[d_k]] = val
    g_k = (g_f / g_m[:, None]).T - np.diag(np.sum(g_f / g_m[:, None], axis=1))
    g_a = g_k - DECAY_CAMBRIDGE * np.eye(11, dtype=np.float64)
    g_b = np.array([_REF_GUTTLER15_PROD.get(k, 0.0) for k in g_keys], dtype=np.float64)
    g_yss = np.linalg.solve(-g_a, ALPHA_ATOMS_CM2_S_TO_KG_YR * DEFAULT_Q_STD * g_b)
    r_ref_g15_np = float(g_yss[g_idx[("troposphere", "none")]] / g_m[g_idx[("troposphere", "none")]])

    assert m_open.sink_rates[0] == 0.0 and m_open.sink_rates[1] == 0.0
    assert m_open.sink_rates[2] > 0.0 and m_open.sink_rates[3] == 0.0
    np.testing.assert_allclose(
        float(m_open.reference_specific_activity()), r_ref_g15_np, rtol=1e-14
    )

    assert m_closed.sink_rates == (0.0, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(np.asarray(m_closed.rate_matrix_K()), k_open, rtol=1e-15)

    assert m_v1.sink_rates == (0.0, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(
        np.asarray(m_v1.flux_matrix()),
        np.asarray(m_closed.flux_matrix()) * LEGACY_V1_C14_OVER_C12_RATIO,
        rtol=1e-15,
    )


# =============================================================================
# 3. Seasonal Window Averaging vs 20,000-Point Composite Trapezoid Quadrature
# =============================================================================


def _scipy_ode_dense(
    system: tuple[np.ndarray, np.ndarray],
    dq_fn: Callable[[float], float],
    t_span: tuple[float, float],
    dy0: np.ndarray,
    max_step: float = 0.02,
):
    """Integrate ``dy/dt = A dy + b_scaled * dq(t)`` with SciPy ``DOP853``."""
    A, b_scaled = system
    sol = spi.solve_ivp(
        lambda t, y: A @ y + b_scaled * dq_fn(t),
        t_span,
        dy0,
        method="DOP853",
        rtol=1e-13,
        atol=1e-14,
        max_step=max_step,
        dense_output=True,
    )
    assert sol.success, f"SciPy DOP853 failed: {sol.message}"
    return sol


def _scipy_boxcar_eval(
    A: np.ndarray,
    b_scaled: np.ndarray,
    t_start: float,
    duration: float,
    fluence: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build exact piecewise SciPy evaluator for a boxcar spike using ``expm`` + ``DOP853``."""
    t_end = t_start + duration
    q_amp = fluence / duration
    a_inv_b = np.linalg.solve(A, b_scaled)
    dy_end = q_amp * (spl.expm(A * duration) @ a_inv_b - a_inv_b)

    sol_active = _scipy_ode_dense(
        (A, b_scaled), lambda _t: q_amp, (t_start, t_end), np.zeros(A.shape[0]), max_step=0.01
    )
    sol_post = _scipy_ode_dense(
        (A, b_scaled), lambda _t: 0.0, (t_end, 780.0), dy_end, max_step=0.05
    )

    def _eval(t_grid: np.ndarray) -> np.ndarray:
        out = np.zeros((t_grid.size, A.shape[0]), dtype=np.float64)
        m_act = (t_grid > t_start) & (t_grid <= t_end)
        m_post = t_grid > t_end
        if np.any(m_act):
            out[m_act] = sol_active.sol(t_grid[m_act]).T
        if np.any(m_post):
            out[m_post] = sol_post.sol(t_grid[m_post]).T
        return out

    return _eval


def _trapezoid_window_average(
    eval_d14c: Callable[[np.ndarray], np.ndarray],
    sub_starts: np.ndarray,
    sub_ends: np.ndarray,
    sub_weights: np.ndarray,
    n_points: int = 20_000,
) -> np.ndarray:
    """Evaluate weighted seasonal averages using an ``n_points`` composite trapezoid rule."""
    n_obs, n_sub = sub_starts.shape
    w = np.asarray(sub_weights, dtype=np.float64)
    w = w / np.sum(w)
    out = np.zeros(n_obs, dtype=np.float64)
    for r in range(n_obs):
        acc = 0.0
        for m in range(n_sub):
            a, b = float(sub_starts[r, m]), float(sub_ends[r, m])
            grid = np.linspace(a, b, n_points, dtype=np.float64)
            acc += w[m] * float(np.trapezoid(eval_d14c(grid), grid) / (b - a))
        out[r] = acc
    return out


def _brehm21_numpy_state() -> tuple[FixedCBM, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return ``Brehm21`` ``FixedCBM`` plus pure-NumPy ``(A, b_scaled, masses, y_ss, r_ref)``."""
    cbm = load_preset("Brehm21", kind="fixed")
    A = np.asarray(cbm.matrix_A(), dtype=np.float64)
    b_scaled = ALPHA_ATOMS_CM2_S_TO_KG_YR * np.asarray(cbm.production_vector(), dtype=np.float64)
    masses = np.asarray(cbm.reservoir_masses(), dtype=np.float64)
    y_ss = np.linalg.solve(-A, DEFAULT_Q_STD * b_scaled)
    trop_idx = np.asarray(cbm.topology.trop_box_indices, dtype=np.int64)
    r_ref = float(np.sum(y_ss[trop_idx]) / np.sum(masses[trop_idx]))
    return cbm, A, b_scaled, masses, y_ss, r_ref


def test_seasonal_window_spikes_vs_20k_trapezoid_nh_and_sh_wraparound() -> None:
    """Compare ``predict_d14c`` against 20,000-point SciPy trapezoid on NH & SH windows."""
    cbm, A, b_scaled, masses, y_ss, r_ref = _brehm21_numpy_state()
    years = np.array([774.0, 775.0, 776.0], dtype=np.float64)
    zeros, ones = np.zeros_like(years), np.ones_like(years)

    sw_sh_nonunif = SeasonalWeights.from_subannual_weights(
        [1.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 3.0],
        year_convention="schulman_start",
        label="sh_nonuniform",
    )
    ds = Dataset(
        [
            TreeRingSeries(
                years, zeros, ones, hemisphere="north", seasonal_weights="april-september", name="nh"
            ),
            TreeRingSeries(
                years, zeros, ones, hemisphere="south",
                year_convention="schulman_start", seasonal_weights="october-march", name="sh_sch",
            ),
            TreeRingSeries(
                years, zeros, ones, hemisphere="south",
                year_convention="end_year", seasonal_weights="september-february", name="sh_end",
            ),
            TreeRingSeries(
                years, zeros, ones, hemisphere="south",
                year_convention="schulman_start", seasonal_weights=sw_sh_nonunif, name="sh_multi",
            ),
        ]
    )
    solver = ExactLinearSolver()

    def _d14c(dy: np.ndarray, bi: int) -> np.ndarray:
        return 1000.0 * (((y_ss[bi] + dy[..., bi]) / masses[bi]) / r_ref - 1.0)

    # 1. Sub-annual boxcar spike inside NH summer (774.45 .. 774.60)
    spike_box = MiyakeSpike(t_start=774.45, duration=0.15, fluence=6.4, profile="boxcar")
    box_eval = _scipy_boxcar_eval(A, b_scaled, 774.45, 0.15, 6.4)
    pred_box = RadiocarbonModel(cbm, spike_box, solver).predict_d14c(ds, concat=False)
    for series, pred_s in zip(ds.series, pred_box):
        bi = cbm.box_index("Troposphere", hemisphere=series.hemisphere)
        ref_s = _trapezoid_window_average(
            lambda g, b_i=bi: _d14c(box_eval(g), b_i),
            np.asarray(series.sub_t_starts),
            np.asarray(series.sub_t_ends),
            np.asarray(series.seasonal_weights.weights),
        )
        np.testing.assert_allclose(np.asarray(pred_s), ref_s, rtol=1e-8, atol=1e-7)

    # 2. Smooth supergaussian spike straddling 775.0 year boundary
    ts_sg, dur_sg, fl_sg, p_sg = 774.92, 0.20, 5.8, 10.0
    ln2 = math.log(2.0)
    norm_sg = (ln2 ** (1.0 / p_sg)) / math.gamma(1.0 + 1.0 / p_sg)

    def dq_sg(t: float) -> float:
        x = 2.0 * (t - ts_sg) / dur_sg - 1.0
        xp = x**10
        return (fl_sg / dur_sg) * norm_sg * math.exp(-ln2 * xp) if xp < 700.0 else 0.0

    sol_sg = _scipy_ode_dense((A, b_scaled), dq_sg, (772.0, 778.5), np.zeros(cbm.n_boxes), max_step=0.01)
    spike_sg = MiyakeSpike(ts_sg, dur_sg, fl_sg, profile="supergaussian", exponent=p_sg)
    pred_sg = RadiocarbonModel(cbm, spike_sg, solver).predict_d14c(ds, concat=False)
    for series, pred_s in zip(ds.series, pred_sg):
        bi = cbm.box_index("Troposphere", hemisphere=series.hemisphere)
        ref_s = _trapezoid_window_average(
            lambda g, b_i=bi: _d14c(sol_sg.sol(g).T, b_i),
            np.asarray(series.sub_t_starts),
            np.asarray(series.sub_t_ends),
            np.asarray(series.seasonal_weights.weights),
        )
        np.testing.assert_allclose(np.asarray(pred_s), ref_s, rtol=1e-9, atol=1e-9)

    # 3. Causal Gamma(k=3) spike (pure SciPy FWHM root + pure math forcing)
    ts_g, dur_g, fl_g = 774.30, 0.25, 5.2
    x_l = spo.brentq(lambda x: 2.0 * math.log(x / 2.0) - (x - 2.0) + ln2, 1e-6, 2.0)
    x_r = spo.brentq(lambda x: 2.0 * math.log(x / 2.0) - (x - 2.0) + ln2, 2.0, 20.0)
    c_gam = float(x_r - x_l)

    def dq_gam(t: float) -> float:
        u = (t - ts_g) / dur_g
        if u <= 0.0:
            return 0.0
        return (fl_g / dur_g) * (0.5 * c_gam**3) * (u * u) * math.exp(-c_gam * u)

    sol_gam = _scipy_ode_dense(
        (A, b_scaled), dq_gam, (ts_g, 778.5), np.zeros(cbm.n_boxes), max_step=0.01
    )
    spike_gam = MiyakeSpike(ts_g, dur_g, fl_g, profile="gamma", gamma_shape=3.0)
    pred_gam = RadiocarbonModel(cbm, spike_gam, solver).predict_d14c(ds, concat=False)
    for series, pred_s in zip(ds.series, pred_gam):
        bi = cbm.box_index("Troposphere", hemisphere=series.hemisphere)

        def _eval_gam(g: np.ndarray, b_i: int = bi) -> np.ndarray:
            dy = np.zeros((g.size, cbm.n_boxes), dtype=np.float64)
            mask = g >= ts_g
            if np.any(mask):
                dy[mask] = sol_gam.sol(g[mask]).T
            return _d14c(dy, b_i)

        ref_s = _trapezoid_window_average(
            _eval_gam,
            np.asarray(series.sub_t_starts),
            np.asarray(series.sub_t_ends),
            np.asarray(series.seasonal_weights.weights),
        )
        np.testing.assert_allclose(np.asarray(pred_s), ref_s, rtol=1e-8, atol=1e-8)


def test_seasonal_window_harmonic_and_gp_vs_20k_trapezoid_and_convergence_order() -> None:
    """Verify harmonic & GP seasonal averaging vs 20k trapezoid and ``O(h^2)`` convergence."""
    cbm, A, b_scaled, masses, y_ss, r_ref = _brehm21_numpy_state()
    years = np.array([774.0, 775.0, 776.0], dtype=np.float64)
    zeros, ones = np.zeros_like(years), np.ones_like(years)
    s_nh = TreeRingSeries(
        years, zeros, ones, hemisphere="north", seasonal_weights="april-september", name="nh"
    )
    s_sh = TreeRingSeries(
        years, zeros, ones, hemisphere="south",
        year_convention="schulman_start", seasonal_weights="october-march", name="sh",
    )
    ds = Dataset([s_nh, s_sh])

    def _d14c(dy: np.ndarray, bi: int) -> np.ndarray:
        return 1000.0 * (((y_ss[bi] + dy[..., bi]) / masses[bi]) / r_ref - 1.0)

    periods = (11.0, 22.0, 1.3)
    c_cos, c_sin = (0.35, -0.20, 0.45), (-0.28, 0.15, 0.30)
    omegas = [2.0 * math.pi / p for p in periods]
    t0_h = 772.0
    dy0_h = np.zeros(cbm.n_boxes, dtype=np.float64)
    for w, cc, cs in zip(omegas, c_cos, c_sin):
        z = np.linalg.solve(1j * w * np.eye(cbm.n_boxes) - A, b_scaled)
        th = w * (t0_h - 774.0)
        dy0_h += math.cos(th) * (cc * z.real + cs * z.imag) + math.sin(th) * (
            cs * z.real - cc * z.imag
        )

    def dq_harm(t: float) -> float:
        dt = t - 774.0
        return sum(cc * math.cos(w * dt) + cs * math.sin(w * dt) for w, cc, cs in zip(omegas, c_cos, c_sin))

    sol_harm = _scipy_ode_dense((A, b_scaled), dq_harm, (t0_h, 777.5), dy0_h, max_step=0.02)
    solar = HarmonicSolar(periods=periods, c_cos=c_cos, c_sin=c_sin, t_ref=774.0)
    pred_harm = RadiocarbonModel(cbm, solar, ExactLinearSolver()).predict_d14c(ds, concat=False)
    for series, pred_s in zip(ds.series, pred_harm):
        bi = cbm.box_index("Troposphere", hemisphere=series.hemisphere)
        ref_s = _trapezoid_window_average(
            lambda g, b_i=bi: _d14c(sol_harm.sol(g).T, b_i),
            np.asarray(series.sub_t_starts),
            np.asarray(series.sub_t_ends),
            np.asarray(series.seasonal_weights.weights),
        )
        np.testing.assert_allclose(np.asarray(pred_s), ref_s, rtol=1e-9, atol=1e-9)

    # Explicit O(h^2) trapezoid convergence-order verification on NH 774 window
    box_nh = cbm.box_index("Troposphere", hemisphere="north")
    exact_nh_774 = float(pred_harm[0][1])
    a_w, b_w = float(s_nh.sub_t_starts[1, 0]), float(s_nh.sub_t_ends[1, 0])
    errs = []
    for n_pts in (17, 33, 65, 129, 257):
        g = np.linspace(a_w, b_w, n_pts, dtype=np.float64)
        errs.append(abs(float(np.trapezoid(_d14c(sol_harm.sol(g).T, box_nh), g) / (b_w - a_w)) - exact_nh_774))
    orders = [math.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    for p_obs in orders:
        assert 1.98 <= p_obs <= 2.02, f"Expected O(h^2) trapezoid order, got {orders}"

    # Realized GPProduction (Solin–Särkkä Matern-3/2 in pure NumPy)
    rng = np.random.default_rng(314)
    coeffs = rng.standard_normal(12)
    a_dom, b_dom = 772.0, 778.0
    half_len = 0.5 * (b_dom - a_dom)
    w_gp = np.arange(1, 13, dtype=np.float64) * (math.pi / (2.0 * half_len))
    ell, sig_gp = 3.0, 0.45
    psd_gp = (sig_gp**2) * (12.0 * math.sqrt(3.0) * ell) / ((3.0 + (ell * w_gp) ** 2) ** 2)
    amp_gp = np.sqrt(psd_gp / half_len) * coeffs

    sol_gp = _scipy_ode_dense(
        (A, b_scaled),
        lambda t: float(np.dot(amp_gp, np.sin((t - a_dom) * w_gp))),
        (a_dom, b_dom),
        np.zeros(cbm.n_boxes),
        max_step=0.02,
    )
    ref_gp = np.concatenate(
        [
            _trapezoid_window_average(
                lambda g, b_i=cbm.box_index("Troposphere", hemisphere=s.hemisphere): _d14c(
                    sol_gp.sol(g).T, b_i
                ),
                np.asarray(s.sub_t_starts),
                np.asarray(s.sub_t_ends),
                np.asarray(s.seasonal_weights.weights),
            )
            for s in ds.series
        ]
    )

    gp_term = GPProduction(
        sig_gp**2 * tgk.Matern32(scale=ell), n_modes=12, coefficients=coeffs, domain=(a_dom, b_dom)
    )
    pred_48 = np.asarray(
        RadiocarbonModel(cbm, gp_term, ExactLinearSolver(oversample=48)).predict_d14c(ds)
    )
    pred_96 = np.asarray(
        RadiocarbonModel(cbm, gp_term, ExactLinearSolver(oversample=96)).predict_d14c(ds)
    )
    err_48 = float(np.max(np.abs(pred_48 - ref_gp)))
    err_96 = float(np.max(np.abs(pred_96 - ref_gp)))
    assert err_48 < 1e-4 and err_96 < 2.5e-5
    assert err_48 / err_96 > 3.8, f"Expected O(dt^2) FOH convergence, ratio={err_48 / err_96}"


# =============================================================================
# 4. Solver Eager vs jax.jit Consistency (Modal, Van Loan, and Diffrax)
# =============================================================================


_THREE_BOX_TOPO = Topology(
    box_names=("Stratosphere", "Troposphere", "Sink"),
    edges_src=(0, 1, 1, 2),
    edges_dst=(1, 0, 2, 0),
    prod_box_indices=(0, 1),
    trop_box_indices=(1,),
)


def _make_three_box_regime_cbm(regime: str) -> BalancedCBM:
    """Build a 3-box ``BalancedCBM`` on ``_THREE_BOX_TOPO`` in one of 3 spectral regimes.

    The non-zero eigenvalue discriminant is ``(k01 + k10 + k12 - k20)^2 - 4 k01 k12``:
    - ``"real"``: strictly positive discriminant (real float64 modal path eagerly)
    - ``"complex"``: strictly negative discriminant (complex128 modal path eagerly)
    - ``"near_degenerate"``: near-zero discriminant (``cond(V) > 1e6``, Van Loan fallback eagerly)
    """
    rates_map = {
        "real": [0.5, 0.2, 0.4, 0.05],
        "complex": [1.0, 0.1, 1.0, 1.0],
        "near_degenerate": [1.0, 0.5, 1.0, 0.5 + 1e-13],
    }
    cbm = BalancedCBM(
        topology=_THREE_BOX_TOPO,
        log_rates_free=jnp.log(jnp.array(rates_map[regime], dtype=jnp.float64)),
        log_total_mass=jnp.log(2000.0),
        prod_weights=(0.7, 0.3),
        decay_rate=DECAY_CAMBRIDGE,
    )
    eigvals, evecs = np.linalg.eig(np.asarray(cbm.matrix_A(), dtype=np.float64))
    modal = _modal_decomposition(cbm.matrix_A(), cbm=cbm)
    if regime == "real":
        assert np.max(np.abs(eigvals.imag)) == 0.0
        assert modal is not None and not jnp.iscomplexobj(modal[0])
    elif regime == "complex":
        assert np.max(np.abs(eigvals.imag)) > 0.5
        assert float(np.linalg.cond(evecs)) < 1e4
        assert modal is not None and jnp.iscomplexobj(modal[0])
    else:
        assert float(np.linalg.cond(evecs)) >= 1e6
        assert modal is None
    return cbm


def _make_spike(profile: str, params: jax.Array, skewness: float | jax.Array = 2.2) -> MiyakeSpike:
    """Construct a ``MiyakeSpike`` from a 3-element ``(t_start, duration, fluence)`` vector."""
    if profile == "skewgaussian":
        return MiyakeSpike(
            params[0], params[1], params[2], profile="skewgaussian", skewness=skewness
        )
    return MiyakeSpike(params[0], params[1], params[2], profile=profile)  # type: ignore[arg-type]


@pytest.mark.parametrize("profile", ("boxcar", "supergaussian", "skewgaussian", "gamma"))
def test_exact_linear_solver_eager_vs_jit_all_profiles_and_regimes(profile: str) -> None:
    """Assert ``rtol=1e-13`` eager vs ``jax.jit`` consistency across spectral regimes."""
    solver = ExactLinearSolver()
    t_eval = jnp.array([774.45, 774.60, 775.5], dtype=jnp.float64)
    use_subwin = profile == "boxcar"
    a_win = (
        jnp.array([[774.25, 774.50], [775.25, 775.50]], dtype=jnp.float64)
        if use_subwin
        else jnp.array([774.25, 775.25], dtype=jnp.float64)
    )
    b_win = a_win + (0.25 if use_subwin else 0.50)
    sub_w = jnp.array([0.65, 0.35], dtype=jnp.float64) if use_subwin else None
    dy0_3 = jnp.array([-0.4, 0.2, 0.3], dtype=jnp.float64)
    p_init = jnp.array([774.40, 0.24, 6.3], dtype=jnp.float64)
    w_obs = jnp.array([1.3, 0.8], dtype=jnp.float64)

    def loss_pw(cbm_in, p_in: jax.Array) -> tuple[jax.Array, jax.Array]:
        spike = _make_spike(profile, p_in)
        y0_arg = (cbm_in.steady_state_c14() + dy0_3) if use_subwin else None
        pw = solver.solve_delta_trajectory(cbm_in, spike, t_eval, t0=773.5, y0=y0_arg)
        return jnp.sum(jnp.square(pw[:, 1])), pw

    def loss_bn(cbm_in, p_in: jax.Array) -> tuple[jax.Array, jax.Array]:
        spike = _make_spike(profile, p_in)
        y0_arg = (cbm_in.steady_state_c14() + dy0_3) if use_subwin else None
        bn = solver.solve_delta_binned(
            cbm_in, spike, a_win, b_win, sub_weights=sub_w, t0=773.0, y0=y0_arg
        )
        return jnp.sum(w_obs * jnp.square(bn[:, 1])), bn

    # Shared XLA compilations per profile reused across all 3 spectral regimes:
    # eager closes over concrete cbm (modal path for real/complex, Van Loan for near_degenerate);
    # filter_jit traces cbm (always Van Loan path).
    jit_vag_pw = eqx.filter_jit(jax.value_and_grad(loss_pw, argnums=1, has_aux=True))
    jit_vag_bn = eqx.filter_jit(jax.value_and_grad(loss_bn, argnums=1, has_aux=True))

    for regime in ("real", "complex", "near_degenerate"):
        cbm = _make_three_box_regime_cbm(regime)
        (_, pw_e), g_pw_e = jax.value_and_grad(lambda p: loss_pw(cbm, p), has_aux=True)(p_init)
        (_, pw_j), g_pw_j = jit_vag_pw(cbm, p_init)
        _assert_norm_close(np.asarray(pw_e), np.asarray(pw_j))
        _assert_norm_close(np.asarray(g_pw_e), np.asarray(g_pw_j))

        (_, bn_e), g_bn_e = jax.value_and_grad(lambda p: loss_bn(cbm, p), has_aux=True)(p_init)
        (_, bn_j), g_bn_j = jit_vag_bn(cbm, p_init)
        _assert_norm_close(np.asarray(bn_e), np.asarray(bn_j))
        _assert_norm_close(np.asarray(g_bn_e), np.asarray(g_bn_j))
    jax.clear_caches()


def test_exact_linear_solver_eager_vs_jit_guttler15_and_effective_cbm() -> None:
    """Assert ``rtol=1e-13`` eager vs ``jax.jit`` on nominal 11-box ``Guttler15`` and ``EffectiveCBM``."""
    solver = ExactLinearSolver()
    t_eval = jnp.array([774.45, 774.60, 775.5], dtype=jnp.float64)
    a_win = jnp.array([774.25, 775.25], dtype=jnp.float64)
    b_win = a_win + 0.50
    p_init = jnp.array([774.40, 0.24, 6.3], dtype=jnp.float64)
    solar = HarmonicSolar(periods=11.0, c_cos=0.25, c_sin=-0.18, t_ref=774.0)

    # 1. Nominal real-spectrum 11-box Guttler15 (real modal eagerly vs Van Loan under jit)
    guttler = load_preset("Guttler15", kind="balanced")
    modal_g = _modal_decomposition(guttler.matrix_A(), cbm=guttler)
    assert modal_g is not None and not jnp.iscomplexobj(modal_g[0])

    def g15_loss(c: BalancedCBM, p: jax.Array) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        prod = ProductionSum(steady=SteadyProduction(), spike=_make_spike("boxcar", p), solar=solar)
        pw = solver.solve_delta_trajectory(c, prod, t_eval)
        bn = solver.solve_delta_binned(c, prod, a_win, b_win)
        return jnp.sum(jnp.square(pw[:, 1])) + jnp.sum(jnp.square(bn[:, 1])), (pw, bn)

    (_, (pw_e, bn_e)), g_e = jax.value_and_grad(lambda p: g15_loss(guttler, p), has_aux=True)(p_init)
    (_, (pw_j, bn_j)), g_j = eqx.filter_jit(
        jax.value_and_grad(g15_loss, argnums=1, has_aux=True)
    )(guttler, p_init)
    _assert_norm_close(np.asarray(pw_e), np.asarray(pw_j))
    _assert_norm_close(np.asarray(bn_e), np.asarray(bn_j))
    _assert_norm_close(np.asarray(g_e), np.asarray(g_j))

    # 2. EffectiveCBM (symmetric eigh modal path both eagerly and under jit, CBM + spike grad)
    eff = EffectiveCBM(K=4)
    assert _modal_decomposition(eff.matrix_A(), cbm=eff) is not None

    def eff_loss(c: EffectiveCBM, p: jax.Array) -> tuple[jax.Array, jax.Array]:
        prod = ProductionSum(
            steady=SteadyProduction(), spike=_make_spike("boxcar", p), solar=solar
        )
        bn = solver.solve_delta_binned(c, prod, a_win, b_win)
        return jnp.sum(jnp.square(bn[:, 1])), bn

    (_, eb_e), eg_e = jax.value_and_grad(eff_loss, argnums=(0, 1), has_aux=True)(eff, p_init)
    (_, eb_j), eg_j = eqx.filter_jit(
        jax.value_and_grad(eff_loss, argnums=(0, 1), has_aux=True)
    )(eff, p_init)
    _assert_norm_close(np.asarray(eb_e), np.asarray(eb_j))
    _assert_norm_close(np.asarray(eg_e[1]), np.asarray(eg_j[1]))
    _assert_norm_close(
        np.asarray(eg_e[0].log_k_trop_to_sinks), np.asarray(eg_j[0].log_k_trop_to_sinks)
    )
    jax.clear_caches()


@pytest.mark.parametrize("method", ("Tsit5", "Kvaerno5"))
def test_diffrax_solver_eager_vs_jit_forward_and_non_spike_grad(method: str) -> None:
    """Assert eager vs ``jax.jit`` agreement for ``DiffraxSolver`` forward and smooth-forcing ``grad``."""
    cbm = _make_three_box_regime_cbm("complex")
    solver = DiffraxSolver(method=method, rtol=1e-12, atol=1e-12, dt0=0.05)

    t_eval = jnp.array([774.4, 775.2], dtype=jnp.float64)
    a_win = jnp.array([774.25, 775.25], dtype=jnp.float64)
    b_win = a_win + 0.50
    c_init = jnp.array([0.25, -0.18], dtype=jnp.float64)

    def make_prod(cc: jax.Array) -> ProductionSum:
        return ProductionSum(
            steady=SteadyProduction(),
            solar=HarmonicSolar(periods=11.0, c_cos=cc[0], c_sin=cc[1], t_ref=774.0),
        )

    def eval_bn(cc: jax.Array) -> jax.Array:
        return solver.predict_d14c(cbm, make_prod(cc), a_win, b_win, box_indices=1, t0=774.0)

    def loss_pw(cc: jax.Array) -> tuple[jax.Array, jax.Array]:
        pw = solver.solve_delta_trajectory(cbm, make_prod(cc), t_eval, t0=774.0)
        return jnp.sum(jnp.square(pw[:, 1])), pw

    _assert_norm_close(np.asarray(eval_bn(c_init)), np.asarray(jax.jit(eval_bn)(c_init)), atol_floor=1e-12)
    (_, pw_e), g_e = jax.value_and_grad(loss_pw, has_aux=True)(c_init)
    (_, pw_j), g_j = jax.jit(jax.value_and_grad(loss_pw, has_aux=True))(c_init)
    _assert_norm_close(np.asarray(pw_e), np.asarray(pw_j))
    _assert_norm_close(np.asarray(g_e), np.asarray(g_j))


@pytest.mark.xfail(
    strict=True,
    reason="DiffraxSolver PIDController step_ts lacks lax.stop_gradient on smooth spikes (solvers.py:1667-1716)",
)
def test_diffrax_solver_smooth_spike_grad_eager_vs_jit_stepts_bug() -> None:
    """Document ``DiffraxSolver`` ``step_ts`` differentiation bug on smooth ``MiyakeSpike`` ``grad``."""
    cbm = _make_three_box_regime_cbm("near_degenerate")
    solver = DiffraxSolver(method="Tsit5", rtol=1e-12, atol=1e-12, dt0=0.05)
    t_eval = jnp.array([773.8, 774.5, 775.2, 776.4], dtype=jnp.float64)
    p_init = jnp.array([774.40, 0.25, 5.5], dtype=jnp.float64)
    solar = HarmonicSolar(periods=11.0, c_cos=0.2, c_sin=-0.15, t_ref=774.0)

    def loss_spike(p: jax.Array) -> jax.Array:
        prod = ProductionSum(
            steady=SteadyProduction(),
            spike=MiyakeSpike(p[0], p[1], p[2], profile="supergaussian", exponent=10.0),
            solar=solar,
        )
        return jnp.sum(jnp.square(solver.solve_delta_trajectory(cbm, prod, t_eval, t0=773.5)[:, 1]))

    g_eager = np.asarray(jax.grad(loss_spike)(p_init))
    g_jit = np.asarray(jax.jit(jax.grad(loss_spike))(p_init))
    _assert_norm_close(g_eager, g_jit, rtol=1e-13)
