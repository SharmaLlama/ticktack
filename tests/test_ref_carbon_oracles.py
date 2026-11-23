"""Independent mathematical and statistical reference oracles (Section 3 Carbon tests).

Verifies four core inference and linear-systems identities in ``ticktack`` against
independent NumPy/SciPy closed-form and state-space references without calling internal
design-matrix or marginalization helpers:

1. **Fisher information vs Monte Carlo score covariance and negative Hessian**:
   For ``y^{(s)} ~ N(mu(theta_0), Sigma)`` (both diagonal ``sigma`` and correlated
   ``Sigma``) on nonlinear ``RadiocarbonModel`` parameters spanning CBM rates and
   ``MiyakeSpike`` shape parameters, verifies Wishart ``O(1/sqrt(S))`` score covariance
   convergence, analytical Wishart entrywise z-scores and ``chi^2`` calibration, and
   ``E[-hess_theta log p(y | theta_0)] == expected_fisher`` to ``rtol=1e-11``.
2. **Stationary initial-state covariance vs ``scipy.linalg.solve_continuous_lyapunov``**:
   Builds CBM transition matrices from reservoir inventories/fluxes and GP continuous-time
   companion SDEs from textbook formulas independently in NumPy/SciPy across ``Guttler15``,
   ``Brehm21``, and ``Miyake17`` coupled to ``Matern32``, ``Matern52``, ``SHO``, ``Exp``,
   sum, and ``Matern32 * Cosine`` product kernels, asserting ``rtol=1e-10`` agreement
   with ``stationary_cbm_gp_cov``.
3. **Marginal likelihood, linear posterior, and gradients vs independent dense MVN**:
   Constructs multi-series NH + SH datasets with differing seasonal profiles, hierarchical
   offsets, ``HarmonicSolar``, ``LinearTrend``, and ``GPProduction`` (with stationary
   initial-state covariance), pushes basis modes through augmented real ``scipy.linalg.expm``
   ODE solves from scratch, and asserts ``rtol=1e-11`` agreement of
   ``scipy.stats.multivariate_normal.logpdf`` and Woodbury posteriors/gradients with
   ``marginal_log_likelihood``, ``linear_posterior``, and ``jax.grad(marginal_log_likelihood)``.
4. **NUTS and SMC importance weights on a closed-form conjugate linear-Gaussian problem**:
   Reweights a 2D linear-Gaussian ``(fluence, slope)`` parameter space from a truncated
   low-rank GP + widened-jitter surrogate ``Sigma_surr`` to a full-rank GP target
   ``Sigma_target``, checking pointwise ``log_weights`` to ``atol=1e-12``, ``Delta log Z``
   evidence correction, and reweighted posterior moments against closed-form conjugate
   Gaussian formulas ``N(mu_1, V_1)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Callable, NamedTuple

import jax

jax.config.update("jax_enable_x64", True)

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla
import scipy.special as sps
import scipy.stats as sstats
import tinygp.kernels as tgk
from tinygp.kernels import quasisep as tgq

import ticktack as tt
from ticktack.data import SeasonalWeights
import ticktack.sampling as tts


# =============================================================================
# Shared independent NumPy/SciPy LTI & continuous-time SDE reference kernels
# =============================================================================


class CompanionSDE(NamedTuple):
    """Continuous-time LTI SDE representation ``dx = F x dt + L dW``, ``q = h^T x``."""

    drift: np.ndarray
    diffusion_lql: np.ndarray
    obs_vec: np.ndarray
    state_scales: np.ndarray


def _sde_from_textbook(kind: str, params: dict[str, float]) -> CompanionSDE:
    """Construct companion SDE matrices independently from Särkkä & Solin (2019, §12.3)."""
    if kind == "exp":
        ell, sig = float(params["scale"]), float(params["sigma"])
        lam = 1.0 / ell
        return CompanionSDE(
            drift=np.array([[-lam]], dtype=np.float64),
            diffusion_lql=np.array([[2.0 * lam * sig**2]], dtype=np.float64),
            obs_vec=np.array([1.0], dtype=np.float64),
            state_scales=np.array([sig], dtype=np.float64),
        )
    if kind == "matern32":
        ell, sig = float(params["scale"]), float(params["sigma"])
        lam = math.sqrt(3.0) / ell
        return CompanionSDE(
            drift=np.array([[0.0, 1.0], [-(lam**2), -2.0 * lam]], dtype=np.float64),
            diffusion_lql=np.array([[0.0, 0.0], [0.0, 4.0 * (lam**3) * (sig**2)]], dtype=np.float64),
            obs_vec=np.array([1.0, 0.0], dtype=np.float64),
            state_scales=np.array([sig, sig * lam], dtype=np.float64),
        )
    if kind == "matern52":
        ell, sig = float(params["scale"]), float(params["sigma"])
        lam = math.sqrt(5.0) / ell
        lql = np.zeros((3, 3), dtype=np.float64)
        lql[2, 2] = (16.0 / 3.0) * (lam**5) * (sig**2)
        return CompanionSDE(
            drift=np.array(
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-(lam**3), -3.0 * (lam**2), -3.0 * lam]],
                dtype=np.float64,
            ),
            diffusion_lql=lql,
            obs_vec=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            state_scales=np.array([sig, sig * lam, sig * (lam**2)], dtype=np.float64),
        )
    if kind == "sho":
        w0, qual, sig = float(params["omega"]), float(params["quality"]), float(params["sigma"])
        return CompanionSDE(
            drift=np.array([[0.0, 1.0], [-(w0**2), -w0 / qual]], dtype=np.float64),
            diffusion_lql=np.array(
                [[0.0, 0.0], [0.0, 2.0 * (sig**2) * (w0**3) / qual]], dtype=np.float64
            ),
            obs_vec=np.array([1.0, 0.0], dtype=np.float64),
            state_scales=np.array([sig, sig * w0], dtype=np.float64),
        )
    if kind == "matern32_cosine":
        ell, period, sig = float(params["scale"]), float(params["period"]), float(params["sigma"])
        base = _sde_from_textbook("matern32", {"scale": ell, "sigma": sig})
        w_c = 2.0 * math.pi / period
        f_cos = np.array([[0.0, -w_c], [w_c, 0.0]], dtype=np.float64)
        p_base = sla.solve_continuous_lyapunov(base.drift, -base.diffusion_lql)
        f_prod = np.kron(base.drift, np.eye(2)) + np.kron(np.eye(2), f_cos)
        p_prod = 0.5 * (np.kron(p_base, np.eye(2)) + np.kron(p_base, np.eye(2)).T)
        raw_q = -(f_prod @ p_prod + p_prod @ f_prod.T)
        return CompanionSDE(
            drift=f_prod,
            diffusion_lql=0.5 * (raw_q + raw_q.T),
            obs_vec=np.kron(base.obs_vec, np.array([1.0, 0.0], dtype=np.float64)),
            state_scales=np.kron(base.state_scales, np.ones(2, dtype=np.float64)),
        )
    raise ValueError(f"Unsupported SDE kind {kind!r}.")


def _sum_sdes(parts: list[CompanionSDE]) -> CompanionSDE:
    """Form the block-diagonal direct sum of independent continuous-time SDEs."""
    return CompanionSDE(
        drift=sla.block_diag(*(p.drift for p in parts)),
        diffusion_lql=sla.block_diag(*(p.diffusion_lql for p in parts)),
        obs_vec=np.concatenate([p.obs_vec for p in parts]),
        state_scales=np.concatenate([p.state_scales for p in parts]),
    )


def _cbm_lti_from_scratch_numpy(
    cbm: tt.CarbonBoxModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(A_np, b_np, M_np)`` in plain NumPy without ``matrix_A`` or ``rate_matrix_K``."""
    n_box = cbm.n_boxes
    src_idx = np.asarray(cbm.topology.edges_src, dtype=np.int64)
    dst_idx = np.asarray(cbm.topology.edges_dst, dtype=np.int64)

    if hasattr(cbm, "edge_rates") and hasattr(cbm, "_perron_simplex"):
        k_edges = np.asarray(cbm.edge_rates(), dtype=np.float64)
        k_mat = np.zeros((n_box, n_box), dtype=np.float64)
        np.add.at(k_mat, (dst_idx, src_idx), k_edges)
        np.add.at(k_mat, (src_idx, src_idx), -k_edges)
        d_exit = np.abs(np.diag(k_mat))
        a_ls = np.vstack([k_mat / d_exit[None, :], np.ones((1, n_box), dtype=np.float64)])
        b_ls = np.zeros(n_box + 1, dtype=np.float64)
        b_ls[-1] = 1.0
        q_vec, _, _, _ = sla.lstsq(a_ls, b_ls)
        q_vec = q_vec - sla.lstsq(a_ls, a_ls @ q_vec - b_ls)[0]
        pi_raw = q_vec / d_exit
        log_mtot = float(getattr(cbm, "log_total_mass", getattr(cbm, "base_log_total_mass", 0.0)))
        masses = math.exp(log_mtot) * (pi_raw / np.sum(pi_raw))
    else:
        masses = np.asarray(cbm.reservoir_masses(), dtype=np.float64)
        flux_mat = np.asarray(cbm.flux_matrix(), dtype=np.float64)
        rates_out = flux_mat / masses[:, None]
        k_mat = rates_out.T - np.diag(np.sum(rates_out, axis=1))

    sinks = np.asarray(cbm.sink_rates(), dtype=np.float64)
    a_np = k_mat - np.diag(sinks) - float(cbm.decay_rate) * np.eye(n_box, dtype=np.float64)
    if hasattr(cbm, "prod_weights"):
        b_np = np.zeros(n_box, dtype=np.float64)
        pw = np.asarray(cbm.prod_weights, dtype=np.float64)
        b_np[np.asarray(cbm.topology.prod_box_indices, dtype=np.int64)] = pw / np.sum(pw)
    else:
        b_np = np.asarray(cbm.production_vector(), dtype=np.float64)
    return a_np, b_np, masses


def _solve_joint_lyapunov_scipy(
    a_np: np.ndarray,
    b_eff: np.ndarray,
    masses_np: np.ndarray,
    sde: CompanionSDE,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve ``F_aug P_inf + P_inf F_aug^T + L Q L^T = 0`` via equilibrated SciPy Lyapunov."""
    n_box = a_np.shape[0]
    d_gp = sde.drift.shape[0]
    f_aug = np.block(
        [
            [a_np, np.outer(b_eff, sde.obs_vec)],
            [np.zeros((d_gp, n_box), dtype=np.float64), sde.drift],
        ]
    )
    lql_aug = sla.block_diag(np.zeros((n_box, n_box), dtype=np.float64), sde.diffusion_lql)
    d_scale = np.concatenate([masses_np, sde.state_scales])
    f_bal = (f_aug / d_scale[:, None]) * d_scale[None, :]
    lql_bal = lql_aug / (d_scale[:, None] * d_scale[None, :])
    p_bal = sla.solve_continuous_lyapunov(f_bal, -lql_bal)
    p_bal = 0.5 * (p_bal + p_bal.T)
    p_inf = 0.5 * ((p_bal * (d_scale[:, None] * d_scale[None, :])) + (p_bal * (d_scale[:, None] * d_scale[None, :])).T)
    bal_resid = float(np.max(np.abs(f_bal @ p_bal + p_bal @ f_bal.T + lql_bal))) / max(
        float(np.max(np.abs(lql_bal))), 1e-30
    )
    return p_inf[:n_box, :n_box], p_inf, bal_resid


# =============================================================================
# 1. Fisher Matrix vs Monte Carlo Sampled Score Covariance and Negative Hessian
# =============================================================================


@dataclass(frozen=True)
class FisherOracleSetup:
    """Compiled score/Hessian kernels and reference model for the MC Fisher oracle."""

    forward_fn: Callable[[jax.Array], jax.Array]
    score_batch_fn: Callable[[jax.Array, jax.Array], jax.Array]
    neg_hess_batch_fn: Callable[[jax.Array, jax.Array], jax.Array]
    theta0: jax.Array
    mu0: np.ndarray
    sigma_1d: np.ndarray
    cov_corr: np.ndarray


@pytest.fixture(scope="module")
def fisher_oracle_setup() -> FisherOracleSetup:
    """Build a 7-parameter nonlinear ``EffectiveCBM(K=3) + MiyakeSpike`` NH+SH setup."""
    cbm_eff = tt.EffectiveCBM(K=3)
    prod = tt.ProductionSum(
        steady=tt.SteadyProduction(),
        spike=tt.MiyakeSpike(t_start=774.35, duration=0.30, fluence=7.5, profile="boxcar"),
    )
    model = tt.RadiocarbonModel(cbm_eff, prod)
    years = np.arange(770.0, 792.0, 1.0)
    s_nh = tt.TreeRingSeries(
        years,
        np.zeros_like(years),
        np.linspace(1.1, 1.9, len(years)),
        hemisphere="north",
        seasonal_weights="april-september",
        name="nh",
    )
    s_sh = tt.TreeRingSeries(
        years,
        np.zeros_like(years),
        np.linspace(1.3, 1.7, len(years)),
        hemisphere="south",
        year_convention="schulman_start",
        seasonal_weights="october-march",
        name="sh",
    )
    ds = tt.Dataset([s_nh, s_sh])
    ps = tt.ParameterSpace(
        model,
        [
            tt.ParamSpec("cbm.log_k_strat_to_trop", "identity", float(cbm_eff.log_k_strat_to_trop), 0.3),
            tt.ParamSpec("cbm.log_k_trop_to_strat", "identity", float(cbm_eff.log_k_trop_to_strat), 0.3),
            tt.ParamSpec("cbm.log_k_trop_to_sinks", "identity", cbm_eff.log_k_trop_to_sinks, 0.3),
            tt.ParamSpec("cbm.sink_tau_params", "identity", cbm_eff.sink_tau_params, 0.4),
            tt.ParamSpec("production.spike.t_start", "identity", 774.35, 0.5),
            tt.ParamSpec("production.spike.duration", "exp", math.log(0.30), 0.4),
            tt.ParamSpec("production.spike.fluence", "exp", math.log(7.5), 0.4),
        ],
    )
    theta0 = ps.unconstrain(model)

    def forward_fn(th: jax.Array) -> jax.Array:
        return ps.constrain(th, model).predict_d14c(ds, concat=True)

    def quad_loglike(y_obs: jax.Array, prec_mat: jax.Array, th: jax.Array) -> jax.Array:
        resid = y_obs - forward_fn(th)
        return -0.5 * jnp.dot(resid, prec_mat @ resid)

    @jax.jit
    def score_batch_fn(y_batch: jax.Array, prec_mat: jax.Array) -> jax.Array:
        return jax.lax.map(
            lambda y_s: jax.grad(lambda th: quad_loglike(y_s, prec_mat, th))(theta0),
            y_batch,
            batch_size=64,
        )

    @jax.jit
    def neg_hess_batch_fn(y_batch: jax.Array, prec_mat: jax.Array) -> jax.Array:
        return jax.vmap(lambda y_s: -jax.hessian(lambda th: quad_loglike(y_s, prec_mat, th))(theta0))(
            y_batch
        )

    mu0 = np.asarray(forward_fn(theta0), dtype=np.float64)
    sigma_1d = np.asarray(ds.sigma_flat, dtype=np.float64)
    rng = np.random.default_rng(20261123)
    b_factor = rng.normal(0.0, 0.45, size=(ds.n_obs, 5))
    cov_corr = np.diag(sigma_1d**2) + b_factor @ b_factor.T
    return FisherOracleSetup(
        forward_fn, score_batch_fn, neg_hess_batch_fn, theta0, mu0, sigma_1d, cov_corr
    )


def _assert_wishart_score_and_hessian_oracle(
    setup: FisherOracleSetup,
    cov_mat: np.ndarray,
    fisher_expected: np.ndarray,
    seed: int,
) -> None:
    """Verify Wishart score covariance convergence/calibration and ``E[-Hess] == F``."""
    n_obs = setup.mu0.shape[0]
    p_dim = int(setup.theta0.shape[0])
    l_cov = np.linalg.cholesky(cov_mat)
    prec_jax = jnp.asarray(
        sla.cho_solve((l_cov, True), np.eye(n_obs, dtype=np.float64)), dtype=jnp.float64
    )

    rng = np.random.default_rng(seed)
    s_max = 3072
    y_mc = jnp.asarray(setup.mu0[None, :] + rng.standard_normal((s_max, n_obs)) @ l_cov.T)
    scores = np.asarray(setup.score_batch_fn(y_mc, prec_jax), dtype=np.float64)

    l_fisher = np.linalg.cholesky(fisher_expected)
    scores_white = sla.solve_triangular(l_fisher, scores.T, lower=True).T
    fro_raw_expected = math.sqrt(float(np.trace(fisher_expected) ** 2 + np.sum(fisher_expected**2)))
    fro_white_expected = math.sqrt(float(p_dim * (p_dim + 1)))

    errs_white: list[float] = []
    for s_sub in (128, 512, 1536, s_max):
        cov_raw_s = np.cov(scores[:s_sub], rowvar=False, ddof=1)
        cov_w_s = np.cov(scores_white[:s_sub], rowvar=False, ddof=1)
        err_raw = float(np.linalg.norm(cov_raw_s - fisher_expected, ord="fro"))
        err_w = float(np.linalg.norm(cov_w_s - np.eye(p_dim), ord="fro"))
        errs_white.append(err_w)
        assert 0.35 < err_raw / (fro_raw_expected / math.sqrt(s_sub - 1)) < 2.2
        assert 0.45 < err_w / (fro_white_expected / math.sqrt(s_sub - 1)) < 1.85
    assert errs_white[-1] < 0.35 * errs_white[0]

    cov_w_full = np.cov(scores_white, rowvar=False, ddof=1)
    z_diag = (np.diag(cov_w_full) - 1.0) / math.sqrt(2.0 / (s_max - 1))
    iu = np.triu_indices(p_dim, k=1)
    z_off = cov_w_full[iu] / math.sqrt(1.0 / (s_max - 1))
    z_all = np.concatenate([z_diag, z_off])
    assert float(np.max(np.abs(z_all))) < 3.8
    cdf_val = float(sstats.chi2.cdf(float(np.sum(z_all**2)), df=p_dim * (p_dim + 1) // 2))
    assert 0.005 < cdf_val < 0.995

    cov_raw_full = np.cov(scores, rowvar=False, ddof=1)
    var_raw = (np.outer(np.diag(fisher_expected), np.diag(fisher_expected)) + fisher_expected**2) / (
        s_max - 1
    )
    assert float(np.max(np.abs((cov_raw_full - fisher_expected) / np.sqrt(var_raw)))) < 3.8

    eps_anti = rng.standard_normal((6, n_obs)) @ l_cov.T
    y_anti = jnp.asarray(
        np.concatenate([setup.mu0[None, :] + eps_anti, setup.mu0[None, :] - eps_anti], axis=0)
    )
    hess_draws = np.asarray(setup.neg_hess_batch_fn(y_anti, prec_jax), dtype=np.float64)
    assert np.linalg.norm(hess_draws[0] - fisher_expected) / np.linalg.norm(fisher_expected) > 1e-3
    np.testing.assert_allclose(np.mean(hess_draws, axis=0), fisher_expected, rtol=1e-11, atol=1e-12)


class TestFisherMonteCarloScoreAndHessian:
    """Section 3 line 91: ``expected_fisher`` vs sampled score covariance and ``E[-Hess]``."""

    def test_diagonal_noise_score_wishart_and_expected_hessian(
        self, fisher_oracle_setup: FisherOracleSetup
    ) -> None:
        f_sigma = np.asarray(
            tt.expected_fisher(
                fisher_oracle_setup.forward_fn,
                fisher_oracle_setup.theta0,
                fisher_oracle_setup.sigma_1d,
                kind="sigma",
            )
        )
        f_auto = np.asarray(
            tt.expected_fisher(
                fisher_oracle_setup.forward_fn,
                fisher_oracle_setup.theta0,
                fisher_oracle_setup.sigma_1d,
                kind="auto",
            )
        )
        np.testing.assert_allclose(f_sigma, f_auto, rtol=1e-14, atol=1e-14)
        _assert_wishart_score_and_hessian_oracle(
            fisher_oracle_setup,
            np.diag(fisher_oracle_setup.sigma_1d**2),
            f_sigma,
            seed=104729,
        )

    def test_correlated_noise_score_wishart_expected_hessian_and_pytree(
        self, fisher_oracle_setup: FisherOracleSetup
    ) -> None:
        cov_mat = fisher_oracle_setup.cov_corr
        l_cov = np.linalg.cholesky(cov_mat)
        prec_mat = sla.cho_solve((l_cov, True), np.eye(cov_mat.shape[0], dtype=np.float64))

        f_cov = np.asarray(
            tt.expected_fisher(
                fisher_oracle_setup.forward_fn, fisher_oracle_setup.theta0, cov_mat, kind="cov"
            )
        )
        f_prec = np.asarray(
            tt.expected_fisher(
                fisher_oracle_setup.forward_fn, fisher_oracle_setup.theta0, prec_mat, kind="precision"
            )
        )
        f_auto = np.asarray(
            tt.expected_fisher(
                fisher_oracle_setup.forward_fn, fisher_oracle_setup.theta0, prec_mat, kind="auto"
            )
        )
        np.testing.assert_allclose(f_cov, f_prec, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(f_cov, f_auto, rtol=1e-14, atol=1e-14)
        _assert_wishart_score_and_hessian_oracle(
            fisher_oracle_setup, cov_mat, f_cov, seed=209713
        )

        cbm_eff = tt.EffectiveCBM(K=2)
        spike0 = tt.MiyakeSpike(t_start=774.4, duration=0.4, fluence=7.0, profile="boxcar")
        s_ring = tt.TreeRingSeries(
            np.arange(772.0, 779.0), np.zeros(7), np.full(7, 1.3), hemisphere="north"
        )
        f_tree, unravel_fn = tt.expected_fisher(
            lambda p: tt.RadiocarbonModel(p["cbm"], tt.SteadyProduction() + p["spike"]).predict_d14c(
                s_ring
            ),
            {"cbm": cbm_eff, "spike": spike0},
            s_ring.sigma_d14c,
            return_unravel=True,
        )
        assert f_tree.shape == (5, 5)
        rebuilt = unravel_fn(
            jnp.array(
                [
                    float(cbm_eff.log_k_strat_to_trop),
                    float(cbm_eff.log_k_trop_to_strat),
                    774.4,
                    0.4,
                    7.0,
                ]
            )
        )
        assert float(rebuilt["spike"].fluence) == pytest.approx(7.0, rel=1e-14)


# =============================================================================
# 2. Stationary Initial-State Covariance vs scipy.linalg.solve_continuous_lyapunov
# =============================================================================


class LyapunovCase(NamedTuple):
    """Test case pairing a CBM preset with a Markov GP kernel and textbook SDE."""

    preset_name: str
    kernel_label: str
    kernel: tgk.Kernel
    sde: CompanionSDE


def _build_lyapunov_cases() -> list[LyapunovCase]:
    """Assemble multi-box CBM + Markov SDE test cases spanning stiff multi-decade scales."""
    sde_m32 = _sde_from_textbook("matern32", {"scale": 3.7, "sigma": 0.45})
    sde_m52 = _sde_from_textbook("matern52", {"scale": 8.5, "sigma": 0.62})
    sde_sho = _sde_from_textbook("sho", {"omega": 2.0 * math.pi / 11.0, "quality": 4.5, "sigma": 0.38})
    sde_exp = _sde_from_textbook("exp", {"scale": 1.8, "sigma": 0.25})
    sde_sum = _sum_sdes([sde_m52, sde_sho, sde_exp])
    sde_prod = _sde_from_textbook("matern32_cosine", {"scale": 18.0, "period": 11.0, "sigma": 0.40})

    kern_m32 = tgq.Matern32(scale=3.7, sigma=0.45)
    kern_m52 = (0.62**2) * tgk.Matern52(scale=8.5)
    kern_sho = tgq.SHO(omega=2.0 * math.pi / 11.0, quality=4.5, sigma=0.38)
    kern_sum = (
        (0.62**2) * tgk.Matern52(scale=8.5)
        + tgq.SHO(omega=2.0 * math.pi / 11.0, quality=4.5, sigma=0.38)
        + tgq.Exp(scale=1.8, sigma=0.25)
    )
    kern_prod = (0.40**2) * (tgk.Matern32(scale=18.0) * tgk.Cosine(scale=11.0))

    return [
        LyapunovCase("Guttler15", "matern32", kern_m32, sde_m32),
        LyapunovCase("Guttler15", "sho_solar", kern_sho, sde_sho),
        LyapunovCase("Guttler15", "prod_m32_cos", kern_prod, sde_prod),
        LyapunovCase("Brehm21", "matern52", kern_m52, sde_m52),
        LyapunovCase("Brehm21", "sum_m52_sho_exp", kern_sum, sde_sum),
        LyapunovCase("Brehm21", "prod_m32_cos", kern_prod, sde_prod),
        LyapunovCase("Miyake17", "sum_m52_sho_exp", kern_sum, sde_sum),
    ]


LYAPUNOV_CASES = _build_lyapunov_cases()


@pytest.mark.parametrize(
    "case",
    LYAPUNOV_CASES,
    ids=[f"{c.preset_name}-{c.kernel_label}" for c in LYAPUNOV_CASES],
)
def test_stationary_cbm_gp_cov_against_scipy_continuous_lyapunov(case: LyapunovCase) -> None:
    """Section 3 line 94: ``stationary_cbm_gp_cov`` matches ``solve_continuous_lyapunov``."""
    cbm = tt.load_preset(case.preset_name)
    a_np, b_np, masses_np = _cbm_lti_from_scratch_numpy(cbm)
    alpha_u = 1.37 * float(tt.ALPHA_ATOMS_CM2_S_TO_KG_YR)

    sig_y0_ref, p_inf_ref, bal_resid = _solve_joint_lyapunov_scipy(
        a_np, alpha_u * b_np, masses_np, case.sde
    )
    assert bal_resid < 1e-13
    sig_conc_ref = sig_y0_ref / (masses_np[:, None] * masses_np[None, :])
    evals_conc = np.linalg.eigvalsh(sig_conc_ref)
    assert float(evals_conc[-1]) > 0.0
    assert float(evals_conc[0]) >= -1e-14 * float(evals_conc[-1])

    sig_y0_default, _ = tt.stationary_cbm_gp_cov(
        cbm.matrix_A(), cbm.production_vector(), case.kernel, alpha_unit=alpha_u
    )
    sig_y0_tt, root_y0_tt, sig_joint_tt = tt.stationary_cbm_gp_cov(
        cbm.matrix_A(),
        cbm.production_vector(),
        case.kernel,
        alpha_unit=alpha_u,
        jitter=0.0,
        return_full_state=True,
    )
    sig_y0_tt_np = np.asarray(sig_y0_tt, dtype=np.float64)
    root_y0_tt_np = np.asarray(root_y0_tt, dtype=np.float64)

    np.testing.assert_allclose(np.asarray(sig_y0_default), sig_y0_ref, rtol=1e-10, atol=0.0)
    np.testing.assert_allclose(sig_y0_tt_np, sig_y0_ref, rtol=1e-10, atol=0.0)
    np.testing.assert_allclose(
        root_y0_tt_np @ root_y0_tt_np.T, sig_y0_ref, rtol=1e-10, atol=1e-22
    )
    assert sig_joint_tt.shape == p_inf_ref.shape


# =============================================================================
# 3. Marginal Likelihood, Linear Posterior & Gradients vs Independent Dense MVN
# =============================================================================


class DenseMVNReference(NamedTuple):
    """Outputs of the from-scratch NumPy/SciPy dense marginal Gaussian oracle."""

    m_dense: np.ndarray
    c_dense: np.ndarray
    c_gp_obs: np.ndarray
    x_canon: np.ndarray
    post_mean_canon: np.ndarray
    post_cov_canon: np.ndarray
    ring_pred_mean: np.ndarray


def _int_exp_a_scipy(a_np: np.ndarray, dt: float) -> np.ndarray:
    """Compute ``int_0^{dt} exp(A s) ds`` via a ``2n x 2n`` SciPy ``expm``."""
    n_box = a_np.shape[0]
    if dt <= 0.0:
        return np.zeros((n_box, n_box), dtype=np.float64)
    blk = np.zeros((2 * n_box, 2 * n_box), dtype=np.float64)
    blk[:n_box, n_box:] = np.eye(n_box, dtype=np.float64)
    blk[n_box:, n_box:] = a_np
    return sla.expm(blk * dt)[:n_box, n_box:]


def _boxcar_spike_mean_scipy(
    a_np: np.ndarray,
    b_eff: np.ndarray,
    spike: tt.MiyakeSpike,
    subwindows: list[list[tuple[float, float, float]]],
    obs_boxes: np.ndarray,
) -> np.ndarray:
    """Integrate a boxcar ``MiyakeSpike`` across arbitrary seasonal sub-windows in SciPy."""
    n_box = a_np.shape[0]
    t0_sp = float(spike.t_start)
    t1_sp = t0_sp + float(spike.duration)
    rate_sp = float(spike.fluence) / float(spike.duration)
    out = np.empty(len(subwindows), dtype=np.float64)

    for i, wins in enumerate(subwindows):
        box_c = int(obs_boxes[i])
        acc_i = 0.0
        for ar, br, w_sub in wins:
            knots = sorted({ar, br, min(max(t0_sp, ar), br), min(max(t1_sp, ar), br)})
            if ar <= t0_sp:
                y_cur = np.zeros(n_box, dtype=np.float64)
            elif ar <= t1_sp:
                y_cur = rate_sp * (_int_exp_a_scipy(a_np, ar - t0_sp) @ b_eff)
            else:
                y_end = rate_sp * (_int_exp_a_scipy(a_np, t1_sp - t0_sp) @ b_eff)
                y_cur = sla.expm(a_np * (ar - t1_sp)) @ y_end
            int_y = np.zeros(n_box, dtype=np.float64)
            for ka, kb in zip(knots[:-1], knots[1:]):
                dt = kb - ka
                if dt <= 0.0:
                    continue
                q_seg = rate_sp if (t0_sp < 0.5 * (ka + kb) < t1_sp) else 0.0
                aug = np.zeros((2 * n_box + 1, 2 * n_box + 1), dtype=np.float64)
                aug[:n_box, n_box : 2 * n_box] = np.eye(n_box)
                aug[n_box : 2 * n_box, n_box : 2 * n_box] = a_np
                aug[n_box : 2 * n_box, 2 * n_box] = b_eff * q_seg
                st1 = sla.expm(aug * dt) @ np.concatenate([np.zeros(n_box), y_cur, [1.0]])
                int_y += st1[:n_box]
                y_cur = st1[n_box : 2 * n_box]
            acc_i += w_sub * float(int_y[box_c]) / (br - ar)
        out[i] = acc_i
    return out


def _boxcar_spike_sensitivities_scipy(
    a_np: np.ndarray,
    b_eff: np.ndarray,
    spike: tt.MiyakeSpike,
    subwindows: list[list[tuple[float, float, float]]],
    obs_boxes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact analytical ``(d_dy/d_tstart, d_dy/d_duration)`` via Leibniz rule."""
    n_box = a_np.shape[0]
    t0_sp = float(spike.t_start)
    dur_sp = float(spike.duration)
    t1_sp = t0_sp + dur_sp
    rate_sp = float(spike.fluence) / dur_sp
    dy_val = _boxcar_spike_mean_scipy(a_np, b_eff, spike, subwindows, obs_boxes)

    def _impulse_window_avg(tau_imp: float, ar: float, br: float) -> np.ndarray:
        if br <= tau_imp:
            return np.zeros(n_box, dtype=np.float64)
        if ar <= tau_imp:
            return (_int_exp_a_scipy(a_np, br - tau_imp) @ b_eff) / (br - ar)
        return (
            (_int_exp_a_scipy(a_np, br - tau_imp) - _int_exp_a_scipy(a_np, ar - tau_imp)) @ b_eff
        ) / (br - ar)

    d_t0 = np.empty(len(subwindows), dtype=np.float64)
    d_dur = np.empty(len(subwindows), dtype=np.float64)
    for i, wins in enumerate(subwindows):
        box_c = int(obs_boxes[i])
        g_t0 = sum(w_s * float(_impulse_window_avg(t0_sp, ar, br)[box_c]) for ar, br, w_s in wins)
        g_t1 = sum(w_s * float(_impulse_window_avg(t1_sp, ar, br)[box_c]) for ar, br, w_s in wins)
        d_t0[i] = rate_sp * (g_t1 - g_t0)
        d_dur[i] = -dy_val[i] / dur_sp + rate_sp * g_t1
    return d_t0, d_dur


def _propagate_lti_chain_scipy(
    mat_op: np.ndarray, st0: np.ndarray, t_start: float, sorted_ts: list[float]
) -> np.ndarray:
    """Propagate ``d(st)/dt = mat_op @ st`` across ``sorted_ts`` with ``dt <= 1.5`` substeps."""
    out = np.empty((len(sorted_ts), *st0.shape), dtype=np.float64)
    cur = st0.copy()
    t_prev = t_start
    for idx, tv in enumerate(sorted_ts):
        dt = tv - t_prev
        if dt > 0.0:
            n_sub = max(1, int(math.ceil(dt / 1.5)))
            step_m = sla.expm(mat_op * (dt / n_sub))
            for _ in range(n_sub):
                cur = step_m @ cur
        out[idx] = cur
        t_prev = tv
    return out


def _unique_window_times(
    subwindows: list[list[tuple[float, float, float]]],
) -> tuple[float, list[float], dict[float, int]]:
    """Collect sorted unique subwindow boundary times and earliest observation ``t0_obs``."""
    pts: set[float] = set()
    for wins in subwindows:
        for ar, br, _ in wins:
            pts.add(ar)
            pts.add(br)
    uniq_t = sorted(pts)
    return uniq_t[0], uniq_t, {tv: k for k, tv in enumerate(uniq_t)}


def _window_avg_antideriv(
    st_ut: np.ndarray,
    t_idx: dict[float, int],
    subwindows: list[list[tuple[float, float, float]]],
    n_box: int,
) -> np.ndarray:
    """Average propagated antiderivative states ``st_ut[:, :n_box, ...]`` over ``subwindows``."""
    tail_shape = st_ut.shape[2:]
    out = np.zeros((len(subwindows), n_box, *tail_shape), dtype=np.float64)
    for i, wins in enumerate(subwindows):
        for ar, br, w_sub in wins:
            out[i] += w_sub * (st_ut[t_idx[br], :n_box] - st_ut[t_idx[ar], :n_box]) / (br - ar)
    return out


def _harmonic_and_trend_designs_scipy(
    a_np: np.ndarray,
    b_eff: np.ndarray,
    solar: tt.HarmonicSolar,
    trend: tt.LinearTrend,
    subwindows: list[list[tuple[float, float, float]]],
) -> tuple[np.ndarray, np.ndarray]:
    """Build unscaled ``(X_sol_raw, X_tr_raw)`` ``(N, n_box, k)`` via forward ``expm`` ODEs."""
    n_box = a_np.shape[0]
    n_obs = len(subwindows)
    t0_obs, uniq_t, t_idx = _unique_window_times(subwindows)
    omegas = 2.0 * math.pi / np.asarray(solar.periods, dtype=np.float64)

    x_sol_raw = np.zeros((n_obs, n_box, 2 * len(omegas)), dtype=np.float64)
    for m_idx, wm in enumerate(omegas):
        blk_2n = np.block([[-a_np, -wm * np.eye(n_box)], [wm * np.eye(n_box), -a_np]])
        uv = sla.solve(blk_2n, np.concatenate([b_eff, np.zeros(n_box)]))
        c0 = math.cos(wm * (t0_obs - solar.t_ref))
        s0 = math.sin(wm * (t0_obs - solar.t_ref))
        u_t0 = uv[:n_box] * c0 - uv[n_box:] * s0
        v_t0 = uv[n_box:] * c0 + uv[:n_box] * s0
        osc = np.zeros((2 * n_box + 2, 2 * n_box + 2), dtype=np.float64)
        osc[:n_box, n_box : 2 * n_box] = np.eye(n_box)
        osc[n_box : 2 * n_box, n_box : 2 * n_box] = a_np
        osc[n_box : 2 * n_box, 2 * n_box] = b_eff
        osc[2 * n_box, 2 * n_box + 1] = -wm
        osc[2 * n_box + 1, 2 * n_box] = wm
        init_pair = np.column_stack(
            [
                np.concatenate([np.zeros(n_box), u_t0, [c0, s0]]),
                np.concatenate([np.zeros(n_box), v_t0, [s0, -c0]]),
            ]
        )
        st_ut = _propagate_lti_chain_scipy(osc, init_pair, t0_obs, uniq_t)
        x_sol_raw[:, :, 2 * m_idx : 2 * m_idx + 2] = _window_avg_antideriv(
            st_ut, t_idx, subwindows, n_box
        )

    x_tr_raw = np.zeros((n_obs, n_box, 1), dtype=np.float64)
    if trend.switched_on:
        ramp = np.zeros((2 * n_box + 2, 2 * n_box + 2), dtype=np.float64)
        ramp[:n_box, n_box : 2 * n_box] = np.eye(n_box)
        ramp[n_box : 2 * n_box, n_box : 2 * n_box] = a_np
        ramp[n_box : 2 * n_box, 2 * n_box] = b_eff
        ramp[2 * n_box, 2 * n_box + 1] = 1.0
        st0_r = np.zeros(2 * n_box + 2, dtype=np.float64)
        st0_r[2 * n_box + 1] = 1.0
        pos_ts = sorted({0.0} | {max(tv - trend.t_ref, 0.0) for tv in uniq_t})
        st_r = _propagate_lti_chain_scipy(ramp, st0_r, 0.0, pos_ts)
        remapped = {tv: pos_ts.index(max(tv - trend.t_ref, 0.0)) for tv in uniq_t}
        x_tr_raw[:, :, 0] = _window_avg_antideriv(st_r, remapped, subwindows, n_box)
    else:
        v_ss = sla.solve(-a_np, b_eff)
        for i, wins in enumerate(subwindows):
            mid_avg = sum(w_sub * (0.5 * (ar + br) - trend.t_ref) for ar, br, w_sub in wins)
            x_tr_raw[i, :, 0] = mid_avg * v_ss
    return x_sol_raw, x_tr_raw


def _unweighted_gp_modes_scipy(
    a_np: np.ndarray,
    b_eff: np.ndarray,
    gp_term: tt.GPProduction,
    subwindows: list[list[tuple[float, float, float]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate unit-amplitude Solin--Särkkä sine modes via SciPy ``expm``."""
    n_box = a_np.shape[0]
    n_obs = len(subwindows)
    assert gp_term.domain is not None
    a_dom, b_dom = gp_term.domain
    half_l = 0.5 * (b_dom - a_dom)
    om_gp = np.arange(1, gp_term.n_modes + 1, dtype=np.float64) * (math.pi / (2.0 * half_l))
    t0_obs, uniq_t, t_idx = _unique_window_times(subwindows)

    unit_design = np.zeros((n_obs, n_box, gp_term.n_modes), dtype=np.float64)
    y_modes_t0 = np.zeros((n_box, gp_term.n_modes), dtype=np.float64)
    for j_idx, wj in enumerate(om_gp):
        osc_j = np.zeros((2 * n_box + 2, 2 * n_box + 2), dtype=np.float64)
        osc_j[:n_box, n_box : 2 * n_box] = np.eye(n_box)
        osc_j[n_box : 2 * n_box, n_box : 2 * n_box] = a_np
        osc_j[n_box : 2 * n_box, 2 * n_box] = b_eff / math.sqrt(half_l)
        osc_j[2 * n_box, 2 * n_box + 1] = wj
        osc_j[2 * n_box + 1, 2 * n_box] = -wj
        st_init = np.zeros(2 * n_box + 2, dtype=np.float64)
        st_init[2 * n_box + 1] = 1.0
        st_ut = _propagate_lti_chain_scipy(osc_j, st_init, float(a_dom), uniq_t)
        y_modes_t0[:, j_idx] = st_ut[t_idx[t0_obs], n_box : 2 * n_box]
        unit_design[:, :, j_idx] = _window_avg_antideriv(st_ut, t_idx, subwindows, n_box)
    return om_gp, unit_design, y_modes_t0


def _gp_design_and_cov_scipy(
    a_np: np.ndarray,
    b_eff: np.ndarray,
    masses_np: np.ndarray,
    gp_term: tt.GPProduction,
    subwindows: list[list[tuple[float, float, float]]],
) -> tuple[np.ndarray, np.ndarray]:
    """Build unscaled ``X_gp_raw`` and initial-state covariance via SciPy ``expm`` + Lyapunov."""
    om_gp, unit_design, y_modes_t0 = _unweighted_gp_modes_scipy(a_np, b_eff, gp_term, subwindows)
    ell_gp, sig_gp = float(gp_term.kernel.scale), float(gp_term.kernel.sigma)
    if isinstance(gp_term.kernel, tgq.Matern52):
        f_m = math.sqrt(5.0) / ell_gp
        psd_gp = (sig_gp**2) * ((16.0 / 3.0) * (f_m**5)) / ((f_m**2 + om_gp**2) ** 3)
        sde_gp = _sde_from_textbook("matern52", {"scale": ell_gp, "sigma": sig_gp})
    else:
        f_m = math.sqrt(3.0) / ell_gp
        psd_gp = (sig_gp**2) * (4.0 * (f_m**3)) / ((f_m**2 + om_gp**2) ** 2)
        sde_gp = _sde_from_textbook("matern32", {"scale": ell_gp, "sigma": sig_gp})

    x_gp_raw = unit_design * np.sqrt(psd_gp)[None, None, :]
    sig_y0, _, _ = _solve_joint_lyapunov_scipy(a_np, b_eff, masses_np, sde_gp)
    delta_cov = 0.5 * (
        (sig_y0 - (y_modes_t0 * psd_gp[None, :]) @ y_modes_t0.T)
        + (sig_y0 - (y_modes_t0 * psd_gp[None, :]) @ y_modes_t0.T).T
    )
    t0_obs, uniq_t, t_idx = _unique_window_times(subwindows)
    int_hom = np.array([_int_exp_a_scipy(a_np, tv - t0_obs) for tv in uniq_t])
    w_hom_raw = _window_avg_antideriv(int_hom, t_idx, subwindows, a_np.shape[0])
    return x_gp_raw, np.einsum("ibk,kl,jdl->ibjd", w_hom_raw, delta_cov, w_hom_raw)


_LITERAL_CALENDAR_RULES: dict[str, tuple[str, tuple[tuple[float, float, float], ...]]] = {
    "nh_std": ("north", ((3.0 / 12.0, 9.0 / 12.0, 1.0),)),
    "c_nh": ("north", ((3.0 / 12.0, 9.0 / 12.0, 1.0),)),
    "nh_ewlw": ("north", ((3.0 / 12.0, 5.0 / 12.0, 0.65), (6.0 / 12.0, 9.0 / 12.0, 0.35))),
    "sh_schul": ("south", ((9.0 / 12.0, 15.0 / 12.0, 1.0),)),
    "c_sh": ("south", ((9.0 / 12.0, 15.0 / 12.0, 1.0),)),
    "sh_end": ("south", ((-4.0 / 12.0, 2.0 / 12.0, 1.0),)),
}


def _extract_dataset_subwindows(
    cbm: tt.CarbonBoxModel, dataset: tt.Dataset
) -> tuple[np.ndarray, np.ndarray, list[list[tuple[float, float, float]]]]:
    """Build literal calendar ``(obs_boxes, obs_series, subwindows)`` without package window getters."""
    obs_boxes_l: list[int] = []
    obs_series_l: list[int] = []
    subwindows: list[list[tuple[float, float, float]]] = []
    names = cbm.topology.box_names
    hemis = cbm.topology.hemispheres
    for s_idx, ser in enumerate(dataset.series):
        hemi_req, rel_wins = _LITERAL_CALENDAR_RULES[str(ser.name)]
        matches = [
            i for i, (bn, bh) in enumerate(zip(names, hemis))
            if bn == "Troposphere" and bh in (hemi_req, "none")
        ]
        assert len(matches) == 1
        box_c = matches[0]
        raw_y = np.asarray(ser.raw_years, dtype=np.float64)
        exp_ts = np.column_stack([raw_y + a_rel for a_rel, _, _ in rel_wins])
        exp_te = np.column_stack([raw_y + b_rel for _, b_rel, _ in rel_wins])
        exp_w = np.array([w_rel for _, _, w_rel in rel_wins], dtype=np.float64)
        np.testing.assert_allclose(np.asarray(ser.sub_t_starts), exp_ts, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(np.asarray(ser.sub_t_ends), exp_te, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(np.asarray(ser.seasonal_weights.weights), exp_w, rtol=1e-14, atol=1e-14)
        for y_val in raw_y:
            obs_boxes_l.append(box_c)
            obs_series_l.append(s_idx)
            subwindows.append([(float(y_val + a_r), float(y_val + b_r), w_r) for a_r, b_r, w_r in rel_wins])
    return np.array(obs_boxes_l, dtype=np.int64), np.array(obs_series_l, dtype=np.int64), subwindows


def _build_dense_mvn_scipy(
    model: tt.RadiocarbonModel,
    dataset: tt.Dataset,
    cfg: tt.LinearMarginalConfig,
    sigma_jitter: float = 0.0,
) -> DenseMVNReference:
    """Build ``(m_dense, C_dense, posteriors)`` from scratch in NumPy/SciPy."""
    a_np, b_np, masses_np = _cbm_lti_from_scratch_numpy(model.cbm)
    q_std, b_eff = float(model.q_std), float(model.alpha_unit) * b_np
    y_ss = sla.solve(-a_np, q_std * b_eff)
    trop = list(model.cbm.topology.trop_box_indices)
    r_ref = float(np.sum(y_ss[trop])) / float(np.sum(masses_np[trop]))

    obs_boxes, obs_series, subwindows = _extract_dataset_subwindows(model.cbm, dataset)
    n_obs, n_ser, idx_n = len(obs_boxes), dataset.n_series, np.arange(len(obs_boxes))
    r_act = (y_ss[obs_boxes] / masses_np[obs_boxes]) / r_ref
    obs_scale = 1000.0 / (r_ref * masses_np[obs_boxes])
    dy_steady = sla.solve(-a_np, (float(model.production.steady.rate) - q_std) * b_eff)[obs_boxes]
    dy_sp = _boxcar_spike_mean_scipy(a_np, b_eff, model.production.spike, subwindows, obs_boxes)
    m0_np = 1000.0 * (r_act - 1.0) + (dy_steady + dy_sp) * obs_scale

    x_off = np.zeros((n_obs, 1 + n_ser), dtype=np.float64)
    x_off[:, 0] = r_act
    x_off[idx_n, 1 + obs_series] = 1.0
    mu_off = np.array([cfg.global_prior_mean] + [0.0] * n_ser, dtype=np.float64)
    sd_off = np.array([cfg.global_prior_sd] + [cfg.lab_prior_sd] * n_ser, dtype=np.float64)

    x_sol_raw, x_tr_raw = _harmonic_and_trend_designs_scipy(
        a_np, b_eff, model.production.solar, model.production.trend, subwindows
    )
    x_sol = x_sol_raw[idx_n, obs_boxes, :] * obs_scale[:, None]
    x_tr = x_tr_raw[idx_n, obs_boxes, :] * obs_scale[:, None]
    if not cfg.marginalize_solar:
        sol_c = np.stack([model.production.solar.c_cos, model.production.solar.c_sin], axis=-1).reshape(-1)
        m0_np = m0_np + x_sol @ sol_c
    if not cfg.marginalize_trend:
        m0_np = m0_np + x_tr[:, 0] * float(model.production.trend.slope)

    x_gp_raw, hom_cov_box = _gp_design_and_cov_scipy(
        a_np, b_eff, masses_np, model.production.gp, subwindows
    )
    x_gp = x_gp_raw[idx_n, obs_boxes, :] * obs_scale[:, None]
    c_hom_obs = hom_cov_box[idx_n[:, None], obs_boxes[:, None], idx_n[None, :], obs_boxes[None, :]] * (
        obs_scale[:, None] * obs_scale[None, :]
    )
    c_gp_obs = (x_gp @ x_gp.T + c_hom_obs) if cfg.include_gp_initial_cov else (x_gp @ x_gp.T)

    sig_vec = np.sqrt(np.asarray(dataset.sigma_flat, dtype=np.float64) ** 2 + sigma_jitter**2)
    c_dense = np.diag(sig_vec**2) + (x_off * (sd_off**2)[None, :]) @ x_off.T
    x_parts, sd_parts = [x_off], [sd_off]
    if cfg.marginalize_solar:
        sd_sol = np.full(x_sol.shape[1], cfg.solar_prior_sd, dtype=np.float64)
        c_dense = c_dense + (x_sol * (sd_sol**2)[None, :]) @ x_sol.T
        x_parts, sd_parts = [*x_parts, x_sol], [*sd_parts, sd_sol]
    if cfg.marginalize_trend:
        sd_tr = np.array([cfg.trend_prior_sd], dtype=np.float64)
        c_dense = c_dense + (x_tr * (sd_tr**2)[None, :]) @ x_tr.T
        x_parts, sd_parts = [*x_parts, x_tr], [*sd_parts, sd_tr]
    if cfg.marginalize_gp:
        c_dense = c_dense + c_gp_obs
        x_parts, sd_parts = [*x_parts, x_gp], [*sd_parts, np.ones(x_gp.shape[1], dtype=np.float64)]
    c_dense, m_dense = 0.5 * (c_dense + c_dense.T), m0_np + x_off @ mu_off
    x_canon = np.concatenate(x_parts, axis=1)
    pmean, pcov, rpred = _dense_woodbury_posterior(
        np.asarray(dataset.d14c_flat, dtype=np.float64),
        (m_dense, c_dense, sig_vec),
        x_canon,
        np.concatenate([s**2 for s in sd_parts]),
        mu_off,
    )
    return DenseMVNReference(m_dense, c_dense, c_gp_obs, x_canon, pmean, pcov, rpred)


def _dense_woodbury_posterior(
    y_obs: np.ndarray,
    mvn_triple: tuple[np.ndarray, np.ndarray, np.ndarray],
    x_canon: np.ndarray,
    var_canon: np.ndarray,
    mu_off: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute closed-form linear Gaussian posterior ``(mean, cov, ring_pred_mean)``."""
    m_dense, c_dense, sig_vec = mvn_triple
    sig_canon = np.diag(var_canon)
    mu_canon = np.zeros(x_canon.shape[1], dtype=np.float64)
    mu_canon[: len(mu_off)] = mu_off
    l_c = np.linalg.cholesky(c_dense)
    alpha_sol = sla.cho_solve((l_c, True), y_obs - m_dense)
    pmean = mu_canon + sig_canon @ (x_canon.T @ alpha_sol)
    pcov = sig_canon - sig_canon @ x_canon.T @ sla.cho_solve((l_c, True), x_canon @ sig_canon)
    rpred = m_dense + (c_dense - np.diag(sig_vec**2)) @ alpha_sol
    return pmean, pcov, rpred


def _build_multi_series_nhsh_dataset() -> tt.Dataset:
    """Build a 4-series (2 NH + 2 SH) dataset with heterogeneous seasonal profiles."""
    sw_ewlw = SeasonalWeights(
        [3.0 / 12.0, 6.0 / 12.0], [5.0 / 12.0, 9.0 / 12.0], [0.65, 0.35], label="ew_lw"
    )
    rng = np.random.default_rng(202611)
    s0 = tt.TreeRingSeries(
        np.arange(771.0, 780.0),
        rng.normal(-14.0, 3.0, 9),
        np.linspace(1.1, 1.7, 9),
        hemisphere="north",
        seasonal_weights="april-september",
        name="nh_std",
    )
    s1 = tt.TreeRingSeries(
        np.arange(772.0, 780.0),
        rng.normal(-13.5, 3.0, 8),
        np.linspace(1.2, 1.8, 8),
        hemisphere="north",
        seasonal_weights=sw_ewlw,
        name="nh_ewlw",
    )
    s2 = tt.TreeRingSeries(
        np.arange(771.0, 779.0),
        rng.normal(-18.5, 3.0, 8),
        np.linspace(1.3, 1.9, 8),
        hemisphere="south",
        year_convention="schulman_start",
        seasonal_weights="october-march",
        name="sh_schul",
    )
    s3 = tt.TreeRingSeries(
        np.arange(772.0, 779.0),
        rng.normal(-18.0, 3.0, 7),
        np.linspace(1.15, 1.65, 7),
        hemisphere="south",
        year_convention="end_year",
        seasonal_weights="september-february",
        name="sh_end",
    )
    return tt.Dataset([s0, s1, s2, s3])


def _dense_chol_logpdf(y_obs: np.ndarray, m_dense: np.ndarray, c_dense: np.ndarray) -> float:
    """Evaluate exact MVN log-PDF via dense NumPy Cholesky without ``eigh`` conditioning loss."""
    l_c = np.linalg.cholesky(c_dense)
    whitened = sla.solve_triangular(l_c, y_obs - m_dense, lower=True)
    return float(
        -0.5 * np.sum(whitened**2)
        - np.sum(np.log(np.diag(l_c)))
        - 0.5 * len(y_obs) * math.log(2.0 * math.pi)
    )


def _build_two_hemisphere_6box_open_cbm(
    sink_nh: float = 0.0024, sink_sh: float = 0.0031
) -> tt.OpenCBM:
    """Build a 6-box 2-hemisphere ``OpenCBM`` with asymmetric NH-SH ocean exchange."""
    box_names = (
        "Stratosphere",
        "Troposphere",
        "Surface Ocean",
        "Stratosphere",
        "Troposphere",
        "Surface Ocean",
    )
    hemispheres = ("north", "north", "north", "south", "south", "south")
    edges_spec = (
        (0, 1, 22.0),
        (1, 0, 22.0),
        (1, 2, 48.0),
        (2, 1, 48.0),
        (3, 4, 20.0),
        (4, 3, 20.0),
        (4, 5, 62.0),
        (5, 4, 62.0),
        (0, 3, 6.0),
        (3, 0, 6.0),
        (1, 4, 35.0),
        (4, 1, 35.0),
        (2, 5, 16.0),
        (5, 2, 16.0),
    )
    masses = np.array([45.0, 365.0, 820.0, 45.0, 365.0, 940.0], dtype=np.float64)
    sinks = np.array([0.0, 0.0, sink_nh, 0.0, 0.0, sink_sh], dtype=np.float64)
    topo = tt.Topology(
        box_names=box_names,
        hemispheres=hemispheres,
        edges_src=tuple(e[0] for e in edges_spec),
        edges_dst=tuple(e[1] for e in edges_spec),
        trop_box_indices=(1, 4),
        prod_box_indices=(0, 1, 3, 4),
    )
    return tt.OpenCBM.from_masses_and_fluxes(
        topology=topo,
        masses=masses,
        edge_fluxes=[e[2] for e in edges_spec],
        sink_rates=sinks,
        prod_weights=[0.40, 0.16, 0.29, 0.15],
    )


def _assert_nhsh_analytic_gradients(
    model: tt.RadiocarbonModel,
    ds: tt.Dataset,
    cfg: tt.LinearMarginalConfig,
) -> None:
    """Verify exact matrix-calculus gradients wrt ``fluence``, ``steady``, ``jitter``, and ``gp.sigma``."""
    y_obs = np.asarray(ds.d14c_flat, dtype=np.float64)
    cfg_noinit = tt.LinearMarginalConfig(
        offset_mode=cfg.offset_mode,
        global_prior_mean=cfg.global_prior_mean,
        global_prior_sd=cfg.global_prior_sd,
        lab_prior_sd=cfg.lab_prior_sd,
        solar_prior_sd=cfg.solar_prior_sd,
        trend_prior_sd=cfg.trend_prior_sd,
        marginalize_solar=True,
        marginalize_trend=True,
        marginalize_gp=True,
        include_gp_initial_cov=False,
    )
    ref_ni = _build_dense_mvn_scipy(model, ds, cfg_noinit, 0.25)
    lc_ni = np.linalg.cholesky(ref_ni.c_dense)
    al_ni = sla.cho_solve((lc_ni, True), y_obs - ref_ni.m_dense)
    wg_ni = np.outer(al_ni, al_ni) - sla.cho_solve((lc_ni, True), np.eye(len(y_obs)))
    ref_fl0 = _build_dense_mvn_scipy(model.set("production.spike.fluence", 0.0), ds, cfg_noinit, 0.25)
    ref_st1 = _build_dense_mvn_scipy(model.set("production.steady.rate", 2.76), ds, cfg_noinit, 0.25)
    g_4_scipy = np.array(
        [
            float(((ref_ni.m_dense - ref_fl0.m_dense) / 7.1) @ al_ni),
            float((ref_st1.m_dense - ref_ni.m_dense) @ al_ni),
            0.5 * float(np.sum(wg_ni * (2.0 * 0.25 * np.eye(len(y_obs))))),
            0.5 * float(np.sum(wg_ni * ((2.0 / 0.18) * ref_ni.c_gp_obs))),
        ],
        dtype=np.float64,
    )
    g_4_tt = np.asarray(
        jax.grad(
            lambda th: tt.marginal_log_likelihood(
                model.set("production.spike.fluence", th[0])
                .set("production.steady.rate", th[1])
                .set("production.gp.kernel", tgq.Matern32(scale=3.8, sigma=th[3])),
                ds,
                cfg_noinit,
                sigma_jitter=th[2],
            )
        )(jnp.array([7.1, 1.76, 0.25, 0.18], dtype=jnp.float64)),
        dtype=np.float64,
    )
    np.testing.assert_allclose(g_4_tt, g_4_scipy, rtol=1e-11, atol=1e-11)


def _matern52_scale_dcov_scipy(model: tt.RadiocarbonModel, dataset: tt.Dataset) -> np.ndarray:
    """Compute exact analytical ``d(C_gp_obs)/d(scale)`` via differentiated SciPy Lyapunov."""
    cbm = model.cbm
    a_np, b_np, masses_np = _cbm_lti_from_scratch_numpy(cbm)
    b_eff = float(model.alpha_unit) * b_np
    y_ss = sla.solve(-a_np, float(model.q_std) * b_eff)
    trop_idx = list(cbm.topology.trop_box_indices)
    r_ref = float(np.sum(y_ss[trop_idx])) / float(np.sum(masses_np[trop_idx]))
    obs_boxes, _, subwindows = _extract_dataset_subwindows(cbm, dataset)
    n_obs, n_box = len(obs_boxes), a_np.shape[0]
    idx_n = np.arange(n_obs)
    obs_scale = 1000.0 / (r_ref * masses_np[obs_boxes])

    gp_t = model.production.gp
    om_gp, unit_design, y_modes_t0 = _unweighted_gp_modes_scipy(a_np, b_eff, gp_t, subwindows)
    ell, sig = float(gp_t.kernel.scale), float(gp_t.kernel.sigma)
    f_m = math.sqrt(5.0) / ell
    psd_gp = (sig**2) * ((16.0 / 3.0) * (f_m**5)) / ((f_m**2 + om_gp**2) ** 3)
    dlog_psd = -5.0 / ell + (6.0 * (f_m**2)) / (ell * (f_m**2 + om_gp**2))

    sde_gp = _sde_from_textbook("matern52", {"scale": ell, "sigma": sig})
    _, p_inf, _ = _solve_joint_lyapunov_scipy(a_np, b_eff, masses_np, sde_gp)
    f_aug = np.block([[a_np, np.outer(b_eff, sde_gp.obs_vec)], [np.zeros((3, n_box)), sde_gp.drift]])
    df_gp = (-f_m / ell) * np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-3.0 * f_m**2, -6.0 * f_m, -3.0]])
    df_aug = sla.block_diag(np.zeros((n_box, n_box)), df_gp)
    dq_aug = sla.block_diag(np.zeros((n_box, n_box)), (-5.0 / ell) * sde_gp.diffusion_lql)
    rhs_tan = 0.5 * ((df_aug @ p_inf + p_inf @ df_aug.T + dq_aug) + (df_aug @ p_inf + p_inf @ df_aug.T + dq_aug).T)
    d_sc = np.concatenate([masses_np, sde_gp.state_scales])
    f_bal = (f_aug / d_sc[:, None]) * d_sc[None, :]
    dp_bal = sla.solve_continuous_lyapunov(f_bal, -(rhs_tan / (d_sc[:, None] * d_sc[None, :])))
    dsig_y0 = (0.5 * (dp_bal + dp_bal.T) * (d_sc[:, None] * d_sc[None, :]))[:n_box, :n_box]

    x_gp = (unit_design * np.sqrt(psd_gp)[None, None, :])[idx_n, obs_boxes, :] * obs_scale[:, None]
    ddelta = 0.5 * (
        (dsig_y0 - (y_modes_t0 * (psd_gp * dlog_psd)[None, :]) @ y_modes_t0.T)
        + (dsig_y0 - (y_modes_t0 * (psd_gp * dlog_psd)[None, :]) @ y_modes_t0.T).T
    )
    t0_obs, uniq_t, t_idx = _unique_window_times(subwindows)
    int_hom = np.array([_int_exp_a_scipy(a_np, tv - t0_obs) for tv in uniq_t])
    w_hom_raw = _window_avg_antideriv(int_hom, t_idx, subwindows, n_box)
    w_hom_obs = w_hom_raw[idx_n, obs_boxes, :] * obs_scale[:, None]
    return (x_gp * dlog_psd[None, :]) @ x_gp.T + w_hom_obs @ ddelta @ w_hom_obs.T


class TestMarginalLikelihoodIndependentDenseMVN:
    """Section 3 line 96: ``marginal_log_likelihood``, ``linear_posterior``, and ``grad`` vs SciPy MVN."""

    def test_nhsh_multi_series_dense_mvn_posteriors_and_gradients(self) -> None:
        ds = _build_multi_series_nhsh_dataset()
        t0_lit = 771.0 + 3.0 / 12.0
        cbm_open = _build_two_hemisphere_6box_open_cbm()
        prod = tt.ProductionSum(
            steady=tt.SteadyProduction(),
            spike=tt.MiyakeSpike(t_start=774.55, duration=0.16, fluence=7.1, profile="boxcar"),
            solar=tt.HarmonicSolar(periods=(11.0, 22.0), c_cos=(0.0, 0.0), c_sin=(0.0, 0.0), t_ref=775.0),
            trend=tt.LinearTrend(slope=0.0, t_ref=770.0, switched_on=True),
            gp=tt.GPProduction(tgq.Matern32(scale=3.8, sigma=0.18), n_modes=14, domain=(t0_lit, 790.0)),
        )
        model = tt.RadiocarbonModel(cbm_open, prod)
        cfg = tt.LinearMarginalConfig(
            offset_mode="hierarchical",
            global_prior_mean=-15.0,
            global_prior_sd=18.0,
            lab_prior_sd=1.4,
            solar_prior_sd=0.35,
            trend_prior_sd=0.18,
            marginalize_solar=True,
            marginalize_trend=True,
            marginalize_gp=True,
            include_gp_initial_cov=True,
        )

        ref = _build_dense_mvn_scipy(model, ds, cfg, sigma_jitter=0.25)
        y_obs = np.asarray(ds.d14c_flat, dtype=np.float64)
        ll_scipy = float(sstats.multivariate_normal.logpdf(y_obs, mean=ref.m_dense, cov=ref.c_dense))
        ll_chol = _dense_chol_logpdf(y_obs, ref.m_dense, ref.c_dense)
        np.testing.assert_allclose(ll_scipy, ll_chol, rtol=1e-11, atol=1e-11)

        for method in ("cholesky", "qr", "rank_one"):
            cfg_m = tt.LinearMarginalConfig(
                offset_mode=cfg.offset_mode,
                global_prior_mean=cfg.global_prior_mean,
                global_prior_sd=cfg.global_prior_sd,
                lab_prior_sd=cfg.lab_prior_sd,
                solar_prior_sd=cfg.solar_prior_sd,
                trend_prior_sd=cfg.trend_prior_sd,
                marginalize_solar=True,
                marginalize_trend=True,
                marginalize_gp=True,
                include_gp_initial_cov=True,
                whiten_method=method,
            )
            ll_tt = float(tt.marginal_log_likelihood(model, ds, cfg_m, sigma_jitter=0.25))
            np.testing.assert_allclose(ll_tt, ll_scipy, rtol=1e-11, atol=1e-11)
            np.testing.assert_allclose(ll_tt, ll_chol, rtol=1e-11, atol=1e-11)

        lp = tt.linear_posterior(model, ds, cfg, sigma_jitter=0.25)
        k_c = ref.post_mean_canon.shape[0]
        np.testing.assert_allclose(np.asarray(lp.coeff_mean[:k_c]), ref.post_mean_canon, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(
            np.asarray(lp.coeff_cov[:k_c, :k_c]), ref.post_cov_canon, rtol=1e-11, atol=1e-11
        )
        np.testing.assert_allclose(float(lp.offset_global_mean), ref.post_mean_canon[0], rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(
            np.asarray(lp.offset_lab_means), ref.post_mean_canon[1 : 1 + ds.n_series], rtol=1e-11, atol=1e-11
        )
        np.testing.assert_allclose(np.asarray(lp.ring_pred_mean), ref.ring_pred_mean, rtol=1e-11, atol=1e-11)
        _assert_nhsh_analytic_gradients(model, ds, cfg)

    def test_pre_observation_gp_domain_subtraction_and_nonlinear_gradients(self) -> None:
        """Verify ``a < t_0`` residual initial-covariance subtraction and nonlinear ``jax.grad``."""
        ds = _build_multi_series_nhsh_dataset()
        cbm_eff = tt.EffectiveCBM(K=3)
        prod = tt.ProductionSum(
            steady=tt.SteadyProduction(),
            spike=tt.MiyakeSpike(t_start=774.55, duration=0.35, fluence=7.1, profile="boxcar"),
            solar=tt.HarmonicSolar(periods=(11.0,), c_cos=(0.0,), c_sin=(0.0,), t_ref=775.0),
            trend=tt.LinearTrend(slope=0.0, t_ref=775.0, switched_on=False),
            gp=tt.GPProduction(tgq.Matern52(scale=5.5, sigma=0.35), n_modes=48, domain=(769.5, 790.0)),
        )
        model = tt.RadiocarbonModel(cbm_eff, prod)
        cfg = tt.LinearMarginalConfig(
            offset_mode="hierarchical",
            global_prior_mean=-14.0,
            global_prior_sd=14.0,
            lab_prior_sd=1.3,
            solar_prior_sd=0.3,
            trend_prior_sd=0.2,
            marginalize_solar=True,
            marginalize_trend=True,
            marginalize_gp=True,
            include_gp_initial_cov=True,
        )

        ref = _build_dense_mvn_scipy(model, ds, cfg, sigma_jitter=0.15)
        y_obs = np.asarray(ds.d14c_flat, dtype=np.float64)
        ll_scipy = float(sstats.multivariate_normal.logpdf(y_obs, mean=ref.m_dense, cov=ref.c_dense))
        ll_tt = float(tt.marginal_log_likelihood(model, ds, cfg, sigma_jitter=0.15))
        np.testing.assert_allclose(ll_tt, ll_scipy, rtol=1e-11, atol=1e-11)

        lp = tt.linear_posterior(model, ds, cfg, sigma_jitter=0.15)
        k_c = ref.post_mean_canon.shape[0]
        np.testing.assert_allclose(np.asarray(lp.coeff_mean[:k_c]), ref.post_mean_canon, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(
            np.asarray(lp.coeff_cov[:k_c, :k_c]), ref.post_cov_canon, rtol=1e-11, atol=1e-11
        )

        a_np, b_np, masses_np = _cbm_lti_from_scratch_numpy(cbm_eff)
        b_eff = float(model.alpha_unit) * b_np
        y_ss = sla.solve(-a_np, float(model.q_std) * b_eff)
        r_ref = float(y_ss[1]) / float(masses_np[1])
        obs_boxes, _, subwindows = _extract_dataset_subwindows(cbm_eff, ds)
        obs_scale = 1000.0 / (r_ref * masses_np[obs_boxes])
        d_dy_t0, d_dy_dur = _boxcar_spike_sensitivities_scipy(
            a_np, b_eff, model.production.spike, subwindows, obs_boxes
        )
        dc_gp_ell = _matern52_scale_dcov_scipy(model, ds)
        l_c = np.linalg.cholesky(ref.c_dense)
        alpha_r = sla.cho_solve((l_c, True), y_obs - ref.m_dense)
        w_grad = np.outer(alpha_r, alpha_r) - sla.cho_solve((l_c, True), np.eye(len(y_obs)))
        g_nl_ref = np.array(
            [
                float((d_dy_t0 * obs_scale) @ alpha_r),
                float((d_dy_dur * obs_scale) @ alpha_r),
                0.5 * float(np.sum(w_grad * dc_gp_ell)),
            ],
            dtype=np.float64,
        )
        g_nl_tt = np.asarray(
            jax.grad(
                lambda th: tt.marginal_log_likelihood(
                    model.set("production.spike.t_start", th[0])
                    .set("production.spike.duration", th[1])
                    .set("production.gp.kernel", tgq.Matern52(scale=th[2], sigma=0.35)),
                    ds,
                    cfg,
                    sigma_jitter=0.15,
                )
            )(jnp.array([774.55, 0.35, 5.5], dtype=jnp.float64)),
            dtype=np.float64,
        )
        np.testing.assert_allclose(g_nl_tt, g_nl_ref, rtol=1e-11, atol=1e-11)


# =============================================================================
# 4. Conjugate Linear-Gaussian NUTS and SMC Importance-Reweighting Oracle
# =============================================================================


class ConjugateGaussianOracle(NamedTuple):
    """Exact closed-form quantities for 2D linear-Gaussian surrogate -> target IS."""

    model: tt.RadiocarbonModel
    dataset: tt.Dataset
    param_space: tt.ParameterSpace
    cfg_surr: tt.LinearMarginalConfig
    cfg_targ: tt.LinearMarginalConfig
    jit_surr: float
    jit_targ: float
    m_base: np.ndarray
    jac_u: np.ndarray
    cov_surr: np.ndarray
    cov_targ: np.ndarray
    mu_surr: np.ndarray
    v_surr: np.ndarray
    ln_z_surr: float
    mu_targ: np.ndarray
    v_targ: np.ndarray
    ln_z_targ: float
    delta_ln_z: float


@dataclass(frozen=True)
class LinearGaussianSystem:
    """Affine observation model ``y = m_base + J u + eps`` with Gaussian prior on ``u``."""

    y_obs: np.ndarray
    m_base: np.ndarray
    jac_u: np.ndarray
    prior_mean: np.ndarray
    prior_cov: np.ndarray


def _conjugate_posterior_from_cov(
    sys_lg: LinearGaussianSystem, cov_data: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute closed-form conjugate posterior ``(mu_post, V_post, ln_Z)``."""
    l_d = np.linalg.cholesky(cov_data)
    d_inv_j = sla.cho_solve((l_d, True), sys_lg.jac_u)
    prec_prior = sla.inv(sys_lg.prior_cov)
    prec_post = prec_prior + sys_lg.jac_u.T @ d_inv_j
    v_post = sla.inv(0.5 * (prec_post + prec_post.T))
    v_post = 0.5 * (v_post + v_post.T)
    rhs = prec_prior @ sys_lg.prior_mean + sys_lg.jac_u.T @ sla.cho_solve(
        (l_d, True), sys_lg.y_obs - sys_lg.m_base
    )
    mu_post = v_post @ rhs
    marg_cov = 0.5 * (
        (cov_data + sys_lg.jac_u @ sys_lg.prior_cov @ sys_lg.jac_u.T)
        + (cov_data + sys_lg.jac_u @ sys_lg.prior_cov @ sys_lg.jac_u.T).T
    )
    marg_mean = sys_lg.m_base + sys_lg.jac_u @ sys_lg.prior_mean
    ln_z = _dense_chol_logpdf(sys_lg.y_obs, marg_mean, marg_cov)
    return mu_post, v_post, ln_z


def _build_conjugate_model_and_dataset() -> tuple[tt.RadiocarbonModel, tt.Dataset, np.ndarray]:
    """Build the 2-box NH/SH conjugate ``(fluence, slope)`` model and 16-ring dataset."""
    cbm = tt.EffectiveCBM(
        K=2, log_k_strat_to_trop=math.log(0.18), log_k_trop_to_strat=math.log(0.18)
    )
    years = np.arange(771.0, 779.0, 1.0)
    s0 = tt.TreeRingSeries(
        years,
        np.zeros_like(years),
        np.full(len(years), 1.15),
        hemisphere="north",
        seasonal_weights="april-september",
        name="c_nh",
    )
    s1 = tt.TreeRingSeries(
        years,
        np.zeros_like(years),
        np.full(len(years), 1.35),
        hemisphere="south",
        year_convention="schulman_start",
        seasonal_weights="october-march",
        name="c_sh",
    )
    ds_tmpl = tt.Dataset([s0, s1])
    prod = tt.ProductionSum(
        steady=tt.SteadyProduction(),
        spike=tt.MiyakeSpike(t_start=774.4, duration=0.4, fluence=7.2, profile="boxcar"),
        solar=tt.HarmonicSolar(periods=(11.0,), c_cos=(0.0,), c_sin=(0.0,), t_ref=775.0),
        trend=tt.LinearTrend(slope=0.0, t_ref=775.0, switched_on=False),
        gp=tt.GPProduction(tgq.Matern32(scale=3.2, sigma=0.65), n_modes=18, domain=(763.0, 790.0)),
    )
    model = tt.RadiocarbonModel(cbm, prod)
    rng = np.random.default_rng(98765)
    m_true = model.set("production.spike.fluence", 8.05).set("production.trend.slope", 0.0048)
    y_clean = np.asarray(m_true.predict_d14c(ds_tmpl, offsets=jnp.array([5.4, 4.7]), concat=True))
    t_mids = np.concatenate([years + 0.5, years + 0.75])
    wiggle = 3.8 * np.cos(2.0 * math.pi * (t_mids - 774.5) / 5.5)
    y_obs = y_clean + wiggle + rng.normal(0.0, 0.9, size=y_clean.shape)
    ds = tt.Dataset(
        [
            tt.TreeRingSeries(
                years,
                y_obs[: len(years)],
                np.full(len(years), 1.15),
                hemisphere="north",
                seasonal_weights="april-september",
                name="c_nh",
            ),
            tt.TreeRingSeries(
                years,
                y_obs[len(years) :],
                np.full(len(years), 1.35),
                hemisphere="south",
                year_convention="schulman_start",
                seasonal_weights="october-march",
                name="c_sh",
            ),
        ]
    )
    return model, ds, y_obs


@pytest.fixture(scope="module")
def conjugate_is_oracle() -> ConjugateGaussianOracle:
    """Construct a 2D conjugate linear-Gaussian ``(fluence, slope)`` surrogate/target pair."""
    model, ds, y_obs = _build_conjugate_model_and_dataset()
    ps = tt.ParameterSpace(
        model,
        [
            tt.ParamSpec("production.spike.fluence", "identity", prior_mean=7.2, prior_sd=0.65),
            tt.ParamSpec("production.trend.slope", "identity", prior_mean=0.002, prior_sd=0.003),
        ],
    )
    cfg_surr = tt.LinearMarginalConfig(
        offset_mode="hierarchical",
        global_prior_mean=5.0,
        global_prior_sd=3.5,
        lab_prior_sd=1.4,
        marginalize_solar=False,
        marginalize_trend=False,
        marginalize_gp=True,
        n_gp_modes=3,
        include_gp_initial_cov=False,
    )
    cfg_targ = tt.LinearMarginalConfig(
        offset_mode="hierarchical",
        global_prior_mean=5.0,
        global_prior_sd=3.5,
        lab_prior_sd=1.4,
        marginalize_solar=False,
        marginalize_trend=False,
        marginalize_gp=True,
        n_gp_modes=18,
        include_gp_initial_cov=False,
    )
    jit_surr, jit_targ = 1.65, 0.0

    m_zero = model.set("production.spike.fluence", 0.0).set("production.trend.slope", 0.0)
    mod_surr_gp = m_zero.set(
        "production.gp",
        tt.GPProduction(tgq.Matern32(scale=3.2, sigma=0.65), n_modes=3, domain=(763.0, 790.0)),
    )
    ref_surr = _build_dense_mvn_scipy(mod_surr_gp, ds, cfg_surr, sigma_jitter=jit_surr)
    ref_targ = _build_dense_mvn_scipy(m_zero, ds, cfg_targ, sigma_jitter=jit_targ)
    ref_e1 = _build_dense_mvn_scipy(m_zero.set("production.spike.fluence", 1.0), ds, cfg_targ, jit_targ)
    ref_e2 = _build_dense_mvn_scipy(m_zero.set("production.trend.slope", 1.0), ds, cfg_targ, jit_targ)

    m_base = ref_targ.m_dense
    jac_u = np.column_stack([ref_e1.m_dense - m_base, ref_e2.m_dense - m_base])
    sys_lg = LinearGaussianSystem(
        y_obs=y_obs,
        m_base=m_base,
        jac_u=jac_u,
        prior_mean=np.array([7.2, 0.002], dtype=np.float64),
        prior_cov=np.diag([0.65**2, 0.003**2]),
    )
    mu_0, v_0, lnz_0 = _conjugate_posterior_from_cov(sys_lg, ref_surr.c_dense)
    mu_1, v_1, lnz_1 = _conjugate_posterior_from_cov(sys_lg, ref_targ.c_dense)
    return ConjugateGaussianOracle(
        model=model,
        dataset=ds,
        param_space=ps,
        cfg_surr=cfg_surr,
        cfg_targ=cfg_targ,
        jit_surr=jit_surr,
        jit_targ=jit_targ,
        m_base=m_base,
        jac_u=jac_u,
        cov_surr=ref_surr.c_dense,
        cov_targ=ref_targ.c_dense,
        mu_surr=mu_0,
        v_surr=v_0,
        ln_z_surr=lnz_0,
        mu_targ=mu_1,
        v_targ=v_1,
        ln_z_targ=lnz_1,
        delta_ln_z=lnz_1 - lnz_0,
    )


def _exact_mvnorm_loglikes_batch(
    orc: ConjugateGaussianOracle, cov_mat: np.ndarray, u_draws: np.ndarray
) -> np.ndarray:
    """Evaluate exact MVN log-likelihood ``log N(y; m_base + J u_s, cov_mat)`` across draws."""
    y_obs = np.asarray(orc.dataset.d14c_flat, dtype=np.float64)
    l_c = np.linalg.cholesky(cov_mat)
    resids = y_obs[:, None] - (orc.m_base[:, None] + orc.jac_u @ u_draws.T)
    whitened = sla.solve_triangular(l_c, resids, lower=True)
    log_norm = float(np.sum(np.log(np.diag(l_c))) + 0.5 * len(y_obs) * math.log(2.0 * math.pi))
    return -0.5 * np.sum(whitened**2, axis=0) - log_norm


class TestConjugateLinearGaussianImportanceWeights:
    """Section 3 line 98: NUTS & SMC importance weights vs closed-form conjugate Gaussian."""

    def test_smc_single_bridge_and_tempered_reweighting_oracle(
        self, conjugate_is_oracle: ConjugateGaussianOracle
    ) -> None:
        orc = conjugate_is_oracle
        assert float(np.min(np.linalg.eigvalsh(2.0 * sla.inv(orc.v_targ) - sla.inv(orc.v_surr)))) > 0.0

        smc_1 = tt.sample_smc(
            orc.model,
            orc.dataset,
            orc.param_space,
            orc.cfg_surr,
            num_particles=512,
            target_ess=0.0,
            num_mcmc_steps=0,
            sigma_jitter=orc.jit_surr,
            rng_key=jax.random.PRNGKey(202),
        )
        u_p1 = np.asarray(smc_1.particles, dtype=np.float64)
        ll_surr_p1 = _exact_mvnorm_loglikes_batch(orc, orc.cov_surr, u_p1)
        w_p1 = np.asarray(smc_1.weights, dtype=np.float64)
        assert np.all(w_p1 > 0.0)
        log_w_p1 = np.log(w_p1)
        centered_diff = (log_w_p1 - np.mean(log_w_p1)) - (ll_surr_p1 - np.mean(ll_surr_p1))
        np.testing.assert_allclose(centered_diff, 0.0, atol=1e-12)
        lnz_1step_ref = float(sps.logsumexp(ll_surr_p1) - math.log(len(u_p1)))
        np.testing.assert_allclose(float(smc_1.log_evidence), lnz_1step_ref, rtol=1e-12, atol=1e-12)

        smc_temp = tt.sample_smc(
            orc.model,
            orc.dataset,
            orc.param_space,
            orc.cfg_surr,
            num_particles=320,
            target_ess=0.75,
            num_mcmc_steps=4,
            sigma_jitter=orc.jit_surr,
            rng_key=jax.random.PRNGKey(303),
        )
        u_pt = np.asarray(smc_temp.particles, dtype=np.float64)
        fast_surr = tts._make_fast_unconstrained_loglike(
            orc.model, orc.dataset, orc.param_space, orc.cfg_surr, orc.jit_surr
        )
        fast_targ = tts._make_fast_unconstrained_loglike(
            orc.model, orc.dataset, orc.param_space, orc.cfg_targ, orc.jit_targ
        )
        ll_surr_tt = np.asarray(tts._vmap_fast_lk_jit(fast_surr, smc_temp.particles), dtype=np.float64)
        ll_targ_tt = np.asarray(tts._vmap_fast_lk_jit(fast_targ, smc_temp.particles), dtype=np.float64)
        lw_ratio_tt = ll_targ_tt - ll_surr_tt

        lw_ratio_ref = _exact_mvnorm_loglikes_batch(
            orc, orc.cov_targ, u_pt
        ) - _exact_mvnorm_loglikes_batch(orc, orc.cov_surr, u_pt)
        np.testing.assert_allclose(lw_ratio_tt, lw_ratio_ref, rtol=1e-12, atol=1e-12)

        log_comb = np.log(np.asarray(smc_temp.weights, dtype=np.float64)) + lw_ratio_tt
        dlnz_hat = float(sps.logsumexp(log_comb))
        w_targ = np.exp(log_comb - dlnz_hat)
        ess_targ = 1.0 / float(np.sum(w_targ**2))
        assert ess_targ > 100.0

        lnz_targ_hat = float(smc_temp.log_evidence) + dlnz_hat
        se_lnz = math.hypot(float(smc_temp.log_evidence_err), math.sqrt(1.0 / ess_targ))
        assert abs(dlnz_hat - orc.delta_ln_z) < 4.0 * math.sqrt(2.0 / ess_targ)
        assert abs(lnz_targ_hat - orc.ln_z_targ) < 4.0 * se_lnz

        mu_hat = np.sum(w_targ[:, None] * u_pt, axis=0)
        diffs = u_pt - mu_hat[None, :]
        v_hat = np.einsum("ni,nj,n->ij", diffs, diffs, w_targ) / (1.0 - float(np.sum(w_targ**2)))
        assert np.all(np.abs(mu_hat - orc.mu_targ) < 3.8 * np.sqrt(np.diag(orc.v_targ) / ess_targ))
        np.testing.assert_allclose(v_hat, orc.v_targ, rtol=0.25, atol=3e-4)

    def test_nuts_surrogate_to_target_importance_reweighting_oracle(
        self, conjugate_is_oracle: ConjugateGaussianOracle
    ) -> None:
        orc = conjugate_is_oracle

        idata = tt.sample_nuts(
            orc.model,
            orc.dataset,
            orc.param_space,
            orc.cfg_surr,
            num_warmup=100,
            num_samples=350,
            num_chains=2,
            reconstruct_linear=False,
            sigma_jitter=orc.jit_surr,
            rng_key=jax.random.PRNGKey(404),
        )
        u_draws = np.stack(
            [
                np.asarray(idata.posterior["production__spike__fluence"]).reshape(-1),
                np.asarray(idata.posterior["production__trend__slope"]).reshape(-1),
            ],
            axis=-1,
        )
        ll_surr_stored = np.asarray(idata.log_likelihood["obs"]).reshape(-1)
        lp_stored = np.asarray(idata.sample_stats["lp"]).reshape(-1)

        ll_surr_ref = _exact_mvnorm_loglikes_batch(orc, orc.cov_surr, u_draws)
        ll_targ_ref = _exact_mvnorm_loglikes_batch(orc, orc.cov_targ, u_draws)
        np.testing.assert_allclose(ll_surr_stored, ll_surr_ref, rtol=1e-12, atol=1e-12)

        z_draws = (u_draws - np.array([7.2, 0.002])) / np.array([0.65, 0.003])
        lp_ref = -0.5 * np.sum(z_draws**2, axis=1) - math.log(2.0 * math.pi) + ll_surr_ref
        np.testing.assert_allclose(lp_stored, lp_ref, rtol=1e-12, atol=1e-12)

        fast_targ = tts._make_fast_unconstrained_loglike(
            orc.model, orc.dataset, orc.param_space, orc.cfg_targ, orc.jit_targ
        )
        ll_targ_tt = np.asarray(tts._vmap_fast_lk_jit(fast_targ, jnp.asarray(u_draws)), dtype=np.float64)
        lw_is = ll_targ_tt - ll_surr_stored
        lw_ref = ll_targ_ref - ll_surr_ref
        np.testing.assert_allclose(lw_is, lw_ref, rtol=1e-12, atol=1e-12)

        mu_unweighted = np.mean(u_draws, axis=0)
        se_surr = np.sqrt(np.diag(orc.v_surr) / len(u_draws))
        assert np.max(np.abs(orc.mu_targ - orc.mu_surr) / se_surr) > 4.0
        assert np.all(np.abs(mu_unweighted - orc.mu_surr) < 4.0 * se_surr)

        lse_w = float(sps.logsumexp(lw_is))
        dlnz_is = lse_w - math.log(len(lw_is))
        w_norm = np.exp(lw_is - lse_w)
        ess_is = 1.0 / float(np.sum(w_norm**2))
        assert ess_is > 150.0

        var_w_rel = float(len(lw_is) * np.sum(w_norm**2) - 1.0)
        se_dlnz = math.sqrt(max(var_w_rel, 1e-6) / len(lw_is))
        assert abs(dlnz_is - orc.delta_ln_z) < 4.0 * max(se_dlnz, 0.02)

        mu_rw = np.sum(w_norm[:, None] * u_draws, axis=0)
        diffs_rw = u_draws - mu_rw[None, :]
        v_rw = np.einsum("ni,nj,n->ij", diffs_rw, diffs_rw, w_norm) / (1.0 - float(np.sum(w_norm**2)))
        assert np.all(np.abs(mu_rw - orc.mu_targ) < 3.8 * np.sqrt(np.diag(orc.v_targ) / ess_is))
        np.testing.assert_allclose(v_rw, orc.v_targ, rtol=0.22, atol=2.5e-4)

    @pytest.mark.skipif(
        not hasattr(tts, "reweight_exact_marginal"),
        reason="reweight_exact_marginal lands with PR #57",
    )
    def test_reweight_exact_marginal_helper_when_available(
        self, conjugate_is_oracle: ConjugateGaussianOracle
    ) -> None:
        """Directly verify ``reweight_exact_marginal`` once PR #57 merges onto ``main``."""
        orc = conjugate_is_oracle
        idata = tt.sample_nuts(
            orc.model,
            orc.dataset,
            orc.param_space,
            orc.cfg_surr,
            num_warmup=100,
            num_samples=160,
            num_chains=1,
            reconstruct_linear=False,
            rng_key=jax.random.PRNGKey(505),
        )
        idata_rw, diag = tts.reweight_exact_marginal(
            idata, orc.model, orc.dataset, orc.param_space, orc.cfg_targ, orc.cfg_surr
        )
        u_draws = np.stack(
            [
                np.asarray(idata_rw.posterior["production__spike__fluence"]).reshape(-1),
                np.asarray(idata_rw.posterior["production__trend__slope"]).reshape(-1),
            ],
            axis=-1,
        )
        ref_surr_nojit = _build_dense_mvn_scipy(
            orc.model.set("production.spike.fluence", 0.0)
            .set("production.trend.slope", 0.0)
            .set(
                "production.gp",
                tt.GPProduction(tgq.Matern32(scale=3.2, sigma=0.65), n_modes=3, domain=(763.0, 790.0)),
            ),
            orc.dataset,
            orc.cfg_surr,
            sigma_jitter=0.0,
        )
        lw_ref = _exact_mvnorm_loglikes_batch(
            orc, orc.cov_targ, u_draws
        ) - _exact_mvnorm_loglikes_batch(orc, ref_surr_nojit.c_dense, u_draws)
        lw_got = np.asarray(idata_rw.sample_stats["log_weight_is"]).reshape(-1)
        np.testing.assert_allclose(lw_got, lw_ref, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            float(diag["log_z_correction"]),
            float(sps.logsumexp(lw_ref) - math.log(len(lw_ref))),
            rtol=1e-12,
            atol=1e-12,
        )
