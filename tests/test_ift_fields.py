"""Comprehensive scientific and numerical validation suite for ``ticktack.fields``."""

from __future__ import annotations

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jax.test_util import check_grads
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import optimistix as optx
import pytest
from tinygp import kernels as tk

from ticktack.cbm import CarbonBoxModel
from ticktack.constants import DEFAULT_Q_STD
from ticktack.data import Dataset, TreeRingSeries, load_bundled_event, load_bundled_series
from ticktack.fields import (
    _DCT_OPERATOR_CACHE,
    CorrelatedFieldSpectrum,
    GaussianField,
    LogGaussianField,
    MaternSpectrum,
    QuasiPeriodicSpectrum,
    SquaredExpSpectrum,
    SumSpectrum,
    apply_error_scale,
    build_dct_response_matrix,
    dct2,
    dct_knot_variance,
    dct_laplacian_eigenvalues,
    error_scale,
    evidence_grid,
    field_spectrum,
    fit_ift_field,
    idct2,
    laplace_samples,
    log_evidence,
    optimize_gp_kernel,
    predict_field_d14c,
    select_gp_kernel,
    stationary_cbm_spectral_cov,
)
from ticktack.plotting import plot_evidence_grid, plot_field_draws
from ticktack.fitting import fit_map
from ticktack.likelihood import (
    LinearMarginalConfig,
    ParamSpec,
    ParameterSpace,
    _reconstruct_initial_state_posterior,
    linear_posterior,
    marginal_log_likelihood,
)
from ticktack.linear import (
    LinearMarginal,
    hierarchical_offset_design,
    project_causal_gp,
    stationary_cbm_gp_cov,
)
from ticktack.model import RadiocarbonModel
from ticktack.presets import load_preset
from ticktack.production import GPProduction, GriddedProduction, MiyakeSpike, SteadyProduction
from ticktack.sampling import sample_nuts
from ticktack.solvers import ExactLinearSolver


# =============================================================================
# 1. DCT-II Spectra, Gradients, & Exact 1D GMRF TSV+L2 Equivalence
# =============================================================================


def test_dct2_orthonormality_and_laplacian_diagonalization() -> None:
    n = 48
    dt = 0.5
    key = jax.random.PRNGKey(11)
    x = jax.random.normal(key, (n,), dtype=jnp.float64)

    # Orthonormal inverse identity
    x_rec = idct2(dct2(x))
    np.testing.assert_allclose(x_rec, x, rtol=1e-13, atol=1e-13)

    # Parseval energy conservation
    np.testing.assert_allclose(jnp.sum(x**2), jnp.sum(dct2(x) ** 2), rtol=1e-14, atol=1e-14)

    # Exact diagonalization of 1D Neumann discrete Laplacian L = dt^{-2} D^T D
    d_mat = jnp.zeros((n - 1, n), dtype=jnp.float64)
    idx = jnp.arange(n - 1)
    d_mat = d_mat.at[idx, idx].set(-1.0).at[idx, idx + 1].set(1.0)
    lap_mat = (d_mat.T @ d_mat) / (dt**2)

    c_t = idct2(jnp.eye(n, dtype=jnp.float64), axis=0)
    diag_check = c_t.T @ lap_mat @ c_t
    lam = dct_laplacian_eigenvalues(n, dt)
    np.testing.assert_allclose(diag_check, jnp.diag(lam), rtol=1e-12, atol=1e-12)


def test_field_spectrum_normalization_zero_mode_and_knot_variance() -> None:
    n = 64
    dt = 0.75
    sigma = 0.42
    length_yr = 6.0

    for zm in (False, True):
        for order in (1.0, 1.7, 2.0, 3.0, float("inf")):
            spec = field_spectrum(n, dt, sigma, length_yr, order=order, zero_mode=zm)
            assert spec.dtype == jnp.float64
            assert spec.shape == (n,)
            np.testing.assert_allclose(jnp.mean(spec), sigma**2, rtol=1e-14, atol=1e-14)
            if not zm:
                assert float(spec[0]) == 0.0
            else:
                assert float(spec[0]) > 0.0

    # Verify O(N log N) dct_knot_variance matches diag(C^T diag(S) C) for both even and odd N
    for n_test in (32, 33):
        gf = GaussianField(
            jnp.zeros((n_test,), dtype=jnp.float64),
            760.0,
            790.0,
            sigma=0.35,
            length_yr=4.5,
            order=2.2,
            zero_mode=False,
        )
        np.testing.assert_allclose(
            gf.knot_variance(),
            jnp.diag(gf.prior_covariance()),
            rtol=1e-13,
            atol=1e-14,
        )


def test_field_spectrum_finite_gradients_at_zero_mode_and_continuous_order() -> None:
    z = jax.random.normal(jax.random.PRNGKey(3), (40,), dtype=jnp.float64)

    def _loss(sigma: jax.Array, ell: jax.Array, order: jax.Array) -> jax.Array:
        gf = GaussianField(z, 750.0, 790.0, sigma=sigma, length_yr=ell, order=order, zero_mode=False)
        eta = gf.evaluate_grid()
        return jnp.sum(eta**2) + jnp.sum(gf.power_spectrum())

    g_s, g_l, g_o = jax.grad(_loss, argnums=(0, 1, 2))(
        jnp.asarray(0.3, dtype=jnp.float64),
        jnp.asarray(5.0, dtype=jnp.float64),
        jnp.asarray(1.85, dtype=jnp.float64),
    )
    assert jnp.all(jnp.isfinite(jnp.array([g_s, g_l, g_o])))

    # Also check sigma = 0 boundary has finite gradient (double-where guard)
    g_zero_sig = jax.grad(lambda s: _loss(s, jnp.asarray(5.0), jnp.asarray(2.0)))(
        jnp.asarray(0.0, dtype=jnp.float64)
    )
    assert jnp.isfinite(g_zero_sig)

    check_grads(
        lambda u: _loss(jnp.exp(u[0]), jnp.exp(u[1]), 0.5 + jnp.exp(u[2])),
        (jnp.array([math.log(0.3), math.log(5.0), math.log(1.4)], dtype=jnp.float64),),
        order=2,
        modes=["fwd", "rev"],
        rtol=1e-5,
        atol=1e-6,
    )


def test_exact_gmrf_tsv_plus_l2_equivalence_at_order_one() -> None:
    n = 50
    t_start, t_end = 700.0, 749.0
    dt = (t_end - t_start) / (n - 1)
    sigma = 0.4
    ell = 4.0

    # With zero_mode=True, S_j = c * (1 + ell^2 * lambda_j)^{-1} for all j = 0 .. N - 1,
    # so ||z||_2^2 == (1 / c) * (||eta||_2^2 + (ell / dt)^2 * sum (eta_{g+1} - eta_g)^2).
    lam = dct_laplacian_eigenvalues(n, dt)
    raw = 1.0 / (1.0 + (ell**2) * lam)
    c_scale = (sigma**2) * n / jnp.sum(raw)

    z = jax.random.normal(jax.random.PRNGKey(9), (n,), dtype=jnp.float64)
    gf = GaussianField(z, t_start, t_end, sigma=sigma, length_yr=ell, order=1.0, zero_mode=True)
    eta = gf.evaluate_grid()

    tsv = jnp.sum(jnp.diff(eta) ** 2) / (dt**2)
    gmrf_quad = (jnp.sum(eta**2) + (ell**2) * tsv) / c_scale
    np.testing.assert_allclose(jnp.sum(z**2), gmrf_quad, rtol=1e-13, atol=1e-13)


def test_composable_spectral_kernels() -> None:
    n, dt = 80, 0.5
    m_k = MaternSpectrum(sigma=0.3, length_yr=6.0, order=2.0)
    q_k = QuasiPeriodicSpectrum(sigma=0.2, period_yr=11.0, quality=5.0)
    se_k = SquaredExpSpectrum(sigma=0.15, length_yr=8.0)
    cf_k = CorrelatedFieldSpectrum(sigma=0.25, slope=-2.2, n_modes=5)

    comp = m_k + q_k + se_k
    assert isinstance(comp, SumSpectrum)
    assert len(comp.parts) == 3
    s_comp = comp.spectrum(n, dt)
    expected_var = 0.3**2 + 0.2**2 + 0.15**2
    np.testing.assert_allclose(jnp.mean(s_comp), expected_var, rtol=1e-13, atol=1e-13)

    s_cf = cf_k.spectrum(n, dt)
    np.testing.assert_allclose(jnp.mean(s_cf), 0.25**2, rtol=1e-13, atol=1e-13)
    assert jnp.isfinite(cf_k.hyperprior_log_prob())


# =============================================================================
# 2. Exact GaussianField Marginal Log-Evidence, DCT Operator, & check_grads
# =============================================================================


def _make_two_hemisphere_dataset(cbm: CarbonBoxModel, seed: int = 7) -> Dataset:
    years_nh = np.arange(765.0, 790.0, 1.0)
    years_sh = np.arange(766.0, 789.0, 1.0)
    rng = np.random.default_rng(seed)

    t_grid = jnp.linspace(755.0, 792.0, 64, dtype=jnp.float64)
    dq_true = 0.6 * jnp.exp(-0.5 * ((t_grid - 774.6) / 1.5) ** 2)
    solver = ExactLinearSolver()
    prod_true = GriddedProduction(t_grid, dq_true, is_delta=True)

    s_nh_0 = TreeRingSeries(
        years_nh,
        np.zeros_like(years_nh),
        np.full_like(years_nh, 1.2),
        hemisphere="north",
        name="NH_Oak",
    )
    s_sh_0 = TreeRingSeries(
        years_sh,
        np.zeros_like(years_sh),
        np.full_like(years_sh, 1.4),
        hemisphere="south",
        year_convention="schulman_start",
        name="SH_Pine",
    )
    nh_clean = solver.predict_d14c(
        cbm,
        prod_true,
        s_nh_0.sub_t_starts,
        s_nh_0.sub_t_ends,
        sub_weights=s_nh_0.seasonal_weights.weights,
        box_indices=cbm.box_index("Troposphere", hemisphere="north"),
    )
    sh_clean = solver.predict_d14c(
        cbm,
        prod_true,
        s_sh_0.sub_t_starts,
        s_sh_0.sub_t_ends,
        sub_weights=s_sh_0.seasonal_weights.weights,
        box_indices=cbm.box_index("Troposphere", hemisphere="south"),
    )
    s_nh = TreeRingSeries(
        years_nh,
        np.asarray(nh_clean) + 2.5 + rng.normal(0.0, 1.2, size=years_nh.size),
        np.full_like(years_nh, 1.2),
        hemisphere="north",
        name="NH_Oak",
    )
    s_sh = TreeRingSeries(
        years_sh,
        np.asarray(sh_clean) + 1.8 + rng.normal(0.0, 1.4, size=years_sh.size),
        np.full_like(years_sh, 1.4),
        hemisphere="south",
        year_convention="schulman_start",
        name="SH_Pine",
    )
    return Dataset((s_nh, s_sh), name="synth_2hem")


def test_dct_response_matrix_matches_exact_linear_solver() -> None:
    cbm = load_preset("Brehm21")
    ds = _make_two_hemisphere_dataset(cbm)
    t_start, t_end, n_grid = 755.0, 792.0, 60

    op = build_dct_response_matrix(
        cbm, ds, t_start, t_end, n_grid, return_operator=True
    )
    z = jax.random.normal(jax.random.PRNGKey(21), (n_grid,), dtype=jnp.float64)
    gf = GaussianField(z, t_start, t_end, sigma=0.45, length_yr=4.0, order=2.0)

    pred_dct = op.ss_d14c + op.w_dct @ ( jnp.sqrt(gf.power_spectrum()) * z )

    solver = ExactLinearSolver()
    gridded = gf.to_gridded()
    parts = []
    for s in ds.series:
        sub_s, sub_e, w_sub = s.quadrature_windows()
        box_i = cbm.box_index(s.box_name, hemisphere=s.hemisphere)
        parts.append(
            solver.predict_d14c(
                cbm, gridded, sub_s, sub_e, sub_weights=w_sub, box_indices=box_i
            )
        )
    pred_exact = jnp.concatenate(parts, axis=0)
    np.testing.assert_allclose(pred_dct, pred_exact, rtol=1e-11, atol=2e-11)


def test_gaussian_field_log_evidence_matches_dense_marginal_and_check_grads() -> None:
    cbm = load_preset("Brehm21")
    ds = _make_two_hemisphere_dataset(cbm)
    t_start, t_end, n_grid = 755.0, 792.0, 56
    op = build_dct_response_matrix(cbm, ds, t_start, t_end, n_grid, return_operator=True)

    gf = GaussianField(
        jnp.zeros((n_grid,), dtype=jnp.float64),
        t_start,
        t_end,
        sigma=0.38,
        length_yr=5.5,
        order=2.3,
        zero_mode=False,
    )
    model = RadiocarbonModel(cbm, gf)

    # 1. offset_mode="none", zero initial state: compare against dense N(d; ss_d14c, W_grid Sigma_eta W_grid^T + D)
    lnz_none = log_evidence(model, ds, offset_mode="none", include_field_initial_cov=False)
    cov_eta = gf.prior_covariance()
    cov_d_none = op.w_grid @ cov_eta @ op.w_grid.T + jnp.diag(ds.sigma_flat**2)
    r_none = ds.d14c_flat - op.ss_d14c
    sign_n, logdet_n = jnp.linalg.slogdet(cov_d_none)
    assert float(sign_n) > 0.0
    dense_lnz_none = -0.5 * (
        r_none @ jnp.linalg.solve(cov_d_none, r_none)
        + logdet_n
        + ds.n_obs * math.log(2.0 * math.pi)
    )
    np.testing.assert_allclose(lnz_none, dense_lnz_none, rtol=1e-11, atol=1e-10)

    # 2. offset_mode="hierarchical", zero initial state: compare against dense N(d; ss + X_off mu_off, ...)
    lnz_hier = log_evidence(
        model,
        ds,
        offset_mode="hierarchical",
        global_offset_prior_sd=200.0,
        lab_offset_prior_sd=1.5,
        include_field_initial_cov=False,
    )
    fit_res = fit_ift_field(
        model,
        ds,
        offset_mode="hierarchical",
        global_offset_prior_sd=200.0,
        lab_offset_prior_sd=1.5,
        include_field_initial_cov=False,
    )
    np.testing.assert_allclose(fit_res.log_evidence, lnz_hier, rtol=1e-12, atol=1e-11)
    assert fit_res.initial_state_mean is None
    assert fit_res.initial_state_sd is None

    x_off, mu_off, sd_off = hierarchical_offset_design(
        ds.series_indices,
        ds.n_series,
        box_activity_ratios=op.box_activity_ratios,
        global_prior_sd=200.0,
        lab_prior_sd=1.5,
    )
    cov_d_hier = (
        cov_d_none + (x_off * (sd_off**2)[None, :]) @ x_off.T
    )
    r_hier = ds.d14c_flat - op.ss_d14c - x_off @ mu_off
    sign_h, logdet_h = jnp.linalg.slogdet(cov_d_hier)
    assert float(sign_h) > 0.0
    dense_lnz_hier = -0.5 * (
        r_hier @ jnp.linalg.solve(cov_d_hier, r_hier)
        + logdet_h
        + ds.n_obs * math.log(2.0 * math.pi)
    )
    np.testing.assert_allclose(lnz_hier, dense_lnz_hier, rtol=1e-11, atol=1e-10)

    # 3. 2nd-order check_grads wrt (ln sigma, ln ell, order)
    def _lnz_hyper(u: jax.Array) -> jax.Array:
        gf_u = GaussianField(
            jnp.zeros((n_grid,), dtype=jnp.float64),
            t_start,
            t_end,
            sigma=jnp.exp(u[0]),
            length_yr=jnp.exp(u[1]),
            order=u[2],
            zero_mode=False,
        )
        return log_evidence(RadiocarbonModel(cbm, gf_u), ds, operator=op)

    u_test = jnp.array([math.log(0.35), math.log(5.0), 2.1], dtype=jnp.float64)
    check_grads(_lnz_hyper, (u_test,), order=2, modes=["fwd", "rev"], rtol=1e-5, atol=1e-6)


# =============================================================================
# 3. LogGaussianField Positivity, Fast LM Convergence, & NumPyro NUTS Validation
# =============================================================================


def test_log_gaussian_field_positivity_ito_mean_and_nuts_agreement() -> None:
    cbm = load_preset("Guttler15")
    t_start, t_end, n_grid = 760.0, 790.0, 24

    # Strict positivity even under extreme negative excitations (ito_corrected=True by default)
    extreme = LogGaussianField(
        jnp.full((n_grid,), -12.0, dtype=jnp.float64),
        t_start,
        t_end,
        sigma=0.8,
        length_yr=4.0,
    )
    assert extreme.ito_corrected is True
    q_ext = extreme.evaluate_q_grid()
    assert jnp.all(q_ext > 0.0)
    assert jnp.all(jnp.isfinite(q_ext))

    # Default Itô variance correction preserves E[Q(t)] == q_mean across interior & boundary knots
    lgf_ito = LogGaussianField(
        jnp.zeros((n_grid,), dtype=jnp.float64),
        t_start,
        t_end,
        sigma=0.45,
        length_yr=4.0,
        q_mean=DEFAULT_Q_STD,
    )
    assert lgf_ito.ito_corrected is True
    draws_prior = lgf_ito.sample_prior(jax.random.PRNGKey(42), n_samples=6000)
    np.testing.assert_allclose(
        jnp.mean(draws_prior, axis=0),
        jnp.full((n_grid,), DEFAULT_Q_STD),
        rtol=0.03,
        atol=0.03,
    )

    # Explicit ito_corrected=False is median-anchored at z=0 with E[Q] = q_mean * exp(0.5 * var)
    lgf_uncorr = LogGaussianField(
        jnp.zeros((n_grid,), dtype=jnp.float64),
        t_start,
        t_end,
        sigma=0.45,
        length_yr=4.0,
        q_mean=DEFAULT_Q_STD,
        ito_corrected=False,
    )
    assert lgf_uncorr.ito_corrected is False
    np.testing.assert_allclose(
        lgf_uncorr.evaluate_q_grid(),
        jnp.full((n_grid,), DEFAULT_Q_STD),
        rtol=1e-14,
        atol=1e-14,
    )
    knot_var = dct_knot_variance(lgf_uncorr.power_spectrum())
    np.testing.assert_allclose(
        lgf_ito.evaluate_q_grid(),
        DEFAULT_Q_STD * jnp.exp(-0.5 * knot_var),
        rtol=1e-14,
        atol=1e-14,
    )
    draws_uncorr = lgf_uncorr.sample_prior(jax.random.PRNGKey(42), n_samples=6000)
    np.testing.assert_allclose(
        jnp.mean(draws_uncorr, axis=0),
        DEFAULT_Q_STD * jnp.exp(0.5 * knot_var),
        rtol=0.03,
        atol=0.03,
    )

    # Generate nonlinear synthetic tree-ring series from a known LogGaussianField
    z_true = jax.random.normal(jax.random.PRNGKey(101), (n_grid,), dtype=jnp.float64)
    lgf_true = LogGaussianField(
        z_true,
        t_start,
        t_end,
        sigma=0.35,
        length_yr=5.0,
        order=2.0,
        q_mean=DEFAULT_Q_STD,
        zero_mode=True,
    )
    years = np.arange(765.0, 789.0, 1.0)
    sig_obs = 0.8
    s_proto = TreeRingSeries(
        years, np.zeros_like(years), np.full_like(years, sig_obs), hemisphere="north", name="Synth"
    )
    op = build_dct_response_matrix(cbm, s_proto, t_start, t_end, n_grid, return_operator=True)
    d_clean = op.ss_d14c + op.w_grid @ lgf_true.evaluate_grid(q_std=DEFAULT_Q_STD)
    rng = np.random.default_rng(102)
    d_noisy = np.asarray(d_clean) + rng.normal(0.0, sig_obs, size=years.size)
    series = TreeRingSeries(
        years, d_noisy, np.full_like(years, sig_obs), hemisphere="north", name="Synth"
    )

    lgf_init = lgf_true.with_latent(jnp.zeros((n_grid,), dtype=jnp.float64))
    mdl = RadiocarbonModel(cbm, lgf_init)
    fit_res = fit_ift_field(
        mdl, series, offset_mode="none", operator=op, include_field_initial_cov=False
    )

    assert fit_res.converged
    assert fit_res.n_steps < 40
    assert jnp.all(fit_res.q_grid > 0.0)

    lap_draws = laplace_samples(
        fit_res, series, n_samples=1500, rng_key=jax.random.PRNGKey(5), offset_mode="none"
    )
    lap_q_mean = np.asarray(jnp.mean(lap_draws["q"], axis=0))
    lap_q_sd = np.asarray(jnp.std(lap_draws["q"], axis=0))

    # Compare against exact NumPyro NUTS posterior over whitened latents z
    def _numpyro_model() -> None:
        z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([n_grid]).to_event(1))
        q_g = lgf_init.with_latent(z).evaluate_q_grid()
        mu_d = op.ss_d14c + op.w_grid @ (q_g - DEFAULT_Q_STD)
        numpyro.sample("obs", dist.Normal(mu_d, sig_obs), obs=series.d14c)

    mcmc = MCMC(NUTS(_numpyro_model), num_warmup=300, num_samples=500, num_chains=1, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(88))
    z_nuts = mcmc.get_samples()["z"]
    q_nuts = np.asarray(jax.vmap(lambda z: lgf_init.with_latent(z).evaluate_q_grid())(z_nuts))
    nuts_q_mean = np.mean(q_nuts, axis=0)
    nuts_q_sd = np.std(q_nuts, axis=0)

    # Interior knots (away from un-observed burn-in padding) match NUTS closely
    interior = slice(5, -2)
    np.testing.assert_allclose(lap_q_mean[interior], nuts_q_mean[interior], atol=0.12, rtol=0.08)
    np.testing.assert_allclose(lap_q_sd[interior], nuts_q_sd[interior], atol=0.10, rtol=0.22)

    # Verify Laplace log_evidence via importance-sampling correction around Laplace proposal:
    # ln Z_exact = ln Z_Laplace + ln E_{q_Laplace}[p(d, z) / q_Laplace(z) / Z_Laplace]
    z_prop = lap_draws["latents"]  # drawn from N(z_hat, (I + J^T J)^{-1})
    # Evaluate exact log p(d, z) vs quadratic Taylor expansion around z_hat
    inv_sig = 1.0 / series.sigma_d14c

    def _r_vec(z_in: jax.Array) -> jax.Array:
        return inv_sig * (
            op.ss_d14c
            + op.w_grid @ lgf_init.with_latent(z_in).evaluate_grid(q_std=DEFAULT_Q_STD)
            - series.d14c
        )

    r_hat = _r_vec(fit_res.latent)
    j_hat = jax.jacfwd(_r_vec)(fit_res.latent)
    log_norm_d = 0.5 * series.n_obs * math.log(2.0 * math.pi) + jnp.sum(jnp.log(series.sigma_d14c))

    def _log_weight_delta(z_s: jax.Array) -> jax.Array:
        r_s = _r_vec(z_s)
        r_lin = r_hat + j_hat @ (z_s - fit_res.latent)
        # Difference between true joint log-density and Gauss-Newton quadratic surrogate
        return -0.5 * ((r_s @ r_s) - (r_lin @ r_lin))

    deltas = jax.vmap(_log_weight_delta)(z_prop)
    is_correction = float(
        jax.scipy.special.logsumexp(deltas) - math.log(float(z_prop.shape[0]))
    )
    assert abs(is_correction) < 0.25
    assert jnp.isfinite(log_norm_d)


# =============================================================================
# 4. MacKay error_scale Recovery (Global & Per-Series)
# =============================================================================


def test_error_scale_global_and_by_series_recovery() -> None:
    cbm = load_preset("Brehm21")
    rng = np.random.default_rng(2026)

    t_start, t_end, n_grid = 700.0, 820.0, 128
    years = np.arange(710.0, 815.0, 1.0)
    true_sigma_d14c = 1.0

    # Generate synthetic single-series dataset with true noise = 1.0 permil,
    # but quote 2.0 permil error bars (2x overstated -> true s == 0.50).
    s_template = TreeRingSeries(
        years, np.zeros_like(years), np.full_like(years, 2.0), hemisphere="north", name="Oak"
    )
    op = build_dct_response_matrix(cbm, s_template, t_start, t_end, n_grid, return_operator=True)
    gf_true = GaussianField(
        jax.random.normal(jax.random.PRNGKey(31), (n_grid,), dtype=jnp.float64),
        t_start,
        t_end,
        sigma=0.35,
        length_yr=6.0,
        order=2.0,
    )
    clean_nh = np.asarray(op.ss_d14c + op.w_grid @ gf_true.evaluate_grid())
    noisy_nh = clean_nh + 3.0 + rng.normal(0.0, true_sigma_d14c, size=years.size)

    overstated_series = TreeRingSeries(
        years, noisy_nh, np.full_like(years, 2.0), hemisphere="north", name="Oak"
    )
    model_0 = RadiocarbonModel(
        cbm,
        GaussianField(
            jnp.zeros((n_grid,), dtype=jnp.float64),
            t_start,
            t_end,
            sigma=0.35,
            length_yr=6.0,
            order=2.0,
        ),
    )
    fit_0 = fit_ift_field(model_0, overstated_series)
    s_est = error_scale(fit_0, overstated_series)
    assert isinstance(s_est, float)
    assert 0.42 <= s_est <= 0.58

    # Rescale error bars by s_est and refit -> error_scale must be 1.000!
    rescaled_series = apply_error_scale(overstated_series, s_est)
    fit_1 = fit_ift_field(model_0, rescaled_series)
    s_refit = error_scale(fit_1, rescaled_series)
    np.testing.assert_allclose(s_refit, 1.0, rtol=1e-6, atol=1e-6)

    # Multi-series test: NH 2x overstated (quoted 2.0, true 1.0 -> s ~ 0.50),
    # SH 0.5x understated (quoted 0.6, true 1.2 -> s ~ 2.00).
    s_sh_template = TreeRingSeries(
        years,
        np.zeros_like(years),
        np.full_like(years, 0.6),
        hemisphere="south",
        year_convention="schulman_start",
        name="Kauri",
    )
    op_sh = build_dct_response_matrix(cbm, s_sh_template, t_start, t_end, n_grid, return_operator=True)
    clean_sh = np.asarray(op_sh.ss_d14c + op_sh.w_grid @ gf_true.evaluate_grid())
    noisy_sh = clean_sh - 1.5 + rng.normal(0.0, 1.2, size=years.size)
    understated_sh = TreeRingSeries(
        years,
        noisy_sh,
        np.full_like(years, 0.6),
        hemisphere="south",
        year_convention="schulman_start",
        name="Kauri",
    )
    ds_multi = Dataset((overstated_series, understated_sh), name="multi_scale")
    fit_multi = fit_ift_field(model_0, ds_multi)
    scales_dict = error_scale(fit_multi, ds_multi, by_series=True)
    assert isinstance(scales_dict, dict)
    assert set(scales_dict.keys()) == {"Oak", "Kauri"}
    assert 0.40 <= scales_dict["Oak"] <= 0.60
    assert 1.65 <= scales_dict["Kauri"] <= 2.35

    ds_multi_rescaled = apply_error_scale(ds_multi, scales_dict)
    fit_multi_refit = fit_ift_field(model_0, ds_multi_rescaled)
    scales_refit = error_scale(fit_multi_refit, ds_multi_rescaled, by_series=True)
    assert isinstance(scales_refit, dict)
    np.testing.assert_allclose(scales_refit["Oak"], 1.0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(scales_refit["Kauri"], 1.0, rtol=1e-5, atol=1e-5)


# =============================================================================
# 5. Empirical-Bayes Kernel Optimization & Truth Recovery (5a, 5b, 5c)
# =============================================================================


def test_select_gp_kernel_solar_plus_continuum_truth_recovery() -> None:
    """5(a): Synthetic 11-yr solar cycle + red continuum selects matern32_plus_solar by dlnZ > 5."""
    cbm = load_preset("Brehm21")
    t_start, t_end, n_grid = 1790.0, 1950.0, 168
    years = np.arange(1805.0, 1945.0, 1.0)
    sig_obs = 0.65

    s_proto = TreeRingSeries(
        years, np.zeros_like(years), np.full_like(years, sig_obs), hemisphere="north", name="SolarSynth"
    )
    op = build_dct_response_matrix(cbm, s_proto, t_start, t_end, n_grid, return_operator=True)

    true_kernel = MaternSpectrum(sigma=0.14, length_yr=9.0, order=2.0) + QuasiPeriodicSpectrum(
        sigma=0.38, period_yr=11.0, quality=7.0
    )
    gf_true = GaussianField(
        jax.random.normal(jax.random.PRNGKey(77), (n_grid,), dtype=jnp.float64),
        t_start,
        t_end,
        kernel=true_kernel,
    )
    d_clean = np.asarray(op.ss_d14c + op.w_grid @ gf_true.evaluate_grid())
    rng = np.random.default_rng(78)
    series = TreeRingSeries(
        years,
        d_clean + rng.normal(0.0, sig_obs, size=years.size),
        np.full_like(years, sig_obs),
        hemisphere="north",
        name="SolarSynth",
    )

    sel = select_gp_kernel(
        cbm,
        series,
        candidates=("matern12", "matern32", "matern52", "squared_exp", "matern32_plus_solar"),
        n_grid=n_grid,
        pad_years=15.0,
        multi_start=2,
        rng_key=jax.random.PRNGKey(9),
    )
    assert sel.best_name == "matern32_plus_solar"
    best_lnz = float(sel.get_candidate("matern32_plus_solar").log_hyper_evidence)
    for pure_k in ("matern12", "matern32", "matern52", "squared_exp"):
        diff = best_lnz - float(sel.get_candidate(pure_k).log_hyper_evidence)
        assert diff > 5.0, f"Expected dlnZ > 5 over {pure_k}, got {diff:.2f}"


def test_select_gp_kernel_rough_ou_vs_smooth_squared_exp() -> None:
    """5(b): Rough OU (order=1) vs smooth SquaredExp data selects matching smoothness kernel."""
    cbm = load_preset("Brehm21")
    t_start, t_end, n_grid = 1790.0, 1950.0, 168
    years = np.arange(1805.0, 1945.0, 1.0)
    sig_obs = 0.35

    s_proto = TreeRingSeries(
        years, np.zeros_like(years), np.full_like(years, sig_obs), hemisphere="north", name="Smoothness"
    )
    op = build_dct_response_matrix(cbm, s_proto, t_start, t_end, n_grid, return_operator=True)
    rng = np.random.default_rng(303)

    # Case 1: Rough OU (order = 1.0)
    gf_rough = GaussianField(
        jax.random.normal(jax.random.PRNGKey(304), (n_grid,), dtype=jnp.float64),
        t_start,
        t_end,
        sigma=0.55,
        length_yr=2.5,
        order=1.0,
    )
    d_rough = np.asarray(op.ss_d14c + op.w_grid @ gf_rough.evaluate_grid())
    s_rough = TreeRingSeries(
        years,
        d_rough + rng.normal(0.0, sig_obs, size=years.size),
        np.full_like(years, sig_obs),
        hemisphere="north",
        name="RoughOU",
    )
    sel_rough = select_gp_kernel(
        cbm,
        s_rough,
        candidates=("matern12", "squared_exp", "continuous_matern"),
        n_grid=n_grid,
        multi_start=2,
        rng_key=jax.random.PRNGKey(1),
    )
    lnz_m12_r = float(sel_rough.get_candidate("matern12").log_hyper_evidence)
    lnz_se_r = float(sel_rough.get_candidate("squared_exp").log_hyper_evidence)
    assert lnz_m12_r > lnz_se_r

    # Case 2: Smooth Squared-Exponential
    gf_smooth = GaussianField(
        jax.random.normal(jax.random.PRNGKey(305), (n_grid,), dtype=jnp.float64),
        t_start,
        t_end,
        kernel=SquaredExpSpectrum(sigma=0.50, length_yr=5.5),
    )
    d_smooth = np.asarray(op.ss_d14c + op.w_grid @ gf_smooth.evaluate_grid())
    s_smooth = TreeRingSeries(
        years,
        d_smooth + rng.normal(0.0, sig_obs, size=years.size),
        np.full_like(years, sig_obs),
        hemisphere="north",
        name="SmoothSE",
    )
    sel_smooth = select_gp_kernel(
        cbm,
        s_smooth,
        candidates=("matern12", "squared_exp", "continuous_matern"),
        n_grid=n_grid,
        multi_start=2,
        rng_key=jax.random.PRNGKey(2),
    )
    lnz_se_s = float(sel_smooth.get_candidate("squared_exp").log_hyper_evidence)
    lnz_m12_s = float(sel_smooth.get_candidate("matern12").log_hyper_evidence)
    assert lnz_se_s > lnz_m12_s
    ord_rough = sel_rough.get_candidate("continuous_matern").hyperparameters["order"]
    ord_smooth = sel_smooth.get_candidate("continuous_matern").hyperparameters["order"]
    assert ord_smooth > ord_rough


def test_real_775ce_bundled_kernel_selection_accuracy() -> None:
    """5(c): Real bundled 775CE dataset with MiyakeSpike + GaussianField recovers physical spike & weights."""
    cbm = load_preset("Brehm21")
    ds_775 = load_bundled_event("775CE")
    spike = MiyakeSpike(t_start=774.55, duration=0.25, fluence=6.4, profile="boxcar")

    sel = select_gp_kernel(
        cbm,
        ds_775,
        extra_production=spike,
        n_grid=96,
        multi_start=2,
        rng_key=jax.random.PRNGKey(42),
    )

    assert len(sel.rankings) == 8
    assert "sho_solar" in sel.names
    assert sel.get_candidate("sho_solar").name == "sho_solar"
    np.testing.assert_allclose(jnp.sum(sel.model_weights), 1.0, rtol=1e-12, atol=1e-12)
    assert jnp.all(jnp.isfinite(sel.log_hyper_evidences))
    best = sel.best_result
    assert best.converged
    assert jnp.all(jnp.linalg.eigvalsh(best.hyper_hessian) > 0.0)
    fitted_fluence = float(best.fit.model.get("production.spike.fluence"))
    assert 4.5 <= fitted_fluence <= 9.5
    s_glob = error_scale(best.fit, ds_775)
    assert 0.5 <= s_glob <= 2.0
    table_str = sel.format_table()
    assert sel.best_name in table_str
    assert "MacKay s_error" in table_str


# =============================================================================
# 6. Evidence Grid, Stateless Plotting, & 5-Cell Notebook Audit
# =============================================================================


def test_evidence_grid_and_stateless_plotting_helpers() -> None:
    cbm = load_preset("Brehm21")
    s_tas = load_bundled_series("775AD-early-S/Büntgen18_TAS01.csv")
    spike = MiyakeSpike(t_start=774.55, duration=0.25, fluence=6.4, profile="boxcar")

    grid = evidence_grid(
        cbm,
        s_tas,
        sigmas=np.geomspace(0.05, 0.8, 5),
        lengths_yr=np.geomspace(1.0, 20.0, 6),
        extra_production=spike,
        n_grid=64,
    )
    assert grid.log_evidence.shape == (5, 6)
    assert jnp.all(jnp.isfinite(grid.log_evidence))

    rc_before = dict(matplotlib.rcParams)
    fig1, ax1 = plot_evidence_grid(grid, title="Evidence Surface")
    assert fig1 is not None and ax1 is not None
    plt.close(fig1)

    z0 = jnp.zeros((64,), dtype=jnp.float64)
    f1 = GaussianField(z0, 760.0, 790.0, sigma=0.3, length_yr=1.5)
    f2 = GaussianField(z0, 760.0, 790.0, sigma=0.3, length_yr=12.0)
    fig2, axes2 = plot_field_draws({"Short": f1, "Decadal": f2}, jax.random.PRNGKey(0))
    assert axes2.shape == (2, 2)
    plt.close(fig2)

    fig3, axes3 = plot_field_draws(f1, jax.random.PRNGKey(1))
    assert axes3.size == 4
    plt.close(fig3)
    assert dict(matplotlib.rcParams) == rc_before


def test_notebook_04_ift_kernel_selection_structure_and_execution() -> None:
    nb_path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "v2"
        / "04_ift_kernel_selection.ipynb"
    )
    assert nb_path.exists(), f"Missing required notebook {nb_path}."

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    md_cells = [c for c in cells if c["cell_type"] == "markdown"]

    assert len(code_cells) == 5, f"Expected 5 code cells, got {len(code_cells)}."
    assert len(md_cells) >= 6, "Expected opening, interleaving, and closing prose cells."
    assert cells[0]["cell_type"] == "markdown"
    assert cells[-1]["cell_type"] == "markdown"

    png_count = 0
    for idx, c in enumerate(code_cells):
        assert c.get("execution_count") is not None, f"Code cell {idx + 1} was not executed."
        outputs = c.get("outputs", [])
        assert len(outputs) >= 1, f"Code cell {idx + 1} produced no output."
        has_error = any(o.get("output_type") == "error" for o in outputs)
        assert not has_error, f"Code cell {idx + 1} raised an error."

        has_png = any(
            "image/png" in o.get("data", {})
            for o in outputs
            if o.get("output_type") in ("display_data", "execute_result")
        )
        has_text_stream = any(o.get("output_type") == "stream" for o in outputs)
        # Strictly ONE clear output kind per cell: never mix stdout print and PNG figure
        assert not (has_png and has_text_stream), (
            f"Code cell {idx + 1} mixes text stream output and an inline PNG figure."
        )
        if has_png:
            png_count += 1

    assert png_count == 4, f"Expected 4 figure cells + 1 table cell across 5 cells, got {png_count} PNGs."


def test_duplicate_series_names_error_scale_and_apply_guard() -> None:
    """Duplicate default series names must not drop rows in global error_scale."""
    cbm = load_preset("Brehm21")
    rng = np.random.default_rng(109)
    years = np.arange(765.0, 785.0, 1.0)
    d1 = rng.normal(0.0, 1.2, size=years.size)
    d2 = rng.normal(-2.0, 1.5, size=years.size)
    sig1 = np.full_like(years, 2.4)
    sig2 = np.full_like(years, 3.0)

    s_a_named = TreeRingSeries(years, d1, sig1, hemisphere="north", name="S1")
    s_b_named = TreeRingSeries(years, d2, sig2, hemisphere="south", year_convention="schulman_start", name="S2")
    ds_named = Dataset((s_a_named, s_b_named))

    # Same data, both keeping default name="series"
    s_a_dup = TreeRingSeries(years, d1, sig1, hemisphere="north")
    s_b_dup = TreeRingSeries(years, d2, sig2, hemisphere="south", year_convention="schulman_start")
    ds_dup = Dataset((s_a_dup, s_b_dup))

    gf = GaussianField(jnp.zeros((48,), dtype=jnp.float64), 755.0, 790.0, sigma=0.25, length_yr=5.0)
    model = RadiocarbonModel(cbm, gf, ExactLinearSolver())

    s_glob_named = float(error_scale(model, ds_named, by_series=False))
    s_glob_dup = float(error_scale(model, ds_dup, by_series=False))
    np.testing.assert_allclose(s_glob_dup, s_glob_named, rtol=1e-13, atol=1e-13)

    with pytest.raises(ValueError, match="unique"):
        _ = error_scale(model, ds_dup, by_series=True)
    with pytest.raises(ValueError, match="unique"):
        _ = apply_error_scale(ds_dup, {"series": 0.5})


def test_symmetric_and_multipart_custom_sum_spectrum_optimization() -> None:
    """Custom SumSpectrum optimization must be symmetric in summand order and support >=3 parts."""
    cbm = load_preset("Brehm21")
    s_tas = load_bundled_series("775AD-early-S/Büntgen18_TAS01.csv")
    spike = MiyakeSpike(t_start=774.55, duration=0.25, fluence=6.4, profile="boxcar")

    m_part = MaternSpectrum(sigma=0.2, length_yr=6.0, order=2.0)
    qp_part = QuasiPeriodicSpectrum(sigma=0.2, period_yr=11.0, quality=4.0)
    se_part = SquaredExpSpectrum(sigma=0.1, length_yr=12.0)

    res_mq = optimize_gp_kernel(
        cbm, s_tas, kernel=m_part + qp_part, extra_production=spike, n_grid=48, multi_start=1
    )
    res_qm = optimize_gp_kernel(
        cbm, s_tas, kernel=qp_part + m_part, extra_production=spike, n_grid=48, multi_start=1
    )
    assert res_mq.n_hyperparams == 5
    assert res_qm.n_hyperparams == 5
    np.testing.assert_allclose(res_qm.log_evidence, res_mq.log_evidence, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(
        res_qm.log_hyper_evidence, res_mq.log_hyper_evidence, rtol=1e-4, atol=1e-3
    )

    sel_custom = select_gp_kernel(
        cbm,
        s_tas,
        candidates={"solar_first": qp_part + m_part, "triple": qp_part + m_part + se_part},
        extra_production=spike,
        n_grid=48,
        multi_start=1,
    )
    assert sel_custom.get_candidate("triple").n_hyperparams == 7
    assert jnp.all(jnp.isfinite(sel_custom.log_hyper_evidences))


def test_log_evidence_extra_priors_invariant_to_linear_latent() -> None:
    """Off-MAP fluence=3.0 extra_priors matches exact closed-form Gaussian marginal likelihood."""
    cbm = load_preset("Brehm21")
    s_tas = load_bundled_series("775AD-early-S/Büntgen18_TAS01.csv")
    ds_tas = Dataset((s_tas,))
    spike_off = MiyakeSpike(t_start=774.55, duration=0.25, fluence=3.0, profile="boxcar")
    spike_init2 = MiyakeSpike(t_start=774.55, duration=0.25, fluence=11.5, profile="boxcar")

    z_zero = jnp.zeros((48,), dtype=jnp.float64)
    z_rand = 4.5 * jax.random.normal(jax.random.PRNGKey(77), (48,), dtype=jnp.float64)
    gf_0 = GaussianField(z_zero, 760.0, 795.0, sigma=0.25, length_yr=5.0)
    m_off = RadiocarbonModel(cbm, spike_off + gf_0, ExactLinearSolver())
    m_rand = RadiocarbonModel(
        cbm, spike_init2 + GaussianField(z_rand, 760.0, 795.0, sigma=0.25, length_yr=5.0), ExactLinearSolver()
    )
    fit = fit_ift_field(m_off, ds_tas)
    extra_p = {"production.spike.fluence": dist.Normal(6.4, 2.0)}

    lnz_off = log_evidence(m_off, ds_tas, extra_priors=extra_p)
    lnz_rand = log_evidence(m_rand, ds_tas, extra_priors=extra_p)
    lnz_fit = log_evidence(fit.model, ds_tas, extra_priors=extra_p)
    np.testing.assert_allclose(lnz_rand, lnz_off, rtol=1e-11, atol=1e-10)
    np.testing.assert_allclose(lnz_fit, lnz_off, rtol=1e-11, atol=1e-10)

    # Compare against exact closed-form linear-Gaussian marginal log-likelihood over (fluence, z, offsets)
    op = build_dct_response_matrix(cbm, ds_tas, 760.0, 795.0, 48, return_operator=True)
    m_unit = RadiocarbonModel(
        cbm, MiyakeSpike(t_start=774.55, duration=0.25, fluence=1.0, profile="boxcar"), ExactLinearSolver()
    )
    m_steady = RadiocarbonModel(cbm, MiyakeSpike(t_start=774.55, duration=0.25, fluence=0.0, profile="boxcar"), ExactLinearSolver())
    w_spike = (m_unit.predict_d14c(ds_tas) - m_steady.predict_d14c(ds_tas))[:, None]
    x_off, mu_off, sd_off = hierarchical_offset_design(
        ds_tas.series_indices, ds_tas.n_series, box_activity_ratios=op.box_activity_ratios, global_prior_sd=200.0, lab_prior_sd=1.5
    )
    x_dct = op.w_dct * jnp.sqrt(gf_0.power_spectrum())[None, :]
    _, root_y0_0 = stationary_cbm_spectral_cov(
        cbm.matrix_A(), cbm.production_vector(), gf_0.effective_kernel, alpha_unit=m_off.alpha_unit
    )
    x_init = op.init_state_basis @ root_y0_0
    m_init = x_init.shape[1]
    design_full = jnp.concatenate([w_spike, x_dct, x_off, x_init], axis=1)
    mean_full = jnp.concatenate([jnp.array([6.4]), jnp.zeros((48,)), mu_off, jnp.zeros((m_init,))])
    sd_full = jnp.concatenate([jnp.array([2.0]), jnp.ones((48,)), sd_off, jnp.ones((m_init,))])
    lm_exact = LinearMarginal(design_full, prior_mean=mean_full, prior_sd=sd_full, method="qr")
    lnz_closed = lm_exact.loglike(ds_tas.d14c_flat - op.ss_d14c, ds_tas.sigma_flat)
    np.testing.assert_allclose(lnz_off, lnz_closed, rtol=1e-10, atol=1e-9)

    # Positive bijector equivalence: dist.LogNormal(loc, scale) == ParamSpec(..., prior_mean=loc, prior_sd=scale, bijector="exp")
    ln_dist = {"production.spike.fluence": dist.LogNormal(math.log(6.4), 0.35)}
    ln_spec = {
        "production.spike.fluence": ParamSpec(
            "production.spike.fluence", prior_mean=math.log(6.4), prior_sd=0.35, bijector="exp"
        )
    }
    np.testing.assert_allclose(
        log_evidence(m_off, ds_tas, extra_priors=ln_dist),
        log_evidence(m_off, ds_tas, extra_priors=ln_spec),
        rtol=1e-10,
        atol=1e-9,
    )


def test_step4_shared_wiring_solver_data_and_top_level_exports() -> None:
    """Verify ExactLinearSolver hook, Dataset/TreeRingSeries.with_error_scale, and tt.* exports."""
    import ticktack as tt
    import ticktack.plotting as tt_plotting
    from ticktack.fields import predict_field_d14c

    for sym in (
        "GaussianField",
        "LogGaussianField",
        "MaternSpectrum",
        "SquaredExpSpectrum",
        "QuasiPeriodicSpectrum",
        "CorrelatedFieldSpectrum",
        "SumSpectrum",
        "fit_ift_field",
        "log_evidence",
        "laplace_samples",
        "error_scale",
        "evidence_grid",
        "optimize_gp_kernel",
        "select_gp_kernel",
        "plot_field_draws",
        "plot_evidence_grid",
    ):
        assert hasattr(tt, sym), f"Missing top-level export tt.{sym}"
    assert hasattr(tt_plotting, "plot_field_draws")
    assert hasattr(tt_plotting, "plot_evidence_grid")

    cbm = tt.load_preset("Brehm21")
    s_tas = tt.load_bundled_series("775AD-early-S/Büntgen18_TAS01.csv")
    s_kauri = tt.load_bundled_series("notebooks/775CE/Guttler15_Kauri.csv")
    ds = tt.Dataset((s_tas, s_kauri))

    # Pre-dataset burn-in [755, 795] must be honoured identically by RadiocarbonModel.predict_d14c
    z_exc = jax.random.normal(jax.random.PRNGKey(314), (64,), dtype=jnp.float64)
    gf = tt.GaussianField(z_exc, 755.0, 795.0, sigma=0.35, length_yr=6.0)
    lgf = tt.LogGaussianField(0.4 * z_exc, 755.0, 795.0, sigma=0.25, length_yr=6.0)

    for field_term in (gf, lgf):
        model = tt.RadiocarbonModel(cbm, field_term, tt.ExactLinearSolver())
        pred_model = model.predict_d14c(ds)
        pred_ift = predict_field_d14c(model, ds, include_offsets=False)
        np.testing.assert_allclose(pred_model, pred_ift, rtol=1e-11, atol=2e-11)

    # Native TreeRingSeries.with_error_scale and Dataset.with_error_scale
    s_scaled = s_tas.with_error_scale(0.5)
    np.testing.assert_allclose(s_scaled.sigma_d14c, 0.5 * s_tas.sigma_d14c, rtol=1e-15)
    ds_scaled = ds.with_error_scale({s_tas.name: 0.75, s_kauri.name: 1.25})
    np.testing.assert_allclose(ds_scaled.series[0].sigma_d14c, 0.75 * s_tas.sigma_d14c, rtol=1e-15)
    np.testing.assert_allclose(ds_scaled.series[1].sigma_d14c, 1.25 * s_kauri.sigma_d14c, rtol=1e-15)


def test_prehistory_kernel_dense_marginal_and_propagation() -> None:
    """prehistory_kernel unpacks stationary_cbm_gp_cov tuple and matches dense Sigma_y0 MVN."""
    cbm = load_preset("Brehm21")
    ds = _make_two_hemisphere_dataset(cbm)
    t_start, t_end, n_grid = 755.0, 792.0, 48
    op = build_dct_response_matrix(cbm, ds, t_start, t_end, n_grid, return_operator=True)
    gf = GaussianField(jnp.zeros((n_grid,), dtype=jnp.float64), t_start, t_end, sigma=0.32, length_yr=5.0)
    model = RadiocarbonModel(cbm, gf, ExactLinearSolver())
    pre_k = tk.quasisep.Matern32(scale=12.0, sigma=0.45)

    sigma_y0, root_y0 = stationary_cbm_gp_cov(
        cbm.matrix_A(), cbm.production_vector(), pre_k, alpha_unit=float(model.alpha_unit)
    )
    assert sigma_y0.shape == (cbm.n_boxes, cbm.n_boxes)
    assert root_y0.shape[0] == cbm.n_boxes

    lnz_pre = log_evidence(
        model, ds, offset_mode="hierarchical", global_offset_prior_sd=200.0, lab_offset_prior_sd=1.5, prehistory_kernel=pre_k
    )
    x_off, mu_off, sd_off = hierarchical_offset_design(
        ds.series_indices, ds.n_series, box_activity_ratios=op.box_activity_ratios, global_prior_sd=200.0, lab_prior_sd=1.5
    )
    cov_d = (
        op.w_grid @ gf.prior_covariance() @ op.w_grid.T
        + (x_off * (sd_off**2)[None, :]) @ x_off.T
        + op.init_state_basis @ sigma_y0 @ op.init_state_basis.T
        + jnp.diag(ds.sigma_flat**2)
    )
    r_d = ds.d14c_flat - op.ss_d14c - x_off @ mu_off
    sign_p, logdet_p = jnp.linalg.slogdet(cov_d)
    assert float(sign_p) > 0.0
    dense_lnz = -0.5 * (
        r_d @ jnp.linalg.solve(cov_d, r_d) + logdet_p + ds.n_obs * math.log(2.0 * math.pi)
    )
    np.testing.assert_allclose(lnz_pre, dense_lnz, rtol=1e-11, atol=1e-10)

    fit_pre = fit_ift_field(model, ds, prehistory_kernel=pre_k)
    assert fit_pre.prehistory_root_y0 is not None
    np.testing.assert_allclose(fit_pre.log_evidence, lnz_pre, rtol=1e-12, atol=1e-11)
    # Implicit reuse of fit_pre.prehistory_root_y0 without re-passing prehistory_kernel
    np.testing.assert_allclose(log_evidence(fit_pre, ds), lnz_pre, rtol=1e-12, atol=1e-11)
    draws_pre = laplace_samples(fit_pre, ds, n_samples=16, rng_key=jax.random.PRNGKey(3))
    assert draws_pre["q"].shape == (16, n_grid)
    s_pre = error_scale(fit_pre, ds)
    assert isinstance(s_pre, float) and s_pre > 0.0

    opt_pre = optimize_gp_kernel(
        cbm, ds, kernel="matern32", n_grid=32, multi_start=1, prehistory_kernel=pre_k
    )
    assert opt_pre.fit.prehistory_root_y0 is not None
    sel_pre = select_gp_kernel(
        cbm, ds, candidates=("matern12", "matern32"), n_grid=32, multi_start=1, prehistory_kernel=pre_k
    )
    assert sel_pre.best_result.fit.prehistory_root_y0 is not None


def test_spike_profiling_schur_evidence_and_log_gaussian_hessian() -> None:
    """Spike profiling uses full (n_k + 1) Schur-complement evidence, AIC/BIC, and exact Hessians."""
    cbm = load_preset("Brehm21")
    s_tas = load_bundled_series("775AD-early-S/Büntgen18_TAS01.csv")

    sp_low = MiyakeSpike(t_start=774.55, duration=0.25, fluence=1.2, profile="boxcar")
    sp_high = MiyakeSpike(t_start=774.55, duration=0.25, fluence=14.0, profile="boxcar")
    r_low = optimize_gp_kernel(cbm, s_tas, kernel="matern32", extra_production=sp_low, n_grid=40, multi_start=1)
    r_high = optimize_gp_kernel(cbm, s_tas, kernel="matern32", extra_production=sp_high, n_grid=40, multi_start=1)

    assert r_low.n_hyperparams == 2
    k_eff = r_low.n_hyperparams + 1
    np.testing.assert_allclose(r_low.aic, -2.0 * r_low.log_evidence + 2.0 * k_eff, rtol=1e-12)
    np.testing.assert_allclose(r_low.log_evidence, r_high.log_evidence, rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(r_low.log_hyper_evidence, r_high.log_hyper_evidence, rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(
        r_low.fit.model.get("production.spike.fluence"),
        r_high.fit.model.get("production.spike.fluence"),
        rtol=1e-5,
        atol=1e-5,
    )

    # Unclipped hyperprior restoring gradient at extreme |u| = 25 > 20 and dataset-span independence
    proto = MaternSpectrum(sigma=0.3, length_yr=5.0, order=2.0)
    g_extreme = jax.grad(lambda u: proto.from_unconstrained_hyperparams(u).hyperprior_log_prob(u))(
        jnp.array([25.0, -25.0], dtype=jnp.float64)
    )
    assert jnp.all(jnp.abs(g_extreme) > 1.0)

    # Non-PD Hessian returns -inf and converged=False (no +5.52 nats eigenvalue clamping bonus)
    r_non_pd = optimize_gp_kernel(
        cbm,
        s_tas,
        kernel="matern32",
        n_grid=32,
        multi_start=1,
        hyperprior=lambda u: 500.0 * jnp.sum(u**2),
    )
    assert not r_non_pd.converged
    assert float(r_non_pd.log_hyper_evidence) == float("-inf")

    # LogGaussianField multi_start + spike profiling + implicit Hessian finite-difference check
    r_lg = optimize_gp_kernel(
        cbm,
        s_tas,
        kernel="matern32",
        field_kind="log_gaussian",
        extra_production=sp_low,
        n_grid=28,
        multi_start=3,
        rng_key=jax.random.PRNGKey(12),
    )
    assert float(r_lg.fit.model.get("production.spike.fluence")) > 4.0
    r_lg_nosp = optimize_gp_kernel(
        cbm, s_tas, kernel="matern32", field_kind="log_gaussian", n_grid=24, multi_start=2
    )
    proto_lg = r_lg_nosp.kernel
    u_opt = r_lg_nosp.phi_map
    t0, t1 = r_lg_nosp.fit.operator.t_start, r_lg_nosp.fit.operator.t_end

    def _lg_neg_post(u_vec: jax.Array) -> float:
        k_u = proto_lg.from_unconstrained_hyperparams(u_vec)
        lgf_u = LogGaussianField(jnp.zeros((24,), dtype=jnp.float64), t0, t1, kernel=k_u)
        f_u = fit_ift_field(RadiocarbonModel(cbm, lgf_u), s_tas)
        return float(-(f_u.log_evidence + k_u.hyperprior_log_prob(u_vec)))

    eps = 1e-3
    h_fd = np.zeros((2, 2), dtype=np.float64)
    f0 = _lg_neg_post(u_opt)
    for i in range(2):
        ei = jnp.zeros((2,), dtype=jnp.float64).at[i].set(eps)
        h_fd[i, i] = (_lg_neg_post(u_opt + ei) - 2.0 * f0 + _lg_neg_post(u_opt - ei)) / (eps**2)
    np.testing.assert_allclose(np.diag(r_lg_nosp.hyper_hessian), np.diag(h_fd), rtol=2e-2, atol=2e-2)


def test_optimize_spike_both_fields_and_dataset_digest_non_aliasing() -> None:
    """optimize_spike=True supports Gaussian & LogGaussian fields with explicit priors; digest prevents aliasing."""
    cbm = load_preset("Brehm21")
    s_tas = load_bundled_series("775AD-early-S/Büntgen18_TAS01.csv")
    z0 = jnp.zeros((40,), dtype=jnp.float64)
    sp_a = MiyakeSpike(t_start=774.25, duration=0.35, fluence=5.0, profile="supergaussian")
    sp_b = MiyakeSpike(t_start=774.85, duration=0.45, fluence=8.0, profile="supergaussian")
    custom_sp = {
        "t_start": dist.Normal(774.6, 2.0),
        "duration": dist.LogNormal(math.log(0.4), 0.6),
        "fluence": dist.LogNormal(math.log(6.4), 0.6),
    }

    gf_a = RadiocarbonModel(cbm, sp_a + GaussianField(z0, 760.0, 795.0, sigma=0.25, length_yr=5.0))
    gf_b = RadiocarbonModel(cbm, sp_b + GaussianField(z0, 760.0, 795.0, sigma=0.25, length_yr=5.0))
    fit_ga = fit_ift_field(gf_a, s_tas, optimize_spike=True, spike_priors=custom_sp)
    fit_gb = fit_ift_field(gf_b, s_tas, optimize_spike=True, spike_priors=custom_sp)
    assert fit_ga.converged and fit_gb.converged
    np.testing.assert_allclose(fit_ga.log_evidence, fit_gb.log_evidence, rtol=1e-4, atol=1e-3)
    # Joint Schur-complement evidence matches log_evidence(..., extra_priors=...) closely
    ext_map = {f"production.spike.{k}": v for k, v in custom_sp.items()}
    np.testing.assert_allclose(
        fit_ga.log_evidence, log_evidence(fit_ga.model, s_tas, extra_priors=ext_map), rtol=2e-3, atol=5e-2
    )

    lgf_a = RadiocarbonModel(cbm, sp_a + LogGaussianField(z0, 760.0, 795.0, sigma=0.25, length_yr=5.0))
    fit_la = fit_ift_field(lgf_a, s_tas, optimize_spike=True, spike_priors=custom_sp)
    assert fit_la.converged
    assert 4.0 <= float(fit_la.model.get("production.spike.fluence")) <= 10.0

    # Template scaling: LogGaussianField with custom template preserves E[Q] == q_mean * template
    tmpl = jnp.linspace(0.5, 2.5, 32, dtype=jnp.float64)
    lgf_tmpl = LogGaussianField(
        jnp.zeros((32,), dtype=jnp.float64), 760.0, 790.0, sigma=0.2, length_yr=4.0, q_mean=2.0, template=tmpl
    )
    np.testing.assert_allclose(
        jnp.mean(lgf_tmpl.sample_prior(jax.random.PRNGKey(99), n_samples=4000), axis=0),
        2.0 * tmpl,
        rtol=0.04,
        atol=0.04,
    )

    # Dataset geometry digest non-aliasing on held-out dataset with identical n_obs
    s_shifted = TreeRingSeries(
        np.asarray(s_tas.raw_years) + 3.0,
        np.asarray(s_tas.d14c),
        np.asarray(s_tas.sigma_d14c),
        hemisphere=s_tas.hemisphere,
        name="Shifted",
    )
    assert s_shifted.n_obs == s_tas.n_obs
    fit_tas = fit_ift_field(RadiocarbonModel(cbm, GaussianField(z0, 755.0, 805.0, sigma=0.3, length_yr=5.0)), s_tas)
    assert fit_tas.initial_state_mean is not None
    op_shifted = build_dct_response_matrix(cbm, s_shifted, 755.0, 805.0, 40, return_operator=True)
    np.testing.assert_allclose(
        predict_field_d14c(fit_tas, s_shifted, include_offsets=False),
        predict_field_d14c(fit_tas.model, s_shifted, include_offsets=False)
        + op_shifted.init_state_basis @ fit_tas.initial_state_mean,
        rtol=1e-12,
        atol=1e-12,
    )
    fit_tas_noinit = fit_ift_field(
        RadiocarbonModel(cbm, GaussianField(z0, 755.0, 805.0, sigma=0.3, length_yr=5.0)),
        s_tas,
        include_field_initial_cov=False,
    )
    np.testing.assert_allclose(
        predict_field_d14c(fit_tas_noinit, s_shifted, include_offsets=False),
        predict_field_d14c(fit_tas_noinit.model, s_shifted, include_offsets=False),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        log_evidence(fit_tas, s_shifted), log_evidence(fit_tas.model, s_shifted), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        error_scale(fit_tas, s_shifted), error_scale(fit_tas.model, s_shifted), rtol=1e-12, atol=1e-12
    )

    # laplace_samples(RadiocarbonModel(cbm, lgf_zero), ds) fits MAP first rather than sampling around z=0
    lgf_unfitted = RadiocarbonModel(cbm, sp_a + LogGaussianField(z0, 760.0, 795.0, sigma=0.35, length_yr=4.0))
    fit_lgf_ref = fit_ift_field(lgf_unfitted, s_tas)
    draws_from_model = laplace_samples(lgf_unfitted, s_tas, n_samples=200, rng_key=jax.random.PRNGKey(7))
    draws_from_fit = laplace_samples(fit_lgf_ref, s_tas, n_samples=200, rng_key=jax.random.PRNGKey(7))
    np.testing.assert_allclose(draws_from_model["q"], draws_from_fit["q"], rtol=1e-10, atol=1e-10)


def test_likelihood_and_fit_map_ift_integration_and_edge_cases() -> None:
    """ticktack.likelihood & fit_map support GaussianField/LogGaussianField + 1-ring & dup-name edge cases."""
    cbm = load_preset("Brehm21")
    s_tas = load_bundled_series("775AD-early-S/Büntgen18_TAS01.csv")
    ds_tas = Dataset((s_tas,))
    z0 = jnp.zeros((40,), dtype=jnp.float64)
    sp = MiyakeSpike(t_start=774.55, duration=0.25, fluence=5.2, profile="boxcar")

    gf = GaussianField(z0, 760.0, 795.0, sigma=0.3, length_yr=5.0)
    m_gf = RadiocarbonModel(cbm, sp + gf, ExactLinearSolver())
    cfg = LinearMarginalConfig(
        offset_mode="hierarchical", global_prior_mean=0.0, global_prior_sd=200.0, lab_prior_sd=1.5
    )
    np.testing.assert_allclose(
        marginal_log_likelihood(m_gf, ds_tas, cfg), log_evidence(m_gf, ds_tas), rtol=1e-11, atol=1e-10
    )

    t_fine = jnp.linspace(765.0, 790.0, 50, dtype=jnp.float64)
    lp_gf = linear_posterior(m_gf, ds_tas, cfg, rng_key=jax.random.PRNGKey(4), n_draws=8, t_fine=t_fine)
    assert lp_gf.q_fine_draws is not None and lp_gf.q_fine_draws.shape == (8, 50)
    assert lp_gf.d14c_fine_draws is not None and lp_gf.d14c_fine_draws.shape == (8, 50)

    ps_sp = ParameterSpace(m_gf, [ParamSpec("production.spike.fluence", prior_mean=6.4, prior_sd=2.0)])
    map_gf = fit_map(m_gf, ds_tas, ps_sp, cfg=cfg, method="BFGS")
    assert map_gf.converged
    assert 5.0 <= float(map_gf.theta_map["production.spike.fluence"]) <= 9.0

    lgf = LogGaussianField(z0, 760.0, 795.0, sigma=0.28, length_yr=5.0)
    m_lgf = RadiocarbonModel(cbm, sp + lgf, ExactLinearSolver())
    fit_lgf = fit_ift_field(m_lgf, ds_tas)
    np.testing.assert_allclose(
        marginal_log_likelihood(m_lgf, ds_tas, cfg), log_evidence(fit_lgf, ds_tas), rtol=1e-8, atol=1e-7
    )
    ps_sp_lgf = ParameterSpace(m_lgf, [ParamSpec("production.spike.fluence", prior_mean=6.4, prior_sd=2.0)])
    map_lgf = fit_map(m_lgf, ds_tas, ps_sp_lgf, cfg=cfg, method="BFGS")
    assert 4.5 <= float(map_lgf.theta_map["production.spike.fluence"]) <= 9.0

    # 1-ring series + duplicate series names in error_scale and select_gp_kernel
    s_one = TreeRingSeries([776.0], [10.5], [1.8], hemisphere="north", name="OneRing")
    ds_1ring = Dataset((s_tas, s_one))
    s_glob_1ring = error_scale(m_gf, ds_1ring, by_series=False)
    assert isinstance(s_glob_1ring, float) and s_glob_1ring > 0.0
    sel_1ring = select_gp_kernel(cbm, ds_1ring, candidates=("matern12", "matern32"), n_grid=28, multi_start=1)
    assert "MacKay s_error" in sel_1ring.format_table()

    s_dup1 = TreeRingSeries(np.asarray(s_tas.raw_years), np.asarray(s_tas.d14c), np.asarray(s_tas.sigma_d14c), hemisphere="south")
    s_dup2 = TreeRingSeries(np.asarray(s_tas.raw_years), np.asarray(s_tas.d14c) + 0.5, np.asarray(s_tas.sigma_d14c), hemisphere="south")
    ds_dup = Dataset((s_dup1, s_dup2))
    sel_dup = select_gp_kernel(cbm, ds_dup, candidates=("matern12", "matern32"), n_grid=28, multi_start=1)
    assert "MacKay s_error" in sel_dup.format_table()


def test_spike_on_marginalized_gp_fit_nuts_and_select_kernel() -> None:
    """Spike on marginalized GP: select_gp_kernel(optimize_spike=True), cache reuse, fit_map, sample_nuts, linear_posterior."""
    cbm = load_preset("Miyake17")
    t_start_g, t_end_g, n_grid = 745.0, 801.0, 40
    years = np.arange(755.0, 800.0, 1.0)
    sig_obs = 0.7

    s_proto = TreeRingSeries(
        years, np.zeros_like(years), np.full_like(years, sig_obs), hemisphere="north", name="SynthSpikeGP"
    )
    true_spike = MiyakeSpike(t_start=774.2, duration=0.8, fluence=6.2, profile="boxcar")
    true_field = GaussianField(
        jax.random.normal(jax.random.PRNGKey(51), (n_grid,), dtype=jnp.float64),
        t_start_g,
        t_end_g,
        kernel=QuasiPeriodicSpectrum(sigma=0.35, period_yr=11.0, quality=6.0),
    )
    true_model = RadiocarbonModel(cbm, SteadyProduction() + true_field + true_spike, ExactLinearSolver())
    d_clean = np.asarray(true_model.predict_d14c(s_proto))
    rng = np.random.default_rng(52)
    series = TreeRingSeries(
        years,
        d_clean + 1.8 + rng.normal(0.0, sig_obs, size=years.size),
        np.full_like(years, sig_obs),
        hemisphere="north",
        name="SynthSpikeGP",
    )
    ds = Dataset((series,), name="synth_spike_gp")

    # 1. select_gp_kernel(..., optimize_spike=True, spike_priors=...) recovers spike & sho_solar
    rough_spike = MiyakeSpike(t_start=774.5, duration=0.4, fluence=5.0, profile="boxcar")
    sp_priors = {
        "t_start": (774.3, 1.5),
        "duration": dist.LogNormal(math.log(0.7), 0.8),
        "fluence": dist.LogNormal(math.log(6.0), 0.8),
    }
    sel = select_gp_kernel(
        cbm,
        ds,
        candidates=("matern32", "sho_solar"),
        extra_production=rough_spike,
        optimize_spike=True,
        spike_priors=sp_priors,
        n_grid=n_grid,
        pad_years=10.0,
        multi_start=1,
        rng_key=jax.random.PRNGKey(11),
    )
    assert sel.best_name == "sho_solar"
    rec_t = float(sel.best_model.get("production.spike.t_start"))
    rec_d = float(sel.best_model.get("production.spike.duration"))
    rec_f = float(sel.best_model.get("production.spike.fluence"))
    assert abs(rec_t - 774.2) < 0.35, f"Expected t_start ~ 774.2, got {rec_t:.3f}"
    assert 0.30 <= rec_d <= 1.50, f"Expected duration ~ 0.8, got {rec_d:.3f}"
    assert 4.8 <= rec_f <= 7.8, f"Expected fluence ~ 6.2, got {rec_f:.3f}"

    opt_sp = optimize_gp_kernel(
        cbm,
        ds,
        kernel="sho_solar",
        extra_production=rough_spike,
        optimize_spike=True,
        spike_priors=sp_priors,
        n_grid=n_grid,
        pad_years=10.0,
        multi_start=1,
        rng_key=jax.random.PRNGKey(13),
    )
    assert opt_sp.converged
    assert abs(float(opt_sp.fit.model.get("production.spike.t_start")) - 774.2) < 0.35
    jax.clear_caches()

    # 2. SteadyProduction + GaussianField + MiyakeSpike (marginalize_field=True):
    #    both fixed-kernel and joint spike + GP-hyperparameter inference + operator cache reuse
    gf_init = GaussianField(jnp.zeros((n_grid,), dtype=jnp.float64), t_start_g, t_end_g, sigma=0.28, length_yr=5.0)
    m_gf = RadiocarbonModel(cbm, SteadyProduction() + gf_init + rough_spike, ExactLinearSolver())
    cfg_gf = LinearMarginalConfig(marginalize_field=True, offset_mode="hierarchical")
    spike_specs = [
        ParamSpec("production.spike.t_start", prior_mean=774.4, prior_sd=1.0),
        ParamSpec("production.spike.duration", bijector="exp", prior_mean=math.log(0.6), prior_sd=0.8),
        ParamSpec("production.spike.fluence", bijector="exp", prior_mean=math.log(6.0), prior_sd=0.8),
    ]
    ps_fixed = ParameterSpace(m_gf, spike_specs)
    map_fixed = fit_map(m_gf, ds, ps_fixed, cfg=cfg_gf, method="BFGS")
    assert map_fixed.converged
    assert abs(float(map_fixed.theta_map["production.spike.t_start"]) - 774.2) < 0.40
    assert 4.5 <= float(map_fixed.theta_map["production.spike.fluence"]) <= 8.0

    idata_fixed = sample_nuts(
        map_fixed.model,
        ds,
        ps_fixed,
        cfg=cfg_gf,
        init_map=False,
        num_warmup=25,
        num_samples=25,
        num_chains=1,
        max_tree_depth=6,
        rng_key=jax.random.PRNGKey(21),
        progress_bar=False,
    )
    assert jnp.all(jnp.isfinite(jnp.asarray(idata_fixed.posterior["production__spike__fluence"])))

    ps_joint = ParameterSpace(
        map_fixed.model,
        spike_specs
        + [
            ParamSpec("production.field.sigma", bijector="exp", prior_mean=math.log(0.28), prior_sd=0.8),
            ParamSpec("production.field.length_yr", bijector="exp", prior_mean=math.log(5.0), prior_sd=0.8),
        ],
    )
    _DCT_OPERATOR_CACHE.clear()
    map_joint = fit_map(map_fixed.model, ds, ps_joint, cfg=cfg_gf, method="BFGS", compute_laplace=False)
    assert map_joint.converged
    idata_joint = sample_nuts(
        map_joint.model,
        ds,
        ps_joint,
        cfg=cfg_gf,
        init_map=False,
        num_warmup=25,
        num_samples=25,
        num_chains=1,
        max_tree_depth=6,
        rng_key=jax.random.PRNGKey(22),
        progress_bar=False,
    )
    # Concrete-leaf operator cache must be populated and reused inside sample_nuts!
    assert len(_DCT_OPERATOR_CACHE) >= 1
    for var_name in (
        "production__spike__t_start",
        "production__spike__duration",
        "production__spike__fluence",
        "production__field__sigma",
        "production__field__length_yr",
    ):
        arr = np.asarray(idata_joint.posterior[var_name])
        assert np.all(np.isfinite(arr))

    t_fine = jnp.linspace(756.0, 799.0, 300, dtype=jnp.float64)
    lp_gf = linear_posterior(
        map_joint.model, ds, cfg_gf, rng_key=jax.random.PRNGKey(23), n_draws=16, t_fine=t_fine
    )
    assert lp_gf.q_fine_mean is not None and lp_gf.q_fine_std is not None
    assert jnp.all(jnp.isfinite(lp_gf.q_fine_mean)) and jnp.all(lp_gf.q_fine_std > 0.0)
    assert float(jnp.max(lp_gf.q_fine_mean)) > DEFAULT_Q_STD + 2.5
    jax.clear_caches()

    # 3. SteadyProduction + GPProduction + MiyakeSpike (marginalize_gp=True)
    gp_term = GPProduction(tk.quasisep.Matern32(scale=5.0, sigma=0.28), n_modes=40, pad_years=10.0)
    m_gp = RadiocarbonModel(cbm, SteadyProduction() + gp_term + rough_spike, ExactLinearSolver())
    cfg_gp = LinearMarginalConfig(marginalize_gp=True, offset_mode="hierarchical")
    ps_gp = ParameterSpace(m_gp, spike_specs)
    map_gp = fit_map(m_gp, ds, ps_gp, cfg=cfg_gp, method="BFGS")
    assert map_gp.converged
    assert abs(float(map_gp.theta_map["production.spike.t_start"]) - 774.2) < 0.40
    idata_gp = sample_nuts(
        map_gp.model,
        ds,
        ps_gp,
        cfg=cfg_gp,
        init_map=False,
        num_warmup=25,
        num_samples=25,
        num_chains=1,
        max_tree_depth=6,
        rng_key=jax.random.PRNGKey(24),
        progress_bar=False,
    )
    assert jnp.all(jnp.isfinite(jnp.asarray(idata_gp.posterior["production__spike__fluence"])))
    lp_gp = linear_posterior(
        map_gp.model, ds, cfg_gp, rng_key=jax.random.PRNGKey(25), n_draws=16, t_fine=t_fine
    )
    assert lp_gp.q_fine_mean is not None and lp_gp.q_fine_std is not None
    assert jnp.all(jnp.isfinite(lp_gp.q_fine_mean)) and jnp.all(lp_gp.q_fine_std > 0.0)
    assert float(jnp.max(lp_gp.q_fine_mean)) > DEFAULT_Q_STD + 2.5
    jax.clear_caches()


def test_fit_ift_field_optimize_spike_woodbury_matches_dense_mvn() -> None:
    """O(N_obs K) QR square-root Woodbury varpro matches dense N_obs x N_obs MVN when N_obs > K."""
    cbm = load_preset("Guttler15")
    t_start_g, t_end_g, n_grid = 755.0, 795.0, 20
    years = np.arange(762.0, 787.0, 1.0)
    rng = np.random.default_rng(7751)

    true_spike = MiyakeSpike(t_start=774.45, duration=0.45, fluence=5.8, profile="boxcar")
    true_field = GaussianField(
        jax.random.normal(jax.random.PRNGKey(7752), (n_grid,), dtype=jnp.float64),
        t_start_g,
        t_end_g,
        sigma=0.28,
        length_yr=5.5,
    )
    true_model = RadiocarbonModel(cbm, true_spike + true_field, ExactLinearSolver())

    series_list = []
    for idx, (hemi, off, sig) in enumerate(
        (("north", 2.4, 1.1), ("north", 1.1, 1.3), ("south", 1.9, 1.2))
    ):
        proto = TreeRingSeries(
            years, np.zeros_like(years), np.full_like(years, sig), hemisphere=hemi, name=f"S{idx}"
        )
        d_clean = np.asarray(true_model.predict_d14c(proto))
        series_list.append(
            TreeRingSeries(
                years,
                d_clean + off + rng.normal(0.0, sig, size=years.size),
                np.full_like(years, sig),
                hemisphere=hemi,
                name=f"S{idx}",
            )
        )
    ds = Dataset(tuple(series_list), name="tall_spike_varpro")
    # 3 series x 25 rings = 75 observations > n_grid (20) + k_off (4) + n_boxes (11) = 35
    k_nuis = 1 + ds.n_series + cbm.n_boxes
    assert ds.n_obs == 75
    assert ds.n_obs > n_grid + k_nuis

    op = build_dct_response_matrix(cbm, ds, t_start_g, t_end_g, n_grid, return_operator=True)
    sp_init = MiyakeSpike(t_start=774.2, duration=0.35, fluence=4.8, profile="boxcar")
    gf_init = GaussianField(
        jnp.zeros((n_grid,), dtype=jnp.float64),
        t_start_g,
        t_end_g,
        sigma=0.28,
        length_yr=5.5,
    )
    model_init = RadiocarbonModel(cbm, sp_init + gf_init, ExactLinearSolver())
    sp_priors = {
        "t_start": dist.Normal(774.5, 1.5),
        "duration": dist.LogNormal(math.log(0.45), 0.7),
        "fluence": dist.LogNormal(math.log(5.5), 0.7),
    }

    fit_res = fit_ift_field(
        model_init,
        ds,
        offset_mode="hierarchical",
        global_offset_prior_sd=5.0,
        lab_offset_prior_sd=1.5,
        operator=op,
        optimize_spike=True,
        spike_priors=sp_priors,
        max_steps=100,
        tol=1e-14,
    )
    assert fit_res.converged

    # Build independent dense N_obs x N_obs MVN marginal covariance reference
    inv_sig = 1.0 / ds.sigma_flat
    x_off, mu_off, sd_off = hierarchical_offset_design(
        ds.series_indices,
        ds.n_series,
        box_activity_ratios=op.box_activity_ratios,
        global_prior_sd=5.0,
        lab_prior_sd=1.5,
    )
    _, root_y0 = stationary_cbm_spectral_cov(
        cbm.matrix_A(),
        cbm.production_vector(),
        gf_init.effective_kernel,
        alpha_unit=model_init.alpha_unit,
    )
    u_off = (inv_sig[:, None] * x_off) * sd_off[None, :]
    u_init = (inv_sig[:, None] * op.init_state_basis) @ root_y0
    u_nuis = jnp.concatenate([u_off, u_init], axis=1)
    j_raw = inv_sig[:, None] * (op.w_dct * jnp.sqrt(jnp.maximum(gf_init.power_spectrum(), 0.0))[None, :])
    u_all = jnp.concatenate([u_nuis, j_raw], axis=1)
    assert u_all.shape == (ds.n_obs, n_grid + k_nuis)

    cov_dense = jnp.eye(ds.n_obs, dtype=jnp.float64) + u_all @ u_all.T
    chol_dense = jnp.linalg.cholesky(cov_dense)
    sign_c, logdet_c = jnp.linalg.slogdet(cov_dense)
    assert float(sign_c) > 0.0
    log_norm_d = 0.5 * ds.n_obs * math.log(2.0 * math.pi) + jnp.sum(jnp.log(ds.sigma_flat))
    nuis_shift = x_off @ mu_off

    sp_prior_means = jnp.array([774.5, math.log(0.45), math.log(5.5)], dtype=jnp.float64)
    sp_prior_sds = jnp.array([1.5, 0.7, 0.7], dtype=jnp.float64)

    def _raw_resid(u_sp: jax.Array) -> jax.Array:
        sp_cur = MiyakeSpike(
            t_start=u_sp[0],
            duration=jnp.exp(u_sp[1]),
            fluence=jnp.exp(u_sp[2]),
            profile="boxcar",
        )
        m_sp = RadiocarbonModel(cbm, sp_cur, ExactLinearSolver())
        d_sp = m_sp.predict_d14c(ds)
        return inv_sig * (d_sp + nuis_shift - ds.d14c_flat)

    def _dense_varpro_res(u_sp: jax.Array, _args: object) -> jax.Array:
        r_raw = _raw_resid(u_sp)
        r_marg = jsl.solve_triangular(chol_dense, r_raw, lower=True)
        sp_pen = (u_sp - sp_prior_means) / sp_prior_sds
        return jnp.concatenate([r_marg, sp_pen], axis=0)

    u_sp0 = jnp.array([774.2, math.log(0.35), math.log(4.8)], dtype=jnp.float64)
    sol_dense = optx.least_squares(
        _dense_varpro_res,
        optx.LevenbergMarquardt(rtol=1e-12, atol=1e-12),
        u_sp0,
        max_steps=80,
        throw=False,
    )
    assert sol_dense.result == optx.RESULTS.successful
    u_sp_dense = sol_dense.value

    grad_dense = jax.jit(
        jax.grad(lambda u_in: 0.5 * jnp.sum(jnp.square(_dense_varpro_res(u_in, None))))
    )

    def _dense_hess_at(u_pt: jax.Array) -> jax.Array:
        eps_h = 1e-6
        cols = []
        for j_idx in range(3):
            ej = jnp.zeros((3,), dtype=jnp.float64).at[j_idx].set(eps_h)
            cols.append((grad_dense(u_pt + ej) - grad_dense(u_pt - ej)) / (2.0 * eps_h))
        h_m = jnp.stack(cols, axis=1)
        return 0.5 * (h_m + h_m.T)

    # Polish dense MAP to machine precision (matching fit_ift_field's 2-step Newton polish)
    for _ in range(2):
        u_sp_dense = u_sp_dense - jnp.linalg.solve(_dense_hess_at(u_sp_dense), grad_dense(u_sp_dense))

    # Verify QR square-root Woodbury quadratic form matches dense MVN quadratic form
    k_all = u_all.shape[1]
    eye_k = jnp.eye(k_all, dtype=jnp.float64)
    _, r_upper = jnp.linalg.qr(jnp.concatenate([u_all, eye_k], axis=0), mode="reduced")
    signs = jnp.where(jnp.diag(r_upper) >= 0.0, 1.0, -1.0)
    chol_k = (signs[:, None] * r_upper).T
    l_inv_t = jsl.solve_triangular(chol_k, eye_k, lower=True).T
    w_mat = u_all @ l_inv_t
    p_right = jsl.solve_triangular(eye_k + l_inv_t, w_mat.T, lower=False)

    for u_eval in (u_sp0, u_sp_dense):
        r_raw = _raw_resid(u_eval)
        r_qr = r_raw - w_mat @ (p_right @ r_raw)
        q_qr = r_qr @ r_qr
        q_dense = r_raw @ jsl.cho_solve((chol_dense, True), r_raw)
        np.testing.assert_allclose(q_qr, q_dense, rtol=1e-10, atol=1e-10)

    # Compare MAP spike parameters
    sp_opt_fit = jnp.array(
        [
            fit_res.model.get("production.spike.t_start"),
            fit_res.model.get("production.spike.duration"),
            fit_res.model.get("production.spike.fluence"),
        ],
        dtype=jnp.float64,
    )
    sp_opt_dense = jnp.array(
        [u_sp_dense[0], jnp.exp(u_sp_dense[1]), jnp.exp(u_sp_dense[2])], dtype=jnp.float64
    )
    np.testing.assert_allclose(sp_opt_fit, sp_opt_dense, rtol=1e-10, atol=1e-10)

    # Compare exact dense marginal log-evidence including Laplace spike_occam_term
    r_opt = _raw_resid(u_sp_dense)
    q_opt_dense = r_opt @ jsl.cho_solve((chol_dense, True), r_opt)
    ln_z_cond_dense = -0.5 * q_opt_dense - 0.5 * logdet_c - log_norm_d
    h_sp_dense = _dense_hess_at(u_sp_dense)
    _, logdet_h_dense = jnp.linalg.slogdet(h_sp_dense)
    lp_sp_dense = jnp.sum(
        -0.5 * jnp.square((u_sp_dense - sp_prior_means) / sp_prior_sds)
        - jnp.log(sp_prior_sds)
        - 0.5 * math.log(2.0 * math.pi)
    )
    spike_occam_dense = lp_sp_dense + 1.5 * math.log(2.0 * math.pi) - 0.5 * logdet_h_dense
    ln_z_total_dense = ln_z_cond_dense + spike_occam_dense
    np.testing.assert_allclose(fit_res.log_evidence, ln_z_total_dense, rtol=1e-10, atol=1e-10)
    jax.clear_caches()


def test_stationary_cbm_spectral_cov_and_initial_state_posteriors() -> None:
    """stationary_cbm_spectral_cov, default include_field_initial_cov=True, and initial-state posteriors."""
    cbm = load_preset("Brehm21")
    ds = _make_two_hemisphere_dataset(cbm, seed=19)
    a_mat = cbm.matrix_A()
    b_vec = cbm.production_vector()

    # 1. MaternSpectrum(order=2) matches stationary_cbm_gp_cov(Matern32) to < 1e-10
    sig, scale = 0.42, 12.0
    m32_spec = MaternSpectrum(sigma=sig, length_yr=scale / math.sqrt(3.0), order=2.0)
    m32_tinygp = tk.quasisep.Matern32(scale=scale, sigma=sig)
    alpha = float(RadiocarbonModel(cbm, SteadyProduction()).alpha_unit)
    sig_y0_spec, root_y0_spec = stationary_cbm_spectral_cov(
        a_mat, b_vec, m32_spec, alpha_unit=alpha
    )
    sig_y0_gp, _ = stationary_cbm_gp_cov(a_mat, b_vec, m32_tinygp, alpha_unit=alpha)
    np.testing.assert_allclose(sig_y0_spec, sig_y0_gp, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(root_y0_spec @ root_y0_spec.T, sig_y0_spec, rtol=1e-9, atol=1e-9)

    # All elementary SpectralKernel classes + SumSpectrum produce finite PSD covariances
    for kern in (
        SquaredExpSpectrum(sigma=0.35, length_yr=8.0),
        QuasiPeriodicSpectrum(sigma=0.25, period_yr=11.0, quality=4.0),
        CorrelatedFieldSpectrum(sigma=0.30, slope=-2.2),
        m32_spec + QuasiPeriodicSpectrum(sigma=0.20, period_yr=11.0, quality=5.0),
    ):
        s_k, r_k = stationary_cbm_spectral_cov(a_mat, b_vec, kern, alpha_unit=alpha)
        assert s_k.shape == (cbm.n_boxes, cbm.n_boxes)
        assert jnp.all(jnp.isfinite(s_k)) and jnp.all(jnp.isfinite(r_k))
        evals_k = jnp.linalg.eigvalsh(s_k)
        assert float(jnp.min(evals_k)) >= -1e-12 * max(float(jnp.max(evals_k)), 1.0)
        np.testing.assert_allclose(r_k @ r_k.T, s_k, rtol=1e-9, atol=1e-9)

    # 2. Default include_field_initial_cov=True matches dense MVN & marginal_log_likelihood
    t_start, t_end, n_grid = 755.0, 792.0, 44
    op = build_dct_response_matrix(cbm, ds, t_start, t_end, n_grid, return_operator=True)
    gf = GaussianField(
        jnp.zeros((n_grid,), dtype=jnp.float64),
        t_start,
        t_end,
        kernel=m32_spec,
        zero_mode=False,
    )
    model = RadiocarbonModel(cbm, gf, ExactLinearSolver())
    cfg = LinearMarginalConfig(
        offset_mode="hierarchical",
        global_prior_sd=200.0,
        lab_prior_sd=1.5,
        marginalize_field=True,
        include_field_initial_cov=True,
    )

    lnz_ift = log_evidence(model, ds, operator=op)
    lnz_lik = marginal_log_likelihood(model, ds, cfg)
    fit_res = fit_ift_field(model, ds, operator=op)
    assert fit_res.include_field_initial_cov is True
    assert fit_res.prehistory_root_y0 is not None
    # Passing a SpectralKernel directly to prehistory_kernel also works identically
    lnz_direct = log_evidence(model, ds, operator=op, prehistory_kernel=m32_spec)

    x_off, mu_off, sd_off = hierarchical_offset_design(
        ds.series_indices,
        ds.n_series,
        box_activity_ratios=op.box_activity_ratios,
        global_prior_sd=200.0,
        lab_prior_sd=1.5,
    )
    cov_d = (
        op.w_grid @ gf.prior_covariance() @ op.w_grid.T
        + op.init_state_basis @ sig_y0_spec @ op.init_state_basis.T
        + (x_off * (sd_off**2)[None, :]) @ x_off.T
        + jnp.diag(ds.sigma_flat**2)
    )
    r_d = ds.d14c_flat - op.ss_d14c - x_off @ mu_off
    sign_c, logdet_c = jnp.linalg.slogdet(cov_d)
    assert float(sign_c) > 0.0
    dense_lnz = -0.5 * (
        r_d @ jnp.linalg.solve(cov_d, r_d) + logdet_c + ds.n_obs * math.log(2.0 * math.pi)
    )
    np.testing.assert_allclose(lnz_ift, dense_lnz, rtol=1e-11, atol=1e-10)
    np.testing.assert_allclose(fit_res.log_evidence, dense_lnz, rtol=1e-11, atol=1e-10)
    np.testing.assert_allclose(lnz_lik, dense_lnz, rtol=1e-11, atol=1e-10)
    np.testing.assert_allclose(lnz_direct, dense_lnz, rtol=1e-11, atol=1e-10)

    # 3. Initial-state posterior mean & SD match linear_posterior AND dense Schur complement
    t_fine = jnp.linspace(760.0, 790.0, 40, dtype=jnp.float64)
    lp_res = linear_posterior(
        model, ds, cfg, rng_key=jax.random.PRNGKey(71), n_draws=12, t_fine=t_fine
    )
    lp_nofine = linear_posterior(model, ds, cfg)
    assert fit_res.initial_state_mean is not None
    assert fit_res.initial_state_sd is not None
    assert fit_res.initial_state_d14c_mean is not None
    assert fit_res.initial_state_d14c_sd is not None
    assert lp_res.initial_state_mean is not None
    assert lp_res.initial_state_sd is not None
    assert lp_res.initial_state_d14c_mean is not None
    assert lp_res.initial_state_d14c_sd is not None
    assert lp_nofine.initial_state_mean is not None

    # Dense Schur complement for posterior of delta_y0 | d using effective root prior cov:
    # mean = Sigma_{y0} H_y0^T cov_d^{-1} r_d
    # cov  = Sigma_{y0} - Sigma_{y0} H_y0^T cov_d^{-1} H_y0 Sigma_{y0}
    sig_y0_eff = root_y0_spec @ root_y0_spec.T
    cov_d_eff = (
        op.w_grid @ gf.prior_covariance() @ op.w_grid.T
        + op.init_state_basis @ sig_y0_eff @ op.init_state_basis.T
        + (x_off * (sd_off**2)[None, :]) @ x_off.T
        + jnp.diag(ds.sigma_flat**2)
    )
    h_y0 = op.init_state_basis
    cross_cov = sig_y0_eff @ h_y0.T
    dense_dy0_mean = cross_cov @ jnp.linalg.solve(cov_d_eff, r_d)
    dense_dy0_cov = sig_y0_eff - cross_cov @ jnp.linalg.solve(cov_d_eff, cross_cov.T)
    dense_dy0_sd = jnp.sqrt(jnp.maximum(jnp.diag(0.5 * (dense_dy0_cov + dense_dy0_cov.T)), 0.0))

    y_ss = cbm.steady_state_c14(q_std=model.q_std, alpha_unit=model.alpha_unit)
    y_trop = jnp.sum(y_ss[jnp.asarray(cbm.topology.trop_box_indices, dtype=jnp.int32)])
    scale_boxes = 1000.0 / (y_trop * cbm.relative_reservoir_masses())
    dense_d14c_mean = scale_boxes * dense_dy0_mean
    dense_d14c_sd = scale_boxes * dense_dy0_sd

    np.testing.assert_allclose(fit_res.initial_state_mean, dense_dy0_mean, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(fit_res.initial_state_sd, dense_dy0_sd, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(fit_res.initial_state_d14c_mean, dense_d14c_mean, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(fit_res.initial_state_d14c_sd, dense_d14c_sd, rtol=1e-10, atol=1e-10)

    np.testing.assert_allclose(fit_res.initial_state_mean, lp_res.initial_state_mean, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(fit_res.initial_state_sd, lp_res.initial_state_sd, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(fit_res.initial_state_d14c_mean, lp_res.initial_state_d14c_mean, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(fit_res.initial_state_d14c_sd, lp_res.initial_state_d14c_sd, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(lp_nofine.initial_state_mean, lp_res.initial_state_mean, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(fit_res.d14c_pred, lp_res.ring_pred_mean, rtol=1e-10, atol=1e-10)

    # Disabling include_field_initial_cov sets all initial_state_* fields to None
    cfg_noinit = LinearMarginalConfig(
        offset_mode="hierarchical", marginalize_field=True, include_field_initial_cov=False
    )
    lp_noinit = linear_posterior(model, ds, cfg_noinit)
    assert lp_noinit.initial_state_mean is None
    assert lp_noinit.initial_state_sd is None
    assert lp_noinit.initial_state_d14c_mean is None
    assert lp_noinit.initial_state_d14c_sd is None

    # GPProduction also populates LinearPosteriorResult initial_state_*
    m_gp = RadiocarbonModel(
        cbm, SteadyProduction() + GPProduction(m32_tinygp, n_modes=32, pad_years=10.0), ExactLinearSolver()
    )
    lp_gp = linear_posterior(m_gp, ds, LinearMarginalConfig(include_gp_initial_cov=True))
    assert lp_gp.initial_state_mean is not None and lp_gp.initial_state_mean.shape == (cbm.n_boxes,)
    assert lp_gp.initial_state_sd is not None and jnp.all(lp_gp.initial_state_sd >= 0.0)
    assert lp_gp.initial_state_d14c_mean is not None and lp_gp.initial_state_d14c_sd is not None
    lp_gp_noinit = linear_posterior(m_gp, ds, LinearMarginalConfig(include_gp_initial_cov=False))
    assert lp_gp_noinit.initial_state_mean is None

    # laplace_samples propagates initial-state draws matching fit_res moments
    draws = laplace_samples(fit_res, ds, n_samples=1500, rng_key=jax.random.PRNGKey(72))
    assert "initial_state" in draws and "initial_state_d14c" in draws and "d14c_pred" in draws
    assert draws["initial_state"].shape == (1500, cbm.n_boxes)
    np.testing.assert_allclose(
        jnp.mean(draws["d14c_pred"], axis=0), fit_res.d14c_pred, rtol=0.05, atol=0.15
    )
    np.testing.assert_allclose(
        jnp.std(draws["initial_state_d14c"], axis=0),
        fit_res.initial_state_d14c_sd,
        rtol=0.10,
        atol=0.15,
    )


def test_linear_posterior_multi_term_initial_state_anchors() -> None:
    """Multi-term initial-state posteriors return None when anchors differ and match dense Schur when anchors coincide."""
    cbm = load_preset("Brehm21")
    ds = _make_two_hemisphere_dataset(cbm, seed=23)
    a_mat = cbm.matrix_A()
    b_vec = cbm.production_vector()
    alpha = float(RadiocarbonModel(cbm, SteadyProduction()).alpha_unit)

    y_ss = cbm.steady_state_c14(q_std=DEFAULT_Q_STD, alpha_unit=alpha)
    y_trop = jnp.sum(y_ss[jnp.asarray(cbm.topology.trop_box_indices, dtype=jnp.int32)])
    m_rel = cbm.relative_reservoir_masses()
    scale_boxes = 1000.0 / (y_trop * m_rel)

    t0_obs = float(min(jnp.min(s.sub_t_starts) for s in ds.series))
    m32_spec = MaternSpectrum(sigma=0.35, length_yr=10.0 / math.sqrt(3.0), order=2.0)
    se_spec = SquaredExpSpectrum(sigma=0.25, length_yr=6.0)
    m32_tinygp = tk.quasisep.Matern32(scale=10.0, sigma=0.35)

    _, root_m32 = stationary_cbm_spectral_cov(a_mat, b_vec, m32_spec, alpha_unit=alpha)
    _, root_se = stationary_cbm_spectral_cov(a_mat, b_vec, se_spec, alpha_unit=alpha)
    sig_m32 = root_m32 @ root_m32.T
    sig_se = root_se @ root_se.T

    n_grid = 36
    t_end = 792.0
    gf_early = GaussianField(
        jnp.zeros((n_grid,), dtype=jnp.float64),
        755.0,
        t_end,
        kernel=m32_spec,
        zero_mode=False,
    )
    gf_earlier = GaussianField(
        jnp.zeros((n_grid,), dtype=jnp.float64),
        750.0,
        t_end,
        kernel=se_spec,
        zero_mode=False,
    )
    gf_same = GaussianField(
        jnp.zeros((n_grid,), dtype=jnp.float64),
        755.0,
        t_end,
        kernel=se_spec,
        zero_mode=False,
    )
    gf_at_t0 = GaussianField(
        jnp.zeros((n_grid,), dtype=jnp.float64),
        t0_obs,
        t_end,
        kernel=se_spec,
        zero_mode=False,
    )
    gp_term = GPProduction(m32_tinygp, n_modes=28, pad_years=10.0)

    cfg = LinearMarginalConfig(
        offset_mode="hierarchical",
        global_prior_sd=200.0,
        lab_prior_sd=1.5,
        marginalize_gp=True,
        marginalize_field=True,
        include_gp_initial_cov=True,
        include_field_initial_cov=True,
    )

    op_755 = build_dct_response_matrix(cbm, ds, 755.0, t_end, n_grid, return_operator=True)
    op_750 = build_dct_response_matrix(cbm, ds, 750.0, t_end, n_grid, return_operator=True)
    op_t0 = build_dct_response_matrix(cbm, ds, t0_obs, t_end, n_grid, return_operator=True)

    # Independent offset and GP projection pieces for dense MVN / Schur checks
    x_off, mu_off, sd_off = hierarchical_offset_design(
        ds.series_indices,
        ds.n_series,
        box_activity_ratios=op_755.box_activity_ratios,
        global_prior_sd=200.0,
        lab_prior_sd=1.5,
    )
    cov_off = (x_off * (sd_off**2)[None, :]) @ x_off.T
    cov_noise = jnp.diag(ds.sigma_flat**2)
    r_d = ds.d14c_flat - op_755.ss_d14c - x_off @ mu_off

    # Build padded observation windows for independent project_causal_gp reference
    m_max = max(s.sub_t_starts.shape[1] for s in ds.series)
    ts_parts, te_parts, w_parts, box_parts = [], [], [], []
    for s in ds.series:
        n_s, m_s = s.sub_t_starts.shape
        w_s = jnp.broadcast_to(jnp.asarray(s.seasonal_weights.weights, dtype=jnp.float64), (n_s, m_s))
        if m_s < m_max:
            pad_k = m_max - m_s
            ts_parts.append(jnp.concatenate([s.sub_t_starts, jnp.repeat(s.sub_t_starts[:, -1:], pad_k, axis=1)], axis=1))
            te_parts.append(jnp.concatenate([s.sub_t_ends, jnp.repeat(s.sub_t_ends[:, -1:], pad_k, axis=1)], axis=1))
            w_parts.append(jnp.concatenate([w_s, jnp.zeros((n_s, pad_k), dtype=jnp.float64)], axis=1))
        else:
            ts_parts.append(s.sub_t_starts)
            te_parts.append(s.sub_t_ends)
            w_parts.append(w_s)
        box_idx = cbm.box_index("Troposphere", hemisphere=s.hemisphere)
        box_parts.extend([box_idx] * n_s)
    box_arr = jnp.asarray(box_parts, dtype=jnp.int32)
    obs_scale_flat = 1000.0 / (y_trop * m_rel[box_arr])
    proj_gp = project_causal_gp(
        A=a_mat,
        b=b_vec,
        t_starts=jnp.concatenate(ts_parts, axis=0),
        t_ends=jnp.concatenate(te_parts, axis=0),
        sub_weights=jnp.concatenate(w_parts, axis=0),
        pad=gp_term.pad_years,
        n_modes=gp_term.n_modes,
        kernel=gp_term.kernel,
        alpha_unit=alpha,
        box_indices=box_arr,
        obs_scale=obs_scale_flat,
        include_initial_cov=True,
        cbm=cbm,
    )
    assert proj_gp.spectral_weights is not None and proj_gp.anchor_cov_root is not None
    cov_gp_modes = (
        proj_gp.causal_basis_design * proj_gp.spectral_weights[None, :]
    ) @ proj_gp.causal_basis_design.T
    sig_gp_anc = proj_gp.anchor_cov_root @ proj_gp.anchor_cov_root.T
    cov_gp_init = proj_gp.hom_design_anchor @ sig_gp_anc @ proj_gp.hom_design_anchor.T

    # 1. Differing anchors: GPProduction (t_anchor = t0_obs) + GaussianField (t_start = 755.0)
    m_gp_gf_diff = RadiocarbonModel(cbm, SteadyProduction() + gp_term + gf_early, ExactLinearSolver())
    lp_gp_gf_diff = linear_posterior(m_gp_gf_diff, ds, cfg)
    assert lp_gp_gf_diff.initial_state_mean is None
    assert lp_gp_gf_diff.initial_state_sd is None
    assert lp_gp_gf_diff.initial_state_d14c_mean is None
    assert lp_gp_gf_diff.initial_state_d14c_sd is None
    cov_sig_diff = (
        cov_gp_modes
        + cov_gp_init
        + op_755.w_grid @ gf_early.prior_covariance() @ op_755.w_grid.T
        + op_755.init_state_basis @ sig_m32 @ op_755.init_state_basis.T
        + cov_off
    )
    dense_pred_diff = op_755.ss_d14c + x_off @ mu_off + cov_sig_diff @ jnp.linalg.solve(cov_sig_diff + cov_noise, r_d)
    np.testing.assert_allclose(lp_gp_gf_diff.ring_pred_mean, dense_pred_diff, rtol=1e-10, atol=1e-10)

    # 2. Differing anchors: two GaussianFields (t_start = 750.0 vs 755.0)
    m_two_gf_diff = RadiocarbonModel(cbm, gf_earlier + gf_early, ExactLinearSolver())
    lp_two_gf_diff = linear_posterior(m_two_gf_diff, ds, cfg)
    assert lp_two_gf_diff.initial_state_mean is None
    assert lp_two_gf_diff.initial_state_sd is None
    assert lp_two_gf_diff.initial_state_d14c_mean is None
    assert lp_two_gf_diff.initial_state_d14c_sd is None
    cov_sig_two_diff = (
        op_750.w_grid @ gf_earlier.prior_covariance() @ op_750.w_grid.T
        + op_750.init_state_basis @ sig_se @ op_750.init_state_basis.T
        + op_755.w_grid @ gf_early.prior_covariance() @ op_755.w_grid.T
        + op_755.init_state_basis @ sig_m32 @ op_755.init_state_basis.T
        + cov_off
    )
    dense_pred_two_diff = op_755.ss_d14c + x_off @ mu_off + cov_sig_two_diff @ jnp.linalg.solve(cov_sig_two_diff + cov_noise, r_d)
    np.testing.assert_allclose(lp_two_gf_diff.ring_pred_mean, dense_pred_two_diff, rtol=1e-10, atol=1e-10)

    # 3. Matching anchors: two GaussianFields sharing t_start = 755.0 -> matches dense Schur sum
    m_two_gf_same = RadiocarbonModel(cbm, gf_early + gf_same, ExactLinearSolver())
    lp_two_gf_same = linear_posterior(m_two_gf_same, ds, cfg)
    assert lp_two_gf_same.initial_state_mean is not None
    assert lp_two_gf_same.initial_state_sd is not None
    assert lp_two_gf_same.initial_state_d14c_mean is not None
    assert lp_two_gf_same.initial_state_d14c_sd is not None
    sig_tot_gf = sig_m32 + sig_se
    cov_d_same = (
        op_755.w_grid @ (gf_early.prior_covariance() + gf_same.prior_covariance()) @ op_755.w_grid.T
        + op_755.init_state_basis @ sig_tot_gf @ op_755.init_state_basis.T
        + cov_off
        + cov_noise
    )
    cross_gf = sig_tot_gf @ op_755.init_state_basis.T
    dense_mean_gf = cross_gf @ jnp.linalg.solve(cov_d_same, r_d)
    dense_cov_gf = sig_tot_gf - cross_gf @ jnp.linalg.solve(cov_d_same, cross_gf.T)
    dense_sd_gf = jnp.sqrt(jnp.maximum(jnp.diag(0.5 * (dense_cov_gf + dense_cov_gf.T)), 0.0))
    np.testing.assert_allclose(lp_two_gf_same.initial_state_mean, dense_mean_gf, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(lp_two_gf_same.initial_state_sd, dense_sd_gf, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(lp_two_gf_same.initial_state_d14c_mean, scale_boxes * dense_mean_gf, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(lp_two_gf_same.initial_state_d14c_sd, scale_boxes * dense_sd_gf, rtol=1e-8, atol=1e-9)

    # Independent full GP initial-state map R_gp(t0) = [Y_SS(t0) sqrt(S_k), anchor_cov_root]
    dt_gp_anc = float(proj_gp.t_anchor - proj_gp.a)
    a_np = np.asarray(a_mat, dtype=np.float64)
    b_eff_np = alpha * np.asarray(b_vec, dtype=np.float64)
    eye_np = np.eye(cbm.n_boxes, dtype=np.complex128)
    om_gp_np = np.asarray(proj_gp.omegas, dtype=np.float64)
    u_cols, v_cols = [], []
    for w_j in om_gp_np:
        r_j = np.linalg.solve(1j * w_j * eye_np - a_np, b_eff_np)
        u_cols.append(np.real(r_j))
        v_cols.append(np.imag(r_j))
    u_gp_np = np.column_stack(u_cols)
    v_gp_np = np.column_stack(v_cols)
    expm_dt_np = np.asarray(jsl.expm(a_mat * dt_gp_anc), dtype=np.float64)
    y_ss_gp_t0 = (
        v_gp_np * np.cos(om_gp_np * dt_gp_anc)[None, :]
        + u_gp_np * np.sin(om_gp_np * dt_gp_anc)[None, :]
        - expm_dt_np @ v_gp_np
    ) / math.sqrt(float(proj_gp.half_length))
    r_gp_full = jnp.concatenate(
        [
            jnp.asarray(y_ss_gp_t0, dtype=jnp.float64)
            * jnp.sqrt(jnp.maximum(proj_gp.spectral_weights, 0.0))[None, :],
            proj_gp.anchor_cov_root,
        ],
        axis=1,
    )
    sig_gp_full = r_gp_full @ r_gp_full.T
    assert proj_gp.whitened_design is not None

    # 4. Matching anchors: GPProduction + GaussianField with gf.t_start == t0_obs -> matches dense Schur sum
    m_gp_gf_same = RadiocarbonModel(cbm, SteadyProduction() + gp_term + gf_at_t0, ExactLinearSolver())
    lp_gp_gf_same = linear_posterior(m_gp_gf_same, ds, cfg)
    assert lp_gp_gf_same.initial_state_mean is not None
    assert lp_gp_gf_same.initial_state_sd is not None
    sig_tot_mix = sig_gp_full + sig_se
    cov_d_mix = (
        cov_gp_modes
        + cov_gp_init
        + op_t0.w_grid @ gf_at_t0.prior_covariance() @ op_t0.w_grid.T
        + op_t0.init_state_basis @ sig_se @ op_t0.init_state_basis.T
        + cov_off
        + cov_noise
    )
    cross_mix = r_gp_full @ proj_gp.whitened_design.T + sig_se @ op_t0.init_state_basis.T
    dense_mean_mix = cross_mix @ jnp.linalg.solve(cov_d_mix, r_d)
    dense_cov_mix = sig_tot_mix - cross_mix @ jnp.linalg.solve(cov_d_mix, cross_mix.T)
    dense_sd_mix = jnp.sqrt(jnp.maximum(jnp.diag(0.5 * (dense_cov_mix + dense_cov_mix.T)), 0.0))
    np.testing.assert_allclose(lp_gp_gf_same.initial_state_mean, dense_mean_mix, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(lp_gp_gf_same.initial_state_sd, dense_sd_mix, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(lp_gp_gf_same.initial_state_d14c_mean, scale_boxes * dense_mean_mix, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(lp_gp_gf_same.initial_state_d14c_sd, scale_boxes * dense_sd_mix, rtol=1e-8, atol=1e-9)

    # 5. Single-term GPProduction initial-state recovery against dense Schur complement
    m_gp_only = RadiocarbonModel(cbm, SteadyProduction() + gp_term, ExactLinearSolver())
    lp_gp_only = linear_posterior(m_gp_only, ds, cfg)
    assert lp_gp_only.initial_state_mean is not None
    assert lp_gp_only.initial_state_sd is not None
    cov_d_gp = cov_gp_modes + cov_gp_init + cov_off + cov_noise
    cross_gp = r_gp_full @ proj_gp.whitened_design.T
    dense_mean_gp = cross_gp @ jnp.linalg.solve(cov_d_gp, r_d)
    dense_cov_gp = sig_gp_full - cross_gp @ jnp.linalg.solve(cov_d_gp, cross_gp.T)
    dense_sd_gp = jnp.sqrt(jnp.maximum(jnp.diag(0.5 * (dense_cov_gp + dense_cov_gp.T)), 0.0))
    np.testing.assert_allclose(lp_gp_only.initial_state_mean, dense_mean_gp, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(lp_gp_only.initial_state_sd, dense_sd_gp, rtol=1e-8, atol=1e-9)

    # 6. Mixed model with only one initial-state block enabled -> single anchor, returns exact field initial state
    cfg_gp_noinit = LinearMarginalConfig(
        offset_mode="hierarchical",
        global_prior_sd=200.0,
        lab_prior_sd=1.5,
        marginalize_gp=True,
        marginalize_field=True,
        include_gp_initial_cov=False,
        include_field_initial_cov=True,
    )
    lp_one_active = linear_posterior(m_gp_gf_diff, ds, cfg_gp_noinit)
    assert lp_one_active.initial_state_mean is not None
    assert lp_one_active.initial_state_sd is not None
    cov_d_one = (
        cov_gp_modes
        + op_755.w_grid @ gf_early.prior_covariance() @ op_755.w_grid.T
        + op_755.init_state_basis @ sig_m32 @ op_755.init_state_basis.T
        + cov_off
        + cov_noise
    )
    cross_one = sig_m32 @ op_755.init_state_basis.T
    dense_mean_one = cross_one @ jnp.linalg.solve(cov_d_one, r_d)
    dense_cov_one = sig_m32 - cross_one @ jnp.linalg.solve(cov_d_one, cross_one.T)
    dense_sd_one = jnp.sqrt(jnp.maximum(jnp.diag(0.5 * (dense_cov_one + dense_cov_one.T)), 0.0))
    np.testing.assert_allclose(lp_one_active.initial_state_mean, dense_mean_one, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(lp_one_active.initial_state_sd, dense_sd_one, rtol=1e-8, atol=1e-9)

    # 7. Direct kernel test of _reconstruct_initial_state_posterior (empty, matching, differing anchors)
    n_b = cbm.n_boxes
    dummy_m = RadiocarbonModel(cbm, SteadyProduction(), ExactLinearSolver())
    c_mean = jnp.arange(1, 2 * n_b + 1, dtype=jnp.float64) * 0.01
    c_cov = jnp.eye(2 * n_b, dtype=jnp.float64) * 0.04
    assert _reconstruct_initial_state_posterior(dummy_m, [], c_mean, c_cov) == (
        None,
        None,
        None,
        None,
    )
    blk_a = (slice(0, n_b), root_m32, jnp.asarray(755.0, dtype=jnp.float64))
    blk_b_same = (slice(n_b, 2 * n_b), root_se, jnp.asarray(755.0, dtype=jnp.float64))
    blk_b_diff = (slice(n_b, 2 * n_b), root_se, jnp.asarray(760.0, dtype=jnp.float64))
    assert _reconstruct_initial_state_posterior(
        dummy_m, [blk_a, blk_b_diff], c_mean, c_cov
    ) == (None, None, None, None)
    m_same, sd_same, dm_same, dsd_same = _reconstruct_initial_state_posterior(
        dummy_m, [blk_a, blk_b_same], c_mean, c_cov
    )
    assert m_same is not None and sd_same is not None and dm_same is not None and dsd_same is not None
    expected_m = root_m32 @ c_mean[:n_b] + root_se @ c_mean[n_b:]
    expected_cov = 0.04 * (sig_m32 + sig_se)
    expected_sd = jnp.sqrt(jnp.maximum(jnp.diag(expected_cov), 0.0))
    np.testing.assert_allclose(m_same, expected_m, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sd_same, expected_sd, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dm_same, scale_boxes * expected_m, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dsd_same, scale_boxes * expected_sd, rtol=1e-12, atol=1e-12)


def test_stationary_cbm_spectral_cov_high_q_ou_correlated_field_and_t_fine_prestart() -> None:
    """Verify high-Q SHO, short-ell OU, CorrelatedFieldSpectrum band alignment, and t_fine < t_start masking."""
    import scipy.linalg as sla

    from ticktack.fields import _spectral_part_log_raw

    cbm = load_preset("Guttler15")
    a_mat = np.asarray(cbm.matrix_A(), dtype=np.float64)
    b_vec = np.asarray(cbm.production_vector(), dtype=np.float64)
    alpha = float(RadiocarbonModel(cbm, SteadyProduction()).alpha_unit)
    b_eff = alpha * b_vec
    n_b = cbm.n_boxes

    # 1. QuasiPeriodicSpectrum vs independent scipy solve_continuous_lyapunov (SHO state-space) at Q=3.5..40
    period = 11.0
    w0 = 2.0 * math.pi / period
    sig = 0.2
    for q_val in (3.5, 10.0, 20.0, 40.0):
        f_aug = np.zeros((n_b + 2, n_b + 2), dtype=np.float64)
        f_aug[:n_b, :n_b] = a_mat
        f_aug[:n_b, n_b] = b_eff
        f_aug[n_b, n_b + 1] = 1.0
        f_aug[n_b + 1, n_b] = -(w0**2)
        f_aug[n_b + 1, n_b + 1] = -w0 / q_val
        q_diff = np.zeros((n_b + 2, n_b + 2), dtype=np.float64)
        q_diff[-1, -1] = 2.0 * (sig**2) * (w0**3) / q_val
        p_lyap = sla.solve_continuous_lyapunov(f_aug, -q_diff)[:n_b, :n_b]

        kern_qp = QuasiPeriodicSpectrum(sigma=sig, period_yr=period, quality=q_val)
        sig_spec, _ = stationary_cbm_spectral_cov(a_mat, b_vec, kern_qp, alpha_unit=alpha)
        rel_fro = np.linalg.norm(np.asarray(sig_spec) - p_lyap) / np.linalg.norm(p_lyap)
        assert rel_fro < 1e-8, f"QuasiPeriodic Q={q_val} relFro error {rel_fro:.3e} >= 1e-8"

    # 2. Short-ell OU (Matern order=1) vs independent scipy solve_continuous_lyapunov
    for ell in (0.3, 5.0):
        f_ou = np.zeros((n_b + 1, n_b + 1), dtype=np.float64)
        f_ou[:n_b, :n_b] = a_mat
        f_ou[:n_b, n_b] = b_eff
        f_ou[n_b, n_b] = -1.0 / ell
        q_ou = np.zeros((n_b + 1, n_b + 1), dtype=np.float64)
        q_ou[-1, -1] = 2.0 * (sig**2) / ell
        p_ou = sla.solve_continuous_lyapunov(f_ou, -q_ou)[:n_b, :n_b]

        kern_ou = MaternSpectrum(sigma=sig, length_yr=ell, order=1.0)
        sig_ou, _ = stationary_cbm_spectral_cov(a_mat, b_vec, kern_ou, alpha_unit=alpha)
        rel_fro = np.linalg.norm(np.asarray(sig_ou) - p_ou) / np.linalg.norm(p_ou)
        assert rel_fro < 1e-4, f"OU ell={ell} relFro error {rel_fro:.3e} >= 1e-4"

    # 3. CorrelatedFieldSpectrum prehistory matches in-window DCT shape and is band-limited for slope > -1
    n_grid, dt = 64, 1.0
    lam_dct = dct_laplacian_eigenvalues(n_grid, dt)
    om_dct = jnp.sqrt(lam_dct[1:])
    bounds = (float(om_dct[0]), float(om_dct[-1]))
    devs = jnp.array([0.6, -0.4, 0.3, -0.2], dtype=jnp.float64)
    cf_kern = CorrelatedFieldSpectrum(sigma=0.3, slope=-0.5, log_deviations=devs, flexibility=0.5)
    in_win_spec = cf_kern.spectrum(n_grid, dt)[1:]
    cont_log = _spectral_part_log_raw(cf_kern, om_dct, omega_bounds=bounds)
    ratio = jnp.exp(cont_log - jnp.max(cont_log)) / (in_win_spec / jnp.max(in_win_spec))
    np.testing.assert_allclose(ratio / jnp.mean(ratio), jnp.ones_like(ratio), rtol=1e-12, atol=1e-12)

    # Independent high-order Gauss-Legendre reference strictly on [w_lo, w_hi] (zero_mode=False)
    gl_x, gl_w = np.polynomial.legendre.leggauss(256)
    xi = 0.5 * (gl_x + 1.0)
    w_xi = 0.5 * gl_w
    w_lo, w_hi = bounds
    ln_span = math.log(w_hi / w_lo)
    om_ref = w_lo * np.exp(ln_span * xi)
    dw_ref = w_xi * om_ref * ln_span
    log_raw_ref = np.asarray(_spectral_part_log_raw(cf_kern, jnp.asarray(om_ref), omega_bounds=bounds))
    raw_ref = np.exp(log_raw_ref - np.max(log_raw_ref))
    qw_ref = (0.3**2) * (dw_ref * raw_ref) / np.sum(dw_ref * raw_ref)
    sig_ref = np.zeros((n_b, n_b), dtype=np.float64)
    eye_n = np.eye(n_b, dtype=np.complex128)
    for w_k, q_k in zip(om_ref, qw_ref):
        r_vec = np.linalg.solve(1j * w_k * eye_n - a_mat, b_eff)
        sig_ref += q_k * np.real(np.outer(r_vec, np.conj(r_vec)))

    sig_cf_128, _ = stationary_cbm_spectral_cov(
        a_mat, b_vec, cf_kern, alpha_unit=alpha, omega_bounds=bounds, n_quad=128
    )
    rel_fro_cf = np.linalg.norm(np.asarray(sig_cf_128) - sig_ref) / np.linalg.norm(sig_ref)
    assert rel_fro_cf < 1e-9, f"CorrelatedField slope=-0.5 relFro error {rel_fro_cf:.3e} >= 1e-9"

    # 4. linear_posterior t_fine < gf.t_start has zero pre-history response (matching op.init_state_basis)
    cbm_b = load_preset("Brehm21")
    ds = _make_two_hemisphere_dataset(cbm_b, seed=29)
    gf = GaussianField(
        jnp.zeros((32,), dtype=jnp.float64),
        760.0,
        792.0,
        kernel=MaternSpectrum(sigma=0.35, length_yr=8.0, order=2.0),
        zero_mode=False,
    )
    m_gf = RadiocarbonModel(cbm_b, gf, ExactLinearSolver())
    cfg_no_off = LinearMarginalConfig(
        offset_mode="none",
        marginalize_field=True,
        include_field_initial_cov=True,
    )
    t_fine_pre = jnp.array([750.0, 759.5, 760.0, 765.0], dtype=jnp.float64)
    lp_pre = linear_posterior(m_gf, ds, cfg_no_off, t_fine=t_fine_pre)
    assert lp_pre.d14c_fine_mean is not None and lp_pre.d14c_fine_std is not None
    ss_nh = float(cbm_b.steady_state_d14c(q_std=DEFAULT_Q_STD)[cbm_b.box_index("Troposphere", hemisphere="north")])
    np.testing.assert_allclose(lp_pre.d14c_fine_mean[:2], ss_nh, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(lp_pre.d14c_fine_std[:2], 0.0, atol=1e-12)
    assert float(lp_pre.d14c_fine_std[2]) > 0.0


def test_audit_gp_init_state_predict_field_fractional_matern_and_cf_band() -> None:
    """Verify GP full initial state at t0, predict_field_d14c initial state, fractional Matern, and CF [w_1, w_{N-1}] band."""
    import scipy.integrate as spi
    import scipy.special as sps

    from ticktack.fields import (
        _field_omega_bounds,
        _spectral_kernel_quad_weights,
        _spectral_part_log_raw,
        _spectral_quad_grid,
    )
    from ticktack.linear import _solve_complex_resolvents

    cbm = load_preset("Guttler15")
    a_mat = np.asarray(cbm.matrix_A(), dtype=np.float64)
    b_vec = np.asarray(cbm.production_vector(), dtype=np.float64)
    alpha = float(RadiocarbonModel(cbm, SteadyProduction()).alpha_unit)
    b_eff = alpha * b_vec
    n_b = cbm.n_boxes
    trop_idx = cbm.box_index("Troposphere")
    ss_trop = float(cbm.steady_state_d14c(q_std=DEFAULT_Q_STD)[trop_idx])

    years = np.arange(770.0, 785.0, 1.0)
    d14c_obs = ss_trop + 8.0 * np.exp(-(years - 770.0) / 6.0)
    s_decay = TreeRingSeries(years, d14c_obs, np.ones_like(years), hemisphere="north", name="Decay")
    ds_decay = Dataset([s_decay])
    t0_obs = float(jnp.min(s_decay.sub_t_starts))

    # 1. Single GPProduction(pad_years=15): initial_state_d14c[trop] matches d14c_fine at t_fine=[t0_obs]
    gp = GPProduction(0.25 * tk.quasisep.Matern32(scale=5.0), n_modes=32, pad_years=15.0)
    m_gp = RadiocarbonModel(cbm, SteadyProduction() + gp, ExactLinearSolver())
    cfg_gp = LinearMarginalConfig(
        offset_mode="none",
        marginalize_gp=True,
        include_gp_initial_cov=True,
    )
    lp_gp = linear_posterior(m_gp, ds_decay, cfg_gp, t_fine=jnp.array([t0_obs], dtype=jnp.float64))
    assert lp_gp.initial_state_d14c_mean is not None and lp_gp.initial_state_d14c_sd is not None
    assert lp_gp.d14c_fine_mean is not None and lp_gp.d14c_fine_std is not None
    np.testing.assert_allclose(
        lp_gp.initial_state_d14c_mean[trop_idx],
        lp_gp.d14c_fine_mean[0] - ss_trop,
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        lp_gp.initial_state_d14c_sd[trop_idx],
        lp_gp.d14c_fine_std[0],
        rtol=1e-11,
        atol=1e-11,
    )

    # 2. predict_field_d14c includes fit_res.initial_state_mean on operator=, validation ds, and t_fine=
    gf = GaussianField(
        jnp.zeros((24,), dtype=jnp.float64),
        770.0,
        786.0,
        kernel=MaternSpectrum(sigma=0.4, length_yr=6.0, order=2.0),
        zero_mode=False,
    )
    m_gf = RadiocarbonModel(cbm, gf, ExactLinearSolver())
    fit_res = fit_ift_field(m_gf, ds_decay, offset_mode="none", include_field_initial_cov=True)
    assert fit_res.initial_state_mean is not None
    pred_with_op = predict_field_d14c(fit_res, ds_decay, operator=fit_res.operator)
    np.testing.assert_allclose(pred_with_op, fit_res.d14c_pred, rtol=1e-12, atol=1e-12)

    s_val = TreeRingSeries(years[:5], d14c_obs[:5], np.ones(5), hemisphere="north", name="Val")
    pred_val = predict_field_d14c(fit_res, Dataset([s_val]))
    np.testing.assert_allclose(pred_val, fit_res.d14c_pred[:5], rtol=1e-11, atol=1e-11)

    t_fine_chk = jnp.array([765.0, 770.0, 772.5, 778.0], dtype=jnp.float64)
    pred_fine = predict_field_d14c(fit_res, t_fine=t_fine_chk)
    cfg_gf = LinearMarginalConfig(
        offset_mode="none",
        marginalize_field=True,
        include_field_initial_cov=True,
    )
    lp_gf = linear_posterior(m_gf, ds_decay, cfg_gf, t_fine=t_fine_chk)
    assert lp_gf.d14c_fine_mean is not None
    np.testing.assert_allclose(pred_fine, lp_gf.d14c_fine_mean, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(pred_fine[0], ss_trop, rtol=1e-12, atol=1e-12)

    # 3. Rough fractional MaternSpectrum(order=0.55) matches scipy.integrate.quad_vec over [0, inf)
    ell_f, p_f, sig_f = 5.0, 0.55, 0.3
    exact_norm = (math.sqrt(math.pi) / (2.0 * ell_f)) * math.exp(
        sps.gammaln(p_f - 0.5) - sps.gammaln(p_f)
    )
    eye_n = np.eye(n_b, dtype=np.complex128)

    def _integrand(om: float) -> np.ndarray:
        r_vec = np.linalg.solve(1j * om * eye_n - a_mat, b_eff)
        w_spec = (sig_f**2) * ((1.0 + (ell_f * om) ** 2) ** (-p_f)) / exact_norm
        return w_spec * np.real(np.outer(r_vec, np.conj(r_vec)))

    sig_quad, _ = spi.quad_vec(_integrand, 0.0, np.inf, epsrel=1e-9, epsabs=1e-12, limit=200)
    kern_frac = MaternSpectrum(sigma=sig_f, length_yr=ell_f, order=p_f)
    sig_frac, _ = stationary_cbm_spectral_cov(a_mat, b_vec, kern_frac, alpha_unit=alpha)
    rel_fro_frac = np.linalg.norm(np.asarray(sig_frac) - sig_quad) / np.linalg.norm(sig_quad)
    assert rel_fro_frac < 1e-6, f"Fractional Matern order=0.55 relFro {rel_fro_frac:.3e} >= 1e-6"

    # 4. CorrelatedFieldSpectrum(zero_mode=False) strictly band-limited to [w_1, w_{N-1}]
    n_g, dt_g = 64, 1.0
    bounds = _field_omega_bounds(n_g, dt_g)
    w_lo, w_hi = bounds
    devs = jnp.array([1.0, -1.0, 1.0, -1.0], dtype=jnp.float64)
    for slope_val in (-2.0, 0.5):
        cf_k = CorrelatedFieldSpectrum(
            sigma=0.3, slope=slope_val, log_deviations=devs, flexibility=0.5, zero_mode=False
        )
        gl_x, gl_w = np.polynomial.legendre.leggauss(256)
        xi = 0.5 * (gl_x + 1.0)
        w_xi = 0.5 * gl_w
        ln_span = math.log(w_hi / w_lo)
        om_band = w_lo * np.exp(ln_span * xi)
        dw_band = w_xi * om_band * ln_span
        log_r = np.asarray(_spectral_part_log_raw(cf_k, jnp.asarray(om_band), omega_bounds=bounds))
        raw_r = np.exp(log_r - np.max(log_r))
        qw_r = (0.3**2) * (dw_band * raw_r) / np.sum(dw_band * raw_r)
        sig_band_ref = np.zeros((n_b, n_b), dtype=np.float64)
        for w_k, q_k in zip(om_band, qw_r):
            r_vec = np.linalg.solve(1j * w_k * eye_n - a_mat, b_eff)
            sig_band_ref += q_k * np.real(np.outer(r_vec, np.conj(r_vec)))

        sig_cf, _ = stationary_cbm_spectral_cov(
            a_mat, b_vec, cf_k, alpha_unit=alpha, omega_bounds=bounds
        )
        rel_cf = np.linalg.norm(np.asarray(sig_cf) - sig_band_ref) / np.linalg.norm(sig_band_ref)
        assert rel_cf < 1e-10, f"CF slope={slope_val} relFro {rel_cf:.3e} >= 1e-10"

        # Shared pre_resolvents grid also agrees to < 1e-10
        om_q, dw_q = _spectral_quad_grid(0.5, n_quad=128, omega_bounds=bounds)
        u_r, v_r = _solve_complex_resolvents(jnp.asarray(a_mat), jnp.asarray(b_eff), om_q)
        qw_q = _spectral_kernel_quad_weights(cf_k, om_q, dw_q, omega_bounds=bounds)
        sig_pre = (u_r * qw_q[None, :]) @ u_r.T + (v_r * qw_q[None, :]) @ v_r.T
        rel_pre = np.linalg.norm(np.asarray(sig_pre) - sig_band_ref) / np.linalg.norm(sig_band_ref)
        assert rel_pre < 1e-10, f"CF pre_resolvents slope={slope_val} relFro {rel_pre:.3e} >= 1e-10"

    # Explicit prehistory_kernel=CorrelatedFieldSpectrum matches field's effective_kernel
    cf_k0 = CorrelatedFieldSpectrum(sigma=0.3, slope=-2.0, log_deviations=devs, flexibility=0.5)
    gf_cf = GaussianField(jnp.zeros((24,), dtype=jnp.float64), 770.0, 786.0, kernel=cf_k0, zero_mode=False)
    m_cf = RadiocarbonModel(cbm, gf_cf, ExactLinearSolver())
    op_cf = build_dct_response_matrix(cbm, ds_decay, 770.0, 786.0, 24, return_operator=True)
    lnz_default = log_evidence(m_cf, ds_decay, operator=op_cf, offset_mode="none")
    lnz_explicit = log_evidence(
        m_cf, ds_decay, operator=op_cf, offset_mode="none", prehistory_kernel=cf_k0
    )
    np.testing.assert_allclose(lnz_explicit, lnz_default, rtol=1e-12, atol=1e-12)











