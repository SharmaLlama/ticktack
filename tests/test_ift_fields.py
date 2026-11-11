"""Comprehensive scientific and numerical validation suite for ``ticktack.fields``."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time

import jax
import jax.numpy as jnp
from jax.test_util import check_grads
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import pytest

from ticktack.cbm import CarbonBoxModel
from ticktack.constants import DEFAULT_Q_STD
from ticktack.data import Dataset, TreeRingSeries, load_bundled_event, load_bundled_series
from ticktack.fields import (
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
    dct_laplacian_eigenvalues,
    error_scale,
    evidence_grid,
    field_spectrum,
    fit_ift_field,
    idct2,
    laplace_samples,
    log_evidence,
    optimize_gp_kernel,
    plot_evidence_grid,
    plot_field_draws,
    select_gp_kernel,
)
from ticktack.linear import hierarchical_offset_design
from ticktack.model import RadiocarbonModel
from ticktack.presets import load_preset
from ticktack.production import GriddedProduction, MiyakeSpike
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

    # 1. offset_mode="none": compare against dense N(d; ss_d14c, W_grid Sigma_eta W_grid^T + D)
    lnz_none = log_evidence(model, ds, offset_mode="none")
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

    # 2. offset_mode="hierarchical": compare against dense N(d; ss + X_off mu_off, ...)
    lnz_hier = log_evidence(
        model, ds, offset_mode="hierarchical", global_offset_prior_sd=200.0, lab_offset_prior_sd=1.5
    )
    fit_res = fit_ift_field(
        model, ds, offset_mode="hierarchical", global_offset_prior_sd=200.0, lab_offset_prior_sd=1.5
    )
    np.testing.assert_allclose(fit_res.log_evidence, lnz_hier, rtol=1e-12, atol=1e-11)

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

    # Strict positivity even under extreme negative excitations
    extreme = LogGaussianField(
        jnp.full((n_grid,), -12.0, dtype=jnp.float64),
        t_start,
        t_end,
        sigma=0.8,
        length_yr=4.0,
    )
    q_ext = extreme.evaluate_q_grid()
    assert jnp.all(q_ext > 0.0)
    assert jnp.all(jnp.isfinite(q_ext))

    # Itô variance correction preserves E[Q(t)] == q_mean
    lgf_ito = LogGaussianField(
        jnp.zeros((n_grid,), dtype=jnp.float64),
        t_start,
        t_end,
        sigma=0.45,
        length_yr=4.0,
        q_mean=DEFAULT_Q_STD,
        ito_corrected=True,
    )
    draws_prior = lgf_ito.sample_prior(jax.random.PRNGKey(42), n_samples=6000)
    np.testing.assert_allclose(
        jnp.mean(draws_prior, axis=0),
        jnp.full((n_grid,), DEFAULT_Q_STD),
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
    fit_res = fit_ift_field(mdl, series, offset_mode="none", operator=op)

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


def test_real_775ce_bundled_kernel_selection_under_two_seconds() -> None:
    """5(c): Real bundled 775CE dataset with MiyakeSpike + GaussianField runs in < 2 s."""
    cbm = load_preset("Brehm21")
    ds_775 = load_bundled_event("775CE")
    spike = MiyakeSpike(t_start=774.55, duration=0.25, fluence=6.4, profile="boxcar")

    # Warm JIT caches once so we measure true execution speed alongside full 7-kernel run
    _ = optimize_gp_kernel(
        cbm,
        ds_775,
        kernel="matern32",
        extra_production=spike,
        n_grid=96,
        multi_start=2,
        rng_key=jax.random.PRNGKey(0),
    )
    # Also warm remaining distinct parameter-shape kernels once
    _ = select_gp_kernel(
        cbm,
        ds_775,
        extra_production=spike,
        n_grid=96,
        multi_start=2,
        rng_key=jax.random.PRNGKey(1),
    )

    t0 = time.perf_counter()
    sel = select_gp_kernel(
        cbm,
        ds_775,
        extra_production=spike,
        n_grid=96,
        multi_start=2,
        rng_key=jax.random.PRNGKey(42),
    )
    elapsed = time.perf_counter() - t0

    assert elapsed < 2.0, f"7-kernel 775CE selection took {elapsed:.3f} s (expected < 2.0 s)."
    assert len(sel.rankings) == 7
    np.testing.assert_allclose(jnp.sum(sel.model_weights), 1.0, rtol=1e-12, atol=1e-12)
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
    if not nb_path.exists():
        pytest.skip("Notebook 04_ift_kernel_selection.ipynb not yet generated.")

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
    """log_evidence(..., extra_priors=...) on linear GaussianField must not depend on initial latent."""
    cbm = load_preset("Brehm21")
    s_tas = load_bundled_series("775AD-early-S/Büntgen18_TAS01.csv")
    spike = MiyakeSpike(t_start=774.55, duration=0.25, fluence=6.4, profile="boxcar")

    z_zero = jnp.zeros((48,), dtype=jnp.float64)
    z_rand = 4.5 * jax.random.normal(jax.random.PRNGKey(77), (48,), dtype=jnp.float64)
    m_zero = RadiocarbonModel(
        cbm, spike + GaussianField(z_zero, 760.0, 795.0, sigma=0.25, length_yr=5.0), ExactLinearSolver()
    )
    m_rand = RadiocarbonModel(
        cbm, spike + GaussianField(z_rand, 760.0, 795.0, sigma=0.25, length_yr=5.0), ExactLinearSolver()
    )
    fit = fit_ift_field(m_zero, s_tas)
    extra_p = {"production.spike.fluence": dist.Normal(6.4, 2.0)}

    lnz_zero = log_evidence(m_zero, s_tas, extra_priors=extra_p)
    lnz_rand = log_evidence(m_rand, s_tas, extra_priors=extra_p)
    lnz_fit = log_evidence(fit.model, s_tas, extra_priors=extra_p)
    np.testing.assert_allclose(lnz_rand, lnz_zero, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(lnz_fit, lnz_zero, rtol=1e-12, atol=1e-12)


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


