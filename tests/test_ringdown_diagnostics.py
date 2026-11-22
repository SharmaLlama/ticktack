"""Tests for post-spike GP ringdown diagnostics and comparison plots."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

import ticktack as tt
from ticktack.plotting import (
    RingdownDiagnosticResult,
    _build_ordered_z_grid,
    plot_ringdown_comparison,
    ringdown_diagnostic,
)

matplotlib.use("Agg")


def _make_synthetic_ringdown_problem():
    """Build a small, fast NH+SH synthetic ringdown test problem with a data gap."""
    years_full = np.arange(760, 801, dtype=np.float64)
    # Introduce a deliberate 3-year data gap in one SH series to test NaN masking.
    years_gap = np.concatenate([years_full[:18], years_full[21:]])
    sigma_val = 0.9

    cbm0 = tt.load_preset("Brehm21", kind="balanced")
    # Build a deterministic 3-mode SubspaceCBM perturbation from orthogonal unit directions.
    p_free = int(np.asarray(cbm0.log_rates_free).size)
    rng = np.random.default_rng(1234)
    basis_raw = rng.normal(size=(p_free, 3))
    q_mat, _ = np.linalg.qr(basis_raw)
    basis_3 = jnp.asarray(0.35 * q_mat, dtype=jnp.float64)

    alpha_true = jnp.array([1.8, -1.6, 1.4], dtype=jnp.float64)
    cbm_sub_true = tt.SubspaceCBM(
        topology=cbm0.topology,
        base_log_rates_free=cbm0.log_rates_free,
        basis=basis_3,
        base_log_total_mass=float(cbm0.log_total_mass),
        alpha=alpha_true,
        prod_weights=cbm0.prod_weights,
        tie_matrix=cbm0.tie_matrix_rows,
        decay_rate=cbm0.decay_rate,
    )

    kernel = tt.QuasiPeriodicSpectrum(sigma=0.25, period_yr=11.0, quality=4.0)
    n_grid = 56
    z_true = jnp.asarray(rng.normal(size=(n_grid,)), dtype=jnp.float64)
    gf_true = tt.GaussianField(z_true, t_start=745.0, t_end=801.0, kernel=kernel)
    spike_true = tt.MiyakeSpike(
        t_start=774.25, duration=0.85, fluence=6.4, profile="boxcar"
    )
    prod_true = tt.SteadyProduction() + gf_true + spike_true
    model_true = tt.RadiocarbonModel(cbm_sub_true, prod_true)

    # Interleave NH and SH series deliberately to test NH-top / SH-bottom ordering.
    dummy_series = [
        tt.TreeRingSeries(
            years=years_full,
            d14c=np.zeros_like(years_full),
            sigma_d14c=np.full_like(years_full, sigma_val),
            hemisphere="north",
            seasonal_weights="april-september",
            name="NH_1",
        ),
        tt.TreeRingSeries(
            years=years_gap,
            d14c=np.zeros_like(years_gap),
            sigma_d14c=np.full_like(years_gap, sigma_val),
            hemisphere="south",
            year_convention="schulman_start",
            seasonal_weights="october-march",
            name="SH_1",
        ),
        tt.TreeRingSeries(
            years=years_full,
            d14c=np.zeros_like(years_full),
            sigma_d14c=np.full_like(years_full, sigma_val),
            hemisphere="north",
            seasonal_weights="april-september",
            name="NH_2",
        ),
        tt.TreeRingSeries(
            years=years_full,
            d14c=np.zeros_like(years_full),
            sigma_d14c=np.full_like(years_full, sigma_val),
            hemisphere="south",
            year_convention="schulman_start",
            seasonal_weights="october-march",
            name="SH_2",
        ),
    ]
    ds_dummy = tt.Dataset(dummy_series, name="synthetic_ringdown")
    pred_parts = model_true.predict_d14c(ds_dummy, concat=False)
    noise_rng = np.random.default_rng(42)
    obs_series = [
        tt.TreeRingSeries(
            years=s.years_ce,
            d14c=np.asarray(p, dtype=np.float64)
            + noise_rng.normal(0.0, sigma_val, size=s.n_obs),
            sigma_d14c=np.full(s.n_obs, sigma_val, dtype=np.float64),
            hemisphere=s.hemisphere,
            year_convention=s.year_convention,
            seasonal_weights=s.seasonal_weights,
            name=s.name,
        )
        for s, p in zip(dummy_series, pred_parts)
    ]
    dataset = tt.Dataset(obs_series, name="synthetic_ringdown")

    gf_zero = tt.GaussianField(
        jnp.zeros((n_grid,), dtype=jnp.float64),
        t_start=745.0,
        t_end=801.0,
        kernel=kernel,
    )
    model_m0 = tt.RadiocarbonModel(cbm0, tt.SteadyProduction() + gf_zero + spike_true)
    model_sub = tt.RadiocarbonModel(
        cbm_sub_true, tt.SteadyProduction() + gf_zero + spike_true
    )

    t_fine = np.linspace(760.0, 800.0, 161, dtype=np.float64)
    q_true_fine = np.asarray(
        (tt.SteadyProduction() + gf_true)(jnp.asarray(t_fine), q_std=model_true.q_std),
        dtype=np.float64,
    )
    cfg = tt.LinearMarginalConfig(marginalize_field=True, offset_mode="hierarchical")
    return model_m0, model_sub, dataset, cfg, t_fine, q_true_fine, spike_true


@pytest.fixture(scope="module")
def synthetic_problem():
    return _make_synthetic_ringdown_problem()


def test_ringdown_diagnostic_pytree_spike_subtraction_and_metrics(synthetic_problem):
    model_m0, model_sub, dataset, cfg, t_fine, q_true_fine, spike_true = (
        synthetic_problem
    )

    diag = ringdown_diagnostic(
        model_m0,
        model_sub,
        dataset,
        cfg,
        pre_window=(760.0, 773.0),
        post_window=(775.0, 792.0),
        t_fine=t_fine,
        n_draws=24,
        rng_key=jax.random.PRNGKey(7),
        q_true_fine=q_true_fine,
    )

    # 1. Frozen dataclass, JAX PyTree & top-level export checks
    assert tt.RingdownDiagnosticResult is RingdownDiagnosticResult
    assert tt.ringdown_diagnostic is ringdown_diagnostic
    assert tt.plot_ringdown_comparison is plot_ringdown_comparison
    assert isinstance(diag, RingdownDiagnosticResult)
    assert dataclasses.is_dataclass(diag)
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
        diag.n_obs = 999  # type: ignore[misc]
    leaves = jax.tree_util.tree_leaves(diag)
    assert len(leaves) >= 10

    # 2. Shapes and clean parametric spike subtraction
    assert diag.t_fine.shape == t_fine.shape
    assert diag.q_field_m0_mean.shape == t_fine.shape
    assert diag.q_field_m0_std.shape == t_fine.shape
    assert diag.q_field_subspace_mean.shape == t_fine.shape
    assert diag.q_field_subspace_std.shape == t_fine.shape
    assert diag.q_field_m0_draws is not None
    assert diag.q_field_subspace_draws is not None
    assert diag.q_field_m0_draws.shape == (24, t_fine.size)
    assert diag.q_field_subspace_draws.shape == (24, t_fine.size)
    assert diag.m_modes == 3
    assert diag.n_obs == dataset.n_obs

    lp_sub = tt.linear_posterior(
        model_sub, dataset, cfg, n_draws=0, t_fine=jnp.asarray(t_fine)
    )
    spike_dq = spike_true.delta_q(jnp.asarray(t_fine), q_std=model_sub.q_std)
    expected_q_bg = np.asarray(lp_sub.q_fine_mean - spike_dq, dtype=np.float64)
    np.testing.assert_allclose(
        np.asarray(diag.q_field_subspace_mean), expected_q_bg, atol=1e-12, rtol=1e-12
    )
    # Spike peak height is ~7.5 atoms cm^-2 s^-1; subtracted background stays O(Q_0).
    spike_mask = (t_fine >= 774.25) & (t_fine <= 775.10)
    assert float(np.max(np.abs(np.asarray(diag.q_field_subspace_mean)[spike_mask]))) < 4.0

    # 3. Raw chi2 / N and z-score residuals on unrescaled sigma_flat
    expected_z_m0 = (
        np.asarray(dataset.d14c_flat) - np.asarray(diag.ring_pred_m0)
    ) / np.asarray(dataset.sigma_flat)
    np.testing.assert_allclose(
        np.asarray(diag.z_residuals_m0), expected_z_m0, atol=1e-12
    )
    assert np.isclose(diag.chi2_m0, float(np.sum(expected_z_m0**2)), rtol=1e-12)
    assert np.isclose(diag.chi2_per_n_m0, diag.chi2_m0 / dataset.n_obs, rtol=1e-12)

    # 4. Physical diagnostics: joint SubspaceCBM suppresses misspecification ringdown
    assert diag.chi2_per_n_subspace < diag.chi2_per_n_m0
    assert diag.chi2_per_n_subspace < 1.5
    assert diag.rmse_pre_m0 is not None
    assert diag.rmse_post_m0 is not None
    assert diag.rmse_pre_subspace is not None
    assert diag.rmse_post_subspace is not None
    assert diag.max_abs_err_post_m0 is not None
    assert diag.max_abs_err_post_subspace is not None
    assert diag.rmse_post_subspace < diag.rmse_post_m0
    assert diag.max_abs_err_post_subspace < diag.max_abs_err_post_m0
    assert diag.rms_pre_m0 > 0.0
    assert diag.rms_post_m0 > 0.0
    assert np.isclose(
        diag.excitation_ratio_m0, diag.rms_post_m0 / diag.rms_pre_m0, rtol=1e-12
    )
    assert np.isclose(
        diag.excitation_ratio_subspace,
        diag.rms_post_subspace / diag.rms_pre_subspace,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(diag.q_field_diff),
        np.asarray(diag.q_field_m0_mean - diag.q_field_subspace_mean),
        atol=1e-12,
    )
    summary_md = diag.format_summary()
    assert "Raw $\\chi^2 / N$" in summary_md
    assert "Post-spike truth RMSE" in summary_md


def test_ringdown_diagnostic_defaults_and_ensemble_marginalization(synthetic_problem):
    model_m0, model_sub, dataset, cfg, t_fine, _, _ = synthetic_problem

    # Default t_fine=None, marginal_config_subspace=None, n_draws=0, no q_true_fine
    diag_default = ringdown_diagnostic(
        model_m0,
        model_sub,
        dataset,
        cfg,
        n_draws=0,
    )
    assert diag_default.q_field_m0_draws is None
    assert diag_default.q_field_subspace_draws is None
    assert diag_default.rmse_post_m0 is None
    assert diag_default.t_fine.size >= 201
    assert diag_default.spike_window is not None
    assert np.isclose(diag_default.spike_window[0], 774.25)

    # Ensemble marginalization via Law of Total Variance
    m_pert = model_sub.set("cbm.alpha", model_sub.cbm.alpha + jnp.array([0.2, -0.2, 0.1]))
    diag_ens = ringdown_diagnostic(
        model_m0,
        model_sub,
        dataset,
        cfg,
        t_fine=t_fine,
        n_draws=10,
        rng_key=jax.random.PRNGKey(11),
        subspace_models=[model_sub, m_pert],
    )
    assert diag_ens.q_field_subspace_draws is not None
    assert diag_ens.q_field_subspace_draws.shape == (10, t_fine.size)
    assert np.all(np.isfinite(np.asarray(diag_ens.q_field_subspace_std)))
    assert np.all(np.asarray(diag_ens.q_field_subspace_std) > 0.0)


def test_ordered_z_grid_nh_first_and_nan_gaps(synthetic_problem):
    _, _, dataset, _, _, _, _ = synthetic_problem
    z_flat = np.linspace(-2.0, 2.0, dataset.n_obs)
    grid, ordered_series, n_nh, year_min, year_max = _build_ordered_z_grid(
        dataset, z_flat
    )
    assert n_nh == 2
    assert [s.hemisphere for s in ordered_series] == ["north", "north", "south", "south"]
    assert year_min == 760
    assert year_max == 800
    assert grid.shape == (4, 41)
    # SH_1 (row 2 after NH-first ordering) has 3 missing years (778, 779, 780)
    assert int(np.sum(np.isnan(grid[2]))) == 3
    assert int(np.sum(np.isnan(grid[0]))) == 0


def test_plot_ringdown_comparison_stateless_truth_and_real_modes(synthetic_problem):
    model_m0, model_sub, dataset, cfg, t_fine, q_true_fine, _ = synthetic_problem
    diag = ringdown_diagnostic(
        model_m0,
        model_sub,
        dataset,
        cfg,
        t_fine=t_fine,
        n_draws=0,
        q_true_fine=q_true_fine,
    )
    diag_no_truth = ringdown_diagnostic(
        model_m0,
        model_sub,
        dataset,
        cfg,
        t_fine=t_fine,
        n_draws=0,
    )

    rc_before = dict(plt.rcParams)

    # Synthetic mode (with q_true_fine)
    fig_truth = plot_ringdown_comparison(
        diag,
        dataset,
        spike_window=(774.25, 775.10),
        q_true_fine=q_true_fine,
        title="Synthetic ringdown comparison",
    )
    assert isinstance(fig_truth, Figure)
    # 4 main panels + 1 colourbar axis = 5 axes
    assert len(fig_truth.axes) == 5
    fig_truth.canvas.draw()
    plt.close(fig_truth)

    # Real-data mode (without q_true_fine -> Panel B shows Q_m0 - Q_subspace)
    fig_diff = plot_ringdown_comparison(
        diag_no_truth,
        dataset,
        title="Real-data ringdown comparison",
    )
    assert isinstance(fig_diff, Figure)
    assert len(fig_diff.axes) == 5
    fig_diff.canvas.draw()
    plt.close(fig_diff)

    # Global rcParams must remain untouched
    assert plt.rcParams["font.size"] == rc_before["font.size"]
    assert plt.rcParams["axes.spines.top"] == rc_before["axes.spines.top"]


def test_ringdown_validation_errors(synthetic_problem):
    model_m0, model_sub, dataset, cfg, t_fine, _, _ = synthetic_problem

    with pytest.raises(ValueError, match="pre_window"):
        ringdown_diagnostic(model_m0, model_sub, dataset, cfg, pre_window=(775.0, 770.0))

    with pytest.raises(ValueError, match="post_window"):
        ringdown_diagnostic(model_m0, model_sub, dataset, cfg, post_window=(792.0, 775.0))

    with pytest.raises(ValueError, match="contains no points"):
        ringdown_diagnostic(
            model_m0,
            model_sub,
            dataset,
            cfg,
            pre_window=(700.0, 710.0),
            t_fine=t_fine,
        )

    with pytest.raises(ValueError, match="q_true_fine shape"):
        ringdown_diagnostic(
            model_m0,
            model_sub,
            dataset,
            cfg,
            t_fine=t_fine,
            q_true_fine=np.zeros((5,)),
        )

    diag = ringdown_diagnostic(model_m0, model_sub, dataset, cfg, t_fine=t_fine, n_draws=0)
    with pytest.raises(ValueError, match="vmax"):
        plot_ringdown_comparison(diag, dataset, vmax=-1.0)
