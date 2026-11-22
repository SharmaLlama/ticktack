"""Deterministic and statistical tests for GP-marginalized CBM Fisher spectra,
MAP-centred tangent-linear DCT operators, joint CBM+GP kernel selection, and
two-stage SubspaceCBM+GP sampling with exact importance-sampling reweighting.
"""

from __future__ import annotations

import gc
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tinygp.kernels as tk

import ticktack as tt


def _make_synthetic_nhsh_datasets(
    years: np.ndarray,
    sigma_val: float,
    d14c_by_series: list[np.ndarray] | None = None,
    n_nh: int = 3,
    n_sh: int = 2,
) -> tuple[tt.Dataset, tt.Dataset]:
    """Construct paired NH+SH and NH-only synthetic tree-ring datasets."""
    n_yr = len(years)
    series_list: list[tt.TreeRingSeries] = []
    for i in range(n_nh):
        y = np.zeros(n_yr) if d14c_by_series is None else d14c_by_series[i]
        series_list.append(
            tt.TreeRingSeries(
                years=years,
                d14c=y,
                sigma_d14c=np.full(n_yr, sigma_val),
                hemisphere="north",
                seasonal_weights="april-september",
                name=f"NH_{i + 1}",
            )
        )
    for j in range(n_sh):
        y = np.zeros(n_yr) if d14c_by_series is None else d14c_by_series[n_nh + j]
        series_list.append(
            tt.TreeRingSeries(
                years=years,
                d14c=y,
                sigma_d14c=np.full(n_yr, sigma_val),
                hemisphere="south",
                year_convention="schulman_start",
                seasonal_weights="october-march",
                name=f"SH_{j + 1}",
            )
        )
    return tt.Dataset(series_list, name="sim_NHSH"), tt.Dataset(
        series_list[:n_nh], name="sim_NH"
    )


@pytest.fixture(scope="module")
def synthetic_m3_bundle() -> dict:
    """Module-scoped synthetic SubspaceCBM(m=3) + MiyakeSpike + stationary GP dataset."""
    cbm0 = tt.load_preset("Brehm21", kind="balanced")
    years = np.arange(762, 798, dtype=np.float64)
    n_yr = len(years)
    n_nh, n_sh = 3, 2
    sigma_val = 0.75
    t_grid_start = 747.0
    t_grid_end = 798.5
    n_grid = 44

    ds_tmpl, ds_nh_tmpl = _make_synthetic_nhsh_datasets(
        years, sigma_val, n_nh=n_nh, n_sh=n_sh
    )
    kp_true = tt.QuasiPeriodicSpectrum(sigma=0.26, period_yr=10.8, quality=4.5)
    spike_true = tt.MiyakeSpike(t_start=774.25, duration=0.85, fluence=6.4, profile="boxcar")

    # GP-marginalized Fisher spectra (uninformative per-series offsets) on NH+SH and NH-only
    spec_gp_nhsh_per = tt.cbm_gp_fisher_spectrum(
        cbm0,
        ds_tmpl,
        kp_true,
        spike=spike_true,
        marginal_config=tt.LinearMarginalConfig(offset_mode="per_series"),
        per_series_prior_sd=None,
        prior_sd=0.35,
        n_grid=n_grid,
        pad_years=15.0,
    )
    spec_gp_nh_per = tt.cbm_gp_fisher_spectrum(
        cbm0,
        ds_nh_tmpl,
        kp_true,
        spike=spike_true,
        marginal_config=tt.LinearMarginalConfig(offset_mode="per_series"),
        per_series_prior_sd=None,
        prior_sd=0.35,
        n_grid=n_grid,
        pad_years=15.0,
    )

    # Inject m=3 SubspaceCBM truth + stationary GP + MiyakeSpike + hierarchical offsets
    cbm_sub3_0 = tt.SubspaceCBM.from_fisher(cbm0, spec_gp_nhsh_per, n_modes=3)
    alpha_true = jnp.array([1.5, -1.4, 2.2], dtype=jnp.float64)
    cbm_true = cbm_sub3_0.set("alpha", alpha_true)

    z_true = jax.random.normal(jax.random.PRNGKey(2026), (n_grid,), dtype=jnp.float64)
    gf_true = tt.GaussianField(z_true, t_start=t_grid_start, t_end=t_grid_end, kernel=kp_true)

    delta_global_true = -18.0
    delta_lab_true = 1.0 * jax.random.normal(
        jax.random.PRNGKey(2027), (n_nh + n_sh,), dtype=jnp.float64
    )
    noise_true = sigma_val * jax.random.normal(
        jax.random.PRNGKey(2028), (ds_tmpl.n_obs,), dtype=jnp.float64
    )

    op_true = tt.build_dct_response_operator(
        cbm_true, ds_tmpl, t_start=t_grid_start, t_end=t_grid_end, n_grid=n_grid
    )
    d14c_gf_true = op_true.w_dct @ (jnp.sqrt(gf_true.power_spectrum()) * z_true)
    d14c_sp_true = tt.RadiocarbonModel(
        cbm_true, tt.SteadyProduction() + spike_true
    ).predict_d14c(ds_tmpl, delta_global=delta_global_true, offsets=delta_lab_true)
    d14c_obs = np.asarray(d14c_sp_true + d14c_gf_true + noise_true)

    ds_sim, ds_nh_sim = _make_synthetic_nhsh_datasets(
        years,
        sigma_val,
        [d14c_obs[i * n_yr : (i + 1) * n_yr] for i in range(n_nh + n_sh)],
        n_nh=n_nh,
        n_sh=n_sh,
    )

    return {
        "cbm0": cbm0,
        "cbm_sub3_0": cbm_sub3_0,
        "cbm_true": cbm_true,
        "alpha_true": alpha_true,
        "kp_true": kp_true,
        "gf_true": gf_true,
        "spike_true": spike_true,
        "years": years,
        "n_nh": n_nh,
        "n_sh": n_sh,
        "sigma_val": sigma_val,
        "t_grid_start": t_grid_start,
        "t_grid_end": t_grid_end,
        "n_grid": n_grid,
        "ds_tmpl": ds_tmpl,
        "ds_nh_tmpl": ds_nh_tmpl,
        "ds_sim": ds_sim,
        "ds_nh_sim": ds_nh_sim,
        "spec_gp_nhsh_per": spec_gp_nhsh_per,
        "spec_gp_nh_per": spec_gp_nh_per,
    }


@pytest.fixture(autouse=True)
def _release_python_gc() -> None:
    yield
    gc.collect()


# ---------------------------------------------------------------------------
# 1. cbm_gp_fisher_spectrum: exact Hessian identity & NH vs NH+SH identifiability
# ---------------------------------------------------------------------------


def test_cbm_gp_fisher_matches_marginal_loglike_hessian(synthetic_m3_bundle: dict) -> None:
    """Finite-difference Hessian of marginal_log_likelihood wrt log_rates_free matches Fisher."""
    cbm0 = synthetic_m3_bundle["cbm0"]
    ds_tmpl = synthetic_m3_bundle["ds_tmpl"]
    kp_true = synthetic_m3_bundle["kp_true"]
    spike_true = synthetic_m3_bundle["spike_true"]
    years = synthetic_m3_bundle["years"]
    n_yr = len(years)
    n_nh = synthetic_m3_bundle["n_nh"]
    n_sh = synthetic_m3_bundle["n_sh"]
    sigma_val = synthetic_m3_bundle["sigma_val"]
    t0 = synthetic_m3_bundle["t_grid_start"]
    t1 = synthetic_m3_bundle["t_grid_end"]
    n_grid = synthetic_m3_bundle["n_grid"]

    d14c_noiseless = np.asarray(
        tt.RadiocarbonModel(cbm0, tt.SteadyProduction() + spike_true).predict_d14c(ds_tmpl)
    )
    ds_clean, _ = _make_synthetic_nhsh_datasets(
        years,
        sigma_val,
        [d14c_noiseless[i * n_yr : (i + 1) * n_yr] for i in range(n_nh + n_sh)],
        n_nh=n_nh,
        n_sh=n_sh,
    )
    d14c_ss = np.asarray(tt.RadiocarbonModel(cbm0, tt.SteadyProduction()).predict_d14c(ds_clean))
    ds_ss, _ = _make_synthetic_nhsh_datasets(
        years,
        sigma_val,
        [d14c_ss[i * n_yr : (i + 1) * n_yr] for i in range(n_nh + n_sh)],
        n_nh=n_nh,
        n_sh=n_sh,
    )

    gf_zero = tt.GaussianField(
        jnp.zeros((n_grid,), dtype=jnp.float64), t_start=t0, t_end=t1, kernel=kp_true
    )
    op0 = tt.build_dct_response_operator(cbm0, ds_clean, t_start=t0, t_end=t1, n_grid=n_grid)

    # 1a. Steady-state + GP without spike (offset_mode="none" and "per_series" finite prior)
    lr0 = jnp.asarray(cbm0.log_rates_free, dtype=jnp.float64)
    p_free = int(lr0.shape[0])
    for cfg in (
        tt.LinearMarginalConfig(offset_mode="none", dct_operator=op0),
        tt.LinearMarginalConfig(
            offset_mode="per_series", global_prior_sd=50.0, dct_operator=op0
        ),
    ):
        res_fisher = tt.cbm_gp_fisher_spectrum(
            cbm0,
            ds_ss,
            gf_zero,
            spike=None,
            marginal_config=cfg,
            prior_sd=0.35,
        )
        q_dirs = res_fisher.parameter_loadings[:p_free, :3] / 0.35

        def _nll_dir(
            c_vec: jax.Array,
            _cfg: tt.LinearMarginalConfig = cfg,
            _q: jax.Array = q_dirs,
        ) -> jax.Array:
            cbm_v = eqx.tree_at(lambda m: m.log_rates_free, cbm0, lr0 + _q @ c_vec)
            mod_v = tt.RadiocarbonModel(cbm_v, tt.SteadyProduction() + gf_zero)
            return -tt.marginal_log_likelihood(mod_v, ds_ss, _cfg)

        grad_fn = jax.jit(jax.grad(_nll_dir))
        c0 = jnp.zeros((3,), dtype=jnp.float64)
        eps = 1e-6
        eye3 = jnp.eye(3, dtype=jnp.float64)
        h_fd = jnp.column_stack(
            [
                (grad_fn(c0 + eps * eye3[i]) - grad_fn(c0 - eps * eye3[i])) / (2.0 * eps)
                for i in range(3)
            ]
        )
        h_fd = 0.5 * (h_fd + h_fd.T)
        f_sub = q_dirs.T @ res_fisher.marginal_fisher[:p_free, :p_free] @ q_dirs
        scale_f = max(float(np.max(np.abs(f_sub))), 1.0)
        np.testing.assert_allclose(
            np.asarray(h_fd), np.asarray(f_sub), rtol=1e-5, atol=1e-6 * scale_f
        )

    # 1b. With MiyakeSpike projected via Schur complement
    cfg_sp = tt.LinearMarginalConfig(
        offset_mode="per_series", global_prior_sd=50.0, dct_operator=op0
    )
    res_sp = tt.cbm_gp_fisher_spectrum(
        cbm0,
        ds_clean,
        gf_zero,
        spike=spike_true,
        marginal_config=cfg_sp,
        prior_sd=0.35,
    )
    q_dirs = res_sp.parameter_loadings[:p_free, :3] / 0.35
    sp0 = jnp.array(
        [float(spike_true.t_start), float(spike_true.duration), float(spike_true.fluence)],
        dtype=jnp.float64,
    )

    def _nll_joint(z_vec: jax.Array) -> jax.Array:
        c_v = z_vec[:3]
        s_v = z_vec[3:]
        cbm_v = eqx.tree_at(lambda m: m.log_rates_free, cbm0, lr0 + q_dirs @ c_v)
        sp_v = tt.MiyakeSpike(t_start=s_v[0], duration=s_v[1], fluence=s_v[2], profile="boxcar")
        mod_v = tt.RadiocarbonModel(cbm_v, tt.SteadyProduction() + sp_v + gf_zero)
        return -tt.marginal_log_likelihood(mod_v, ds_clean, cfg_sp)

    grad_joint = jax.jit(jax.grad(_nll_joint))
    z0 = jnp.concatenate([jnp.zeros((3,), dtype=jnp.float64), sp0])
    eps = 1e-6
    eye6 = jnp.eye(6, dtype=jnp.float64)
    h_j = jnp.column_stack(
        [
            (grad_joint(z0 + eps * eye6[i]) - grad_joint(z0 - eps * eye6[i])) / (2.0 * eps)
            for i in range(6)
        ]
    )
    h_j = 0.5 * (h_j + h_j.T)
    h_cc = h_j[:3, :3]
    h_cs = h_j[:3, 3:]
    h_ss = h_j[3:, 3:]
    f_schur_fd = h_cc - h_cs @ jnp.linalg.solve(h_ss, h_cs.T)
    f_schur_api = q_dirs.T @ res_sp.marginal_fisher[:p_free, :p_free] @ q_dirs
    scale_sp = max(float(np.max(np.abs(f_schur_api))), 1.0)
    np.testing.assert_allclose(
        np.asarray(f_schur_fd), np.asarray(f_schur_api), rtol=1e-5, atol=1e-6 * scale_sp
    )


def test_cbm_gp_fisher_nh_collapse_vs_nhsh_rescue_and_inputs(synthetic_m3_bundle: dict) -> None:
    """Confirm NH-only collapse (d_eff < 1.1) vs NH+SH rescue (d_eff > 3.2) and GPProduction input."""
    cbm0 = synthetic_m3_bundle["cbm0"]
    ds_tmpl = synthetic_m3_bundle["ds_tmpl"]
    kp_true = synthetic_m3_bundle["kp_true"]
    spike_true = synthetic_m3_bundle["spike_true"]
    n_grid = synthetic_m3_bundle["n_grid"]

    spec_nh = synthetic_m3_bundle["spec_gp_nh_per"]
    spec_nhsh_per = synthetic_m3_bundle["spec_gp_nhsh_per"]
    spec_nhsh_hier = tt.cbm_gp_fisher_spectrum(
        cbm0,
        ds_tmpl,
        kp_true,
        spike=spike_true,
        prior_sd=0.35,
        n_grid=n_grid,
        pad_years=15.0,
    )

    assert isinstance(spec_nh, tt.FisherSpectrumResult)
    assert float(spec_nh.d_eff) < 1.1
    assert float(spec_nhsh_per.d_eff) > 3.2
    assert float(spec_nhsh_hier.d_eff) > 3.2
    assert float(spec_nhsh_hier.d_eff) >= float(spec_nhsh_per.d_eff)

    # SubspaceCBM.from_fisher compatibility
    cbm_sub = tt.SubspaceCBM.from_fisher(cbm0, spec_nhsh_hier, n_modes=4)
    assert cbm_sub.n_modes == 4

    # Also verify GPProduction input works cleanly and gives finite positive eigenvalues
    gp_prod = tt.GPProduction(tk.quasisep.Matern32(scale=5.0, sigma=0.25), n_modes=24)
    spec_gp_prod = tt.cbm_gp_fisher_spectrum(
        cbm0,
        ds_tmpl,
        gp_prod,
        spike=spike_true,
        prior_sd=0.35,
    )
    assert isinstance(spec_gp_prod, tt.FisherSpectrumResult)
    assert bool(jnp.all(jnp.isfinite(spec_gp_prod.eigenvalues)))
    assert float(spec_gp_prod.d_eff) > 2.5


# ---------------------------------------------------------------------------
# 2. LinearizedDCTOperator: O(eps^2) JVP convergence & posterior-ball accuracy
# ---------------------------------------------------------------------------


def test_linearized_dct_operator_jvp_convergence_and_likelihood(
    synthetic_m3_bundle: dict,
) -> None:
    """Verify O(eps^2) Taylor convergence and exact match at alpha_ref in likelihood & posterior."""
    cbm_sub0 = synthetic_m3_bundle["cbm_sub3_0"]
    ds_sim = synthetic_m3_bundle["ds_sim"]
    kp_true = synthetic_m3_bundle["kp_true"]
    spike_true = synthetic_m3_bundle["spike_true"]
    t0 = synthetic_m3_bundle["t_grid_start"]
    t1 = synthetic_m3_bundle["t_grid_end"]
    n_grid = synthetic_m3_bundle["n_grid"]

    alpha_ref = jnp.array([0.8, -0.6, 0.5], dtype=jnp.float64)
    cbm_ref = cbm_sub0.set("alpha", alpha_ref)
    lin_op = tt.build_linearized_dct_operator(
        cbm_ref, ds_sim, t_start=t0, t_end=t1, n_grid=n_grid
    )
    exact_op = tt.build_dct_response_operator(
        cbm_ref, ds_sim, t_start=t0, t_end=t1, n_grid=n_grid
    )
    assert isinstance(lin_op, tt.LinearizedDCTOperator)
    assert lin_op.dw_dalpha.shape == (3, ds_sim.n_obs, n_grid)
    np.testing.assert_allclose(
        np.asarray(lin_op.evaluate_w_dct(alpha_ref)),
        np.asarray(exact_op.w_dct),
        rtol=1e-12,
        atol=1e-12,
    )

    # Directional O(eps^2) Taylor convergence check
    direction = jnp.array([0.6, -0.5, 0.62], dtype=jnp.float64)
    errs = []
    for eps in (2e-2, 1e-2, 5e-3):
        a_step = alpha_ref + eps * direction
        w_exact = tt.build_dct_response_operator(
            cbm_sub0.set("alpha", a_step), ds_sim, t_start=t0, t_end=t1, n_grid=n_grid
        ).w_dct
        w_tan = lin_op.evaluate_w_dct(a_step)
        errs.append(float(jnp.linalg.norm(w_exact - w_tan)))
    ratio_1 = errs[0] / errs[1]
    ratio_2 = errs[1] / errs[2]
    assert 3.7 < ratio_1 < 4.3, f"Expected O(eps^2) ratio ~4, got {ratio_1}"
    assert 3.8 < ratio_2 < 4.2, f"Expected O(eps^2) ratio ~4, got {ratio_2}"

    # Verify marginal_log_likelihood & linear_posterior match identically at alpha_ref
    gf_zero = tt.GaussianField(jnp.zeros((n_grid,), dtype=jnp.float64), t0, t1, kernel=kp_true)
    mod_ref = tt.RadiocarbonModel(cbm_ref, tt.SteadyProduction() + gf_zero + spike_true)
    cfg_exact = tt.LinearMarginalConfig(marginalize_field=True, offset_mode="hierarchical")
    cfg_tan = eqx.tree_at(
        lambda c: c.dct_operator, cfg_exact, lin_op, is_leaf=lambda x: x is None
    )

    ll_exact = tt.marginal_log_likelihood(mod_ref, ds_sim, cfg_exact)
    ll_tan = tt.marginal_log_likelihood(mod_ref, ds_sim, cfg_tan)
    np.testing.assert_allclose(float(ll_tan), float(ll_exact), rtol=1e-12, atol=1e-11)

    lp_exact = tt.linear_posterior(mod_ref, ds_sim, cfg_exact)
    lp_tan = tt.linear_posterior(mod_ref, ds_sim, cfg_tan)
    np.testing.assert_allclose(
        np.asarray(lp_tan.coeff_mean), np.asarray(lp_exact.coeff_mean), rtol=1e-11, atol=1e-11
    )
    np.testing.assert_allclose(
        np.asarray(lp_tan.ring_pred_mean),
        np.asarray(lp_exact.ring_pred_mean),
        rtol=1e-11,
        atol=1e-11,
    )


# ---------------------------------------------------------------------------
# 3. End-to-end fit_subspace_cbm_gp: recovery, ringdown reduction & joint GP hyperparams
# ---------------------------------------------------------------------------


def test_fit_subspace_cbm_gp_recovery_and_ringdown_reduction(
    synthetic_m3_bundle: dict,
) -> None:
    """Synthetic SubspaceCBM(m=3)+Spike+GP recovery (<2.5σ), IS efficiency >0.95, >2.5x ringdown drop."""
    cbm0 = synthetic_m3_bundle["cbm0"]
    cbm_sub3_0 = synthetic_m3_bundle["cbm_sub3_0"]
    alpha_true = np.asarray(synthetic_m3_bundle["alpha_true"])
    kp_true = synthetic_m3_bundle["kp_true"]
    gf_true = synthetic_m3_bundle["gf_true"]
    spike_true = synthetic_m3_bundle["spike_true"]
    ds_sim = synthetic_m3_bundle["ds_sim"]
    t0 = synthetic_m3_bundle["t_grid_start"]
    t1 = synthetic_m3_bundle["t_grid_end"]
    n_grid = synthetic_m3_bundle["n_grid"]

    gf_zero = tt.GaussianField(jnp.zeros((n_grid,), dtype=jnp.float64), t0, t1, kernel=kp_true)
    spike_init = tt.MiyakeSpike(774.4, 0.5, 6.0, profile="boxcar")
    cfg_gf = tt.LinearMarginalConfig(marginalize_field=True, offset_mode="hierarchical")

    # Baseline fixed-CBM (m = 0) fit
    mod_m0 = tt.RadiocarbonModel(cbm0, tt.SteadyProduction() + gf_zero + spike_init)
    ps_m0 = tt.ParameterSpace(
        mod_m0,
        [
            tt.ParamSpec("production.spike.t_start", bijector="identity", prior_mean=774.5, prior_sd=2.0),
            tt.ParamSpec("production.spike.duration", bijector="exp", prior_mean=0.0, prior_sd=1.5),
            tt.ParamSpec("production.spike.fluence", bijector="exp", prior_mean=math.log(6.0), prior_sd=1.0),
        ],
    )
    map_m0 = tt.fit_map(mod_m0, ds_sim, ps_m0, cfg_gf, method="BFGS", max_steps=300, tol=1e-5)
    t_fine = np.linspace(762.0, 797.0, 71)
    lp_m0 = tt.linear_posterior(map_m0.model, ds_sim, cfg_gf, t_fine=t_fine)
    q_true_fine = float(tt.DEFAULT_Q_STD) + np.asarray(gf_true.delta_q(t_fine))
    q_field_m0 = np.asarray(lp_m0.q_fine_mean) - np.asarray(
        map_m0.model.production.spike.delta_q(t_fine)
    )
    ring_mask = (t_fine >= 775.0) & (t_fine <= 792.0)
    rmse_post_m0 = float(np.sqrt(np.mean((q_field_m0[ring_mask] - q_true_fine[ring_mask]) ** 2)))

    # Joint SubspaceCBM(m = 3) + MiyakeSpike + marginalized GP via fit_subspace_cbm_gp
    mod_sub3 = tt.RadiocarbonModel(cbm_sub3_0, tt.SteadyProduction() + gf_zero + spike_init)
    ps_sub3 = tt.ParameterSpace(
        mod_sub3,
        [
            tt.ParamSpec("cbm.alpha", bijector="identity", prior_mean=0.0, prior_sd=1.0),
            tt.ParamSpec("production.spike.t_start", bijector="identity", prior_mean=774.5, prior_sd=2.0),
            tt.ParamSpec("production.spike.duration", bijector="exp", prior_mean=0.0, prior_sd=1.5),
            tt.ParamSpec("production.spike.fluence", bijector="exp", prior_mean=math.log(6.0), prior_sd=1.0),
        ],
    )

    fit_res = tt.fit_subspace_cbm_gp(
        mod_sub3,
        ds_sim,
        ps_sub3,
        cfg_gf,
        sampler="nuts",
        reweight_exact=True,
        is_max_draws=60,
        rng_key=jax.random.PRNGKey(42),
        num_warmup=40,
        num_samples=60,
        num_chains=1,
        max_tree_depth=5,
        map_tol=1e-5,
        reconstruct_linear=False,
    )
    assert isinstance(fit_res, tt.SubspaceGPFitResult)
    # Also verify 4-tuple unpacking works
    idata, map_res, lin_op, diag = fit_res
    assert "log_weight_is" in idata.sample_stats
    lw_arr = np.asarray(idata.sample_stats["log_weight_is"])
    assert np.all(np.isfinite(lw_arr))

    # Posterior-ball tangent accuracy & exact IS efficiency
    assert diag["median_abs_dlnL"] < 0.08, f"median|ΔlnL|={diag['median_abs_dlnL']}"
    assert diag["kish_ess_ratio"] > 0.95, f"kish_ess_ratio={diag['kish_ess_ratio']}"

    # Weighted posterior recovery within 2.5σ of injected truth
    w_is = diag["weights_is"]
    alpha_draws = np.asarray(idata.posterior["cbm__alpha"]).reshape(-1, 3)
    alpha_mean = np.sum(w_is[:, None] * alpha_draws, axis=0)
    alpha_sd = np.sqrt(np.sum(w_is[:, None] * (alpha_draws - alpha_mean[None, :]) ** 2, axis=0))
    z_alpha = np.abs(alpha_mean - alpha_true) / alpha_sd
    assert np.all(z_alpha < 2.5), f"alpha z-scores exceeded 2.5σ: {z_alpha}"

    for key, truth_val in (
        ("production__spike__t_start", float(spike_true.t_start)),
        ("production__spike__duration", float(spike_true.duration)),
        ("production__spike__fluence", float(spike_true.fluence)),
    ):
        draws_k = np.asarray(idata.posterior[key]).reshape(-1)
        mu_k = float(np.sum(w_is * draws_k))
        sd_k = float(np.sqrt(np.sum(w_is * (draws_k - mu_k) ** 2)))
        assert abs(mu_k - truth_val) / sd_k < 2.5, (
            f"{key}: {mu_k:.3f} ± {sd_k:.3f} vs {truth_val:.3f}"
        )

    # Verify > 2.5x post-spike Q_field ringdown RMSE reduction vs fixed m = 0
    cfg_tan = eqx.tree_at(lambda c: c.dct_operator, cfg_gf, lin_op, is_leaf=lambda x: x is None)
    lp_sub3 = tt.linear_posterior(map_res.model, ds_sim, cfg_tan, t_fine=t_fine)
    q_field_sub3 = np.asarray(lp_sub3.q_fine_mean) - np.asarray(
        map_res.model.production.spike.delta_q(t_fine)
    )
    rmse_post_sub3 = float(np.sqrt(np.mean((q_field_sub3[ring_mask] - q_true_fine[ring_mask]) ** 2)))
    assert rmse_post_m0 / rmse_post_sub3 > 2.5, (
        f"Expected >2.5x post-spike RMSE reduction, got m0={rmse_post_m0:.4f} vs m3={rmse_post_sub3:.4f}"
    )


def test_fit_subspace_cbm_gp_joint_gp_hyperparameters_and_laplace(
    synthetic_m3_bundle: dict,
) -> None:
    """Verify joint GP hyperparameter sampling (production.field.kernel.*) with LinearizedDCTOperator."""
    cbm_sub3_0 = synthetic_m3_bundle["cbm_sub3_0"]
    alpha_true = np.asarray(synthetic_m3_bundle["alpha_true"])
    ds_sim = synthetic_m3_bundle["ds_sim"]
    t0 = synthetic_m3_bundle["t_grid_start"]
    t1 = synthetic_m3_bundle["t_grid_end"]
    n_grid = synthetic_m3_bundle["n_grid"]

    kp_init = tt.QuasiPeriodicSpectrum(sigma=0.25, period_yr=11.0, quality=4.5)
    gf_zero = tt.GaussianField(jnp.zeros((n_grid,), dtype=jnp.float64), t0, t1, kernel=kp_init)
    spike_init = tt.MiyakeSpike(774.4, 0.5, 6.0, profile="boxcar")
    mod_joint = tt.RadiocarbonModel(cbm_sub3_0, tt.SteadyProduction() + gf_zero + spike_init)
    cfg_gf = tt.LinearMarginalConfig(marginalize_field=True, offset_mode="hierarchical")

    ps_joint = tt.ParameterSpace(
        mod_joint,
        [
            tt.ParamSpec("cbm.alpha", bijector="identity", prior_mean=0.0, prior_sd=1.0),
            tt.ParamSpec("production.spike.t_start", bijector="identity", prior_mean=774.5, prior_sd=2.0),
            tt.ParamSpec("production.spike.duration", bijector="exp", prior_mean=0.0, prior_sd=1.5),
            tt.ParamSpec("production.spike.fluence", bijector="exp", prior_mean=math.log(6.0), prior_sd=1.0),
            tt.ParamSpec(
                "production.field.kernel.sigma",
                bijector="exp",
                prior_mean=math.log(0.25),
                prior_sd=1.0,
            ),
            tt.ParamSpec(
                "production.field.kernel.period_yr",
                bijector="exp",
                prior_mean=math.log(11.0),
                prior_sd=0.35,
            ),
        ],
    )

    # Joint NUTS with GP kernel hyperparameters varying dynamically at every leapfrog step
    res_nuts = tt.fit_subspace_cbm_gp(
        mod_joint,
        ds_sim,
        ps_joint,
        cfg_gf,
        sampler="nuts",
        reweight_exact=True,
        is_max_draws=40,
        rng_key=jax.random.PRNGKey(7),
        num_warmup=35,
        num_samples=40,
        num_chains=1,
        max_tree_depth=5,
        map_tol=1e-5,
        reconstruct_linear=False,
    )
    assert "production__field__kernel__sigma" in res_nuts.idata.posterior
    assert "production__field__kernel__period_yr" in res_nuts.idata.posterior
    assert res_nuts.diagnostics["median_abs_dlnL"] < 0.08
    assert res_nuts.diagnostics["kish_ess_ratio"] > 0.95

    sig_mean = float(np.mean(res_nuts.idata.posterior["production__field__kernel__sigma"]))
    per_mean = float(np.mean(res_nuts.idata.posterior["production__field__kernel__period_yr"]))
    assert 0.10 < sig_mean < 0.60
    assert 8.0 < per_mean < 14.0

    a_mean = np.mean(np.asarray(res_nuts.idata.posterior["cbm__alpha"]).reshape(-1, 3), axis=0)
    a_sd = np.std(np.asarray(res_nuts.idata.posterior["cbm__alpha"]).reshape(-1, 3), axis=0, ddof=1)
    assert np.all(np.abs(a_mean - alpha_true) / a_sd < 2.5)


# ---------------------------------------------------------------------------
# 4. select_gp_kernel(..., optimize_cbm=True, optimize_spike=True)
# ---------------------------------------------------------------------------


def test_select_gp_kernel_optimize_cbm(synthetic_m3_bundle: dict) -> None:
    """Joint optimize_cbm=True + optimize_spike=True recovers alpha_opt and improves evidence."""
    cbm0 = synthetic_m3_bundle["cbm0"]
    cbm_sub3_0 = synthetic_m3_bundle["cbm_sub3_0"]
    alpha_true = np.asarray(synthetic_m3_bundle["alpha_true"])
    ds_sim = synthetic_m3_bundle["ds_sim"]
    n_grid = synthetic_m3_bundle["n_grid"]

    spike_init = tt.MiyakeSpike(774.4, 0.5, 6.0, profile="boxcar")

    sel_fixed = tt.select_gp_kernel(
        cbm0,
        ds_sim,
        candidates=("matern32", "sho_solar"),
        extra_production=spike_init,
        optimize_spike=True,
        optimize_cbm=False,
        multi_start=1,
        n_grid=n_grid,
        pad_years=15.0,
        n_posterior_samples=20,
    )
    sel_joint = tt.select_gp_kernel(
        cbm_sub3_0,
        ds_sim,
        candidates=("matern32", "sho_solar"),
        extra_production=spike_init,
        optimize_spike=True,
        optimize_cbm=True,
        cbm_prior_sd=1.0,
        multi_start=1,
        n_grid=n_grid,
        pad_years=15.0,
        n_posterior_samples=20,
    )

    best_j = sel_joint.best_result
    assert best_j.converged
    assert isinstance(best_j.cbm_opt, tt.SubspaceCBM)
    assert best_j.alpha_opt is not None
    assert best_j.alpha_opt.shape == (3,)
    assert float(best_j.log_evidence_hyper) == float(best_j.log_hyper_evidence)
    assert math.isfinite(float(best_j.log_hyper_evidence))

    # Parameter count d includes the m=3 SubspaceCBM modes
    fixed_sho = sel_fixed.get_candidate("sho_solar")
    joint_sho = sel_joint.get_candidate("sho_solar")
    assert joint_sho.n_hyperparams == fixed_sho.n_hyperparams + 3

    # Joint CBM optimization dramatically improves Laplace hyper-evidence and MacKay error scale
    assert float(best_j.log_hyper_evidence) > float(sel_fixed.best_result.log_hyper_evidence) + 50.0
    assert sel_joint.error_scale_global < sel_fixed.error_scale_global - 0.25
    assert sel_joint.error_scale_global < 1.25

    # Recovered alpha_opt tracks alpha_true closely
    alpha_opt_np = np.asarray(best_j.alpha_opt)
    assert float(np.linalg.norm(alpha_opt_np - alpha_true)) < 1.0
