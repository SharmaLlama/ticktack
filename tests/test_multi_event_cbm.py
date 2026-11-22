"""Unit and statistical recovery tests for multi-event shared-CBM pooling."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ticktack as tt
from ticktack.multievent import (
    EventWindow,
    MultiEventDataset,
    MultiEventModel,
    fit_multi_event_subspace_cbm,
    multi_event_cbm_gp_fisher_spectrum,
    multi_event_marginal_log_likelihood,
    multi_event_marginal_log_posterior,
    reweight_multi_event_exact,
)


def _make_synthetic_dataset(
    t_center: float,
    *,
    n_nh: int = 2,
    n_sh: int = 2,
    n_pre: int = 10,
    n_post: int = 14,
    sigma: float = 0.9,
    d14c_by_series: list[np.ndarray] | None = None,
    name: str = "synth",
) -> tt.Dataset:
    years = np.arange(
        int(math.floor(t_center)) - n_pre,
        int(math.floor(t_center)) + n_post + 1,
        dtype=np.float64,
    )
    n_yr = len(years)
    series_list: list[tt.TreeRingSeries] = []
    for i in range(n_nh):
        y = np.zeros(n_yr) if d14c_by_series is None else d14c_by_series[i]
        series_list.append(
            tt.TreeRingSeries(
                years=years,
                d14c=y,
                sigma_d14c=np.full(n_yr, sigma),
                hemisphere="north",
                seasonal_weights="april-september",
                name=f"{name}_NH_{i + 1}",
            )
        )
    for j in range(n_sh):
        y = np.zeros(n_yr) if d14c_by_series is None else d14c_by_series[n_nh + j]
        series_list.append(
            tt.TreeRingSeries(
                years=years,
                d14c=y,
                sigma_d14c=np.full(n_yr, sigma),
                hemisphere="south",
                year_convention="schulman_start",
                seasonal_weights="october-march",
                name=f"{name}_SH_{j + 1}",
            )
        )
    return tt.Dataset(series_list, name=name)


def _make_event_window(
    name: str,
    t_spike: float,
    fluence: float,
    *,
    duration: float = 0.6,
    n_grid: int = 32,
    sigma_gp: float = 0.22,
    period_yr: float = 11.0,
    quality: float = 4.0,
    offset_mode: str = "hierarchical",
    n_nh: int = 2,
    n_sh: int = 2,
) -> EventWindow:
    ds = _make_synthetic_dataset(t_spike, n_nh=n_nh, n_sh=n_sh, name=name)
    t_lo, t_hi = ds.time_span
    kernel = tt.QuasiPeriodicSpectrum(
        sigma=sigma_gp, period_yr=period_yr, quality=quality
    )
    gf = tt.GaussianField(
        jnp.zeros((n_grid,), dtype=jnp.float64),
        t_start=t_lo - 10.0,
        t_end=t_hi + 0.5,
        kernel=kernel,
    )
    spike = tt.MiyakeSpike(
        t_start=t_spike, duration=duration, fluence=fluence, profile="boxcar"
    )
    prod = tt.ProductionSum(steady=tt.SteadyProduction(), field=gf, spike=spike)
    cfg = tt.LinearMarginalConfig(offset_mode=offset_mode, marginalize_field=True)
    return EventWindow(
        name=name,
        dataset=ds,
        production=prod,
        marginal_config=cfg,
        spike_path="spike",
    )


# ---------------------------------------------------------------------------
# 1. Container & validation tests
# ---------------------------------------------------------------------------


def test_event_window_and_multi_event_dataset_validation() -> None:
    # Verify top-level ticktack exports
    for sym in (
        "EventWindow",
        "MultiEventDataset",
        "MultiEventFitResult",
        "MultiEventModel",
        "multi_event_cbm_gp_fisher_spectrum",
        "multi_event_marginal_log_likelihood",
        "multi_event_marginal_log_posterior",
        "reweight_multi_event_exact",
        "fit_multi_event_subspace_cbm",
    ):
        assert hasattr(tt, sym), f"Missing top-level export ticktack.{sym}"

    ev1 = _make_event_window("e774", 774.25, 6.4)
    ev2 = _make_event_window("e993", 993.3, 4.0)
    ev3 = _make_event_window("e660bce", -663.2, 5.2)

    mds = MultiEventDataset((ev1, ev2, ev3), name="holocene3")
    assert mds.n_events == 3
    assert len(mds) == 3
    assert mds.n_obs_total == ev1.n_obs + ev2.n_obs + ev3.n_obs
    assert mds.event_names == ("e774", "e993", "e660bce")
    assert "e774" in mds
    assert "missing" not in mds
    assert mds["e993"].name == "e993"
    assert mds[0].name == "e774"
    assert mds.get_event("e660bce").get_spike() is not None
    assert float(mds.get_event("e774").get_spike().fluence) == pytest.approx(6.4)

    # Duplicate names rejected
    with pytest.raises(ValueError, match="Duplicate"):
        MultiEventDataset((ev1, _make_event_window("e774", 993.3, 4.0)))

    # Overlapping windows rejected by default
    ev_overlap = _make_event_window("e775_dup", 776.0, 5.0)
    with pytest.raises(ValueError, match="overlaps"):
        MultiEventDataset((ev1, ev_overlap))

    # Overlapping allowed explicitly when check_disjoint=False
    mds_ov = MultiEventDataset((ev1, ev_overlap), check_disjoint=False)
    assert mds_ov.n_events == 2

    # Invalid event names rejected (dots break zodiax paths)
    with pytest.raises(ValueError, match=r"\."):
        EventWindow("e.774", ev1.dataset, ev1.production)
    with pytest.raises(ValueError, match="non-empty"):
        EventWindow("   ", ev1.dataset, ev1.production)

    # Unknown lookup raises KeyError
    with pytest.raises(KeyError):
        mds.get_event("no_such_event")


def test_multi_event_model_zodiax_parameter_space_and_single_prior() -> None:
    cbm0 = _make_compact_balanced_cbm()
    ev1 = _make_event_window("775CE", 774.25, 6.4, n_grid=20, n_nh=1, n_sh=0)
    ev2 = _make_event_window("993CE", 993.3, 4.0, n_grid=20, n_nh=1, n_sh=0)
    mds = MultiEventDataset((ev1, ev2))

    # Build a 2-mode SubspaceCBM directly (no redundant Fisher build needed here)
    p_free = int(cbm0.log_rates_free.shape[0])
    basis = np.zeros((p_free, 2), dtype=np.float64)
    basis[0, 0] = 0.35
    basis[1, 1] = 0.35
    cbm_sub = tt.SubspaceCBM(
        topology=cbm0.topology,
        base_log_rates_free=np.asarray(cbm0.log_rates_free, dtype=np.float64),
        basis=basis,
        base_log_total_mass=float(cbm0.log_total_mass),
        prod_weights=cbm0.prod_weights,
        tie_matrix=cbm0.tie_matrix_rows,
        decay_rate=cbm0.decay_rate,
    )
    m_multi = MultiEventModel.from_multi_dataset(cbm_sub, mds)

    assert m_multi.n_events == 2
    assert m_multi.event_names == ("775CE", "993CE")

    ps = tt.ParameterSpace(
        m_multi,
        [
            tt.ParamSpec("cbm.alpha", "identity", prior_mean=0.0, prior_sd=1.0),
            tt.ParamSpec("events.775CE.spike.fluence", "exp", prior_mean=math.log(6.4), prior_sd=0.5),
            tt.ParamSpec("events.993CE.spike.fluence", "exp", prior_mean=math.log(4.0), prior_sd=0.5),
        ],
    )
    assert ps.n_params == 4

    u_test = jnp.array([0.4, -0.3, math.log(7.0), math.log(4.5)], dtype=jnp.float64)
    m_c = ps.constrain(u_test, m_multi)
    np.testing.assert_allclose(np.asarray(m_c.cbm.alpha), [0.4, -0.3], rtol=1e-12)
    assert float(m_c.get("events.775CE.spike.fluence")) == pytest.approx(7.0, rel=1e-12)
    assert float(m_c.get("events.993CE.spike.fluence")) == pytest.approx(4.5, rel=1e-12)
    np.testing.assert_allclose(np.asarray(ps.unconstrain(m_c)), np.asarray(u_test), rtol=1e-12)

    # Exact block-diagonal likelihood additivity
    ll_multi = multi_event_marginal_log_likelihood(m_c, mds)
    ll_sum = sum(
        tt.marginal_log_likelihood(m_c.event_model(ev.name), ev.dataset, ev.marginal_config)
        for ev in mds.events
    )
    assert float(ll_multi) == pytest.approx(float(ll_sum), abs=1e-11)

    # Prior on cbm.alpha is evaluated ONCE (not E times!)
    lp_multi = multi_event_marginal_log_posterior(m_c, mds, ps, u_test)
    prior_diff = float(lp_multi - ll_multi)
    expected_prior = (
        -0.5 * (0.4**2 + (-0.3) ** 2) - 2.0 * 0.5 * math.log(2.0 * math.pi)
        - 0.5 * ((math.log(7.0) - math.log(6.4)) / 0.5) ** 2
        - math.log(0.5)
        - 0.5 * math.log(2.0 * math.pi)
        - 0.5 * ((math.log(4.5) - math.log(4.0)) / 0.5) ** 2
        - math.log(0.5)
        - 0.5 * math.log(2.0 * math.pi)
    )
    assert prior_diff == pytest.approx(expected_prior, abs=1e-12)


# ---------------------------------------------------------------------------
# 2. Fisher additivity, prior-once covariance, and monotonic d_eff growth
# ---------------------------------------------------------------------------


def _make_compact_balanced_cbm() -> tt.BalancedCBM:
    """Compact 4-box strongly connected BalancedCBM for fast multi-event tests."""
    topo = tt.Topology(
        box_names=("Stratosphere", "Troposphere", "Biosphere", "Ocean"),
        edges_src=(0, 1, 1, 2, 1, 3),
        edges_dst=(1, 0, 2, 1, 3, 1),
        prod_box_indices=(0, 1),
        trop_box_indices=(1,),
    )
    return tt.BalancedCBM.from_masses_and_fluxes(
        topology=topo,
        masses=np.array([90.0, 500.0, 700.0, 18000.0], dtype=np.float64),
        edge_fluxes=np.array([45.0, 45.0, 60.0, 60.0, 80.0, 80.0], dtype=np.float64),
        prod_weights=(0.7, 0.3),
    )


@pytest.fixture(scope="module")
def _shared_compact_events():
    cbm0 = _make_compact_balanced_cbm()
    prior_sd = 0.35
    n_grid = 20
    ev1 = _make_event_window("e774", 774.25, 6.4, duration=0.75, n_grid=n_grid, n_nh=2, n_sh=0)
    ev2 = _make_event_window("e993", 993.25, 5.0, duration=0.70, n_grid=n_grid, n_nh=2, n_sh=0)
    ev3 = _make_event_window("e660bce", -663.2, 5.1, duration=0.75, n_grid=n_grid, n_nh=2, n_sh=0)
    res1 = multi_event_cbm_gp_fisher_spectrum(cbm0, MultiEventDataset((ev1,)), prior_sd=prior_sd)
    res2 = multi_event_cbm_gp_fisher_spectrum(cbm0, MultiEventDataset((ev2,)), prior_sd=prior_sd)
    res12 = multi_event_cbm_gp_fisher_spectrum(
        cbm0,
        MultiEventDataset((ev1, ev2)),
        prior_sd=prior_sd,
        mass_fn=lambda m: m.reservoir_masses(),
    )
    res123 = multi_event_cbm_gp_fisher_spectrum(
        cbm0,
        MultiEventDataset((ev1, ev2, ev3)),
        prior_sd=prior_sd,
    )
    return cbm0, prior_sd, n_grid, ev1, ev2, ev3, res1, res2, res12, res123


def test_multi_event_fisher_additivity_and_monotonic_deff(_shared_compact_events) -> None:
    cbm0, prior_sd, _n_grid, _ev1, _ev2, _ev3, res1, res2, res12, res123 = _shared_compact_events

    # Exact linear additivity: F_tot == F_1 + F_2 (both whitened and marginal)
    np.testing.assert_allclose(
        np.asarray(res12.whitened_fisher),
        np.asarray(res1.whitened_fisher + res2.whitened_fisher),
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        np.asarray(res12.marginal_fisher),
        np.asarray(res1.marginal_fisher + res2.marginal_fisher),
        rtol=1e-11,
        atol=1e-11,
    )

    # Prior precision enters posterior_cov ONCE (not E times!)
    act = np.asarray(res12.active_mask, dtype=bool)
    f_raw_act = np.asarray(res12.marginal_fisher)[np.ix_(act, act)]
    pi_prior_once = np.eye(f_raw_act.shape[0]) / (prior_sd**2)
    expected_post_cov = np.linalg.inv(f_raw_act + pi_prior_once)
    np.testing.assert_allclose(
        np.asarray(res12.posterior_cov),
        expected_post_cov,
        rtol=1e-9,
        atol=1e-11,
    )

    # Monotonic d_eff and info_bits growth from 1 -> 2 -> 3 events
    assert float(res1.d_eff) < float(res12.d_eff) < float(res123.d_eff)
    assert float(res1.info_bits) < float(res12.info_bits) < float(res123.info_bits)

    # Directly compatible with SubspaceCBM.from_fisher and preserves Kirchhoff balance
    cbm_sub = tt.SubspaceCBM.from_fisher(cbm0, res12, n_modes=3)
    assert cbm_sub.n_modes == 3
    cbm_pert = cbm_sub.set("alpha", jnp.array([1.0, -0.8, 0.5]))
    assert float(jnp.max(jnp.abs(cbm_pert.kirchhoff_residual()))) < 1e-10
    assert res12.mass_loadings is not None
    assert res12.mass_loadings.shape[0] == cbm0.n_boxes


# ---------------------------------------------------------------------------
# 3. Synthetic multi-event MAP + NUTS recovery & exact IS reweighting
# ---------------------------------------------------------------------------


def test_fit_multi_event_subspace_cbm_map_nuts_and_is_recovery(_shared_compact_events) -> None:
    jax.clear_caches()
    cbm0, _prior_sd, n_grid, ev1_tmpl, ev2_tmpl, _ev3, _r1, _r2, spec_multi, _r123 = (
        _shared_compact_events
    )
    sigma_val = 0.85
    mds_tmpl = MultiEventDataset((ev1_tmpl, ev2_tmpl))
    cbm_sub0 = tt.SubspaceCBM.from_fisher(cbm0, spec_multi, n_modes=2)

    alpha_true = jnp.array([1.2, -1.0], dtype=jnp.float64)
    cbm_true = cbm_sub0.set("alpha", alpha_true)

    # Generate synthetic observations for both events with independent GP draws, offsets, and noise
    key = jax.random.PRNGKey(42)
    synth_events: list[EventWindow] = []
    true_spikes = {
        "e774": tt.MiyakeSpike(t_start=774.25, duration=0.75, fluence=6.4, profile="boxcar"),
        "e993": tt.MiyakeSpike(t_start=993.25, duration=0.70, fluence=5.0, profile="boxcar"),
    }
    for ev_t in mds_tmpl.events:
        key, k_gp, k_off, k_noise = jax.random.split(key, 4)
        gf_tmpl = ev_t.production.field
        z_true = jax.random.normal(k_gp, (n_grid,), dtype=jnp.float64)
        gf_true = gf_tmpl.with_latent(z_true)
        sp_true = true_spikes[ev_t.name]

        op_true = tt.build_dct_response_operator(
            cbm_true,
            ev_t.dataset,
            t_start=gf_true.t_start,
            t_end=gf_true.t_end,
            n_grid=n_grid,
        )
        d14c_gp = op_true.w_dct @ (jnp.sqrt(gf_true.power_spectrum()) * z_true)
        delta_lab = 0.8 * jax.random.normal(k_off, (ev_t.n_series,), dtype=jnp.float64)
        d14c_sp = tt.RadiocarbonModel(cbm_true, tt.SteadyProduction() + sp_true).predict_d14c(
            ev_t.dataset, delta_global=-15.0, offsets=delta_lab
        )
        noise = sigma_val * jax.random.normal(k_noise, (ev_t.n_obs,), dtype=jnp.float64)
        y_obs = np.asarray(d14c_sp + d14c_gp + noise)

        n_yr = ev_t.dataset.series[0].n_obs
        ds_obs = _make_synthetic_dataset(
            float(sp_true.t_start),
            n_nh=2,
            n_sh=0,
            sigma=sigma_val,
            d14c_by_series=[y_obs[i * n_yr : (i + 1) * n_yr] for i in range(2)],
            name=ev_t.name,
        )
        # Perturbed initial spike guesses for inference
        sp_init = tt.MiyakeSpike(
            t_start=float(sp_true.t_start) + 0.15,
            duration=float(sp_true.duration),
            fluence=float(sp_true.fluence) * 0.9,
            profile="boxcar",
        )
        prod_init = tt.ProductionSum(
            steady=tt.SteadyProduction(),
            field=gf_tmpl,
            spike=sp_init,
        )
        synth_events.append(
            EventWindow(
                name=ev_t.name,
                dataset=ds_obs,
                production=prod_init,
                marginal_config=ev_t.marginal_config,
                spike_path="spike",
            )
        )

    mds_obs = MultiEventDataset(tuple(synth_events))
    model_init = MultiEventModel.from_multi_dataset(cbm_sub0, mds_obs)
    ps_joint = tt.ParameterSpace(
        model_init,
        [
            tt.ParamSpec("cbm.alpha", "identity", prior_mean=0.0, prior_sd=1.0),
            tt.ParamSpec("events.e774.spike.t_start", "identity", prior_mean=774.3, prior_sd=1.5),
            tt.ParamSpec("events.e774.spike.fluence", "exp", prior_mean=math.log(6.0), prior_sd=0.8),
            tt.ParamSpec("events.e993.spike.t_start", "identity", prior_mean=993.3, prior_sd=1.5),
            tt.ParamSpec("events.e993.spike.fluence", "exp", prior_mean=math.log(5.0), prior_sd=0.8),
        ],
    )

    # First verify single-event vs multi-event Laplace shrinkage on shared alpha
    mds_single = MultiEventDataset((synth_events[0],))
    m_single = MultiEventModel.from_multi_dataset(cbm_sub0, mds_single)
    ps_single = tt.ParameterSpace(
        m_single,
        [
            tt.ParamSpec("cbm.alpha", "identity", prior_mean=0.0, prior_sd=1.0),
            tt.ParamSpec("events.e774.spike.t_start", "identity", prior_mean=774.3, prior_sd=1.5),
            tt.ParamSpec("events.e774.spike.fluence", "exp", prior_mean=math.log(6.0), prior_sd=0.8),
        ],
    )
    fit_lap_1 = fit_multi_event_subspace_cbm(
        m_single,
        mds_single,
        ps_single,
        sampler="laplace",
        compute_linear_posterior=False,
    )
    assert fit_lap_1.idata is None
    assert fit_lap_1.map_converged

    # Joint 2-event NUTS + exact IS reweighting
    fit_joint = fit_multi_event_subspace_cbm(
        model_init,
        mds_obs,
        ps_joint,
        sampler="nuts",
        reweight_exact=True,
        use_linearized_dct=True,
        is_max_draws=70,
        num_warmup=50,
        num_samples=35,
        num_chains=2,
        max_tree_depth=6,
        rng_key=jax.random.PRNGKey(99),
    )

    assert fit_joint.map_converged
    assert fit_joint.idata is not None
    assert "cbm__alpha" in fit_joint.idata.posterior
    assert "events__e774__spike__fluence" in fit_joint.idata.posterior
    assert "events__e993__spike__fluence" in fit_joint.idata.posterior
    assert "log_weight_is" in fit_joint.idata.sample_stats
    assert fit_joint.diagnostics["n_divergences"] == 0

    # Pooling 2 events tightens Laplace posterior variance on shared alpha
    sd_alpha_1 = np.sqrt(np.diag(np.asarray(fit_lap_1.laplace_cov))[:2])
    sd_alpha_2 = np.sqrt(np.diag(np.asarray(fit_joint.laplace_cov))[:2])
    assert np.all(sd_alpha_2 < sd_alpha_1)

    # Tangent-linear DCT accuracy & exact IS efficiency
    assert fit_joint.kish_ess_ratio is not None and fit_joint.kish_ess_ratio > 0.90
    assert fit_joint.median_abs_dlnL is not None and fit_joint.median_abs_dlnL < 0.15

    # Posterior recovery of shared alpha and both events' spike parameters within 3 sigma
    alpha_draws = np.asarray(fit_joint.idata.posterior["cbm__alpha"]).reshape(-1, 2)
    alpha_mean = np.mean(alpha_draws, axis=0)
    alpha_sd = np.std(alpha_draws, axis=0, ddof=1)
    assert np.all(np.abs(alpha_mean - np.asarray(alpha_true)) < 3.0 * alpha_sd)

    for ev_name, sp_true in true_spikes.items():
        fl_draws = np.asarray(
            fit_joint.idata.posterior[f"events__{ev_name}__spike__fluence"]
        ).reshape(-1)
        ts_draws = np.asarray(
            fit_joint.idata.posterior[f"events__{ev_name}__spike__t_start"]
        ).reshape(-1)
        assert abs(float(np.mean(fl_draws)) - float(sp_true.fluence)) < 3.0 * float(np.std(fl_draws, ddof=1))
        assert abs(float(np.mean(ts_draws)) - float(sp_true.t_start)) < 3.0 * float(np.std(ts_draws, ddof=1))

    # Raw chi2 / N reasonable on both events (noting GP effective DoF on N=50)
    for ev_name, chi2_n in fit_joint.diagnostics["per_event_chi2_per_n"].items():
        assert 0.25 < chi2_n < 1.6, f"Unexpected raw chi2/N={chi2_n} on {ev_name}"

    # Verify exact IS weight sign & zero error at alpha_MAP and PR 1 LinearizedDCTOperator parity
    assert all(isinstance(op, tt.LinearizedDCTOperator) for op in fit_joint.dct_operators.values())
    ll_ex_map = multi_event_marginal_log_likelihood(fit_joint.model, mds_obs)
    assert float(ll_ex_map - fit_joint.log_likelihood_map) == pytest.approx(0.0, abs=1e-10)
    mds_tan = mds_obs.with_marginal_configs(
        {
            ev.name: tt.LinearMarginalConfig(
                offset_mode=ev.marginal_config.offset_mode,
                marginalize_field=True,
                dct_operator=fit_joint.dct_operators[ev.name],
            )
            for ev in mds_obs.events
        }
    )
    ll_tan_cfg = multi_event_marginal_log_likelihood(fit_joint.model, mds_tan)
    assert float(ll_tan_cfg - ll_ex_map) == pytest.approx(0.0, abs=1e-10)
    is_at_map = reweight_multi_event_exact(
        jnp.broadcast_to(fit_joint.u_map[None, :], (2, ps_joint.n_params)),
        model_init,
        mds_obs,
        ps_joint,
        fit_joint.dct_operators,
    )
    np.testing.assert_allclose(np.asarray(is_at_map["log_weights"]), 0.0, atol=1e-10)
