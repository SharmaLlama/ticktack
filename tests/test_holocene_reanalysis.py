"""End-to-end scientific validation across all 6 bundled Holocene Miyake events and IntCal20."""

from __future__ import annotations

import json
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import tinygp.kernels as tk

import ticktack as tt

ROOT = Path(__file__).resolve().parents[1]

MIYAKE_EVENTS: tuple[str, ...] = (
    "775CE",
    "993CE",
    "660BCE",
    "5259BCE",
    "5410BCE",
    "7176BCE",
)

EVENT_PRIORS: dict[str, tuple[float, float, float, float]] = {
    # (t_start_init, dur_init, fluence_init, dur_upper)
    "775CE": (774.4, 0.35, 6.5, 2.0),
    "993CE": (993.3, 0.35, 3.8, 2.0),
    "660BCE": (-663.2, 1.8, 5.0, 5.0),
    "5259BCE": (-5258.6, 0.5, 7.0, 3.0),
    "5410BCE": (-5409.6, 0.6, 2.8, 3.0),
    "7176BCE": (-7175.5, 0.6, 6.8, 3.0),
}


@pytest.fixture(scope="module")
def brehm21_cbm() -> tt.CarbonBoxModel:
    return tt.load_preset("Brehm21")


@pytest.fixture(scope="module")
def six_event_fits(brehm21_cbm: tt.CarbonBoxModel) -> dict[str, dict[str, object]]:
    """Fit all 6 bundled Holocene Miyake events once under Brehm21 + marginalized solar + GP."""
    cfg_gp = tt.LinearMarginalConfig(
        offset_mode="hierarchical",
        solar_prior_sd=0.25,
        marginalize_solar=True,
        marginalize_gp=True,
    )
    results: dict[str, dict[str, object]] = {}
    for ev in MIYAKE_EVENTS:
        t0_init, dur_init, fl_init, dur_hi = EVENT_PRIORS[ev]
        ds = tt.load_bundled_event(ev)
        prod = tt.ProductionSum(
            steady=tt.SteadyProduction(),
            solar=tt.HarmonicSolar.from_polar(amplitudes=0.0, phases=0.0),
            gp=tt.GPProduction(tk.quasisep.Matern32(scale=6.0, sigma=0.08), n_modes=16),
            spike=tt.MiyakeSpike(
                t_start=t0_init, duration=dur_init, fluence=fl_init, profile="boxcar"
            ),
        )
        model = tt.RadiocarbonModel(brehm21_cbm, prod)
        t_sd = 0.45 if ev in ("775CE", "993CE") else 1.2
        ps = tt.ParameterSpace(
            model,
            [
                tt.ParamSpec("production.spike.t_start", "identity", prior_mean=t0_init, prior_sd=t_sd),
                tt.ParamSpec(
                    "production.spike.duration",
                    "sigmoid",
                    prior_mean=-0.5,
                    prior_sd=1.4,
                    low=0.05,
                    high=dur_hi,
                ),
                tt.ParamSpec(
                    "production.spike.fluence",
                    "exp",
                    prior_mean=math.log(fl_init),
                    prior_sd=0.6,
                ),
            ],
        )
        map_res = tt.fit_map(
            model,
            ds,
            ps,
            cfg_gp,
            method="BFGS",
            max_steps=60,
            compute_laplace=False,
            compute_linear_posterior=False,
        )
        lp = tt.linear_posterior(map_res.fitted_model, ds, cfg_gp)
        results[ev] = {
            "dataset": ds,
            "map_res": map_res,
            "linear_post": lp,
            "cfg": cfg_gp,
        }
    return results


# ===========================================================================
# 1. Bundled Catalogue, Proleptic BCE Conversion, & SH Ring Window Audit
# ===========================================================================


class TestBundledHoloceneCatalogueAudit:
    def test_all_bundled_events_and_intcal20_load_cleanly(self) -> None:
        pkg_events = {
            entry.event
            for entry in tt.BUNDLED_DATASET_CATALOGUE.values()
            if entry.location == "package_data"
        }
        expected = set(MIYAKE_EVENTS) | {"IntCal20"}
        assert pkg_events == expected

        expected_series_counts = {
            "775CE": 28,
            "993CE": 9,
            "660BCE": 2,
            "5259BCE": 4,
            "5410BCE": 3,
            "7176BCE": 3,
            "IntCal20": 1,
        }
        for ev, exp_s in expected_series_counts.items():
            ds = tt.load_bundled_event(ev)
            assert ds.n_series == exp_s
            assert ds.n_obs > 0
            assert len(set(ds.series_names)) == ds.n_series
            assert bool(jnp.all(jnp.isfinite(ds.d14c_flat)))
            assert bool(jnp.all(jnp.isfinite(ds.sigma_flat)))
            assert bool(jnp.all(ds.sigma_flat > 0.0))
            assert bool(jnp.all(ds.t_ends_flat > ds.t_starts_flat))
            for s in ds.series:
                diffs = np.diff(np.asarray(s.t_starts))
                assert np.all(diffs >= -1e-12), f"{ev}:{s.name} not chronologically sorted"

        ds_775_nb = tt.load_bundled_event("775CE", include_repo_notebooks=True)
        assert ds_775_nb.n_series == 42
        assert ds_775_nb.n_obs > 500
        assert "south" in ds_775_nb.hemispheres and "north" in ds_775_nb.hemispheres

    def test_historical_bce_proleptic_conversion_and_subannual_660bce_windows(self) -> None:
        for ev in ("660BCE", "5259BCE", "5410BCE", "7176BCE"):
            ds = tt.load_bundled_event(ev)
            for s in ds.series:
                assert s.calendar == "historical_bce"
                assert np.allclose(np.asarray(s.years_ce), np.asarray(s.raw_years) + 1.0, atol=1e-12)

        ds_660 = tt.load_bundled_event("660BCE")
        by_name = {s.name: s for s in ds_660.series}
        ew = by_name["Sakurai20_CedarEw"]
        lw = by_name["Sakurai20_CedarLw"]
        assert np.allclose(np.asarray(ew.t_starts - ew.years_ce), 0.25, atol=1e-12)
        assert np.allclose(np.asarray(ew.t_ends - ew.years_ce), 0.50, atol=1e-12)
        assert np.allclose(np.asarray(lw.t_starts - lw.years_ce), 0.50, atol=1e-12)
        assert np.allclose(np.asarray(lw.t_ends - lw.years_ce), 0.75, atol=1e-12)

    def test_southern_hemisphere_schulman_vs_completion_year_alignment(self) -> None:
        tas = tt.load_bundled_series("775AD-early-S/Büntgen18_TAS01.csv")
        kauri = tt.load_bundled_series("notebooks/775CE/Guttler15_Kauri.csv")
        assert tas.hemisphere == "south" and tas.year_convention == "schulman_start"
        assert kauri.hemisphere == "south" and kauri.year_convention == "end_year"

        tas_idx = int(np.flatnonzero(np.isclose(np.asarray(tas.raw_years), 774.0))[0])
        kauri_idx = int(np.flatnonzero(np.isclose(np.asarray(kauri.raw_years), 775.0))[0])
        assert float(tas.t_starts[tas_idx]) == pytest.approx(774.75, abs=1e-12)
        assert float(tas.t_ends[tas_idx]) == pytest.approx(775.25, abs=1e-12)
        assert float(kauri.t_starts[kauri_idx]) == pytest.approx(774.75, abs=1e-12)
        assert float(kauri.t_ends[kauri_idx]) == pytest.approx(775.25, abs=1e-12)


# ===========================================================================
# 2. Joint NH+SH Brehm21 Semiparametric Re-Analysis (775CE & 993CE)
# ===========================================================================


class TestTwoHemisphereSemiparametricReanalysis:
    def test_brehm21_interhemispheric_steady_state_gradient(
        self, brehm21_cbm: tt.CarbonBoxModel
    ) -> None:
        steady_model = tt.RadiocarbonModel(brehm21_cbm, tt.SteadyProduction())
        t_eval = jnp.array([770.0, 990.0])
        nh_ss = steady_model.predict_fine(t_eval, hemisphere="north")
        sh_ss = steady_model.predict_fine(t_eval, hemisphere="south")
        assert float(nh_ss[0]) == pytest.approx(2.194, abs=0.02)
        assert float(sh_ss[0]) == pytest.approx(-2.194, abs=0.02)
        assert float(nh_ss[0] - sh_ss[0]) == pytest.approx(4.388, abs=0.02)

    def test_joint_nh_sh_fit_775ce_and_993ce_and_sh_convention_consistency(
        self,
        brehm21_cbm: tt.CarbonBoxModel,
        six_event_fits: dict[str, dict[str, object]],
    ) -> None:
        fit_775 = six_event_fits["775CE"]
        res_775: tt.MAPResult = fit_775["map_res"]  # type: ignore[assignment]
        lp_775: tt.LinearPosteriorResult = fit_775["linear_post"]  # type: ignore[assignment]
        t0_775 = float(res_775.theta_map["production.spike.t_start"])
        fl_775 = float(res_775.theta_map["production.spike.fluence"])
        d0_775 = float(lp_775.baseline_d14c)
        assert 773.8 <= t0_775 <= 774.7
        assert 5.5 <= fl_775 <= 7.8
        assert -30.0 <= d0_775 <= -12.0

        fit_993 = six_event_fits["993CE"]
        res_993: tt.MAPResult = fit_993["map_res"]  # type: ignore[assignment]
        lp_993: tt.LinearPosteriorResult = fit_993["linear_post"]  # type: ignore[assignment]
        t0_993 = float(res_993.theta_map["production.spike.t_start"])
        fl_993 = float(res_993.theta_map["production.spike.fluence"])
        d0_993 = float(lp_993.baseline_d14c)
        assert 992.7 <= t0_993 <= 993.8
        assert 3.0 <= fl_993 <= 5.2
        assert -30.0 <= d0_993 <= -12.0

        # Compare Schulman SH (Büntgen18_TAS01) vs completion-year SH (Guttler15_Kauri)
        nh_a = tt.load_bundled_series("775AD-early-N/Büntgen18_SWE02.csv")
        nh_b = tt.load_bundled_series("775AD-late-N/Büntgen18_USA02.csv")
        sh_tas = tt.load_bundled_series("775AD-early-S/Büntgen18_TAS01.csv")
        sh_kauri = tt.load_bundled_series("notebooks/775CE/Guttler15_Kauri.csv")

        prod_pair = tt.ProductionSum(
            steady=tt.SteadyProduction(),
            solar=tt.HarmonicSolar.from_polar(amplitudes=0.0, phases=0.0),
            spike=tt.MiyakeSpike(t_start=774.4, duration=0.35, fluence=6.4, profile="boxcar"),
        )
        m_pair = tt.RadiocarbonModel(brehm21_cbm, prod_pair)
        ps_pair = tt.ParameterSpace(
            m_pair,
            [
                tt.ParamSpec("production.spike.t_start", "identity", prior_mean=774.4, prior_sd=0.8),
                tt.ParamSpec("production.spike.fluence", "exp", prior_mean=math.log(6.4), prior_sd=0.6),
            ],
        )
        cfg_pair = tt.LinearMarginalConfig(
            offset_mode="hierarchical", solar_prior_sd=0.25, marginalize_solar=True
        )
        res_tas = tt.fit_map(
            m_pair,
            tt.Dataset((nh_a, nh_b, sh_tas)),
            ps_pair,
            cfg_pair,
            method="BFGS",
            max_steps=50,
            compute_laplace=False,
            compute_linear_posterior=False,
        )
        res_kauri = tt.fit_map(
            m_pair,
            tt.Dataset((nh_a, nh_b, sh_kauri)),
            ps_pair,
            cfg_pair,
            method="BFGS",
            max_steps=50,
            compute_laplace=False,
            compute_linear_posterior=False,
        )
        ts_tas = float(res_tas.theta_map["production.spike.t_start"])
        ts_kauri = float(res_kauri.theta_map["production.spike.t_start"])
        assert abs(ts_tas - ts_kauri) < 0.25
        assert ts_tas < 774.85 and ts_kauri < 774.85


# ===========================================================================
# 3. Holocene Ambient-Baseline Yield Scaling (standard vs ambient)
# ===========================================================================


class TestHoloceneAmbientBaselineYieldScaling:
    def test_solver_standard_vs_ambient_exact_scaling_identity(
        self, brehm21_cbm: tt.CarbonBoxModel
    ) -> None:
        solver = tt.ExactLinearSolver()
        ds_sub = tt.load_bundled_event("5259BCE")
        s0 = ds_sub.series[0]
        nh_box = brehm21_cbm.box_index("Troposphere", "north")
        fluence_std = 7.2

        for delta_global in (-18.5, 0.0, 92.0):
            ambient_factor = 1.0 + delta_global / 1000.0
            sp_std = tt.MiyakeSpike(t_start=-5258.6, duration=0.5, fluence=fluence_std)
            sp_amb = tt.MiyakeSpike(
                t_start=-5258.6, duration=0.5, fluence=fluence_std / ambient_factor
            )
            pred_std = solver.predict_d14c(
                brehm21_cbm,
                sp_std,
                s0.sub_t_starts,
                s0.sub_t_ends,
                sub_weights=s0.seasonal_weights.weights,
                box_indices=nh_box,
                delta_global=delta_global,
                relative_to="standard",
            )
            pred_amb = solver.predict_d14c(
                brehm21_cbm,
                sp_amb,
                s0.sub_t_starts,
                s0.sub_t_ends,
                sub_weights=s0.seasonal_weights.weights,
                box_indices=nh_box,
                delta_global=delta_global,
                relative_to="ambient",
            )
            assert np.allclose(np.asarray(pred_std), np.asarray(pred_amb), rtol=1e-12, atol=1e-12)
            q0_std = float(tt.RadiocarbonModel.fluence_q0_years(fluence_std))
            q0_amb = float(
                tt.RadiocarbonModel.fluence_q0_years(
                    fluence_std, baseline_d14c_permil=delta_global
                )
            )
            assert q0_std / q0_amb == pytest.approx(ambient_factor, rel=1e-12)

    def test_six_holocene_miyake_events_baseline_and_yield_scaling(
        self, six_event_fits: dict[str, dict[str, object]]
    ) -> None:
        q0_std_map: dict[str, float] = {}
        q0_amb_map: dict[str, float] = {}
        delta0_map: dict[str, float] = {}

        for ev in MIYAKE_EVENTS:
            entry = six_event_fits[ev]
            res: tt.MAPResult = entry["map_res"]  # type: ignore[assignment]
            lp: tt.LinearPosteriorResult = entry["linear_post"]  # type: ignore[assignment]
            fl = float(res.theta_map["production.spike.fluence"])
            d0 = float(lp.baseline_d14c)
            q_std = float(tt.RadiocarbonModel.fluence_q0_years(fl))
            q_amb = float(tt.RadiocarbonModel.fluence_q0_years(fl, baseline_d14c_permil=d0))
            assert (q_std - q_amb) / q_amb == pytest.approx(d0 / 1000.0, rel=1e-12)
            q0_std_map[ev] = q_std
            q0_amb_map[ev] = q_amb
            delta0_map[ev] = d0

        # Near-modern events (775CE, 993CE, 660BCE): |Delta_global| <= ~25 permil (~0.3%-2.5% shift)
        for ev in ("775CE", "993CE"):
            rel_shift = abs(q0_amb_map[ev] - q0_std_map[ev]) / q0_std_map[ev]
            assert -28.0 <= delta0_map[ev] <= -12.0
            assert 0.012 <= rel_shift <= 0.028
        assert abs(delta0_map["660BCE"]) <= 18.0
        dur_660 = float(
            six_event_fits["660BCE"]["map_res"].theta_map["production.spike.duration"]  # type: ignore[attr-defined]
        )
        assert dur_660 > 1.2

        # Mid-Holocene events (5259BCE, 5410BCE, 7176BCE): Delta_global ~ +75 to +95 permil (~7.2%-9.5% shift)
        for ev in ("5259BCE", "5410BCE", "7176BCE"):
            assert 70.0 <= delta0_map[ev] <= 105.0
            rel_excess_over_amb = (q0_std_map[ev] - q0_amb_map[ev]) / q0_amb_map[ev]
            rel_reduction_from_std = (q0_std_map[ev] - q0_amb_map[ev]) / q0_std_map[ev]
            assert 0.070 <= rel_excess_over_amb <= 0.105
            assert 0.065 <= rel_reduction_from_std <= 0.096

        # Giant-event convergence in epoch-local q0(t_ev)-years
        giants_std = [q0_std_map[e] for e in ("775CE", "5259BCE", "7176BCE")]
        giants_amb = [q0_amb_map[e] for e in ("775CE", "5259BCE", "7176BCE")]
        spread_std = (max(giants_std) - min(giants_std)) / float(np.mean(giants_std))
        spread_amb = (max(giants_amb) - min(giants_amb)) / float(np.mean(giants_amb))
        assert spread_amb < spread_std
        assert all(3.3 <= g <= 4.0 for g in giants_amb)


# ===========================================================================
# 4. MacKay error_scale(..., by_series=True) & kalman_gp_marginal on Bundled Data
# ===========================================================================


class TestMacKayErrorScaleAndKalmanGPMarginal:
    def test_mackay_error_scale_by_series_on_bundled_events(
        self, brehm21_cbm: tt.CarbonBoxModel
    ) -> None:
        for ev, sp_t0, sp_fl in (("993CE", 993.0, 4.0), ("5259BCE", -5258.7, 7.2)):
            ds = tt.load_bundled_event(ev)
            t_min, t_max = ds.time_span
            f_prod = tt.ProductionSum(
                steady=tt.SteadyProduction(),
                field=tt.GaussianField(
                    jnp.zeros(96),
                    t_min - 5.0,
                    t_max + 2.0,
                    sigma=0.12,
                    length_yr=6.0,
                    order=2.0,
                ),
                spike=tt.MiyakeSpike(
                    t_start=sp_t0, duration=0.5, fluence=sp_fl, profile="boxcar"
                ),
            )
            f_model = tt.RadiocarbonModel(brehm21_cbm, f_prod)
            fit_res = tt.fit_ift_field(f_model, ds)
            s_glob = float(tt.error_scale(fit_res, ds, by_series=False))
            scales_by_s = tt.error_scale(fit_res, ds, by_series=True)

            assert math.isfinite(s_glob) and 0.5 <= s_glob <= 3.0
            assert isinstance(scales_by_s, dict)
            assert set(scales_by_s.keys()) == set(ds.series_names)
            assert all(math.isfinite(v) and v > 0.0 for v in scales_by_s.values())

            rescaled_ds = ds.with_error_scale(scales_by_s)
            for orig_s, new_s in zip(ds.series, rescaled_ds.series):
                assert np.allclose(
                    np.asarray(new_s.sigma_d14c),
                    np.asarray(orig_s.sigma_d14c) * scales_by_s[orig_s.name],
                    rtol=1e-12,
                )

            if ev == "993CE":
                assert scales_by_s["Büntgen18_SWE02"] > scales_by_s["Büntgen18_CHI01"]

    def test_kalman_gp_marginal_on_intcal20_and_multiseries_993ce(
        self,
        brehm21_cbm: tt.CarbonBoxModel,
        six_event_fits: dict[str, dict[str, object]],
    ) -> None:
        steady_m = tt.RadiocarbonModel(brehm21_cbm, tt.SteadyProduction())
        a_mat = brehm21_cbm.matrix_A()
        b_vec = brehm21_cbm.production_vector()
        y_ss = brehm21_cbm.steady_state_c14(
            q_std=steady_m.q_std, alpha_unit=steady_m.alpha_unit
        )
        trop_idx = jnp.asarray(brehm21_cbm.topology.trop_box_indices, dtype=jnp.int32)
        y_trop = jnp.sum(y_ss[trop_idx])
        m_rel = brehm21_cbm.relative_reservoir_masses()
        r_ss = (y_ss / y_trop) / m_rel
        nh_box = brehm21_cbm.box_index("Troposphere", "north")
        obs_scale_nh = float(1000.0 / (y_trop * m_rel[nh_box]))
        ss_nh = float(1000.0 * (r_ss[nh_box] - 1.0))

        # (a) Holocene slice of bundled IntCal20 [765, 785] CE
        ic = tt.load_bundled_event("IntCal20").series[0]
        mask = (np.asarray(ic.years_ce) >= 765.0) & (np.asarray(ic.years_ce) <= 785.0)
        resid_ic = ic.d14c[mask] - ss_nh
        sig_ic = ic.sigma_d14c[mask]
        ts_ic = ic.t_starts[mask]
        te_ic = ic.t_ends[mask]
        kern = tk.quasisep.Matern32(scale=6.0, sigma=0.15 * steady_m.q_std)
        nuis_ic = jnp.ones((resid_ic.shape[0], 1), dtype=jnp.float64)
        nuis_sd_ic = jnp.array([150.0], dtype=jnp.float64)

        kal_ic = tt.kalman_gp_marginal(
            a_mat,
            b_vec,
            kern,
            resid_ic,
            sig_ic,
            ts_ic,
            te_ic,
            dt=1.0 / 24.0,
            alpha_unit=steady_m.alpha_unit,
            box_indices=nh_box,
            obs_scale=obs_scale_nh,
            nuisance_design=nuis_ic,
            nuisance_prior_sd=nuis_sd_ic,
        )
        assert bool(jnp.isfinite(kal_ic.loglike))
        assert bool(jnp.all(kal_ic.production_var > 0.0))
        assert bool(jnp.all(kal_ic.cbm_state_var > 0.0))
        assert kal_ic.nuisance_mean is not None and kal_ic.nuisance_mean.shape == (1,)

        proj_ic = tt.project_causal_gp(
            a_mat,
            b_vec,
            ts_ic,
            te_ic,
            n_modes=48,
            kernel=kern,
            alpha_unit=steady_m.alpha_unit,
            box_indices=nh_box,
            obs_scale=obs_scale_nh,
        )
        lm_ic = proj_ic.to_linear_marginal(
            nuisance_design=nuis_ic, nuisance_prior_sd=nuis_sd_ic
        )
        ll_pod = float(lm_ic.loglike(resid_ic, sig_ic))
        assert float(kal_ic.loglike) == pytest.approx(ll_pod, abs=1.5)

        # (b) Multi-series NH+SH 993CE after subtracting MAP MiyakeSpike + NH/SH steady state
        fit_993 = six_event_fits["993CE"]
        ds_993: tt.Dataset = fit_993["dataset"]  # type: ignore[assignment]
        res_993: tt.MAPResult = fit_993["map_res"]  # type: ignore[assignment]
        sp_993 = res_993.fitted_model.production.spike
        sp_only_m = tt.RadiocarbonModel(
            brehm21_cbm,
            tt.ProductionSum(steady=tt.SteadyProduction(), spike=sp_993),
        )
        pred_sp = sp_only_m.predict_d14c(ds_993, concat=True)
        resid_993 = ds_993.d14c_flat - pred_sp

        box_idx_993 = jnp.array(
            [steady_m._box_index_for_series(s) for s in ds_993.series for _ in range(s.n_obs)],
            dtype=jnp.int32,
        )
        obs_scale_993 = 1000.0 / (y_trop * m_rel[box_idx_993])
        x_off, pm_off, ps_off = tt.hierarchical_offset_design(
            ds_993.series_indices,
            n_series=ds_993.n_series,
            box_activity_ratios=r_ss[box_idx_993],
        )
        kal_993 = tt.kalman_gp_marginal(
            a_mat,
            b_vec,
            tk.quasisep.Matern32(scale=6.0, sigma=0.08 * steady_m.q_std),
            resid_993,
            ds_993.sigma_flat,
            ds_993.t_starts_flat,
            ds_993.t_ends_flat,
            dt=1.0 / 24.0,
            alpha_unit=steady_m.alpha_unit,
            box_indices=box_idx_993,
            obs_scale=obs_scale_993,
            nuisance_design=x_off,
            nuisance_prior_mean=pm_off,
            nuisance_prior_sd=ps_off,
        )
        assert bool(jnp.isfinite(kal_993.loglike))
        assert kal_993.nuisance_mean is not None
        assert kal_993.nuisance_mean.shape == (1 + ds_993.n_series,)


# ===========================================================================
# 5. Notebook 05 Structure & Execution Audit (stdlib json)
# ===========================================================================


class TestHoloceneReanalysisNotebookAudit:
    def test_notebook_05_five_cell_structure_and_outputs(self) -> None:
        nb_path = ROOT / "notebooks" / "v2" / "05_holocene_miyake_reanalysis.ipynb"
        assert nb_path.exists(), f"Missing notebook: {nb_path}"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        assert nb.get("nbformat") == 4
        cells = nb.get("cells", [])
        assert len(cells) >= 11
        assert cells[0].get("cell_type") == "markdown"
        assert cells[-1].get("cell_type") == "markdown"

        exec_counts: list[int] = []
        n_png_cells = 0
        prev_type = None
        for idx, cell in enumerate(cells):
            ctype = cell.get("cell_type")
            src = "".join(cell.get("source", []))
            assert ".png" not in src and "savefig" not in src
            if ctype == "markdown":
                assert "—" not in src, f"Markdown cell {idx} contains em-dash instead of OUP en-dash"
            elif ctype == "code":
                assert prev_type == "markdown", f"Code cell {idx} not preceded by markdown"
                ec = cell.get("execution_count")
                assert isinstance(ec, int)
                exec_counts.append(ec)
                outputs = cell.get("outputs", [])
                assert len(outputs) == 1, f"Code cell {idx} must emit strictly 1 output, got {len(outputs)}"
                out = outputs[0]
                assert out.get("output_type") != "error"
                if out.get("output_type") in ("display_data", "execute_result") and "image/png" in out.get("data", {}):
                    assert len(out["data"]["image/png"]) > 1000
                    n_png_cells += 1
                else:
                    assert out.get("output_type") == "stream" and out.get("name") == "stdout"
                    assert len("".join(out.get("text", []))) > 10
            prev_type = ctype

        assert exec_counts == [1, 2, 3, 4, 5]
        assert n_png_cells >= 2
        assert list((ROOT / "notebooks" / "v2").glob("*.png")) == []
