"""Comprehensive unit and scientific regression tests for ``presets``, ``data``, and ``plotting``."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# On standalone Wave 1 branches before `src/ticktack/__init__.py` is updated,
# legacy `ticktack.py` imports removed `pkg_resources` on package init.
if importlib.util.find_spec("pkg_resources") is None:
    sys.modules["pkg_resources"] = types.ModuleType("pkg_resources")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ticktack.data import (
    BUNDLED_DATASET_CATALOGUE,
    Dataset,
    SeasonalWeights,
    TreeRingSeries,
    convert_calendar_years,
    load_bundled_event,
    load_bundled_series,
    parse_growth_season_bounds,
)
from ticktack.plotting import (
    plot_cbm_eigenmodes,
    plot_d14c_fit,
    plot_fisher_spectrum,
    plot_production_posterior,
)
from ticktack.presets import (
    LEGACY_V1_C14_OVER_C12_RATIO,
    PRESET_NAMES,
    PresetSpec,
    get_preset_spec,
    load_preset_hdf5,
    save_preset_hdf5,
)

# ---------------------------------------------------------------------------
# 1. Preset Topology, Kirchhoff Balance, Hurwitz Stability, & Symmetry Tying
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", PRESET_NAMES)
def test_kirchhoff_balance_and_hurwitz_stability_all_presets(preset_name: str) -> None:
    spec = get_preset_spec(preset_name)
    assert isinstance(spec, PresetSpec)

    # Strict 12C Kirchhoff circulation balance < 1e-13 across every reservoir
    imbalance = spec.kirchhoff_imbalance()
    assert imbalance.dtype == jnp.float64
    assert float(jnp.max(jnp.abs(imbalance))) < 1e-13

    # Column-conservation of Metzler transfer-rate matrix K: 1^T K == 0
    k_mat = spec.rate_matrix_K()
    assert k_mat.dtype == jnp.float64
    assert float(jnp.max(jnp.abs(jnp.sum(k_mat, axis=0)))) < 1e-14
    off_diag = k_mat - jnp.diag(jnp.diag(k_mat))
    assert bool(jnp.all(off_diag >= 0.0))

    # Strict Hurwitz stability: Re(lambda_k(A)) <= -decay_rate < 0
    decay_cambridge = float(np.log(2.0) / 5730.0)
    a_mat = spec.matrix_A(decay_rate=decay_cambridge)
    eigvals = np.linalg.eigvals(np.asarray(a_mat))
    assert np.max(eigvals.real) <= -decay_cambridge + 1e-14

    # Positive steady-state 14C inventories
    y_ss = spec.steady_state_c14()
    assert y_ss.dtype == jnp.float64
    assert bool(jnp.all(y_ss > 0.0))


def test_buntgen18_filters_zero_flux_placeholder_and_symmetric_groups() -> None:
    brehm = get_preset_spec("Brehm21")
    buntgen = get_preset_spec("Buntgen18")

    # Brehm21 has 56 positive edges; Buntgen18 filters out f36 = Flow(SwN, SwS, 0) -> 55 edges
    assert brehm.n_boxes == 22
    assert buntgen.n_boxes == 22
    assert brehm.n_edges == 56
    assert buntgen.n_edges == 55
    assert all(flux > 0.0 for _, _, flux in buntgen.edges)
    # Specifically verify (SwN=13 -> SwS=2) is absent in Buntgen18 and present in Brehm21
    assert (13, 2) in {(s, d) for s, d, _ in brehm.edges}
    assert (13, 2) not in {(s, d) for s, d, _ in buntgen.edges}

    # Verify N/S symmetry-tying groups reproduce exact literature transfer rates
    for spec in (brehm, buntgen):
        assert spec.symmetric_edge_groups is not None
        assert len(spec.symmetric_edge_groups) >= 15
        tie_mat = spec.edge_tie_matrix(tie_symmetric_edges=True)
        assert tie_mat.shape[0] == spec.n_edges
        assert tie_mat.shape[1] < spec.n_edges
        # Row sums must be 1 (each edge belongs to exactly one free parameter)
        np.testing.assert_allclose(np.sum(np.asarray(tie_mat), axis=1), 1.0, atol=1e-15)

        masses = np.asarray(spec.masses, dtype=np.float64)
        log_rates = np.array([np.log(f / masses[s]) for s, _, f in spec.edges])
        theta_free, *_ = np.linalg.lstsq(np.asarray(tie_mat), log_rates, rcond=None)
        reconstructed = np.asarray(tie_mat) @ theta_free
        np.testing.assert_allclose(reconstructed, log_rates, atol=1e-13, rtol=1e-13)


def test_brehm21_preserves_interhemispheric_gradient() -> None:
    brehm = get_preset_spec("Brehm21")
    y_ss = brehm.steady_state_c14()
    r_ref = brehm.reference_specific_activity()
    masses = jnp.asarray(brehm.masses, dtype=jnp.float64)

    idx_tps = brehm.box_index("Troposphere", "south")
    idx_tpn = brehm.box_index("Troposphere", "north")
    d14c_south = 1000.0 * ((y_ss[idx_tps] / masses[idx_tps]) / r_ref - 1.0)
    d14c_north = 1000.0 * ((y_ss[idx_tpn] / masses[idx_tpn]) / r_ref - 1.0)
    nh_minus_sh_permil = float(d14c_north - d14c_south)

    # Southern Ocean deep-water upwelling depletes TpS relative to TpN by ~4.38 permil
    assert nh_minus_sh_permil == pytest.approx(4.378, abs=0.05)


# ---------------------------------------------------------------------------
# 2. Miyake17 Double-Bug Fix vs Legacy v1 Regression
# ---------------------------------------------------------------------------


def test_miyake17_rate_fix_and_r_ref_calibration_vs_legacy_v1() -> None:
    guttler = get_preset_spec("Guttler15")
    miyake_fixed = get_preset_spec("Miyake17")  # default: legacy_v1_units=False, open_ocean_sink
    miyake_closed = get_preset_spec(
        "Miyake17", legacy_v1_units=False, miyake17_closure="closed_shallow"
    )
    miyake_v1 = get_preset_spec("Miyake17", legacy_v1_units=True)

    # 1. Verify 14.003242/12 rate inflation is removed by default
    k_fixed = np.asarray(miyake_fixed.rate_matrix_K())
    assert k_fixed[1, 0] == pytest.approx(0.5, rel=1e-15)
    assert k_fixed[2, 1] == pytest.approx(1.0 / 11.0, rel=1e-15)
    assert k_fixed[3, 1] == pytest.approx(1.0 / 23.0, rel=1e-15)

    f_fixed = np.asarray(miyake_fixed.flux_matrix())
    assert f_fixed[0, 1] == pytest.approx(56.25, rel=1e-15)
    assert f_fixed[1, 0] == pytest.approx(56.25, rel=1e-15)
    assert f_fixed[1, 2] == pytest.approx(637.5 / 11.0, rel=1e-15)
    assert f_fixed[2, 1] == pytest.approx(637.5 / 11.0, rel=1e-15)
    assert f_fixed[1, 3] == pytest.approx(637.5 / 23.0, rel=1e-15)
    assert f_fixed[3, 1] == pytest.approx(637.5 / 23.0, rel=1e-15)

    # 2. Verify open deep-ocean sink on Marine surface (box 2) matches Guttler15 R_ref
    assert miyake_fixed.sink_rates[0] == 0.0
    assert miyake_fixed.sink_rates[1] == 0.0
    assert miyake_fixed.sink_rates[2] > 0.0
    assert miyake_fixed.sink_rates[3] == 0.0
    assert 0.004 < miyake_fixed.sink_rates[2] < 0.008

    r_ref_guttler = float(guttler.reference_specific_activity())
    r_ref_miyake_fixed = float(miyake_fixed.reference_specific_activity())
    assert r_ref_guttler == pytest.approx(1.3195, rel=5e-3)
    assert r_ref_miyake_fixed == pytest.approx(r_ref_guttler, rel=1e-14)

    # 3. Verify closed_shallow exhibits the ~13-16x over-inventory suppression hazard
    r_ref_closed = float(miyake_closed.reference_specific_activity())
    assert r_ref_closed / r_ref_miyake_fixed > 12.0

    # 4. Verify legacy_v1_units=True reproduces exact ticktack 1.x rate inflation & y_ss
    k_v1 = np.asarray(miyake_v1.rate_matrix_K())
    assert k_v1[1, 0] == pytest.approx(0.5 * LEGACY_V1_C14_OVER_C12_RATIO, rel=1e-15)
    assert k_v1[2, 1] == pytest.approx((1.0 / 11.0) * LEGACY_V1_C14_OVER_C12_RATIO, rel=1e-15)
    assert k_v1[3, 1] == pytest.approx((1.0 / 23.0) * LEGACY_V1_C14_OVER_C12_RATIO, rel=1e-15)
    assert miyake_v1.sink_rates == (0.0, 0.0, 0.0, 0.0)

    alpha_v1 = 14.003242 / 6.022 * 5.11 * 31536.0 / 1.0e5
    decay_v1 = float(np.log(2.0) / 5700.0)
    y_ss_v1 = np.asarray(
        miyake_v1.steady_state_c14(q_std=1.76, alpha_unit=alpha_v1, decay_rate=decay_v1)
    )
    assert y_ss_v1[1] == pytest.approx(10679.0, rel=2e-3)


@pytest.mark.parametrize("preset_name", PRESET_NAMES)
def test_hdf5_importlib_resources_loader_and_roundtrip(
    preset_name: str, tmp_path: Path
) -> None:
    spec_code = get_preset_spec(preset_name)
    spec_hd5 = load_preset_hdf5(preset_name)

    assert spec_hd5.box_names == spec_code.box_names
    assert spec_hd5.hemispheres == spec_code.hemispheres
    np.testing.assert_allclose(spec_hd5.masses, spec_code.masses, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(
        np.asarray(spec_hd5.flux_matrix()),
        np.asarray(spec_code.flux_matrix()),
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        spec_hd5.sink_rates, spec_code.sink_rates, rtol=1e-14, atol=1e-14
    )

    # Verify legacy Miyake17.hd5 matches legacy_v1_units=True directly
    if preset_name == "Miyake17":
        spec_v1_code = get_preset_spec("Miyake17", legacy_v1_units=True)
        spec_v1_hd5 = load_preset_hdf5("Miyake17", legacy_v1_units=True)
        np.testing.assert_allclose(
            np.asarray(spec_v1_hd5.flux_matrix()),
            np.asarray(spec_v1_code.flux_matrix()),
            rtol=1e-14,
            atol=1e-14,
        )

    # Save and reload round-trip
    out_file = tmp_path / f"{preset_name}_v2.hd5"
    save_preset_hdf5(spec_code, out_file)
    reloaded = load_preset_hdf5(out_file)
    np.testing.assert_allclose(
        np.asarray(reloaded.flux_matrix()),
        np.asarray(spec_code.flux_matrix()),
        rtol=1e-15,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        reloaded.sink_rates, spec_code.sink_rates, rtol=1e-15, atol=1e-15
    )


# ---------------------------------------------------------------------------
# 3. Southern Hemisphere Calendar Conventions & Seasonal Quadrature
# ---------------------------------------------------------------------------


def test_seasonal_weights_bounds_and_subannual_arrays() -> None:
    nh = SeasonalWeights.from_string("april-september")
    assert nh.bounds == pytest.approx((0.25, 0.75), abs=1e-15)

    # Cross-New-Year SH seasons must require explicit year_convention
    with pytest.raises(ValueError, match="year_convention"):
        parse_growth_season_bounds("october-march")

    sh_schulman = SeasonalWeights.from_string(
        "october-march", year_convention="schulman_start"
    )
    assert sh_schulman.bounds == pytest.approx((0.75, 1.25), abs=1e-15)

    sh_end_oct_mar = SeasonalWeights.from_string(
        "october-march", year_convention="end_year"
    )
    assert sh_end_oct_mar.bounds == pytest.approx((-0.25, 0.25), abs=1e-15)

    sh_end_sep_feb = SeasonalWeights.from_string(
        "september-february", year_convention="end_year"
    )
    assert sh_end_sep_feb.bounds == pytest.approx((-4.0 / 12.0, 2.0 / 12.0), abs=1e-15)

    # Cyclic 12-month binary vector for Oct-Mar collapses to 1 exact sub-window
    oct_mar_mask = [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    sw_mask_schulman = SeasonalWeights.from_subannual_weights(
        oct_mar_mask, year_convention="schulman_start"
    )
    assert sw_mask_schulman.n_subwindows == 1
    assert sw_mask_schulman.bounds == pytest.approx((0.75, 1.25), abs=1e-14)

    sw_mask_end = SeasonalWeights.from_subannual_weights(
        oct_mar_mask, year_convention="end_year"
    )
    assert sw_mask_end.n_subwindows == 1
    assert sw_mask_end.bounds == pytest.approx((-0.25, 0.25), abs=1e-14)

    # Non-uniform subannual photosynthesis profile
    nonuniform = SeasonalWeights.from_subannual_weights(
        [0, 0, 0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0, 0, 0]
    )
    assert float(jnp.sum(nonuniform.weights)) == pytest.approx(1.0, abs=1e-15)
    assert float(nonuniform.midpoint_offset) == pytest.approx(0.5, abs=1e-15)


def test_sh_tas01_schulman_and_guttler15_kauri_end_year_align_to_same_summer() -> None:
    tas01 = load_bundled_series("775AD-early-S/Büntgen18_TAS01.csv")
    kauri = load_bundled_series("notebooks/775CE/Guttler15_Kauri.csv")

    assert tas01.hemisphere == "south"
    assert tas01.year_convention == "schulman_start"
    assert kauri.hemisphere == "south"
    assert kauri.year_convention == "end_year"

    # Locate the onset jump of the 774 CE cosmic-ray spike in each series.
    # In TAS01: baseline ~ -17.9 (773) -> jumps to -12.7 at nominal year 774, then -5.8 at 775.
    # In Kauri: baseline ~ -23.44 (774) -> jumps to -11.62 at nominal year 775, then -6.46 at 776.
    tas_774_idx = int(jnp.where(tas01.raw_years == 774.0)[0][0])
    kauri_775_idx = int(jnp.where(kauri.raw_years == 775.0)[0][0])
    kauri_jump_idx = int(jnp.argmax(jnp.diff(kauri.d14c))) + 1
    assert kauri_jump_idx == kauri_775_idx

    tas_onset_window = (
        float(tas01.t_starts[tas_774_idx]),
        float(tas01.t_ends[tas_774_idx]),
    )
    kauri_onset_window = (
        float(kauri.t_starts[kauri_775_idx]),
        float(kauri.t_ends[kauri_775_idx]),
    )

    # Both map to the EXACT same physical Austral summer: Oct 774 - Mar 775 CE (774.75 .. 775.25)!
    assert tas_onset_window == pytest.approx((774.75, 775.25), abs=1e-14)
    assert kauri_onset_window == pytest.approx((774.75, 775.25), abs=1e-14)
    assert tas_onset_window == pytest.approx(kauri_onset_window, abs=1e-14)

    # Peak rings (TAS01 year=775 and Kauri year=776) also align to 775.75 .. 776.25
    tas_peak_idx = int(jnp.argmax(tas01.d14c))
    kauri_peak_idx = int(jnp.argmax(kauri.d14c))
    assert float(tas01.raw_years[tas_peak_idx]) == 775.0
    assert float(kauri.raw_years[kauri_peak_idx]) == 776.0
    assert float(tas01.t_starts[tas_peak_idx]) == pytest.approx(
        float(kauri.t_starts[kauri_peak_idx]), abs=1e-14
    )
    assert float(tas01.t_ends[tas_peak_idx]) == pytest.approx(
        float(kauri.t_ends[kauri_peak_idx]), abs=1e-14
    )

    # Loading an SH CSV directly without year_convention must raise ValueError
    sh_csv = REPO_ROOT / "notebooks" / "775CE" / "Guttler15_Kauri.csv"
    with pytest.raises(ValueError, match="year_convention"):
        TreeRingSeries.from_csv(sh_csv, hemisphere="south")


# ---------------------------------------------------------------------------
# 4. Calendar Conversions, Explicit Sub-Annual Bounds, & zodiax Dataset
# ---------------------------------------------------------------------------


def test_calendar_conversions_and_historical_bce_year_zero_guard() -> None:
    # 1. Astronomical CE identity
    ce = convert_calendar_years([-5258.0, 0.0, 775.0], calendar="astronomical_ce")
    assert ce.dtype == jnp.float64
    np.testing.assert_allclose(np.asarray(ce), [-5258.0, 0.0, 775.0], atol=1e-15)

    # 2. Calibrated BP (1950.0 - bp)
    bp = convert_calendar_years([0.0, 1175.0, 1950.0, 7208.0], calendar="cal_bp")
    np.testing.assert_allclose(
        np.asarray(bp), [1950.0, 775.0, 0.0, -5258.0], atol=1e-15
    )

    # 3. Historical BCE: negative Y -> Y + 1, positive Y -> Y, uniform spacing across 1 BCE / 1 CE
    hist = convert_calendar_years([-7176, -5259, -664, -2, -1, 1, 2, 775], calendar="historical_bce")
    np.testing.assert_allclose(
        np.asarray(hist),
        [-7175.0, -5258.0, -663.0, -1.0, 0.0, 1.0, 2.0, 775.0],
        atol=1e-15,
    )
    # Consecutive historical years 2 BCE (-2), 1 BCE (-1), 1 CE (+1) are spaced by 1.00 yr
    np.testing.assert_allclose(np.diff(np.asarray(hist)[3:7]), [1.0, 1.0, 1.0], atol=1e-15)

    # Year 0 does not exist in historical_bce -> must raise ValueError
    with pytest.raises(ValueError, match="Year 0"):
        convert_calendar_years([-1, 0, 1], calendar="historical_bce")


def test_explicit_bounds_bundled_catalogue_and_zodiax_dataset() -> None:
    # Explicit sub-annual bounds
    ew_lw = TreeRingSeries(
        years=[774, 774, 775, 775],
        d14c=[-18.0, -15.0, -4.0, -1.0],
        sigma_d14c=[1.5, 1.5, 1.5, 1.5],
        hemisphere="north",
        calendar="astronomical_ce",
        year_convention="explicit_bounds",
        t_starts=[774.25, 774.50, 775.25, 775.50],
        t_ends=[774.50, 774.75, 775.50, 775.75],
        name="subannual_test",
    )
    assert ew_lw.n_obs == 4
    np.testing.assert_allclose(
        np.asarray(ew_lw.t_midpoints), [774.375, 774.625, 775.375, 775.625], atol=1e-15
    )

    # Audit Sakurai20 660 BCE earlywood & latewood ordering
    cedar_ew = load_bundled_series("660BCE-Ew/Sakurai20_CedarEw.csv")
    cedar_lw = load_bundled_series("660BCE-Lw/Sakurai20_CedarLw.csv")
    assert cedar_ew.calendar == "historical_bce"
    # Historical -664 -> astronomical -663; Ew [0.25, 0.50] precedes Lw [0.50, 0.75]
    idx_664 = int(jnp.where(cedar_ew.raw_years == -664.0)[0][0])
    assert float(cedar_ew.years_ce[idx_664]) == -663.0
    assert float(cedar_ew.t_ends[idx_664]) == pytest.approx(
        float(cedar_lw.t_starts[idx_664]), abs=1e-15
    )

    # Audit 5259 BCE and IntCal20 sorting
    ds_5259 = load_bundled_event("5259BCE")
    assert ds_5259.n_series == 4
    intcal = load_bundled_series("IntCal20/Intcal20.csv")
    assert bool(jnp.all(jnp.diff(intcal.t_starts) >= 0.0))

    # Check zodiax PyTree addressing and JIT compatibility on Dataset
    ds = Dataset([cedar_ew, cedar_lw], name="660BCE_joint")
    assert ds.n_series == 2
    assert ds.n_obs == cedar_ew.n_obs + cedar_lw.n_obs
    h_lab = ds.lab_offset_design_matrix()
    assert h_lab.shape == (ds.n_obs, 2)
    np.testing.assert_allclose(np.sum(np.asarray(h_lab), axis=1), 1.0, atol=1e-15)

    # zodiax .get / .set roundtrip and jax.jit
    got_d14c = ds.get("series")[0].d14c
    np.testing.assert_allclose(np.asarray(got_d14c), np.asarray(cedar_ew.d14c))

    @jax.jit
    def total_weighted_mean(dataset: Dataset) -> jax.Array:
        w = 1.0 / jnp.square(dataset.sigma_flat)
        return jnp.sum(w * dataset.d14c_flat) / jnp.sum(w)

    val = float(total_weighted_mean(ds))
    assert np.isfinite(val)

    # Confirm all 64 catalogued CSVs exist and load cleanly
    assert len(BUNDLED_DATASET_CATALOGUE) == 64
    for k in BUNDLED_DATASET_CATALOGUE:
        s = load_bundled_series(k)
        assert s.n_obs > 0


# ---------------------------------------------------------------------------
# 5. Stateless Plotting Smoke Tests (Agg Backend)
# ---------------------------------------------------------------------------


def test_stateless_plotting_functions_agg() -> None:
    rc_before = dict(mpl.rcParams)

    tas01 = load_bundled_series("775AD-early-S/Büntgen18_TAS01.csv")
    mon03 = load_bundled_series("775AD-late-N/Büntgen18_MON03.csv")
    ds = Dataset([mon03, tas01], name="775CE_NH_SH")

    t_fine = np.linspace(769.5, 781.5, 240)
    nh_curve = -19.0 + 18.0 * (t_fine > 774.5) * np.exp(-(t_fine - 774.5) / 8.0)
    sh_curve = nh_curve - 4.38

    # 1. plot_d14c_fit (multi-hemisphere + residuals)
    fig1, axes1 = plot_d14c_fit(
        ds,
        t_fine=t_fine,
        d14c_pred={"north": nh_curve, "south": sh_curve},
        d14c_band={
            "north": (nh_curve - 1.2, nh_curve + 1.2),
            "south": (sh_curve - 1.2, sh_curve + 1.2),
        },
        show_residuals=True,
        title="774/775 CE NH+SH Fit",
    )
    assert isinstance(fig1, mpl.figure.Figure)
    assert len(axes1) == 2
    fig1.canvas.draw()
    plt.close(fig1)

    # Also test 2D sample draws on single-panel plot_d14c_fit
    rng = np.random.default_rng(42)
    draws_d14c = nh_curve[None, :] + rng.normal(0.0, 0.8, size=(25, t_fine.size))
    fig1b, ax1b = plot_d14c_fit(mon03, t_fine=t_fine, d14c_pred=draws_d14c)
    assert isinstance(fig1b, mpl.figure.Figure)
    assert isinstance(ax1b, mpl.axes.Axes)
    fig1b.canvas.draw()
    plt.close(fig1b)

    # 2. plot_production_posterior (Gaussian POD mean+std and 2D MCMC draws)
    q_mean = 1.76 + 6.0 * np.exp(-0.5 * ((t_fine - 774.6) / 0.15) ** 2)
    q_std_arr = np.full_like(t_fine, 0.25)
    fig2, ax2 = plot_production_posterior(
        t_fine, q_mean, std=q_std_arr, q_std=1.76, truth=q_mean
    )
    assert isinstance(fig2, mpl.figure.Figure)
    assert isinstance(ax2, mpl.axes.Axes)
    fig2.canvas.draw()
    plt.close(fig2)

    q_samples = q_mean[None, :] + rng.normal(0.0, 0.2, size=(30, t_fine.size))
    fig2b, ax2b = plot_production_posterior(
        t_fine, q_samples, q_std=1.76, n_spaghetti=8
    )
    assert isinstance(ax2b, mpl.axes.Axes)
    fig2b.canvas.draw()
    plt.close(fig2b)

    # 3. plot_fisher_spectrum
    eigenvalues = np.array([420.0, 38.0, 4.2, 0.45, 0.03, 1e-4, 1e-9])
    analytical = 450.0 * np.exp(-np.pi * np.arange(eigenvalues.size))
    fig3, axes3 = plot_fisher_spectrum(
        eigenvalues,
        analytical_spectrum=analytical,
        title="CBM Fisher Spectrum",
    )
    assert isinstance(fig3, mpl.figure.Figure)
    assert len(axes3) == 2
    fig3.canvas.draw()
    plt.close(fig3)

    # 4. plot_cbm_eigenmodes (discrete bars, heatmap, and continuous log-tau curves)
    modes_discrete = rng.standard_normal((11, 5))
    spec_guttler = get_preset_spec("Guttler15")
    fig4a, ax4a = plot_cbm_eigenmodes(
        modes_discrete,
        eigenvalues=eigenvalues[:5],
        n_modes=3,
        param_names=spec_guttler.box_names,
        style="bars",
    )
    assert isinstance(ax4a, mpl.axes.Axes)
    fig4a.canvas.draw()
    plt.close(fig4a)

    fig4b, ax4b = plot_cbm_eigenmodes(
        rng.standard_normal((28, 6)),
        eigenvalues=eigenvalues[:6],
        n_modes=4,
        style="heatmap",
    )
    assert isinstance(ax4b, mpl.axes.Axes)
    fig4b.canvas.draw()
    plt.close(fig4b)

    tau_grid = np.geomspace(0.1, 2000.0, 64)
    modes_cont = np.column_stack(
        [np.sin((k + 1) * np.log(tau_grid / 0.1) / 3.5) for k in range(4)]
    )
    fig4c, ax4c = plot_cbm_eigenmodes(
        modes_cont,
        eigenvalues=eigenvalues[:4],
        n_modes=4,
        x_values=tau_grid,
    )
    assert isinstance(ax4c, mpl.axes.Axes)
    fig4c.canvas.draw()
    plt.close(fig4c)

    # Confirm zero global rcParams side effects
    assert dict(mpl.rcParams) == rc_before
