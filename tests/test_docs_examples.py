"""Tests for v2/docs-examples: correlation & residual-map plots, whitespace CSVs, and notebooks."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import ticktack as tt

ROOT = Path(__file__).resolve().parents[1]

DOCS_EXAMPLE_NOTEBOOKS = (
    "notebooks/v2/docs_examples/01_single_series_774ce.ipynb",
    "notebooks/v2/docs_examples/02_multiseries_774ce_nh_sh.ipynb",
    "notebooks/v2/docs_examples/03_nonparametric_inverse_reconstruction.ipynb",
    "notebooks/v2/docs_examples/04_gp_injection_recovery.ipynb",
    "notebooks/v2/docs_examples/05_flat_baseline_smc_evidence.ipynb",
    "notebooks/v2/docs_examples/06_direct_cbm_inference.ipynb",
)


class TestWhitespaceCSVFallback:
    def test_miyake12_space_delimited_matches_comma_roundtrip(self, tmp_path: Path):
        ws_src = ROOT / "notebooks" / "miyake12.csv"
        cedar_ws = tt.TreeRingSeries.from_csv(
            ws_src,
            hemisphere="north",
            growth_season="april-september",
            name="Miyake12_Cedar",
        )
        comma_copy = tmp_path / "miyake12_comma.csv"
        comma_copy.write_text(
            "\n".join(",".join(ln.split()) for ln in ws_src.read_text(encoding="utf-8").splitlines() if ln.strip()) + "\n",
            encoding="utf-8",
        )
        cedar_comma = tt.TreeRingSeries.from_csv(
            comma_copy,
            hemisphere="north",
            growth_season="april-september",
            name="Miyake12_Cedar",
        )
        assert cedar_ws.n_obs == 28
        np.testing.assert_allclose(cedar_ws.years_ce, np.arange(760.0, 788.0))
        assert cedar_ws.d14c[0] == pytest.approx(-21.63)
        assert cedar_ws.d14c[15] == pytest.approx(-9.335)
        assert cedar_ws.d14c[-1] == pytest.approx(-14.44)
        np.testing.assert_allclose(cedar_ws.d14c, cedar_comma.d14c, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            cedar_ws.sigma_d14c, cedar_comma.sigma_d14c, rtol=1e-12, atol=1e-12
        )

    def test_tabs_comments_and_missing_column_errors(self, tmp_path: Path):
        ws_file = tmp_path / "tab_series.txt"
        ws_file.write_text(
            "\n  year\td14c   sigma_d14c  growth_season\n"
            "# comment row\n"
            "  774   -14.2   1.5   june-august\n"
            "  775    +3.8   1.4   june-august\n",
            encoding="utf-8",
        )
        s = tt.TreeRingSeries.from_csv(ws_file, hemisphere="north")
        assert s.n_obs == 2
        assert s.seasonal_weights.label == "june-august"
        np.testing.assert_allclose(s.d14c, [-14.2, 3.8])

        bad_file = tmp_path / "missing_cols.txt"
        bad_file.write_text("year d14c\n774 -14.0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="sig_d14c"):
            tt.TreeRingSeries.from_csv(bad_file, hemisphere="north")


class TestDiagnosticPlottingHelpers:
    def test_plot_data_model_correlation_structured_and_raw(self):
        rc_before = dict(plt.rcParams)
        s_nh = tt.TreeRingSeries(
            [773.0, 774.0, 775.0, 776.0],
            [-15.0, -14.5, 2.0, 4.5],
            [1.2, 1.1, 1.3, 1.2],
            hemisphere="north",
            name="NH_Oak",
        )
        s_sh = tt.TreeRingSeries(
            [773.0, 774.0, 775.0],
            [-19.2, -18.8, -2.5],
            [1.4, 1.5, 1.3],
            hemisphere="south",
            year_convention="schulman_start",
            name="SH_Kauri",
        )
        ds = tt.Dataset([s_nh, s_sh], name="pair")
        pred = np.array([-14.8, -14.7, 1.6, 4.1, -19.0, -18.5, -2.8])

        fig, ax = tt.plot_data_model_correlation(ds, pred, title="1:1 Check")
        try:
            assert ax.get_title() == "1:1 Check"
            xlo, xhi = ax.get_xlim()
            ylo, yhi = ax.get_ylim()
            assert (xlo, xhi) == pytest.approx((ylo, yhi))
            assert any(ln.get_linestyle() == "--" for ln in ax.lines)
            texts = "\n".join(t.get_text() for t in ax.texts)
            assert "chi^2" in texts and "WRMS" in texts
        finally:
            plt.close(fig)

        # Multi-series (>6) hemisphere grouping + caller-supplied Axes + raw array mode
        many = tt.Dataset(
            [
                tt.TreeRingSeries([774.0, 775.0], [-14.0 + i, 2.0 + i], [1.0, 1.0], hemisphere="north", name=f"S{i}")
                for i in range(7)
            ]
            + [s_sh]
        )
        fig2, ext_axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
        try:
            tt.plot_data_model_correlation(many, many.d14c_flat, ax=ext_axes[0])
            tt.plot_data_model_correlation(np.array([1.0, 2.0, 3.0]), np.array([1.1, 1.9, 3.05]), ax=ext_axes[1])
            assert len(ext_axes[0].get_legend().get_texts()) == 3  # NH, SH, 1:1 identity
        finally:
            plt.close(fig2)

        with pytest.raises(ValueError, match="length"):
            f_err, ax_err = plt.subplots()
            try:
                tt.plot_data_model_correlation(ds, pred[:-1], ax=ax_err)
            finally:
                plt.close(f_err)

        assert dict(plt.rcParams) == rc_before

    def test_plot_residual_map_symmetry_nan_masking_and_colourbar(self):
        s0 = tt.TreeRingSeries(
            [773.0, 774.0, 776.0],
            [-14.0, -13.0, 5.0],
            [1.0, 2.0, 1.0],
            hemisphere="north",
            name="Cedar",
        )
        s1 = tt.TreeRingSeries(
            [775.0, 776.0],
            [-18.0, -2.0],
            [1.5, 2.0],
            hemisphere="south",
            year_convention="end_year",
            name="Kauri",
        )
        ds = tt.Dataset([s0, s1], name="gapped")
        # s0 years_start: 773, 774, 776; s1 end_year 775, 776 -> t_starts 774.75, 775.75 -> cols 774, 775!
        pred = np.array([-15.0, -9.0, 5.5, -15.0, -2.0])

        fig, ax = tt.plot_residual_map(ds, pred, zscore=True, cmap="RdBu_r", title="Pull Map")
        try:
            assert len(fig.axes) == 2  # main axes + colourbar axes
            im = ax.images[0]
            vmin, vmax = im.get_clim()
            assert vmin == pytest.approx(-vmax)
            assert vmax >= 3.0
            arr = np.ma.filled(im.get_array(), np.nan)
            assert arr.shape == (2, 4)  # years 773, 774, 775, 776
            assert np.isnan(arr[0, 2])  # Cedar missing 775
            assert np.isnan(arr[1, 0]) and np.isnan(arr[1, 3])  # Kauri missing 773 & 776
            assert arr[0, 0] == pytest.approx(1.0)
            assert arr[0, 1] == pytest.approx(-2.0)
            assert "pull" in fig.axes[1].get_ylabel().lower()
        finally:
            plt.close(fig)

        fig_raw, ax_raw = tt.plot_residual_map(ds, pred, zscore=False, vmax=4.5)
        try:
            vmin_r, vmax_r = ax_raw.images[0].get_clim()
            assert (vmin_r, vmax_r) == pytest.approx((-4.5, 4.5))
            assert "‰" in fig_raw.axes[1].get_ylabel()
        finally:
            plt.close(fig_raw)

    def test_plot_corner_idata_weighted_smc_and_statelessness(self):
        import arviz as az

        rc_before = dict(plt.rcParams)
        rng = np.random.default_rng(42)

        # 1. InferenceData input with param_names mapping + truth crosshairs
        idata = az.from_dict(
            {
                "posterior": {
                    "t_start": rng.normal(774.5, 0.15, size=(2, 120)),
                    "duration": np.abs(rng.normal(0.5, 0.12, size=(2, 120))) + 0.05,
                    "fluence": rng.normal(6.3, 0.30, size=(2, 120)),
                }
            }
        )
        pnames = {
            "t_start": r"$t_{\mathrm{start}}$ (CE)",
            "duration": r"$\tau_d$ (yr)",
            "fluence": r"$F$",
        }
        fig1, axes1 = tt.plot_corner(
            idata,
            param_names=pnames,
            name="NUTS",
            truth=[774.5, 0.5, 6.3],
            truth_name="MAP",
        )
        try:
            assert len(fig1.axes) >= 9
            assert axes1.size == len(fig1.axes)
        finally:
            plt.close(fig1)

        # 2. Multi-chain weighted SMC overlay
        n_p = 180
        p_flat = np.column_stack(
            [
                rng.normal(774.5, 0.15, n_p),
                np.abs(rng.normal(0.5, 0.12, n_p)) + 0.05,
                rng.normal(3.4, 0.18, n_p),
            ]
        )
        p_gp = np.column_stack(
            [
                rng.normal(774.6, 0.18, n_p),
                np.abs(rng.normal(0.55, 0.14, n_p)) + 0.05,
                rng.normal(3.5, 0.40, n_p),
            ]
        )
        w_flat = rng.dirichlet(np.ones(n_p))
        w_gp = rng.dirichlet(np.ones(n_p))
        fig2, axes2 = tt.plot_corner(
            {"Flat Baseline": (p_flat, w_flat), "Marginalized GP": (p_gp, w_gp)},
            param_names=[r"$t_{\mathrm{start}}$ (CE)", r"$\tau_d$ (yr)", r"$Q / q_0$ (yr)"],
        )
        try:
            assert len(fig2.axes) >= 9
            assert axes2.size == len(fig2.axes)
        finally:
            plt.close(fig2)

        assert dict(plt.rcParams) == rc_before


class TestDocsExampleNotebooks:
    @pytest.mark.parametrize("rel_path", DOCS_EXAMPLE_NOTEBOOKS)
    def test_five_cell_notebook_invariants_and_outputs(self, rel_path: str):
        nb_path = ROOT / rel_path
        assert nb_path.exists(), f"Missing notebook {nb_path}"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        assert nb.get("nbformat") == 4
        cells = nb.get("cells", [])
        assert len(cells) >= 11

        assert cells[0]["cell_type"] == "markdown"
        assert cells[-1]["cell_type"] == "markdown"

        n_code = 0
        n_png = 0
        all_stdout: list[str] = []

        for idx, cell in enumerate(cells):
            expected_type = "markdown" if idx % 2 == 0 else "code"
            assert cell["cell_type"] == expected_type, (
                f"{rel_path} cell {idx}: expected {expected_type}, got {cell['cell_type']}"
            )
            src = "".join(cell.get("source", []))
            assert src.strip(), f"{rel_path} cell {idx} is empty"
            assert ".png" not in src and "savefig" not in src, (
                f"{rel_path} cell {idx} writes .png files instead of inline rendering"
            )

            if expected_type == "code":
                n_code += 1
                n_lines = len(src.strip().splitlines())
                assert 3 <= n_lines <= 16, (
                    f"{rel_path} code cell {idx} has {n_lines} lines (expected concise <=16 lines)"
                )
                assert cell.get("execution_count") == n_code
                outputs = cell.get("outputs", [])
                assert len(outputs) == 1, (
                    f"{rel_path} code cell {idx} must emit strictly 1 output, got {len(outputs)}"
                )
                out = outputs[0]
                otype = out.get("output_type")
                assert otype in ("stream", "display_data")
                if otype == "stream":
                    assert out.get("name") == "stdout"
                    txt = "".join(out.get("text", []))
                    assert txt.strip()
                    all_stdout.append(txt)
                else:
                    b64 = out.get("data", {}).get("image/png", "")
                    raw_png = base64.b64decode(b64)
                    assert raw_png.startswith(b"\x89PNG\r\n\x1a\n")
                    n_png += 1

        assert 5 <= n_code <= 7
        assert n_png >= 2
        joined_stdout = "\n".join(all_stdout)
        assert "v1" in joined_stdout and "warm=" in joined_stdout

    def test_no_stray_png_files_in_docs_examples(self):
        pngs = list((ROOT / "notebooks").rglob("*.png"))
        assert pngs == [], f"Found stray .png files under notebooks/: {pngs}"
