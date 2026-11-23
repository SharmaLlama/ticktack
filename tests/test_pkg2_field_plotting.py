"""Regression tests for Package 2: field plotting moved to ``ticktack.plotting``."""

from __future__ import annotations

from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import ticktack
import ticktack.fields as tt_fields
import ticktack.plotting as tt_plotting
from ticktack.fields import EvidenceGridResult, GaussianField, LogGaussianField
from ticktack.plotting import plot_evidence_grid, plot_field_draws


def test_fields_module_has_no_matplotlib_or_plotting_exports() -> None:
    """``ticktack.fields`` must not import matplotlib or export plotting helpers."""
    assert "plot_field_draws" not in tt_fields.__all__
    assert "plot_evidence_grid" not in tt_fields.__all__
    assert not hasattr(tt_fields, "plot_field_draws")
    assert not hasattr(tt_fields, "plot_evidence_grid")
    assert not hasattr(tt_fields, "plt")
    assert hasattr(tt_plotting, "plot_field_draws")
    assert hasattr(tt_plotting, "plot_evidence_grid")
    assert hasattr(ticktack, "plot_field_draws")
    assert hasattr(ticktack, "plot_evidence_grid")


@pytest.mark.parametrize("as_total_q", [True, False])
def test_plot_field_draws_single_field_axes_artists_and_rcparams(as_total_q: bool) -> None:
    """Single-field mode renders 3 top panels + 1 ensemble panel without mutating rcParams."""
    rc_before = dict(matplotlib.rcParams)
    z0 = jnp.zeros((32,), dtype=jnp.float64)
    gf = GaussianField(z0, 765.0, 785.0, sigma=0.25, length_yr=3.0)
    truth = np.full(32, 1.8 if as_total_q else 0.0, dtype=np.float64)

    fig, axes = plot_field_draws(
        gf,
        jax.random.PRNGKey(7),
        n_faint=12,
        as_total_q=as_total_q,
        truth=truth,
        title="Single field",
    )
    try:
        assert axes.size == 4
        for idx in range(3):
            # 1 draw line + 1 reference axhline
            assert len(axes[idx].lines) == 2
        # 12 ensemble lines + 1 truth line
        assert len(axes[3].lines) == 13
    finally:
        plt.close(fig)

    # Caller-supplied axes path
    fig_ext = plt.figure(figsize=(8.0, 4.0))
    ext_axes = [fig_ext.add_subplot(2, 2, i + 1) for i in range(4)]
    lgf = LogGaussianField(z0, 765.0, 785.0, sigma=0.2, length_yr=3.0)
    fig_ret, axes_ret = plot_field_draws(lgf, jax.random.PRNGKey(8), n_faint=8, axes=ext_axes)
    try:
        assert fig_ret is fig_ext
        assert axes_ret.size == 4
        assert len(axes_ret[3].lines) == 8
    finally:
        plt.close(fig_ext)

    assert dict(matplotlib.rcParams) == rc_before


def test_plot_field_draws_multi_field_mapping_and_sequence() -> None:
    """Multi-field mode renders a 2 x N grid sharing latents and preserving rcParams."""
    rc_before = dict(matplotlib.rcParams)
    z0 = jnp.zeros((32,), dtype=jnp.float64)
    f_short = GaussianField(z0, 765.0, 785.0, sigma=0.25, length_yr=1.5)
    f_long = LogGaussianField(z0, 765.0, 785.0, sigma=0.25, length_yr=8.0)
    truth = np.full(32, 1.8, dtype=np.float64)

    fig, axes = plot_field_draws(
        {"Short": f_short, "Long": f_long},
        jax.random.PRNGKey(11),
        n_faint=10,
        truth=truth,
        title="Comparison",
    )
    try:
        assert axes.shape == (2, 2)
        for col in range(2):
            assert len(axes[0, col].lines) == 3
            assert len(axes[1, col].lines) == 11
        assert axes[0, 0].get_title() == "Short"
        assert axes[0, 1].get_title() == "Long"
    finally:
        plt.close(fig)

    # Sequence input + supplied 2x2 axes
    fig_ext, ext_axes = plt.subplots(2, 2)
    fig_ret, axes_ret = plot_field_draws(
        [f_short, f_long],
        jax.random.PRNGKey(12),
        n_faint=6,
        axes=ext_axes,
    )
    try:
        assert fig_ret is fig_ext
        assert axes_ret.shape == (2, 2)
        assert axes_ret[0, 0].get_title() == "Field 1"
        assert axes_ret[0, 1].get_title() == "Field 2"
    finally:
        plt.close(fig_ext)

    assert dict(matplotlib.rcParams) == rc_before


@pytest.mark.parametrize("delta", [True, False])
def test_plot_evidence_grid_artists_and_rcparams(delta: bool) -> None:
    """``plot_evidence_grid`` renders mesh, annotations, peak/truth markers, and preserves rcParams."""
    rc_before = dict(matplotlib.rcParams)
    sigs = np.geomspace(0.05, 0.5, 4)
    lens = np.geomspace(1.0, 16.0, 5)
    lnz = -np.add.outer((np.log(sigs) - np.log(0.2)) ** 2, (np.log(lens) - np.log(4.0)) ** 2)
    grid = EvidenceGridResult(
        sigmas=jnp.asarray(sigs),
        lengths_yr=jnp.asarray(lens),
        log_evidence=jnp.asarray(lnz),
        best_sigma=jnp.asarray(sigs[2]),
        best_length_yr=jnp.asarray(lens[2]),
        best_log_evidence=jnp.asarray(np.max(lnz)),
        best_fit=None,  # type: ignore[arg-type]
    )

    fig, ax = plt.subplots(1, 1)
    fig_ret, ax_ret = plot_evidence_grid(
        grid, delta=delta, annotate=True, truth=(0.2, 4.0), title="Grid", ax=ax
    )
    try:
        assert fig_ret is fig
        assert ax_ret is ax
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"
        # 1 QuadMesh + 2 peak scatter layers + 1 truth scatter layer = 4 collections
        assert len(ax.collections) == 4
        assert len(ax.texts) == sigs.size * lens.size
        assert ax.get_title() == "Grid"
    finally:
        plt.close(fig)

    assert dict(matplotlib.rcParams) == rc_before
