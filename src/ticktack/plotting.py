"""Stateless scientific diagnostic plotting for ``ticktack 2.0``.

Every plotting function in this module is strictly stateless:
- Never mutates ``matplotlib.rcParams`` or global styles.
- Never relies on ``plt.gca()`` / ``plt.gcf()`` implicit state.
- Accepts optional caller-supplied ``Axes`` and always returns ``(fig, axes)``.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from chainconsumer import Chain, ChainConsumer, PlotConfig, Truth
from jax.typing import ArrayLike
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from .constants import DEFAULT_Q_STD
from .data import Dataset, TreeRingSeries
from .fields import EvidenceGridResult, GaussianField, LogGaussianField

_PALETTE: tuple[str, ...] = (
    "#0072B2",  # blue (NH / primary)
    "#D55E00",  # vermilion (SH / secondary)
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#333333",  # charcoal
)


def _extract_series_list(
    dataset_or_series: Dataset | TreeRingSeries | Sequence[TreeRingSeries] | None,
) -> list[TreeRingSeries]:
    if dataset_or_series is None:
        return []
    if isinstance(dataset_or_series, Dataset):
        return list(dataset_or_series.series)
    if isinstance(dataset_or_series, TreeRingSeries):
        return [dataset_or_series]
    return list(dataset_or_series)


def plot_d14c_fit(
    dataset_or_series: Dataset | TreeRingSeries | Sequence[TreeRingSeries] | None = None,
    *,
    t_fine: Any | None = None,
    d14c_pred: Any | Mapping[str, Any] | None = None,
    d14c_band: tuple[Any, Any] | Mapping[str, tuple[Any, Any]] | None = None,
    ring_pred: Any | Sequence[Any] | None = None,
    t_obs: Any | None = None,
    d14c_obs: Any | None = None,
    sigma_obs: Any | None = None,
    show_windows: bool = True,
    show_residuals: bool = False,
    title: str | None = None,
    xlabel: str = "Calendar Year (Astronomical CE)",
    ylabel: str = r"$\Delta^{14}\mathrm{C}$ (‰)",
    figsize: tuple[float, float] | None = None,
    axes: Axes | Sequence[Axes] | None = None,
) -> tuple[Figure, Axes | np.ndarray]:
    """Plot observed tree-ring ``Delta14C`` series and model posterior predictions.

    Parameters
    ----------
    dataset_or_series : Dataset, TreeRingSeries, sequence of TreeRingSeries, optional
        Structured ``ticktack 2.0`` observations with audited seasonal windows.
    t_fine : array-like, optional
        1D fine time grid ``(N_fine,)`` for continuous ``Delta14C(t)`` curves.
    d14c_pred : array-like or Mapping[str, array-like], optional
        Either a 1D curve ``(N_fine,)``, a 2D sample array ``(N_draws, N_fine)``,
        or a mapping keyed by hemisphere/label (e.g. ``{"north": ..., "south": ...}``).
    d14c_band : tuple ``(low, high)`` or Mapping[str, tuple], optional
        Explicit credible band bounds on ``t_fine`` when ``d14c_pred`` is 1D.
    ring_pred : array-like or sequence of array-like, optional
        Predicted seasonal-binned ``Delta14C`` at each ring (flat ``(N_tot,)`` or
        per-series list), used for plotting binned markers or exact residuals.
    t_obs, d14c_obs, sigma_obs : array-like, optional
        Raw arrays when ``dataset_or_series`` is omitted.
    show_windows : bool, default True
        If ``True``, render horizontal ``[t_start, t_end]`` growth-season spans.
    show_residuals : bool, default False
        If ``True``, include a lower standardized-residual panel ``(d - m) / sigma``.
    axes : Axes or sequence of Axes, optional
        Pre-existing axes to draw into.

    Returns
    -------
    tuple[Figure, Axes | np.ndarray]
        The matplotlib ``(fig, axes)`` pair.
    """
    if axes is None:
        if show_residuals:
            eff_size = (8.5, 6.0) if figsize is None else figsize
            fig, ax_arr = plt.subplots(
                2,
                1,
                figsize=eff_size,
                sharex=True,
                gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.08},
            )
            main_ax, resid_ax = ax_arr[0], ax_arr[1]
            out_axes: Axes | np.ndarray = ax_arr
        else:
            eff_size = (8.5, 4.6) if figsize is None else figsize
            fig, main_ax = plt.subplots(1, 1, figsize=eff_size)
            resid_ax = None
            out_axes = main_ax
    else:
        if show_residuals:
            if not isinstance(axes, (Sequence, np.ndarray)) or len(axes) < 2:
                raise ValueError("show_residuals=True requires a sequence of 2 Axes.")
            main_ax, resid_ax = axes[0], axes[1]
            fig = main_ax.figure
            out_axes = np.asarray(axes, dtype=object)
        else:
            main_ax = axes[0] if isinstance(axes, (Sequence, np.ndarray)) else axes
            resid_ax = None
            fig = main_ax.figure
            out_axes = main_ax

    series_list = _extract_series_list(dataset_or_series)

    # Unpack optional per-series ring predictions for residual plotting.
    per_series_pred: list[np.ndarray] | None = None
    if ring_pred is not None and series_list:
        if isinstance(ring_pred, (list, tuple)):
            per_series_pred = [np.asarray(rp, dtype=np.float64) for rp in ring_pred]
        else:
            flat_rp = np.asarray(ring_pred, dtype=np.float64).reshape(-1)
            if flat_rp.size == sum(s.n_obs for s in series_list):
                per_series_pred = []
                cursor = 0
                for s in series_list:
                    per_series_pred.append(flat_rp[cursor : cursor + s.n_obs])
                    cursor += s.n_obs

    # 1. Plot observations
    if series_list:
        for idx, s in enumerate(series_list):
            color = (
                _PALETTE[0]
                if s.hemisphere == "north" and len(series_list) <= 2
                else (
                    _PALETTE[1]
                    if s.hemisphere == "south" and len(series_list) <= 2
                    else _PALETTE[idx % len(_PALETTE)]
                )
            )
            t_mid = np.asarray(s.t_midpoints, dtype=np.float64)
            y_val = np.asarray(s.d14c, dtype=np.float64)
            y_err = np.asarray(s.sigma_d14c, dtype=np.float64)
            if show_windows:
                t_lo = np.asarray(s.t_starts, dtype=np.float64)
                t_hi = np.asarray(s.t_ends, dtype=np.float64)
                xerr = np.vstack([np.maximum(t_mid - t_lo, 0.0), np.maximum(t_hi - t_mid, 0.0)])
            else:
                xerr = None
            marker = "o" if s.hemisphere == "north" else "s"
            label = f"{s.name} ({'NH' if s.hemisphere == 'north' else 'SH'})"
            main_ax.errorbar(
                t_mid,
                y_val,
                yerr=y_err,
                xerr=xerr,
                fmt=marker,
                color=color,
                markersize=4.5,
                elinewidth=1.1,
                capsize=2.0,
                alpha=0.85,
                label=label,
            )
            if resid_ax is not None:
                pred_s = _resolve_series_ring_pred(
                    idx, s, per_series_pred, t_fine, d14c_pred
                )
                if pred_s is not None:
                    std_res = (y_val - pred_s) / y_err
                    resid_ax.errorbar(
                        t_mid,
                        std_res,
                        yerr=np.ones_like(std_res),
                        fmt=marker,
                        color=color,
                        markersize=4.0,
                        elinewidth=0.9,
                        capsize=1.5,
                        alpha=0.85,
                    )
    elif t_obs is not None and d14c_obs is not None:
        t_arr = np.asarray(t_obs, dtype=np.float64)
        y_arr = np.asarray(d14c_obs, dtype=np.float64)
        s_arr = (
            np.ones_like(y_arr)
            if sigma_obs is None
            else np.asarray(sigma_obs, dtype=np.float64)
        )
        main_ax.errorbar(
            t_arr,
            y_arr,
            yerr=None if sigma_obs is None else s_arr,
            fmt="o",
            color=_PALETTE[7],
            markersize=4.5,
            elinewidth=1.1,
            capsize=2.0,
            label=r"Observed $\Delta^{14}\mathrm{C}$",
        )
        if resid_ax is not None:
            if ring_pred is not None:
                rp_arr = np.asarray(ring_pred, dtype=np.float64)
            elif t_fine is not None and d14c_pred is not None and not isinstance(d14c_pred, Mapping):
                tf_arr = np.asarray(t_fine, dtype=np.float64)
                dp_arr = np.asarray(d14c_pred, dtype=np.float64)
                mean_curve = dp_arr if dp_arr.ndim == 1 else np.mean(dp_arr, axis=0)
                rp_arr = np.interp(t_arr, tf_arr, mean_curve)
            else:
                rp_arr = None
            if rp_arr is not None:
                resid_ax.errorbar(
                    t_arr,
                    (y_arr - rp_arr) / s_arr,
                    yerr=np.ones_like(y_arr),
                    fmt="o",
                    color=_PALETTE[7],
                    markersize=4.0,
                    elinewidth=0.9,
                )

    # 2. Plot continuous fine-grid model prediction(s)
    if t_fine is not None and d14c_pred is not None:
        tf = np.asarray(t_fine, dtype=np.float64)
        if isinstance(d14c_pred, Mapping):
            for m_idx, (key, arr_like) in enumerate(d14c_pred.items()):
                col = (
                    _PALETTE[0]
                    if key.lower() in ("north", "nh")
                    else (_PALETTE[1] if key.lower() in ("south", "sh") else _PALETTE[m_idx % len(_PALETTE)])
                )
                band_k = (
                    d14c_band.get(key)
                    if isinstance(d14c_band, Mapping)
                    else None
                )
                _draw_prediction_curve(main_ax, tf, arr_like, band_k, color=col, label=f"Model ({key})")
        else:
            band_single = (
                None if isinstance(d14c_band, Mapping) else d14c_band
            )
            _draw_prediction_curve(
                main_ax,
                tf,
                d14c_pred,
                band_single,
                color=_PALETTE[0],
                label=r"Model $\Delta^{14}\mathrm{C}(t)$",
            )

    main_ax.set_ylabel(ylabel)
    main_ax.grid(True, alpha=0.25, linestyle="--")
    handles, _labels = main_ax.get_legend_handles_labels()
    if handles:
        main_ax.legend(loc="best", frameon=True, fontsize=8.5)
    if title is not None:
        main_ax.set_title(title)

    if resid_ax is not None:
        resid_ax.axhline(0.0, color="#333333", lw=1.0, ls="--")
        resid_ax.set_ylabel(r"Pull $(\mathrm{d}-\mathrm{m})/\sigma$")
        resid_ax.set_xlabel(xlabel)
        resid_ax.grid(True, alpha=0.25, linestyle="--")
    else:
        main_ax.set_xlabel(xlabel)

    return fig, out_axes


def _resolve_series_ring_pred(
    idx: int,
    series: TreeRingSeries,
    per_series_pred: list[np.ndarray] | None,
    t_fine: Any | None,
    d14c_pred: Any | Mapping[str, Any] | None,
) -> np.ndarray | None:
    if per_series_pred is not None and idx < len(per_series_pred):
        return per_series_pred[idx]
    if t_fine is None or d14c_pred is None:
        return None
    tf = np.asarray(t_fine, dtype=np.float64)
    if isinstance(d14c_pred, Mapping):
        raw = (
            d14c_pred[series.hemisphere]
            if series.hemisphere in d14c_pred
            else d14c_pred.get(series.name)
        )
        if raw is None:
            return None
        arr = np.asarray(raw, dtype=np.float64)
    else:
        arr = np.asarray(d14c_pred, dtype=np.float64)
    curve = arr if arr.ndim == 1 else np.mean(arr, axis=0)
    return np.interp(np.asarray(series.t_midpoints, dtype=np.float64), tf, curve)


def _draw_prediction_curve(
    ax: Axes,
    tf: np.ndarray,
    pred_like: Any,
    band: tuple[Any, Any] | None,
    *,
    color: str,
    label: str,
) -> None:
    arr = np.asarray(pred_like, dtype=np.float64)
    if arr.ndim == 2:
        median = np.percentile(arr, 50.0, axis=0)
        lo68, hi68 = np.percentile(arr, [16.0, 84.0], axis=0)
        lo95, hi95 = np.percentile(arr, [2.5, 97.5], axis=0)
        ax.fill_between(tf, lo95, hi95, color=color, alpha=0.14)
        ax.fill_between(tf, lo68, hi68, color=color, alpha=0.28)
        ax.plot(tf, median, color=color, lw=1.8, label=label)
    elif arr.ndim == 1:
        if band is not None:
            lo = np.asarray(band[0], dtype=np.float64)
            hi = np.asarray(band[1], dtype=np.float64)
            ax.fill_between(tf, lo, hi, color=color, alpha=0.24)
        ax.plot(tf, arr, color=color, lw=1.8, label=label)
    else:
        raise ValueError(f"Expected 1D or 2D prediction array, got shape {arr.shape}.")


def plot_production_posterior(
    t: Any,
    mean_or_samples: Any,
    *,
    std: Any | None = None,
    cov: Any | None = None,
    q_std: float | None = None,
    truth: Any | None = None,
    n_spaghetti: int = 0,
    color: str = "#0072B2",
    title: str | None = None,
    xlabel: str = "Calendar Year (Astronomical CE)",
    ylabel: str = r"Production rate $Q(t)$ ($\mathrm{atoms\,cm^{-2}\,s^{-1}}$)",
    figsize: tuple[float, float] = (8.5, 4.2),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot cosmic-ray 14C production posterior trajectory and credible bands.

    Supports both analytical Gaussian POD posteriors (1D ``mean`` + ``std`` or
    2D ``cov``) and MCMC/SMC sample ensembles (2D ``(N_draws, N_t)``).
    """
    if ax is None:
        fig, target_ax = plt.subplots(1, 1, figsize=figsize)
    else:
        target_ax = ax
        fig = target_ax.figure

    t_arr = np.asarray(t, dtype=np.float64).reshape(-1)
    q_arr = np.asarray(mean_or_samples, dtype=np.float64)

    if q_arr.ndim == 2:
        if q_arr.shape[1] != t_arr.size:
            raise ValueError(
                f"Sample array shape {q_arr.shape} incompatible with t length {t_arr.size}."
            )
        center = np.percentile(q_arr, 50.0, axis=0)
        lo68, hi68 = np.percentile(q_arr, [16.0, 84.0], axis=0)
        lo95, hi95 = np.percentile(q_arr, [2.5, 97.5], axis=0)
        target_ax.fill_between(t_arr, lo95, hi95, color=color, alpha=0.14, label="95% CI")
        target_ax.fill_between(t_arr, lo68, hi68, color=color, alpha=0.28, label="68% CI")
        if n_spaghetti > 0:
            step = max(1, q_arr.shape[0] // n_spaghetti)
            for draw in q_arr[::step][:n_spaghetti]:
                target_ax.plot(t_arr, draw, color=color, lw=0.6, alpha=0.18)
        target_ax.plot(t_arr, center, color=color, lw=1.9, label="Posterior median")
    elif q_arr.ndim == 1:
        if q_arr.size != t_arr.size:
            raise ValueError("mean and t must have equal length.")
        sigma_q: np.ndarray | None = None
        if std is not None:
            sigma_q = np.asarray(std, dtype=np.float64).reshape(-1)
        elif cov is not None:
            cov_arr = np.asarray(cov, dtype=np.float64)
            sigma_q = np.sqrt(np.maximum(np.diag(cov_arr), 0.0))
        if sigma_q is not None:
            target_ax.fill_between(
                t_arr,
                q_arr - 1.959964 * sigma_q,
                q_arr + 1.959964 * sigma_q,
                color=color,
                alpha=0.14,
                label="95% CI",
            )
            target_ax.fill_between(
                t_arr,
                q_arr - sigma_q,
                q_arr + sigma_q,
                color=color,
                alpha=0.28,
                label="68% CI",
            )
        target_ax.plot(t_arr, q_arr, color=color, lw=1.9, label="Posterior mean")
    else:
        raise ValueError(f"Expected 1D or 2D mean_or_samples, got shape {q_arr.shape}.")

    if q_std is not None:
        target_ax.axhline(
            float(q_std),
            color="#555555",
            lw=1.1,
            ls="--",
            label=rf"$Q_{{\mathrm{{std}}}} = {float(q_std):.2f}$",
        )
    if truth is not None:
        truth_arr = np.asarray(truth, dtype=np.float64).reshape(-1)
        target_ax.plot(t_arr, truth_arr, color="#D55E00", lw=1.5, ls="-.", label="Injected truth")

    target_ax.set_xlabel(xlabel)
    target_ax.set_ylabel(ylabel)
    if title is not None:
        target_ax.set_title(title)
    target_ax.grid(True, alpha=0.25, linestyle="--")
    target_ax.legend(loc="best", frameon=True, fontsize=8.5)
    return fig, target_ax


def plot_fisher_spectrum(
    spectrum_or_eigenvalues: Any,
    *,
    analytical_spectrum: Any | None = None,
    show_cumulative: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    axes: Axes | Sequence[Axes] | None = None,
) -> tuple[Figure, Axes | np.ndarray]:
    """Plot prior-whitened carbon-cycle Fisher scree spectrum and information gain.

    Accepts either a ``FisherSpectrumResult`` object (exposing ``.eigenvalues``,
    ``.d_eff``, ``.info_nats``, ``.info_bits``) or a 1D array of non-negative
    prior-whitened Fisher eigenvalues ``mu_k``.
    """
    if hasattr(spectrum_or_eigenvalues, "eigenvalues"):
        mu_raw = np.asarray(spectrum_or_eigenvalues.eigenvalues, dtype=np.float64).reshape(-1)
        d_eff_val = float(
            getattr(
                spectrum_or_eigenvalues,
                "d_eff",
                np.sum(np.maximum(mu_raw, 0.0) / (1.0 + np.maximum(mu_raw, 0.0))),
            )
        )
        info_nats_val = float(
            getattr(
                spectrum_or_eigenvalues,
                "info_nats",
                0.5 * np.sum(np.log1p(np.maximum(mu_raw, 0.0))),
            )
        )
        info_bits_val = float(
            getattr(
                spectrum_or_eigenvalues,
                "info_bits",
                info_nats_val / np.log(2.0),
            )
        )
    else:
        mu_raw = np.asarray(spectrum_or_eigenvalues, dtype=np.float64).reshape(-1)
        mu_pos = np.maximum(mu_raw, 0.0)
        d_eff_val = float(np.sum(mu_pos / (1.0 + mu_pos)))
        info_nats_val = float(0.5 * np.sum(np.log1p(mu_pos)))
        info_bits_val = float(info_nats_val / np.log(2.0))

    if mu_raw.size == 0:
        raise ValueError("Fisher spectrum eigenvalues must be non-empty.")

    mu = np.sort(np.maximum(mu_raw, 0.0))[::-1]
    idx = np.arange(1, mu.size + 1)
    floor = 1e-18
    mu_plot = np.maximum(mu, floor)

    if axes is None:
        if show_cumulative:
            eff_size = (9.5, 4.2) if figsize is None else figsize
            fig, ax_arr = plt.subplots(1, 2, figsize=eff_size)
            scree_ax, cum_ax = ax_arr[0], ax_arr[1]
            out_axes: Axes | np.ndarray = ax_arr
        else:
            eff_size = (5.4, 4.2) if figsize is None else figsize
            fig, scree_ax = plt.subplots(1, 1, figsize=eff_size)
            cum_ax = None
            out_axes = scree_ax
    else:
        if show_cumulative:
            if not isinstance(axes, (Sequence, np.ndarray)) or len(axes) < 2:
                raise ValueError("show_cumulative=True requires a sequence of 2 Axes.")
            scree_ax, cum_ax = axes[0], axes[1]
            fig = scree_ax.figure
            out_axes = np.asarray(axes, dtype=object)
        else:
            scree_ax = axes[0] if isinstance(axes, (Sequence, np.ndarray)) else axes
            cum_ax = None
            fig = scree_ax.figure
            out_axes = scree_ax

    scree_ax.semilogy(
        idx,
        mu_plot,
        marker="o",
        color=_PALETTE[0],
        lw=1.7,
        markersize=5.0,
        label=r"Whitened Fisher $\mu_k$",
    )
    scree_ax.axhline(
        1.0, color="#D55E00", lw=1.2, ls="--", label=r"Prior threshold $\mu_k = 1$"
    )
    if analytical_spectrum is not None:
        th_arr = np.maximum(
            np.asarray(analytical_spectrum, dtype=np.float64).reshape(-1), floor
        )
        th_idx = np.arange(1, th_arr.size + 1)
        scree_ax.semilogy(
            th_idx,
            th_arr,
            color="#009E73",
            lw=1.5,
            ls="-.",
            label=r"Analytical Mellin decay",
        )

    summary_txt = (
        rf"$d_{{\mathrm{{eff}}}} = {d_eff_val:.2f}$"
        "\n"
        rf"$\mathcal{{I}} = {info_nats_val:.2f}\,\mathrm{{nats}}\;({info_bits_val:.2f}\,\mathrm{{bits}})$"
    )
    scree_ax.text(
        0.97,
        0.95,
        summary_txt,
        transform=scree_ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85, "edgecolor": "#CCCCCC"},
    )
    scree_ax.set_xlabel("Mode index $k$")
    scree_ax.set_ylabel(r"Prior-whitened eigenvalue $\mu_k$")
    scree_ax.grid(True, which="both", alpha=0.25, linestyle="--")
    scree_ax.legend(loc="lower left", frameon=True, fontsize=8.0)
    if title is not None and cum_ax is None:
        scree_ax.set_title(title)

    if cum_ax is not None:
        cum_deff = np.cumsum(mu / (1.0 + mu))
        cum_bits = np.cumsum(0.5 * np.log1p(mu) / np.log(2.0))
        cum_ax.plot(
            idx,
            cum_deff,
            marker="o",
            color=_PALETTE[0],
            lw=1.7,
            markersize=4.5,
            label=r"Cumulative $d_{\mathrm{eff}}(\leq k)$",
        )
        cum_ax.plot(
            idx,
            cum_bits,
            marker="s",
            color=_PALETTE[2],
            lw=1.7,
            markersize=4.5,
            ls="--",
            label=r"Cumulative $\mathcal{I}(\leq k)$ [bits]",
        )
        cum_ax.set_xlabel("Mode index $k$")
        cum_ax.set_ylabel("Cumulative constrained modes / bits")
        cum_ax.grid(True, alpha=0.25, linestyle="--")
        cum_ax.legend(loc="best", frameon=True, fontsize=8.0)
        if title is not None:
            fig.suptitle(title)

    return fig, out_axes


def plot_cbm_eigenmodes(
    modes_or_result: Any,
    *,
    eigenvalues: Any | None = None,
    n_modes: int = 4,
    x_values: Any | None = None,
    param_names: Sequence[str] | None = None,
    style: Literal["auto", "bars", "heatmap", "curves"] = "auto",
    log_x: bool = True,
    xlabel: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Visualize leading carbon-cycle Fisher or Laplace-relaxation eigenmodes.

    Parameters
    ----------
    modes_or_result : FisherSpectrumResult or 2D array ``(n_coords, n_total_modes)``
        Eigenvector matrix whose columns are ordered from largest to smallest
        Fisher eigenvalue ``mu_k``.
    eigenvalues : array-like, optional
        Corresponding ``mu_k`` values for legend/tick annotations.
    n_modes : int, default 4
        Maximum number of leading modes to display.
    x_values : array-like, optional
        Continuous coordinate axis (e.g. residence times ``tau_res`` in years
        from ``relaxation_fisher_modes``). Triggers smooth curve plotting.
    param_names : sequence of str, optional
        Labels for discrete CBM reservoir or transfer-rate coordinates.
    style : {"auto", "bars", "heatmap", "curves"}, default "auto"
        Rendering mode.
    """
    if hasattr(modes_or_result, "prior_whitened_modes"):
        modes_mat = np.asarray(modes_or_result.prior_whitened_modes, dtype=np.float64)
        if eigenvalues is None and hasattr(modes_or_result, "eigenvalues"):
            eigenvalues = modes_or_result.eigenvalues
    elif hasattr(modes_or_result, "parameter_loadings"):
        modes_mat = np.asarray(modes_or_result.parameter_loadings, dtype=np.float64)
        if eigenvalues is None and hasattr(modes_or_result, "eigenvalues"):
            eigenvalues = modes_or_result.eigenvalues
    else:
        modes_mat = np.asarray(modes_or_result, dtype=np.float64)

    if modes_mat.ndim == 1:
        modes_mat = modes_mat[:, None]
    if modes_mat.ndim != 2 or modes_mat.size == 0:
        raise ValueError(f"Expected non-empty 2D modes matrix, got shape {modes_mat.shape}.")

    n_coords, total_modes = modes_mat.shape
    k_show = max(1, min(int(n_modes), total_modes))
    lead_modes = modes_mat[:, :k_show]
    mu_arr = (
        None
        if eigenvalues is None
        else np.asarray(eigenvalues, dtype=np.float64).reshape(-1)[:k_show]
    )

    if style == "auto":
        if x_values is not None:
            eff_style: Literal["bars", "heatmap", "curves"] = "curves"
        elif n_coords > 24 or k_show > 5:
            eff_style = "heatmap"
        else:
            eff_style = "bars"
    else:
        eff_style = style

    if ax is None:
        eff_size = (8.5, 4.5) if figsize is None else figsize
        fig, target_ax = plt.subplots(1, 1, figsize=eff_size)
    else:
        target_ax = ax
        fig = target_ax.figure

    mode_labels = [
        (
            rf"Mode {j + 1} ($\mu_{{{j + 1}}} = {mu_arr[j]:.2g}$)"
            if mu_arr is not None and j < mu_arr.size
            else f"Mode {j + 1}"
        )
        for j in range(k_show)
    ]

    if eff_style == "curves":
        x_arr = (
            np.arange(n_coords, dtype=np.float64)
            if x_values is None
            else np.asarray(x_values, dtype=np.float64).reshape(-1)
        )
        if x_arr.size != n_coords:
            raise ValueError("x_values length must match modes_mat.shape[0].")
        for j in range(k_show):
            plot_fn = target_ax.semilogx if (log_x and np.all(x_arr > 0.0)) else target_ax.plot
            plot_fn(
                x_arr,
                lead_modes[:, j],
                color=_PALETTE[j % len(_PALETTE)],
                lw=1.8,
                label=mode_labels[j],
            )
        target_ax.axhline(0.0, color="#555555", lw=0.9, ls="--")
        target_ax.set_xlabel(
            r"Residence time $\tau_{\mathrm{res}}$ (yr)" if xlabel is None else xlabel
        )
        target_ax.set_ylabel("Normalized eigenmode amplitude")
        target_ax.grid(True, which="both", alpha=0.25, linestyle="--")
        target_ax.legend(loc="best", frameon=True, fontsize=8.5)

    elif eff_style == "bars":
        base_x = np.arange(n_coords, dtype=np.float64)
        width = 0.8 / k_show
        for j in range(k_show):
            offset = (j - 0.5 * (k_show - 1)) * width
            target_ax.bar(
                base_x + offset,
                lead_modes[:, j],
                width=width,
                color=_PALETTE[j % len(_PALETTE)],
                alpha=0.88,
                label=mode_labels[j],
            )
        target_ax.axhline(0.0, color="#555555", lw=0.9)
        target_ax.set_xticks(base_x)
        if param_names is not None:
            target_ax.set_xticklabels(list(param_names), rotation=35, ha="right", fontsize=8.0)
        else:
            target_ax.set_xticklabels([f"p{i}" for i in range(n_coords)], fontsize=8.0)
        target_ax.set_xlabel("CBM parameter" if xlabel is None else xlabel)
        target_ax.set_ylabel("Mode loading")
        target_ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        target_ax.legend(loc="best", frameon=True, fontsize=8.5)

    elif eff_style == "heatmap":
        vmax = float(np.max(np.abs(lead_modes)))
        vmax = 1.0 if vmax == 0.0 else vmax
        im = target_ax.imshow(
            lead_modes.T,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        target_ax.set_yticks(np.arange(k_show))
        target_ax.set_yticklabels(mode_labels, fontsize=8.5)
        target_ax.set_xticks(np.arange(n_coords))
        if param_names is not None:
            target_ax.set_xticklabels(list(param_names), rotation=45, ha="right", fontsize=7.5)
        target_ax.set_xlabel("CBM coordinate" if xlabel is None else xlabel)
        fig.colorbar(im, ax=target_ax, fraction=0.046, pad=0.04, label="Loading")
    else:
        raise ValueError(f"Unknown style={style!r}.")

    if title is not None:
        target_ax.set_title(title)

    return fig, target_ax


def _unpack_ring_predictions(
    series_list: Sequence[TreeRingSeries],
    ring_pred: Any,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Normalize flat or per-series ring predictions against ``series_list``."""
    n_tot = sum(s.n_obs for s in series_list)
    if (
        isinstance(ring_pred, (list, tuple))
        and len(ring_pred) == len(series_list)
        and (len(ring_pred) == 0 or np.ndim(ring_pred[0]) > 0)
    ):
        parts = [np.asarray(rp, dtype=np.float64).reshape(-1) for rp in ring_pred]
        for s, p in zip(series_list, parts):
            if p.size != s.n_obs:
                raise ValueError(
                    f"Series {s.name!r} has {s.n_obs} rings, got {p.size} predictions."
                )
        flat = np.concatenate(parts, axis=0) if parts else np.zeros((0,), dtype=np.float64)
        return parts, flat

    flat = np.asarray(ring_pred, dtype=np.float64).reshape(-1)
    if flat.size != n_tot:
        raise ValueError(
            f"ring_pred length {flat.size} does not match total observations {n_tot}."
        )
    parts = []
    cursor = 0
    for s in series_list:
        parts.append(flat[cursor : cursor + s.n_obs])
        cursor += s.n_obs
    return parts, flat


def plot_data_model_correlation(
    dataset_or_obs: Dataset | TreeRingSeries | Sequence[TreeRingSeries] | Any,
    ring_pred: Any,
    *,
    sigma_obs: Any | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (5.2, 4.6),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    r"""Plot 1-to-1 observed versus model-predicted ``Delta14C`` (‰) correlation.

    Renders ``1-sigma`` observation error bars (colour-coded by series or hemisphere
    when given a :class:`~ticktack.data.Dataset` or :class:`~ticktack.data.TreeRingSeries`),
    the dashed ``1:1`` identity line, matched square axis limits, and a compact
    goodness-of-fit summary (reduced ``chi^2 / N_obs`` and WRMS pull).

    Parameters
    ----------
    dataset_or_obs : Dataset, TreeRingSeries, sequence of TreeRingSeries, or array-like
        Structured tree-ring dataset/series or 1D observed ``Delta14C`` array (‰).
    ring_pred : array-like or sequence of array-like
        Model-predicted seasonal-binned ``Delta14C`` (‰) at each observation.
    sigma_obs : array-like, optional
        Observation ``1-sigma`` uncertainties (‰). Extracted automatically from
        ``dataset_or_obs`` when omitted for structured inputs.
    title : str, optional
        Optional axes title.
    figsize : tuple[float, float], default (5.2, 4.6)
        Figure size when ``ax`` is ``None``.
    ax : Axes, optional
        Existing Matplotlib ``Axes`` to draw into.

    Returns
    -------
    tuple[Figure, Axes]
        The ``(fig, ax)`` pair.
    """
    if ax is None:
        fig, target_ax = plt.subplots(1, 1, figsize=figsize)
    else:
        target_ax = ax
        fig = target_ax.figure

    is_structured = isinstance(dataset_or_obs, (Dataset, TreeRingSeries)) or (
        isinstance(dataset_or_obs, Sequence)
        and len(dataset_or_obs) > 0
        and isinstance(dataset_or_obs[0], TreeRingSeries)
    )

    if is_structured:
        series_list = _extract_series_list(dataset_or_obs)
        if not series_list:
            raise ValueError("dataset_or_obs contains no TreeRingSeries.")
        per_pred, pred_all = _unpack_ring_predictions(series_list, ring_pred)
        obs_all = np.concatenate([np.asarray(s.d14c, dtype=np.float64) for s in series_list])
        if sigma_obs is None:
            sig_all = np.concatenate(
                [np.asarray(s.sigma_d14c, dtype=np.float64) for s in series_list]
            )
            per_sig = [np.asarray(s.sigma_d14c, dtype=np.float64) for s in series_list]
        else:
            sig_all = np.broadcast_to(
                np.asarray(sigma_obs, dtype=np.float64), obs_all.shape
            ).reshape(-1)
            per_sig = []
            cur = 0
            for s in series_list:
                per_sig.append(sig_all[cur : cur + s.n_obs])
                cur += s.n_obs

        if len(series_list) <= 6:
            for idx, (s, p_s, sig_s) in enumerate(zip(series_list, per_pred, per_sig)):
                col = (
                    _PALETTE[0]
                    if s.hemisphere == "north" and len(series_list) <= 2
                    else (
                        _PALETTE[1]
                        if s.hemisphere == "south" and len(series_list) <= 2
                        else _PALETTE[idx % len(_PALETTE)]
                    )
                )
                marker = "o" if s.hemisphere == "north" else "s"
                hem_lbl = "NH" if s.hemisphere == "north" else "SH"
                target_ax.errorbar(
                    p_s,
                    np.asarray(s.d14c, dtype=np.float64),
                    yerr=sig_s,
                    fmt=marker,
                    color=col,
                    markersize=4.5,
                    elinewidth=1.0,
                    capsize=1.8,
                    alpha=0.85,
                    label=f"{s.name} ({hem_lbl})",
                )
        else:
            for hem, col, marker, tag in (
                ("north", _PALETTE[0], "o", "NH"),
                ("south", _PALETTE[1], "s", "SH"),
            ):
                idxs = [i for i, s in enumerate(series_list) if s.hemisphere == hem]
                if not idxs:
                    continue
                p_h = np.concatenate([per_pred[i] for i in idxs])
                o_h = np.concatenate(
                    [np.asarray(series_list[i].d14c, dtype=np.float64) for i in idxs]
                )
                s_h = np.concatenate([per_sig[i] for i in idxs])
                target_ax.errorbar(
                    p_h,
                    o_h,
                    yerr=s_h,
                    fmt=marker,
                    color=col,
                    markersize=4.2,
                    elinewidth=0.95,
                    capsize=1.6,
                    alpha=0.8,
                    label=f"{tag} ({len(idxs)} series)",
                )
    else:
        obs_all = np.asarray(dataset_or_obs, dtype=np.float64).reshape(-1)
        pred_all = np.asarray(ring_pred, dtype=np.float64).reshape(-1)
        if obs_all.size == 0 or obs_all.size != pred_all.size:
            raise ValueError(
                f"Observed ({obs_all.size}) and predicted ({pred_all.size}) must be non-empty 1D arrays of equal length."
            )
        sig_all = (
            None
            if sigma_obs is None
            else np.broadcast_to(
                np.asarray(sigma_obs, dtype=np.float64), obs_all.shape
            ).reshape(-1)
        )
        target_ax.errorbar(
            pred_all,
            obs_all,
            yerr=sig_all,
            fmt="o",
            color=_PALETTE[0],
            markersize=4.5,
            elinewidth=1.0,
            capsize=1.8,
            alpha=0.85,
            label=r"Observed $\Delta^{14}\mathrm{C}$",
        )

    obs_lo = obs_all if sig_all is None else (obs_all - sig_all)
    obs_hi = obs_all if sig_all is None else (obs_all + sig_all)
    val_min = float(min(np.min(pred_all), np.min(obs_lo)))
    val_max = float(max(np.max(pred_all), np.max(obs_hi)))
    span = val_max - val_min
    pad = 0.06 * span if span > 0.0 else 1.0
    lim_lo, lim_hi = val_min - pad, val_max + pad

    target_ax.plot(
        [lim_lo, lim_hi],
        [lim_lo, lim_hi],
        ls="--",
        lw=1.15,
        color="#333333",
        zorder=1,
        label="1:1 identity",
    )
    target_ax.set_xlim(lim_lo, lim_hi)
    target_ax.set_ylim(lim_lo, lim_hi)
    target_ax.set_aspect("equal", adjustable="box")

    resid = obs_all - pred_all
    if sig_all is not None:
        pulls = resid / sig_all
        chi2_red = float(np.mean(pulls**2))
        wrms_pull = float(np.sqrt(chi2_red))
        inv_var = 1.0 / (sig_all**2)
        wrms_permil = float(np.sqrt(np.sum(inv_var * (resid**2)) / np.sum(inv_var)))
        stat_txt = (
            rf"$\chi^2 / N_{{\mathrm{{obs}}}} = {chi2_red:.2f}$"
            "\n"
            rf"WRMS pull $= {wrms_pull:.2f}\sigma$ ({wrms_permil:.2f}‰)"
        )
    else:
        rms_permil = float(np.sqrt(np.mean(resid**2)))
        stat_txt = f"N = {obs_all.size}\nRMS = {rms_permil:.2f}‰"

    target_ax.text(
        0.04,
        0.96,
        stat_txt,
        transform=target_ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.2,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.9},
    )
    target_ax.set_xlabel(r"Predicted $\Delta^{14}\mathrm{C}$ (‰)")
    target_ax.set_ylabel(r"Observed $\Delta^{14}\mathrm{C}$ (‰)")
    target_ax.grid(True, alpha=0.25, linestyle="--")
    target_ax.legend(loc="lower right", frameon=True, fontsize=8.0)
    if title is not None:
        target_ax.set_title(title)

    return fig, target_ax


def plot_residual_map(
    dataset: Dataset | TreeRingSeries | Sequence[TreeRingSeries],
    ring_pred: Any,
    *,
    zscore: bool = True,
    cmap: str = "RdBu_r",
    vmax: float | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (9.5, 4.8),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    r"""Plot a 2D ``series × calendar year`` radiocarbon residual heatmap.

    Unobserved series-year cells remain ``NaN`` and are rendered in neutral grey so
    data gaps are immediately distinguishable from near-zero residuals. Uses a
    symmetric diverging colour scale ``vmin = -vmax`` with a labelled colourbar.

    Parameters
    ----------
    dataset : Dataset, TreeRingSeries, or sequence of TreeRingSeries
        Observed tree-ring series with audited calendar windows.
    ring_pred : array-like or sequence of array-like
        Model-predicted seasonal-binned ``Delta14C`` (‰) at each observation.
    zscore : bool, default True
        When ``True``, plot standardized pulls ``(d - m) / sigma``; when ``False``,
        plot raw residuals ``d - m`` (‰).
    cmap : str, default "RdBu_r"
        Diverging Matplotlib colourmap name.
    vmax : float, optional
        Positive colour-scale bound (``vmin = -vmax``). Defaults to
        ``max(3.0, np.nanpercentile(np.abs(grid), 98))`` when ``zscore=True``.
    title : str, optional
        Optional axes title.
    figsize : tuple[float, float], default (9.5, 4.8)
        Figure size when ``ax`` is ``None``.
    ax : Axes, optional
        Existing Matplotlib ``Axes`` to draw into.

    Returns
    -------
    tuple[Figure, Axes]
        The ``(fig, ax)`` pair.
    """
    series_list = _extract_series_list(dataset)
    if not series_list:
        raise ValueError("dataset must contain at least one TreeRingSeries.")

    per_pred, _ = _unpack_ring_predictions(series_list, ring_pred)
    series_years = [
        np.floor(np.asarray(s.t_starts, dtype=np.float64) + 1e-9).astype(np.int64)
        for s in series_list
    ]
    year_min = int(min(int(np.min(yrs)) for yrs in series_years))
    year_max = int(max(int(np.max(yrs)) for yrs in series_years))
    n_series = len(series_list)
    n_years = year_max - year_min + 1

    num_acc = np.zeros((n_series, n_years), dtype=np.float64)
    w_acc = np.zeros((n_series, n_years), dtype=np.float64)
    counts = np.zeros((n_series, n_years), dtype=np.int64)

    for row_idx, (s, pred_s, yrs) in enumerate(zip(series_list, per_pred, series_years)):
        obs_s = np.asarray(s.d14c, dtype=np.float64)
        sig_s = np.asarray(s.sigma_d14c, dtype=np.float64)
        inv_var_s = 1.0 / (sig_s**2)
        res_s = obs_s - pred_s
        cols = yrs - year_min
        np.add.at(num_acc[row_idx], cols, inv_var_s * res_s)
        np.add.at(w_acc[row_idx], cols, inv_var_s)
        np.add.at(counts[row_idx], cols, 1)

    grid = np.full((n_series, n_years), np.nan, dtype=np.float64)
    mask_obs = counts > 0
    if zscore:
        grid[mask_obs] = num_acc[mask_obs] / np.sqrt(w_acc[mask_obs])
    else:
        grid[mask_obs] = num_acc[mask_obs] / w_acc[mask_obs]

    if vmax is not None:
        eff_vmax = float(vmax)
        if not (np.isfinite(eff_vmax) and eff_vmax > 0.0):
            raise ValueError(f"vmax must be finite and positive, got {vmax!r}.")
    elif np.any(mask_obs):
        p98 = float(np.nanpercentile(np.abs(grid), 98.0))
        eff_vmax = max(3.0, p98) if zscore else (p98 if p98 > 0.0 else 1.0)
    else:
        eff_vmax = 3.0 if zscore else 1.0

    if ax is None:
        fig, target_ax = plt.subplots(1, 1, figsize=figsize)
    else:
        target_ax = ax
        fig = target_ax.figure

    cmap_obj = plt.get_cmap(cmap).with_extremes(bad="#E6E6E6")

    im = target_ax.imshow(
        grid,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap_obj,
        vmin=-eff_vmax,
        vmax=eff_vmax,
        extent=(year_min - 0.5, year_max + 0.5, n_series - 0.5, -0.5),
    )
    target_ax.set_yticks(np.arange(n_series))
    target_ax.set_yticklabels(
        [f"{s.name} ({'NH' if s.hemisphere == 'north' else 'SH'})" for s in series_list],
        fontsize=8.2,
    )
    target_ax.set_xlabel("Calendar Year (Astronomical CE)")
    cbar_label = (
        r"Standardized pull $(d - m) / \sigma$" if zscore else r"Residual $d - m$ (‰)"
    )
    fig.colorbar(im, ax=target_ax, pad=0.02, label=cbar_label)
    if title is not None:
        target_ax.set_title(title)

    return fig, target_ax


# ---------------------------------------------------------------------------
# Parametric Cornerplots via ChainConsumer >= 1.0 (stateless, silent logging)
# ---------------------------------------------------------------------------

_INDEXED_VAR_RE = re.compile(r"^(.+)\[(\d+)\]$")


def _extract_idata_df(
    idata: Any,
    param_names: Sequence[str] | Mapping[str, str] | None,
) -> pd.DataFrame:
    post = idata.posterior
    cols: dict[str, np.ndarray] = {}

    if isinstance(param_names, Mapping):
        for var_key, label in param_names.items():
            m = _INDEXED_VAR_RE.match(str(var_key))
            if var_key in post.data_vars:
                arr = np.asarray(post[var_key], dtype=np.float64)
                if arr.ndim <= 2:
                    cols[str(label)] = arr.reshape(-1)
                else:
                    flat = arr.reshape(-1, int(np.prod(arr.shape[2:])))
                    if flat.shape[1] == 1:
                        cols[str(label)] = flat[:, 0]
                    else:
                        for j in range(flat.shape[1]):
                            cols[f"{label}[{j}]"] = flat[:, j]
            elif m is not None and m.group(1) in post.data_vars:
                base, idx = m.group(1), int(m.group(2))
                arr = np.asarray(post[base], dtype=np.float64)
                flat = arr.reshape(-1, int(np.prod(arr.shape[2:])))
                cols[str(label)] = flat[:, idx]
            else:
                raise KeyError(
                    f"Posterior variable {var_key!r} not found in InferenceData.posterior "
                    f"(available: {list(post.data_vars)})."
                )
        return pd.DataFrame(cols)

    # Sequence or None: extract variables from posterior
    var_keys: list[str]
    rename_seq: list[str] | None = None
    if param_names is not None:
        seq = [str(x) for x in param_names]
        if all((k in post.data_vars) or (_INDEXED_VAR_RE.match(k) and _INDEXED_VAR_RE.match(k).group(1) in post.data_vars) for k in seq):  # type: ignore[union-attr]
            var_keys = seq
        else:
            var_keys = [str(k) for k in post.data_vars]
            rename_seq = seq
    else:
        var_keys = [str(k) for k in post.data_vars]

    for var_key in var_keys:
        m = _INDEXED_VAR_RE.match(var_key)
        if var_key in post.data_vars:
            arr = np.asarray(post[var_key], dtype=np.float64)
            if arr.ndim <= 2:
                cols[var_key] = arr.reshape(-1)
            else:
                flat = arr.reshape(-1, int(np.prod(arr.shape[2:])))
                if flat.shape[1] == 1:
                    cols[var_key] = flat[:, 0]
                else:
                    for j in range(flat.shape[1]):
                        cols[f"{var_key}[{j}]"] = flat[:, j]
        elif m is not None and m.group(1) in post.data_vars:
            base, idx = m.group(1), int(m.group(2))
            arr = np.asarray(post[base], dtype=np.float64)
            flat = arr.reshape(-1, int(np.prod(arr.shape[2:])))
            cols[var_key] = flat[:, idx]

    df = pd.DataFrame(cols)
    if rename_seq is not None:
        if len(rename_seq) != df.shape[1]:
            raise ValueError(
                f"param_names length ({len(rename_seq)}) does not match "
                f"posterior column count ({df.shape[1]})."
            )
        df.columns = rename_seq
    return df


def _to_chain_df_and_weights(
    obj: Any,
    *,
    param_names: Sequence[str] | Mapping[str, str] | None,
    weights: Any | None,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    w_arr = (
        None if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    )

    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        if isinstance(param_names, Mapping):
            keep = [k for k in param_names if k in df.columns]
            if keep:
                df = df[keep].rename(columns={k: str(param_names[k]) for k in keep})
        elif param_names is not None:
            seq = [str(x) for x in param_names]
            if all(k in df.columns for k in seq):
                df = df[seq]
            elif len(seq) == df.shape[1]:
                df.columns = seq
            else:
                raise ValueError(
                    f"param_names length ({len(seq)}) incompatible with DataFrame columns ({list(df.columns)})."
                )
        return df, w_arr

    # SMCResult: prefer resampled constrained draws in idata.posterior
    if hasattr(obj, "idata") and hasattr(obj, "particles") and hasattr(obj, "weights"):
        if getattr(obj, "idata", None) is not None and hasattr(obj.idata, "posterior"):
            df = _extract_idata_df(obj.idata, param_names)
            return df, w_arr
        arr = np.asarray(obj.particles, dtype=np.float64)
        w_smc = np.asarray(obj.weights, dtype=np.float64).reshape(-1) if w_arr is None else w_arr
        if isinstance(param_names, Mapping):
            names = [str(v) for v in param_names.values()]
        elif param_names is not None:
            names = [str(v) for v in param_names]
        else:
            names = [f"p{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=names), w_smc

    # arviz.InferenceData
    if hasattr(obj, "posterior"):
        return _extract_idata_df(obj, param_names), w_arr

    # Mapping of 1D arrays {label: samples}
    if isinstance(obj, Mapping):
        if isinstance(param_names, Mapping):
            cols = {
                str(lbl): np.asarray(obj[k], dtype=np.float64).reshape(-1)
                for k, lbl in param_names.items()
                if k in obj
            }
            if not cols:
                cols = {
                    str(k): np.asarray(v, dtype=np.float64).reshape(-1)
                    for k, v in obj.items()
                }
        else:
            cols = {
                str(k): np.asarray(v, dtype=np.float64).reshape(-1)
                for k, v in obj.items()
            }
            if param_names is not None:
                seq = [str(x) for x in param_names]
                if all(k in cols for k in seq):
                    cols = {k: cols[k] for k in seq}
                elif len(seq) == len(cols):
                    cols = dict(zip(seq, cols.values()))
                else:
                    raise ValueError(
                        f"param_names length ({len(seq)}) incompatible with mapping keys ({list(cols)})."
                    )
        return pd.DataFrame(cols), w_arr

    # 2D numeric array (n_samples, n_params)
    arr = np.asarray(obj, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] == 0:
        raise ValueError(
            f"Expected 2D sample array (n_samples, n_params), got shape {arr.shape}."
        )
    if isinstance(param_names, Mapping):
        names = [str(v) for v in param_names.values()]
    elif param_names is not None:
        names = [str(v) for v in param_names]
    else:
        names = [f"p{i}" for i in range(arr.shape[1])]
    if len(names) != arr.shape[1]:
        raise ValueError(
            f"param_names length ({len(names)}) must match sample dimension ({arr.shape[1]})."
        )
    return pd.DataFrame(arr, columns=names), w_arr


def _is_chain_entry(val: Any) -> bool:
    if isinstance(val, (Mapping, pd.DataFrame)):
        return True
    if hasattr(val, "posterior") or (hasattr(val, "particles") and hasattr(val, "weights")):
        return True
    if isinstance(val, tuple) and len(val) in (2, 3) and not np.isscalar(val[0]):
        return True
    if hasattr(val, "ndim") and getattr(val, "ndim") == 2:
        return True
    return False


def _normalize_chainconsumer_math_label(label: str) -> str:
    """Convert mixed ``'$math$ suffix'`` into a single outer ``'$...$'`` for ChainConsumer summaries."""
    s = str(label).strip()
    if "$" not in s:
        return s
    if s.startswith("$") and s.endswith("$") and s.count("$") == 2:
        return s
    # Split on $...$ math segments and wrap plain text segments in \mathrm{...}
    parts = re.split(r"(\$[^$]*\$)", s)
    inner_parts: list[str] = []
    for p in parts:
        if not p:
            continue
        if p.startswith("$") and p.endswith("$"):
            inner_parts.append(p[1:-1])
        else:
            escaped = p.replace(" ", r"\;")
            inner_parts.append(rf"\mathrm{{{escaped}}}")
    return "$" + "".join(inner_parts) + "$"


def plot_corner(
    chains_or_samples: Any,
    *,
    param_names: Sequence[str] | Mapping[str, str] | None = None,
    weights: Any | None = None,
    name: str = "Posterior",
    truth: Mapping[str, float] | Sequence[float] | None = None,
    truth_name: str = "MAP",
    colors: Sequence[str] | None = None,
    figsize: tuple[float, float] | None = None,
    summarize: bool = False,
    plot_kwargs: Mapping[str, Any] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Render a stateless parametric cornerplot via ``ChainConsumer >= 1.0``.

    Accepts a single posterior chain (``Mapping[str, array]``, ``DataFrame``,
    ``arviz.InferenceData``, ``SMCResult``, or 2D array ``(N, D)``) or multiple
    chains (``Mapping[chain_label, chain_input]`` or ``Sequence`` of chain
    inputs / ``(samples, weights, label)`` tuples).

    Strictly preserves global ``matplotlib.rcParams`` via ``plt.rc_context()``
    and silences ``chainconsumer`` / ``matplotlib.font_manager`` warnings so
    zero ``stderr`` text leaks into Jupyter notebooks.
    """
    chain_specs: list[tuple[str, pd.DataFrame, np.ndarray | None]] = []

    if (
        isinstance(chains_or_samples, pd.DataFrame)
        or hasattr(chains_or_samples, "posterior")
        or (
            hasattr(chains_or_samples, "particles")
            and hasattr(chains_or_samples, "weights")
        )
    ):
        df_c, w_c = _to_chain_df_and_weights(
            chains_or_samples, param_names=param_names, weights=weights
        )
        chain_specs.append((str(name), df_c, w_c))
    elif isinstance(chains_or_samples, Mapping) and chains_or_samples:
        first_val = next(iter(chains_or_samples.values()))
        if _is_chain_entry(first_val):
            for ch_label, entry in chains_or_samples.items():
                if isinstance(entry, tuple) and len(entry) == 2 and not np.isscalar(entry[0]):
                    df_c, w_c = _to_chain_df_and_weights(
                        entry[0], param_names=param_names, weights=entry[1]
                    )
                else:
                    df_c, w_c = _to_chain_df_and_weights(
                        entry, param_names=param_names, weights=None
                    )
                chain_specs.append((str(ch_label), df_c, w_c))
        else:
            df_c, w_c = _to_chain_df_and_weights(
                chains_or_samples, param_names=param_names, weights=weights
            )
            chain_specs.append((str(name), df_c, w_c))
    elif (
        isinstance(chains_or_samples, (list, tuple))
        and len(chains_or_samples) > 0
        and _is_chain_entry(chains_or_samples[0])
    ):
        for idx, entry in enumerate(chains_or_samples):
            if isinstance(entry, tuple) and len(entry) == 3:
                raw_s, raw_w, raw_lbl = entry
                df_c, w_c = _to_chain_df_and_weights(
                    raw_s, param_names=param_names, weights=raw_w
                )
                chain_specs.append((str(raw_lbl), df_c, w_c))
            elif isinstance(entry, tuple) and len(entry) == 2:
                raw_s, second = entry
                if isinstance(second, str):
                    df_c, w_c = _to_chain_df_and_weights(
                        raw_s, param_names=param_names, weights=None
                    )
                    chain_specs.append((second, df_c, w_c))
                else:
                    df_c, w_c = _to_chain_df_and_weights(
                        raw_s, param_names=param_names, weights=second
                    )
                    chain_specs.append((f"Chain {idx + 1}", df_c, w_c))
            else:
                df_c, w_c = _to_chain_df_and_weights(
                    entry, param_names=param_names, weights=None
                )
                chain_specs.append((f"Chain {idx + 1}", df_c, w_c))
    else:
        df_c, w_c = _to_chain_df_and_weights(
            chains_or_samples, param_names=param_names, weights=weights
        )
        chain_specs.append((str(name), df_c, w_c))

    if not chain_specs or chain_specs[0][1].shape[1] == 0:
        raise ValueError("plot_corner requires at least one non-empty parameter column.")

    palette = list(_PALETTE) if colors is None else list(colors)
    consumer = ChainConsumer()

    eff_summarize = bool(
        (plot_kwargs or {}).get(
            "summarise", (plot_kwargs or {}).get("summarize", summarize)
        )
    )
    raw_cols = list(chain_specs[0][1].columns)
    col_map = (
        {c: _normalize_chainconsumer_math_label(c) for c in raw_cols}
        if eff_summarize
        else {c: c for c in raw_cols}
    )
    param_cols = [col_map[c] for c in raw_cols]
    for idx, (ch_label, df_c, w_c) in enumerate(chain_specs):
        df_use = df_c.rename(columns=col_map).copy()
        if w_c is not None:
            if w_c.shape[0] != len(df_use):
                raise ValueError(
                    f"weights length ({w_c.shape[0]}) must match sample count ({len(df_use)})."
                )
            df_use["weight"] = w_c
        consumer.add_chain(
            Chain(
                samples=df_use,
                name=ch_label,
                color=palette[idx % len(palette)],
                shade=True,
                shade_alpha=0.45 if len(chain_specs) > 1 else 0.55,
                smooth=3,
            )
        )

    if truth is not None:
        truth_loc: dict[str, float] = {}
        if isinstance(truth, Mapping):
            for k, v in truth.items():
                mapped_k = (
                    str(param_names[k])
                    if isinstance(param_names, Mapping) and k in param_names
                    else str(k)
                )
                norm_k = col_map.get(mapped_k, mapped_k)
                if norm_k in param_cols:
                    truth_loc[norm_k] = float(v)
        else:
            t_arr = np.asarray(truth, dtype=np.float64).reshape(-1)
            if t_arr.size != len(param_cols):
                raise ValueError(
                    f"truth sequence length ({t_arr.size}) must match parameter count ({len(param_cols)})."
                )
            truth_loc = {col: float(val) for col, val in zip(param_cols, t_arr)}
        if truth_loc:
            consumer.add_truth(
                Truth(
                    location=truth_loc,
                    name=truth_name,
                    color="#333333",
                    line_style="--",
                    line_width=1.2,
                )
            )

    cfg_dict: dict[str, Any] = {
        "usetex": False,
        "diagonal_tick_labels": False,
        "spacing": 0.0,
        "max_ticks": 4,
        "summarise": eff_summarize,
        "show_legend": len(chain_specs) > 1 or truth is not None,
    }
    if plot_kwargs:
        extra_kw = dict(plot_kwargs)
        if "summarize" in extra_kw:
            extra_kw["summarise"] = bool(extra_kw.pop("summarize"))
        cfg_dict.update(extra_kw)
    consumer.set_plot_config(PlotConfig(**cfg_dict))

    cc_logger = logging.getLogger("chainconsumer")
    fm_logger = logging.getLogger("matplotlib.font_manager")
    prev_cc_level = cc_logger.level
    prev_fm_level = fm_logger.level
    cc_logger.setLevel(logging.ERROR)
    fm_logger.setLevel(logging.ERROR)
    try:
        with plt.rc_context():
            if figsize is not None:
                fig = consumer.plotter.plot(columns=param_cols, figsize=figsize)
            else:
                fig = consumer.plotter.plot(columns=param_cols)
    finally:
        cc_logger.setLevel(prev_cc_level)
        fm_logger.setLevel(prev_fm_level)

    return fig, np.asarray(fig.axes, dtype=object)


# ---------------------------------------------------------------------------
# arXiv:2210.13775 TikZ Carbon-Box Geometry & Reservoir-Node Alpha Modes
# ---------------------------------------------------------------------------

_TIKZ_SHORT_LABELS: dict[str, str] = {
    "stratosphere": "S",
    "troposphere": "T",
    "biosphere": "B",
    "short-lived biota": "SLB",
    "short lived biota": "SLB",
    "long-lived biota": "LLB",
    "long lived biota": "LLB",
    "litter": "L",
    "soil": "So",
    "peat": "P",
    "marine surface": "MS",
    "surface water": "SW",
    "surface biota": "SB",
    "intermediate & deep water": "I&DW",
    "intermediate and deep water": "I&DW",
    "sedimentary sink": "SS",
}


def _tikz_short_label(box_name: str) -> str:
    key = box_name.strip().lower()
    if key in _TIKZ_SHORT_LABELS:
        return _TIKZ_SHORT_LABELS[key]
    if key.startswith("sink_") or key.startswith("sink "):
        suffix = key.replace("sink_", "").replace("sink ", "").strip()
        return f"Sk{suffix}"
    words = [w for w in re.split(r"[\s_\-]+", box_name.strip()) if w]
    if len(words) == 1:
        return words[0][:3]
    return "".join(w[0].upper() for w in words[:4])


def _tikz_node_layout(
    topology: Any,
) -> tuple[np.ndarray, list[str], str]:
    """Compute exact arXiv:2210.13775 TikZ reservoir coordinates ``(n_boxes, 2)``."""
    n_boxes = int(topology.n_boxes)
    names = list(topology.box_names)
    labels = [_tikz_short_label(nm) for nm in names]
    radius = 0.8

    def _pol(deg: float, cy: float = 0.0) -> tuple[float, float]:
        rad = math.radians(deg)
        return (radius * math.cos(rad), cy + radius * math.sin(rad))

    # 22-box two-hemisphere (Brehm21 / Buntgen18): NH on TOP (0, 0), SH on BOTTOM (0, -2.15),
    # mirrored across the equator so S, T, SW, SB, and I&DW face each other directly.
    if n_boxes == 22 and getattr(topology, "is_hemispheric", False):
        coords = np.zeros((22, 2), dtype=np.float64)
        # Northern Hemisphere (boxes 11..21, j = 1..11) in top ring centred at (0, 0)
        for j in (1, 2):
            coords[10 + j] = _pol((j + 8) * 360.0 / 11.0, 0.0)
        for j in (3, 4, 5):
            coords[10 + j] = _pol((j + 3) * 360.0 / 11.0, 0.0)
        for j in range(6, 11):
            coords[10 + j] = _pol(((j + 6) % 11) * 360.0 / 11.0, 0.0)
        coords[21] = _pol(0.0, 0.0)
        # Southern Hemisphere (boxes 0..10, i = 1..11) vertically mirrored at (0, -2.15)
        cy_sh = -2.15
        for i in range(1, 12):
            coords[i - 1] = (coords[10 + i, 0], cy_sh - coords[10 + i, 1])
        return coords, labels, "two_hemisphere_22"

    # 11-box global (Guttler15)
    if n_boxes == 11 and labels[:3] == ["S", "T", "SW"]:
        coords = np.zeros((11, 2), dtype=np.float64)
        for i in (1, 2):
            coords[i - 1] = _pol((i + 8) * 360.0 / 11.0, 0.0)
        for i in (3, 4, 5):
            coords[i - 1] = _pol((i + 3) * 360.0 / 11.0, 0.0)
        for i in range(6, 11):
            coords[i - 1] = _pol(((i + 6) % 11) * 360.0 / 11.0, 0.0)
        coords[10] = _pol(0.0, 0.0)
        return coords, labels, "global_11"

    # 4-box (Miyake17)
    if n_boxes == 4 and labels == ["S", "T", "MS", "B"]:
        coords = np.array([_pol(i * 90.0, 0.0) for i in range(1, 5)], dtype=np.float64)
        return coords, labels, "miyake_4"

    # Generic circular fallback
    coords = np.array(
        [_pol(k * 360.0 / max(n_boxes, 1), 0.0) for k in range(n_boxes)],
        dtype=np.float64,
    )
    return coords, labels, "generic"


def plot_cbm_tikz_modes(
    cbm_or_topology: Any,
    spectrum_or_loadings: Any,
    *,
    modes: Sequence[int] = (0, 1, 2, 3, 4, 5),
    loading: Literal["auto", "mass", "net_rate", "incident_rate"] = "auto",
    eigenvalues: Any | None = None,
    ncols: int = 3,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    axes: Axes | Sequence[Axes] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Render Fisher eigenmodes as alpha-weighted reservoir nodes on the ``arXiv:2210.13775`` TikZ layout.

    Uses the exact reservoir node coordinates and abbreviations of Zhang et al.
    (2022, ``arXiv:2210.13775``) for 11-box (``Guttler15``), 22-box two-hemisphere
    (``Brehm21``, ``Buntgen18``), and 4-box (``Miyake17``) carbon-box models, with
    a circular fallback for custom topologies such as ``EffectiveCBM``.

    Each mode's signed reservoir loading ``u_b = w_b / max(|w_b|)`` in ``[-1, 1]``
    controls both symmetric diverging ``RdBu_r`` node colour (red positive, blue
    negative) and node fill opacity ``alpha = 0.12 + 0.88 * |u_b|``, while directed
    edges fade by ``alpha = 0.06 + 0.70 * |d ln k_e| / max(|d ln k|)``.
    """
    topology = (
        cbm_or_topology.topology
        if hasattr(cbm_or_topology, "topology")
        else cbm_or_topology
    )
    n_boxes = int(topology.n_boxes)
    edges_src = tuple(int(s) for s in topology.edges_src)
    edges_dst = tuple(int(d) for d in topology.edges_dst)
    n_edges = len(edges_src)

    mu_arr: np.ndarray | None = None
    if eigenvalues is not None:
        mu_arr = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    elif hasattr(spectrum_or_loadings, "eigenvalues"):
        mu_arr = np.asarray(spectrum_or_loadings.eigenvalues, dtype=np.float64).reshape(-1)

    edge_loadings: np.ndarray | None = None
    if hasattr(spectrum_or_loadings, "parameter_loadings"):
        p_load = np.asarray(spectrum_or_loadings.parameter_loadings, dtype=np.float64)
        tie_rows = getattr(cbm_or_topology, "tie_matrix_rows", None)
        if tie_rows is not None:
            m_tie = np.asarray(tie_rows, dtype=np.float64)
            p_free = m_tie.shape[1]
            if p_load.shape[0] >= p_free:
                edge_loadings = m_tie @ p_load[:p_free, :]
        elif p_load.shape[0] >= n_edges and n_edges > 0:
            edge_loadings = p_load[:n_edges, :]

    if hasattr(spectrum_or_loadings, "parameter_loadings") or hasattr(
        spectrum_or_loadings, "mass_loadings"
    ):
        mass_l = getattr(spectrum_or_loadings, "mass_loadings", None)
        eff_loading = loading
        if eff_loading == "auto":
            eff_loading = "mass" if mass_l is not None else "net_rate"
        if eff_loading == "mass":
            if mass_l is None:
                raise ValueError(
                    "loading='mass' requested, but spectrum.mass_loadings is None; "
                    "pass mass_fn=lambda m: m.reservoir_masses() to cbm_fisher_spectrum "
                    "or use loading='net_rate'."
                )
            w_mat = np.asarray(mass_l, dtype=np.float64)
        elif eff_loading in ("net_rate", "incident_rate"):
            if edge_loadings is None:
                raise ValueError(
                    f"loading={eff_loading!r} requires edge rate loadings compatible with topology."
                )
            w_mat = np.zeros((n_boxes, edge_loadings.shape[1]), dtype=np.float64)
            for e_idx, (src, dst) in enumerate(zip(edges_src, edges_dst)):
                w_mat[dst, :] += edge_loadings[e_idx, :]
                if eff_loading == "net_rate":
                    w_mat[src, :] -= edge_loadings[e_idx, :]
                else:
                    w_mat[src, :] += edge_loadings[e_idx, :]
        else:
            raise ValueError(
                f"loading must be 'auto', 'mass', 'net_rate', or 'incident_rate'; got {loading!r}."
            )
    else:
        w_mat = np.asarray(spectrum_or_loadings, dtype=np.float64)
        if w_mat.ndim == 1:
            w_mat = w_mat[:, None]
        if w_mat.ndim != 2 or w_mat.shape[0] != n_boxes:
            raise ValueError(
                f"Explicit node loadings must have shape ({n_boxes}, n_modes), got {w_mat.shape}."
            )

    n_avail = int(w_mat.shape[1])
    modes_list = [int(k) for k in modes]
    if not modes_list:
        raise ValueError("modes must contain at least one mode index.")
    for k in modes_list:
        if k < 0 or k >= n_avail:
            raise ValueError(
                f"Mode index {k} out of range for {n_avail} available modes."
            )

    coords, labels, layout_kind = _tikz_node_layout(topology)
    is_two_hem = layout_kind == "two_hemisphere_22"
    n_panels = len(modes_list)
    eff_ncols = max(1, min(int(ncols), n_panels))
    nrows = int(math.ceil(n_panels / eff_ncols))

    if axes is None:
        if figsize is None:
            if is_two_hem:
                eff_size = (3.3 * eff_ncols + 0.85, 4.9 * nrows + 0.55)
            else:
                eff_size = (3.4 * eff_ncols + 0.85, 3.4 * nrows + 0.55)
        else:
            eff_size = figsize
        fig, ax_grid = plt.subplots(nrows, eff_ncols, figsize=eff_size, squeeze=False)
        flat_axes = list(ax_grid.ravel())
        for extra_ax in flat_axes[n_panels:]:
            extra_ax.set_visible(False)
        active_axes = flat_axes[:n_panels]
        out_axes = np.asarray(active_axes, dtype=object)
    else:
        if isinstance(axes, Axes):
            active_axes = [axes]
        else:
            active_axes = list(np.asarray(axes, dtype=object).ravel())
        if len(active_axes) < n_panels:
            raise ValueError(
                f"Provided {len(active_axes)} Axes for {n_panels} requested modes."
            )
        active_axes = active_axes[:n_panels]
        fig = active_axes[0].figure
        out_axes = np.asarray(active_axes, dtype=object)

    cmap = plt.get_cmap("RdBu_r")
    norm = Normalize(vmin=-1.0, vmax=1.0)
    node_r = 0.145 if n_boxes >= 11 else 0.185
    font_sz = 6.8 if n_boxes >= 11 else 8.5

    for ax, mode_k in zip(active_axes, modes_list):
        if is_two_hem:
            nh_patch = FancyBboxPatch(
                (-1.05, -1.02),
                2.10,
                2.04,
                boxstyle="round,pad=0.04,rounding_size=0.08",
                facecolor="#ffe0b2",
                edgecolor="#e0a96d",
                linewidth=0.7,
                alpha=0.28,
                zorder=0,
            )
            sh_patch = FancyBboxPatch(
                (-1.05, -3.17),
                2.10,
                2.04,
                boxstyle="round,pad=0.04,rounding_size=0.08",
                facecolor="#cceeff",
                edgecolor="#7fb3d5",
                linewidth=0.7,
                alpha=0.28,
                zorder=0,
            )
            ax.add_patch(nh_patch)
            ax.add_patch(sh_patch)
            ax.text(
                -0.98,
                0.95,
                "NH",
                fontsize=8.5,
                fontweight="bold",
                color="#8c4b00",
                ha="left",
                va="top",
                zorder=1,
            )
            ax.text(
                -0.98,
                -3.10,
                "SH",
                fontsize=8.5,
                fontweight="bold",
                color="#1f5c7a",
                ha="left",
                va="bottom",
                zorder=1,
            )

        # Draw directed edges weighted by |d ln k_e|
        dk_abs = (
            np.abs(edge_loadings[:, mode_k])
            if edge_loadings is not None and mode_k < edge_loadings.shape[1]
            else None
        )
        dk_max = float(np.max(dk_abs)) if (dk_abs is not None and dk_abs.size > 0) else 0.0

        for e_idx, (src, dst) in enumerate(zip(edges_src, edges_dst)):
            p0 = coords[src]
            p1 = coords[dst]
            diff = p1 - p0
            dist = float(np.hypot(diff[0], diff[1]))
            if dist <= 2.0 * node_r:
                continue
            unit = diff / dist
            start = p0 + unit * (node_r * 1.04)
            end = p1 - unit * (node_r * 1.04)
            if dk_abs is not None and dk_max > 0.0:
                rel_e = float(dk_abs[e_idx]) / dk_max
                e_alpha = 0.06 + 0.70 * rel_e
                e_lw = 0.55 + 1.35 * rel_e
            else:
                e_alpha = 0.28
                e_lw = 0.8
            arrow = FancyArrowPatch(
                posA=(float(start[0]), float(start[1])),
                posB=(float(end[0]), float(end[1])),
                connectionstyle="arc3,rad=0.14",
                arrowstyle="-|>,head_length=3.2,head_width=1.8",
                linewidth=e_lw,
                color="#222222",
                alpha=e_alpha,
                zorder=2,
            )
            ax.add_patch(arrow)

        # Normalize signed per-reservoir loading u_b in [-1, 1]
        w_vec = w_mat[:, mode_k]
        w_max = float(np.max(np.abs(w_vec)))
        u_vec = w_vec / w_max if w_max > 0.0 else np.zeros_like(w_vec)

        for b_idx in range(n_boxes):
            x_b, y_b = float(coords[b_idx, 0]), float(coords[b_idx, 1])
            u_val = float(u_vec[b_idx])
            abs_u = abs(u_val)
            rgba = cmap(norm(u_val))
            fill_alpha = 0.12 + 0.88 * abs_u

            # White backing circle prevents arrows crossing behind transparent nodes
            ax.add_patch(
                Circle(
                    (x_b, y_b),
                    radius=node_r,
                    facecolor="white",
                    edgecolor="none",
                    zorder=3,
                )
            )
            ax.add_patch(
                Circle(
                    (x_b, y_b),
                    radius=node_r,
                    facecolor=(rgba[0], rgba[1], rgba[2], fill_alpha),
                    edgecolor="#222222",
                    linewidth=0.75 + 0.85 * abs_u,
                    zorder=4,
                )
            )
            txt_color = "white" if abs_u > 0.72 else "#111111"
            lbl_str = labels[b_idx]
            eff_fs = font_sz * 0.82 if len(lbl_str) >= 4 else font_sz
            ax.text(
                x_b,
                y_b,
                lbl_str,
                fontsize=eff_fs,
                fontweight="bold" if abs_u > 0.35 else "normal",
                color=txt_color,
                ha="center",
                va="center",
                zorder=5,
            )

        if mu_arr is not None and mode_k < mu_arr.size:
            mu_v = float(mu_arr[mode_k])
            if 0.0 < mu_v < 1e-2:
                exp_p = int(math.floor(math.log10(mu_v)))
                mant = mu_v / (10.0**exp_p)
                mu_tex = rf"{mant:.1f} \times 10^{{{exp_p}}}"
            else:
                mu_tex = f"{mu_v:.2g}"
            ax.set_title(
                rf"Mode {mode_k + 1} ($\mu_{{{mode_k + 1}}} = {mu_tex}$)",
                fontsize=9.5,
                pad=4.0,
            )
        else:
            ax.set_title(f"Mode {mode_k + 1}", fontsize=9.5, pad=4.0)

        ax.set_aspect("equal")
        ax.axis("off")
        if is_two_hem:
            ax.set_xlim(-1.14, 1.14)
            ax.set_ylim(-3.26, 1.12)
        else:
            ax.set_xlim(-1.14, 1.14)
            ax.set_ylim(-1.14, 1.14)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    if axes is None:
        fig.subplots_adjust(
            left=0.02,
            right=0.89,
            bottom=0.03,
            top=0.90 if title is not None else 0.94,
            wspace=0.08,
            hspace=0.14,
        )
        cbar_ax = fig.add_axes([0.915, 0.14, 0.022, 0.72])
        cbar = fig.colorbar(sm, cax=cbar_ax)
    else:
        cbar = fig.colorbar(sm, ax=active_axes, fraction=0.03, pad=0.03)
    cbar.set_label(
        r"Normalized signed reservoir loading (opacity $\propto |\alpha|$)",
        fontsize=9.0,
    )

    if title is not None:
        fig.suptitle(title, fontsize=11.0, y=0.975)

    return fig, out_axes


def _format_mode_metric_tex(val: float) -> str:
    """Format an eigenvalue or singular value cleanly for math-mode titles."""
    if not math.isfinite(val):
        return r"\mathrm{NaN}"
    if val == 0.0:
        return "0"
    if 0.0 < abs(val) < 1e-2 or abs(val) >= 1e3:
        exp_p = int(math.floor(math.log10(abs(val))))
        mant = val / (10.0**exp_p)
        return rf"{mant:.1f} \times 10^{{{exp_p}}}"
    return f"{val:.2g}"


# shape-exempt(carbon): public plotting entry point forwarding explicit prior_cbm_sd in SVD mode
def plot_cbm_mode_impulse_responses(
    cbm_or_result: Any,
    spectrum: Any | None = None,
    *,
    t_lag: Any | None = None,
    modes: Sequence[int] = (0, 1, 2, 3, 5, 8),
    prior_cbm_sd: float | Any | None = None,
    obs_window_yr: tuple[float, float] | None = (0.0, 6.5),
    ncols: int = 3,
    show_baseline: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot per-mode tropospheric impulse-response perturbations ``δh_k(τ)``.

    Accepts either a precomputed :class:`~ticktack.fisher.CBMModeImpulseResult`
    or a ``(cbm, spectrum)`` pair (calling
    :func:`~ticktack.fisher.cbm_mode_impulse_responses` on ``t_lag``).
    When ``spectrum is None`` and a CBM is passed directly, displays the
    continuous impulse-response Jacobian SVD modes (requiring an explicit
    ``prior_cbm_sd``).

    When ``show_baseline=True``, renders a compact full-width reference panel
    across the top showing the nominal tropospheric impulse responses
    ``h_{NH,0}(τ)`` and ``h_{SH,0}(τ)`` above the grid of signed ``+1 σ_prior``
    mode perturbations ``δh_k(τ)`` so both fast stratospheric rollover,
    observation-window sensitivity, and slow multi-decadal ventilation tails
    are immediately legible.

    Strictly stateless: wraps rendering in ``plt.rc_context()`` and never
    mutates global ``matplotlib.rcParams``.
    """
    from .fisher import CBMModeImpulseResult, cbm_mode_impulse_responses

    if isinstance(cbm_or_result, CBMModeImpulseResult):
        res = cbm_or_result
        avail_idx = list(res.mode_indices)
        if modes is not None:
            pos_list = [avail_idx.index(int(m)) for m in modes if int(m) in avail_idx]
            if not pos_list:
                pos_list = list(range(len(avail_idx)))
        else:
            pos_list = list(range(len(avail_idx)))
    else:
        lag_grid = (
            np.linspace(0.0, 30.0, 301, dtype=np.float64)
            if t_lag is None
            else np.asarray(t_lag, dtype=np.float64).reshape(-1)
        )
        if spectrum is not None:
            n_avail = int(np.asarray(spectrum.parameter_loadings).shape[1])
            req_modes = tuple(int(m) for m in modes if 0 <= int(m) < n_avail)
            if not req_modes:
                req_modes = tuple(range(min(6, n_avail)))
            res = cbm_mode_impulse_responses(
                cbm_or_result, lag_grid, spectrum, modes=req_modes
            )
            pos_list = list(range(len(res.mode_indices)))
        else:
            res_all = cbm_mode_impulse_responses(cbm_or_result, lag_grid, None, prior_cbm_sd=prior_cbm_sd)
            avail_idx = list(res_all.mode_indices)
            pos_list = [avail_idx.index(int(m)) for m in modes if int(m) in avail_idx]
            if not pos_list:
                pos_list = list(range(min(6, len(avail_idx))))
            res = res_all

    n_modes = len(pos_list)
    if n_modes == 0:
        raise ValueError("plot_cbm_mode_impulse_responses requires at least one mode.")

    tau = np.asarray(res.t_lag, dtype=np.float64)
    hems = list(res.hemispheres)
    evals = np.asarray(res.eigenvalues, dtype=np.float64)
    svals = np.asarray(res.singular_values, dtype=np.float64)

    has_nh = "north" in res.baseline
    has_sh = "south" in res.baseline
    # Collapse duplicate curves for single-hemisphere CBMs where NH == SH identically:
    single_hemi = (
        has_nh
        and has_sh
        and np.allclose(
            np.asarray(res.baseline["north"], dtype=np.float64),
            np.asarray(res.baseline["south"], dtype=np.float64),
            rtol=1e-12,
            atol=1e-14,
        )
    )

    if res.units == "permil_per_q0_yr":
        y_unit = r"$\text{‰}\,(q_0\,\mathrm{yr})^{-1}$"
    elif res.units == "permil_per_fluence":
        y_unit = r"$\text{‰}\,(\mathrm{atom\,cm^{-2}\,s^{-1}\,yr})^{-1}$"
    else:
        y_unit = r"$\mathrm{yr}^{-1}$"

    ncols_eff = max(1, min(int(ncols), n_modes))
    nrows_modes = int(math.ceil(n_modes / ncols_eff))

    nh_color = "#c44e52"
    sh_color = "#1f77b4"
    win_color = "#f0e442"

    with plt.rc_context():
        if figsize is None:
            base_h = 2.15 if show_baseline else 0.0
            figsize = (3.45 * ncols_eff + 0.35, base_h + 2.55 * nrows_modes + 0.55)

        fig = plt.figure(figsize=figsize)
        if show_baseline:
            gs = fig.add_gridspec(
                nrows_modes + 1,
                ncols_eff,
                height_ratios=[0.78] + [1.0] * nrows_modes,
                hspace=0.40,
                wspace=0.28,
            )
            ax_base = fig.add_subplot(gs[0, :])
            mode_axes = np.empty((nrows_modes, ncols_eff), dtype=object)
            for r in range(nrows_modes):
                for c in range(ncols_eff):
                    mode_axes[r, c] = fig.add_subplot(gs[r + 1, c], sharex=ax_base)
        else:
            ax_base = None
            gs = fig.add_gridspec(
                nrows_modes,
                ncols_eff,
                hspace=0.38,
                wspace=0.28,
            )
            mode_axes = np.empty((nrows_modes, ncols_eff), dtype=object)
            for r in range(nrows_modes):
                for c in range(ncols_eff):
                    share = mode_axes[0, 0] if (r > 0 or c > 0) else None
                    mode_axes[r, c] = fig.add_subplot(gs[r, c], sharex=share)

        win_label = None
        if obs_window_yr is not None:
            w0, w1 = float(obs_window_yr[0]), float(obs_window_yr[1])
            win_label = f"Post-spike window ({w0:g}–{w1:g} yr)"

        if ax_base is not None:
            if obs_window_yr is not None:
                ax_base.axvspan(
                    float(obs_window_yr[0]),
                    float(obs_window_yr[1]),
                    color=win_color,
                    alpha=0.22,
                    label=win_label,
                    zorder=0,
                )
            if single_hemi or (has_nh and not has_sh):
                h_n = np.asarray(res.baseline["north"], dtype=np.float64)
                ax_base.plot(tau, h_n, color=nh_color, lw=2.0, label=r"Troposphere $h_0(\tau)$")
            elif has_nh and has_sh:
                h_n = np.asarray(res.baseline["north"], dtype=np.float64)
                h_s = np.asarray(res.baseline["south"], dtype=np.float64)
                ax_base.plot(tau, h_n, color=nh_color, lw=2.0, label=r"NH troposphere $h_{\mathrm{NH},0}(\tau)$")
                ax_base.plot(tau, h_s, color=sh_color, lw=1.8, ls="--", label=r"SH troposphere $h_{\mathrm{SH},0}(\tau)$")
            else:
                for hi, h_key in enumerate(hems):
                    ax_base.plot(
                        tau,
                        np.asarray(res.baseline[h_key], dtype=np.float64),
                        color=_PALETTE[hi % len(_PALETTE)],
                        lw=1.8,
                        label=h_key.upper(),
                    )
            ax_base.axhline(0.0, color="#666666", lw=0.75, ls=":")
            ax_base.set_xlim(float(tau[0]), float(tau[-1]))
            ax_base.set_ylabel(rf"$h_0(\tau)$ ({y_unit})", fontsize=8.8)
            ax_base.set_title(
                r"Nominal tropospheric $\Delta^{14}\mathrm{C}$ impulse response $h_0(\tau)$",
                fontsize=9.5,
                fontweight="bold",
                pad=4.0,
            )
            ax_base.grid(True, alpha=0.22, ls="--")
            ax_base.legend(
                loc="upper right",
                ncol=3 if (has_nh and has_sh and not single_hemi) else 2,
                frameon=True,
                fontsize=8.0,
            )
            ax_base.tick_params(labelsize=8.2)

        flat_mode_axes = mode_axes.ravel()
        for slot, pos in enumerate(pos_list):
            ax = flat_mode_axes[slot]
            mode_k = int(res.mode_indices[pos])
            if obs_window_yr is not None:
                ax.axvspan(
                    float(obs_window_yr[0]),
                    float(obs_window_yr[1]),
                    color=win_color,
                    alpha=0.22,
                    label=(win_label if (slot == 0 and ax_base is None) else None),
                    zorder=0,
                )
            ax.axhline(0.0, color="#555555", lw=0.8, ls="--", zorder=1)

            if single_hemi or (has_nh and not has_sh):
                dh_n = np.asarray(res.mode_contributions["north"][pos], dtype=np.float64)
                ax.plot(
                    tau,
                    dh_n,
                    color=nh_color,
                    lw=1.8,
                    label=(r"$\delta h_k(\tau)$" if (slot == 0 and ax_base is None) else None),
                    zorder=3,
                )
                ax.fill_between(tau, 0.0, dh_n, color=nh_color, alpha=0.10, zorder=2)
            elif has_nh and has_sh:
                dh_n = np.asarray(res.mode_contributions["north"][pos], dtype=np.float64)
                dh_s = np.asarray(res.mode_contributions["south"][pos], dtype=np.float64)
                ax.plot(
                    tau,
                    dh_n,
                    color=nh_color,
                    lw=1.8,
                    label=(r"NH $\delta h_{\mathrm{NH},k}$" if (slot == 0 and ax_base is None) else None),
                    zorder=3,
                )
                ax.plot(
                    tau,
                    dh_s,
                    color=sh_color,
                    lw=1.65,
                    ls="--",
                    label=(r"SH $\delta h_{\mathrm{SH},k}$" if (slot == 0 and ax_base is None) else None),
                    zorder=3,
                )
                ax.fill_between(tau, 0.0, dh_n, color=nh_color, alpha=0.09, zorder=2)
                ax.fill_between(tau, 0.0, dh_s, color=sh_color, alpha=0.08, zorder=2)
            else:
                for hi, h_key in enumerate(hems):
                    dh_h = np.asarray(res.mode_contributions[h_key][pos], dtype=np.float64)
                    ax.plot(
                        tau,
                        dh_h,
                        color=_PALETTE[hi % len(_PALETTE)],
                        lw=1.7,
                        label=(h_key.upper() if (slot == 0 and ax_base is None) else None),
                        zorder=3,
                    )

            if res.source == "impulse_svd":
                sv_tex = _format_mode_metric_tex(float(svals[pos]))
                ax.set_title(
                    rf"SVD Mode {mode_k + 1} ($s_{{{mode_k + 1}}} = {sv_tex}$)",
                    fontsize=9.2,
                    pad=4.0,
                )
            else:
                mu_tex = _format_mode_metric_tex(float(evals[pos]))
                ax.set_title(
                    rf"Mode {mode_k + 1} ($\mu_{{{mode_k + 1}}} = {mu_tex}$)",
                    fontsize=9.2,
                    pad=4.0,
                )

            ax.set_xlim(float(tau[0]), float(tau[-1]))
            ax.grid(True, alpha=0.22, ls="--")
            ax.tick_params(labelsize=8.2)
            row_i, col_i = divmod(slot, ncols_eff)
            if col_i == 0:
                ax.set_ylabel(rf"$\delta h_k(\tau)$ ({y_unit})", fontsize=8.6)
            if row_i == nrows_modes - 1 or slot + ncols_eff >= n_modes:
                ax.set_xlabel(r"Lag $\tau$ (yr)", fontsize=8.8)
            if slot == 0 and ax_base is None:
                ax.legend(loc="best", frameon=True, fontsize=7.6)

        for slot in range(n_modes, nrows_modes * ncols_eff):
            flat_mode_axes[slot].set_visible(False)

        if title is not None:
            fig.suptitle(title, fontsize=11.0, fontweight="bold", y=0.985)
            fig.subplots_adjust(top=0.90, bottom=0.08, left=0.09, right=0.98)
        else:
            fig.subplots_adjust(top=0.94, bottom=0.08, left=0.09, right=0.98)

    if ax_base is not None:
        out_axes = np.empty(1 + n_modes, dtype=object)
        out_axes[0] = ax_base
        for i in range(n_modes):
            out_axes[i + 1] = flat_mode_axes[i]
        return fig, out_axes
    return fig, mode_axes


def _eval_field_realizations(
    field: GaussianField | LogGaussianField,
    latents: jax.Array,
    *,
    as_total_q: bool,
    q_std: float,
) -> np.ndarray:
    """Evaluate ``(N_draws, N_grid)`` curves for ``field`` across ``latents``."""
    if isinstance(field, GaussianField):
        dq = jax.vmap(lambda z: field.with_latent(z).evaluate_grid())(latents)
        vals = (q_std + dq) if as_total_q else dq
    elif isinstance(field, LogGaussianField):
        q = jax.vmap(lambda z: field.with_latent(z).evaluate_q_grid())(latents)
        vals = q if as_total_q else (q - q_std)
    else:
        raise TypeError(f"Expected GaussianField or LogGaussianField, got {type(field)!r}.")
    return np.asarray(vals, dtype=np.float64)


def _normalize_field_draws_input(
    field_or_fields: (
        GaussianField
        | LogGaussianField
        | Sequence[GaussianField | LogGaussianField]
        | Mapping[str, GaussianField | LogGaussianField]
    ),
) -> tuple[bool, list[str], list[GaussianField | LogGaussianField]]:
    """Normalize single, sequence, or mapping field inputs and verify shared ``n_grid``."""
    if isinstance(field_or_fields, (GaussianField, LogGaussianField)):
        return True, [""], [field_or_fields]
    if isinstance(field_or_fields, Mapping):
        labels, fields = list(field_or_fields.keys()), list(field_or_fields.values())
    else:
        fields = list(field_or_fields)
        labels = [f"Field {i + 1}" for i in range(len(fields))]
    if not fields:
        raise ValueError("field_or_fields cannot be empty.")
    n_grid = fields[0].n_grid
    if any(f.n_grid != n_grid for f in fields):
        raise ValueError("All compared fields must share the same n_grid.")
    return False, labels, fields


def _resolve_field_axes(
    is_single: bool,
    n_cols: int,
    figsize: tuple[float, float] | None,
    axes: np.ndarray | Sequence[Axes] | None,
) -> tuple[Figure, np.ndarray]:
    """Create or unpack ``Axes`` for single-field (4 axes) or multi-field (``2 x n_cols``)."""
    if is_single:
        if axes is None:
            fig = plt.figure(figsize=figsize or (9.0, 5.2), constrained_layout=True)
            gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.35])
            out = np.array([*[fig.add_subplot(gs[0, c]) for c in range(3)], fig.add_subplot(gs[1, :])], dtype=object)
            return fig, out
        arr_ax = np.asarray(axes, dtype=object).ravel()
        if arr_ax.size < 4:
            raise ValueError("Single-field plot_field_draws requires 4 Axes when supplied.")
        return arr_ax[3].figure, arr_ax
    if axes is None:
        fig, ax_grid = plt.subplots(
            2, n_cols, figsize=figsize or (3.3 * n_cols, 5.2), sharex="col", sharey="row",
            gridspec_kw={"height_ratios": [1.0, 1.25]}, constrained_layout=True,
        )
        return fig, np.atleast_2d(ax_grid).reshape(2, n_cols)
    out_axes = np.asarray(axes, dtype=object).reshape(2, n_cols)
    return out_axes[0, 0].figure, out_axes


def _plot_single_field_top_row(
    top_axes: Sequence[Axes],
    t_np: np.ndarray,
    curves: np.ndarray,
    ref_val: float,
    eff_ylabel: str,
) -> None:
    """Render the 3 individual draw panels in the top row of single-field mode."""
    y_min, y_max = float(np.percentile(curves, 1.0)), float(np.percentile(curves, 99.0))
    pad = 0.12 * max(y_max - y_min, 1e-6)
    for c in range(3):
        ax_c = top_axes[c]
        ax_c.plot(t_np, curves[c], color=_PALETTE[c], lw=1.5)
        ax_c.axhline(ref_val, color="#666666", lw=0.7, ls=":")
        ax_c.set_xlim(float(t_np[0]), float(t_np[-1]))
        ax_c.set_ylim(y_min - pad, y_max + pad)
        ax_c.set_title(f"Draw {c + 1}", fontsize=10)
        if c == 0:
            ax_c.set_ylabel(eff_ylabel)
        else:
            ax_c.tick_params(labelleft=False)


def _plot_field_ensemble(
    ax: Axes,
    t_np: np.ndarray,
    curves: np.ndarray,
    style: tuple[str, int, str | None],
    truth: ArrayLike | None = None,
) -> None:
    """Plot faint ensemble draws, bold Draw 1, and optional truth curve on ``ax``."""
    color, n_faint, draw1_label = style
    lw_faint, lw_truth = (0.75, 1.5) if draw1_label is not None else (0.7, 1.3)
    n_show = min(int(n_faint), curves.shape[0])
    for m in range(1, n_show):
        ax.plot(t_np, curves[m], color=color, lw=lw_faint, alpha=0.14)
    ax.plot(t_np, curves[0], color=color, lw=2.0, label=draw1_label or "Draw 1")
    if truth is not None:
        ax.plot(
            t_np, np.asarray(truth, dtype=np.float64), color="#111111", lw=lw_truth, ls="--",
            label="Truth" if draw1_label is not None else None,
        )
    ax.set_xlim(float(t_np[0]), float(t_np[-1]))
    if draw1_label is not None:
        ax.legend(loc="upper right", frameon=False)


def plot_field_draws(  # shape-exempt(carbon): existing public plotting API moved from fields.py preserving keyword signature
    field_or_fields: (
        GaussianField
        | LogGaussianField
        | Sequence[GaussianField | LogGaussianField]
        | Mapping[str, GaussianField | LogGaussianField]
    ),
    rng_key: jax.Array | None = None,
    *,
    latents: ArrayLike | None = None,
    n_faint: int = 50,
    as_total_q: bool = True,
    q_std: float = DEFAULT_Q_STD,
    truth: ArrayLike | None = None,
    title: str | None = None,
    xlabel: str = "Calendar Year (CE)",
    ylabel: str | None = None,
    figsize: tuple[float, float] | None = None,
    axes: np.ndarray | Sequence[Axes] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot 3 individual field draws on top and ``n_faint`` realizations (draw 1 bold) below.

    Radiocarbon 1D counterpart of ``virgil.plotting.plot_field_draws``:

    * **Single field**: top row shows draws 1, 2, and 3 in three panels; bottom panel
      spans all columns showing ``n_faint`` (default 50) faint realizations with draw 1
      highlighted in bold.
    * **Sequence or mapping of fields** (sharing the same grid length ``N``): draws a
      common bank of standard-normal excitations ``z ~ N(0, I_N)`` so every column maps
      the **identical** underlying random latents through its own power spectrum ``S(omega)``
      (top row: first 3 draws; bottom row: 50 faint draws + draw 1 bold).
    """
    n_draws = max(int(n_faint), 3)
    key = jax.random.PRNGKey(0) if rng_key is None else rng_key
    default_ylabel = (
        r"$Q(t)$ ($\mathrm{atoms\,cm^{-2}\,s^{-1}}$)"
        if as_total_q
        else r"$\delta q(t)$ ($\mathrm{atoms\,cm^{-2}\,s^{-1}}$)"
    )
    eff_ylabel = default_ylabel if ylabel is None else ylabel

    is_single, labels, fields = _normalize_field_draws_input(field_or_fields)
    n_grid = fields[0].n_grid
    z_bank = (
        jax.random.normal(key, (n_draws, n_grid), dtype=jnp.float64)
        if latents is None
        else jnp.asarray(latents, dtype=jnp.float64)
    )
    if z_bank.ndim != 2 or z_bank.shape[1] != n_grid or z_bank.shape[0] < 3:
        raise ValueError(f"latents must have shape (n_draws >= 3, {n_grid}), got {z_bank.shape}.")

    fig, out_axes = _resolve_field_axes(is_single, len(fields), figsize, axes)
    if is_single:
        curves = _eval_field_realizations(fields[0], z_bank, as_total_q=as_total_q, q_std=q_std)
        t_np = np.asarray(fields[0].t_grid, dtype=np.float64)
        _plot_single_field_top_row(
            out_axes[:3], t_np, curves, q_std if as_total_q else 0.0, eff_ylabel
        )
        _plot_field_ensemble(
            out_axes[3], t_np, curves, (_PALETTE[0], n_faint, "Draw 1 (highlighted)"), truth
        )
        out_axes[3].set_xlabel(xlabel)
        out_axes[3].set_ylabel(eff_ylabel)
    else:
        n_show = min(int(n_faint), z_bank.shape[0])
        for col, (lbl, f_col) in enumerate(zip(labels, fields)):
            col_color = _PALETTE[col % len(_PALETTE)]
            curves_c = _eval_field_realizations(f_col, z_bank, as_total_q=as_total_q, q_std=q_std)
            t_np = np.asarray(f_col.t_grid, dtype=np.float64)
            ax_top, ax_bot = out_axes[0, col], out_axes[1, col]
            for r_idx, ls_style in zip(range(3), ("-", "--", "-.")):
                ax_top.plot(
                    t_np, curves_c[r_idx], color=col_color, lw=1.35, ls=ls_style,
                    alpha=0.9 - 0.15 * r_idx, label=f"Draw {r_idx + 1}",
                )
            ax_top.set_title(lbl, fontsize=10)
            ax_top.set_xlim(float(t_np[0]), float(t_np[-1]))
            _plot_field_ensemble(ax_bot, t_np, curves_c, (col_color, n_faint, None), truth)
            ax_bot.set_xlabel(xlabel)
            if col == 0:
                ax_top.set_ylabel(f"3 draws\n{eff_ylabel}")
                ax_top.legend(loc="upper right", frameon=False, fontsize=8)
                ax_bot.set_ylabel(f"{n_show} draws\n{eff_ylabel}")

    if title:
        fig.suptitle(title, fontsize=11)
    return fig, out_axes


def _cell_edges(centres: np.ndarray, *, log_scale: bool) -> np.ndarray:
    """Compute ``N + 1`` cell edges around 1D monotone grid ``centres``."""
    c = np.asarray(centres, dtype=np.float64).reshape(-1)
    if c.size == 1:
        return np.array([c[0] / 1.25, c[0] * 1.25] if log_scale else [c[0] - 0.5, c[0] + 0.5], dtype=np.float64)
    vals = np.log(c) if log_scale else c
    mids = 0.5 * (vals[:-1] + vals[1:])
    edges = np.concatenate([[vals[0] - 0.5 * (vals[1] - vals[0])], mids, [vals[-1] + 0.5 * (vals[-1] - vals[-2])]])
    return np.exp(edges) if log_scale else edges


def _is_log_spaced(arr: np.ndarray) -> bool:
    if arr.size < 3 or np.any(arr <= 0.0):
        return False
    lin_diff, log_diff = np.diff(arr), np.diff(np.log(arr))
    lin_cv = float(np.std(lin_diff) / max(abs(float(np.mean(lin_diff))), 1e-12))
    log_cv = float(np.std(log_diff) / max(abs(float(np.mean(log_diff))), 1e-12))
    return log_cv < 0.25 * lin_cv


def _annotate_evidence_cells(
    ax: Axes, lens: np.ndarray, sigs: np.ndarray, surface: np.ndarray, mid_val: float,
) -> None:
    """Annotate 2D evidence grid cells with formatted numeric values."""
    for i, y_c in enumerate(sigs):
        for j, x_c in enumerate(lens):
            val = float(surface[i, j])
            ax.text(
                float(x_c), float(y_c), f"{val:.1f}", ha="center", va="center",
                fontsize=7.5, color="#111111" if val > mid_val else "white",
            )


def plot_evidence_grid(  # shape-exempt(carbon): existing public plotting API moved from fields.py preserving keyword signature
    grid: EvidenceGridResult,
    *,
    delta: bool = True,
    vmin: float | None = -25.0,
    annotate: bool = True,
    cmap: str = "magma",
    truth: tuple[float, float] | None = None,
    title: str | None = None,
    xlabel: str = r"Correlation timescale $\ell$ (yr)",
    ylabel: str = r"Field standard deviation $\sigma$ ($\mathrm{atoms\,cm^{-2}\,s^{-1}}$)",
    figsize: tuple[float, float] = (6.8, 5.0),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot annotated 2D ``(sigma, length_yr)`` ``Delta ln Z`` heatmap with white ``x`` at peak.

    Inspired by ``virgil/docs/imaging_gp.md``: renders ``Delta ln Z = ln Z - max ln Z``
    (or raw ``ln Z`` when ``delta=False``), formats logarithmic or linear axes cleanly,
    annotates cell values on compact grids (``<= 100`` cells), and marks the empirical-Bayes
    optimum ``(best_length_yr, best_sigma)`` with a high-contrast white ``x``.
    """
    fig, target_ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True) if ax is None else (ax.figure, ax)
    sigs = np.asarray(grid.sigmas, dtype=np.float64)
    lens = np.asarray(grid.lengths_yr, dtype=np.float64)
    lnz = np.asarray(grid.log_evidence, dtype=np.float64)
    surface = (lnz - float(np.nanmax(lnz))) if delta else lnz
    log_x, log_y = _is_log_spaced(lens), _is_log_spaced(sigs)
    eff_vmin, eff_vmax = (vmin, 0.0) if delta else (None, None)
    mesh = target_ax.pcolormesh(
        _cell_edges(lens, log_scale=log_x), _cell_edges(sigs, log_scale=log_y),
        surface, cmap=cmap, vmin=eff_vmin, vmax=eff_vmax, shading="flat",
    )
    fig.colorbar(mesh, ax=target_ax, label=r"$\Delta \ln Z$" if delta else r"$\ln Z$", pad=0.02)
    if log_x:
        target_ax.set_xscale("log")
    if log_y:
        target_ax.set_yscale("log")
    if annotate and surface.size <= 100:
        lo_val = float(eff_vmin) if eff_vmin is not None else float(np.nanmin(surface))
        hi_val = float(eff_vmax) if eff_vmax is not None else float(np.nanmax(surface))
        _annotate_evidence_cells(target_ax, lens, sigs, surface, 0.5 * (lo_val + hi_val))

    best_x, best_y = float(grid.best_length_yr), float(grid.best_sigma)
    # Dark contrast backing beneath the white 'x' marker so it stands out on bright peaks
    target_ax.scatter([best_x], [best_y], marker="x", s=125, c="#111111", linewidths=3.4, zorder=5)
    target_ax.scatter(
        [best_x], [best_y], marker="x", s=95, c="white", linewidths=2.0, zorder=6,
        label=rf"Max $\ln Z$ ($\sigma={best_y:.2g}$, $\ell={best_x:.2g}$ yr)",
    )
    if truth is not None:
        target_ax.scatter(
            [float(truth[1])], [float(truth[0])], marker="o", s=70,
            facecolors="none", edgecolors="#56B4E9", linewidths=1.8, zorder=6, label="Truth",
        )
    target_ax.set_xlabel(xlabel)
    target_ax.set_ylabel(ylabel)
    if title:
        target_ax.set_title(title)
    target_ax.legend(loc="lower right", frameon=True, framealpha=0.82, fontsize=8.5)
    return fig, target_ax


__all__ = [
    "plot_cbm_eigenmodes",
    "plot_cbm_mode_impulse_responses",
    "plot_cbm_tikz_modes",
    "plot_corner",
    "plot_d14c_fit",
    "plot_data_model_correlation",
    "plot_evidence_grid",
    "plot_field_draws",
    "plot_fisher_spectrum",
    "plot_production_posterior",
    "plot_residual_map",
]
