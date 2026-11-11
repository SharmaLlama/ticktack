"""Stateless scientific diagnostic plotting for ``ticktack 2.0``.

Every plotting function in this module is strictly stateless:
- Never mutates ``matplotlib.rcParams`` or global styles.
- Never relies on ``plt.gca()`` / ``plt.gcf()`` implicit state.
- Accepts optional caller-supplied ``Axes`` and always returns ``(fig, axes)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .data import Dataset, TreeRingSeries

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


from .fields import plot_evidence_grid, plot_field_draws  # noqa: E402

__all__ = [
    "plot_cbm_eigenmodes",
    "plot_d14c_fit",
    "plot_evidence_grid",
    "plot_field_draws",
    "plot_fisher_spectrum",
    "plot_production_posterior",
]

