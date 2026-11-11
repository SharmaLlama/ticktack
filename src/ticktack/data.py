"""Tree-ring radiocarbon data structures, seasonal weights, and calendar audit.

Defines immutable :mod:`zodiax` PyTrees (:class:`SeasonalWeights`,
:class:`TreeRingSeries`, and :class:`Dataset`) with mandatory calendar and
Southern Hemisphere ring-dating metadata:

* **Calendar conventions** (:data:`CalendarType`):
  - ``"astronomical_ce"``: continuous ISO 8601 / astronomical numbering
    (includes year ``0 = 1 BCE``, ``-5258 = 5259 BCE``).
  - ``"cal_bp"``: calibrated years Before Present (``1950.0 - bp``).
  - ``"historical_bce"``: traditional BC/AD numbering without year zero
    (raises :class:`ValueError` on ``year == 0`` and maps ``year < 0`` to
    ``year + 1`` on the continuous astronomical CE axis).
* **Ring-year conventions** (:data:`YearConvention`):
  - ``"schulman_start"``: Edmund Schulman's (1956) dendrochronological
    convention assigning a cross-New-Year Southern Hemisphere ring (Oct of
    ``Y`` through Mar of ``Y + 1``) to nominal year ``Y``, i.e. offsets
    ``[+0.75, +1.25]`` (used by ``Büntgen18_TAS01.csv``, where the 774 CE
    Miyake spike jumps at ``year = 774``).
  - ``"end_year"``: growth-completion calendar year convention assigning
    Oct of ``Y - 1`` through Mar of ``Y`` to nominal year ``Y``, i.e. offsets
    ``[-0.25, +0.25]`` (or ``[-4/12, +2/12]`` for ``"september-february"``;
    used by ``Guttler15_Kauri.csv``, where the 774 CE spike jumps at
    ``year = 775``, mapping to the exact same physical Austral summer
    ``774.75 .. 775.25``!).
  - ``"explicit_bounds"``: per-ring ``(t_starts, t_ends)`` in astronomical CE
    supplied directly (e.g. sub-annual earlywood/latewood slices).
"""

from __future__ import annotations

import csv
import dataclasses
import math
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import zodiax as zdx

CalendarType = Literal["astronomical_ce", "cal_bp", "historical_bce"]
YearConvention = Literal["schulman_start", "end_year", "explicit_bounds"]
HemisphereType = Literal["north", "south"]

VALID_CALENDARS: tuple[CalendarType, ...] = (
    "astronomical_ce",
    "cal_bp",
    "historical_bce",
)
VALID_YEAR_CONVENTIONS: tuple[YearConvention, ...] = (
    "schulman_start",
    "end_year",
    "explicit_bounds",
)
VALID_HEMISPHERES: tuple[HemisphereType, ...] = ("north", "south")

_MONTH_INDEX: dict[str, int] = {
    "january": 0,
    "jan": 0,
    "february": 1,
    "feb": 1,
    "march": 2,
    "mar": 2,
    "april": 3,
    "apr": 3,
    "may": 4,
    "june": 5,
    "jun": 5,
    "july": 6,
    "jul": 6,
    "august": 7,
    "aug": 7,
    "september": 8,
    "sep": 8,
    "sept": 8,
    "october": 9,
    "oct": 9,
    "november": 10,
    "nov": 10,
    "december": 11,
    "dec": 11,
}

_NAMED_SEASON_ALIASES: dict[str, str] = {
    "annual": "january-december",
    "full_year": "january-december",
    "earlywood": "april-june",
    "latewood": "july-september",
}


def convert_calendar_years(
    years: Sequence[float] | np.ndarray | jax.Array,
    calendar: CalendarType = "astronomical_ce",
) -> jax.Array:
    """Convert nominal calendar years to continuous astronomical CE (``float64``).

    Parameters
    ----------
    years : array-like
        Input nominal year labels.
    calendar : {"astronomical_ce", "cal_bp", "historical_bce"}
        Input calendar convention.

    Returns
    -------
    jax.Array
        Continuous ``float64`` astronomical CE years (where ``0.0`` is 1 BCE,
        ``-1.0`` is 2 BCE, and ``1950.0`` is 0 cal BP).
    """
    if calendar not in VALID_CALENDARS:
        raise ValueError(
            f"Invalid calendar={calendar!r}; expected one of {VALID_CALENDARS}."
        )
    arr_np = np.asarray(years, dtype=np.float64)
    if np.any(~np.isfinite(arr_np)):
        raise ValueError("Calendar years must be finite numbers.")

    if calendar == "astronomical_ce":
        return jnp.asarray(arr_np, dtype=jnp.float64)
    if calendar == "cal_bp":
        return jnp.asarray(1950.0 - arr_np, dtype=jnp.float64)

    # calendar == "historical_bce"
    if np.any(arr_np == 0.0):
        raise ValueError(
            "Year 0 does not exist in 'historical_bce' calendar numbering; "
            "1 BCE is -1 (astronomical 0) and 1 CE is +1."
        )
    converted = np.where(arr_np < 0.0, arr_np + 1.0, arr_np)
    return jnp.asarray(converted, dtype=jnp.float64)


def parse_growth_season_bounds(
    season: str,
    *,
    year_convention: YearConvention | None = None,
) -> tuple[float, float]:
    """Parse a month-range string into ``(offset_start, offset_end)`` in years.

    Each calendar month is treated as ``1/12`` of a year (January = ``[0, 1/12]``,
    ..., December = ``[11/12, 1]``). Cross-New-Year Southern Hemisphere ranges
    such as ``"october-march"`` or ``"september-february"`` require an explicit
    ``year_convention`` (``"schulman_start"`` or ``"end_year"``).
    """
    cleaned = season.strip().lower()
    if not cleaned:
        raise ValueError("Empty growth_season string.")
    cleaned = _NAMED_SEASON_ALIASES.get(cleaned, cleaned)

    parts = [p.strip() for p in cleaned.split("-")]
    if len(parts) == 1:
        if parts[0] not in _MONTH_INDEX:
            raise ValueError(f"Unrecognized month name in growth_season={season!r}.")
        m_start = m_end = _MONTH_INDEX[parts[0]]
    elif len(parts) == 2:
        if parts[0] not in _MONTH_INDEX or parts[1] not in _MONTH_INDEX:
            raise ValueError(
                f"Unrecognized month range {season!r}; expected 'StartMonth-EndMonth'."
            )
        m_start = _MONTH_INDEX[parts[0]]
        m_end = _MONTH_INDEX[parts[1]]
    else:
        raise ValueError(
            f"Invalid growth_season={season!r}; expected 'StartMonth-EndMonth'."
        )

    if year_convention is not None and year_convention not in VALID_YEAR_CONVENTIONS:
        raise ValueError(
            f"Invalid year_convention={year_convention!r}; expected one of "
            f"{VALID_YEAR_CONVENTIONS}."
        )
    if year_convention == "explicit_bounds":
        raise ValueError(
            "parse_growth_season_bounds cannot use year_convention='explicit_bounds'."
        )

    if m_start <= m_end:
        return (m_start / 12.0, (m_end + 1) / 12.0)

    # Cross-New-Year Southern Hemisphere growth season (m_start > m_end).
    if year_convention is None:
        raise ValueError(
            f"Cross-New-Year growth season {season!r} requires explicit "
            "year_convention='schulman_start' (e.g. [0.75, 1.25] for Büntgen18 SH) "
            "or year_convention='end_year' (e.g. [-0.25, 0.25] for Guttler15_Kauri)."
        )
    if year_convention == "schulman_start":
        return (m_start / 12.0, 1.0 + (m_end + 1) / 12.0)
    # year_convention == "end_year"
    return ((m_start - 12) / 12.0, (m_end + 1) / 12.0)


class SeasonalWeights(zdx.Base):
    """Sub-annual cellulose photosynthesis quadrature specification (:class:`zodiax.Base`).

    Represents the ring-formation operator for nominal astronomical year ``Y`` as
    a convex combination of ``M_sub >= 1`` sub-window boxcar averages:

    .. math::
        \\mathcal{B}_{\\mathrm{season}}[f](Y)
        = \\sum_{m=1}^{M_{\\mathrm{sub}}} w_m
          \\left( \\frac{1}{b_m - a_m} \\int_{Y + a_m}^{Y + b_m} f(t)\\,dt \\right),
        \\qquad \\sum_{m=1}^{M_{\\mathrm{sub}}} w_m = 1.

    For standard contiguous growing seasons (e.g. ``"april-september"`` or
    ``"october-march"``), ``M_sub = 1`` with ``weights = [1.0]``.
    """

    offset_starts: jax.Array
    offset_ends: jax.Array
    weights: jax.Array
    label: str = eqx.field(static=True)

    def __init__(
        self,
        offset_starts: Sequence[float] | np.ndarray | jax.Array,
        offset_ends: Sequence[float] | np.ndarray | jax.Array,
        weights: Sequence[float] | np.ndarray | jax.Array | None = None,
        *,
        label: str = "custom",
    ) -> None:
        starts_np = np.atleast_1d(np.asarray(offset_starts, dtype=np.float64))
        ends_np = np.atleast_1d(np.asarray(offset_ends, dtype=np.float64))
        if starts_np.ndim != 1 or ends_np.ndim != 1 or starts_np.shape != ends_np.shape:
            raise ValueError("offset_starts and offset_ends must be 1D arrays of equal length.")
        if starts_np.size == 0:
            raise ValueError("SeasonalWeights requires at least one sub-window.")
        if np.any(ends_np <= starts_np):
            raise ValueError("Every sub-window must satisfy offset_end > offset_start.")

        if weights is None:
            durations = ends_np - starts_np
            weights_np = durations / np.sum(durations)
        else:
            weights_np = np.atleast_1d(np.asarray(weights, dtype=np.float64))
            if weights_np.shape != starts_np.shape:
                raise ValueError("weights must have the same shape as offset_starts.")
            if np.any(weights_np <= 0.0):
                raise ValueError("All active sub-window weights must be strictly positive.")
            weights_np = weights_np / np.sum(weights_np)

        self.offset_starts = jnp.asarray(starts_np, dtype=jnp.float64)
        self.offset_ends = jnp.asarray(ends_np, dtype=jnp.float64)
        self.weights = jnp.asarray(weights_np, dtype=jnp.float64)
        self.label = str(label)

    @property
    def n_subwindows(self) -> int:
        """Number of active piecewise-constant sub-windows ``M_sub``."""
        return int(self.offset_starts.shape[0])

    @property
    def offset_start(self) -> jax.Array:
        """Overall start offset ``min(offset_starts)`` relative to year ``Y``."""
        return jnp.min(self.offset_starts)

    @property
    def offset_end(self) -> jax.Array:
        """Overall end offset ``max(offset_ends)`` relative to year ``Y``."""
        return jnp.max(self.offset_ends)

    @property
    def bounds(self) -> tuple[float, float]:
        """Python float tuple ``(offset_start, offset_end)``."""
        return (float(self.offset_start), float(self.offset_end))

    @property
    def midpoint_offset(self) -> jax.Array:
        """Weighted mean seasonal midpoint offset relative to year ``Y``."""
        mids = 0.5 * (self.offset_starts + self.offset_ends)
        return jnp.sum(self.weights * mids)

    @classmethod
    def from_bounds(
        cls,
        offset_start: float,
        offset_end: float,
        *,
        label: str = "bounds",
    ) -> SeasonalWeights:
        """Construct a single-window boxcar :class:`SeasonalWeights`."""
        return cls([offset_start], [offset_end], [1.0], label=label)

    @classmethod
    def from_string(
        cls,
        season: str,
        *,
        year_convention: YearConvention | None = None,
    ) -> SeasonalWeights:
        """Parse ``"StartMonth-EndMonth"`` into an exact single-window :class:`SeasonalWeights`."""
        start, end = parse_growth_season_bounds(season, year_convention=year_convention)
        return cls([start], [end], [1.0], label=season.strip().lower())

    @classmethod
    def from_subannual_weights(
        cls,
        bin_weights: Sequence[float] | np.ndarray | jax.Array,
        *,
        year_convention: YearConvention | None = None,
        offset_origin: float | None = None,
        period: float = 1.0,
        label: str = "subannual_weights",
    ) -> SeasonalWeights:
        """Build :class:`SeasonalWeights` from an ``M``-bin sub-annual density array.

        Supports arbitrary ``M``-bin photosynthesis profiles (e.g. 12 monthly or
        24 half-monthly weights), automatically merging adjacent equal-density
        bins into exact contiguous sub-windows and handling cross-New-Year cyclic
        Southern Hemisphere masks when ``bin_weights[0] > 0`` and ``bin_weights[-1] > 0``
        around an inactive mid-year winter gap.
        """
        w_np = np.asarray(bin_weights, dtype=np.float64).reshape(-1)
        n_bins = w_np.size
        if n_bins == 0:
            raise ValueError("bin_weights must be non-empty.")
        if np.any(w_np < 0.0) or np.sum(w_np) <= 0.0:
            raise ValueError("bin_weights must be non-negative with positive sum.")
        if period <= 0.0:
            raise ValueError("period must be positive.")

        dt = float(period) / n_bins
        # Detect cross-New-Year wrap (e.g. [1, 1, 1, 0, ..., 0, 1, 1, 1] for Oct-Mar)
        # when no explicit offset_origin was forced.
        is_cyclic_wrap = (
            offset_origin is None
            and n_bins >= 3
            and w_np[0] > 0.0
            and w_np[-1] > 0.0
            and np.any(w_np == 0.0)
        )

        raw_intervals: list[tuple[float, float, float]] = []
        if is_cyclic_wrap:
            if year_convention not in ("schulman_start", "end_year"):
                raise ValueError(
                    "Cross-New-Year subannual bin_weights require explicit "
                    "year_convention='schulman_start' or 'end_year'."
                )
            zero_indices = np.flatnonzero(w_np == 0.0)
            last_zero = int(zero_indices[-1])
            wrap_start = last_zero + 1  # in 1 .. n_bins - 1
            base_shift = 0.0 if year_convention == "schulman_start" else -float(period)
            ordered_indices = list(range(wrap_start, n_bins)) + list(range(last_zero + 1))
            for pos, b_idx in enumerate(ordered_indices):
                dens = float(w_np[b_idx])
                if dens > 0.0:
                    a = base_shift + (wrap_start + pos) * dt
                    b = a + dt
                    raw_intervals.append((a, b, dens))
        else:
            origin = 0.0 if offset_origin is None else float(offset_origin)
            for b_idx in range(n_bins):
                dens = float(w_np[b_idx])
                if dens > 0.0:
                    a = origin + b_idx * dt
                    b = a + dt
                    raw_intervals.append((a, b, dens))

        # Merge contiguous bins that share identical density to minimize M_sub.
        merged: list[tuple[float, float, float]] = []
        for a, b, dens in raw_intervals:
            if (
                merged
                and abs(a - merged[-1][1]) <= 1e-14
                and abs(dens - merged[-1][2]) <= 1e-14 * max(1.0, abs(dens))
            ):
                prev_a, _, prev_d = merged[-1]
                merged[-1] = (prev_a, b, prev_d)
            else:
                merged.append((a, b, dens))

        starts = [m[0] for m in merged]
        ends = [m[1] for m in merged]
        masses = [m[2] * (m[1] - m[0]) for m in merged]
        return cls(starts, ends, masses, label=label)

    def windows_for_years(
        self, years_ce: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return ``(sub_t_starts, sub_t_ends, weights)`` of shapes ``(N, M_sub)`` and ``(M_sub,)``."""
        y = jnp.asarray(years_ce, dtype=jnp.float64).reshape(-1)
        sub_starts = y[:, None] + self.offset_starts[None, :]
        sub_ends = y[:, None] + self.offset_ends[None, :]
        return sub_starts, sub_ends, self.weights


class TreeRingSeries(zdx.Base):
    """Single audited tree-ring ``Delta14C`` time series (:class:`zodiax.Base`).

    Stores both the continuous astronomical CE year labels ``years_ce`` and the
    exact physical integration windows ``(t_starts, t_ends)`` and sub-window
    quadrature ``(sub_t_starts, sub_t_ends)`` so solvers never re-parse strings
    or guess hemisphere dating conventions at evaluation time.
    """

    raw_years: jax.Array
    years_ce: jax.Array
    d14c: jax.Array
    sigma_d14c: jax.Array
    t_starts: jax.Array
    t_ends: jax.Array
    sub_t_starts: jax.Array
    sub_t_ends: jax.Array
    seasonal_weights: SeasonalWeights
    name: str = eqx.field(static=True)
    hemisphere: HemisphereType = eqx.field(static=True)
    calendar: CalendarType = eqx.field(static=True)
    year_convention: YearConvention = eqx.field(static=True)
    box_name: str = eqx.field(static=True)

    def __init__(
        self,
        years: Sequence[float] | np.ndarray | jax.Array,
        d14c: Sequence[float] | np.ndarray | jax.Array,
        sigma_d14c: Sequence[float] | np.ndarray | jax.Array,
        *,
        hemisphere: HemisphereType,
        calendar: CalendarType = "astronomical_ce",
        year_convention: YearConvention = "schulman_start",
        seasonal_weights: SeasonalWeights | str | None = None,
        t_starts: Sequence[float] | np.ndarray | jax.Array | None = None,
        t_ends: Sequence[float] | np.ndarray | jax.Array | None = None,
        name: str = "series",
        box_name: str = "Troposphere",
    ) -> None:
        if hemisphere not in VALID_HEMISPHERES:
            raise ValueError(
                f"Invalid hemisphere={hemisphere!r}; expected one of {VALID_HEMISPHERES}."
            )
        if calendar not in VALID_CALENDARS:
            raise ValueError(
                f"Invalid calendar={calendar!r}; expected one of {VALID_CALENDARS}."
            )
        if year_convention not in VALID_YEAR_CONVENTIONS:
            raise ValueError(
                f"Invalid year_convention={year_convention!r}; expected one of "
                f"{VALID_YEAR_CONVENTIONS}."
            )

        raw_np = np.asarray(years, dtype=np.float64).reshape(-1)
        d14c_np = np.asarray(d14c, dtype=np.float64).reshape(-1)
        sig_np = np.asarray(sigma_d14c, dtype=np.float64).reshape(-1)
        n = raw_np.size
        if n == 0:
            raise ValueError("TreeRingSeries cannot be empty.")
        if d14c_np.size != n or sig_np.size != n:
            raise ValueError("years, d14c, and sigma_d14c must have identical 1D length.")
        if np.any(~np.isfinite(d14c_np)) or np.any(~np.isfinite(sig_np)):
            raise ValueError("d14c and sigma_d14c must contain only finite values.")
        if np.any(sig_np <= 0.0):
            raise ValueError("sigma_d14c uncertainties must be strictly positive.")

        years_ce_arr = convert_calendar_years(raw_np, calendar=calendar)

        if year_convention == "explicit_bounds":
            if t_starts is None or t_ends is None:
                raise ValueError(
                    "year_convention='explicit_bounds' requires explicit t_starts and t_ends."
                )
            ts_np = np.asarray(t_starts, dtype=np.float64).reshape(-1)
            te_np = np.asarray(t_ends, dtype=np.float64).reshape(-1)
            if ts_np.size != n or te_np.size != n:
                raise ValueError("t_starts and t_ends must match length of years.")
            if np.any(te_np <= ts_np):
                raise ValueError("Explicit bounds must satisfy t_ends > t_starts.")
            self.t_starts = jnp.asarray(ts_np, dtype=jnp.float64)
            self.t_ends = jnp.asarray(te_np, dtype=jnp.float64)
            self.sub_t_starts = self.t_starts[:, None]
            self.sub_t_ends = self.t_ends[:, None]
            self.seasonal_weights = SeasonalWeights([0.0], [1.0], [1.0], label="explicit_bounds")
        else:
            if seasonal_weights is None:
                default_season = (
                    "april-september" if hemisphere == "north" else "october-march"
                )
                sw = SeasonalWeights.from_string(
                    default_season, year_convention=year_convention
                )
            elif isinstance(seasonal_weights, str):
                sw = SeasonalWeights.from_string(
                    seasonal_weights, year_convention=year_convention
                )
            elif isinstance(seasonal_weights, SeasonalWeights):
                sw = seasonal_weights
            else:
                raise TypeError(
                    "seasonal_weights must be a SeasonalWeights instance, month string, or None."
                )
            sub_s, sub_e, _ = sw.windows_for_years(years_ce_arr)
            self.seasonal_weights = sw
            self.sub_t_starts = sub_s
            self.sub_t_ends = sub_e
            self.t_starts = jnp.min(sub_s, axis=1)
            self.t_ends = jnp.max(sub_e, axis=1)

        self.raw_years = jnp.asarray(raw_np, dtype=jnp.float64)
        self.years_ce = years_ce_arr
        self.d14c = jnp.asarray(d14c_np, dtype=jnp.float64)
        self.sigma_d14c = jnp.asarray(sig_np, dtype=jnp.float64)
        self.name = str(name)
        self.hemisphere = hemisphere
        self.calendar = calendar
        self.year_convention = year_convention
        self.box_name = str(box_name)

    @property
    def n_obs(self) -> int:
        """Number of tree-ring observations in this series."""
        return int(self.d14c.shape[0])

    @property
    def t_midpoints(self) -> jax.Array:
        """Weighted physical growth-window midpoint times in astronomical CE."""
        sub_mids = 0.5 * (self.sub_t_starts + self.sub_t_ends)
        return jnp.sum(sub_mids * self.seasonal_weights.weights[None, :], axis=1)

    def quadrature_windows(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return ``(sub_t_starts, sub_t_ends, sub_weights)`` for seasonal integration."""
        return self.sub_t_starts, self.sub_t_ends, self.seasonal_weights.weights

    def with_error_scale(self, scale: float | Mapping[str, float]) -> TreeRingSeries:
        """Return a copy of this series with ``sigma_d14c`` multiplied by ``scale > 0``."""
        if isinstance(scale, Mapping):
            if self.name not in scale:
                raise KeyError(
                    f"Series {self.name!r} missing from scale mapping {list(scale)}."
                )
            s_val = float(scale[self.name])
        else:
            s_val = float(scale)
        if not (math.isfinite(s_val) and s_val > 0.0):
            raise ValueError(f"Error scale must be finite and positive, got {s_val}.")
        return eqx.tree_at(lambda s: s.sigma_d14c, self, self.sigma_d14c * s_val)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        hemisphere: HemisphereType,
        calendar: CalendarType = "astronomical_ce",
        year_convention: YearConvention | None = None,
        growth_season: SeasonalWeights | str | None = None,
        name: str | None = None,
        box_name: str = "Troposphere",
        sort_by_time: bool = True,
    ) -> TreeRingSeries:
        """Load a :class:`TreeRingSeries` from CSV using stdlib :mod:`csv` only.

        Recognized columns (case-insensitive, stripped):
        - ``year`` (required)
        - ``d14c`` (required)
        - ``sig_d14c`` (or ``sigma_d14c`` / ``d14c_err``, required)
        - ``growth_season`` (optional; first non-empty value used if ``growth_season`` arg is ``None``)
        - ``t_start``, ``t_end`` (required when ``year_convention="explicit_bounds"``)
        """
        csv_path = Path(path)
        series_name = csv_path.stem if name is None else name

        years_list: list[float] = []
        d14c_list: list[float] = []
        sig_list: list[float] = []
        t_start_list: list[float] = []
        t_end_list: list[float] = []
        csv_season: str | None = None

        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV file {csv_path} has no header row.")
            norm_map = {h.strip().lower(): h for h in reader.fieldnames if h is not None}

            year_col = norm_map.get("year")
            d14c_col = norm_map.get("d14c")
            sig_col = (
                norm_map.get("sig_d14c")
                or norm_map.get("sigma_d14c")
                or norm_map.get("d14c_err")
            )
            season_col = norm_map.get("growth_season")
            ts_col = norm_map.get("t_start")
            te_col = norm_map.get("t_end")

            if year_col is None or d14c_col is None or sig_col is None:
                raise ValueError(
                    f"CSV {csv_path} must contain 'year', 'd14c', and 'sig_d14c' columns; "
                    f"found {list(norm_map.keys())}."
                )

            for row in reader:
                if not row or all((v is None or not v.strip()) for v in row.values()):
                    continue
                years_list.append(float(row[year_col].strip()))
                d14c_list.append(float(row[d14c_col].strip()))
                sig_list.append(float(row[sig_col].strip()))
                if season_col is not None and csv_season is None:
                    raw_s = row.get(season_col)
                    if raw_s is not None and raw_s.strip():
                        csv_season = raw_s.strip()
                if ts_col is not None and te_col is not None:
                    raw_ts = row.get(ts_col, "")
                    raw_te = row.get(te_col, "")
                    if raw_ts and raw_te and raw_ts.strip() and raw_te.strip():
                        t_start_list.append(float(raw_ts.strip()))
                        t_end_list.append(float(raw_te.strip()))

        eff_season = growth_season if growth_season is not None else csv_season
        if year_convention is None:
            if ts_col is not None and te_col is not None and len(t_start_list) == len(years_list):
                eff_convention: YearConvention = "explicit_bounds"
            elif hemisphere == "south":
                raise ValueError(
                    f"Southern Hemisphere CSV {csv_path.name!r} requires explicit "
                    "year_convention ('schulman_start', 'end_year', or 'explicit_bounds') "
                    "to prevent a 1.00-year calendar shift."
                )
            else:
                eff_convention = "schulman_start"
        else:
            eff_convention = year_convention

        series = cls(
            years_list,
            d14c_list,
            sig_list,
            hemisphere=hemisphere,
            calendar=calendar,
            year_convention=eff_convention,
            seasonal_weights=eff_season,
            t_starts=t_start_list if eff_convention == "explicit_bounds" else None,
            t_ends=t_end_list if eff_convention == "explicit_bounds" else None,
            name=series_name,
            box_name=box_name,
        )

        if not sort_by_time or series.n_obs <= 1:
            return series

        # Ensure chronological ordering (e.g. descending IntCal20 blocks) using
        # stable mergesort so co-eval duplicate years keep their file order.
        order = np.argsort(np.asarray(series.t_starts), kind="mergesort")
        if np.all(order == np.arange(series.n_obs)):
            return series

        idx = jnp.asarray(order, dtype=jnp.int32)
        if eff_convention == "explicit_bounds":
            return cls(
                series.raw_years[idx],
                series.d14c[idx],
                series.sigma_d14c[idx],
                hemisphere=hemisphere,
                calendar=calendar,
                year_convention=eff_convention,
                t_starts=series.t_starts[idx],
                t_ends=series.t_ends[idx],
                name=series_name,
                box_name=box_name,
            )
        return cls(
            series.raw_years[idx],
            series.d14c[idx],
            series.sigma_d14c[idx],
            hemisphere=hemisphere,
            calendar=calendar,
            year_convention=eff_convention,
            seasonal_weights=series.seasonal_weights,
            name=series_name,
            box_name=box_name,
        )


class Dataset(zdx.Base):
    """Multi-series tree-ring radiocarbon dataset (:class:`zodiax.Base`).

    Unifies single-series (``S = 1``) and multi-series (``S > 1``) observations
    so the carbon-box model is solved once on a shared physical time span and
    projected onto all ``S`` series via their audited seasonal windows.
    """

    series: tuple[TreeRingSeries, ...]
    name: str = eqx.field(static=True)

    def __init__(
        self,
        series: Sequence[TreeRingSeries] | TreeRingSeries,
        *,
        name: str = "dataset",
    ) -> None:
        if isinstance(series, TreeRingSeries):
            items = (series,)
        else:
            items = tuple(series)
        if not items:
            raise ValueError("Dataset must contain at least one TreeRingSeries.")
        for s in items:
            if not isinstance(s, TreeRingSeries):
                raise TypeError(f"Expected TreeRingSeries, got {type(s)!r}.")
        self.series = items
        self.name = str(name)

    @property
    def n_series(self) -> int:
        """Number of constituent :class:`TreeRingSeries` ``S``."""
        return len(self.series)

    @property
    def n_obs(self) -> int:
        """Total observation count across all ``S`` series."""
        return sum(s.n_obs for s in self.series)

    @property
    def series_names(self) -> tuple[str, ...]:
        """Tuple of constituent series names."""
        return tuple(s.name for s in self.series)

    @property
    def hemispheres(self) -> tuple[HemisphereType, ...]:
        """Unique hemispheres represented in the dataset (in order of appearance)."""
        seen: list[HemisphereType] = []
        for s in self.series:
            if s.hemisphere not in seen:
                seen.append(s.hemisphere)
        return tuple(seen)

    @property
    def d14c_flat(self) -> jax.Array:
        """Concatenated ``(N_tot,)`` ``float64`` vector of observed ``Delta14C``."""
        return jnp.concatenate([s.d14c for s in self.series], axis=0)

    @property
    def sigma_flat(self) -> jax.Array:
        """Concatenated ``(N_tot,)`` ``float64`` vector of ``1-sigma`` errors."""
        return jnp.concatenate([s.sigma_d14c for s in self.series], axis=0)

    @property
    def t_starts_flat(self) -> jax.Array:
        """Concatenated ``(N_tot,)`` physical ring start times (astronomical CE)."""
        return jnp.concatenate([s.t_starts for s in self.series], axis=0)

    @property
    def t_ends_flat(self) -> jax.Array:
        """Concatenated ``(N_tot,)`` physical ring end times (astronomical CE)."""
        return jnp.concatenate([s.t_ends for s in self.series], axis=0)

    @property
    def t_midpoints_flat(self) -> jax.Array:
        """Concatenated ``(N_tot,)`` weighted ring midpoint times (astronomical CE)."""
        return jnp.concatenate([s.t_midpoints for s in self.series], axis=0)

    @property
    def series_indices(self) -> jax.Array:
        """Integer ``(N_tot,)`` array mapping each observation to series ``0 .. S - 1``."""
        blocks = [
            jnp.full((s.n_obs,), i, dtype=jnp.int32) for i, s in enumerate(self.series)
        ]
        return jnp.concatenate(blocks, axis=0)

    @property
    def time_span(self) -> tuple[float, float]:
        """Overall physical integration span ``(min(t_starts), max(t_ends))`` in CE."""
        t_min = min(float(jnp.min(s.t_starts)) for s in self.series)
        t_max = max(float(jnp.max(s.t_ends)) for s in self.series)
        return (t_min, t_max)

    def lab_offset_design_matrix(self) -> jax.Array:
        """Return ``(N_tot, S)`` indicator matrix ``H_lab`` for per-series lab offsets."""
        idx = self.series_indices
        return (idx[:, None] == jnp.arange(self.n_series, dtype=jnp.int32)[None, :]).astype(
            jnp.float64
        )

    def with_error_scale(self, scale: float | Mapping[str, float]) -> Dataset:
        """Return a copy of this dataset with ``sigma_d14c`` multiplied by ``scale``."""
        if isinstance(scale, Mapping):
            if len(set(self.series_names)) != self.n_series:
                raise ValueError(
                    "Per-series error scaling via a Mapping requires unique TreeRingSeries "
                    f"names in Dataset; got {self.series_names}."
                )
            unknown = set(scale) - set(self.series_names)
            if unknown:
                raise KeyError(
                    f"Unknown series names {sorted(unknown)} in scale mapping; "
                    f"dataset has {self.series_names}."
                )
            new_series = tuple(
                s.with_error_scale(scale[s.name]) if s.name in scale else s
                for s in self.series
            )
        else:
            s_val = float(scale)
            new_series = tuple(s.with_error_scale(s_val) for s in self.series)
        return Dataset(new_series, name=self.name)

    @classmethod
    def from_csvs(
        cls,
        paths: Sequence[str | Path],
        *,
        hemisphere: HemisphereType | Sequence[HemisphereType],
        calendar: CalendarType | Sequence[CalendarType] = "astronomical_ce",
        year_convention: YearConvention | Sequence[YearConvention | None] | None = None,
        growth_season: SeasonalWeights | str | Sequence[SeasonalWeights | str | None] | None = None,
        name: str = "dataset",
        sort_by_time: bool = True,
    ) -> Dataset:
        """Load multiple CSV files into a single :class:`Dataset`."""
        n = len(paths)
        if n == 0:
            raise ValueError("paths must be non-empty.")

        hems = [hemisphere] * n if isinstance(hemisphere, str) else list(hemisphere)
        cals = [calendar] * n if isinstance(calendar, str) else list(calendar)
        convs = (
            [year_convention] * n
            if (year_convention is None or isinstance(year_convention, str))
            else list(year_convention)
        )
        seasons = (
            [growth_season] * n
            if (growth_season is None or isinstance(growth_season, (str, SeasonalWeights)))
            else list(growth_season)
        )
        if not (len(hems) == len(cals) == len(convs) == len(seasons) == n):
            raise ValueError("Metadata sequence lengths must match paths length.")

        loaded = [
            TreeRingSeries.from_csv(
                p,
                hemisphere=h,
                calendar=c,
                year_convention=yc,
                growth_season=gs,
                sort_by_time=sort_by_time,
            )
            for p, h, c, yc, gs in zip(paths, hems, cals, convs, seasons)
        ]
        return cls(loaded, name=name)


# ---------------------------------------------------------------------------
# Audited Catalogue of All Bundled Repository CSV Datasets
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class BundledDatasetEntry:
    """Audited calendar, hemisphere, and ring-dating metadata for a bundled CSV."""

    key: str
    location: Literal["package_data", "repo_notebooks"]
    relative_path: str
    event: str
    hemisphere: HemisphereType
    calendar: CalendarType
    year_convention: YearConvention
    growth_season: str
    notes: str = ""


def _build_bundled_catalogue() -> dict[str, BundledDatasetEntry]:
    entries: list[BundledDatasetEntry] = []

    def _add(
        rel: str,
        *,
        loc: Literal["package_data", "repo_notebooks"] = "package_data",
        event: str,
        hem: HemisphereType,
        cal: CalendarType,
        conv: YearConvention,
        season: str,
        notes: str = "",
    ) -> None:
        key = rel if loc == "package_data" else f"notebooks/{rel}"
        entries.append(
            BundledDatasetEntry(
                key=key,
                location=loc,
                relative_path=rel,
                event=event,
                hemisphere=hem,
                calendar=cal,
                year_convention=conv,
                growth_season=season,
                notes=notes,
            )
        )

    # 1. 775 CE Northern Hemisphere (early-N & late-N COSMIC + Scifo19)
    for fname in (
        "Büntgen18_ALT01.csv",
        "Büntgen18_AUT02.csv",
        "Büntgen18_CAN06.csv",
        "Büntgen18_GRE02.csv",
        "Büntgen18_MON09.csv",
        "Büntgen18_PAK04.csv",
        "Büntgen18_RUS15.csv",
        "Büntgen18_RUS17.csv",
        "Büntgen18_RUS20.csv",
        "Büntgen18_SWE02.csv",
        "Büntgen18_TIB01.csv",
    ):
        _add(
            f"775AD-early-N/{fname}",
            event="775CE",
            hem="north",
            cal="astronomical_ce",
            conv="schulman_start",
            season="april-september",
            notes="Büntgen et al. (2018) NH 775 CE series (early rise).",
        )

    for fname in (
        "Büntgen18_CHN01.csv",
        "Büntgen18_GER01.csv",
        "Büntgen18_GER07.csv",
        "Büntgen18_JAP01.csv",
        "Büntgen18_MON03.csv",
        "Büntgen18_RUS04.csv",
        "Büntgen18_SWE05.csv",
        "Büntgen18_USA02.csv",
        "Büntgen18_USA11.csv",
        "Büntgen18_USA18.csv",
        "Scifo19_Oak.csv",
    ):
        _add(
            f"775AD-late-N/{fname}",
            event="775CE",
            hem="north",
            cal="astronomical_ce",
            conv="schulman_start",
            season="april-september",
            notes="Büntgen et al. (2018) / Scifo et al. (2019) NH 775 CE series.",
        )

    # 2. 775 CE Southern Hemisphere (Büntgen18 COSMIC: Schulman start-year = 774!)
    for fname in (
        "Büntgen18_DAR01.csv",
        "Büntgen18_DAR06.csv",
        "Büntgen18_DAR07.csv",
        "Büntgen18_PAT02.csv",
        "Büntgen18_PAT03.csv",
        "Büntgen18_TAS01.csv",
    ):
        _add(
            f"775AD-early-S/{fname}",
            event="775CE",
            hem="south",
            cal="astronomical_ce",
            conv="schulman_start",
            season="october-march",
            notes=(
                "Schulman start-year convention: ring 774 spans Oct 774 - Mar 775 CE "
                "(774.75..775.25), where the 774 CE Miyake jump first appears."
            ),
        )

    # 3. 993 CE Northern & Southern Hemisphere (Büntgen18 COSMIC)
    for fname in (
        "Büntgen18_ALT02.csv",
        "Büntgen18_CHI01.csv",
        "Büntgen18_MON05.csv",
        "Büntgen18_SWE01.csv",
        "Büntgen18_SWE02.csv",
        "Büntgen18_SWE03.csv",
        "Büntgen18_SWE04.csv",
    ):
        _add(
            f"993AD-N/{fname}",
            event="993CE",
            hem="north",
            cal="astronomical_ce",
            conv="schulman_start",
            season="april-september",
            notes="Büntgen et al. (2018) NH 993 CE series.",
        )

    for fname in ("Büntgen18_DAR01.csv", "Büntgen18_PAT03.csv"):
        _add(
            f"993AD-S/{fname}",
            event="993CE",
            hem="south",
            cal="astronomical_ce",
            conv="schulman_start",
            season="october-march",
            notes=(
                "Schulman start-year convention: ring 993 spans Oct 993 - Mar 994 CE "
                "(993.75..994.25), matching the 993 CE spring spike."
            ),
        )

    # 4. ~660 BCE Sub-Annual Earlywood & Latewood (Sakurai et al. 2020: historical BCE!)
    _add(
        "660BCE-Ew/Sakurai20_CedarEw.csv",
        event="660BCE",
        hem="north",
        cal="historical_bce",
        conv="schulman_start",
        season="april-june",
        notes="Sakurai et al. (2020) Yakushima cedar earlywood; CSV encodes 664 BCE as -664.",
    )
    _add(
        "660BCE-Lw/Sakurai20_CedarLw.csv",
        event="660BCE",
        hem="north",
        cal="historical_bce",
        conv="schulman_start",
        season="july-september",
        notes=(
            "Sakurai et al. (2020) Yakushima cedar latewood; CSV encodes 664 BCE as -664 "
            "(first jump in 664 BCE latewood = astronomical -663 + [0.50, 0.75])."
        ),
    )

    # 5. 5259 BCE, 5410 BCE, 7176 BCE (Brehm21 & Miyake21: historical BCE!)
    for fname in (
        "Brehm21_AlpineLarch.csv",
        "Brehm21_IrishOak.csv",
        "Brehm21_SibirianLarch.csv",
        "Brehm21_USABristlecone.csv",
    ):
        _add(
            f"5259BCE/{fname}",
            event="5259BCE",
            hem="north",
            cal="historical_bce",
            conv="schulman_start",
            season="april-september",
            notes="Brehm et al. (2021) 5259 BCE event; CSV encodes 5259 BCE as -5259 (astronomical -5258).",
        )

    for fname in (
        "Miyake21_Bristlecone.csv",
        "Miyake21_Finland1.csv",
        "Miyake21_Switzerland.csv",
    ):
        _add(
            f"5410BCE/{fname}",
            event="5410BCE",
            hem="north",
            cal="historical_bce",
            conv="schulman_start",
            season="april-september",
            notes="Miyake et al. (2021) 5410 BCE event; CSV encodes 5410 BCE as -5410 (astronomical -5409).",
        )

    for fname in (
        "Brehm21_AlpineLarch.csv",
        "Brehm21_GermanOak.csv",
        "Brehm21_USABristlecone.csv",
    ):
        _add(
            f"7176BCE/{fname}",
            event="7176BCE",
            hem="north",
            cal="historical_bce",
            conv="schulman_start",
            season="april-september",
            notes="Brehm et al. (2021) 7176 BCE event; CSV encodes 7176 BCE as -7176 (astronomical -7175).",
        )

    # 6. IntCal20 raw tree-ring compilation
    _add(
        "IntCal20/Intcal20.csv",
        event="IntCal20",
        hem="north",
        cal="astronomical_ce",
        conv="schulman_start",
        season="january-december",
        notes="IntCal20 compilation in astronomical CE (1950 - cal BP); unsorted blocks sorted on load.",
    )

    # 7. Repository `notebooks/775CE/` CSVs (notably Guttler15_Kauri end_year!)
    _add(
        "775CE/Guttler15_Kauri.csv",
        loc="repo_notebooks",
        event="775CE",
        hem="south",
        cal="astronomical_ce",
        conv="end_year",
        season="october-march",
        notes=(
            "Growth-completion year convention: ring 775 spans Oct 774 - Mar 775 CE "
            "(775 + [-0.25, 0.25] = 774.75..775.25), matching Büntgen18_TAS01 ring 774!"
        ),
    )
    _add(
        "775CE/Jull14_Bristlecone.csv",
        loc="repo_notebooks",
        event="775CE",
        hem="north",
        cal="astronomical_ce",
        conv="schulman_start",
        season="june-august",
        notes="Jull et al. (2014) bristlecone pine with high-elevation June-August season.",
    )
    for fname in (
        "Büntgen18_CHN01.csv",
        "Büntgen18_GER01.csv",
        "Büntgen18_GER07.csv",
        "Büntgen18_JAP01.csv",
        "Büntgen18_MON03.csv",
        "Büntgen18_SWE05.csv",
        "Büntgen18_USA02.csv",
        "Büntgen18_USA11.csv",
        "Büntgen18_USA18.csv",
        "Miyake12_Cedar.csv",
        "Rakowski15_Oak.csv",
        "Scifo19_Oak.csv",
    ):
        _add(
            f"775CE/{fname}",
            loc="repo_notebooks",
            event="775CE",
            hem="north",
            cal="astronomical_ce",
            conv="schulman_start",
            season="april-september",
            notes="NH 775 CE series in notebooks/775CE/.",
        )

    return {entry.key: entry for entry in entries}


BUNDLED_DATASET_CATALOGUE: dict[str, BundledDatasetEntry] = _build_bundled_catalogue()


def load_bundled_series(
    key: str,
    *,
    repo_root: str | Path | None = None,
    sort_by_time: bool = True,
) -> TreeRingSeries:
    """Load an audited :class:`TreeRingSeries` by key from :data:`BUNDLED_DATASET_CATALOGUE`."""
    if key not in BUNDLED_DATASET_CATALOGUE:
        raise KeyError(
            f"Unknown bundled dataset key {key!r}. Available keys include e.g. "
            "'775AD-early-S/Büntgen18_TAS01.csv' and 'notebooks/775CE/Guttler15_Kauri.csv'."
        )
    entry = BUNDLED_DATASET_CATALOGUE[key]
    if entry.location == "package_data":
        res = resources.files("ticktack").joinpath(
            "data", "datasets", entry.relative_path
        )
        with resources.as_file(res) as resolved_path:
            return TreeRingSeries.from_csv(
                resolved_path,
                hemisphere=entry.hemisphere,
                calendar=entry.calendar,
                year_convention=entry.year_convention,
                growth_season=entry.growth_season,
                name=Path(entry.relative_path).stem,
                sort_by_time=sort_by_time,
            )

    # repo_notebooks location
    if repo_root is not None:
        nb_base = Path(repo_root) / "notebooks"
    else:
        nb_base = Path(__file__).resolve().parents[2] / "notebooks"
    full_path = nb_base / entry.relative_path
    if not full_path.exists():
        raise FileNotFoundError(
            f"Notebook dataset {full_path} not found on disk (only available in full git checkout)."
        )
    return TreeRingSeries.from_csv(
        full_path,
        hemisphere=entry.hemisphere,
        calendar=entry.calendar,
        year_convention=entry.year_convention,
        growth_season=entry.growth_season,
        name=Path(entry.relative_path).stem,
        sort_by_time=sort_by_time,
    )


def load_bundled_event(
    event: str,
    *,
    include_repo_notebooks: bool = False,
    hemisphere: HemisphereType | None = None,
) -> Dataset:
    """Load all audited bundled series for a given cosmic-ray event into a :class:`Dataset`."""
    matched = [
        entry
        for entry in BUNDLED_DATASET_CATALOGUE.values()
        if entry.event == event
        and (include_repo_notebooks or entry.location == "package_data")
        and (hemisphere is None or entry.hemisphere == hemisphere)
    ]
    if not matched:
        raise ValueError(f"No bundled datasets matched event={event!r}, hemisphere={hemisphere!r}.")
    series_list = [load_bundled_series(entry.key) for entry in matched]
    return Dataset(series_list, name=event)
