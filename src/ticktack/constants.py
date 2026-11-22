"""Physical, isotopic, and radiometric constants for ``ticktack 2.0``.

All decay rates are expressed in inverse Julian years (``yr^-1``) and production
unit conversions map global mean surface cosmic-ray ``14C`` production rates in
``atoms cm^-2 s^-1`` to global ``14C`` mass production rates in ``kg yr^-1``.
"""

from __future__ import annotations

import math

__all__ = [
    "HALF_LIFE_CAMBRIDGE",
    "HALF_LIFE_KUTSCHERA",
    "MEAN_LIFE_STUIVER_POLACH",
    "DECAY_CAMBRIDGE",
    "DECAY_STUIVER_POLACH",
    "DECAY_KUTSCHERA",
    "DEFAULT_DECAY_RATE",
    "AVOGADRO_CODATA",
    "ATOMIC_MASS_C14",
    "ATOMIC_MASS_C12",
    "EARTH_MEAN_RADIUS_CM",
    "EARTH_SURFACE_AREA_CM2",
    "SECONDS_PER_JULIAN_YEAR",
    "ALPHA_ATOMS_CM2_S_TO_KG_YR",
    "ALPHA_LEGACY_V1",
    "DEFAULT_Q_STD",
    "GUTTLER15_TROP_MASS_GTC",
    "GUTTLER15_TROP_C14_KG",
    "GUTTLER15_R_REF_KG_PER_GTC",
]

# Radiocarbon half-lives and decay constants (yr^-1)
HALF_LIFE_CAMBRIDGE: float = 5730.0
HALF_LIFE_KUTSCHERA: float = 5700.0
MEAN_LIFE_STUIVER_POLACH: float = 8267.0

DECAY_CAMBRIDGE: float = math.log(2.0) / HALF_LIFE_CAMBRIDGE
DECAY_STUIVER_POLACH: float = 1.0 / MEAN_LIFE_STUIVER_POLACH
DECAY_KUTSCHERA: float = math.log(2.0) / HALF_LIFE_KUTSCHERA
DEFAULT_DECAY_RATE: float = DECAY_CAMBRIDGE

# CODATA / IUGG physical constants for 14C production conversion
AVOGADRO_CODATA: float = 6.02214076e23  # mol^-1 (exact SI)
ATOMIC_MASS_C14: float = 14.0032419884  # g mol^-1 (AME2020)
ATOMIC_MASS_C12: float = 12.0  # g mol^-1 (exact carbon-12 scale)
EARTH_MEAN_RADIUS_CM: float = 6.3710088e8  # cm (IUGG volumetric/mean radius)
EARTH_SURFACE_AREA_CM2: float = 4.0 * math.pi * EARTH_MEAN_RADIUS_CM**2  # cm^2
SECONDS_PER_JULIAN_YEAR: float = 365.25 * 86400.0  # s yr^-1

# Conversion factor: 1 atom cm^-2 s^-1 (global mean) -> kg(14C) yr^-1
ALPHA_ATOMS_CM2_S_TO_KG_YR: float = (
    (ATOMIC_MASS_C14 * 1e-3 / AVOGADRO_CODATA)
    * EARTH_SURFACE_AREA_CM2
    * SECONDS_PER_JULIAN_YEAR
)

# Legacy ticktack 1.x conversion factor (365-day year, 5.11e18 cm^2, 6.022e23)
ALPHA_LEGACY_V1: float = 14.003242 / 6.022 * 5.11 * 31536.0 / 1e5

# Standard pre-industrial 1950 CE reference production rate (atoms cm^-2 s^-1)
DEFAULT_Q_STD: float = 1.76

# Pre-industrial tropospheric reference values in Guttler et al. (2015) at Q_std = 1.76
# (evaluated with DECAY_CAMBRIDGE and ALPHA_ATOMS_CM2_S_TO_KG_YR)
GUTTLER15_TROP_MASS_GTC: float = 501.5
GUTTLER15_TROP_C14_KG: float = 663.5388282832537
GUTTLER15_R_REF_KG_PER_GTC: float = GUTTLER15_TROP_C14_KG / GUTTLER15_TROP_MASS_GTC
