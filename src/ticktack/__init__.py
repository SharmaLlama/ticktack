"""ticktack 2.0 — Bayesian radiocarbon inference with ``zodiax`` CBMs.

This package provides a fully differentiable, JAX-native framework for
inferring cosmic-ray production histories from tree-ring radiocarbon data.

.. rubric:: Quick start

.. code-block:: python

    import ticktack as tt

    cbm        = tt.load_preset("Brehm21")
    production = tt.ProductionSum(
        steady = tt.SteadyProduction(),
        spike  = tt.MiyakeSpike(t_start=774.3, duration=1.0, fluence=120.0),
        solar  = tt.HarmonicSolar.from_polar(amplitudes=0.0, phases=0.0),
    )
    model = tt.RadiocarbonModel(cbm, production)

"""

# ---------------------------------------------------------------------------
# Enable float64 before any JAX computations (must come first).
# ---------------------------------------------------------------------------
import warnings as _warnings

import jax as _jax

_jax.config.update("jax_enable_x64", True)

import jax.numpy as _jnp  # noqa: E402

if _jnp.zeros(1).dtype != _jnp.float64:
    _warnings.warn(
        "jax_enable_x64 failed; ticktack 2.0 requires float64 throughout. "
        "Make sure no code imported JAX before ticktack.",
        RuntimeWarning,
        stacklevel=2,
    )

# ---------------------------------------------------------------------------
# Package version
# ---------------------------------------------------------------------------

__version__ = "2.0.0a0"

# ---------------------------------------------------------------------------
# Public API — submodule exports
# ---------------------------------------------------------------------------

# Constants
from .constants import (  # noqa: E402
    ALPHA_ATOMS_CM2_S_TO_KG_YR,
    DECAY_CAMBRIDGE,
    DEFAULT_Q_STD,
    SECONDS_PER_JULIAN_YEAR,
)

# Carbon Box Models
from .cbm import (  # noqa: E402
    BalancedCBM,
    CarbonBoxModel,
    EffectiveCBM,
    FixedCBM,
    OpenCBM,
    ReversibleCBM,
    Topology,
)

# Presets
from .presets import load_preset  # noqa: E402

# Production terms
from .production import (  # noqa: E402
    GPProduction,
    GriddedProduction,
    HarmonicSolar,
    LinearTrend,
    LogGPProduction,
    MiyakeSpike,
    ProductionSum,
    ProductionTerm,
    SteadyProduction,
    kernel_spectral_density,
    solin_sarkka_eigenpairs,
)

# Solvers
from .solvers import (  # noqa: E402
    AbstractSolver,
    DiffraxSolver,
    ExactLinearSolver,
    ModaxSolver,
    wrap_jvp_only_scalar_fn,
)

# Data
from .data import (  # noqa: E402
    BUNDLED_DATASET_CATALOGUE,
    Dataset,
    TreeRingSeries,
    load_bundled_event,
    load_bundled_series,
)

# Linear algebra / marginalisation helpers
from .linear import (  # noqa: E402
    ArrowProblem,
    CanonicalPODResult,
    CausalGPProjection,
    LinearMarginal,
    SolinSarkkaBasis,
    arrow_at,
    arrow_problem,
    canonical_pod_modes,
    harmonic_solar_design,
    hierarchical_offset_design,
    kalman_gp_marginal,
    posterior,
    project_causal_gp,
    solin_sarkka_basis,
    spectral_density,
    stationary_cbm_gp_cov,
    whiten_blocks,
    whiten_cholesky,
    whiten_qr,
    whiten_rank_one,
)

# Fisher / information-geometry
from .fisher import (  # noqa: E402
    FisherSpectrumResult,
    RelaxationModesResult,
    cbm_fisher_spectrum,
    expected_fisher,
    fisher_projection,
    laplace_covariance,
    regularized_inverse,
    relaxation_fisher_modes,
)

# Top-level model
from .model import RadiocarbonModel  # noqa: E402

# Likelihoods
from .likelihood import (  # noqa: E402
    LinearMarginalConfig,
    LinearPosteriorResult,
    ParamSpec,
    ParameterSpace,
    joint_log_likelihood,
    joint_log_posterior,
    linear_posterior,
    marginal_log_likelihood,
    marginal_log_posterior,
)

# MAP fitting
from .fitting import MAPResult, fit_map  # noqa: E402

# Samplers
from .sampling import (  # noqa: E402
    SMCResult,
    SVIResult,
    fit_svi,
    sample_nuts,
    sample_smc,
)

# Plotting
from .plotting import (  # noqa: E402
    plot_cbm_eigenmodes,
    plot_d14c_fit,
    plot_fisher_spectrum,
    plot_production_posterior,
)

__all__ = [
    # version
    "__version__",
    # constants
    "ALPHA_ATOMS_CM2_S_TO_KG_YR",
    "DECAY_CAMBRIDGE",
    "DEFAULT_Q_STD",
    "SECONDS_PER_JULIAN_YEAR",
    # cbm
    "BalancedCBM",
    "CarbonBoxModel",
    "EffectiveCBM",
    "FixedCBM",
    "OpenCBM",
    "ReversibleCBM",
    "Topology",
    # presets
    "load_preset",
    # production
    "GPProduction",
    "GriddedProduction",
    "HarmonicSolar",
    "LinearTrend",
    "LogGPProduction",
    "MiyakeSpike",
    "ProductionSum",
    "ProductionTerm",
    "SteadyProduction",
    "kernel_spectral_density",
    "solin_sarkka_eigenpairs",
    # solvers
    "AbstractSolver",
    "DiffraxSolver",
    "ExactLinearSolver",
    "ModaxSolver",
    "wrap_jvp_only_scalar_fn",
    # data
    "BUNDLED_DATASET_CATALOGUE",
    "Dataset",
    "TreeRingSeries",
    "load_bundled_event",
    "load_bundled_series",
    # linear
    "ArrowProblem",
    "CanonicalPODResult",
    "CausalGPProjection",
    "LinearMarginal",
    "SolinSarkkaBasis",
    "arrow_at",
    "arrow_problem",
    "canonical_pod_modes",
    "harmonic_solar_design",
    "hierarchical_offset_design",
    "kalman_gp_marginal",
    "posterior",
    "project_causal_gp",
    "solin_sarkka_basis",
    "spectral_density",
    "stationary_cbm_gp_cov",
    "whiten_blocks",
    "whiten_cholesky",
    "whiten_qr",
    "whiten_rank_one",
    # fisher
    "FisherSpectrumResult",
    "RelaxationModesResult",
    "cbm_fisher_spectrum",
    "expected_fisher",
    "fisher_projection",
    "laplace_covariance",
    "regularized_inverse",
    "relaxation_fisher_modes",
    # model
    "RadiocarbonModel",
    # likelihood
    "LinearMarginalConfig",
    "LinearPosteriorResult",
    "ParamSpec",
    "ParameterSpace",
    "joint_log_likelihood",
    "joint_log_posterior",
    "linear_posterior",
    "marginal_log_likelihood",
    "marginal_log_posterior",
    # fitting
    "MAPResult",
    "fit_map",
    # sampling
    "SMCResult",
    "SVIResult",
    "fit_svi",
    "sample_nuts",
    "sample_smc",
    # plotting
    "plot_cbm_eigenmodes",
    "plot_d14c_fit",
    "plot_fisher_spectrum",
    "plot_production_posterior",
]
