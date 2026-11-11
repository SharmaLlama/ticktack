"""Information Field Theory (IFT) production fields, MacKay evidence, and GP kernel selection.

Adapts the reflecting-boundary cosine (DCT-II) field architecture and empirical-Bayes
diagnostics of ``virgil`` (:mod:`virgil.fields`, :mod:`virgil.imaging`, and
``docs/gp_and_ift.md``; Enßlin et al. 2009; Arras et al. 2022; Edenhofer et al. 2024;
MacKay 1992) to 1D cosmic-ray ``14C`` production histories in ``ticktack 2.0``:

* **1D DCT-II spectra & composable spectral kernels**:
  :func:`dct_laplacian_eigenvalues`, :func:`field_spectrum`, :class:`SpectralKernel`,
  :class:`MaternSpectrum`, :class:`SquaredExpSpectrum`, :class:`QuasiPeriodicSpectrum`,
  :class:`CorrelatedFieldSpectrum`, and :class:`SumSpectrum`.
* **Whitened signal-field production terms**:
  :class:`GaussianField` (linear standardized DCT-II excitations ``z ~ N(0, I)``) and
  :class:`LogGaussianField` (strictly positive log-normal field ``Q(t) > 0``), backed
  by the precomputed first-order-hold operator :func:`build_dct_response_matrix`.
* **Wiener / MAP fitting, Laplace evidence, posterior draws, & noise diagnostics**:
  :func:`fit_ift_field`, :func:`log_evidence`, :func:`laplace_samples`, and MacKay's
  fixed-point :func:`error_scale` (global ``float`` and per-series ``dict[str, float]``).
* **Automatic empirical-Bayes GP kernel optimization & model selection**:
  :func:`evidence_grid`, :func:`optimize_gp_kernel`, and :func:`select_gp_kernel`
  across seven candidate spectral families with Laplace hyper-Occam penalties.
* **Stateless scientific diagnostic plotting**:
  :func:`plot_field_draws` and :func:`plot_evidence_grid`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any, Literal
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.fft import dct as _jax_dct
from jax.scipy.fft import idct as _jax_idct
import jax.scipy.linalg as jsl
from jax.typing import ArrayLike
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import optimistix as optx
from scipy.linalg import solve_triangular as _scipy_solve_triangular
from scipy.optimize import minimize as _scipy_minimize
import zodiax as zdx

from .cbm import CarbonBoxModel
from .constants import ALPHA_ATOMS_CM2_S_TO_KG_YR, DEFAULT_Q_STD
from .data import Dataset, TreeRingSeries
from .linear import (
    hierarchical_offset_design,
    posterior as _linear_posterior_std,
    stationary_cbm_gp_cov,
    whiten_qr,
)
from .model import RadiocarbonModel
from .production import (
    GriddedProduction,
    MiyakeSpike,
    ProductionSum,
    ProductionTerm,
    SteadyProduction,
)
from .solvers import ExactLinearSolver, _foh_cumulative_integral

__all__ = [
    "dct2",
    "idct2",
    "dct_laplacian_eigenvalues",
    "dct_knot_variance",
    "field_spectrum",
    "SpectralKernel",
    "MaternSpectrum",
    "SquaredExpSpectrum",
    "QuasiPeriodicSpectrum",
    "SumSpectrum",
    "CorrelatedFieldSpectrum",
    "GaussianField",
    "LogGaussianField",
    "DCTResponseOperator",
    "build_dct_response_operator",
    "build_dct_response_matrix",
    "predict_field_d14c",
    "IFTFieldFitResult",
    "fit_ift_field",
    "fit_field",
    "log_evidence",
    "laplace_samples",
    "error_scale",
    "apply_error_scale",
    "EvidenceGridResult",
    "evidence_grid",
    "OptimizedKernelResult",
    "optimize_gp_kernel",
    "KernelSelectionResult",
    "select_gp_kernel",
    "plot_field_draws",
    "plot_evidence_grid",
]


def _concrete_np(value: Any) -> np.ndarray | None:
    """Return ``value`` as a host NumPy array when concrete, else ``None``."""
    try:
        return np.asarray(value, dtype=np.float64)
    except (
        jax.errors.TracerArrayConversionError,
        jax.errors.ConcretizationTypeError,
    ):
        return None


def _check_positive_concrete(name: str, value: Any, *, allow_zero: bool = False) -> None:
    """Validate concrete numeric inputs while staying transparent to JAX tracers."""
    arr = _concrete_np(value)
    if arr is None:
        return
    ok = np.all(np.isfinite(arr)) and (np.all(arr >= 0.0) if allow_zero else np.all(arr > 0.0))
    if not ok:
        rel = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {rel}; got {value!r}.")


def _safe_sqrt(x: jax.Array) -> jax.Array:
    """Differentiable square root with double-``where`` guard at ``x <= 0``."""
    pos = x > 0.0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, x, 1.0)), 0.0)


def _as_dataset(dataset_or_series: Dataset | TreeRingSeries) -> Dataset:
    """Wrap a single :class:`TreeRingSeries` into a :class:`Dataset` if needed."""
    if isinstance(dataset_or_series, Dataset):
        return dataset_or_series
    if isinstance(dataset_or_series, TreeRingSeries):
        return Dataset((dataset_or_series,), name=dataset_or_series.name)
    raise TypeError(
        f"Expected Dataset or TreeRingSeries, got {type(dataset_or_series)!r}."
    )


# =============================================================================
# 1. Orthonormal 1D DCT-II Transforms & Reflecting-Boundary Spectral Kernels
# =============================================================================


def dct2(x: ArrayLike, *, axis: int = -1) -> jax.Array:
    """Orthonormal 1D Discrete Cosine Transform (DCT-II) in ``float64``."""
    return _jax_dct(jnp.asarray(x, dtype=jnp.float64), type=2, axis=axis, norm="ortho")


def idct2(x: ArrayLike, *, axis: int = -1) -> jax.Array:
    """Orthonormal 1D Inverse Discrete Cosine Transform (IDCT-II, transpose of :func:`dct2`)."""
    return _jax_idct(jnp.asarray(x, dtype=jnp.float64), type=2, axis=axis, norm="ortho")


def dct_laplacian_eigenvalues(n_grid: int, dt: ArrayLike) -> jax.Array:
    r"""Exact eigenvalues of the 1D Neumann discrete Laplacian under :func:`idct2`.

    For the reflecting-boundary discrete second-difference operator ``L`` satisfying
    ``x^T L x = dt^{-2} sum_{g=0}^{N-2} (x_{g+1} - x_g)^2``, the orthonormal DCT-II
    modes ``j = 0, ..., N - 1`` have exact eigenvalues

    .. math::

        \lambda_j = \left(\frac{2}{\Delta t}\right)^2 \sin^2\!\left(\frac{\pi j}{2 N}\right).
    """
    n = int(n_grid)
    if n < 1:
        raise ValueError(f"n_grid must be >= 1, got {n_grid}.")
    _check_positive_concrete("dt", dt)
    dt_arr = jnp.asarray(dt, dtype=jnp.float64)
    j = jnp.arange(n, dtype=jnp.float64)
    return jnp.square((2.0 / dt_arr) * jnp.sin(math.pi * j / (2.0 * n)))


def dct_knot_variance(spectrum: ArrayLike) -> jax.Array:
    r"""Exact per-knot marginal variance ``diag(C^T diag(S) C)`` in ``O(N log N)``.

    Near reflecting boundaries a knot correlates with its mirror image so its marginal
    variance rises to ``~2 sigma^2`` while the grid average remains ``(1/N) sum_j S_j``.
    Uses the trigonometric identity ``2 cos^2 theta = 1 + cos(2 theta)`` to fold mode
    variances ``S_j`` into a single orthonormal :func:`idct2` evaluation without
    allocating an ``N x N`` matrix.
    """
    s = jnp.asarray(spectrum, dtype=jnp.float64).reshape(-1)
    n = s.shape[0]
    if n == 0:
        return jnp.zeros((0,), dtype=jnp.float64)
    if n == 1:
        return s.copy()

    coeffs = jnp.zeros((n,), dtype=jnp.float64)
    coeffs = coeffs.at[0].set(jnp.sum(s) / math.sqrt(float(n)))

    # Positive frequencies 1 <= j <= N - 1 map 2*j either to k = 2*j < N (+)
    # or to k = 2*N - 2*j < N (-), while 2*j == N vanishes identically.
    j_low = jnp.arange(1, (n + 1) // 2, dtype=jnp.int32)
    if j_low.size > 0:
        coeffs = coeffs.at[2 * j_low].add(s[j_low] / math.sqrt(2.0 * float(n)))

    j_high = jnp.arange(n // 2 + 1, n, dtype=jnp.int32)
    if j_high.size > 0:
        coeffs = coeffs.at[2 * n - 2 * j_high].add(-s[j_high] / math.sqrt(2.0 * float(n)))

    return idct2(coeffs)


def _normalize_from_log_spectrum(
    log_raw: jax.Array,
    sigma: ArrayLike,
    *,
    zero_mode: bool,
) -> jax.Array:
    """Overflow-free normalization of log-power ``log_raw`` to grid-averaged variance ``sigma^2``."""
    n = log_raw.shape[0]
    sig = jnp.asarray(sigma, dtype=jnp.float64).reshape(())
    if n == 0:
        return jnp.zeros((0,), dtype=jnp.float64)
    if n == 1:
        return jnp.where(zero_mode, jnp.square(sig)[None], jnp.zeros((1,), dtype=jnp.float64))

    if zero_mode:
        shift = jnp.max(log_raw)
        spec = jnp.exp(log_raw - shift)
    else:
        shift = jnp.max(log_raw[1:])
        spec = jnp.concatenate(
            [jnp.zeros((1,), dtype=jnp.float64), jnp.exp(log_raw[1:] - shift)],
            axis=0,
        )
    total = jnp.sum(spec)
    pos_total = total > 0.0
    safe_total = jnp.where(pos_total, total, 1.0)
    scale = jnp.where(pos_total, jnp.square(sig) * float(n) / safe_total, 0.0)
    return spec * scale


def field_spectrum(
    n_grid: int,
    dt: ArrayLike,
    sigma: ArrayLike,
    length_yr: ArrayLike,
    order: ArrayLike = 2.0,
    *,
    zero_mode: bool = False,
) -> jax.Array:
    r"""Prior variances ``S_j`` of a 1D DCT-II signal field's cosine modes.

    Direct 1D counterpart of ``virgil.fields.field_spectrum``:

    .. math::

        S_j \propto \left(\ell^{-2} + \lambda_j\right)^{-\mathrm{order}}
        \propto \left(1 + \ell^2 \lambda_j\right)^{-\mathrm{order}},

    where ``lambda_j`` are the reflecting-boundary Laplacian eigenvalues from
    :func:`dct_laplacian_eigenvalues`. Supports both integer ``order``
    (``1`` = Ornstein–Uhlenbeck / exact GMRF total-squared-variation + ``L_2`` penalty,
    ``2`` = Matérn-3/2, ``3`` = Matérn-5/2) and continuous real ``order > 0.5``
    (``nu = order - 1/2`` in 1D, ``C^inf``-differentiable with respect to ``order``),
    as well as ``order = inf`` (Squared-Exponential limit).

    When ``zero_mode=False`` (default), ``S_0 = 0`` so a global constant production
    shift is separated cleanly from ``SteadyProduction`` / baseline offsets. The
    spectrum is normalized so the grid-averaged prior variance satisfies
    ``(1 / N) sum_{j=0}^{N-1} S_j = sigma^2``.
    """
    _check_positive_concrete("sigma", sigma, allow_zero=True)
    _check_positive_concrete("length_yr", length_yr)
    ord_np = _concrete_np(order)
    if ord_np is not None and not (np.isposinf(ord_np) or float(ord_np) > 0.5):
        raise ValueError(f"order must be > 0.5 (or inf), got {order!r}.")

    lam = dct_laplacian_eigenvalues(n_grid, dt)
    ell = jnp.asarray(length_yr, dtype=jnp.float64).reshape(())
    if ord_np is not None and bool(np.isposinf(ord_np)):
        log_raw = -0.5 * jnp.square(ell) * lam
    else:
        ord_arr = jnp.asarray(order, dtype=jnp.float64).reshape(())
        log_raw = -ord_arr * jnp.log1p(jnp.square(ell) * lam)
    return _normalize_from_log_spectrum(log_raw, sigma, zero_mode=zero_mode)


class SpectralKernel(zdx.Base):
    """Abstract composable 1D DCT-II power-spectrum kernel (:class:`zodiax.Base`)."""

    def spectrum(self, n_grid: int, dt: ArrayLike) -> jax.Array:
        """Evaluate DCT-II mode variances ``S_j`` of shape ``(n_grid,)``."""
        raise NotImplementedError

    def hyperprior_log_prob(self) -> jax.Array:
        """Optional internal log-hyperprior contribution (e.g. non-parametric deviations)."""
        return jnp.asarray(0.0, dtype=jnp.float64)

    def __add__(self, other: SpectralKernel) -> SumSpectrum:
        if not isinstance(other, SpectralKernel):
            return NotImplemented
        left = self.parts if isinstance(self, SumSpectrum) else (self,)
        right = other.parts if isinstance(other, SumSpectrum) else (other,)
        return SumSpectrum(left + right)


class MaternSpectrum(SpectralKernel):
    """Matérn-like reflecting-boundary DCT-II spectrum supporting continuous ``order > 0.5``."""

    sigma: jax.Array
    length_yr: jax.Array
    order: jax.Array
    learn_order: bool = eqx.field(static=True)
    zero_mode: bool = eqx.field(static=True)

    def __init__(
        self,
        sigma: ArrayLike = 0.3,
        length_yr: ArrayLike = 5.0,
        order: ArrayLike = 2.0,
        *,
        learn_order: bool = False,
        zero_mode: bool = False,
    ) -> None:
        _check_positive_concrete("sigma", sigma, allow_zero=True)
        _check_positive_concrete("length_yr", length_yr)
        ord_np = _concrete_np(order)
        if ord_np is not None and not (float(ord_np) > 0.5):
            raise ValueError(f"MaternSpectrum order must be > 0.5, got {order!r}.")
        self.sigma = jnp.asarray(sigma, dtype=jnp.float64).reshape(())
        self.length_yr = jnp.asarray(length_yr, dtype=jnp.float64).reshape(())
        self.order = jnp.asarray(order, dtype=jnp.float64).reshape(())
        self.learn_order = bool(learn_order)
        self.zero_mode = bool(zero_mode)

    def spectrum(self, n_grid: int, dt: ArrayLike) -> jax.Array:
        return field_spectrum(
            n_grid,
            dt,
            self.sigma,
            self.length_yr,
            self.order,
            zero_mode=self.zero_mode,
        )


class SquaredExpSpectrum(SpectralKernel):
    r"""Smooth Squared-Exponential (RBF) DCT-II spectrum ``S_j \propto exp(-0.5 ell^2 lambda_j)``."""

    sigma: jax.Array
    length_yr: jax.Array
    zero_mode: bool = eqx.field(static=True)

    def __init__(
        self,
        sigma: ArrayLike = 0.3,
        length_yr: ArrayLike = 5.0,
        *,
        zero_mode: bool = False,
    ) -> None:
        _check_positive_concrete("sigma", sigma, allow_zero=True)
        _check_positive_concrete("length_yr", length_yr)
        self.sigma = jnp.asarray(sigma, dtype=jnp.float64).reshape(())
        self.length_yr = jnp.asarray(length_yr, dtype=jnp.float64).reshape(())
        self.zero_mode = bool(zero_mode)

    def spectrum(self, n_grid: int, dt: ArrayLike) -> jax.Array:
        lam = dct_laplacian_eigenvalues(n_grid, dt)
        log_raw = -0.5 * jnp.square(self.length_yr) * lam
        return _normalize_from_log_spectrum(log_raw, self.sigma, zero_mode=self.zero_mode)


class QuasiPeriodicSpectrum(SpectralKernel):
    r"""Damped harmonic oscillator (Lorentzian) DCT-II spectrum for quasi-periodic solar cycles.

    Models a wandering solar cycle with angular frequency ``omega_0 = 2 pi / period_yr``
    and quality factor ``Q = quality``:

    .. math::

        S_j \propto \left[(\lambda_j - \omega_0^2)^2 + \frac{\omega_0^2 \lambda_j}{Q^2}\right]^{-1}.
    """

    sigma: jax.Array
    period_yr: jax.Array
    quality: jax.Array
    zero_mode: bool = eqx.field(static=True)

    def __init__(
        self,
        sigma: ArrayLike = 0.2,
        period_yr: ArrayLike = 11.0,
        quality: ArrayLike = 4.0,
        *,
        zero_mode: bool = False,
    ) -> None:
        _check_positive_concrete("sigma", sigma, allow_zero=True)
        _check_positive_concrete("period_yr", period_yr)
        _check_positive_concrete("quality", quality)
        self.sigma = jnp.asarray(sigma, dtype=jnp.float64).reshape(())
        self.period_yr = jnp.asarray(period_yr, dtype=jnp.float64).reshape(())
        self.quality = jnp.asarray(quality, dtype=jnp.float64).reshape(())
        self.zero_mode = bool(zero_mode)

    def spectrum(self, n_grid: int, dt: ArrayLike) -> jax.Array:
        lam = dct_laplacian_eigenvalues(n_grid, dt)
        w0_sq = jnp.square(2.0 * math.pi / self.period_yr)
        denom = jnp.square(lam - w0_sq) + (w0_sq * lam) / jnp.square(self.quality)
        log_raw = -jnp.log(jnp.maximum(denom, 1e-300))
        return _normalize_from_log_spectrum(log_raw, self.sigma, zero_mode=self.zero_mode)


class SumSpectrum(SpectralKernel):
    """Additive composition ``S(omega) = sum_a S_a(omega)`` of :class:`SpectralKernel` terms."""

    parts: tuple[SpectralKernel, ...]

    def __init__(self, parts: Sequence[SpectralKernel]) -> None:
        items: list[SpectralKernel] = []
        for p in parts:
            if isinstance(p, SumSpectrum):
                items.extend(p.parts)
            elif isinstance(p, SpectralKernel):
                items.append(p)
            else:
                raise TypeError(f"Expected SpectralKernel, got {type(p)!r}.")
        if not items:
            raise ValueError("SumSpectrum requires at least one SpectralKernel.")
        self.parts = tuple(items)

    def spectrum(self, n_grid: int, dt: ArrayLike) -> jax.Array:
        acc = jnp.zeros((int(n_grid),), dtype=jnp.float64)
        for p in self.parts:
            acc = acc + p.spectrum(n_grid, dt)
        return acc

    def hyperprior_log_prob(self) -> jax.Array:
        acc = jnp.asarray(0.0, dtype=jnp.float64)
        for p in self.parts:
            acc = acc + p.hyperprior_log_prob()
        return acc


class CorrelatedFieldSpectrum(SpectralKernel):
    r"""NIFTy-inspired non-parametric log-amplitude power spectrum on ``ln omega``.

    Models the log-power spectrum across non-zero discrete wavenumbers
    ``omega_j = sqrt(lambda_j)`` (``j = 1, ..., N - 1``) as a power-law slope plus
    smooth integrated cosine basis deviations ``xi_m ~ N(0, 1)`` (Enßlin & Frommert 2011;
    Arras et al. 2022; Edenhofer et al. 2024):

    .. math::

        \ln P_j = \mathrm{slope}\,(\ln\omega_j - \overline{\ln\omega})
        + \mathrm{flexibility}\,\sum_{m=1}^{M_{\mathrm{spec}}} \frac{\xi_m}{m}\sqrt{2}\cos(\pi m u_j),

    where ``u_j in [0, 1]`` linearly spans ``[ln omega_1, ln omega_{N-1}]`` and the
    spectrum is normalized to grid-averaged variance ``sigma^2``.
    """

    sigma: jax.Array
    slope: jax.Array
    log_deviations: jax.Array
    flexibility: float = eqx.field(static=True)
    zero_mode: bool = eqx.field(static=True)

    def __init__(
        self,
        sigma: ArrayLike = 0.3,
        slope: ArrayLike = -2.0,
        log_deviations: ArrayLike | None = None,
        *,
        n_modes: int = 4,
        flexibility: float = 0.5,
        zero_mode: bool = False,
    ) -> None:
        _check_positive_concrete("sigma", sigma, allow_zero=True)
        if float(flexibility) < 0.0:
            raise ValueError(f"flexibility must be >= 0, got {flexibility}.")
        if log_deviations is None:
            devs = jnp.zeros((int(n_modes),), dtype=jnp.float64)
        else:
            devs = jnp.asarray(log_deviations, dtype=jnp.float64).reshape(-1)
            if devs.shape[0] < 1:
                raise ValueError("log_deviations must be non-empty.")
        self.sigma = jnp.asarray(sigma, dtype=jnp.float64).reshape(())
        self.slope = jnp.asarray(slope, dtype=jnp.float64).reshape(())
        self.log_deviations = devs
        self.flexibility = float(flexibility)
        self.zero_mode = bool(zero_mode)

    def spectrum(self, n_grid: int, dt: ArrayLike) -> jax.Array:
        n = int(n_grid)
        if n <= 1:
            return _normalize_from_log_spectrum(
                jnp.zeros((n,), dtype=jnp.float64), self.sigma, zero_mode=self.zero_mode
            )
        lam = dct_laplacian_eigenvalues(n, dt)
        log_w = 0.5 * jnp.log(jnp.maximum(lam[1:], 1e-300))
        centred = log_w - jnp.mean(log_w)
        span = jnp.maximum(log_w[-1] - log_w[0], 1e-12)
        u = (log_w - log_w[0]) / span  # (N - 1,) in [0, 1]

        m_idx = jnp.arange(1, self.log_deviations.shape[0] + 1, dtype=jnp.float64)
        basis = (math.sqrt(2.0) / m_idx[:, None]) * jnp.cos(
            math.pi * m_idx[:, None] * u[None, :]
        )
        log_pos = self.slope * centred + self.flexibility * (self.log_deviations @ basis)
        log_raw = jnp.concatenate([log_pos[:1], log_pos], axis=0)
        return _normalize_from_log_spectrum(log_raw, self.sigma, zero_mode=self.zero_mode)

    def hyperprior_log_prob(self) -> jax.Array:
        m = self.log_deviations.shape[0]
        return -0.5 * jnp.sum(jnp.square(self.log_deviations)) - 0.5 * m * math.log(
            2.0 * math.pi
        )


# =============================================================================
# 2. Standardized Signal Field ProductionTerms (GaussianField & LogGaussianField)
# =============================================================================


class GaussianField(ProductionTerm):
    r"""Linear IFT Gaussian signal-field production perturbation ``delta_q(t) = eta(t)``.

    Represents cosmic-ray ``14C`` production excursions on ``[t_start, t_end]`` via
    standardized cosine excitations ``latent`` ``z ~ N(0, I_N)`` mapped through the
    orthonormal inverse DCT-II:

    .. math::

        \eta = \operatorname{IDCT}_{\mathrm{II}}\!\left(\sqrt{S(\phi)} \odot z\right),

    optionally modulated by a temporal envelope ``template``.
    """

    latent: jax.Array
    sigma: jax.Array
    length_yr: jax.Array
    order: jax.Array
    kernel: SpectralKernel | None
    template: jax.Array | None
    t_start: float = eqx.field(static=True)
    t_end: float = eqx.field(static=True)
    zero_mode: bool = eqx.field(static=True)

    def __init__(
        self,
        latent: ArrayLike,
        t_start: float,
        t_end: float,
        sigma: ArrayLike = 0.3,
        length_yr: ArrayLike = 5.0,
        order: ArrayLike = 2.0,
        *,
        kernel: SpectralKernel | None = None,
        zero_mode: bool = False,
        template: ArrayLike | None = None,
    ) -> None:
        z = jnp.asarray(latent, dtype=jnp.float64)
        if z.ndim != 1 or z.shape[0] < 2:
            raise ValueError(
                f"GaussianField latent must be 1D with length >= 2, got shape {z.shape}."
            )
        t0, t1 = float(t_start), float(t_end)
        if not (math.isfinite(t0) and math.isfinite(t1) and t1 > t0):
            raise ValueError(f"Expected finite t_end > t_start, got [{t_start}, {t_end}].")
        _check_positive_concrete("sigma", sigma, allow_zero=True)
        _check_positive_concrete("length_yr", length_yr)
        if kernel is not None and not isinstance(kernel, SpectralKernel):
            raise TypeError(f"kernel must be a SpectralKernel or None, got {type(kernel)!r}.")

        if template is None:
            tmpl = None
        else:
            tmpl = jnp.asarray(template, dtype=jnp.float64)
            if tmpl.shape != z.shape:
                raise ValueError(
                    f"template shape {tmpl.shape} must match latent shape {z.shape}."
                )

        self.latent = z
        self.t_start = t0
        self.t_end = t1
        self.sigma = jnp.asarray(sigma, dtype=jnp.float64).reshape(())
        self.length_yr = jnp.asarray(length_yr, dtype=jnp.float64).reshape(())
        self.order = jnp.asarray(order, dtype=jnp.float64).reshape(())
        self.kernel = kernel
        self.zero_mode = bool(zero_mode)
        self.template = tmpl

    @property
    def is_linear_marginalizable(self) -> bool:
        return True

    def _default_slug_name(self) -> str:
        return "field"

    @property
    def n_grid(self) -> int:
        return int(self.latent.shape[0])

    @property
    def dt(self) -> float:
        return (self.t_end - self.t_start) / float(self.n_grid - 1)

    @property
    def t_grid(self) -> jax.Array:
        return jnp.linspace(self.t_start, self.t_end, self.n_grid, dtype=jnp.float64)

    @property
    def effective_kernel(self) -> SpectralKernel:
        if self.kernel is not None:
            return self.kernel
        return MaternSpectrum(
            sigma=self.sigma,
            length_yr=self.length_yr,
            order=self.order,
            zero_mode=self.zero_mode,
        )

    def power_spectrum(self) -> jax.Array:
        """Return DCT-II mode variances ``S_j`` of shape ``(n_grid,)``."""
        return self.effective_kernel.spectrum(self.n_grid, self.dt)

    def knot_variance(self) -> jax.Array:
        """Exact per-knot prior variance ``Var[eta(t_g)]`` of shape ``(n_grid,)``."""
        var = dct_knot_variance(self.power_spectrum())
        if self.template is not None:
            var = jnp.square(self.template) * var
        return var

    def evaluate_grid(self) -> jax.Array:
        """Evaluate perturbation ``delta_q`` at the ``n_grid`` uniform knots."""
        amp = _safe_sqrt(self.power_spectrum())
        eta = idct2(amp * self.latent)
        if self.template is not None:
            eta = self.template * eta
        return eta

    def with_latent(self, latent: ArrayLike) -> GaussianField:
        """Return a copy of this field with updated standardized excitations ``latent``."""
        z = jnp.asarray(latent, dtype=jnp.float64)
        if z.shape != self.latent.shape:
            raise ValueError(f"Expected latent shape {self.latent.shape}, got {z.shape}.")
        return eqx.tree_at(lambda f: f.latent, self, z)

    def sample_prior(
        self,
        rng_key: jax.Array,
        n_samples: int | None = None,
    ) -> jax.Array:
        """Draw prior realizations of ``delta_q(t_grid)`` from ``z ~ N(0, I_N)``."""
        if n_samples is None:
            z = jax.random.normal(rng_key, (self.n_grid,), dtype=jnp.float64)
            return self.with_latent(z).evaluate_grid()
        zs = jax.random.normal(rng_key, (int(n_samples), self.n_grid), dtype=jnp.float64)
        return jax.vmap(lambda z: self.with_latent(z).evaluate_grid())(zs)

    def prior_covariance(self) -> jax.Array:
        """Dense ``(n_grid, n_grid)`` prior covariance ``C^T diag(S) C`` (for validation)."""
        spec = self.power_spectrum()
        eye = jnp.eye(self.n_grid, dtype=jnp.float64)
        c_t = idct2(eye, axis=0)  # columns are inverse DCT basis vectors
        cov = (c_t * spec[None, :]) @ c_t.T
        if self.template is not None:
            cov = (self.template[:, None] * cov) * self.template[None, :]
        return 0.5 * (cov + cov.T)

    def to_gridded(self, q_std: ArrayLike = DEFAULT_Q_STD) -> GriddedProduction:
        """Materialize as a piecewise-linear :class:`GriddedProduction` on ``t_grid``."""
        del q_std
        return GriddedProduction(self.t_grid, self.evaluate_grid(), is_delta=True)

    def delta_q(
        self,
        t: ArrayLike,
        q_std: ArrayLike = DEFAULT_Q_STD,
    ) -> jax.Array:
        del q_std
        t_arr = jnp.asarray(t, dtype=jnp.float64)
        return jnp.interp(t_arr, self.t_grid, self.evaluate_grid())


class LogGaussianField(ProductionTerm):
    r"""Strictly positive IFT log-Gaussian production field ``Q(t) > 0``.

    Direct 1D counterpart of ``virgil``'s positive ``Image(GaussianField(...))``:

    .. math::

        Q(t_g) = q_{\mathrm{mean}}\,\tilde{\mu}_g\,\exp\!\left(\eta_g - \tfrac{1}{2} c_{\mathrm{Ito}}\,\sigma_{\eta,g}^2\right) > 0,

    where ``eta = IDCT_II(sqrt(S) * z)`` with standardized excitations ``z ~ N(0, I_N)``.
    When ``ito_corrected=True``, the exact per-knot variance correction
    ``-0.5 * dct_knot_variance(S)`` ensures ``E_{z ~ N(0, I)}[Q(t_g)] == q_mean * tilde_mu_g``.
    """

    latent: jax.Array
    sigma: jax.Array
    length_yr: jax.Array
    order: jax.Array
    q_mean: jax.Array
    kernel: SpectralKernel | None
    template: jax.Array | None
    t_start: float = eqx.field(static=True)
    t_end: float = eqx.field(static=True)
    mean_floor: float = eqx.field(static=True)
    zero_mode: bool = eqx.field(static=True)
    ito_corrected: bool = eqx.field(static=True)

    def __init__(
        self,
        latent: ArrayLike,
        t_start: float,
        t_end: float,
        sigma: ArrayLike = 0.3,
        length_yr: ArrayLike = 5.0,
        order: ArrayLike = 2.0,
        *,
        kernel: SpectralKernel | None = None,
        q_mean: ArrayLike = DEFAULT_Q_STD,
        template: ArrayLike | None = None,
        mean_floor: float = 1e-3,
        zero_mode: bool = True,
        ito_corrected: bool = False,
    ) -> None:
        z = jnp.asarray(latent, dtype=jnp.float64)
        if z.ndim != 1 or z.shape[0] < 2:
            raise ValueError(
                f"LogGaussianField latent must be 1D with length >= 2, got shape {z.shape}."
            )
        t0, t1 = float(t_start), float(t_end)
        if not (math.isfinite(t0) and math.isfinite(t1) and t1 > t0):
            raise ValueError(f"Expected finite t_end > t_start, got [{t_start}, {t_end}].")
        _check_positive_concrete("sigma", sigma, allow_zero=True)
        _check_positive_concrete("length_yr", length_yr)
        _check_positive_concrete("q_mean", q_mean)
        _check_positive_concrete("mean_floor", mean_floor)
        if kernel is not None and not isinstance(kernel, SpectralKernel):
            raise TypeError(f"kernel must be a SpectralKernel or None, got {type(kernel)!r}.")

        if template is None:
            tmpl = None
        else:
            tmpl = jnp.asarray(template, dtype=jnp.float64)
            if tmpl.shape != z.shape:
                raise ValueError(
                    f"template shape {tmpl.shape} must match latent shape {z.shape}."
                )
            t_np = _concrete_np(tmpl)
            if t_np is not None and not (np.all(np.isfinite(t_np)) and np.max(t_np) > 0.0):
                raise ValueError("template must be finite with a strictly positive peak.")

        self.latent = z
        self.t_start = t0
        self.t_end = t1
        self.sigma = jnp.asarray(sigma, dtype=jnp.float64).reshape(())
        self.length_yr = jnp.asarray(length_yr, dtype=jnp.float64).reshape(())
        self.order = jnp.asarray(order, dtype=jnp.float64).reshape(())
        self.q_mean = jnp.asarray(q_mean, dtype=jnp.float64).reshape(())
        self.kernel = kernel
        self.template = tmpl
        self.mean_floor = float(mean_floor)
        self.zero_mode = bool(zero_mode)
        self.ito_corrected = bool(ito_corrected)

    def _default_slug_name(self) -> str:
        return "log_field"

    @property
    def n_grid(self) -> int:
        return int(self.latent.shape[0])

    @property
    def dt(self) -> float:
        return (self.t_end - self.t_start) / float(self.n_grid - 1)

    @property
    def t_grid(self) -> jax.Array:
        return jnp.linspace(self.t_start, self.t_end, self.n_grid, dtype=jnp.float64)

    @property
    def effective_kernel(self) -> SpectralKernel:
        if self.kernel is not None:
            return self.kernel
        return MaternSpectrum(
            sigma=self.sigma,
            length_yr=self.length_yr,
            order=self.order,
            zero_mode=self.zero_mode,
        )

    def power_spectrum(self) -> jax.Array:
        """Return DCT-II mode variances ``S_j`` of shape ``(n_grid,)``."""
        return self.effective_kernel.spectrum(self.n_grid, self.dt)

    def knot_variance(self) -> jax.Array:
        """Exact per-knot variance ``Var[eta(t_g)]`` of shape ``(n_grid,)``."""
        return dct_knot_variance(self.power_spectrum())

    def evaluate_log_field(self) -> jax.Array:
        """Evaluate log-rate perturbation ``eta(t_grid)`` (including optional template & Itô shift)."""
        spec = self.power_spectrum()
        eta = idct2(_safe_sqrt(spec) * self.latent)
        if self.ito_corrected:
            eta = eta - 0.5 * dct_knot_variance(spec)
        if self.template is not None:
            norm_t = jnp.maximum(self.template, 0.0) / jnp.max(self.template)
            eta = eta + jnp.log((norm_t + self.mean_floor) / (1.0 + self.mean_floor))
        return eta

    def evaluate_q_grid(self) -> jax.Array:
        """Evaluate strictly positive production rate ``Q(t_grid) > 0`` at uniform knots."""
        return self.q_mean * jnp.exp(self.evaluate_log_field())

    def evaluate_grid(self, q_std: ArrayLike = DEFAULT_Q_STD) -> jax.Array:
        """Evaluate ``delta_q(t_grid) = Q(t_grid) - q_std`` at uniform knots."""
        return self.evaluate_q_grid() - jnp.asarray(q_std, dtype=jnp.float64)

    def with_latent(self, latent: ArrayLike) -> LogGaussianField:
        """Return a copy of this field with updated standardized excitations ``latent``."""
        z = jnp.asarray(latent, dtype=jnp.float64)
        if z.shape != self.latent.shape:
            raise ValueError(f"Expected latent shape {self.latent.shape}, got {z.shape}.")
        return eqx.tree_at(lambda f: f.latent, self, z)

    def sample_prior(
        self,
        rng_key: jax.Array,
        n_samples: int | None = None,
    ) -> jax.Array:
        """Draw strictly positive prior realizations ``Q(t_grid) > 0`` from ``z ~ N(0, I_N)``."""
        if n_samples is None:
            z = jax.random.normal(rng_key, (self.n_grid,), dtype=jnp.float64)
            return self.with_latent(z).evaluate_q_grid()
        zs = jax.random.normal(rng_key, (int(n_samples), self.n_grid), dtype=jnp.float64)
        return jax.vmap(lambda z: self.with_latent(z).evaluate_q_grid())(zs)

    def to_gridded(self, q_std: ArrayLike = DEFAULT_Q_STD) -> GriddedProduction:
        """Materialize as a piecewise-linear :class:`GriddedProduction` on ``t_grid``."""
        del q_std
        return GriddedProduction(self.t_grid, self.evaluate_q_grid(), is_delta=False)

    def delta_q(
        self,
        t: ArrayLike,
        q_std: ArrayLike = DEFAULT_Q_STD,
    ) -> jax.Array:
        t_arr = jnp.asarray(t, dtype=jnp.float64)
        q_interp = jnp.interp(t_arr, self.t_grid, self.evaluate_q_grid())
        return q_interp - jnp.asarray(q_std, dtype=jnp.float64)


# =============================================================================
# 3. Precomputed First-Order-Hold DCT-II Response Operator
# =============================================================================


class DCTResponseOperator(zdx.Base):
    """Precomputed exact FOH seasonal-window CBM response on a uniform DCT-II grid."""

    w_grid: jax.Array
    w_dct: jax.Array
    ss_d14c: jax.Array
    box_activity_ratios: jax.Array
    steady_unit_response: jax.Array
    init_state_basis: jax.Array
    t_start: float = eqx.field(static=True)
    t_end: float = eqx.field(static=True)
    n_grid: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)

    @property
    def t_grid(self) -> jax.Array:
        return jnp.linspace(self.t_start, self.t_end, self.n_grid, dtype=jnp.float64)


def _pack_dataset_quadrature(
    cbm: CarbonBoxModel,
    dataset: Dataset,
) -> tuple[jax.Array, jax.Array, tuple[tuple[jax.Array, int, int, jax.Array, int], ...]]:
    """Pack deduplicated sub-window start/end times and series gather indices."""
    starts_all: list[np.ndarray] = []
    ends_all: list[np.ndarray] = []
    raw_specs: list[tuple[int, int, int, jax.Array, int]] = []
    cursor = 0
    for s in dataset.series:
        sub_s, sub_e, w_sub = s.quadrature_windows()  # (N_s, M_s), (N_s, M_s), (M_s,)
        n_s, m_s = int(sub_s.shape[0]), int(sub_s.shape[1])
        box_idx = cbm.box_index(s.box_name, hemisphere=s.hemisphere)
        starts_all.append(np.asarray(sub_s, dtype=np.float64).reshape(-1))
        ends_all.append(np.asarray(sub_e, dtype=np.float64).reshape(-1))
        raw_specs.append((cursor, n_s, m_s, w_sub, box_idx))
        cursor += n_s * m_s

    s_cat = np.concatenate(starts_all, axis=0)
    e_cat = np.concatenate(ends_all, axis=0)
    pairs = np.column_stack([np.round(s_cat, 11), np.round(e_cat, 11)])
    _, first_idx, inv_all = np.unique(
        pairs, axis=0, return_index=True, return_inverse=True
    )
    uniq_s = jnp.asarray(s_cat[first_idx], dtype=jnp.float64)
    uniq_e = jnp.asarray(e_cat[first_idx], dtype=jnp.float64)
    specs: list[tuple[jax.Array, int, int, jax.Array, int]] = []
    for start_idx, n_s, m_s, w_sub, box_idx in raw_specs:
        end_idx = start_idx + n_s * m_s
        inv_slice = jnp.asarray(inv_all[start_idx:end_idx], dtype=jnp.int32)
        specs.append((inv_slice, n_s, m_s, w_sub, box_idx))
    return uniq_s, uniq_e, tuple(specs)


def _foh_excursion_from_packed(
    cbm: CarbonBoxModel,
    q_std: ArrayLike,
    alpha_unit: ArrayLike,
    t0: jax.Array,
    dt: jax.Array,
    dq_grid: jax.Array,
    dy0: jax.Array,
    flat_starts: jax.Array,
    flat_ends: jax.Array,
    specs: tuple[tuple[jax.Array, int, int, jax.Array, int], ...],
) -> jax.Array:
    """Exact seasonal-binned ``Delta14C`` excursion (permil) for ``(dq_grid, dy0)``."""
    A = cbm.matrix_A()
    b_scaled = jnp.asarray(alpha_unit, dtype=jnp.float64) * cbm.production_vector()
    trop_idx = jnp.asarray(cbm.topology.trop_box_indices, dtype=jnp.int32)
    y_ss = cbm.steady_state_c14(q_std=q_std, alpha_unit=alpha_unit)
    m_rel = cbm.relative_reservoir_masses()
    y_trop = jnp.sum(y_ss[trop_idx])

    m_tot = flat_starts.shape[0]
    queries = jnp.concatenate([flat_starts, flat_ends], axis=0)
    _, Y_all = _foh_cumulative_integral(A, b_scaled, t0, dt, dq_grid, dy0, queries)
    Y_a = Y_all[:m_tot]
    Y_b = Y_all[m_tot:]
    widths = (flat_ends - flat_starts)[:, None]
    dy_sub = (Y_b - Y_a) / widths  # (M_uniq, n_boxes)

    series_out: list[jax.Array] = []
    for inv_slice, n_s, m_s, w_sub, box_idx in specs:
        block = dy_sub[inv_slice, box_idx].reshape(n_s, m_s)
        dy_ring = jnp.sum(block * w_sub[None, :], axis=1)
        d14c_exc = 1000.0 * ((dy_ring / y_trop) / m_rel[box_idx])
        series_out.append(d14c_exc)
    return jnp.concatenate(series_out, axis=0)


@eqx.filter_jit
def _compute_dct_operator_jit(
    cbm: CarbonBoxModel,
    q_std: jax.Array,
    alpha_unit: jax.Array,
    t_start: float,
    t_end: float,
    n_grid: int,
    flat_starts: jax.Array,
    flat_ends: jax.Array,
    specs: tuple[tuple[jax.Array, int, int, jax.Array, int], ...],
    template: jax.Array | None,
) -> DCTResponseOperator:
    dt_f = (t_end - t_start) / float(n_grid - 1)
    t0_arr = jnp.asarray(t_start, dtype=jnp.float64)
    dt_arr = jnp.asarray(dt_f, dtype=jnp.float64)
    dq_zero = jnp.zeros((n_grid,), dtype=jnp.float64)
    dy0_zero = jnp.zeros((cbm.n_boxes,), dtype=jnp.float64)

    w_grid, init_basis = jax.jacfwd(
        lambda dq, dy0: _foh_excursion_from_packed(
            cbm, q_std, alpha_unit, t0_arr, dt_arr, dq, dy0, flat_starts, flat_ends, specs
        ),
        argnums=(0, 1),
    )(dq_zero, dy0_zero)

    w_eff = w_grid if template is None else w_grid * template[None, :]
    w_dct = dct2(w_eff, axis=1)

    trop_idx = jnp.asarray(cbm.topology.trop_box_indices, dtype=jnp.int32)
    y_ss = cbm.steady_state_c14(q_std=q_std, alpha_unit=alpha_unit)
    m_rel = cbm.relative_reservoir_masses()
    y_trop = jnp.sum(y_ss[trop_idx])
    r_ss_ratio = (y_ss / y_trop) / m_rel
    ss_all = 1000.0 * (r_ss_ratio - 1.0)

    A = cbm.matrix_A()
    b_scaled = alpha_unit * cbm.production_vector()
    v_unit = jnp.linalg.solve(-A, b_scaled)
    unit_d14c_boxes = 1000.0 * ((v_unit / y_trop) / m_rel)

    ss_blocks: list[jax.Array] = []
    ratio_blocks: list[jax.Array] = []
    unit_blocks: list[jax.Array] = []
    for _, n_s, _, _, box_idx in specs:
        ss_blocks.append(jnp.full((n_s,), ss_all[box_idx], dtype=jnp.float64))
        ratio_blocks.append(jnp.full((n_s,), r_ss_ratio[box_idx], dtype=jnp.float64))
        unit_blocks.append(jnp.full((n_s,), unit_d14c_boxes[box_idx], dtype=jnp.float64))

    return DCTResponseOperator(
        w_grid=w_grid,
        w_dct=w_dct,
        ss_d14c=jnp.concatenate(ss_blocks, axis=0),
        box_activity_ratios=jnp.concatenate(ratio_blocks, axis=0),
        steady_unit_response=jnp.concatenate(unit_blocks, axis=0),
        init_state_basis=init_basis,
        t_start=t_start,
        t_end=t_end,
        n_grid=n_grid,
        dt=dt_f,
    )


def build_dct_response_operator(
    cbm: CarbonBoxModel,
    dataset: Dataset | TreeRingSeries,
    t_start: float | None = None,
    t_end: float | None = None,
    n_grid: int = 128,
    *,
    pad_years: float = 15.0,
    q_std: ArrayLike = DEFAULT_Q_STD,
    alpha_unit: ArrayLike = ALPHA_ATOMS_CM2_S_TO_KG_YR,
    template: ArrayLike | None = None,
) -> DCTResponseOperator:
    """Precompute the exact FOH seasonal-window response operator :class:`DCTResponseOperator`.

    Once built for ``(cbm, dataset, t_start, t_end, n_grid)``, evaluating exact marginal
    log-evidence across any :class:`SpectralKernel` ``S_j(phi)`` requires only column
    rescaling ``w_dct * sqrt(S(phi))`` with zero additional matrix exponentials or ODE scans.
    """
    ds = _as_dataset(dataset)
    n_g = int(n_grid)
    if n_g < 2:
        raise ValueError(f"n_grid must be >= 2, got {n_grid}.")
    span_lo, span_hi = ds.time_span
    t0 = float(span_lo - float(pad_years)) if t_start is None else float(t_start)
    t1 = float(span_hi + 0.5) if t_end is None else float(t_end)
    if not (t1 > t0):
        raise ValueError(f"Expected t_end > t_start, got [{t0}, {t1}].")

    tmpl_arr = None if template is None else jnp.asarray(template, dtype=jnp.float64)
    if tmpl_arr is not None and tmpl_arr.shape != (n_g,):
        raise ValueError(f"template shape {tmpl_arr.shape} must be ({n_g},).")

    flat_s, flat_e, specs = _pack_dataset_quadrature(cbm, ds)
    return _compute_dct_operator_jit(
        cbm,
        jnp.asarray(q_std, dtype=jnp.float64).reshape(()),
        jnp.asarray(alpha_unit, dtype=jnp.float64).reshape(()),
        t0,
        t1,
        n_g,
        flat_s,
        flat_e,
        specs,
        tmpl_arr,
    )


def build_dct_response_matrix(
    cbm: CarbonBoxModel,
    dataset: Dataset | TreeRingSeries,
    t_start: float,
    t_end: float,
    n_grid: int,
    *,
    q_std: ArrayLike = DEFAULT_Q_STD,
    alpha_unit: ArrayLike = ALPHA_ATOMS_CM2_S_TO_KG_YR,
    template: ArrayLike | None = None,
    return_operator: bool = False,
) -> jax.Array | DCTResponseOperator:
    """Return the ``(N_obs, n_grid)`` DCT-II response matrix ``W_DCT`` (or full operator).

    Satisfies ``Delta(Delta14C) == W_DCT @ (sqrt(S) * z)`` to machine precision against
    :class:`~ticktack.solvers.ExactLinearSolver` on ``GriddedProduction(t_grid, dq, is_delta=True)``.
    """
    op = build_dct_response_operator(
        cbm,
        dataset,
        t_start=t_start,
        t_end=t_end,
        n_grid=n_grid,
        q_std=q_std,
        alpha_unit=alpha_unit,
        template=template,
    )
    return op if return_operator else op.w_dct


# =============================================================================
# 4. Model Unpacking, Wiener / LM Fitting, Evidence, Samples, & Error Scale
# =============================================================================


def _extract_field_and_extras(
    production: ProductionTerm,
    prefix: str = "production",
) -> tuple[GaussianField | LogGaussianField, str, tuple[tuple[str, ProductionTerm], ...]]:
    """Locate the single IFT field term and any sibling ``ProductionTerm`` components."""
    found_fields: list[tuple[str, GaussianField | LogGaussianField]] = []
    extras: list[tuple[str, ProductionTerm]] = []

    def _walk(term: ProductionTerm, path: str) -> None:
        if isinstance(term, (GaussianField, LogGaussianField)):
            found_fields.append((path, term))
        elif isinstance(term, ProductionSum):
            for name, part in zip(term.names, term.parts):
                _walk(part, f"{path}.{name}")
        else:
            extras.append((path, term))

    _walk(production, prefix)
    if len(found_fields) != 1:
        raise ValueError(
            f"Expected exactly one GaussianField or LogGaussianField in model.production; "
            f"found {len(found_fields)}."
        )
    field_path, field_obj = found_fields[0]
    return field_obj, field_path, tuple(extras)


@eqx.filter_jit
def _eval_packed_term_d14c(
    cbm: CarbonBoxModel,
    solver: ExactLinearSolver,
    term: ProductionTerm,
    flat_starts: jax.Array,
    flat_ends: jax.Array,
    specs: tuple[tuple[jax.Array, int, int, jax.Array, int], ...],
    y_trop: jax.Array,
    m_rel: jax.Array,
    q_std: ArrayLike,
    alpha_unit: ArrayLike,
) -> jax.Array:
    """Evaluate a single ``ProductionTerm`` across all deduplicated dataset sub-windows."""
    dy_sub = solver.solve_delta_binned(
        cbm,
        term,
        flat_starts,
        flat_ends,
        q_std=q_std,
        alpha_unit=alpha_unit,
    )
    series_out: list[jax.Array] = []
    for inv_slice, n_s, m_s, w_sub, box_idx in specs:
        block = dy_sub[inv_slice, box_idx].reshape(n_s, m_s)
        dy_ring = jnp.sum(block * w_sub[None, :], axis=1)
        series_out.append(1000.0 * ((dy_ring / y_trop) / m_rel[box_idx]))
    return jnp.concatenate(series_out, axis=0)


def _eval_extra_terms_d14c(
    cbm: CarbonBoxModel,
    solver: ExactLinearSolver,
    dataset: Dataset,
    extras: tuple[tuple[str, ProductionTerm], ...],
    q_std: ArrayLike,
    alpha_unit: ArrayLike,
    steady_unit_response: jax.Array,
) -> jax.Array:
    """Evaluate additive ``Delta14C`` excursion from non-field ``ProductionTerm`` siblings."""
    n_obs = dataset.n_obs
    if not extras:
        return jnp.zeros((n_obs,), dtype=jnp.float64)

    trop_idx = jnp.asarray(cbm.topology.trop_box_indices, dtype=jnp.int32)
    y_ss = cbm.steady_state_c14(q_std=q_std, alpha_unit=alpha_unit)
    m_rel = cbm.relative_reservoir_masses()
    y_trop = jnp.sum(y_ss[trop_idx])

    non_steady = [term for _, term in extras if not isinstance(term, SteadyProduction)]
    packed = _pack_dataset_quadrature(cbm, dataset) if non_steady else None

    total_exc = jnp.zeros((n_obs,), dtype=jnp.float64)
    for _, term in extras:
        if isinstance(term, SteadyProduction):
            dq = term.rate - jnp.asarray(q_std, dtype=jnp.float64)
            total_exc = total_exc + dq * steady_unit_response
        else:
            assert packed is not None
            flat_s, flat_e, specs = packed
            total_exc = total_exc + _eval_packed_term_d14c(
                cbm, solver, term, flat_s, flat_e, specs, y_trop, m_rel, q_std, alpha_unit
            )
    return total_exc


def predict_field_d14c(
    model_or_result: RadiocarbonModel | IFTFieldFitResult,
    dataset_or_series: Dataset | TreeRingSeries | None = None,
    *,
    t_fine: ArrayLike | None = None,
    hemisphere: str | None = "north",
    box_name: str = "troposphere",
    include_offsets: bool = True,
    baseline_d14c: ArrayLike | None = None,
    operator: DCTResponseOperator | None = None,
) -> jax.Array:
    """Predict ring-binned or continuous fine-grid ``Delta14C`` (‰) for an IFT field model."""
    if isinstance(model_or_result, IFTFieldFitResult):
        fit_res: IFTFieldFitResult | None = model_or_result
        model = fit_res.model
        eff_op = fit_res.operator if operator is None else operator
    elif isinstance(model_or_result, RadiocarbonModel):
        fit_res = None
        model = model_or_result
        eff_op = operator
    else:
        raise TypeError(
            f"Expected RadiocarbonModel or IFTFieldFitResult, got {type(model_or_result)!r}."
        )

    solver = (
        model.solver
        if isinstance(model.solver, ExactLinearSolver)
        else ExactLinearSolver()
    )
    field, _, extras = _extract_field_and_extras(model.production)

    if t_fine is not None:
        t_arr = jnp.atleast_1d(jnp.asarray(t_fine, dtype=jnp.float64))
        trop_idx = jnp.asarray(model.cbm.topology.trop_box_indices, dtype=jnp.int32)
        y_ss = model.cbm.steady_state_c14(q_std=model.q_std, alpha_unit=model.alpha_unit)
        m_rel = model.cbm.relative_reservoir_masses()
        y_trop = jnp.sum(y_ss[trop_idx])
        box_idx = model.cbm.box_index(box_name, hemisphere=hemisphere)
        r_ratio = (y_ss[box_idx] / y_trop) / m_rel[box_idx]
        ss_val = 1000.0 * (r_ratio - 1.0)

        grid_term: ProductionTerm = field.to_gridded(q_std=model.q_std)
        for _, ext_t in extras:
            grid_term = grid_term + ext_t
        dy_traj = solver.solve_delta_trajectory(
            model.cbm,
            grid_term,
            t_arr,
            q_std=model.q_std,
            alpha_unit=model.alpha_unit,
        )
        d14c_curve = ss_val + 1000.0 * ((dy_traj[:, box_idx] / y_trop) / m_rel[box_idx])
        if baseline_d14c is not None:
            d14c_curve = d14c_curve + jnp.asarray(baseline_d14c, dtype=jnp.float64)
        elif include_offsets and fit_res is not None:
            d14c_curve = d14c_curve + fit_res.global_offset * r_ratio
        return d14c_curve

    if dataset_or_series is None:
        if fit_res is not None:
            return fit_res.d14c_pred if include_offsets else (fit_res.d14c_pred - fit_res.baseline_d14c)
        raise ValueError("dataset_or_series or t_fine must be supplied when passing a RadiocarbonModel.")

    ds = _as_dataset(dataset_or_series)
    if (
        fit_res is not None
        and operator is None
        and baseline_d14c is None
        and ds.n_obs == fit_res.d14c_pred.shape[0]
    ):
        return fit_res.d14c_pred if include_offsets else (fit_res.d14c_pred - fit_res.baseline_d14c)

    tmpl = field.template if isinstance(field, GaussianField) else None
    op = (
        build_dct_response_operator(
            model.cbm,
            ds,
            t_start=field.t_start,
            t_end=field.t_end,
            n_grid=field.n_grid,
            q_std=model.q_std,
            alpha_unit=model.alpha_unit,
            template=tmpl,
        )
        if eff_op is None or eff_op.w_grid.shape[0] != ds.n_obs
        else eff_op
    )
    extra_exc = _eval_extra_terms_d14c(
        model.cbm,
        solver,
        ds,
        extras,
        model.q_std,
        model.alpha_unit,
        op.steady_unit_response,
    )
    if isinstance(field, GaussianField):
        field_exc = op.w_dct @ (_safe_sqrt(field.power_spectrum()) * field.latent)
    else:
        field_exc = op.w_grid @ field.evaluate_grid(q_std=model.q_std)

    pred = op.ss_d14c + extra_exc + field_exc
    if baseline_d14c is not None:
        pred = pred + jnp.asarray(baseline_d14c, dtype=jnp.float64)
    elif include_offsets and fit_res is not None and fit_res.baseline_d14c.shape[0] == ds.n_obs:
        pred = pred + fit_res.baseline_d14c
    return pred


def _build_offset_whitening(
    dataset: Dataset,
    box_activity_ratios: jax.Array,
    *,
    offset_mode: Literal["hierarchical", "none"],
    global_offset_prior_sd: float,
    lab_offset_prior_sd: float,
    prehistory_cov_y0: jax.Array | None = None,
    init_state_basis: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return ``(X_nuis, mu_nuis, root_nuis, U_nuis)`` for QR whitening of offsets & ``y_0``."""
    n_obs = dataset.n_obs
    inv_sig = 1.0 / dataset.sigma_flat

    designs: list[jax.Array] = []
    means: list[jax.Array] = []
    roots: list[jax.Array] = []

    if offset_mode == "hierarchical":
        x_off, mu_off, sd_off = hierarchical_offset_design(
            dataset.series_indices,
            dataset.n_series,
            box_activity_ratios=box_activity_ratios,
            global_prior_sd=global_offset_prior_sd,
            lab_prior_sd=lab_offset_prior_sd if dataset.n_series > 1 else 0.0,
        )
        designs.append(x_off)
        means.append(mu_off)
        roots.append(jnp.diag(sd_off))
    elif offset_mode != "none":
        raise ValueError(
            f"offset_mode must be 'hierarchical' or 'none', got {offset_mode!r}."
        )

    if prehistory_cov_y0 is not None and init_state_basis is not None:
        cov0 = 0.5 * (prehistory_cov_y0 + prehistory_cov_y0.T)
        evals, evecs = jnp.linalg.eigh(cov0)
        pos = evals > 1e-30
        l_y0 = evecs * jnp.where(pos, jnp.sqrt(jnp.where(pos, evals, 1.0)), 0.0)[None, :]
        designs.append(init_state_basis)
        means.append(jnp.zeros((init_state_basis.shape[1],), dtype=jnp.float64))
        roots.append(l_y0)

    if not designs:
        empty_x = jnp.zeros((n_obs, 0), dtype=jnp.float64)
        empty_m = jnp.zeros((0,), dtype=jnp.float64)
        empty_r = jnp.zeros((0, 0), dtype=jnp.float64)
        return empty_x, empty_m, empty_r, empty_x

    x_nuis = jnp.concatenate(designs, axis=1)
    mu_nuis = jnp.concatenate(means, axis=0)
    root_nuis = jsl.block_diag(*roots)
    u_nuis = (inv_sig[:, None] * x_nuis) @ root_nuis
    return x_nuis, mu_nuis, root_nuis, u_nuis


@jax.jit
def _whiten_vec_and_mat_qr(
    vec: jax.Array,
    mat: jax.Array,
    u_nuis: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Apply exact QR square-root whitening ``W_nuis`` to ``vec`` ``(N,)`` and ``mat`` ``(N, P)``."""
    k = u_nuis.shape[1]
    if k == 0:
        return vec, mat, jnp.asarray(0.0, dtype=jnp.float64)
    combined = jnp.concatenate([vec[:, None], mat], axis=1)
    whitened, half_logdet = whiten_qr(combined, u_nuis)
    return whitened[:, 0], whitened[:, 1:], half_logdet


class IFTFieldFitResult(zdx.Base):
    """Result of :func:`fit_ift_field` (exact Wiener solution or Levenberg–Marquardt MAP)."""

    model: RadiocarbonModel
    latent: jax.Array
    field_grid: jax.Array
    q_grid: jax.Array
    t_grid: jax.Array
    d14c_pred: jax.Array
    baseline_d14c: jax.Array
    global_offset: jax.Array
    lab_offsets: jax.Array
    chi2: jax.Array
    latent_norm_sq: jax.Array
    log_evidence: jax.Array
    gamma: jax.Array
    operator: DCTResponseOperator
    offset_mode: str = eqx.field(static=True)
    global_offset_prior_sd: float = eqx.field(static=True)
    lab_offset_prior_sd: float = eqx.field(static=True)
    converged: bool = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)


@jax.jit
def _wiener_solve_from_svd(
    x_target: jax.Array,
    j_mat: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Exact SVD Wiener solve ``z_hat = (I + J^T J)^{-1} J^T x_target``."""
    u_svd, s_vals, vt_svd = jnp.linalg.svd(j_mat, full_matrices=False)
    ut_x = u_svd.T @ x_target
    z_hat = vt_svd.T @ ((s_vals / (1.0 + jnp.square(s_vals))) * ut_x)
    r_post = x_target - j_mat @ z_hat
    chi2_eff = r_post @ r_post
    return z_hat, s_vals, vt_svd, chi2_eff


def fit_ift_field(
    model: RadiocarbonModel,
    dataset: Dataset | TreeRingSeries,
    *,
    offset_mode: Literal["hierarchical", "none"] = "hierarchical",
    global_offset_prior_sd: float = 200.0,
    lab_offset_prior_sd: float = 1.5,
    optimize_spike: bool = False,
    max_steps: int = 50,
    tol: float = 1e-8,
    operator: DCTResponseOperator | None = None,
    prehistory_kernel: Any | None = None,
) -> IFTFieldFitResult:
    """Fit a :class:`GaussianField` (exact Wiener) or :class:`LogGaussianField` (whitened LM).

    Minimizes the whitened IFT Hamiltonian ``H(d, z) = 0.5 ||r_eff(z)||^2 + 0.5 ||z||^2``
    with analytical QR marginalization over hierarchical global and per-series laboratory
    baseline offsets ``(Delta_global, delta_{lab,s})``.

    * For linear :class:`GaussianField` (with ``optimize_spike=False``), computes the exact
      closed-form SVD Wiener filter in a single pass (``n_steps = 1``).
    * For nonlinear :class:`LogGaussianField` (or joint ``MiyakeSpike`` optimization when
      ``optimize_spike=True``), runs ``optimistix.LevenbergMarquardt`` on the full vector
      residual ``[r_eff(z), z]``, typically converging to ``1e-8`` in 10–25 steps.
    """
    ds = _as_dataset(dataset)
    field, field_path, extras = _extract_field_and_extras(model.production)
    tmpl = field.template if isinstance(field, GaussianField) else None
    op = (
        build_dct_response_operator(
            model.cbm,
            ds,
            t_start=field.t_start,
            t_end=field.t_end,
            n_grid=field.n_grid,
            q_std=model.q_std,
            alpha_unit=model.alpha_unit,
            template=tmpl,
        )
        if operator is None
        else operator
    )
    pre_cov = (
        stationary_cbm_gp_cov(
            model.cbm.matrix_A(),
            model.alpha_unit * model.cbm.production_vector(),
            prehistory_kernel,
        )
        if prehistory_kernel is not None
        else None
    )
    x_nuis, mu_nuis, root_nuis, u_nuis = _build_offset_whitening(
        ds,
        op.box_activity_ratios,
        offset_mode=offset_mode,
        global_offset_prior_sd=global_offset_prior_sd,
        lab_offset_prior_sd=lab_offset_prior_sd,
        prehistory_cov_y0=pre_cov,
        init_state_basis=op.init_state_basis,
    )

    solver = (
        model.solver
        if isinstance(model.solver, ExactLinearSolver)
        else ExactLinearSolver()
    )
    inv_sig = 1.0 / ds.sigma_flat
    d_obs = ds.d14c_flat
    n_obs = ds.n_obs
    log_norm_d = 0.5 * n_obs * math.log(2.0 * math.pi) + jnp.sum(jnp.log(ds.sigma_flat))

    spike_entries = [
        (p, t) for p, t in extras if optimize_spike and isinstance(t, MiyakeSpike)
    ]

    if isinstance(field, GaussianField) and not spike_entries:
        extra_exc = _eval_extra_terms_d14c(
            model.cbm, solver, ds, extras, model.q_std, model.alpha_unit, op.steady_unit_response
        )
        m_base = op.ss_d14c + extra_exc + (x_nuis @ mu_nuis if x_nuis.shape[1] else 0.0)
        x_raw = inv_sig * (d_obs - m_base)
        j_raw = inv_sig[:, None] * (op.w_dct * _safe_sqrt(field.power_spectrum())[None, :])
        x_eff, j_eff, half_logdet_nuis = _whiten_vec_and_mat_qr(x_raw, j_raw, u_nuis)
        z_hat, s_vals, _, chi2_eff = _wiener_solve_from_svd(x_eff, j_eff)
        converged = True
        n_steps = 1
        fitted_model = model.set(f"{field_path}.latent", z_hat)
    elif isinstance(field, GaussianField) and spike_entries:
        # Golub-Pereyra variable projection: marginalize the n_z linear GaussianField
        # latents AND hierarchical offsets analytically outside LM so LM only solves
        # a 3-variable nonlinear least-squares problem for (t_start, log_dur, log_fl)!
        sp_path, sp_term = spike_entries[0]
        u_sp0 = jnp.array(
            [
                sp_term.t_start,
                jnp.log(jnp.maximum(sp_term.duration, 1e-4)),
                jnp.log(jnp.maximum(sp_term.fluence, 1e-4)),
            ],
            dtype=jnp.float64,
        )
        t_mid_prior = sp_term.t_start
        fixed_extras = tuple((p, t) for p, t in extras if p != sp_path)
        fixed_exc = _eval_extra_terms_d14c(
            model.cbm,
            solver,
            ds,
            fixed_extras,
            model.q_std,
            model.alpha_unit,
            op.steady_unit_response,
        )
        nuis_mean_shift = x_nuis @ mu_nuis if x_nuis.shape[1] else 0.0
        j_raw = inv_sig[:, None] * (op.w_dct * _safe_sqrt(field.power_spectrum())[None, :])
        u_all = jnp.concatenate([u_nuis, j_raw], axis=1)
        cov_marg = jnp.eye(n_obs, dtype=jnp.float64) + u_all @ u_all.T
        chol_marg = jnp.linalg.cholesky(cov_marg)

        flat_s, flat_e, packed_specs = _pack_dataset_quadrature(model.cbm, ds)
        trop_idx = jnp.asarray(model.cbm.topology.trop_box_indices, dtype=jnp.int32)
        y_ss = model.cbm.steady_state_c14(q_std=model.q_std, alpha_unit=model.alpha_unit)
        m_rel = model.cbm.relative_reservoir_masses()
        y_trop = jnp.sum(y_ss[trop_idx])
        r_fixed_raw = inv_sig * (op.ss_d14c + fixed_exc + nuis_mean_shift - d_obs)

        def _varpro_spike_residual(u_sp: jax.Array, _args: Any) -> jax.Array:
            sp_cur = eqx.tree_at(
                lambda s: (s.t_start, s.duration, s.fluence),
                sp_term,
                (u_sp[0], jnp.exp(u_sp[1]), jnp.exp(u_sp[2])),
            )
            sp_exc = _eval_packed_term_d14c(
                model.cbm,
                solver,
                sp_cur,
                flat_s,
                flat_e,
                packed_specs,
                y_trop,
                m_rel,
                model.q_std,
                model.alpha_unit,
            )
            r_raw = r_fixed_raw + inv_sig * sp_exc
            r_marg = jsl.solve_triangular(chol_marg, r_raw, lower=True)
            sp_pen = jnp.array(
                [
                    (u_sp[0] - t_mid_prior) / 5.0,
                    (u_sp[1] - math.log(0.25)) / 1.5,
                    (u_sp[2] - math.log(6.5)) / 1.5,
                ],
                dtype=jnp.float64,
            )
            return jnp.concatenate([r_marg, sp_pen], axis=0)

        sol = optx.least_squares(
            _varpro_spike_residual,
            optx.LevenbergMarquardt(rtol=tol, atol=tol),
            u_sp0,
            max_steps=max_steps,
            throw=False,
        )
        u_sp_opt = sol.value
        converged = bool(sol.result == optx.RESULTS.successful)
        try:
            n_steps = int(sol.stats.get("num_steps", max_steps))
        except Exception:
            n_steps = max_steps

        sp_fitted = eqx.tree_at(
            lambda s: (s.t_start, s.duration, s.fluence),
            sp_term,
            (u_sp_opt[0], jnp.exp(u_sp_opt[1]), jnp.exp(u_sp_opt[2])),
        )
        sp_exc_opt = _eval_packed_term_d14c(
            model.cbm,
            solver,
            sp_fitted,
            flat_s,
            flat_e,
            packed_specs,
            y_trop,
            m_rel,
            model.q_std,
            model.alpha_unit,
        )
        x_raw_opt = -(r_fixed_raw + inv_sig * sp_exc_opt)
        x_eff, j_eff, half_logdet_nuis = _whiten_vec_and_mat_qr(x_raw_opt, j_raw, u_nuis)
        z_hat, s_vals, _, chi2_eff = _wiener_solve_from_svd(x_eff, j_eff)
        fitted_model = (
            model.set(f"{field_path}.latent", z_hat)
            .set(f"{sp_path}.t_start", u_sp_opt[0])
            .set(f"{sp_path}.duration", jnp.exp(u_sp_opt[1]))
            .set(f"{sp_path}.fluence", jnp.exp(u_sp_opt[2]))
        )
    else:
        # Nonlinear LM over latent z (and optional MiyakeSpike parameters)
        z0 = field.latent
        if spike_entries:
            sp_path, sp_term = spike_entries[0]
            u_sp0 = jnp.array(
                [
                    sp_term.t_start,
                    jnp.log(jnp.maximum(sp_term.duration, 1e-4)),
                    jnp.log(jnp.maximum(sp_term.fluence, 1e-4)),
                ],
                dtype=jnp.float64,
            )
            y0_vec = jnp.concatenate([z0, u_sp0], axis=0)
            t_mid_prior = sp_term.t_start
            flat_s, flat_e, packed_specs = _pack_dataset_quadrature(model.cbm, ds)
            trop_idx = jnp.asarray(model.cbm.topology.trop_box_indices, dtype=jnp.int32)
            y_ss = model.cbm.steady_state_c14(q_std=model.q_std, alpha_unit=model.alpha_unit)
            m_rel = model.cbm.relative_reservoir_masses()
            y_trop = jnp.sum(y_ss[trop_idx])
        else:
            sp_path = ""
            y0_vec = z0
            t_mid_prior = jnp.asarray(0.0, dtype=jnp.float64)

        n_z = field.n_grid
        fixed_extras = tuple((p, t) for p, t in extras if p != sp_path)
        fixed_exc = _eval_extra_terms_d14c(
            model.cbm,
            solver,
            ds,
            fixed_extras,
            model.q_std,
            model.alpha_unit,
            op.steady_unit_response,
        )
        nuis_mean_shift = x_nuis @ mu_nuis if x_nuis.shape[1] else 0.0

        def _lm_residual(y_vec: jax.Array, _args: Any) -> jax.Array:
            z_cur = y_vec[:n_z]
            f_cur = field.with_latent(z_cur)
            if isinstance(f_cur, GaussianField):
                f_exc = op.w_dct @ (_safe_sqrt(f_cur.power_spectrum()) * z_cur)
            else:
                f_exc = op.w_grid @ f_cur.evaluate_grid(q_std=model.q_std)

            if spike_entries:
                u_sp = y_vec[n_z:]
                sp_cur = eqx.tree_at(
                    lambda s: (s.t_start, s.duration, s.fluence),
                    spike_entries[0][1],
                    (u_sp[0], jnp.exp(u_sp[1]), jnp.exp(u_sp[2])),
                )
                sp_exc = _eval_packed_term_d14c(
                    model.cbm,
                    solver,
                    sp_cur,
                    flat_s,
                    flat_e,
                    packed_specs,
                    y_trop,
                    m_rel,
                    model.q_std,
                    model.alpha_unit,
                )
                sp_pen = jnp.array(
                    [
                        (u_sp[0] - t_mid_prior) / 5.0,
                        (u_sp[1] - math.log(0.25)) / 1.5,
                        (u_sp[2] - math.log(6.5)) / 1.5,
                    ],
                    dtype=jnp.float64,
                )
            else:
                sp_exc = 0.0
                sp_pen = jnp.zeros((0,), dtype=jnp.float64)

            r_raw = inv_sig * (op.ss_d14c + fixed_exc + sp_exc + f_exc + nuis_mean_shift - d_obs)
            if u_nuis.shape[1] > 0:
                r_eff, _ = whiten_qr(r_raw, u_nuis)
            else:
                r_eff = r_raw
            return jnp.concatenate([r_eff, z_cur, sp_pen], axis=0)

        lm_solver = optx.LevenbergMarquardt(rtol=tol, atol=tol)
        sol = optx.least_squares(
            _lm_residual,
            lm_solver,
            y0_vec,
            max_steps=max_steps,
            throw=False,
        )
        y_opt = sol.value
        z_hat = y_opt[:n_z]
        converged = bool(sol.result == optx.RESULTS.successful)
        try:
            n_steps = int(sol.stats.get("num_steps", max_steps))
        except Exception:
            n_steps = max_steps

        fitted_model = model.set(f"{field_path}.latent", z_hat)
        if spike_entries:
            u_sp = y_opt[n_z:]
            fitted_model = fitted_model.set(
                f"{sp_path}.t_start", u_sp[0]
            ).set(
                f"{sp_path}.duration", jnp.exp(u_sp[1])
            ).set(
                f"{sp_path}.fluence", jnp.exp(u_sp[2])
            )

        # Evaluate Laplace SVD at MAP z_hat
        fitted_field, _, fitted_extras = _extract_field_and_extras(fitted_model.production)
        extra_exc = _eval_extra_terms_d14c(
            fitted_model.cbm,
            solver,
            ds,
            fitted_extras,
            fitted_model.q_std,
            fitted_model.alpha_unit,
            op.steady_unit_response,
        )

        def _r_raw_of_z(z_in: jax.Array) -> jax.Array:
            f_in = fitted_field.with_latent(z_in)
            if isinstance(f_in, GaussianField):
                fe = op.w_dct @ (_safe_sqrt(f_in.power_spectrum()) * z_in)
            else:
                fe = op.w_grid @ f_in.evaluate_grid(q_std=fitted_model.q_std)
            return inv_sig * (op.ss_d14c + extra_exc + fe + nuis_mean_shift - d_obs)

        r_raw_map = _r_raw_of_z(z_hat)
        j_raw_map = jax.jacfwd(_r_raw_of_z)(z_hat)
        r_eff_map, j_eff, half_logdet_nuis = _whiten_vec_and_mat_qr(
            r_raw_map, j_raw_map, u_nuis
        )
        s_vals = jnp.linalg.svd(j_eff, full_matrices=False, compute_uv=False)
        chi2_eff = r_eff_map @ r_eff_map
        extras = fitted_extras
        field = fitted_field

    # Reconstruct conditional posterior mean of hierarchical offsets given z_hat
    fitted_field, _, fitted_extras = _extract_field_and_extras(fitted_model.production)
    extra_exc_final = _eval_extra_terms_d14c(
        fitted_model.cbm,
        solver,
        ds,
        fitted_extras,
        fitted_model.q_std,
        fitted_model.alpha_unit,
        op.steady_unit_response,
    )
    if isinstance(fitted_field, GaussianField):
        f_grid = fitted_field.evaluate_grid()
        q_grid = fitted_model.q_std + f_grid
        f_exc_final = op.w_dct @ (_safe_sqrt(fitted_field.power_spectrum()) * z_hat)
    else:
        q_grid = fitted_field.evaluate_q_grid()
        f_grid = q_grid - fitted_model.q_std
        f_exc_final = op.w_grid @ f_grid

    pred_no_off = op.ss_d14c + extra_exc_final + f_exc_final
    if u_nuis.shape[1] > 0:
        resid_for_nuis = inv_sig * (d_obs - pred_no_off - x_nuis @ mu_nuis)
        w_std, _ = _linear_posterior_std(resid_for_nuis, u_nuis)
        w_nuis = mu_nuis + root_nuis @ w_std
        if offset_mode == "hierarchical":
            n_off = 1 + (ds.n_series if ds.n_series > 1 else 0)
            w_off = w_nuis[:n_off]
            g_off = w_off[0]
            l_off = (
                w_off[1:]
                if ds.n_series > 1
                else jnp.zeros((1,), dtype=jnp.float64)
            )
            base_d14c = x_nuis[:, :n_off] @ w_off
        else:
            g_off = jnp.asarray(0.0, dtype=jnp.float64)
            l_off = jnp.zeros((ds.n_series,), dtype=jnp.float64)
            base_d14c = jnp.zeros((n_obs,), dtype=jnp.float64)
    else:
        g_off = jnp.asarray(0.0, dtype=jnp.float64)
        l_off = jnp.zeros((ds.n_series,), dtype=jnp.float64)
        base_d14c = jnp.zeros((n_obs,), dtype=jnp.float64)

    d14c_pred = pred_no_off + base_d14c
    z_norm_sq = z_hat @ z_hat
    half_logdet_field = 0.5 * jnp.sum(jnp.log1p(jnp.square(s_vals)))
    ln_z = (
        -0.5 * chi2_eff
        - 0.5 * z_norm_sq
        - log_norm_d
        - half_logdet_nuis
        - half_logdet_field
    )
    gamma = jnp.sum(jnp.square(s_vals) / (1.0 + jnp.square(s_vals)))

    return IFTFieldFitResult(
        model=fitted_model,
        latent=z_hat,
        field_grid=f_grid,
        q_grid=q_grid,
        t_grid=op.t_grid,
        d14c_pred=d14c_pred,
        baseline_d14c=base_d14c,
        global_offset=g_off,
        lab_offsets=l_off,
        chi2=chi2_eff,
        latent_norm_sq=z_norm_sq,
        log_evidence=ln_z,
        gamma=gamma,
        operator=op,
        offset_mode=offset_mode,
        global_offset_prior_sd=float(global_offset_prior_sd),
        lab_offset_prior_sd=float(lab_offset_prior_sd),
        converged=converged,
        n_steps=n_steps,
    )


fit_field = fit_ift_field


def _smooth_logdet_and_gamma(j_eff: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return ``(0.5 * ln det(I + J^T J), tr(J^T J (I + J^T J)^{-1}))`` smoothly via Cholesky."""
    n_obs, n_lat = j_eff.shape
    if n_obs <= n_lat:
        eye = jnp.eye(n_obs, dtype=jnp.float64)
        chol = jnp.linalg.cholesky(eye + j_eff @ j_eff.T)
        half_logdet = jnp.sum(jnp.log(jnp.diag(chol)))
        l_inv = jsl.solve_triangular(chol, eye, lower=True)
        gamma = float(n_obs) - jnp.sum(jnp.square(l_inv))
        return half_logdet, gamma
    eye = jnp.eye(n_lat, dtype=jnp.float64)
    chol = jnp.linalg.cholesky(eye + j_eff.T @ j_eff)
    half_logdet = jnp.sum(jnp.log(jnp.diag(chol)))
    l_inv = jsl.solve_triangular(chol, eye, lower=True)
    gamma = float(n_lat) - jnp.sum(jnp.square(l_inv))
    return half_logdet, gamma


def _exact_linear_evidence_core(
    x_eff: jax.Array,
    j_eff: jax.Array,
    log_norm_d: jax.Array,
    half_logdet_nuis: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Exact Gaussian marginal ``(log_evidence, gamma)`` with ``C^inf`` Cholesky autodiff."""
    n_obs, n_lat = j_eff.shape
    if n_obs <= n_lat:
        eye = jnp.eye(n_obs, dtype=jnp.float64)
        chol = jnp.linalg.cholesky(eye + j_eff @ j_eff.T)
        v = jsl.solve_triangular(chol, x_eff, lower=True)
        quad = v @ v
        half_logdet = jnp.sum(jnp.log(jnp.diag(chol)))
        l_inv = jsl.solve_triangular(chol, eye, lower=True)
        gamma = float(n_obs) - jnp.sum(jnp.square(l_inv))
    else:
        eye = jnp.eye(n_lat, dtype=jnp.float64)
        chol = jnp.linalg.cholesky(eye + j_eff.T @ j_eff)
        w = jsl.solve_triangular(chol, j_eff.T @ x_eff, lower=True)
        quad = jnp.maximum((x_eff @ x_eff) - (w @ w), 0.0)
        half_logdet = jnp.sum(jnp.log(jnp.diag(chol)))
        l_inv = jsl.solve_triangular(chol, eye, lower=True)
        gamma = float(n_lat) - jnp.sum(jnp.square(l_inv))
    ln_z = -0.5 * quad - log_norm_d - half_logdet_nuis - half_logdet
    return ln_z, gamma


def log_evidence(
    model_or_result: RadiocarbonModel | IFTFieldFitResult,
    dataset: Dataset | TreeRingSeries | None = None,
    *,
    offset_mode: Literal["hierarchical", "none"] | None = None,
    global_offset_prior_sd: float = 200.0,
    lab_offset_prior_sd: float = 1.5,
    extra_priors: Mapping[str, Any] | None = None,
    operator: DCTResponseOperator | None = None,
    prehistory_kernel: Any | None = None,
) -> jax.Array:
    r"""MacKay / ``virgil`` log-evidence ``ln Z`` of a :class:`GaussianField` or :class:`LogGaussianField`.

    Evaluates

    .. math::

        \ln Z = -\frac{1}{2}\chi_{\mathrm{eff}}^2
        - \frac{1}{2}\sum_{i=1}^{N_d}\ln(2\pi\sigma_i^2)
        - \frac{1}{2}\|z\|^2
        - \frac{1}{2}\ln\det(I + U_{\mathrm{off}}^\top U_{\mathrm{off}})
        - \frac{1}{2}\ln\det(I + J_{\mathrm{eff}}^\top J_{\mathrm{eff}}),

    which is **exact** for linear :class:`GaussianField` models (independent of initial
    ``latent`` values) and equals the Laplace approximation around the MAP ``model`` for
    nonlinear :class:`LogGaussianField` models. Optional ``extra_priors`` ``{path: dist}``
    enter via their ``log_prob`` and Schur-complement curvature block, mirroring
    ``virgil.imaging.log_evidence``.
    """
    if isinstance(model_or_result, IFTFieldFitResult):
        if dataset is None and not extra_priors and offset_mode is None and prehistory_kernel is None:
            return model_or_result.log_evidence
        model = model_or_result.model
        eff_offset_mode: Literal["hierarchical", "none"] = (
            model_or_result.offset_mode  # type: ignore[assignment]
            if offset_mode is None
            else offset_mode
        )
        eff_op = model_or_result.operator if operator is None else operator
        global_offset_prior_sd = model_or_result.global_offset_prior_sd
        lab_offset_prior_sd = model_or_result.lab_offset_prior_sd
    elif isinstance(model_or_result, RadiocarbonModel):
        model = model_or_result
        eff_offset_mode = "hierarchical" if offset_mode is None else offset_mode
        eff_op = operator
    else:
        raise TypeError(
            f"Expected RadiocarbonModel or IFTFieldFitResult, got {type(model_or_result)!r}."
        )

    if dataset is None:
        raise ValueError("dataset must be provided when evaluating log_evidence from a model.")
    ds = _as_dataset(dataset)
    field, field_path, extras = _extract_field_and_extras(model.production)
    tmpl = field.template if isinstance(field, GaussianField) else None
    op = (
        build_dct_response_operator(
            model.cbm,
            ds,
            t_start=field.t_start,
            t_end=field.t_end,
            n_grid=field.n_grid,
            q_std=model.q_std,
            alpha_unit=model.alpha_unit,
            template=tmpl,
        )
        if eff_op is None
        else eff_op
    )
    pre_cov = (
        stationary_cbm_gp_cov(
            model.cbm.matrix_A(),
            model.alpha_unit * model.cbm.production_vector(),
            prehistory_kernel,
        )
        if prehistory_kernel is not None
        else None
    )
    x_nuis, mu_nuis, _, u_nuis = _build_offset_whitening(
        ds,
        op.box_activity_ratios,
        offset_mode=eff_offset_mode,
        global_offset_prior_sd=global_offset_prior_sd,
        lab_offset_prior_sd=lab_offset_prior_sd,
        prehistory_cov_y0=pre_cov,
        init_state_basis=op.init_state_basis,
    )

    solver = (
        model.solver
        if isinstance(model.solver, ExactLinearSolver)
        else ExactLinearSolver()
    )
    inv_sig = 1.0 / ds.sigma_flat
    d_obs = ds.d14c_flat
    log_norm_d = 0.5 * ds.n_obs * math.log(2.0 * math.pi) + jnp.sum(jnp.log(ds.sigma_flat))
    nuis_shift = x_nuis @ mu_nuis if x_nuis.shape[1] else 0.0

    extra_paths = tuple(
        p for p in (extra_priors or ()) if p != f"{field_path}.latent"
    )

    if isinstance(field, GaussianField) and not extra_paths:
        extra_exc = _eval_extra_terms_d14c(
            model.cbm, solver, ds, extras, model.q_std, model.alpha_unit, op.steady_unit_response
        )
        x_raw = inv_sig * (d_obs - op.ss_d14c - extra_exc - nuis_shift)
        j_raw = inv_sig[:, None] * (op.w_dct * _safe_sqrt(field.power_spectrum())[None, :])
        x_eff, j_eff, half_logdet_nuis = _whiten_vec_and_mat_qr(x_raw, j_raw, u_nuis)
        ln_z, _ = _exact_linear_evidence_core(x_eff, j_eff, log_norm_d, half_logdet_nuis)
        return ln_z

    # Nonlinear LogGaussianField (or extra_priors Schur complement at MAP)
    extra_vals = [jnp.asarray(model.get(p), dtype=jnp.float64).reshape(()) for p in extra_paths]
    extra_vec0 = (
        jnp.stack(extra_vals, axis=0)
        if extra_vals
        else jnp.zeros((0,), dtype=jnp.float64)
    )

    def _r_raw_joint(z_in: jax.Array, e_in: jax.Array) -> jax.Array:
        m_cur = model.set(f"{field_path}.latent", z_in)
        for idx, p in enumerate(extra_paths):
            m_cur = m_cur.set(p, e_in[idx])
        f_cur, _, ext_cur = _extract_field_and_extras(m_cur.production)
        ext_exc = _eval_extra_terms_d14c(
            m_cur.cbm, solver, ds, ext_cur, m_cur.q_std, m_cur.alpha_unit, op.steady_unit_response
        )
        if isinstance(f_cur, GaussianField):
            f_exc = op.w_dct @ (_safe_sqrt(f_cur.power_spectrum()) * z_in)
        else:
            f_exc = op.w_grid @ f_cur.evaluate_grid(q_std=m_cur.q_std)
        return inv_sig * (op.ss_d14c + ext_exc + f_exc + nuis_shift - d_obs)

    if isinstance(field, GaussianField):
        # Solve exact conditional Wiener mode z_hat at extra_vec0 so log_evidence(..., extra_priors=...)
        # is strictly invariant to any initial field.latent values.
        z_zero = jnp.zeros_like(field.latent)
        r_raw0 = _r_raw_joint(z_zero, extra_vec0)
        m0 = model
        for idx, p in enumerate(extra_paths):
            m0 = m0.set(p, extra_vec0[idx])
        f0, _, _ = _extract_field_and_extras(m0.production)
        j_z0 = inv_sig[:, None] * (op.w_dct * _safe_sqrt(f0.power_spectrum())[None, :])
        r_eff0, j_eff0, _ = _whiten_vec_and_mat_qr(r_raw0, j_z0, u_nuis)
        z_val, _, _, _ = _wiener_solve_from_svd(-r_eff0, j_eff0)
    else:
        z_val = field.latent

    r_raw = _r_raw_joint(z_val, extra_vec0)
    j_z_raw, j_ext_raw = jax.jacfwd(_r_raw_joint, argnums=(0, 1))(z_val, extra_vec0)
    j_all_raw = jnp.concatenate([j_z_raw, j_ext_raw], axis=1)
    r_eff, j_all_eff, half_logdet_nuis = _whiten_vec_and_mat_qr(r_raw, j_all_raw, u_nuis)
    n_z = z_val.shape[0]
    j_eff = j_all_eff[:, :n_z]
    half_logdet_field, _ = _smooth_logdet_and_gamma(j_eff)

    chi2_eff = r_eff @ r_eff
    z2 = z_val @ z_val
    base_ln_z = (
        -0.5 * chi2_eff
        - 0.5 * z2
        - log_norm_d
        - half_logdet_nuis
        - half_logdet_field
    )
    if not extra_paths:
        return base_ln_z

    assert extra_priors is not None
    j_ext = j_all_eff[:, n_z:]
    _, s_vals, vt_svd = jnp.linalg.svd(j_eff, full_matrices=False)
    c_mat = vt_svd @ (j_eff.T @ j_ext)
    schur = j_ext.T @ j_ext - (c_mat.T / (1.0 + jnp.square(s_vals))) @ c_mat
    # Include negative Hessian of extra log-priors if differentiable
    def _log_p_extra(ev: jax.Array) -> jax.Array:
        acc = jnp.asarray(0.0, dtype=jnp.float64)
        for idx, p in enumerate(extra_paths):
            acc = acc + jnp.sum(extra_priors[p].log_prob(ev[idx]))
        return acc

    lp_val = _log_p_extra(extra_vec0)
    neg_h_lp = -jax.hessian(_log_p_extra)(extra_vec0)
    h_ext = 0.5 * (schur + schur.T + neg_h_lp + neg_h_lp.T)
    sign, logdet_ext = jnp.linalg.slogdet(h_ext)
    extra_term = jnp.where(
        sign > 0.0,
        lp_val + 0.5 * len(extra_paths) * math.log(2.0 * math.pi) - 0.5 * logdet_ext,
        -jnp.inf,
    )
    return base_ln_z + extra_term


def laplace_samples(
    model_or_result: RadiocarbonModel | IFTFieldFitResult,
    dataset: Dataset | TreeRingSeries | None = None,
    *,
    n_samples: int = 200,
    rng_key: jax.Array | None = None,
    offset_mode: Literal["hierarchical", "none"] | None = None,
    global_offset_prior_sd: float = 200.0,
    lab_offset_prior_sd: float = 1.5,
) -> dict[str, jax.Array]:
    r"""Draw ``z^{(m)} = z_hat + (I + J^T J)^{-1/2} eps^{(m)}`` via economy SVD.

    Direct radiocarbon counterpart of ``virgil.imaging.laplace_samples``: decomposes
    posterior fluctuations into data-informed singular directions
    ``V diag((1 + s^2)^{-1/2}) eps_1`` and prior-dominated null-space directions
    ``(I - V V^T) eps_2`` without ever forming an ``N x N`` covariance matrix.

    * **Exact** Wiener posterior samples for linear :class:`GaussianField` models.
    * **Laplace / MGVI** local Gaussian posterior samples for :class:`LogGaussianField`.

    Returns a dictionary containing ``"latents"`` ``(n_samples, n_grid)``,
    ``"delta_q"`` ``(n_samples, n_grid)``, ``"q"`` ``(n_samples, n_grid)``, and
    ``"t_grid"`` ``(n_grid,)``.
    """
    n_s = int(n_samples)
    if n_s < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}.")
    key = jax.random.PRNGKey(0) if rng_key is None else rng_key

    if isinstance(model_or_result, IFTFieldFitResult):
        fit_res = model_or_result
        model = fit_res.model
        op = fit_res.operator
        eff_offset_mode: Literal["hierarchical", "none"] = (
            fit_res.offset_mode  # type: ignore[assignment]
            if offset_mode is None
            else offset_mode
        )
        global_offset_prior_sd = fit_res.global_offset_prior_sd
        lab_offset_prior_sd = fit_res.lab_offset_prior_sd
        if dataset is None:
            raise ValueError(
                "dataset is required by laplace_samples to evaluate measurement noise weights."
            )
        ds = _as_dataset(dataset)
    elif isinstance(model_or_result, RadiocarbonModel):
        if dataset is None:
            raise ValueError("dataset must be provided when sampling from a RadiocarbonModel.")
        ds = _as_dataset(dataset)
        eff_offset_mode = "hierarchical" if offset_mode is None else offset_mode
        field0, _, _ = _extract_field_and_extras(model_or_result.production)
        if isinstance(field0, GaussianField):
            fit_res = fit_ift_field(
                model_or_result,
                ds,
                offset_mode=eff_offset_mode,
                global_offset_prior_sd=global_offset_prior_sd,
                lab_offset_prior_sd=lab_offset_prior_sd,
            )
            model = fit_res.model
            op = fit_res.operator
        else:
            model = model_or_result
            op = build_dct_response_operator(
                model.cbm,
                ds,
                t_start=field0.t_start,
                t_end=field0.t_end,
                n_grid=field0.n_grid,
                q_std=model.q_std,
                alpha_unit=model.alpha_unit,
            )
    else:
        raise TypeError(
            f"Expected RadiocarbonModel or IFTFieldFitResult, got {type(model_or_result)!r}."
        )

    field, _, extras = _extract_field_and_extras(model.production)
    _, _, _, u_nuis = _build_offset_whitening(
        ds,
        op.box_activity_ratios,
        offset_mode=eff_offset_mode,
        global_offset_prior_sd=global_offset_prior_sd,
        lab_offset_prior_sd=lab_offset_prior_sd,
    )
    inv_sig = 1.0 / ds.sigma_flat
    z_hat = field.latent
    n_z = field.n_grid

    if isinstance(field, GaussianField):
        j_raw = inv_sig[:, None] * (op.w_dct * _safe_sqrt(field.power_spectrum())[None, :])
    else:
        solver = (
            model.solver
            if isinstance(model.solver, ExactLinearSolver)
            else ExactLinearSolver()
        )
        extra_exc = _eval_extra_terms_d14c(
            model.cbm, solver, ds, extras, model.q_std, model.alpha_unit, op.steady_unit_response
        )
        j_raw = jax.jacfwd(
            lambda z_in: inv_sig
            * (
                op.ss_d14c
                + extra_exc
                + op.w_grid @ field.with_latent(z_in).evaluate_grid(q_std=model.q_std)
                - ds.d14c_flat
            )
        )(z_hat)

    _, j_eff, _ = _whiten_vec_and_mat_qr(jnp.zeros((ds.n_obs,), dtype=jnp.float64), j_raw, u_nuis)
    _, s_vals, vt_svd = jnp.linalg.svd(j_eff, full_matrices=False)

    k1, k2 = jax.random.split(key)
    e1 = jax.random.normal(k1, (n_s, s_vals.shape[0]), dtype=jnp.float64)
    e2 = jax.random.normal(k2, (n_s, n_z), dtype=jnp.float64)
    fluct = (e1 / jnp.sqrt(1.0 + jnp.square(s_vals))[None, :]) @ vt_svd
    fluct = fluct + (e2 - (e2 @ vt_svd.T) @ vt_svd)
    z_draws = z_hat[None, :] + fluct

    if isinstance(field, GaussianField):
        dq_draws = jax.vmap(lambda z: field.with_latent(z).evaluate_grid())(z_draws)
        q_draws = model.q_std + dq_draws
    else:
        q_draws = jax.vmap(lambda z: field.with_latent(z).evaluate_q_grid())(z_draws)
        dq_draws = q_draws - model.q_std

    return {
        "latents": z_draws,
        "delta_q": dq_draws,
        "q": q_draws,
        "t_grid": op.t_grid,
    }


def error_scale(
    model_or_result: RadiocarbonModel | IFTFieldFitResult,
    dataset: Dataset | TreeRingSeries | None = None,
    *,
    by_series: bool = False,
    offset_mode: Literal["hierarchical", "none"] | None = None,
    max_iter: int = 200,
    rtol: float = 1e-10,
) -> float | dict[str, float]:
    r"""Re-estimate noise error-bar scaling ``s`` via MacKay's (1992) evidence fixed point.

    Radiocarbon port of ``virgil.imaging.error_scale`` solving the empirical-Bayes
    stationary condition ``s^2 = chi^2(s) / (N_eff - gamma(s))`` (with precision
    ``beta = 1 / s^2`` and ``gamma(s) = tr(B^{1/2} J (I + J^T B J)^{-1} J^T B^{1/2})``)
    on the Gauss–Newton linearized data vector ``x_lin = r_data(z_hat) + J_z z_hat``
    projected orthogonal to unpenalized per-series baseline offsets.

    * ``by_series=False`` (default): returns a single global ``float`` ``s``
      (``s < 1`` means quoted error bars are conservative/overstated; after rescaling
      with ``dataset.with_error_scale(s)`` and refitting, ``error_scale`` yields ``1.0``).
    * ``by_series=True``: returns ``dict[series.name, s_b]`` via monotone block
      fixed-point iteration ``beta_b <- N_{b,eff} / (chi_b^2(beta) + gamma_b(beta)/beta_b)``.
    """
    if dataset is None:
        raise ValueError("dataset must be provided to error_scale.")
    ds = _as_dataset(dataset)

    if isinstance(model_or_result, IFTFieldFitResult):
        fit_res = model_or_result
        model = fit_res.model
        op = fit_res.operator
        eff_offset_mode = (
            fit_res.offset_mode if offset_mode is None else offset_mode
        )
    elif isinstance(model_or_result, RadiocarbonModel):
        eff_offset_mode = "hierarchical" if offset_mode is None else offset_mode
        fit_res = fit_ift_field(model_or_result, ds, offset_mode=eff_offset_mode)  # type: ignore[arg-type]
        model = fit_res.model
        op = fit_res.operator
    else:
        raise TypeError(
            f"Expected RadiocarbonModel or IFTFieldFitResult, got {type(model_or_result)!r}."
        )

    field, _, extras = _extract_field_and_extras(model.production)
    solver = (
        model.solver
        if isinstance(model.solver, ExactLinearSolver)
        else ExactLinearSolver()
    )
    extra_exc = _eval_extra_terms_d14c(
        model.cbm, solver, ds, extras, model.q_std, model.alpha_unit, op.steady_unit_response
    )
    inv_sig = np.asarray(1.0 / ds.sigma_flat, dtype=np.float64)
    d_obs = np.asarray(ds.d14c_flat, dtype=np.float64)
    z_hat = np.asarray(field.latent, dtype=np.float64)

    if isinstance(field, GaussianField):
        x_raw = inv_sig * (d_obs - np.asarray(op.ss_d14c + extra_exc, dtype=np.float64))
        j_raw = inv_sig[:, None] * np.asarray(
            op.w_dct * _safe_sqrt(field.power_spectrum())[None, :], dtype=np.float64
        )
    else:
        f_exc_hat = np.asarray(
            op.w_grid @ field.evaluate_grid(q_std=model.q_std), dtype=np.float64
        )
        r_data = inv_sig * (
            d_obs - np.asarray(op.ss_d14c + extra_exc, dtype=np.float64) - f_exc_hat
        )
        j_raw = np.asarray(
            jax.jacfwd(
                lambda z_in: inv_sig[:, None][:, 0]
                * (op.w_grid @ field.with_latent(z_in).evaluate_grid(q_std=model.q_std))
            )(field.latent),
            dtype=np.float64,
        )
        x_raw = r_data + j_raw @ z_hat

    # Project out per-series constant offsets within each series block (indexed by integer
    # series index 0 .. n_series - 1 so duplicate series.name never drops rows).
    slices_by_idx: list[np.ndarray] = []
    counts_by_idx: list[int] = []
    x_proj = x_raw.copy()
    j_proj = j_raw.copy()
    cursor = 0
    for s in ds.series:
        sl = np.arange(cursor, cursor + s.n_obs, dtype=np.int64)
        slices_by_idx.append(sl)
        if eff_offset_mode == "hierarchical":
            if s.n_obs <= 1:
                raise ValueError(
                    f"Series {s.name!r} has only {s.n_obs} observation; cannot estimate "
                    "error_scale while also projecting a per-series offset."
                )
            w_s = inv_sig[sl] * np.asarray(op.box_activity_ratios[sl], dtype=np.float64)
            q_s = w_s / np.linalg.norm(w_s)
            x_proj[sl] = x_proj[sl] - q_s * float(q_s @ x_proj[sl])
            j_proj[sl] = j_proj[sl] - np.outer(q_s, q_s @ j_proj[sl])
            counts_by_idx.append(s.n_obs - 1)
        else:
            counts_by_idx.append(s.n_obs)
        cursor += s.n_obs

    if not by_series:
        n_eff = float(sum(counts_by_idx))
        u_svd, s_vals, _ = np.linalg.svd(j_proj, full_matrices=False)
        lam = s_vals**2
        proj_sq = (u_svd.T @ x_proj) ** 2
        perp_sq = max(float(x_proj @ x_proj) - float(np.sum(proj_sq)), 0.0)
        total_sq = float(x_proj @ x_proj)
        if total_sq <= 0.0:
            raise ValueError("error_scale: exact zero residual vector; noise scale undefined.")

        # Solve f(beta) = sum_k [beta * proj_sq_k / (1 + beta * lam_k)^2 + beta * lam_k / (1 + beta * lam_k)]
        #                 + beta * perp_sq - n_eff = 0 via safeguarded log-beta Newton/bisection.
        def _f_and_df_log_beta(lb: float) -> tuple[float, float]:
            b = math.exp(lb)
            denom = 1.0 + b * lam
            chi2_b = float(np.sum(proj_sq / (denom**2))) + perp_sq
            gamma_b = float(np.sum((b * lam) / denom))
            val = b * chi2_b + gamma_b - n_eff
            # d/d(ln b) [b * chi2(b) + gamma(b)]
            d_b_chi2 = float(np.sum(proj_sq * (1.0 - b * lam) / (denom**3))) + perp_sq
            d_gamma = float(np.sum((b * lam) / (denom**2)))
            deriv = b * d_b_chi2 + d_gamma
            return val, deriv

        lo, hi = -30.0, 30.0
        while _f_and_df_log_beta(lo)[0] > 0.0 and lo > -80.0:
            lo -= 10.0
        while _f_and_df_log_beta(hi)[0] < 0.0 and hi < 80.0:
            hi += 10.0

        lb = math.log(max(n_eff / max(total_sq, 1e-30), 1e-12))
        lb = min(max(lb, lo), hi)
        for _ in range(max_iter):
            val, deriv = _f_and_df_log_beta(lb)
            if abs(val) <= rtol * max(n_eff, 1.0):
                break
            if val < 0.0:
                lo = max(lo, lb)
            else:
                hi = min(hi, lb)
            if deriv > 1e-14:
                cand = lb - val / deriv
                if lo < cand < hi:
                    step = abs(cand - lb)
                    lb = cand
                else:
                    cand = 0.5 * (lo + hi)
                    step = abs(cand - lb)
                    lb = cand
            else:
                cand = 0.5 * (lo + hi)
                step = abs(cand - lb)
                lb = cand
            if step <= rtol:
                break
        return float(math.exp(-0.5 * lb))

    # Per-series block re-estimation: beta_b <- N_{b,eff} / (chi2_b(beta) + tr_b(beta))
    if len(set(ds.series_names)) != ds.n_series:
        raise ValueError(
            "error_scale(by_series=True) requires unique TreeRingSeries names in Dataset; "
            f"got {ds.series_names}."
        )
    n_vec = np.asarray(counts_by_idx, dtype=np.float64)
    init_sq = np.array(
        [max(float(x_proj[sl] @ x_proj[sl]), 1e-24) for sl in slices_by_idx],
        dtype=np.float64,
    )
    beta = np.clip(n_vec / init_sq, 1e-6, 1e6)
    n_rows, n_lat = j_proj.shape
    eye_lat = np.eye(n_lat, dtype=np.float64)

    for _ in range(max_iter):
        row_b = np.empty(n_rows, dtype=np.float64)
        for idx_k, b_val in enumerate(beta):
            row_b[slices_by_idx[idx_k]] = b_val
        sqrt_b = np.sqrt(row_b)
        j_scaled = sqrt_b[:, None] * j_proj
        x_scaled = sqrt_b * x_proj

        # QR of [j_scaled; I] avoids forming J^T B J directly
        r_fac = np.linalg.qr(np.vstack([j_scaled, eye_lat]), mode="r")
        rhs = j_scaled.T @ x_scaled
        # Solve (R^T R) z_b = rhs in C/LAPACK
        tmp = _scipy_solve_triangular(r_fac, rhs, trans="T", lower=False)
        z_b = _scipy_solve_triangular(r_fac, tmp, lower=False)
        r_unscaled = x_proj - j_proj @ z_b
        v_all = _scipy_solve_triangular(r_fac, j_proj.T, trans="T", lower=False)
        col_tr = np.sum(v_all**2, axis=0)

        new_beta = np.empty_like(beta)
        for idx_k, sl in enumerate(slices_by_idx):
            chi2_k = float(r_unscaled[sl] @ r_unscaled[sl])
            tr_k = float(np.sum(col_tr[sl]))
            denom_k = max(chi2_k + tr_k, 1e-30)
            new_beta[idx_k] = n_vec[idx_k] / denom_k

        rel_change = float(np.max(np.abs(new_beta - beta) / np.maximum(new_beta, 1e-30)))
        beta = new_beta
        if rel_change < rtol:
            break
    else:
        warnings.warn(
            f"error_scale(by_series=True) did not converge in {max_iter} iterations "
            f"(max relative change {rel_change:.2e}).",
            RuntimeWarning,
            stacklevel=2,
        )

    return {
        ds.series[idx_k].name: float(1.0 / math.sqrt(float(b_val)))
        for idx_k, b_val in enumerate(beta)
    }


def apply_error_scale(
    dataset_or_series: Dataset | TreeRingSeries,
    scale: float | Mapping[str, float],
) -> Dataset | TreeRingSeries:
    """Return a copy of ``dataset_or_series`` with ``sigma_d14c`` multiplied by ``scale``.

    Delegates to ``.with_error_scale(scale)`` when available on :class:`Dataset` /
    :class:`TreeRingSeries`, and provides a standalone ``zodiax`` PyTree fallback.
    """
    if hasattr(dataset_or_series, "with_error_scale"):
        return dataset_or_series.with_error_scale(scale)  # type: ignore[no-any-return]

    if isinstance(dataset_or_series, TreeRingSeries):
        if isinstance(scale, Mapping):
            if dataset_or_series.name not in scale:
                raise KeyError(
                    f"Series {dataset_or_series.name!r} missing from scale mapping {list(scale)}."
                )
            s_val = float(scale[dataset_or_series.name])
        else:
            s_val = float(scale)
        if not (math.isfinite(s_val) and s_val > 0.0):
            raise ValueError(f"Error scale must be finite and positive, got {s_val}.")
        return eqx.tree_at(
            lambda s: s.sigma_d14c,
            dataset_or_series,
            dataset_or_series.sigma_d14c * s_val,
        )

    if isinstance(dataset_or_series, Dataset):
        if isinstance(scale, Mapping):
            if len(set(dataset_or_series.series_names)) != dataset_or_series.n_series:
                raise ValueError(
                    "Per-series error scaling via a Mapping requires unique TreeRingSeries "
                    f"names in Dataset; got {dataset_or_series.series_names}."
                )
            unknown = set(scale) - set(dataset_or_series.series_names)
            if unknown:
                raise KeyError(
                    f"Unknown series names {sorted(unknown)} in scale mapping; "
                    f"dataset has {dataset_or_series.series_names}."
                )
            new_series = tuple(
                apply_error_scale(s, scale[s.name])  # type: ignore[misc]
                if s.name in scale
                else s
                for s in dataset_or_series.series
            )
        else:
            new_series = tuple(
                apply_error_scale(s, float(scale))  # type: ignore[misc]
                for s in dataset_or_series.series
            )
        return Dataset(new_series, name=dataset_or_series.name)

    raise TypeError(
        f"Expected Dataset or TreeRingSeries, got {type(dataset_or_series)!r}."
    )


# =============================================================================
# 5. Empirical-Bayes Evidence Grids, Kernel Optimization, & Model Selection
# =============================================================================


class EvidenceGridResult(zdx.Base):
    """2D ``(sigma, length_yr)`` marginal log-evidence surface from :func:`evidence_grid`."""

    sigmas: jax.Array
    lengths_yr: jax.Array
    log_evidence: jax.Array
    best_sigma: jax.Array
    best_length_yr: jax.Array
    best_log_evidence: jax.Array
    best_fit: IFTFieldFitResult


def evidence_grid(
    cbm: CarbonBoxModel,
    dataset: Dataset | TreeRingSeries,
    sigmas: ArrayLike,
    lengths_yr: ArrayLike,
    *,
    order: float = 2.0,
    field_kind: Literal["gaussian", "log_gaussian"] = "gaussian",
    extra_production: ProductionTerm | None = None,
    n_grid: int = 128,
    pad_years: float = 15.0,
    offset_mode: Literal["hierarchical", "none"] = "hierarchical",
    zero_mode: bool | None = None,
    q_std: float = DEFAULT_Q_STD,
) -> EvidenceGridResult:
    """Evaluate the 2D ``(sigma, length_yr)`` log-evidence surface (mirroring ``virgil``).

    Precomputes :class:`DCTResponseOperator` once and sweeps all ``(len(sigmas), len(lengths_yr))``
    grid points via ``jax.vmap`` in milliseconds for ``field_kind="gaussian"`` (or warm-started
    Levenberg–Marquardt fits for ``field_kind="log_gaussian"``).
    """
    ds = _as_dataset(dataset)
    sig_arr = jnp.asarray(sigmas, dtype=jnp.float64).reshape(-1)
    len_arr = jnp.asarray(lengths_yr, dtype=jnp.float64).reshape(-1)
    if sig_arr.size < 1 or len_arr.size < 1:
        raise ValueError("sigmas and lengths_yr must be non-empty 1D sequences.")
    _check_positive_concrete("sigmas", sig_arr)
    _check_positive_concrete("lengths_yr", len_arr)

    op = build_dct_response_operator(
        cbm, ds, n_grid=n_grid, pad_years=pad_years, q_std=q_std
    )
    solver = ExactLinearSolver()
    extras = (
        tuple(
            (f"production.{n}", p)
            for n, p in zip(extra_production.names, extra_production.parts)
        )
        if isinstance(extra_production, ProductionSum)
        else ((("production.extra", extra_production),) if extra_production is not None else ())
    )
    extra_exc = _eval_extra_terms_d14c(
        cbm, solver, ds, extras, q_std, ALPHA_ATOMS_CM2_S_TO_KG_YR, op.steady_unit_response
    )
    x_nuis, mu_nuis, _, u_nuis = _build_offset_whitening(
        ds,
        op.box_activity_ratios,
        offset_mode=offset_mode,
        global_offset_prior_sd=200.0,
        lab_offset_prior_sd=1.5,
    )
    inv_sig = 1.0 / ds.sigma_flat
    log_norm_d = 0.5 * ds.n_obs * math.log(2.0 * math.pi) + jnp.sum(jnp.log(ds.sigma_flat))
    nuis_shift = x_nuis @ mu_nuis if x_nuis.shape[1] else 0.0

    if field_kind == "gaussian":
        zm = False if zero_mode is None else bool(zero_mode)
        x_raw = inv_sig * (ds.d14c_flat - op.ss_d14c - extra_exc - nuis_shift)
        w_dct_scaled = inv_sig[:, None] * op.w_dct
        x_eff, w_dct_eff, half_logdet_nuis = _whiten_vec_and_mat_qr(
            x_raw, w_dct_scaled, u_nuis
        )

        @jax.jit
        def _eval_grid_linear(s_vec: jax.Array, l_vec: jax.Array) -> jax.Array:
            def _one(s_val: jax.Array, l_val: jax.Array) -> jax.Array:
                spec = field_spectrum(op.n_grid, op.dt, s_val, l_val, order, zero_mode=zm)
                j_eff = w_dct_eff * _safe_sqrt(spec)[None, :]
                ln_z, _ = _exact_linear_evidence_core(
                    x_eff, j_eff, log_norm_d, half_logdet_nuis
                )
                return ln_z

            return jax.vmap(lambda s_v: jax.vmap(lambda l_v: _one(s_v, l_v))(l_vec))(s_vec)

        grid_lnz = _eval_grid_linear(sig_arr, len_arr)
    elif field_kind == "log_gaussian":
        zm = True if zero_mode is None else bool(zero_mode)
        rows_out: list[jax.Array] = []
        z_warm = jnp.zeros((op.n_grid,), dtype=jnp.float64)
        for s_val in np.asarray(sig_arr, dtype=np.float64):
            row_vals: list[jax.Array] = []
            for l_val in np.asarray(len_arr, dtype=np.float64):
                lgf = LogGaussianField(
                    z_warm,
                    op.t_start,
                    op.t_end,
                    sigma=s_val,
                    length_yr=l_val,
                    order=order,
                    q_mean=q_std,
                    zero_mode=zm,
                )
                prod = lgf if extra_production is None else (extra_production + lgf)
                mdl = RadiocarbonModel(cbm, prod, solver, q_std=q_std)
                res_ij = fit_ift_field(mdl, ds, offset_mode=offset_mode, operator=op, max_steps=35)
                z_warm = res_ij.latent
                row_vals.append(res_ij.log_evidence)
            rows_out.append(jnp.stack(row_vals, axis=0))
        grid_lnz = jnp.stack(rows_out, axis=0)
    else:
        raise ValueError(
            f"field_kind must be 'gaussian' or 'log_gaussian', got {field_kind!r}."
        )

    flat_idx = int(jnp.argmax(grid_lnz))
    i_best, j_best = divmod(flat_idx, int(len_arr.shape[0]))
    best_sig = sig_arr[i_best]
    best_len = len_arr[j_best]
    best_lnz = grid_lnz[i_best, j_best]

    if field_kind == "gaussian":
        zm_b = False if zero_mode is None else bool(zero_mode)
        best_field: ProductionTerm = GaussianField(
            jnp.zeros((op.n_grid,), dtype=jnp.float64),
            op.t_start,
            op.t_end,
            sigma=best_sig,
            length_yr=best_len,
            order=order,
            zero_mode=zm_b,
        )
    else:
        zm_b = True if zero_mode is None else bool(zero_mode)
        best_field = LogGaussianField(
            jnp.zeros((op.n_grid,), dtype=jnp.float64),
            op.t_start,
            op.t_end,
            sigma=best_sig,
            length_yr=best_len,
            order=order,
            q_mean=q_std,
            zero_mode=zm_b,
        )
    best_prod = best_field if extra_production is None else (extra_production + best_field)
    best_fit = fit_ift_field(
        RadiocarbonModel(cbm, best_prod, solver, q_std=q_std),
        ds,
        offset_mode=offset_mode,
        operator=op,
    )

    return EvidenceGridResult(
        sigmas=sig_arr,
        lengths_yr=len_arr,
        log_evidence=grid_lnz,
        best_sigma=best_sig,
        best_length_yr=best_len,
        best_log_evidence=best_lnz,
        best_fit=best_fit,
    )


DEFAULT_CANDIDATE_KERNELS: tuple[str, ...] = (
    "matern12",
    "matern32",
    "matern52",
    "squared_exp",
    "continuous_matern",
    "matern32_plus_solar",
    "correlated_field",
)

_KERNEL_NAME_ALIASES: dict[str, str] = {
    "exp": "matern12",
    "ou": "matern12",
    "matern12": "matern12",
    "matern32": "matern32",
    "matern52": "matern52",
    "squared_exp": "squared_exp",
    "rbf": "squared_exp",
    "continuous_matern": "continuous_matern",
    "matern_free": "continuous_matern",
    "sho_solar": "sho_solar",
    "quasi_periodic": "sho_solar",
    "matern32_plus_solar": "matern32_plus_solar",
    "composite_solar": "matern32_plus_solar",
    "correlated_field": "correlated_field",
    "nifty_correlated": "correlated_field",
}


def _normal_logpdf(x: jax.Array, mean: float | jax.Array, sd: float | jax.Array) -> jax.Array:
    z = (x - mean) / sd
    return -0.5 * jnp.square(z) - jnp.log(sd) - 0.5 * math.log(2.0 * math.pi)


def _kernel_spec_builders(
    canonical_name: str,
    ell_mid: float,
    *,
    zero_mode: bool,
    n_corr_modes: int = 4,
) -> tuple[
    Any,
    Any,
    tuple[str, ...],
    jax.Array,
]:
    """Return ``(build_kernel_fn, log_hyperprior_fn, param_names, base_phi0)``."""
    log_sig_mu = math.log(0.25)
    log_sig_sd = 1.5
    log_ell_mu = math.log(ell_mid)
    log_ell_sd = 1.5

    if canonical_name in ("matern12", "matern32", "matern52"):
        ord_fixed = {"matern12": 1.0, "matern32": 2.0, "matern52": 3.0}[canonical_name]

        def build_m(phi: jax.Array) -> SpectralKernel:
            return MaternSpectrum(
                sigma=jnp.exp(phi[0]),
                length_yr=jnp.exp(phi[1]),
                order=ord_fixed,
                learn_order=False,
                zero_mode=zero_mode,
            )

        def prior_m(phi: jax.Array) -> jax.Array:
            return _normal_logpdf(phi[0], log_sig_mu, log_sig_sd) + _normal_logpdf(
                phi[1], log_ell_mu, log_ell_sd
            )

        phi0 = jnp.array([math.log(0.25), log_ell_mu], dtype=jnp.float64)
        return build_m, prior_m, ("log_sigma", "log_length_yr"), phi0

    if canonical_name == "squared_exp":

        def build_se(phi: jax.Array) -> SpectralKernel:
            return SquaredExpSpectrum(
                sigma=jnp.exp(phi[0]),
                length_yr=jnp.exp(phi[1]),
                zero_mode=zero_mode,
            )

        def prior_se(phi: jax.Array) -> jax.Array:
            return _normal_logpdf(phi[0], log_sig_mu, log_sig_sd) + _normal_logpdf(
                phi[1], log_ell_mu, log_ell_sd
            )

        phi0 = jnp.array([math.log(0.25), log_ell_mu], dtype=jnp.float64)
        return build_se, prior_se, ("log_sigma", "log_length_yr"), phi0

    if canonical_name == "continuous_matern":

        def build_cm(phi: jax.Array) -> SpectralKernel:
            return MaternSpectrum(
                sigma=jnp.exp(phi[0]),
                length_yr=jnp.exp(phi[1]),
                order=0.5 + jnp.exp(phi[2]),
                learn_order=True,
                zero_mode=zero_mode,
            )

        def prior_cm(phi: jax.Array) -> jax.Array:
            return (
                _normal_logpdf(phi[0], log_sig_mu, log_sig_sd)
                + _normal_logpdf(phi[1], log_ell_mu, log_ell_sd)
                + _normal_logpdf(phi[2], math.log(1.5), 1.0)
            )

        phi0 = jnp.array([math.log(0.25), log_ell_mu, math.log(1.5)], dtype=jnp.float64)
        return build_cm, prior_cm, ("log_sigma", "log_length_yr", "log_nu"), phi0

    if canonical_name == "sho_solar":

        def build_sho(phi: jax.Array) -> SpectralKernel:
            return QuasiPeriodicSpectrum(
                sigma=jnp.exp(phi[0]),
                period_yr=jnp.exp(phi[1]),
                quality=0.5 + jnp.exp(phi[2]),
                zero_mode=zero_mode,
            )

        def prior_sho(phi: jax.Array) -> jax.Array:
            return (
                _normal_logpdf(phi[0], log_sig_mu, log_sig_sd)
                + _normal_logpdf(phi[1], math.log(11.0), 0.35)
                + _normal_logpdf(phi[2], math.log(3.5), 1.0)
            )

        phi0 = jnp.array(
            [math.log(0.22), math.log(11.0), math.log(4.5)], dtype=jnp.float64
        )
        return build_sho, prior_sho, ("log_sigma", "log_period_yr", "log_q_minus_half"), phi0

    if canonical_name == "matern32_plus_solar":

        def build_comp(phi: jax.Array) -> SpectralKernel:
            return MaternSpectrum(
                sigma=jnp.exp(phi[0]),
                length_yr=jnp.exp(phi[1]),
                order=2.0,
                zero_mode=zero_mode,
            ) + QuasiPeriodicSpectrum(
                sigma=jnp.exp(phi[2]),
                period_yr=jnp.exp(phi[3]),
                quality=0.5 + jnp.exp(phi[4]),
                zero_mode=False,
            )

        def prior_comp(phi: jax.Array) -> jax.Array:
            return (
                _normal_logpdf(phi[0], math.log(0.18), log_sig_sd)
                + _normal_logpdf(phi[1], log_ell_mu, log_ell_sd)
                + _normal_logpdf(phi[2], math.log(0.18), log_sig_sd)
                + _normal_logpdf(phi[3], math.log(11.0), 0.35)
                + _normal_logpdf(phi[4], math.log(3.5), 1.0)
            )

        phi0 = jnp.array(
            [
                math.log(0.18),
                log_ell_mu,
                math.log(0.22),
                math.log(11.0),
                math.log(4.5),
            ],
            dtype=jnp.float64,
        )
        names = (
            "log_sigma_matern",
            "log_length_yr",
            "log_sigma_solar",
            "log_period_yr",
            "log_q_minus_half",
        )
        return build_comp, prior_comp, names, phi0

    if canonical_name == "correlated_field":

        def build_cf(phi: jax.Array) -> SpectralKernel:
            return CorrelatedFieldSpectrum(
                sigma=jnp.exp(phi[0]),
                slope=phi[1],
                log_deviations=phi[2:],
                flexibility=0.5,
                zero_mode=zero_mode,
            )

        def prior_cf(phi: jax.Array) -> jax.Array:
            lp_head = _normal_logpdf(phi[0], log_sig_mu, log_sig_sd) + _normal_logpdf(
                phi[1], -2.0, 1.5
            )
            lp_xi = jnp.sum(_normal_logpdf(phi[2:], 0.0, 1.0))
            return lp_head + lp_xi

        phi0 = jnp.concatenate(
            [
                jnp.array([math.log(0.25), -2.0], dtype=jnp.float64),
                jnp.zeros((n_corr_modes,), dtype=jnp.float64),
            ]
        )
        names = ("log_sigma", "slope") + tuple(
            f"xi_spec_{m + 1}" for m in range(n_corr_modes)
        )
        return build_cf, prior_cf, names, phi0

    raise ValueError(
        f"Unknown kernel identifier {canonical_name!r}; expected one of "
        f"{sorted(_KERNEL_NAME_ALIASES)}."
    )


def _custom_kernel_builders(
    kernel_obj: SpectralKernel,
    ell_mid: float,
) -> tuple[Any, Any, tuple[str, ...], jax.Array]:
    """Construct hyperparameter builders recursively for any :class:`SpectralKernel`."""
    if isinstance(kernel_obj, MaternSpectrum):
        if kernel_obj.learn_order:
            b_fn, p_fn, nms, _ = _kernel_spec_builders(
                "continuous_matern", ell_mid, zero_mode=kernel_obj.zero_mode
            )
            phi0_cm = jnp.array(
                [
                    math.log(max(float(kernel_obj.sigma), 1e-3)),
                    math.log(max(float(kernel_obj.length_yr), 1e-2)),
                    math.log(max(float(kernel_obj.order) - 0.5, 1e-2)),
                ],
                dtype=jnp.float64,
            )
            return b_fn, p_fn, nms, phi0_cm
        ord_f = float(kernel_obj.order)

        def build_custom_m(phi: jax.Array) -> SpectralKernel:
            return MaternSpectrum(
                sigma=jnp.exp(phi[0]),
                length_yr=jnp.exp(phi[1]),
                order=ord_f,
                learn_order=False,
                zero_mode=kernel_obj.zero_mode,
            )

        def prior_custom_m(phi: jax.Array) -> jax.Array:
            return _normal_logpdf(phi[0], math.log(0.25), 1.5) + _normal_logpdf(
                phi[1], math.log(ell_mid), 1.5
            )

        phi0 = jnp.array(
            [
                math.log(max(float(kernel_obj.sigma), 1e-3)),
                math.log(max(float(kernel_obj.length_yr), 1e-2)),
            ],
            dtype=jnp.float64,
        )
        return build_custom_m, prior_custom_m, ("log_sigma", "log_length_yr"), phi0

    if isinstance(kernel_obj, SquaredExpSpectrum):
        b_se, p_se, nms_se, _ = _kernel_spec_builders(
            "squared_exp", ell_mid, zero_mode=kernel_obj.zero_mode
        )
        phi0_se = jnp.array(
            [
                math.log(max(float(kernel_obj.sigma), 1e-3)),
                math.log(max(float(kernel_obj.length_yr), 1e-2)),
            ],
            dtype=jnp.float64,
        )
        return b_se, p_se, nms_se, phi0_se

    if isinstance(kernel_obj, QuasiPeriodicSpectrum):
        b_qp, p_qp, nms_qp, _ = _kernel_spec_builders(
            "sho_solar", ell_mid, zero_mode=kernel_obj.zero_mode
        )
        phi0_qp = jnp.array(
            [
                math.log(max(float(kernel_obj.sigma), 1e-3)),
                math.log(max(float(kernel_obj.period_yr), 1e-2)),
                math.log(max(float(kernel_obj.quality) - 0.5, 1e-2)),
            ],
            dtype=jnp.float64,
        )
        return b_qp, p_qp, nms_qp, phi0_qp

    if isinstance(kernel_obj, CorrelatedFieldSpectrum):
        n_m = int(kernel_obj.log_deviations.shape[0])
        flex_f = float(kernel_obj.flexibility)
        zm_cf = kernel_obj.zero_mode

        def build_custom_cf(phi: jax.Array) -> SpectralKernel:
            return CorrelatedFieldSpectrum(
                sigma=jnp.exp(phi[0]),
                slope=phi[1],
                log_deviations=phi[2:],
                flexibility=flex_f,
                zero_mode=zm_cf,
            )

        _, p_cf, nms_cf, _ = _kernel_spec_builders(
            "correlated_field", ell_mid, zero_mode=zm_cf, n_corr_modes=n_m
        )
        phi0_cf = jnp.concatenate(
            [
                jnp.array(
                    [
                        math.log(max(float(kernel_obj.sigma), 1e-3)),
                        float(kernel_obj.slope),
                    ],
                    dtype=jnp.float64,
                ),
                jnp.asarray(kernel_obj.log_deviations, dtype=jnp.float64),
            ],
            axis=0,
        )
        return build_custom_cf, p_cf, nms_cf, phi0_cf

    if isinstance(kernel_obj, SumSpectrum):
        sub_specs = [_custom_kernel_builders(part, ell_mid) for part in kernel_obj.parts]
        sub_builds = tuple(s[0] for s in sub_specs)
        sub_priors = tuple(s[1] for s in sub_specs)
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for s in sub_specs:
            sz = int(s[3].shape[0])
            offsets.append((cursor, cursor + sz))
            cursor += sz
        bounds = tuple(offsets)

        def build_sum(phi: jax.Array) -> SpectralKernel:
            return SumSpectrum(
                tuple(b_fn(phi[lo:hi]) for b_fn, (lo, hi) in zip(sub_builds, bounds))
            )

        def prior_sum(phi: jax.Array) -> jax.Array:
            acc = jnp.asarray(0.0, dtype=jnp.float64)
            for p_fn, (lo, hi) in zip(sub_priors, bounds):
                acc = acc + p_fn(phi[lo:hi])
            return acc

        names_sum = tuple(
            f"p{idx + 1}_{nm}"
            for idx, s in enumerate(sub_specs)
            for nm in s[2]
        )
        phi0_sum = jnp.concatenate([s[3] for s in sub_specs], axis=0)
        return build_sum, prior_sum, names_sum, phi0_sum

    raise TypeError(
        f"Automatic hyperparameter parameterization is not registered for {type(kernel_obj)!r}."
    )


class OptimizedKernelResult(zdx.Base):
    """Empirical-Bayes optimization result for a single GP spectral family."""

    name: str = eqx.field(static=True)
    kernel: SpectralKernel
    fit: IFTFieldFitResult
    phi_map: jax.Array
    hyperparameters: dict[str, float] = eqx.field(static=True)
    log_evidence: jax.Array
    log_hyperprior: jax.Array
    log_joint_hyper: jax.Array
    log_hyper_evidence: jax.Array
    hyper_hessian: jax.Array
    gamma: jax.Array
    aic: jax.Array
    bic: jax.Array
    n_hyperparams: int = eqx.field(static=True)
    converged: bool = eqx.field(static=True)


def _summarize_hyperparams(kernel: SpectralKernel) -> dict[str, float]:
    if isinstance(kernel, MaternSpectrum):
        out = {
            "sigma": float(kernel.sigma),
            "length_yr": float(kernel.length_yr),
            "order": float(kernel.order),
        }
        return out
    if isinstance(kernel, SquaredExpSpectrum):
        return {
            "sigma": float(kernel.sigma),
            "length_yr": float(kernel.length_yr),
        }
    if isinstance(kernel, QuasiPeriodicSpectrum):
        return {
            "sigma": float(kernel.sigma),
            "period_yr": float(kernel.period_yr),
            "quality": float(kernel.quality),
        }
    if isinstance(kernel, CorrelatedFieldSpectrum):
        return {
            "sigma": float(kernel.sigma),
            "slope": float(kernel.slope),
            "rms_xi": float(jnp.sqrt(jnp.mean(jnp.square(kernel.log_deviations)))),
        }
    if isinstance(kernel, SumSpectrum):
        merged: dict[str, float] = {}
        for idx, p in enumerate(kernel.parts):
            for k, v in _summarize_hyperparams(p).items():
                merged[f"p{idx + 1}_{k}"] = v
        return merged
    return {}


def _preoptimize_extra_production(
    cbm: CarbonBoxModel,
    ds: Dataset,
    extra_production: ProductionTerm,
    op: DCTResponseOperator,
    offset_mode: Literal["hierarchical", "none"],
    q_std: float,
) -> tuple[ProductionTerm, jax.Array, jax.Array | None, float]:
    """Extract linear unit-fluence response for any ``MiyakeSpike`` in ``extra_production``."""
    _ = offset_mode
    solver = ExactLinearSolver()
    probe_model = RadiocarbonModel(
        cbm,
        extra_production
        + GaussianField(
            jnp.zeros((op.n_grid,), dtype=jnp.float64),
            op.t_start,
            op.t_end,
            sigma=0.25,
            length_yr=5.0,
            order=2.0,
        ),
        solver,
        q_std=q_std,
    )
    _, _, extras_init = _extract_field_and_extras(probe_model.production)
    has_spike = any(isinstance(t, MiyakeSpike) for _, t in extras_init)

    if not has_spike:
        exc = _eval_extra_terms_d14c(
            cbm,
            solver,
            ds,
            extras_init,
            q_std,
            ALPHA_ATOMS_CM2_S_TO_KG_YR,
            op.steady_unit_response,
        )
        return extra_production, exc, None, 0.0

    non_spike_terms: list[tuple[str, ProductionTerm]] = []
    spike_term: MiyakeSpike | None = None
    updated_parts: dict[str, ProductionTerm] = {}
    for p_str, term in extras_init:
        short_name = p_str.split(".")[-1]
        updated_parts[short_name] = term
        if isinstance(term, MiyakeSpike) and spike_term is None:
            spike_term = term
        else:
            non_spike_terms.append((p_str, term))

    updated_extra: ProductionTerm = (
        list(updated_parts.values())[0]
        if len(updated_parts) == 1
        else ProductionSum(updated_parts)
    )
    assert spike_term is not None
    unit_spike = eqx.tree_at(
        lambda s: s.fluence, spike_term, jnp.asarray(1.0, dtype=jnp.float64)
    )
    unit_spike_vec = _eval_extra_terms_d14c(
        cbm,
        solver,
        ds,
        (("spike", unit_spike),),
        q_std,
        ALPHA_ATOMS_CM2_S_TO_KG_YR,
        op.steady_unit_response,
    )
    fixed_non_spike_exc = _eval_extra_terms_d14c(
        cbm,
        solver,
        ds,
        tuple(non_spike_terms),
        q_std,
        ALPHA_ATOMS_CM2_S_TO_KG_YR,
        op.steady_unit_response,
    )
    return updated_extra, fixed_non_spike_exc, unit_spike_vec, float(spike_term.fluence)


def _gram_neg_log_evidence(
    amp: jax.Array,
    fl: jax.Array,
    gram_dct: jax.Array,
    wt_x_base: jax.Array,
    x_base_sq: jax.Array,
    wt_sp: jax.Array,
    x_dot_sp: jax.Array,
    sp_sq: jax.Array,
    log_norm_d: jax.Array,
    half_logdet_nuis: jax.Array,
) -> jax.Array:
    """Shared ``O(n_grid^3)`` Woodbury negative log-evidence compiled once across all kernels."""
    n_g = amp.shape[0]
    wt_x = wt_x_base - fl * wt_sp
    x_sq = x_base_sq - 2.0 * fl * x_dot_sp + (fl * fl) * sp_sq
    c_mat = jnp.eye(n_g, dtype=jnp.float64) + (amp[:, None] * gram_dct) * amp[None, :]
    chol = jnp.linalg.cholesky(c_mat)
    w_sol = jsl.solve_triangular(chol, amp * wt_x, lower=True)
    quad = jnp.maximum(x_sq - (w_sol @ w_sol), 0.0)
    half_logdet = jnp.sum(jnp.log(jnp.diag(chol)))
    return 0.5 * quad + log_norm_d + half_logdet_nuis + half_logdet


def _custom_kernel_cache_tag(kernel_obj: SpectralKernel) -> str:
    """Deterministic recursive JIT cache tag for any custom :class:`SpectralKernel`."""
    if isinstance(kernel_obj, MaternSpectrum):
        if kernel_obj.learn_order:
            return f"canon:continuous_matern:zm={kernel_obj.zero_mode}"
        return f"custom_matern:ord={round(float(kernel_obj.order), 8)}:zm={kernel_obj.zero_mode}"
    if isinstance(kernel_obj, SquaredExpSpectrum):
        return f"canon:squared_exp:zm={kernel_obj.zero_mode}"
    if isinstance(kernel_obj, QuasiPeriodicSpectrum):
        return f"canon:sho_solar:zm={kernel_obj.zero_mode}"
    if isinstance(kernel_obj, CorrelatedFieldSpectrum):
        return (
            f"custom_cf:m={int(kernel_obj.log_deviations.shape[0])}"
            f":flex={round(float(kernel_obj.flexibility), 8)}"
            f":zm={kernel_obj.zero_mode}"
        )
    if isinstance(kernel_obj, SumSpectrum):
        inner = "+".join(_custom_kernel_cache_tag(p) for p in kernel_obj.parts)
        return f"sum:({inner})"
    return f"custom:{id(kernel_obj)}"


def _finite_diff_hessian_from_grad(
    grad_fn: Any,
    u_opt: np.ndarray,
    eps: float = 1e-5,
) -> jax.Array:
    """Central-difference Hessian from exact analytical gradient ``grad_fn`` (``O(eps^2) ~ 1e-10``)."""
    d = int(u_opt.size)
    hess = np.empty((d, d), dtype=np.float64)
    for j in range(d):
        step = np.zeros(d, dtype=np.float64)
        step[j] = eps
        g_plus = grad_fn(u_opt + step)
        g_minus = grad_fn(u_opt - step)
        hess[:, j] = (g_plus - g_minus) / (2.0 * eps)
    return jnp.asarray(0.5 * (hess + hess.T), dtype=jnp.float64)


_SPEC_JIT_CACHE: dict[tuple[Any, ...], Any] = {}


def _get_cached_gram_grad_fn(
    cache_key: tuple[Any, ...],
    build_kernel_fn: Any,
    prior_fn: Any,
    n_k: int,
    n_g: int,
    dt: float,
    has_spike_amp: bool,
) -> Any:
    """Return a module-cached ``@jax.jit`` ``value_and_grad`` function for ``cache_key``."""
    cached = _SPEC_JIT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    def _loss_u(
        u_vec: jax.Array,
        log_fl0_arr: jax.Array,
        gram_dct: jax.Array,
        wt_x_base: jax.Array,
        x_base_sq: jax.Array,
        wt_sp: jax.Array,
        x_dot_sp: jax.Array,
        sp_sq: jax.Array,
        log_norm_d: jax.Array,
        half_logdet_nuis: jax.Array,
    ) -> jax.Array:
        u_c = jnp.clip(u_vec, -12.0, 12.0)
        phi_cur = u_c[:n_k]
        amp = _safe_sqrt(build_kernel_fn(phi_cur).spectrum(n_g, dt))
        if has_spike_amp:
            fl = jnp.exp(u_c[n_k])
            lp_sp = _normal_logpdf(u_c[n_k], log_fl0_arr, 1.5)
        else:
            fl = jnp.asarray(0.0, dtype=jnp.float64)
            lp_sp = jnp.asarray(0.0, dtype=jnp.float64)
        nll = _gram_neg_log_evidence(
            amp,
            fl,
            gram_dct,
            wt_x_base,
            x_base_sq,
            wt_sp,
            x_dot_sp,
            sp_sq,
            log_norm_d,
            half_logdet_nuis,
        )
        return nll - (prior_fn(phi_cur) + lp_sp)

    fn = jax.jit(jax.value_and_grad(_loss_u, argnums=0))
    _SPEC_JIT_CACHE[cache_key] = fn
    return fn


def _optimize_single_kernel_precomputed(
    label: str,
    cache_tag: str,
    ell_mid: float,
    build_kernel_fn: Any,
    prior_fn: Any,
    phi0: jax.Array,
    cbm: CarbonBoxModel,
    ds: Dataset,
    op: DCTResponseOperator,
    x_eff_base: jax.Array,
    w_dct_eff: jax.Array,
    unit_spike_eff: jax.Array | None,
    fixed_exc: jax.Array,
    unit_sp_raw: jax.Array | None,
    x_nuis: jax.Array,
    mu_nuis: jax.Array,
    root_nuis: jax.Array,
    u_nuis: jax.Array,
    spike_fluence_init: float,
    updated_extra: ProductionTerm | None,
    log_norm_d: jax.Array,
    half_logdet_nuis: jax.Array,
    field_kind: Literal["gaussian", "log_gaussian"],
    offset_mode: Literal["hierarchical", "none"],
    multi_start: int,
    q_std: float,
    rng_key: jax.Array,
) -> OptimizedKernelResult:
    """Optimize ``phi`` via multi-start BFGS using a module-cached JIT Gram kernel."""
    n_k = int(phi0.shape[0])
    has_spike_amp = unit_spike_eff is not None and field_kind == "gaussian"
    if has_spike_amp:
        log_fl0 = math.log(max(spike_fluence_init, 1e-3))
        u0 = jnp.concatenate([phi0, jnp.array([log_fl0], dtype=jnp.float64)], axis=0)
    else:
        log_fl0 = 0.0
        u0 = phi0

    d_u = int(u0.shape[0])
    n_starts = max(int(multi_start), 1)
    if n_starts == 1:
        starts = u0[None, :]
    else:
        pert = 0.45 * jax.random.normal(rng_key, (n_starts - 1, d_u), dtype=jnp.float64)
        if has_spike_amp:
            pert = pert.at[:, -1].multiply(0.2)
        starts = jnp.concatenate([u0[None, :], u0[None, :] + pert], axis=0)

    if field_kind == "gaussian":
        n_g = op.n_grid
        gram_dct = w_dct_eff.T @ w_dct_eff
        wt_x_base = w_dct_eff.T @ x_eff_base
        x_base_sq = x_eff_base @ x_eff_base
        if has_spike_amp:
            assert unit_spike_eff is not None
            wt_sp = w_dct_eff.T @ unit_spike_eff
            x_dot_sp = x_eff_base @ unit_spike_eff
            sp_sq = unit_spike_eff @ unit_spike_eff
        else:
            wt_sp = jnp.zeros((n_g,), dtype=jnp.float64)
            x_dot_sp = jnp.asarray(0.0, dtype=jnp.float64)
            sp_sq = jnp.asarray(0.0, dtype=jnp.float64)

        log_fl0_arr = jnp.asarray(log_fl0, dtype=jnp.float64)
        cache_key = (
            cache_tag,
            n_g,
            round(float(op.dt), 11),
            round(float(ell_mid), 11),
            bool(has_spike_amp),
            n_k,
        )
        jit_vg = _get_cached_gram_grad_fn(
            cache_key, build_kernel_fn, prior_fn, n_k, n_g, op.dt, has_spike_amp
        )

        def _obj_and_grad_np(u_np: np.ndarray) -> tuple[float, np.ndarray]:
            val_j, g_u = jit_vg(
                jnp.asarray(u_np, dtype=jnp.float64),
                log_fl0_arr,
                gram_dct,
                wt_x_base,
                x_base_sq,
                wt_sp,
                x_dot_sp,
                sp_sq,
                log_norm_d,
                half_logdet_nuis,
            )
            total_val = float(val_j)
            grad_arr = np.asarray(g_u, dtype=np.float64)
            if not (math.isfinite(total_val) and np.all(np.isfinite(grad_arr))):
                return 1e30, np.zeros_like(u_np)
            return total_val, grad_arr

        best_val = float("inf")
        best_u_np = np.asarray(starts[0], dtype=np.float64)
        converged = False
        for s_idx in range(n_starts):
            res_s = _scipy_minimize(
                _obj_and_grad_np,
                np.asarray(starts[s_idx], dtype=np.float64),
                method="BFGS",
                jac=True,
                options={"gtol": 1e-6, "maxiter": 80},
            )
            val_s = float(res_s.fun)
            if math.isfinite(val_s) and val_s < best_val:
                best_val = val_s
                best_u_np = np.asarray(res_s.x, dtype=np.float64)
                converged = bool(res_s.success) or float(np.linalg.norm(res_s.jac)) < 1e-4

        u_opt = jnp.asarray(best_u_np, dtype=jnp.float64)
        hess_all = _finite_diff_hessian_from_grad(
            lambda u_in: _obj_and_grad_np(u_in)[1], best_u_np
        )
        phi_opt = u_opt[:n_k]
        kern_opt = build_kernel_fn(phi_opt)

        # Update spike fluence if jointly profiled
        final_extra = updated_extra
        if has_spike_amp and updated_extra is not None:
            fl_opt = jnp.exp(u_opt[n_k])
            if isinstance(updated_extra, MiyakeSpike):
                final_extra = eqx.tree_at(lambda s: s.fluence, updated_extra, fl_opt)
            elif isinstance(updated_extra, ProductionSum):
                new_parts = {
                    n: (eqx.tree_at(lambda s: s.fluence, p, fl_opt) if isinstance(p, MiyakeSpike) else p)
                    for n, p in zip(updated_extra.names, updated_extra.parts)
                }
                final_extra = ProductionSum(new_parts)
            assert unit_spike_eff is not None and unit_sp_raw is not None
            x_eff_final = x_eff_base - fl_opt * unit_spike_eff
            extra_exc_final = fixed_exc + fl_opt * unit_sp_raw
        else:
            x_eff_final = x_eff_base
            extra_exc_final = fixed_exc

        hess_phi = hess_all[:n_k, :n_k]
        if has_spike_amp:
            # Schur-complement curvature wrt phi marginalizing spike log-fluence
            h_ss = jnp.maximum(hess_all[n_k, n_k], 1e-4)
            h_ps = hess_all[:n_k, n_k]
            hess_phi = hess_phi - jnp.outer(h_ps, h_ps) / h_ss

        # Fast closed-form Wiener reconstruction without re-running ODE solves
        spec_opt = kern_opt.spectrum(op.n_grid, op.dt)
        amp_opt = _safe_sqrt(spec_opt)
        j_eff_opt = w_dct_eff * amp_opt[None, :]
        z_hat, s_vals, _, chi2_eff = _wiener_solve_from_svd(x_eff_final, j_eff_opt)

        gf = GaussianField(
            z_hat,
            op.t_start,
            op.t_end,
            kernel=kern_opt,
            zero_mode=getattr(kern_opt, "zero_mode", False),
        )
        prod_final = gf if final_extra is None else (final_extra + gf)
        fitted_model = RadiocarbonModel(cbm, prod_final, ExactLinearSolver(), q_std=q_std)
        f_grid = gf.evaluate_grid()
        q_grid = q_std + f_grid
        f_exc_final = op.w_dct @ (amp_opt * z_hat)
        pred_no_off = op.ss_d14c + extra_exc_final + f_exc_final
        inv_sig = 1.0 / ds.sigma_flat

        if u_nuis.shape[1] > 0:
            resid_nuis = inv_sig * (ds.d14c_flat - pred_no_off - x_nuis @ mu_nuis)
            w_std, _ = _linear_posterior_std(resid_nuis, u_nuis)
            w_nuis = mu_nuis + root_nuis @ w_std
            if offset_mode == "hierarchical":
                n_off = 1 + (ds.n_series if ds.n_series > 1 else 0)
                w_off = w_nuis[:n_off]
                g_off = w_off[0]
                l_off = w_off[1:] if ds.n_series > 1 else jnp.zeros((1,), dtype=jnp.float64)
                base_d14c = x_nuis[:, :n_off] @ w_off
            else:
                g_off = jnp.asarray(0.0, dtype=jnp.float64)
                l_off = jnp.zeros((ds.n_series,), dtype=jnp.float64)
                base_d14c = jnp.zeros((ds.n_obs,), dtype=jnp.float64)
        else:
            g_off = jnp.asarray(0.0, dtype=jnp.float64)
            l_off = jnp.zeros((ds.n_series,), dtype=jnp.float64)
            base_d14c = jnp.zeros((ds.n_obs,), dtype=jnp.float64)

        z_norm_sq = z_hat @ z_hat
        half_logdet_field = 0.5 * jnp.sum(jnp.log1p(jnp.square(s_vals)))
        ln_z_val = (
            -0.5 * chi2_eff
            - 0.5 * z_norm_sq
            - log_norm_d
            - half_logdet_nuis
            - half_logdet_field
        )
        gamma_val = jnp.sum(jnp.square(s_vals) / (1.0 + jnp.square(s_vals)))
        lp_val = prior_fn(phi_opt)
        fit_res = IFTFieldFitResult(
            model=fitted_model,
            latent=z_hat,
            field_grid=f_grid,
            q_grid=q_grid,
            t_grid=op.t_grid,
            d14c_pred=pred_no_off + base_d14c,
            baseline_d14c=base_d14c,
            global_offset=g_off,
            lab_offsets=l_off,
            chi2=chi2_eff,
            latent_norm_sq=z_norm_sq,
            log_evidence=ln_z_val,
            gamma=gamma_val,
            operator=op,
            offset_mode=offset_mode,
            global_offset_prior_sd=200.0,
            lab_offset_prior_sd=1.5,
            converged=converged,
            n_steps=1,
        )
    else:
        # Nonlinear LogGaussianField: alternating inner LM on z and outer BFGS on phi
        best_neg_j = float("inf")
        phi_opt = phi0
        fit_best: IFTFieldFitResult | None = None
        for s_idx in range(min(n_starts, 2)):
            phi_cur = starts[s_idx, :n_k]
            z_cur = jnp.zeros((op.n_grid,), dtype=jnp.float64)
            for _ in range(3):
                k_cur = build_kernel_fn(phi_cur)
                lgf = LogGaussianField(
                    z_cur, op.t_start, op.t_end, kernel=k_cur, q_mean=q_std, zero_mode=True
                )
                prod_c = lgf if updated_extra is None else (updated_extra + lgf)
                f_c = fit_ift_field(
                    RadiocarbonModel(cbm, prod_c, ExactLinearSolver(), q_std=q_std),
                    ds,
                    offset_mode=offset_mode,
                    operator=op,
                    max_steps=30,
                )
                z_cur = f_c.latent

                # Outer BFGS step on phi using local Gauss-Newton linearization around z_cur
                q_c = lgf.with_latent(z_cur).evaluate_q_grid()
                inv_sig = 1.0 / ds.sigma_flat
                solver_tmp = ExactLinearSolver()
                _, _, ext_tmp = _extract_field_and_extras(f_c.model.production)
                ext_exc = _eval_extra_terms_d14c(
                    cbm,
                    solver_tmp,
                    ds,
                    ext_tmp,
                    q_std,
                    ALPHA_ATOMS_CM2_S_TO_KG_YR,
                    op.steady_unit_response,
                )
                x_nuis_t, mu_nuis_t, _, u_nuis_t = _build_offset_whitening(
                    ds,
                    op.box_activity_ratios,
                    offset_mode=offset_mode,
                    global_offset_prior_sd=200.0,
                    lab_offset_prior_sd=1.5,
                )
                nuis_s = x_nuis_t @ mu_nuis_t if x_nuis_t.shape[1] else 0.0
                f_exc_c = op.w_grid @ (q_c - q_std)
                r_raw_c = inv_sig * (ds.d14c_flat - op.ss_d14c - ext_exc - f_exc_c - nuis_s)
                w_local_dct = dct2((inv_sig[:, None] * op.w_grid) * q_c[None, :], axis=1)
                r_eff_c, w_loc_eff, h_ld_nuis = _whiten_vec_and_mat_qr(
                    r_raw_c, w_local_dct, u_nuis_t
                )

                def _surrogate_neg(p_in: jax.Array) -> jax.Array:
                    sp_in = build_kernel_fn(p_in).spectrum(op.n_grid, op.dt)
                    j_loc = w_loc_eff * _safe_sqrt(sp_in)[None, :]
                    x_loc = r_eff_c + j_loc @ z_cur
                    lz, _ = _exact_linear_evidence_core(
                        x_loc, j_loc, log_norm_d, h_ld_nuis
                    )
                    return -(lz + prior_fn(p_in))

                sol_p = optx.minimise(
                    lambda p, _: _surrogate_neg(p),
                    optx.BFGS(rtol=1e-5, atol=1e-5),
                    phi_cur,
                    max_steps=40,
                    throw=False,
                )
                phi_cur = sol_p.value

            k_fin = build_kernel_fn(phi_cur)
            lgf_fin = LogGaussianField(
                z_cur, op.t_start, op.t_end, kernel=k_fin, q_mean=q_std, zero_mode=True
            )
            prod_fin = lgf_fin if updated_extra is None else (updated_extra + lgf_fin)
            f_fin = fit_ift_field(
                RadiocarbonModel(cbm, prod_fin, ExactLinearSolver(), q_std=q_std),
                ds,
                offset_mode=offset_mode,
                operator=op,
                max_steps=35,
            )
            neg_j = float(-(f_fin.log_evidence + prior_fn(phi_cur)))
            if neg_j < best_neg_j:
                best_neg_j = neg_j
                phi_opt = phi_cur
                fit_best = f_fin
                hess_phi = jax.hessian(_surrogate_neg)(phi_cur)

        assert fit_best is not None
        fit_res = fit_best
        kern_opt = build_kernel_fn(phi_opt)
        ln_z_val = fit_res.log_evidence
        lp_val = prior_fn(phi_opt)
        gamma_val = fit_res.gamma
        converged = fit_res.converged

    h_sym = 0.5 * (hess_phi + hess_phi.T)
    evals = jnp.linalg.eigvalsh(h_sym)
    safe_evals = jnp.maximum(evals, 1e-4)
    half_logdet_h = 0.5 * jnp.sum(jnp.log(safe_evals))
    ln_joint = ln_z_val + lp_val
    ln_hyper = ln_joint + 0.5 * n_k * math.log(2.0 * math.pi) - half_logdet_h
    aic_val = -2.0 * ln_z_val + 2.0 * n_k
    bic_val = -2.0 * ln_z_val + float(n_k) * math.log(max(ds.n_obs, 1))

    return OptimizedKernelResult(
        name=label,
        kernel=kern_opt,
        fit=fit_res,
        phi_map=phi_opt,
        hyperparameters=_summarize_hyperparams(kern_opt),
        log_evidence=ln_z_val,
        log_hyperprior=lp_val,
        log_joint_hyper=ln_joint,
        log_hyper_evidence=ln_hyper,
        hyper_hessian=h_sym,
        gamma=gamma_val,
        aic=aic_val,
        bic=bic_val,
        n_hyperparams=n_k,
        converged=converged,
    )


def optimize_gp_kernel(
    cbm: CarbonBoxModel,
    dataset: Dataset | TreeRingSeries,
    kernel: str | SpectralKernel = "matern32",
    *,
    field_kind: Literal["gaussian", "log_gaussian"] = "gaussian",
    extra_production: ProductionTerm | None = None,
    multi_start: int = 3,
    n_grid: int = 128,
    pad_years: float = 15.0,
    offset_mode: Literal["hierarchical", "none"] = "hierarchical",
    q_std: float = DEFAULT_Q_STD,
    rng_key: jax.Array | None = None,
) -> OptimizedKernelResult:
    r"""Optimize GP spectral hyperparameters via empirical Bayes (MAP-II + Laplace Occam).

    Maximizes ``ln Z(d | phi) + ln p(phi)`` using a precomputed :class:`DCTResponseOperator`:
    for linear ``field_kind="gaussian"``, runs multi-start ``scipy.optimize.minimize(...,
    method="BFGS", jac=True)`` over a module-cached ``@jax.jit`` analytical
    ``jax.value_and_grad`` Gram kernel; for nonlinear ``field_kind="log_gaussian"``,
    alternates inner Levenberg–Marquardt MAP solves over ``z`` with outer
    ``optimistix.BFGS`` steps over ``phi``. Returns the Laplace hyper-evidence

    .. math::

        \ln \mathcal{Z}_{\mathrm{hyper}} = \ln Z(d \mid \hat{\phi}) + \ln p(\hat{\phi})
        + \frac{d_\phi}{2}\ln(2\pi) - \frac{1}{2}\ln\det H_\phi.
    """
    ds = _as_dataset(dataset)
    key = jax.random.PRNGKey(0) if rng_key is None else rng_key
    span_lo, span_hi = ds.time_span
    ell_mid = float(np.clip(0.15 * max(span_hi - span_lo, 5.0), 1.5, 25.0))
    zm = field_kind == "log_gaussian"

    if isinstance(kernel, str):
        key_clean = kernel.strip().lower()
        if key_clean not in _KERNEL_NAME_ALIASES:
            raise ValueError(
                f"Unknown kernel name {kernel!r}; expected one of {sorted(_KERNEL_NAME_ALIASES)}."
            )
        canon = _KERNEL_NAME_ALIASES[key_clean]
        build_fn, prior_fn, _, phi0 = _kernel_spec_builders(canon, ell_mid, zero_mode=zm)
        cache_tag = f"canon:{canon}:zm={zm}"
        label = kernel
    elif isinstance(kernel, SpectralKernel):
        build_fn, prior_fn, _, phi0 = _custom_kernel_builders(kernel, ell_mid)
        cache_tag = _custom_kernel_cache_tag(kernel)
        label = type(kernel).__name__
    else:
        raise TypeError(f"kernel must be a str or SpectralKernel, got {type(kernel)!r}.")

    op = build_dct_response_operator(
        cbm, ds, n_grid=n_grid, pad_years=pad_years, q_std=q_std
    )
    if extra_production is not None:
        upd_extra, fixed_exc, unit_sp, sp_fl = _preoptimize_extra_production(
            cbm, ds, extra_production, op, offset_mode, q_std
        )
    else:
        upd_extra = None
        fixed_exc = jnp.zeros((ds.n_obs,), dtype=jnp.float64)
        unit_sp = None
        sp_fl = 0.0

    x_nuis, mu_nuis, root_nuis, u_nuis = _build_offset_whitening(
        ds,
        op.box_activity_ratios,
        offset_mode=offset_mode,
        global_offset_prior_sd=200.0,
        lab_offset_prior_sd=1.5,
    )
    inv_sig = 1.0 / ds.sigma_flat
    log_norm_d = 0.5 * ds.n_obs * math.log(2.0 * math.pi) + jnp.sum(jnp.log(ds.sigma_flat))
    nuis_shift = x_nuis @ mu_nuis if x_nuis.shape[1] else 0.0
    x_raw_base = inv_sig * (ds.d14c_flat - op.ss_d14c - fixed_exc - nuis_shift)
    w_dct_scaled = inv_sig[:, None] * op.w_dct

    if unit_sp is not None:
        aug_mat = jnp.concatenate([w_dct_scaled, (inv_sig * unit_sp)[:, None]], axis=1)
        x_eff_base, aug_eff, half_logdet_nuis = _whiten_vec_and_mat_qr(
            x_raw_base, aug_mat, u_nuis
        )
        w_dct_eff = aug_eff[:, :-1]
        unit_sp_eff: jax.Array | None = aug_eff[:, -1]
    else:
        x_eff_base, w_dct_eff, half_logdet_nuis = _whiten_vec_and_mat_qr(
            x_raw_base, w_dct_scaled, u_nuis
        )
        unit_sp_eff = None

    return _optimize_single_kernel_precomputed(
        label=label,
        cache_tag=cache_tag,
        ell_mid=ell_mid,
        build_kernel_fn=build_fn,
        prior_fn=prior_fn,
        phi0=phi0,
        cbm=cbm,
        ds=ds,
        op=op,
        x_eff_base=x_eff_base,
        w_dct_eff=w_dct_eff,
        unit_spike_eff=unit_sp_eff,
        fixed_exc=fixed_exc,
        unit_sp_raw=unit_sp,
        x_nuis=x_nuis,
        mu_nuis=mu_nuis,
        root_nuis=root_nuis,
        u_nuis=u_nuis,
        spike_fluence_init=sp_fl,
        updated_extra=upd_extra,
        log_norm_d=log_norm_d,
        half_logdet_nuis=half_logdet_nuis,
        field_kind=field_kind,
        offset_mode=offset_mode,
        multi_start=multi_start,
        q_std=q_std,
        rng_key=key,
    )


class KernelSelectionResult(zdx.Base):
    """Ranked empirical-Bayes GP kernel comparison from :func:`select_gp_kernel`."""

    best_name: str = eqx.field(static=True)
    best_result: OptimizedKernelResult
    best_kernel: SpectralKernel
    best_model: RadiocarbonModel
    rankings: tuple[OptimizedKernelResult, ...]
    names: tuple[str, ...] = eqx.field(static=True)
    log_hyper_evidences: jax.Array
    log_evidences: jax.Array
    delta_log_evidence: jax.Array
    model_weights: jax.Array
    aics: jax.Array
    bics: jax.Array
    gammas: jax.Array
    posterior_latents: jax.Array
    posterior_q: jax.Array
    posterior_spectrum: jax.Array
    t_grid: jax.Array
    error_scale_global: float = eqx.field(static=True)
    error_scale_by_series: dict[str, float] = eqx.field(static=True)

    def get_candidate(self, name: str) -> OptimizedKernelResult:
        """Retrieve a specific candidate's :class:`OptimizedKernelResult` by name."""
        for r in self.rankings:
            if r.name == name:
                return r
        raise KeyError(f"Candidate {name!r} not in {self.names}.")

    def format_table(self) -> str:
        """Format a publication-ready Bayesian kernel comparison summary table."""
        header = (
            f"{'Rank':<4} {'Kernel':<22} {'d':>2} {'ln Z(phi)':>11} "
            f"{'ln Z_hyper':>11} {'dlnZ':>8} {'Weight':>8} {'gamma':>7} {'BIC':>10}"
        )
        sep = "-" * len(header)
        lines = [header, sep]
        for idx, r in enumerate(self.rankings):
            d_lnz = float(self.delta_log_evidence[idx])
            w_k = float(self.model_weights[idx])
            lines.append(
                f"{idx + 1:<4d} {r.name:<22} {r.n_hyperparams:>2d} "
                f"{float(r.log_evidence):>11.2f} {float(r.log_hyper_evidence):>11.2f} "
                f"{d_lnz:>+8.2f} {w_k:>8.3f} {float(r.gamma):>7.2f} {float(r.bic):>10.2f}"
            )
        lines.append(sep)
        hp_str = ", ".join(
            f"{k}={v:.3g}" for k, v in self.best_result.hyperparameters.items()
        )
        lines.append(f"Selected kernel : {self.best_name} ({hp_str})")
        lines.append(f"MacKay s_error  : {self.error_scale_global:.3f}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format_table()


def select_gp_kernel(
    cbm: CarbonBoxModel,
    dataset: Dataset | TreeRingSeries,
    candidates: Sequence[str | SpectralKernel] | Mapping[str, str | SpectralKernel] | None = None,
    *,
    field_kind: Literal["gaussian", "log_gaussian"] = "gaussian",
    extra_production: ProductionTerm | None = None,
    multi_start: int = 3,
    n_grid: int = 128,
    pad_years: float = 15.0,
    n_posterior_samples: int = 200,
    offset_mode: Literal["hierarchical", "none"] = "hierarchical",
    q_std: float = DEFAULT_Q_STD,
    rng_key: jax.Array | None = None,
) -> KernelSelectionResult:
    """Optimize and rank candidate GP spectral families via Bayesian model evidence.

    By default compares all seven canonical IFT / GP spectral families:
    ``("matern12", "matern32", "matern52", "squared_exp", "continuous_matern",
    "matern32_plus_solar", "correlated_field")``, sharing a single precomputed
    :class:`DCTResponseOperator` so all seven models optimize in well under ``1 s``.
    """
    ds = _as_dataset(dataset)
    key = jax.random.PRNGKey(0) if rng_key is None else rng_key
    span_lo, span_hi = ds.time_span
    ell_mid = float(np.clip(0.15 * max(span_hi - span_lo, 5.0), 1.5, 25.0))
    zm = field_kind == "log_gaussian"

    if candidates is None:
        cand_items: list[tuple[str, str | SpectralKernel]] = [
            (k, k) for k in DEFAULT_CANDIDATE_KERNELS
        ]
    elif isinstance(candidates, Mapping):
        cand_items = list(candidates.items())
    else:
        cand_items = [
            (c if isinstance(c, str) else type(c).__name__, c) for c in candidates
        ]
    if not cand_items:
        raise ValueError("candidates must contain at least one kernel.")

    op = build_dct_response_operator(
        cbm, ds, n_grid=n_grid, pad_years=pad_years, q_std=q_std
    )
    if extra_production is not None:
        upd_extra, fixed_exc, unit_sp, sp_fl = _preoptimize_extra_production(
            cbm, ds, extra_production, op, offset_mode, q_std
        )
    else:
        upd_extra = None
        fixed_exc = jnp.zeros((ds.n_obs,), dtype=jnp.float64)
        unit_sp = None
        sp_fl = 0.0

    x_nuis, mu_nuis, root_nuis, u_nuis = _build_offset_whitening(
        ds,
        op.box_activity_ratios,
        offset_mode=offset_mode,
        global_offset_prior_sd=200.0,
        lab_offset_prior_sd=1.5,
    )
    inv_sig = 1.0 / ds.sigma_flat
    log_norm_d = 0.5 * ds.n_obs * math.log(2.0 * math.pi) + jnp.sum(jnp.log(ds.sigma_flat))
    nuis_shift = x_nuis @ mu_nuis if x_nuis.shape[1] else 0.0
    x_raw_base = inv_sig * (ds.d14c_flat - op.ss_d14c - fixed_exc - nuis_shift)
    w_dct_scaled = inv_sig[:, None] * op.w_dct

    if unit_sp is not None:
        aug_mat = jnp.concatenate([w_dct_scaled, (inv_sig * unit_sp)[:, None]], axis=1)
        x_eff_base, aug_eff, half_logdet_nuis = _whiten_vec_and_mat_qr(
            x_raw_base, aug_mat, u_nuis
        )
        w_dct_eff = aug_eff[:, :-1]
        unit_sp_eff: jax.Array | None = aug_eff[:, -1]
    else:
        x_eff_base, w_dct_eff, half_logdet_nuis = _whiten_vec_and_mat_qr(
            x_raw_base, w_dct_scaled, u_nuis
        )
        unit_sp_eff = None

    subkeys = jax.random.split(key, len(cand_items) + 1)
    sample_key = subkeys[-1]

    results: list[OptimizedKernelResult] = []
    for idx, (label, spec_item) in enumerate(cand_items):
        if isinstance(spec_item, str):
            k_clean = spec_item.strip().lower()
            if k_clean not in _KERNEL_NAME_ALIASES:
                raise ValueError(
                    f"Unknown candidate kernel {spec_item!r}; expected one of "
                    f"{sorted(_KERNEL_NAME_ALIASES)}."
                )
            canon = _KERNEL_NAME_ALIASES[k_clean]
            build_fn, prior_fn, _, phi0 = _kernel_spec_builders(
                canon, ell_mid, zero_mode=zm
            )
            cache_tag = f"canon:{canon}:zm={zm}"
        elif isinstance(spec_item, SpectralKernel):
            build_fn, prior_fn, _, phi0 = _custom_kernel_builders(spec_item, ell_mid)
            cache_tag = _custom_kernel_cache_tag(spec_item)
        else:
            raise TypeError(
                f"Candidate {label!r} must be a str or SpectralKernel, got {type(spec_item)!r}."
            )

        res_k = _optimize_single_kernel_precomputed(
            label=label,
            cache_tag=cache_tag,
            ell_mid=ell_mid,
            build_kernel_fn=build_fn,
            prior_fn=prior_fn,
            phi0=phi0,
            cbm=cbm,
            ds=ds,
            op=op,
            x_eff_base=x_eff_base,
            w_dct_eff=w_dct_eff,
            unit_spike_eff=unit_sp_eff,
            fixed_exc=fixed_exc,
            unit_sp_raw=unit_sp,
            x_nuis=x_nuis,
            mu_nuis=mu_nuis,
            root_nuis=root_nuis,
            u_nuis=u_nuis,
            spike_fluence_init=sp_fl,
            updated_extra=upd_extra,
            log_norm_d=log_norm_d,
            half_logdet_nuis=half_logdet_nuis,
            field_kind=field_kind,
            offset_mode=offset_mode,
            multi_start=multi_start,
            q_std=q_std,
            rng_key=subkeys[idx],
        )
        results.append(res_k)

    # Sort descending by Laplace hyper-evidence
    ranked = tuple(
        sorted(results, key=lambda r: float(r.log_hyper_evidence), reverse=True)
    )
    best = ranked[0]
    ln_hyper_arr = jnp.array([r.log_hyper_evidence for r in ranked], dtype=jnp.float64)
    ln_ev_arr = jnp.array([r.log_evidence for r in ranked], dtype=jnp.float64)
    delta_lnz = ln_hyper_arr - ln_hyper_arr[0]
    weights = jax.nn.softmax(ln_hyper_arr)
    aics = jnp.array([r.aic for r in ranked], dtype=jnp.float64)
    bics = jnp.array([r.bic for r in ranked], dtype=jnp.float64)
    gammas = jnp.array([r.gamma for r in ranked], dtype=jnp.float64)

    draws = laplace_samples(
        best.fit,
        ds,
        n_samples=n_posterior_samples,
        rng_key=sample_key,
        offset_mode=offset_mode,
    )
    s_glob = float(error_scale(best.fit, ds, by_series=False, offset_mode=offset_mode))  # type: ignore[arg-type]
    try:
        s_by_ser = error_scale(best.fit, ds, by_series=True, offset_mode=offset_mode)
        assert isinstance(s_by_ser, dict)
    except ValueError:
        s_by_ser = {s.name: s_glob for s in ds.series}

    best_field, _, _ = _extract_field_and_extras(best.fit.model.production)
    post_spec = best_field.power_spectrum()

    return KernelSelectionResult(
        best_name=best.name,
        best_result=best,
        best_kernel=best.kernel,
        best_model=best.fit.model,
        rankings=ranked,
        names=tuple(r.name for r in ranked),
        log_hyper_evidences=ln_hyper_arr,
        log_evidences=ln_ev_arr,
        delta_log_evidence=delta_lnz,
        model_weights=weights,
        aics=aics,
        bics=bics,
        gammas=gammas,
        posterior_latents=draws["latents"],
        posterior_q=draws["q"],
        posterior_spectrum=post_spec,
        t_grid=op.t_grid,
        error_scale_global=s_glob,
        error_scale_by_series=s_by_ser,
    )


# =============================================================================
# 6. Stateless IFT Diagnostic Plotting (plot_field_draws & plot_evidence_grid)
# =============================================================================

_PLOT_PALETTE: tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#009E73",  # bluish green
    "#CC79A7",  # purple
    "#E69F00",  # orange
)


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
        raise TypeError(
            f"Expected GaussianField or LogGaussianField, got {type(field)!r}."
        )
    return np.asarray(vals, dtype=np.float64)


def plot_field_draws(
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
    eff_ylabel = (
        (r"$Q(t)$ ($\mathrm{atoms\,cm^{-2}\,s^{-1}}$)" if as_total_q else r"$\delta q(t)$ ($\mathrm{atoms\,cm^{-2}\,s^{-1}}$)")
        if ylabel is None
        else ylabel
    )

    is_single = isinstance(field_or_fields, (GaussianField, LogGaussianField))
    if is_single:
        f_single = field_or_fields
        n_grid = f_single.n_grid
        z_bank = (
            jax.random.normal(key, (n_draws, n_grid), dtype=jnp.float64)
            if latents is None
            else jnp.asarray(latents, dtype=jnp.float64)
        )
        if z_bank.ndim != 2 or z_bank.shape[1] != n_grid or z_bank.shape[0] < 3:
            raise ValueError(
                f"latents must have shape (n_draws >= 3, {n_grid}), got {z_bank.shape}."
            )
        curves = _eval_field_realizations(
            f_single, z_bank, as_total_q=as_total_q, q_std=q_std
        )
        t_np = np.asarray(f_single.t_grid, dtype=np.float64)

        if axes is None:
            eff_size = (9.0, 5.2) if figsize is None else figsize
            fig = plt.figure(figsize=eff_size, constrained_layout=True)
            gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.35])
            top_axes = [fig.add_subplot(gs[0, c]) for c in range(3)]
            bot_ax = fig.add_subplot(gs[1, :])
            out_axes = np.array([*top_axes, bot_ax], dtype=object)
        else:
            arr_ax = np.asarray(axes, dtype=object).ravel()
            if arr_ax.size < 4:
                raise ValueError("Single-field plot_field_draws requires 4 Axes when supplied.")
            top_axes = [arr_ax[0], arr_ax[1], arr_ax[2]]
            bot_ax = arr_ax[3]
            fig = bot_ax.figure
            out_axes = arr_ax

        y_min = float(np.percentile(curves, 1.0))
        y_max = float(np.percentile(curves, 99.0))
        pad = 0.12 * max(y_max - y_min, 1e-6)

        for c in range(3):
            ax_c = top_axes[c]
            ax_c.plot(t_np, curves[c], color=_PLOT_PALETTE[c], lw=1.5)
            if as_total_q:
                ax_c.axhline(q_std, color="#666666", lw=0.7, ls=":")
            else:
                ax_c.axhline(0.0, color="#666666", lw=0.7, ls=":")
            ax_c.set_xlim(float(t_np[0]), float(t_np[-1]))
            ax_c.set_ylim(y_min - pad, y_max + pad)
            ax_c.set_title(f"Draw {c + 1}", fontsize=10)
            if c == 0:
                ax_c.set_ylabel(eff_ylabel)
            else:
                ax_c.tick_params(labelleft=False)

        n_show = min(int(n_faint), curves.shape[0])
        for m in range(1, n_show):
            bot_ax.plot(t_np, curves[m], color=_PLOT_PALETTE[0], lw=0.75, alpha=0.14)
        bot_ax.plot(
            t_np,
            curves[0],
            color=_PLOT_PALETTE[0],
            lw=2.0,
            label="Draw 1 (highlighted)",
        )
        if truth is not None:
            bot_ax.plot(
                t_np,
                np.asarray(truth, dtype=np.float64),
                color="#111111",
                lw=1.5,
                ls="--",
                label="Truth",
            )
        bot_ax.set_xlim(float(t_np[0]), float(t_np[-1]))
        bot_ax.set_xlabel(xlabel)
        bot_ax.set_ylabel(eff_ylabel)
        bot_ax.legend(loc="upper right", frameon=False)
        if title:
            fig.suptitle(title, fontsize=11)
        return fig, out_axes

    # Multi-field side-by-side comparison sharing identical standard-normal latents z
    if isinstance(field_or_fields, Mapping):
        labels = list(field_or_fields.keys())
        fields = list(field_or_fields.values())
    else:
        fields = list(field_or_fields)
        labels = [f"Field {i + 1}" for i in range(len(fields))]
    if not fields:
        raise ValueError("field_or_fields cannot be empty.")

    n_cols = len(fields)
    n_grid = fields[0].n_grid
    for f in fields:
        if f.n_grid != n_grid:
            raise ValueError("All compared fields must share the same n_grid.")

    z_bank = (
        jax.random.normal(key, (n_draws, n_grid), dtype=jnp.float64)
        if latents is None
        else jnp.asarray(latents, dtype=jnp.float64)
    )
    if axes is None:
        eff_size = (3.3 * n_cols, 5.2) if figsize is None else figsize
        fig, ax_grid = plt.subplots(
            2,
            n_cols,
            figsize=eff_size,
            sharex="col",
            sharey="row",
            gridspec_kw={"height_ratios": [1.0, 1.25]},
            constrained_layout=True,
        )
        out_axes = np.atleast_2d(ax_grid)
        if out_axes.shape != (2, n_cols):
            out_axes = out_axes.reshape(2, n_cols)
    else:
        out_axes = np.asarray(axes, dtype=object).reshape(2, n_cols)
        fig = out_axes[0, 0].figure

    for col, (lbl, f_col) in enumerate(zip(labels, fields)):
        col_color = _PLOT_PALETTE[col % len(_PLOT_PALETTE)]
        curves_c = _eval_field_realizations(
            f_col, z_bank, as_total_q=as_total_q, q_std=q_std
        )
        t_np = np.asarray(f_col.t_grid, dtype=np.float64)
        ax_top = out_axes[0, col]
        ax_bot = out_axes[1, col]

        for r_idx, ls_style in zip(range(3), ("-", "--", "-.")):
            ax_top.plot(
                t_np,
                curves_c[r_idx],
                color=col_color,
                lw=1.35,
                ls=ls_style,
                alpha=0.9 - 0.15 * r_idx,
                label=f"Draw {r_idx + 1}",
            )
        ax_top.set_title(lbl, fontsize=10)
        ax_top.set_xlim(float(t_np[0]), float(t_np[-1]))
        if col == 0:
            ax_top.set_ylabel(f"3 draws\n{eff_ylabel}")
            ax_top.legend(loc="upper right", frameon=False, fontsize=8)

        n_show = min(int(n_faint), curves_c.shape[0])
        for m in range(1, n_show):
            ax_bot.plot(t_np, curves_c[m], color=col_color, lw=0.7, alpha=0.14)
        ax_bot.plot(t_np, curves_c[0], color=col_color, lw=2.0, label="Draw 1")
        if truth is not None:
            ax_bot.plot(
                t_np,
                np.asarray(truth, dtype=np.float64),
                color="#111111",
                lw=1.3,
                ls="--",
            )
        ax_bot.set_xlim(float(t_np[0]), float(t_np[-1]))
        ax_bot.set_xlabel(xlabel)
        if col == 0:
            ax_bot.set_ylabel(f"{n_show} draws\n{eff_ylabel}")

    if title:
        fig.suptitle(title, fontsize=11)
    return fig, out_axes


def _cell_edges(centres: np.ndarray, *, log_scale: bool) -> np.ndarray:
    """Compute ``N + 1`` cell edges around 1D monotone grid ``centres``."""
    c = np.asarray(centres, dtype=np.float64).reshape(-1)
    if c.size == 1:
        if log_scale:
            return np.array([c[0] / 1.25, c[0] * 1.25], dtype=np.float64)
        return np.array([c[0] - 0.5, c[0] + 0.5], dtype=np.float64)
    if log_scale:
        lc = np.log(c)
        mids = 0.5 * (lc[:-1] + lc[1:])
        left = lc[0] - 0.5 * (lc[1] - lc[0])
        right = lc[-1] + 0.5 * (lc[-1] - lc[-2])
        return np.exp(np.concatenate([[left], mids, [right]]))
    mids = 0.5 * (c[:-1] + c[1:])
    left = c[0] - 0.5 * (c[1] - c[0])
    right = c[-1] + 0.5 * (c[-1] - c[-2])
    return np.concatenate([[left], mids, [right]])


def _is_log_spaced(arr: np.ndarray) -> bool:
    if arr.size < 3 or np.any(arr <= 0.0):
        return False
    lin_diff = np.diff(arr)
    log_diff = np.diff(np.log(arr))
    lin_cv = float(np.std(lin_diff) / max(abs(float(np.mean(lin_diff))), 1e-12))
    log_cv = float(np.std(log_diff) / max(abs(float(np.mean(log_diff))), 1e-12))
    return log_cv < 0.25 * lin_cv


def plot_evidence_grid(
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
    if ax is None:
        fig, target_ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    else:
        target_ax = ax
        fig = target_ax.figure

    sigs = np.asarray(grid.sigmas, dtype=np.float64)
    lens = np.asarray(grid.lengths_yr, dtype=np.float64)
    lnz = np.asarray(grid.log_evidence, dtype=np.float64)
    surface = (lnz - float(np.nanmax(lnz))) if delta else lnz
    cbar_label = r"$\Delta \ln Z$" if delta else r"$\ln Z$"

    log_x = _is_log_spaced(lens)
    log_y = _is_log_spaced(sigs)
    x_edges = _cell_edges(lens, log_scale=log_x)
    y_edges = _cell_edges(sigs, log_scale=log_y)

    eff_vmin = vmin if delta else None
    eff_vmax = 0.0 if delta else None
    mesh = target_ax.pcolormesh(
        x_edges,
        y_edges,
        surface,
        cmap=cmap,
        vmin=eff_vmin,
        vmax=eff_vmax,
        shading="flat",
    )
    fig.colorbar(mesh, ax=target_ax, label=cbar_label, pad=0.02)

    if log_x:
        target_ax.set_xscale("log")
    if log_y:
        target_ax.set_yscale("log")

    if annotate and surface.size <= 100:
        lo_val = float(eff_vmin) if eff_vmin is not None else float(np.nanmin(surface))
        hi_val = float(eff_vmax) if eff_vmax is not None else float(np.nanmax(surface))
        mid_val = 0.5 * (lo_val + hi_val)
        for i, y_c in enumerate(sigs):
            for j, x_c in enumerate(lens):
                val = float(surface[i, j])
                txt_col = "#111111" if val > mid_val else "white"
                target_ax.text(
                    float(x_c),
                    float(y_c),
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=txt_col,
                )

    best_x = float(grid.best_length_yr)
    best_y = float(grid.best_sigma)
    # Dark contrast backing beneath the white 'x' marker so it stands out on bright peaks
    target_ax.scatter(
        [best_x], [best_y], marker="x", s=125, c="#111111", linewidths=3.4, zorder=5
    )
    target_ax.scatter(
        [best_x],
        [best_y],
        marker="x",
        s=95,
        c="white",
        linewidths=2.0,
        zorder=6,
        label=rf"Max $\ln Z$ ($\sigma={best_y:.2g}$, $\ell={best_x:.2g}$ yr)",
    )
    if truth is not None:
        t_sig, t_len = float(truth[0]), float(truth[1])
        target_ax.scatter(
            [t_len],
            [t_sig],
            marker="o",
            s=70,
            facecolors="none",
            edgecolors="#56B4E9",
            linewidths=1.8,
            zorder=6,
            label="Truth",
        )

    target_ax.set_xlabel(xlabel)
    target_ax.set_ylabel(ylabel)
    if title:
        target_ax.set_title(title)
    target_ax.legend(loc="lower right", frameon=True, framealpha=0.82, fontsize=8.5)
    return fig, target_ax
