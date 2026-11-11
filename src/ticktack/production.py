"""Composable ``zodiax`` cosmic-ray ``14C`` production models for ``ticktack 2.0``.

Follows the named-component ``zodiax.Base`` pattern of ``virgil.models.System`` so
parameter string-path addressing (e.g. ``model.get("production.spike.t_start")`` and
``model.set("production.spike.t_start", 774.6)``) works natively alongside ``+``
algebraic composition and linear-vs-nonlinear term partitioning.
"""

from __future__ import annotations

import dataclasses
import math
import textwrap
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import tinygp.kernels as tk
import zodiax as zdx

from .constants import DEFAULT_Q_STD

__all__ = [
    "kernel_spectral_density",
    "solin_sarkka_eigenpairs",
    "ProductionTerm",
    "ProductionSum",
    "SteadyProduction",
    "HarmonicSolar",
    "LinearTrend",
    "MiyakeSpike",
    "GPProduction",
    "LogGPProduction",
    "GriddedProduction",
]


def _check_part_name(name: str, owner_cls: type) -> None:
    """Validate a component name for ``ProductionSum`` string-path addressing."""
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(
            f"Production component name {name!r} must be a valid Python identifier "
            "so it can be addressed in zodiax paths like 'production.spike.t_start'."
        )
    field_names = {f.name for f in dataclasses.fields(owner_cls)}
    if name.startswith("_") or name in field_names or hasattr(owner_cls, name):
        raise ValueError(
            f"Production component name {name!r} clashes with an attribute or method "
            f"of {owner_cls.__name__}; choose another name."
        )


def _gamma_unit_fwhm(k: float) -> float:
    """Exact FWHM of ``u^{k-1} exp(-u)`` (``k > 1``) relative to its peak at ``u = k - 1``."""
    if k <= 1.0:
        raise ValueError(f"gamma_shape must be > 1.0 for a peaked pulse with well-defined FWHM, got {k}.")
    mode = k - 1.0
    target = -math.log(2.0)

    def f(x: float) -> float:
        if x <= 0.0:
            return -1e300
        return mode * math.log(x / mode) - (x - mode) - target

    # Left root in (0, mode)
    lo, hi = 0.0, mode
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    x_left = 0.5 * (lo + hi)

    # Right root in (mode, mode + 10 * sqrt(k) + 20)
    lo, hi = mode, mode + 10.0 * math.sqrt(k) + 20.0
    while f(hi) > 0.0:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    x_right = 0.5 * (lo + hi)
    return x_right - x_left


def _skewnormal_log_pdf_and_derivs(
    z: jax.Array, alpha: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Overflow-free unnormalized log-density ``g(z)``, ``g'(z)``, ``g''(z)`` for skew-normal."""
    w = alpha * z / math.sqrt(2.0)
    neg_branch = w <= 0.0
    w_neg = jnp.where(neg_branch, w, 0.0)
    w_pos = jnp.where(neg_branch, 0.0, w)

    # For w <= 0: erfc(-w) = exp(-w^2) * erfcx(-w)
    erfcx_neg = jax.scipy.special.erfcx(-w_neg)
    log_erfc_neg = -jnp.square(w_neg) + jnp.log(erfcx_neg)
    ratio_neg = math.sqrt(2.0 / math.pi) / erfcx_neg

    # For w > 0: erfc(-w) = 2 - exp(-w^2) * erfcx(w)
    erfc_pos = 2.0 - jnp.exp(-jnp.square(w_pos)) * jax.scipy.special.erfcx(w_pos)
    log_erfc_pos = jnp.log(erfc_pos)
    ratio_pos = math.sqrt(2.0 / math.pi) * jnp.exp(-jnp.square(w_pos)) / erfc_pos

    log_erfc = jnp.where(neg_branch, log_erfc_neg, log_erfc_pos)
    h = jnp.where(neg_branch, ratio_neg, ratio_pos)
    g = -0.5 * jnp.square(z) + log_erfc
    gp = -z + alpha * h
    gpp = -1.0 - jnp.square(alpha) * h * (alpha * z + h)
    return g, gp, gpp


def _skewnormal_mode_and_fwhm(alpha: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Exact mode ``z_mode`` and FWHM ``z_right - z_left`` of ``2 phi(z) Phi(alpha z)``."""
    delta = alpha / jnp.sqrt(1.0 + jnp.square(alpha))
    mu_z = delta * math.sqrt(2.0 / math.pi)
    sigma_z = jnp.sqrt(jnp.maximum(1.0 - jnp.square(mu_z), 1e-12))
    skew_z = ((4.0 - math.pi) / 2.0) * (mu_z / sigma_z) ** 3
    abs_a = jnp.maximum(jnp.abs(alpha), 1e-6)
    corr = jnp.where(
        jnp.abs(alpha) > 1e-3,
        0.5 * jnp.sign(alpha) * jnp.exp(-2.0 * math.pi / abs_a),
        0.0,
    )
    z_mode = mu_z - 0.5 * skew_z * sigma_z - corr

    # Globally convergent Newton refinement on strictly concave g(z) (g'' <= -1)
    for _ in range(6):
        _, gp, gpp = _skewnormal_log_pdf_and_derivs(z_mode, alpha)
        z_mode = z_mode - gp / gpp

    g_max, _, _ = _skewnormal_log_pdf_and_derivs(z_mode, alpha)
    half_step = math.sqrt(2.0 * math.log(2.0)) * sigma_z
    z_left = z_mode - half_step
    z_right = z_mode + half_step
    ln2 = math.log(2.0)

    for _ in range(7):
        g_l, gp_l, _ = _skewnormal_log_pdf_and_derivs(z_left, alpha)
        z_left = z_left - (g_l - g_max + ln2) / gp_l
        g_r, gp_r, _ = _skewnormal_log_pdf_and_derivs(z_right, alpha)
        z_right = z_right - (g_r - g_max + ln2) / gp_r

    return z_mode, z_right - z_left


def kernel_spectral_density(kernel: Any, omega: ArrayLike) -> jax.Array:
    """Two-sided Fourier power spectral density ``S_k(omega) = int k(tau) e^{-i omega tau} dtau``.

    Satisfies ``k(tau) = (1 / pi) int_0^inf S_k(omega) cos(omega tau) domega``,
    matching Solin & Särkkä (2020) Hilbert-space GP eigenvalues ``lambda_j = S_k(omega_j)``.
    Supports stationary and quasiseparable ``tinygp >= 0.3`` kernels and algebraic
    sums/scalar products.
    """
    w = jnp.asarray(omega, dtype=jnp.float64)

    # Sum kernels (both stationary base.Sum and quasisep.Sum)
    if isinstance(kernel, (tk.base.Sum, tk.quasisep.Sum)):
        return kernel_spectral_density(kernel.kernel1, w) + kernel_spectral_density(
            kernel.kernel2, w
        )

    # Quasiseparable Scale wrapper (note: Scale.scale multiplies covariance k -> scale * k)
    if isinstance(kernel, tk.quasisep.Scale):
        return jnp.asarray(kernel.scale, dtype=jnp.float64) * kernel_spectral_density(
            kernel.kernel, w
        )

    # Product with a Constant kernel (e.g. sigma**2 * tk.Matern32(ell))
    if isinstance(kernel, tk.base.Product):
        if isinstance(kernel.kernel1, tk.base.Constant):
            c = jnp.asarray(kernel.kernel1.value, dtype=jnp.float64)
            return c * kernel_spectral_density(kernel.kernel2, w)
        if isinstance(kernel.kernel2, tk.base.Constant):
            c = jnp.asarray(kernel.kernel2.value, dtype=jnp.float64)
            return c * kernel_spectral_density(kernel.kernel1, w)
        raise TypeError(
            "Product of two non-constant kernels does not have an elementary closed-form "
            f"spectral density; got {type(kernel.kernel1)} * {type(kernel.kernel2)}."
        )

    # Quasiseparable kernels carry explicit amplitude sigma
    if isinstance(kernel, tk.quasisep.Matern32):
        ell = jnp.asarray(kernel.scale, dtype=jnp.float64)
        sig2 = jnp.square(jnp.asarray(kernel.sigma, dtype=jnp.float64))
        return sig2 * (12.0 * math.sqrt(3.0) * ell) / jnp.square(3.0 + jnp.square(ell * w))

    if isinstance(kernel, tk.quasisep.Matern52):
        ell = jnp.asarray(kernel.scale, dtype=jnp.float64)
        sig2 = jnp.square(jnp.asarray(kernel.sigma, dtype=jnp.float64))
        return (
            sig2
            * (400.0 * math.sqrt(5.0) * ell / 3.0)
            / (5.0 + jnp.square(ell * w)) ** 3
        )

    if isinstance(kernel, tk.quasisep.Exp):
        ell = jnp.asarray(kernel.scale, dtype=jnp.float64)
        sig2 = jnp.square(jnp.asarray(kernel.sigma, dtype=jnp.float64))
        return sig2 * (2.0 * ell) / (1.0 + jnp.square(ell * w))

    if isinstance(kernel, tk.quasisep.SHO):
        w0 = jnp.asarray(kernel.omega, dtype=jnp.float64)
        q = jnp.asarray(kernel.quality, dtype=jnp.float64)
        sig2 = jnp.square(jnp.asarray(kernel.sigma, dtype=jnp.float64))
        num = sig2 * (2.0 * w0**3 / q)
        den = jnp.square(jnp.square(w) - jnp.square(w0)) + jnp.square(w0 * w / q)
        return num / den

    # Standard stationary kernels (unit variance unless multiplied by Constant)
    if isinstance(kernel, tk.Matern32):
        ell = jnp.asarray(kernel.scale, dtype=jnp.float64)
        return (12.0 * math.sqrt(3.0) * ell) / jnp.square(3.0 + jnp.square(ell * w))

    if isinstance(kernel, tk.Matern52):
        ell = jnp.asarray(kernel.scale, dtype=jnp.float64)
        return (400.0 * math.sqrt(5.0) * ell / 3.0) / (5.0 + jnp.square(ell * w)) ** 3

    if isinstance(kernel, tk.Exp):
        ell = jnp.asarray(kernel.scale, dtype=jnp.float64)
        return (2.0 * ell) / (1.0 + jnp.square(ell * w))

    if isinstance(kernel, tk.ExpSquared):
        ell = jnp.asarray(kernel.scale, dtype=jnp.float64)
        return math.sqrt(2.0 * math.pi) * ell * jnp.exp(-0.5 * jnp.square(ell * w))

    raise TypeError(
        f"Unsupported kernel type for spectral density evaluation: {type(kernel)!r}."
    )


def solin_sarkka_eigenpairs(
    a: float,
    b: float,
    n_modes: int,
    kernel: Any,
) -> tuple[jax.Array, jax.Array, float]:
    """Dirichlet Laplacian frequencies ``omega_j``, spectral weights ``S_k(omega_j)``, and half-length ``L``."""
    if not (b > a):
        raise ValueError(f"Domain requires b > a, got [{a}, {b}].")
    if n_modes < 1:
        raise ValueError(f"n_modes must be >= 1, got {n_modes}.")
    half_len = 0.5 * float(b - a)
    j = jnp.arange(1, int(n_modes) + 1, dtype=jnp.float64)
    omegas = j * (math.pi / (2.0 * half_len))
    psd = kernel_spectral_density(kernel, omegas)
    return omegas, psd, half_len


class ProductionTerm(zdx.Base):
    """Abstract immutable cosmic-ray ``14C`` production component."""

    @property
    def is_linear_marginalizable(self) -> bool:
        """Whether this term's amplitudes enter ``Delta14C`` linearly and can be marginalized."""
        return False

    def _default_slug_name(self) -> str:
        return "term"

    def delta_q(
        self,
        t: ArrayLike,
        q_std: ArrayLike = DEFAULT_Q_STD,
    ) -> jax.Array:
        """Production perturbation ``delta_q(t) = Q(t) - q_std`` at epoch(s) ``t``."""
        raise NotImplementedError

    def __call__(
        self,
        t: ArrayLike,
        q_std: ArrayLike = DEFAULT_Q_STD,
    ) -> jax.Array:
        """Total cosmic-ray production rate ``Q(t) = q_std + delta_q(t)``."""
        return jnp.asarray(q_std, dtype=jnp.float64) + self.delta_q(t, q_std=q_std)

    def __add__(self, other: ProductionTerm) -> ProductionSum:
        if not isinstance(other, ProductionTerm):
            return NotImplemented
        left_items = (
            list(zip(self.names, self.parts))
            if isinstance(self, ProductionSum)
            else [(self._default_slug_name(), self)]
        )
        right_items = (
            list(zip(other.names, other.parts))
            if isinstance(other, ProductionSum)
            else [(other._default_slug_name(), other)]
        )

        combined: dict[str, ProductionTerm] = {}
        for base_name, part in left_items + right_items:
            candidate = base_name
            suffix = 2
            while candidate in combined:
                candidate = f"{base_name}_{suffix}"
                suffix += 1
            combined[candidate] = part
        return ProductionSum(combined)


class ProductionSum(ProductionTerm):
    """Named sum of :class:`ProductionTerm` components supporting ``zodiax`` string paths.

    Examples
    --------
    >>> prod = ProductionSum(
    ...     steady=SteadyProduction(1.76),
    ...     solar=HarmonicSolar.from_polar(periods=11.0, amplitudes=0.3),
    ...     spike=MiyakeSpike(t_start=774.5, duration=0.25, fluence=6.5),
    ... )
    >>> float(prod.get("spike.t_start"))
    774.5
    """

    names: tuple[str, ...] = eqx.field(static=True)
    parts: tuple[ProductionTerm, ...]

    def __init__(
        self,
        components: dict[str, ProductionTerm] | None = None,
        /,
        **named: ProductionTerm,
    ):
        merged = {**(components or {}), **named}
        if not merged:
            raise ValueError("ProductionSum requires at least one named ProductionTerm.")
        for name, comp in merged.items():
            _check_part_name(name, ProductionSum)
            if not isinstance(comp, ProductionTerm):
                raise TypeError(
                    f"Component {name!r} must be a ProductionTerm, got {comp!r}."
                )
        self.names = tuple(merged.keys())
        self.parts = tuple(merged.values())

    @property
    def components(self) -> dict[str, ProductionTerm]:
        """Named components as an ordered ``{name: term}`` dictionary."""
        return dict(zip(self.names, self.parts))

    def __getattr__(self, name: str) -> Any:
        try:
            names = object.__getattribute__(self, "names")
            parts = object.__getattribute__(self, "parts")
        except AttributeError:
            raise AttributeError(name) from None
        if name in names:
            return parts[names.index(name)]
        raise AttributeError(
            f"{type(self).__name__} has no component or attribute {name!r}; "
            f"available components are {list(names)}."
        )

    def __repr__(self) -> str:
        lines = [f"{name}={part!r}," for name, part in zip(self.names, self.parts)]
        body = textwrap.indent("\n".join(lines), "    ")
        return f"ProductionSum(\n{body}\n)"

    @property
    def is_linear_marginalizable(self) -> bool:
        return all(
            isinstance(p, SteadyProduction) or p.is_linear_marginalizable
            for p in self.parts
        )

    @property
    def linear_terms(self) -> dict[str, ProductionTerm]:
        """Components whose amplitudes can be marginalized analytically in ``LinearMarginal``."""
        return {
            n: p
            for n, p in zip(self.names, self.parts)
            if p.is_linear_marginalizable
        }

    @property
    def nonlinear_terms(self) -> dict[str, ProductionTerm]:
        """Nonlinear trajectory components (``MiyakeSpike``, ``LogGPProduction``, ``GriddedProduction``)."""
        return {
            n: p
            for n, p in zip(self.names, self.parts)
            if not isinstance(p, SteadyProduction) and not p.is_linear_marginalizable
        }

    @property
    def steady_terms(self) -> dict[str, SteadyProduction]:
        """Steady baseline components."""
        return {
            n: p
            for n, p in zip(self.names, self.parts)
            if isinstance(p, SteadyProduction)
        }

    def partition_linear_nonlinear(
        self,
        *,
        include_steady_in_linear: bool = False,
    ) -> tuple[dict[str, ProductionTerm], dict[str, ProductionTerm]]:
        """Partition components into ``(linear_marginalizable, nonlinear_trajectory)`` dicts."""
        lin: dict[str, ProductionTerm] = {}
        nonlin: dict[str, ProductionTerm] = {}
        for name, part in zip(self.names, self.parts):
            if isinstance(part, SteadyProduction):
                if include_steady_in_linear:
                    lin[name] = part
            elif part.is_linear_marginalizable:
                lin[name] = part
            else:
                nonlin[name] = part
        return lin, nonlin

    def delta_q(
        self,
        t: ArrayLike,
        q_std: ArrayLike = DEFAULT_Q_STD,
    ) -> jax.Array:
        t_arr = jnp.asarray(t, dtype=jnp.float64)
        total = jnp.zeros_like(t_arr, dtype=jnp.float64)
        for part in self.parts:
            total = total + part.delta_q(t_arr, q_std=q_std)
        return total


class SteadyProduction(ProductionTerm):
    """Constant cosmic-ray baseline production rate ``Q_0`` (``atoms cm^-2 s^-1``)."""

    rate: jax.Array

    def __init__(
        self,
        rate: ArrayLike = DEFAULT_Q_STD,
        *,
        q_std: ArrayLike | None = None,
    ):
        val = q_std if q_std is not None else rate
        self.rate = jnp.asarray(val, dtype=jnp.float64).reshape(())

    def _default_slug_name(self) -> str:
        return "steady"

    def delta_q(
        self,
        t: ArrayLike,
        q_std: ArrayLike = DEFAULT_Q_STD,
    ) -> jax.Array:
        t_arr = jnp.asarray(t, dtype=jnp.float64)
        offset = self.rate - jnp.asarray(q_std, dtype=jnp.float64)
        return jnp.broadcast_to(offset, t_arr.shape)


class HarmonicSolar(ProductionTerm):
    """Multi-harmonic solar modulation in linear quadrature coordinates ``(c_cos, c_sin)``.

    Models ``delta_q(t) = sum_m [c_cos[m] cos(omega_m (t - t_ref)) + c_sin[m] sin(omega_m (t - t_ref))]``,
    eliminating phase-wrapping singularities and enabling closed-form resolvent
    evaluation and ``LinearMarginal`` marginalization.
    """

    periods: jax.Array
    c_cos: jax.Array
    c_sin: jax.Array
    t_ref: float = eqx.field(static=True)

    def __init__(
        self,
        periods: ArrayLike = 11.0,
        c_cos: ArrayLike = 0.0,
        c_sin: ArrayLike | None = None,
        *,
        t_ref: float = 0.0,
    ):
        p = jnp.atleast_1d(jnp.asarray(periods, dtype=jnp.float64))
        m = p.shape[0]
        cc = jnp.broadcast_to(
            jnp.atleast_1d(jnp.asarray(c_cos, dtype=jnp.float64)), (m,)
        )
        if c_sin is None:
            cs = jnp.zeros((m,), dtype=jnp.float64)
        else:
            cs = jnp.broadcast_to(
                jnp.atleast_1d(jnp.asarray(c_sin, dtype=jnp.float64)), (m,)
            )
        self.periods = p
        self.c_cos = jnp.array(cc, dtype=jnp.float64)
        self.c_sin = jnp.array(cs, dtype=jnp.float64)
        self.t_ref = float(t_ref)

    @classmethod
    def from_polar(
        cls,
        periods: ArrayLike = 11.0,
        amplitudes: ArrayLike = 0.18 * DEFAULT_Q_STD,
        phases: ArrayLike = 0.0,
        *,
        t_ref: float = 0.0,
        convention: Literal["cos", "sin"] = "cos",
    ) -> HarmonicSolar:
        """Construct ``HarmonicSolar`` from polar amplitudes and phases (radians).

        * ``convention="cos"``: ``A * cos(omega * (t - t_ref) - phase)``
          -> ``c_cos = A cos(phase), c_sin = A sin(phase)``.
        * ``convention="sin"``: ``A * sin(omega * (t - t_ref) + phase)``
          -> ``c_cos = A sin(phase), c_sin = A cos(phase)``.
        """
        p = jnp.atleast_1d(jnp.asarray(periods, dtype=jnp.float64))
        m = p.shape[0]
        amp = jnp.broadcast_to(
            jnp.atleast_1d(jnp.asarray(amplitudes, dtype=jnp.float64)), (m,)
        )
        phi = jnp.broadcast_to(
            jnp.atleast_1d(jnp.asarray(phases, dtype=jnp.float64)), (m,)
        )
        if convention == "cos":
            c_cos = amp * jnp.cos(phi)
            c_sin = amp * jnp.sin(phi)
        elif convention == "sin":
            c_cos = amp * jnp.sin(phi)
            c_sin = amp * jnp.cos(phi)
        else:
            raise ValueError(f"convention must be 'cos' or 'sin', got {convention!r}.")
        return cls(periods=p, c_cos=c_cos, c_sin=c_sin, t_ref=t_ref)

    @property
    def is_linear_marginalizable(self) -> bool:
        return True

    def _default_slug_name(self) -> str:
        return "solar"

    @property
    def omegas(self) -> jax.Array:
        return (2.0 * math.pi) / self.periods

    @property
    def amplitudes(self) -> jax.Array:
        return jnp.hypot(self.c_cos, self.c_sin)

    @property
    def phases(self) -> jax.Array:
        return jnp.arctan2(self.c_sin, self.c_cos)

    def delta_q(
        self,
        t: ArrayLike,
        q_std: ArrayLike = DEFAULT_Q_STD,
    ) -> jax.Array:
        del q_std
        t_arr = jnp.asarray(t, dtype=jnp.float64)
        theta = (t_arr[..., None] - self.t_ref) * self.omegas
        return jnp.sum(
            self.c_cos * jnp.cos(theta) + self.c_sin * jnp.sin(theta),
            axis=-1,
        )


class LinearTrend(ProductionTerm):
    """Secular or switched-on linear cosmic-ray drift ``slope * (t - t_ref)``.

    * ``switched_on=False`` (default): eternal drift anchored at ``t_ref`` so
      ``delta_y(t_ref) = 0`` (constant lag absorbed into ``Delta_global``, avoiding
      ill-conditioned ``(-A)^{-2}`` subtraction).
    * ``switched_on=True``: causal ramp ``slope * max(t - t_ref, 0)`` starting from
      rest at ``t = t_ref``, integrated exactly via Van Loan ``psi_2`` / ``psi_3``.
    """

    slope: jax.Array
    t_ref: float = eqx.field(static=True)
    switched_on: bool = eqx.field(static=True)

    def __init__(
        self,
        slope: ArrayLike = 0.0,
        *,
        t_ref: float = 0.0,
        switched_on: bool = False,
    ):
        self.slope = jnp.asarray(slope, dtype=jnp.float64).reshape(())
        self.t_ref = float(t_ref)
        self.switched_on = bool(switched_on)

    @property
    def is_linear_marginalizable(self) -> bool:
        return True

    def _default_slug_name(self) -> str:
        return "trend"

    def delta_q(
        self,
        t: ArrayLike,
        q_std: ArrayLike = DEFAULT_Q_STD,
    ) -> jax.Array:
        del q_std
        dt = jnp.asarray(t, dtype=jnp.float64) - self.t_ref
        if self.switched_on:
            dt = jnp.maximum(dt, 0.0)
        return self.slope * dt


class MiyakeSpike(ProductionTerm):
    """Parametric cosmic-ray event pulse with unified FWHM ``duration`` and exact ``fluence``.

    Supported profiles:
    * ``"boxcar"``: flat pulse on ``[t_start, t_start + duration]`` of height
      ``fluence / duration`` (exact analytical Van Loan integration down to ``duration -> 0+``).
    * ``"supergaussian"``: centred at ``t_start + 0.5 * duration`` with width
      ``w = duration / (2 * (ln 2)^{1/p})`` (exact FWHM ``= duration``) and analytical
      normalization ``fluence / (2 * w * Gamma(1 + 1/p))`` so ``int delta_q dt == fluence``.
    * ``"skewgaussian"``: Azzalini skew-normal pulse with skewness parameter ``skewness``
      and exact unit-integral ``erfc`` normalization.
    * ``"gamma"``: causal ``Gamma(k, theta)`` pulse on ``t >= t_start`` with shape
      ``gamma_shape > 1`` and exact FWHM ``= duration``.
    """

    t_start: jax.Array
    duration: jax.Array
    fluence: jax.Array
    skewness: jax.Array | None
    profile: str = eqx.field(static=True)
    exponent: float = eqx.field(static=True)
    gamma_shape: float = eqx.field(static=True)
    _gamma_fwhm_scale: float = eqx.field(static=True)

    def __init__(
        self,
        t_start: ArrayLike = 774.5,
        duration: ArrayLike = 0.25,
        fluence: ArrayLike | None = None,
        *,
        area: ArrayLike | None = None,
        profile: Literal[
            "boxcar", "supergaussian", "skewgaussian", "gamma"
        ] = "supergaussian",
        exponent: float = 16.0,
        skewness: ArrayLike | None = None,
        gamma_shape: float = 3.0,
    ):
        if fluence is not None and area is not None:
            raise ValueError("Pass either 'fluence' or alias 'area', not both.")
        fl_val = 6.5 if (fluence is None and area is None) else (area if fluence is None else fluence)

        prof = str(profile).strip().lower()
        if prof not in ("boxcar", "supergaussian", "skewgaussian", "gamma"):
            raise ValueError(
                f"Unknown MiyakeSpike profile {profile!r}; expected 'boxcar', "
                "'supergaussian', 'skewgaussian', or 'gamma'."
            )
        if float(exponent) <= 0.0:
            raise ValueError(f"supergaussian exponent must be > 0, got {exponent}.")

        self.t_start = jnp.asarray(t_start, dtype=jnp.float64).reshape(())
        self.duration = jnp.asarray(duration, dtype=jnp.float64).reshape(())
        self.fluence = jnp.asarray(fl_val, dtype=jnp.float64).reshape(())
        self.profile = prof
        self.exponent = float(exponent)
        self.gamma_shape = float(gamma_shape)
        self._gamma_fwhm_scale = (
            _gamma_unit_fwhm(self.gamma_shape) if prof == "gamma" else 1.0
        )

        if prof == "skewgaussian":
            sk = 2.0 if skewness is None else skewness
            self.skewness = jnp.asarray(sk, dtype=jnp.float64).reshape(())
        else:
            if skewness is not None:
                raise ValueError("skewness is only valid when profile='skewgaussian'.")
            self.skewness = None

    def _default_slug_name(self) -> str:
        return "spike"

    def unit_profile(self, u: jax.Array) -> jax.Array:
        """Dimensionless unit-area kernel ``phi_unit(u)`` for ``t = t_start + duration * u``."""
        u_arr = jnp.asarray(u, dtype=jnp.float64)

        if self.profile == "boxcar":
            return jnp.where((u_arr >= 0.0) & (u_arr <= 1.0), 1.0, 0.0)

        if self.profile == "supergaussian":
            p = self.exponent
            ln2 = math.log(2.0)
            norm = (ln2 ** (1.0 / p)) / math.gamma(1.0 + 1.0 / p)
            x = 2.0 * u_arr - 1.0
            p_int = int(round(p))
            if abs(p - p_int) < 1e-12 and p_int % 2 == 0:
                xp = x**p_int
            else:
                safe_abs = jnp.where(jnp.abs(x) > 1e-15, jnp.abs(x), 1.0)
                xp = jnp.where(jnp.abs(x) > 1e-15, safe_abs**p, 0.0)
            return norm * jnp.exp(-ln2 * xp)

        if self.profile == "skewgaussian":
            assert self.skewness is not None
            alpha = self.skewness
            z_mode, fwhm_z = _skewnormal_mode_and_fwhm(alpha)
            z = z_mode + fwhm_z * (u_arr - 0.5)
            g, _, _ = _skewnormal_log_pdf_and_derivs(z, alpha)
            pdf_z = (1.0 / math.sqrt(2.0 * math.pi)) * jnp.exp(g)
            return pdf_z * fwhm_z

        if self.profile == "gamma":
            k = self.gamma_shape
            c_k = self._gamma_fwhm_scale
            pos = u_arr > 0.0
            safe_u = jnp.where(pos, u_arr, 1.0)
            log_norm = k * math.log(c_k) - math.lgamma(k)
            val = jnp.exp(log_norm + (k - 1.0) * jnp.log(safe_u) - c_k * safe_u)
            return jnp.where(pos, val, 0.0)

        raise RuntimeError(f"Unhandled profile {self.profile!r}")

    def unit_support_panels(self) -> tuple[tuple[float, float], ...]:
        """Smooth dimensionless sub-panels ``(u_lo, u_hi)`` covering ``phi_unit`` to ``< 1e-17``."""
        if self.profile == "boxcar":
            return ((0.0, 1.0),)

        if self.profile == "supergaussian":
            p = self.exponent
            h_u = 0.5 * ((42.0 / math.log(2.0)) ** (1.0 / p))
            mid_sh = 0.5 * (h_u - 0.5)
            knots = (
                0.5 - h_u,
                -mid_sh,
                0.0,
                0.25,
                0.5,
                0.75,
                1.0,
                1.0 + mid_sh,
                0.5 + h_u,
            )
            return tuple(zip(knots[:-1], knots[1:]))

        if self.profile == "skewgaussian":
            knots = (-7.5, -3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 4.5, 8.5)
            return tuple(zip(knots[:-1], knots[1:]))

        if self.profile == "gamma":
            k = self.gamma_shape
            c_k = self._gamma_fwhm_scale
            u_mode = (k - 1.0) / c_k
            u_max = (k + 45.0 + 12.0 * math.sqrt(k)) / c_k
            # Geometrically graded toward u=0 to resolve non-integer u^{k-1} branch point to < 1e-14
            knots = (
                0.0,
                0.002 * u_mode,
                0.015 * u_mode,
                0.06 * u_mode,
                0.20 * u_mode,
                0.50 * u_mode,
                u_mode,
                1.80 * u_mode,
                3.20 * u_mode,
                5.50 * u_mode,
                min(9.50 * u_mode, 0.5 * (5.50 * u_mode + u_max)),
                u_max,
            )
            return tuple(zip(knots[:-1], knots[1:]))

        raise RuntimeError(f"Unhandled profile {self.profile!r}")

    def delta_q(
        self,
        t: ArrayLike,
        q_std: ArrayLike = DEFAULT_Q_STD,
    ) -> jax.Array:
        del q_std
        t_arr = jnp.asarray(t, dtype=jnp.float64)
        active = self.duration > 1e-12
        safe_dur = jnp.where(active, self.duration, 1.0)
        u = (t_arr - self.t_start) / safe_dur
        rate = (self.fluence / safe_dur) * self.unit_profile(u)
        return jnp.where(active, rate, 0.0)


class GPProduction(ProductionTerm):
    """Gaussian Process cosmic-ray perturbation ``delta_q(t) ~ GP(0, kernel)``.

    When ``coefficients is None`` (the default), ``GPProduction`` represents the
    prior over Solin–Särkkä modes and is analytically marginalized via
    ``ticktack.linear.LinearMarginal`` with zero latent MCMC parameters. When
    ``coefficients`` (white-noise weights ``z in R^{n_modes}``) and ``domain``
    are supplied, ``delta_q(t)`` evaluates the explicit realization.
    """

    kernel: Any
    coefficients: jax.Array | None
    n_modes: int = eqx.field(static=True)
    pad_years: float = eqx.field(static=True)
    domain: tuple[float, float] | None = eqx.field(static=True)

    def __init__(
        self,
        kernel: Any,
        *,
        n_modes: int = 96,
        pad_years: float = 15.0,
        coefficients: ArrayLike | None = None,
        domain: tuple[float, float] | None = None,
    ):
        self.kernel = kernel
        self.n_modes = int(n_modes)
        self.pad_years = float(pad_years)
        if coefficients is None:
            self.coefficients = None
        else:
            c = jnp.asarray(coefficients, dtype=jnp.float64)
            if c.shape != (self.n_modes,):
                raise ValueError(
                    f"coefficients must have shape ({self.n_modes},), got {c.shape}."
                )
            if domain is None:
                raise ValueError(
                    "GPProduction requires explicit domain=(a, b) when coefficients are provided."
                )
            self.coefficients = c
        self.domain = (
            (float(domain[0]), float(domain[1])) if domain is not None else None
        )

    @property
    def is_linear_marginalizable(self) -> bool:
        return self.coefficients is None

    def _default_slug_name(self) -> str:
        return "gp"

    def delta_q(
        self,
        t: ArrayLike,
        q_std: ArrayLike = DEFAULT_Q_STD,
    ) -> jax.Array:
        del q_std
        t_arr = jnp.asarray(t, dtype=jnp.float64)
        if self.coefficients is None:
            return jnp.zeros_like(t_arr, dtype=jnp.float64)
        assert self.domain is not None
        a, b = self.domain
        omegas, psd, half_len = solin_sarkka_eigenpairs(
            a, b, self.n_modes, self.kernel
        )
        psi = (1.0 / math.sqrt(half_len)) * jnp.sin(
            (t_arr[..., None] - a) * omegas
        )
        weights = jnp.sqrt(jnp.maximum(psd, 0.0)) * self.coefficients
        return jnp.sum(psi * weights, axis=-1)


class LogGPProduction(ProductionTerm):
    """Strictly positive log-Gaussian Process production ``Q(t) = Q_std exp(f(t) - 0.5 sigma_f^2)``.

    Includes the exact Itô / log-normal variance correction ``-0.5 * Var[f(t)]`` so
    that ``E_{z ~ N(0, I)}[Q(t)] == Q_std`` identically across all epochs ``t`` and
    hyperparameters ``sigma_f``, preventing upward amplitude drift.
    """

    kernel: Any
    coefficients: jax.Array
    domain: tuple[float, float] = eqx.field(static=True)
    n_modes: int = eqx.field(static=True)
    exact_basis_variance: bool = eqx.field(static=True)

    def __init__(
        self,
        kernel: Any,
        coefficients: ArrayLike,
        *,
        domain: tuple[float, float],
        exact_basis_variance: bool = True,
    ):
        c = jnp.asarray(coefficients, dtype=jnp.float64)
        if c.ndim != 1 or c.shape[0] < 1:
            raise ValueError(
                f"LogGPProduction coefficients must be a non-empty 1D array, got {c.shape}."
            )
        a, b = float(domain[0]), float(domain[1])
        if not (b > a):
            raise ValueError(f"domain=(a, b) requires b > a, got {domain}.")
        self.kernel = kernel
        self.coefficients = c
        self.n_modes = int(c.shape[0])
        self.domain = (a, b)
        self.exact_basis_variance = bool(exact_basis_variance)

    def _default_slug_name(self) -> str:
        return "log_gp"

    def log_field_and_variance(self, t: ArrayLike) -> tuple[jax.Array, jax.Array]:
        """Evaluate ``(f(t), sigma_f^2(t))`` on the Solin–Särkkä basis."""
        t_arr = jnp.asarray(t, dtype=jnp.float64)
        a, b = self.domain
        omegas, psd, half_len = solin_sarkka_eigenpairs(
            a, b, self.n_modes, self.kernel
        )
        psi = (1.0 / math.sqrt(half_len)) * jnp.sin(
            (t_arr[..., None] - a) * omegas
        )
        pos_psd = jnp.maximum(psd, 0.0)
        f_t = jnp.sum(psi * (jnp.sqrt(pos_psd) * self.coefficients), axis=-1)
        if self.exact_basis_variance:
            var_t = jnp.sum( jnp.square(psi) * pos_psd, axis=-1 )
        else:
            zero = jnp.zeros((), dtype=jnp.float64)
            var_scalar = jnp.asarray(self.kernel.evaluate(zero, zero), dtype=jnp.float64)
            var_t = jnp.broadcast_to(var_scalar, t_arr.shape)
        return f_t, var_t

    def delta_q(
        self,
        t: ArrayLike,
        q_std: ArrayLike = DEFAULT_Q_STD,
    ) -> jax.Array:
        q0 = jnp.asarray(q_std, dtype=jnp.float64)
        f_t, var_t = self.log_field_and_variance(t)
        return q0 * jnp.expm1(f_t - 0.5 * var_t)


class GriddedProduction(ProductionTerm):
    """Piecewise-linear (first-order-hold) production trajectory on a sorted grid ``t_grid``."""

    t_grid: jax.Array
    values: jax.Array
    is_delta: bool = eqx.field(static=True)

    def __init__(
        self,
        t_grid: ArrayLike,
        values: ArrayLike,
        *,
        is_delta: bool = False,
    ):
        tg = jnp.asarray(t_grid, dtype=jnp.float64)
        vals = jnp.asarray(values, dtype=jnp.float64)
        if tg.ndim != 1 or tg.shape[0] < 2:
            raise ValueError(
                f"t_grid must be a 1D array with at least 2 knots, got {tg.shape}."
            )
        if vals.shape != tg.shape:
            raise ValueError(
                f"values shape {vals.shape} must match t_grid shape {tg.shape}."
            )
        self.t_grid = tg
        self.values = vals
        self.is_delta = bool(is_delta)

    def _default_slug_name(self) -> str:
        return "gridded"

    def delta_q(
        self,
        t: ArrayLike,
        q_std: ArrayLike = DEFAULT_Q_STD,
    ) -> jax.Array:
        t_arr = jnp.asarray(t, dtype=jnp.float64)
        interp_val = jnp.interp(t_arr, self.t_grid, self.values)
        if self.is_delta:
            return interp_val
        return interp_val - jnp.asarray(q_std, dtype=jnp.float64)
