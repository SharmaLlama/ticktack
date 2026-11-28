"""Unit and mathematical verification tests for Package 6: ``_FieldBase`` & field UX."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ticktack import _utils as tt_utils
from ticktack import fields as tt_fields
from ticktack.cbm import SubspaceCBM
from ticktack.data import Dataset, TreeRingSeries
from ticktack.fields import (
    DCTResponseOperator,
    GaussianField,
    LinearizedDCTOperator,
    LogGaussianField,
    MaternSpectrum,
    SquaredExpSpectrum,
    _FieldBase,
    _compute_cbm_digest,
    _compute_dataset_digest,
    _compute_template_digest,
    _safe_sqrt,
    build_dct_response_operator,
    build_linearized_dct_operator,
    error_scale,
    field_spectrum,
    fit_ift_field,
    laplace_samples,
    log_evidence,
    predict_field_d14c,
    stationary_cbm_spectral_cov,
)
from ticktack.model import RadiocarbonModel
from ticktack.presets import load_preset
from ticktack.production import ProductionTerm, SteadyProduction


def _make_synthetic_problem(
    n_grid: int = 24,
) -> tuple[RadiocarbonModel, Dataset, DCTResponseOperator]:
    """Build a small deterministic radiocarbon dataset and precomputed DCT operator."""
    cbm = load_preset("Brehm21")
    years = np.arange(765.0, 785.0, 1.0, dtype=np.float64)
    rng = np.random.default_rng(2026)
    d14c = 2.5 * np.sin(0.35 * (years - 765.0)) + rng.normal(0.0, 0.4, size=years.shape)
    sigma = np.full_like(years, 1.3)
    series = TreeRingSeries(
        years,
        d14c,
        sigma,
        hemisphere="north",
        name="syn_nh",
    )
    ds = Dataset((series,), name="syn_pkg6")
    t_start, t_end = 755.0, 786.0
    op = build_dct_response_operator(cbm, ds, t_start=t_start, t_end=t_end, n_grid=n_grid)
    model = RadiocarbonModel(cbm=cbm, production=SteadyProduction(1.76))
    return model, ds, op


def _make_subspace_linearized_operator(
    ds: Dataset,
    t_start: float = 755.0,
    t_end: float = 786.0,
    n_grid: int = 20,
) -> LinearizedDCTOperator:
    """Build a 1-mode SubspaceCBM and its corresponding LinearizedDCTOperator."""
    bal_cbm = load_preset("Brehm21", kind="balanced")
    base_lr = np.asarray(bal_cbm.log_rates_free, dtype=np.float64)
    basis = np.zeros((base_lr.shape[0], 1), dtype=np.float64)
    basis[0, 0] = 0.05
    sub_cbm = SubspaceCBM(
        topology=bal_cbm.topology,
        base_log_rates_free=base_lr,
        basis=basis,
        base_log_total_mass=float(bal_cbm.log_total_mass),
        alpha=jnp.asarray([0.15], dtype=jnp.float64),
        prod_weights=bal_cbm.prod_weights,
        tie_matrix=bal_cbm.tie_matrix_rows,
        decay_rate=bal_cbm.decay_rate,
    )
    return build_linearized_dct_operator(
        sub_cbm,
        ds,
        t_start=t_start,
        t_end=t_end,
        n_grid=n_grid,
    )


def _central_fd_grad(fn, x: jax.Array, eps: float = 5e-6) -> np.ndarray:
    """Compute central finite-difference gradient of scalar ``fn(x)``."""
    x_np = np.asarray(x, dtype=np.float64)
    flat = x_np.reshape(-1)
    grad = np.zeros_like(flat)
    for idx in range(flat.size):
        step = np.zeros_like(flat)
        step[idx] = eps
        f_plus = float(fn(jnp.asarray((flat + step).reshape(x_np.shape), dtype=jnp.float64)))
        f_minus = float(fn(jnp.asarray((flat - step).reshape(x_np.shape), dtype=jnp.float64)))
        grad[idx] = (f_plus - f_minus) / (2.0 * eps)
    return grad.reshape(x_np.shape)


# =============================================================================
# 1. Deduplication, Integer Order Regression, & Class Hierarchy
# =============================================================================


def test_utils_deduplication_and_digest_byte_identity() -> None:
    assert not hasattr(tt_fields, "_concrete_np")
    assert tt_fields._concrete is tt_utils._concrete
    assert tt_fields._safe_psd_root is tt_utils._safe_psd_root

    model, ds, _ = _make_synthetic_problem(n_grid=16)
    dyadic64 = np.linspace(0.5, 1.5, 17, dtype=np.float64)
    dyadic32 = dyadic64.astype(np.float32)
    assert _compute_template_digest(dyadic64) == _compute_template_digest(dyadic32)
    assert _compute_template_digest(dyadic64.tolist()) == _compute_template_digest(dyadic64)
    assert _compute_template_digest(None) == "none"

    d_ds = _compute_dataset_digest(model.cbm, ds)
    assert len(d_ds) == 64

    d_cbm = _compute_cbm_digest(model.cbm)
    assert len(d_cbm) == 64

    traced_digests: list[str] = []

    @jax.jit
    def _traced_probe(arr: jax.Array) -> jax.Array:
        traced_digests.append(_compute_template_digest(arr))
        return arr

    _traced_probe(jnp.ones(16, dtype=jnp.float64))
    assert traced_digests == [""]


def test_integer_order_field_spectrum_and_spectral_cov() -> None:
    """Integer Matérn orders (1, 2, 3, np.int64) must not crash np.isposinf."""
    for ord_int in (1, 2, 3, np.int64(2)):
        spec_int = field_spectrum(64, 0.5, 0.3, 5.0, order=ord_int)
        spec_flt = field_spectrum(64, 0.5, 0.3, 5.0, order=float(ord_int))
        np.testing.assert_allclose(spec_int, spec_flt, rtol=0.0, atol=0.0)

    cbm = load_preset("Brehm21")
    kern_int = MaternSpectrum(sigma=0.35, length_yr=5.0, order=2.0).set("order", 2)
    kern_flt = MaternSpectrum(sigma=0.35, length_yr=5.0, order=2.0)
    cov_int, root_int = stationary_cbm_spectral_cov(
        cbm.matrix_A(), cbm.production_vector(), kern_int
    )
    cov_flt, root_flt = stationary_cbm_spectral_cov(
        cbm.matrix_A(), cbm.production_vector(), kern_flt
    )
    np.testing.assert_allclose(cov_int, cov_flt, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(root_int, root_flt, rtol=1e-14, atol=1e-14)


def test_field_base_hierarchy_and_abstract_stubs() -> None:
    assert issubclass(_FieldBase, ProductionTerm)
    assert issubclass(GaussianField, _FieldBase)
    assert issubclass(LogGaussianField, _FieldBase)

    class _BareField(_FieldBase):
        def __init__(self, n_grid: int = 8) -> None:
            self.latent = jnp.zeros(n_grid, dtype=jnp.float64)
            self.sigma = jnp.asarray(0.3, dtype=jnp.float64)
            self.length_yr = jnp.asarray(5.0, dtype=jnp.float64)
            self.order = jnp.asarray(2.0, dtype=jnp.float64)
            self.kernel = None
            self.template = None
            self.t_start = 755.0
            self.t_end = 786.0
            self.zero_mode = False

    _, _, op = _make_synthetic_problem(n_grid=8)
    bare = _BareField(n_grid=8)
    assert bare._operator_template() is None
    with pytest.raises(NotImplementedError):
        bare._prior_variance_scale()
    with pytest.raises(NotImplementedError):
        bare.excess(op)
    with pytest.raises(NotImplementedError):
        bare._evaluate_grid_and_q()
    with pytest.raises(NotImplementedError):
        bare._evaluate_draws(op, jnp.zeros((2, 8), dtype=jnp.float64))


# =============================================================================
# 2. Ergonomic Constructors: .zeros(...) and .from_operator(...)
# =============================================================================


def test_zeros_constructors_and_defaults() -> None:
    gf = GaussianField.zeros(760.0, 800.0, n_grid=48, sigma=0.45, length_yr=7.5)
    gf_ref = GaussianField(
        jnp.zeros(48, dtype=jnp.float64), 760.0, 800.0, sigma=0.45, length_yr=7.5
    )
    assert isinstance(gf, GaussianField)
    assert gf.zero_mode is False
    assert gf.n_grid == 48
    assert gf.t_start == 760.0
    assert gf.t_end == 800.0
    np.testing.assert_array_equal(gf.latent, jnp.zeros(48, dtype=jnp.float64))
    np.testing.assert_allclose(gf.power_spectrum(), gf_ref.power_spectrum(), rtol=0.0, atol=0.0)

    # Accepts numpy integer types via operator.index
    gf_np_int = GaussianField.zeros(760.0, 800.0, n_grid=np.int64(32))
    assert gf_np_int.n_grid == 32

    tmpl = jnp.linspace(0.6, 1.4, 32, dtype=jnp.float64)
    kern = SquaredExpSpectrum(sigma=0.25, length_yr=4.0, zero_mode=True)
    lgf = LogGaussianField.zeros(
        760.0,
        800.0,
        n_grid=32,
        kernel=kern,
        q_mean=1.95,
        template=tmpl,
        ito_corrected=False,
    )
    assert isinstance(lgf, LogGaussianField)
    assert lgf.zero_mode is True
    assert lgf.ito_corrected is False
    assert float(lgf.q_mean) == pytest.approx(1.95)
    assert lgf.kernel is kern
    np.testing.assert_array_equal(lgf.template, tmpl)
    np.testing.assert_allclose(lgf.evaluate_q_grid(), 1.95 * tmpl, rtol=1e-15, atol=1e-15)

    # Default ito_corrected=True and n_grid=128
    lgf_default = LogGaussianField.zeros(760.0, 800.0)
    assert lgf_default.n_grid == 128
    assert lgf_default.zero_mode is True
    assert lgf_default.ito_corrected is True

    # Invalid n_grid values raise ValueError or TypeError (floats/bools rejected)
    for bad_n in (1, 0, -5):
        with pytest.raises(ValueError, match="n_grid must be >= 2"):
            GaussianField.zeros(760.0, 800.0, n_grid=bad_n)
        with pytest.raises(ValueError, match="n_grid must be >= 2"):
            LogGaussianField.zeros(760.0, 800.0, n_grid=bad_n)
    for bad_type in (64.7, True, False):
        with pytest.raises(TypeError):
            GaussianField.zeros(760.0, 800.0, n_grid=bad_type)  # type: ignore[arg-type]


def test_from_operator_dct_validation_and_defaults() -> None:
    model, ds, op = _make_synthetic_problem(n_grid=24)
    lin_op = _make_subspace_linearized_operator(ds, t_start=752.0, t_end=788.0, n_grid=20)
    tmpl = jnp.linspace(0.5, 1.5, op.n_grid, dtype=jnp.float64)
    op_tmpl = build_dct_response_operator(
        model.cbm,
        ds,
        t_start=op.t_start,
        t_end=op.t_end,
        n_grid=op.n_grid,
        template=tmpl,
    )

    kern = MaternSpectrum(sigma=0.35, length_yr=6.0, order=2.5, zero_mode=False)
    gf = GaussianField.from_operator(op, kernel=kern)
    assert isinstance(gf, GaussianField)
    assert gf.n_grid == op.n_grid
    assert gf.t_start == op.t_start
    assert gf.t_end == op.t_end
    assert gf.zero_mode is False
    assert gf.kernel is kern
    np.testing.assert_array_equal(gf.latent, jnp.zeros(op.n_grid, dtype=jnp.float64))

    # Matching template on GaussianField succeeds; mismatches fail fast via _validate_operator
    gf_tmpl = GaussianField.from_operator(op_tmpl, template=tmpl)
    np.testing.assert_array_equal(gf_tmpl.template, tmpl)

    with pytest.raises(ValueError, match="template_digest"):
        GaussianField.from_operator(op_tmpl)
    with pytest.raises(ValueError, match="template_digest"):
        GaussianField.from_operator(op, template=tmpl)
    with pytest.raises(ValueError, match="template_digest"):
        GaussianField.from_operator(op_tmpl, template=tmpl * 1.1)

    # LogGaussianField accepts un-templated DCTResponseOperator (even with log-space template)
    lgf = LogGaussianField.from_operator(
        op,
        sigma=0.22,
        length_yr=4.5,
        q_mean=1.85,
        template=tmpl,
        ito_corrected=False,
    )
    assert isinstance(lgf, LogGaussianField)
    assert lgf.n_grid == op.n_grid
    assert lgf.t_start == op.t_start
    assert lgf.t_end == op.t_end
    assert lgf.zero_mode is True
    assert lgf.ito_corrected is False
    assert float(lgf.q_mean) == pytest.approx(1.85)
    np.testing.assert_array_equal(lgf.template, tmpl)

    # Both classes reject LinearizedDCTOperator and invalid operator types; LogGaussianField rejects templated op
    with pytest.raises(TypeError, match="DCTResponseOperator"):
        GaussianField.from_operator(lin_op)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="DCTResponseOperator"):
        LogGaussianField.from_operator(lin_op)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="template_digest"):
        LogGaussianField.from_operator(op_tmpl)

    with pytest.raises(TypeError, match="DCTResponseOperator"):
        GaussianField.from_operator("not_an_operator")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="DCTResponseOperator"):
        LogGaussianField.from_operator(None)  # type: ignore[arg-type]


# =============================================================================
# 3. Operator Geometry & Template Validation on Operator-Consuming Methods
# =============================================================================


def test_operator_geometry_and_template_validation_on_methods() -> None:
    model, ds, op = _make_synthetic_problem(n_grid=16)
    tmpl = jnp.linspace(0.5, 1.5, op.n_grid, dtype=jnp.float64)
    op_tmpl = build_dct_response_operator(
        model.cbm, ds, t_start=op.t_start, t_end=op.t_end, n_grid=op.n_grid, template=tmpl
    )
    op_shifted = build_dct_response_operator(
        model.cbm, ds, t_start=op.t_start - 2.0, t_end=op.t_end - 2.0, n_grid=op.n_grid
    )
    op_ngrid = build_dct_response_operator(
        model.cbm, ds, t_start=op.t_start, t_end=op.t_end, n_grid=op.n_grid + 2
    )
    lin_op = _make_subspace_linearized_operator(ds, t_start=op.t_start, t_end=op.t_end, n_grid=16)

    z = jnp.linspace(-0.3, 0.4, op.n_grid, dtype=jnp.float64)
    gf = GaussianField.from_operator(op).with_latent(z)
    gf_tmpl = GaussianField.from_operator(op_tmpl, template=tmpl).with_latent(z)
    lgf = LogGaussianField.from_operator(op, template=tmpl).with_latent(z)

    inv_sig = jnp.ones(ds.n_obs, dtype=jnp.float64)
    inv_sig_np = np.ones(ds.n_obs, dtype=np.float64)
    d_obs = np.asarray(ds.d14c_flat, dtype=np.float64)
    base_pred = np.asarray(op.ss_d14c, dtype=np.float64)
    z_draws = jnp.zeros((3, op.n_grid), dtype=jnp.float64)

    # Shifted calendar window with identical n_grid fails fast
    with pytest.raises(ValueError, match="Operator grid"):
        gf.excess(op_shifted)
    with pytest.raises(ValueError, match="Operator grid"):
        lgf.excess(op_shifted)
    with pytest.raises(ValueError, match="Operator grid"):
        gf._evaluate_draws(op_shifted, z_draws)
    with pytest.raises(ValueError, match="Operator grid"):
        gf._whitened_jacobian(op_shifted, inv_sig)
    with pytest.raises(ValueError, match="Operator grid"):
        gf._linearized_error_scale_inputs(op_shifted, d_obs, base_pred, inv_sig_np)

    # Mismatched n_grid fails fast
    with pytest.raises(ValueError, match="Operator grid"):
        gf.excess(op_ngrid)

    # Mismatched template_digest fails fast in both directions
    with pytest.raises(ValueError, match="template_digest"):
        gf_tmpl.excess(op)
    with pytest.raises(ValueError, match="template_digest"):
        gf_tmpl._evaluate_draws(op, z_draws)
    with pytest.raises(ValueError, match="template_digest"):
        gf_tmpl._whitened_jacobian(op, inv_sig)
    with pytest.raises(ValueError, match="template_digest"):
        gf_tmpl._linearized_error_scale_inputs(op, d_obs, base_pred, inv_sig_np)
    with pytest.raises(ValueError, match="template_digest"):
        gf.excess(op_tmpl)
    with pytest.raises(ValueError, match="template_digest"):
        lgf.excess(op_tmpl)

    # Non-DCTResponseOperator rejected with TypeError
    with pytest.raises(TypeError, match="DCTResponseOperator"):
        gf.excess(lin_op)  # type: ignore[arg-type]

    # Traced template under jax.jit propagates dynamic values & gradients accurately
    @jax.jit
    def _jit_excess(t_arr: jax.Array) -> jax.Array:
        return GaussianField(z, op.t_start, op.t_end, template=t_arr).excess(op_tmpl)

    @jax.jit
    def _jit_draws(t_arr: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        return GaussianField(z, op.t_start, op.t_end, template=t_arr)._evaluate_draws(
            op_tmpl, z_draws
        )

    np.testing.assert_allclose(_jit_excess(tmpl), gf_tmpl.excess(op_tmpl), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(
        _jit_excess(2.3 * tmpl), 2.3 * gf_tmpl.excess(op_tmpl), rtol=1e-14, atol=1e-14
    )
    dq_jit, _, exc_jit = _jit_draws(2.3 * tmpl)
    np.testing.assert_allclose(exc_jit, dq_jit @ op_tmpl.w_grid.T, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(
        exc_jit, 2.3 * gf_tmpl._evaluate_draws(op_tmpl, z_draws)[2], rtol=1e-14, atol=1e-14
    )

    # Traced template paired with un-templated operator still raises ValueError inside @jax.jit
    @jax.jit
    def _jit_from_op_bad(t_arr: jax.Array) -> jax.Array:
        return GaussianField.from_operator(op, template=t_arr).latent

    @jax.jit
    def _jit_excess_bad(t_arr: jax.Array) -> jax.Array:
        return GaussianField(z, op.t_start, op.t_end, template=t_arr).excess(op)

    with pytest.raises(ValueError, match="template_digest"):
        _jit_from_op_bad(tmpl)
    with pytest.raises(ValueError, match="template_digest"):
        _jit_excess_bad(tmpl)


# =============================================================================
# 4. Polymorphic Method Parity at Asymmetric Non-Zero Inputs
# =============================================================================


@pytest.mark.parametrize("use_template", [False, True])
def test_gaussian_field_polymorphic_methods_parity(use_template: bool) -> None:
    model, ds, _ = _make_synthetic_problem(n_grid=20)
    t_start, t_end, n_grid = 755.0, 786.0, 20
    tmpl = jnp.linspace(0.4, 1.7, n_grid, dtype=jnp.float64) if use_template else None
    op = build_dct_response_operator(
        model.cbm, ds, t_start=t_start, t_end=t_end, n_grid=n_grid, template=tmpl
    )

    rng = np.random.default_rng(101 + int(use_template))
    z_vec = jnp.asarray(rng.normal(0.25, 0.9, size=(n_grid,)), dtype=jnp.float64)
    gf = GaussianField.from_operator(
        op, sigma=0.47, length_yr=6.3, order=2.4, template=tmpl
    ).with_latent(z_vec)

    if use_template:
        assert gf._operator_template() is gf.template
        np.testing.assert_allclose(gf._prior_variance_scale(), tmpl[0] ** 2, rtol=1e-15, atol=1e-15)
    else:
        assert gf._operator_template() is None
        np.testing.assert_allclose(gf._prior_variance_scale(), 1.0, rtol=1e-15, atol=1e-15)

    q_std = 1.91
    exc = gf.excess(op, q_std=q_std)
    exc_ref = op.w_dct @ (_safe_sqrt(gf.power_spectrum()) * z_vec)
    np.testing.assert_allclose(exc, exc_ref, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(exc, op.w_grid @ gf.evaluate_grid(), rtol=1e-13, atol=1e-13)

    dq, q = gf._evaluate_grid_and_q(q_std=q_std)
    np.testing.assert_allclose(dq, gf.evaluate_grid(), rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(q, q_std + dq, rtol=1e-15, atol=1e-15)

    z_draws = jnp.asarray(rng.normal(-0.1, 0.8, size=(7, n_grid)), dtype=jnp.float64)
    dq_d, q_d, exc_d = gf._evaluate_draws(op, z_draws, q_std=q_std)
    for i in range(z_draws.shape[0]):
        fi = gf.with_latent(z_draws[i])
        dqi, qi = fi._evaluate_grid_and_q(q_std=q_std)
        np.testing.assert_allclose(dq_d[i], dqi, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(q_d[i], qi, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(exc_d[i], fi.excess(op, q_std=q_std), rtol=1e-14, atol=1e-14)

    inv_sig = jnp.asarray(1.0 / (0.8 + 0.3 * np.arange(ds.n_obs) / ds.n_obs), dtype=jnp.float64)
    jac = gf._whitened_jacobian(op, inv_sig, q_std=q_std)
    jac_ad = jax.jacfwd(lambda z: inv_sig * gf.with_latent(z).excess(op, q_std=q_std))(z_vec)
    np.testing.assert_allclose(jac, jac_ad, rtol=1e-14, atol=1e-14)

    d_obs = np.asarray(ds.d14c_flat, dtype=np.float64)
    base_pred = np.asarray(op.ss_d14c, dtype=np.float64) + 1.25
    inv_sig_np = np.asarray(inv_sig, dtype=np.float64)
    x_raw, j_raw = gf._linearized_error_scale_inputs(
        op, d_obs, base_pred, inv_sig_np, q_std=q_std
    )
    np.testing.assert_allclose(x_raw, inv_sig_np * (d_obs - base_pred), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(j_raw, np.asarray(jac), rtol=1e-14, atol=1e-14)
    r_data = inv_sig_np * (d_obs - base_pred - np.asarray(exc))
    np.testing.assert_allclose(r_data + j_raw @ np.asarray(z_vec), x_raw, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("use_template", [False, True])
@pytest.mark.parametrize("ito_corrected", [True, False])
def test_log_gaussian_field_polymorphic_methods_parity(
    use_template: bool, ito_corrected: bool
) -> None:
    _, ds, op = _make_synthetic_problem(n_grid=20)
    n_grid = op.n_grid
    tmpl = jnp.linspace(0.35, 1.85, n_grid, dtype=jnp.float64) if use_template else None

    rng = np.random.default_rng(202 + 2 * int(use_template) + int(ito_corrected))
    z_vec = jnp.asarray(rng.normal(-0.2, 0.7, size=(n_grid,)), dtype=jnp.float64)
    q_mean = 2.13
    q_std = 1.79
    sigma_val = 0.41
    lgf = LogGaussianField.from_operator(
        op,
        sigma=sigma_val,
        length_yr=5.7,
        order=2.2,
        q_mean=q_mean,
        template=tmpl,
        mean_floor=0.05,
        ito_corrected=ito_corrected,
    ).with_latent(z_vec)

    assert lgf._operator_template() is None

    sq = sigma_val**2
    expected_vscale = (q_mean**2) * (math.expm1(sq) / sq)
    if use_template:
        expected_vscale *= float(max(float(tmpl[0]), 0.05) ** 2)
    np.testing.assert_allclose(
        lgf._prior_variance_scale(), expected_vscale, rtol=1e-14, atol=1e-14
    )

    exc = lgf.excess(op, q_std=q_std)
    exc_ref = op.w_grid @ (lgf.evaluate_q_grid() - q_std)
    np.testing.assert_allclose(exc, exc_ref, rtol=1e-14, atol=1e-14)

    dq, q = lgf._evaluate_grid_and_q(q_std=q_std)
    np.testing.assert_allclose(q, lgf.evaluate_q_grid(), rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(dq, q - q_std, rtol=1e-15, atol=1e-15)

    z_draws = jnp.asarray(rng.normal(0.1, 0.6, size=(6, n_grid)), dtype=jnp.float64)
    dq_d, q_d, exc_d = lgf._evaluate_draws(op, z_draws, q_std=q_std)
    for i in range(z_draws.shape[0]):
        fi = lgf.with_latent(z_draws[i])
        dqi, qi = fi._evaluate_grid_and_q(q_std=q_std)
        np.testing.assert_allclose(dq_d[i], dqi, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(q_d[i], qi, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(exc_d[i], fi.excess(op, q_std=q_std), rtol=1e-14, atol=1e-14)

    inv_sig = jnp.asarray(1.0 / (0.9 + 0.4 * np.arange(ds.n_obs) / ds.n_obs), dtype=jnp.float64)
    jac = lgf._whitened_jacobian(op, inv_sig, q_std=q_std)
    jac_ref = jax.jacfwd(
        lambda z: inv_sig
        * (op.ss_d14c + op.w_grid @ lgf.with_latent(z).evaluate_grid(q_std=q_std) - ds.d14c_flat)
    )(z_vec)
    np.testing.assert_allclose(jac, jac_ref, rtol=1e-14, atol=1e-14)

    d_obs = np.asarray(ds.d14c_flat, dtype=np.float64)
    base_pred = np.asarray(op.ss_d14c, dtype=np.float64) - 0.75
    inv_sig_np = np.asarray(inv_sig, dtype=np.float64)
    x_raw, j_raw = lgf._linearized_error_scale_inputs(
        op, d_obs, base_pred, inv_sig_np, q_std=q_std
    )
    r_data_ref = inv_sig_np * (d_obs - base_pred - np.asarray(exc_ref))
    np.testing.assert_allclose(j_raw, np.asarray(jac_ref), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(x_raw, r_data_ref + j_raw @ np.asarray(z_vec), rtol=1e-14, atol=1e-14)


# =============================================================================
# 5. Autodiff vs Central Finite Differences (latent & kernel hyperparameters)
# =============================================================================


@pytest.mark.parametrize("field_cls", [GaussianField, LogGaussianField])
@pytest.mark.parametrize("use_template", [False, True])
def test_autodiff_excess_and_whitened_jacobian_vs_finite_diff(
    field_cls: type[_FieldBase], use_template: bool
) -> None:
    model, ds, _ = _make_synthetic_problem(n_grid=14)
    t_start, t_end, n_grid = 755.0, 786.0, 14
    tmpl = jnp.linspace(0.5, 1.6, n_grid, dtype=jnp.float64) if use_template else None
    op_tmpl = tmpl if field_cls is GaussianField else None
    op = build_dct_response_operator(
        model.cbm, ds, t_start=t_start, t_end=t_end, n_grid=n_grid, template=op_tmpl
    )

    rng = np.random.default_rng(303 + int(use_template))
    z0 = jnp.asarray(rng.normal(0.15, 0.65, size=(n_grid,)), dtype=jnp.float64)
    inv_sig = jnp.asarray(1.0 / (0.7 + 0.5 * np.arange(ds.n_obs) / ds.n_obs), dtype=jnp.float64)
    w_obs = jnp.asarray(rng.normal(0.0, 1.0, size=(ds.n_obs,)), dtype=jnp.float64)
    w_mat = jnp.asarray(rng.normal(0.0, 1.0, size=(ds.n_obs, n_grid)), dtype=jnp.float64)
    q_std = 1.88

    def _build(
        z: jax.Array,
        sigma: jax.Array,
        length_yr: jax.Array,
        template: jax.Array | None,
    ) -> _FieldBase:
        kwargs: dict[str, object] = {
            "sigma": sigma,
            "length_yr": length_yr,
            "order": 2.3,
            "template": template,
        }
        if field_cls is LogGaussianField:
            kwargs["q_mean"] = 2.05
            kwargs["ito_corrected"] = True
        return field_cls(z, t_start, t_end, **kwargs)  # type: ignore[arg-type]

    s0 = jnp.asarray(0.43, dtype=jnp.float64)
    l0 = jnp.asarray(5.4, dtype=jnp.float64)

    def _unpack(params: jax.Array) -> _FieldBase:
        t_arg = params[n_grid + 2 :] if use_template else None
        return _build(params[:n_grid], params[n_grid], params[n_grid + 1], t_arg)

    def _exc_scalar(params: jax.Array) -> jax.Array:
        return jnp.dot(w_obs, _unpack(params).excess(op, q_std=q_std))

    def _jac_scalar(params: jax.Array) -> jax.Array:
        return jnp.sum(w_mat * _unpack(params)._whitened_jacobian(op, inv_sig, q_std=q_std))

    parts = [z0, jnp.array([s0, l0], dtype=jnp.float64)]
    if use_template:
        assert tmpl is not None
        parts.append(tmpl)
    p0 = jnp.concatenate(parts)

    g_exc_ad = np.asarray(jax.grad(_exc_scalar)(p0), dtype=np.float64)
    g_exc_fd = _central_fd_grad(jax.jit(_exc_scalar), p0, eps=5e-6)
    np.testing.assert_allclose(g_exc_ad, g_exc_fd, rtol=1e-6, atol=1e-6)

    g_jac_ad = np.asarray(jax.grad(_jac_scalar)(p0), dtype=np.float64)
    g_jac_fd = _central_fd_grad(jax.jit(_jac_scalar), p0, eps=5e-6)
    np.testing.assert_allclose(g_jac_ad, g_jac_fd, rtol=1e-6, atol=1e-6)

    if use_template:
        # Ensure template gradients are strictly non-trivial and match analytical w_grid oracle
        assert np.linalg.norm(g_exc_ad[n_grid + 2 :]) > 1e-3
        assert np.linalg.norm(g_jac_ad[n_grid + 2 :]) > 1e-3
        if field_cls is GaussianField:
            f0 = _build(z0, s0, l0, None)
            eta0 = f0.evaluate_grid()
            expected_t_grad = np.asarray((op.w_grid.T @ w_obs) * eta0, dtype=np.float64)
            np.testing.assert_allclose(
                g_exc_ad[n_grid + 2 :], expected_t_grad, rtol=1e-13, atol=1e-13
            )


# =============================================================================
# 6. End-to-End Pipelines (fit_ift_field, log_evidence, laplace_samples, error_scale)
# =============================================================================


@pytest.mark.parametrize(
    ("kind", "use_template", "ito_corrected"),
    [
        ("gaussian", False, False),
        ("gaussian", True, False),
        ("log_gaussian", False, True),
        ("log_gaussian", True, False),
    ],
)
def test_end_to_end_ift_pipelines_with_new_constructors(
    kind: str, use_template: bool, ito_corrected: bool
) -> None:
    base_model, ds, op_plain = _make_synthetic_problem(n_grid=20)
    n_grid = op_plain.n_grid
    tmpl = jnp.linspace(0.6, 1.4, n_grid, dtype=jnp.float64) if use_template else None

    if kind == "gaussian":
        op = (
            build_dct_response_operator(
                base_model.cbm,
                ds,
                t_start=op_plain.t_start,
                t_end=op_plain.t_end,
                n_grid=n_grid,
                template=tmpl,
            )
            if use_template
            else op_plain
        )
        field: _FieldBase = GaussianField.from_operator(
            op, sigma=0.38, length_yr=5.2, template=tmpl
        )
    else:
        op = op_plain
        field = LogGaussianField.from_operator(
            op,
            sigma=0.28,
            length_yr=5.2,
            q_mean=1.92,
            template=tmpl,
            ito_corrected=ito_corrected,
        )

    model = RadiocarbonModel(
        cbm=base_model.cbm,
        production=SteadyProduction(1.76) + field,
        q_std=1.76,
    )

    fit_res = fit_ift_field(model, ds, operator=op)
    assert fit_res.converged
    assert np.isfinite(float(fit_res.log_evidence))

    pred_from_model = predict_field_d14c(
        fit_res.model, ds, operator=op, baseline_d14c=fit_res.baseline_d14c
    )
    if fit_res.initial_state_mean is not None:
        pred_from_model = pred_from_model + op.init_state_basis @ fit_res.initial_state_mean
    np.testing.assert_allclose(pred_from_model, fit_res.d14c_pred, rtol=1e-12, atol=1e-12)

    ln_z = log_evidence(fit_res.model, ds, operator=op)
    np.testing.assert_allclose(float(ln_z), float(fit_res.log_evidence), rtol=1e-10, atol=1e-10)

    draws = laplace_samples(fit_res, ds, n_samples=16, rng_key=jax.random.PRNGKey(7))
    assert draws["latents"].shape == (16, n_grid)
    assert draws["delta_q"].shape == (16, n_grid)
    assert draws["q"].shape == (16, n_grid)
    assert draws["d14c_pred"].shape == (16, ds.n_obs)
    assert np.all(np.isfinite(draws["d14c_pred"]))
    if kind == "log_gaussian":
        assert np.all(np.asarray(draws["q"]) > 0.0)

    s_hat = error_scale(fit_res, ds)
    assert isinstance(s_hat, float)
    assert math.isfinite(s_hat) and s_hat > 0.0


def test_field_base_validation_and_error_cases() -> None:
    """Validate error handling shared by _FieldBase subclasses."""
    with pytest.raises(ValueError, match="1D with length >= 2"):
        GaussianField(jnp.zeros((2, 2)), 760.0, 800.0)
    with pytest.raises(ValueError, match="t_end > t_start"):
        GaussianField.zeros(800.0, 760.0, n_grid=8)
    with pytest.raises(TypeError, match="SpectralKernel or None"):
        GaussianField.zeros(760.0, 800.0, n_grid=8, kernel="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="template shape"):
        GaussianField.zeros(760.0, 800.0, n_grid=8, template=jnp.ones(5))
    with pytest.raises(ValueError, match="strictly positive peak"):
        LogGaussianField.zeros(760.0, 800.0, n_grid=8, template=jnp.zeros(8))
    with pytest.raises(ValueError, match="Expected latent shape"):
        GaussianField.zeros(760.0, 800.0, n_grid=8).with_latent(jnp.zeros(4))
