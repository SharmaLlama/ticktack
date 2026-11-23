"""Equivalence, gradient, conditioning, and JIT caching tests for unified ``FastLogLikeSpec``."""

from __future__ import annotations

import dataclasses
import gc
import inspect
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla
import tinygp.kernels as tgk

import ticktack as tt
import ticktack.fitting as ttf
import ticktack.likelihood as ttl
import ticktack.sampling as tts
from ticktack.likelihood import (
    FastLogLikeSpec,
    LinearMarginalConfig,
    ParamSpec,
    ParameterSpace,
    _build_marginal_design,
    _make_fast_unconstrained_loglike,
    _offset_only_cfg,
    joint_log_likelihood,
    marginal_log_likelihood,
)


def _central_fd_grad(fn, u: jax.Array, eps: float = 1e-6) -> np.ndarray:
    """4th-order Richardson-extrapolated central finite-difference gradient."""
    u_np = np.asarray(u, dtype=np.float64)
    g = np.zeros_like(u_np)
    for i in range(u_np.size):
        step = np.zeros_like(u_np)
        step[i] = eps
        fp1 = float(fn(jnp.asarray(u_np + step, dtype=jnp.float64)))
        fm1 = float(fn(jnp.asarray(u_np - step, dtype=jnp.float64)))
        fp2 = float(fn(jnp.asarray(u_np + 2.0 * step, dtype=jnp.float64)))
        fm2 = float(fn(jnp.asarray(u_np - 2.0 * step, dtype=jnp.float64)))
        g[i] = (8.0 * (fp1 - fm1) - (fp2 - fm2)) / (12.0 * eps)
    return g


def _scipy_qr_woodbury_loglike(
    resid: np.ndarray,
    design: np.ndarray,
    prior_mean: np.ndarray,
    prior_sd: np.ndarray,
    sigma_eff: np.ndarray,
) -> float:
    """Independent SciPy thin-QR Woodbury Gaussian marginal log-likelihood."""
    n_obs, k = design.shape
    x = (resid - design @ prior_mean) / sigma_eff
    half_ld_d = float(np.sum(np.log(sigma_eff)))
    if k == 0:
        return -0.5 * float(x @ x) - half_ld_d - 0.5 * n_obs * math.log(2.0 * math.pi)
    u_mat = (design * prior_sd[None, :]) / sigma_eff[:, None]
    aug = np.vstack([u_mat, np.eye(k, dtype=np.float64)])
    q_mat, r_up = sla.qr(aug, mode="economic")
    half_ld_u = float(np.sum(np.log(np.abs(np.diag(r_up)))))
    q1 = q_mat[:n_obs]
    q2 = q_mat[n_obs:]
    proj = q1.T @ x
    r_perp = x - q1 @ proj
    r_reg = q2 @ proj
    quad = float(r_perp @ r_perp + r_reg @ r_reg)
    return -0.5 * quad - half_ld_d - half_ld_u - 0.5 * n_obs * math.log(2.0 * math.pi)


@pytest.fixture(scope="module")
def multi_series_dataset() -> tt.Dataset:
    s_nh1 = tt.TreeRingSeries(
        years=np.arange(772.0, 780.0),
        d14c=np.array([-14.3, -13.7, -12.8, -1.9, 0.7, -1.6, -3.1, -4.4]),
        sigma_d14c=np.array([1.15, 1.42, 1.08, 1.53, 1.29, 1.61, 1.22, 1.37]),
        hemisphere="north",
        seasonal_weights="Apr-Sep",
        name="nh_lab_a",
    )
    s_nh2 = tt.TreeRingSeries(
        years=np.arange(772.0, 780.0),
        d14c=np.array([-15.2, -14.6, -13.4, -2.8, -0.4, -2.5, -3.9, -5.2]),
        sigma_d14c=np.array([1.31, 1.49, 1.19, 1.68, 1.44, 1.52, 1.28, 1.59]),
        hemisphere="north",
        seasonal_weights="May-Aug",
        name="nh_lab_b",
    )
    return tt.Dataset([s_nh1, s_nh2])


@eqx.filter_jit
def _fjit_spec_vg(
    spec: FastLogLikeSpec, u_vec: jax.Array
) -> tuple[jax.Array, jax.Array]:
    return jax.value_and_grad(spec)(u_vec)


@eqx.filter_jit
def _fjit_ref_vg(
    model: tt.RadiocarbonModel,
    dataset: tt.Dataset,
    ps: ParameterSpace,
    cfg: LinearMarginalConfig,
    jit_arr: jax.Array | None,
    is_joint: bool,
    u_vec: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    def _fn(u: jax.Array) -> jax.Array:
        m = ps.constrain(u, model)
        if is_joint:
            return joint_log_likelihood(m, dataset, cfg=cfg, sigma_jitter=jit_arr)
        return marginal_log_likelihood(m, dataset, cfg, sigma_jitter=jit_arr)

    return jax.value_and_grad(_fn)(u_vec)


def _assert_spec_equiv_and_grad(
    spec: FastLogLikeSpec,
    ref_fn,
    u_vec: jax.Array,
    *,
    expect_static: bool,
    expect_joint: bool,
    rtol: float = 1e-13,
    check_fd: bool = False,
    check_filter_jit: bool = False,
) -> None:
    assert isinstance(spec, FastLogLikeSpec)
    assert spec.is_static is expect_static
    assert spec.joint_mode is expect_joint
    assert (spec.static_cache is not None) is expect_static

    val_eag, grad_eag = jax.value_and_grad(spec)(u_vec)
    val_jit, grad_jit = jax.jit(jax.value_and_grad(spec))(u_vec)
    val_ref_eag, grad_ref_eag = jax.value_and_grad(ref_fn)(u_vec)
    val_ref_jit, grad_ref_jit = jax.jit(jax.value_and_grad(ref_fn))(u_vec)

    # Confirm evaluation is away from any zero-gradient fixed point
    assert float(jnp.linalg.norm(grad_ref_eag)) > 1e-2

    # Eager FastLogLikeSpec vs eager reference (rtol=1e-13)
    np.testing.assert_allclose(
        float(val_eag), float(val_ref_eag), rtol=rtol, atol=1e-13
    )
    np.testing.assert_allclose(
        np.asarray(grad_eag), np.asarray(grad_ref_eag), rtol=rtol, atol=1e-12
    )

    # jax.jit FastLogLikeSpec vs jax.jit reference (rtol=1e-13)
    np.testing.assert_allclose(
        float(val_jit), float(val_ref_jit), rtol=rtol, atol=1e-13
    )
    np.testing.assert_allclose(
        np.asarray(grad_jit), np.asarray(grad_ref_jit), rtol=rtol, atol=1e-12
    )

    if check_filter_jit:
        val_fjit, grad_fjit = _fjit_spec_vg(spec, u_vec)
        val_ref_fjit, grad_ref_fjit = _fjit_ref_vg(
            spec.base_model,
            spec.dataset,
            spec.parameter_space,
            spec.cfg_eff,
            spec.sigma_jitter_arr,
            spec.joint_mode,
            u_vec,
        )
        np.testing.assert_allclose(
            float(val_fjit), float(val_ref_fjit), rtol=rtol, atol=1e-13
        )
        np.testing.assert_allclose(
            np.asarray(grad_fjit), np.asarray(grad_ref_fjit), rtol=rtol, atol=1e-12
        )

    if check_fd:
        grad_fd = _central_fd_grad(ref_fn, u_vec, eps=1e-5)
        np.testing.assert_allclose(
            np.asarray(grad_eag), grad_fd, rtol=2e-7, atol=2e-7
        )


def test_fast_loglike_spec_shape_and_dead_helper_removed() -> None:
    fields = dataclasses.fields(FastLogLikeSpec)
    assert len(fields) <= 8
    sig = inspect.signature(FastLogLikeSpec.__init__)
    n_params = len([p for name, p in sig.parameters.items() if name != "self"])
    assert n_params <= 8
    assert not hasattr(FastLogLikeSpec, "mode")
    assert not hasattr(ttf, "_neg_log_post_fn")


_STATIC_MODE_CASES = [
    ("none", "cholesky"),
    ("hierarchical", "cholesky"),
    ("hierarchical", "qr"),
    ("hierarchical", "rank_one"),
    ("shared", "qr"),
    ("per_series", "rank_one"),
]


@pytest.mark.parametrize("offset_mode,whiten_method", _STATIC_MODE_CASES)
def test_static_joint_modes_equiv_and_grad(
    multi_series_dataset: tt.Dataset,
    offset_mode: str,
    whiten_method: str,
) -> None:
    cbm = tt.EffectiveCBM(K=2)
    prod = tt.ProductionSum(
        spike=tt.MiyakeSpike(
            t_start=774.62, duration=0.33, fluence=6.15, profile="supergaussian"
        ),
        log_gp=tt.LogGPProduction(
            0.22 * tgk.Matern32(scale=3.2),
            jnp.array([-0.31, 0.24, -0.17, 0.39, -0.22]),
            domain=(770.0, 782.0),
        ),
    )
    model = tt.RadiocarbonModel(cbm, prod, q_std=1.83)
    ps = ParameterSpace(
        model,
        [
            ParamSpec("production.spike.fluence", "exp", 1.7, 0.5),
            ParamSpec("production.log_gp.coefficients", "identity", 0.0, 1.0),
        ],
    )
    cfg = LinearMarginalConfig(
        offset_mode=offset_mode,  # type: ignore[arg-type]
        global_prior_mean=-5.7,
        global_prior_sd=135.0,
        lab_prior_sd=1.65,
        whiten_method=whiten_method,  # type: ignore[arg-type]
    )
    u_vec = jnp.array([1.86, -0.28, 0.19, -0.12, 0.34, -0.25])
    jit_s = 0.27
    spec = _make_fast_unconstrained_loglike(
        model, multi_series_dataset, ps, cfg, sigma_jitter=jit_s
    )

    def ref_fn(u: jax.Array) -> jax.Array:
        return joint_log_likelihood(
            ps.constrain(u, model), multi_series_dataset, cfg=cfg, sigma_jitter=jit_s
        )

    _assert_spec_equiv_and_grad(
        spec,
        ref_fn,
        u_vec,
        expect_static=True,
        expect_joint=True,
        rtol=5e-13 if whiten_method == "cholesky" else 1e-13,
        check_fd=(offset_mode == "hierarchical" and whiten_method == "cholesky"),
        check_filter_jit=(offset_mode in ("none", "hierarchical") and whiten_method == "cholesky"),
    )

    m_eval = ps.constrain(u_vec, model)
    resid_fast = np.asarray(spec.nonlinear_residual_from_u(u_vec))
    resid_ref = np.asarray(
        multi_series_dataset.d14c_flat
        - m_eval.predict_d14c(multi_series_dataset, offsets=None, concat=True)
    )
    np.testing.assert_allclose(resid_fast, resid_ref, rtol=1e-14, atol=1e-13)

    off_cfg = _offset_only_cfg(cfg)
    X_np, pm_np, ps_np = (
        np.asarray(arr)
        for arr in _build_marginal_design(m_eval, multi_series_dataset, off_cfg)
    )
    sig_eff = np.hypot(np.asarray(multi_series_dataset.sigma_flat), jit_s)
    ll_scipy = _scipy_qr_woodbury_loglike(resid_ref, X_np, pm_np, ps_np, sig_eff)
    scipy_rtol = 5e-13 if whiten_method == "cholesky" else 1e-13
    np.testing.assert_allclose(
        float(spec(u_vec)), ll_scipy, rtol=scipy_rtol, atol=1e-12
    )
    if offset_mode == "per_series":
        jax.clear_caches()
        gc.collect()


@pytest.mark.parametrize("offset_mode,whiten_method", _STATIC_MODE_CASES)
def test_static_marginal_modes_equiv_and_grad(
    multi_series_dataset: tt.Dataset,
    offset_mode: str,
    whiten_method: str,
) -> None:
    cbm = tt.EffectiveCBM(K=2)
    prod = tt.ProductionSum(
        spike=tt.MiyakeSpike(t_start=774.55, duration=0.35, fluence=5.9),
        solar=tt.HarmonicSolar.from_polar(
            periods=11.0, amplitudes=0.21, phases=0.35, t_ref=775.0
        ),
        trend=tt.LinearTrend(slope=0.018, t_ref=772.0),
        gp=tt.GPProduction(tgk.Matern32(scale=4.2), n_modes=8, domain=(770.0, 782.0)),
    )
    model = tt.RadiocarbonModel(cbm, prod, q_std=1.81)
    ps = ParameterSpace(
        model,
        [
            ParamSpec(
                "production.spike.t_start", "sigmoid", 0.15, 0.8, low=773.5, high=775.5
            ),
            ParamSpec("production.spike.duration", "exp", -0.9, 0.5),
            ParamSpec("production.spike.fluence", "exp", 1.8, 0.4),
        ],
    )
    cfg = LinearMarginalConfig(
        offset_mode=offset_mode,  # type: ignore[arg-type]
        global_prior_mean=-3.8,
        global_prior_sd=95.0,
        lab_prior_sd=1.45,
        whiten_method=whiten_method,  # type: ignore[arg-type]
        include_gp_initial_cov=(offset_mode == "hierarchical"),
    )
    u_vec = jnp.array([0.22, -0.82, 1.79])
    jit_s = 0.23
    spec = _make_fast_unconstrained_loglike(
        model, multi_series_dataset, ps, cfg, sigma_jitter=jit_s
    )

    def ref_fn(u: jax.Array) -> jax.Array:
        return marginal_log_likelihood(
            ps.constrain(u, model), multi_series_dataset, cfg, sigma_jitter=jit_s
        )

    _assert_spec_equiv_and_grad(
        spec,
        ref_fn,
        u_vec,
        expect_static=True,
        expect_joint=False,
        rtol=5e-13 if whiten_method == "cholesky" else 1e-13,
        check_fd=(offset_mode == "hierarchical" and whiten_method == "qr"),
        check_filter_jit=(offset_mode == "hierarchical" and whiten_method == "qr"),
    )

    m_eval = ps.constrain(u_vec, model)
    resid_fast = np.asarray(spec.nonlinear_residual_from_u(u_vec))
    resid_ref = np.asarray(ttl._nonlinear_residual(m_eval, multi_series_dataset, cfg))
    np.testing.assert_allclose(resid_fast, resid_ref, rtol=1e-14, atol=1e-13)

    X_np, pm_np, ps_np = (
        np.asarray(arr)
        for arr in _build_marginal_design(m_eval, multi_series_dataset, cfg)
    )
    sig_eff = np.hypot(np.asarray(multi_series_dataset.sigma_flat), jit_s)
    ll_scipy = _scipy_qr_woodbury_loglike(resid_ref, X_np, pm_np, ps_np, sig_eff)
    scipy_rtol = 5e-13 if whiten_method == "cholesky" else 1e-13
    np.testing.assert_allclose(
        float(spec(u_vec)), ll_scipy, rtol=scipy_rtol, atol=1e-12
    )
    if offset_mode == "per_series":
        jax.clear_caches()
        gc.collect()


def test_static_marginal_gaussian_field_two_hemisphere_and_cached_dct() -> None:
    jax.clear_caches()
    cbm = tt.load_preset("Buntgen18")
    s_nh = tt.TreeRingSeries(
        years=np.arange(772.0, 779.0),
        d14c=np.array([-13.9, -13.2, -12.4, -1.7, 0.8, -1.5, -2.9]),
        sigma_d14c=np.array([1.2, 1.3, 1.1, 1.5, 1.3, 1.4, 1.2]),
        hemisphere="north",
        seasonal_weights="Apr-Sep",
        name="nh",
    )
    s_sh = tt.TreeRingSeries(
        years=np.arange(772.0, 779.0),
        d14c=np.array([-18.1, -17.5, -16.8, -6.2, -3.5, -5.8, -7.1]),
        sigma_d14c=np.array([1.4, 1.5, 1.3, 1.6, 1.4, 1.5, 1.3]),
        hemisphere="south",
        year_convention="schulman_start",
        seasonal_weights="Oct-Mar",
        name="sh",
    )
    ds = tt.Dataset([s_nh, s_sh])
    gf = tt.GaussianField(jnp.zeros(9), 771.0, 780.0, sigma=0.35, length_yr=3.5)
    dct_op = tt.build_dct_response_operator(
        cbm, ds, t_start=771.0, t_end=780.0, n_grid=9, q_std=1.82
    )
    prod = tt.ProductionSum(
        spike=tt.MiyakeSpike(t_start=774.6, duration=0.3, fluence=6.2),
        field=gf,
    )
    model = tt.RadiocarbonModel(cbm, prod, q_std=1.82)
    ps = ParameterSpace(model, [ParamSpec("production.spike.fluence", "exp", 1.8, 0.4)])
    cfg = LinearMarginalConfig(
        offset_mode="hierarchical",
        global_prior_mean=-4.5,
        dct_operator=dct_op,
        include_field_initial_cov=True,
    )
    u_vec = jnp.array([1.87])
    spec = _make_fast_unconstrained_loglike(model, ds, ps, cfg, sigma_jitter=0.18)

    _assert_spec_equiv_and_grad(
        spec,
        lambda u: marginal_log_likelihood(
            ps.constrain(u, model), ds, cfg, sigma_jitter=0.18
        ),
        u_vec,
        expect_static=True,
        expect_joint=False,
        rtol=1e-13,
        check_fd=True,
    )
    jax.clear_caches()


def test_static_diffrax_solver_and_collinear_rank_one_conditioning(
    multi_series_dataset: tt.Dataset,
) -> None:
    cbm = tt.EffectiveCBM(K=2)
    solver = tt.DiffraxSolver(rtol=1e-8, atol=1e-10)
    prod = tt.ProductionSum(
        spike=tt.MiyakeSpike(
            t_start=774.6, duration=0.35, fluence=6.0, profile="supergaussian"
        ),
        solar=tt.HarmonicSolar.from_polar(periods=11.0, amplitudes=0.2, t_ref=775.0),
    )
    model = tt.RadiocarbonModel(cbm, prod, solver=solver, q_std=1.79)
    ps = ParameterSpace(model, [ParamSpec("production.spike.fluence", "exp", 1.8, 0.4)])
    cfg = LinearMarginalConfig(offset_mode="hierarchical", global_prior_mean=-2.5)
    spec = _make_fast_unconstrained_loglike(model, multi_series_dataset, ps, cfg)
    assert spec.is_static is True
    assert spec.static_cache is not None and spec.static_cache.cbm_obs is None

    _assert_spec_equiv_and_grad(
        spec,
        lambda u: marginal_log_likelihood(ps.constrain(u, model), multi_series_dataset, cfg),
        jnp.array([1.84]),
        expect_static=True,
        expect_joint=False,
        rtol=1e-13,
    )

    # Collinear large-prior offset design: rank_one and qr must avoid squaring kappa(U)
    model_ex = tt.RadiocarbonModel(cbm, prod.spike, q_std=1.79)
    ps_ex = ParameterSpace(model_ex, [ParamSpec("production.fluence", "exp", 1.8, 0.4)])
    for wm in ("qr", "rank_one"):
        cfg_ill = LinearMarginalConfig(
            offset_mode="hierarchical",
            global_prior_sd=1e9,
            lab_prior_sd=1e9,
            whiten_method=wm,  # type: ignore[arg-type]
        )
        sp_ill = _make_fast_unconstrained_loglike(
            model_ex, multi_series_dataset, ps_ex, cfg_ill
        )
        u_ill = jnp.array([1.82])
        m_ill = ps_ex.constrain(u_ill, model_ex)
        v_ref = float(marginal_log_likelihood(m_ill, multi_series_dataset, cfg_ill))
        X_i, pm_i, ps_i = (
            np.asarray(a)
            for a in _build_marginal_design(m_ill, multi_series_dataset, cfg_ill)
        )
        r_i = np.asarray(ttl._nonlinear_residual(m_ill, multi_series_dataset, cfg_ill))
        v_qr = _scipy_qr_woodbury_loglike(
            r_i, X_i, pm_i, ps_i, np.asarray(multi_series_dataset.sigma_flat)
        )
        np.testing.assert_allclose(float(sp_ill(u_ill)), v_ref, rtol=1e-9, atol=1e-8)
        np.testing.assert_allclose(float(sp_ill(u_ill)), v_qr, rtol=1e-11, atol=1e-10)
    jax.clear_caches()


def test_dynamic_modes_cbm_subspace_lgf_and_traced_jitter(
    multi_series_dataset: tt.Dataset,
) -> None:
    cbm = tt.EffectiveCBM(K=2)
    gf = tt.GaussianField(jnp.zeros(7), 771.5, 779.5, sigma=0.3, length_yr=3.5)
    prod = tt.ProductionSum(
        spike=tt.MiyakeSpike(t_start=774.6, duration=0.3, fluence=6.0),
        field=gf,
    )
    model = tt.RadiocarbonModel(cbm, prod, q_std=1.76)
    cfg = LinearMarginalConfig(
        offset_mode="hierarchical", include_field_initial_cov=False
    )

    # 1. Dynamic CBM parameter
    ps_cbm = ParameterSpace(
        model,
        [
            ParamSpec("cbm.log_k_strat_to_trop", "identity", -0.5, 0.5),
            ParamSpec("production.spike.fluence", "exp", 1.8, 0.4),
        ],
    )
    sp_cbm = _make_fast_unconstrained_loglike(model, multi_series_dataset, ps_cbm, cfg)
    _assert_spec_equiv_and_grad(
        sp_cbm,
        lambda u: marginal_log_likelihood(
            ps_cbm.constrain(u, model), multi_series_dataset, cfg
        ),
        jnp.array([-0.35, 1.83]),
        expect_static=False,
        expect_joint=False,
        rtol=1e-13,
    )

    # 2. Dynamic field hyperparameter
    ps_hyp = ParameterSpace(
        model, [ParamSpec("production.field.sigma", "exp", -1.2, 0.5)]
    )
    sp_hyp = _make_fast_unconstrained_loglike(model, multi_series_dataset, ps_hyp, cfg)
    _assert_spec_equiv_and_grad(
        sp_hyp,
        lambda u: marginal_log_likelihood(
            ps_hyp.constrain(u, model), multi_series_dataset, cfg
        ),
        jnp.array([-1.05]),
        expect_static=False,
        expect_joint=False,
        rtol=1e-13,
    )

    # 3. SubspaceCBM + LinearizedDCTOperator dynamic mode (compact 3-box balanced CBM)
    topo_sub = tt.Topology(
        box_names=("Stratosphere", "Troposphere", "Ocean"),
        edges_src=(0, 1, 1, 2),
        edges_dst=(1, 0, 2, 1),
        prod_box_indices=(0, 1),
        trop_box_indices=(1,),
        hemispheres=("north", "north", "north"),
    )
    basis = np.array([[0.2, 0.0], [-0.1, 0.15], [0.05, -0.2], [-0.05, 0.1]], dtype=np.float64)
    sub_cbm = tt.SubspaceCBM(
        topo_sub,
        np.array([-0.7, -1.6, -2.1, -4.0], dtype=np.float64),
        basis,
        math.log(40000.0),
        alpha=jnp.array([0.1, -0.1]),
        prod_weights=(0.7, 0.3),
    )
    lin_op = tt.build_linearized_dct_operator(
        sub_cbm, multi_series_dataset, t_start=771.5, t_end=779.5, n_grid=7
    )
    m_sub = tt.RadiocarbonModel(sub_cbm, prod)
    ps_sub = ParameterSpace(m_sub, [ParamSpec("cbm.alpha", "identity", 0.0, 1.0)])
    cfg_sub = LinearMarginalConfig(
        offset_mode="hierarchical", dct_operator=lin_op, include_field_initial_cov=True
    )
    sp_sub = _make_fast_unconstrained_loglike(
        m_sub, multi_series_dataset, ps_sub, cfg_sub
    )
    _assert_spec_equiv_and_grad(
        sp_sub,
        lambda u: marginal_log_likelihood(
            ps_sub.constrain(u, m_sub), multi_series_dataset, cfg_sub
        ),
        jnp.array([0.18, -0.14]),
        expect_static=False,
        expect_joint=False,
        rtol=1e-13,
    )
    jax.clear_caches()

    # 4. LogGaussianField Laplace dynamic vs sampled-latent static joint
    lgf = tt.LogGaussianField(
        jnp.linspace(-0.2, 0.25, 6), 771.5, 779.5, sigma=0.25, length_yr=3.5
    )
    m_lgf = tt.RadiocarbonModel(
        cbm,
        tt.ProductionSum(
            spike=tt.MiyakeSpike(t_start=774.6, duration=0.3, fluence=6.0), lgf=lgf
        ),
    )
    cfg_lgf = LinearMarginalConfig(
        offset_mode="hierarchical", marginalise_field=True, include_field_initial_cov=False
    )
    ps_lgf_lap = ParameterSpace(
        m_lgf, [ParamSpec("production.spike.fluence", "exp", 1.8, 0.4)]
    )
    sp_lgf_lap = _make_fast_unconstrained_loglike(
        m_lgf, multi_series_dataset, ps_lgf_lap, cfg_lgf
    )
    _assert_spec_equiv_and_grad(
        sp_lgf_lap,
        lambda u: marginal_log_likelihood(
            ps_lgf_lap.constrain(u, m_lgf), multi_series_dataset, cfg_lgf
        ),
        jnp.array([1.81]),
        expect_static=False,
        expect_joint=False,
        rtol=1e-13,
    )

    ps_lgf_jnt = ParameterSpace(
        m_lgf, [ParamSpec("production.lgf.latent", "identity", 0.0, 1.0)]
    )
    sp_lgf_jnt = _make_fast_unconstrained_loglike(
        m_lgf, multi_series_dataset, ps_lgf_jnt, cfg_lgf
    )
    _assert_spec_equiv_and_grad(
        sp_lgf_jnt,
        lambda u: joint_log_likelihood(
            ps_lgf_jnt.constrain(u, m_lgf), multi_series_dataset, cfg=cfg_lgf
        ),
        jnp.linspace(-0.25, 0.3, 6),
        expect_static=True,
        expect_joint=True,
        rtol=1e-13,
    )

    # 5. Traced sigma_jitter falls back cleanly to dynamic mode
    m_sp = tt.RadiocarbonModel(cbm, prod.spike)
    ps_sp = ParameterSpace(m_sp, [ParamSpec("production.fluence", "exp", 1.8, 0.4)])
    u_sp = jnp.array([1.82])
    val_tr, grad_tr = jax.value_and_grad(
        lambda jit_s: _make_fast_unconstrained_loglike(
            m_sp, multi_series_dataset, ps_sp, cfg, sigma_jitter=jit_s
        )(u_sp)
    )(jnp.asarray(0.35, dtype=jnp.float64))
    val_tr_ref, grad_tr_ref = jax.value_and_grad(
        lambda jit_s: marginal_log_likelihood(
            ps_sp.constrain(u_sp, m_sp), multi_series_dataset, cfg, sigma_jitter=jit_s
        )
    )(jnp.asarray(0.35, dtype=jnp.float64))
    np.testing.assert_allclose(float(val_tr), float(val_tr_ref), rtol=1e-13)
    np.testing.assert_allclose(float(grad_tr), float(grad_tr_ref), rtol=1e-13)
    jax.clear_caches()


def test_zero_basis_rebuilds_under_jit_and_perturbation_guard(
    multi_series_dataset: tt.Dataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cbm = tt.EffectiveCBM(K=2)
    prod = tt.ProductionSum(
        spike=tt.MiyakeSpike(t_start=774.6, duration=0.3, fluence=6.1),
        gp=tt.GPProduction(tgk.Matern32(scale=3.8), n_modes=8, domain=(770.0, 782.0)),
    )
    model = tt.RadiocarbonModel(cbm, prod, q_std=1.81)
    ps = ParameterSpace(model, [ParamSpec("production.spike.fluence", "exp", 1.8, 0.4)])
    cfg = LinearMarginalConfig(
        offset_mode="hierarchical",
        global_prior_mean=-6.0,
        global_prior_sd=100.0,
        lab_prior_sd=1.5,
    )

    counts = {"design": 0, "modal": 0}
    orig_design = ttl._build_marginal_design
    orig_modal = ttl._modal_decomposition

    def spy_design(*args, **kwargs):
        counts["design"] += 1
        return orig_design(*args, **kwargs)

    def spy_modal(*args, **kwargs):
        counts["modal"] += 1
        return orig_modal(*args, **kwargs)

    monkeypatch.setattr(ttl, "_build_marginal_design", spy_design)
    monkeypatch.setattr(ttl, "_modal_decomposition", spy_modal)

    spec = _make_fast_unconstrained_loglike(
        model, multi_series_dataset, ps, cfg, sigma_jitter=0.3
    )
    assert counts["design"] == 1
    assert counts["modal"] >= 1

    counts["design"] = 0
    counts["modal"] = 0
    step_vg = eqx.filter_jit(lambda sp, u: jax.value_and_grad(sp)(u))
    for u_val in np.linspace(1.4, 2.2, 12):
        val, grad = step_vg(spec, jnp.array([u_val]))
        assert jnp.isfinite(val) and jnp.all(jnp.isfinite(grad))
    assert counts["design"] == 0
    assert counts["modal"] == 0

    # Batch vmap helpers in sampling.py (both static vmap and dynamic lax.map)
    u_batch = jnp.linspace(1.5, 2.1, 5)[:, None]
    lk_batch = tts._vmap_fast_lk_jit(spec, u_batch)
    res_batch = tts._vmap_fast_resid_jit(spec, u_batch)
    assert lk_batch.shape == (5,)
    assert res_batch.shape == (5, multi_series_dataset.n_obs)

    # Exercise fitting._laplace_covariance on >8-box dynamic vs static CBM
    g15_bal = tt.load_preset("Guttler15", kind="balanced")
    m_g15 = tt.RadiocarbonModel(g15_bal, prod.spike)
    ps_g15_dyn = ParameterSpace(
        m_g15, [ParamSpec("cbm.log_total_mass", "identity", 10.6, 0.2)]
    )
    cov_dyn = ttf._laplace_covariance(
        jnp.array([10.6]), m_g15, multi_series_dataset, ps_g15_dyn, cfg, None
    )
    assert cov_dyn.shape == (1, 1) and bool(jnp.all(jnp.isfinite(cov_dyn)))

    # Confirm assertions fail when shift_const, w_mat, or sigma_jitter are dropped
    u0 = jnp.array([1.83])
    exact_val = float(spec(u0))
    assert spec.static_cache is not None
    w = spec.static_cache.whiten

    spec_no_shift = eqx.tree_at(
        lambda s: s.static_cache.whiten.shift_const,
        spec,
        jnp.zeros_like(w.shift_const),
    )
    assert abs(float(spec_no_shift(u0)) - exact_val) > 5e-3

    spec_no_proj = eqx.tree_at(
        lambda s: s.static_cache.whiten.w_mat,
        spec,
        jnp.zeros_like(w.w_mat),
    )
    assert abs(float(spec_no_proj(u0)) - exact_val) > 1.0

    spec_no_jitter = _make_fast_unconstrained_loglike(
        model, multi_series_dataset, ps, cfg, sigma_jitter=None
    )
    assert abs(float(spec_no_jitter(u0)) - exact_val) > 0.1
    jax.clear_caches()
