import numpy as np
import jax.numpy as jnp
import pytest
import ticktack
from ticktack import fitting


@pytest.fixture
def CarbonFitter_creation():
    cbm = ticktack.load_presaved_model('Guttler14', production_rate_units='atoms/cm^2/s')
    cf = fitting.CarbonFitter(cbm)
    cf.time_data = jnp.arange(200, 210)
    cf.d14c_data_error = jnp.ones((cf.time_data.size,))
    cf.d14c_data = jnp.array([-169.81482498, -168.05109886, -163.81278239, -158.13313339,
                              -153.23525037, -150.89762995, -151.11804461, -152.1036496 ,
                              -151.60492619, -151.60492619])
    cf.start = np.nanmin(cf.time_data)
    cf.end = np.nanmax(cf.time_data)
    cf.resolution = 1000
    cf.burn_in_time = jnp.linspace(cf.start - 1000, cf.start, cf.resolution)
    cf.time_grid_fine = jnp.arange(cf.start, cf.end, 0.05)
    cf.time_oversample = 1000
    cf.offset = 0
    cf.annual = jnp.arange(cf.start, cf.end + 1)
    cf.mask = jnp.in1d(cf.annual, cf.time_data)[:-1]
    return cf


def test_miyake_event_fixed_solar(CarbonFitter_creation):
    CarbonFitter_creation.miyake_event_fixed_solar(200, jnp.array([205., 1./12, jnp.pi/2., 81./12]))
    assert True

def test_miyake_event_flexible_solar(CarbonFitter_creation):
    CarbonFitter_creation.miyake_event_flexible_solar(200, jnp.array([205., 1./12, jnp.pi/2., 81./12,
                                                                           2*np.pi/11, 0.18]))
    assert True

def test_interp_gp(CarbonFitter_creation):
    CarbonFitter_creation.prepare_function(use_control_points=True, interp="gp")
    CarbonFitter_creation.interp_gp(201, jnp.ones(CarbonFitter_creation.control_points_time.size))
    assert True

def test_grad_sum_interp_gp(CarbonFitter_creation):
    CarbonFitter_creation.prepare_function(use_control_points=True, interp="gp")
    CarbonFitter_creation.grad_sum_interp_gp(jnp.ones(CarbonFitter_creation.control_points_time.size))
    assert True

def test_neg_gp_log_likelihood(CarbonFitter_creation):
    CarbonFitter_creation.prepare_function(use_control_points=True, interp="gp")
    CarbonFitter_creation.neg_gp_log_likelihood(jnp.ones(CarbonFitter_creation.control_points_time.size))
    assert True

def test_gp_likelihood(CarbonFitter_creation):
    CarbonFitter_creation.prepare_function(use_control_points=True, interp="gp")
    CarbonFitter_creation.gp_likelihood(jnp.ones(CarbonFitter_creation.control_points_time.size))
    assert True

def test_fit_cp(CarbonFitter_creation):
    CarbonFitter_creation.prepare_function(use_control_points=True, interp="gp")
    CarbonFitter_creation.fit_cp(low_bound=0)
    assert True

def test_log_like(CarbonFitter_creation):
    CarbonFitter_creation.prepare_function(production='miyake', fit_solar=False)
    CarbonFitter_creation.log_like(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]))
    CarbonFitter_creation.prepare_function(production='miyake', fit_solar=True)
    CarbonFitter_creation.log_like(jnp.array([205., 1./12, jnp.pi/2., 81./12, 2*np.pi/11, 0.18]))
    CarbonFitter_creation.prepare_function(use_control_points=True, interp="gp")
    CarbonFitter_creation.log_like(jnp.ones(CarbonFitter_creation.control_points_time.size))
    assert True

def test_dc14(CarbonFitter_creation):
    CarbonFitter_creation.prepare_function(production='miyake', fit_solar=False)
    CarbonFitter_creation.dc14(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]))
    CarbonFitter_creation.prepare_function(production='miyake', fit_solar=True)
    CarbonFitter_creation.dc14(jnp.array([205., 1./12, jnp.pi/2., 81./12, 2*np.pi/11, 0.18]))
    CarbonFitter_creation.prepare_function(use_control_points=True, interp="gp")
    CarbonFitter_creation.dc14(jnp.ones(CarbonFitter_creation.control_points_time.size))
    assert True

def test_dc14_fine(CarbonFitter_creation):
    CarbonFitter_creation.prepare_function(production='miyake', fit_solar=False)
    CarbonFitter_creation.dc14_fine(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]))
    CarbonFitter_creation.prepare_function(production='miyake', fit_solar=True)
    CarbonFitter_creation.dc14_fine(jnp.array([205., 1./12, jnp.pi/2., 81./12, 2*np.pi/11, 0.18]))
    CarbonFitter_creation.prepare_function(use_control_points=True, interp="gp")
    CarbonFitter_creation.dc14_fine(jnp.ones(CarbonFitter_creation.control_points_time.size))
    assert True


