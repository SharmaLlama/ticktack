import numpy as np
import jax.numpy as jnp
import pytest
import ticktack
from ticktack import fitting


@pytest.fixture
def SingleFitter_creation():
    cbm = ticktack.load_presaved_model('Guttler14', production_rate_units='atoms/cm^2/s')
    sf = fitting.SingleFitter(cbm)
    sf.time_data = jnp.arange(200, 210)
    sf.d14c_data_error = jnp.ones((sf.time_data.size,))
    sf.d14c_data = jnp.array([-169.81482498, -168.05109886, -163.81278239, -158.13313339,
                              -153.23525037, -150.89762995, -151.11804461, -152.1036496 ,
                              -151.60492619, -151.60492619])
    sf.start = np.nanmin(sf.time_data)
    sf.end = np.nanmax(sf.time_data)
    sf.resolution = 1000
    sf.burn_in_time = jnp.linspace(sf.start - 1000, sf.start, sf.resolution)
    sf.time_grid_fine = jnp.arange(sf.start, sf.end, 0.05)
    sf.time_oversample = 1000
    sf.offset = 0
    sf.annual = jnp.arange(sf.start, sf.end + 1)
    sf.mask = jnp.in1d(sf.annual, sf.time_data)[:-1]
    return sf


def test_miyake_event_fixed_solar(SingleFitter_creation):
    SingleFitter_creation.miyake_event_fixed_solar(200, jnp.array([205., 1./12, jnp.pi/2., 81./12]))
    assert True

def test_miyake_event_flexible_solar(SingleFitter_creation):
    SingleFitter_creation.miyake_event_flexible_solar(200, jnp.array([205., 1./12, jnp.pi/2., 81./12, 0.18]))
    assert True

def test_interp_gp(SingleFitter_creation):
    SingleFitter_creation.prepare_function(use_control_points=True, interp="gp")
    SingleFitter_creation.interp_gp(201, jnp.ones(SingleFitter_creation.control_points_time.size))
    assert True

def test_grad_sum_interp_gp(SingleFitter_creation):
    SingleFitter_creation.prepare_function(use_control_points=True, interp="gp")
    SingleFitter_creation.grad_sum_interp_gp(jnp.ones(SingleFitter_creation.control_points_time.size))
    assert True

def test_neg_gp_log_likelihood(SingleFitter_creation):
    SingleFitter_creation.prepare_function(use_control_points=True, interp="gp")
    SingleFitter_creation.neg_gp_log_likelihood(jnp.ones(SingleFitter_creation.control_points_time.size))
    assert True

def test_gp_likelihood(SingleFitter_creation):
    SingleFitter_creation.prepare_function(use_control_points=True, interp="gp")
    SingleFitter_creation.gp_likelihood(jnp.ones(SingleFitter_creation.control_points_time.size))
    assert True

def test_fit_cp(SingleFitter_creation):
    SingleFitter_creation.prepare_function(use_control_points=True, interp="gp")
    SingleFitter_creation.fit_cp(low_bound=0)
    assert True

def test_log_like(SingleFitter_creation):
    SingleFitter_creation.prepare_function(production='miyake', fit_solar=False)
    SingleFitter_creation.log_like(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]))
    SingleFitter_creation.prepare_function(production='miyake', fit_solar=True)
    SingleFitter_creation.log_like(jnp.array([205., 1./12, jnp.pi/2., 81./12, 0.18]))
    SingleFitter_creation.prepare_function(use_control_points=True, interp="gp")
    SingleFitter_creation.log_like(jnp.ones(SingleFitter_creation.control_points_time.size))
    assert True

def test_dc14(SingleFitter_creation):
    SingleFitter_creation.prepare_function(production='miyake', fit_solar=False)
    SingleFitter_creation.dc14(params=jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]))
    SingleFitter_creation.prepare_function(production='miyake', fit_solar=True)
    SingleFitter_creation.dc14(hemisphere='south', params=(jnp.array([205., 1./12, jnp.pi/2., 81./12, 0.18])))
    SingleFitter_creation.prepare_function(use_control_points=True, interp="gp")
    SingleFitter_creation.dc14(params=jnp.ones(SingleFitter_creation.control_points_time.size))
    assert True

def test_dc14_fine(SingleFitter_creation):
    SingleFitter_creation.prepare_function(production='miyake', fit_solar=False)
    SingleFitter_creation.dc14_fine(params=jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]))
    SingleFitter_creation.prepare_function(production='miyake', fit_solar=True)
    SingleFitter_creation.dc14_fine(params=jnp.array([205., 1./12, jnp.pi/2., 81./12, 0.18]))
    SingleFitter_creation.prepare_function(use_control_points=True, interp="gp")
    SingleFitter_creation.dc14_fine(params=jnp.ones(SingleFitter_creation.control_points_time.size))
    assert True


