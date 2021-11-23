import numpy as np
import jax.numpy as jnp
import pytest
import ticktack
from ticktack import fitting


@pytest.fixture
def SingleFitter_creation():
    cbm = ticktack.load_presaved_model('Guttler14', production_rate_units='atoms/cm^2/s')
    sf = fitting.SingleFitter(cbm, hemisphere="south")
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

@pytest.fixture
def MultiFitter_creation(SingleFitter_creation):
    mf = fitting.MultiFitter()
    SingleFitter_creation.prepare_function(model="simple_sinusoid")
    mf.add_SingleFitter(SingleFitter_creation)
    mf.add_SingleFitter(SingleFitter_creation)
    return mf

def test_multi_likelihood(MultiFitter_creation):
    MultiFitter_creation.multi_likelihood(params=jnp.array([205., 1. / 12, jnp.pi / 2., 81. / 12]))
    assert True

def test_mf_log_prior_simple_sinusoid(MultiFitter_creation):
    MultiFitter_creation.log_prior_simple_sinusoid(params=jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]))
    assert True

def test_mf_log_prior_flexible_sinusoid(MultiFitter_creation):
    MultiFitter_creation.log_prior_flexible_sinusoid(params=jnp.array([205., 1. / 12, jnp.pi / 2., 81./12, 0.18]))
    assert True

def test_mf_log_joint_simple_sinusoid(MultiFitter_creation):
    MultiFitter_creation.log_joint_simple_sinusoid(params=jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]))
    assert True

def test_mf_log_joint_flexible_sinusoid(SingleFitter_creation, MultiFitter_creation):
    mf = fitting.MultiFitter()
    SingleFitter_creation.prepare_function(model="flexible_sinusoid")
    mf.add_SingleFitter(SingleFitter_creation)
    mf.add_SingleFitter(SingleFitter_creation)
    MultiFitter_creation.log_joint_flexible_sinusoid(params=jnp.array([205., 1. / 12, jnp.pi / 2., 81./12, 0.18]))
    assert True

def test_MarkovChainSampler(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="simple_sinusoid")
    SingleFitter_creation.MarkovChainSampler(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]),
                                             SingleFitter_creation.log_joint_simple_sinusoid,
                                             burnin=10,
                                             production=10)
    assert True

def test_NestedSampler(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="simple_sinusoid")
    SingleFitter_creation.NestedSampler(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]),
                                        SingleFitter_creation.log_joint_simple_sinusoid,
                                        low_bound=jnp.array([770., 0., -jnp.pi, 0.]),
                                        high_bound = jnp.array([780., 5., jnp.pi, 15.])
                                        )
    assert True

def test_interp_gp(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="control_points")
    SingleFitter_creation.interp_gp(201, jnp.ones(SingleFitter_creation.control_points_time.size))
    assert True

def test_simple_sinusoid(SingleFitter_creation):
    SingleFitter_creation.simple_sinusoid(200, jnp.array([205., 1./12, jnp.pi/2., 81./12]))
    assert True

def test_flexible_sinusoid(SingleFitter_creation):
    SingleFitter_creation.flexible_sinusoid(200, jnp.array([205., 1./12, jnp.pi/2., 81./12, 0.18]))
    assert True

def test_dc14(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="simple_sinusoid")
    SingleFitter_creation.dc14(params=jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]))
    SingleFitter_creation.prepare_function(model="flexible_sinusoid")
    SingleFitter_creation.dc14(params=(jnp.array([205., 1./12, jnp.pi/2., 81./12, 0.18])))
    SingleFitter_creation.prepare_function(model="control_points")
    SingleFitter_creation.dc14(params=jnp.ones(SingleFitter_creation.control_points_time.size))
    assert True

def test_dc14_fine(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="simple_sinusoid")
    SingleFitter_creation.dc14_fine(params=jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]))
    SingleFitter_creation.prepare_function(model="flexible_sinusoid")
    SingleFitter_creation.dc14_fine(params=jnp.array([205., 1./12, jnp.pi/2., 81./12, 0.18]))
    SingleFitter_creation.prepare_function(model="control_points")
    SingleFitter_creation.dc14_fine(params=jnp.ones(SingleFitter_creation.control_points_time.size))
    assert True

def test_log_prior_simple_sinusoid(SingleFitter_creation):
    SingleFitter_creation.log_prior_simple_sinusoid(params=jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]))
    assert True

def test_log_prior_flexible_sinusoid(SingleFitter_creation):
    SingleFitter_creation.log_prior_flexible_sinusoid(params=jnp.array([205., 1. / 12, jnp.pi / 2., 81./12, 0.18]))
    assert True

def test_log_likelihood(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="simple_sinusoid")
    SingleFitter_creation.log_likelihood(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]))
    assert True

def test_log_joint_simple_sinusoid(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="simple_sinusoid")
    SingleFitter_creation.log_joint_simple_sinusoid(params=jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]))
    assert True

def test_log_joint_flexible_sinusoid(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="flexible_sinusoid")
    SingleFitter_creation.log_joint_flexible_sinusoid(params=jnp.array([205., 1. / 12, jnp.pi / 2., 81./12, 0.18]))
    assert True

def test_neg_log_likelihood_gp(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="control_points")
    SingleFitter_creation.neg_log_likelihood_gp(jnp.ones(SingleFitter_creation.control_points_time.size))
    assert True

def test_log_joint_gp(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="control_points")
    SingleFitter_creation.log_joint_gp(jnp.ones(SingleFitter_creation.control_points_time.size))
    assert True

def test_neg_grad_log_joint_gp(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="control_points")
    SingleFitter_creation.neg_grad_log_joint_gp(jnp.ones(SingleFitter_creation.control_points_time.size))
    assert True

def test_fit_ControlPoints(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="control_points")
    SingleFitter_creation.fit_ControlPoints(low_bound=0)
    assert True

def test_plot_recovery(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="simple_sinusoid")
    chain = SingleFitter_creation.MarkovChainSampler(jnp.array([205., 1. / 12, jnp.pi / 2., 81. / 12]),
                                             SingleFitter_creation.log_joint_simple_sinusoid,
                                             burnin=10,
                                             production=100)
    SingleFitter_creation.plot_recovery(chain,
                                        time_data=SingleFitter_creation.time_grid_fine,
                                        true_production=jnp.ones(SingleFitter_creation.time_grid_fine.size))
    assert True

def test_plot_samples(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="simple_sinusoid")
    chain = SingleFitter_creation.MarkovChainSampler(jnp.array([205., 1. / 12, jnp.pi / 2., 81. / 12]),
                                             SingleFitter_creation.log_joint_simple_sinusoid,
                                             burnin=10,
                                             production=100)
    SingleFitter_creation.plot_samples(chain, 8)
    assert True

# def test_chain_summary(SingleFitter_creation):
#     SingleFitter_creation.prepare_function(model="simple_sinusoid")
#     chain = SingleFitter_creation.MarkovChainSampler(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]),
#                                              SingleFitter_creation.log_joint_simple_sinusoid,
#                                              burnin=10,
#                                              production=10)
#     SingleFitter_creation.chain_summary(chain, 8,
#                                         labels=["Start Date (yr)", "Duration (yr)", "phi (yr)", "Area"])
#     assert True
#
# def test_plot_multiple_chains(SingleFitter_creation):
#     SingleFitter_creation.prepare_function(model="simple_sinusoid")
#     chain = SingleFitter_creation.MarkovChainSampler(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]),
#                                              SingleFitter_creation.log_joint_simple_sinusoid,
#                                              burnin=10,
#                                              production=10)
#     SingleFitter_creation.plot_multiple_chains([chain, chain], 8,
#                                                params_names=["Start Date (yr)", "Duration (yr)", "phi (yr)", "Area"],
#                                                labels=["1", "2"])
#     assert True




