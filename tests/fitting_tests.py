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
    sf.burn_in_time = jnp.arange(sf.start - 1000, sf.start)
    sf.oversample = 1008
    sf.time_data_fine = jnp.linspace(sf.start - 1, sf.end + 1, int(sf.oversample * (sf.end - sf.start + 2)))
    sf.burnin_oversample = 1
    sf.offset = 0
    sf.annual = jnp.arange(sf.start, sf.end + 1)
    sf.mask = jnp.in1d(sf.annual, sf.time_data)
    sf.growth = jnp.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    return sf

@pytest.fixture
def MultiFitter_creation(SingleFitter_creation):
    mf = fitting.MultiFitter()
    SingleFitter_creation.prepare_function(model="simple_sinusoid")
    mf.add_SingleFitter(SingleFitter_creation)
    mf.add_SingleFitter(SingleFitter_creation)
    mf.compile()
    return mf

def test_get_data():
    fitting.get_data(event="993AD", hemisphere="south")
    fitting.get_data(event="993AD", hemisphere="north")
    fitting.get_data(event="5259BCE", hemisphere="north")
    fitting.get_data(event="5410BCE", hemisphere="north")
    fitting.get_data(event="7176BCE", hemisphere="north")
    assert True

def test_fit_event():
    fitting.fit_event(993,
                      event='993AD',
                      production_model='simple_sinusoid',
                      sampler="MCMC",
                      burnin=10,
                      production=10)
    mf = fitting.fit_event(993,
                           event='993AD',
                           sampler=None)
    fitting.fit_event(993,
                      event='993AD',
                      sampler='MCMC', mf=mf)
    assert True

def test_chain_summary(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="simple_sinusoid")
    chain = SingleFitter_creation.MarkovChainSampler(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]),
                                                     SingleFitter_creation.log_joint_simple_sinusoid,
                                                     burnin=10,
                                                     production=10,
                                                     args=(jnp.array([200., 0., -jnp.pi, 0.]),
                                                           jnp.array([210., 5., jnp.pi, 15.]))
                                                     )
    SingleFitter_creation.chain_summary(chain, 8,
                                        labels=["Start Date (yr)", "Duration (yr)", "phi (yr)", "Area"])
    assert True

def test_plot_multiple_chains(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="simple_sinusoid")
    chain = SingleFitter_creation.MarkovChainSampler(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]),
                                                     SingleFitter_creation.log_joint_simple_sinusoid,
                                                     burnin=10,
                                                     production=10,
                                                     args=(jnp.array([200., 0., -jnp.pi, 0.]),
                                                           jnp.array([210., 5., jnp.pi, 15.]))
                                                     )
    SingleFitter_creation.plot_multiple_chains([chain, chain], 8,
                                               params_names=["Start Date (yr)", "Duration (yr)", "phi (yr)", "Area"],
                                               labels=["1", "2"])
    assert True

def test_multi_likelihood(MultiFitter_creation):
    out = MultiFitter_creation.multi_likelihood(params=jnp.array([205., 1. / 12, jnp.pi / 2., 81. / 12]))
    assert jnp.allclose(out, -268373.5993183)

def test_mf_log_prior_simple_sinusoid(MultiFitter_creation):
    out = MultiFitter_creation.log_prior_simple_sinusoid(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]),
                                                         jnp.array([200., 0., -jnp.pi, 0.]),
                                                         jnp.array([210., 5., jnp.pi, 15.])
                                                         )
    assert jnp.allclose(out, 0)

    out = MultiFitter_creation.log_prior_simple_sinusoid(jnp.array([211., 1. / 12, jnp.pi / 2., 81./12]),
                                                         jnp.array([200., 0., -jnp.pi, 0.]),
                                                         jnp.array([210., 5., jnp.pi, 15.])
                                                         )
    assert jnp.allclose(out, -np.inf)

def test_mf_log_prior_flexible_sinusoid(MultiFitter_creation):
    out = MultiFitter_creation.log_prior_flexible_sinusoid(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12, 0.18]),
                                                           jnp.array([200., 0., -jnp.pi, 0., 0.]),
                                                           jnp.array([210., 5., jnp.pi, 15., 2.])
                                                           )
    assert jnp.allclose(out, 0)

    out = MultiFitter_creation.log_prior_flexible_sinusoid(jnp.array([211., 1. / 12, jnp.pi / 2., 81./12, 0.18]),
                                                           jnp.array([200., 0., -jnp.pi, 0., 0.]),
                                                           jnp.array([210., 5., jnp.pi, 15., 2.])
                                                           )
    assert jnp.allclose(out, -np.inf)

def test_mf_log_joint_simple_sinusoid(MultiFitter_creation):
    out = MultiFitter_creation.log_joint_simple_sinusoid(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]),
                                                   jnp.array([200., 0., -jnp.pi, 0.]),
                                                   jnp.array([210., 5., jnp.pi, 15.])
                                                   )
    assert jnp.allclose(out, -268373.5993183)

def test_mf_log_joint_flexible_sinusoid(SingleFitter_creation, MultiFitter_creation):
    mf = fitting.MultiFitter()
    SingleFitter_creation.prepare_function(model="flexible_sinusoid")
    mf.add_SingleFitter(SingleFitter_creation)
    mf.add_SingleFitter(SingleFitter_creation)
    mf.compile()
    out = MultiFitter_creation.log_joint_flexible_sinusoid(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12, 0.18]),
                                                     jnp.array([200., 0., -jnp.pi, 0., 0.]),
                                                     jnp.array([210., 5., jnp.pi, 15., 2.])
                                                     )
    assert jnp.allclose(out, -268373.5993183)

def test_MarkovChainSampler(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="simple_sinusoid")
    SingleFitter_creation.MarkovChainSampler(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]),
                                             SingleFitter_creation.log_joint_simple_sinusoid,
                                             burnin=10,
                                             production=10,
                                             args=(jnp.array([200., 0., -jnp.pi, 0.]),
                                                   jnp.array([210., 5., jnp.pi, 15.]))
                                             )
    assert True

# def test_NestedSampler(SingleFitter_creation):
#     SingleFitter_creation.prepare_function(model="simple_sinusoid")
#     SingleFitter_creation.NestedSampler(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]),
#                                         SingleFitter_creation.log_likelihood,
#                                         low_bound = jnp.array([770., 0., -jnp.pi, 0.]),
#                                         high_bound = jnp.array([780., 5., jnp.pi, 15.])
#                                         )
#     assert True

def test_correlation_plot():
    cf = fitting.CarbonFitter()
    mat = jnp.array([[0.50777277, 0.83949556, 0.28273227, 0.65313255],
                     [0.97509577, 0.06402038, 0.88118395, 0.18990772],
                     [0.87250517, 0.59221863, 0.80177601, 0.77286119],
                     [0.66751632, 0.39030823, 0.3096194 , 0.38781154]])
    cf.correlation_plot(mat)
    assert True

def test_interp_gp(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="control_points")
    out = SingleFitter_creation.interp_gp(201, jnp.ones(SingleFitter_creation.control_points_time.size))
    assert jnp.allclose(out, 1)

def test_simple_sinusoid(SingleFitter_creation):
    out = SingleFitter_creation.simple_sinusoid(200, jnp.array([205., 1./12, jnp.pi/2., 81./12]))
    assert jnp.allclose(out, 2.01314825)

def test_flexible_sinusoid(SingleFitter_creation):
    out = SingleFitter_creation.flexible_sinusoid(200, jnp.array([205., 1./12, jnp.pi/2., 81./12, 0.18]))
    assert jnp.allclose(out, 2.01314825)

def test_dc14(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="simple_sinusoid")
    a = SingleFitter_creation.dc14(params=jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]))
    SingleFitter_creation.prepare_function(model="flexible_sinusoid")
    b = SingleFitter_creation.dc14(params=(jnp.array([205., 1./12, jnp.pi/2., 81./12, 0.18])))
    SingleFitter_creation.prepare_function(model="control_points")
    c = SingleFitter_creation.dc14(params=jnp.ones(SingleFitter_creation.control_points_time.size))
    assert jnp.all(
        jnp.array([
            jnp.allclose(a, jnp.array([ 1.84064124,  2.12501188,  1.99019023,  1.40560938,
                                        0.51876855, -0.40943278, 12.50465155, 15.11260263,
                                        15.70854079, 15.69839605])),
            jnp.allclose(b, jnp.array([ 1.84064124,  2.12501188,  1.99019023,  1.40560938,
                                        0.51876855, -0.40943278, 12.50465155, 15.11260263,
                                        15.70854079, 15.69839605])),
            jnp.allclose(c, jnp.array([-125.6135246 , -125.64595525, -125.68380601, -125.73652927,
                                       -125.78905847, -125.8426928 , -125.89562722, -125.94955076,
                                       -125.99709806, -126.07872079]))
        ])
    )

def test_dc14_fine(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="simple_sinusoid")
    a = SingleFitter_creation.dc14_fine(params=jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]))[-9:]
    SingleFitter_creation.prepare_function(model="flexible_sinusoid")
    b = SingleFitter_creation.dc14_fine(params=jnp.array([205., 1./12, jnp.pi/2., 81./12, 0.18]))[-9:]
    SingleFitter_creation.prepare_function(model="control_points")
    c = SingleFitter_creation.dc14_fine(params=jnp.ones(SingleFitter_creation.control_points_time.size))[-9:]
    assert jnp.all(
        jnp.array([
            jnp.allclose(a, jnp.array([15.43155608, 15.42893274, 15.42629311, 15.42363705,
                                       15.42096444, 15.41827515, 15.41556907, 15.41284606,
                                       15.410106  ])),
            jnp.allclose(b, jnp.array([15.43155608, 15.42893274, 15.42629311, 15.42363705,
                                       15.42096444, 15.41827515, 15.41556907, 15.41284606,
                                       15.410106  ])),
            jnp.allclose(c, jnp.array([-127.12783102, -127.13943935, -127.15110516, -127.16282829,
                                       -127.17460858, -127.18644589, -127.19834006, -127.21029093,
                                       -127.22229834]))
        ])
    )

def test_log_prior_simple_sinusoid(SingleFitter_creation):
    out = SingleFitter_creation.log_prior_simple_sinusoid(jnp.array([205., 1. / 12, jnp.pi / 2., 81. / 12]),
                                                    jnp.array([200., 0., -jnp.pi, 0.]),
                                                    jnp.array([210., 5., jnp.pi, 15.])
                                                    )
    assert jnp.allclose(out, 0)

def test_log_prior_flexible_sinusoid(SingleFitter_creation):
    out = SingleFitter_creation.log_prior_flexible_sinusoid(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12, 0.18]),
                                                     jnp.array([200., 0., -jnp.pi, 0., 0.]),
                                                     jnp.array([210., 5., jnp.pi, 15., 2.])
                                                     )
    assert jnp.allclose(out, 0)

def test_log_likelihood(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="simple_sinusoid")
    out = SingleFitter_creation.log_likelihood(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]))
    assert jnp.allclose(out, -134186.79965915)

def test_log_joint_simple_sinusoid(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="simple_sinusoid")
    out = SingleFitter_creation.log_joint_simple_sinusoid(jnp.array([205., 1. / 12, jnp.pi / 2., 81. / 12]),
                                                    jnp.array([200., 0., -jnp.pi, 0.]),
                                                    jnp.array([210., 5., jnp.pi, 15.])
                                                    )
    assert jnp.allclose(out, -134186.79965915)

def test_log_joint_flexible_sinusoid(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="flexible_sinusoid")
    out = SingleFitter_creation.log_joint_flexible_sinusoid(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12, 0.18]),
                                                     jnp.array([200., 0., -jnp.pi, 0., 0.]),
                                                     jnp.array([210., 5., jnp.pi, 15., 2.])
                                                     )
    assert jnp.allclose(out, -134186.79965915)

def test_neg_log_likelihood_gp(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="control_points")
    out = SingleFitter_creation.neg_log_likelihood_gp(jnp.ones(SingleFitter_creation.control_points_time.size))
    assert jnp.allclose(out, 10.21255349)

def test_log_joint_gp(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="control_points")
    out = SingleFitter_creation.log_joint_gp(jnp.ones(SingleFitter_creation.control_points_time.size))
    assert jnp.allclose(out, -5142.16681948)

def test_neg_grad_log_joint_gp(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="control_points")
    out = SingleFitter_creation.neg_grad_log_joint_gp(jnp.ones(SingleFitter_creation.control_points_time.size))
    assert jnp.allclose(out, jnp.array([42493.36926544,   919.64414454,   344.64406023,
                                        366.25838298,   289.70909586,   239.29340972,
                                        178.21194028,   113.01036065,    53.55456182]))

def test_fit_ControlPoints(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="control_points")
    SingleFitter_creation.fit_ControlPoints(low_bound=0)
    assert True

def test_plot_recovery(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="simple_sinusoid")
    chain = SingleFitter_creation.MarkovChainSampler(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]),
                                                     SingleFitter_creation.log_joint_simple_sinusoid,
                                                     burnin=10,
                                                     production=10,
                                                     args=(jnp.array([200., 0., -jnp.pi, 0.]),
                                                           jnp.array([210., 5., jnp.pi, 15.]))
                                                     )
    SingleFitter_creation.plot_recovery(chain,
                                        time_data=SingleFitter_creation.time_data_fine,
                                        true_production=jnp.ones(SingleFitter_creation.time_data_fine.size))
    assert True

def test_plot_samples(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="simple_sinusoid")
    chain = SingleFitter_creation.MarkovChainSampler(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]),
                                                     SingleFitter_creation.log_joint_simple_sinusoid,
                                                     burnin=10,
                                                     production=10,
                                                     args=(jnp.array([200., 0., -jnp.pi, 0.]),
                                                           jnp.array([210., 5., jnp.pi, 15.]))
                                                     )
    SingleFitter_creation.plot_samples(chain, 8)
    assert True




