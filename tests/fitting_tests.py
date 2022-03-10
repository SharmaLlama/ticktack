import numpy as np
import jax.numpy as jnp
import pytest
import ticktack
from ticktack import fitting

@pytest.fixture
def SingleFitter_creation():
    cbm = ticktack.load_presaved_model('Guttler15', production_rate_units='atoms/cm^2/s')
    sf = fitting.SingleFitter(cbm, 'Guttler15', hemisphere="north")
    sf.time_data = jnp.arange(200, 210)
    sf.d14c_data_error = jnp.ones((sf.time_data.size,))
    sf.d14c_data = jnp.array([-169.81482498, -168.05109886, -163.81278239, -158.13313339,
                              -153.23525037, -150.89762995, -151.11804461, -152.1036496 ,
                              -151.60492619, -151.60492619])
    sf.start = np.nanmin(sf.time_data)
    sf.end = np.nanmax(sf.time_data)
    sf.burn_in_time = jnp.arange(sf.start - 1000, sf.start, 1.)
    sf.oversample = 1008
    sf.annual = jnp.arange(sf.start, sf.end + 1)
    sf.time_data_fine = jnp.linspace(jnp.min(sf.annual), jnp.max(sf.annual) + 2, (sf.annual.size + 1) * sf.oversample)
    sf.offset = 0
    sf.mask = jnp.in1d(sf.annual, sf.time_data)
    sf.growth = jnp.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    return sf

@pytest.fixture
def MultiFitter_creation(SingleFitter_creation):
    mf = fitting.MultiFitter()
    SingleFitter_creation.compile_production_model(model="simple_sinusoid")
    mf.add_SingleFitter(SingleFitter_creation)
    mf.add_SingleFitter(SingleFitter_creation)
    mf.compile()
    return mf

# def test_fit_event():
    # fitting.fit_event(993,
    #                   event='993AD-S',
    #                   production_model='simple_sinusoid',
    #                   sampler="MCMC",
    #                   hemisphere='south',
    #                   burnin=10,
    #                   production=10)
    # mf = fitting.fit_event(993,
    #                        event='993AD-S',
    #                        sampler=None, hemisphere='south')
    # fitting.fit_event(993,
    #                   event='993AD-S',
    #                   sampler='MCMC', mf=mf, hemisphere='south',
    #                   burnin=10,
    #                   production=10)
    # assert True

# def test_chain_summary(SingleFitter_creation):
#     SingleFitter_creation.compile_production_model(model="simple_sinusoid")
#     chain = SingleFitter_creation.MarkovChainSampler(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]),
#                                                      SingleFitter_creation.log_joint_simple_sinusoid,
#                                                      burnin=10,
#                                                      production=10,
#                                                      args=(jnp.array([200., 0., -jnp.pi, 0.]),
#                                                            jnp.array([210., 5., jnp.pi, 15.]))
#                                                      )
#     SingleFitter_creation.chain_summary(chain, 8,
#                                         labels=["Start Date (yr)", "Duration (yr)", "phi (yr)", "Area"])
#     assert True
#
# def test_plot_multiple_chains(SingleFitter_creation):
#     SingleFitter_creation.compile_production_model(model="simple_sinusoid")
#     chain = SingleFitter_creation.MarkovChainSampler(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]),
#                                                      SingleFitter_creation.log_joint_simple_sinusoid,
#                                                      burnin=10,
#                                                      production=10,
#                                                      args=(jnp.array([200., 0., -jnp.pi, 0.]),
#                                                            jnp.array([210., 5., jnp.pi, 15.]))
#                                                      )
#     SingleFitter_creation.plot_multiple_chains([chain, chain], 8,
#                                                params_names=["Start Date (yr)", "Duration (yr)", "phi (yr)", "Area"],
#                                                labels=["1", "2"])
#     assert True

def test_multi_likelihood(MultiFitter_creation):
    out = MultiFitter_creation.multi_likelihood(params=jnp.array([205., 1. / 12, jnp.pi / 2., 81. / 12]))
    assert jnp.allclose(out, -271736.26950342)

def test_mf_log_joint_likelihood(MultiFitter_creation):
    out = MultiFitter_creation.log_joint_likelihood(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]),
                                                         jnp.array([200., 0., -jnp.pi, 0.]),
                                                         jnp.array([210., 5., jnp.pi, 15.])
                                                         )
    assert jnp.allclose(out, -271736.26950342)

    out = MultiFitter_creation.log_joint_likelihood(jnp.array([211., 1. / 12, jnp.pi / 2., 81./12]),
                                                         jnp.array([200., 0., -jnp.pi, 0.]),
                                                         jnp.array([210., 5., jnp.pi, 15.])
                                                         )
    assert jnp.allclose(out, -np.inf)

def test_MarkovChainSampler(SingleFitter_creation):
    SingleFitter_creation.compile_production_model(model="simple_sinusoid")
    SingleFitter_creation.MarkovChainSampler(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]),
                                             SingleFitter_creation.log_joint_likelihood,
                                             burnin=10,
                                             production=10,
                                             args=(jnp.array([200., 0., -jnp.pi, 0.]),
                                                   jnp.array([210., 5., jnp.pi, 15.]))
                                             )
    assert True

# def test_NestedSampler(SingleFitter_creation):
#     SingleFitter_creation.compile_production_model(model="simple_sinusoid")
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
    SingleFitter_creation.compile_production_model(model="control_points")
    out = SingleFitter_creation.interp_gp(201, jnp.ones(SingleFitter_creation.control_points_time.size))
    assert jnp.allclose(out, 1)

def test_simple_sinusoid(SingleFitter_creation):
    out = SingleFitter_creation.simple_sinusoid(200, jnp.array([205., 1./12, jnp.pi/2., 81./12]))
    assert jnp.allclose(out, 2.1823329)

def test_flexible_sinusoid(SingleFitter_creation):
    out = SingleFitter_creation.flexible_sinusoid(200, jnp.array([205., 1./12, jnp.pi/2., 81./12, 0.1]))
    assert jnp.allclose(out, 2.04813439)

def test_dc14(SingleFitter_creation):
    SingleFitter_creation.compile_production_model(model="simple_sinusoid")
    a = SingleFitter_creation.dc14(params=jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]))
    SingleFitter_creation.compile_production_model(model="flexible_sinusoid")
    b = SingleFitter_creation.dc14(params=(jnp.array([205., 1./12, jnp.pi/2., 81./12, 0.1])))
    SingleFitter_creation.compile_production_model(model="control_points")
    c = SingleFitter_creation.dc14(params=jnp.ones(SingleFitter_creation.control_points_time.size))
    assert jnp.all(
        jnp.array([
            jnp.allclose(a, jnp.array([ 0.04804975,  0.63910482,  0.76016559,  0.37527574,
              -0.3795604 , 12.34915479, 14.50537971, 14.54228208,
              14.06061059, 13.67800247])),
            jnp.allclose(b, jnp.array([ 0.02669429,  0.35505823,  0.42231421,  0.20848652,
              -0.21086689, 12.90361593, 15.365865  , 15.52502989,
              14.93705643, 14.24820176])),
            jnp.allclose(c, jnp.array([-126.52752693, -126.57603706, -126.6245375 , -126.67302826,
              -126.72150935, -126.76998077, -126.81844254, -126.86689466,
              -126.91533713, -126.96376997]))
        ])
    )

def test_dc14_fine(SingleFitter_creation):
    SingleFitter_creation.compile_production_model(model="simple_sinusoid")
    a = SingleFitter_creation.dc14_fine(params=jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]))[-9:]
    SingleFitter_creation.compile_production_model(model="flexible_sinusoid")
    b = SingleFitter_creation.dc14_fine(params=jnp.array([205., 1./12, jnp.pi/2., 81./12, 0.1]))[-9:]
    SingleFitter_creation.compile_production_model(model="control_points")
    c = SingleFitter_creation.dc14_fine(params=jnp.ones(SingleFitter_creation.control_points_time.size))[-9:]
    assert jnp.all(
        jnp.array([
            jnp.allclose(a, jnp.array([13.42488917, 13.42474906, 13.42460886, 13.42446859,
              13.42432823, 13.42418779, 13.42404727, 13.42390667,
              13.42376598])),
            jnp.allclose(b, jnp.array([13.37697866, 13.37644154, 13.37590442, 13.37536731,
              13.37483019, 13.37429309, 13.37375598, 13.37321888,
              13.37268179])),
            jnp.allclose(c, jnp.array([-127.03599304, -127.03604107, -127.0360891 , -127.03613713,
              -127.03618515, -127.03623318, -127.03628121, -127.03632924,
              -127.03637727]))
        ])
    )

def test_log_likelihood(SingleFitter_creation):
    SingleFitter_creation.compile_production_model(model="simple_sinusoid")
    out = SingleFitter_creation.log_likelihood(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]))
    assert jnp.allclose(out, -134749.41424852)

def test_log_joint_likelihood(SingleFitter_creation):
    SingleFitter_creation.compile_production_model(model="simple_sinusoid")
    out = SingleFitter_creation.log_joint_likelihood(jnp.array([205., 1. / 12, jnp.pi / 2., 81. / 12]),
                                                    jnp.array([200., 0., -jnp.pi, 0.]),
                                                    jnp.array([210., 5., jnp.pi, 15.])
                                                    )
    assert jnp.allclose(out, -134749.41424852)

    SingleFitter_creation.compile_production_model(model="simple_sinusoid")
    out = SingleFitter_creation.log_joint_likelihood(jnp.array([205., 1. / 12, jnp.pi / 2., 81. / 12]),
                                                    jnp.array([200., 0., -jnp.pi, 0.]),
                                                    jnp.array([204., 5., jnp.pi, 15.])
                                                    )
    assert jnp.allclose(out, -np.inf)

def test_log_likelihood_gp(SingleFitter_creation):
    SingleFitter_creation.compile_production_model(model="control_points")
    out = SingleFitter_creation.log_likelihood_gp(jnp.ones(SingleFitter_creation.control_points_time.size))
    assert jnp.allclose(out, -7.15140844)

def test_log_joint_likelihood_gp(SingleFitter_creation):
    SingleFitter_creation.compile_production_model(model="control_points")
    out = SingleFitter_creation.log_joint_likelihood_gp(jnp.ones(SingleFitter_creation.control_points_time.size),
                                                        jnp.zeros((SingleFitter_creation.control_points_time.size)),
                                                        jnp.ones(SingleFitter_creation.control_points_time.size) * 100)
    assert jnp.allclose(out, -4855.11313943)

def test_grad_neg_log_joint_likelihood_gp(SingleFitter_creation):
    SingleFitter_creation.compile_production_model(model="control_points")
    out = SingleFitter_creation.grad_neg_log_joint_likelihood_gp(jnp.ones(SingleFitter_creation.control_points_time.size))
    assert jnp.allclose(out, jnp.array([41088.42057089,   503.61695263,   472.62284358,
               392.90213745,   334.28333181,   280.16811762,
               228.82478265,   163.3669408 ,   135.70935834]))

def test_fit_ControlPoints(SingleFitter_creation):
    SingleFitter_creation.compile_production_model(model="control_points")
    SingleFitter_creation.fit_ControlPoints(low_bound=0)
    assert True

# def test_plot_recovery(SingleFitter_creation):
#     SingleFitter_creation.compile_production_model(model="simple_sinusoid")
#     chain = SingleFitter_creation.MarkovChainSampler(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]),
#                                                      SingleFitter_creation.log_joint_simple_sinusoid,
#                                                      burnin=10,
#                                                      production=10,
#                                                      args=(jnp.array([200., 0., -jnp.pi, 0.]),
#                                                            jnp.array([210., 5., jnp.pi, 15.]))
#                                                      )
#     SingleFitter_creation.plot_recovery(chain,
#                                         time_data=SingleFitter_creation.time_data_fine,
#                                         true_production=jnp.ones(SingleFitter_creation.time_data_fine.size))
#     assert True
#
# def test_plot_samples(SingleFitter_creation):
#     SingleFitter_creation.compile_production_model(model="simple_sinusoid")
#     chain = SingleFitter_creation.MarkovChainSampler(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]),
#                                                      SingleFitter_creation.log_joint_simple_sinusoid,
#                                                      burnin=10,
#                                                      production=10,
#                                                      args=(jnp.array([200., 0., -jnp.pi, 0.]),
#                                                            jnp.array([210., 5., jnp.pi, 15.]))
#                                                      )
#     SingleFitter_creation.plot_samples(chain, 8)
#     assert True

