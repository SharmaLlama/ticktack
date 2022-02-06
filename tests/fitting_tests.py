import numpy as np
import jax.numpy as jnp
import pytest
import ticktack
from ticktack import fitting

@pytest.fixture
def SingleFitter_creation():
    cbm = ticktack.load_presaved_model('Guttler14', production_rate_units='atoms/cm^2/s')
    sf = fitting.SingleFitter(cbm, 'Guttler14', hemisphere="north")
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

def test_get_data():
    fitting.get_data(event="993AD-N")
    fitting.get_data(event="993AD-S")
    fitting.get_data(event="775AD-early-N")
    fitting.get_data(event="775AD-early-S")
    fitting.get_data(event="775AD-late-N")
    # fitting.get_data(event="660BCE")
    fitting.get_data(event="5259BCE")
    fitting.get_data(event="5410BCE")
    fitting.get_data(event="7176BCE")
    assert True

def test_fit_event():
    fitting.fit_event(993,
                      event='993AD-S',
                      production_model='simple_sinusoid',
                      sampler="MCMC",
                      hemisphere='south',
                      burnin=10,
                      production=10)
    mf = fitting.fit_event(993,
                           event='993AD-S',
                           sampler=None, hemisphere='south')
    fitting.fit_event(993,
                      event='993AD-S',
                      sampler='MCMC', mf=mf, hemisphere='south',
                      burnin=10,
                      production=10)
    assert True

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
    assert jnp.allclose(out, -269500.51096454)

def test_mf_log_joint_likelihood(MultiFitter_creation):
    out = MultiFitter_creation.log_joint_likelihood(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]),
                                                         jnp.array([200., 0., -jnp.pi, 0.]),
                                                         jnp.array([210., 5., jnp.pi, 15.])
                                                         )
    assert jnp.allclose(out, -269500.51096454)

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
            jnp.allclose(a, jnp.array([ 0.04804975,  0.63910481,  0.76016558,  0.37527572,
              -0.37956041, 12.34915482, 14.50537975, 14.54228213,
              14.06061063, 13.67800251])),
            jnp.allclose(b, jnp.array([ 0.02669431,  0.35505823,  0.42231421,  0.20848651,
              -0.2108669 , 12.90361592, 15.36586499, 15.52502989,
              14.93705642, 14.24820176])),
            jnp.allclose(c, jnp.array([-126.51396122, -126.55161042, -126.60292686, -126.65378554,
              -126.70581391, -126.75734421, -126.81007942, -126.856763  ,
              -126.93691473, -127.52508099]))
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
            jnp.allclose(a, jnp.array([13.4248892 , 13.42474909, 13.42460889, 13.42446862,
              13.42432826, 13.42418782, 13.4240473 , 13.4239067 ,
              13.42376602])),
            jnp.allclose(b, jnp.array([13.37697867, 13.37644155, 13.37590443, 13.37536732,
              13.37483021, 13.3742931 , 13.373756  , 13.3732189 ,
              13.3726818 ])),
            jnp.allclose(c, jnp.array([-129.65468674, -129.6565079 , -129.65832947, -129.66015145,
              -129.66197385, -129.66379665, -129.66561987, -129.66744349,
              -129.66926753]))
        ])
    )

def test_log_likelihood(SingleFitter_creation):
    SingleFitter_creation.compile_production_model(model="simple_sinusoid")
    out = SingleFitter_creation.log_likelihood(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]))
    assert jnp.allclose(out, -134749.41427443)

def test_log_joint_likelihood(SingleFitter_creation):
    SingleFitter_creation.compile_production_model(model="simple_sinusoid")
    out = SingleFitter_creation.log_joint_likelihood(jnp.array([205., 1. / 12, jnp.pi / 2., 81. / 12]),
                                                    jnp.array([200., 0., -jnp.pi, 0.]),
                                                    jnp.array([210., 5., jnp.pi, 15.])
                                                    )
    assert jnp.allclose(out, -134749.41427443)

    SingleFitter_creation.compile_production_model(model="simple_sinusoid")
    out = SingleFitter_creation.log_joint_likelihood(jnp.array([205., 1. / 12, jnp.pi / 2., 81. / 12]),
                                                    jnp.array([200., 0., -jnp.pi, 0.]),
                                                    jnp.array([204., 5., jnp.pi, 15.])
                                                    )
    assert jnp.allclose(out, -np.inf)

def test_log_likelihood_gp(SingleFitter_creation):
    SingleFitter_creation.compile_production_model(model="control_points")
    out = SingleFitter_creation.log_likelihood_gp(jnp.ones(SingleFitter_creation.control_points_time.size))
    assert jnp.allclose(out, -10.21255349)

def test_log_joint_likelihood_gp(SingleFitter_creation):
    SingleFitter_creation.compile_production_model(model="control_points")
    out = SingleFitter_creation.log_joint_likelihood_gp(jnp.ones(SingleFitter_creation.control_points_time.size),
                                                        jnp.zeros((SingleFitter_creation.control_points_time.size)),
                                                        jnp.ones(SingleFitter_creation.control_points_time.size) * 100)
    assert jnp.allclose(out, -4848.15836128)

def test_grad_neg_log_joint_likelihood_gp(SingleFitter_creation):
    SingleFitter_creation.compile_production_model(model="control_points")
    out = SingleFitter_creation.grad_neg_log_joint_likelihood_gp(jnp.ones(SingleFitter_creation.control_points_time.size))
    assert jnp.allclose(out, jnp.array([40903.92798108,   603.14553586,   455.32578024,
               400.57668741,   335.71501679,   280.9520573 ,
               233.9208041 ,   148.3065663 ,   162.2101442 ]))

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




