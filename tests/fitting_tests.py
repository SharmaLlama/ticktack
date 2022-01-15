import numpy as np
import jax.numpy as jnp
import pytest
import ticktack
from ticktack import fitting


@pytest.fixture
def SingleFitter_creation():
    cbm = ticktack.load_presaved_model('Guttler14', production_rate_units='atoms/cm^2/s')
    sf = fitting.SingleFitter(cbm, 'Guttler14', hemisphere="south")
    sf.time_data = jnp.arange(200, 210)
    sf.d14c_data_error = jnp.ones((sf.time_data.size,))
    sf.d14c_data = jnp.array([-169.81482498, -168.05109886, -163.81278239, -158.13313339,
                              -153.23525037, -150.89762995, -151.11804461, -152.1036496 ,
                              -151.60492619, -151.60492619])
    sf.start = np.nanmin(sf.time_data)
    sf.end = np.nanmax(sf.time_data)
    sf.burn_in_time = jnp.arange(sf.start - 1000 - 1, sf.start - 1)
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
    fitting.get_data(event="993AD-N")
    fitting.get_data(event="993AD-S")
    fitting.get_data(event="775AD-early-N")
    fitting.get_data(event="775AD-early-S")
    fitting.get_data(event="775AD-late-N")
    fitting.get_data(event="660BCE_Ew")
    fitting.get_data(event="660BCE_Lw")
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
#     SingleFitter_creation.prepare_function(model="simple_sinusoid")
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
#     SingleFitter_creation.prepare_function(model="simple_sinusoid")
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
    assert jnp.allclose(out, -266986.37486027)

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
    assert jnp.allclose(out, -266986.37486027)

def test_mf_log_joint_flexible_sinusoid(SingleFitter_creation):
    mf = fitting.MultiFitter()
    SingleFitter_creation.prepare_function(model="flexible_sinusoid")
    mf.add_SingleFitter(SingleFitter_creation)
    mf.add_SingleFitter(SingleFitter_creation)
    mf.compile()
    out = mf.log_joint_flexible_sinusoid(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12, 0.18]),
                                                     jnp.array([200., 0., -jnp.pi, 0., 0.]),
                                                     jnp.array([210., 5., jnp.pi, 15., 2.])
                                                     )
    assert jnp.allclose(out, -266986.37486027)

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
            jnp.allclose(a, jnp.array([ 1.10286451,  1.56875159,  1.53089673,  1.00132181,
                                        0.14819373, -0.75757673, 12.16878374, 14.79044244,
                                        15.39825221, 15.39806953])),
            jnp.allclose(b, jnp.array([ 1.10286451,  1.56875159,  1.53089673,  1.00132181,
                                        0.14819373, -0.75757673, 12.16878374, 14.79044244,
                                        15.39825221, 15.39806953])),
            jnp.allclose(c, jnp.array([-125.59404218, -125.62859294, -125.66584556, -125.71676144,
                                       -125.76722043, -125.81884901, -125.8699794 , -125.92231592,
                                       -125.96859642, -126.04836994]))
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
            jnp.allclose(a, jnp.array([15.12435115, 15.12406633, 15.12378131, 15.1234961 ,
                                       15.12321069, 15.12292508, 15.12263928, 15.12235328,
                                       15.12206708])),
            jnp.allclose(b, jnp.array([15.12435115, 15.12406633, 15.12378131, 15.1234961 ,
                                       15.12321069, 15.12292508, 15.12263928, 15.12235328,
                                       15.12206708])),
            jnp.allclose(c, jnp.array([-127.18105239, -127.18233528, -127.18361882, -127.18490301,
                                       -127.18618784, -127.18747332, -127.18875945, -127.19004622,
                                       -127.19133364]))
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
    assert jnp.allclose(out, -133505.43112876)

def test_log_joint_simple_sinusoid(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="simple_sinusoid")
    out = SingleFitter_creation.log_joint_simple_sinusoid(jnp.array([205., 1. / 12, jnp.pi / 2., 81. / 12]),
                                                    jnp.array([200., 0., -jnp.pi, 0.]),
                                                    jnp.array([210., 5., jnp.pi, 15.])
                                                    )
    assert jnp.allclose(out, -133505.43112876)

def test_log_joint_flexible_sinusoid(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="flexible_sinusoid")
    out = SingleFitter_creation.log_joint_flexible_sinusoid(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12, 0.18]),
                                                     jnp.array([200., 0., -jnp.pi, 0., 0.]),
                                                     jnp.array([210., 5., jnp.pi, 15., 2.])
                                                     )
    assert jnp.allclose(out, -133505.43112876)

def test_neg_log_likelihood_gp(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="control_points")
    out = SingleFitter_creation.neg_log_likelihood_gp(jnp.ones(SingleFitter_creation.control_points_time.size))
    assert jnp.allclose(out, 10.21255349)

def test_log_joint_gp(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="control_points")
    out = SingleFitter_creation.log_joint_gp(jnp.ones(SingleFitter_creation.control_points_time.size))
    assert jnp.allclose(out, -5149.16859566)

def test_neg_grad_log_joint_gp(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="control_points")
    out = SingleFitter_creation.neg_grad_log_joint_gp(jnp.ones(SingleFitter_creation.control_points_time.size))
    assert jnp.allclose(out, jnp.array([42839.95321812,   535.24664399,   400.76549536,
                                        353.05217581,   293.41853198,   238.71060548,
                                        178.52838771,   112.99122061,    53.38319684]))

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




