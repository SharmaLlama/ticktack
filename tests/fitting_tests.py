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
    assert jnp.allclose(out, -267301.06888754)

def test_mf_log_joint_likelihood(MultiFitter_creation):
    out = MultiFitter_creation.log_joint_likelihood(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]),
                                                         jnp.array([200., 0., -jnp.pi, 0.]),
                                                         jnp.array([210., 5., jnp.pi, 15.])
                                                         )
    assert jnp.allclose(out, -267301.06888754)

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
            jnp.allclose(a, jnp.array([ 0.13688862,  1.00019051,  1.54693847,  1.60351323,
                                        1.15194934,  0.33564009, 13.01094515, 15.12074837,
                                        15.11844421, 14.60362678])),
            jnp.allclose(b, jnp.array([ 0.07604924,  0.55566139,  0.85941026,  0.89084068,
                                        0.63997186,  0.18646672, 13.27147707, 15.7077855 ,
                                        15.84510528, 15.23869322])),
            jnp.allclose(c, jnp.array([-126.62452875, -126.65945078, -126.69707457, -126.74836151,
                                       -126.79919146, -126.85119089, -126.90269201, -126.95539917,
                                       -127.00205019, -127.08219414]))
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
            jnp.allclose(a, jnp.array([14.06854483, 14.06834237, 14.06814007, 14.06793791,
                                       14.06773591, 14.06753405, 14.06733234, 14.06713078,
                                       14.06692937])),
            jnp.allclose(b, jnp.array([14.2185199 , 14.21791158, 14.21730336, 14.21669524,
                                       14.21608721, 14.21547929, 14.21487148, 14.21426376,
                                       14.21365614])),
            jnp.allclose(c, jnp.array([-128.21542846, -128.21671172, -128.21799562, -128.21928017,
                                       -128.22056537, -128.22185122, -128.22313771, -128.22442485,
                                       -128.22571264]))
        ])
    )

def test_log_likelihood(SingleFitter_creation):
    SingleFitter_creation.compile_production_model(model="simple_sinusoid")
    out = SingleFitter_creation.log_likelihood(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]))
    assert jnp.allclose(out, -133676.3781144)

def test_log_joint_likelihood(SingleFitter_creation):
    SingleFitter_creation.compile_production_model(model="simple_sinusoid")
    out = SingleFitter_creation.log_joint_likelihood(jnp.array([205., 1. / 12, jnp.pi / 2., 81. / 12]),
                                                    jnp.array([200., 0., -jnp.pi, 0.]),
                                                    jnp.array([210., 5., jnp.pi, 15.])
                                                    )
    assert jnp.allclose(out, -133676.3781144)

def test_log_likelihood_gp(SingleFitter_creation):
    SingleFitter_creation.compile_production_model(model="control_points")
    out = SingleFitter_creation.log_likelihood_gp(jnp.ones(SingleFitter_creation.control_points_time.size))
    assert jnp.allclose(out, -10.21255349)

def test_log_joint_likelihood_gp(SingleFitter_creation):
    SingleFitter_creation.compile_production_model(model="control_points")
    out = SingleFitter_creation.log_joint_likelihood_gp(jnp.ones(SingleFitter_creation.control_points_time.size),
                                                        jnp.zeros((SingleFitter_creation.control_points_time.size)),
                                                        jnp.ones(SingleFitter_creation.control_points_time.size) * 100)
    assert jnp.allclose(out, -4832.12756846)

def test_grad_neg_log_joint_likelihood_gp(SingleFitter_creation):
    SingleFitter_creation.compile_production_model(model="control_points")
    out = SingleFitter_creation.grad_neg_log_joint_likelihood_gp(jnp.ones(SingleFitter_creation.control_points_time.size))
    assert jnp.allclose(out, jnp.array([41430.18874965,   515.81845717,   385.24052732,
                                        338.96075142,   281.56061807,   229.10357489,
                                        171.36231689,   108.43039156,    51.22351578]))

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




