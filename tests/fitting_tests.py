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
    sf.time_oversample = 1008
    sf.offset = 0
    sf.annual = jnp.arange(sf.start, sf.end + 1)
    sf.mask = jnp.in1d(sf.annual, sf.time_data)
    return sf

@pytest.fixture
def MultiFitter_creation(SingleFitter_creation):
    mf = fitting.MultiFitter()
    SingleFitter_creation.prepare_function(model="simple_sinusoid")
    mf.add_SingleFitter(SingleFitter_creation)
    mf.add_SingleFitter(SingleFitter_creation)
    return mf

def test_get_time_period(SingleFitter_creation):
    mf = fitting.MultiFitter()
    sf = SingleFitter_creation
    mf.add_SingleFitter(sf)
    sf.start = 199
    sf.end = 209
    mf.add_SingleFitter(sf)
    start, end = mf.get_time_period()
    assert jnp.allclose(jnp.array([start, end]), jnp.array([199, 209]))

def test_get_data():
    fitting.get_data(event="775AD", hemisphere="south")
    fitting.get_data(event="775AD", hemisphere="north")
    fitting.get_data(event="993AD", hemisphere="south")
    fitting.get_data(event="993AD", hemisphere="north")
    fitting.get_data(event="660BCE", hemisphere="north")
    fitting.get_data(event="5259BCE", hemisphere="north")
    fitting.get_data(event="5410BCE", hemisphere="north")
    fitting.get_data(event="7176BCE", hemisphere="north")
    assert True

def test_fit_event():
    fitting.fit_event(-660,
                      event='660BCE',
                      production_model='simple_sinusoid',
                      sampler="MCMC",
                      burnin=10,
                      production=10)
    mf = fitting.fit_event(-660,
                           event='660BCE',
                           sampler=None)
    fitting.fit_event(-660,
                      event='660BCE',
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
    assert jnp.allclose(out, -272052.45881277)

def test_mf_log_prior_simple_sinusoid(MultiFitter_creation):
    out = MultiFitter_creation.log_prior_simple_sinusoid(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]),
                                                         jnp.array([200., 0., -jnp.pi, 0.]),
                                                         jnp.array([210., 5., jnp.pi, 15.])
                                                         )
    assert jnp.allclose(out, 0)

def test_mf_log_prior_flexible_sinusoid(MultiFitter_creation):
    out = MultiFitter_creation.log_prior_flexible_sinusoid(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12, 0.18]),
                                                           jnp.array([200., 0., -jnp.pi, 0., 0.]),
                                                           jnp.array([210., 5., jnp.pi, 15., 2.])
                                                           )
    assert jnp.allclose(out, 0)

def test_mf_log_joint_simple_sinusoid(MultiFitter_creation):
    out = MultiFitter_creation.log_joint_simple_sinusoid(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]),
                                                   jnp.array([200., 0., -jnp.pi, 0.]),
                                                   jnp.array([210., 5., jnp.pi, 15.])
                                                   )
    assert jnp.allclose(out, -272052.45881277)

def test_mf_log_joint_flexible_sinusoid(SingleFitter_creation, MultiFitter_creation):
    mf = fitting.MultiFitter()
    SingleFitter_creation.prepare_function(model="flexible_sinusoid")
    mf.add_SingleFitter(SingleFitter_creation)
    mf.add_SingleFitter(SingleFitter_creation)
    out = MultiFitter_creation.log_joint_flexible_sinusoid(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12, 0.18]),
                                                     jnp.array([200., 0., -jnp.pi, 0., 0.]),
                                                     jnp.array([210., 5., jnp.pi, 15., 2.])
                                                     )
    assert jnp.allclose(out, -272052.45881277)

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

def test_NestedSampler(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="simple_sinusoid")
    SingleFitter_creation.NestedSampler(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12]),
                                        SingleFitter_creation.log_likelihood,
                                        low_bound = jnp.array([770., 0., -jnp.pi, 0.]),
                                        high_bound = jnp.array([780., 5., jnp.pi, 15.])
                                        )
    assert True

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
            jnp.allclose(a, jnp.array([ 2.02663583,  2.05373713,  1.58866043,  0.75720779,
                                        -0.19198642, 10.32059596, 14.71941444, 15.62882696,
                                        15.69838433, 15.55767994])),
            jnp.allclose(b, jnp.array([ 2.02663583,  2.05373713,  1.58866043,  0.75720779,
                                        -0.19198642, 10.32059596, 14.71941444, 15.62882696,
                                        15.69838433, 15.55767994])),
            jnp.allclose(c, jnp.array([-125.57659792, -125.60449947, -125.65646164, -125.70621563,
                                       -125.75778369, -125.80939222, -125.85991583, -125.91398731,
                                       -125.95339274, -126.37412093]))
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
            jnp.allclose(a, jnp.array([15.5249451 , 15.51212831, 15.49885171, 15.48508867,
                                       15.47081085, 15.45598851, 15.4405906 , 15.42458504,
                                       15.40793869])),
            jnp.allclose(b, jnp.array([15.5249451 , 15.51212831, 15.49885171, 15.48508867,
                                       15.47081085, 15.45598851, 15.4405906 , 15.42458504,
                                       15.40793869])),
            jnp.allclose(c, jnp.array([-126.50727426, -126.56247561, -126.62035966, -126.68090697,
                                       -126.7440939 , -126.80989303, -126.87827354, -126.94920134,
                                       -127.02263934]))
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
    assert jnp.allclose(out, -136026.22940639)

def test_log_joint_simple_sinusoid(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="simple_sinusoid")
    out = SingleFitter_creation.log_joint_simple_sinusoid(jnp.array([205., 1. / 12, jnp.pi / 2., 81. / 12]),
                                                    jnp.array([200., 0., -jnp.pi, 0.]),
                                                    jnp.array([210., 5., jnp.pi, 15.])
                                                    )
    assert jnp.allclose(out, -136026.22940639)

def test_log_joint_flexible_sinusoid(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="flexible_sinusoid")
    out = SingleFitter_creation.log_joint_flexible_sinusoid(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12, 0.18]),
                                                     jnp.array([200., 0., -jnp.pi, 0., 0.]),
                                                     jnp.array([210., 5., jnp.pi, 15., 2.])
                                                     )
    assert jnp.allclose(out, -136026.22940639)

def test_neg_log_likelihood_gp(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="control_points")
    out = SingleFitter_creation.neg_log_likelihood_gp(jnp.ones(SingleFitter_creation.control_points_time.size))
    assert jnp.allclose(out, 10.21255349)

def test_log_joint_gp(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="control_points")
    out = SingleFitter_creation.log_joint_gp(jnp.ones(SingleFitter_creation.control_points_time.size))
    assert jnp.allclose(out, -5144.73323962)

def test_neg_grad_log_joint_gp(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="control_points")
    out = SingleFitter_creation.neg_grad_log_joint_gp(jnp.ones(SingleFitter_creation.control_points_time.size))
    assert jnp.allclose(out, jnp.array([42388.87425578,   600.34169159,   452.92252801,
                                        399.03381353,   334.53214683,   279.11980775,
                                        227.05368143,   146.48346689,   134.61677252]))

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
                                        time_data=SingleFitter_creation.time_grid_fine,
                                        true_production=jnp.ones(SingleFitter_creation.time_grid_fine.size))
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




