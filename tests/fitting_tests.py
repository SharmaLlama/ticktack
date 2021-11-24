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
    assert jnp.allclose(out, -223756.46745182)

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
    assert jnp.allclose(out, -223756.46745182)

def test_mf_log_joint_flexible_sinusoid(SingleFitter_creation, MultiFitter_creation):
    mf = fitting.MultiFitter()
    SingleFitter_creation.prepare_function(model="flexible_sinusoid")
    mf.add_SingleFitter(SingleFitter_creation)
    mf.add_SingleFitter(SingleFitter_creation)
    out = MultiFitter_creation.log_joint_flexible_sinusoid(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12, 0.18]),
                                                     jnp.array([200., 0., -jnp.pi, 0., 0.]),
                                                     jnp.array([210., 5., jnp.pi, 15., 2.])
                                                     )
    assert jnp.allclose(out, -223756.46745182)

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
            jnp.allclose(a, jnp.array([1.53796943, 1.50055152, 0.97640999, 0.13198755, -0.7645682,
                             -1.42855475, -1.64912179, -1.35622635, -0.64287515])),
            jnp.allclose(b, jnp.array([1.53796943, 1.50055152, 0.97640999, 0.13198755, -0.7645682,
                             -1.42855475, -1.64912179, -1.35622635, -0.64287515])),
            jnp.allclose(c, jnp.array([-125.53269265, -125.56971761, -125.62058337, -125.67101689,
                             -125.72267541, -125.77372637, -125.82646874, -125.87120292, -125.96980777]))
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
            jnp.allclose(a, jnp.array([0.95999231, 1.00453728, 1.04784445, 1.08986208, 1.13054003,
                             1.16982976, 1.20768432, 1.24405849, 1.27890888])),
            jnp.allclose(b, jnp.array([0.95999231, 1.00453728, 1.04784445, 1.08986208, 1.13054003,
                             1.16982976, 1.20768432, 1.24405849, 1.27890888])),
            jnp.allclose(c, jnp.array([-127.53537869, -127.62682645, -127.72037002, -127.81595568,
                             -127.91352902, -128.01303496, -128.11441803, -128.21762246, -128.32259232]))
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
    assert jnp.allclose(out, -111878.23372591)

def test_log_joint_simple_sinusoid(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="simple_sinusoid")
    out = SingleFitter_creation.log_joint_simple_sinusoid(jnp.array([205., 1. / 12, jnp.pi / 2., 81. / 12]),
                                                    jnp.array([200., 0., -jnp.pi, 0.]),
                                                    jnp.array([210., 5., jnp.pi, 15.])
                                                    )
    assert jnp.allclose(out, -111878.23372591)

def test_log_joint_flexible_sinusoid(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="flexible_sinusoid")
    out = SingleFitter_creation.log_joint_flexible_sinusoid(jnp.array([205., 1. / 12, jnp.pi / 2., 81./12, 0.18]),
                                                     jnp.array([200., 0., -jnp.pi, 0., 0.]),
                                                     jnp.array([210., 5., jnp.pi, 15., 2.])
                                                     )
    assert jnp.allclose(out, -111878.23372591)

def test_neg_log_likelihood_gp(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="control_points")
    out = SingleFitter_creation.neg_log_likelihood_gp(jnp.ones(SingleFitter_creation.control_points_time.size))
    assert jnp.allclose(out, 10.21255349)

def test_log_joint_gp(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="control_points")
    out = SingleFitter_creation.log_joint_gp(jnp.ones(SingleFitter_creation.control_points_time.size))
    assert jnp.allclose(out, -4835.77492731)

def test_neg_grad_log_joint_gp(SingleFitter_creation):
    SingleFitter_creation.prepare_function(model="control_points")
    out = SingleFitter_creation.neg_grad_log_joint_gp(jnp.ones(SingleFitter_creation.control_points_time.size))
    assert jnp.allclose(out, jnp.array([39097.58324264, 579.20894195, 428.80250487, 368.67443377, 297.74144928, 238.13667954,
                              178.53660702, 113.40072485, 55.81188067]))

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




