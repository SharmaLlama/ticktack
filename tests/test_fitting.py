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
    sf.d14c_data = jnp.array([-164.90491016, -164.31385509, -164.19279432, -164.57768418,
             -165.33252032, -152.60046413, -150.44363798, -150.40669362,
             -150.88854914, -151.27141576]) # output of simple_sinusoid with data = sf.dc14(true_params)
    # true params = jnp.array([205., np.log10(1. / 12), jnp.pi / 2., np.log10(81. / 12)])
    sf.start = np.nanmin(sf.time_data)
    sf.end = np.nanmax(sf.time_data)
    sf.burn_in_time = jnp.arange(sf.start - 1000, sf.start, 1.)
    sf.oversample = 1008
    sf.annual = jnp.arange(sf.start, sf.end + 1)
    sf.time_data_fine = jnp.linspace(jnp.min(sf.annual), jnp.max(sf.annual) + 2, (sf.annual.size + 1) * sf.oversample)
    sf.offset = jnp.mean(sf.d14c_data[:4])
    sf.mask = jnp.in1d(sf.annual, sf.time_data)
    sf.growth = sf.get_growth_vector("april-september")
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
#     chain = SingleFitter_creation.MarkovChainSampler(jnp.array([205.,np.log10(1. / 12), jnp.pi / 2., np.log10(81./12)]),
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
#     chain = SingleFitter_creation.MarkovChainSampler(jnp.array([205.,np.log10(1. / 12), jnp.pi / 2., np.log10(81./12)]),
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
    out = MultiFitter_creation.multi_likelihood(params=jnp.array([205., np.log10(1. / 12), jnp.pi / 2., np.log10(81. / 12)]))
    assert jnp.allclose(out, -29.38086526, rtol=1e-3)

def test_mf_log_joint_likelihood(MultiFitter_creation):
    out = MultiFitter_creation.log_joint_likelihood(jnp.array([205.,np.log10(1. / 12), jnp.pi / 2., np.log10(81./12)]),
                                                         jnp.array([200., -2, -jnp.pi, 0.]),
                                                         jnp.array([210., 1, jnp.pi, 15.])
                                                         )
    assert jnp.allclose(out, -29.38086526, rtol=1e-3)

    out = MultiFitter_creation.log_joint_likelihood(jnp.array([211.,np.log10(1. / 12), jnp.pi / 2., np.log10(81./12)]),
                                                         jnp.array([200., -2, -jnp.pi, 0.]),
                                                         jnp.array([210., 1, jnp.pi, 15.])
                                                         )
    assert jnp.allclose(out, -np.inf)

def test_MarkovChainSampler(SingleFitter_creation):
    SingleFitter_creation.compile_production_model(model="simple_sinusoid")
    SingleFitter_creation.MarkovChainSampler(jnp.array([205.,np.log10(1. / 12), jnp.pi / 2., np.log10(81./12)]),
                                             SingleFitter_creation.log_joint_likelihood,
                                             burnin=10,
                                             production=10,
                                             args=(jnp.array([200., -2, -jnp.pi, -2]),
                                                   jnp.array([210., 1, jnp.pi, 1.5]))
                                             )
    assert True

# def test_NestedSampler(SingleFitter_creation):
#     SingleFitter_creation.compile_production_model(model="simple_sinusoid")
#     SingleFitter_creation.NestedSampler(jnp.array([205.,np.log10(1. / 12), jnp.pi / 2., np.log10(81./12)]),
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
    out = SingleFitter_creation.simple_sinusoid(200, jnp.array([205., np.log10(1./12), jnp.pi/2., np.log10(81./12)]))
    assert jnp.allclose(out, 2.04261539, rtol=1e-4)

def test_flexible_sinusoid(SingleFitter_creation):
    out = SingleFitter_creation.flexible_sinusoid(200, jnp.array([205., np.log10(1./12), jnp.pi/2., np.log10(81./12), np.log10(0.1)]))
    assert jnp.allclose(out, 1.91700855, rtol=1e-4)

def test_dc14(SingleFitter_creation):
    SingleFitter_creation.compile_production_model(model="simple_sinusoid")
    a = SingleFitter_creation.dc14(params=jnp.array([205.,np.log10(1. / 12), jnp.pi / 2., np.log10(81./12)]))
    SingleFitter_creation.compile_production_model(model="flexible_sinusoid")
    b = SingleFitter_creation.dc14(params=(jnp.array([205., np.log10(1./12), jnp.pi/2., np.log10(81./12), np.log10(0.1)])))
    SingleFitter_creation.compile_production_model(model="control_points")
    c = SingleFitter_creation.dc14(params=jnp.ones(SingleFitter_creation.control_points_time.size))
    assert jnp.all(
        jnp.array([
            jnp.allclose(a-SingleFitter_creation.offset, jnp.array([ 0.04804975,  0.63910481,  0.76016558,  0.37527572,
              -0.37956041, 13.27817203, 15.62874622, 15.68695714,
              15.1560305 , 14.70020043]), rtol=1e-3),
            jnp.allclose(b-SingleFitter_creation.offset, jnp.array([ 0.02669431,  0.35505823,  0.42231421,  0.20848651,
              -0.2108669 , 13.83314212, 16.48983922, 16.6703216 ,
              16.03306547, 15.27094911]), rtol=1e-3),
            jnp.allclose(c-SingleFitter_creation.offset, jnp.array([-116.69701825, -116.7417594 , -116.78649162, -116.83121491,
              -116.87592927, -116.92063473, -116.96533128, -117.01001892,
              -117.05469768, -117.09936755]), rtol=1e-3)
        ])
    )

def test_dc14_fine(SingleFitter_creation):
    SingleFitter_creation.compile_production_model(model="simple_sinusoid")
    a = SingleFitter_creation.dc14_fine(params=jnp.array([205.,np.log10(1. / 12), jnp.pi / 2., np.log10(81./12)]))[-9:]
    SingleFitter_creation.compile_production_model(model="flexible_sinusoid")
    b = SingleFitter_creation.dc14_fine(params=jnp.array([205., np.log10(1./12), jnp.pi/2., np.log10(81./12), np.log10(0.1)]))[-9:]
    SingleFitter_creation.compile_production_model(model="control_points")
    c = SingleFitter_creation.dc14_fine(params=jnp.ones(SingleFitter_creation.control_points_time.size))[-9:]
    assert jnp.all(
        jnp.array([
            jnp.allclose(a-SingleFitter_creation.offset, jnp.array([14.33477116, 14.33456044, 14.33434965, 14.33413878,
              14.33392784, 14.33371683, 14.33350574, 14.33329458,
              14.33308335]), rtol=1e-3),
            jnp.allclose(b-SingleFitter_creation.offset, jnp.array([14.28734964, 14.28674187, 14.28613412, 14.28552638,
              14.28491865, 14.28431093, 14.28370322, 14.28309552,
              14.28248783]), rtol=1e-3),
            jnp.allclose(c-SingleFitter_creation.offset, jnp.array([-117.16597927, -117.16602357, -117.16606786, -117.16611216,
              -117.16615646, -117.16620076, -117.16624506, -117.16628935,
              -117.16633365]), rtol=1e-3)
        ])
    )

def test_log_likelihood(SingleFitter_creation):
    SingleFitter_creation.compile_production_model(model="simple_sinusoid")
    out = SingleFitter_creation.log_likelihood(jnp.array([205.,np.log10(1. / 12), jnp.pi / 2., np.log10(81./12)]))
    assert jnp.allclose(out, -6.27145615, rtol=1e-4)

def test_log_joint_likelihood(SingleFitter_creation):
    SingleFitter_creation.compile_production_model(model="simple_sinusoid")
    out = SingleFitter_creation.log_joint_likelihood(jnp.array([205.,np.log10(1. / 12), jnp.pi / 2., np.log10(81. / 12)]),
                                                    jnp.array([200., -2, -jnp.pi, -2.]),
                                                    jnp.array([210., 1., jnp.pi, 1.5])
                                                    )
    assert jnp.allclose(out, -6.27145615, rtol=1e-4)

    SingleFitter_creation.compile_production_model(model="simple_sinusoid")
    out = SingleFitter_creation.log_joint_likelihood(jnp.array([205.,np.log10(1. / 12), jnp.pi / 2., np.log10(81. / 12)]),
                                                    jnp.array([200., -2, -jnp.pi, -2.]),
                                                    jnp.array([204., 1., jnp.pi, 1.5])
                                                    )
    assert jnp.allclose(out, -np.inf)

def test_log_likelihood_gp(SingleFitter_creation):
    SingleFitter_creation.compile_production_model(model="control_points")
    out = SingleFitter_creation.log_likelihood_gp(jnp.ones(SingleFitter_creation.control_points_time.size))
    assert jnp.allclose(out, -7.15140844, rtol=1e-4)

def test_log_joint_likelihood_gp(SingleFitter_creation):
    SingleFitter_creation.compile_production_model(model="control_points")
    out = SingleFitter_creation.log_joint_likelihood_gp(jnp.ones(SingleFitter_creation.control_points_time.size),
                                                        jnp.zeros((SingleFitter_creation.control_points_time.size)),
                                                        jnp.ones(SingleFitter_creation.control_points_time.size) * 100)
    assert jnp.allclose(out, -76510.23162792, rtol=1e-4) # seems big, used to be -4855.11313943

# def test_grad_neg_log_joint_likelihood_gp(SingleFitter_creation):
#     SingleFitter_creation.compile_production_model(model="control_points")
#     out = SingleFitter_creation.grad_neg_log_joint_likelihood_gp(jnp.ones(SingleFitter_creation.control_points_time.size))
#     assert jnp.allclose(out, jnp.array([41088.42057089,   503.61695263,   472.62284358,
#                392.90213745,   334.28333181,   280.16811762,
#                228.82478265,   163.3669408 ,   135.70935834]))

def test_fit_ControlPoints(SingleFitter_creation):
    SingleFitter_creation.compile_production_model(model="control_points")
    SingleFitter_creation.fit_ControlPoints(low_bound=0)
    assert True

# def test_plot_recovery(SingleFitter_creation):
#     SingleFitter_creation.compile_production_model(model="simple_sinusoid")
#     chain = SingleFitter_creation.MarkovChainSampler(jnp.array([205.,np.log10(1. / 12), jnp.pi / 2., np.log10(81./12)]),
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
#     chain = SingleFitter_creation.MarkovChainSampler(jnp.array([205.,np.log10(1. / 12), jnp.pi / 2., np.log10(81./12)]),
#                                                      SingleFitter_creation.log_joint_simple_sinusoid,
#                                                      burnin=10,
#                                                      production=10,
#                                                      args=(jnp.array([200., 0., -jnp.pi, 0.]),
#                                                            jnp.array([210., 5., jnp.pi, 15.]))
#                                                      )
#     SingleFitter_creation.plot_samples(chain, 8)
#     assert True

