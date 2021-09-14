import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
import celerite2.jax
from celerite2.jax import terms as jax_terms
import re
import jax.numpy as jnp
import jax
from jax import grad, jit, partial
import ticktack
from astropy.table import Table
from tqdm import tqdm
import emcee
import corner
import scipy

rcParams['figure.figsize'] = (16.0, 8.0)

class CarbonFitter():
    """
    """
    def __init__(self, cbm, production_rate_units='atoms/cm^2/s',target_C_14=707.):
        if isinstance(cbm, str):
            try:
                if cbm in ['Guttler14', 'Brehm21', 'Miyake17', 'Buntgen18']:
                    cbm = ticktack.load_presaved_model(cbm, production_rate_units=production_rate_units)
                else:
                    cbm = load_model(cbm, production_rate_units=production_rate_units)
            except:
                raise ValueError('Must be a valid CBM model')
        self.cbm = cbm
        self.cbm.compile()
        self.steady_state_production = self.cbm.equilibrate(target_C_14=target_C_14) # 707 default for Guttler
        self.steady_state_y0 = self.cbm.equilibrate(production_rate=self.steady_state_production)

    def load_data(self, file_name, resolution=1000, fine_grid=0.05, time_oversample=1000, num_offset=4):
        data = Table.read(file_name, format="ascii")
        self.time_data = jnp.array(data["year"])
        self.d14c_data = jnp.array(data["d14c"])
        self.d14c_data_error = jnp.array(data["sig_d14c"])
        self.start = np.nanmin(self.time_data)
        self.end = np.nanmax(self.time_data)
        self.resolution = resolution
        self.burn_in_time = jnp.linspace(self.start-1000, self.start, self.resolution)
        self.time_grid_fine = jnp.arange(self.start, self.end, fine_grid)
        self.time_oversample = time_oversample
        self.offset = jnp.mean(self.d14c_data[:num_offset])
        self.annual = jnp.arange(self.start, self.end + 1)
        self.mask = jnp.in1d(self.annual, self.time_data)[:-1]

    def prepare_function(self, **kwargs):
        self.production = None
        self.gp = None
        try:
            custom_function = kwargs['custom_function']
            f = kwargs['f']
        except:
            custom_function = False
            f = None

        try:
            use_control_points = kwargs['use_control_points']
        except:
            use_control_points = False

        try:
            interp = kwargs['interp']
        except:
            interp = 'linear'

        try:
            dense_years = kwargs['dense_years']
        except:
            dense_years = 3

        try:
            gap_years = kwargs['gap_years']
        except:
            gap_years = 5

        try:
            production = kwargs['production']
        except:
            production = None

        try:
            fit_solar_params = kwargs['fit_solar']
        except:
            fit_solar_params = None

        if production == 'miyake':
            if fit_solar_params:
                self.production = self.miyake_event_flexible_solar
            else:
                self.production = self.miyake_event_fixed_solar

        if custom_function is True and f is not None:
            self.production = f

        if use_control_points is True:
            if interp == "linear":
                control_points_time = [self.start - 1, self.start]

                n = len(self.time_data[:-1])
                for i in range(n):
                    if (self.time_data[i] - control_points_time[-1]<=2) & (self.time_data[i] - control_points_time[-1]>0):
                        control_points_time.append(float(self.time_data[i]) + dense_years)
                    elif self.time_data[i] >= control_points_time[-1] + gap_years:
                        control_points_time.append(float(self.time_data[i]))
                control_points_time = np.array(control_points_time)[control_points_time <= self.end]

                self.control_points_time = jnp.array(control_points_time)
                self.production = self.interp_linear

            elif interp == "gp":
                self.control_points_time = jnp.arange(self.start, self.end)
                self.production = self.interp_gp
                self.gp = True


    @partial(jit, static_argnums=(0,))
    def interp_linear(self, tval, *args):
        control_points = jnp.squeeze(jnp.array(list(args)))
        return jnp.interp(tval, self.control_points_time, control_points)

    @partial(jit, static_argnums=(0,))
    def gp_neg_log_likelihood(self, params):
        control_points = params
        mean = params[0]
        kernel = jax_terms.Matern32Term(sigma=2., rho=2.)
        gp = celerite2.jax.GaussianProcess(kernel, mean=mean)
        gp.compute(self.control_points_time)
        return -gp.log_likelihood(control_points)

    @partial(jit, static_argnums=(0,))
    def gp_log_likelihood(self, params):
        control_points = params
        mean = params[0]
        kernel = jax_terms.Matern32Term(sigma=2., rho=2.)
        gp = celerite2.jax.GaussianProcess(kernel, mean=mean)
        gp.compute(self.control_points_time)
        return gp.log_likelihood(control_points)

    @partial(jit, static_argnums=(0,))
    def interp_gp(self, tval, *args):
        tval = tval.reshape(-1)
        params = jnp.squeeze(jnp.array(list(args)))
        control_points = params
        mean = params[0]

        kernel = jax_terms.Matern32Term(sigma=2., rho=2.)
        gp = celerite2.jax.GaussianProcess(kernel, mean=mean)
        gp.compute(self.control_points_time)
        mu = gp.predict(control_points, t=tval, return_var=False)
        mu = (tval > self.start) * mu +  (tval <= self.start) * mean
        return mu

    @partial(jit, static_argnums=(0,))
    def sum_interp_gp(self, *args):
        mu = self.interp_gp(self.annual, *args)
        return jnp.sum(mu)

    @partial(jit, static_argnums=(0,))
    def grad_sum_interp_gp(self, *args):
        return grad(self.sum_interp_gp)(*args)

    @partial(jit, static_argnums=(0,)) 
    def super_gaussian(self, t, start_time, duration, area):
        middle = start_time+duration/2.
        height = area/duration
        return height*jnp.exp(- ((t-middle)/(1./1.93516*duration))**16.)

    @partial(jit, static_argnums=(0,)) 
    def miyake_event_fixed_solar(self, t, start_time, duration, phase, area):
        height = self.super_gaussian(t, start_time, duration, area)
        prod = self.steady_state_production + 0.18 * self.steady_state_production * jnp.sin(2 * np.pi / 11 * t + phase) + height
        return prod

    @partial(jit, static_argnums=(0,))
    def miyake_event_flexible_solar(self, t, start_time, duration, phase, area, omega, amplitude):
        height = self.super_gaussian(t, start_time, duration, area)
        prod = self.steady_state_production + amplitude * self.steady_state_production * jnp.sin(
            omega * t + phase) + height
        return prod

    @partial(jit, static_argnums=(0,)) 
    def run(self, time_values, y0, params=()):
        burn_in, _ = self.cbm.run(time_values, production=self.production, args=params, y0=y0)
        return burn_in

    @partial(jit, static_argnums=(0, 2))
    def run_D_14_C_values(self, time_out, time_oversample, y0, params=()):
        d_14_c = self.cbm.run_D_14_C_values(time_out, time_oversample, 
                                       production=self.production, args=params, y0=y0,
                                       steady_state_solutions=self.steady_state_y0)
        return d_14_c


    @partial(jit, static_argnums=(0,))
    def dc14(self, params=()):
        burn_in = self.run(self.burn_in_time, self.steady_state_y0, params=params)
        d_14_c = self.run_D_14_C_values(self.annual, self.time_oversample, burn_in[-1, :], params=params)
        d_14_c = d_14_c[self.mask]
        return d_14_c + self.offset

    @partial(jit, static_argnums=(0,))
    def dc14_fine(self, params=()):
        burn_in = self.run(self.burn_in_time, self.steady_state_y0, params=params)
        data, solution = self.cbm.run(self.time_grid_fine, production=self.production, args=params, y0=burn_in[-1,:])
        d_14_c = self.cbm._to_d14c(data,self.steady_state_y0)
        return d_14_c + self.offset

    @partial(jit, static_argnums=(0,))
    def log_like(self, params=()):
        d_14_c = self.dc14(params=params)
        
        chi2 = jnp.sum(((self.d14c_data[:-1] - d_14_c)/self.d14c_data_error[:-1])**2)
        like = -0.5*chi2
        return like

    @partial(jit, static_argnums=(0,))
    def log_prior(self, params=()):
        lp = jnp.sum(jnp.where((params<=0), -np.inf, 0))
        return lp

    @partial(jit, static_argnums=(0,))
    def log_prob(self, params=()):
        lp = self.log_prior(params=params)
        pos = self.log_like(params=params)
        return lp + pos

    @partial(jit, static_argnums=(0,))
    def loss_chi2(self, params=()):
        d_14_c = self.dc14(params=params)
        chi2 = jnp.sum(((self.d14c_data[:-1] - d_14_c) / self.d14c_data_error[:-1]) ** 2)
        # chi2 += 10 * jnp.sum(((self.d14c_data[:4] - d_14_c[:4]) / self.d14c_data_error[:4]) ** 2)
        return 0.5*chi2

    @partial(jit, static_argnums=(0,))
    def loss_chi2_avg(self, params=(), k=1):
        d_14_c = self.dc14(params=params)
        chi2 = jnp.sum(((self.d14c_data[:-1] - d_14_c) / self.d14c_data_error[:-1]) ** 2)
        return 0.5*chi2/(k*len(d_14_c))

    @partial(jit, static_argnums=(0,))
    def gp_likelihood(self, params=()):
        chi2 = self.loss_chi2(params=params)
        return chi2 + self.gp_neg_log_likelihood(params)

    @partial(jit, static_argnums=(0,))
    def gp_sampling_likelihood(self, params=()):
        chi2 = self.loss_chi2(params=params)
        return -chi2 + self.gp_log_likelihood(params)

    @partial(jit, static_argnums=(0,))
    def grad_gp_likelihood(self, params=()):
        return grad(self.gp_likelihood)(params)

    @partial(jit, static_argnums=(0,))
    def gp_likelihood_avg(self, params=(), k=1):
        chi2 = self.loss_chi2_avg(params=params, k=k)
        return chi2 + self.gp_neg_log_likelihood(params)

    def fit_cp(self, low_bound=0, avg=False, k=1):
        steady_state = self.steady_state_production * jnp.ones((len(self.control_points_time),))
        params = steady_state
        bounds = tuple([(low_bound, None)] * len(steady_state))

        if self.gp:
            if avg:
                soln = scipy.optimize.minimize(self.gp_likelihood_avg, params, args=(k), bounds=bounds,)
            else:
                soln = scipy.optimize.minimize(self.gp_likelihood, params, bounds=bounds,
                                               options={'maxiter': 20000})
        else:
            if avg:
                soln = scipy.optimize.minimize(self.loss_chi2_avg, params, args=(k), bounds=bounds,
                                               method="L-BFGS-B", options={'maxiter': 20000})
            else:
                soln = scipy.optimize.minimize(self.loss_chi2, params, bounds=bounds,
                                               method="L-BFGS-B", options={'maxiter': 20000})
        return soln


    def sampling(self, params, likelihood, burnin=500, production=1000):
        initial = params
        ndim, nwalkers = len(initial), 5*len(initial)
        if self.gp:
            ndim, nwalkers = len(initial), 2*len(initial)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihood)

        print("Running burn-in...")
        p0 = initial + 1e-5 * np.random.rand(nwalkers, ndim)
        p0, lp, _ = sampler.run_mcmc(p0, burnin, progress=True);

        print("Running production...")
        sampler.reset()
        sampler.run_mcmc(p0, production, progress=True);
        return sampler

    def plot_recovery(self, sampler, time_data=None, true_production=None):
        mean = np.mean(sampler.flatchain, axis=0)
        fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 10), sharex=True)
        n = 100
        top_n = np.random.permutation(len(sampler.flatchain))[:n]
        ax1.errorbar(self.time_data[:-1], self.d14c_data[:-1], yerr=self.d14c_data_error[:-1],
                     fmt="o", color="k", fillstyle="full", capsize=3, markersize=4, label="true d14c")
        for i in tqdm(top_n):
            d14c = self.dc14_fine(params=sampler.flatchain[i, :])
            ax1.plot(self.time_grid_fine, d14c, color="g", alpha=0.2)
            control_points_fine = self.production(self.time_grid_fine, (sampler.flatchain[i, :],))
            ax2.plot(self.time_grid_fine, control_points_fine, color="g", alpha=0.2)
        control_points_fine = self.production(self.time_grid_fine, (mean,))
        ax2.plot(self.time_grid_fine, control_points_fine, "r", label="sample mean production rate")
        ax1.set_ylabel("$\Delta^{14}$C (‰)");
        ax2.set_ylabel("Production rate ($cm^2s^{-1}$)")
        ax2.set_xlabel("Calendar Year")
        if (true_production is not None) & (time_data is not None):
            ax2.plot(time_data, true_production, 'k', label="true production rate")
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), fancybox=True);
        ax1.legend();

    def corner_plot(self, sampler, labels=None, savefile=None):
        ndim = sampler.flatchain.shape[1]
        if labels is not None:
            labels = labels
            figure = corner.corner(sampler.flatchain, labels=labels)
        else:
            figure = corner.corner(sampler.flatchain, labels=labels)
        value = np.mean(sampler.flatchain, axis=0)
        axes = np.array(figure.axes).reshape((ndim, ndim))
        for i in range(ndim):
            ax = axes[i, i]
            ax.axvline(value[i], color="r")

        for yi in range(ndim):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(value[xi], color="r")
                ax.axhline(value[yi], color="r")
                ax.plot(value[xi], value[yi], "sr")
        if savefile is not None:
            figure.savefig(savefile,bbox_inches='tight')

    def plot_samples(self, sampler, savefile=None):
        value = jnp.mean(sampler.flatchain, axis=0)
        fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        samples = sampler.flatchain
        for s in samples[np.random.randint(len(samples), size=100)]:
            d_c_14_fine = self.dc14_fine(params=s)
            ax1.plot(self.time_grid_fine, d_c_14_fine, alpha=0.2, color="g")

        d_c_14_coarse = self.dc14(params=value)
        d_c_14_fine = self.dc14_fine(params=value)
        ax1.plot(self.time_grid_fine, d_c_14_fine, alpha=1, color="k")

        ax1.plot(self.time_data[:-1], d_c_14_coarse, "o", color="k", fillstyle="none", markersize=7)
        ax1.errorbar(self.time_data, self.d14c_data, 
            yerr=self.d14c_data_error, fmt="o", color="k", fillstyle="full", capsize=3, markersize=7)
        ax1.set_ylabel("$\delta^{14}$C (‰)")
        fig.subplots_adjust(hspace=0.05)


        for s in samples[np.random.randint(len(samples), size=10)]:
            production_rate = self.production(self.time_grid_fine,
                                           s[0], s[1], s[2], s[3])
            ax2.plot(self.time_grid_fine, production_rate, alpha=0.25, color="g")

        mean_draw = self.production(self.time_grid_fine, value[0], value[1], value[2], value[3])
        ax2.plot(self.time_grid_fine, mean_draw, color="k", lw=2)
        ax2.set_ylim(jnp.min(mean_draw)*0.8, jnp.max(mean_draw)*1.1);
        ax2.set_xlabel("Calendar Year (CE)");
        ax2.set_ylabel("Production rate ($cm^2s^{-1}$)");
        if savefile is not None:
            figure.savefig(savefile,bbox_inches='tight')