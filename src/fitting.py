import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import rcParams
import celerite
from celerite import terms
from celerite.modeling import Model
import re
import jax.numpy as jnp
from jax import grad, jit, partial
import ticktack
from astropy.table import Table
import emcee
import corner

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

    def load_data(self, file_name, resolution=1000, fine_grid=0.02, time_oversample=1000):
        data = Table.read(file_name, format="ascii")
        self.time_data = jnp.array(data["year"])
        self.d14c_data = jnp.array(data["d14c"])
        self.d14c_data_error = jnp.array(data["sig_d14c"])
        self.start = np.nanmin(self.time_data)
        self.end = np.nanmax(self.time_data)
        self.resolution = resolution
        self.burn_in_time = np.linspace(self.start-1000, self.start, self.resolution)
        self.time_grid_fine = np.arange(self.start, self.end, fine_grid)
        self.time_oversample = time_oversample

    def prepare_function(self, **kwargs):
        self.production = None
        try:
            custom_function = kwargs['custom_function']
            f = kwargs['f']
        except:
            custom_function = False
            f = None

        try:
            use_control_points = kwargs['use_control_points']
            years = kwargs['years']
            control_points = kwargs['control_points']
        except:
            use_control_points = False
            control_points = None
            years = None
        try:
            production = kwargs['production']
        except:
            production = None
        try:
            fit_solar_params = kwargs['fit_solar']
        except:
            fit_solar_params = False

        if production == 'miyake':
            if fit_solar_params == True:
                self.production = self.miyake_event_flexible_solar
            else:
                self.production = self.miyake_event_fixed_solar

        if custom_function == True and f != None:
            self.production = f

        if use_control_points == True and control_points != None:
            def f(tval, *args):
                control_points = jnp.array(list(args))
                t = jnp.linspace(self.start, self.end, num=len(args), endpoint=True)
                return jnp.interp(tval, self.time_data, control_points)
            self.production = f

        if self.production == None:
            self.production = self.miyake_event_fixed_solar
            print("No matching production function, use default "
                  "miyake production with fixed solar cycle (11 yrs) and amplitude (0.18)\n")

    @partial(jit, static_argnums=(0,)) 
    def super_gaussian(self, t, start_time, duration, area):
        middle = start_time+duration/2.
        height = area/duration
        return height*jnp.exp(- ((t-middle)/(1./1.93516*duration))**16.)

    @partial(jit, static_argnums=(0,)) 
    def miyake_event_fixed_solar(self, t, start_time, duration, phase, area):
        height = self.super_gaussian(t, start_time, duration, area)
        prod =  self.steady_state_production + 0.18 * self.steady_state_production * jnp.sin(2 * np.pi / 11 * t + phase) + height
        return prod

    @partial(jit, static_argnums=(0,))
    def miyake_event_flexible_solar(self, t, start_time, duration, phase, area, omega, amplitude):
        height = self.super_gaussian(t, start_time, duration, area)
        prod = self.steady_state_production + amplitude * self.steady_state_production * jnp.sin(
            omega * t + phase) + height
        return prod

    @partial(jit, static_argnums=(0,)) 
    def run(self, time_values, params, y0):
        burn_in, _ = self.cbm.run(time_values, production=self.production, args=params, y0=y0)
        return burn_in

    @partial(jit, static_argnums=(0, 2))
    def run_D_14_C_values(self, time_out, time_oversample, params, y0):
        d_14_c = self.cbm.run_D_14_C_values(time_out, time_oversample, 
                                       production=self.production, args=params, y0=y0,
                                       steady_state_solutions=self.steady_state_y0)
        return d_14_c

    @partial(jit, static_argnums=(0,))
    def dc14(self, params):
    # calls CBM on production_rate of params 
        burn_in = self.run(self.burn_in_time, params, self.steady_state_y0)
        d_14_c = self.run_D_14_C_values(self.time_data, self.time_oversample, params, burn_in[-1, :])
        return d_14_c - 22.72

    @partial(jit, static_argnums=(0,))
    def dc14_fine(self, params):
    # calls CBM on production_rate of params 
        burn_in = self.run(self.burn_in_time, params, self.steady_state_y0)
        d_14_c = self.run_D_14_C_values(self.time_grid_fine, self.time_oversample, params, burn_in[-1, :])
        return d_14_c - 22.72

    @partial(jit, static_argnums=(0,))
    def log_like(self, params):
        # calls dc14 and compare to data, (can be gp or gaussian loglikelihood)
        d_14_c = self.dc14(params)
        
        chi2 = jnp.sum(((self.d14c_data[:-1] - d_14_c)/self.d14c_data_error[:-1])**2)
        like = -0.5*chi2
        return like

    @partial(jit, static_argnums=(0,))
    def log_prior(self, params):
        lp = jnp.where(((params[1]<=0)|(params[1]>=3)), -np.inf, 0)
        return lp

    @partial(jit, static_argnums=(0,))
    def log_prob(self, params):
        # call log_like and log_prior, for later MCMC
        lp = self.log_prior(params)
        pos = self.log_like(params)
        return lp + pos

    def sampling(self, params, burnin=500, production=2000):
        initial = params
        ndim, nwalkers = len(initial), 5*len(initial)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_prob)

        print("Running burn-in...")
        p0 = initial + 1e-5 * np.random.rand(nwalkers, ndim)
        p0, lp, _ = sampler.run_mcmc(p0, burnin, progress=True);

        print("Running production...")
        sampler.reset()
        sampler.run_mcmc(p0, production, progress=True);
        return sampler

    def corner_plot(self, sampler, labels=None, save=False):
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
        if save:
            figure.savefig("corner.jpg")


    def plot_samples(self, sampler, save=False):
        value = jnp.mean(sampler.flatchain, axis=0)
        fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        samples = sampler.flatchain
        for s in samples[np.random.randint(len(samples), size=100)]:
            d_c_14_fine = self.dc14_fine(s)
            ax1.plot(self.time_grid_fine[:-1], d_c_14_fine, alpha=0.2, color="g")

        d_c_14_coarse = self.dc14(value)
        d_c_14_fine = self.dc14_fine(value)
        ax1.plot(self.time_grid_fine[:-1], d_c_14_fine, alpha=1, color="k")

        ax1.plot(self.time_data[:-1], d_c_14_coarse, "o", color="k", fillstyle="none", markersize=7)
        ax1.errorbar(self.time_data, self.d14c_data, 
            yerr=self.d14c_data_error, fmt="o", color="k", fillstyle="full", capsize=3, markersize=7)
        ax1.set_ylabel("$\delta^{14}$C (‰)")
        fig.subplots_adjust(hspace=0.05)


        for s in samples[np.random.randint(len(samples), size=10)]:
            production_rate = self.miyake_event_fixed_solar(self.time_grid_fine,
                                           s[0], s[1], s[2], s[3])
            ax2.plot(self.time_grid_fine, production_rate, alpha=0.25, color="g")

        mean_draw = self.miyake_event(self.time_grid_fine, value[0], value[1], value[2], value[3])
        ax2.plot(self.time_grid_fine, mean_draw, color="k", lw=2)
        ax2.set_ylim(jnp.min(mean_draw)*0.8, jnp.max(mean_draw)*1.1);
        ax2.set_xlabel("Calendar Year (CE)");
        ax2.set_ylabel("Production rate ($cm^2s^{-1}$)");
        if save:
            fig.savefig("samples.jpg")